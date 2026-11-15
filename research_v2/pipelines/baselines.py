"""Six presentation baseline retrieval configurations on the archived G0 snapshot.

WARNING: The default corpus is made by verbalizing G0 edges. These are
GRAPH-DERIVED records, NOT independently verified medical source documents.
Run results on the 100 unreviewed pilot are exploratory, not clinical accuracy.

Usage:
  python3 -m research_v2.presentation_baselines --pipeline bm25_rag \
    --input presentation_20261118/benchmark/questions_only.jsonl --limit 3
  python3 -m research_v2.presentation_baselines --pipeline all \
    --input presentation_20261118/benchmark/questions_only.jsonl --limit 100
  # Add --generate to run the shared Ollama generator (requires local Ollama).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from ..base_pipeline import BaseRAGPipeline, PipelineRequest
from ..experiment_runner import PROJECT_ROOT, load_questions, run_experiment
from ..schema import CausalEdge, CausalPath, EvidenceDocument, PipelineRecord, StageTimings

G0 = PROJECT_ROOT / "research_artifacts/graph-G0"
PRESENTATION_ROOT = PROJECT_ROOT / "presentation_20261118"
BASELINE_NAMES = (
    "naive_rag", "bm25_rag", "dense_rag", "hybrid_rag",
    "graph_rag", "causal_kg_rag",
)
WORD = re.compile(r"[a-z0-9]+")
STOP = frozenset(
    "a an and are as at be can does for from how i in is it of on or "
    "the their these this to what which why with would might about "
    "disease condition factors risk cause causes associated explain".split()
)


def tokenize(text):
    return [token for token in WORD.findall(text.lower()) if token not in STOP]


def similarity(a, b):
    left, right = set(tokenize(a)), set(tokenize(b))
    return len(left & right) / len(left | right) if left and right else 0.0


class G0Corpus:
    """A fixed reproducibility/smoke corpus; no claims of document provenance."""

    def __init__(self, graph_dir: Path = G0):
        with (Path(graph_dir) / "nodes.csv").open(newline="", encoding="utf-8") as handle:
            self.nodes = [row["name"].strip() for row in csv.DictReader(handle)]
        with (Path(graph_dir) / "edges.csv").open(newline="", encoding="utf-8") as handle:
            self.edges = [(row["cause"].strip(), row["effect"].strip()) for row in csv.DictReader(handle)]
        if not self.nodes or not self.edges:
            raise ValueError("Empty graph")
        if any(a not in self.nodes or b not in self.nodes for a, b in self.edges):
            raise ValueError("Graph edge references an unknown concept")

        self.documents = [
            EvidenceDocument(
                document_id=f"G0_EDGE_{index:04d}",
                text=f"{cause} is linked as a cause of {effect} in the G0 graph snapshot.",
                source_id="G0_GRAPH_DERIVED_UNVALIDATED",
            )
            for index, (cause, effect) in enumerate(self.edges, 1)
        ]
        self.graph_documents = list(self.documents)
        self.tokens = [tokenize(item.text) for item in self.documents]
        self.lengths = [len(tokens) for tokens in self.tokens]
        self.avg_length = sum(self.lengths) / len(self.lengths)
        self.df = Counter(word for tokens in self.tokens for word in set(tokens))
        self.reverse = defaultdict(list)
        self.neighbors = defaultdict(set)
        for idx, (cause, effect) in enumerate(self.edges):
            self.reverse[effect].append((cause, idx))
            self.neighbors[cause].add(idx)
            self.neighbors[effect].add(idx)

    def concept(self, query):
        q = query.lower()
        exact = [node for node in self.nodes if node.lower() in q]
        if exact:
            return max(exact, key=lambda node: len(node))
        scored = [(similarity(query, node), node) for node in self.nodes]
        value, node = max(scored, default=(0, None))
        return node if value >= 0.40 else None

    def bm25_scores(self, query):
        terms = tokenize(query)
        scores = []
        count = len(self.documents)
        for length, words in zip(self.lengths, self.tokens):
            freqs = Counter(words)
            score = 0.0
            for word in terms:
                freq = freqs[word]
                if freq == 0:
                    continue
                df = self.df[word]
                idf = math.log(1 + (count - df + 0.5) / (df + 0.5))
                score += idf * freq * 2.2 / (
                    freq + 1.2 * (0.25 + 0.75 * length / self.avg_length)
                )
            scores.append(score)
        return scores

    def lexical_scores(self, query):
        terms = set(tokenize(query))
        return [len(terms & set(tokens)) for tokens in self.tokens]


class CandidateCorpus(G0Corpus):
    """Source-attributed candidate summaries for B1-B4; graph stays fixed at G0."""

    def __init__(self, documents_path: Path):
        super().__init__()
        path = Path(documents_path).expanduser().resolve()
        if not path.is_relative_to(PROJECT_ROOT):
            raise ValueError("Document corpus must be inside CDF-RAG-2")
        documents = []
        seen = set()
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                document = EvidenceDocument.model_validate(row)
                if document.document_id in seen:
                    raise ValueError(f"Duplicate document ID: {document.document_id}")
                if not document.text.strip() or not document.source_id:
                    raise ValueError("Source-attributed text required")
                seen.add(document.document_id)
                documents.append(document)
        if not documents:
            raise ValueError("Document corpus is empty")
        self.documents = documents
        self.tokens = [tokenize(doc.text) for doc in documents]
        self.lengths = [len(tokens) for tokens in self.tokens]
        self.avg_length = sum(self.lengths) / len(self.lengths)
        if self.avg_length <= 0:
            raise ValueError("Document corpus has no usable tokens")
        self.df = Counter(word for tokens in self.tokens for word in set(tokens))


class LSAEncoder:
    """Small, explicitly labeled TF-IDF + LSA dense baseline for offline pilots.

    This is not a pretrained sentence-transformer model. Use the identical
    LSA encoder for dense and hybrid retrieval when comparing those methods.
    """

    def __init__(self, dimensions=64):
        self.dimensions = dimensions
        self.vectorizer = None
        self.reducer = None

    def encode(self, texts, normalize_embeddings=True):
        import numpy as np
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD
        from sklearn.preprocessing import normalize

        if self.vectorizer is None:
            self.vectorizer = TfidfVectorizer(
                lowercase=True, token_pattern=r"(?u)\b\w+\b"
            )
            vectors = self.vectorizer.fit_transform(texts)
            upper = min(vectors.shape) - 1
            if upper >= 2:
                self.reducer = TruncatedSVD(
                    n_components=min(self.dimensions, upper),
                    random_state=2026,
                )
                values = self.reducer.fit_transform(vectors)
            else:
                values = vectors.toarray()
        else:
            vectors = self.vectorizer.transform(texts)
            values = (
                self.reducer.transform(vectors)
                if self.reducer is not None else vectors.toarray()
            )
        return normalize(values).astype(float) if normalize_embeddings else np.asarray(values, dtype=float)


class BaselinePipeline(BaseRAGPipeline):
    """One of six distinct retrieval strategies; optional shared Ollama generation."""

    def __init__(self, name, corpus=None, top_k=5, generate=False, dense_encoder=None):
        if name not in BASELINE_NAMES:
            raise ValueError(f"Unsupported baseline: {name}")
        super().__init__(name)
        self.corpus = corpus if corpus is not None else G0Corpus()
        self.top_k = top_k
        if top_k < 1:
            raise ValueError("top_k must be positive")
        self.generate = generate
        self._encoder = dense_encoder
        self._dense_vectors = None
        if name in {"dense_rag", "hybrid_rag"}:
            if self._encoder is None:
                from sentence_transformers import SentenceTransformer
                from ..sentence_model_assets import resolve_cached_snapshot
                snapshot = resolve_cached_snapshot()
                self._encoder = SentenceTransformer(str(snapshot), local_files_only=True)
            vectors = self._encoder.encode([d.text for d in self.corpus.documents], normalize_embeddings=True)
            self._dense_vectors = vectors

    @staticmethod
    def rank(scores, top_k):
        ranked = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
        return [i for i in ranked if scores[i] > 0][:top_k]

    def dense_scores(self, query):
        if self._dense_vectors is None:
            raise ValueError("Dense model not initialized")
        q = self._encoder.encode([query], normalize_embeddings=True)[0]
        return [float(sum(float(a)*float(b) for a, b in zip(q, row)))
                for row in self._dense_vectors]

    def select(self, query):
        """Return document indices and the resolved graph concept."""
        c = self.corpus
        concept = c.concept(query)
        if self.pipeline_id == "naive_rag":
            return self.rank(c.lexical_scores(query), self.top_k), concept
        if self.pipeline_id == "bm25_rag":
            return self.rank(c.bm25_scores(query), self.top_k), concept
        if self.pipeline_id == "dense_rag":
            return self.rank(self.dense_scores(query), self.top_k), concept
        if self.pipeline_id == "hybrid_rag":
            lexical = self.rank(c.bm25_scores(query), max(self.top_k * 3, 10))
            semantic = self.rank(self.dense_scores(query), max(self.top_k * 3, 10))
            rrf = defaultdict(float)
            for ids in (lexical, semantic):
                for position, index in enumerate(ids, 1):
                    rrf[index] += 1.0 / (60 + position)
            ranked = sorted(rrf, key=lambda index: (-rrf[index], index))
            return ranked[:self.top_k], concept
        if concept is None:
            return [], None
        if self.pipeline_id == "graph_rag":
            # Undirected one-hop neighborhood: general graph proximity.
            return sorted(c.neighbors[concept])[:self.top_k], concept

        # Directed causal evidence. Reserve an edge budget for *complete*
        # two-hop paths; simply truncating all direct edges first can
        # permanently exclude every intermediate ancestor at top_k=5.
        direct = [index for _, index in c.reverse[concept]]
        complete_paths = [
            (direct_index, ancestor_index)
            for parent, direct_index in c.reverse[concept]
            for _, ancestor_index in c.reverse[parent]
        ]
        chosen = []
        if self.top_k >= 2 and complete_paths:
            # First complete causal path has two connected directed edges.
            chosen.extend(complete_paths[0])
        for index in direct:
            if index not in chosen and len(chosen) < self.top_k:
                chosen.append(index)
        for first, second in complete_paths:
            if len(chosen) >= self.top_k:
                break
            if first in chosen and second not in chosen:
                chosen.append(second)
        return chosen[:self.top_k], concept

    def _run(self, request):
        start = time.perf_counter()
        indices, concept = self.select(request.query)
        retrieval_ms = (time.perf_counter() - start) * 1000.0
        graph_mode = self.pipeline_id in {"graph_rag", "causal_kg_rag"}
        document_mode = isinstance(self.corpus, CandidateCorpus)
        documents = [
            (self.corpus.graph_documents if graph_mode else self.corpus.documents)[i]
            for i in indices
        ]
        # Text documents are not silently converted to causal graph edges.
        edges = [
            CausalEdge(cause=self.corpus.edges[i][0], effect=self.corpus.edges[i][1])
            for i in indices
        ] if graph_mode or not document_mode else []
        paths = []
        if self.pipeline_id == "causal_kg_rag" and concept:
            present = set(indices)
            for parent, direct_index in self.corpus.reverse[concept]:
                for grandparent, second_index in self.corpus.reverse[parent]:
                    if direct_index in present and second_index in present:
                        paths.append(CausalPath(nodes=[grandparent, parent, concept]))

        answer = None
        generation_ms = None
        status = "ok" if documents else "abstained"
        if self.generate and documents:
            generation_start = time.perf_counter()
            answer = self._generate_answer(request.query, documents)
            generation_ms = (time.perf_counter() - generation_start) * 1000.0
        total_ms = (time.perf_counter() - start) * 1000.0
        return PipelineRecord(
            run_id=request.run_id,
            query_id=request.query_id,
            pipeline_id=self.pipeline_id,
            original_query=request.query,
            graph_version=request.graph_version,
            matched_concept=concept,
            retrieved_documents=documents,
            causal_edges=edges,
            causal_paths=paths,
            final_answer=answer,
            status=status,
            generator_model=(os.getenv("OLLAMA_MODEL", "llama3.2:latest") if self.generate else None),
            random_seed=request.random_seed,
            timings=StageTimings(
                retrieval_ms=retrieval_ms,
                generation_ms=generation_ms,
                total_ms=total_ms,
            ),
        )

    @staticmethod
    def _generate_answer(query, documents):
        import requests
        context = "\n".join(f"[{d.document_id}] {d.text}" for d in documents)
        prompt = (
            "Answer using only the evidence below. If insufficient, state that. "
            "Evidence is unverified G0 graph data or AI-assisted source-candidate summaries, NOT clinical guidance. "
            "Do not claim independent medical certainty. Cite evidence IDs.\n\n"
            f"Evidence:\n{context}\n\nQuestion: {query}\nAnswer:"
        )
        base = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
        response = requests.post(
            base + "/api/generate",
            json={
                "model": os.getenv("OLLAMA_MODEL", "llama3.2:latest"),
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0, "num_predict": 256},
            },
            timeout=180,
        )
        response.raise_for_status()
        return response.json()["response"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline", choices=[*BASELINE_NAMES, "all"], required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--document-corpus", type=Path, help="Source-attributed JSONL for B1-B4")
    parser.add_argument("--encoder", choices=["sentence", "lsa"], default="sentence", help="Dense retrieval encoder; lsa requires scikit-learn")
    parser.add_argument("--output-root", type=Path, default=PRESENTATION_ROOT / "runs")
    args = parser.parse_args()
    if Path.cwd().resolve() != PROJECT_ROOT:
        parser.error("Run from the CDF-RAG-2 project root")
    questions = load_questions(args.input)
    if args.limit is not None:
        if args.limit < 1:
            parser.error("--limit must be positive")
        questions = questions[:args.limit]
    if not args.output_root.expanduser().resolve().is_relative_to(PROJECT_ROOT):
        parser.error("Outputs must remain inside the repository")
    graph_corpus = G0Corpus()
    document_corpus = CandidateCorpus(args.document_corpus) if args.document_corpus else graph_corpus
    for name in BASELINE_NAMES if args.pipeline == "all" else [args.pipeline]:
        corpus = graph_corpus if name in {"graph_rag", "causal_kg_rag"} else document_corpus
        dense_encoder = LSAEncoder() if args.encoder == "lsa" and name in {"dense_rag", "hybrid_rag"} else None
        pipeline = BaselinePipeline(
            name, corpus, top_k=args.top_k, generate=args.generate,
            dense_encoder=dense_encoder,
        )
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        summary, output_dir = run_experiment(
            pipeline=pipeline, questions=questions,
            run_id=f"presentation_{name}_{stamp}",
            source_file=args.input,
            output_root=args.output_root,
            graph_version="G0",
        )
        print(json.dumps({"output_dir": str(output_dir), **summary}, indent=2))


if __name__ == "__main__":
    main()
