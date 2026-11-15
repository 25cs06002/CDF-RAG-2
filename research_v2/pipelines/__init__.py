"""B1–B8 model registry: one discoverable import path for all pipelines.

The implementation remains in existing tested modules to avoid breaking
historic code. Use `python3 -m research_v2.pipelines` to list all models.
"""
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PipelineSpec:
    model_id: str
    name: str
    implementation: str
    evidence: str
    generator_required: bool


MODEL_REGISTRY = {
    "B1": PipelineSpec("B1", "Naive lexical RAG", "naive_rag", "phase2_candidate_summaries", False),
    "B2": PipelineSpec("B2", "BM25 RAG", "bm25_rag", "phase2_candidate_summaries", False),
    "B3": PipelineSpec("B3", "Dense Vector RAG", "dense_rag", "phase2_candidate_summaries", False),
    "B4": PipelineSpec("B4", "Hybrid BM25 + Dense RAG", "hybrid_rag", "phase2_candidate_summaries", False),
    "B5": PipelineSpec("B5", "Graph-neighborhood RAG", "graph_rag", "g0_graph", False),
    "B6": PipelineSpec("B6", "Directed Causal KG-RAG", "causal_kg_rag", "g0_graph", False),
    "B7": PipelineSpec("B7", "CDF-RAG No-RL (legacy)", "no_rl", "g0_neo4j", True),
    "B8": PipelineSpec("B8", "CDF-RAG PPO RL (legacy)", "rl", "g0_neo4j", True),
}


def create_pipeline(model_id: str, *, document_corpus: Path | None = None,
                    top_k: int = 5, generate: bool = False, encoder: str = "sentence"):
    """Instantiate a model by B-number; never load PPO or embeddings on import.

    B1-B4 use the same source-summary corpus when supplied; B5-B6 use G0
    graph edges; B7-B8 use their original live Neo4j/Ollama adapters.
    """
    if model_id not in MODEL_REGISTRY:
        raise ValueError("Unknown pipeline ID; choose B1-B8")
    item = MODEL_REGISTRY[model_id]
    if item.model_id == "B7":
        from .legacy_no_rl import LegacyNoRLPipeline
        return LegacyNoRLPipeline()
    if item.model_id == "B8":
        from .legacy_rl import LegacyRLPipeline
        return LegacyRLPipeline()

    from .baselines import (
        BaselinePipeline, G0Corpus, CandidateCorpus, LSAEncoder
    )
    if item.evidence == "phase2_candidate_summaries":
        if document_corpus is None:
            raise ValueError("B1-B4 require a frozen source-attributed document corpus")
        corpus = CandidateCorpus(document_corpus)
    else:
        corpus = G0Corpus()
    if encoder not in {"sentence", "lsa"}:
        raise ValueError("Encoder must be sentence or lsa")
    dense_encoder = (
        LSAEncoder() if encoder == "lsa" and model_id in {"B3", "B4"} else None
    )
    return BaselinePipeline(
        item.implementation,
        corpus=corpus,
        top_k=top_k,
        generate=generate,
        dense_encoder=dense_encoder,
    )


__all__ = ["PipelineSpec", "MODEL_REGISTRY", "create_pipeline"]
