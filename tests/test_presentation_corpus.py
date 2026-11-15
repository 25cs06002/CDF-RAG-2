"""Offline tests for the provisional Phase 2 source summary corpus."""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from research_v2.presentation_corpus import build_candidate_documents, prepare
from research_v2.presentation_baselines import CandidateCorpus, BaselinePipeline
from research_v2.base_pipeline import PipelineRequest


class TestPresentationCorpus(unittest.TestCase):
    def test_documents_have_source_provenance_and_no_leaked_labels(self):
        documents, metadata = build_candidate_documents()
        self.assertGreaterEqual(len(documents), 20)
        self.assertEqual(len(documents), len(metadata))
        self.assertEqual(len({d["document_id"] for d in documents}), len(documents))
        self.assertTrue(all(d["source_id"].startswith("https://") for d in documents))
        self.assertTrue(all(not item["medical_gold"] for item in metadata))
        for doc in documents:
            self.assertEqual(set(doc), {"document_id", "text", "source_id"})
            self.assertNotIn("gold_answer", doc)
            self.assertNotIn("query_id", doc)

    def test_corpus_is_frozen_and_document_evidence_is_not_graph_edges(self):
        with tempfile.TemporaryDirectory(prefix="candidate_corpus_", dir=__import__("research_v2.presentation_corpus", fromlist=["PROJECT_ROOT"]).PROJECT_ROOT) as temp:
            output = Path(temp) / "collection"
            with contextlib.redirect_stdout(io.StringIO()):
                prepare(output)
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertFalse(manifest["human_verified"])
            self.assertFalse(manifest["medical_gold_created"])
            corpus = CandidateCorpus(output / "documents.jsonl")
            pipeline = BaselinePipeline("bm25_rag", corpus=corpus, top_k=5)
            req = PipelineRequest(
                run_id="test",
                query_id="candidate",
                query="What factors contribute to asthma?",
                graph_version="G0",
            )
            record = pipeline.run(req)
            self.assertEqual(record.pipeline_id, "bm25_rag")
            self.assertEqual(record.causal_edges, [])
            self.assertEqual(record.causal_paths, [])
            self.assertEqual(record.status, "ok")
            with self.assertRaises(FileExistsError):
                prepare(output)


if __name__ == "__main__":
    unittest.main()
