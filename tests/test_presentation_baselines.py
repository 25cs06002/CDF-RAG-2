"""Deterministic tests for six graph-derived presentation baselines.

No local LLM, Neo4j, model download or user data required.
"""
import tempfile
import unittest
from pathlib import Path

from research_v2.base_pipeline import PipelineRequest
from research_v2.presentation_baselines import (
    BASELINE_NAMES, BaselinePipeline, G0Corpus
)


class FakeEncoder:
    def encode(self, texts, normalize_embeddings=True):
        # Deliberately deterministic, not a real model.
        return [
            [float("heart" in text.lower()), float("asthma" in text.lower()), 0.5]
            for text in texts
        ]


class TestPresentationBaselines(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.corpus = G0Corpus()

    def req(self, query):
        return PipelineRequest(
            run_id="test",
            query_id="T001",
            query=query,
            graph_version="G0",
        )

    def test_six_independent_strategies_exist(self):
        self.assertEqual(len(BASELINE_NAMES), 6)
        self.assertEqual(len(set(BASELINE_NAMES)), 6)

    def test_known_snapshot_record_counts(self):
        self.assertEqual(len(self.corpus.nodes), 35)
        self.assertEqual(len(self.corpus.edges), 37)
        self.assertEqual(len(self.corpus.documents), 37)

    def test_all_baselines_return_valid_records_without_llm(self):
        for name in BASELINE_NAMES:
            with self.subTest(pipeline=name):
                pipeline = BaselinePipeline(
                    name,
                    corpus=self.corpus,
                    dense_encoder=FakeEncoder(),
                    top_k=3,
                    generate=False,
                )
                record = pipeline.run(self.req("What causes heart disease?"))
                self.assertEqual(record.pipeline_id, name)
                self.assertIn(record.status, {"ok", "abstained"})
                self.assertLessEqual(len(record.retrieved_documents), 3)
                self.assertIsNone(record.final_answer)
                self.assertIsNone(record.timings.generation_ms)
                self.assertGreaterEqual(record.timings.retrieval_ms, 0)
                self.assertIsNotNone(record.timings.total_ms)

    def test_unsupported_graph_query_abstains(self):
        for name in ("graph_rag", "causal_kg_rag"):
            with self.subTest(pipeline=name):
                p = BaselinePipeline(name, corpus=self.corpus)
                r = p.run(self.req("What is xyzzqplk syndrome?"))
                self.assertEqual(r.status, "abstained")
                self.assertEqual(r.causal_edges, [])

    def test_causal_graph_only_returns_edges_into_or_ancestors_of_target(self):
        p = BaselinePipeline("causal_kg_rag", corpus=self.corpus, top_k=10)
        r = p.run(self.req("What causes heart disease?"))
        self.assertIn(r.status, {"ok", "abstained"})
        if r.status == "ok":
            self.assertEqual(r.matched_concept.lower(), "heart disease")
            self.assertTrue(any(e.effect.lower() == "heart disease" for e in r.causal_edges))

    def test_top_k_validation(self):
        with self.assertRaisesRegex(ValueError, "positive"):
            BaselinePipeline("bm25_rag", corpus=self.corpus, top_k=0)


if __name__ == "__main__":
    unittest.main()
