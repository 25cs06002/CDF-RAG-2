"""Controlled G0 structural benchmark regression tests (offline)."""
import unittest

from research_v2.controlled_g0 import (
    MODEL_NAMES, parse_models, parse_top_k, describe,
)
from research_v2.base_pipeline import PipelineRequest
from research_v2.pipelines.baselines import BaselinePipeline, G0Corpus
from research_v2.schema import PipelineRecord, StageTimings
from research_v2.evaluation import GoldAnnotation
from research_v2.schema import CausalEdge


class ControlledG0Tests(unittest.TestCase):
    def test_knobs(self):
        self.assertEqual(parse_models("B1,B6"), ["B1", "B6"])
        self.assertEqual(parse_top_k("1,3,5,10"), [1, 3, 5, 10])
        for bad in ("0", "-1", "1,1", "5,x", ""):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                parse_top_k(bad)
        for bad in ("B7", "B1,B1", ""):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                parse_models(bad)

    def test_all_six_have_same_graph_edge_corpus(self):
        corpus = G0Corpus()
        self.assertEqual(len(corpus.documents), len(corpus.edges))
        self.assertEqual(len(corpus.edges), 37)
        self.assertEqual(len(MODEL_NAMES), 6)
        for name in ("naive_rag", "bm25_rag", "graph_rag", "causal_kg_rag"):
            pipeline = BaselinePipeline(name, corpus=corpus, top_k=5)
            record = pipeline.run(PipelineRequest(
                run_id="t", query_id="q", query="What causes heart disease?"
            ))
            self.assertTrue(record.causal_edges)
            self.assertTrue(all(edge in corpus.edges for edge in
                                [(e.cause, e.effect) for e in record.causal_edges]))

    def test_describe_scores_direct_g0_edges_only(self):
        record = PipelineRecord(
            run_id="t", query_id="q", pipeline_id="bm25_rag",
            original_query="What causes heart disease?", status="ok",
            causal_edges=[CausalEdge(cause="smoking", effect="heart disease")],
            timings=StageTimings(total_ms=2, retrieval_ms=1),
        )
        annotation = GoldAnnotation(
            query_id="q", reference_basis="graph_snapshot",
            causal_edges=[CausalEdge(cause="smoking", effect="heart disease")],
        )
        summary, details = describe([record], [annotation])
        self.assertEqual(summary["mapped_questions"], 1)
        self.assertEqual(summary["g0_edge_micro_f1"], 1.0)
        self.assertEqual(summary["mean_query_latency_ms"], 2)
        self.assertEqual(len(details), 1)

    def test_no_scoring_without_mapped_subset(self):
        record = PipelineRecord(
            run_id="t", query_id="unmapped", pipeline_id="bm25_rag",
            original_query="example", status="abstained",
        )
        summary, details = describe([record], [
            GoldAnnotation(query_id="different", reference_basis="graph_snapshot",
                           causal_edges=[])
        ])
        self.assertIsNone(summary["g0_edge_micro_f1"])
        self.assertFalse(details)


if __name__ == "__main__":
    unittest.main()
