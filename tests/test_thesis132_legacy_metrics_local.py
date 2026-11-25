"""Synthetic 132-row legacy-metric bridge regression tests."""
import unittest

from research_v2.schema import PipelineRecord, EvidenceDocument, CausalEdge, StageTimings
from research_v2.thesis132_legacy_metrics import LEGACY_METRICS, summarize, _value_metrics


class LegacyMetricBridgeTests(unittest.TestCase):
    def synthetic(self):
        data = []
        for i in range(132):
            data.append(PipelineRecord(
                run_id="synthetic", query_id=f"Q{i:03d}", pipeline_id="B1",
                original_query=f"Synthetic {i}", status="ok",
                retrieved_documents=[EvidenceDocument(document_id="D", source_id="SRC",
                                                      text="synthetic")],
                causal_edges=[CausalEdge(cause="A", effect="B", relation="CAUSES")],
                final_answer="synthetic two words", timings=StageTimings(total_ms=1000),
            ))
        return data

    def test_complete_132_has_legacy_names_but_no_fake_accuracy(self):
        result = summarize(self.synthetic(), wall_seconds=264.0)
        self.assertEqual(set(result["metrics"]), set(LEGACY_METRICS))
        self.assertEqual(result["queries"], 132)
        self.assertEqual(result["metrics"]["Retrieval Success Rate"]["value"], 1)
        self.assertEqual(result["metrics"]["Average Throughput (QPS)"]["value"], .5)
        self.assertEqual(result["metrics"]["Total Execution Time (s)"]["value"], 264.0)
        for unavailable in ("Groundedness", "Context Relevance", "Hallucination Rate",
                            "Semantic Refinement Score", "Refinement Success Rate"):
            self.assertIsNone(result["metrics"][unavailable]["value"])
        self.assertEqual(result["metrics"]["Average Direct Causes"]["value"], 1)

    def test_incomplete_or_duplicate_session_rejected(self):
        records = self.synthetic()
        with self.assertRaisesRegex(ValueError, "exactly 132"):
            summarize(records[:-1])
        records[-1] = records[0]
        with self.assertRaisesRegex(ValueError, "exactly 132"):
            summarize(records)

    def test_missing_wall_time_is_null_not_fake_qps(self):
        computed = _value_metrics(self.synthetic(), wall_seconds=None)
        self.assertIsNone(computed["Average Throughput (QPS)"][0])
        self.assertIsNone(computed["Total Execution Time (s)"][0])


if __name__ == "__main__":
    unittest.main()
