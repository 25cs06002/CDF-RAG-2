import json
import unittest

from research_v2.base_pipeline import PipelineRequest
from research_v2.legacy_no_rl import LegacyNoRLPipeline
from research_v2.schema import PipelineRecord


class TestLegacyNoRLAdapter(unittest.TestCase):

    def setUp(self):
        self.request = PipelineRequest(
            run_id="step9_test",
            query_id="Q001",
            query="What causes condition X?",
            graph_version="G0",
            random_seed=42,
        )

    def fake_legacy_state(self):
        return {
            "query": "What causes condition X?",
            "refinement_type": "No Refinement",
            "refined_query_text": (
                "What causes condition X?"
            ),
            "refined_query": "Condition X",
            "matched_concept": "Condition X",
            "retrieved_docs": [
                "Factor A",
                "Factor B",
            ],
            "causal_docs": [
                "Factor C → Factor A → Condition X",
            ],
            "rewritten_knowledge": (
                "Synthetic explanation."
            ),
            "final_response": (
                "Synthetic answer."
            ),
            "is_hallucination": False,
            "validator_passed": True,
            "supported_claims": [
                {
                    "cause": "Factor A",
                    "effect": "Condition X",
                    "sentence": "Synthetic claim.",
                }
            ],
            "unsupported_claims": [],
        }

    def test_conversion_to_common_record(self):
        pipeline = LegacyNoRLPipeline(
            executor=lambda query: (
                self.fake_legacy_state()
            )
        )

        result = pipeline.run(self.request)

        self.assertIsInstance(
            result, PipelineRecord
        )
        self.assertEqual(
            result.pipeline_id,
            "cdf_rag_no_rl_legacy"
        )
        self.assertEqual(
            result.matched_concept,
            "Condition X"
        )

        self.assertEqual(
            len(result.causal_edges), 2
        )
        self.assertEqual(
            len(result.causal_paths), 1
        )

        self.assertEqual(
            result.causal_edges[0].cause,
            "Factor A"
        )
        self.assertEqual(
            result.causal_edges[0].effect,
            "Condition X"
        )

        self.assertEqual(
            result.causal_paths[0].nodes,
            [
                "Factor C",
                "Factor A",
                "Condition X",
            ]
        )

        # Cause names must not be mislabeled as
        # source-document evidence.
        self.assertEqual(
            result.retrieved_documents, []
        )

        self.assertFalse(
            result.hallucination_detected
        )
        self.assertTrue(
            result.validator_passed
        )

        self.assertGreaterEqual(
            result.timings.total_ms, 0
        )

    def test_claim_dictionary_is_preserved(self):
        pipeline = LegacyNoRLPipeline(
            executor=lambda query: (
                self.fake_legacy_state()
            )
        )

        result = pipeline.run(self.request)

        claim = json.loads(
            result.supported_claims[0]
        )

        self.assertEqual(
            claim["cause"],
            "Factor A"
        )

        self.assertEqual(
            claim["effect"],
            "Condition X"
        )

    def test_no_evidence_result(self):
        pipeline = LegacyNoRLPipeline(
            executor=lambda query: {
                "refinement_type": "No Refinement",
                "matched_concept": "",
                "retrieved_docs": [],
                "causal_docs": [],
                "final_response": (
                    "No supported evidence found."
                ),
            }
        )

        result = pipeline.run(self.request)

        self.assertEqual(
            result.causal_edges, []
        )
        self.assertEqual(
            result.causal_paths, []
        )
        self.assertIsNone(
            result.hallucination_detected
        )

    def test_invalid_path_is_rejected(self):
        state = self.fake_legacy_state()

        state["causal_docs"] = [
            "Invalid single-node path"
        ]

        pipeline = LegacyNoRLPipeline(
            executor=lambda query: state
        )

        with self.assertRaises(ValueError):
            pipeline.run(self.request)

    def test_missing_target_is_rejected(self):
        state = self.fake_legacy_state()
        state["matched_concept"] = ""

        pipeline = LegacyNoRLPipeline(
            executor=lambda query: state
        )

        with self.assertRaises(ValueError):
            pipeline.run(self.request)

    def test_executor_errors_propagate(self):
        def failing_executor(query):
            raise RuntimeError(
                "Simulated Neo4j failure"
            )

        pipeline = LegacyNoRLPipeline(
            executor=failing_executor
        )

        with self.assertRaises(RuntimeError):
            pipeline.run(self.request)


if __name__ == "__main__":
    unittest.main()
