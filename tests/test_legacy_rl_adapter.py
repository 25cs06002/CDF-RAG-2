import unittest

from research_v2.base_pipeline import PipelineRequest
from research_v2.legacy_rl import LegacyRLPipeline
from research_v2.legacy_no_rl import LegacyNoRLPipeline
from research_v2.schema import PipelineRecord


class TestLegacyRLAdapter(unittest.TestCase):

    def setUp(self):
        self.request = PipelineRequest(
            run_id="step10_test",
            query_id="Q001",
            query="What causes condition X?",
            graph_version="G0",
            random_seed=42,
        )

    def fake_rl_state(self):
        return {
            "query": self.request.query,
            "refinement_type": "Expand Query",
            "refined_query_text": (
                "What factors contribute to condition X?"
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
                "Synthetic evidence."
            ),
            "final_response": (
                "Synthetic answer."
            ),
            "is_hallucination": False,
            "validator_passed": True,
            "supported_claims": [],
            "unsupported_claims": [],
        }

    def test_rl_output_contract(self):
        pipeline = LegacyRLPipeline(
            executor=lambda query: self.fake_rl_state()
        )

        result = pipeline.run(self.request)

        self.assertIsInstance(
            result, PipelineRecord
        )
        self.assertEqual(
            result.pipeline_id,
            "cdf_rag_rl_legacy"
        )
        self.assertEqual(
            result.refinement_action,
            "Expand Query"
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

    def test_refined_query_preserved(self):
        result = LegacyRLPipeline(
            executor=lambda query: self.fake_rl_state()
        ).run(self.request)

        self.assertEqual(
            result.refined_query,
            "What factors contribute to condition X?"
        )
        self.assertEqual(
            result.original_query,
            self.request.query
        )

    def test_rl_and_no_rl_ids_are_distinct(self):
        rl = LegacyRLPipeline(
            executor=lambda query: self.fake_rl_state()
        )

        no_rl = LegacyNoRLPipeline(
            executor=lambda query: self.fake_rl_state()
        )

        self.assertNotEqual(
            rl.pipeline_id,
            no_rl.pipeline_id
        )

    def test_serialization(self):
        result = LegacyRLPipeline(
            executor=lambda query: self.fake_rl_state()
        ).run(self.request)

        restored = PipelineRecord.model_validate_json(
            result.model_dump_json()
        )

        self.assertEqual(restored, result)

    def test_executor_error_propagates(self):
        def fail(query):
            raise RuntimeError("Simulated failure")

        pipeline = LegacyRLPipeline(executor=fail)

        with self.assertRaises(RuntimeError):
            pipeline.run(self.request)

    def test_invalid_causal_path_rejected(self):
        state = self.fake_rl_state()
        state["causal_docs"] = ["Invalid path"]

        pipeline = LegacyRLPipeline(
            executor=lambda query: state
        )

        with self.assertRaises(ValueError):
            pipeline.run(self.request)


if __name__ == "__main__":
    unittest.main()
