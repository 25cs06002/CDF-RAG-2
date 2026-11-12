import unittest

from pydantic import ValidationError

from research_v2.base_pipeline import (
    BaseRAGPipeline,
    PipelineRequest,
)
from research_v2.schema import PipelineRecord


class EchoPipeline(BaseRAGPipeline):
    """
    Lightweight test-only pipeline.

    No actual retrieval or generation.
    """

    def __init__(self):
        super().__init__(
            pipeline_id="echo_test"
        )

    def _run(self, request):

        return PipelineRecord(
            run_id=request.run_id,
            query_id=request.query_id,
            pipeline_id=self.pipeline_id,
            original_query=request.query,
            graph_version=request.graph_version,
            random_seed=request.random_seed,
            final_answer=(
                "Test response; no retrieval performed."
            ),
        )


class WrongIdPipeline(BaseRAGPipeline):

    def __init__(self):
        super().__init__(
            pipeline_id="wrong_test"
        )

    def _run(self, request):

        return PipelineRecord(
            run_id="wrong_run",
            query_id=request.query_id,
            pipeline_id=self.pipeline_id,
            original_query=request.query,
        )


class WrongTypePipeline(BaseRAGPipeline):

    def __init__(self):
        super().__init__(
            pipeline_id="wrong_type"
        )

    def _run(self, request):
        return {
            "answer": "not a PipelineRecord"
        }


class TestPipelineInterface(unittest.TestCase):

    def setUp(self):

        self.request = PipelineRequest(
            run_id="smoke_001",
            query_id="Q001",
            query="What causes condition X?",
            graph_version="G0",
            random_seed=42,
        )

    def test_pipeline_contract_and_json_roundtrip(self):

        record = EchoPipeline().run(
            self.request
        )

        self.assertEqual(
            record.pipeline_id,
            "echo_test",
        )

        self.assertEqual(
            record.graph_version,
            "G0",
        )

        self.assertEqual(
            record.original_query,
            self.request.query,
        )

        restored = PipelineRecord.model_validate_json(
            record.model_dump_json()
        )

        self.assertEqual(restored, record)

    def test_blank_query_is_rejected(self):

        with self.assertRaises(ValidationError):
            PipelineRequest(
                run_id="x",
                query_id="y",
                query="   ",
            )

    def test_mismatched_identifiers_are_rejected(self):

        with self.assertRaisesRegex(
            ValueError,
            "run_id",
        ):
            WrongIdPipeline().run(
                self.request
            )

    def test_invalid_return_type_is_rejected(self):

        with self.assertRaises(TypeError):
            WrongTypePipeline().run(
                self.request
            )

    def test_invalid_request_type_is_rejected(self):

        with self.assertRaises(TypeError):
            EchoPipeline().run({
                "query": "test"
            })


if __name__ == "__main__":
    unittest.main()
