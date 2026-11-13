import json
import tempfile
import unittest

from pathlib import Path

from research_v2.base_pipeline import BaseRAGPipeline
from research_v2.schema import PipelineRecord

from research_v2.experiment_runner import (
    load_questions,
    run_experiment,
)


class MockPipeline(BaseRAGPipeline):

    def __init__(self, fail_query=None):
        super().__init__("mock_pipeline")
        self.fail_query = fail_query

    def _run(self, request):
        if request.query_id == self.fail_query:
            raise RuntimeError("Synthetic failure")

        return PipelineRecord(
            run_id=request.run_id,
            query_id=request.query_id,
            pipeline_id=self.pipeline_id,
            original_query=request.query,
            final_answer="Synthetic answer",
        )


class TestExperimentRunner(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

        self.root = Path(self.tmp.name)
        self.input_file = self.root / "questions.jsonl"

        rows = [
            {
                "query_id": "Q001",
                "query": "Question one?",
                "gold_answer": "Hidden from pipeline",
            },
            {
                "query_id": "Q002",
                "query": "Question two?",
            },
            {
                "query_id": "Q003",
                "query": "Question three?",
            },
        ]

        self.input_file.write_text(
            "\n".join(json.dumps(x) for x in rows)
            + "\n",
            encoding="utf-8",
        )

    def run_mock(self, run_id, fail_query=None):
        questions = load_questions(self.input_file)

        return run_experiment(
            pipeline=MockPipeline(fail_query),
            questions=questions,
            run_id=run_id,
            source_file=self.input_file,
            output_root=self.root / "results",
        )

    def test_load_questions(self):
        questions = load_questions(self.input_file)

        self.assertEqual(len(questions), 3)
        self.assertEqual(
            questions[0].query_id, "Q001"
        )

        # Gold answers never reach PipelineRequest.
        self.assertFalse(
            hasattr(questions[0], "gold_answer")
        )

    def test_successful_batch(self):
        summary, folder = self.run_mock("test_success")

        self.assertEqual(
            summary["successful"], 3
        )
        self.assertEqual(
            summary["errors"], 0
        )

        lines = (
            folder / "records.jsonl"
        ).read_text().splitlines()

        self.assertEqual(len(lines), 3)

        records = [
            PipelineRecord.model_validate_json(x)
            for x in lines
        ]

        self.assertTrue(
            all(x.status == "ok" for x in records)
        )

        self.assertTrue(
            (folder / "manifest.json").exists()
        )
        self.assertTrue(
            (folder / "summary.json").exists()
        )

    def test_error_is_recorded_and_batch_continues(self):
        summary, folder = self.run_mock(
            "test_error",
            fail_query="Q002",
        )

        self.assertEqual(
            summary["successful"], 2
        )
        self.assertEqual(
            summary["errors"], 1
        )

        records = [
            json.loads(line)
            for line in (
                folder / "records.jsonl"
            ).read_text().splitlines()
        ]

        self.assertEqual(
            records[1]["status"], "error"
        )
        self.assertEqual(
            records[2]["query_id"], "Q003"
        )

    def test_previous_run_cannot_be_overwritten(self):
        self.run_mock("same_run")

        with self.assertRaises(FileExistsError):
            self.run_mock("same_run")

    def test_duplicate_query_ids_rejected(self):
        self.input_file.write_text(
            '{"query_id":"Q001","query":"A"}\n'
            '{"query_id":"Q001","query":"B"}\n',
            encoding="utf-8",
        )

        with self.assertRaises(ValueError):
            load_questions(self.input_file)


if __name__ == "__main__":
    unittest.main()
