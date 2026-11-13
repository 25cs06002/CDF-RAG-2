import json
import tempfile
import unittest

from pathlib import Path

from research_v2.base_pipeline import (
    BaseRAGPipeline,
)
from research_v2.configured_experiment import (
    run_configured_experiment,
)
from research_v2.experiment_runner import (
    BenchmarkQuestion,
    file_sha256,
    run_experiment,
)
from research_v2.graph_drift import (
    GraphDriftMonitor,
)
from research_v2.schema import PipelineRecord


GOOD_HASH = "a" * 64
BAD_HASH = "b" * 64


class CountingPipeline(BaseRAGPipeline):
    def __init__(self):
        super().__init__("mock_graph_pipeline")
        self.calls = 0

    def _run(self, request):
        self.calls += 1
        return PipelineRecord(
            run_id=request.run_id,
            query_id=request.query_id,
            pipeline_id=self.pipeline_id,
            original_query=request.query,
            final_answer="Synthetic answer",
        )


class TestGraphDrift(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

        self.root = Path(self.tmp.name)

        self.dataset = (
            self.root / "questions.jsonl"
        )

        self.dataset.write_text(
            '{"query_id":"Q1","query":"Question one?"}\n'
            '{"query_id":"Q2","query":"Question two?"}\n'
            '{"query_id":"Q3","query":"Question three?"}\n',
            encoding="utf-8",
        )

        self.questions = [
            BenchmarkQuestion(
                query_id=f"Q{i}",
                query=f"Question {i}?",
            )
            for i in range(1, 4)
        ]

    def read_journal(self, folder):
        path = folder / "graph_checks.jsonl"

        return [
            json.loads(line)
            for line in path.read_text(
                encoding="utf-8"
            ).splitlines()
        ]

    def test_monitor_records_successful_checks(self):
        monitor = GraphDriftMonitor(
            GOOD_HASH,
            reader=lambda: {
                "sha256": GOOD_HASH
            },
        )

        self.assertTrue(
            monitor("before", 1, self.root)
        )
        self.assertTrue(
            monitor("after", 1, self.root)
        )

        journal = self.read_journal(self.root)

        self.assertEqual(len(journal), 2)
        self.assertEqual(
            [row["phase"] for row in journal],
            ["before", "after"],
        )
        self.assertTrue(
            all(
                row["status"] == "pass"
                for row in journal
            )
        )

    def test_changed_graph_detected(self):
        monitor = GraphDriftMonitor(
            GOOD_HASH,
            reader=lambda: {
                "sha256": BAD_HASH
            },
        )

        self.assertFalse(
            monitor("after", 1, self.root)
        )

        journal = self.read_journal(self.root)

        self.assertEqual(
            journal[0]["status"],
            "drift",
        )

    def test_graph_read_error_fails_closed(self):
        def broken_reader():
            raise RuntimeError(
                "Simulated unavailable database"
            )

        monitor = GraphDriftMonitor(
            GOOD_HASH,
            reader=broken_reader,
        )

        self.assertFalse(
            monitor("before", 1, self.root)
        )

        journal = self.read_journal(self.root)

        self.assertEqual(
            journal[0]["status"],
            "error",
        )
        self.assertEqual(
            journal[0]["error_type"],
            "RuntimeError",
        )

    def run_batch(self, checker):
        pipeline = CountingPipeline()

        summary, folder = run_experiment(
            pipeline=pipeline,
            questions=self.questions,
            run_id="test_graph_run",
            source_file=self.dataset,
            output_root=self.root / "runs",
            graph_version="G0",
            graph_checker=checker,
        )

        records = [
            PipelineRecord.model_validate_json(
                line
            )
            for line in (
                folder / "records.jsonl"
            ).read_text(
                encoding="utf-8"
            ).splitlines()
        ]

        return pipeline, summary, records

    def test_drift_before_query_blocks_execution(self):
        pipeline, summary, records = (
            self.run_batch(
                checker=lambda phase, i, folder: False
            )
        )

        self.assertEqual(pipeline.calls, 0)
        self.assertTrue(summary["aborted"])
        self.assertEqual(
            summary["executed_queries"], 0
        )
        self.assertEqual(
            summary["total_queries"], 1
        )
        self.assertEqual(len(records), 1)
        self.assertEqual(
            records[0].error_message,
            "GraphIntegrityFailedBeforeQuery",
        )

    def test_drift_after_query_stops_batch(self):
        def checker(phase, index, folder):
            return not (
                index == 1
                and phase == "after"
            )

        pipeline, summary, records = (
            self.run_batch(checker)
        )

        self.assertEqual(pipeline.calls, 1)
        self.assertTrue(summary["aborted"])
        self.assertEqual(
            summary["planned_queries"], 3
        )
        self.assertEqual(
            summary["total_queries"], 1
        )
        self.assertEqual(
            summary["executed_queries"], 1
        )
        self.assertEqual(
            summary["errors"], 1
        )
        self.assertEqual(len(records), 1)
        self.assertEqual(
            records[0].status,
            "error",
        )

    def test_configured_launcher_uses_graph_guard(self):
        config_path = (
            self.root / "config.json"
        )

        config_path.write_text(
            json.dumps({
                "schema_version": 1,
                "experiment_name": "drift_test",
                "pipeline": "no_rl",
                "dataset_path": "questions.jsonl",
                "dataset_sha256": file_sha256(
                    self.dataset
                ),
                "graph_version": "G0",
                "limit": 1,
            }),
            encoding="utf-8",
        )

        pipeline = CountingPipeline()

        monitor = GraphDriftMonitor(
            GOOD_HASH,
            reader=lambda: {
                "sha256": GOOD_HASH
            },
        )

        summary, folder = (
            run_configured_experiment(
                config_path=config_path,
                output_root=self.root / "configured",
                run_id="configured_guard_test",
                project_root=self.root,
                pipeline_factory=lambda name: pipeline,
                preflight_checker=lambda p, g: {
                    "status": "pass",
                },
                graph_checker=monitor,
            )
        )

        self.assertEqual(
            summary["successful"], 1
        )
        self.assertFalse(
            summary["aborted"]
        )
        self.assertEqual(pipeline.calls, 1)

        journal = self.read_journal(folder)

        self.assertEqual(len(journal), 2)

        provenance = json.loads(
            (
                folder / "config_provenance.json"
            ).read_text(encoding="utf-8")
        )

        self.assertTrue(
            provenance["graph_checked_per_query"]
        )


if __name__ == "__main__":
    unittest.main()
