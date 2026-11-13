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
    file_sha256,
)

from research_v2.schema import PipelineRecord


class InspectingPipeline(BaseRAGPipeline):

    def __init__(self, run_dir, fail=False):
        super().__init__(
            "cdf_rag_no_rl_legacy"
        )

        self.run_dir = run_dir
        self.fail = fail
        self.metadata_visible = False

    def _run(self, request):
        required_files = [
            "manifest.json",
            "runtime_preflight.json",
            "experiment_config.json",
            "config_provenance.json",
        ]

        self.metadata_visible = all(
            (self.run_dir / name).is_file()
            for name in required_files
        )

        if not self.metadata_visible:
            raise AssertionError(
                "Provenance was not saved before inference"
            )

        if self.fail:
            raise RuntimeError(
                "Simulated interrupted query"
            )

        return PipelineRecord(
            run_id=request.run_id,
            query_id=request.query_id,
            pipeline_id=self.pipeline_id,
            original_query=request.query,
            final_answer="Synthetic answer",
        )


class TestPreRunProvenance(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

        self.root = Path(self.tmp.name)

        dataset = self.root / "questions.jsonl"

        dataset.write_text(
            '{"query_id":"Q001","query":"Test question?"}\n',
            encoding="utf-8",
        )

        self.config_path = (
            self.root / "config.json"
        )

        self.config_path.write_text(
            json.dumps({
                "schema_version": 1,
                "experiment_name": "provenance_test",
                "pipeline": "no_rl",
                "dataset_path": "questions.jsonl",
                "dataset_sha256": file_sha256(
                    dataset
                ),
                "graph_version": "G0",
                "limit": 1,
            }),
            encoding="utf-8",
        )

    def run_case(self, fail=False):
        run_dir = (
            self.root / "runs" / "test_run"
        )

        pipeline = InspectingPipeline(
            run_dir,
            fail=fail,
        )

        summary, output_dir = (
            run_configured_experiment(
                config_path=self.config_path,
                output_root=self.root / "runs",
                project_root=self.root,
                run_id="test_run",
                pipeline_factory=lambda _: pipeline,
                preflight_checker=lambda p, g: {
                    "status": "pass",
                    "pipeline": p,
                    "graph_version": g,
                },
            )
        )

        return summary, output_dir, pipeline

    def test_metadata_exists_before_first_query(self):
        summary, folder, pipeline = (
            self.run_case()
        )

        self.assertTrue(
            pipeline.metadata_visible
        )

        self.assertEqual(
            summary["successful"], 1
        )

        provenance = json.loads(
            (
                folder / "config_provenance.json"
            ).read_text(encoding="utf-8")
        )

        self.assertTrue(
            provenance[
                "provenance_written_before_queries"
            ]
        )

    def test_metadata_survives_failed_query(self):
        summary, folder, pipeline = (
            self.run_case(fail=True)
        )

        self.assertTrue(
            pipeline.metadata_visible
        )

        self.assertEqual(
            summary["errors"], 1
        )

        self.assertTrue(
            (
                folder / "runtime_preflight.json"
            ).exists()
        )

        self.assertTrue(
            (
                folder / "experiment_config.json"
            ).exists()
        )

        records = (
            folder / "records.jsonl"
        ).read_text(
            encoding="utf-8"
        ).splitlines()

        self.assertEqual(
            len(records), 1
        )

        record = PipelineRecord.model_validate_json(
            records[0]
        )

        self.assertEqual(
            record.status,
            "error"
        )


if __name__ == "__main__":
    unittest.main()
