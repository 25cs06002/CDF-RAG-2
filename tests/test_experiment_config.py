import json
import tempfile
import unittest

from pathlib import Path

from pydantic import ValidationError

from research_v2.base_pipeline import BaseRAGPipeline
from research_v2.schema import PipelineRecord

from research_v2.experiment_config import (
    prepare_experiment,
)

from research_v2.configured_experiment import (
    run_configured_experiment,
)

from research_v2.experiment_runner import (
    file_sha256,
)


class MockPipeline(BaseRAGPipeline):

    def __init__(self):
        super().__init__(
            "cdf_rag_no_rl_legacy"
        )

    def _run(self, request):
        return PipelineRecord(
            run_id=request.run_id,
            query_id=request.query_id,
            pipeline_id=self.pipeline_id,
            original_query=request.query,
            final_answer="Synthetic test answer.",
        )


class TestExperimentConfig(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

        self.root = Path(self.tmp.name)

        self.dataset = self.root / "questions.jsonl"

        self.dataset.write_text(
            '{"query_id":"Q001","query":"Question one?"}\n'
            '{"query_id":"Q002","query":"Question two?"}\n',
            encoding="utf-8",
        )

        self.config_data = {
            "schema_version": 1,
            "experiment_name": "unit_test_v1",
            "pipeline": "no_rl",
            "dataset_path": "questions.jsonl",
            "dataset_sha256": file_sha256(
                self.dataset
            ),
            "graph_version": "G0",
            "random_seed": None,
            "limit": 1,
            "purpose": "Unit testing only",
        }

        self.config_path = self.root / "config.json"
        self.write_config()

    def write_config(self):
        self.config_path.write_text(
            json.dumps(self.config_data, indent=2),
            encoding="utf-8",
        )

    def prepare(self):
        return prepare_experiment(
            self.config_path,
            project_root=self.root,
        )

    def test_valid_configuration(self):
        config, dataset, questions = self.prepare()

        self.assertEqual(config.schema_version, 1)
        self.assertEqual(config.pipeline, "no_rl")
        self.assertEqual(dataset, self.dataset)
        self.assertEqual(len(questions), 1)

    def test_modified_dataset_is_rejected(self):
        self.dataset.write_text(
            '{"query_id":"Q001","query":"Changed?"}\n',
            encoding="utf-8",
        )

        with self.assertRaisesRegex(
            ValueError,
            "SHA256 mismatch",
        ):
            self.prepare()

    def test_invalid_schema_version_is_rejected(self):
        self.config_data["schema_version"] = 2
        self.write_config()

        with self.assertRaises(ValidationError):
            self.prepare()

    def test_unknown_pipeline_is_rejected(self):
        self.config_data["pipeline"] = "unknown"
        self.write_config()

        with self.assertRaises(ValidationError):
            self.prepare()

    def test_path_escape_is_rejected(self):
        self.config_data["dataset_path"] = (
            "../outside.jsonl"
        )
        self.write_config()

        with self.assertRaisesRegex(
            ValueError,
            "escapes",
        ):
            self.prepare()

    def test_extra_config_fields_are_rejected(self):
        self.config_data["unexpected_field"] = True
        self.write_config()

        with self.assertRaises(ValidationError):
            self.prepare()

    def test_configured_mock_experiment(self):
        summary, output_dir = run_configured_experiment(
            config_path=self.config_path,
            output_root=self.root / "results",
            project_root=self.root,
            run_id="unit_test_run",
            pipeline_factory=lambda name: MockPipeline(),
        )

        self.assertEqual(
            summary["successful"], 1
        )
        self.assertEqual(
            summary["errors"], 0
        )

        saved_config = (
            output_dir / "experiment_config.json"
        )

        self.assertEqual(
            file_sha256(saved_config),
            file_sha256(self.config_path),
        )

        provenance = json.loads(
            (
                output_dir / "config_provenance.json"
            ).read_text(encoding="utf-8")
        )

        self.assertEqual(
            provenance["config_schema_version"],
            1,
        )

        records = (
            output_dir / "records.jsonl"
        ).read_text(encoding="utf-8").splitlines()

        self.assertEqual(len(records), 1)

        record = PipelineRecord.model_validate_json(
            records[0]
        )

        self.assertEqual(record.query_id, "Q001")


if __name__ == "__main__":
    unittest.main()
