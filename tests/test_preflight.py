import copy
import json
import tempfile
import unittest

from pathlib import Path

from research_v2.base_pipeline import BaseRAGPipeline
from research_v2.configured_experiment import (
    run_configured_experiment,
)
from research_v2.experiment_runner import file_sha256
from research_v2.preflight import (
    DEFAULT_LOCK,
    compare_runtime,
)
from research_v2.schema import PipelineRecord


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
            final_answer="Synthetic answer",
        )


class TestRuntimePreflight(unittest.TestCase):

    def setUp(self):
        self.lock = json.loads(
            DEFAULT_LOCK.read_text(
                encoding="utf-8"
            )
        )

        graph_hash = (
            self.lock["graph_structural_sha256"]
        )

        self.actual = {
            "python_version": (
                self.lock["python_version"]
            ),
            "graph": {
                "match": True,
                "live_structural_sha256": graph_hash,
                "export_structural_sha256": graph_hash,
            },
            "nodes_csv_sha256": (
                self.lock["nodes_csv_sha256"]
            ),
            "edges_csv_sha256": (
                self.lock["edges_csv_sha256"]
            ),
            "ollama_version": (
                self.lock["ollama_version"]
            ),
            "model_selection": {
                "generation": (
                    self.lock["models"]["generation"]["name"]
                ),
                "refinement": (
                    self.lock["models"]["refinement"]["name"]
                ),
                "embedding": (
                    self.lock["models"]["embedding"]["name"]
                ),
                "configured_embedding": (
                    self.lock["models"]["embedding"]["name"]
                ),
            },
            "installed_models": {
                item["name"]: item["digest"]
                for item in self.lock["models"].values()
            },
            "package_versions": copy.deepcopy(
                self.lock["package_versions"]
            ),
            "ppo_checkpoint_sha256": (
                self.lock["ppo_checkpoint_sha256"]
            ),
        }

    def check(self, pipeline="no_rl"):
        return compare_runtime(
            self.lock,
            self.actual,
            pipeline,
            "G0",
        )

    def test_valid_no_rl_environment(self):
        report = self.check()

        self.assertEqual(report["status"], "pass")
        self.assertEqual(report["failed_checks"], [])
        self.assertFalse(
            report["ppo_checkpoint_checked"]
        )

    def test_valid_rl_environment(self):
        report = self.check("rl")

        self.assertEqual(report["status"], "pass")
        self.assertTrue(
            report["ppo_checkpoint_checked"]
        )

    def test_changed_graph_is_detected(self):
        self.actual["graph"][
            "live_structural_sha256"
        ] = "different"

        report = self.check()

        self.assertEqual(report["status"], "fail")
        self.assertIn(
            "live_graph_fingerprint",
            report["failed_checks"],
        )

    def test_changed_model_digest_is_detected(self):
        model_name = (
            self.lock["models"]["generation"]["name"]
        )

        self.actual["installed_models"][
            model_name
        ] = "different"

        report = self.check()

        self.assertEqual(report["status"], "fail")
        self.assertIn(
            "generation_model_digest",
            report["failed_checks"],
        )

    def test_embedding_mismatch_is_detected(self):
        self.actual["model_selection"][
            "configured_embedding"
        ] = "another-model"

        report = self.check()

        self.assertEqual(report["status"], "fail")
        self.assertIn(
            "embedding_environment_consistency",
            report["failed_checks"],
        )

    def test_ppo_checkpoint_required_for_rl(self):
        self.actual[
            "ppo_checkpoint_sha256"
        ] = None

        self.assertEqual(
            self.check("no_rl")["status"],
            "pass",
        )

        self.assertEqual(
            self.check("rl")["status"],
            "fail",
        )

    def test_launcher_saves_preflight_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            dataset = root / "questions.jsonl"
            dataset.write_text(
                '{"query_id":"Q1","query":"Test?"}\n',
                encoding="utf-8",
            )

            config_path = root / "config.json"
            config_path.write_text(
                json.dumps({
                    "schema_version": 1,
                    "experiment_name": "preflight_test",
                    "pipeline": "no_rl",
                    "dataset_path": "questions.jsonl",
                    "dataset_sha256": (
                        file_sha256(dataset)
                    ),
                    "graph_version": "G0",
                    "limit": 1,
                }),
                encoding="utf-8",
            )

            def verified(pipeline, graph_version):
                return self.check(pipeline)

            summary, folder = (
                run_configured_experiment(
                    config_path=config_path,
                    output_root=root / "runs",
                    run_id="preflight_test",
                    project_root=root,
                    pipeline_factory=(
                        lambda name: MockPipeline()
                    ),
                    preflight_checker=verified,
                )
            )

            self.assertEqual(
                summary["successful"], 1
            )

            report = json.loads(
                (
                    folder / "runtime_preflight.json"
                ).read_text(encoding="utf-8")
            )

            self.assertEqual(
                report["status"], "pass"
            )

    def test_failed_preflight_blocks_execution(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            dataset = root / "questions.jsonl"
            dataset.write_text(
                '{"query_id":"Q1","query":"Test?"}\n',
                encoding="utf-8",
            )

            config_path = root / "config.json"
            config_path.write_text(
                json.dumps({
                    "schema_version": 1,
                    "experiment_name": "blocked_test",
                    "pipeline": "no_rl",
                    "dataset_path": "questions.jsonl",
                    "dataset_sha256": (
                        file_sha256(dataset)
                    ),
                    "graph_version": "G0",
                }),
                encoding="utf-8",
            )

            calls = []

            def factory(name):
                calls.append(name)
                return MockPipeline()

            with self.assertRaisesRegex(
                RuntimeError,
                "preflight",
            ):
                run_configured_experiment(
                    config_path=config_path,
                    output_root=root / "runs",
                    run_id="blocked_test",
                    project_root=root,
                    pipeline_factory=factory,
                    preflight_checker=(
                        lambda p, g: {
                            "status": "fail"
                        }
                    ),
                )

            self.assertEqual(calls, [])
            self.assertFalse(
                (root / "runs").exists()
            )


if __name__ == "__main__":
    unittest.main()
