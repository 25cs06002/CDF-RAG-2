import json
import random
import tempfile
import unittest

from pathlib import Path

import numpy as np
import torch

from pydantic import ValidationError

from research_v2.base_pipeline import BaseRAGPipeline
from research_v2.configured_experiment import (
    run_configured_experiment,
)
from research_v2.experiment_config import ExperimentConfig
from research_v2.experiment_runner import (
    BenchmarkQuestion,
    file_sha256,
    run_experiment,
)
from research_v2.schema import PipelineRecord
from research_v2.seeding import (
    seed_libraries,
    validate_seed,
)


class RandomProbePipeline(BaseRAGPipeline):

    def __init__(self):
        super().__init__("rng_probe")
        self.initialization_value = random.random()

    def _run(self, request):
        values = [
            random.random(),
            float(np.random.random()),
            float(torch.rand(1).item()),
        ]

        return PipelineRecord(
            run_id=request.run_id,
            query_id=request.query_id,
            pipeline_id=self.pipeline_id,
            original_query=request.query,
            final_answer=json.dumps(values),
            random_seed=request.random_seed,
        )


class TestSeeding(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

        self.root = Path(self.tmp.name)

        self.dataset = self.root / "questions.jsonl"

        self.dataset.write_text(
            '{"query_id":"Q1","query":"One?"}\n'
            '{"query_id":"Q2","query":"Two?"}\n',
            encoding="utf-8",
        )

        self.questions = [
            BenchmarkQuestion(
                query_id="Q1",
                query="One?",
            ),
            BenchmarkQuestion(
                query_id="Q2",
                query="Two?",
            ),
        ]

    def test_same_seed_reproduces_all_three_rngs(self):
        def sample():
            return (
                random.random(),
                float(np.random.random()),
                float(torch.rand(1).item()),
            )

        seed_libraries(2026)
        first = sample()

        seed_libraries(2026)
        second = sample()

        self.assertEqual(first, second)

    def test_invalid_seed_rejected(self):
        for invalid in (
            -1,
            2**32,
            True,
            "2026",
            1.5,
        ):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    validate_seed(invalid)

    def test_runner_reseeds_every_query(self):
        pipeline = RandomProbePipeline()

        _, folder = run_experiment(
            pipeline=pipeline,
            questions=self.questions,
            run_id="seeded_run",
            source_file=self.dataset,
            output_root=self.root / "runs",
            random_seed=2026,
        )

        records = [
            PipelineRecord.model_validate_json(line)
            for line in (
                folder / "records.jsonl"
            ).read_text(
                encoding="utf-8"
            ).splitlines()
        ]

        self.assertEqual(len(records), 2)

        self.assertEqual(
            records[0].final_answer,
            records[1].final_answer,
        )

        self.assertEqual(
            records[0].random_seed,
            2026,
        )

        events = [
            json.loads(line)
            for line in (
                folder / "seed_events.jsonl"
            ).read_text(
                encoding="utf-8"
            ).splitlines()
        ]

        self.assertEqual(len(events), 2)
        self.assertEqual(
            [event["seed"] for event in events],
            [2026, 2026],
        )
        self.assertTrue(
            all(
                event["torch_global_rng"]
                for event in events
            )
        )

    def test_seed_applied_before_pipeline_factory(self):
        config_path = self.root / "config.json"

        config_path.write_text(
            json.dumps({
                "schema_version": 1,
                "experiment_name": "seeded_factory_test",
                "pipeline": "no_rl",
                "dataset_path": "questions.jsonl",
                "dataset_sha256": file_sha256(
                    self.dataset
                ),
                "graph_version": "G0",
                "random_seed": 2026,
                "limit": 1,
            }),
            encoding="utf-8",
        )

        constructed = []

        def factory(name):
            pipeline = RandomProbePipeline()
            constructed.append(
                pipeline.initialization_value
            )
            return pipeline

        outputs = []

        for run_id in ("run_a", "run_b"):
            summary, folder = run_configured_experiment(
                config_path=config_path,
                output_root=self.root / "configured",
                project_root=self.root,
                run_id=run_id,
                pipeline_factory=factory,
                preflight_checker=lambda p, g: {
                    "status": "pass"
                },
            )

            self.assertEqual(summary["successful"], 1)

            records = (
                folder / "records.jsonl"
            ).read_text(
                encoding="utf-8"
            ).splitlines()

            outputs.append(
                PipelineRecord.model_validate_json(
                    records[0]
                ).final_answer
            )

            provenance = json.loads(
                (
                    folder / "config_provenance.json"
                ).read_text(encoding="utf-8")
            )

            self.assertTrue(
                provenance[
                    "seed_applied_before_pipeline_init"
                ]
            )
            self.assertFalse(
                provenance["ollama_seed_enforced"]
            )

        self.assertEqual(
            constructed[0],
            constructed[1],
        )
        self.assertEqual(
            outputs[0],
            outputs[1],
        )

    def test_seed_config_bounds(self):
        data = {
            "schema_version": 1,
            "experiment_name": "seed_validation",
            "pipeline": "no_rl",
            "dataset_path": "questions.jsonl",
            "dataset_sha256": "a" * 64,
            "graph_version": "G0",
            "random_seed": 2**32,
        }

        with self.assertRaises(ValidationError):
            ExperimentConfig.model_validate(data)

    def test_unseeded_run_has_no_seed_journal(self):
        _, folder = run_experiment(
            pipeline=RandomProbePipeline(),
            questions=self.questions[:1],
            run_id="unseeded_run",
            source_file=self.dataset,
            output_root=self.root / "runs",
            random_seed=None,
        )

        self.assertFalse(
            (folder / "seed_events.jsonl").exists()
        )


if __name__ == "__main__":
    unittest.main()
