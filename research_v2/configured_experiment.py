"""
Launch CDF-RAG experiments using versioned JSON configs.

Run:
python3 -m research_v2.configured_experiment \
    --config research_v2/configs/g0_no_rl_smoke_v1.json
"""

import argparse
import json
import shutil
import uuid

from datetime import datetime, timezone
from pathlib import Path

from .experiment_config import prepare_experiment
from .seeding import seed_libraries
from .preflight import run_preflight
from .graph_drift import GraphDriftMonitor

from .experiment_runner import (
    PROJECT_ROOT,
    DEFAULT_OUTPUT_ROOT,
    create_pipeline,
    file_sha256,
    run_experiment,
)


def run_configured_experiment(
    config_path: Path,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    run_id: str | None = None,
    project_root: Path = PROJECT_ROOT,
    pipeline_factory=None,
    preflight_checker=None,
    graph_checker=None,
):
    """
    Execute one versioned experiment.

    Allows a mock pipeline factory for unit tests.
    """

    config_path = Path(config_path).resolve()

    config, dataset_path, questions = (
        prepare_experiment(
            config_path,
            project_root=project_root,
        )
    )

    selected_factory = (
        pipeline_factory
        if pipeline_factory is not None
        else create_pipeline
    )


    # Real pipelines require fingerprint verification.
    # Mock pipelines used by unit tests remain isolated.
    if preflight_checker is not None:
        preflight_report = preflight_checker(
            config.pipeline,
            config.graph_version,
        )
    elif pipeline_factory is None:
        preflight_report = run_preflight(
            config.pipeline,
            config.graph_version,
        )
    else:
        preflight_report = None

    if (
        preflight_report is not None
        and preflight_report.get("status") != "pass"
    ):
        raise RuntimeError(
            "Experiment blocked by runtime preflight"
        )

    if graph_checker is not None:
        active_graph_checker = graph_checker

    elif pipeline_factory is None:
        # The live preflight has already verified
        # this fingerprint against the runtime lock.
        expected_hash = preflight_report[
            "checks"
        ]["live_graph_fingerprint"]["expected"]

        active_graph_checker = GraphDriftMonitor(
            expected_hash
        )

    else:
        # Isolated unit tests need no live Neo4j.
        active_graph_checker = None

    # Seed before constructing the model/pipeline.
    if config.random_seed is not None:
        seed_libraries(config.random_seed)

    pipeline = selected_factory(config.pipeline)

    if run_id is None:
        timestamp = datetime.now(
            timezone.utc
        ).strftime("%Y%m%dT%H%M%SZ")

        run_id = (
            f"{timestamp}_{uuid.uuid4().hex[:8]}"
        )

    def save_metadata_before_run(output_dir: Path):
        # This callback is called before the first query.

        if preflight_report is not None:
            (
                output_dir / "runtime_preflight.json"
            ).write_text(
                json.dumps(
                    preflight_report,
                    indent=2,
                ) + "\n",
                encoding="utf-8",
            )

        saved_config = (
            output_dir / "experiment_config.json"
        )

        shutil.copyfile(
            config_path,
            saved_config,
        )

        provenance = {
            "experiment_name": config.experiment_name,
            "config_schema_version": config.schema_version,
            "config_sha256": file_sha256(config_path),
            "dataset_sha256": config.dataset_sha256,
            "config_file": saved_config.name,
            "graph_version_is_label_only": (
                preflight_report is None
            ),
            "graph_identity_checked_before_run": (
                preflight_report is not None
            ),
            "graph_locked_during_run": False,
            "graph_checked_per_query": (
                active_graph_checker is not None
            ),
            "graph_checks_file": (
                "graph_checks.jsonl"
                if active_graph_checker is not None
                else None
            ),
            "runtime_preflight_file": (
                "runtime_preflight.json"
                if preflight_report is not None
                else None
            ),
            "random_seed_is_label_only": (
                config.random_seed is None
            ),
            "local_random_seed_applied": (
                config.random_seed is not None
            ),
            "local_seed_scope": [
                "python_random",
                "numpy_legacy_global_rng",
                "torch_global_rng",
            ],
            "seed_applied_before_pipeline_init": (
                config.random_seed is not None
            ),
            "seed_reapplied_before_each_query": (
                config.random_seed is not None
            ),
            "ollama_seed_enforced": False,
            "python_hash_seed_enforced": False,
            "torch_deterministic_algorithms_enforced": False,
            "seed_journal_file": (
                "seed_events.jsonl"
                if config.random_seed is not None
                else None
            ),
            "provenance_written_before_queries": True,
            "purpose": config.purpose,
        }

        (
            output_dir / "config_provenance.json"
        ).write_text(
            json.dumps(
                provenance,
                indent=2,
            ) + "\n",
            encoding="utf-8",
        )

    summary, output_dir = run_experiment(
        pipeline=pipeline,
        questions=questions,
        run_id=run_id,
        source_file=dataset_path,
        output_root=Path(output_root),
        graph_version=config.graph_version,
        random_seed=config.random_seed,
        on_run_ready=save_metadata_before_run,
        graph_checker=active_graph_checker,
    )

    return summary, output_dir


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--validate-only",
        action="store_true",
    )

    parser.add_argument(
        "--run-id",
        default=None,
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )

    args = parser.parse_args()

    if Path.cwd().resolve() != PROJECT_ROOT:
        parser.error(
            "Run from the CDF-RAG-2 project root"
        )

    config, dataset, questions = (
        prepare_experiment(args.config)
    )

    print("=" * 60)
    print("VERSIONED EXPERIMENT CONFIGURATION")
    print("=" * 60)

    print("Experiment:", config.experiment_name)
    print("Schema version:", config.schema_version)
    print("Pipeline:", config.pipeline)
    print("Dataset:", dataset)
    print("Questions selected:", len(questions))
    print("Dataset SHA256:", config.dataset_sha256)
    print("Graph label:", config.graph_version)
    print("Seed label:", config.random_seed)

    if args.validate_only:
        print("\nCONFIG VALIDATION PASSED")
        print("No model or database execution performed.")
        return

    summary, output_dir = run_configured_experiment(
        config_path=args.config,
        output_root=args.output_root,
        run_id=args.run_id,
    )

    print("\nEXPERIMENT COMPLETED")
    print(json.dumps(summary, indent=2))
    print("Results:", output_dir)

    if summary.get("aborted"):
        print(
            "EXPERIMENT INVALID: graph integrity check failed."
        )
        raise SystemExit(2)


if __name__ == "__main__":
    main()
