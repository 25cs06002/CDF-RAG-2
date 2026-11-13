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
from .preflight import run_preflight

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

    pipeline = selected_factory(config.pipeline)

    if run_id is None:
        timestamp = datetime.now(
            timezone.utc
        ).strftime("%Y%m%dT%H%M%SZ")

        run_id = (
            f"{timestamp}_{uuid.uuid4().hex[:8]}"
        )

    summary, output_dir = run_experiment(
        pipeline=pipeline,
        questions=questions,
        run_id=run_id,
        source_file=dataset_path,
        output_root=Path(output_root),
        graph_version=config.graph_version,
        random_seed=config.random_seed,
    )

    # Preserve successful preflight evidence.
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

    # Preserve the exact bytes of the selected config.
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
        "graph_version_is_label_only": preflight_report is None,
        "graph_identity_checked_before_run": preflight_report is not None,
        "graph_locked_during_run": False,
        "runtime_preflight_file": "runtime_preflight.json" if preflight_report is not None else None,
        "random_seed_is_label_only": True,
        "purpose": config.purpose,
    }

    (
        output_dir / "config_provenance.json"
    ).write_text(
        json.dumps(
            provenance,
            indent=2,
        ),
        encoding="utf-8",
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


if __name__ == "__main__":
    main()
