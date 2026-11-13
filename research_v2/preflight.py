"""
Fail-fast reproducibility verification for G0 experiments.

Checks graph identity, Ollama model digests, dependencies,
Python version and (for RL) the PPO checkpoint.
No inference or graph writes are performed.
"""

import argparse
import importlib.metadata
import json
import os
import sys

from datetime import datetime, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv

from .experiment_config import prepare_experiment
from .experiment_runner import PROJECT_ROOT, file_sha256
from .graph_identity import (
    compare_graphs,
    read_export,
    read_live_graph,
)


DEFAULT_LOCK = (
    PROJECT_ROOT
    / "research_v2/configs/g0_runtime_lock_v1.json"
)


class PreflightError(RuntimeError):
    pass


def collect_runtime(project_root=PROJECT_ROOT):
    root = Path(project_root).resolve()

    load_dotenv(
        dotenv_path=root / ".env",
        override=False,
    )

    graph_dir = root.parent / "CDF-RAG-2-graph-G0"

    archived = read_export(graph_dir)
    live = read_live_graph()

    comparison = compare_graphs(
        archived,
        live,
    )

    base_url = os.getenv(
        "OLLAMA_BASE_URL",
        "http://127.0.0.1:11434",
    ).rstrip("/")

    version_response = requests.get(
        base_url + "/api/version",
        timeout=15,
    )
    version_response.raise_for_status()

    tags_response = requests.get(
        base_url + "/api/tags",
        timeout=15,
    )
    tags_response.raise_for_status()

    models = {
        model["name"]: model.get("digest")
        for model in tags_response.json().get(
            "models", []
        )
        if model.get("name")
    }

    packages = {}

    package_names = [
        "torch",
        "stable-baselines3",
        "gymnasium",
        "langgraph",
        "neo4j",
        "pydantic",
        "sentence-transformers",
        "transformers",
        "numpy",
        "python-dotenv",
        "requests",
    ]

    for package in package_names:
        try:
            packages[package] = (
                importlib.metadata.version(package)
            )
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None

    checkpoint = (
        root / "ppo_query_refiner_2.zip"
    )

    return {
        "python_version": (
            sys.version.split()[0]
        ),
        "graph": comparison,
        "nodes_csv_sha256": file_sha256(
            graph_dir / "nodes.csv"
        ),
        "edges_csv_sha256": file_sha256(
            graph_dir / "edges.csv"
        ),
        "ollama_version": (
            version_response.json().get("version")
        ),
        "installed_models": models,
        "model_selection": {
            "generation": os.getenv(
                "OLLAMA_MODEL",
                "llama3.2:latest",
            ),
            "refinement": os.getenv(
                "QUERY_REFINEMENT_MODEL",
                "llama3.2:latest",
            ),
            "embedding": os.getenv(
                "ENCODER_MODEL",
                "nomic-embed-text:latest",
            ),
            "configured_embedding": os.getenv(
                "EMBEDDING_MODEL",
                "nomic-embed-text:latest",
            ),
        },
        "ppo_checkpoint_sha256": (
            file_sha256(checkpoint)
            if checkpoint.is_file()
            else None
        ),
        "package_versions": packages,
    }


def compare_runtime(
    lock,
    actual,
    pipeline,
    graph_version,
):
    """
    Pure comparison, also suitable for unit tests.
    """

    if pipeline not in {"rl", "no_rl"}:
        raise ValueError(
            "Unsupported pipeline"
        )

    checks = {}

    def check(name, expected, observed):
        checks[name] = {
            "expected": expected,
            "actual": observed,
            "passed": expected == observed,
        }

    check(
        "lock_schema_version",
        1,
        lock.get("schema_version"),
    )

    check(
        "graph_version",
        lock["graph_version"],
        graph_version,
    )

    check(
        "export_matches_live",
        True,
        actual["graph"]["match"],
    )

    check(
        "live_graph_fingerprint",
        lock["graph_structural_sha256"],
        actual["graph"]["live_structural_sha256"],
    )

    check(
        "archived_graph_fingerprint",
        lock["graph_structural_sha256"],
        actual["graph"]["export_structural_sha256"],
    )

    for name in [
        "nodes_csv_sha256",
        "edges_csv_sha256",
        "python_version",
        "ollama_version",
    ]:
        check(
            name,
            lock[name],
            actual[name],
        )

    for role in [
        "generation",
        "refinement",
        "embedding",
    ]:
        expected_model = lock["models"][role]

        check(
            f"{role}_model_name",
            expected_model["name"],
            actual["model_selection"][role],
        )

        check(
            f"{role}_model_digest",
            expected_model["digest"],
            actual["installed_models"].get(
                expected_model["name"]
            ),
        )

    check(
        "embedding_environment_consistency",
        actual["model_selection"]["embedding"],
        actual["model_selection"][
            "configured_embedding"
        ],
    )

    for package, version in (
        lock["package_versions"].items()
    ):
        check(
            f"package_{package}",
            version,
            actual["package_versions"].get(
                package
            ),
        )

    if pipeline == "rl":
        check(
            "ppo_checkpoint_sha256",
            lock["ppo_checkpoint_sha256"],
            actual["ppo_checkpoint_sha256"],
        )

    failures = [
        name
        for name, result in checks.items()
        if not result["passed"]
    ]

    return {
        "status": (
            "pass" if not failures else "fail"
        ),
        "pipeline": pipeline,
        "graph_version": graph_version,
        "checked_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "checks": checks,
        "failed_checks": failures,
        "ppo_checkpoint_checked": (
            pipeline == "rl"
        ),
        "graph_checked_before_run": True,
        "graph_locked_during_run": False,
        "random_seed_enforced": False,
        "sentence_encoder_weights_verified": False,
    }


def run_preflight(
    pipeline,
    graph_version,
    lock_path=DEFAULT_LOCK,
):
    lock_path = Path(lock_path)

    lock = json.loads(
        lock_path.read_text(
            encoding="utf-8"
        )
    )

    actual = collect_runtime()

    report = compare_runtime(
        lock,
        actual,
        pipeline,
        graph_version,
    )

    report["runtime_lock_sha256"] = (
        file_sha256(lock_path)
    )

    if report["status"] != "pass":
        raise PreflightError(
            "Runtime preflight failed: "
            + ", ".join(
                report["failed_checks"]
            )
        )

    return report


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--runtime-lock",
        type=Path,
        default=DEFAULT_LOCK,
    )

    args = parser.parse_args()

    config, _, _ = prepare_experiment(
        args.config
    )

    try:
        report = run_preflight(
            config.pipeline,
            config.graph_version,
            args.runtime_lock,
        )
    except PreflightError as exc:
        print("PREFLIGHT FAILED:", exc)
        raise SystemExit(1)

    passed = sum(
        item["passed"]
        for item in report["checks"].values()
    )

    print("=" * 60)
    print("CDF-RAG RUNTIME PREFLIGHT")
    print("=" * 60)
    print("Pipeline:", config.pipeline)
    print("Graph:", config.graph_version)
    print("Checks passed:", passed)
    print("Failed checks:", 0)
    print("PPO checkpoint checked:",
          report["ppo_checkpoint_checked"])
    print("PREFLIGHT PASSED")
    print("No experiment was executed.")


if __name__ == "__main__":
    main()
