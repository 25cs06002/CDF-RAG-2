"""Create a versioned, local presentation workspace without touching legacy experiments.

Usage:
  python3 -m research_v2.presentation_sprint prepare
  python3 -m research_v2.presentation_sprint status

Every generated path remains inside the CDF-RAG-2 repository.
"""
import argparse
import csv
import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from .pilot_preparation import validate_pilot
from .experiment_runner import load_questions

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WORKSPACE = ROOT / "presentation_20261118"
PILOT = ROOT / "data/benchmark/medical_causal_pilot100_draft_v1.jsonl"
G0 = ROOT / "research_artifacts/graph-G0"
PIPELINES = {
    "B1": "naive_rag",
    "B2": "bm25_rag",
    "B3": "dense_rag",
    "B4": "hybrid_rag",
    "B5": "graph_rag",
    "B6": "causal_kg_rag",
    "B7": "no_rl",
    "B8": "rl",
}
LEGACY_RESULT_NAMES = (
    "evaluation_summary_no_rl_deterministic.csv",
    "evaluation_summary_rl_deterministic.csv",
    "rl_vs_no_rl_comparison.csv",
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def workspace_path(value):
    path = Path(value).expanduser().resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError("Presentation workspace must stay inside CDF-RAG-2")
    if path == ROOT:
        raise ValueError("Presentation workspace must be a subdirectory")
    return path


def prepare(workspace=DEFAULT_WORKSPACE):
    workspace = workspace_path(workspace)
    # Validate all input data before creating any output.
    records, profile = validate_pilot(PILOT)
    node_path, edge_path = G0 / "nodes.csv", G0 / "edges.csv"
    counts = {}
    for label, path in (("nodes", node_path), ("edges", edge_path)):
        with path.open(newline="", encoding="utf-8") as handle:
            counts[label] = sum(1 for _ in csv.DictReader(handle))
    if counts != {"nodes": 35, "edges": 37}:
        raise ValueError(f"G0 snapshot differs from documented baseline: {counts}")

    benchmark = workspace / "benchmark"
    logs = workspace / "logs"
    legacy = workspace / "legacy_results"
    for directory in (benchmark, logs, legacy, workspace / "runs", workspace / "metrics", workspace / "figures"):
        directory.mkdir(parents=True, exist_ok=True)

    questions = benchmark / "questions_only.jsonl"
    # Never silently replace a previously frozen experimental input.
    if questions.exists():
        raise FileExistsError(f"Already prepared: {questions}")

    questions.write_text(
        "".join(json.dumps({"query_id": r.query_id, "query": r.query}, ensure_ascii=False) + "\n" for r in records),
        encoding="utf-8",
    )
    if len(load_questions(questions)) != 100:
        raise AssertionError("Expected exactly 100 exported questions")

    copied = {}
    for name in LEGACY_RESULT_NAMES:
        original = ROOT / name
        if original.is_file():
            target = legacy / name
            if target.exists():
                raise FileExistsError(f"Frozen result already exists: {target}")
            shutil.copy2(original, target)
            copied[name] = sha256(target)

    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT,
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "project_commit": commit,
        "dataset_role": "unreviewed_pilot_for_behavioral_and_runtime_experiments",
        "medical_gold_verified": False,
        "questions": 100,
        "query_distribution": profile["query_type_counts"],
        "question_family_pair_count": profile["question_family_pair_count"],
        "dataset_sha256": sha256(PILOT),
        "inference_input_sha256": sha256(questions),
        "graph_version": "G0",
        "graph_counts": counts,
        "graph_nodes_sha256": sha256(node_path),
        "graph_edges_sha256": sha256(edge_path),
        "legacy_result_sha256": copied,
        "planned_pipelines": PIPELINES,
        "evaluation_warning": (
            "The full pilot has no independently adjudicated medical gold labels. "
            "G0 structural scores are not medical accuracy scores. "
            "G0 has limited domain coverage."
        ),
    }
    (benchmark / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "status": "PREPARED", "workspace": str(workspace),
        "questions": 100, "graph": counts,
        "legacy_results_copied": sorted(copied),
        "manifest": str(benchmark / "manifest.json"),
    }, indent=2))


def status(workspace=DEFAULT_WORKSPACE):
    workspace = workspace_path(workspace)
    manifest = workspace / "benchmark/manifest.json"
    if not manifest.is_file():
        raise FileNotFoundError("Run 'prepare' first")
    metadata = json.loads(manifest.read_text(encoding="utf-8"))
    question_file = workspace / "benchmark/questions_only.jsonl"
    if sha256(question_file) != metadata["inference_input_sha256"]:
        raise RuntimeError("Presentation input checksum mismatch")
    runs = []
    for summary in sorted((workspace / "runs").glob("*/summary.json")):
        try:
            entry = json.loads(summary.read_text(encoding="utf-8"))
            runs.append({
                "run": summary.parent.name,
                "pipeline": entry.get("pipeline_id"),
                "total": entry.get("total_queries"),
                "success": entry.get("successful"),
                "errors": entry.get("errors"),
                "aborted": entry.get("aborted"),
            })
        except (ValueError, OSError) as exc:
            runs.append({"run": summary.parent.name, "read_error": str(exc)})
    print(json.dumps({
        "workspace": str(workspace),
        "questions": metadata["questions"],
        "planned_pipelines": metadata["planned_pipelines"],
        "completed_run_summaries": runs,
        "note": "No model execution is performed by this status command.",
    }, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "status"])
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.workspace)
    else:
        status(args.workspace)


if __name__ == "__main__":
    main()
