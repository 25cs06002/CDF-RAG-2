"""Construct G0-only structural reference labels from the presentation pilot.

These are NOT independently reviewed medical gold labels. Their purpose is
to test recovery of edges already present in the fixed G0 snapshot.
Avoid treating graph-snapshot agreement as out-of-sample clinical accuracy.

Use:
  python3 -m research_v2.presentation_g0_reference prepare
  python3 -m research_v2.presentation_g0_reference score
"""
import argparse
import csv
import json
from pathlib import Path

from .evaluation import GoldAnnotation, evaluate, load_jsonl
from .experiment_runner import PROJECT_ROOT
from .g0_regression import build_g0_gold
from .presentation_matrix import (
    INPUT, MODEL_IDS, PILOT, WORKSPACE, EXPECTED_PIPELINE_IDS,
    validate_inputs,
)
from .schema import PipelineRecord

G0 = PROJECT_ROOT / "research_artifacts/graph-G0"
REFERENCE_DIR = WORKSPACE / "metrics/g0_structural_pilot_v1"
SELECTED_TYPES = frozenset({"direct", "paraphrased", "indirect", "risk_factor"})


def prepare():
    questions, metadata = validate_inputs()
    with (G0 / "nodes.csv").open(newline="", encoding="utf-8") as handle:
        nodes = {
            row["name"].casefold(): row["name"]
            for row in csv.DictReader(handle)
        }
    selection = [
        {"query_id": q.query_id,
         "target_concept": nodes[metadata[q.query_id]["target_concept"].casefold()]}
        for q in questions
        if metadata[q.query_id]["query_type"] in SELECTED_TYPES
        and metadata[q.query_id].get("target_concept", "").casefold() in nodes
    ]
    if not selection:
        raise ValueError("No G0-mapped query targets found")
    root = WORKSPACE / "benchmark"
    root.mkdir(parents=True, exist_ok=True)
    mapping = root / "g0_structural_mapping_v1.jsonl"
    if mapping.exists() or REFERENCE_DIR.exists():
        raise FileExistsError("Existing structural reference; preserve frozen inputs")
    mapping.write_text(
        "".join(json.dumps(row) + "\n" for row in selection),
        encoding="utf-8"
    )
    annotations, provenance = build_g0_gold(
        graph_dir=G0, mapping_file=mapping, output_dir=REFERENCE_DIR
    )
    detail = {
        "reference_basis": "graph_snapshot",
        "clinical_accuracy_verified": False,
        "source_of_target_concepts": "AI-authored pilot draft metadata",
        "task_scope": sorted(SELECTED_TYPES),
        "total_pilot_questions": len(questions),
        "structurally_mapped_subset": len(annotations),
        "not_mapped_or_excluded": len(questions) - len(annotations),
        "graph_snapshot": provenance,
    }
    (REFERENCE_DIR / "scope.json").write_text(
        json.dumps(detail, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(detail, indent=2))


def score():
    if not REFERENCE_DIR.is_dir():
        raise FileNotFoundError("Run 'prepare' first")
    gold = load_jsonl(REFERENCE_DIR / "g0_gold.jsonl", GoldAnnotation)
    by_id = {row.query_id: row for row in gold}
    journal = WORKSPACE / "matrix_runs.jsonl"
    if not journal.is_file():
        raise FileNotFoundError("Run presentation_matrix first")
    chosen = {}
    for line in journal.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        run = json.loads(line)
        if run["query_count"] == 100 and not run["summary"].get("aborted"):
            chosen[run["model_id"]] = run
    results = []
    for model_id in MODEL_IDS:
        if model_id not in chosen:
            continue
        run = chosen[model_id]
        file = PROJECT_ROOT / run["run_dir"] / "records.jsonl"
        records = [
            PipelineRecord.model_validate_json(line)
            for line in file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if len(records) != 100 or {r.pipeline_id for r in records} != {EXPECTED_PIPELINE_IDS[model_id]}:
            raise ValueError(f"Wrong records for {model_id}")
        subset = [r for r in records if r.query_id in by_id]
        if {r.query_id for r in subset} != set(by_id):
            raise ValueError(f"Missing mapped query IDs for {model_id}")
        summary, per_query = evaluate(subset, gold)
        edge_metrics = summary["metrics"]["edges"]
        results.append({
            "model": model_id,
            "pipeline_id": summary["pipeline_id"],
            "run_id": run["run_id"],
            "mapped_queries": len(subset),
            "edge_macro_f1": edge_metrics["macro_f1"],
            "edge_micro_f1": edge_metrics["micro_f1"],
            "edge_micro_precision": edge_metrics["micro_precision"],
            "edge_micro_recall": edge_metrics["micro_recall"],
            "successful_latency_seconds": summary["successful_latency_seconds"],
            "reference_basis": "graph_snapshot",
            "medical_gold": False,
        })
    if not results:
        raise ValueError("No completed 100-query runs available to score")
    output = REFERENCE_DIR / "structural_scores.json"
    if output.exists():
        raise FileExistsError(f"Already scored: {output}")
    output.write_text(
        json.dumps({
            "mapped_questions": len(by_id),
            "models_scored": len(results),
            "warning": "Measures G0 structural recovery only; unreviewed pilot metadata and graph relations are not medical gold.",
            "results": results,
        }, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(output), "models_scored": len(results)}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "score"])
    args = parser.parse_args()
    if Path.cwd().resolve() != PROJECT_ROOT:
        parser.error("Run from CDF-RAG-2 project root")
    {"prepare": prepare, "score": score}[args.command]()


if __name__ == "__main__":
    main()
