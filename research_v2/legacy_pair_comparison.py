"""Offline, provenance-audited B7 vs B8 100-query paired comparison.

Uses EXISTING local saved runs. No Neo4j, Ollama, or PPO execution.
Do not call structural G0 scores 'clinical accuracy'. An observational paired
comparison is not a randomized demonstration of RL causal benefit.

Usage:
    python -m research_v2.legacy_pair_comparison
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from .evaluation import GoldAnnotation, evaluate, load_jsonl
from .experiment_runner import PROJECT_ROOT, file_sha256
from .presentation_export import _select_runs
from .presentation_matrix import INPUT, WORKSPACE, validate_inputs
from .presentation_g0_reference import REFERENCE_DIR
from .schema import PipelineRecord

SHARED_CHECKS_EXCEPTIONS = frozenset({"ppo_checkpoint_sha256"})
EXPECTED_PIPELINES = {"B7": "cdf_rag_no_rl_legacy", "B8": "cdf_rag_rl_legacy"}


def audit_run(run_dir: Path, records: list[PipelineRecord], *, expected_sha: str) -> tuple[dict, list[str]]:
    """Check stored run identity, successful preflight and each graph check."""
    failures = []
    data = {}
    try:
        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        config = json.loads((run_dir / "experiment_config.json").read_text(encoding="utf-8"))
        preflight = json.loads((run_dir / "runtime_preflight.json").read_text(encoding="utf-8"))
        checks_file = run_dir / "graph_checks.jsonl"
        graph_checks = [
            json.loads(line) for line in checks_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, ValueError, KeyError) as exc:
        return {}, [f"missing_or_invalid_run_provenance:{type(exc).__name__}"]

    data = {"manifest": manifest, "summary": summary, "config": config,
            "preflight": preflight}
    if manifest.get("input_sha256") != expected_sha:
        failures.append("dataset_hash_mismatch")
    if manifest.get("question_count") != 100 or len(records) != 100:
        failures.append("not_exactly_100_queries")
    if manifest.get("graph_version_label") != "G0" or config.get("graph_version") != "G0":
        failures.append("not_frozen_G0")
    if config.get("dataset_sha256") != expected_sha or config.get("limit") != 100:
        failures.append("config_dataset_mismatch")
    if config.get("random_seed") != 2026:
        failures.append("seed_label_mismatch")
    if (summary.get("total_queries") != 100 or summary.get("errors") != 0
            or summary.get("aborted") or summary.get("successful") != 100):
        failures.append("run_has_errors_or_missing_queries")
    if {record.status for record in records} != {"ok"}:
        failures.append("per_query_status_not_all_ok")
    if preflight.get("status") != "pass" or preflight.get("failed_checks"):
        failures.append("preflight_not_passed")
    expected_checks = {
        "live_graph_fingerprint", "archived_graph_fingerprint",
        "export_matches_live", "nodes_csv_sha256", "edges_csv_sha256",
        "generation_model_name", "generation_model_digest",
        "refinement_model_name", "refinement_model_digest",
        "embedding_model_name", "embedding_model_digest",
    }
    checks = preflight.get("checks", {})
    if not expected_checks.issubset(checks):
        failures.append("missing_required_preflight_checks")
    if any(value.get("passed") is not True for value in checks.values()):
        failures.append("one_or_more_preflight_checks_failed")
    graph_hash = checks.get("live_graph_fingerprint", {}).get("expected")
    if not graph_hash or graph_hash != checks.get("live_graph_fingerprint", {}).get("actual"):
        failures.append("missing_or_invalid_graph_fingerprint")
    lock_sha = preflight.get("runtime_lock_sha256")
    if not isinstance(lock_sha, str) or len(lock_sha) != 64:
        failures.append("missing_runtime_lock_digest")
    entries = Counter((item.get("query_index"), item.get("phase")) for item in graph_checks)
    if (len(graph_checks) != 200 or set(entries.values()) != {1}
            or set(entries) != {(i, phase) for i in range(1, 101)
                                for phase in ("before", "after")}):
        failures.append("incomplete_graph_drift_journal")
    if any(item.get("status") != "pass"
           or item.get("expected_sha256") != graph_hash
           or item.get("observed_sha256") != graph_hash for item in graph_checks):
        failures.append("graph_drift_or_failed_graph_check")

    data["graph_check_count"] = len(graph_checks)
    data["runtime_lock_sha256"] = lock_sha
    data["graph_fingerprint"] = graph_hash
    return data, failures


def pair_audit(b7: dict, b8: dict) -> list[str]:
    issues = []
    if not b7 or not b8:
        return ["unavailable_provenance"]
    for field in ("git_commit", "input_sha256", "graph_version_label"):
        if (not b7["manifest"].get(field)
                or b7["manifest"].get(field) != b8["manifest"].get(field)):
            issues.append("mismatched_" + field)
    for field in ("dataset_sha256", "graph_version", "random_seed", "limit"):
        if (b7["config"].get(field) is None
                or b7["config"].get(field) != b8["config"].get(field)):
            issues.append("mismatched_config_" + field)
    if b7["config"].get("pipeline") != "no_rl" or b8["config"].get("pipeline") != "rl":
        issues.append("incorrect_pipeline_configs")
    if not b7.get("graph_fingerprint") or b7["graph_fingerprint"] != b8.get("graph_fingerprint"):
        issues.append("mismatched_graph_fingerprint")
    if not b7.get("runtime_lock_sha256") or b7["runtime_lock_sha256"] != b8.get("runtime_lock_sha256"):
        issues.append("mismatched_runtime_lock_sha")
    left = {k: v for k, v in b7["preflight"].get("checks", {}).items()
            if k not in SHARED_CHECKS_EXCEPTIONS}
    right = {k: v for k, v in b8["preflight"].get("checks", {}).items()
             if k not in SHARED_CHECKS_EXCEPTIONS}
    if left != right:
        issues.append("mismatched_shared_runtime_checks")
    if b8["preflight"].get("ppo_checkpoint_checked") is not True:
        issues.append("rl_checkpoint_not_checked")
    return issues


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    position = fraction * (len(values) - 1)
    low = int(position)
    return values[low] * (1 - (position - low)) + values[
        min(low + 1, len(values) - 1)
    ] * (position - low)


def has_evidence(record: PipelineRecord) -> bool:
    return bool(record.retrieved_documents or record.causal_edges or record.causal_paths)


def paired_rows(left: list[PipelineRecord], right: list[PipelineRecord],
                query_types: dict[str, str]) -> tuple[list[dict], dict]:
    if len(left) != 100 or len(right) != 100:
        raise ValueError("Both runs must contain exactly 100 queries")
    if [r.query_id for r in left] != [r.query_id for r in right]:
        raise ValueError("B7/B8 query order/IDs mismatch")
    if any(r.status != "ok" for r in left + right):
        raise ValueError("B7/B8 includes failed or abstained queries")
    outcomes, deltas, rows = Counter(), [], []
    for a, b in zip(left, right):
        l, r = has_evidence(a), has_evidence(b)
        label = ("both" if l and r else "B7_only" if l else
                 "B8_only" if r else "neither")
        outcomes[label] += 1
        a_ms, b_ms = a.timings.total_ms, b.timings.total_ms
        delta = b_ms - a_ms if a_ms is not None and b_ms is not None else None
        if delta is not None:
            deltas.append(delta)
        rows.append({
            "query_id": a.query_id,
            "query_type": query_types[a.query_id],
            "B7_status": a.status, "B8_status": b.status,
            "B7_evidence": l, "B8_evidence": r, "evidence_outcome": label,
            "B7_edges": len(a.causal_edges), "B8_edges": len(b.causal_edges),
            "B7_paths": len(a.causal_paths), "B8_paths": len(b.causal_paths),
            "B7_answer_present": bool(a.final_answer and a.final_answer.strip()),
            "B8_answer_present": bool(b.final_answer and b.final_answer.strip()),
            "B7_latency_ms": a_ms, "B8_latency_ms": b_ms,
            "B8_minus_B7_ms": delta,
        })
    summary = {
        "paired_queries": len(rows),
        "evidence_outcomes": dict(outcomes),
        "B7_evidence_yield": sum(has_evidence(r) for r in left),
        "B8_evidence_yield": sum(has_evidence(r) for r in right),
        "B7_causal_paths_total": sum(len(r.causal_paths) for r in left),
        "B8_causal_paths_total": sum(len(r.causal_paths) for r in right),
        "B7_answers_present": sum(bool(r.final_answer and r.final_answer.strip()) for r in left),
        "B8_answers_present": sum(bool(r.final_answer and r.final_answer.strip()) for r in right),
        "B7_mean_total_latency_ms": statistics.mean(r.timings.total_ms for r in left if r.timings.total_ms is not None),
        "B8_mean_total_latency_ms": statistics.mean(r.timings.total_ms for r in right if r.timings.total_ms is not None),
        "B8_minus_B7_mean_paired_latency_ms": statistics.mean(deltas) if deltas else None,
        "B8_minus_B7_p50_paired_latency_ms": percentile(deltas, 0.5),
        "B8_minus_B7_p95_paired_latency_ms": percentile(deltas, 0.95),
        "query_types": {
            category: {
                "total": sum(row["query_type"] == category for row in rows),
                "B7_evidence": sum(row["query_type"] == category and row["B7_evidence"] for row in rows),
                "B8_evidence": sum(row["query_type"] == category and row["B8_evidence"] for row in rows),
            }
            for category in sorted(set(query_types.values()))
        },
    }
    return rows, summary


def g0_mapped_structural(left: list[PipelineRecord], right: list[PipelineRecord]) -> dict:
    gold_path = REFERENCE_DIR / "g0_gold.jsonl"
    if not gold_path.is_file():
        return {"status": "not_available", "reason": "G0 snapshot reference missing"}
    gold = load_jsonl(gold_path, GoldAnnotation)
    mapped = {g.query_id for g in gold}
    result = {"status": "G0_snapshot_only_not_clinical_gold", "mapped_questions": len(gold)}
    for label, records in (("B7", left), ("B8", right)):
        subset = [r for r in records if r.query_id in mapped]
        metrics, _ = evaluate(subset, gold)
        result[label] = metrics["metrics"]["edges"]
    return result


def generate(workspace: Path = WORKSPACE, output: Path | None = None) -> tuple[Path, dict]:
    workspace = Path(workspace).expanduser().resolve()
    if not workspace.is_relative_to(PROJECT_ROOT):
        raise ValueError("Workspace must remain within repository")
    questions, metadata = validate_inputs(workspace / "benchmark/questions_only.jsonl")
    ids = [q.query_id for q in questions]
    input_sha = file_sha256(workspace / "benchmark/questions_only.jsonl")
    selected = _select_runs(workspace / "runs", ids, input_sha)
    if not all(key in selected for key in EXPECTED_PIPELINES):
        raise ValueError("B7 and B8 saved 100-query runs missing; do not infer they ran")
    runs = {}
    audit = {}
    checks = []
    for label, pipeline in EXPECTED_PIPELINES.items():
        bundle = selected[label]
        run_dir = bundle[2].parent
        records = bundle[3]
        if {r.pipeline_id for r in records} != {pipeline}:
            raise ValueError(f"{label} saved run has unexpected pipeline identity")
        run_info, failures = audit_run(run_dir, records, expected_sha=input_sha)
        runs[label] = {"dir": run_dir, "records": records,
                       "run_id": records[0].run_id}
        audit[label] = {
            "run_id": records[0].run_id, "failures": failures,
            "graph_check_count": run_info.get("graph_check_count"),
            "graph_fingerprint": run_info.get("graph_fingerprint"),
            "runtime_lock_sha256": run_info.get("runtime_lock_sha256"),
        }
        checks.extend(f"{label}:{e}" for e in failures)
        runs[label]["audit"] = run_info
    checks.extend(pair_audit(runs["B7"]["audit"], runs["B8"]["audit"]))
    valid = not checks
    if output is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        output = workspace / "metrics" / f"legacy_pair_{stamp}_{uuid.uuid4().hex[:6]}"
    output = Path(output).expanduser().resolve()
    if not output.is_relative_to(PROJECT_ROOT):
        raise ValueError("Comparison output must stay in repository")
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "purpose": "Existing B7 vs B8 100-query observation, no model rerun",
        "provenance_verified": valid,
        "comparison_reasons_if_invalid": checks,
        "run_audit": audit,
        "query_count": 100,
        "dataset_sha256": input_sha,
        "medical_gold_verified": False,
        "clinical_accuracy_measured": False,
        "B7_B8_retrieval_method_and_RL_differ": True,
        "warnings": [
            "Only observational same-query operational comparison; not a controlled causal estimate of PPO benefit.",
            "Original legacy generator/Neo4j/PPO architecture differs from the B1-B6 retrieval-only top-k sweep.",
            "G0 structural edge recovery on mapped queries is not clinical correctness.",
            "Do not disclose raw generated answers or patient-specific details without manual review.",
        ],
    }
    if valid:
        paired, metrics = paired_rows(
            runs["B7"]["records"], runs["B8"]["records"],
            {qid: metadata[qid]["query_type"] for qid in ids}
        )
        with (output / "paired_by_query.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(paired[0]))
            writer.writeheader()
            writer.writerows(paired)
        report["paired_operational_metrics"] = metrics
        report["g0_mapped_structural_metrics"] = g0_mapped_structural(
            runs["B7"]["records"], runs["B8"]["records"]
        )
    (output / "comparison.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(output), "provenance_verified": valid,
        "failure_reasons": checks,
        "paired_queries": report.get("paired_operational_metrics", {}).get("paired_queries", 0),
        "B7_run": audit["B7"]["run_id"], "B8_run": audit["B8"]["run_id"],
    }, indent=2))
    return output, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=WORKSPACE)
    args = parser.parse_args()
    if Path.cwd().resolve() != PROJECT_ROOT:
        parser.error("Run from repository root")
    generate(args.workspace)


if __name__ == "__main__":
    main()
