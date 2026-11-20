"""Bridge the historical 22-query metric names to full 132-query run records.

This exports measured operational quantities and explicit unavailable values,
rather than inventing legacy semantic scores or medical gold. Comparisons
between different evidence bases must be reported as descriptive only.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .experiment_runner import file_sha256
from .schema import PipelineRecord
from .thesis132 import _inside, _write_json

LEGACY_METRICS = (
    "Causal Retrieval Coverage",
    "Causal Chain Depth",
    "Semantic Refinement Score",
    "Context Relevance",
    "Groundedness",
    "Average Answer Length",
    "Hallucination Rate",
    "Retrieval Success Rate",
    "Average Direct Causes",
    "Total Direct Causes",
    "Total Execution Time (s)",
    "Average Latency (s)",
    "Average Latency (ms)",
    "Median Latency (ms)",
    "Average Throughput (QPS)",
    "Refinement Success Rate",
)
NOT_COMPARABLE = "Operational definition differs from the 22-question legacy evaluation."
NEEDS_ADJUDICATION = "Unavailable without separately reviewed answer/evidence labels."
NEEDS_REFINEMENT_EVAL = "Unavailable without an independent refinement-quality evaluator."


def _mean(items):
    return statistics.mean(items) if items else None


def _value_metrics(records: list[PipelineRecord], wall_seconds: float | None) -> dict:
    if len(records) != 132 or len({r.query_id for r in records}) != 132:
        raise ValueError("Metrics bridge accepts exactly 132 unique records per model")
    n = len(records)
    paths = [max(0, len(p.nodes) - 1) for r in records for p in r.causal_paths]
    query_ms = [r.timings.total_ms for r in records if r.timings.total_ms is not None]
    causal_edges = [sum(e.relation == "CAUSES" for e in r.causal_edges) for r in records]
    words = [len((r.final_answer or "").split()) for r in records if r.final_answer]
    evidence = [
        bool(r.retrieved_documents or r.causal_edges or r.causal_paths)
        for r in records
    ]
    causal_coverage = [
        bool(r.causal_edges or r.causal_paths)
        for r in records
    ]
    return {
        "Causal Retrieval Coverage": (sum(causal_coverage) / n,
                                      "fraction", NOT_COMPARABLE),
        "Causal Chain Depth": (_mean(paths), "mean_hops_of_returned_paths",
                               NOT_COMPARABLE),
        "Semantic Refinement Score": (None, "score", NEEDS_REFINEMENT_EVAL),
        "Context Relevance": (None, "score", NEEDS_ADJUDICATION),
        "Groundedness": (None, "score", NEEDS_ADJUDICATION),
        "Average Answer Length": (_mean(words), "whitespace_words_if_answer_present",
                                  NOT_COMPARABLE),
        "Hallucination Rate": (None, "rate", NEEDS_ADJUDICATION),
        "Retrieval Success Rate": (sum(evidence) / n, "candidate_yield_fraction",
                                   NOT_COMPARABLE),
        "Average Direct Causes": (_mean(causal_edges), "recorded_CAUSES_edges_per_query",
                                  NOT_COMPARABLE),
        "Total Direct Causes": (sum(causal_edges), "recorded_CAUSES_edges",
                                NOT_COMPARABLE),
        "Total Execution Time (s)": (
            wall_seconds if wall_seconds and wall_seconds > 0 else None,
            "run_wall_seconds", "Null if wall time was not recorded by original runner"),
        "Average Latency (s)": (
            _mean(query_ms) / 1000 if query_ms else None, "seconds",
            "Query-level mean excludes initialization"),
        "Average Latency (ms)": (_mean(query_ms), "milliseconds",
                                  "Query-level mean excludes initialization"),
        "Median Latency (ms)": (
            statistics.median(query_ms) if query_ms else None, "milliseconds",
            "Query-level median excludes initialization"),
        "Average Throughput (QPS)": (
            n / wall_seconds if wall_seconds and wall_seconds > 0 else None,
            "queries_per_run_wall_second", "Null if measured wall time unavailable"),
        "Refinement Success Rate": (None, "rate", NEEDS_REFINEMENT_EVAL),
    }


def summarize(records: list[PipelineRecord], wall_seconds: float | None = None) -> dict:
    computed = _value_metrics(records, wall_seconds)
    if set(computed) != set(LEGACY_METRICS):
        raise ValueError("Metric bridge definition has drifted")
    detectors = [r.hallucination_detected for r in records
                 if r.hallucination_detected is not None]
    validators = [r.validator_passed for r in records
                  if r.validator_passed is not None]
    result = {
        "queries": 132,
        "status_ok": sum(r.status == "ok" for r in records),
        "status_abstained": sum(r.status == "abstained" for r in records),
        "status_error": sum(r.status == "error" for r in records),
        "detector_flag_rate_not_gold": sum(detectors) / len(detectors)
            if detectors else None,
        "detector_observations": len(detectors),
        "validator_flag_pass_rate_not_gold": sum(validators) / len(validators)
            if validators else None,
        "validator_observations": len(validators),
        "mean_feedback_iterations_recorded": _mean(
            [r.feedback_iterations for r in records]
        ),
        "metrics": {
            key: {"value": value, "unit": unit, "limitation": limitation}
            for key, (value, unit, limitation) in computed.items()
        },
    }
    return result


def build(session_dir: Path, output_root: Path | None = None) -> tuple[Path, dict]:
    session_dir = _inside(session_dir)
    meta = json.loads((session_dir / "session.json").read_text(encoding="utf-8"))
    if meta.get("query_count") != 132 or not meta.get("runs"):
        raise ValueError("Metrics bridge refuses 1- or 2-query smoke sessions")
    # Filename and content are authenticated against the session manifest.
    reports, table = {}, []
    for model, entry in sorted(meta["runs"].items()):
        folder = _inside(session_dir / entry["folder"])
        records_file = folder / "records.jsonl"
        if file_sha256(records_file) != entry["record_sha256"]:
            raise ValueError("Recorded model artifacts changed: " + model)
        records = [PipelineRecord.model_validate_json(line)
                   for line in records_file.read_text(encoding="utf-8").splitlines()
                   if line.strip()]
        summary_path = folder / "summary.json"
        raw = json.loads(summary_path.read_text(encoding="utf-8"))
        measured = raw.get("wall_clock_seconds")
        if measured is None:
            measured = raw.get("elapsed_seconds")
        if measured is None:
            measured = raw.get("total_wall_seconds")
        if measured is not None and (isinstance(measured, bool) or
                                     not isinstance(measured, (int, float))):
            measured = None
        reports[model] = summarize(records, measured)
        for key, detail in reports[model]["metrics"].items():
            table.append({"model": model, "metric": key, **detail,
                          "status": ("unavailable" if detail["value"] is None else
                                     "descriptive_operational")})
    output_root = _inside(output_root) if output_root else session_dir
    out = output_root / ("legacy22_metric_bridge_132_" +
                         datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") +
                         "_" + uuid.uuid4().hex[:6])
    out.mkdir(parents=True, exist_ok=False)
    with (out / "legacy_metric_bridge.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)
    result = {
        "scope": "fixed_132_development_questions",
        "session_sha256": file_sha256(session_dir / "session.json"),
        "evidence_bases_differ": meta.get("evidence_bases_differ", True),
        "not_historical_unseen_test": True,
        "not_comparable_to_legacy_22_proxy_definitions": True,
        "models": reports,
        "warning": "Unavailable medical-quality fields are null, NOT zero or imputed. Context relevance, groundedness and hallucination require independently adjudicated references/claims.",
    }
    _write_json(out / "summary.json", result)
    print(json.dumps({"output": str(out), "models": list(reports),
                      "questions": 132, "medical_gold_generated": False}, indent=2))
    return out, result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--session-dir", type=Path, required=True)
    p.add_argument("--output-root", type=Path)
    args = p.parse_args()
    build(args.session_dir, args.output_root)


if __name__ == "__main__":
    main()
