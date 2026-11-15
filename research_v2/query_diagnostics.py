"""Evidence-yield diagnostics for all 100 queries and B1-B8.

No clinical gold labels are assumed. Absent models are NOT_RUN, not failures.
Pairwise comparisons require matched source, run settings, query IDs and code.
Usage:
    python3 -m research_v2.query_diagnostics
    python3 evaluate_all_baselines.py diagnose
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from .experiment_runner import PROJECT_ROOT, file_sha256
from .pipelines import MODEL_REGISTRY
from .presentation_export import _select_runs
from .presentation_matrix import MODEL_IDS, WORKSPACE, validate_inputs
from .schema import PipelineRecord

GROUPS = {
    "document_candidate": ("B1", "B2", "B3", "B4"),
    "graph_snapshot": ("B5", "B6"),
    "legacy_neo4j": ("B7", "B8"),
}


def _evidence(record: PipelineRecord | None) -> bool | None:
    if record is None:
        return None
    return bool(record.retrieved_documents or record.causal_edges or record.causal_paths)


def _write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_query_diagnostics(questions, metadata: dict, records_by_model: dict) -> list[dict]:
    """Preserve one row per question; missing runs must never become negatives."""
    ids = [q.query_id for q in questions]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate question IDs")
    for model, records in records_by_model.items():
        if model not in MODEL_IDS:
            raise ValueError(f"Unknown model: {model}")
        if set(records) != set(ids):
            raise ValueError(f"Incomplete or misaligned run: {model}")
    rows = []
    for question in questions:
        qid = question.query_id
        meta = metadata[qid]
        row = {
            "query_id": qid,
            "query_type": meta["query_type"],
            "medical_domain": meta.get("medical_domain"),
            "target_concept_draft": meta.get("target_concept"),
            "question": question.query,
        }
        doc_yes, graph_yes = 0, 0
        for model in MODEL_IDS:
            record = records_by_model.get(model, {}).get(qid)
            present = _evidence(record)
            row[f"{model}_status"] = record.status if record else "NOT_RUN"
            row[f"{model}_evidence"] = present
            row[f"{model}_documents"] = len(record.retrieved_documents) if record else None
            row[f"{model}_edges"] = len(record.causal_edges) if record else None
            row[f"{model}_paths"] = len(record.causal_paths) if record else None
            row[f"{model}_latency_ms"] = record.timings.total_ms if record else None
            if present and model in GROUPS["document_candidate"]:
                doc_yes += 1
            if present and model in GROUPS["graph_snapshot"]:
                graph_yes += 1
        row["document_models_returning_evidence"] = doc_yes
        row["graph_models_returning_evidence"] = graph_yes
        row["missing_models"] = ",".join(
            model for model in MODEL_IDS if row[f"{model}_status"] == "NOT_RUN"
        )

        # Priority is a *retrieval coverage triage*, not a clinical verdict.
        if "B5" not in records_by_model or "B6" not in records_by_model:
            bucket = "GRAPH_NOT_RUN"
        elif row["B5_evidence"] and not row["B6_evidence"]:
            bucket = "DIRECTED_RETRIEVAL_GAP"
        elif not row["B5_evidence"] and not row["B6_evidence"]:
            bucket = "G0_SNAPSHOT_COVERAGE_GAP"
        elif not row["B5_evidence"] and row["B6_evidence"]:
            bucket = "CAUSAL_ONLY_EVIDENCE"
        else:
            bucket = "BOTH_GRAPH_METHODS_RETURN_EVIDENCE"
        if bucket == "BOTH_GRAPH_METHODS_RETURN_EVIDENCE" and 0 < doc_yes < 4:
            bucket = "DOCUMENT_METHOD_DISAGREEMENT"
        row["retrieval_review_bucket"] = bucket
        rows.append(row)
    return rows


def _run_settings(workspace: Path, selected: dict) -> dict:
    journal_path = workspace / "matrix_runs.jsonl"
    journal = {}
    if journal_path.is_file():
        for line in journal_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                item = json.loads(line)
                journal[item["run_id"]] = item
    settings = {}
    for model, bundle in selected.items():
        _, _, summary_path, records, summary = bundle
        run_dir = summary_path.parent
        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        run_id = records[0].run_id
        row = journal.get(run_id, {})
        settings[model] = {
            "run_id": run_id,
            "pipeline_id": records[0].pipeline_id,
            "code_sha": manifest.get("git_commit"),
            "dataset_sha256": manifest.get("input_sha256"),
            "query_count": summary.get("total_queries"),
            "execution_mode": row.get("execution_mode") or (
                "end_to_end" if bundle[0] else "retrieval_only"
            ),
            "corpus_sha256": row.get("corpus_sha256"),
            "embedding_backend": row.get("embedding_backend"),
            "top_k": row.get("top_k"),
            "graph_version": manifest.get("graph_version_label"),
            "has_matrix_journal": bool(row),
            "aborted": bool(summary.get("aborted")),
        }
    return settings


def pair_comparability(left: str, right: str, settings: dict) -> tuple[bool, str]:
    """Only descriptive paired retrieval/latency comparison, never accuracy."""
    group = next((g for g, ms in GROUPS.items() if left in ms and right in ms), None)
    if group is None:
        return False, "different_evidence_families"
    if left not in settings or right not in settings:
        return False, "missing_run"
    a, b = settings[left], settings[right]
    for name in ("has_matrix_journal", "code_sha", "dataset_sha256", "execution_mode",
                 "query_count", "graph_version"):
        if name == "has_matrix_journal" and not (a[name] and b[name]):
            return False, "missing_run_provenance"
        if name != "has_matrix_journal" and (a.get(name) is None or a[name] != b.get(name)):
            return False, f"mismatched_{name}"
    if a["aborted"] or b["aborted"]:
        return False, "aborted_run"
    if group == "document_candidate":
        if not a["corpus_sha256"] or a["corpus_sha256"] != b["corpus_sha256"]:
            return False, "mismatched_document_corpus"
        if not a["top_k"] or a["top_k"] != b["top_k"]:
            return False, "mismatched_top_k"
        if left in {"B3", "B4"} and right in {"B3", "B4"}:
            if not a["embedding_backend"] or a["embedding_backend"] != b["embedding_backend"]:
                return False, "mismatched_dense_encoder"
    if group == "graph_snapshot" and (a["top_k"] is None or a["top_k"] != b["top_k"]):
        return False, "mismatched_top_k"
    if group == "legacy_neo4j":
        # These runs need per-query Neo4j graph-fingerprint and runtime-lock
        # equivalence, not just the G0 label. Avoid unsupported pairing.
        return False, "legacy_runtime_fingerprint_not_checked"
    return True, group


def build_pairwise(rows: list[dict], settings: dict) -> tuple[list[dict], list[dict]]:
    summaries, query_rows = [], []
    for group, models in GROUPS.items():
        for left, right in itertools.combinations(models, 2):
            valid, reason = pair_comparability(left, right, settings)
            if not valid:
                summaries.append({
                    "left_model": left, "right_model": right,
                    "basis": group, "comparable": False, "reason": reason,
                    "paired_queries": 0,
                    "both_evidence": None, "left_only": None,
                    "right_only": None, "neither": None,
                    "mean_right_minus_left_ms": None,
                })
                continue
            counts = Counter()
            latency_deltas = []
            for row in rows:
                l, r = row[f"{left}_evidence"], row[f"{right}_evidence"]
                if l is None or r is None:
                    raise ValueError("Comparable pair contains an absent query")
                outcome = (
                    "both_evidence" if l and r else
                    "left_only" if l else "right_only" if r else "neither"
                )
                counts[outcome] += 1
                a, b = row[f"{left}_latency_ms"], row[f"{right}_latency_ms"]
                delta = b - a if a is not None and b is not None else None
                if delta is not None:
                    latency_deltas.append(delta)
                query_rows.append({
                    "query_id": row["query_id"], "query_type": row["query_type"],
                    "left_model": left, "right_model": right, "basis": group,
                    "left_status": row[f"{left}_status"],
                    "right_status": row[f"{right}_status"],
                    "evidence_outcome": outcome,
                    "right_minus_left_ms": delta,
                })
            summaries.append({
                "left_model": left, "right_model": right, "basis": group,
                "comparable": True, "reason": "matched_pilot_and_runtime_settings",
                "paired_queries": len(rows),
                **{kind: counts[kind] for kind in
                   ("both_evidence", "left_only", "right_only", "neither")},
                "mean_right_minus_left_ms": (
                    statistics.mean(latency_deltas) if latency_deltas else None
                ),
            })
    return summaries, query_rows


def _category_summary(rows: list[dict]) -> list[dict]:
    output = []
    for model in MODEL_IDS:
        for category in sorted({row["query_type"] for row in rows}):
            group = [row for row in rows if row["query_type"] == category]
            measured = [row for row in group if row[f"{model}_status"] != "NOT_RUN"]
            output.append({
                "model_id": model,
                "pipeline": MODEL_REGISTRY[model].name,
                "query_type": category,
                "total_queries": len(group),
                "measured": len(measured),
                "not_run": len(group) - len(measured),
                "ok": sum(r[f"{model}_status"] == "ok" for r in measured),
                "abstained": sum(r[f"{model}_status"] == "abstained" for r in measured),
                "errors": sum(r[f"{model}_status"] == "error" for r in measured),
                "evidence_returned": sum(r[f"{model}_evidence"] is True for r in measured),
            })
    return output


def diagnose(workspace: Path = WORKSPACE, output_dir: Path | None = None):
    workspace = Path(workspace).expanduser().resolve()
    if not workspace.is_relative_to(PROJECT_ROOT):
        raise ValueError("Workspace must be within the repository")
    input_file = workspace / "benchmark/questions_only.jsonl"
    questions, metadata = validate_inputs(input_file)
    ids = [question.query_id for question in questions]
    selected = _select_runs(workspace / "runs", ids, file_sha256(input_file))
    records_by_model = {
        model: {r.query_id: r for r in bundle[3]}
        for model, bundle in selected.items()
    }
    rows = build_query_diagnostics(questions, metadata, records_by_model)
    settings = _run_settings(workspace, selected)
    comparisons, per_query_pairs = build_pairwise(rows, settings)
    category = _category_summary(rows)
    if output_dir is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        output_dir = workspace / "metrics" / f"query_diagnostics_{timestamp}"
    output_dir = Path(output_dir).expanduser().resolve()
    if not output_dir.is_relative_to(PROJECT_ROOT):
        raise ValueError("Output must remain inside the repository")
    output_dir.mkdir(parents=True, exist_ok=False)
    _write_csv(output_dir / "query_coverage_B1_B8.csv", rows, list(rows[0]))
    _write_csv(output_dir / "model_by_query_type.csv", category, list(category[0]))
    _write_csv(output_dir / "matched_pairwise_summary.csv", comparisons, list(comparisons[0]))
    pair_cols = ("query_id", "query_type", "left_model", "right_model", "basis",
                 "left_status", "right_status", "evidence_outcome", "right_minus_left_ms")
    _write_csv(output_dir / "matched_pairwise_by_query.csv", per_query_pairs, list(pair_cols))
    priority = {
        "DIRECTED_RETRIEVAL_GAP": 0,
        "G0_SNAPSHOT_COVERAGE_GAP": 1,
        "CAUSAL_ONLY_EVIDENCE": 2,
        "DOCUMENT_METHOD_DISAGREEMENT": 3,
        "BOTH_GRAPH_METHODS_RETURN_EVIDENCE": 4,
        "GRAPH_NOT_RUN": 5,
    }
    review = sorted(rows, key=lambda row: (priority[row["retrieval_review_bucket"]], row["query_id"]))
    _write_csv(output_dir / "graph_gap_review_queue.csv", review, list(review[0]))
    manifest = {
        "question_count": len(rows),
        "model_count": len(MODEL_IDS),
        "measured_models": sorted(selected),
        "missing_models": [m for m in MODEL_IDS if m not in selected],
        "input_sha256": file_sha256(input_file),
        "clinical_gold_verified": False,
        "clinical_accuracy_computed": False,
        "group_rules": GROUPS,
        "review_bucket_counts": dict(Counter(r["retrieval_review_bucket"] for r in rows)),
        "paired_comparisons_accepted": sum(r["comparable"] for r in comparisons),
        "paired_comparisons_excluded": [
            {"models": [r["left_model"], r["right_model"]], "reason": r["reason"]}
            for r in comparisons if not r["comparable"]
        ],
        "provenance": settings,
        "caveat": (
            "Evidence presence, graph coverage and latency are not medical "
            "correctness. Matched pairs compare operational retrieval only; "
            "the pilot and its target concepts are unreviewed. Missing B7/B8 "
            "runs remain NOT_RUN rather than being scored as errors."
        ),
    }
    (output_dir / "diagnostics_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output_dir": str(output_dir), **{
        key: manifest[key] for key in
        ("question_count", "measured_models", "missing_models",
         "review_bucket_counts", "paired_comparisons_accepted")
    }}, indent=2))
    return output_dir, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=WORKSPACE)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    diagnose(args.workspace, args.output_dir)


if __name__ == "__main__":
    main()
