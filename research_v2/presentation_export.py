"""Build a supervisor-readable 100-query x B1-B8 experiment matrix.

Inputs are existing, non-overwriting standardized PipelineRecord JSONL runs.
Missing pipelines remain explicitly NOT_RUN; no fabricated predictions or scores.
No medical gold metrics are computed here.

Usage:
  python3 -m research_v2.presentation_export
  python3 -m research_v2.presentation_export --output-dir presentation_20261118/metrics/latest_matrix
"""
import argparse
import csv
import json
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from .experiment_runner import PROJECT_ROOT, file_sha256
from .presentation_matrix import (
    EXPECTED_PIPELINE_IDS, INPUT, MODEL_IDS, WORKSPACE,
    load_metadata, validate_inputs,
)
from .presentation_sprint import PIPELINES
from .schema import PipelineRecord


def _write_csv(path, headers, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=headers, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _select_runs(runs_dir, expected_ids, input_hash):
    """Newest complete 100-record run for each model; prefer end-to-end."""
    eligible = defaultdict(list)
    journal_modes = {}
    journal = runs_dir.parent / "matrix_runs.jsonl"
    if journal.is_file():
        for line in journal.read_text(encoding="utf-8").splitlines():
            if line.strip():
                item = json.loads(line)
                journal_modes[item["run_id"]] = item.get("execution_mode")
    for path in sorted(runs_dir.glob("*/summary.json")):
        try:
            info = json.loads(path.read_text(encoding="utf-8"))
            records_file = path.parent / "records.jsonl"
            manifest_file = path.parent / "manifest.json"
            manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
            if info.get("total_queries") != 100 or info.get("aborted"):
                continue
            if manifest.get("input_sha256") != input_hash:
                continue
            records = [
                PipelineRecord.model_validate_json(line)
                for line in records_file.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            if len(records) != 100:
                continue
            if [r.query_id for r in records] != expected_ids:
                continue
            for model in MODEL_IDS:
                if {r.pipeline_id for r in records} == {EXPECTED_PIPELINE_IDS[model]}:
                    # A generation crash can leave every answer empty. Use the
                    # declared run mode rather than inferring it from answers.
                    declared = journal_modes.get(info.get("run_id"))
                    e2e = (declared == "end_to_end") if declared else any(
                        r.generator_model or r.final_answer for r in records
                    )
                    eligible[model].append((bool(e2e), path.stat().st_mtime_ns, path, records, info))
                    break
        except (OSError, ValueError, KeyError):
            continue
    # Successful 100-query runs take precedence over all-error end-to-end
    # attempts. Within the same error count, prefer end-to-end and freshness.
    return {
        model: max(items, key=lambda item: (
            -item[4].get("errors", 100), item[0], item[1]
        ))
        for model, items in eligible.items()
    }


def export(workspace=WORKSPACE, output_dir=None):
    workspace = Path(workspace).expanduser().resolve()
    if not workspace.is_relative_to(PROJECT_ROOT):
        raise ValueError("Workspace must be inside CDF-RAG-2")
    questions, metadata = validate_inputs(workspace / "benchmark/questions_only.jsonl")
    ids = [row.query_id for row in questions]
    selected = _select_runs(
        workspace / "runs", ids,
        file_sha256(workspace / "benchmark/questions_only.jsonl")
    )
    if output_dir is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        output_dir = workspace / "metrics" / f"supervisor_matrix_{stamp}"
    output_dir = Path(output_dir).expanduser().resolve()
    if not output_dir.is_relative_to(PROJECT_ROOT):
        raise ValueError("Report path must be inside CDF-RAG-2")
    output_dir.mkdir(parents=True, exist_ok=False)

    records_by_model = {
        model: {r.query_id: r for r in bundle[3]}
        for model, bundle in selected.items()
    }
    long_rows, wide_rows, detailed_rows = [], [], []
    for question in questions:
        meta = metadata[question.query_id]
        wide = {
            "query_id": question.query_id,
            "query_type": meta["query_type"],
            "medical_domain": meta.get("medical_domain"),
            "question": question.query,
        }
        for model in MODEL_IDS:
            record = records_by_model.get(model, {}).get(question.query_id)
            result = {
                "query_id": question.query_id,
                "query_type": meta["query_type"],
                "medical_domain": meta.get("medical_domain"),
                "question": question.query,
                "model_id": model,
                "pipeline": PIPELINES[model],
                "status": record.status if record else "NOT_RUN",
                "returned_evidence": (
                    bool(record.retrieved_documents or record.causal_edges or record.causal_paths)
                    if record else None
                ),
                "retrieved_documents": len(record.retrieved_documents) if record else None,
                "causal_edges": len(record.causal_edges) if record else None,
                "causal_paths": len(record.causal_paths) if record else None,
                "answer_present": bool(record.final_answer and record.final_answer.strip()) if record else None,
                "latency_ms": round(record.timings.total_ms, 5)
                              if record and record.timings.total_ms is not None else None,
                "retrieval_ms": round(record.timings.retrieval_ms, 5)
                                if record and record.timings.retrieval_ms is not None else None,
                "generation_ms": round(record.timings.generation_ms, 5)
                                 if record and record.timings.generation_ms is not None else None,
                "run_id": record.run_id if record else None,
            }
            long_rows.append(result)
            detailed_rows.append({
                **result,
                # Full evidence and generated answers are kept in a separate
                # JSONL for local research inspection, never silently
                # promoted into medical-gold evaluation.
                "matched_concept": record.matched_concept if record else None,
                "refined_query": record.refined_query if record else None,
                "refinement_action": record.refinement_action if record else None,
                "document_ids": [d.document_id for d in record.retrieved_documents] if record else None,
                "source_ids": [d.source_id for d in record.retrieved_documents] if record else None,
                "causal_edge_tuples": [
                    [edge.cause, edge.effect, edge.relation]
                    for edge in record.causal_edges
                ] if record else None,
                "causal_path_nodes": [
                    path.nodes for path in record.causal_paths
                ] if record else None,
                "final_answer": record.final_answer if record else None,
                "feedback_iterations": record.feedback_iterations if record else None,
                "action_history": record.action_history if record else None,
                "hallucination_detected_proxy": record.hallucination_detected if record else None,
                "validator_passed_proxy": record.validator_passed if record else None,
                "error_message": record.error_message if record else None,
                "timings": record.timings.model_dump() if record else None,
            })
            wide.update({
                f"{model}_status": result["status"],
                f"{model}_evidence": result["returned_evidence"],
                f"{model}_edges": result["causal_edges"],
                f"{model}_paths": result["causal_paths"],
                f"{model}_latency_ms": result["latency_ms"],
                f"{model}_answer": result["answer_present"],
            })
        wide_rows.append(wide)

    long_cols = list(long_rows[0])
    wide_cols = list(wide_rows[0])
    _write_csv(output_dir / "all_800_slots_long.csv", long_cols, long_rows)
    _write_csv(output_dir / "all_100_questions_wide.csv", wide_cols, wide_rows)
    # Full per-question payloads are intended for local analysis. Public
    # repository publication requires manual privacy/copyright review.
    (output_dir / "query_model_details.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in detailed_rows),
        encoding="utf-8",
    )

    summary = []
    for model in MODEL_IDS:
        bundle = selected.get(model)
        records = list(records_by_model.get(model, {}).values())
        success = sum(r.status == "ok" for r in records)
        abstain = sum(r.status == "abstained" for r in records)
        errors = sum(r.status == "error" for r in records)
        times = [r.timings.total_ms for r in records if r.timings.total_ms is not None]
        paths = sum(len(r.causal_paths) for r in records)
        summary.append({
            "model_id": model,
            "pipeline": PIPELINES[model],
            "execution_status": (
                "COMPLETE_100" if bundle and errors == 0 else
                "COMPLETE_100_WITH_ERRORS" if bundle else "NOT_RUN"
            ),
            "experiment_mode": (
                "end_to_end" if bundle and bundle[0] else "retrieval_only"
                if bundle else "not_run"
            ),
            "evaluated": len(records),
            "evidence_returned": sum(bool(r.retrieved_documents or r.causal_edges or r.causal_paths) for r in records),
            "abstained": abstain,
            "errors": errors,
            "answer_present": sum(bool(r.final_answer and r.final_answer.strip()) for r in records),
            "causal_paths": paths,
            "mean_latency_ms": round(statistics.mean(times), 5) if times else None,
            "run_id": bundle[3][0].run_id if bundle else None,
        })
    _write_csv(output_dir / "model_summary_B1_B8.csv", list(summary[0]), summary)
    manifest = {
        "questions": 100,
        "model_slots": 8,
        "expected_rows": 800,
        "actual_completed_model_runs": len(selected),
        "missing_models": [m for m in MODEL_IDS if m not in selected],
        "dataset_sha256": file_sha256(workspace / "benchmark/questions_only.jsonl"),
        "medical_gold_verified": False,
        "detailed_jsonl_contains_unreviewed_answers": True,
        "publish_detailed_jsonl_only_after_manual_review": True,
        "output_files": [
            "all_800_slots_long.csv",
            "all_100_questions_wide.csv",
            "model_summary_B1_B8.csv",
            "query_model_details.jsonl",
        ],
        "validity_warning": (
            "A retrieved document is not proof of relevance or correctness. "
            "Graph and text baselines use different evidence. Retrieval-only "
            "timings are incomparable to full LLM generation timings. "
            "No medical accuracy, hallucination or groundedness claim is made."
        ),
        "selected": {
            key: {"run_dir": str(bundle[2].parent.relative_to(PROJECT_ROOT)),
                  "mode": "end_to_end" if bundle[0] else "retrieval_only"}
            for key, bundle in selected.items()
        },
    }
    (output_dir / "report_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(output_dir), **manifest}, indent=2))
    return output_dir, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=WORKSPACE)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if Path.cwd().resolve() != PROJECT_ROOT:
        parser.error("Run from CDF-RAG-2 project root")
    export(args.workspace, args.output_dir)


if __name__ == "__main__":
    main()
