"""Export validated fixed-132 B1–B8 experiments to separate, auditable CSV files.

All data come from completed research_v2.thesis132 run records; this module does
not rerun models, invent quality scores, or use evaluator labels at inference.
Multiple sessions are accepted only with identical dataset/code/run settings.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .experiment_runner import PROJECT_ROOT as ROOT, file_sha256
from .pipelines import MODEL_REGISTRY
from .schema import PipelineRecord
from .thesis132 import _inside, _read_json
from .thesis132_dataset_release import RELEASE, verify_release

DEFAULT_OUTPUT = ROOT / "presentation_20261118/thesis132/csv_reports"
MODELS = tuple(f"B{i}" for i in range(1, 9))
FIELDS = [
    "model", "model_name", "query_number", "query_id", "query", "query_type",
    "medical_domain", "semantic_group", "fold", "mode", "run_id",
    "graph_version", "status", "error_message",
    "refinement_type", "refined_query_text", "matched_concept",
    "num_retrieved_documents", "retrieved_document_ids_json",
    "retrieved_source_ids_json", "retrieved_scores_json",
    "direct_causes", "num_direct_causes", "other_relation_count",
    "multi_hop_paths", "num_multi_hop_paths", "causal_chain_depth_hops_mean",
    "retrieval_success", "rewritten_knowledge", "final_response",
    "answer_present", "is_hallucination", "validator_passed",
    "supported_claim_count", "unsupported_claim_count",
    "feedback_iterations_recorded", "action_history_json",
    "latency_seconds", "latency_ms", "refinement_ms", "retrieval_ms",
    "rewriting_ms", "generation_ms", "validation_ms",
    "query_throughput_qps_proxy", "generator_model", "embedding_model",
    "random_seed",
]
SUMMARY_FIELDS = [
    "model", "model_name", "mode", "source_session", "run_id",
    "question_count", "status_ok", "status_abstained", "status_error",
    "aborted", "operational_run_valid",
    "candidate_evidence_count", "candidate_evidence_rate",
    "answer_present_count", "answer_present_rate",
    "recorded_causal_edge_count", "recorded_causal_path_count",
    "mean_direct_causes", "mean_multi_hop_paths", "mean_path_depth_hops_if_any",
    "mean_feedback_iterations_recorded",
    "mean_latency_ms", "median_latency_ms", "p95_latency_ms",
    "mean_refinement_ms", "mean_retrieval_ms", "mean_rewriting_ms",
    "mean_generation_ms", "mean_validation_ms",
    "wall_time_seconds", "actual_run_throughput_qps",
    "detector_observations", "detector_flagged", "detector_flag_rate_not_gold",
    "validator_observations", "validator_passed",
    "independent_medical_accuracy", "clinical_groundedness_score",
    "evidence_source_type", "input_sha256", "code_commit",
]
WIDE_PER_MODEL = [
    "status", "latency_ms", "retrieval_success",
    "num_direct_causes", "num_multi_hop_paths", "answer_present",
]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError("THESIS132_CSV_EXPORT: " + message)


def _json(value) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _mean(values):
    measured = [v for v in values if v is not None]
    return statistics.mean(measured) if measured else None


def _percentile(values, p):
    measured = sorted(v for v in values if v is not None)
    if not measured:
        return None
    position = (len(measured) - 1) * p
    low = int(position)
    high = min(low + 1, len(measured) - 1)
    return measured[low] + (position - low) * (measured[high] - measured[low])


def _csv(path: Path, fields: list[str], rows: list[dict]):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def result_to_row(model: str, record: PipelineRecord, metadata: dict,
                  query_number: int, mode: str) -> dict:
    """Convert a recorded response. None stays missing, never false or 0."""
    docs = record.retrieved_documents
    edges = [edge for edge in record.causal_edges if edge.relation == "CAUSES"]
    paths = record.causal_paths
    hops = [len(p.nodes) - 1 for p in paths]
    causal_edges = [edge.model_dump(mode="json") for edge in edges]
    total = record.timings.total_ms
    evidence = bool(docs or record.causal_edges or paths)
    return {
        "model": model,
        "model_name": MODEL_REGISTRY[model].name,
        "query_number": query_number, "query_id": record.query_id,
        "query": record.original_query, "query_type": metadata["query_type"],
        "medical_domain": metadata["medical_domain"],
        "semantic_group": metadata["semantic_group"],
        "fold": metadata["fold"], "mode": mode, "run_id": record.run_id,
        "graph_version": record.graph_version or "",
        "status": record.status, "error_message": record.error_message,
        "refinement_type": record.refinement_action,
        "refined_query_text": record.refined_query,
        "matched_concept": record.matched_concept,
        "num_retrieved_documents": len(docs),
        "retrieved_document_ids_json": _json([d.document_id for d in docs]),
        "retrieved_source_ids_json": _json([d.source_id for d in docs]),
        "retrieved_scores_json": _json([d.score for d in docs]),
        "direct_causes": _json(causal_edges),
        "num_direct_causes": len(edges),
        "other_relation_count": len(record.causal_edges) - len(edges),
        "multi_hop_paths": _json([path.model_dump(mode="json") for path in paths]),
        "num_multi_hop_paths": len(paths),
        "causal_chain_depth_hops_mean": _mean(hops),
        "retrieval_success": evidence,
        "rewritten_knowledge": record.rewritten_knowledge,
        "final_response": record.final_answer,
        "answer_present": bool(record.final_answer and record.final_answer.strip()),
        # This is a pipeline detector flag, NOT verified medical hallucination.
        "is_hallucination": record.hallucination_detected,
        "validator_passed": record.validator_passed,
        "supported_claim_count": len(record.supported_claims),
        "unsupported_claim_count": len(record.unsupported_claims),
        "feedback_iterations_recorded": record.feedback_iterations,
        "action_history_json": _json(record.action_history),
        "latency_seconds": total / 1000 if total is not None else None,
        "latency_ms": total,
        "refinement_ms": record.timings.refinement_ms,
        "retrieval_ms": record.timings.retrieval_ms,
        "rewriting_ms": record.timings.rewriting_ms,
        "generation_ms": record.timings.generation_ms,
        "validation_ms": record.timings.validation_ms,
        # Reciprocal latency is NOT measured batch throughput.
        "query_throughput_qps_proxy": 1000 / total if total and total > 0 else None,
        "generator_model": record.generator_model,
        "embedding_model": record.embedding_model,
        "random_seed": record.random_seed,
    }


def _model_summary(model: str, rows: list[dict], summary: dict,
                   session: Path, code_commit: str, input_sha: str,
                   mode: str, run_id: str) -> dict:
    wall = summary.get("wall_time_seconds")
    if isinstance(wall, bool) or not isinstance(wall, (int, float)) or wall <= 0:
        wall = None
    n = len(rows)
    observed_det = [r["is_hallucination"] for r in rows
                    if r["is_hallucination"] is not None]
    observed_val = [r["validator_passed"] for r in rows
                    if r["validator_passed"] is not None]
    paths = [r for r in rows if r["causal_chain_depth_hops_mean"] is not None]
    return {
        "model": model, "model_name": MODEL_REGISTRY[model].name, "mode": mode,
        "source_session": str(session.relative_to(ROOT)), "run_id": run_id,
        "question_count": n,
        "status_ok": sum(r["status"] == "ok" for r in rows),
        "status_abstained": sum(r["status"] == "abstained" for r in rows),
        "status_error": sum(r["status"] == "error" for r in rows),
        "aborted": bool(summary.get("aborted")),
        "operational_run_valid": not summary.get("aborted") and
            not any(r["status"] == "error" for r in rows),
        "candidate_evidence_count": sum(bool(r["retrieval_success"]) for r in rows),
        "candidate_evidence_rate": sum(bool(r["retrieval_success"]) for r in rows) / n,
        "answer_present_count": sum(bool(r["answer_present"]) for r in rows),
        "answer_present_rate": sum(bool(r["answer_present"]) for r in rows) / n,
        "recorded_causal_edge_count": sum(r["num_direct_causes"] for r in rows),
        "recorded_causal_path_count": sum(r["num_multi_hop_paths"] for r in rows),
        "mean_direct_causes": _mean(r["num_direct_causes"] for r in rows),
        "mean_multi_hop_paths": _mean(r["num_multi_hop_paths"] for r in rows),
        "mean_path_depth_hops_if_any": _mean(
            r["causal_chain_depth_hops_mean"] for r in paths),
        "mean_feedback_iterations_recorded": _mean(
            r["feedback_iterations_recorded"] for r in rows),
        "mean_latency_ms": _mean(r["latency_ms"] for r in rows),
        "median_latency_ms": _percentile([r["latency_ms"] for r in rows], .5),
        "p95_latency_ms": _percentile([r["latency_ms"] for r in rows], .95),
        "mean_refinement_ms": _mean(r["refinement_ms"] for r in rows),
        "mean_retrieval_ms": _mean(r["retrieval_ms"] for r in rows),
        "mean_rewriting_ms": _mean(r["rewriting_ms"] for r in rows),
        "mean_generation_ms": _mean(r["generation_ms"] for r in rows),
        "mean_validation_ms": _mean(r["validation_ms"] for r in rows),
        "wall_time_seconds": wall,
        "actual_run_throughput_qps": n / wall if wall else None,
        "detector_observations": len(observed_det),
        "detector_flagged": sum(observed_det),
        "detector_flag_rate_not_gold": (
            sum(observed_det) / len(observed_det) if observed_det else None),
        "validator_observations": len(observed_val),
        "validator_passed": sum(observed_val),
        "independent_medical_accuracy": None,
        "clinical_groundedness_score": None,
        "evidence_source_type": (
            "provisional_source_attributed_documents"
            if model in {"B1", "B2", "B3", "B4"} else "unverified_G0_causal_graph"
        ),
        "input_sha256": input_sha, "code_commit": code_commit,
    }


def export(sessions: list[Path], output_root: Path = DEFAULT_OUTPUT,
           require_all: bool = False, fail_on_errors: bool = False) -> tuple[Path, dict]:
    """Create new timestamped folder; never replace 22-query CSV results."""
    audit = verify_release()
    _require(bool(sessions), "At least one --session-dir is required")
    release_questions = [
        json.loads(line) for line in (RELEASE / "questions_only.jsonl").read_text(
            encoding="utf-8").splitlines() if line.strip()
    ]
    with (RELEASE / "metadata_review_only.csv").open(newline="", encoding="utf-8") as f:
        review_meta = {r["query_id"]: r for r in csv.DictReader(f)}
    reference_ids = [q["query_id"] for q in release_questions]
    _require(len(review_meta) == 132, "Incomplete evaluation metadata")
    all_rows = {}
    summaries = {}
    sources = {}
    reference_settings = None
    for raw_session in sessions:
        session = _inside(raw_session)
        session_meta = _read_json(session / "session.json")
        _require(session_meta.get("schema") == "THESIS132_SESSION_V1",
                 "Invalid thesis132 session " + str(session))
        settings = tuple(session_meta.get(k) for k in (
            "dataset_sha256", "query_count", "mode", "encoder", "top_k",
            "code_commit", "candidate_corpus_sha256", "G0_nodes_sha256",
            "G0_edges_sha256",
        ))
        _require(settings[0] == audit["query_sha256"] and settings[1] == 132,
                 "Session does not use exact fixed 132-query release")
        if reference_settings is None:
            reference_settings = settings
        _require(settings == reference_settings,
                 "Different input, mode, code commit, corpus or G0 graph across sessions")
        for model, item in session_meta.get("runs", {}).items():
            _require(model in MODELS and model not in all_rows,
                     "Unknown or duplicate model across sessions: " + model)
            folder = _inside(session / item["folder"])
            _require(folder.is_relative_to(session),
                     "Run records outside session: " + model)
            records_path = folder / "records.jsonl"
            _require(records_path.is_file(), "Missing records: " + model)
            _require(item["record_sha256"] == file_sha256(records_path),
                     "Modified experimental records: " + model)
            raw_records = [PipelineRecord.model_validate_json(line)
                           for line in records_path.read_text(encoding="utf-8").splitlines()
                           if line.strip()]
            _require(len(raw_records) == 132, "Not exactly 132 results: " + model)
            _require([r.query_id for r in raw_records] == reference_ids,
                     "Question ID/order mismatch: " + model)
            for r, q in zip(raw_records, release_questions):
                _require(r.original_query == q["query"], "Question wording changed: " + model)
                _require(r.run_id == item["run_id"], "Record run ID mismatch: " + model)
            run_manifest = _read_json(folder / "manifest.json")
            _require(run_manifest.get("input_sha256") == audit["query_sha256"] and
                     run_manifest.get("git_commit") == settings[5] and
                     run_manifest.get("run_id") == item["run_id"],
                     "Run manifest provenance mismatch: " + model)
            run_summary = _read_json(folder / "summary.json")
            _require(run_summary.get("total_queries") == 132,
                     "Incomplete run summary: " + model)
            counts = {
                "successful": sum(r.status == "ok" for r in raw_records),
                "abstained": sum(r.status == "abstained" for r in raw_records),
                "errors": sum(r.status == "error" for r in raw_records),
            }
            for key, value in counts.items():
                _require(run_summary.get(key) == value,
                         "Run summary status mismatch " + model + ": " + key)
            rows = [result_to_row(model, rec, review_meta[rec.query_id],
                                  i, session_meta["mode"])
                    for i, rec in enumerate(raw_records, 1)]
            all_rows[model] = rows
            summaries[model] = _model_summary(
                model, rows, run_summary, session, settings[5],
                audit["query_sha256"], session_meta["mode"], item["run_id"],
            )
            sources[model] = {
                "session": str(session.relative_to(ROOT)),
                "records_path": str(records_path.relative_to(ROOT)),
                "records_sha256": item["record_sha256"],
                "run_id": item["run_id"],
            }
    _require(bool(all_rows), "No completed model records in these sessions")
    missing = [model for model in MODELS if model not in all_rows]
    invalid = [model for model in MODELS if model in summaries and
               not summaries[model]["operational_run_valid"]]
    if require_all:
        _require(not missing, "Incomplete set; missing model CSVs: " + ", ".join(missing))
    if fail_on_errors:
        _require(not invalid, "Models with execution errors: " + ", ".join(invalid))

    output_root = _inside(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    out = output_root / (
        "batch132_csv_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") +
        "_" + uuid.uuid4().hex[:6]
    )
    out.mkdir(parents=True, exist_ok=False)
    files = {}
    for model in MODELS:
        if model not in all_rows:
            continue
        path = out / f"{model}_queries_132.csv"
        _csv(path, FIELDS, all_rows[model])
        files[path.name] = file_sha256(path)
    summary_rows = [summaries[model] for model in MODELS if model in summaries]
    _csv(out / "pipeline_summary_132.csv", SUMMARY_FIELDS, summary_rows)
    files["pipeline_summary_132.csv"] = file_sha256(out / "pipeline_summary_132.csv")
    long_rows = [r for model in MODELS if model in all_rows for r in all_rows[model]]
    _csv(out / "all_models_per_query_132.csv", FIELDS, long_rows)
    files["all_models_per_query_132.csv"] = file_sha256(out / "all_models_per_query_132.csv")

    wide = []
    wide_fields = ["query_id", "query", "query_type", "medical_domain", "semantic_group", "fold"]
    for model in MODELS:
        wide_fields.extend(f"{model}_{field}" for field in WIDE_PER_MODEL)
    for i, q in enumerate(release_questions):
        row = {"query_id": q["query_id"], "query": q["query"],
               **{k: review_meta[q["query_id"]][k] for k in
                  ("query_type", "medical_domain", "semantic_group", "fold")}}
        for model in MODELS:
            if model in all_rows:
                record = all_rows[model][i]
                row.update({f"{model}_{field}": record[field] for field in WIDE_PER_MODEL})
            else:
                row.update({f"{model}_{field}": None for field in WIDE_PER_MODEL})
        wide.append(row)
    _csv(out / "comparison_by_query_132.csv", wide_fields, wide)
    files["comparison_by_query_132.csv"] = file_sha256(out / "comparison_by_query_132.csv")
    report = {
        "schema": "THESIS132_BATCH_CSV_EXPORT_V1",
        "status": ("COMPLETE_8_MODELS" if not missing and not invalid
                   else "PARTIAL_OR_INVALID_OPERATIONAL_RUNS"),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "query_count_per_model": 132, "models_present": [
            m for m in MODELS if m in all_rows
        ], "models_missing": missing, "models_with_errors": invalid,
        "full_8_models_successful": not missing and not invalid,
        "dataset_sha256": audit["query_sha256"],
        "session_configuration": {
            key: value for key, value in zip((
                "dataset_sha256", "query_count", "mode", "encoder", "top_k",
                "code_commit", "candidate_corpus_sha256", "G0_nodes_sha256",
                "G0_edges_sha256",
            ), reference_settings)
        },
        "source_runs": sources, "generated_csv_sha256": files,
        "medical_accuracy": None, "clinical_groundedness": None,
        "historically_unseen_test": False,
        "warnings": [
            "Identical 132 inputs do not imply common evidence: B1-B4 use candidate documents; B5-B8 use G0.",
            "Retrieval success measures candidate evidence availability, not medical correctness.",
            "Hallucination detector flags are not independently adjudicated hallucinations.",
            "Missing detector/timing/quality fields remain empty, never imputed as zero.",
            "The 132 questions influenced prior G1 development; prospective family folds are not an external holdout.",
            "Historical 22-question semantic similarity proxies are not automatically valid medical quality scores.",
            "Reciprocal query latency is a proxy, not measured run throughput; run QPS includes recorded wall time.",
        ],
    }
    (out / "export_manifest.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(out), "models": report["models_present"],
                      "missing": missing, "invalid": invalid, "dataset_sha256":
                      audit["query_sha256"]}, indent=2))
    return out, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-dir", required=True, action="append", type=Path,
                        help="Repeat for separate complete 132-query sessions")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--require-all", action="store_true",
                        help="Require B1-B8 to all have 132 recorded queries")
    parser.add_argument("--fail-on-errors", action="store_true",
                        help="Reject results with actual errors or graph-drift aborts")
    args = parser.parse_args()
    export(args.session_dir, args.output_root, args.require_all, args.fail_on_errors)


if __name__ == "__main__":
    main()
