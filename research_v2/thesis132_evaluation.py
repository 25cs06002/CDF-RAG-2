"""Strict, zero-LLM evaluation of the frozen 132-question B1–B8 CSV export.

Root entry points: evaluate_metrics_132.py, compare_evaluation_132.py.
Outputs are a new immutable analysis folder next to the source export. The old
22-query scores and result files remain untouched.

No medically verified accuracy, hallucination rate, groundedness or F1 scores
are inferred from unreviewed candidate evidence or pipeline detector flags.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "data/benchmark/final_132_v1/questions_only.jsonl"
REPORTS = ROOT / "presentation_20261118/thesis132/csv_reports"
MODELS = tuple(f"B{i}" for i in range(1, 9))
MISSING = ""
COUNTER_FIELDS = ("num_direct_causes", "num_multi_hop_paths", "num_retrieved_documents")
TIME_FIELDS = ("latency_ms", "refinement_ms", "retrieval_ms",
               "rewriting_ms", "generation_ms", "validation_ms")
PER_QUERY_DERIVED = (
    "candidate_evidence_available", "causal_evidence_available",
    "answer_length_words", "latency_for_answered_ms",
    "latency_for_abstained_ms", "evidence_without_answer",
    "answer_without_recorded_evidence",
)
SUMMARY_FIELDS = (
    "model", "model_name", "mode", "queries", "ok", "abstained", "errors",
    "candidate_evidence_count", "candidate_evidence_rate",
    "causal_evidence_count", "causal_evidence_rate",
    "answer_present_count", "answer_present_rate",
    "answers_without_recorded_evidence",
    "total_CAUSES_edges", "total_causal_paths",
    "mean_direct_causes_per_question", "mean_paths_per_question",
    "mean_path_hops_for_questions_with_paths",
    "mean_latency_ms", "median_latency_ms", "p95_latency_ms",
    "min_latency_ms", "max_latency_ms",
    "mean_latency_ok_ms", "median_latency_ok_ms",
    "mean_latency_abstained_ms", "median_latency_abstained_ms",
    "mean_refinement_ms", "mean_retrieval_ms", "mean_rewriting_ms",
    "mean_generation_ms", "mean_validation_ms",
    "wall_time_seconds", "run_throughput_qps",
    "detector_observations", "detector_flagged", "detector_flag_rate_NOT_gold",
    "validator_observations", "validator_passed",
    "mean_answer_words_if_present", "mean_recorded_feedback_iterations",
    "medical_accuracy", "clinical_groundedness", "source_type",
)
METRIC_DEFS = (
    ("Queries Evaluated", "queries", "count", "Fixed, frozen 132-question development set"),
    ("Successful Executions", "ok", "count", "Pipeline status, not answer correctness"),
    ("Abstained Executions", "abstained", "count", "Explicit abstention, not execution error"),
    ("Execution Errors", "errors", "count", "Pipeline errors"),
    ("Retrieval Success Rate", "candidate_evidence_rate", "fraction", "Candidate evidence present, not correct/relevant retrieval"),
    ("Causal Retrieval Coverage", "causal_evidence_rate", "fraction", "Any recorded causal edge or path, not gold causal accuracy"),
    ("Causal Chain Depth", "mean_path_hops_for_questions_with_paths", "mean_hops", "Mean of reported per-query mean path hops only for queries with paths"),
    ("Semantic Refinement Score", None, "unavailable", "Not calculated: no independently validated refinement metric"),
    ("Context Relevance", None, "unavailable", "Not calculated: no independently reviewed relevance reference"),
    ("Groundedness", None, "unavailable", "Not calculated: no independently verified claim/evidence reference"),
    ("Average Answer Length", "mean_answer_words_if_present", "words", "Among nonempty answers only"),
    ("Hallucination Rate", None, "unavailable", "Not medical gold; automated detector flag rate reported separately"),
    ("Detector Flag Rate (Not Gold)", "detector_flag_rate_NOT_gold", "fraction", "Fraction flagged among nonempty detector observations only"),
    ("Retrieval Success Count", "candidate_evidence_count", "count", "Queries with candidate evidence"),
    ("Answer Present Rate", "answer_present_rate", "fraction", "Nonempty answer text, not correct answer"),
    ("Answers Without Recorded Evidence", "answers_without_recorded_evidence", "count", "Recorded answers with zero candidate document/edge/path"),
    ("Average Direct Causes", "mean_direct_causes_per_question", "CA US ES/query", "Reported CAUSES edges per all 132 queries; not validated facts"),
    ("Total Direct Causes", "total_CAUSES_edges", "CA US ES", "Recorded CAUSES edges, not clinically adjudicated"),
    ("Total Causal Paths", "total_causal_paths", "paths", "Reported path count, may contain repeated evidence"),
    ("Total Execution Time (s)", "wall_time_seconds", "seconds", "Recorded per-model wall time, includes measured initialization"),
    ("Average Latency (s)", "mean_latency_ms", "seconds", "All 132 queries, including abstentions; converted ms/1000"),
    ("Average Latency (ms)", "mean_latency_ms", "ms", "All statuses"),
    ("Minimum Latency (ms)", "min_latency_ms", "ms", "All statuses"),
    ("Maximum Latency (ms)", "max_latency_ms", "ms", "All statuses"),
    ("Median Latency (ms)", "median_latency_ms", "ms", "All statuses"),
    ("P95 Latency (ms)", "p95_latency_ms", "ms", "All statuses; interpolated percentile"),
    ("Mean Latency if OK (ms)", "mean_latency_ok_ms", "ms", "Conditioned on status=ok, avoids abstention confounding"),
    ("Mean Latency if Abstained (ms)", "mean_latency_abstained_ms", "ms", "Conditioned on status=abstained"),
    ("Average Throughput (QPS)", "run_throughput_qps", "queries/second", "132 divided by recorded wall seconds; NOT mean reciprocal per-query latency"),
    ("Minimum Throughput (QPS)", None, "unavailable", "Not measured separately; do not substitute per-query reciprocal latency"),
    ("Maximum Throughput (QPS)", None, "unavailable", "Not measured separately; do not substitute per-query reciprocal latency"),
    ("Refinement Success Rate", None, "unavailable", "No independent refinement success reference"),
    ("Mean Recorded Feedback Iterations", "mean_recorded_feedback_iterations", "count", "Legacy B7/B8 adapters record zero; not proof no PPO action occurred"),
)
# Correct human-readable causal edge units while keeping metric names frozen.
CAUSAL_UNITS = {"Average Direct Causes": "CAUSES edges/query",
                "Total Direct Causes": "CAUSES edges"}


def _check(ok, message):
    if not ok:
        raise ValueError("THESIS132_EVALUATION: " + message)


def _hash(path):
    sha = hashlib.sha256()
    with path.open("rb") as handle:
        for part in iter(lambda: handle.read(1024 * 1024), b""):
            sha.update(part)
    return sha.hexdigest()


def _number(raw, label, *, integer=False):
    if raw is None or raw == "":
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        raise ValueError(f"Invalid numeric {label}: {raw!r}")
    _check(math.isfinite(value), f"Non-finite {label}")
    if integer:
        _check(value >= 0 and value.is_integer(), f"Invalid count {label}")
        return int(value)
    return value


def _boolean(raw, label):
    if raw is None or raw == "":
        return None
    if raw in ("True", "true", "1", True):
        return True
    if raw in ("False", "false", "0", False):
        return False
    raise ValueError(f"Invalid boolean {label}: {raw!r}")


def _mean(items):
    items = [x for x in items if x is not None]
    return statistics.mean(items) if items else None


def _quantile(items, q):
    a = sorted(x for x in items if x is not None)
    if not a:
        return None
    p = (len(a) - 1) * q
    lo = int(p)
    hi = min(lo + 1, len(a) - 1)
    return a[lo] + (p - lo) * (a[hi] - a[lo])


def _write_csv(path, fields, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        _check(reader.fieldnames is not None, "CSV has no header: " + str(path))
        _check(len(set(reader.fieldnames)) == len(reader.fieldnames),
               "Duplicate CSV headers: " + str(path))
        return list(reader), reader.fieldnames


def _resolve_dir(value):
    path = Path(value).expanduser().resolve()
    _check(path.is_relative_to(ROOT), "Input/output must be inside the repository")
    return path


def _load(input_dir):
    base = _resolve_dir(input_dir)
    manifest_file = base / "export_manifest.json"
    _check(manifest_file.is_file(), "Missing export_manifest.json")
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    _check(manifest.get("schema") == "THESIS132_BATCH_CSV_EXPORT_V1",
           "Not a versioned 132-query batch CSV export")
    _check(manifest.get("status") == "COMPLETE_8_MODELS" and
           manifest.get("models_present") == list(MODELS) and
           not manifest.get("models_missing") and
           not manifest.get("models_with_errors"),
           "Need a complete, zero-execution-error B1-B8 export")
    frozen_hash = _hash(RELEASE)
    # The exported hash covers the question-only byte representation; those
    # exact bytes were previously frozen and must remain unchanged.
    _check(manifest.get("dataset_sha256") == frozen_hash,
           "Frozen question file SHA does not match exported session")
    release_rows = [json.loads(line) for line in RELEASE.read_text(
        encoding="utf-8").splitlines() if line.strip()]
    _check(len(release_rows) == 132, "Canonical question count !=132")
    ids = [q["query_id"] for q in release_rows]
    _check(len(set(ids)) == 132, "Duplicate released question IDs")
    checksums = manifest.get("generated_csv_sha256", {})
    summary_file = base / "pipeline_summary_132.csv"
    _check(summary_file.is_file() and _hash(summary_file) == checksums.get(
        "pipeline_summary_132.csv"), "Summary CSV integrity mismatch")
    raw_summaries, summary_fields = _read_csv(summary_file)
    _check([r["model"] for r in raw_summaries] == list(MODELS),
           "Original pipeline summary lacks B1-B8 order")
    summary = {r["model"]: r for r in raw_summaries}
    data = {}
    for model in MODELS:
        path = base / f"{model}_queries_132.csv"
        _check(path.is_file() and _hash(path) == checksums.get(path.name),
               "Missing or modified model CSV: " + model)
        rows, fields = _read_csv(path)
        needed = {"model", "query_id", "query", "query_type", "medical_domain",
                  "semantic_group", "fold", "status", "error_message",
                  "retrieval_success", "answer_present", "num_direct_causes",
                  "num_multi_hop_paths", "num_retrieved_documents", "final_response",
                  "causal_chain_depth_hops_mean", "latency_ms", "is_hallucination",
                  "validator_passed", "feedback_iterations_recorded", "run_id"}
        needed.update(TIME_FIELDS)
        _check(needed.issubset(fields), "Missing required CSV columns: " + model)
        _check(len(rows) == 132, "Wrong row count: " + model)
        _check([r["query_id"] for r in rows] == ids,
               "Question ID/order mismatch: " + model)
        _check(all(r["model"] == model for r in rows),
               "Model identity mismatch: " + model)
        _check(all(r["query"] == q["query"] for r, q in zip(rows, release_rows)),
               "Question wording mismatch: " + model)
        _check(all(r["run_id"] == summary[model]["run_id"] for r in rows),
               "Run ID mismatch: " + model)
        _check(summary[model].get("input_sha256") == frozen_hash,
               "Summary dataset SHA mismatch: " + model)
        _check(summary[model].get("code_commit") == manifest.get(
            "session_configuration", {}).get("code_commit"),
            "Code commit mismatch: " + model)
        _check(summary[model].get("mode") == manifest.get(
            "session_configuration", {}).get("mode"), "Execution mode mismatch")
        # Metadata must match for paired and category analyses. These labels
        # never serve as verified answer/evidence ground truth.
        if data:
            base_rows = next(iter(data.values()))
            for old, new in zip(base_rows, rows):
                for label in ("query_type", "medical_domain", "semantic_group", "fold"):
                    _check(old[label] == new[label],
                           "Inconsistent metadata " + label + ": " + model)
        data[model] = rows
    return base, manifest, summary, data, fields


def _derive(model, rows):
    parsed = []
    for row in rows:
        status = row["status"]
        _check(status in ("ok", "abstained", "error"), "Invalid status")
        evidence = _boolean(row["retrieval_success"], "retrieval_success")
        answer = _boolean(row["answer_present"], "answer_present")
        _check(evidence is not None and answer is not None, "Missing evidence/answer marker")
        counts = {field: _number(row[field], field, integer=True)
                  for field in COUNTER_FIELDS}
        _check(all(v is not None for v in counts.values()), "Missing evidence counts")
        _check(evidence == any(v > 0 for v in counts.values()),
               "Candidate evidence/count conflict: " + model)
        _check(answer == bool(row["final_response"].strip()),
               "Answer presence/text conflict: " + model)
        causal = counts["num_direct_causes"] + counts["num_multi_hop_paths"] > 0
        times = {key: _number(row[key], key) for key in TIME_FIELDS}
        _check(times["latency_ms"] is not None and times["latency_ms"] >= 0,
               "Missing/negative latency: " + model)
        _check(all(v is None or v >= 0 for v in times.values()),
               "Negative stage time")
        detector = _boolean(row["is_hallucination"], "is_hallucination")
        validator = _boolean(row["validator_passed"], "validator_passed")
        depth = _number(row["causal_chain_depth_hops_mean"], "path depth")
        _check(depth is None or depth >= 1, "Invalid path hops")
        if counts["num_multi_hop_paths"]:
            _check(depth is not None, "Paths reported without mean depth")
        feedback = _number(row["feedback_iterations_recorded"], "feedback_iterations_recorded", integer=True)
        _check(feedback is not None, "Missing feedback iterations")
        words = len(row["final_response"].split()) if answer else None
        entry = {
            **row,
            "candidate_evidence_available": evidence,
            "causal_evidence_available": causal,
            "answer_length_words": words,
            "latency_for_answered_ms": times["latency_ms"] if status == "ok" else None,
            "latency_for_abstained_ms": times["latency_ms"] if status == "abstained" else None,
            "evidence_without_answer": evidence and not answer,
            "answer_without_recorded_evidence": answer and not evidence,
        }
        entry["_detector"] = detector
        entry["_validator"] = validator
        entry["_timings"] = times
        entry["_counts"] = counts
        entry["_depth"] = depth
        entry["_feedback"] = feedback
        parsed.append(entry)
    return parsed


def _aggregate(model, rows, original):
    n = len(rows)
    ok = sum(r["status"] == "ok" for r in rows)
    abstained = sum(r["status"] == "abstained" for r in rows)
    errors = sum(r["status"] == "error" for r in rows)
    evidence = sum(r["candidate_evidence_available"] for r in rows)
    causal = sum(r["causal_evidence_available"] for r in rows)
    answers = sum(bool(r["answer_present"] == "True") for r in rows)
    edges = sum(r["_counts"]["num_direct_causes"] for r in rows)
    paths = sum(r["_counts"]["num_multi_hop_paths"] for r in rows)
    det = [r["_detector"] for r in rows if r["_detector"] is not None]
    validators = [r["_validator"] for r in rows if r["_validator"] is not None]
    measured = [r["_timings"]["latency_ms"] for r in rows]
    wall = _number(original["wall_time_seconds"], "wall_time_seconds")
    _check(wall is not None and wall > 0, "Missing measured wall time: " + model)
    expected = {
        "status_ok": ok, "status_abstained": abstained, "status_error": errors,
        "candidate_evidence_count": evidence,
        "answer_present_count": answers,
        "recorded_causal_edge_count": edges,
        "recorded_causal_path_count": paths,
        "detector_observations": len(det), "detector_flagged": sum(det),
        "validator_observations": len(validators), "validator_passed": sum(validators),
    }
    for key, count in expected.items():
        _check(_number(original[key], key, integer=True) == count,
               f"Original summary mismatch: {model}.{key}")
    _check(not errors and ok + abstained == n,
           "Only zero-error complete experiment supported")
    return {
        "model": model, "model_name": original["model_name"],
        "mode": original["mode"], "queries": n, "ok": ok,
        "abstained": abstained, "errors": errors,
        "candidate_evidence_count": evidence, "candidate_evidence_rate": evidence/n,
        "causal_evidence_count": causal, "causal_evidence_rate": causal/n,
        "answer_present_count": answers, "answer_present_rate": answers/n,
        "answers_without_recorded_evidence": sum(r["answer_without_recorded_evidence"] for r in rows),
        "total_CAUSES_edges": edges, "total_causal_paths": paths,
        "mean_direct_causes_per_question": edges/n,
        "mean_paths_per_question": paths/n,
        "mean_path_hops_for_questions_with_paths": _mean(
            r["_depth"] for r in rows if r["_counts"]["num_multi_hop_paths"]),
        "mean_latency_ms": _mean(measured), "median_latency_ms": _quantile(measured, .5),
        "p95_latency_ms": _quantile(measured, .95),
        "min_latency_ms": min(measured), "max_latency_ms": max(measured),
        "mean_latency_ok_ms": _mean(r["_timings"]["latency_ms"] for r in rows if r["status"] == "ok"),
        "median_latency_ok_ms": _quantile(
            [r["_timings"]["latency_ms"] for r in rows if r["status"] == "ok"], .5),
        "mean_latency_abstained_ms": _mean(
            r["_timings"]["latency_ms"] for r in rows if r["status"] == "abstained"),
        "median_latency_abstained_ms": _quantile(
            [r["_timings"]["latency_ms"] for r in rows if r["status"] == "abstained"], .5),
        **{"mean_"+key+"_ms": _mean(r["_timings"][key+"_ms"] for r in rows)
           for key in ("refinement", "retrieval", "rewriting", "generation", "validation")},
        "wall_time_seconds": wall, "run_throughput_qps": n/wall,
        "detector_observations": len(det), "detector_flagged": sum(det),
        "detector_flag_rate_NOT_gold": _mean(det),
        "validator_observations": len(validators), "validator_passed": sum(validators),
        "mean_answer_words_if_present": _mean(
            r["answer_length_words"] for r in rows if r["answer_length_words"] is not None),
        "mean_recorded_feedback_iterations": _mean(r["_feedback"] for r in rows),
        "medical_accuracy": None, "clinical_groundedness": None,
        "source_type": original["evidence_source_type"],
    }


def _metric_rows(summary):
    output = []
    for name, key, unit, note in METRIC_DEFS:
        value = summary.get(key) if key else None
        if name == "Average Latency (s)" and value is not None:
            value /= 1000
        output.append({"Metric": name, "Value": value,
                       "Unit": CAUSAL_UNITS.get(name, unit),
                       "Notes": note})
    return output


def _categorize(data, field):
    out = []
    for model in MODELS:
        groups = defaultdict(list)
        for row in data[model]:
            groups[row[field]].append(row)
        for category, rows in sorted(groups.items()):
            n = len(rows)
            out.append({
                "model": model, field: category, "questions": n,
                "ok": sum(r["status"] == "ok" for r in rows),
                "abstained": sum(r["status"] == "abstained" for r in rows),
                "candidate_evidence_count": sum(r["candidate_evidence_available"] for r in rows),
                "candidate_evidence_rate": _mean(r["candidate_evidence_available"] for r in rows),
                "causal_evidence_rate": _mean(r["causal_evidence_available"] for r in rows),
                "answer_present_rate": _mean(_boolean(r["answer_present"], "answer_present") for r in rows),
                "mean_latency_ms": _mean(r["_timings"]["latency_ms"] for r in rows),
                "mean_latency_ok_ms": _mean(
                    r["_timings"]["latency_ms"] for r in rows if r["status"] == "ok"),
            })
    return out


def _paired(data):
    pairs = []
    a, b = data["B7"], data["B8"]
    for old, new in zip(a, b):
        _check(old["query_id"] == new["query_id"], "Misaligned B7/B8 query IDs")
        ea, eb = old["candidate_evidence_available"], new["candidate_evidence_available"]
        classification = ("both" if ea and eb else "B7_only" if ea
                          else "B8_only" if eb else "neither")
        pairs.append({
            "query_id": old["query_id"], "query": old["query"],
            "query_type": old["query_type"], "medical_domain": old["medical_domain"],
            "fold": old["fold"], "B7_status": old["status"], "B8_status": new["status"],
            "B7_matched_concept": old["matched_concept"],
            "B8_matched_concept": new["matched_concept"],
            "B7_refined_query": old["refined_query_text"],
            "B8_refined_query": new["refined_query_text"],
            "B7_candidate_evidence": ea, "B8_candidate_evidence": eb,
            "evidence_category": classification,
            "B7_direct_causes": old["_counts"]["num_direct_causes"],
            "B8_direct_causes": new["_counts"]["num_direct_causes"],
            "delta_direct_causes": (
                new["_counts"]["num_direct_causes"]-old["_counts"]["num_direct_causes"]),
            "B7_causal_paths": old["_counts"]["num_multi_hop_paths"],
            "B8_causal_paths": new["_counts"]["num_multi_hop_paths"],
            "delta_causal_paths": (
                new["_counts"]["num_multi_hop_paths"]-old["_counts"]["num_multi_hop_paths"]),
            "B7_latency_ms": old["_timings"]["latency_ms"],
            "B8_latency_ms": new["_timings"]["latency_ms"],
            "B8_minus_B7_latency_ms": (
                new["_timings"]["latency_ms"]-old["_timings"]["latency_ms"]),
            "B7_answer_present": _boolean(old["answer_present"], "answer_present"),
            "B8_answer_present": _boolean(new["answer_present"], "answer_present"),
            "B7_answer_without_evidence": old["answer_without_recorded_evidence"],
            "B8_answer_without_evidence": new["answer_without_recorded_evidence"],
            "B7_detector_flag": old["_detector"],
            "B8_detector_flag": new["_detector"],
            "B7_validator_pass": old["_validator"],
            "B8_validator_pass": new["_validator"],
        })
    categories = {k:sum(r["evidence_category"]==k for r in pairs)
                  for k in ("both","B7_only","B8_only","neither")}
    n_disc = categories["B7_only"] + categories["B8_only"]
    k = min(categories["B7_only"], categories["B8_only"])
    mcnemar_p = (min(1.0, 2 * sum(math.comb(n_disc,i) for i in range(k+1)) /
                     (2**n_disc)) if n_disc else 1.0)
    stats = {
        "questions": 132,
        "candidate_evidence_pair_counts": categories,
        "B8_minus_B7_net_evidence_questions": categories["B8_only"]-categories["B7_only"],
        "B8_minus_B7_evidence_percentage_points": (
            100 * (categories["B8_only"]-categories["B7_only"])/132),
        "B8_minus_B7_mean_paired_latency_ms": _mean(
            r["B8_minus_B7_latency_ms"] for r in pairs),
        "B8_minus_B7_median_paired_latency_ms": _quantile(
            [r["B8_minus_B7_latency_ms"] for r in pairs], .5),
        "B8_slower_query_count": sum(r["B8_minus_B7_latency_ms"]>0 for r in pairs),
        "B7_slower_query_count": sum(r["B8_minus_B7_latency_ms"]<0 for r in pairs),
        "exact_discordant_pair_p_DESCRIPTIVE_ONLY": mcnemar_p,
        "PPO_causal_effect_established": False,
        "medically_verified_relevance_established": False,
    }
    return pairs, stats


def analyze(input_dir, output_dir=None):
    base, manifest, existing_summary, raw, source_fields = _load(input_dir)
    parsed = {model: _derive(model, raw[model]) for model in MODELS}
    aggregated = {model: _aggregate(model, parsed[model], existing_summary[model])
                  for model in MODELS}
    pair, paired_stats = _paired(parsed)
    kinds = _categorize(parsed, "query_type")
    domains = _categorize(parsed, "medical_domain")
    if output_dir:
        output = _resolve_dir(output_dir)
        _check(not output.exists(), "Refusing to overwrite existing analysis folder")
    else:
        output = base / ("evaluation_132_" +
                         datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") +
                         "_" + uuid.uuid4().hex[:6])
    output.mkdir(parents=True, exist_ok=False)
    generated = {}
    def emit(filename, fields, rows):
        path = output / filename
        _write_csv(path, list(fields), rows)
        generated[filename] = _hash(path)
    for model in MODELS:
        cleaned = [{key: r.get(key) for key in source_fields+list(PER_QUERY_DERIVED)}
                   for r in parsed[model]]
        emit(f"evaluation_metrics_{model}_132.csv",
             source_fields + list(PER_QUERY_DERIVED), cleaned)
        emit(f"evaluation_summary_{model}_132.csv",
             ["Metric","Value","Unit","Notes"], _metric_rows(aggregated[model]))
    emit("evaluation_summary_all_models_132.csv", SUMMARY_FIELDS,
         [aggregated[model] for model in MODELS])
    metric_fields = ["Metric", "Unit"] + list(MODELS) + ["B8_minus_B7", "Notes"]
    comparison = []
    for i, (metric, key, unit, note) in enumerate(METRIC_DEFS):
        row = {"Metric":metric, "Unit":CAUSAL_UNITS.get(metric,unit), "Notes":note}
        for model in MODELS:
            row[model] = _metric_rows(aggregated[model])[i]["Value"]
        a, b = row["B7"], row["B8"]
        row["B8_minus_B7"] = b-a if isinstance(a,(int,float)) and isinstance(
            b,(int,float)) else None
        comparison.append(row)
    emit("comparison_metrics_B1_B8_132.csv", metric_fields, comparison)
    emit("comparison_by_query_type_132.csv", list(kinds[0]), kinds)
    emit("comparison_by_medical_domain_132.csv", list(domains[0]), domains)
    emit("comparison_B7_vs_B8_paired_132.csv", list(pair[0]), pair)
    changes = [r for r in pair if r["evidence_category"] in ("B7_only","B8_only")]
    emit("B7_B8_evidence_changed_questions_132.csv", list(pair[0]), changes)
    notes = [
        "All 132 questions were previously exposed to G1 development: not independent clinical gold.",
        "B1-B4 use provisional candidate documents, while B5-B8 use G0 graph evidence: unlike evidence bases.",
        "Candidate evidence returned is NOT retrieval accuracy, evidence relevance or verified medical correctness.",
        "Hallucination detector flags are automated output, NOT ground-truth hallucination rate.",
        "Legacy B7/B8 output adapters do not record their actual refinement iteration counts.",
        "B5/B6 abstain frequently; compare latency within status=ok as well as across all questions.",
        "Actual throughput is 132 / run wall time; never average 1/query_latency as run throughput.",
        "No verified citation precision, groundedness, clinical answer accuracy, SRS or gold label metrics available.",
        "Pairwise B7/B8 differences are descriptive, not causal attribution to PPO alone.",
    ]
    report = {
        "schema": "THESIS132_EVALUATION_V1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_export": str(base.relative_to(ROOT)),
        "source_manifest_sha256": _hash(base/"export_manifest.json"),
        "dataset_sha256": manifest["dataset_sha256"],
        "model_count": 8, "records_per_model": 132,
        "total_model_query_records": 1056,
        "paired_B7_B8": paired_stats,
        "per_model_summary": aggregated,
        "generated_sha256": generated,
        "clinical_gold": False, "warnings": notes,
    }
    report_path = output / "analysis_manifest.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                           encoding="utf-8")
    print(json.dumps({
        "evaluation_output": str(output),
        "models": list(MODELS), "records": 1056,
        "B7_B8_net_evidence_questions": paired_stats["B8_minus_B7_net_evidence_questions"],
        "B7_B8_mean_latency_delta_ms": paired_stats["B8_minus_B7_mean_paired_latency_ms"],
        "medical_gold": False,
    }, indent=2))
    return output, report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", type=Path, required=True,
                   help="Exact batch132_csv_* directory with export_manifest.json")
    p.add_argument("--output-dir", type=Path,
                   help="New, nonexistent destination folder inside this repository")
    args = p.parse_args()
    analyze(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
