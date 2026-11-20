"""Immutable 132-question development release for B1–B9.

This validates provenance, exact question bytes/order, family groups and
source-ID references. It cannot certify medical answer correctness.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

from .benchmark_v2_validation import (
    DATASET, PILOT, EXT, SOURCE, validate as validate_benchmark,
)
from .experiment_runner import PROJECT_ROOT as ROOT, file_sha256

RELEASE = ROOT / "data/benchmark/final_132_v1"
INPUT = RELEASE / "questions_only.jsonl"
EXPECTED_TYPES = {
    "direct": 19, "paraphrased": 19, "indirect": 19, "multi_hop": 19,
    "comparative": 16, "risk_factor": 16, "ambiguous": 12, "unsupported": 12,
}
METADATA_FIELDS = (
    "query_id", "cohort", "query_type", "medical_domain", "difficulty_legacy",
    "task_objective", "expected_answerability", "expected_response_type",
    "semantic_group", "fold", "source_reference_ids_json", "split",
    "annotation_review_status",
)


def _require(condition: bool, why: str) -> None:
    if not condition:
        raise ValueError("THESIS132_RELEASE_VALIDATION_FAILED: " + why)


def _load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _git_blob_sha(path: Path) -> str:
    raw = path.read_bytes()
    return hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()


def verify_release(base: Path = RELEASE) -> dict:
    """Validate all tracked release files against original benchmark v2."""
    base = Path(base)
    report, rows, source_by_id = validate_benchmark()
    m = json.loads((base / "manifest.json").read_text(encoding="utf-8"))
    contract = json.loads((base / "pipeline_contract.json").read_text(encoding="utf-8"))
    url_audit = json.loads(
        (base / "source_url_access_audit_20261121.json").read_text(encoding="utf-8")
    )
    query_file = base / "questions_only.jsonl"
    query_csv = base / "questions_only.csv"
    metadata_file = base / "metadata_review_only.csv"

    _require(m["schema"] == "THESIS132_FINAL_QUERY_INPUT_V1", "Unexpected schema")
    _require(m["query_count"] == 132, "Manifest count")
    _require(m["source_dataset_sha256"] == file_sha256(DATASET),
             "Original v2 annotations changed after release")
    _require(m["source_v2_git_blob_sha"] == _git_blob_sha(DATASET),
             "V2 source blob changed")
    for key, path in (
        ("original_pilot_git_blob_sha", PILOT),
        ("original_extension_git_blob_sha", EXT),
        ("original_source_catalog_git_blob_sha", SOURCE),
    ):
        _require(m[key] == _git_blob_sha(path), "Original source changed: " + key)
    for key, path in (
        ("query_only_sha256", query_file),
        ("query_only_csv_sha256", query_csv),
        ("metadata_csv_sha256", metadata_file),
    ):
        _require(m[key] == file_sha256(path), "Release hash mismatch: " + key)

    # The runtime file must match the established thesis132.freeze() serializer.
    serialized = "".join(json.dumps(
        {"query_id": r["query_id"], "query": r["query"]},
        ensure_ascii=False, separators=(",", ":"
        )) + "\n" for r in rows)
    _require(query_file.read_bytes() == serialized.encode("utf-8"),
             "Query text, order, encoding or field leakage differs from v2")
    questions = _load_jsonl(query_file)
    _require(len(questions) == 132, "Not 132 inference questions")
    _require(all(set(q) == {"query_id", "query"} for q in questions),
             "Inference contains non-query/answer/target metadata")
    _require(len({r["query_id"] for r in questions}) == 132, "Duplicate query_id")
    normalized = {" ".join(q["query"].casefold().split()) for q in questions}
    _require(len(normalized) == 132, "Normalized duplicate questions")
    _require(dict(Counter(r["query_type"] for r in rows)) == EXPECTED_TYPES,
             "Original question-type distribution changed")
    _require(m["query_types"] == EXPECTED_TYPES,
             "Manifest question types do not match source")

    from .thesis132 import _grouped_folds
    folds, fold_sizes = _grouped_folds(rows)
    _require(fold_sizes == m["fold_counts"] == [27, 27, 26, 26, 26],
             "Previously frozen folds have drifted")
    with metadata_file.open(newline="", encoding="utf-8") as stream:
        csv_reader = csv.DictReader(stream)
        _require(tuple(csv_reader.fieldnames or ()) == METADATA_FIELDS,
                 "Metadata columns changed")
        meta = list(csv_reader)
    _require(len(meta) == 132, "Metadata length differs")
    for q, expected in zip(meta, rows):
        for k in METADATA_FIELDS:
            actual = q[k]
            desired = (
                str(folds[expected["query_id"]])
                if k == "fold" else
                json.dumps(expected["source_reference_ids"], ensure_ascii=False,
                           separators=(",", ":"))
                if k == "source_reference_ids_json" else str(expected[k])
            )
            _require(actual == desired, f"Metadata mismatch for {expected['query_id']} {k}")
    with query_csv.open(newline="", encoding="utf-8") as stream:
        csv_reader = csv.DictReader(stream)
        _require(tuple(csv_reader.fieldnames or ()) == ("query_id", "query"),
                 "CSV inference field leakage")
        csv_rows = list(csv_reader)
    _require(csv_rows == questions, "CSV and JSONL inputs differ")

    _require(url_audit["total"] == 24 and
             url_audit["opened_in_web_reader"] == 22 and
             url_audit["reader_http_403"] == 2,
             "Recorded web source access summary changed")
    access = url_audit["sources"]
    _require(len(access) == 24 and
             len({s["source_id"] for s in access}) == 24,
             "Source URL access audit incomplete")
    for s in access:
        sid = s["source_id"]
        _require(sid in source_by_id, "Unknown URL audit source " + sid)
        _require(s["original_url"] == source_by_id[sid]["url"],
                 "Source URL was silently remapped: " + sid)
        _require(s["source_claim_level_verified"] is False,
                 "Source URL access incorrectly promoted to medical gold")
    _require(sum(s["reader_status"] == "web_reader_opened_2026-11-21"
                 for s in access) == 22, "Web-reader-open status count mismatch")
    restricted = {s["source_id"] for s in access
                  if s["reader_status"] == "web_reader_http_403_search_index_found"}
    _require(restricted == {"MIGRAINE_NIH", "EPILEPSY_NINDS"},
             "Reader-restricted source registry changed")

    _require(contract["input_sha256"] == m["query_only_sha256"],
             "Different pipeline input hash")
    _require(contract["rows"] == 132 and
             contract["exact_fields"] == ["query_id", "query"],
             "Pipeline input contract permits leakage")
    _require(set(contract["all_pipelines"]) ==
             {f"B{i}" for i in range(1, 10)}, "B1–B9 contract incomplete")
    _require(set(contract["run_spec"]) == set(contract["all_pipelines"]),
             "Missing pipeline runner specification")
    _require(m["independent_clinical_gold"] is False and
             m["independent_holdout"] is False and
             report["clinical_gold_questions"] == 0,
             "Falsely marked independent clinical gold or test")
    _require(all(r["gold_answer"] is None and
                 r["independent_review_count"] == 0 for r in rows),
             "Unadjudicated medical reference was changed")

    return {
        "status": "PASS_FROZEN_132_INPUT_AND_METADATA_NOT_CLINICAL_GOLD",
        "query_count": 132,
        "pilot100": 100, "extension32": 32,
        "source_screened_questions": 122,
        "negative_questions_without_topical_sources": 10,
        "query_sha256": file_sha256(query_file),
        "v2_source_sha256": file_sha256(DATASET),
        "family_groups": len({r["semantic_group"] for r in rows}),
        "fold_counts": fold_sizes,
        "source_catalog_count": 24,
        "source_urls_opened_by_web_reader": 22,
        "source_urls_reader_restricted": 2,
        "independent_medical_gold": False,
        "historically_unseen_holdout": False,
        "pipelines": contract["all_pipelines"],
    }


def verify_model_records(records_file: Path, model: str, base: Path = RELEASE) -> dict:
    """Fail if a B1–B9 experiment skipped, reordered or modified questions.

    A recorded execution error is not an abstention and fails the ready gate.
    """
    report = verify_release(base)
    _require(model in report["pipelines"], "Unknown model: " + model)
    reference = _load_jsonl(base / "questions_only.jsonl")
    submitted = _load_jsonl(Path(records_file))
    _require(len(submitted) == 132, "Missing or extra pipeline records")
    for idx, (observed, expected) in enumerate(zip(submitted, reference), 1):
        _require(observed.get("query_id") == expected["query_id"],
                 f"Record {idx} query ID/order mismatch")
        question = observed.get("original_query", observed.get("query"))
        _require(question == expected["query"],
                 f"Record {idx} question text mismatch")
        _require(observed.get("status") != "error",
                 f"Record {idx} execution error; do not relabel as abstention")
    return {
        "status": "PASS_COMPLETE_ORDERED_132_MODEL_RECORDS",
        "model": model, "records": 132,
        "query_sha256": report["query_sha256"],
        "medical_accuracy": None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-records", type=Path,
                        help="Validate a B1–B9 records.jsonl from a local run")
    parser.add_argument("--model", choices=[f"B{i}" for i in range(1, 10)])
    args = parser.parse_args()
    if args.check_records and not args.model:
        parser.error("--check-records needs --model B1 ... B9")
    summary = (verify_model_records(args.check_records, args.model)
               if args.check_records else verify_release())
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
