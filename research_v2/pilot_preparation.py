"""Prepare a draft-only causal medical QA pilot for human source review.

Pilot questions and metadata are NOT verified medical ground truth.
"""

import argparse
import json
from pathlib import Path

from .medical_benchmark import audit_benchmark, file_sha256, load_benchmark


PILOT_COUNTS = {
    "direct": 15,
    "paraphrased": 15,
    "indirect": 15,
    "multi_hop": 15,
    "comparative": 12,
    "risk_factor": 12,
    "ambiguous": 8,
    "unsupported": 8,
}


def validate_pilot(path: Path):
    records = load_benchmark(path)
    audit = audit_benchmark(records)

    if len(records) != 100:
        raise ValueError(f"Pilot requires 100 records; found {len(records)}")
    if audit["query_type_counts"] != PILOT_COUNTS:
        raise ValueError("Pilot query-type distribution does not match the protocol")

    for record in records:
        if record.review_status != "draft" or record.split is not None:
            raise ValueError(f"Pilot must remain an unsplit draft: {record.query_id}")
        if record.answerable is not None or record.gold_answer is not None:
            raise ValueError(f"Draft contains premature gold labels: {record.query_id}")
        if record.gold_entities or record.gold_causal_edges or record.gold_paths:
            raise ValueError(f"Draft contains premature structured gold: {record.query_id}")
        if record.evidence_sources:
            raise ValueError(
                "Draft contains source entries: record these in a reviewed edition: "
                + record.query_id
            )
        if not record.medical_domain:
            raise ValueError(f"Missing medical_domain: {record.query_id}")
        if not record.question_family or not record.concept_family:
            raise ValueError(f"Missing family IDs: {record.query_id}")

    paired_families = {}
    for record in records:
        if record.query_type in {"direct", "paraphrased"}:
            paired_families.setdefault(record.question_family, set()).add(
                record.query_type
            )

    if len(paired_families) != 15 or any(
        types != {"direct", "paraphrased"}
        for types in paired_families.values()
    ):
        raise ValueError("Expected 15 paired direct/paraphrased question families")

    profile = dict(audit)
    profile.update({
        "data_role": "unreviewed_question_drafts",
        "question_family_pair_count": len(paired_families),
        "dataset_sha256": file_sha256(path),
        "gold_labels_created": False,
        "medical_accuracy_verified": False,
        "final_splits_assigned": False,
    })
    return records, profile


def write_pilot_inputs(source: Path, output_dir: Path):
    records, profile = validate_pilot(source)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)

    # No target concepts, reviewer IDs, metadata, labels or evidence
    # may flow into the query-only inference file.
    questions = [
        {"query_id": record.query_id, "query": record.query}
        for record in records
    ]
    (output_dir / "questions_only.jsonl").write_text(
        "".join(
            json.dumps(question, ensure_ascii=False) + "\n"
            for question in questions
        ),
        encoding="utf-8",
    )
    (output_dir / "pilot_profile.json").write_text(
        json.dumps(profile, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return profile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    if args.output_dir is None:
        _, profile = validate_pilot(args.input)
    else:
        profile = write_pilot_inputs(args.input, args.output_dir)

    print(json.dumps(profile, indent=2))


if __name__ == "__main__":
    main()
