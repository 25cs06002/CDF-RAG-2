"""Phase 2 human-review worksheets with immutable candidate provenance.

Generates editable CSV worksheets for candidate claims and question-level
answerability. Validation checks workflow completeness and tampering, not
clinical accuracy. It NEVER exports gold labels automatically.
"""

import argparse
import csv
import io
import json
from collections import Counter
from datetime import date
from pathlib import Path

from .evidence_proposal_review import build_evidence_review_pack
from .medical_benchmark import file_sha256, load_benchmark


DEFAULT_PILOT = Path("data/benchmark/medical_causal_pilot100_draft_v1.jsonl")
DEFAULT_SOURCES = Path("data/benchmark/source_candidates15_v2.jsonl")
DEFAULT_PROPOSALS = Path("data/benchmark/evidence_proposals15_v2.jsonl")

CLAIM_REVIEW_FIELDS = (
    "review_decision", "corrected_cause", "corrected_effect",
    "corrected_relation", "source_passage_locator",
    "reviewer_id", "reviewed_date", "review_note",
)
QUESTION_REVIEW_FIELDS = (
    "answerability_decision", "draft_reference_answer",
    "supporting_source_ids", "reviewer_id", "reviewed_date",
    "review_note",
)

CLAIM_DECISIONS = {
    "accept", "revise", "reject", "needs_more_evidence",
}
QUESTION_DECISIONS = {
    "answerable", "unanswerable", "needs_more_evidence",
}
RELATIONS = {
    "CAUSES", "RISK_FACTOR_FOR", "ASSOCIATED_WITH",
    "PROTECTS_AGAINST", "UNCERTAIN",
}


def build_review_tables(pilot_file=DEFAULT_PILOT,
                        sources_file=DEFAULT_SOURCES,
                        proposals_file=DEFAULT_PROPOSALS):
    """Return reference rows with empty editable decision columns."""
    worklist, _ = build_evidence_review_pack(
        pilot_file, sources_file, proposals_file
    )
    questions_by_id = {
        q.query_id: q for q in load_benchmark(pilot_file)
    }
    claims, questions = [], []
    for item in worklist:
        source = item["candidate_source"]
        common = {
            "proposal_id": item["proposal_id"],
            "candidate_source_id": source["source_id"],
            "candidate_source_url": source["url"],
            "source_section": item["source_section"],
            "unresolved_scope_issue": item["unresolved_scope_issue"] or "",
        }
        for index, relation in enumerate(item["proposed_relationships"], 1):
            row = {
                "task_id": f'{item["proposal_id"]}:R{index:02d}',
                "task_type": "relationship",
                "query_ids": ";".join(item["query_ids"]),
                **common,
                "proposed_cause": relation["cause"],
                "proposed_effect": relation["effect"],
                "proposed_relation": relation["relation"],
                "source_support_summary": relation["source_support_summary"],
                "review_question": relation["review_question"],
            }
            row.update({key: "" for key in CLAIM_REVIEW_FIELDS})
            claims.append(row)

        if item["proposal_outcome"] == "insufficient_for_question":
            row = {
                "task_id": f'{item["proposal_id"]}:GAP',
                "task_type": "source_gap",
                "query_ids": ";".join(item["query_ids"]),
                **common,
                "proposed_cause": "",
                "proposed_effect": "",
                "proposed_relation": "",
                "source_support_summary": "",
                "review_question": (
                    "Does the candidate source support disease onset? "
                    "If not, request another source; do not invent a causal edge."
                ),
            }
            row.update({key: "" for key in CLAIM_REVIEW_FIELDS})
            claims.append(row)

        for query_id in item["query_ids"]:
            question = questions_by_id[query_id]
            row = {
                "query_id": query_id,
                "query": question.query,
                "query_type": question.query_type,
                "medical_domain": question.medical_domain,
                "target_concept": question.target_concept or "",
                "question_family": question.question_family or "",
                "concept_family": question.concept_family or "",
                **common,
                "proposal_outcome": item["proposal_outcome"],
            }
            row.update({key: "" for key in QUESTION_REVIEW_FIELDS})
            questions.append(row)

    return claims, questions


def _csv_text(rows):
    if not rows:
        raise ValueError("No worksheet rows")
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def _read_csv(path, expected_fields):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != expected_fields:
            raise ValueError(f"Worksheet columns changed: {path}")
        rows = list(reader)
    if any(None in row for row in rows):
        raise ValueError(f"Malformed CSV row: {path}")
    return rows


def create_review_workspace(output_dir, pilot_file=DEFAULT_PILOT,
                            sources_file=DEFAULT_SOURCES,
                            proposals_file=DEFAULT_PROPOSALS):
    claims, questions = build_review_tables(
        pilot_file, sources_file, proposals_file
    )
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=False)
    (path / "claim_review.csv").write_text(
        _csv_text(claims), encoding="utf-8"
    )
    (path / "question_review.csv").write_text(
        _csv_text(questions), encoding="utf-8"
    )
    manifest = {
        "schema_version": 1,
        "purpose": "human_review_not_medical_gold",
        "pilot_sha256": file_sha256(pilot_file),
        "sources_sha256": file_sha256(sources_file),
        "proposals_sha256": file_sha256(proposals_file),
        "claim_tasks": len(claims),
        "question_tasks": len(questions),
        "initial_review_decisions": 0,
        "gold_labels_created": False,
    }
    (path / "review_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def _check_noneditable(expected_rows, actual_rows, key_field, editable):
    if len(actual_rows) != len(expected_rows):
        raise ValueError("Worksheet row count changed")
    expected = {row[key_field]: row for row in expected_rows}
    if len(expected) != len(expected_rows):
        raise ValueError("Duplicate expected review task IDs")
    seen = set()
    for row in actual_rows:
        key = row[key_field]
        if key in seen or key not in expected:
            raise ValueError(f"Duplicate or unknown worksheet ID: {key}")
        seen.add(key)
        original = expected[key]
        for field, value in original.items():
            if field not in editable and row[field] != value:
                raise ValueError(
                    f"Original candidate field modified: {key}.{field}"
                )
    return expected


def _date_valid(text):
    try:
        date.fromisoformat(text)
        return len(text) == 10
    except ValueError:
        return False


def audit_review_workspace(review_dir, pilot_file=DEFAULT_PILOT,
                           sources_file=DEFAULT_SOURCES,
                           proposals_file=DEFAULT_PROPOSALS, strict=False):
    root = Path(review_dir)
    manifest = json.loads(
        (root / "review_manifest.json").read_text(encoding="utf-8")
    )
    for name, path in (
        ("pilot_sha256", pilot_file),
        ("sources_sha256", sources_file),
        ("proposals_sha256", proposals_file),
    ):
        if manifest.get(name) != file_sha256(path):
            raise ValueError(f"Input provenance hash changed: {name}")

    expected_claims, expected_questions = build_review_tables(
        pilot_file, sources_file, proposals_file
    )
    claims = _read_csv(
        root / "claim_review.csv", list(expected_claims[0])
    )
    questions = _read_csv(
        root / "question_review.csv", list(expected_questions[0])
    )
    _check_noneditable(
        expected_claims, claims, "task_id", CLAIM_REVIEW_FIELDS
    )
    _check_noneditable(
        expected_questions, questions, "query_id", QUESTION_REVIEW_FIELDS
    )

    claim_status = Counter()
    question_status = Counter()
    incomplete = []

    for row in claims:
        key = row["task_id"]
        decision = row["review_decision"].strip()
        if not decision:
            claim_status["pending"] += 1
            incomplete.append(key)
            continue
        if decision not in CLAIM_DECISIONS:
            raise ValueError(f"Invalid claim decision: {key}")
        if row["task_type"] == "source_gap" and decision not in {
            "needs_more_evidence", "reject"
        }:
            raise ValueError(f"Cannot accept a source gap: {key}")
        if not row["reviewer_id"].strip() or not _date_valid(row["reviewed_date"]):
            raise ValueError(f"Missing reviewer ID or ISO date: {key}")
        if not row["review_note"].strip():
            raise ValueError(f"Missing review rationale: {key}")
        if decision in {"accept", "revise"}:
            if not row["source_passage_locator"].strip():
                raise ValueError(f"Missing evidence passage locator: {key}")
        if decision == "revise":
            if not row["corrected_cause"].strip() or not row["corrected_effect"].strip():
                raise ValueError(f"Revised claim needs corrected concepts: {key}")
            if row["corrected_relation"].strip() not in RELATIONS:
                raise ValueError(f"Revised claim needs valid relation: {key}")
        claim_status[decision] += 1

    for row in questions:
        key = row["query_id"]
        decision = row["answerability_decision"].strip()
        if not decision:
            question_status["pending"] += 1
            incomplete.append(key)
            continue
        if decision not in QUESTION_DECISIONS:
            raise ValueError(f"Invalid answerability decision: {key}")
        if not row["reviewer_id"].strip() or not _date_valid(row["reviewed_date"]):
            raise ValueError(f"Missing reviewer ID or ISO date: {key}")
        if not row["review_note"].strip():
            raise ValueError(f"Missing review rationale: {key}")
        if decision == "answerable":
            if not row["draft_reference_answer"].strip():
                raise ValueError(f"Answerable question lacks reference answer: {key}")
            ids = {
                sid.strip() for sid in row["supporting_source_ids"].split(";")
                if sid.strip()
            }
            # The first review iteration only knows this question's
            # candidate source. Additional sources must enter a
            # versioned source registry before they can be referenced.
            allowed = {row["candidate_source_id"]}
            if not ids or not ids.issubset(allowed):
                raise ValueError(
                    f"Unknown or unrelated reference source IDs: {key}"
                )
        question_status[decision] += 1

    if strict and incomplete:
        raise ValueError(
            f"Incomplete review decisions ({len(incomplete)}): "
            + ", ".join(incomplete[:5])
        )

    return {
        "schema_version": 1,
        "data_role": "human_decision_workflow_audit_not_gold",
        "claim_tasks": len(claims),
        "question_tasks": len(questions),
        "claim_decision_counts": dict(sorted(claim_status.items())),
        "question_decision_counts": dict(sorted(question_status.items())),
        "pending_decisions": len(incomplete),
        "strict_decisions_required": bool(strict),
        "reviewer_identity_verified_by_code": False,
        "medical_correctness_verified_by_code": False,
        "gold_labels_created": False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--audit-dir", type=Path)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--pilot", type=Path, default=DEFAULT_PILOT)
    parser.add_argument("--sources", type=Path, default=DEFAULT_SOURCES)
    parser.add_argument("--proposals", type=Path, default=DEFAULT_PROPOSALS)
    args = parser.parse_args()
    if bool(args.output_dir) == bool(args.audit_dir):
        parser.error("Provide exactly one of --output-dir or --audit-dir")
    if args.output_dir:
        result = create_review_workspace(
            args.output_dir, args.pilot, args.sources, args.proposals
        )
    else:
        result = audit_review_workspace(
            args.audit_dir, args.pilot, args.sources, args.proposals,
            strict=args.strict
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
