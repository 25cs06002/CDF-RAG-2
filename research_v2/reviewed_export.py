"""Export Phase 2 reviewed *candidates* without promoting them to medical gold.

The original reviewer_workspace strict audit is a mandatory precondition for
CLI export. Medical correctness and reviewer identity are NOT verified by code.
"""

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


INPUT_FILES = ("claim_review.csv", "question_review.csv", "review_manifest.json")
OUTPUT_FILES = (
    "reviewed_relationships_v1.jsonl",
    "unresolved_relationships_v1.jsonl",
    "reviewed_questions_v1.jsonl",
    "unresolved_questions_v1.jsonl",
)
SCHEMA_VERSION = 1


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        if not reader.fieldnames or any(None in x for x in rows):
            raise ValueError(f"Malformed CSV: {path}")
        return rows


def verify_checksums(review_dir, checksums_file):
    """Check all three frozen inputs against an existing SHA256SUMS file."""
    review_dir = Path(review_dir)
    entries = {}
    for line in Path(checksums_file).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        match = re.fullmatch(r"([a-fA-F0-9]{64})\s+\*?([^\\/]+)", line.strip())
        if not match:
            raise ValueError(f"Invalid SHA256SUMS entry: {line!r}")
        expected_digest, name = match.groups()
        if name not in INPUT_FILES or name in entries:
            raise ValueError(f"Unexpected/duplicate frozen input: {name}")
        entries[name] = expected_digest.lower()
    if set(entries) != set(INPUT_FILES):
        raise ValueError("SHA256SUMS must contain exactly the three review input files")
    for name, expected_digest in entries.items():
        if sha256(review_dir / name) != expected_digest:
            raise ValueError(f"Frozen file SHA256 mismatch: {name}")
    return entries


def _source(row):
    return {
        "source_id": row["candidate_source_id"],
        "url": row["candidate_source_url"],
        "section": row["source_section"],
        "passage_locator": row.get("source_passage_locator", ""),
    }


def _review(row, field):
    return {
        "decision": row[field],
        "reviewer_id_recorded": row["reviewer_id"],
        "reviewed_date_recorded": row["reviewed_date"],
        "review_note": row["review_note"],
        "reviewer_identity_verified_by_code": False,
    }


def _jsonl(rows):
    return "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)


def build_reviewed_export(claims, questions, manifest, input_hashes, audit_profile):
    """Pure transformation of already strictly audited reviewer worksheets."""
    if audit_profile.get("pending_decisions") != 0 or not audit_profile.get("strict_decisions_required"):
        raise ValueError("Full strict reviewer_workspace audit is required before export")
    if audit_profile.get("claim_tasks") != len(claims) or audit_profile.get("question_tasks") != len(questions):
        raise ValueError("Strict-audit row counts differ from supplied data")
    if len({r["task_id"] for r in claims}) != len(claims):
        raise ValueError("Duplicate claim review task")
    if len({r["query_id"] for r in questions}) != len(questions):
        raise ValueError("Duplicate question review ID")

    retained, unresolved, answers, unanswerable = [], [], [], []
    by_proposal_retained = defaultdict(list)
    by_proposal_unresolved = defaultdict(list)
    source_by_proposal = {}

    for row in claims:
        decision = row["review_decision"].strip()
        task_id = row["task_id"]
        prop_id = row["proposal_id"]
        source_id = row["candidate_source_id"]
        if prop_id in source_by_proposal and source_by_proposal[prop_id] != source_id:
            raise ValueError(f"Inconsistent claim source for {prop_id}")
        source_by_proposal[prop_id] = source_id
        if decision not in {"accept", "revise", "reject", "needs_more_evidence"}:
            raise ValueError(f"Invalid claim decision: {task_id}")
        if row["task_type"] == "source_gap" and decision not in {"reject", "needs_more_evidence"}:
            raise ValueError(f"Source gap cannot be retained: {task_id}")
        original = {
            "cause": row["proposed_cause"] or None,
            "effect": row["proposed_effect"] or None,
            "relation": row["proposed_relation"] or None,
        }
        common = {
            "schema_version": SCHEMA_VERSION,
            "task_id": task_id,
            "task_type": row["task_type"],
            "proposal_id": prop_id,
            "query_ids": [x for x in row["query_ids"].split(";") if x],
            "original_candidate": original,
            "source": _source(row),
            "source_support_summary": row["source_support_summary"],
            "unresolved_scope_issue": row["unresolved_scope_issue"],
            "review": _review(row, "review_decision"),
        }
        if decision in {"accept", "revise"}:
            is_revised = decision == "revise"
            effective = {
                "cause": row["corrected_cause"] if is_revised else row["proposed_cause"],
                "effect": row["corrected_effect"] if is_revised else row["proposed_effect"],
                "relation": row["corrected_relation"] if is_revised else row["proposed_relation"],
            }
            if not all(effective.values()) or not row["source_passage_locator"].strip():
                raise ValueError(f"Incomplete retained claim: {task_id}")
            retained.append({
                **common,
                "data_role": "reviewed_candidate_relationship_not_medical_gold",
                "effective_candidate": effective,
                "revision_applied": is_revised,
            })
            by_proposal_retained[prop_id].append(task_id)
        else:
            unresolved.append({
                **common,
                "data_role": "excluded_or_unresolved_claim_not_gold",
                "resolution_status": decision,
                "suggested_action": "obtain_additional_source_or_adjudicate" if decision == "needs_more_evidence" else "retain_rejection_record",
            })
            by_proposal_unresolved[prop_id].append(task_id)

    warnings = []
    warning_question_ids = []
    for row in questions:
        decision = row["answerability_decision"].strip()
        qid, prop_id = row["query_id"], row["proposal_id"]
        if decision not in {"answerable", "unanswerable", "needs_more_evidence"}:
            raise ValueError(f"Invalid question decision: {qid}")
        if prop_id in source_by_proposal and source_by_proposal[prop_id] != row["candidate_source_id"]:
            raise ValueError(f"Claim/question registered source mismatch: {qid}")
        common = {
            "schema_version": SCHEMA_VERSION,
            "query_id": qid,
            "query": row["query"],
            "query_type": row["query_type"],
            "medical_domain": row["medical_domain"],
            "target_concept": row["target_concept"],
            "question_family": row["question_family"],
            "concept_family": row["concept_family"],
            "source_family": None,
            "split": None,
            "proposal_id": prop_id,
            "proposal_outcome": row["proposal_outcome"],
            "candidate_source": _source(row),
            "unresolved_scope_issue": row["unresolved_scope_issue"],
            "linked_retained_claim_task_ids": by_proposal_retained[prop_id],
            "linked_unresolved_claim_task_ids": by_proposal_unresolved[prop_id],
            "review": _review(row, "answerability_decision"),
        }
        if decision == "answerable":
            draft = row["draft_reference_answer"].strip()
            source_ids = [s.strip() for s in row["supporting_source_ids"].split(";") if s.strip()]
            if not draft or not source_ids or set(source_ids) != {row["candidate_source_id"]}:
                raise ValueError(f"Unsupported/missing reference draft for {qid}")
            if not by_proposal_retained[prop_id]:
                warning_question_ids.append(qid)
            answers.append({
                **common,
                "data_role": "reviewed_answer_draft_not_medical_gold",
                "answerability_decision": "answerable",
                "reference_answer_draft": draft,
                "supporting_source_ids": sorted(set(source_ids)),
                "no_retained_claim_for_proposal": not bool(by_proposal_retained[prop_id]),
            })
        else:
            unanswerable.append({
                **common,
                "data_role": "unresolved_or_unanswerable_question_not_medical_gold",
                "answerability_decision": decision,
                "draft_reference_answer": row["draft_reference_answer"] or None,
                "supporting_source_ids": [x.strip() for x in row["supporting_source_ids"].split(";") if x.strip()],
            })
    if warning_question_ids:
        warnings.append({
            "code": "answerable_questions_without_retained_causal_claim",
            "query_ids": warning_question_ids,
            "explanation": "An answerable question has no retained causal relationship within its corresponding proposal. This can be valid for explanatory QA, but requires separate adjudication before graph-edge or path evaluation.",
        })

    summary = {
        "schema_version": SCHEMA_VERSION,
        "data_role": "strictly_audited_reviewed_candidates_not_medical_gold",
        "source_review_manifest": manifest,
        "input_sha256": input_hashes,
        "strict_audit": audit_profile,
        "counts": {
            "input_claim_tasks": len(claims),
            "accepted_claim_tasks": sum(r["review_decision"] == "accept" for r in claims),
            "revised_claim_tasks": sum(r["review_decision"] == "revise" for r in claims),
            "retained_reviewed_candidate_relationships": len(retained),
            "unresolved_or_rejected_claim_tasks": len(unresolved),
            "input_question_tasks": len(questions),
            "answerable_question_drafts": len(answers),
            "unresolved_or_unanswerable_questions": len(unanswerable),
        },
        "warnings": warnings,
        "provenance_limitations": {
            "medical_correctness_verified_by_code": False,
            "reviewer_identity_verified_by_code": False,
            "medical_gold_created": False,
            "final_splits_assigned": False,
            "ready_for_automatic_graph_import": False,
            "annotation_adjudication_required": True,
        },
    }
    return {
        OUTPUT_FILES[0]: retained,
        OUTPUT_FILES[1]: unresolved,
        OUTPUT_FILES[2]: answers,
        OUTPUT_FILES[3]: unanswerable,
    }, summary


def export_reviewed_batch(review_dir, output_dir, checksums_file=None):
    """Validate the original inputs strictly, then create a fresh export dir."""
    from .reviewer_workspace import audit_review_workspace

    root = Path(review_dir)
    expected = verify_checksums(root, checksums_file) if checksums_file else None
    strict_profile = audit_review_workspace(root, strict=True)
    input_hashes = {name: sha256(root / name) for name in INPUT_FILES}
    if expected and expected != input_hashes:
        raise ValueError("The frozen checksum file does not match the input files")
    manifest = json.loads((root / "review_manifest.json").read_text(encoding="utf-8"))
    rows, profile = build_reviewed_export(
        read_csv(root / "claim_review.csv"),
        read_csv(root / "question_review.csv"),
        manifest,
        input_hashes,
        strict_profile,
    )
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=False)
    for name, values in rows.items():
        (out / name).write_text(_jsonl(values), encoding="utf-8")
    profile["export_sha256"] = {name: sha256(out / name) for name in OUTPUT_FILES}
    (out / "export_profile.json").write_text(
        json.dumps(profile, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    files = list(OUTPUT_FILES) + ["export_profile.json"]
    (out / "SHA256SUMS.txt").write_text(
        "".join(f"{sha256(out / name)}  {name}\n" for name in files),
        encoding="utf-8",
    )
    return profile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checksums-file", type=Path,
                        help="Frozen SHA256SUMS.txt with all three original input hashes")
    args = parser.parse_args()
    profile = export_reviewed_batch(
        args.review_dir, args.output_dir, checksums_file=args.checksums_file
    )
    print(json.dumps(profile, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
