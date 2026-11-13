"""Phase 2I-C2: non-destructive, source-traceable combined review snapshot.

Combines audited remaining70 reviews and five strictly reviewed supplemental
sources. This is a frozen handoff snapshot, NOT a gold-label export or a
replacement for a future 94-source editable review validator.
"""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path

from .indirect_source_extension import (
    REGISTRY, audit as audit_supplement, expected_rows as supplement_expected,
    read_csv as read_supplement_csv,
)
from .remaining_review_workspace import (
    audit_remaining_review_workspace, build_remaining_review_tables,
    _read_csv, _csv_text,
)

NAMES = ("question_review.csv", "source_review.csv", "link_review.csv", "review_manifest.json")
EXTRA = ("supplemental_source_review.csv", "supplement_manifest.json")

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def _read_original(base):
    qs, ss, ls = build_remaining_review_tables()
    return (
        _read_csv(base / NAMES[0], list(qs[0])),
        _read_csv(base / NAMES[1], list(ss[0])),
        _read_csv(base / NAMES[2], list(ls[0])),
    )

def _supplement(supp):
    expected = supplement_expected()
    return read_supplement_csv(supp / EXTRA[0], list(expected[0]))

def _joined(base, supp):
    q, s, links = _read_original(base)
    extra = _supplement(supp)
    seen = {row["source_task_id"] for row in s}
    qmap = {row["query_id"]: row for row in q}
    for row in extra:
        qid, sid = row["query_id"], row["candidate_source_id"]
        if row["source_task_id"] in seen or qid not in qmap:
            raise ValueError("Invalid or duplicate additional source task")
        seen.add(row["source_task_id"])
        question = qmap[qid]
        existing = question["candidate_source_ids"].split(";") if question["candidate_source_ids"] else []
        if sid in existing:
            raise ValueError("Duplicate source ID for question: " + qid)
        question["candidate_source_ids"] = ";".join(existing + [sid])
        urls = question["candidate_source_urls"].split(";") if question["candidate_source_urls"] else []
        question["candidate_source_urls"] = ";".join(urls + [row["candidate_source_url"]])
        s.append({
            "source_task_id": row["source_task_id"],
            "query_id": qid,
            "query_type": question["query_type"],
            "review_route": question["review_route"],
            "candidate_source_id": sid,
            "candidate_source_url": row["candidate_source_url"],
            "candidate_source_title": row["candidate_source_title"],
            "candidate_origin": "phase2ic1_supplement",
            "needs_additional_sources": question["needs_additional_sources"],
            "scope_note": row["scope_note"],
            "review_decision": row["review_decision"],
            "source_passage_locator": row["source_passage_locator"],
            "supported_claim_summary": row["supported_claim_summary"],
            "limitation_note": row["limitation_note"],
            "reviewer_id": row["reviewer_id"],
            "reviewed_date": row["reviewed_date"],
            "review_note": row["review_note"],
        })
    if len(q) != 70 or len(s) != 94 or len(links) != 30:
        raise ValueError("Incorrect combined review counts")
    return q, s, links

def _input_hashes(base, supp, registry):
    return {
        "baseline": {name: digest(base / name) for name in NAMES},
        "supplement": {name: digest(supp / name) for name in EXTRA},
        "registry_sha256": digest(registry),
    }

def create(output_dir, baseline_dir, supplement_dir, registry=REGISTRY):
    base, supp, dest = Path(baseline_dir), Path(supplement_dir), Path(output_dir)
    audit_remaining_review_workspace(base)
    audit_supplement(supp, base, registry=registry, strict=True)
    q, s, links = _joined(base, supp)
    if dest.exists():
        raise FileExistsError("Refusing to overwrite combined workspace: " + str(dest))
    # Establish complete combined content before creating the directory.
    data = {
        "question_review.csv": _csv_text(q),
        "source_review.csv": _csv_text(s),
        "link_review.csv": _csv_text(links),
    }
    manifest = {
        "schema_version": 1,
        "purpose": "combined_70_question_94_source_handoff_not_medical_gold",
        "inputs": _input_hashes(base, supp, registry),
        "question_tasks": 70, "source_tasks": 94, "link_slots": 30,
        "source_review_decisions_copied_not_promoted": True,
        "question_decisions_copied_not_promoted": True,
        "existing_review_sources_rewritten": False,
        "medical_gold_created": False,
        "editable_combined_review_validator_implemented": False,
    }
    dest.mkdir(parents=True, exist_ok=False)
    try:
        for name, content in data.items():
            (dest / name).write_text(content, encoding="utf-8")
        (dest / "migration_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
    except Exception:
        # Avoid claiming a fully created result after partial writes.
        raise
    return audit(dest, base, supp, registry)

def audit(combined_dir, baseline_dir, supplement_dir, registry=REGISTRY):
    base, supp, dest = Path(baseline_dir), Path(supplement_dir), Path(combined_dir)
    audit_remaining_review_workspace(base)
    audit_supplement(supp, base, registry=registry, strict=True)
    manifest = json.loads((dest / "migration_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("purpose") != "combined_70_question_94_source_handoff_not_medical_gold":
        raise ValueError("Unexpected combined workspace purpose")
    if manifest.get("inputs") != _input_hashes(base, supp, registry):
        raise ValueError("Migration input hashes do not match the original snapshots")
    expected = _joined(base, supp)
    for name, rows in zip(("question_review.csv", "source_review.csv", "link_review.csv"), expected):
        actual = _read_csv(dest / name, list(rows[0]))
        if actual != rows:
            raise ValueError("Combined review differs from approved migration inputs: " + name)
    from collections import Counter
    q, s, links = expected
    return {
        "status": "audit_passed",
        "question_tasks": len(q), "source_tasks": len(s), "link_slots": len(links),
        "question_decisions": dict(sorted(Counter(x["answerability_decision"] for x in q).items())),
        "new_source_decisions": dict(sorted(Counter(x["review_decision"] for x in s if x["candidate_origin"] == "phase2ic1_supplement").items())),
        "inputs_unchanged": True, "migrated_review_answers_changed": False,
        "combined_snapshot_editable": False, "medical_gold_created": False,
    }

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--output-dir", type=Path)
    group.add_argument("--audit-dir", type=Path)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--supplement-dir", type=Path, required=True)
    parser.add_argument("--registry", type=Path, default=REGISTRY)
    args = parser.parse_args()
    if args.output_dir:
        result = create(args.output_dir, args.baseline_dir, args.supplement_dir, args.registry)
    else:
        result = audit(args.audit_dir, args.baseline_dir, args.supplement_dir, args.registry)
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
