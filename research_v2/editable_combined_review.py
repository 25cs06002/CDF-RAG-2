"""Phase 2I-C3: editable 94-source review workspace with strict provenance.

Creates an independent review copy of a verified C2 handoff. The validator
checks structure, citations, and reviewer-supplied decisions, not medicine.
"""
import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

from .combined_review_handoff import audit as audit_handoff
from .indirect_source_extension import REGISTRY
from .remaining_review_workspace import (
    QUESTION_EDITABLE, SOURCE_EDITABLE, LINK_EDITABLE,
    QUESTION_DECISIONS, SOURCE_DECISIONS, LINK_DECISIONS,
    RELATIONS, PATH_STATUSES, _read_csv, _validate_immutable,
    _review_identity, _parse_ids, _csv_text,
)

REVIEW_FILES = ("question_review.csv", "source_review.csv", "link_review.csv")
REQUIRED_INPUTS = (*REVIEW_FILES, "migration_manifest.json")
PURPOSE = "editable_94_source_review_not_medical_gold"

def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def _read_base(combined):
    return [ _read_csv(combined / file, None) for file in REVIEW_FILES ]

def _read_table(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError("Invalid CSV header: " + str(path))
        rows = list(reader)
    if not rows or any(None in row for row in rows):
        raise ValueError("Malformed or empty CSV: " + str(path))
    return rows

def _expected(combined):
    return [_read_table(combined / name) for name in REVIEW_FILES]

def create(output_dir, combined_dir, baseline_dir, supplement_dir, registry=REGISTRY):
    combined, dest = Path(combined_dir), Path(output_dir)
    audit_handoff(combined, baseline_dir, supplement_dir, registry=registry)
    if dest.exists():
        raise FileExistsError("Refusing to overwrite review workspace: " + str(dest))
    rows = _expected(combined)
    if list(map(len, rows)) != [70, 94, 30]:
        raise ValueError("Unexpected C2 handoff review counts")
    manifest = {
        "schema_version": 1,
        "purpose": PURPOSE,
        "combined_input_hashes": {name: _sha(combined / name) for name in REQUIRED_INPUTS},
        "question_tasks": 70, "source_tasks": 94, "link_slots": 30,
        "review_decisions_auto_promoted": False,
        "clinical_gold_created": False,
        "baseline_workspace_modified": False,
    }
    dest.mkdir(parents=True, exist_ok=False)
    for name, table in zip(REVIEW_FILES, rows):
        (dest / name).write_text(_csv_text(table), encoding="utf-8")
    (dest / "editable_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return audit(dest, combined, baseline_dir, supplement_dir, registry)

def audit(review_dir, combined_dir, baseline_dir, supplement_dir, registry=REGISTRY, strict=False):
    root, combined = Path(review_dir), Path(combined_dir)
    audit_handoff(combined, baseline_dir, supplement_dir, registry=registry)
    manifest = json.loads((root / "editable_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("purpose") != PURPOSE:
        raise ValueError("Unexpected editable workspace purpose")
    expected_hashes = {name: _sha(combined / name) for name in REQUIRED_INPUTS}
    if manifest.get("combined_input_hashes") != expected_hashes:
        raise ValueError("Original combined handoff hash changed")
    expected = _expected(combined)
    actual = [
        _read_csv(root / name, list(table[0]))
        for name, table in zip(REVIEW_FILES, expected)
    ]
    for old, new, key, editable in zip(
        expected, actual, ("query_id", "source_task_id", "link_task_id"),
        (QUESTION_EDITABLE, SOURCE_EDITABLE, LINK_EDITABLE)
    ):
        _validate_immutable(old, new, key, editable)
    questions, sources, links = actual
    source_map, link_map = {}, {}
    source_counts, link_counts, question_counts = Counter(), Counter(), Counter()
    pending = []
    for row in sources:
        qid, sid, key = row["query_id"], row["candidate_source_id"], row["source_task_id"]
        source_map.setdefault(qid, {})[sid] = row
        decision = row["review_decision"].strip()
        if not decision:
            pending.append(key)
            source_counts["pending"] += 1
            continue
        if decision not in SOURCE_DECISIONS:
            raise ValueError("Invalid source decision: " + key)
        _review_identity(row, key)
        if decision in ("supports", "partial") and (
            not row["source_passage_locator"].strip() or not row["supported_claim_summary"].strip()
        ):
            raise ValueError("Supported/partial source requires a passage and summary: " + key)
        source_counts[decision] += 1
    for row in links:
        qid, key = row["query_id"], row["link_task_id"]
        link_map.setdefault(qid, []).append(row)
        decision = row["review_decision"].strip()
        if not decision:
            pending.append(key)
            link_counts["pending"] += 1
            continue
        if decision not in LINK_DECISIONS:
            raise ValueError("Invalid link decision: " + key)
        _review_identity(row, key)
        source_ids = _parse_ids(row["supporting_source_ids"])
        allowed = source_map.get(qid, {})
        if not set(source_ids).issubset(allowed):
            raise ValueError("Unregistered link source: " + key)
        if decision == "supports":
            if not all(row[x].strip() for x in ("cause_node", "effect_node", "relation", "source_passage_locator")):
                raise ValueError("Source-supported link missing nodes, relation or passage: " + key)
            if row["relation"] not in RELATIONS or row["relation"] == "UNCERTAIN":
                raise ValueError("Invalid supported link relation: " + key)
            if not source_ids or not all(allowed[s]["review_decision"] == "supports" for s in source_ids):
                raise ValueError("Supported link uses non-supporting sources: " + key)
        link_counts[decision] += 1
    source_gap_flags = []
    for row in questions:
        qid, decision = row["query_id"], row["answerability_decision"].strip()
        if not decision:
            pending.append(qid)
            question_counts["pending"] += 1
            continue
        if decision not in QUESTION_DECISIONS:
            raise ValueError("Invalid question decision: " + qid)
        if decision == "answerable" and row["query_type"] == "unsupported":
            raise ValueError("Unsupported exact-person prediction cannot be answerable: " + qid)
        _review_identity(row, qid)
        ids = _parse_ids(row["supporting_source_ids"])
        allowed = source_map.get(qid, {})
        if not set(ids).issubset(allowed):
            raise ValueError("Question cites unregistered source: " + qid)
        if decision == "answerable":
            if not row["draft_reference_answer"].strip() or not ids:
                raise ValueError("Answerable question requires sourced answer: " + qid)
            if not all(allowed[s]["review_decision"] == "supports" for s in ids):
                raise ValueError("Answerable question cites partial/non-supporting source: " + qid)
            if row["needs_additional_sources"] == "true":
                if not row["scope_resolution_note"].strip():
                    raise ValueError("Original source gap requires scope resolution: " + qid)
                source_gap_flags.append(qid)
            if row["query_type"] == "risk_factor" and not row["risk_vs_cause_note"].strip():
                raise ValueError("Missing risk versus causation note: " + qid)
            if row["query_type"] == "comparative":
                if not all(row[f].strip() for f in ("comparison_topic_a", "comparison_topic_b", "comparison_evidence_note")):
                    raise ValueError("Missing comparison evidence fields: " + qid)
                side_a = _parse_ids(row["comparison_source_ids_a"])
                side_b = _parse_ids(row["comparison_source_ids_b"])
                if not side_a or not side_b or not set(side_a+side_b).issubset(ids):
                    raise ValueError("Comparison lacks both source-backed sides: " + qid)
            if row["query_type"] == "multi_hop":
                status = row["candidate_path_status"]
                if status not in PATH_STATUSES:
                    raise ValueError("Invalid candidate path status: " + qid)
                slots = sorted(link_map.get(qid, []), key=lambda link: int(link["link_slot"]))
                if len(slots) != 2 or any(not x["review_decision"] for x in slots):
                    raise ValueError("Multi-hop needs two reviewed link slots: " + qid)
                supported = [x for x in slots if x["review_decision"] == "supports"]
                if status == "two_links_source_supported":
                    if len(supported) != 2:
                        raise ValueError("Two-link status needs two supported links: " + qid)
                    if slots[0]["effect_node"].strip().casefold() != slots[1]["cause_node"].strip().casefold():
                        raise ValueError("Two link nodes do not connect: " + qid)
                if status == "partial_links_supported" and len(supported) != 1:
                    raise ValueError("Partial-link status needs exactly one supported link: " + qid)
        if decision == "unanswerable" and not row["abstention_reason"].strip():
            raise ValueError("Unanswerable question needs abstention reason: " + qid)
        question_counts[decision] += 1
    if strict and pending:
        raise ValueError("Incomplete editable reviews: " + ", ".join(pending[:8]))
    return {
        "status": "audit_passed", "question_tasks": len(questions), "source_tasks": len(sources),
        "link_slots": len(links), "question_decisions": dict(sorted(question_counts.items())),
        "source_decisions": dict(sorted(source_counts.items())),
        "link_decisions": dict(sorted(link_counts.items())),
        "pending_decisions": len(pending),
        "original_gap_flags_still_present": sorted(source_gap_flags),
        "strict": strict, "original_handoff_unchanged": True,
        "medical_correctness_verified_by_code": False, "gold_labels_created": False,
    }

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--output-dir", type=Path)
    group.add_argument("--audit-dir", type=Path)
    parser.add_argument("--combined-dir", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--supplement-dir", type=Path, required=True)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    action = create if args.output_dir else audit
    root = args.output_dir if args.output_dir else args.audit_dir
    result = action(root, args.combined_dir, args.baseline_dir, args.supplement_dir,
                    strict=args.strict) if not args.output_dir else action(root, args.combined_dir, args.baseline_dir, args.supplement_dir)
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
