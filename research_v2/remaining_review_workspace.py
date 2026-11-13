"""Phase 2I-A: evidence-review workspace for the remaining 70 pilot questions.

The source queue contains candidate URLs, not verified answer/evidence links.
The multi-hop rows are blank TWO-SLOT review prompts, NOT proposed causal
edges. The validator verifies workflow integrity, not clinical correctness.
"""

import argparse
import csv
import io
import json
from collections import Counter
from datetime import date
from pathlib import Path

from .medical_benchmark import file_sha256, load_benchmark
from .remaining_source_queue import (
    PILOT, BASE_SOURCES, EXTRA_SOURCES, SOURCE_MAP, build_remaining_queue
)

QUESTION_EDITABLE = (
    "answerability_decision", "draft_reference_answer",
    "supporting_source_ids", "abstention_reason",
    "scope_resolution_note", "candidate_path_status",
    "comparison_topic_a", "comparison_topic_b",
    "comparison_source_ids_a", "comparison_source_ids_b",
    "comparison_evidence_note", "risk_vs_cause_note",
    "reviewer_id", "reviewed_date", "review_note",
)
SOURCE_EDITABLE = (
    "review_decision", "source_passage_locator",
    "supported_claim_summary", "limitation_note",
    "reviewer_id", "reviewed_date", "review_note",
)
LINK_EDITABLE = (
    "review_decision", "cause_node", "effect_node",
    "relation", "supporting_source_ids", "source_passage_locator",
    "reviewer_id", "reviewed_date", "review_note",
)
QUESTION_DECISIONS = {"answerable", "unanswerable", "needs_more_evidence"}
SOURCE_DECISIONS = {"supports", "partial", "does_not_support", "needs_more_evidence"}
LINK_DECISIONS = SOURCE_DECISIONS
RELATIONS = {
    "CAUSES", "RISK_FACTOR_FOR", "ASSOCIATED_WITH",
    "PROTECTS_AGAINST", "UNCERTAIN",
}
PATH_STATUSES = {
    "two_links_source_supported", "partial_links_supported",
    "no_supported_chain",
}
INPUT_FILES = (
    ("pilot_sha256", PILOT),
    ("base_sources_sha256", BASE_SOURCES),
    ("extra_sources_sha256", EXTRA_SOURCES),
    ("source_map_sha256", SOURCE_MAP),
)


def _empty(row, fields):
    return {**row, **{key: "" for key in fields}}


def _parse_ids(raw):
    ids = [x.strip() for x in raw.split(";") if x.strip()]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate source IDs within a reviewed field")
    return ids


def _csv_text(rows):
    if not rows:
        raise ValueError("No worksheet rows")
    buff = io.StringIO(newline="")
    writer = csv.DictWriter(buff, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return buff.getvalue()


def _read_csv(path, fields):
    with Path(path).open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != fields:
            raise ValueError(f"Worksheet columns changed: {path}")
        rows = list(reader)
    if any(None in row for row in rows):
        raise ValueError(f"Malformed CSV row: {path}")
    return rows


def _validate_immutable(expected, actual, key_field, editable):
    if len(expected) != len(actual):
        raise ValueError(f"Review worksheet row count changed: {key_field}")
    reference = {row[key_field]: row for row in expected}
    if len(reference) != len(expected):
        raise ValueError(f"Duplicate original worksheet keys: {key_field}")
    seen = set()
    for row in actual:
        key = row[key_field]
        if key in seen or key not in reference:
            raise ValueError(f"Unknown/duplicate review task: {key}")
        seen.add(key)
        for field, original in reference[key].items():
            if field not in editable and row[field] != original:
                raise ValueError(f"Original candidate field modified: {key}.{field}")


def _valid_date(raw):
    try:
        return len(raw) == 10 and date.fromisoformat(raw).isoformat() == raw
    except ValueError:
        return False


def _review_identity(row, key):
    if not row["reviewer_id"].strip() or not _valid_date(row["reviewed_date"]):
        raise ValueError(f"Missing real reviewer ID or ISO date: {key}")
    if not row["review_note"].strip():
        raise ValueError(f"Missing review rationale: {key}")


def build_remaining_review_tables(
    pilot_file=PILOT, base_sources_file=BASE_SOURCES,
    extra_sources_file=EXTRA_SOURCES, source_map_file=SOURCE_MAP,
):
    queue, _ = build_remaining_queue(
        pilot_file, base_sources_file, extra_sources_file, source_map_file,
    )
    all_pilot = {r.query_id: r for r in load_benchmark(pilot_file)}
    question_rows, source_rows, link_rows = [], [], []
    for item in queue:
        qid = item["query_id"]
        question = all_pilot[qid]
        sources = item["candidate_sources"]
        ids = ";".join(src["source_id"] for src in sources)
        base = {
            "query_id": qid,
            "query": item["query"],
            "query_type": item["query_type"],
            "medical_domain": item["medical_domain"],
            "target_concept": item["target_concept"] or "",
            "question_family": question.question_family or "",
            "concept_family": question.concept_family or "",
            "review_route": item["review_route"],
            "candidate_source_ids": ids,
            "candidate_source_urls": ";".join(src["url"] for src in sources),
            "needs_additional_sources": str(item["needs_additional_sources"]).lower(),
            "scope_note": item["scope_note"],
        }
        question_rows.append(_empty(base, QUESTION_EDITABLE))

        for src in sources:
            source_rows.append(_empty({
                "source_task_id": f"{qid}:{src['source_id']}",
                "query_id": qid,
                "query_type": item["query_type"],
                "review_route": item["review_route"],
                "candidate_source_id": src["source_id"],
                "candidate_source_url": src["url"],
                "candidate_source_title": src["title"],
                "candidate_origin": src["candidate_origin"],
                "needs_additional_sources": str(item["needs_additional_sources"]).lower(),
                "scope_note": item["scope_note"],
            }, SOURCE_EDITABLE))

        if item["review_route"] == "multi_hop_link_evidence":
            for slot in (1, 2):
                link_rows.append(_empty({
                    "link_task_id": f"{qid}:LINK{slot}",
                    "query_id": qid,
                    "link_slot": str(slot),
                    "candidate_source_ids": ids,
                    "slot_role": "unverified_minimum_two_link_review_placeholder",
                    "scope_note": item["scope_note"],
                }, LINK_EDITABLE))
    if len(question_rows) != 70 or len(source_rows) != 89 or len(link_rows) != 30:
        raise ValueError("Unexpected remaining source mapping worksheet counts")
    return question_rows, source_rows, link_rows


def create_remaining_review_workspace(
    output_dir, pilot_file=PILOT, base_sources_file=BASE_SOURCES,
    extra_sources_file=EXTRA_SOURCES, source_map_file=SOURCE_MAP,
):
    questions, sources, links = build_remaining_review_tables(
        pilot_file, base_sources_file, extra_sources_file, source_map_file,
    )
    hashes = {
        name: file_sha256(path) for name, path in (
            ("pilot_sha256", pilot_file),
            ("base_sources_sha256", base_sources_file),
            ("extra_sources_sha256", extra_sources_file),
            ("source_map_sha256", source_map_file),
        )
    }
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=False)
    (root / "question_review.csv").write_text(
        _csv_text(questions), encoding="utf-8"
    )
    (root / "source_review.csv").write_text(
        _csv_text(sources), encoding="utf-8"
    )
    (root / "link_review.csv").write_text(
        _csv_text(links), encoding="utf-8"
    )
    manifest = {
        "schema_version": 1,
        "purpose": "unreviewed_remaining70_evidence_workflow_not_medical_gold",
        **hashes,
        "question_tasks": len(questions),
        "question_source_review_tasks": len(sources),
        "two_link_review_slots": len(links),
        "link_slots_are_proposals": False,
        "human_review_completed": False,
        "medical_gold_created": False,
        "previous_30_question_worklist_not_modified": True,
    }
    (root / "review_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def audit_remaining_review_workspace(
    review_dir, pilot_file=PILOT, base_sources_file=BASE_SOURCES,
    extra_sources_file=EXTRA_SOURCES, source_map_file=SOURCE_MAP,
    strict=False,
):
    root = Path(review_dir)
    manifest = json.loads(
        (root / "review_manifest.json").read_text(encoding="utf-8")
    )
    if manifest.get("purpose") != "unreviewed_remaining70_evidence_workflow_not_medical_gold":
        raise ValueError("Incorrect remaining70 workspace purpose")
    for name, path in (
        ("pilot_sha256", pilot_file),
        ("base_sources_sha256", base_sources_file),
        ("extra_sources_sha256", extra_sources_file),
        ("source_map_sha256", source_map_file),
    ):
        if manifest.get(name) != file_sha256(path):
            raise ValueError(f"Source registry provenance hash changed: {name}")
    expected_questions, expected_sources, expected_links = build_remaining_review_tables(
        pilot_file, base_sources_file, extra_sources_file, source_map_file,
    )
    questions = _read_csv(root / "question_review.csv", list(expected_questions[0]))
    sources = _read_csv(root / "source_review.csv", list(expected_sources[0]))
    links = _read_csv(root / "link_review.csv", list(expected_links[0]))
    for expected, actual, key, editable in (
        (expected_questions, questions, "query_id", QUESTION_EDITABLE),
        (expected_sources, sources, "source_task_id", SOURCE_EDITABLE),
        (expected_links, links, "link_task_id", LINK_EDITABLE),
    ):
        _validate_immutable(expected, actual, key, editable)

    sources_by_qid = {}
    source_decisions = Counter()
    incomplete = []
    for row in sources:
        key = row["source_task_id"]
        source_id = row["candidate_source_id"]
        qid = row["query_id"]
        sources_by_qid.setdefault(qid, {})[source_id] = row
        decision = row["review_decision"].strip()
        if not decision:
            source_decisions["pending"] += 1
            incomplete.append(key)
            continue
        if decision not in SOURCE_DECISIONS:
            raise ValueError(f"Unknown source review decision: {key}")
        _review_identity(row, key)
        if decision in {"supports", "partial"}:
            if not row["source_passage_locator"].strip() or not row["supported_claim_summary"].strip():
                raise ValueError(f"Source support requires passage and summary: {key}")
        source_decisions[decision] += 1

    links_by_qid = {}
    link_decisions = Counter()
    for row in links:
        key, qid = row["link_task_id"], row["query_id"]
        links_by_qid.setdefault(qid, []).append(row)
        decision = row["review_decision"].strip()
        if not decision:
            link_decisions["pending"] += 1
            incomplete.append(key)
            continue
        if decision not in LINK_DECISIONS:
            raise ValueError(f"Unknown link review decision: {key}")
        _review_identity(row, key)
        ids = _parse_ids(row["supporting_source_ids"])
        allowed = sources_by_qid.get(qid, {})
        if not set(ids).issubset(allowed):
            raise ValueError(f"Unknown supporting source for link: {key}")
        if decision == "supports":
            if not all(row[field].strip() for field in (
                "cause_node", "effect_node", "relation", "source_passage_locator"
            )):
                raise ValueError(f"Supported link requires typed nodes and passage: {key}")
            if row["relation"] not in RELATIONS or row["relation"] == "UNCERTAIN":
                raise ValueError(f"Invalid verified candidate link relation: {key}")
            if not ids or not all(allowed[s]["review_decision"] == "supports" for s in ids):
                raise ValueError(f"Supported link requires separately supported sources: {key}")
        link_decisions[decision] += 1

    question_decisions = Counter()
    warning_ids = []
    for row in questions:
        qid = row["query_id"]
        decision = row["answerability_decision"].strip()
        if not decision:
            question_decisions["pending"] += 1
            incomplete.append(qid)
            continue
        if decision not in QUESTION_DECISIONS:
            raise ValueError(f"Unknown answerability decision: {qid}")
        # Apply the categorical prediction guard before source-reference checks.
        # PILOT_093 deliberately has no registered source, so the unsupported
        # exact-person prediction rule is the more informative primary error.
        if decision == "answerable" and row["query_type"] == "unsupported":
            raise ValueError(f"Unsupported exact-person prediction cannot be labeled answerable: {qid}")
        _review_identity(row, qid)
        source_ids = _parse_ids(row["supporting_source_ids"])
        allowed = sources_by_qid.get(qid, {})
        if not set(source_ids).issubset(allowed):
            raise ValueError(f"Question cites unregistered source: {qid}")
        if decision == "answerable":
            if not row["draft_reference_answer"].strip() or not source_ids:
                raise ValueError(f"Answerable question needs a source-backed draft: {qid}")
            if not all(allowed[s]["review_decision"] == "supports" for s in source_ids):
                raise ValueError(f"Answerable question cites unreviewed/partial source: {qid}")
            if row["needs_additional_sources"] == "true":
                if not row["scope_resolution_note"].strip():
                    raise ValueError(f"Source discovery gap needs explanation before answerability: {qid}")
                warning_ids.append(qid)
            if row["query_type"] == "risk_factor" and not row["risk_vs_cause_note"].strip():
                raise ValueError(f"Risk factor question needs causal-scope note: {qid}")
            if row["query_type"] == "comparative":
                for name in (
                    "comparison_topic_a", "comparison_topic_b",
                    "comparison_evidence_note",
                ):
                    if not row[name].strip():
                        raise ValueError(f"Comparison needs both sides and evidence note: {qid}")
                side_a = _parse_ids(row["comparison_source_ids_a"])
                side_b = _parse_ids(row["comparison_source_ids_b"])
                if not side_a or not side_b or not set(side_a + side_b).issubset(source_ids):
                    raise ValueError(f"Comparison needs both registered evidence sides: {qid}")
            if row["query_type"] == "multi_hop":
                status = row["candidate_path_status"]
                if status not in PATH_STATUSES:
                    raise ValueError(f"Multi-hop answer needs candidate path status: {qid}")
                lrows = sorted(links_by_qid[qid], key=lambda x: int(x["link_slot"]))
                if any(not link["review_decision"] for link in lrows):
                    raise ValueError(f"Answerable multi-hop requires two reviewed link slots: {qid}")
                supported = [link for link in lrows if link["review_decision"] == "supports"]
                if status == "two_links_source_supported":
                    if len(supported) != 2:
                        raise ValueError(f"Two-link status needs two supported links: {qid}")
                    if supported[0]["effect_node"].casefold().strip() != supported[1]["cause_node"].casefold().strip():
                        raise ValueError(f"Two-link candidate nodes do not connect: {qid}")
                if status == "partial_links_supported" and len(supported) != 1:
                    raise ValueError(f"Partial-link status requires exactly one supported link: {qid}")
                if status == "no_supported_chain" and len(supported) == 2:
                    # Both links can be individually supported but fail to connect.
                    pass
        elif decision == "unanswerable":
            if not row["abstention_reason"].strip():
                raise ValueError(f"Unanswerable question needs abstention reason: {qid}")
        question_decisions[decision] += 1

    if strict and incomplete:
        raise ValueError(
            f"Incomplete remaining70 review decisions ({len(incomplete)}): "
            + ", ".join(incomplete[:7])
        )
    return {
        "schema_version": 1,
        "data_role": "remaining70_review_workflow_audit_not_medical_gold",
        "question_tasks": len(questions),
        "source_tasks": len(sources),
        "unverified_link_review_slots": len(links),
        "question_decision_counts": dict(sorted(question_decisions.items())),
        "source_decision_counts": dict(sorted(source_decisions.items())),
        "link_decision_counts": dict(sorted(link_decisions.items())),
        "pending_decisions": len(incomplete),
        "strict_decisions_required": bool(strict),
        "answerability_with_original_source_gap_flag": sorted(warning_ids),
        "reviewer_identity_verified_by_code": False,
        "medical_correctness_verified_by_code": False,
        "causal_paths_verified_by_code": False,
        "gold_labels_created": False,
        "final_splits_assigned": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--audit-dir", type=Path)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--pilot", type=Path, default=PILOT)
    parser.add_argument("--base-sources", type=Path, default=BASE_SOURCES)
    parser.add_argument("--extra-sources", type=Path, default=EXTRA_SOURCES)
    parser.add_argument("--source-map", type=Path, default=SOURCE_MAP)
    args = parser.parse_args()
    if bool(args.output_dir) == bool(args.audit_dir):
        parser.error("Exactly one of --output-dir or --audit-dir is required")
    kwargs = {
        "pilot_file": args.pilot,
        "base_sources_file": args.base_sources,
        "extra_sources_file": args.extra_sources,
        "source_map_file": args.source_map,
    }
    if args.output_dir:
        profile = create_remaining_review_workspace(args.output_dir, **kwargs)
    else:
        profile = audit_remaining_review_workspace(
            args.audit_dir, strict=args.strict, **kwargs
        )
    print(json.dumps(profile, indent=2))


if __name__ == "__main__":
    main()
