"""Phase 2I-C1: separate, provenance-checked supplemental source review.

Does not migrate or mutate the existing 70-question decisions. Additional
sources remain candidates until passage-level human review is recorded.
"""
import argparse
import csv
import hashlib
import io
import json
from collections import Counter
from datetime import date
from pathlib import Path

REGISTRY = Path(__file__).resolve().parents[1] / "data/benchmark/indirect5_source_extension_v1.jsonl"
EDITABLE = ("review_decision", "source_passage_locator", "supported_claim_summary",
            "limitation_note", "reviewer_id", "reviewed_date", "review_note")
DECISIONS = {"supports", "partial", "does_not_support", "needs_more_evidence"}
REQUIRED_IDS = {"PILOT_034", "PILOT_035", "PILOT_036", "PILOT_038", "PILOT_044"}

def checksum(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def registry_rows(registry=REGISTRY):
    rows = [json.loads(line) for line in Path(registry).read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != 5 or {r["query_id"] for r in rows} != REQUIRED_IDS:
        raise ValueError("Expected exactly five distinct indirect question candidates")
    ids = [r["source_id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate supplemental source IDs")
    for r in rows:
        if r["candidate_status"] != "needs_human_source_review" or r["medical_gold_created"] is not False:
            raise ValueError("Supplemental sources must remain unverified candidates")
    return rows

def expected_rows(registry=REGISTRY):
    return [{
        "source_task_id": r["query_id"] + ":" + r["source_id"],
        "query_id": r["query_id"], "candidate_source_id": r["source_id"],
        "candidate_source_title": r["title"], "candidate_source_url": r["url"],
        "scope_note": r["scope_note"],
        **{field: "" for field in EDITABLE}
    } for r in registry_rows(registry)]

def read_csv(path, fields):
    with Path(path).open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != fields:
            raise ValueError("Review worksheet columns changed")
        rows = list(reader)
    if any(None in row for row in rows):
        raise ValueError("Malformed review CSV")
    return rows

def create(output_dir, baseline_dir, registry=REGISTRY):
    baseline = Path(baseline_dir)
    needed = ("question_review.csv", "source_review.csv", "link_review.csv", "review_manifest.json")
    if not all((baseline / p).is_file() for p in needed):
        raise ValueError("Missing existing audited 70-question baseline files")
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=False)
    rows = expected_rows(registry)
    with (root / "supplemental_source_review.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    manifest = {
        "schema_version": 1,
        "purpose": "supplemental_source_review_not_medical_gold",
        "registry_sha256": checksum(registry),
        "baseline_checksums": {p: checksum(baseline / p) for p in needed},
        "task_count": len(rows), "gold_labels_created": False,
        "existing_reviews_migrated": False, "existing_reviews_modified": False,
    }
    (root / "supplement_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest

def audit(review_dir, baseline_dir, registry=REGISTRY, strict=False):
    root, baseline = Path(review_dir), Path(baseline_dir)
    manifest = json.loads((root / "supplement_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("purpose") != "supplemental_source_review_not_medical_gold":
        raise ValueError("Unexpected supplement purpose")
    if checksum(registry) != manifest.get("registry_sha256"):
        raise ValueError("Supplemental source registry provenance changed")
    for name, original_hash in manifest["baseline_checksums"].items():
        if checksum(baseline / name) != original_hash:
            raise ValueError("Baseline review snapshot modified: " + name)
    expected = expected_rows(registry)
    actual = read_csv(root / "supplemental_source_review.csv", list(expected[0]))
    if len(expected) != len(actual):
        raise ValueError("Supplement worksheet row count changed")
    reference = {r["source_task_id"]: r for r in expected}
    counts = Counter()
    visited = set()
    for row in actual:
        key = row["source_task_id"]
        if key not in reference or key in visited:
            raise ValueError("Unknown or duplicate source review task: " + key)
        visited.add(key)
        for field in reference[key]:
            if field not in EDITABLE and row[field] != reference[key][field]:
                raise ValueError("Original supplemental candidate modified: " + key + "." + field)
        decision = row["review_decision"].strip()
        if not decision:
            counts["pending"] += 1
            continue
        if decision not in DECISIONS:
            raise ValueError("Invalid supplemental review decision: " + key)
        try:
            date_ok = date.fromisoformat(row["reviewed_date"]).isoformat() == row["reviewed_date"]
        except ValueError:
            date_ok = False
        if not row["reviewer_id"].strip() or not date_ok or not row["review_note"].strip():
            raise ValueError("Reviewer identity, ISO date and rationale required: " + key)
        if decision in {"supports", "partial"}:
            if not row["source_passage_locator"].strip() or not row["supported_claim_summary"].strip():
                raise ValueError("Passage and supported claim required: " + key)
        counts[decision] += 1
    if strict and counts["pending"]:
        raise ValueError("Pending supplemental reviews: " + str(counts["pending"]))
    return {
        "source_tasks": len(actual), "decisions": dict(sorted(counts.items())),
        "baseline_snapshot_unchanged": True, "existing_reviews_migrated": False,
        "medical_correctness_verified_by_code": False, "gold_labels_created": False,
    }

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--output-dir", type=Path)
    group.add_argument("--audit-dir", type=Path)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--registry", type=Path, default=REGISTRY)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    if args.output_dir:
        result = create(args.output_dir, args.baseline_dir, args.registry)
    else:
        result = audit(args.audit_dir, args.baseline_dir, args.registry, args.strict)
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
