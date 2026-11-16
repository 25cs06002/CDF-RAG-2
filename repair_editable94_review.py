#!/usr/bin/env python3
"""Restore immutable fields in Phase 2I-C3 editable reviews from C2 handoff.

Only restores metadata and missing ADD001..ADD005 source rows. Never changes
review/answerability decisions already present in the editable worksheets.
The combined handoff is read-only. Run --apply to write (backs up first).
"""
import argparse
import csv
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

QUESTION_EDITABLE = {
    "answerability_decision", "draft_reference_answer", "supporting_source_ids",
    "abstention_reason", "scope_resolution_note", "candidate_path_status",
    "comparison_topic_a", "comparison_topic_b", "comparison_source_ids_a",
    "comparison_source_ids_b", "comparison_evidence_note", "risk_vs_cause_note",
    "reviewer_id", "reviewed_date", "review_note",
}
SOURCE_EDITABLE = {
    "review_decision", "source_passage_locator", "supported_claim_summary",
    "limitation_note", "reviewer_id", "reviewed_date", "review_note",
}
ALLOWED_SUPPLEMENT_IDS = {"ADD001", "ADD002", "ADD003", "ADD004", "ADD005"}


def read_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError(f"Missing or duplicate CSV headers in {path}")
        rows = list(reader)
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError(f"Malformed rows in {path}")
    return reader.fieldnames, rows


def merge(original, current, fieldnames, key, editable, allow_additions=False):
    if fieldnames != list(original[0]):
        raise ValueError(f"Headers/order differ from combined snapshot for {key}")
    def indexed(rows):
        result = {}
        for r in rows:
            ident = r[key]
            if not ident or ident in result:
                raise ValueError(f"Missing/duplicate {key}: {ident!r}")
            result[ident] = r
        return result
    old = indexed(original)
    user = indexed(current)
    unexpected = set(user) - set(old)
    if unexpected:
        raise ValueError(f"Unexpected {key} entries: {sorted(unexpected)[:5]}")
    missing = set(old) - set(user)
    if missing and not allow_additions:
        raise ValueError(f"Missing question rows: {sorted(missing)[:10]}")
    if allow_additions and any(x.split(":")[-1] not in ALLOWED_SUPPLEMENT_IDS for x in missing):
        raise ValueError(f"Missing non-supplement source-review rows: {sorted(missing)}")
    changes = []
    output = []
    for base in original:
        ident = base[key]
        if ident not in user:
            output.append(base.copy())
            continue
        row = user[ident]
        for column in base:
            if column not in editable and row[column] != base[column]:
                changes.append((ident, column, row[column], base[column]))
        restored = base.copy()
        for column in editable:
            restored[column] = row[column]
        output.append(restored)
    return output, changes, sorted(missing)


def write_csv_atomic(path, fields, rows):
    temp = path.with_name(path.name + ".repair-tmp")
    try:
        with temp.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--combined-dir", type=Path, required=True)
    p.add_argument("--editable-dir", type=Path, required=True)
    p.add_argument("--apply", action="store_true", help="Write after backing up both CSVs")
    args = p.parse_args()
    combined, editable = args.combined_dir, args.editable_dir
    if combined.resolve() == editable.resolve():
        raise SystemExit("ERROR: combined and editable directories must be different")
    outputs = []
    for filename, key, editfields, can_add in (
        ("question_review.csv", "query_id", QUESTION_EDITABLE, False),
        ("source_review.csv", "source_task_id", SOURCE_EDITABLE, True),
    ):
        reference_path = combined / filename
        active_path = editable / filename
        ref_fields, ref_rows = read_csv(reference_path)
        active_fields, active_rows = read_csv(active_path)
        if ref_fields != active_fields:
            raise ValueError(f"Column headers differ: {filename}")
        merged, changes, additions = merge(ref_rows, active_rows, ref_fields, key, editfields, can_add)
        outputs.append((filename, active_path, ref_fields, merged, changes, additions))
        print(f"\n{filename}: current={len(active_rows)}, restored={len(merged)}")
        print(f"  Protected-field repairs: {len(changes)}")
        print(f"  Source rows restored from verified handoff: {len(additions)}")
        for ident, field, old, new in changes[:15]:
            print(f"  {ident}.{field}: {old!r} -> {new!r}")
        if len(changes) > 15:
            print(f"  ... and {len(changes)-15} more protected-field repairs")
        if additions:
            print("  Appended:", ", ".join(additions))
    if not args.apply:
        print("\nDRY RUN ONLY: No files changed. Re-run with --apply to repair.")
        return
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    for filename, path, fields, rows, changes, additions in outputs:
        backup = path.with_name(filename + ".bak-before-repair-" + timestamp)
        if backup.exists():
            raise FileExistsError(f"Backup already exists: {backup}")
    for filename, path, fields, rows, changes, additions in outputs:
        backup = path.with_name(filename + ".bak-before-repair-" + timestamp)
        backup.write_bytes(path.read_bytes())
        write_csv_atomic(path, fields, rows)
        print("Saved", path, "| Backup", backup)
    print("\nREPAIRED. Run the editable_combined_review audit next.")
    print("No reviewer decision or editable review note was automatically changed.")


if __name__ == "__main__":
    main()
