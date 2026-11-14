"""Phase 2K: reusable, independent multi-hop source/link review workspaces.

This does not mutate the original 70-question snapshot and never generates
medical gold labels. Registry records are candidate data, not instructions.

Usage:
 python3 -m research_v2.multihop_evidence_review --create --question-id PILOT_047 \
   --snapshot-dir SNAP --registry REGISTRY.jsonl --workspace-dir OUT
 python3 -m research_v2.multihop_evidence_review --audit --question-id PILOT_047 \
   --snapshot-dir SNAP --registry REGISTRY.jsonl --workspace-dir OUT --strict
"""
import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from datetime import date
from pathlib import Path
from urllib.parse import urlparse

SOURCE_EDIT = ("review_decision", "verified_passage_locator", "verified_claim", "verified_relation", "limitation_note", "reviewer_id", "reviewed_date", "review_note")
LINK_EDIT = ("review_decision", "cause_node", "effect_node", "relation", "supporting_source_ids", "source_passage_locator", "reviewer_id", "reviewed_date", "review_note")
DECISIONS = {"supports", "partial", "does_not_support", "needs_more_evidence"}
RELATIONS = {"CAUSES", "RISK_FACTOR_FOR", "ASSOCIATED_WITH", "PROTECTS_AGAINST"}
SNAP_FILES = ("question_review.csv", "source_review.csv", "link_review.csv", "editable_manifest.json")
QID_RE = re.compile(r"^PILOT_(0(?:4[6-9]|5[0-9])|060)$")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_csv(path):
    with Path(path).open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError(f"Invalid CSV header: {path}")
        rows = list(reader)
        if any(None in r for r in rows):
            raise ValueError(f"Malformed CSV rows: {path}")
        return rows, reader.fieldnames


def write_csv(path, rows):
    if not rows:
        raise ValueError("No review rows")
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def checkpoint(snapshot, qid):
    checks = snapshot / "SHA256SUMS.txt"
    if not checks.is_file():
        raise ValueError("Missing frozen SHA256SUMS.txt")
    pairs = {}
    for line in checks.read_text(encoding="utf-8").splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})\s+\*?([^/\\]+)", line.strip())
        if not match or match.group(2) in pairs:
            raise ValueError("Malformed or duplicate frozen checksum entry")
        pairs[match.group(2)] = match.group(1)
    for name in SNAP_FILES:
        if not (snapshot / name).is_file() or pairs.get(name) != digest(snapshot / name):
            raise ValueError("Frozen checkpoint missing or modified: " + name)
    qrows, _ = read_csv(snapshot / "question_review.csv")
    srows, _ = read_csv(snapshot / "source_review.csv")
    lrows, _ = read_csv(snapshot / "link_review.csv")
    if tuple(map(len, (qrows, srows, lrows))) != (70, 94, 30):
        raise ValueError("Unexpected frozen checkpoint dimensions")
    if sum(r["query_id"] == qid for r in qrows) != 1:
        raise ValueError("Question missing or duplicated: " + qid)
    relevant = [r for r in lrows if r["query_id"] == qid]
    if {r["link_task_id"] for r in relevant} != {qid + ":LINK1", qid + ":LINK2"}:
        raise ValueError("Frozen question does not contain exactly two link slots")
    existing = {}
    for r in srows:
        existing.setdefault(r["query_id"], {})[r["candidate_source_id"]] = r
    return {name: digest(snapshot / name) for name in SNAP_FILES}, existing.get(qid, {})


def registry_rows(path, qid, existing):
    entries = [json.loads(s) for s in Path(path).read_text(encoding="utf-8").splitlines() if s.strip()]
    if not entries:
        raise ValueError("Empty source registry")
    seen_pairs, new_ids = set(), {}
    for r in entries:
        sid, link, url = r["source_id"], r["link_task_id"], r["source_url"]
        if r.get("query_id") != qid or link not in (qid + ":LINK1", qid + ":LINK2"):
            raise ValueError("Registry question or link mismatch")
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{1,39}", sid):
            raise ValueError("Invalid source ID")
        parsed = urlparse(url)
        if parsed.scheme != "https" or not parsed.netloc or parsed.username or parsed.password:
            raise ValueError("Source URL must be HTTPS and valid")
        if (sid, link) in seen_pairs:
            raise ValueError("Duplicate registry source/link pair")
        seen_pairs.add((sid, link))
        if r.get("medical_gold_created") is not False or r.get("human_confirmation_pending") is not True:
            raise ValueError("Registry must remain provisional")
        if sid in existing:
            if r.get("registration_status") != "existing_source" or existing[sid]["candidate_source_url"] != url:
                raise ValueError("Existing source mismatch: " + sid)
        else:
            if r.get("registration_status") != "proposed_new_source":
                raise ValueError("New source must be proposed: " + sid)
            if sid in new_ids and new_ids[sid] != url:
                raise ValueError("Source ID reused for different URLs: " + sid)
            new_ids[sid] = url
    if {r["link_task_id"] for r in entries} != {qid + ":LINK1", qid + ":LINK2"}:
        raise ValueError("Both link slots require candidate source entries")
    return entries


def expected_source_rows(entries):
    return [{
        "source_task_id": r["link_task_id"] + ":" + r["source_id"],
        "link_task_id": r["link_task_id"], "source_id": r["source_id"],
        "source_url": r["source_url"], "registration_status": r["registration_status"],
        "provisional_decision": r.get("review_decision", ""),
        "provisional_passage": r.get("passage_section", ""),
        "provisional_claim": r.get("claim_supported", ""),
        **{name: "" for name in SOURCE_EDIT},
    } for r in entries]


def expected_link_rows(qid, entries):
    return [{
        "link_task_id": f"{qid}:LINK{i}", "query_id": qid, "link_slot": str(i),
        "expected_source_ids": ";".join(dict.fromkeys(r["source_id"] for r in entries if r["link_task_id"] == f"{qid}:LINK{i}")),
        **{name: "" for name in LINK_EDIT},
    } for i in (1, 2)]


def identity(row):
    if not row["reviewer_id"].strip() or not row["review_note"].strip():
        raise ValueError("Reviewer ID and rationale required")
    try:
        if date.fromisoformat(row["reviewed_date"]).isoformat() != row["reviewed_date"]:
            raise ValueError("Bad ISO review date")
    except ValueError as ex:
        raise ValueError("Review date must be YYYY-MM-DD") from ex


def create(snapshot, registry, out, qid):
    hashes, existing = checkpoint(snapshot, qid)
    entries = registry_rows(registry, qid, existing)
    if out.exists():
        raise FileExistsError("Refusing to overwrite " + str(out))
    out.mkdir(parents=True)
    write_csv(out / "source_review.csv", expected_source_rows(entries))
    write_csv(out / "link_review.csv", expected_link_rows(qid, entries))
    manifest = {
        "schema_version": 1, "purpose": "phase2k_multihop_candidate_review_not_medical_gold",
        "question_id": qid, "snapshot_hashes": hashes,
        "registry_sha256": digest(registry), "auto_promoted_decisions": False,
        "medical_gold_created": False,
    }
    (out / "review_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return audit(snapshot, registry, out, qid)


def audit(snapshot, registry, out, qid, strict=False):
    hashes, existing = checkpoint(snapshot, qid)
    entries = registry_rows(registry, qid, existing)
    manifest = json.loads((out / "review_manifest.json").read_text(encoding="utf-8"))
    if (manifest.get("purpose") != "phase2k_multihop_candidate_review_not_medical_gold"
        or manifest.get("question_id") != qid or manifest.get("snapshot_hashes") != hashes
        or manifest.get("registry_sha256") != digest(registry)
        or manifest.get("medical_gold_created") is not False):
        raise ValueError("Manifest provenance mismatch")
    source_expect = expected_source_rows(entries)
    link_expect = expected_link_rows(qid, entries)
    source_rows, source_fields = read_csv(out / "source_review.csv")
    link_rows, link_fields = read_csv(out / "link_review.csv")
    for actual, expected, key, editable, fields in (
        (source_rows, source_expect, "source_task_id", SOURCE_EDIT, source_fields),
        (link_rows, link_expect, "link_task_id", LINK_EDIT, link_fields),
    ):
        if len(actual) != len(expected) or fields != list(expected[0]):
            raise ValueError("Wrong worksheet count/columns")
        exp = {r[key]: r for r in expected}
        if len(exp) != len(expected) or len({r[key] for r in actual}) != len(actual):
            raise ValueError("Duplicate worksheet ID")
        for r in actual:
            if r[key] not in exp:
                raise ValueError("Unregistered task: " + r[key])
            for field, v in exp[r[key]].items():
                if field not in editable and r[field] != v:
                    raise ValueError("Protected field altered: " + r[key] + "." + field)
    source_counts, link_counts = Counter(), Counter()
    sources = {r["source_task_id"]: r for r in source_rows}
    for r in source_rows:
        decision = r["review_decision"].strip()
        source_counts[decision or "pending"] += 1
        if not decision:
            continue
        if decision not in DECISIONS:
            raise ValueError("Invalid source decision")
        identity(r)
        if decision in ("supports", "partial"):
            if not r["verified_passage_locator"].strip() or not r["verified_claim"].strip() or r["verified_relation"] not in RELATIONS:
                raise ValueError("Source requires verified passage, claim and relation")
    links = {}
    for r in link_rows:
        key = r["link_task_id"]
        links[key] = r
        decision = r["review_decision"].strip()
        link_counts[decision or "pending"] += 1
        if not decision:
            continue
        if decision not in DECISIONS:
            raise ValueError("Invalid link decision")
        identity(r)
        ids = [v.strip() for v in r["supporting_source_ids"].split(";") if v.strip()]
        if len(ids) != len(set(ids)) or not set(ids).issubset(r["expected_source_ids"].split(";")):
            raise ValueError("Unknown or duplicate supporting source: " + key)
        if decision == "supports":
            if not ids or not all(r[field].strip() for field in ("cause_node", "effect_node", "source_passage_locator")) or r["relation"] not in RELATIONS:
                raise ValueError("Supported link missing evidence fields: " + key)
            for sid in ids:
                sr = sources[f"{key}:{sid}"]
                if sr["review_decision"] != "supports":
                    raise ValueError("Link cites source without support: " + sid)
                # Source-level and link-level relations must agree when declaring source support.
                if sr["verified_relation"] != r["relation"]:
                    raise ValueError("Source/link relation mismatch: " + sid)
    if strict and (source_counts["pending"] or link_counts["pending"]):
        raise ValueError("Review decisions pending")
    a, b = links[qid + ":LINK1"], links[qid + ":LINK2"]
    connected = None
    if a["review_decision"] == b["review_decision"] == "supports":
        connected = a["effect_node"].strip().casefold() == b["cause_node"].strip().casefold()
        if not connected:
            raise ValueError("Two supported links do not connect")
    return {
        "status": "audit_passed", "question_id": qid,
        "source_tasks": len(source_rows), "link_tasks": len(link_rows),
        "source_decisions": dict(source_counts), "link_decisions": dict(link_counts),
        "two_link_chain_connected": connected, "strict": strict,
        "frozen_snapshot_unchanged": True, "medical_correctness_verified_by_code": False,
        "gold_labels_created": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--create", action="store_true")
    mode.add_argument("--audit", action="store_true")
    parser.add_argument("--question-id", required=True)
    parser.add_argument("--snapshot-dir", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--workspace-dir", type=Path, required=True)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    if not QID_RE.fullmatch(args.question_id):
        parser.error("Question ID must be PILOT_046 through PILOT_060")
    result = (create(args.snapshot_dir, args.registry, args.workspace_dir, args.question_id)
              if args.create else audit(args.snapshot_dir, args.registry, args.workspace_dir, args.question_id, args.strict))
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
