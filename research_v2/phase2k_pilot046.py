"""Phase 2K: isolated PILOT_046 source/link adjudication workspace.

No existing benchmark source, frozen review, or gold labels are modified.
Run --create to create a new workspace, --audit to validate it.
"""
import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import date
from pathlib import Path
from urllib.parse import urlparse

EXPECTED = {
    "K046S01": ("PILOT_046:LINK1", "https://www.cdc.gov/tobacco/about/cigarettes-and-cardiovascular-disease.html"),
    "K046S02": ("PILOT_046:LINK1", "https://www.nhlbi.nih.gov/health/heart/smoking/risks"),
    "SRC001": ("PILOT_046:LINK2", "https://www.nhlbi.nih.gov/health/coronary-heart-disease/causes"),
}
SOURCE_EDIT = ("review_decision", "verified_passage_locator", "verified_claim", "verified_relation", "limitation_note", "reviewer_id", "reviewed_date", "review_note")
LINK_EDIT = ("review_decision", "cause_node", "effect_node", "relation", "supporting_source_ids", "source_passage_locator", "reviewer_id", "reviewed_date", "review_note")
DECISIONS = {"supports", "partial", "does_not_support", "needs_more_evidence"}
RELATIONS = {"CAUSES", "RISK_FACTOR_FOR", "ASSOCIATED_WITH", "PROTECTS_AGAINST"}
FILES = ("question_review.csv", "source_review.csv", "link_review.csv", "editable_manifest.json", "SHA256SUMS.txt")

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read_csv(path):
    with Path(path).open(encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        if not r.fieldnames or len(r.fieldnames) != len(set(r.fieldnames)):
            raise ValueError(f"Invalid CSV header: {path}")
        rows = list(r)
        if any(None in x for x in rows):
            raise ValueError(f"Malformed CSV: {path}")
        return rows, r.fieldnames

def write_csv(path, fields, rows):
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fields)
        w.writeheader()
        w.writerows(rows)

def registry_rows(path):
    data = [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]
    if len(data) != 3 or {r["source_id"] for r in data} != set(EXPECTED):
        raise ValueError("Expected exactly K046S01, K046S02, SRC001")
    for r in data:
        sid = r["source_id"]
        if (r["link_task_id"], r["source_url"]) != EXPECTED[sid] or r["query_id"] != "PILOT_046":
            raise ValueError(f"Unexpected identity/URL for {sid}")
        if r.get("registration_status") != ("existing_source" if sid == "SRC001" else "proposed_new_source"):
            raise ValueError(f"Bad registration status for {sid}")
        if r.get("medical_gold_created") is not False or r.get("human_confirmation_pending") is not True:
            raise ValueError(f"Registry must remain provisional: {sid}")
        if urlparse(r["source_url"]).scheme != "https":
            raise ValueError("Non-HTTPS source")
    return data

def assert_checkpoint(snapshot):
    for f in FILES:
        if not (snapshot / f).is_file():
            raise ValueError(f"Missing frozen checkpoint file: {f}")
    expected = {}
    for line in (snapshot / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            raise ValueError("Malformed checksum file")
        expected[parts[1].lstrip("* ")] = parts[0]
    for f in FILES[:-1]:
        if expected.get(f) != sha(snapshot / f):
            raise ValueError(f"Frozen checkpoint checksum mismatch: {f}")
    qs, _ = read_csv(snapshot / "question_review.csv")
    ss, _ = read_csv(snapshot / "source_review.csv")
    ls, _ = read_csv(snapshot / "link_review.csv")
    if (len(qs), len(ss), len(ls)) != (70, 94, 30):
        raise ValueError("Unexpected frozen checkpoint dimensions")
    if not any(x["query_id"] == "PILOT_046" for x in qs):
        raise ValueError("PILOT_046 not in checkpoint")
    for link in ls:
        if link["query_id"] == "PILOT_046" and link["review_decision"] != "needs_more_evidence":
            raise ValueError("PILOT_046 baseline links are not both evidence gaps")
    baseline = [x for x in ss if x["source_task_id"] == "PILOT_046:SRC001"]
    if len(baseline) != 1 or baseline[0]["candidate_source_url"] != EXPECTED["SRC001"][1]:
        raise ValueError("SRC001 URL does not match frozen source review")
    if baseline[0]["review_decision"] != "partial":
        raise ValueError("Expected existing SRC001 overall-source decision to be partial")
    # Avoid collision with any other registered source identifier in the checkpoint.
    used = {x["candidate_source_id"] for x in ss}
    if {"K046S01", "K046S02"} & used:
        raise ValueError("Proposed source ID collides with existing source")
    return {f: sha(snapshot / f) for f in FILES}

def expected_source_rows(reg):
    result = []
    for r in reg:
        result.append({
            "source_task_id": "PILOT_046:" + r["source_id"],
            "link_task_id": r["link_task_id"],
            "source_id": r["source_id"],
            "source_url": r["source_url"],
            "registration_status": r["registration_status"],
            "provisional_decision": r.get("review_decision", ""),
            "provisional_passage": r.get("passage_section", ""),
            "provisional_claim": r.get("claim_supported", ""),
            **{k: "" for k in SOURCE_EDIT},
        })
    return result

def expected_link_rows():
    return [
        {"link_task_id": f"PILOT_046:LINK{i}", "query_id": "PILOT_046", "link_slot": str(i),
         "expected_source_ids": "K046S01;K046S02" if i == 1 else "SRC001",
         **{k: "" for k in LINK_EDIT}} for i in (1, 2)
    ]

def validate_identity(row):
    if not row["reviewer_id"].strip() or not row["review_note"].strip():
        raise ValueError("Reviewed task needs reviewer ID and note")
    try:
        if date.fromisoformat(row["reviewed_date"]).isoformat() != row["reviewed_date"]:
            raise ValueError("Invalid review date")
    except ValueError:
        raise ValueError("Review date must be YYYY-MM-DD")

def create(snapshot, registry, out):
    hashes = assert_checkpoint(snapshot)
    reg = registry_rows(registry)
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite {out}")
    out.mkdir(parents=True)
    sources = expected_source_rows(reg)
    links = expected_link_rows()
    write_csv(out / "pilot046_source_review.csv", list(sources[0]), sources)
    write_csv(out / "pilot046_link_review.csv", list(links[0]), links)
    manifest = {
        "schema_version": 1,
        "purpose": "pilot046_provisional_source_and_link_adjudication_not_medical_gold",
        "snapshot_hashes": hashes,
        "registry_sha256": sha(registry),
        "source_count": 3, "link_count": 2,
        "auto_promoted_decisions": False,
        "medical_gold_created": False,
    }
    (out / "pilot046_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return audit(snapshot, registry, out)

def audit(snapshot, registry, out, strict=False):
    hashes = assert_checkpoint(snapshot)
    reg = registry_rows(registry)
    manifest = json.loads((out / "pilot046_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("purpose") != "pilot046_provisional_source_and_link_adjudication_not_medical_gold":
        raise ValueError("Wrong manifest purpose")
    if manifest.get("snapshot_hashes") != hashes or manifest.get("registry_sha256") != sha(registry):
        raise ValueError("Source provenance hash mismatch")
    source_rows, source_fields = read_csv(out / "pilot046_source_review.csv")
    link_rows, link_fields = read_csv(out / "pilot046_link_review.csv")
    source_expect = expected_source_rows(reg)
    link_expect = expected_link_rows()
    if (len(source_rows), len(link_rows)) != (3, 2):
        raise ValueError("Unexpected review task count")
    for actual, expected, key, editable, actual_fields in (
        (source_rows, source_expect, "source_task_id", SOURCE_EDIT, source_fields),
        (link_rows, link_expect, "link_task_id", LINK_EDIT, link_fields),
    ):
        if actual_fields != list(expected[0]):
            raise ValueError("Column mismatch")
        by_id = {r[key]: r for r in expected}
        if len({r[key] for r in actual}) != len(expected):
            raise ValueError("Duplicate task IDs")
        for row in actual:
            if row[key] not in by_id:
                raise ValueError("Unknown task " + row[key])
            for f in by_id[row[key]]:
                if f not in editable and row[f] != by_id[row[key]][f]:
                    raise ValueError(f"Protected field altered: {row[key]}.{f}")
    source_decisions, link_decisions = Counter(), Counter()
    by_sid = {}
    for row in source_rows:
        sid = row["source_id"]
        by_sid[sid] = row
        d = row["review_decision"].strip()
        source_decisions[d or "pending"] += 1
        if not d:
            continue
        if d not in DECISIONS:
            raise ValueError("Bad source decision: " + sid)
        validate_identity(row)
        if d in ("supports", "partial"):
            if not row["verified_passage_locator"].strip() or not row["verified_claim"].strip():
                raise ValueError("Missing verified source locator/claim: " + sid)
            if row["verified_relation"] not in RELATIONS:
                raise ValueError("Bad source relation: " + sid)
    for row in link_rows:
        key = row["link_task_id"]
        d = row["review_decision"].strip()
        link_decisions[d or "pending"] += 1
        if not d:
            continue
        if d not in DECISIONS:
            raise ValueError("Bad link decision: " + key)
        validate_identity(row)
        ids = [s.strip() for s in row["supporting_source_ids"].split(";") if s.strip()]
        if len(ids) != len(set(ids)) or not set(ids).issubset(set(row["expected_source_ids"].split(";"))):
            raise ValueError("Unknown/duplicate link source: " + key)
        if d == "supports":
            if not all(row[x].strip() for x in ("cause_node", "effect_node", "relation", "source_passage_locator")):
                raise ValueError("Supported link lacks nodes/relation/locator: " + key)
            if row["relation"] not in RELATIONS or not ids:
                raise ValueError("Invalid supported link relation/sources: " + key)
            if not all(by_sid[s]["review_decision"] == "supports" for s in ids):
                raise ValueError("Supported link cites non-supporting source: " + key)
    if strict and (source_decisions.get("pending", 0) or link_decisions.get("pending", 0)):
        raise ValueError("Pending review decisions")
    supported = {r["link_task_id"]: r for r in link_rows if r["review_decision"] == "supports"}
    chain_connected = None
    if len(supported) == 2:
        a, b = supported["PILOT_046:LINK1"], supported["PILOT_046:LINK2"]
        chain_connected = a["effect_node"].strip().casefold() == b["cause_node"].strip().casefold()
    return {
        "status": "audit_passed", "source_tasks": 3, "link_tasks": 2,
        "source_decisions": dict(source_decisions), "link_decisions": dict(link_decisions),
        "two_link_chain_connected": chain_connected,
        "strict": strict, "frozen_snapshot_unchanged": True,
        "medical_correctness_verified_by_code": False, "gold_labels_created": False,
    }

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--create", action="store_true")
    mode.add_argument("--audit", action="store_true")
    parser.add_argument("--snapshot-dir", required=True, type=Path)
    parser.add_argument("--registry", required=True, type=Path)
    parser.add_argument("--workspace-dir", required=True, type=Path)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    result = (create(args.snapshot_dir, args.registry, args.workspace_dir) if args.create
              else audit(args.snapshot_dir, args.registry, args.workspace_dir, args.strict))
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
