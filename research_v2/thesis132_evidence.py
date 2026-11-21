"""Versioned, independently reviewable G1 evidence ledger for 132-query MTP.

prepare: copy immutable candidate identities to an all-pending review ledger.
audit: check that review records, claimed source spans and approvals are complete.
freeze: emit a frozen verified/uncertain/excluded corpus only after declared
        external review and a separate supervisor sign-off. This software
        CANNOT authenticate reviewers, licenses, or the authenticity of sources.
It never modifies G0, G1, B1-B8 results or original benchmark questions.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import date
from pathlib import Path
from urllib.parse import urlparse

from .experiment_runner import PROJECT_ROOT as ROOT, file_sha256
from .g1_typed_retrieval import G1TypedIndex

BASE = ROOT / "presentation_20261118/thesis132/evidence_review"
FIELDS = (
    "edge_id", "subject", "relation", "object", "source_id",
    "candidate_digest", "decision", "source_url", "source_snapshot_path",
    "exact_source_passage", "passage_locator", "relation_direction_checked",
    "reviewer_1", "reviewer_2", "independence_declared", "adjudicator",
    "review_date", "review_notes",
)
DECISIONS = {"pending", "verified", "uncertain", "excluded"}


def _inside(path: Path) -> Path:
    result = Path(path).expanduser().resolve()
    if not result.is_relative_to(ROOT):
        raise ValueError("Evidence review path must remain in the working checkout")
    return result


def _write_csv(path: Path, rows: list[dict], fields=FIELDS) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        data = list(csv.DictReader(f))
    if not data or set(data[0]) != set(FIELDS):
        raise ValueError("Evidence ledger missing rows or required columns")
    return data


def _identity(row: dict) -> dict:
    return {key: row[key] for key in ("edge_id", "subject", "relation", "object", "source_id")}


def _digest(row: dict) -> str:
    canonical = json.dumps(_identity(row), sort_keys=True, ensure_ascii=False,
                           separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def prepare(graph_dir: Path, output_root: Path = BASE) -> Path:
    graph_dir = _inside(graph_dir)
    index = G1TypedIndex(graph_dir)   # checks frozen typed-edge and alias SHA
    output_root = _inside(output_root)
    manifest_sha = file_sha256(graph_dir / "manifest.json")
    out = output_root / ("G1_EVIDENCE_REVIEW_" + manifest_sha[:16])
    if out.exists():
        raise FileExistsError("Review workspace exists; never overwrite: " + str(out))
    out.mkdir(parents=True, exist_ok=False)
    rows = []
    ids = set()
    for edge in sorted(index.edges, key=lambda x: x["edge_id"]):
        if edge["edge_id"] in ids:
            raise ValueError("Duplicate candidate edge identifier")
        ids.add(edge["edge_id"])
        row = {k: edge.get(k, "") for k in FIELDS}
        row["candidate_digest"] = _digest(row)
        row["decision"] = "pending"
        for key in FIELDS[7:]:
            row[key] = ""
        rows.append(row)
    if not rows:
        raise ValueError("G1 has no candidate typed links; no ledger to review")
    _write_csv(out / "edge_adjudication.csv", rows)
    manifest = {
        "schema": "THESIS132_EVIDENCE_CANDIDATE_REVIEW_V1",
        "graph_manifest_sha256": manifest_sha,
        "typed_relationships_sha256": file_sha256(graph_dir / "typed_relationships.csv"),
        "candidate_count": len(rows),
        "candidate_identity_sha256": hashlib.sha256(
            "\n".join(r["candidate_digest"] for r in rows).encode()).hexdigest(),
        "current_status": "PENDING_INDEPENDENT_REVIEW",
        "independent_medical_gold": False,
        "warning": "All G1 approvals are candidates. Prepare never creates medical gold.",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"workspace": str(out), "candidates": len(rows),
                      "verified": 0, "status": "REQUIRES_EXTERNAL_REVIEW"}, indent=2))
    return out


def _date(value: str) -> bool:
    try:
        date.fromisoformat(value)
        return True
    except ValueError:
        return False


def _span_valid(workspace: Path, row: dict) -> bool:
    rel = row["source_snapshot_path"].strip()
    if not rel:
        return False
    source = (workspace / rel).resolve()
    if not source.is_relative_to(workspace.resolve()) or not source.is_file():
        return False
    try:
        text = source.read_text(encoding="utf-8")
        return bool(row["exact_source_passage"].strip()) and row["exact_source_passage"] in text
    except (UnicodeError, OSError):
        return False


def audit(workspace: Path, graph_dir: Path) -> tuple[list[dict], dict]:
    workspace = _inside(workspace)
    graph_dir = _inside(graph_dir)
    index = G1TypedIndex(graph_dir)
    manifest = json.loads((workspace / "manifest.json").read_text(encoding="utf-8"))
    if manifest["schema"] != "THESIS132_EVIDENCE_CANDIDATE_REVIEW_V1":
        raise ValueError("Unrecognized evidence review manifest")
    if (file_sha256(graph_dir / "manifest.json") != manifest["graph_manifest_sha256"]
        or file_sha256(graph_dir / "typed_relationships.csv") != manifest["typed_relationships_sha256"]):
        raise ValueError("G1 candidate graph changed after workspace preparation")
    rows = _rows(workspace / "edge_adjudication.csv")
    original = {e["edge_id"]: e for e in index.edges}
    if len(rows) != manifest["candidate_count"] or {x["edge_id"] for x in rows} != set(original):
        raise ValueError("Candidate edge IDs or row count changed")
    failures = {}
    decisions = Counter()
    signatures = []
    for r in rows:
        eid = r["edge_id"]
        errors = []
        if _identity(r) != _identity(original[eid]) or r["candidate_digest"] != _digest(r):
            errors.append("candidate_identity_modified")
        signatures.append(r["candidate_digest"])
        decision = r["decision"].strip().lower()
        decisions[decision] += 1
        if decision not in DECISIONS:
            errors.append("invalid_decision")
        if decision == "verified":
            url = urlparse(r["source_url"])
            if url.scheme != "https" or not url.netloc:
                errors.append("missing_https_source_url")
            if not r["passage_locator"].strip() or not _span_valid(workspace, r):
                errors.append("exact_source_passage_not_found_in_local_snapshot")
            if r["relation_direction_checked"].strip().lower() != "yes":
                errors.append("relation_direction_not_checked")
        if decision in {"verified", "uncertain", "excluded"}:
            reviewers = [r["reviewer_1"].strip(), r["reviewer_2"].strip()]
            if not all(reviewers) or reviewers[0].casefold() == reviewers[1].casefold():
                errors.append("two_distinct_reviewers_required")
            if r["independence_declared"].strip().lower() != "yes":
                errors.append("independence_declaration_missing")
            if not r["adjudicator"].strip() or r["adjudicator"].strip().casefold() in {
                x.casefold() for x in reviewers if x
            }:
                errors.append("independent_adjudicator_required")
            if not _date(r["review_date"].strip()):
                errors.append("invalid_review_date")
            if not r["review_notes"].strip():
                errors.append("review_rationale_missing")
        if errors:
            failures[eid] = errors
    digest = hashlib.sha256("\n".join(signatures).encode()).hexdigest()
    if digest != manifest["candidate_identity_sha256"]:
        raise ValueError("Frozen candidate identity set/order modified")
    result = {
        "status": ("READY_FOR_EXTERNAL_SIGNOFF" if not failures and not decisions["pending"]
                   else "INCOMPLETE_REVIEW"),
        "candidate_count": len(rows), "decisions": dict(decisions),
        "failed_edges": failures, "clinical_gold_certified_by_code": False,
        "warning": "Offline matching of passages to reviewer-provided snapshots cannot authenticate publication, clinical correctness or reviewer credentials.",
    }
    return rows, result


def freeze(workspace: Path, graph_dir: Path, signoff: Path) -> Path:
    workspace = _inside(workspace)
    signoff = _inside(signoff)
    rows, result = audit(workspace, graph_dir)
    if result["status"] != "READY_FOR_EXTERNAL_SIGNOFF":
        raise ValueError("Cannot freeze an incomplete evidence review: " + json.dumps(result))
    approval = json.loads(signoff.read_text(encoding="utf-8"))
    required = {"supervisor_name", "approval_date", "reviewer_credentials_checked",
                "source_authenticity_checked", "causal_directions_checked",
                "independent_external_adjudication_completed"}
    if not required.issubset(approval):
        raise ValueError("Sign-off missing required fields")
    if not approval["supervisor_name"] or not _date(approval["approval_date"]) or not all(
        approval[k] is True for k in required - {"supervisor_name", "approval_date"}
    ):
        raise ValueError("Sign-off declarations must be genuine and affirmative")
    digest = file_sha256(workspace / "edge_adjudication.csv")
    out = workspace / ("FROZEN_EVIDENCE_" + digest[:16])
    if out.exists():
        raise FileExistsError("Refusing to overwrite a frozen corpus")
    out.mkdir()
    cols = ("edge_id", "subject", "relation", "object", "source_id",
            "source_url", "source_snapshot_path", "exact_source_passage",
            "passage_locator", "decision")
    for decision in ("verified", "uncertain", "excluded"):
        filtered = [{k: r[k] for k in cols} for r in rows if r["decision"] == decision]
        _write_csv(out / (decision + "_relationships.csv"), filtered, cols)
    release = {
        "status": "EXTERNAL_SIGNOFF_DECLARED_REQUIRES_HUMAN_AUTHENTICATION",
        "input_ledger_sha256": digest,
        "candidate_graph_manifest_sha256": file_sha256(_inside(graph_dir) / "manifest.json"),
        "supervisor_signoff_sha256": file_sha256(signoff),
        "counts": {k: sum(r["decision"] == k for r in rows)
                   for k in ("verified", "uncertain", "excluded")},
        "warnings": [
            "Only externally reviewed and signed claims may be treated as verified.",
            "This program cannot establish clinical gold or authenticate source provenance by itself.",
            "Do not confuse this graph-edge release with independently reviewed gold answers for the 132 questions.",
        ],
    }
    (out / "manifest.json").write_text(json.dumps(release, indent=2) + "\n", encoding="utf-8")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    for action in ("prepare", "audit", "freeze"):
        q = sub.add_parser(action)
        q.add_argument("--graph-dir", type=Path, required=True)
        if action == "prepare":
            q.add_argument("--output-root", type=Path, default=BASE)
        else:
            q.add_argument("--workspace", type=Path, required=True)
            if action == "freeze":
                q.add_argument("--signoff", type=Path, required=True)
    args = p.parse_args()
    if args.command == "prepare":
        prepare(args.graph_dir, args.output_root)
    elif args.command == "audit":
        _, result = audit(args.workspace, args.graph_dir)
        print(json.dumps(result, indent=2))
        if result["status"] != "READY_FOR_EXTERNAL_SIGNOFF":
            raise SystemExit(2)
    else:
        out = freeze(args.workspace, args.graph_dir, args.signoff)
        print(json.dumps({"frozen_evidence": str(out)}))


if __name__ == "__main__":
    main()
