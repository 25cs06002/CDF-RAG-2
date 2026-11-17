"""Read-only Phase 2M independent review audit; no automatic G1 promotion.

Checks 52 source tasks and 30 link tasks with cross-references. Does NOT
claim that a reviewer genuinely read sources or that claims are clinically true.
Never modifies worksheets, manifests, frozen evidence, G0 or benchmark labels.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from datetime import date, datetime, timezone
from pathlib import Path

from .experiment_runner import PROJECT_ROOT, file_sha256

HANDOFF = PROJECT_ROOT / "research_artifacts/experiment-artifacts/phase2/phase2m_full_adjudication_handoff_v1"
SRC_FILE = "all52_source_review_tasks.csv"
LINK_FILE = "all30_link_adjudication.csv"
SOURCE_DECISIONS = {"supports", "partial", "does_not_support", "needs_more_evidence"}
RELATIONS = {"CAUSES", "RISK_FACTOR_FOR", "ASSOCIATED_WITH", "PROTECTS_AGAINST",
             "TRIGGERS_EXACERBATION", "UNCERTAIN"}
SOURCE_REVIEW_COLS = ("independent_reviewer_id","independent_review_date",
                      "independent_source_decision","independent_relation",
                      "independent_verified_passage","independent_review_note")
LINK_REVIEW_COLS = ("independent_reviewer_id","independent_review_date",
                    "independent_relation","independent_link_decision",
                    "independent_supporting_source_ids","independent_evidence_notes")


def _rows(path, required):
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if not set(required).issubset(reader.fieldnames or ()):
            raise ValueError("Missing expected columns in " + str(path))
        rows=list(reader)
    if any(None in row for row in rows):
        raise ValueError("Malformed CSV rows in " + str(path))
    return rows


def _ident(row, field):
    return row.get(field,"").strip()


def _date_check(value):
    try:
        d = date.fromisoformat(value)
        return d <= datetime.now(timezone.utc).date()
    except ValueError:
        return False


def analyze(directory=HANDOFF):
    root=Path(directory).expanduser().resolve()
    if not root.is_relative_to(PROJECT_ROOT):
        raise ValueError("Review folder must remain inside repository")
    sources=_rows(root/SRC_FILE, (*SOURCE_REVIEW_COLS,"source_task_id","link_task_id","source_id","source_url"))
    links=_rows(root/LINK_FILE, (*LINK_REVIEW_COLS,"link_task_id","cause_node","effect_node","adjudication_status"))
    if (len(sources), len(links)) != (52,30):
        raise ValueError("Expected exactly 52 source tasks and 30 link tasks")
    if len({s["source_task_id"] for s in sources})!=52 or len({l["link_task_id"] for l in links})!=30:
        raise ValueError("Duplicate task IDs")
    source_index=defaultdict(list)
    for s in sources:
        source_index[(s["link_task_id"],s["source_id"])].append(s)
    findings=[]
    source_decisions=Counter()
    link_decisions=Counter()
    complete_sources=0
    complete_links=0
    for s in sources:
        sid=s["source_task_id"]
        entries=[_ident(s,c) for c in SOURCE_REVIEW_COLS]
        empty=not any(entries)
        missing=[c for c in SOURCE_REVIEW_COLS if not _ident(s,c)]
        decision=_ident(s,"independent_source_decision")
        valid=True
        if empty:
            findings.append({"kind":"source","task_id":sid,"severity":"PENDING",
                             "reason":"Independent review not recorded"})
            continue
        for col in ("independent_reviewer_id","independent_review_date",
                    "independent_source_decision","independent_review_note"):
            if not _ident(s,col):
                findings.append({"kind":"source","task_id":sid,"severity":"ERROR",
                                 "reason":"Missing "+col});valid=False
        if decision not in SOURCE_DECISIONS:
            findings.append({"kind":"source","task_id":sid,"severity":"ERROR",
                             "reason":"Unsupported source decision: "+decision});valid=False
        if _ident(s,"independent_review_date") and not _date_check(_ident(s,"independent_review_date")):
            findings.append({"kind":"source","task_id":sid,"severity":"ERROR",
                             "reason":"Invalid/future review date"});valid=False
        if decision=="supports":
            if not _ident(s,"independent_relation") or not _ident(s,"independent_verified_passage"):
                findings.append({"kind":"source","task_id":sid,"severity":"ERROR",
                                 "reason":"Supports requires verified relation and passage locator"});valid=False
        relation=_ident(s,"independent_relation")
        if relation and relation not in RELATIONS:
            findings.append({"kind":"source","task_id":sid,"severity":"ERROR",
                             "reason":"Unrecognized relation: "+relation});valid=False
        if valid:
            complete_sources+=1;source_decisions[decision]+=1
    for l in links:
        lid=l["link_task_id"]
        entries=[_ident(l,c) for c in LINK_REVIEW_COLS]
        status=_ident(l,"adjudication_status")
        if not any(entries) and status in ("", "pending_independent_review"):
            findings.append({"kind":"link","task_id":lid,"severity":"PENDING",
                             "reason":"Independent link adjudication not recorded"})
            continue
        valid=True
        for col in ("independent_reviewer_id","independent_review_date",
                    "independent_link_decision","independent_evidence_notes"):
            if not _ident(l,col):
                findings.append({"kind":"link","task_id":lid,"severity":"ERROR",
                                 "reason":"Missing "+col});valid=False
        decision=_ident(l,"independent_link_decision")
        if decision not in SOURCE_DECISIONS:
            findings.append({"kind":"link","task_id":lid,"severity":"ERROR",
                             "reason":"Unsupported link decision: "+decision});valid=False
        if _ident(l,"independent_review_date") and not _date_check(_ident(l,"independent_review_date")):
            findings.append({"kind":"link","task_id":lid,"severity":"ERROR",
                             "reason":"Invalid/future review date"});valid=False
        if decision=="supports":
            if not _ident(l,"independent_relation"):
                findings.append({"kind":"link","task_id":lid,"severity":"ERROR",
                                 "reason":"Supports requires typed relation"});valid=False
            if not _ident(l,"independent_supporting_source_ids"):
                findings.append({"kind":"link","task_id":lid,"severity":"ERROR",
                                 "reason":"Supports requires supporting source IDs"});valid=False
        relation=_ident(l,"independent_relation")
        if relation and relation not in RELATIONS:
            findings.append({"kind":"link","task_id":lid,"severity":"ERROR",
                             "reason":"Unrecognized link relation: "+relation});valid=False
        ids=[x.strip() for x in re.split(r"[;,|]",_ident(l,"independent_supporting_source_ids")) if x.strip()]
        if len(ids)!=len(set(ids)):
            findings.append({"kind":"link","task_id":lid,"severity":"ERROR",
                             "reason":"Duplicate supporting source IDs"});valid=False
        for sid in ids:
            candidates=source_index.get((lid,sid),[])
            if not candidates:
                findings.append({"kind":"link","task_id":lid,"severity":"ERROR",
                                 "reason":"Supporting source ID missing from this link: "+sid});valid=False
            elif decision=="supports" and not any(
                _ident(src,"independent_source_decision")=="supports" for src in candidates):
                findings.append({"kind":"link","task_id":lid,"severity":"ERROR",
                                 "reason":"Cited source not independently marked supports: "+sid});valid=False
            elif decision=="supports" and not any(
                _ident(src,"independent_relation")==relation for src in candidates):
                findings.append({"kind":"link","task_id":lid,"severity":"WARNING",
                                 "reason":"Link/source relationship types differ: "+sid})
        if status=="pending_independent_review":
            findings.append({"kind":"link","task_id":lid,"severity":"WARNING",
                             "reason":"adjudication_status still pending despite review fields"})
        if valid:
            complete_links+=1;link_decisions[decision]+=1
    summary={
        "status":"review_complete_structural" if complete_sources==52 and complete_links==30 and not any(
            item["severity"] in {"ERROR","PENDING"} for item in findings) else "review_incomplete_or_requires_reconciliation",
        "sources_total":52,"links_total":30,
        "structurally_complete_sources":complete_sources,
        "structurally_complete_links":complete_links,
        "source_decisions":dict(source_decisions),
        "link_decisions":dict(link_decisions),
        "issues_by_severity":dict(Counter(x["severity"] for x in findings)),
        "source_sha256":file_sha256(root/SRC_FILE),
        "links_sha256":file_sha256(root/LINK_FILE),
        "medical_gold_verified":False,
        "reviewer_identity_verified":False,
        "source_passages_verified_automatically":False,
        "approved_edges_auto_promoted":False,
        "warnings":["Structure and cross-references do not independently verify medical claims.",
                    "A partial source does not automatically support the full causal link.",
                    "Relationship types must be preserved on eventual G1 import."]
    }
    return summary,findings


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory",type=Path,default=HANDOFF)
    parser.add_argument("--out",type=Path)
    args=parser.parse_args()
    summary, findings=analyze(args.directory)
    out=Path(args.out or args.directory/"independent_audit_report.json").expanduser().resolve()
    if not out.is_relative_to(PROJECT_ROOT):parser.error("Output must stay within repository")
    if out.exists():parser.error("Refuse to overwrite an existing audit report")
    out.write_text(json.dumps({"summary":summary,"issues":findings},indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"report":str(out),**summary},indent=2))


if __name__=="__main__":
    main()
