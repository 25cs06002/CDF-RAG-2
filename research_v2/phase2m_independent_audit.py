"""Audit completed Phase 2M 52-source/30-link independent reviewer handoff.

Checks structural completeness, source-to-link consistency, relation typing
and provenance. DOES NOT validate medical truth or certify expertise.
Original input files are not modified. Produces review-only G1 proposals.
No Ollama, Neo4j, PPO, or offline sprint execution.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import uuid
from collections import Counter, defaultdict
from datetime import date, datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

from .experiment_runner import PROJECT_ROOT, file_sha256

BASE = PROJECT_ROOT / "research_artifacts/experiment-artifacts/phase2/phase2m_full_adjudication_handoff_v1"
SOURCE = "all52_source_review_tasks.csv"
LINK = "all30_link_adjudication.csv"
SUPPORTED = {"supports", "supported", "approved"}
UNSUPPORTED = {"partial", "partially_supports", "does_not_support", "not_supported",
               "needs_more_evidence", "rejected", "unavailable", "inaccessible", "uncertain"}
RELATIONS = {"CAUSES", "RISK_FACTOR_FOR", "ASSOCIATED_WITH", "TRIGGERS_EXACERBATION",
             "PROTECTS_AGAINST", "IS_A", "UNCERTAIN"}
SOURCE_FIELDS = ("query_id","source_task_id","link_task_id","source_id","source_url",
                 "independent_reviewer_id","independent_review_date",
                 "independent_source_decision","independent_relation",
                 "independent_verified_passage","independent_review_note")
LINK_FIELDS = ("query_id","link_task_id","cause_node","effect_node","original_relation",
               "independent_reviewer_id","independent_review_date",
               "independent_relation","independent_link_decision",
               "independent_supporting_source_ids","independent_evidence_notes",
               "adjudication_status")


def read_rows(path, columns):
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not set(columns).issubset(set(reader.fieldnames or [])):
            raise ValueError(f"{path.name} missing required columns")
        return list(reader)


def norm(value):
    return (value or "").strip().casefold()


def ids(value):
    return [v.strip() for v in re.split(r"[;|]", value or "") if v.strip()]


def iso_date(value):
    if not value:
        return False
    try:
        v = date.fromisoformat(value.strip())
        return v <= datetime.now(timezone.utc).date()
    except ValueError:
        return False


def analyze(sources, links):
    errors=[]
    warnings=[]
    if len(sources)!=52: errors.append(f"Expected 52 source rows, got {len(sources)}")
    if len(links)!=30: errors.append(f"Expected 30 link rows, got {len(links)}")
    source_index=defaultdict(list)
    source_ids=set()
    source_tasks=set()
    link_ids=set()
    source_decisions=Counter()
    link_decisions=Counter()
    proposals=[]
    for i,row in enumerate(sources,2):
        prefix=f"source_row_{i}"
        sid=row["source_id"].strip()
        task=row["source_task_id"].strip()
        key=(row["query_id"].strip(), row["link_task_id"].strip())
        if not sid or not task or not all(key): errors.append(prefix+": missing source/link IDs")
        if task in source_tasks:errors.append(prefix+": duplicate source_task_id")
        source_tasks.add(task)
        source_ids.add(sid)
        source_index[(key,sid)].append(row)
        u=urlparse(row["source_url"].strip())
        if u.scheme!="https" or not u.netloc:
            errors.append(prefix+": source_url must use HTTPS")
        decision=norm(row["independent_source_decision"])
        source_decisions[decision or "pending"]+=1
        if not decision:
            errors.append(prefix+": independent source decision incomplete")
            continue
        if decision not in SUPPORTED|UNSUPPORTED:
            warnings.append(prefix+": nonstandard source decision "+decision)
        for field in ("independent_reviewer_id","independent_verified_passage",
                      "independent_review_note"):
            if not row[field].strip():
                errors.append(prefix+": missing "+field)
        if not iso_date(row["independent_review_date"]):
            errors.append(prefix+": invalid independent_review_date")
        rel=row["independent_relation"].strip().upper()
        if decision in SUPPORTED and rel not in RELATIONS:
            errors.append(prefix+": supported source requires typed independent_relation")
        elif rel and rel not in RELATIONS:
            errors.append(prefix+": unrecognized independent_relation "+rel)
    for i,row in enumerate(links,2):
        prefix=f"link_row_{i}"
        qid=row["query_id"].strip()
        lid=row["link_task_id"].strip()
        if lid in link_ids:errors.append(prefix+": duplicate link_task_id")
        link_ids.add(lid)
        if not qid or not lid or not row["cause_node"].strip() or not row["effect_node"].strip():
            errors.append(prefix+": incomplete link identity")
        decision=norm(row["independent_link_decision"])
        link_decisions[decision or "pending"]+=1
        if not decision:
            errors.append(prefix+": independent link decision incomplete")
            continue
        for field in ("independent_reviewer_id","independent_evidence_notes"):
            if not row[field].strip():
                errors.append(prefix+": missing "+field)
        if not iso_date(row["independent_review_date"]):
            errors.append(prefix+": invalid independent_review_date")
        relation=row["independent_relation"].strip().upper()
        if relation and relation not in RELATIONS:
            errors.append(prefix+": unknown independent_relation "+relation)
        support=ids(row["independent_supporting_source_ids"])
        if len(support)!=len(set(support)):
            errors.append(prefix+": duplicate supporting source IDs")
        source_entries=[]
        for sid in support:
            entries=source_index.get(((qid,lid),sid),[])
            if not entries:
                errors.append(prefix+": source ID not registered for this link "+sid)
                continue
            source_entries.extend(entries)
        if decision in SUPPORTED:
            if relation not in RELATIONS:
                errors.append(prefix+": supported link requires typed relation")
            if not support:
                errors.append(prefix+": supported link has no independent source IDs")
            for entry in source_entries:
                if norm(entry["independent_source_decision"]) not in SUPPORTED:
                    errors.append(prefix+": cited source not independently supporting: "+entry["source_id"])
                if entry["independent_relation"].strip().upper()!=relation:
                    errors.append(prefix+": link/source relation type mismatch: "+entry["source_id"])
            if support and len(source_entries)==len(support) and relation in RELATIONS:
                # Proposal only; never auto-import approved edges.
                proposals.append({
                    "query_id":qid, "link_task_id":lid,
                    "subject":row["cause_node"].strip(), "relation":relation,
                    "object":row["effect_node"].strip(),
                    "source_ids":"|".join(support),
                    "reviewer":row["independent_reviewer_id"].strip(),
                    "review_date":row["independent_review_date"].strip(),
                    "proposed_import_status":"requires_g1_concept_source_review",
                })
        elif support:
            warnings.append(prefix+": unsupported/partial link cites independent sources; verify scope")
        if not row["adjudication_status"].strip():
            warnings.append(prefix+": adjudication_status blank")
        elif "pending" in norm(row["adjudication_status"]) and decision:
            warnings.append(prefix+": review completed but adjudication_status still pending")
    return {
        "source_rows":len(sources),"link_rows":len(links),
        "source_decisions":dict(source_decisions),
        "link_decisions":dict(link_decisions),
        "supported_link_proposals":len(proposals),
        "error_count":len(errors),"warning_count":len(warnings),
        "errors":errors,"warnings":warnings,
        "structural_audit_passed":len(errors)==0,
        "medically_verified":False,"independently_certified_gold":False,
        "auto_graph_import_permitted":False,
        "note":"Human reviewer attestations and structural checks do not prove clinical correctness."
    }, proposals


def run(folder=BASE, output=None):
    folder=Path(folder).expanduser().resolve()
    if not folder.is_relative_to(PROJECT_ROOT):
        raise ValueError("Source handoff must remain inside repository")
    sources=read_rows(folder/SOURCE,SOURCE_FIELDS)
    links=read_rows(folder/LINK,LINK_FIELDS)
    report,proposals=analyze(sources,links)
    report["source_sha256"]=file_sha256(folder/SOURCE)
    report["link_sha256"]=file_sha256(folder/LINK)
    report["source_folder"]=folder.relative_to(PROJECT_ROOT).as_posix()
    dest=(Path(output) if output else
          PROJECT_ROOT/"research_artifacts/experiment-artifacts/g1_review_workspaces"/
          ("PHASE2M_AUDIT_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")+
           "_"+uuid.uuid4().hex[:6])).expanduser().resolve()
    if not dest.is_relative_to(PROJECT_ROOT):
        raise ValueError("Output must remain inside repository")
    dest.mkdir(parents=True,exist_ok=False)
    (dest/"review_audit.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    with (dest/"g1_link_import_candidates.csv").open("w",newline="",encoding="utf-8") as handle:
        fields=("query_id","link_task_id","subject","relation","object",
                "source_ids","reviewer","review_date","proposed_import_status")
        writer=csv.DictWriter(handle,fieldnames=fields)
        writer.writeheader();writer.writerows(proposals)
    print(json.dumps({"output":str(dest),"source_rows":report["source_rows"],
        "link_rows":report["link_rows"],"error_count":report["error_count"],
        "warning_count":report["warning_count"],"supported_link_proposals":len(proposals),
        "structural_audit_passed":report["structural_audit_passed"]},indent=2))
    return dest,report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--folder",type=Path,default=BASE)
    p.add_argument("--output",type=Path)
    p.add_argument("--require-complete",action="store_true")
    args=p.parse_args()
    if Path.cwd().resolve()!=PROJECT_ROOT:p.error("Run from repository root")
    _,report=run(args.folder,args.output)
    if args.require_complete and not report["structural_audit_passed"]:
        raise SystemExit(f"Review audit failed: {report['error_count']} structural errors")


if __name__=="__main__":
    main()
