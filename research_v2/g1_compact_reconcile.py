"""Diagnose/reconcile edited G1 compact-review tasks without losing reviews.

'diagnose' is read-only. 'repair' changes ONLY a task_id when all other
immutable fields exactly match a canonical task. With --restore-missing,
missing tasks are recreated BLANK for later manual review. Extra or
conflicting tasks are never dropped, merged, overwritten, or auto-reviewed.

Backups of both task CSVs are created before writing. Original review
worksheets and the G1 graph are never modified by this module.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from .experiment_runner import PROJECT_ROOT, file_sha256
from .g1_review_compact import (
    ROOT, CASE_FILE, EVIDENCE_FILE, CASE_TASK_COLS, EVIDENCE_TASK_COLS,
    CASE_REVIEW, EVIDENCE_REVIEW, _read, _write, _source_rows,
    _group_cases, _group_evidence,
)

CONFIG=(
    ("case_tasks.csv", CASE_TASK_COLS, CASE_REVIEW),
    ("evidence_tasks.csv", EVIDENCE_TASK_COLS, EVIDENCE_REVIEW),
)


def context(compact:Path):
    compact=Path(compact).expanduser().resolve()
    if not compact.is_relative_to(ROOT):
        raise ValueError("Compact folder must be inside repository")
    manifest=json.loads((compact/"manifest.json").read_text(encoding="utf-8"))
    parent=ROOT/manifest["original_folder"]
    if not parent.is_relative_to(ROOT):
        raise ValueError("Invalid original review path in manifest")
    for fname,expected_sha in (
        (CASE_FILE,manifest["original_case_sha256"]),
        (EVIDENCE_FILE,manifest["original_evidence_sha256"]),
    ):
        if file_sha256(parent/fname)!=expected_sha:
            raise ValueError("Original "+fname+" changed after compact prepare; stop to protect reviews")
    _,cases,_,evidence,by_case=_source_rows(parent)
    canonical_cases,_=_group_cases(cases)
    canonical_evidence,_=_group_evidence(evidence,by_case)
    canonical=(canonical_cases,canonical_evidence)
    actual=[]
    for (fname,cols,_),expected in zip(CONFIG,canonical):
        headers,rows=_read(compact/fname)
        if set(headers)!=set(cols) or len(headers)!=len(cols):
            raise ValueError(fname+": header mismatch (do not remove review columns)")
        actual.append(rows)
    return compact,manifest,canonical,actual


def identity(row,cols,review_fields):
    return tuple((row.get(key) or "") for key in cols
                 if key!="task_id" and key not in review_fields)


def inspect_rows(name,cols,review_fields,canonical,edited):
    expected_by_id={row["task_id"]:row for row in canonical}
    identity_to_ids={}
    for row in canonical:
        identity_to_ids.setdefault(identity(row,cols,review_fields),[]).append(row["task_id"])
    actual_ids=[row.get("task_id","") for row in edited]
    count=Counter(actual_ids)
    unknown=[]
    identity_proposals=[]
    conflict=[]
    for index,row in enumerate(edited,1):
        task_id=row.get("task_id","")
        if task_id in expected_by_id:
            original=expected_by_id[task_id]
            differences=[field for field in cols
                         if field not in review_fields and row.get(field,"")!=original.get(field,"")]
            if differences:
                conflict.append({"row":index,"task_id":task_id,"immutable_fields_changed":differences})
            continue
        matches=identity_to_ids.get(identity(row,cols,review_fields),[])
        if len(matches)==1:
            identity_proposals.append({"row":index,"from":task_id,"to":matches[0]})
        else:
            unknown.append({"row":index,"task_id":task_id,
                            "candidate_matches":matches,
                            "query_id":row.get("query_id","")})
    missing=[tid for tid in expected_by_id if tid not in actual_ids]
    report={
        "file":name,
        "expected_rows":len(canonical),
        "actual_rows":len(edited),
        "missing_task_ids":len(missing),
        "unknown_task_ids":len([tid for tid in actual_ids if tid not in expected_by_id]),
        "duplicate_task_ids":{k:v for k,v in count.items() if v>1},
        "exact_identity_id_repair_candidates":len(identity_proposals),
        "immutable_identity_conflicts":len(conflict),
        "unmatched_rows":len(unknown),
        "missing_sample":missing[:12],
        "id_repair_sample":identity_proposals[:12],
        "conflict_sample":conflict[:12],
        "unmatched_sample":unknown[:12],
    }
    return report


def diagnose(compact):
    _,manifest,canonical,actual=context(compact)
    details=[inspect_rows(name,cols,review,expected,rows)
             for (name,cols,review),expected,rows in zip(CONFIG,canonical,actual)]
    report={
        "compact_original_folder":manifest["original_folder"],
        "files":details,
        "note":"No files changed. task_id and immutable task metadata must match canonical tasks. Never discard manual review columns.",
    }
    print(json.dumps(report,indent=2))
    return report


def repair_rows(cols,review_fields,canonical,edited,restore_missing,filename):
    expected={r["task_id"]:r for r in canonical}
    by_identity={}
    for row in canonical:
        key=identity(row,cols,review_fields)
        if key in by_identity:
            raise ValueError(filename+": canonical identity is not unique")
        by_identity[key]=row["task_id"]
    repaired={}
    id_fixes=[]
    for i,row in enumerate(edited,1):
        old_id=row.get("task_id","")
        target_id=old_id
        if old_id not in expected:
            target_id=by_identity.get(identity(row,cols,review_fields))
            if not target_id:
                raise ValueError(
                    filename+f": row {i} has an unknown task_id AND changed/unknown immutable fields; "+
                    "do not delete its annotation. Inspect diagnostic output.")
        canonical_row=expected[target_id]
        changed=[c for c in cols if c!="task_id" and c not in review_fields
                 and row.get(c,"")!=canonical_row.get(c,"")]
        if changed:
            raise ValueError(filename+f": row {i}, task {target_id}, immutable fields changed: "+",".join(changed))
        if target_id in repaired:
            raise ValueError(filename+": duplicate or conflicting row for task "+target_id)
        rebuilt=dict(row)
        rebuilt["task_id"]=target_id
        repaired[target_id]=rebuilt
        if old_id!=target_id:
            id_fixes.append({"row":i,"from":old_id,"to":target_id})
    missing=[tid for tid in expected if tid not in repaired]
    if missing and not restore_missing:
        raise ValueError(
            filename+": "+str(len(missing))+" tasks missing. "
            "Use --restore-missing only if you accept recreating those tasks as BLANK, "
            "without any inferred approval. No files changed.")
    if restore_missing:
        for task_id in missing:
            repaired[task_id]=dict(expected[task_id])
    return [repaired[row["task_id"]] for row in canonical],id_fixes,missing


def repair(compact,restore_missing=False):
    compact,manifest,canonical,actual=context(compact)
    validated=[]
    summary={}
    for (name,cols,review),expected,rows in zip(CONFIG,canonical,actual):
        restored,fixes,missing=repair_rows(
            cols,review,expected,rows,restore_missing,name)
        validated.append((name,cols,restored))
        summary[name]={
            "task_ids_restored_from_unchanged_identity":len(fixes),
            "blank_missing_tasks_recreated":len(missing),
            "remaining_rows":len(restored),
            "id_changes":fixes[:20],
        }

    # Validate BOTH CSVs before touching either. Back up even if only one
    # needs a change. Preserve handwritten labels exactly.
    stamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    backups={}
    for name,_,_ in validated:
        source=compact/name
        backup=compact/(name+".pre_reconcile_"+stamp+".bak")
        shutil.copy2(source,backup)
        backups[name]=str(backup)
    for name,cols,rows in validated:
        temp=compact/(name+".reconcile_tmp_"+uuid.uuid4().hex[:8])
        try:
            _write(temp,cols,rows)
            os.replace(temp,compact/name)
        finally:
            if temp.exists():
                temp.unlink()
    report={
        "compact_dir":str(compact.relative_to(ROOT)),
        "per_file":summary,
        "backups":backups,
        "review_fields_preserved":True,
        "automatic_reviews_or_approvals":False,
        "next":"Review any restored blank tasks, then re-run g1_review_compact apply --require-complete",
    }
    (compact/("reconcile_"+stamp+".json")).write_text(
        json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(report,indent=2))
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="command",required=True)
    d=sub.add_parser("diagnose")
    d.add_argument("--compact-dir",type=Path,required=True)
    r=sub.add_parser("repair")
    r.add_argument("--compact-dir",type=Path,required=True)
    r.add_argument("--restore-missing",action="store_true",
                   help="Recreate missing tasks as blank; manual review still required")
    args=parser.parse_args()
    if Path.cwd().resolve()!=PROJECT_ROOT:
        parser.error("Run from repository root")
    if args.command=="diagnose":diagnose(args.compact_dir)
    else:repair(args.compact_dir,args.restore_missing)


if __name__=="__main__":
    main()
