"""Compact duplicate manual G1 relevance tasks, then propagate only explicit annotations.

Read-only 'prepare' groups identical question/target and question/evidence
judgments across B5/B6, stages and duplicate returned edges. It DOES NOT
infer relevance, medical correctness or gold labels.

'apply' checks the original review CSV hashes, validates manual decisions,
backs up the originals and copies only completed labels to matching rows.
No graph changes, no large experiments or automatic approvals.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import uuid
from collections import defaultdict
from datetime import date, datetime, timezone
from pathlib import Path

from .experiment_runner import PROJECT_ROOT, file_sha256

ROOT=PROJECT_ROOT
CASE_FILE="review_cases.csv"
EVIDENCE_FILE="review_evidence.csv"
CASE_REVIEW=("review_query_intent","review_stable_target","reviewer","review_date","review_note")
EVIDENCE_REVIEW=("review_evidence_relevance","review_relation_direction",
                 "review_source_checked","reviewer","review_date","review_note")
CASE_VALUES={"review_query_intent":{"answerable","ambiguous","unsupported","unknown"},
             "review_stable_target":{"appropriate","wrong","uncertain","none_expected"}}
EVIDENCE_VALUES={"review_evidence_relevance":{"relevant","irrelevant","uncertain"},
                 "review_relation_direction":{"correct","incorrect","uncertain"},
                 "review_source_checked":{"checked","not_checked","not_applicable"}}
CASE_TASK_COLS=("task_id","query_id","question","stable_target",
                "case_ids","families","selection_reasons","occurrences",*CASE_REVIEW)
EVIDENCE_TASK_COLS=("task_id","query_id","question",
                    "subject","relation","object","evidence_provenance_group",
                    "original_source_ids","g1_source_ids","source_urls",
                    "reviewer_paraphrase","case_ids","families","stages",
                    "document_ids","occurrences",*EVIDENCE_REVIEW)


def _read(path):
    with path.open(newline="",encoding="utf-8") as f:
        reader=csv.DictReader(f)
        cols=reader.fieldnames
        rows=list(reader)
    if not cols:raise ValueError("CSV lacks header: "+str(path))
    if any(None in r or any(v is None for v in r.values()) for r in rows):
        raise ValueError("Malformed CSV: "+str(path))
    return cols,rows


def _write(path,cols,rows):
    with path.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def _norm(value):
    return " ".join((value or "").casefold().split())


def _key_hash(prefix,key):
    token=json.dumps(key,ensure_ascii=False,separators=(",",":"))
    return prefix+"_"+hashlib.sha256(token.encode("utf-8")).hexdigest()[:18]


def _provenance(row):
    v=row["evidence_provenance"]
    if v in {"G1_new_candidate_CAUSES","G1_approved_typed_candidate"}:
        return "G1_reviewed_candidate"
    if v=="inherited_G0_unverified":
        return "G0_unverified"
    raise ValueError("Unrecognized evidence provenance: "+v)


def _case_key(row):
    # Different target choices need different judgments; identical choices
    # can be reused across B5 and B6 for the same query.
    return (row["query_id"],row["question"],_norm(row["g1_stable_target"]))


def _evidence_key(row):
    # Do not collapse risk factors into causation. Source/provenance must match.
    return (row["query_id"],_norm(row["subject"]),row["relation"],
            _norm(row["object"]),_provenance(row),
            row["original_supporting_source_ids"],
            row["g1_supporting_source_ids"],row["source_urls"],
            row["reviewer_paraphrase"])


def _group_cases(cases):
    groups=defaultdict(list)
    for idx,row in enumerate(cases):
        groups[_case_key(row)].append((idx,row))
    tasks=[]
    mapping={}
    for key,items in sorted(groups.items(),key=lambda pair:pair[0]):
        task_id=_key_hash("C",key)
        if task_id in mapping:raise ValueError("Task hash collision")
        mapping[task_id]=[i for i,_ in items]
        first=items[0][1]
        tasks.append({
            "task_id":task_id,"query_id":first["query_id"],
            "question":first["question"],
            "stable_target":first["g1_stable_target"],
            "case_ids":"|".join(dict.fromkeys(r["case_id"] for _,r in items)),
            "families":"|".join(dict.fromkeys(r["family"] for _,r in items)),
            "selection_reasons":"|".join(dict.fromkeys(
                r["stable_selection_reason"] for _,r in items)),
            "occurrences":str(len(items)),
            **{field:"" for field in CASE_REVIEW},
        })
    return tasks,mapping


def _group_evidence(evidence,by_case):
    groups=defaultdict(list)
    for idx,row in enumerate(evidence):
        if row["case_id"] not in by_case:
            raise ValueError("Evidence links to missing case: "+row["case_id"])
        groups[_evidence_key(row)].append((idx,row))
    tasks=[]
    mapping={}
    for key,items in sorted(groups.items(),key=lambda pair:tuple(str(x) for x in pair[0])):
        task_id=_key_hash("E",key)
        if task_id in mapping:raise ValueError("Task hash collision")
        mapping[task_id]=[i for i,_ in items]
        first=items[0][1]
        tasks.append({
            "task_id":task_id,
            "query_id":first["query_id"],
            "question":by_case[first["case_id"]]["question"],
            "subject":first["subject"],"relation":first["relation"],
            "object":first["object"],
            "evidence_provenance_group":_provenance(first),
            "original_source_ids":first["original_supporting_source_ids"],
            "g1_source_ids":first["g1_supporting_source_ids"],
            "source_urls":first["source_urls"],
            "reviewer_paraphrase":first["reviewer_paraphrase"],
            "case_ids":"|".join(dict.fromkeys(r["case_id"] for _,r in items)),
            "families":"|".join(dict.fromkeys(r["family"] for _,r in items)),
            "stages":"|".join(dict.fromkeys(r["stage"] for _,r in items)),
            "document_ids":"|".join(dict.fromkeys(r["document_id"] for _,r in items)),
            "occurrences":str(len(items)),
            **{field:"" for field in EVIDENCE_REVIEW},
        })
    return tasks,mapping


def _source_rows(folder):
    case_cols,cases=_read(folder/CASE_FILE)
    ev_cols,evidence=_read(folder/EVIDENCE_FILE)
    if len({r["case_id"] for r in cases})!=len(cases):
        raise ValueError("Duplicate case ID")
    by_case={r["case_id"]:r for r in cases}
    if any(not all(k in r for k in CASE_REVIEW) for r in cases):
        raise ValueError("Missing case review columns")
    if any(not all(k in r for k in EVIDENCE_REVIEW) for r in evidence):
        raise ValueError("Missing evidence review columns")
    return case_cols,cases,ev_cols,evidence,by_case


def prepare(folder, output=None):
    folder=Path(folder).resolve()
    if not folder.is_relative_to(ROOT):raise ValueError("Review folder outside repo")
    case_cols,cases,ev_cols,evidence,by_case=_source_rows(folder)
    if any(any(r[f] for f in CASE_REVIEW) for r in cases) or any(
        any(r[f] for f in EVIDENCE_REVIEW) for r in evidence):
        raise ValueError("Original rows already annotated; compact only before annotation")
    case_tasks,case_map=_group_cases(cases)
    ev_tasks,ev_map=_group_evidence(evidence,by_case)
    dest=(Path(output).resolve() if output else
          folder/("compact_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")+
                  "_"+uuid.uuid4().hex[:6]))
    if not dest.is_relative_to(ROOT):raise ValueError("Output folder outside repo")
    dest.mkdir(parents=True,exist_ok=False)
    _write(dest/"case_tasks.csv",CASE_TASK_COLS,case_tasks)
    _write(dest/"evidence_tasks.csv",EVIDENCE_TASK_COLS,ev_tasks)
    meta={
        "purpose":"manual_deduplicated_annotation_no_medical_gold",
        "original_folder":str(folder.relative_to(ROOT)),
        "original_case_sha256":file_sha256(folder/CASE_FILE),
        "original_evidence_sha256":file_sha256(folder/EVIDENCE_FILE),
        "case_rows":len(cases),"case_tasks":len(case_tasks),
        "evidence_rows":len(evidence),"evidence_tasks":len(ev_tasks),
        "no_auto_approvals":True,
        "note":"Review every unique task; identical source-supported edge assertions across stages can reuse the same manual judgment. Inherited G0 remains unverified.",
    }
    (dest/"manifest.json").write_text(json.dumps(meta,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"output":str(dest),**meta},indent=2))
    return dest


def _date_ok(value):
    try:
        return bool(value) and date.fromisoformat(value)<=datetime.now(timezone.utc).date()
    except (ValueError,TypeError):return False


def _validate_review(row,fields,choices,kind):
    vals=[(row.get(f) or "").strip() for f in fields]
    if not any(vals):return False
    if not all(vals):raise ValueError("Partial "+kind+" annotation: "+row["task_id"])
    for field,options in choices.items():
        if row[field] not in options:
            raise ValueError("Invalid "+field+" for "+row["task_id"])
    if not _date_ok(row["review_date"]):
        raise ValueError("Invalid reviewer date for "+row["task_id"])
    if len(row["review_note"].strip())<10:
        raise ValueError("Reviewer note must explain decision for "+row["task_id"])
    if kind=="evidence" and row["review_source_checked"]=="checked" and not row["source_urls"]:
        raise ValueError("Cannot claim URL checked without a source URL: "+row["task_id"])
    return True


def apply(compact_dir, require_complete=False):
    compact_dir=Path(compact_dir).resolve()
    if not compact_dir.is_relative_to(ROOT):raise ValueError("Compact folder outside repo")
    meta=json.loads((compact_dir/"manifest.json").read_text(encoding="utf-8"))
    folder=ROOT/meta["original_folder"]
    if (file_sha256(folder/CASE_FILE)!=meta["original_case_sha256"] or
        file_sha256(folder/EVIDENCE_FILE)!=meta["original_evidence_sha256"]):
        raise ValueError("Original worksheets changed; do not overwrite newer annotations")
    case_cols,cases,ev_cols,evidence,by_case=_source_rows(folder)
    case_tasks,case_map=_group_cases(cases)
    ev_tasks,ev_map=_group_evidence(evidence,by_case)
    cols1,labeled_cases=_read(compact_dir/"case_tasks.csv")
    cols2,labeled_evidence=_read(compact_dir/"evidence_tasks.csv")
    if set(CASE_TASK_COLS)!=set(cols1) or set(EVIDENCE_TASK_COLS)!=set(cols2):
        raise ValueError("Task CSV headers changed")
    expected1={r["task_id"]:r for r in case_tasks}
    expected2={r["task_id"]:r for r in ev_tasks}
    if (len(labeled_cases)!=len(expected1) or len(labeled_evidence)!=len(expected2) or
        {r["task_id"] for r in labeled_cases}!=set(expected1) or
        {r["task_id"] for r in labeled_evidence}!=set(expected2)):
        raise ValueError("Tasks added/removed; regenerate or restore")
    reviewed_cases=reviewed_evidence=0
    for rows,expected,fields in (
        (labeled_cases,expected1,CASE_REVIEW),
        (labeled_evidence,expected2,EVIDENCE_REVIEW)):
        for row in rows:
            # Identity/provenance fields are immutable; reviewers may only
            # change the explicit review columns.
            for field in expected[row["task_id"]]:
                if field not in fields and row[field]!=expected[row["task_id"]][field]:
                    raise ValueError("Task identity changed: "+row["task_id"]+" field "+field)
    for row in labeled_cases:
        if _validate_review(row,CASE_REVIEW,CASE_VALUES,"case"):
            reviewed_cases+=1
            for i in case_map[row["task_id"]]:
                for field in CASE_REVIEW:cases[i][field]=row[field]
    for row in labeled_evidence:
        if _validate_review(row,EVIDENCE_REVIEW,EVIDENCE_VALUES,"evidence"):
            reviewed_evidence+=1
            for i in ev_map[row["task_id"]]:
                for field in EVIDENCE_REVIEW:evidence[i][field]=row[field]
    if require_complete and (reviewed_cases!=len(labeled_cases) or
                              reviewed_evidence!=len(labeled_evidence)):
        raise ValueError("Compact annotations incomplete; originals unchanged")
    stamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    for file in (CASE_FILE,EVIDENCE_FILE):
        shutil.copy2(folder/file, folder/(file+".pre_compact_apply_"+stamp+".bak"))
    _write(folder/CASE_FILE,case_cols,cases)
    _write(folder/EVIDENCE_FILE,ev_cols,evidence)
    report={
        "case_tasks_reviewed":reviewed_cases,
        "case_tasks_total":len(labeled_cases),
        "evidence_tasks_reviewed":reviewed_evidence,
        "evidence_tasks_total":len(labeled_evidence),
        "reviewed_case_rows_propagated":sum(all(c[field] for field in CASE_REVIEW) for c in cases),
        "reviewed_evidence_rows_propagated":sum(all(e[field] for field in EVIDENCE_REVIEW) for e in evidence),
        "gold_or_accuracy_created":False,
        "original_csv_backups_created":True,
    }
    (compact_dir/"apply_report.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(report,indent=2))
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest="command",required=True)
    prep=sub.add_parser("prepare")
    prep.add_argument("--folder",type=Path,required=True)
    prep.add_argument("--output",type=Path)
    ap=sub.add_parser("apply")
    ap.add_argument("--compact-dir",type=Path,required=True)
    ap.add_argument("--require-complete",action="store_true")
    a=p.parse_args()
    if Path.cwd().resolve()!=ROOT:p.error("Run from repo root")
    if a.command=="prepare":prepare(a.folder,a.output)
    else:apply(a.compact_dir,a.require_complete)


if __name__=="__main__":
    main()
