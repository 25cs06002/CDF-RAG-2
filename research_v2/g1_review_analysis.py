"""Summarize completed G1 change-case evidence judgments (not clinical accuracy).

Consumes original, human-reviewed review_cases.csv and review_evidence.csv.
Never re-runs models, never uses draft gold/target labels, and never changes
review worksheets or G1. Returned evidence occurrences are NOT unique facts;
this selected changed-case sample is NOT an independent held-out benchmark.
"""
from __future__ import annotations

import argparse
import csv
import json
import uuid
from collections import Counter, defaultdict
from datetime import datetime, date, timezone
from pathlib import Path

from .experiment_runner import PROJECT_ROOT, file_sha256
from .g1_relevance_handoff import (
    CASE_FIELDS, EVIDENCE_FIELDS, INTENT, TARGET, RELEVANCE, DIRECTION, CHECKED,
    load, write,
)

ROOT=PROJECT_ROOT
OUT=ROOT/"presentation_20261118/g1_review_analysis"
STAGES=("G0_LEGACY","G1_LEGACY","G1_STABLE","G1_TYPED")
CHANGES={"gained","lost","both","neither"}
CASE_OUTPUT=(
    "case_id","question_set","query_id","family","question","phase2m_seed",
    "original_G1_change","g0_to_stable_change","stable_selection_reason",
    "review_query_intent","review_stable_target",
    "g0_relevant_correct","stable_relevant_correct",
    "stable_relevant_correct_checked_G1_candidate","flags",
    *tuple(f"{stage}_{metric}" for stage in STAGES for metric in (
        "evidence_rows","relevant_rows","relevant_correct_rows",
        "relevant_correct_checked_G1_candidate_rows")),
)
UNIQUE_OUTPUT=(
    "query_id","subject","relation","object","evidence_provenance",
    "original_supporting_source_ids","g1_supporting_source_ids","source_urls",
    "reviewer_paraphrase","occurrences","families","stages",
    "review_evidence_relevance","review_relation_direction",
    "review_source_checked","reviewer","review_date","review_note",
)


def _true(value):
    if value=="true" or value=="True":return True
    if value=="false" or value=="False":return False
    raise ValueError("Expected boolean text, got "+repr(value))


def _date(value):
    try:return bool(value) and date.fromisoformat(value)<=datetime.now(timezone.utc).date()
    except ValueError:return False


def _norm(value):
    return " ".join(value.strip().casefold().split())


def _unique_key(row):
    # Same query + exact edge assertion + same reviewed source provenance;
    # stage and model occurrence are deliberately not part of identity.
    provenance=("G0_unverified" if row["evidence_provenance"]=="inherited_G0_unverified"
                else "G1_reviewed_candidate")
    return (
        row["query_id"],_norm(row["subject"]),row["relation"],
        _norm(row["object"]),provenance,
        row["original_supporting_source_ids"],
        row["g1_supporting_source_ids"],row["source_urls"],
        row["reviewer_paraphrase"],
    )


def _review_columns_valid(row,case):
    if case:
        return (row["review_query_intent"] in INTENT
                and row["review_stable_target"] in TARGET
                and all(row[f].strip() for f in ("reviewer","review_date","review_note"))
                and _date(row["review_date"]))
    return (
        row["review_evidence_relevance"] in RELEVANCE
        and row["review_relation_direction"] in DIRECTION
        and row["review_source_checked"] in CHECKED
        and all(row[f].strip() for f in ("reviewer","review_date","review_note"))
        and _date(row["review_date"])
        and not (row["review_source_checked"]=="checked"
                 and not row["source_urls"].strip())
    )


def _has_candidate_check(row):
    return (
        row["evidence_provenance"]!="inherited_G0_unverified"
        and row["source_urls"].strip()
        and row["review_source_checked"]=="checked"
        and row["review_evidence_relevance"]=="relevant"
        and row["review_relation_direction"]=="correct"
    )


def _stage_stats(rows):
    return {
        "evidence_rows":len(rows),
        "relevant_rows":sum(r["review_evidence_relevance"]=="relevant" for r in rows),
        "irrelevant_rows":sum(r["review_evidence_relevance"]=="irrelevant" for r in rows),
        "uncertain_relevance_rows":sum(r["review_evidence_relevance"]=="uncertain" for r in rows),
        "direction_incorrect_rows":sum(r["review_relation_direction"]=="incorrect" for r in rows),
        "source_checked_rows":sum(r["review_source_checked"]=="checked" for r in rows),
        "relevant_correct_rows":sum(
            r["review_evidence_relevance"]=="relevant" and
            r["review_relation_direction"]=="correct" for r in rows
        ),
        "relevant_correct_checked_G1_candidate_rows":sum(
            _has_candidate_check(r) for r in rows
        ),
    }


def _summarize_cases(cases,per_case_stage,filters):
    chosen=[c for c in cases if filters(c)]
    out={"cases":len(chosen)}
    for stage in STAGES:
        pools=[per_case_stage[(c["case_id"],stage)] for c in chosen]
        out[stage]={
            "cases_with_any_evidence":sum(m["evidence_rows"]>0 for m in pools),
            "cases_with_any_relevant_evidence":sum(m["relevant_rows"]>0 for m in pools),
            "cases_with_relevant_direction_correct_evidence":sum(
                m["relevant_correct_rows"]>0 for m in pools
            ),
            "cases_with_relevant_direction_correct_checked_G1_candidate":sum(
                m["relevant_correct_checked_G1_candidate_rows"]>0 for m in pools
            ),
            "evidence_occurrences":sum(m["evidence_rows"] for m in pools),
            "relevant_occurrences":sum(m["relevant_rows"] for m in pools),
            "incorrect_direction_occurrences":sum(m["direction_incorrect_rows"] for m in pools),
            "source_checked_occurrences":sum(m["source_checked_rows"] for m in pools),
        }
    out["intent_labels"]=dict(Counter(c["review_query_intent"] for c in chosen))
    out["stable_target_labels"]=dict(Counter(c["review_stable_target"] for c in chosen))
    out["transition_labels"]=dict(Counter(c["g0_to_stable_change"] for c in chosen))
    return out


def analyze(folder:Path,output_root:Path=OUT):
    folder=Path(folder).expanduser().resolve()
    output_root=Path(output_root).expanduser().resolve()
    for path in (folder,output_root):
        if not path.is_relative_to(ROOT):
            raise ValueError("Review inputs and outputs must remain inside the repository")
    manifest=json.loads((folder/"manifest.json").read_text(encoding="utf-8"))
    if manifest.get("mode")!="manual_relevance_and_target_review_not_medical_gold":
        raise ValueError("Not a G1 manual relevance workspace")
    cases=load(folder/"review_cases.csv")
    evidence=load(folder/"review_evidence.csv")
    if (len(cases)!=manifest["review_case_count"] or
        len(evidence)!=manifest["evidence_review_rows"]):
        raise ValueError("Handoff review counts differ from frozen manifest")
    if not cases:raise ValueError("No review cases found")
    if not set(CASE_FIELDS).issubset(cases[0]) or not set(EVIDENCE_FIELDS).issubset(
        evidence[0] if evidence else {}):
        raise ValueError("Review worksheet columns are missing")
    by_id={}
    for c in cases:
        if c["case_id"] in by_id:raise ValueError("Duplicate reviewed case ID")
        if not _review_columns_valid(c,True):
            raise ValueError("Missing/invalid case review: "+c["case_id"])
        if c["family"] not in {"B5","B6"}:
            raise ValueError("Unexpected model family: "+c["case_id"])
        if c["g0_to_stable_change"] not in CHANGES or c["original_G1_change"] not in CHANGES:
            raise ValueError("Unexpected transition label: "+c["case_id"])
        by_id[c["case_id"]]=c
    stage_rows=defaultdict(list)
    seen_occurrences=set()
    unique_groups=defaultdict(list)
    for r in evidence:
        case_id=r["case_id"]
        if case_id not in by_id:raise ValueError("Evidence for unknown case "+case_id)
        if r["stage"] not in STAGES:raise ValueError("Unrecognized review stage")
        if not _review_columns_valid(r,False):
            raise ValueError("Missing/invalid evidence review: "+
                             case_id+" "+r["stage"]+" "+r["document_id"])
        c=by_id[case_id]
        if r["family"]!=c["family"] or r["query_id"]!=c["query_id"]:
            raise ValueError("Evidence family/query mismatch: "+case_id)
        if ((r["stage"]=="G0_LEGACY" or r["evidence_provenance"]=="inherited_G0_unverified")
            and r["stage"]=="G0_LEGACY" and
            r["evidence_provenance"]!="inherited_G0_unverified"):
            raise ValueError("G0 legacy contains unexpected reviewed evidence")
        key=(case_id,r["stage"],r["document_id"])
        if key in seen_occurrences:raise ValueError("Duplicate evidence occurrence: "+str(key))
        seen_occurrences.add(key)
        stage_rows[(case_id,r["stage"])].append(r)
        unique_groups[_unique_key(r)].append(r)
    per_stage={
        (c["case_id"],stage):_stage_stats(stage_rows[(c["case_id"],stage)])
        for c in cases for stage in STAGES
    }
    for c in cases:
        has={
            "G0_LEGACY":_true(c["g0_evidence"]),
            "G1_LEGACY":_true(c["g1_legacy_evidence"]),
            "G1_STABLE":_true(c["g1_stable_evidence"]),
            "G1_TYPED":_true(c["g1_typed_evidence"]),
        }
        for stage,present in has.items():
            if present!=(per_stage[(c["case_id"],stage)]["evidence_rows"]>0):
                raise ValueError("Evidence row presence contradicts reviewed case: "+
                                 c["case_id"]+" "+stage)
        original=("gained" if has["G1_LEGACY"] and not has["G0_LEGACY"] else
                  "lost" if has["G0_LEGACY"] and not has["G1_LEGACY"] else
                  "both" if has["G0_LEGACY"] else "neither")
        stable=("gained" if has["G1_STABLE"] and not has["G0_LEGACY"] else
                "lost" if has["G0_LEGACY"] and not has["G1_STABLE"] else
                "both" if has["G0_LEGACY"] else "neither")
        if original!=c["original_G1_change"] or stable!=c["g0_to_stable_change"]:
            raise ValueError("Transition mismatch: "+c["case_id"])
    case_out=[]
    attention=[]
    for c in cases:
        look=lambda stage:per_stage[(c["case_id"],stage)]
        base=look("G0_LEGACY")
        stable=look("G1_STABLE")
        flags=[]
        if c["review_stable_target"]=="wrong":
            flags.append("wrong_stable_target")
        if c["review_stable_target"]=="uncertain":
            flags.append("stable_target_uncertain")
        if c["review_query_intent"]=="unsupported" and stable["evidence_rows"]>0:
            flags.append("unsupported_question_returns_evidence")
        if c["review_query_intent"]=="ambiguous" and stable["evidence_rows"]>0:
            flags.append("ambiguous_question_returns_evidence")
        if c["g0_to_stable_change"]=="gained":
            if not stable["relevant_correct_rows"]:
                flags.append("gain_without_relevant_direction_correct_evidence")
            if not stable["relevant_correct_checked_G1_candidate_rows"]:
                flags.append("gain_without_source_checked_relevant_G1_candidate")
        if c["g0_to_stable_change"]=="lost" and base["relevant_correct_rows"]:
            flags.append("lost_previously_relevant_G0_unverified_edge")
        if any(r["review_relation_direction"]=="incorrect" for r in
               stage_rows[(c["case_id"],"G1_STABLE")]):
            flags.append("stable_incorrect_edge_direction")
        row={
            "case_id":c["case_id"],"question_set":c["question_set"],
            "query_id":c["query_id"],"family":c["family"],"question":c["question"],
            "phase2m_seed":c["phase2m_graph_construction_seed"],
            "original_G1_change":c["original_G1_change"],
            "g0_to_stable_change":c["g0_to_stable_change"],
            "stable_selection_reason":c["stable_selection_reason"],
            "review_query_intent":c["review_query_intent"],
            "review_stable_target":c["review_stable_target"],
            "g0_relevant_correct":base["relevant_correct_rows"]>0,
            "stable_relevant_correct":stable["relevant_correct_rows"]>0,
            "stable_relevant_correct_checked_G1_candidate":
                stable["relevant_correct_checked_G1_candidate_rows"]>0,
            "flags":"|".join(flags),
        }
        for stage in STAGES:
            m=look(stage)
            for field in ("evidence_rows","relevant_rows","relevant_correct_rows",
                          "relevant_correct_checked_G1_candidate_rows"):
                row[f"{stage}_{field}"]=m[field]
        case_out.append(row)
        if flags:attention.append(row)
    deduplicated=[]
    for key,group in unique_groups.items():
        labels=("review_evidence_relevance","review_relation_direction",
                "review_source_checked")
        if len({tuple(r[k] for k in labels) for r in group})!=1:
            raise ValueError("Conflicting judgments for repeated evidence: "+repr(key))
        first=group[0]
        deduplicated.append({
            "query_id":first["query_id"],"subject":first["subject"],
            "relation":first["relation"],"object":first["object"],
            "evidence_provenance":(
                "G0_unverified" if first["evidence_provenance"]=="inherited_G0_unverified"
                else "G1_reviewed_candidate"),
            "original_supporting_source_ids":first["original_supporting_source_ids"],
            "g1_supporting_source_ids":first["g1_supporting_source_ids"],
            "source_urls":first["source_urls"],
            "reviewer_paraphrase":first["reviewer_paraphrase"],
            "occurrences":len(group),
            "families":"|".join(sorted({r["family"] for r in group})),
            "stages":"|".join(sorted({r["stage"] for r in group})),
            **{field:first[field] for field in (
                "review_evidence_relevance","review_relation_direction",
                "review_source_checked","reviewer","review_date","review_note")},
        })
    summary={
        "analysis_scope":"manually_selected_changed_G1_model_query_cases_only",
        "review_folder":str(folder.relative_to(ROOT)),
        "source_case_sha256":file_sha256(folder/"review_cases.csv"),
        "source_evidence_sha256":file_sha256(folder/"review_evidence.csv"),
        "reviewed_cases":len(cases),
        "distinct_questions":len({c["query_id"] for c in cases}),
        "reviewed_evidence_occurrences":len(evidence),
        "distinct_question_evidence_assertions":len(deduplicated),
        "by_family":{
            f:_summarize_cases(cases,per_stage,lambda c,f=f:c["family"]==f)
            for f in ("B5","B6")
        },
        "all_cases":_summarize_cases(cases,per_stage,lambda c:True),
        "case_flags":dict(Counter(
            fl for row in attention for fl in row["flags"].split("|") if fl)),
        "attention_case_count":len(attention),
        "independent_holdout":False,
        "clinical_accuracy":False,
        "source_check_is_not_medical_claim_validation":True,
        "warnings":[
            "Metrics are conditional on deliberately selected behavior-change cases, not all 132 questions.",
            "The same evidence can appear in multiple stages and model families: occurrences are not unique claims.",
            "Source checked means reviewer attestation; it is not independent clinical validation.",
            "Legacy G0 statements remain unverified regardless of relevance labels.",
            "Do not report case/occurrence fractions as general accuracy, precision, recall or hallucination rate.",
            "Construction-seed questions and draft benchmark are not a clean independent holdout.",
        ],
    }
    dest=output_root/("analysis_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")+
                      "_"+uuid.uuid4().hex[:6])
    dest.mkdir(parents=True,exist_ok=False)
    write(dest/"per_case_reviewed.csv",CASE_OUTPUT,case_out)
    write(dest/"attention_queue.csv",CASE_OUTPUT,attention)
    write(dest/"unique_evidence_annotations.csv",UNIQUE_OUTPUT,deduplicated)
    (dest/"summary.json").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"output":str(dest),"reviewed_cases":len(cases),
        "reviewed_evidence_occurrences":len(evidence),
        "distinct_question_evidence_assertions":len(deduplicated),
        "attention_case_count":len(attention),
        "case_flags":summary["case_flags"],
        "by_family":summary["by_family"]},indent=2))
    return dest,summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--folder",type=Path,required=True)
    p.add_argument("--output-root",type=Path,default=OUT)
    a=p.parse_args()
    if Path.cwd().resolve()!=ROOT:p.error("Run from repository root")
    analyze(a.folder,a.output_root)


if __name__=="__main__":
    main()
