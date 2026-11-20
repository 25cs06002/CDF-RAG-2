"""Validate the 132-question source-screened development benchmark (v2).

The v2 dataset includes structured AI-assisted annotations and vetted-domain
bibliographic links, *not* medical gold. The validator checks file integrity,
question identity, task/label schema, source IDs, semantic grouping, and the
absence of false claims of independent clinician review.

Optional --prepare-review produces a worksheet for qualified independent
clinical reviewers; no reviewer decisions are invented or auto-approved.
No network calls, model inference, graph writes, or historical run changes.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/"data/benchmark/v2/source_catalog_screened_20261121.json"
DATASET=ROOT/"data/benchmark/v2/medical_causal_132_source_screened_v2.jsonl"
MANIFEST=ROOT/"data/benchmark/v2/manifest.json"
PILOT=ROOT/"data/benchmark/medical_causal_pilot100_draft_v1.jsonl"
EXT=ROOT/"data/benchmark/medical_causal_expansion32_draft_v1.jsonl"
OUTPUT=ROOT/"presentation_20261118/benchmark_v2_audit"
NEGATIVE={"ambiguous","unsupported"}
VALID_TASKS={
 "direct":"explain_causal_or_risk_evidence",
 "paraphrased":"explain_causal_or_risk_evidence",
 "indirect":"identify_condition_from_description",
 "multi_hop":"assess_multi_link_path",
 "comparative":"compare_multiple_conditions_or_mechanisms",
 "risk_factor":"distinguish_risk_association_from_causation",
 "ambiguous":"recognize_insufficient_individual_evidence",
 "unsupported":"recognize_insufficient_individual_evidence",
}
REQUIRED={
 "schema_version","query_id","query","cohort","query_type","medical_domain",
 "difficulty_legacy","task_objective","semantic_group",
 "original_target_concept","focus_entities","target_role",
 "expected_answerability","expected_response_type","required_relation_semantics",
 "source_reference_ids","evidence_reference_status","response_rubric_candidate",
 "answerability_rationale","legacy_question_family","legacy_concept_family",
 "legacy_source_family","annotation_origin","annotation_review_status",
 "independent_reviewer_id","independent_review_date","independent_review_count",
 "split","gold_answer","gold_entities","gold_causal_edges","gold_paths",
 "gold_evidence_spans","gold_reference_status","review_issue_tags",
}
ALLOWED_DOMAINS={"cdc.gov","nih.gov","niddk.nih.gov","nhlbi.nih.gov",
                 "nimh.nih.gov","ninds.nih.gov","medlineplus.gov"}


def _require(ok,message):
    if not ok:raise ValueError("BENCHMARK_V2_VALIDATION_FAILED: "+message)


def _read_jsonl(path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _git_blob_sha(path):
    payload=path.read_bytes()
    return hashlib.sha1(b"blob "+str(len(payload)).encode()+b"\\0"+payload).hexdigest()


def _trusted(url):
    u=urlparse(url)
    host=(u.hostname or "").lower()
    return u.scheme=="https" and (
        host in ALLOWED_DOMAINS or
        any(host.endswith("."+parent) for parent in ALLOWED_DOMAINS)
    )


def validate(pilot=PILOT,ext=EXT,dataset=DATASET,catalog=SOURCE,manifest=MANIFEST):
    pilot=Path(pilot);ext=Path(ext);dataset=Path(dataset)
    catalog=Path(catalog);manifest=Path(manifest)
    m=json.loads(manifest.read_text(encoding="utf-8"))
    sources=json.loads(catalog.read_text(encoding="utf-8"))["sources"]
    originals=_read_jsonl(pilot)+_read_jsonl(ext)
    items=_read_jsonl(dataset)
    _require(m["schema_version"]=="2.0.0","Unexpected manifest schema")
    _require(_git_blob_sha(pilot)==m["original_input_blob_sha"]["pilot100"],
             "Original 100 queries were modified")
    _require(_git_blob_sha(ext)==m["original_input_blob_sha"]["extension32"],
             "Original 32 queries were modified")
    _require(_git_blob_sha(catalog)==m["source_catalog_blob_sha"],
             "Source catalog was changed without refreshing manifest")
    _require(len(originals)==len(items)==m["count"]==132,"Expected 132 questions")
    _require(len({x["query_id"] for x in originals})==132,"Original IDs overlap")
    _require(len({x["query_id"] for x in items})==132,"Duplicate v2 question ID")
    _require(len({x["query"] for x in items})==132,"Exact duplicate v2 question text")
    _require(len(sources)==m["sources_screened"]==24,"Source catalog count")
    _require(len({s["source_id"] for s in sources})==len(sources),
             "Duplicate bibliographic source ID")
    by_source={s["source_id"]:s for s in sources}
    for s in sources:
        _require(_trusted(s["url"]),"Source not an approved government medical domain: "+s["url"])
        _require(s["independent_clinician_review"] is False,
                 "Bibliographic source improperly marked clinician reviewed")
        _require(s["claim_level_verification"]=="NOT_COMPLETED_FOR_EVERY_BENCHMARK_QUESTION",
                 "Source cannot be represented as claim-adjudicated gold")
        _require(bool(s["scope_note"].strip()),"Missing source scope")
    originals_by_id={x["query_id"]:x for x in originals}
    by_type=Counter()
    by_cohort=Counter()
    by_group=defaultdict(list)
    missing_source_ids=[]
    refs_count=0
    for x in items:
        qid=x["query_id"]
        _require(set(x)==REQUIRED,"Unexpected/missing fields: "+qid+" "+repr(sorted(REQUIRED^set(x))))
        _require(x["schema_version"]=="2.0.0","Wrong schema: "+qid)
        _require(qid in originals_by_id,"Unrecognized question: "+qid)
        original=originals_by_id[qid]
        for field in ("query","query_type","medical_domain"):
            _require(x[field]==original[field],"Original "+field+" was modified: "+qid)
        _require(x["original_target_concept"]==original["target_concept"],
                 "Original target concept modified: "+qid)
        _require(x["cohort"]==("pilot100" if qid.startswith("PILOT_") else "extension32"),
                 "Cohort label mismatch: "+qid)
        _require(x["difficulty_legacy"]==original["difficulty"],"Difficulty changed: "+qid)
        _require(x["task_objective"]==VALID_TASKS.get(x["query_type"]),
                 "Task objective mismatch: "+qid)
        _require(isinstance(x["focus_entities"],list) and
                 (bool(x["focus_entities"]) or x["query_type"] in NEGATIVE),
                 "Question lacks focus entities: "+qid)
        _require(isinstance(x["required_relation_semantics"],list) and
                 bool(x["required_relation_semantics"]), "Missing relation semantics: "+qid)
        _require(isinstance(x["semantic_group"],str) and
                 bool(x["semantic_group"].strip()),"Missing semantic group: "+qid)
        _require(x["annotation_origin"]=="AI_ASSISTED_WEB_SOURCE_SCREENING_2026_11_21",
                 "Unrecognized annotation provenance: "+qid)
        _require(x["annotation_review_status"]==
                 "AI_PREANNOTATED_INDEPENDENT_CLINICIAN_ADJUDICATION_PENDING",
                 "AI-only annotation falsely promoted: "+qid)
        _require(x["independent_review_count"]==0 and
                 x["independent_reviewer_id"] is None and
                 x["independent_review_date"] is None,
                 "Invented independent reviewer: "+qid)
        _require(x["split"]=="development_not_independent_test",
                 "Cannot mark question that influenced G1 as held-out: "+qid)
        for field in ("gold_answer","gold_entities","gold_causal_edges",
                      "gold_paths","gold_evidence_spans"):
            _require(x[field] is None,"Unadjudicated "+field+" improperly populated: "+qid)
        _require(x["gold_reference_status"]=="WITHHELD_UNTIL_INDEPENDENT_CLAIM_ADJUDICATION",
                 "False gold status: "+qid)
        _require(bool(x["response_rubric_candidate"].strip()),
                 "Candidate answer rubric missing: "+qid)
        _require(isinstance(x["review_issue_tags"],list),
                 "Invalid issue tags: "+qid)
        ids=x["source_reference_ids"]
        _require(isinstance(ids,list) and len(ids)==len(set(ids)),
                 "Source IDs invalid/duplicated: "+qid)
        for sid in ids:
            _require(sid in by_source,"Unknown source "+str(sid)+" for "+qid)
        refs_count+=len(ids)
        if not ids:
            missing_source_ids.append(qid)
            _require(x["query_type"] in NEGATIVE,
                     "Nonnegative task lacks topical sources: "+qid)
        _require(x["evidence_reference_status"]==(
            "NO_PATIENT_SPECIFIC_EVIDENCE_EXPECTED"
            if x["query_type"] in NEGATIVE
            else "BACKGROUND_TOPICAL_SOURCES_NOT_CLAIM_LEVEL_GOLD"
        ),"Evidence certification mismatch: "+qid)
        if x["query_type"] in NEGATIVE:
            _require(x["expected_answerability"]==
                     "cannot_determine_individual_specific_answer",
                     "Negative needs abstention/uncertainty: "+qid)
            _require(x["expected_response_type"]=="explicit_uncertainty_or_abstention",
                     "Negative response type mismatch: "+qid)
            _require(x["target_role"]=="mentioned_or_hypothesized_not_confirmed",
                     "Negative label would imply clinical diagnosis: "+qid)
        elif x["query_type"]=="multi_hop":
            _require(x["expected_answerability"]==
                     "conditionally_answerable_after_link_by_link_evidence_review",
                     "Multi-hop gold links not adjudicated: "+qid)
        else:
            _require(x["expected_answerability"]==
                     "general_education_answerable_with_cited_support",
                     "Unreviewed answerability mismatch: "+qid)
        if qid=="PILOT_050":
            _require("legacy_target_role_mismatch" in x["review_issue_tags"] and
                     "kidney complications" in x["focus_entities"],
                     "PILOT_050 downstream target-role correction missing")
        by_type[x["query_type"]]+=1
        by_cohort[x["cohort"]]+=1
        by_group[x["semantic_group"]].append(qid)
    _require(dict(by_type)==m["by_query_type"],"Question-type histogram differs")
    _require(dict(by_cohort)==m["cohorts"],"Question-cohort counts differ")
    across={
        group:sorted(qids)
        for group,qids in by_group.items()
        if any(q.startswith("PILOT_") for q in qids)
        and any(q.startswith("EXTQ_") for q in qids)
    }
    _require(m["clinical_gold_question_count"]==0 and
             m["independent_clinician_reviewed_question_count"]==0,
             "False expert-review/gold claim in manifest")
    result={
        "status":"PASS_SOURCE_SCREENED_METADATA_INTEGRITY",
        "questions":132,"cohorts":dict(by_cohort),
        "by_type":dict(by_type),
        "source_catalog_entries":len(sources),
        "question_source_refs":refs_count,
        "questions_with_source_refs":132-len(missing_source_ids),
        "negative_questions_with_no_topic_specific_sources":missing_source_ids,
        "cross_cohort_semantic_groups":across,
        "clinical_gold_questions":0,
        "independent_clinician_adjudicated_questions":0,
        "g1_independent_holdout":False,
        "medical_accuracy_calculation_permitted":False,
        "warnings":[
            "This validates structure and screened source identifiers, NOT medicine.",
            "All response rubrics are AI-prepared candidates; every link in a multi-hop answer needs individual source adjudication.",
            "All 132 questions were used in G1 development/history, so they belong to the development cohort.",
            "Two independent qualified reviewers and an external adjudicator are recommended before gold release.",
            "The screening catalog supplies topic-level bibliographic support, not independently approved claim-level citations.",
        ],
    }
    return result,items,by_source


def write_review_queue(items,by_source,output_root=OUTPUT):
    out=Path(output_root)/(
        "screening_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")+
        "_"+uuid.uuid4().hex[:6])
    out.mkdir(parents=True,exist_ok=False)
    fields=["query_id","query_type","medical_domain","query","focus_entities",
            "expected_response_type","source_ids","source_urls",
            "response_rubric_candidate","reviewer_id","reviewer_qualification",
            "reviewer_independence_declaration","source_claims_checked",
            "entity_roles_approved","relation_directions_approved",
            "response_rubric_approved","decision","review_notes","review_date"]
    with (out/"independent_adjudication_worksheet.csv").open(
            "w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=fields)
        writer.writeheader()
        for row in items:
            writer.writerow({
                "query_id":row["query_id"],"query_type":row["query_type"],
                "medical_domain":row["medical_domain"],"query":row["query"],
                "focus_entities":json.dumps(row["focus_entities"],ensure_ascii=False),
                "expected_response_type":row["expected_response_type"],
                "source_ids":";".join(row["source_reference_ids"]),
                "source_urls":";".join(by_source[x]["url"] for x in row["source_reference_ids"]),
                "response_rubric_candidate":row["response_rubric_candidate"],
                **{col:"" for col in fields[9:]}
            })
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepare-review",action="store_true",
                   help="Create unfilled worksheet for real independent qualified reviewers")
    p.add_argument("--output-root",type=Path,default=OUTPUT)
    args=p.parse_args()
    result,rows,sources=validate()
    if args.prepare_review:
        dest=write_review_queue(rows,sources,args.output_root)
        result["review_worksheet"]=str(dest/"independent_adjudication_worksheet.csv")
    print(json.dumps(result,indent=2,ensure_ascii=False))


if __name__=="__main__":
    main()
