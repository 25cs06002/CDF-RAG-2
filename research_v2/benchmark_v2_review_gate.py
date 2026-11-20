"""Check two-reviewer clinical annotation worksheet; never auto-certify gold.

This verifies recorded reviewer *completeness* and agreement declarations
only. An external qualified supervisor must authenticate reviewer identity,
credentials, source passages, judgments and resolve disputes before any
gold labels or clinical-accuracy measurements can be released.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter,defaultdict
from datetime import datetime
from pathlib import Path

from .benchmark_v2_validation import validate


def _text(row,key):
    return str(row.get(key,"")).strip()


def _non_ai_reviewer(name):
    value=name.strip().lower()
    return bool(value) and not any(tag in value for tag in (
        "ai_", "gpt", "llm", "assistant", "automated", "synthetic", "bot_"
    ))


def _parse_list(value):
    try:
        data=json.loads(value)
    except (ValueError,TypeError):
        return None
    return data if isinstance(data,list) else None


def analyze(review_file:Path):
    benchmark,questions,_=validate()
    expected={r["query_id"]:r for r in questions}
    with Path(review_file).open(newline="",encoding="utf-8") as f:
        rows=list(csv.DictReader(f))
    by_id=defaultdict(dict)
    problems=defaultdict(list)
    for row in rows:
        qid=_text(row,"query_id")
        slot=_text(row,"review_slot")
        if qid not in expected:
            problems[qid].append("unknown_question")
            continue
        if slot not in {"1","2"}:
            problems[qid].append("unknown_review_slot")
            continue
        if slot in by_id[qid]:
            problems[qid].append("duplicate_review_slot")
            continue
        by_id[qid][slot]=row
        if _text(row,"query")!=expected[qid]["query"]:
            problems[qid].append("question_text_changed")
        if _text(row,"source_ids")!=";".join(expected[qid]["source_reference_ids"]):
            problems[qid].append("source_ids_changed")
        if not _non_ai_reviewer(_text(row,"reviewer_id")):
            problems[qid].append("missing_or_automated_reviewer_id")
        if not _text(row,"reviewer_qualification"):
            problems[qid].append("missing_reviewer_qualification")
        if _text(row,"reviewer_independence_declaration").lower() not in {
                "yes","independent","i independently checked the sources"}:
            problems[qid].append("missing_independence_declaration")
        if _text(row,"source_claims_checked").lower()!="yes":
            problems[qid].append("claims_not_checked")
        if _text(row,"entity_roles_approved").lower() not in {"yes","not_applicable"}:
            problems[qid].append("entity_roles_not_approved")
        if _text(row,"relation_directions_approved").lower() not in {
                "yes","not_applicable"}:
            problems[qid].append("relation_directions_not_approved")
        if _text(row,"response_rubric_approved").lower()!="yes":
            problems[qid].append("response_rubric_not_approved")
        if _text(row,"decision").lower()!="approve":
            problems[qid].append("review_not_approved")
        if not _text(row,"verified_answer"):
            problems[qid].append("verified_answer_missing")
        if _parse_list(_text(row,"verified_edges_json")) is None:
            problems[qid].append("verified_edges_not_json_list")
        if _parse_list(_text(row,"verified_paths_json")) is None:
            problems[qid].append("verified_paths_not_json_list")
        if not _text(row,"verified_source_spans"):
            problems[qid].append("claim_evidence_spans_missing")
        try:
            datetime.fromisoformat(_text(row,"review_date"))
        except ValueError:
            problems[qid].append("invalid_review_date")
    for qid in expected:
        these=by_id[qid]
        if set(these)!={"1","2"}:
            problems[qid].append("two_independent_review_slots_required")
            continue
        if (_text(these["1"],"reviewer_id").lower()==
            _text(these["2"],"reviewer_id").lower()):
            problems[qid].append("reviewers_not_distinct")
    counters=Counter(reason for values in problems.values() for reason in values)
    formally_complete=[qid for qid in expected if not problems[qid]]
    return {
        "status":("FORMALLY_COMPLETE_REQUIRES_EXTERNAL_CREDENTIAL_AND_CLAIM_ADJUDICATION"
                  if len(formally_complete)==132 else "INCOMPLETE_EXTERNAL_REVIEW"),
        "questions_total":132,
        "expected_review_slots":264,
        "submitted_review_rows":len(rows),
        "questions_with_formally_complete_double_review":len(formally_complete),
        "questions_pending":132-len(formally_complete),
        "failure_reasons":dict(counters),
        "pending_question_ids":[qid for qid in expected if problems[qid]],
        "clinical_gold_certified":False,
        "independent_expertise_authentication_automatic":False,
        "remaining_manual_requirements":[
            "A qualified independent supervisor verifies reviewers are distinct real clinicians with suitable expertise.",
            "Every asserted edge/path is checked against the recorded source passage and causal direction.",
            "Disagreements or medically uncertain labels are adjudicated and documented.",
            "Approval and benchmark version are signed and frozen externally before claims of expert gold.",
        ],
    }


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--review-file",required=True,type=Path)
    p.add_argument("--report-json",type=Path,
                   help="Optional new diagnostic output file; does not modify worksheet")
    args=p.parse_args()
    report=analyze(args.review_file)
    if args.report_json:
        if args.report_json.exists():
            p.error("Refusing to overwrite an existing review diagnostic")
        args.report_json.parent.mkdir(parents=True,exist_ok=True)
        args.report_json.write_text(
            json.dumps(report,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print(json.dumps(report,indent=2,ensure_ascii=False))


if __name__=="__main__":
    main()
