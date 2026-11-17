"""Prepare a focused, read-only root-cause queue from completed G1 reviews.

Inputs: human-reviewed relevance handoff and the already-generated
g1_review_analysis output. Never uses reviewer labels for inference;
labels are used only for retrospective error analysis and task prioritization.
No generation, graph edits, model reruns, or clinical gold claims.
"""
from __future__ import annotations

import argparse
import csv
import json
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from .experiment_runner import PROJECT_ROOT, file_sha256

ROOT=PROJECT_ROOT
OUTPUT=ROOT/"presentation_20261118/g1_error_triage"
DETAIL_COLUMNS=(
    "priority","issue_flags","case_id","query_id","family",
    "question","stable_target","review_query_intent",
    "review_stable_target","selection_reason",
    "g0_target","g1_legacy_target","original_G1_change",
    "g0_to_stable_change","phase2m_seed",
    "stage","document_id","subject","relation","object",
    "provenance","source_urls","source_checked",
    "evidence_relevance","relation_direction",
    "review_note",
)
CASE_COLUMNS=(
    "priority","case_id","query_id","family","question",
    "issue_flags","review_query_intent","review_stable_target",
    "g0_target","g1_legacy_target","stable_target","selection_reason",
    "original_G1_change","g0_to_stable_change","phase2m_seed",
    "G0_LEGACY_evidence","G1_LEGACY_evidence",
    "G1_STABLE_evidence","G1_TYPED_evidence",
    "g1_stable_checked_relevant_direction_correct","review_note",
)


def _read(path):
    with path.open(newline="",encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write(path,fields,rows):
    with path.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def priority(flags):
    if "wrong_stable_target" in flags:return "P0_WRONG_TARGET"
    if "ambiguous_question_returns_evidence" in flags or "unsupported_question_returns_evidence" in flags:
        return "P0_UNSAFE_RETURN"
    if "gain_without_relevant_direction_correct_evidence" in flags:
        return "P1_UNSUPPORTED_GAIN"
    if "stable_target_uncertain" in flags:return "P2_TARGET_UNCERTAIN"
    if "stable_incorrect_edge_direction" in flags:
        return "P1_INCORRECT_EDGE_DIRECTION"
    return "P3_OTHER"


def run(analysis_dir:Path, output_root:Path=OUTPUT):
    analysis_dir=Path(analysis_dir).expanduser().resolve()
    output_root=Path(output_root).expanduser().resolve()
    if not analysis_dir.is_relative_to(ROOT) or not output_root.is_relative_to(ROOT):
        raise ValueError("Analysis/output must stay inside repository")
    summary=json.loads((analysis_dir/"summary.json").read_text(encoding="utf-8"))
    if summary.get("analysis_scope")!="manually_selected_changed_G1_model_query_cases_only":
        raise ValueError("Not the reviewed G1 changed-case analysis")
    source=ROOT/summary["review_folder"]
    if not source.is_relative_to(ROOT):
        raise ValueError("Unsafe reviewed source folder")
    for name,key in (("review_cases.csv","source_case_sha256"),
                     ("review_evidence.csv","source_evidence_sha256")):
        if file_sha256(source/name)!=summary[key]:
            raise ValueError("Reviewed source changed after analysis: "+name)
    original_cases={r["case_id"]:r for r in _read(source/"review_cases.csv")}
    original_evidence=defaultdict(list)
    for e in _read(source/"review_evidence.csv"):
        original_evidence[e["case_id"]].append(e)
    annotated=_read(analysis_dir/"attention_queue.csv")
    expected=summary["attention_case_count"]
    if len(annotated)!=expected or len({r["case_id"] for r in annotated})!=len(annotated):
        raise ValueError("Unexpected attention queue or duplicate cases")
    annotated.sort(key=lambda r:(priority(set(r["flags"].split("|"))),
                                r["query_id"],r["family"]))
    cases=[]
    details=[]
    flags_count=Counter()
    reason_count=Counter()
    reviewed_question_ids=set()
    for finding in annotated:
        cid=finding["case_id"]
        c=original_cases.get(cid)
        if c is None:raise ValueError("Unknown reviewed case "+cid)
        issues=set(finding["flags"].split("|"))-{" "}
        p=priority(issues)
        if not issues or not any(issues):
            raise ValueError("Attention case without an issue")
        for name in issues:flags_count[name]+=1
        reason_count[c["stable_selection_reason"]]+=1
        reviewed_question_ids.add(c["query_id"])
        evidence_by_stage=Counter(e["stage"] for e in original_evidence[cid])
        if any(x not in {"G0_LEGACY","G1_LEGACY","G1_STABLE","G1_TYPED"}
               for x in evidence_by_stage):
            raise ValueError("Unknown evidence stage")
        cases.append({
            "priority":p,"case_id":cid,"query_id":c["query_id"],
            "family":c["family"],"question":c["question"],
            "issue_flags":finding["flags"],"review_query_intent":c["review_query_intent"],
            "review_stable_target":c["review_stable_target"],
            "g0_target":c["g0_target"],"g1_legacy_target":c["g1_legacy_target"],
            "stable_target":c["g1_stable_target"],
            "selection_reason":c["stable_selection_reason"],
            "original_G1_change":c["original_G1_change"],
            "g0_to_stable_change":c["g0_to_stable_change"],
            "phase2m_seed":c["phase2m_graph_construction_seed"],
            "G0_LEGACY_evidence":evidence_by_stage["G0_LEGACY"],
            "G1_LEGACY_evidence":evidence_by_stage["G1_LEGACY"],
            "G1_STABLE_evidence":evidence_by_stage["G1_STABLE"],
            "G1_TYPED_evidence":evidence_by_stage["G1_TYPED"],
            "g1_stable_checked_relevant_direction_correct":
                finding["stable_relevant_correct_checked_G1_candidate"],
            "review_note":c["review_note"],
        })
        for e in original_evidence[cid]:
            details.append({
                "priority":p,"issue_flags":finding["flags"],
                "case_id":cid,"query_id":c["query_id"],
                "family":c["family"],"question":c["question"],
                "stable_target":c["g1_stable_target"],
                "review_query_intent":c["review_query_intent"],
                "review_stable_target":c["review_stable_target"],
                "selection_reason":c["stable_selection_reason"],
                "g0_target":c["g0_target"],
                "g1_legacy_target":c["g1_legacy_target"],
                "original_G1_change":c["original_G1_change"],
                "g0_to_stable_change":c["g0_to_stable_change"],
                "phase2m_seed":c["phase2m_graph_construction_seed"],
                "stage":e["stage"],"document_id":e["document_id"],
                "subject":e["subject"],"relation":e["relation"],
                "object":e["object"],
                "provenance":e["evidence_provenance"],
                "source_urls":e["source_urls"],
                "source_checked":e["review_source_checked"],
                "evidence_relevance":e["review_evidence_relevance"],
                "relation_direction":e["review_relation_direction"],
                "review_note":e["review_note"],
            })
    out=output_root/("triage_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")+
                     "_"+uuid.uuid4().hex[:6])
    out.mkdir(parents=True,exist_ok=False)
    _write(out/"priority_cases.csv",CASE_COLUMNS,cases)
    _write(out/"reviewed_edge_details.csv",DETAIL_COLUMNS,details)
    result={
        "mode":"reviewed_changed_case_root_cause_triage_only",
        "source_analysis":str(analysis_dir.relative_to(ROOT)),
        "attention_cases":len(cases),
        "distinct_attention_queries":len(reviewed_question_ids),
        "evidence_occurrences_on_attention_cases":len(details),
        "case_priority_counts":dict(Counter(c["priority"] for c in cases)),
        "case_flag_counts":dict(flags_count),
        "selection_reason_counts":dict(reason_count),
        "medical_accuracy":False,
        "independent_holdout":False,
        "reviewer_labels_used_for_inference":False,
        "warnings":[
            "A case can have multiple overlapping flags; counts are NOT additive.",
            "B5/B6 model-query cases can repeat the same question.",
            "These are selected change cases, not a representative benchmark sample.",
            "Do not automatically change graph edge direction, aliases, reviewer labels or approval status.",
            "Frozen G0 relationship relevance does not imply independently verified causation.",
        ],
    }
    (out/"summary.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"output":str(out),**result},indent=2))
    return out,result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir",type=Path,required=True)
    parser.add_argument("--output-root",type=Path,default=OUTPUT)
    args=parser.parse_args()
    if Path.cwd().resolve()!=ROOT:
        parser.error("Run from repository root")
    run(args.analysis_dir,args.output_root)


if __name__=="__main__":
    main()
