"""G1 target and candidate evidence sufficiency diagnostic for fixed 132.

Compares existing G1 stable B5/B6 retrieval with an intentionally cautious,
review-gated typed retrieval/early-stop prototype. Does NOT read reviewer
labels during inference or claim medical answer correctness. Optionally joins
the independently recorded selected 17-case triage *after* predictions to
quantify changes and possible regressions; all results are developmental.
"""
from __future__ import annotations
import argparse
import csv
import json
import re
import time
import uuid
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path

from .experiment_runner import PROJECT_ROOT as ROOT,file_sha256
from .base_pipeline import PipelineRequest
from .g1_stable_retrieval import StableTargetG1
from .g1_typed_retrieval import G1TypedIndex
from .pipelines.baselines import G0Corpus,BaselinePipeline
from .thesis132 import _cohort,_inside,_write_json

G0=ROOT/"research_artifacts/graph-G0"
BASE=ROOT/"presentation_20261118/thesis132/g1_target_diagnostics"
COMPARATIVE=re.compile(r"\b(compared?|versus|vs\.?|difference|contrast|distinguish)\b",re.I)
MULTIHOP=re.compile(r"\b(pathways?|intermediate|through|chain|multi.hop|sequence|leads? to)\b",re.I)
SUBTYPE=re.compile(r"\btype\s*[12]\s*diabetes\b",re.I)
UNSUPPORTED=re.compile(r"\b(exact (?:diagnosis|disease|measurement)|without .*?(?:records|measurements|symptoms)|from (?:their occupation|this alone))\b",re.I)

def _bool(x):
    if isinstance(x,bool):return x
    if x in ("True","true","1","yes"):return True
    if x in ("False","false","0","no",""):return False
    raise ValueError("Unrecognized boolean: "+str(x))

def _read_csv(path):
    with Path(path).open(newline="",encoding="utf-8") as f:
        return list(csv.DictReader(f))

def _safe_gate(index,question, *, top_k=5,query_id="Q"):
    """Pure question-text/evidence gate; never reads labels or review triage."""
    start=time.perf_counter()
    # Answerability regex is intentionally conservative, not a clinical
    # classifier. Ambiguous diagnosis requests should abstain.
    if UNSUPPORTED.search(question):
        return {"target":"","evidence_ids":[],"status":"abstained",
                "decision":"withhold","reason":"under_determined_question",
                "rounds":0,"retrieval_ms":(time.perf_counter()-start)*1000}
    mentioned=[node for node in index.nodes if index.by_effect[node.casefold()]
               and (node.casefold() in question.casefold())]
    if COMPARATIVE.search(question) and len(set(mentioned))>=2:
        return {"target":"","evidence_ids":[],"status":"abstained",
                "decision":"clarification_or_multi_target_required",
                "reason":"comparative_multi_entity_not_single_target",
                "rounds":0,"retrieval_ms":(time.perf_counter()-start)*1000}
    direct,reason=index.retrieve(question,causal_multihop=False,top_k=top_k,
                                 query_id=query_id,run_id="THESIS132_CANDIDATE_GATE")
    rounds=1
    ids=[d.document_id for d in direct.retrieved_documents]
    target=direct.matched_concept or ""
    if SUBTYPE.search(question) and target.lower()=="diabetes":
        ids=[]
        target=""
        reason="subtype_not_generic_diabetes"
    # A source-reviewed candidate typed relationship is evidence to inspect,
    # not independently expert-confirmed medical gold.
    if not ids and MULTIHOP.search(question):
        multi,reason2=index.retrieve(question,causal_multihop=True,top_k=top_k,
                                     query_id=query_id,run_id="THESIS132_CANDIDATE_GATE")
        rounds=2
        if multi.causal_paths and multi.matched_concept:
            target=multi.matched_concept
            ids=list(dict.fromkeys(d.document_id for d in multi.retrieved_documents))
            ids+=["PATH:"+">".join(p.nodes) for p in multi.causal_paths]
            reason="typed_candidate_twohop_path_found"
        else:
            reason="no_approved_candidate_twohop_after_refinement"
    enough=bool(ids)
    return {"target":target,"evidence_ids":ids,
            "status":"candidate_evidence" if enough else "abstained",
            "decision":"candidate_for_review" if enough else "withhold",
            "reason":reason,"rounds":rounds,
            "retrieval_ms":(time.perf_counter()-start)*1000}

def run(graph_dir,triage_dir=None,output_root=BASE,top_k=5):
    graph_dir=_inside(graph_dir)
    output_root=_inside(output_root)
    graph_file=graph_dir/"manifest.json"
    if not graph_file.is_file():
        raise FileNotFoundError("Local G1 candidate snapshot missing: "+str(graph_file))
    if top_k<1:raise ValueError("top_k must be positive")
    index=G1TypedIndex(graph_dir)
    g0=G0Corpus(G0)
    stable={}
    stable_pipes={}
    for model,name in (("B5","graph_rag"),("B6","causal_kg_rag")):
        stable[model]=StableTargetG1(graph_dir,frozen=g0,pipeline=name)
        stable_pipes[model]=BaselinePipeline(name,corpus=stable[model],
                                            top_k=top_k,generate=False)
    frozen,input_file,questions=_cohort()
    rows=[]
    for q in questions:
        # Research labels are never loaded here; only frozen question text.
        gate=_safe_gate(index,q.query,top_k=top_k,query_id=q.query_id)
        for model in ("B5","B6"):
            rec=stable_pipes[model].run(PipelineRequest(
                run_id="THESIS132_STABLE_SHADOW",
                query_id=q.query_id,query=q.query))
            baseline=bool(rec.retrieved_documents or rec.causal_paths)
            evidence=gate["status"]=="candidate_evidence"
            rows.append({
                "model":model,"query_id":q.query_id,"question":q.query,
                "previous_stable_target":rec.matched_concept or "",
                "previous_stable_reason":stable[model].last_reason,
                "previous_stable_evidence":baseline,
                "guarded_target":gate["target"],"guarded_decision":gate["decision"],
                "guarded_reason":gate["reason"],"guarded_candidate_evidence":evidence,
                "guarded_typed_evidence_ids":"|".join(gate["evidence_ids"]),
                "guarded_rounds":gate["rounds"],
                "guarded_retrieval_ms":gate["retrieval_ms"],
                "previous_retrieval_ms":rec.timings.retrieval_ms,
                "change_vs_stable":"new_candidate" if evidence and not baseline else
                    "withheld_previous_evidence" if baseline and not evidence else
                    "candidate_in_both" if baseline else "no_candidate_in_both",
            })
    if len(rows)!=264:
        raise ValueError("Expected 2 × 132 model-query records")
    summary={
        "scope":"fixed_132_source_screened_development_questions",
        "source_query_sha256":file_sha256(input_file),
        "g1_candidate_manifest_sha256":file_sha256(graph_file),
        "query_model_records":264,"independent_clinical_gold":False,
        "inference_used_reviewer_labels":False,
        "approach":"conservative_typed_candidate_sufficiency_0_1_2_rounds",
        "full_adaptive_rag_or_ppo":False,
        "models":{
            m:{
                "previous_stable_evidence":sum(r["previous_stable_evidence"] for r in rows if r["model"]==m),
                "guarded_candidate_evidence":sum(r["guarded_candidate_evidence"] for r in rows if r["model"]==m),
                "withheld_previous_evidence":sum(r["change_vs_stable"]=="withheld_previous_evidence"
                                                 for r in rows if r["model"]==m),
                "new_candidate":sum(r["change_vs_stable"]=="new_candidate" for r in rows if r["model"]==m),
            } for m in ("B5","B6")
        },
        "warnings":[
            "Typed links were review-gated as source candidates, not independently clinical gold.",
            "Withholding an unverified result is not automatically a correct answer.",
            "No learned PPO policy or final grounded generator is implemented in this diagnostic.",
            "Comparing typed retrieval against stable causal-format graph changes both target policy and evidence source.",
        ],
    }
    review_rows=None
    if triage_dir:
        triage_dir=_inside(triage_dir)
        triage=_read_csv(triage_dir/"priority_cases.csv")
        triage_summary=_read_json(triage_dir/"summary.json")
        if len(triage)!=17 or triage_summary.get("attention_cases")!=17:
            raise ValueError("Expected 17 reviewed attention cases")
        keyed={(r["model"],r["query_id"]):r for r in rows}
        selected=[]
        for case in triage:
            key=(case["family"],case["query_id"])
            if key not in keyed:
                raise ValueError("Reviewed case absent from 132 run: "+str(key))
            observed=keyed[key]
            selected.append({
                "case_id":case["case_id"],"priority":case["priority"],
                "family":case["family"],"query_id":case["query_id"],
                "selection_reason":case["selection_reason"],
                "review_stable_target":case["review_stable_target"],
                "review_flags":case["issue_flags"],
                "prior_target":case["stable_target"],
                "guarded_target":observed["guarded_target"],
                "prior_evidence":observed["previous_stable_evidence"],
                "guarded_candidate_evidence":observed["guarded_candidate_evidence"],
                "guarded_decision":observed["guarded_decision"],
                "changed":observed["change_vs_stable"],
                "remains_unadjudicated_medical_gold":True,
            })
        review_rows=selected
        summary["retrospective_selected_cases"]={
            "n":len(selected),
            "priority_counts":dict(Counter(c["priority"] for c in selected)),
            "guarded_withholds":sum(c["guarded_decision"]=="withhold" for c in selected),
            "new_candidate_count":sum(c["guarded_candidate_evidence"] for c in selected),
            "note":"Posthoc 17-case diagnostic only; selected review labels NOT used for inference or as an independent heldout."
        }
    out=output_root/("diagnostic_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")+
                     "_"+uuid.uuid4().hex[:6])
    out.mkdir(parents=True,exist_ok=False)
    with (out/"per_query.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    if review_rows:
        with (out/"reviewed_17_cases.csv").open("w",newline="",encoding="utf-8") as f:
            writer=csv.DictWriter(f,fieldnames=list(review_rows[0]))
            writer.writeheader();writer.writerows(review_rows)
    _write_json(out/"summary.json",summary)
    print(json.dumps({"output":str(out),**summary},indent=2))
    return out,summary

def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-dir",type=Path,required=True)
    parser.add_argument("--triage-dir",type=Path)
    parser.add_argument("--output-root",type=Path,default=BASE)
    parser.add_argument("--top-k",type=int,default=5)
    a=parser.parse_args()
    if Path.cwd().resolve()!=ROOT:parser.error("Run from project root")
    run(a.graph_dir,a.triage_dir,a.output_root,a.top_k)

if __name__=="__main__":
    main()
