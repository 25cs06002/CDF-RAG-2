"""Descriptive/paired analysis of NEW 132-question run sessions.

Reports operational performance, model status, evidence yield, answer
presence, causal paths, detector/validator outputs, and category/fold views.
Uses semantic-group bootstrap for descriptive uncertainty; the 132 questions
were previously exposed to G1 development, so these are *not* independent
clinical accuracy CIs. No medical gold labels or fabricated reference F1.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import random
import statistics
import uuid
from collections import defaultdict,Counter
from datetime import datetime,timezone
from pathlib import Path

from .thesis132 import _inside,_cohort,_read_json,_write_json,summarize
from .schema import PipelineRecord

BASE=Path(__file__).resolve().parents[1]/"presentation_20261118/thesis132"

def _quantile(values,q):
    values=sorted(values)
    if not values:return None
    loc=(len(values)-1)*q
    lo=int(loc);hi=min(lo+1,len(values)-1)
    return values[lo]+(loc-lo)*(values[hi]-values[lo])

def _mean(values):
    vals=[v for v in values if v is not None]
    return statistics.mean(vals) if vals else None

def _summarize(records):
    n=len(records)
    if not n: return {"n":0}
    statuses=Counter(row["status"] for row in records)
    ts=[row["total_ms"] for row in records if row["total_ms"] is not None]
    detector=[row["hallucination_detected"] for row in records
              if row["hallucination_detected"] is not None]
    validated=[row["validator_passed"] for row in records
               if row["validator_passed"] is not None]
    return {
        "n":n,"status_ok":statuses["ok"],"status_abstained":statuses["abstained"],
        "status_error":statuses["error"],
        "evidence_yield":sum(r["evidence"] for r in records),
        "evidence_yield_pct":100*sum(r["evidence"] for r in records)/n,
        "answers_present":sum(r["answer"] for r in records),
        "answers_present_pct":100*sum(r["answer"] for r in records)/n,
        "mean_total_ms":_mean(ts),"p50_total_ms":_quantile(ts,.5),
        "p95_total_ms":_quantile(ts,.95),
        "mean_retrieval_ms":_mean(r["retrieval_ms"] for r in records),
        "mean_generation_ms":_mean(r["generation_ms"] for r in records),
        "mean_paths_per_question":_mean(r["paths"] for r in records),
        "hallucination_detector_reported":len(detector),
        "hallucination_detector_flagged":sum(detector),
        "validator_reported":len(validated),
        "validator_passed":sum(validated),
        "clinical_correctness":None,
    }

def _bootstrap_by_group(all_rows,group_key="semantic_group",iterations=2000,seed=2026):
    """Group-reweighted average yield: descriptive not true external CI."""
    groups=defaultdict(list)
    for r in all_rows:groups[r[group_key]].append(r)
    keys=sorted(groups)
    rng=random.Random(seed)
    observed=sum(r["evidence"] for r in all_rows)/len(all_rows)
    samples=[]
    for _ in range(iterations):
        chosen=[rng.choice(keys) for __ in keys]
        drawn=[r for k in chosen for r in groups[k]]
        samples.append(sum(r["evidence"] for r in drawn)/len(drawn))
    return {"observed":observed,"lower_95":_quantile(samples,.025),
            "upper_95":_quantile(samples,.975),
            "iterations":iterations,"unit":"semantic_group",
            "interpretation":"exploratory_resampling_on_retrospectively_exposed_132_not_external_generalization"}

def _paired_b7_b8(rows_by_model,meta):
    if "B7" not in rows_by_model or "B8" not in rows_by_model:
        return None
    a={r["query_id"]:r for r in rows_by_model["B7"]}
    b={r["query_id"]:r for r in rows_by_model["B8"]}
    if set(a)!=set(b):raise ValueError("Paired B7/B8 query IDs differ")
    ids=sorted(a)
    n=len(ids)
    outcomes=Counter()
    per=[]
    for qid in ids:
        x=a[qid];y=b[qid]
        key=("both" if x["evidence"] and y["evidence"] else
             "B7_only" if x["evidence"] else
             "B8_only" if y["evidence"] else "neither")
        outcomes[key]+=1
        per.append({
            "query_id":qid,"semantic_group":meta[qid]["semantic_group"],
            "delta_evidence":int(y["evidence"])-int(x["evidence"]),
            "delta_total_ms":(y["total_ms"]-x["total_ms"])
                if y["total_ms"] is not None and x["total_ms"] is not None else None,
        })
    # Exact two-sided discordant-pair sign test (McNemar exact under symmetry).
    ndis=outcomes["B7_only"]+outcomes["B8_only"]
    k=min(outcomes["B7_only"],outcomes["B8_only"])
    p_exact=min(1.0,2*sum(math.comb(ndis,i) for i in range(k+1))/2**ndis) if ndis else 1.0
    groups=defaultdict(list)
    for r in per:groups[r["semantic_group"]].append(r)
    keys=sorted(groups);rng=random.Random(2026)
    deltas=[]
    for _ in range(2000):
        sample=[r for g in (rng.choice(keys) for __ in keys) for r in groups[g]]
        deltas.append(sum(x["delta_evidence"] for x in sample)/len(sample))
    times=[r["delta_total_ms"] for r in per if r["delta_total_ms"] is not None]
    return {
        "n":n,"evidence_outcome_counts":dict(outcomes),
        "B8_minus_B7_evidence_percentage_points":100*sum(r["delta_evidence"] for r in per)/n,
        "bootstrap_95_pp":[100*_quantile(deltas,.025),100*_quantile(deltas,.975)],
        "B8_minus_B7_mean_latency_ms":_mean(times),
        "B8_minus_B7_p50_latency_ms":_quantile(times,.5),
        "B8_minus_B7_p95_latency_ms":_quantile(times,.95),
        "exact_discordant_pair_p_descriptive_only":p_exact,
        "not_a_PPO_causal_estimate":True,
        "clinical_accuracy":False,
    }

def analyze(session_dir,output_root=None):
    work=_inside(session_dir)
    summary=summarize(work)
    conf=_read_json(work/"session.json")
    _,_,questions=_cohort()
    ids=[q.query_id for q in questions[:conf["query_count"]]]
    # Annotation metadata is consulted AFTER inference only.
    frozen_dir,_,_= _cohort()
    with (frozen_dir/"fold_metadata.csv").open(newline="",encoding="utf-8") as f:
        meta={r["query_id"]:r for r in csv.DictReader(f)}
    by_model={}
    for model,entry in conf["runs"].items():
        folder=_inside(work/entry["folder"])
        records=[PipelineRecord.model_validate_json(line)
                 for line in (folder/"records.jsonl").read_text(encoding="utf-8").splitlines()
                 if line.strip()]
        if [r.query_id for r in records]!=ids:
            raise ValueError("Run/model missing ordered 132 cohort: "+model)
        by_model[model]=[{
            "query_id":r.query_id,
            "query_type":meta[r.query_id]["query_type"],
            "medical_domain":meta[r.query_id]["medical_domain"],
            "fold":meta[r.query_id]["fold"],
            "semantic_group":meta[r.query_id]["semantic_group"],
            "status":r.status,
            "evidence":bool(r.retrieved_documents or r.causal_edges or r.causal_paths),
            "answer":bool(r.final_answer),
            "paths":len(r.causal_paths),
            "total_ms":r.timings.total_ms,
            "retrieval_ms":r.timings.retrieval_ms,
            "generation_ms":r.timings.generation_ms,
            "hallucination_detected":r.hallucination_detected,
            "validator_passed":r.validator_passed,
        } for r in records]
    category=[]
    for model,rows in sorted(by_model.items()):
        for kind in sorted(set(r["query_type"] for r in rows)):
            category.append({"model":model,"query_type":kind,**_summarize([
                r for r in rows if r["query_type"]==kind])})
    folds=[]
    for model,rows in sorted(by_model.items()):
        for fold in sorted(set(r["fold"] for r in rows)):
            folds.append({"model":model,"fold":fold,**_summarize([
                r for r in rows if r["fold"]==fold])})
    aggregated={
        "analysis_type":"132_query_development_operational_and_performance",
        "source_session":str(work.relative_to(BASE.parent.parent)),
        "source_dataset_sha256":summary["source_dataset_sha256"],
        "source_mode":summary["mode"],
        "question_count":summary["question_count"],
        "all_132_evaluated":summary["question_count"]==132,
        "model_metrics":summary["models"],
        "by_query_type":category,
        "by_fold":folds,
        "evidence_yield_group_bootstrap":{
            model:_bootstrap_by_group(rows)
            for model,rows in sorted(by_model.items())
        },
        "paired_B7_B8":_paired_b7_b8(by_model,meta),
        "clinical_gold":False,
        "independent_external_holdout":False,
        "no_PPO_only_attribution":True,
        "warnings":[
            "The available 132 questions were exposed to previous graph development.",
            "Evidence yield and detector flags are not medically adjudicated correct answers.",
            "Group bootstrap uncertainty is descriptive and does not remove data leakage.",
            "B1-B4 candidate documents and B5-B8 G0 graph are nonidentical evidence sources.",
            "A full adaptive multi-round architecture must be evaluated separately.",
        ]
    }
    output=(_inside(output_root) if output_root else work)/(
        "analysis_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")+
        "_"+uuid.uuid4().hex[:6])
    output.mkdir(parents=True,exist_ok=False)
    _write_json(output/"analysis.json",aggregated)
    for filename,records in (("by_query_type.csv",category),("by_fold.csv",folds)):
        if not records:continue
        keys=list(records[0])
        with (output/filename).open("w",newline="",encoding="utf-8") as f:
            writer=csv.DictWriter(f,fieldnames=keys)
            writer.writeheader();writer.writerows(records)
    print(json.dumps({"output":str(output),"models":sorted(by_model),
                      "query_count":summary["question_count"],
                      "paired_B7_B8":aggregated["paired_B7_B8"],
                      "medical_gold":False},indent=2))
    return output,aggregated

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--session-dir",type=Path,required=True)
    p.add_argument("--output-root",type=Path)
    a=p.parse_args()
    analyze(a.session_dir,a.output_root)

if __name__=="__main__":
    main()
