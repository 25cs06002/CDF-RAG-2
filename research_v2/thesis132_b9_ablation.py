"""Controlled operational ablation of frozen B9-v2 policies, no inference.

Compare fixed1, fixed3 and adaptive with identical inputs/configuration.
Do not reuse AI labels for changed outputs or equate availability with quality.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import statistics
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
from compare_b9_versions_132 import load
from research_v2.thesis132_b9 import ROOT,sha,dump
from research_v2.thesis132_b9_review import RUN,META,readcsv,writecsv,validate,aggregate

POLICIES=("fixed1","fixed3","adaptive")
OUTPUT=ROOT/"presentation_20261118/thesis132/b9_ablation_v2"
SCHEMA="THESIS132_B9_E2E_V2"

def require(ok,msg):
    if not ok:raise ValueError(msg)
def mean(xs):
    return statistics.mean(xs) if xs else None
def quantile(xs,p):
    if not xs:return None
    x=sorted(xs);i=(len(x)-1)*p;lo=int(i)
    return x[lo]+(i-lo)*(x[min(lo+1,len(x)-1)]-x[lo])
def rate(n,d):return n/d if d else None

def read_run(path,policy):
    path=Path(path).resolve()
    m,rows,traces=load(path,SCHEMA)
    require(m["policy"]==policy and m["variant"]=="B9_v2","Unexpected policy/variant: "+policy)
    summary=json.loads((path/"summary.json").read_text())
    require(summary["policy"]==policy and summary["questions"]==132 and summary["errors"]==0,"Summary mismatch")
    statuses=Counter(r["status"] for r in rows)
    require(set(statuses)<= {"ok","abstained"},"Unexpected status")
    require(statuses["ok"]==summary["ok"] and statuses["abstained"]==summary["abstained"],"Status counts mismatch")
    data=[]
    for r,t in zip(rows,traces):
        require(t["policy"]==policy,"Trace policy mismatch")
        require(bool(r["final_answer"])==(r["status"]=="ok"),"Answer/status mismatch")
        rounds=t["rounds"];calls=t["selector_calls"]
        require(len(rounds)<= (1 if policy=="fixed1" else 3),"Round budget exceeded")
        require(calls==sum(bool(z["selector_attempted"]) for z in rounds),"Selector call count mismatch")
        require(policy=="adaptive" or calls<=1,"Fixed policy made multiple selector calls")
        if t["route"] in ("guard","unsupported_comparison"):
            require(not rounds and not calls and r["status"]=="abstained","Guard mismatch")
        elif policy!="adaptive":
            require(len(rounds)==(1 if policy=="fixed1" else 3),"Fixed policy budget mismatch")
        timing=r["timings"]
        require(all(v is None or (math.isfinite(v) and v>=0) for v in timing.values()),"Invalid timing")
        data.append(dict(query_id=r["query_id"],query=r["original_query"],policy=policy,
            route=t["route"],status=r["status"],scope=t["answer_scope"],stop_reason=t["stopping_reason"],
            answer=r["final_answer"] or "",evidence=bool(r["retrieved_documents"] or r["causal_paths"]),
            evidence_count=len(r["retrieved_documents"]),rounds=len(rounds),selector_calls=calls,
            latency_ms=timing["total_ms"],retrieval_ms=timing.get("retrieval_ms"),
            generation_ms=timing.get("generation_ms"),validation_ms=timing.get("validation_ms"),
            refinement_ms=timing.get("refinement_ms")))
    require(sum(r["evidence"] for r in data)==summary["candidate_evidence_count"],"Evidence count mismatch")
    require(sum(r["selector_calls"] for r in data)==summary["total_selector_calls"],"Selector summary mismatch")
    require(math.isclose(mean([r["latency_ms"] for r in data]),summary["mean_latency_ms"],rel_tol=1e-9),"Latency summary mismatch")
    require(summary["wall_time_seconds"]>0 and math.isclose(132/summary["wall_time_seconds"],summary["actual_run_throughput_qps"],rel_tol=1e-9),"Throughput mismatch")
    return m,summary,data

def matching(manifests):
    ref=manifests["adaptive"]
    for policy,m in manifests.items():
        for key in ("dataset_sha256","inputs","top_k","model_digest","generation_settings","generation_method","variant","reference_labels_used_for_inference"):
            require(m[key]==ref[key],f"{policy}: unmatched {key}")
        require(not m["reference_labels_used_for_inference"],"Reference leakage")

def metrics(rows,wall=None):
    ok=[r for r in rows if r["status"]=="ok"]
    negative=[r for r in rows if r["query_type"] in ("ambiguous","unsupported")]
    nonnegative=[r for r in rows if r["query_type"] not in ("ambiguous","unsupported")]
    lat=[r["latency_ms"] for r in rows]
    active=[r for r in rows if r["rounds"]>0]
    out=dict(questions=len(rows),answered=len(ok),abstained=len(rows)-len(ok),
        candidate_evidence_count=sum(r["evidence"] for r in rows),
        candidate_evidence_rate=rate(sum(r["evidence"] for r in rows),len(rows)),
        answer_coverage=rate(len(ok),len(rows)),
        nonnegative_answered=sum(r["status"]=="ok" for r in nonnegative),
        nonnegative_count=len(nonnegative),
        negative_answered=sum(r["status"]=="ok" for r in negative),
        negative_count=len(negative),
        partial_path_answers=sum(r["scope"]=="partial_path_evidence" for r in ok),
        strict_typed_path_answers=sum(r["scope"]=="strict_typed_path" for r in ok),
        comparison_answers=sum(r["scope"]=="comparison_evidence" for r in ok),
        mean_latency_ms=mean(lat),median_latency_ms=quantile(lat,.5),p95_latency_ms=quantile(lat,.95),
        mean_answered_latency_ms=mean([r["latency_ms"] for r in ok]),
        mean_abstained_latency_ms=mean([r["latency_ms"] for r in rows if r["status"]=="abstained"]),
        total_retrieval_rounds=sum(r["rounds"] for r in rows),
        mean_rounds=mean([r["rounds"] for r in rows]),
        mean_rounds_when_retrieval=mean([r["rounds"] for r in active]),
        total_selector_calls=sum(r["selector_calls"] for r in rows),
        mean_selector_calls=mean([r["selector_calls"] for r in rows]))
    for key in ("retrieval_ms","generation_ms","validation_ms","refinement_ms"):
        xs=[r[key] for r in rows if r[key] is not None]
        out["mean_"+key+"_observed"]=mean(xs)
        out[key+"_observations"]=len(xs)
    out.update(wall_seconds=wall,throughput_qps=rate(len(rows),wall),candidate_answers_per_second=rate(len(ok),wall))
    return out

def paired(a,b):
    require([r["query_id"] for r in a]==[r["query_id"] for r in b],"Paired IDs differ")
    both=[(x,y) for x,y in zip(a,b) if x["status"]==y["status"]=="ok"]
    transitions=Counter(x["status"]+"_to_"+y["status"] for x,y in zip(a,b))
    return dict(question_pairs=len(a),both_answered=len(both),transitions=dict(transitions),
        identical_answers_among_both=sum(x["answer"]==y["answer"] for x,y in both),
        mean_all_query_latency_delta_ms=mean([y["latency_ms"]-x["latency_ms"] for x,y in zip(a,b)]),
        mean_common_answered_latency_delta_ms=mean([y["latency_ms"]-x["latency_ms"] for x,y in both]),
        mean_common_answered_from_ms=mean([x["latency_ms"] for x,y in both]),
        mean_common_answered_to_ms=mean([y["latency_ms"] for x,y in both]),
        mean_round_delta=mean([y["rounds"]-x["rounds"] for x,y in zip(a,b)]),
        mean_selector_call_delta=mean([y["selector_calls"]-x["selector_calls"] for x,y in zip(a,b)]))

def report(paths,output=None,package=None,review=None):
    require(set(paths)==set(POLICIES),"Require all three policies")
    manifests={};summaries={};data={}
    metadata={r["query_id"]:r for r in readcsv(META)}
    for policy in POLICIES:
        manifests[policy],summaries[policy],rows=read_run(paths[policy],policy)
        data[policy]=[dict(r,query_type=metadata[r["query_id"]]["query_type"],
                           medical_domain=metadata[r["query_id"]]["medical_domain"]) for r in rows]
    matching(manifests)
    ref=data["adaptive"]
    for policy,rows in data.items():
        require([(r["query_id"],r["query"],r["route"]) for r in rows]==[(r["query_id"],r["query"],r["route"]) for r in ref],"Query/route drift")
    summary=[dict(policy=p,**metrics(data[p],summaries[p]["wall_time_seconds"])) for p in POLICIES]
    common={r["query_id"] for r in ref if r["status"]=="ok"}
    for rows in data.values():common &= {r["query_id"] for r in rows if r["status"]=="ok"}
    # The common answered subset avoids comparing an answer with a cheap abstention.
    for r in summary:
        r["common_all_three_answered_n"]=len(common)
        r["mean_common_all_three_latency_ms"]=mean([x["latency_ms"] for x in data[r["policy"]] if x["query_id"] in common])
    # Quality only for the original answer-specific validated adaptive review.
    # Fixed policy labels stay unavailable even if selected text coincides.
    quality=None
    if package or review:
        require(package is not None and review is not None,"Supply both review and package")
        _,rows=validate(Path(paths["adaptive"]),Path(package),Path(review))
        quality=aggregate(rows)
    for r in summary:
        for k in ("hallucination_rate_ai","hallucination_rate_substantive_ai","supported_substantive_yield","strict_supported_and_fully_grounded_yield","label_coverage"):
            r[k]=quality[k] if quality is not None and r["policy"]=="adaptive" else None
    dest=Path(output).resolve() if output else OUTPUT/("ablation_132_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    dest.mkdir(parents=True,exist_ok=False)
    writecsv(dest/"ablation_summary_132.csv",summary)
    writecsv(dest/"ablation_per_query_132.csv",[r for p in POLICIES for r in data[p]])
    for field in ("query_type","route"):
        groups=[]
        for p in POLICIES:
            for k in sorted({r[field] for r in data[p]}):
                groups.append(dict(policy=p,**{field:k},**metrics([r for r in data[p] if r[field]==k])))
        writecsv(dest/f"ablation_by_{field}_132.csv",groups)
    pairs=[dict(query_id=r["query_id"],query=r["query"],query_type=r["query_type"],
        **{p+"_"+k:data[p][i][k] for p in POLICIES for k in ("status","scope","latency_ms","rounds","selector_calls","answer")})
        for i,r in enumerate(ref)]
    writecsv(dest/"ablation_paired_132.csv",pairs)
    dump(dest/"paired_summary.json",{p+"_to_adaptive":paired(data[p],ref) for p in ("fixed1","fixed3")})
    dump(dest/"manifest.json",dict(schema="THESIS132_B9_ROUND_ABLATION_V1",records=396,
        run_manifest_sha256={p:sha(Path(paths[p])/"manifest.json") for p in POLICIES},
        adaptive_review_audit_sha256=sha(Path(review)/"ai_evaluation_audit.json") if quality is not None else None,
        limitations=["Fixed1 uses the first scheduled action, which may be typed retrieval with no evidence; fixed3 runs all scheduled actions before selection. Adaptive selects during retrieval and may retry; this is a policy-level comparison, not an isolated round-count effect.",
          "Guarded queries have zero rounds in every policy. Same inputs and settings do not guarantee identical thermal/model-cache conditions across separate runs.",
          "Candidate answers and partial pathways are not verified medical correctness.",
          "AI quality unavailable for fixed policies pending answer-specific review; no human gold; fixed development cohort."],
        outputs_sha256={f.name:sha(f) for f in sorted(dest.iterdir())}))
    print("\nB9-v2 POLICY ABLATION — OPERATIONAL METRICS")
    print(f"{'Policy':<10}{'Answers':>10}{'Evidence':>10}{'Rounds':>10}{'LLM calls':>12}{'Mean s':>12}{'P95 s':>12}{'QPS':>10}")
    for r in summary:
        print(f"{r['policy']:<10}{r['answered']:>10}{r['candidate_evidence_count']:>10}{r['mean_rounds']:>10.3f}{r['total_selector_calls']:>12}{r['mean_latency_ms']/1000:>12.3f}{r['p95_latency_ms']/1000:>12.3f}{r['throughput_qps']:>10.4f}")
    print("\nCommon answered questions:",len(common))
    print("Output:",dest)
    return dest

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fixed1-dir",required=True,type=Path)
    p.add_argument("--fixed3-dir",required=True,type=Path)
    p.add_argument("--adaptive-dir",default=RUN,type=Path)
    p.add_argument("--package-dir",type=Path)
    p.add_argument("--review-dir",type=Path)
    p.add_argument("--output-dir",type=Path)
    a=p.parse_args()
    report(dict(fixed1=a.fixed1_dir,fixed3=a.fixed3_dir,adaptive=a.adaptive_dir),a.output_dir,a.package_dir,a.review_dir)
if __name__=="__main__":main()
