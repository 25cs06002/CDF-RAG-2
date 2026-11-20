"""Compare four 132-query B9 fixed/adaptive retrieval policy runs.

Requires exactly four frozen policy artifacts with identical input SHA,
G1 snapshot, source corpus and top_k. No medical truth claims: 'candidate
accepted' is an operational proxy, never clinical correctness.
"""
from __future__ import annotations
import argparse
import json
import statistics
import uuid
from datetime import datetime,timezone
from pathlib import Path

from .thesis132 import _inside,_write_json
from .schema import PipelineRecord

POLICIES={"fixed1","fixed2","fixed3","adaptive"}

def compare(dirs,output_root=None):
    if len(dirs)!=4:raise ValueError("Provide one path for each of four policies")
    outcomes={}
    records={}
    for user_path in dirs:
        folder=_inside(user_path)
        summary=json.loads((folder/"summary.json").read_text(encoding="utf-8"))
        policy=summary["policy"]
        if policy not in POLICIES or policy in outcomes:
            raise ValueError("Missing/duplicate policy run: "+policy)
        if summary["queries"]!=132 or summary["medical_accuracy"] is not False:
            raise ValueError("Not a 132-query non-gold policy experiment")
        rows=[PipelineRecord.model_validate_json(line) for line in
              (folder/"records.jsonl").read_text(encoding="utf-8").splitlines()
              if line.strip()]
        if len(rows)!=132 or len({r.query_id for r in rows})!=132:
            raise ValueError("Missing/duplicate policy query")
        if summary["status_ok"]!=sum(r.status=="ok" for r in rows):
            raise ValueError("Policy summary conflicts with records")
        outcomes[policy]=summary
        records[policy]={r.query_id:r for r in rows}
    if set(outcomes)!=POLICIES:raise ValueError("Need fixed1, fixed2, fixed3, adaptive")
    keys=("input_sha256","graph_manifest_sha256","document_corpus_sha256","top_k")
    for key in keys:
        if len({x[key] for x in outcomes.values()})!=1:
            raise ValueError("Incomparable policy experiment: "+key)
    qids=set(records["adaptive"])
    if any(set(x)!=qids for x in records.values()):
        raise ValueError("Different question IDs across policies")
    per_policy=[]
    for policy in sorted(POLICIES):
        rows=[records[policy][qid] for qid in sorted(qids)]
        accepted=sum(r.status=="ok" for r in rows)
        stage_latency=[r.timings.total_ms for r in rows if r.timings.total_ms is not None]
        # Pre-run model loading is excluded from individual query latency.
        per_policy.append({
            "policy":policy,"queries":132,"candidate_evidence_accepted":accepted,
            "abstained":sum(r.status=="abstained" for r in rows),
            "errors":sum(r.status=="error" for r in rows),
            "mean_total_ms_excluding_init":statistics.mean(stage_latency),
            "median_total_ms_excluding_init":statistics.median(stage_latency),
            "mean_recorded_extra_rounds":statistics.mean(r.feedback_iterations for r in rows),
            "early_stopped":sum("early_stop_candidate_evidence" in r.action_history for r in rows),
        })
    adaptive=records["adaptive"];fixed3=records["fixed3"]
    quality_counterfactual={
        "adaptive_only_candidate":sum(adaptive[x].status=="ok" and fixed3[x].status!="ok" for x in qids),
        "fixed3_only_candidate":sum(fixed3[x].status=="ok" and adaptive[x].status!="ok" for x in qids),
        "both_candidate":sum(adaptive[x].status=="ok" and fixed3[x].status=="ok" for x in qids),
        "neither_candidate":sum(adaptive[x].status!="ok" and fixed3[x].status!="ok" for x in qids),
        "mean_adaptive_minus_fixed3_latency_ms":statistics.mean(
            adaptive[x].timings.total_ms-fixed3[x].timings.total_ms for x in qids),
    }
    result={
        "experiment":"B9_RULE_ONLY_132_POLICY_ABLATION",
        "source_hashes":{k:outcomes["adaptive"][k] for k in keys},
        "per_policy":per_policy,"paired_adaptive_vs_fixed3":quality_counterfactual,
        "gold":False,"clinical_accuracy":None,"learned_PPO":False,
        "warning":"Candidate evidence yield is not medical correctness; performance timings are process/corpus-dependent and initialization excluded. Policy effects are conditional on fixed G1+documents."
    }
    base=_inside(output_root) if output_root else _inside(dirs[0]).parent
    out=base/("ablation_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")+
              "_"+uuid.uuid4().hex[:6])
    out.mkdir(parents=True,exist_ok=False)
    _write_json(out/"ablation.json",result)
    print(json.dumps({"output":str(out),**result},indent=2))
    return out,result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dirs",type=Path,nargs=4,required=True)
    p.add_argument("--output-root",type=Path)
    a=p.parse_args()
    compare(a.run_dirs,a.output_root)

if __name__=="__main__":
    main()
