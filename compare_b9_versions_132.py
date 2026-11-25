"""Compare real B9-v1 and B9-v2 runs on exactly the same frozen 132 inputs."""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from research_v2.thesis132_b9 import ROOT, sha, dump, csvwrite

V1=ROOT/"presentation_20261118/thesis132/b9_end_to_end/b9_adaptive_20261125T180912711211Z_e6efb5"
Q=ROOT/"data/benchmark/final_132_v1/questions_only.jsonl"

def load(folder,schema):
    m=json.loads((folder/"manifest.json").read_text())
    if m["schema"]!=schema or m["status"]!="COMPLETE_132" or m["questions"]!=132 or m["smoke"]:
        raise ValueError("Require completed full live run")
    if m["dataset_sha256"]!=sha(Q):raise ValueError("Different dataset")
    for k,v in m["outputs_sha256"].items():
        if Path(k).name!=k or sha(folder/k)!=v:raise ValueError("Output changed: "+k)
    rows=[json.loads(x) for x in (folder/"records.jsonl").read_text().splitlines()]
    traces=[json.loads(x) for x in (folder/"round_traces.jsonl").read_text().splitlines()]
    qs=[json.loads(x) for x in Q.read_text().splitlines()]
    if len(rows)!=132 or len(traces)!=132:raise ValueError("Missing rows")
    for q,r,t in zip(qs,rows,traces):
        if (q["query_id"],q["query"])!=(r["query_id"],r["original_query"]) or t["query_id"]!=q["query_id"]:
            raise ValueError("Alignment failure")
        if r["status"]=="error":raise ValueError("Run contains errors")
    return m,rows,traces

def compare(v2,v1=V1):
    a,x,tx=load(v1,"THESIS132_B9_E2E_V1")
    b,y,ty=load(v2,"THESIS132_B9_E2E_V2")
    if a["model_digest"]!=b["model_digest"] or a["policy"]!=b["policy"] or a["top_k"]!=b["top_k"]:
        raise ValueError("Incompatible model or policy")
    for k in a["inputs"]:
        if k not in b["inputs"] or a["inputs"][k]!=b["inputs"][k]:raise ValueError("V1 input/code fingerprint drift: "+k)
    with (ROOT/"data/benchmark/final_132_v1/metadata_review_only.csv").open(newline="") as f:
        meta={r["query_id"]:r for r in csv.DictReader(f)}
    rows=[]
    for r,s,t,u in zip(x,y,tx,ty):
        qid=r["query_id"];q=meta[qid]
        rows.append(dict(query_id=qid,query=r["original_query"],query_type=q["query_type"],medical_domain=q["medical_domain"],
            v1_status=r["status"],v2_status=s["status"],v1_stop=t["stopping_reason"],v2_stop=u["stopping_reason"],
            v2_route=u["route"],v2_answer_scope=u["answer_scope"],v1_answer=r["final_answer"],v2_answer=s["final_answer"],
            latency_delta_ms=s["timings"]["total_ms"]-r["timings"]["total_ms"]))
    out=v2.parent/(v2.name+"_comparison_v1")
    out.mkdir(exist_ok=False)
    csvwrite(out/"paired_v1_v2_132.csv",rows)
    groups=[]
    for k in sorted({r["query_type"] for r in rows}):
        g=[r for r in rows if r["query_type"]==k]
        groups.append(dict(query_type=k,queries=len(g),v1_ok=sum(r["v1_status"]=="ok" for r in g),
            v2_ok=sum(r["v2_status"]=="ok" for r in g),v2_partial_path=sum(r["v2_answer_scope"]=="partial_path_evidence" for r in g)))
    csvwrite(out/"by_query_type.csv",groups)
    sa=json.loads((v1/"summary.json").read_text());sb=json.loads((v2/"summary.json").read_text())
    result=dict(pairs=132,transitions=dict(Counter(r["v1_status"]+"_to_"+r["v2_status"] for r in rows)),
        v2_partial_path=sum(r["v2_answer_scope"]=="partial_path_evidence" for r in rows),
        v2_nonnegative_candidate_answers=sum(r["v2_status"]=="ok" and r["query_type"] not in ("ambiguous","unsupported") for r in rows),
        v2_negative_answered=sum(r["v2_status"]=="ok" and r["query_type"] in ("ambiguous","unsupported") for r in rows),
        v1_mean_latency_ms=sa["mean_latency_ms"],v2_mean_latency_ms=sb["mean_latency_ms"],
        v1_qps=sa["actual_run_throughput_qps"],v2_qps=sb["actual_run_throughput_qps"],
        v1_manifest_sha256=sha(v1/"manifest.json"),v2_manifest_sha256=sha(v2/"manifest.json"),
        medical_accuracy=None,warning="Candidate availability and abstention behavior only. Partial evidence is not complete answer correctness. Development-exposed 132.")
    dump(out/"summary.json",result)
    print(json.dumps({"output":str(out),**result},indent=2))
if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--v2-dir",type=Path,required=True)
    p.add_argument("--v1-dir",type=Path,default=V1)
    a=p.parse_args()
    compare(a.v2_dir.resolve(),a.v1_dir.resolve())
