"""Component ablations of frozen B9-v2. Preserve guard, route, renderer and model.
No reference labels are used for inference. Candidate coverage is not accuracy.
"""
from __future__ import annotations
import argparse,csv,json,statistics,time,subprocess
from collections import Counter
from pathlib import Path
from research_v2.thesis132_b9_v2 import B9V2Controller,GRAPH,CORPUS,ROOT,fingerprints,generate,generation_settings,check_ollama,_cohort,_stamp,G1TypedIndex,PipelineRequest,sha,dump
from research_v2.thesis132_b9_review import RUN,META,readcsv,writecsv
from research_v2.thesis132_b9_ablation import read_run,metrics,paired
from research_v2.pipelines import create_pipeline

OUT=ROOT/"presentation_20261118/thesis132/b9_component_ablation_v2"
VARIANTS=("graph_off","document_off")

def need(ok,msg):
    if not ok:raise ValueError(msg)
def reference():
    m,s,r=read_run(RUN,"adaptive")
    return m,s,r
def frozen():
    m,_,_=reference()
    need(fingerprints(GRAPH)==m["inputs"],"Frozen B9-v2 source/evidence drift")
    return m

class ComponentController(B9V2Controller):
    def __init__(self,*args,component,**kwargs):
        super().__init__(*args,**kwargs)
        need(component in (*VARIANTS,"full"),"Unknown component")
        self.component=component
    def _direct(self,q,target):
        return [] if self.component=="graph_off" else super()._direct(q,target)
    def _paths(self,q,target):
        return [] if self.component=="graph_off" else super()._paths(q,target)
    def retrieve_docs(self,q,target=None,focus=False,exclude=()):
        if self.component=="document_off":return [],False
        return super().retrieve_docs(q,target,focus,exclude)

def controller(variant,selector):
    return ComponentController(G1TypedIndex(GRAPH),create_pipeline("B2",document_corpus=CORPUS,top_k=5,generate=False),
        selector,component=variant,policy="adaptive",top_k=5,model="llama3.2:latest")

def evidence_check(variant,r,t):
    if variant=="graph_off":
        need(not r.causal_edges and not r.causal_paths,"Graph leak")
        need(all(x["kind"]=="candidate_document" for x in t["selected_evidence"]),"Graph selection leak")
    if variant=="document_off":
        need(all(x["kind"]!="candidate_document" for x in t["selected_evidence"]),"Document selection leak")
        need(not any(x.document_id.startswith("PHASE2_CAND_") for x in r.retrieved_documents),"Document retrieval leak")

def row(r,t):
    return dict(query_id=r.query_id,query=r.original_query,route=t["route"],status=r.status,
        scope=t["answer_scope"],stop_reason=t["stopping_reason"],answer=r.final_answer or "",
        evidence=bool(r.retrieved_documents or r.causal_paths),evidence_count=len(r.retrieved_documents),
        rounds=len(t["rounds"]),selector_calls=t["selector_calls"],latency_ms=r.timings.total_ms,
        retrieval_ms=r.timings.retrieval_ms,generation_ms=r.timings.generation_ms,
        validation_ms=r.timings.validation_ms,refinement_ms=r.timings.refinement_ms)

def annotate(rows):
    m={r["query_id"]:r for r in readcsv(META)}
    return [dict(r,query_type=m[r["query_id"]]["query_type"],medical_domain=m[r["query_id"]]["medical_domain"]) for r in rows]

def run(variant,allow_live=False):
    need(allow_live,"Explicit --allow-live required")
    need(variant in VARIANTS,"Unknown variant")
    m=frozen();cfg=generation_settings()
    need(cfg==m["generation_settings"],"Generation settings differ from archived adaptive run")
    check_ollama(cfg)
    import requests
    resp=requests.get(cfg["endpoint"]+"/api/tags",timeout=10);resp.raise_for_status()
    digest=next((x["digest"] for x in resp.json()["models"] if x["name"]==cfg["model"]),None)
    need(digest==m["model_digest"],"Model digest mismatch")
    _,source,qs=_cohort()
    need(sha(source)==m["dataset_sha256"] and len(qs)==132,"Dataset drift")
    c=controller(variant,lambda p:generate(p,settings=cfg))
    dest=OUT/(variant+"_"+_stamp());dest.mkdir(parents=True,exist_ok=False)
    selfsha=sha(Path(__file__))
    manifest=dict(schema="B9_COMPONENT_V1",variant=variant,questions=132,status="RUNNING",
        adaptive_manifest_sha256=sha(RUN/"manifest.json"),dataset_sha256=sha(source),
        inputs=m["inputs"],generation_settings=cfg,model_digest=digest,
        component_code_sha256=selfsha,code_commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
        reference_labels_used_for_inference=False)
    dump(dest/"manifest.json",manifest)
    rows=[]
    wall=time.perf_counter()
    with (dest/"records.jsonl").open("w",encoding="utf-8") as rf,(dest/"round_traces.jsonl").open("w",encoding="utf-8") as tf:
        for i,q in enumerate(qs,1):
            r,t=c.run(PipelineRequest(run_id=dest.name,query_id=q.query_id,query=q.query,graph_version="G1_CANDIDATE"))
            evidence_check(variant,r,t)
            rf.write(r.model_dump_json()+"\n");rf.flush()
            tf.write(json.dumps(t,ensure_ascii=False)+"\n");tf.flush()
            rows.append(row(r,t))
            print(f"[{i:03d}/132] {variant} {r.query_id} {r.status} {r.timings.total_ms/1000:.3f}s",flush=True)
    seconds=time.perf_counter()-wall
    need(fingerprints(GRAPH)==m["inputs"] and sha(Path(__file__))==selfsha,"Code/evidence changed during run")
    rows=annotate(rows);writecsv(dest/"per_query.csv",rows)
    errors=sum(r["status"]=="error" for r in rows)
    summary=dict(variant=variant,errors=errors,**metrics(rows,seconds))
    dump(dest/"summary.json",summary)
    manifest["status"]="COMPLETE_WITH_ERRORS" if errors else "COMPLETE_132"
    manifest["outputs_sha256"]={p.name:sha(p) for p in sorted(dest.iterdir()) if p.name!="manifest.json"}
    dump(dest/"manifest.json",manifest)
    print(json.dumps({"output":str(dest),**summary},indent=2))
    if errors:raise SystemExit(1)
    return dest

def load_variant(path,variant,m):
    manifest=json.loads((path/"manifest.json").read_text())
    need(manifest["schema"]=="B9_COMPONENT_V1" and manifest["variant"]==variant and manifest["status"]=="COMPLETE_132" and manifest["questions"]==132,"Incomplete/wrong component run")
    for k in ("dataset_sha256","inputs","generation_settings","model_digest"):
        need(manifest[k]==m[k],"Component mismatch: "+k)
    need(manifest["adaptive_manifest_sha256"]==sha(RUN/"manifest.json") and not manifest["reference_labels_used_for_inference"],"Wrong reference/provenance")
    for name,digest in manifest["outputs_sha256"].items():
        need(Path(name).name==name and sha(path/name)==digest,"Changed output "+name)
    from research_v2.schema import PipelineRecord
    recs=[PipelineRecord.model_validate_json(x) for x in (path/"records.jsonl").read_text().splitlines()]
    traces=[json.loads(x) for x in (path/"round_traces.jsonl").read_text().splitlines()]
    need(len(recs)==len(traces)==132,"Missing records")
    rows=[]
    for r,t in zip(recs,traces):
        need(r.query_id==t["query_id"] and r.status in ("ok","abstained") and bool(r.final_answer)==(r.status=="ok"),"Record mismatch")
        evidence_check(variant,r,t)
        rows.append(row(r,t))
    summary=json.loads((path/"summary.json").read_text())
    need(summary["errors"]==0 and summary["answered"]==sum(r["status"]=="ok" for r in rows),"Summary mismatch")
    return annotate(rows),summary["wall_seconds"],manifest

def compare(graph_off,document_off):
    m,s,ref=reference();ref=annotate(ref)
    data={"full":ref};wall={"full":s["wall_time_seconds"]};manifests={}
    for v,p in (("graph_off",Path(graph_off)),("document_off",Path(document_off))):
        data[v],wall[v],manifests[v]=load_variant(p,v,m)
        need([(r["query_id"],r["query"],r["route"]) for r in data[v]]==[(r["query_id"],r["query"],r["route"]) for r in ref],"Query/route drift")
    need(manifests["graph_off"]["component_code_sha256"]==manifests["document_off"]["component_code_sha256"],"Different component code")
    common=set.intersection(*[{r["query_id"] for r in rs if r["status"]=="ok"} for rs in data.values()])
    summary=[]
    for v,rs in data.items():
        summary.append(dict(component=v,**metrics(rs,wall[v]),common_answered_n=len(common),
            common_answered_mean_ms=statistics.mean([r["latency_ms"] for r in rs if r["query_id"] in common]) if common else None,
            ai_hallucination_rate=None,strict_grounded_yield=None))
    dest=OUT/("comparison_"+_stamp());dest.mkdir(parents=True,exist_ok=False)
    writecsv(dest/"component_summary_132.csv",summary)
    writecsv(dest/"component_per_query_132.csv",[dict(component=v,**r) for v,rs in data.items() for r in rs])
    writecsv(dest/"component_by_query_type_132.csv",[dict(component=v,query_type=k,**metrics([r for r in rs if r["query_type"]==k])) for v,rs in data.items() for k in sorted({r["query_type"] for r in rs})])
    dump(dest/"paired_summary.json",{v+"_to_full":paired(rs,ref) for v,rs in data.items() if v!="full"})
    dump(dest/"manifest.json",dict(schema="B9_COMPONENT_COMPARISON_V1",records=396,reference_manifest_sha256=sha(RUN/"manifest.json"),
        component_manifest_sha256={"graph_off":sha(Path(graph_off)/"manifest.json"),"document_off":sha(Path(document_off)/"manifest.json")},
        limitations=["Graph-off disables typed evidence, not graph-derived target routing; document-off disables all document retrieval, including comparisons.",
            "Unchanged scheduled rounds can be empty; policy-level dependency comparison, not optimal standalone graph/document baselines.",
            "Separate run timing may differ in thermal/model state. Candidate answers are not correctness. No fixed/component AI labels reused."],
        outputs_sha256={p.name:sha(p) for p in sorted(dest.iterdir())}))
    print(f"{'Component':<16}{'Answers':>10}{'Evidence':>10}{'Mean s':>12}{'QPS':>10}{'Rounds':>10}")
    for r in summary:print(f"{r['component']:<16}{r['answered']:>10}{r['candidate_evidence_count']:>10}{r['mean_latency_ms']/1000:>12.3f}{r['throughput_qps']:>10.4f}{r['mean_rounds']:>10.3f}")
    print("Output:",dest)

def diagnostics():
    """Actual feedback/cache exercise and isolated repeat-query retrieval benchmark."""
    m=frozen()
    traces=[json.loads(x) for x in (RUN/"round_traces.jsonl").read_text().splitlines()]
    need(len(traces)==132,"Trace count")
    _,source,qs=_cohort()
    c=controller("full",lambda _: (_ for _ in ()).throw(RuntimeError("No LLM in benchmark")))
    # Same ordered document retrieval workload, not full pipeline inference.
    # Three passes: cold no cache per call; one warmed cache across all passes.
    workload=[(q.query,None,False,()) for q in qs]*3
    def bench(cache):
        c.cache.clear();c.cache_hits=0;ids=[]
        start=time.perf_counter()
        for q,target,focus,exclude in workload:
            if not cache:c.cache.clear()
            docs,_=c.retrieve_docs(q,target,focus,exclude)
            ids.append([x["id"] for x in docs])
        return time.perf_counter()-start,c.cache_hits,ids
    # Alternate trial order to reduce fixed ordering bias. No answer caching.
    trials=[]
    for i in range(5):
        a,b=(bench(False),bench(True)) if i%2==0 else (lambda x,y:(y,x))(bench(True),bench(False))
        need(a[2]==b[2],"Cache changed retrieved IDs")
        trials.append(dict(trial=i+1,uncached_seconds=a[0],cached_seconds=b[0],cache_hits=b[1],calls=len(workload)))
    dest=OUT/("diagnostics_"+_stamp());dest.mkdir(parents=True,exist_ok=False)
    writecsv(dest/"cache_trials.csv",trials)
    result=dict(reference_manifest_sha256=sha(RUN/"manifest.json"),questions=132,
        adaptive_selector_calls=sum(t["selector_calls"] for t in traces),
        multi_selector_queries=sum(t["selector_calls"]>1 for t in traces),
        valid_empty_selections=sum(h["valid"] and h["selected_ids"]==[] for t in traces for h in t["selector_history"]),
        actual_round_cache_hits=sum(r["cache_hit"] for t in traces for r in t["rounds"]),
        median_uncached_retrieval_seconds=statistics.median(r["uncached_seconds"] for r in trials),
        median_cached_retrieval_seconds=statistics.median(r["cached_seconds"] for r in trials),
        calls_per_trial=len(workload),note="Synthetic repeated-query document retrieval workload only; no measured end-to-end speedup, no LLM calls. Feedback-off is not a measured counterfactual.")
    dump(dest/"diagnostics.json",result);print(json.dumps({"output":str(dest),**result},indent=2))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("command",choices=("run","compare","diagnostics"));p.add_argument("--variant",choices=VARIANTS);p.add_argument("--allow-live",action="store_true");p.add_argument("--graph-off-dir",type=Path);p.add_argument("--document-off-dir",type=Path);a=p.parse_args()
    if a.command=="run":
        if not a.variant:p.error("--variant required")
        run(a.variant,a.allow_live)
    elif a.command=="compare":
        if not a.graph_off_dir or not a.document_off_dir:p.error("both run directories required")
        compare(a.graph_off_dir,a.document_off_dir)
    else:diagnostics()
if __name__=="__main__":main()
