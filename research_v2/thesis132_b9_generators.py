"""Controlled compact-ID generator comparison for the frozen B9-v2 controller.

All models share prompts, selection validation and candidate evidence. The adaptive
controller can choose further retrieval based on the selector, so later prompts
may differ. No AI or human gold labels are used during inference.
"""
from __future__ import annotations
import argparse,json,os,subprocess,time
from collections import Counter
from pathlib import Path
import requests
from research_v2.thesis132_b9_profile import Selector,Monitor,profile_summary,meminfo
from research_v2.thesis132_b9_v2 import B9V2Controller,ROOT,GRAPH,CORPUS,fingerprints,generation_settings,check_ollama,_cohort,_stamp,G1TypedIndex,PipelineRequest,sha,dump
from research_v2.thesis132_b9_review import RUN,writecsv,readcsv
from research_v2.thesis132_b9_components import row,annotate
from research_v2.thesis132_b9_ablation import metrics,paired
from research_v2.pipelines import create_pipeline

OUT=ROOT/"presentation_20261118/thesis132/b9_generator_comparison_v2"
MODELS={"llama3b":"llama3.2:latest","llama1b":"llama3.2:1b","qwen15b":"qwen2.5:1.5b"}
SCHEMA="B9_GENERATOR_PROFILE_V1"
def need(ok,msg):
    if not ok:raise ValueError(msg)
def get(url,**kw):
    r=requests.get(url,timeout=10,**kw);r.raise_for_status();return r.json()
def settings(key,ref):
    cfg=generation_settings()
    need({k:v for k,v in cfg.items() if k!="model"}=={k:v for k,v in ref["generation_settings"].items() if k!="model"},"Non-model Ollama settings differ from frozen reference")
    return dict(cfg,model=MODELS[key])
def check_reference():
    ref=json.loads((RUN/"manifest.json").read_text())
    need(ref["status"]=="COMPLETE_132" and fingerprints(GRAPH)==ref["inputs"],"Frozen B9-v2 inputs changed")
    return ref
def installed(cfg):
    check_ollama(cfg)
    tags=get(cfg["endpoint"]+"/api/tags")["models"]
    model=next((m for m in tags if m["name"]==cfg["model"]),None)
    need(model is not None and bool(model.get("digest")),"Model digest unavailable")
    return model
def cold_start(cfg):
    # Unload only experiment models. Refuse unrelated loaded models rather
    # than evicting them or including their memory in the comparison.
    endpoint=cfg["endpoint"]
    loaded=get(endpoint+"/api/ps")["models"]
    for m in loaded:
        name=m.get("name",m.get("model"))
        need(name in MODELS.values(),"Unrelated Ollama model loaded: "+str(name)+"; stop it manually")
    for m in loaded:
        name=m.get("name",m.get("model"))
        r=requests.post(endpoint+"/api/generate",json={"model":name,"keep_alive":0},timeout=cfg["timeout_seconds"]);r.raise_for_status()
    need(not get(endpoint+"/api/ps")["models"],"Ollama still has a loaded model; stop it before continuing")
def host():
    return dict(platform=os.uname().sysname,machine=os.uname().machine,cpu_count=os.cpu_count(),mem_total_kib=meminfo()["MemTotal"])
def run(key,allow_live=False,smoke=False):
    need(allow_live,"Explicit --allow-live required")
    ref=check_reference();cfg=settings(key,ref);model=installed(cfg)
    if key=="llama3b":need(model["digest"]==ref["model_digest"],"Control model digest differs from original B9")
    version=get(cfg["endpoint"]+"/api/version")["version"]
    _,source,qs=_cohort()
    need(len(qs)==132 and sha(source)==ref["dataset_sha256"],"Frozen dataset drift")
    cohort=[qs[i] for i in (0,30,45,60,77,84)] if smoke else qs
    cold_start(cfg)
    selector=Selector("compact_ids",cfg)
    c=B9V2Controller(G1TypedIndex(GRAPH),create_pipeline("B2",document_corpus=CORPUS,top_k=5,generate=False),selector,policy="adaptive",top_k=5,model=cfg["model"])
    dest=OUT/(("smoke_" if smoke else "")+key+"_"+_stamp());dest.mkdir(parents=True,exist_ok=False)
    codehash=sha(Path(__file__))
    profiler=ROOT/"research_v2/thesis132_b9_profile.py"
    profilehash=sha(profiler)
    manifest=dict(schema=SCHEMA,model_key=key,model_name=cfg["model"],model_digest=model["digest"],
        model_size_bytes=model.get("size"),model_details=model.get("details"),ollama_version=version,
        status="RUNNING",questions=len(cohort),smoke=smoke,reference_manifest_sha256=sha(RUN/"manifest.json"),
        inputs=ref["inputs"],dataset_sha256=sha(source),generation_settings=cfg,selector_variant="compact_ids",
        policy="adaptive",top_k=5,runner_sha256=codehash,profiler_sha256=profilehash,host=host(),
        code_commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
        resource_interval_seconds=.25,initial_model_state="no models reported by api/ps",reference_labels_used_for_inference=False)
    dump(dest/"manifest.json",manifest)
    rows=[];wall=time.perf_counter()
    with Monitor() as monitor:
        with (dest/"records.jsonl").open("w",encoding="utf-8") as rf,(dest/"round_traces.jsonl").open("w",encoding="utf-8") as tf,(dest/"selector_telemetry.jsonl").open("w",encoding="utf-8") as ef:
            for i,q in enumerate(cohort,1):
                selector.query_id=q.query_id;before=len(selector.events)
                r,t=c.run(PipelineRequest(run_id=dest.name,query_id=q.query_id,query=q.query,graph_version="G1_CANDIDATE"))
                rf.write(r.model_dump_json()+"\n");rf.flush()
                tf.write(json.dumps(t,ensure_ascii=False)+"\n");tf.flush()
                for e in selector.events[before:]:ef.write(json.dumps(e,ensure_ascii=False)+"\n")
                ef.flush();rows.append(row(r,t))
                print(f"[{i:03d}/{len(cohort)}] {key} {q.query_id} {r.status} {r.timings.total_ms/1000:.3f}s",flush=True)
    seconds=time.perf_counter()-wall
    need(fingerprints(GRAPH)==ref["inputs"] and sha(Path(__file__))==codehash and sha(profiler)==profilehash,"Code/evidence changed during run")
    need(installed(cfg)["digest"]==model["digest"],"Model digest changed during run")
    rows=annotate(rows);writecsv(dest/"per_query.csv",rows);writecsv(dest/"resource_samples.csv",monitor.samples)
    errors=sum(r["status"]=="error" for r in rows)
    summary=dict(model_key=key,model_name=cfg["model"],errors=errors,**metrics(rows,seconds),
        **{k:v for k,v in profile_summary(selector.events,monitor.samples).items() if k!="selector_calls"})
    dump(dest/"summary.json",summary)
    manifest["status"]="COMPLETE_WITH_ERRORS" if errors else "SMOKE_ONLY" if smoke else "COMPLETE_132"
    manifest["outputs_sha256"]={p.name:sha(p) for p in sorted(dest.iterdir()) if p.name!="manifest.json"}
    dump(dest/"manifest.json",manifest)
    print(json.dumps({"output":str(dest),**summary},indent=2))
    if errors:raise SystemExit(1)
    return dest

def load_profile(path,key):
    path=Path(path).resolve();m=json.loads((path/"manifest.json").read_text())
    need(m["schema"]==SCHEMA and m["model_key"]==key and m["model_name"]==MODELS[key] and m["status"]=="COMPLETE_132" and m["questions"]==132 and not m["smoke"],"Wrong or incomplete model run: "+key)
    for k,v in m["outputs_sha256"].items():need(Path(k).name==k and sha(path/k)==v,"Changed file: "+k)
    s=json.loads((path/"summary.json").read_text())
    from research_v2.schema import PipelineRecord
    rs=[PipelineRecord.model_validate_json(l) for l in (path/"records.jsonl").read_text().splitlines()]
    ts=[json.loads(l) for l in (path/"round_traces.jsonl").read_text().splitlines()]
    need(len(rs)==len(ts)==132 and all(r.query_id==t["query_id"] and r.generator_model==m["model_name"] for r,t in zip(rs,ts)),"Record/model alignment")
    rows=annotate([row(r,t) for r,t in zip(rs,ts)])
    need(all(r["status"] in ("ok","abstained") and bool(r["answer"])==(r["status"]=="ok") for r in rows),"Bad statuses")
    events=[json.loads(l) for l in (path/"selector_telemetry.jsonl").read_text().splitlines()]
    need(Counter(e["query_id"] for e in events)==Counter({r["query_id"]:r["selector_calls"] for r in rows if r["selector_calls"]}),"Telemetry alignment")
    need(all(e["variant"]=="compact_ids" for e in events),"Wrong selector mode")
    samples=readcsv(path/"resource_samples.csv")
    for z in samples:
        for k,v in z.items():z[k]=float(v) if v!="" else None
    result=dict(model_key=key,model_name=m["model_name"],model_digest=m["model_digest"],model_size_bytes=m["model_size_bytes"],
        **metrics(rows,s["wall_seconds"]),**{k:v for k,v in profile_summary(events,samples).items() if k!="selector_calls"})
    need(s["errors"]==0 and result["answered"]==s["answered"],"Summary mismatch")
    return m,result,rows

def comparable(a,b):
    for k in ("reference_manifest_sha256","inputs","dataset_sha256","selector_variant","policy","top_k","runner_sha256","profiler_sha256","host","resource_interval_seconds","initial_model_state","ollama_version","reference_labels_used_for_inference"):
        need(a[k]==b[k],"Comparison mismatch: "+k)
    need({k:v for k,v in a["generation_settings"].items() if k!="model"}=={k:v for k,v in b["generation_settings"].items() if k!="model"},"Non-model settings differ")
    need(not a["reference_labels_used_for_inference"],"Reference leakage")

def compare(paths):
    need(set(paths)==set(MODELS),"Require three model runs")
    results={k:load_profile(paths[k],k) for k in MODELS}
    a,x,ar=results["llama3b"];ref=check_reference()
    need(a["model_digest"]==ref["model_digest"] and a["reference_manifest_sha256"]==sha(RUN/"manifest.json"),"Control reference mismatch")
    _,_,qs=_cohort();expected=[(q.query_id,q.query) for q in qs]
    for m,s,rs in results.values():
        comparable(a,m)
        need([(r["query_id"],r["query"]) for r in rs]==expected,"Dataset alignment")
    common=set.intersection(*[{r["query_id"] for r in rs if r["status"]=="ok"} for _,_,rs in results.values()])
    summary=[];pair={}
    for k,(m,s,rs) in results.items():
        common_lat=[r["latency_ms"] for r in rs if r["query_id"] in common]
        summary.append(dict(**s,common_answered_n=len(common),common_answered_mean_ms=sum(common_lat)/len(common_lat) if common_lat else None,ai_hallucination_rate=None,strict_supported_grounded_yield=None))
        if k!="llama3b":
            p=paired(ar,rs);p["exact_answer_equal_all"]=sum(r["answer"]==t["answer"] for r,t in zip(ar,rs));p["changed_answers"]=132-p["exact_answer_equal_all"];pair[k]=p
    dest=OUT/("comparison_"+_stamp());dest.mkdir(parents=True,exist_ok=False)
    writecsv(dest/"generator_summary_132.csv",summary)
    writecsv(dest/"generator_per_query_132.csv",[dict(model_key=k,**r) for k,(_,_,rs) in results.items() for r in rs])
    writecsv(dest/"generator_by_query_type_132.csv",[dict(model_key=k,query_type=q,**metrics([r for r in rs if r["query_type"]==q])) for k,(_,_,rs) in results.items() for q in sorted({r["query_type"] for r in rs})])
    dump(dest/"paired_summary.json",pair)
    dump(dest/"manifest.json",dict(schema="B9_GENERATOR_COMPARISON_V1",records=396,run_manifest_sha256={k:sha(Path(paths[k])/"manifest.json") for k in MODELS},
        limitations=["Same controller, compact IDs and evidence snapshot; model-driven feedback can change later evidence sets.","Token counts depend on model tokenizer; not identical token units across families.","Separate single-run measurements, no controlled OS memory/thermal state. System RAM and CPU include unrelated processes; RSS sum can miss restricted processes or double count shared pages.","All quality metrics require fresh answer-specific review; no gold or clinical accuracy claimed."],outputs_sha256={p.name:sha(p) for p in sorted(dest.iterdir())}))
    print(f"{'Model':<14}{'Answers':>9}{'Mean s':>11}{'P95 s':>11}{'QPS':>10}{'Peak RAM MiB':>16}{'Common n':>11}")
    for r in summary:print(f"{r['model_key']:<14}{r['answered']:>9}{r['mean_latency_ms']/1000:>11.3f}{r['p95_latency_ms']/1000:>11.3f}{r['throughput_qps']:>10.4f}{r['peak_sampled_system_ram_used_kib']/1024:>16.1f}{r['common_answered_n']:>11}")
    print("Output:",dest);return dest

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("command",choices=("run","compare"));p.add_argument("--model",choices=MODELS);p.add_argument("--allow-live",action="store_true");p.add_argument("--smoke",action="store_true")
    for k in MODELS:p.add_argument("--"+k+"-dir",type=Path)
    a=p.parse_args()
    if a.command=="run":
        if not a.model:p.error("--model required")
        run(a.model,a.allow_live,a.smoke)
    else:
        paths={k:getattr(a,k+"_dir") for k in MODELS}
        if not all(paths.values()):p.error("All three directories required")
        compare(paths)
if __name__=="__main__":main()
