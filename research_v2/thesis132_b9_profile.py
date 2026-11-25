"""Frozen B9-v2 inference profiling and compact evidence-ID optimization.
Linux /proc resource sampling; actual Ollama response telemetry. No clinical labels.
"""
from __future__ import annotations
import argparse,json,math,os,statistics,subprocess,threading,time
from collections import Counter
from pathlib import Path
import requests
from research_v2.thesis132_b9_v2 import (
    B9V2Controller,ROOT,GRAPH,CORPUS,fingerprints,generation_settings,check_ollama,
    _cohort,_stamp,G1TypedIndex,PipelineRequest,parse_selection,sha,dump)
from research_v2.thesis132_b9_review import RUN,writecsv,readcsv
from research_v2.thesis132_b9_components import row,annotate
from research_v2.thesis132_b9_ablation import metrics,paired
from research_v2.pipelines import create_pipeline

OUT=ROOT/"presentation_20261118/thesis132/b9_inference_optimization_v2"
VARIANTS=("original_ids","compact_ids")
def need(ok,msg):
    if not ok:raise ValueError(msg)
def mean(x):return statistics.mean(x) if x else None

def compact_prompt(prompt):
    head,sep,body=prompt.partition("\n")
    need(bool(sep),"Unexpected selector prompt")
    obj=json.loads(body)
    mapping={"E"+str(i+1):d["id"] for i,d in enumerate(obj["items"])}
    reverse={v:k for k,v in mapping.items()}
    for d in obj["items"]:d["id"]=reverse[d["id"]]
    obj["groups"]={k:[reverse[x] for x in v] for k,v in obj["groups"].items()}
    return head+"\n"+json.dumps(obj,ensure_ascii=False),mapping

def restore_selection(raw,mapping):
    # Never promote malformed output to valid selection.
    ids=parse_selection(raw,mapping,len(mapping))
    if ids is None:return raw if parse_selection(raw,mapping.values(),len(mapping)) is None else "INVALID_COMPACT_SELECTION"
    return json.dumps({"selected_ids":[mapping[x] for x in ids]})

def call_model(prompt,cfg):
    """Same payload/retry semantics as the frozen generator, with telemetry."""
    payload=dict(model=cfg["model"],prompt=prompt,stream=False,keep_alive=cfg["keep_alive"],
        options=dict(temperature=cfg["temperature"],num_predict=cfg["num_predict"],num_ctx=cfg["num_ctx"]))
    last=None;start=time.perf_counter()
    for i in range(cfg["attempts"]):
        try:
            resp=requests.post(cfg["endpoint"]+"/api/generate",json=payload,timeout=cfg["timeout_seconds"])
            resp.raise_for_status()
            result=resp.json();raw=result.get("response")
            if not isinstance(raw,str) or not raw.strip():raise RuntimeError("empty_generation_response")
            tel={k:result.get(k) for k in ("total_duration","load_duration","prompt_eval_count","prompt_eval_duration","eval_count","eval_duration","done_reason")}
            tel.update(request_wall_ms=(time.perf_counter()-start)*1000,http_attempts=i+1,prompt_chars=len(prompt),response_chars=len(raw))
            return raw,tel
        except (requests.exceptions.ConnectionError,requests.exceptions.Timeout) as e:last=e
        except requests.exceptions.HTTPError as e:
            if e.response is None or e.response.status_code not in {429,500,502,503,504}:raise
            last=e
        except (ValueError,KeyError) as e:raise RuntimeError("invalid_ollama_json") from e
        if i+1<cfg["attempts"]:time.sleep(min(2**i,5))
    raise RuntimeError("ollama_unavailable_after_retries") from last

class Selector:
    def __init__(self,variant,cfg,call=call_model):
        self.variant=variant;self.cfg=cfg;self.call=call;self.query_id=None;self.events=[]
    def __call__(self,prompt):
        sent,mapping=compact_prompt(prompt) if self.variant=="compact_ids" else (prompt,None)
        raw,tel=self.call(sent,self.cfg)
        self.events.append(dict(query_id=self.query_id,variant=self.variant,raw_response=raw,
            mapping=mapping,**tel))
        return restore_selection(raw,mapping) if mapping is not None else raw

def meminfo():
    data={}
    for line in Path("/proc/meminfo").read_text().splitlines():
        k,v=line.split(":",1);data[k]=int(v.split()[0])
    return data
def cpu_ticks():
    a=list(map(int,Path("/proc/stat").read_text().splitlines()[0].split()[1:9]))
    return sum(a),a[3]+a[4]
def rss(pid):
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):return int(line.split()[1])
    except (OSError,ValueError):pass
    return 0
def ollama_rss():
    # Sum RSS (may double-count shared pages). Include server and runner names.
    total=0;n=0
    for p in Path("/proc").iterdir():
        if not p.name.isdigit():continue
        try:
            comm=(p/"comm").read_text().strip().lower()
            cmd=(p/"cmdline").read_bytes().split(b"\0")
            exe=Path(os.fsdecode(cmd[0])).name.lower() if cmd and cmd[0] else ""
            if comm.startswith("ollama") or exe.startswith("ollama"):
                total+=rss(p.name);n+=1
        except (OSError,ValueError):continue
    return total,n

class Monitor:
    def __init__(self,interval=.25):
        self.interval=interval;self.stop=threading.Event();self.samples=[];self.error=None
        self.start_time=time.perf_counter()
        self.thread=threading.Thread(target=self.loop,daemon=True)
    def sample(self):
        m=meminfo();ticks,idle=cpu_ticks();o,n=ollama_rss()
        prev=self.samples[-1] if self.samples else None
        cpu=None
        if prev and ticks>prev["cpu_ticks"]:
            cpu=100*(1-(idle-prev["idle_ticks"])/(ticks-prev["cpu_ticks"]))
        self.samples.append(dict(seconds=time.perf_counter()-self.start_time,
            system_ram_used_kib=m["MemTotal"]-m["MemAvailable"],system_ram_total_kib=m["MemTotal"],
            swap_used_kib=m["SwapTotal"]-m["SwapFree"],python_rss_kib=rss(os.getpid()),
            ollama_sum_rss_kib=o,ollama_processes=n,cpu_ticks=ticks,idle_ticks=idle,system_cpu_percent=cpu))
    def loop(self):
        try:
            while not self.stop.wait(self.interval):self.sample()
        except Exception as e:self.error=repr(e)
    def __enter__(self):
        self.sample();self.thread.start();return self
    def __exit__(self,*args):
        self.stop.set();self.thread.join();self.sample()
        if self.error:raise RuntimeError("Resource sampler failed: "+self.error)

def profile_summary(events,samples):
    result=dict(selector_calls=len(events),resource_samples=len(samples))
    for k in ("load_duration","prompt_eval_duration","eval_duration","total_duration"):
        values=[e[k]/1e6 for e in events if isinstance(e.get(k),(int,float))]
        result["mean_"+k+"_ms"]=mean(values);result[k+"_observations"]=len(values)
    for k in ("prompt_eval_count","eval_count","prompt_chars","response_chars","request_wall_ms"):
        values=[e[k] for e in events if isinstance(e.get(k),(int,float))]
        result["mean_"+k]=mean(values);result[k+"_observations"]=len(values)
    tokens=sum(e["eval_count"] for e in events if isinstance(e.get("eval_count"),int) and isinstance(e.get("eval_duration"),int) and e["eval_duration"]>0)
    seconds=sum(e["eval_duration"]/1e9 for e in events if isinstance(e.get("eval_count"),int) and isinstance(e.get("eval_duration"),int) and e["eval_duration"]>0)
    result["decode_tokens_per_second"]=tokens/seconds if seconds else None
    result["length_stop_count"]=sum(e.get("done_reason")=="length" for e in events)
    result["total_http_attempts"]=sum(e["http_attempts"] for e in events)
    for k in ("system_ram_used_kib","swap_used_kib","python_rss_kib","ollama_sum_rss_kib"):
        result["start_"+k]=samples[0][k];result["peak_sampled_"+k]=max(s[k] for s in samples)
    total=samples[-1]["cpu_ticks"]-samples[0]["cpu_ticks"]
    idle=samples[-1]["idle_ticks"]-samples[0]["idle_ticks"]
    result["system_cpu_percent_time_weighted"]=100*(1-idle/total) if total>0 else None
    return result

def run(variant,allow_live=False):
    need(allow_live,"Explicit --allow-live required")
    need(variant in VARIANTS,"Invalid variant")
    ref=json.loads((RUN/"manifest.json").read_text());cfg=generation_settings()
    need(ref["status"]=="COMPLETE_132" and fingerprints(GRAPH)==ref["inputs"],"Frozen reference changed")
    need(cfg==ref["generation_settings"],"Generation settings differ from archived B9")
    check_ollama(cfg)
    tags=requests.get(cfg["endpoint"]+"/api/tags",timeout=10);tags.raise_for_status()
    digest=next((m["digest"] for m in tags.json()["models"] if m["name"]==cfg["model"]),None)
    need(digest==ref["model_digest"],"Model digest changed")
    _,source,qs=_cohort();need(len(qs)==132 and sha(source)==ref["dataset_sha256"],"Dataset changed")
    # Explicitly unload before each profiled run to start from comparable cold model state.
    res=requests.post(cfg["endpoint"]+"/api/generate",json={"model":cfg["model"],"keep_alive":0},timeout=cfg["timeout_seconds"]);res.raise_for_status()
    selector=Selector(variant,cfg)
    c=B9V2Controller(G1TypedIndex(GRAPH),create_pipeline("B2",document_corpus=CORPUS,top_k=5,generate=False),selector,policy="adaptive",top_k=5,model=cfg["model"])
    dest=OUT/(variant+"_"+_stamp());dest.mkdir(parents=True,exist_ok=False)
    selfsha=sha(Path(__file__))
    manifest=dict(schema="B9_INFERENCE_PROFILE_V1",variant=variant,status="RUNNING",questions=132,
        reference_manifest_sha256=sha(RUN/"manifest.json"),inputs=ref["inputs"],dataset_sha256=sha(source),
        generation_settings=cfg,model_digest=digest,profiler_sha256=selfsha,host=dict(platform=os.uname().sysname,
        machine=os.uname().machine,cpu_count=os.cpu_count(),mem_total_kib=meminfo()["MemTotal"]),
        code_commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
        resource_interval_seconds=.25,cold_model_unload_requested=True,reference_labels_used_for_inference=False)
    dump(dest/"manifest.json",manifest)
    rows=[]
    wall=time.perf_counter()
    with Monitor() as monitor:
        with (dest/"records.jsonl").open("w",encoding="utf-8") as rf,(dest/"round_traces.jsonl").open("w",encoding="utf-8") as tf,(dest/"selector_telemetry.jsonl").open("w",encoding="utf-8") as ef:
            for i,q in enumerate(qs,1):
                selector.query_id=q.query_id;before=len(selector.events)
                r,t=c.run(PipelineRequest(run_id=dest.name,query_id=q.query_id,query=q.query,graph_version="G1_CANDIDATE"))
                rf.write(r.model_dump_json()+"\n");rf.flush()
                tf.write(json.dumps(t,ensure_ascii=False)+"\n");tf.flush()
                for e in selector.events[before:]:ef.write(json.dumps(e,ensure_ascii=False)+"\n")
                ef.flush();rows.append(row(r,t))
                print(f"[{i:03d}/132] {variant} {q.query_id} {r.status} {r.timings.total_ms/1000:.3f}s",flush=True)
    seconds=time.perf_counter()-wall
    need(fingerprints(GRAPH)==ref["inputs"] and sha(Path(__file__))==selfsha,"Inputs changed during run")
    rows=annotate(rows);writecsv(dest/"per_query.csv",rows);writecsv(dest/"resource_samples.csv",monitor.samples)
    errors=sum(r["status"]=="error" for r in rows)
    summary=dict(variant=variant,errors=errors,**metrics(rows,seconds),**{k:v for k,v in profile_summary(selector.events,monitor.samples).items() if k!="selector_calls"})
    dump(dest/"summary.json",summary)
    manifest["status"]="COMPLETE_WITH_ERRORS" if errors else "COMPLETE_132"
    manifest["outputs_sha256"]={p.name:sha(p) for p in sorted(dest.iterdir()) if p.name!="manifest.json"}
    dump(dest/"manifest.json",manifest)
    print(json.dumps({"output":str(dest),**summary},indent=2))
    if errors:raise SystemExit(1)
    return dest

def load_profile(path,variant):
    path=Path(path).resolve();m=json.loads((path/"manifest.json").read_text())
    need(m["schema"]=="B9_INFERENCE_PROFILE_V1" and m["variant"]==variant and m["status"]=="COMPLETE_132" and m["questions"]==132,"Invalid completed profile")
    for k,v in m["outputs_sha256"].items():need(Path(k).name==k and sha(path/k)==v,"Changed profile file "+k)
    s=json.loads((path/"summary.json").read_text())
    from research_v2.schema import PipelineRecord
    rs=[PipelineRecord.model_validate_json(l) for l in (path/"records.jsonl").read_text().splitlines()]
    ts=[json.loads(l) for l in (path/"round_traces.jsonl").read_text().splitlines()]
    need(len(rs)==len(ts)==132 and all(r.query_id==t["query_id"] for r,t in zip(rs,ts)),"Record alignment")
    rows=annotate([row(r,t) for r,t in zip(rs,ts)])
    need(all(r["status"] in ("ok","abstained") and bool(r["answer"])==(r["status"]=="ok") for r in rows),"Bad statuses")
    events=[json.loads(l) for l in (path/"selector_telemetry.jsonl").read_text().splitlines()]
    need(len(events)==sum(r["selector_calls"] for r in rows),"Missing selector telemetry")
    need(Counter(e["query_id"] for e in events)==Counter({r["query_id"]:r["selector_calls"] for r in rows if r["selector_calls"]}),"Telemetry identity mismatch")
    samples=readcsv(path/"resource_samples.csv")
    for z in samples:
        for k,v in z.items():z[k]=float(v) if v!="" else None
    result=dict(variant=variant,**metrics(rows,s["wall_seconds"]),**{k:v for k,v in profile_summary(events,samples).items() if k!="selector_calls"})
    need(s["errors"]==0 and result["answered"]==s["answered"],"Summary mismatch")
    return m,result,rows

def compare(original,compact):
    a,x,ar=load_profile(original,"original_ids");b,y,br=load_profile(compact,"compact_ids")
    for k in ("reference_manifest_sha256","inputs","dataset_sha256","generation_settings","model_digest","profiler_sha256","host","resource_interval_seconds","cold_model_unload_requested"):
        need(a[k]==b[k],"Unmatched profiles: "+k)
    _,_,qs=_cohort()
    expected=[(q.query_id,q.query) for q in qs]
    need([(r["query_id"],r["query"]) for r in ar]==[(r["query_id"],r["query"]) for r in br]==expected,"Different cohort")
    p=paired(ar,br)
    p["exact_answer_equal_all"]=sum(r["answer"]==s["answer"] for r,s in zip(ar,br))
    p["changed_answers"]=132-p["exact_answer_equal_all"]
    p["mean_latency_reduction_fraction"]=1-y["mean_latency_ms"]/x["mean_latency_ms"] if x["mean_latency_ms"] else None
    p["quality_not_evaluated"]=True
    dest=OUT/("comparison_"+_stamp());dest.mkdir(parents=True,exist_ok=False)
    writecsv(dest/"inference_comparison_132.csv",[x,y])
    writecsv(dest/"paired_inference_132.csv",[dict(query_id=r["query_id"],query=r["query"],original_status=r["status"],compact_status=s["status"],original_answer=r["answer"],compact_answer=s["answer"],identical=r["answer"]==s["answer"],original_latency_ms=r["latency_ms"],compact_latency_ms=s["latency_ms"]) for r,s in zip(ar,br)])
    dump(dest/"paired_summary.json",p)
    dump(dest/"manifest.json",dict(schema="B9_INFERENCE_COMPARISON_V1",original_sha256=sha(Path(original)/"manifest.json"),compact_sha256=sha(Path(compact)/"manifest.json"),
        limitations=["Single cold-start request per run, not verified identical thermal/load state; repeat for variability.","Compact IDs change selector prompts, not just runtime; changed answers require fresh review.","Ollama durations are server-reported; system CPU and RAM include other processes. Summed RSS can double-count shared pages; sampled peaks are lower bounds and not energy measurements.","Sampling overhead is included in both wall times. No medical gold or quality improvement claimed."],
        outputs_sha256={p.name:sha(p) for p in sorted(dest.iterdir())}))
    print(f"{'Variant':<16}{'Answers':>9}{'Mean s':>10}{'QPS':>10}{'Decode tok/s':>15}{'Peak RAM MiB':>16}")
    for r in (x,y):print(f"{r['variant']:<16}{r['answered']:>9}{r['mean_latency_ms']/1000:>10.3f}{r['throughput_qps']:>10.4f}{str(round(r['decode_tokens_per_second'],2) if r['decode_tokens_per_second'] is not None else 'N/A'):>15}{r['peak_sampled_system_ram_used_kib']/1024:>16.1f}")
    print(json.dumps(p,indent=2));print("Output:",dest)
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("command",choices=("run","compare"));p.add_argument("--variant",choices=VARIANTS);p.add_argument("--allow-live",action="store_true");p.add_argument("--original-dir",type=Path);p.add_argument("--compact-dir",type=Path);a=p.parse_args()
    if a.command=="run":
        if not a.variant:p.error("--variant required")
        run(a.variant,a.allow_live)
    else:
        if not a.original_dir or not a.compact_dir:p.error("Both profile directories required")
        compare(a.original_dir,a.compact_dir)
if __name__=="__main__":main()
