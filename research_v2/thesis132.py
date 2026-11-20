"""Frozen 132-question MTP thesis experiment runner.

Existing historical runs remain unmodified. Freeze from v2 (AI screened,
NOT clinical gold) with query-only inference data; create deterministic
family-grouped folds for *prospective* evaluations, without claiming the
questions were unseen during previous G1 development.

run: B1-B6 (retrieval-only / optional shared generator) and B7/B8 (live
Neo4j+Ollama/PPO end-to-end, with runtime preflight and graph drift check).
Runs are sequential and versioned. Errors are recorded, never relabeled
as abstentions. Summaries separate output availability from correctness.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from .benchmark_v2_validation import validate as validate_benchmark, DATASET
from .experiment_runner import PROJECT_ROOT as ROOT, file_sha256, load_questions

BASE=ROOT/"presentation_20261118/thesis132"
CORPUS=ROOT/"presentation_20261118/corpus/provisional_source_summaries_v1/documents.jsonl"
MODELS=tuple(f"B{i}" for i in range(1,9))
LEGACY={"B7","B8"}
FOLDS=5

def _write_json(path, obj):
    path.write_text(json.dumps(obj,indent=2,sort_keys=True,ensure_ascii=False)+"\n",encoding="utf-8")

def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def _inside(p):
    p=Path(p).expanduser().resolve()
    if not p.is_relative_to(ROOT):
        raise ValueError("Input/output must remain in this checkout")
    return p

def _stamp():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")+"_"+uuid.uuid4().hex[:6]

def _grouped_folds(rows):
    """Stable group disjointness; future-only CV, not retrospective independence."""
    groups=defaultdict(list)
    for item in rows:
        groups[item["semantic_group"]].append(item)
    # Balance large groups before small ones, using reproducible SHA ordering.
    def key(group):
        items=groups[group]
        sha=hashlib.sha256(group.encode()).hexdigest()
        return (-len(items),sha)
    counts=[0]*FOLDS
    folds={}
    for group in sorted(groups,key=key):
        i=min(range(FOLDS),key=lambda n:(counts[n],n))
        counts[i]+=len(groups[group])
        folds[group]=i
    result={x["query_id"]:folds[x["semantic_group"]] for x in rows}
    if set(result)!={x["query_id"] for x in rows} or set(result.values())!=set(range(FOLDS)):
        raise ValueError("Invalid five-fold assignment")
    for group,items in groups.items():
        if len({result[r["query_id"]] for r in items})!=1:
            raise ValueError("Group leakage detected: "+group)
    return result,counts

def freeze():
    report,rows,_=validate_benchmark()
    if len(rows)!=132 or report["clinical_gold_questions"]!=0:
        raise ValueError("Invalid canonical 132-question development cohort")
    folds,fold_counts=_grouped_folds(rows)
    digest=file_sha256(DATASET)
    directory=BASE/("frozen_"+digest[:16])
    queries="".join(json.dumps({"query_id":r["query_id"],"query":r["query"]},
        ensure_ascii=False,separators=(",",":"))+"\n" for r in rows)
    data={
        "schema_version":"THESIS132_FREEZE_V1",
        "source_dataset_sha256":digest,
        "source_dataset_role":"source_screened_development_not_independent_medical_gold",
        "questions":132,
        "question_ids":[r["query_id"] for r in rows],
        "semantic_groups":len(set(r["semantic_group"] for r in rows)),
        "fold_counts":fold_counts,
        "folds":FOLDS,
        "historical_g1_exposure":True,
        "independent_clinical_gold":False,
        "fold_strategy":"prospective family-grouped five-fold allocation after prior G1 exposure",
        "warning":"These folds do NOT retroactively erase prior exposure of the 132 questions to G1 design.",
    }
    # Freeze is idempotent and refuses any silent replacement.
    if directory.exists():
        input_file=directory/"questions_only.jsonl"
        metadata=directory/"manifest.json"
        if (not input_file.is_file() or not metadata.is_file()
                or input_file.read_text(encoding="utf-8")!=queries
                or _read_json(metadata)!=data):
            raise ValueError("Frozen cohort already exists with different contents")
    else:
        directory.mkdir(parents=True,exist_ok=False)
        (directory/"questions_only.jsonl").write_text(queries,encoding="utf-8")
        with (directory/"fold_metadata.csv").open("w",encoding="utf-8",newline="") as f:
            columns=("query_id","cohort","query_type","medical_domain","semantic_group","fold")
            writer=csv.DictWriter(f,fieldnames=columns)
            writer.writeheader()
            for row in rows:
                writer.writerow({k:(folds[row["query_id"]] if k=="fold" else row[k]) for k in columns})
        _write_json(directory/"manifest.json",data)
    print(json.dumps({"status":"FROZEN_132_DEVELOPMENT","folder":str(directory),
                      "inference_sha256":file_sha256(directory/"questions_only.jsonl"),
                      "fold_sizes":fold_counts,"independent_medical_gold":False},indent=2))
    return directory

def _cohort():
    frozen=freeze()
    data=_read_json(frozen/"manifest.json")
    input_file=frozen/"questions_only.jsonl"
    records=load_questions(input_file)
    if len(records)!=132 or [x.query_id for x in records]!=data["question_ids"]:
        raise ValueError("Frozen query IDs don't match source")
    return frozen,input_file,records

def _selected(models):
    ids=MODELS if models=="all" else tuple(x.strip() for x in models.split(","))
    if not ids or len(set(ids))!=len(ids) or not set(ids).issubset(MODELS):
        raise ValueError("Choose unique pipeline IDs from B1,...,B8")
    return ids

def run(models,mode,encoder,limit,top_k,allow_live,session_dir=None):
    if mode not in {"retrieval_only","end_to_end"}:
        raise ValueError("Choose retrieval_only or end_to_end")
    if encoder not in {"sentence","lsa"}:
        raise ValueError("Unsupported encoder")
    if top_k<1:
        raise ValueError("top_k must be positive")
    requested=_selected(models)
    if mode=="retrieval_only" and any(m in LEGACY for m in requested):
        raise ValueError("B7/B8 legacy pipeline only supports full end-to-end execution")
    if mode=="end_to_end" and not allow_live:
        raise ValueError("End-to-end LLM/Neo4j/PPO runs require --allow-live")
    frozen,input_file,questions=_cohort()
    if limit is not None:
        if not 1<=limit<=132:
            raise ValueError("Limit must be 1..132")
        questions=questions[:limit]
    snapshot={
        "schema":"THESIS132_SESSION_V1","dataset_path":str(input_file.relative_to(ROOT)),
        "dataset_sha256":file_sha256(input_file),
        "query_count":len(questions),"mode":mode,"encoder":encoder,"top_k":top_k,
        "graph_version":"G0",
        "evidence_bases_differ":True,
        "medical_gold":False,"runs":{}
    }
    if session_dir is None:
        work=BASE/"runs"/("batch_"+_stamp())
        work.mkdir(parents=True,exist_ok=False)
        _write_json(work/"session.json",snapshot)
    else:
        work=_inside(session_dir)
        if not work.is_dir():
            raise FileNotFoundError(work)
        old=_read_json(work/"session.json")
        for key in ("dataset_sha256","query_count","mode","encoder","top_k"):
            if old.get(key)!=snapshot[key]:
                raise ValueError("Resume session does not match "+key)
        snapshot=old
    if any(m in {"B1","B2","B3","B4"} for m in requested):
        if not CORPUS.is_file():
            raise FileNotFoundError("Frozen candidate corpus missing: "+str(CORPUS))
    if mode=="end_to_end" and any(m not in LEGACY for m in requested):
        from .ollama_generation import check_ollama
        check_ollama()
    for model in requested:
        if model in snapshot["runs"]:
            prior=snapshot["runs"][model]
            original_dir=work/prior["folder"]
            if not original_dir.is_dir() or not (original_dir/"summary.json").is_file():
                raise ValueError("Session references missing model output "+model)
            old_summary=_read_json(original_dir/"summary.json")
            if (old_summary.get("errors") or old_summary.get("aborted")
                    or old_summary.get("total_queries")!=len(questions)):
                raise ValueError("Prior run has errors/incomplete; create a new session for rerun: "+model)
            print("Skipping completed model",model,prior["folder"])
            continue
        name="THESIS132_"+model+"_"+_stamp()
        if model in LEGACY:
            # Enforces original G0 runtime lock and checks Neo4j graph drift
            # before/after each question through configured experiment runner.
            from .configured_experiment import run_configured_experiment
            cfg={
                "schema_version":1,
                "experiment_name":name,
                "pipeline":"no_rl" if model=="B7" else "rl",
                "dataset_path":str(input_file.relative_to(ROOT)),
                "dataset_sha256":file_sha256(input_file),
                "graph_version":"G0",
                "random_seed":2026,
                "limit":len(questions),
                "purpose":"Fixed 132-question legacy end-to-end research evaluation; no clinical gold.",
            }
            cfg_dir=work/"configs"
            cfg_dir.mkdir(exist_ok=True)
            cfg_path=cfg_dir/(name+".json")
            _write_json(cfg_path,cfg)
            summary,out=run_configured_experiment(
                config_path=cfg_path,output_root=work/"records",run_id=name)
        else:
            from .pipelines import create_pipeline
            from .experiment_runner import run_experiment
            pipe=create_pipeline(model,
                document_corpus=CORPUS if model in {"B1","B2","B3","B4"} else None,
                top_k=top_k,generate=(mode=="end_to_end"),encoder=encoder)
            summary,out=run_experiment(
                pipeline=pipe,questions=questions,run_id=name,
                source_file=input_file,output_root=work/"records",
                graph_version="G0",random_seed=None)
        snapshot["runs"][model]={
            "folder":str(out.relative_to(work)),
            "run_id":name,"summary":summary,
            "record_sha256":file_sha256(out/"records.jsonl"),
        }
        _write_json(work/"session.json",snapshot)
        print(json.dumps({"model":model,"summary":summary,"session":str(work)},indent=2))
        if summary["errors"] or summary["aborted"] or summary["total_queries"]!=len(questions):
            raise RuntimeError("Model "+model+" has a recorded error/incomplete run; see "+str(out))
    print(json.dumps({"session":str(work),"complete_models":sorted(snapshot["runs"]),
                      "scope":"132" if len(questions)==132 else "SMOKE_ONLY",
                      "accuracy_gold":False},indent=2))
    return work

def summarize(session_dir):
    work=_inside(session_dir)
    meta=_read_json(work/"session.json")
    if meta["schema"]!="THESIS132_SESSION_V1":
        raise ValueError("Wrong experiment session")
    frozen,input_file,questions=_cohort()
    if meta["dataset_sha256"]!=file_sha256(input_file):
        raise ValueError("Input changed")
    subset=questions[:meta["query_count"]]
    ids=[r.query_id for r in subset]
    from .schema import PipelineRecord
    rows=[]
    for model,info in sorted(meta["runs"].items()):
        folder=_inside(work/info["folder"])
        data=[PipelineRecord.model_validate_json(x) for x in
              (folder/"records.jsonl").read_text(encoding="utf-8").splitlines()
              if x.strip()]
        if file_sha256(folder/"records.jsonl")!=info["record_sha256"]:
            raise ValueError("Archived result changed: "+model)
        if len(data)!=len(ids) or [x.query_id for x in data]!=ids:
            raise ValueError("Question order/count mismatch: "+model)
        vals=[x.timings.total_ms for x in data if x.timings.total_ms is not None]
        yield_count=sum(bool(x.retrieved_documents or x.causal_edges or x.causal_paths) for x in data)
        def mean(values):
            values=[v for v in values if v is not None]
            return statistics.mean(values) if values else None
        def quantile(values,q):
            if not values:return None
            sorted_vals=sorted(values)
            pos=(len(sorted_vals)-1)*q
            lo=int(pos);hi=min(lo+1,len(sorted_vals)-1)
            return sorted_vals[lo]+(pos-lo)*(sorted_vals[hi]-sorted_vals[lo])
        summary=_read_json(folder/"summary.json")
        rows.append({
            "model":model,"mode":meta["mode"],"n":len(data),
            "ok":sum(x.status=="ok" for x in data),
            "abstained":sum(x.status=="abstained" for x in data),
            "errors":sum(x.status=="error" for x in data),
            "evidence_returned":yield_count,
            "answer_present":sum(bool(x.final_answer) for x in data),
            "mean_total_ms":mean(vals),"p50_total_ms":quantile(vals,.5),
            "p95_total_ms":quantile(vals,.95),
            "mean_retrieval_ms":mean(x.timings.retrieval_ms for x in data),
            "mean_generation_ms":mean(x.timings.generation_ms for x in data),
            "throughput_qps_with_init":len(data)/summary["wall_time_seconds"]
                if summary["wall_time_seconds"]>0 else None,
            "corpus_type":"candidate_documents" if model in {"B1","B2","B3","B4"}
                else "unverified_G0_graph",
            "clinical_accuracy":None,
            "run_id":info["run_id"],
        })
    output={
        "session":str(work.relative_to(ROOT)), "mode":meta["mode"],
        "question_count":meta["query_count"],
        "source_dataset_sha256":meta["dataset_sha256"],
        "models":rows,"all_8_e2e_comparable_execution_mode":
            meta["mode"]=="end_to_end" and
            {r["model"] for r in rows}==set(MODELS),
        "common_evidence_corpus":False,
        "independent_clinical_gold":False,
        "note":"Different evidence bases and retrieval implementations. Do not report evidence availability as medical correctness; no PPO-only causal attribution.",
    }
    _write_json(work/"summary_current.json",output)
    print(json.dumps(output,indent=2))
    return output

def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest="command",required=True)
    sub.add_parser("freeze")
    runp=sub.add_parser("run")
    runp.add_argument("--models",default="all")
    runp.add_argument("--mode",choices=("retrieval_only","end_to_end"),default="retrieval_only")
    runp.add_argument("--encoder",choices=("sentence","lsa"),default="sentence")
    runp.add_argument("--limit",type=int,default=None)
    runp.add_argument("--top-k",type=int,default=5)
    runp.add_argument("--allow-live",action="store_true")
    runp.add_argument("--session-dir",type=Path)
    sum_p=sub.add_parser("summarize")
    sum_p.add_argument("--session-dir",type=Path,required=True)
    a=p.parse_args()
    if Path.cwd().resolve()!=ROOT:
        p.error("Run from repository root")
    if a.command=="freeze":freeze()
    elif a.command=="run":run(a.models,a.mode,a.encoder,a.limit,a.top_k,a.allow_live,a.session_dir)
    else:summarize(a.session_dir)

if __name__=="__main__":
    main()
