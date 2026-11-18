"""Fast, provenance-aware partial-thesis submission for existing 100+32 queries.

COMMANDS:
  verify    Verify existing frozen 100-question B1-B8 archived runs.
  run32     Execute B1-B6 retrieval-only on the existing 32 expansion queries.
  assemble  Combine verified archived 100, optional completed 32, G0/G1
            shadow results and reviewed-error analysis into slide/report-ready
            Markdown+CSV+JSON; NEVER combine unlike evidence/latency modes.

Never launches Ollama, Neo4j, B7/B8, PPO training, tests, or network access.
No answer/medical accuracy is computed. Existing results are never overwritten.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from .experiment_runner import PROJECT_ROOT, file_sha256, load_questions, run_experiment
from .base_pipeline import PipelineRequest
from .pipelines import create_pipeline, MODEL_REGISTRY
from .presentation_export import _select_runs
from .presentation_matrix import EXPECTED_PIPELINE_IDS, validate_inputs
from .schema import PipelineRecord

ROOT=PROJECT_ROOT
WORKSPACE=ROOT/"presentation_20261118"
OUTPUT=WORKSPACE/"partial_submission_sprint"
Q100=WORKSPACE/"benchmark/questions_only.jsonl"
Q32=ROOT/"data/benchmark/medical_causal_expansion32_draft_v1.jsonl"
CORPUS=WORKSPACE/"corpus/provisional_source_summaries_v1/documents.jsonl"
G0=ROOT/"research_artifacts/graph-G0"
LEGACY22=WORKSPACE/"legacy_results/rl_vs_no_rl_comparison.csv"
STRUCTURAL20=WORKSPACE/"metrics/g0_structural_pilot_v1/structural_scores.json"
ANALYSIS=WORKSPACE/"g1_review_analysis/analysis_20261117T183906316580Z_5537c4"
STABLE=WORKSPACE/"g1_stable_retrieval/stable_20261117T165851832115Z_2a02a0"
MODEL100=tuple(f"B{i}" for i in range(1,9))
MODEL32=tuple(f"B{i}" for i in range(1,7))
SCHEMA=("model","pipeline","n","status_ok","status_abstained","status_error",
        "evidence_returned","answer_present","causal_path_queries","mean_total_ms",
        "mean_retrieval_ms","mode","evidence_basis","run_id")
HEADER32=("query_id","question","query_type","model_id","pipeline",
          "status","returned_evidence","matched_concept",
          "document_count","causal_edge_count","causal_path_count",
          "retrieval_ms","total_ms","run_id")


def _stamp(prefix):
    return prefix+"_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")+"_"+uuid.uuid4().hex[:6]


def _safe(path):
    value=Path(path).expanduser().resolve()
    if not value.is_relative_to(ROOT):
        raise ValueError("All inputs/outputs must stay in repository")
    return value


def _dump(path,value):
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write_rows(path,columns,rows):
    with path.open("w",encoding="utf-8",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _read_rows(path):
    with path.open(encoding="utf-8",newline="") as f:
        return list(csv.DictReader(f))


def _average(vals):
    return round(statistics.mean(vals),5) if vals else None


def _measured(records,model,mode,evidence_basis):
    n=len(records)
    return {
        "model":model,"pipeline":MODEL_REGISTRY[model].name,
        "n":n,
        "status_ok":sum(r.status=="ok" for r in records),
        "status_abstained":sum(r.status=="abstained" for r in records),
        "status_error":sum(r.status=="error" for r in records),
        "evidence_returned":sum(bool(r.retrieved_documents or r.causal_edges or r.causal_paths) for r in records),
        "answer_present":sum(bool(r.final_answer and r.final_answer.strip()) for r in records),
        "causal_path_queries":sum(bool(r.causal_paths) for r in records),
        "mean_total_ms":_average([r.timings.total_ms for r in records if r.timings.total_ms is not None]),
        "mean_retrieval_ms":_average([r.timings.retrieval_ms for r in records if r.timings.retrieval_ms is not None]),
        "mode":mode,"evidence_basis":evidence_basis,
        "run_id":records[0].run_id if records else "",
    }


def _cohort():
    questions,metadata=validate_inputs(Q100)
    if len(questions)!=100:
        raise ValueError("Frozen pilot should contain exactly 100 queries")
    originals=[q.query_id for q in questions]
    selected=_select_runs(WORKSPACE/"runs",originals,file_sha256(Q100))
    # The run journal records DECLARED execution mode. Do not infer it
    # from a null generator_model on all-error B2/B4 attempts.
    journal={}
    file=WORKSPACE/"matrix_runs.jsonl"
    if file.exists():
        for raw in file.read_text(encoding="utf-8").splitlines():
            if raw.strip():
                info=json.loads(raw)
                journal[info["run_id"]]=info
    summary=[]
    for model in MODEL100:
        if model not in selected:continue
        end_to_end,_,summary_path,records,info=selected[model]
        # presentation_export._select_runs returns a *summary.json file*.
        # Keep its parent as the archived run directory.
        folder=summary_path.parent
        if not (folder/"records.jsonl").is_file():
            raise FileNotFoundError("Missing archived records: "+str(folder/"records.jsonl"))
        if [r.query_id for r in records]!=originals:
            raise ValueError("100-query archive order differs for "+model)
        # A completed archive may still contain execution errors.
        # Preserve it, label errors, and perform separate retrieval-only repairs.
        run_id=records[0].run_id
        mode=journal.get(run_id,{}).get("execution_mode")
        if mode not in {"end_to_end","retrieval_only"}:
            # Incomplete journal: preserve uncertainty rather than asserting
            # that all-error end-to-end runs were retrieval-only.
            mode="unverified_execution_mode"
        basis="provisional_AI_assisted_source_summaries" if model in MODEL32[:4] else "G0_unverified_graph"
        item=_measured(records,model,mode,basis)
        item["run_dir"]=str(folder.relative_to(ROOT))
        item["source_file_sha256"]=file_sha256(folder/"records.jsonl")
        summary.append(item)
    return summary,metadata


def verify():
    rows,metadata=_cohort()
    result={
        "models_found":[r["model"] for r in rows],
        "missing_models":sorted(set(MODEL100)-{r["model"] for r in rows}),
        "question_count":100,
        "dataset_sha256":file_sha256(Q100),
        "total_archived_model_query_records":sum(r["n"] for r in rows),
        "per_model":rows,
        "all_B1_to_B8_100_slots_present":len(rows)==8,
        "error_free_archived_models":[r["model"] for r in rows if r["status_error"]==0],
        "archived_models_requiring_retrieval_repair":[r["model"] for r in rows if r["status_error"]>0],
        "archived_abstentions_by_model":{r["model"]:r["status_abstained"] for r in rows},
        "archive_error_count":sum(r["status_error"] for r in rows),
        "medical_accuracy":False,
        "warning":(
            "Archived slots and successful executions are counted separately; "
            "older failures remain preserved but selected newer runs may be error-free. "
            "Modes are taken from the experiment journal. Document-summary and "
            "G0 graph evidence bases differ, so retrieval yield is not medical accuracy."
        ),
    }
    print(json.dumps(result,indent=2))
    return result


def doctor(encoder="sentence"):
    """Smoke all six B1-B6 retrievers in isolation without Ollama or writes."""
    questions,_=validate_inputs(Q100)
    if not CORPUS.is_file():
        raise FileNotFoundError("Missing provisional document corpus: "+str(CORPUS))
    evidence_manifest=_read_json(CORPUS.parent/"manifest.json")
    if evidence_manifest.get("documents_sha256")!=file_sha256(CORPUS):
        raise ValueError("Provisional evidence corpus checksum mismatch")
    if encoder not in {"sentence","lsa"}:
        raise ValueError("Unsupported embedding backend")
    outcome=[]
    for model in MODEL32:
        try:
            pipeline=create_pipeline(
                model,document_corpus=CORPUS if model in MODEL32[:4] else None,
                top_k=5,generate=False,encoder=encoder)
            q=questions[0]
            record=pipeline.run(PipelineRequest(run_id="DOCTOR",query_id=q.query_id,
                 query=q.query,graph_version="G0"))
            outcome.append({"model":model,"status":record.status,
                "evidence":len(record.retrieved_documents),
                "error":record.error_message})
        except Exception as exc:
            outcome.append({"model":model,"status":"error",
                "evidence":None,"error":type(exc).__name__+": "+str(exc)})
    report={"mode":"one_query_retrieval_only_preflight",
            "encoder":encoder,"models":outcome,
            "all_error_free":all(row["status"]!="error" for row in outcome),
            "warning":"An abstention is a legitimate outcome on G0; a doctor preflight is not a full experiment."}
    print(json.dumps(report,indent=2))
    if not report["all_error_free"]:
        raise SystemExit("Fix reported pipeline/environment problem before a 100 or 32 run")
    return report


def _expansion():
    data=[json.loads(line) for line in Q32.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(data)!=32 or len({d["query_id"] for d in data})!=32:
        raise ValueError("Expected the fixed 32 unique expansion queries")
    questions=load_questions(Q32)
    if [q.query_id for q in questions]!=[r["query_id"] for r in data]:
        raise ValueError("Parsed question identity mismatch")
    return questions,{r["query_id"]:r["query_type"] for r in data}


def run_retrieval(encoder="sentence",models=MODEL32,scope="32",resume_dir=None):
    if encoder not in {"sentence","lsa"}:
        raise ValueError("Choose sentence or lsa explicitly")
    if scope=="32":
        questions,types=_expansion()
        source_file=Q32
    elif scope=="100":
        questions,metadata=validate_inputs(Q100)
        types={q.query_id:metadata[q.query_id]["query_type"] for q in questions}
        source_file=Q100
    else:
        raise ValueError("Select only frozen 100 or existing 32 query cohorts")
    expected_count=len(questions)
    archived,_=_cohort()
    if len(archived)!=8:
        raise ValueError("Verify archived 100-query B1-B8 results before the expansion")
    for model in models:
        if model not in MODEL32:
            raise ValueError("B7/B8 cannot be launched by retrieval-only run100/run32")
    if not models:
        raise ValueError("Select at least one B1-B6 model")
    if not CORPUS.is_file():
        raise FileNotFoundError("Frozen provisional B1-B4 corpus missing")
    corpus_manifest=_read_json(CORPUS.parent/"manifest.json")
    if corpus_manifest["documents_sha256"]!=file_sha256(CORPUS):
        raise ValueError("Provisional document corpus was modified")
    if not (G0/"edges.csv").is_file():
        raise FileNotFoundError("G0 snapshot missing")
    root=_safe(resume_dir) if resume_dir else OUTPUT/_stamp("run32" if scope=="32" else "run100")
    if resume_dir and not root.is_dir():
        raise FileNotFoundError("Resume directory does not exist: "+str(root))
    if not resume_dir:
        root.mkdir(parents=True,exist_ok=False)
    records_by_model={}
    expected_index={
        "mode":"same_existing_queries_retrieval_only",
        "scope":scope,
        "expected_count":expected_count,
        "models_requested":list(models),
        "input_file":str(source_file.relative_to(ROOT)),
        "input_sha256":file_sha256(source_file),
        "corpus_sha256":file_sha256(CORPUS),
        "G0_nodes_sha256":file_sha256(G0/"nodes.csv"),
        "G0_edges_sha256":file_sha256(G0/"edges.csv"),
        "dense_encoder":encoder,
        "top_k":5,
        "archived_100_question_sha256":file_sha256(Q100),
        "model_runs":{},
        "accuracy_gold":False,
        "warnings":[
            "B1-B4 use source-candidate summaries, B5-B6 use unverified G0 graph.",
            "Retrieval-only experimental runs are not directly comparable in generation latency to archived end-to-end runs.",
            "Both cohorts contain draft questions also used in G1 concept inventory and are not unseen held-out questions.",
            "An evidence-returned flag is not medical correctness.",
        ],
    }
    if resume_dir:
        index=_read_json(root/"run_index.json")
        frozen_fields=("scope","expected_count","input_sha256","corpus_sha256",
                       "G0_nodes_sha256","G0_edges_sha256","dense_encoder","top_k")
        for field in frozen_fields:
            if index.get(field)!=expected_index.get(field):
                raise ValueError("Resume experiment provenance mismatch: "+field)
        if set(index.get("models_requested",[]))!=set(models):
            raise ValueError("Resume requires same original model set")
    else:
        index=expected_index
        _dump(root/"run_index.json",index)
    for model in models:
        previous=index["model_runs"].get(model)
        if previous:
            old_run=root/previous["folder"]
            if (old_run/"records.jsonl").is_file() and (
                    file_sha256(old_run/"records.jsonl")==previous["records_sha256"]
                    and file_sha256(old_run/"summary.json")==previous["summary_sha256"]):
                old_records=[PipelineRecord.model_validate_json(line)
                    for line in (old_run/"records.jsonl").read_text(encoding="utf-8").splitlines()
                    if line.strip()]
                if (len(old_records)==expected_count
                        and [r.query_id for r in old_records]==[q.query_id for q in questions]
                        and not any(r.status=="error" for r in old_records)):
                    print(json.dumps({"reused_verified_model":model,"run_dir":str(old_run)},indent=2))
                    records_by_model[model]=old_records
                    continue
            raise ValueError("Prior stored model output is invalid; preserve it and investigate: "+model)
        documents=CORPUS if model in MODEL32[:4] else None
        pipeline=create_pipeline(model,document_corpus=documents,
                                 top_k=5,generate=False,encoder=encoder)
        name=("EXT32_" if scope=="32" else "PILOT100_")+model+"_"+uuid.uuid4().hex[:8]
        def provenance(folder,mid=model):
            _dump(folder/"sprint_provenance.json",{
                "model":mid,"encoder":encoder,"generation":False,
                "query_sha256":file_sha256(source_file),
                "corpus_sha256":file_sha256(CORPUS) if mid in MODEL32[:4] else None,
                "graph_sha256":file_sha256(G0/"edges.csv") if mid in MODEL32[4:] else None,
                "top_k":5,
            })
        try:
            summary,run_dir=run_experiment(
                pipeline=pipeline,questions=questions,run_id=name,
                source_file=source_file,output_root=root/"runs",
                graph_version="G0",on_run_ready=provenance,
            )
        except Exception:
            # The experiment runner preserves individual records. Do not
            # pretend a failed model completed, and never overwrite runs.
            index["failed_model"]=model
            _dump(root/"run_index.json",index)
            raise
        if summary["aborted"] or summary["errors"] or summary["total_queries"]!=expected_count:
            raise RuntimeError("Incomplete "+scope+"-query retrieval: "+model+
                               "; inspect preserved "+str(run_dir))
        records=[PipelineRecord.model_validate_json(x)
                 for x in (run_dir/"records.jsonl").read_text(encoding="utf-8").splitlines()
                 if x.strip()]
        if len(records)!=expected_count or [r.query_id for r in records]!=[q.query_id for q in questions]:
            raise ValueError("Missing/misaligned "+scope+"-query results for "+model)
        records_by_model[model]=records
        index["model_runs"][model]={
            "folder":str(run_dir.relative_to(root)),
            "records_sha256":file_sha256(run_dir/"records.jsonl"),
            "summary_sha256":file_sha256(run_dir/"summary.json"),
            "execution_mode":"retrieval_only",
        }
        _dump(root/"run_index.json",index)
    rows=[]
    summary_rows=[]
    for model in models:
        rs=records_by_model[model]
        summary_rows.append(_measured(
            rs,model,"retrieval_only",
            "provisional_AI_assisted_source_summaries" if model in MODEL32[:4] else "G0_unverified_graph"))
        for r in rs:
            rows.append({
                "query_id":r.query_id,"question":r.original_query,
                "query_type":types[r.query_id],"model_id":model,
                "pipeline":r.pipeline_id,"status":r.status,
                "returned_evidence":bool(r.retrieved_documents or r.causal_edges or r.causal_paths),
                "matched_concept":r.matched_concept or "",
                "document_count":len(r.retrieved_documents),
                "causal_edge_count":len(r.causal_edges),
                "causal_path_count":len(r.causal_paths),
                "retrieval_ms":r.timings.retrieval_ms,
                "total_ms":r.timings.total_ms,
                "run_id":r.run_id,
            })
    _write_rows(root/(scope+"_query_model_records.csv"),HEADER32,rows)
    _write_rows(root/(scope+"_query_model_summary.csv"),SCHEMA,summary_rows)
    index["models_complete"]=len(summary_rows)
    index["model_query_records"]=len(rows)
    index["all_B1_B6_complete"]=len(summary_rows)==6 and len(rows)==6*expected_count
    _dump(root/"run_index.json",index)
    print(json.dumps({"output":str(root),"models_complete":len(summary_rows),
                     "rows":len(rows),"summary":summary_rows},indent=2))
    return root


def _verify_retrieval(folder,scope):
    folder=_safe(folder)
    ix=_read_json(folder/"run_index.json")
    if ix.get("scope")!=scope:raise ValueError("Retrieval run is from a different query cohort")
    input_file=Q32 if scope=="32" else Q100
    expected_count=32 if scope=="32" else 100
    if ix.get("input_sha256")!=file_sha256(input_file):
        raise ValueError("Frozen "+scope+"-query input changed after experiment")
    if ix.get("corpus_sha256")!=file_sha256(CORPUS) or ix.get("G0_edges_sha256")!=file_sha256(G0/"edges.csv"):
        raise ValueError("Retrieval evidence corpus changed after experiment")
    rows=_read_rows(folder/(scope+"_query_model_summary.csv"))
    if len(rows)!=6 or {r["model"] for r in rows}!=set(MODEL32):
        raise ValueError("Need six complete B1-B6 retrieval runs")
    if any(int(r["n"])!=expected_count or int(r["status_error"])!=0 for r in rows):
        raise ValueError(scope+"-query retrieval run incomplete or contains errors")
    if len(_read_rows(folder/(scope+"_query_model_records.csv")))!=expected_count*6:
        raise ValueError("Wrong number of B1-B6 retrieval records")
    for model,data in ix["model_runs"].items():
        p=folder/data["folder"]
        if file_sha256(p/"records.jsonl")!=data["records_sha256"]:
            raise ValueError("Run records changed: "+model)
    return rows,ix


def _resolve_summary_dir(requested, parent, prefix, kind):
    """Discover a versioned summary only when the frozen default is absent."""
    requested=_safe(requested)
    if (requested/"summary.json").is_file():
        _safe_summary(requested/"summary.json",kind)
        return requested
    if requested != (STABLE if kind=="stable" else ANALYSIS).resolve():
        raise FileNotFoundError("Explicit "+kind+" result path is absent: "+str(requested))
    parent=_safe(parent)
    matches=[]
    for candidate in parent.glob(prefix+"*/summary.json"):
        try:
            item=_safe_summary(candidate,kind)
            if kind=="stable" and (
                    item.get("query_model_rows")!=264 or
                    len(item.get("per_set",{}))<2):
                continue
            if kind=="analysis" and (
                    item.get("reviewed_cases")!=31 or
                    item.get("reviewed_evidence_occurrences")!=132):
                continue
            matches.append(candidate.parent)
        except (OSError,KeyError,ValueError,TypeError,AttributeError):
            continue
    if matches:
        return sorted(matches,key=lambda folder:folder.name)[-1]
    raise FileNotFoundError("No valid "+kind+" summary found under "+str(parent))


def _combine_retrieval_132(run100_dir,run32_dir):
    """Combine exact query-model records; do not mix generation modes."""
    q100,_=validate_inputs(Q100)
    q32,_=_expansion()
    expected={q.query_id for q in [*q100,*q32]}
    if len(expected)!=132:
        raise ValueError("Frozen 100/32 question IDs must not overlap")
    rows=_read_rows(_safe(run100_dir)/"100_query_model_records.csv")+_read_rows(
        _safe(run32_dir)/"32_query_model_records.csv"
    )
    if len(rows)!=792:
        raise ValueError("Expected 792 combined query-model records")
    seen=set()
    by_model={name:[] for name in MODEL32}
    by_type={}
    for row in rows:
        key=(row["model_id"],row["query_id"])
        if key in seen:
            raise ValueError("Duplicate query/model pair: "+str(key))
        seen.add(key)
        if row["model_id"] not in by_model or row["query_id"] not in expected:
            raise ValueError("Unknown model or query in 132-question run")
        if row["status"] not in {"ok","abstained"}:
            raise ValueError("Incomplete query execution: "+str(key))
        if row["returned_evidence"] not in {"True","False"}:
            raise ValueError("Unrecognized evidence flag: "+str(key))
        if ((row["status"]=="abstained") != (row["returned_evidence"]=="False")):
            raise ValueError("Abstention/evidence mismatch: "+str(key))
        by_model[row["model_id"]].append(row)
        by_type.setdefault((row["model_id"],row["query_type"]),[]).append(row)
    summary=[]
    for model,group in by_model.items():
        if len(group)!=132 or {r["query_id"] for r in group}!=expected:
            raise ValueError("Incomplete cohort for "+model)
        times=[float(r["retrieval_ms"]) for r in group
               if r["retrieval_ms"] not in ("",None)]
        if len(times)!=132:
            raise ValueError("Missing measured retrieval timings for "+model)
        summary.append({
            "model":model,"questions":132,
            "evidence_returned":sum(r["returned_evidence"]=="True" for r in group),
            "abstained":sum(r["status"]=="abstained" for r in group),
            "errors":0,"mean_retrieval_ms":_average(times),
            "evidence_basis":(
                "provisional_AI_assisted_source_summaries"
                if model in MODEL32[:4] else "G0_unverified_graph"),
            "mode":"retrieval_only",
        })
    categories=[]
    for (model,typ),group in sorted(by_type.items()):
        categories.append({
            "model":model,"query_type":typ,
            "questions":len(group),
            "evidence_returned":sum(r["returned_evidence"]=="True" for r in group),
            "abstained":sum(r["status"]=="abstained" for r in group),
        })
    return rows,summary,categories


def _safe_summary(path,kind):
    if not path.exists():return None
    item=_read_json(path)
    if kind=="analysis" and item.get("analysis_scope")!="manually_selected_changed_G1_model_query_cases_only":
        raise ValueError("Wrong review analysis format")
    if kind=="stable" and item.get("mode")!="G1_legacy_stable_anchor_plus_explicit_new_targets":
        raise ValueError("Wrong stable retrieval summary")
    return item


def _fmt(v):
    return "—" if v is None else str(v)


def _table(rows,columns):
    return "| "+" | ".join(columns)+" |\n|"+ "|".join("---" for _ in columns)+"|\n"+ "".join(
        "| "+" | ".join(_fmt(row.get(k,"—")) for k in columns)+" |\n" for row in rows
    )


def _figures(folder,historical,clean100,graph132):
    """Optional ready-to-insert PNGs. Never imply retrieval yield is accuracy."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return {"generated":[],"warning":"matplotlib not installed; numerical CSVs remain available"}
    folder=folder/"figures"
    folder.mkdir(exist_ok=True)
    paths=[]
    xs=list(range(len(historical)))
    fig,ax=plt.subplots(figsize=(11,4.5))
    ok=[r["status_ok"] for r in historical]
    err=[r["status_error"] for r in historical]
    abst=[r["status_abstained"] for r in historical]
    ax.bar(xs,ok,label="Completed successfully")
    ax.bar(xs,abst,bottom=ok,label="Abstained")
    ax.bar(xs,err,bottom=[a+b for a,b in zip(ok,abst)],label="Execution error")
    ax.set_xticks(xs,[r["model"] for r in historical])
    ax.set_ylim(0,105)
    ax.set_ylabel("Outcomes / 100 queries")
    ax.set_title("Historical B1–B8: recorded slots ≠ successful runs")
    ax.legend(loc="upper center",ncol=3)
    fig.tight_layout()
    fn=folder/"historical100_execution_status.png"
    fig.savefig(fn,dpi=160)
    plt.close(fig)
    paths.append(str(fn))
    if clean100:
        fig,ax=plt.subplots(figsize=(10,4.2))
        ax.bar(range(6),[int(r["evidence_returned"]) for r in clean100])
        ax.set_xticks(range(6),[r["model"] for r in clean100])
        ax.set_ylim(0,105)
        ax.set_ylabel("Questions with evidence / 100")
        ax.set_title("B1–B6 clean retrieval-only, provisional corpus vs G0 graph")
        fig.text(0.5,0.01,"Evidence availability only; evidence basis differs by model; not medical accuracy.",
                 ha="center",fontsize=8)
        fig.tight_layout(rect=(0,0.04,1,1))
        fn=folder/"clean100_evidence_yield.png"
        fig.savefig(fn,dpi=160)
        plt.close(fig)
        paths.append(str(fn))
    fig,ax=plt.subplots(figsize=(9.5,4.5))
    offsets=(-0.25,0,0.25)
    conditions=(("G0_yield_132","Frozen G0"),
                ("G1_legacy_yield_132","G1 legacy"),
                ("G1_stable_yield_132","G1 stabilized"))
    for dx,(col,label) in zip(offsets,conditions):
        ax.bar([i+dx for i in range(len(graph132))],
               [r[col] for r in graph132],width=0.22,label=label)
    ax.set_xticks(range(len(graph132)),[r["model"] for r in graph132])
    ax.set_ylim(0,132)
    ax.set_ylabel("Queries returning evidence / 132")
    ax.set_title("Paired graph retrieval: evidence yield, not causal accuracy")
    ax.legend()
    fig.tight_layout()
    fn=folder/"G0_G1_stable_132_evidence_yield.png"
    fig.savefig(fn,dpi=160)
    plt.close(fig)
    paths.append(str(fn))
    return {"generated":paths}


def assemble(run32_dir=None,retrieval100_dir=None,stable=STABLE,analysis=ANALYSIS):
    all100,_=_cohort()
    if len(all100)!=8:raise ValueError("All eight archived B1-B8 100-question records are required")
    thirtytwo=None
    thirtytwo_provenance=None
    if run32_dir:
        thirtytwo,thirtytwo_provenance=_verify_retrieval(run32_dir,"32")
    retrieval100=None
    retrieval100_provenance=None
    if retrieval100_dir:
        retrieval100,retrieval100_provenance=_verify_retrieval(retrieval100_dir,"100")
    stable_dir=_resolve_summary_dir(
        stable,WORKSPACE/"g1_stable_retrieval","stable_","stable")
    analysis_dir=_resolve_summary_dir(
        analysis,WORKSPACE/"g1_review_analysis","analysis_","analysis")
    stable_summary=_safe_summary(stable_dir/"summary.json","stable")
    review_summary=_safe_summary(analysis_dir/"summary.json","analysis")
    if stable_summary is None or review_summary is None:
        raise FileNotFoundError("Existing stable + reviewed-analysis summaries are required")
    if review_summary["reviewed_cases"]!=31 or review_summary["reviewed_evidence_occurrences"]!=132:
        raise ValueError("Unexpected reviewed sample; verify provenance")
    original_review=ROOT/review_summary["review_folder"]
    for name,key in (("review_cases.csv","source_case_sha256"),
                     ("review_evidence.csv","source_evidence_sha256")):
        if file_sha256(original_review/name)!=review_summary[key]:
            raise ValueError("Human review changed after analysis; rerun analysis")
    out=OUTPUT/_stamp("submission")
    out.mkdir(parents=True,exist_ok=False)
    _write_rows(out/"historical100_B1_B8.csv",tuple(all100[0]),all100)
    if thirtytwo:_write_rows(out/"new32_B1_B6.csv",SCHEMA,thirtytwo)
    if retrieval100:_write_rows(out/"clean100_retrieval_only_B1_B6.csv",SCHEMA,retrieval100)
    combined132=None
    if retrieval100 is not None and thirtytwo is not None:
        detailed,combined132,categories=_combine_retrieval_132(retrieval100_dir,run32_dir)
        _write_rows(out/"all132_retrieval_only_B1_B6.csv",HEADER32,detailed)
        _write_rows(out/"combined132_retrieval_only_B1_B6.csv",tuple(combined132[0]),combined132)
        _write_rows(out/"by_query_type132_B1_B6.csv",tuple(categories[0]),categories)
    selection=[]
    for model in ("B5","B6"):
        total_old=total_old_g1=total_stable=0
        for data in stable_summary["per_set"].values():
            item=data[model]
            total_old+=item["g0_evidence_yield"]
            total_old_g1+=item["legacy_g1_evidence_yield"]
            total_stable+=item["stable_g1_evidence_yield"]
        selection.append({"model":model,"G0_yield_132":total_old,
                          "G1_legacy_yield_132":total_old_g1,
                          "G1_stable_yield_132":total_stable})
    _write_rows(out/"G0_G1_132_shadow.csv",tuple(selection[0]),selection)
    historical22=_read_rows(LEGACY22) if LEGACY22.exists() else []
    structurals=_read_json(STRUCTURAL20) if STRUCTURAL20.exists() else {}
    structural_b7b8=[{
        "model":r["model"],"mapped_queries":r["mapped_queries"],
        "edge_micro_f1":r.get("edge_micro_f1"),
        "edge_micro_precision":r.get("edge_micro_precision"),
        "edge_micro_recall":r.get("edge_micro_recall"),
    } for r in structurals.get("results",[]) if r.get("model") in {"B7","B8"}]
    if historical22:
        _write_rows(out/"legacy22_RL_vs_NoRL.csv",tuple(historical22[0]),historical22)
    if structural_b7b8:
        _write_rows(out/"G0_structural20_B7_B8.csv",tuple(structural_b7b8[0]),structural_b7b8)
    submission={
        "mode":"reused_archived100_optional_retrieval32_and_reviewed_G1",
        "frozen_100_sha256":file_sha256(Q100),
        "frozen_32_sha256":file_sha256(Q32),
        "historical_B1_B8_all_eight":True,
        "historical_800_model_query_records":sum(r["n"] for r in all100),
        "historical_archive_execution_errors":sum(r["status_error"] for r in all100),
        "legacy22_summary_available":bool(historical22),
        "G0_structural20_B7_B8_available":len(structural_b7b8)==2,
        "historical_error_free_models":[r["model"] for r in all100 if r["status_error"]==0],
        "new32_B1_B6_complete":thirtytwo is not None,
        "clean100_retrieval_B1_B6_complete":retrieval100 is not None,
        "clean132_retrieval_B1_B6_complete":combined132 is not None,
        "clean132_model_query_records":792 if combined132 is not None else None,
        "stable_summary_dir":str(stable_dir.relative_to(ROOT)),
        "review_analysis_dir":str(analysis_dir.relative_to(ROOT)),
        "clean100_provenance":retrieval100_provenance,
        "new32_provenance":thirtytwo_provenance,
        "G0_G1_132_yield":selection,
        "reviewed_case_count":review_summary["reviewed_cases"],
        "reviewed_evidence_occurrences":review_summary["reviewed_evidence_occurrences"],
        "reviewed_distinct_assertions":review_summary["distinct_question_evidence_assertions"],
        "attention_cases":review_summary["attention_case_count"],
        "review_flags":review_summary["case_flags"],
        "graph_expansion_is_not_accuracy":True,
        "limitations":[
            "B1-B4 document summaries differ from B5-B8 graph-derived evidence, so evidence yield alone is not a fair model accuracy ranking.",
            "Older historical attempts contained errors, but selected archived runs can be error-free; preserve run provenance and do not compare end-to-end with retrieval-only latency.",
            "32 additional questions were not executed on B7 or B8.",
            "132 questions are drafts and influenced G1 concept inventory; no independent holdout or medical gold.",
            "Manually reviewed sample consists of cases selected for changed retrieval behavior, not representative test distribution.",
            "G1 inherited 37 unverified G0 edges; new candidate statements are not clinically certified.",
        ],
    }
    _dump(out/"submission_manifest.json",submission)
    history=[{key:r[key] for key in ("model","n","status_error","status_abstained","evidence_returned","answer_present","mode","mean_total_ms")}
             for r in all100]
    md=[
        "# CDF-RAG-2: partial experimental results (non-clinical pilot)","",
        "## A. Archived B1–B8 experiment on frozen 100 queries","",
        _table(history,("model","n","status_error","status_abstained","evidence_returned","answer_present","mode","mean_total_ms")),"",
        "**Interpretation:** These are the newest verified selected archived runs, including actual execution-error counts. "
        "Older failed attempts remain archived separately. The table is not medical accuracy. "
        "B1–B4 retrieve provisional AI-assisted source summaries; B5–B8 use G0 graph records. "
        "Do not compare generation latency against retrieval-only timings.","",
        "## B. Repaired retrieval-only comparison on the original 100 queries","",
    ]
    if retrieval100:
        md.extend([_table(retrieval100,("model","n","status_error","evidence_returned","mean_retrieval_ms")),"",
                   "B1–B6 are all retrieval-only, with recorded corpus/encoder checksums. B7/B8 are not retrieval-only."])
    else:
        md.append("**Pending:** Run a clean retrieval-only B1-B6 comparison on the original 100 queries before claiming a complete baseline comparison.")
    if combined132 is not None:
        md.extend(["","## B2. Combined 132-question B1–B6 retrieval-only results","",
            _table(combined132,("model","questions","evidence_returned",
                                "abstained","errors","mean_retrieval_ms")),"",
            "Means use each query's measured retrieval time; no generation or clinical accuracy is claimed.",""])
    md.extend(["","## C. Additional 32 draft queries",""])
    if thirtytwo:
        md.extend([_table(thirtytwo,("model","n","evidence_returned","mode","mean_retrieval_ms")),"",
                   "No B7/B8 execution on the 32-question extension. The six results are retrieval-only."])
    else:
        md.append("**Pending:** run32 has not yet been verified and is not represented as a completed experiment.")
    md.extend([
        "","## D. Paired G0 vs G1 retrieval on 132 draft questions","",
        _table(selection,("model","G0_yield_132","G1_legacy_yield_132","G1_stable_yield_132")),"",
        "This compares evidence coverage only. Stable resolver changes query processing, so gains cannot be attributed solely to graph expansion.","",
        "## E. Reviewed changed-case analysis","",
        f"- Review cases: {review_summary['reviewed_cases']} across {review_summary['distinct_questions']} distinct questions.",
        f"- Evidence occurrences: {review_summary['reviewed_evidence_occurrences']}; distinct question-evidence assertions: {review_summary['distinct_question_evidence_assertions']}.",
        f"- Cases with review flags: {review_summary['attention_case_count']}.",
        f"- Review issue flags: {json.dumps(review_summary['case_flags'],sort_keys=True)}.","",
        "This manual sample was selected because retrieval behavior changed; reviewer judgments do NOT establish general model accuracy.","",
        "## F. Historical 22-query experiments","",
        "Preserve historical RL/No-RL values as a separate 22-query feasibility result.","",
    ])
    if historical22:
        md.extend([_table(historical22,("Metric","No-RL","RL","RL - No-RL")),"",
                   "Historical 22-query metrics are implementation diagnostics, not medical accuracy."])
    else:
        md.append("Legacy 22-query CSV unavailable locally; values omitted.")
    md.extend(["","## F2. G0 structural reconstruction on 20 mapped draft-target queries",""])
    if structural_b7b8:
        md.extend([_table(structural_b7b8,("model","mapped_queries",
                                          "edge_micro_f1","edge_micro_precision",
                                          "edge_micro_recall")),"",
                   "These scores measure agreement with the unverified G0 snapshot and draft target mapping, not medical correctness."])
    else:
        md.append("20-query structural scores unavailable locally; values omitted.")
    md.extend([
        "## G. Defensible preliminary contribution","",
        "Built a review-gated typed G1 graph; demonstrated that graph expansion changes query target resolution and may reduce evidence coverage; "
        "introduced a conservative stability-first resolver; used manual judgments to identify and classify retrieval errors. "
        "Proposed next: typed evidence sufficiency and adaptive multi-round retrieval, evaluated on a separate adjudicated holdout.","",
        "## Limitations","",
        *("- "+x for x in submission["limitations"]),"",
    ])
    (out/"submission_results.md").write_text("\n".join(md)+"\n",encoding="utf-8")
    report=[
        "# Partial-work report structure",
        "1. Title, scope and research questions (explicitly: partial-stage contribution).",
        "2. Abstract: motivation, implemented prototype and verified pilot results.",
        "3. Introduction: reliability and cost of multi-round causal RAG.",
        "4. Literature survey: MRD-RAG, GARMLE-G, CDF-RAG; comparative gap.",
        "5. Existing B7/B8 architecture and frozen G0 baseline (35 nodes, 37 unverified edges).",
        "6. Dataset: 100+32 draft query categories, sample construction, limitations/no gold.",
        "7. Methods: B1–B8 registry, retrieval modes, G1 candidate provenance (78 nodes, 24 typed relations).",
        "8. Experimental protocol and fair-comparison caveats.",
        "9. Results A: selected archived 100-query B1–B8 matrix (with actual error/abstention counts).",
        "10. Results A2: clean B1–B6 retrieval-only comparison on 100+32 queries.",
        "11. Results B: 32-query B1–B6 retrieval-only (only after executed).",
        "12. Results C: 132-query G0/G1 regression, stabilized resolver, paired changes.",
        "13. Results D: manually reviewed 31 changed cases and 58 distinct assertions.",
        "14. Error analysis: 17 flagged cases, wrong targets, unsupported gains, causal direction.",
        "15. Preliminary contribution and bounded claims; threats to validity.",
        "16. Remaining implementation: feedback, sufficiency threshold, graph validation, holdout and statistical testing.",
        "17. References and appendix (code commit, snapshot hashes, commands, model configs).",
    ]
    (out/"partial_report_outline.md").write_text("\n\n".join(report)+"\n",encoding="utf-8")
    ppt=[
        "# Partial MTP presentation outline (7–9 slides)",
        "1. Title + problem: reliable domain-specific causal RAG; accuracy-latency trade-off.",
        "2. Research gap + objectives: single-pass retrieval/semantic matching/causation-vs-risk.",
        "3. Existing CDF-RAG and eight baselines B1–B8; differentiate legacy PPO and current contributions.",
        "4. Experimental assets: 100 + 32 draft queries; G0 vs G1 graph counts and provenance.",
        "5. Verified B1–B8 end-to-end archived 100-query matrix and clean B1–B6 retrieval-only comparison.",
        "6. Core investigation: graph expansion causes target drift; show paired B5/B6 G0→G1→stable counts.",
        "7. Manual evidence audit: 31 cases, 132 occurrences, 58 unique assertions; reviewed relevance, wrong targets.",
        "8. Contributions & limitations: reviewed G1 graph + matched comparison + stable resolver; no medical gold yet.",
        "9. Concrete next phase and supervisor questions: typed sufficiency, adaptive feedback, independent holdout, latency ablation.",
    ]
    (out/"partial_ppt_outline.md").write_text("\n\n".join(ppt)+"\n",encoding="utf-8")
    fig_report=_figures(out,all100,retrieval100,selection)
    _dump(out/"figure_manifest.json",fig_report)
    print(json.dumps({"output":str(out),"files":[p.name for p in out.iterdir()],
                      "historical_models":len(all100),"retrieval100_verified":retrieval100 is not None,
                      "extension_verified":thirtytwo is not None,
                      "reviewed_cases":review_summary["reviewed_cases"]},indent=2))
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    subs=p.add_subparsers(dest="command",required=True)
    subs.add_parser("verify")
    smoke=subs.add_parser("doctor")
    smoke.add_argument("--encoder",choices=("sentence","lsa"),default="sentence")
    run=subs.add_parser("run32")
    run100=subs.add_parser("run100")
    for command in (run,run100):
        command.add_argument("--encoder",choices=("sentence","lsa"),default="sentence")
        command.add_argument("--models",default="B1,B2,B3,B4,B5,B6")
        command.add_argument("--resume-dir",type=Path,
                             help="Resume failed/incomplete run without repeating successful models")
    build=subs.add_parser("assemble")
    build.add_argument("--run32-dir",type=Path)
    build.add_argument("--retrieval100-dir",type=Path)
    build.add_argument("--stable-dir",type=Path,default=STABLE)
    build.add_argument("--analysis-dir",type=Path,default=ANALYSIS)
    args=p.parse_args()
    if Path.cwd().resolve()!=ROOT:p.error("Run from project root")
    if args.command=="verify":verify()
    elif args.command=="doctor":doctor(args.encoder)
    elif args.command in {"run32","run100"}:
        models=tuple(x.strip() for x in args.models.split(",") if x.strip())
        if len(models)!=len(set(models)):p.error("Duplicate model IDs")
        run_retrieval(args.encoder,models,"32" if args.command=="run32" else "100",
                      args.resume_dir)
    else:assemble(args.run32_dir,args.retrieval100_dir,args.stable_dir,args.analysis_dir)


if __name__=="__main__":
    main()
