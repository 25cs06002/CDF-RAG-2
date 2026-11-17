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
    summary=[]
    for model in MODEL100:
        if model not in selected:continue
        end_to_end,_,folder,records,info=selected[model]
        if [r.query_id for r in records]!=originals:
            raise ValueError("100-query archive order differs for "+model)
        # A completed archive may still contain execution errors.
        # Preserve it, label errors, and perform separate retrieval-only repairs.
        mode="end_to_end" if end_to_end else "retrieval_only"
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
        "archive_error_count":sum(r["status_error"] for r in rows),
        "medical_accuracy":False,
        "warning":"Existing 100-query run may contain execution errors; completed slots are not successful runs. Historical B1-B6 include connection/generation errors and B1-B4/B5-B8 have different evidence bases. Use fresh retrieval-only 100 for B1-B6.",
    }
    print(json.dumps(result,indent=2))
    return result


def _expansion():
    data=[json.loads(line) for line in Q32.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(data)!=32 or len({d["query_id"] for d in data})!=32:
        raise ValueError("Expected the fixed 32 unique expansion queries")
    questions=load_questions(Q32)
    if [q.query_id for q in questions]!=[r["query_id"] for r in data]:
        raise ValueError("Parsed question identity mismatch")
    return questions,{r["query_id"]:r["query_type"] for r in data}


def run_retrieval(encoder="sentence",models=MODEL32,scope="32"):
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
            raise ValueError("B7/B8 cannot be launched by retrieval-only run32")
    if not CORPUS.is_file():
        raise FileNotFoundError("Frozen provisional B1-B4 corpus missing")
    corpus_manifest=_read_json(CORPUS.parent/"manifest.json")
    if corpus_manifest["documents_sha256"]!=file_sha256(CORPUS):
        raise ValueError("Provisional document corpus was modified")
    if not (G0/"edges.csv").is_file():
        raise FileNotFoundError("G0 snapshot missing")
    root=OUTPUT/_stamp("run32" if scope=="32" else "run100")
    root.mkdir(parents=True,exist_ok=False)
    records_by_model={}
    index={
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
    _dump(root/"run_index.json",index)
    for model in models:
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
        summary,run_dir=run_experiment(
            pipeline=pipeline,questions=questions,run_id=name,
            source_file=source_file,output_root=root/"runs",
            graph_version="G0",on_run_ready=provenance,
        )
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


def assemble(run32_dir=None,retrieval100_dir=None,stable=STABLE,analysis=ANALYSIS):
    all100,_=_cohort()
    if len(all100)!=8:raise ValueError("Eight complete B1-B8 original runs required")
    thirtytwo=None
    thirtytwo_provenance=None
    if run32_dir:
        thirtytwo,thirtytwo_provenance=_verify_retrieval(run32_dir,"32")
    retrieval100=None
    retrieval100_provenance=None
    if retrieval100_dir:
        retrieval100,retrieval100_provenance=_verify_retrieval(retrieval100_dir,"100")
    stable_dir=_safe(stable)
    analysis_dir=_safe(analysis)
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
            "100-query archive has execution errors for several B1-B6 models and mixed modes; do not treat completed slots as successful baseline runs or pool latency.",
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
        "**Important:** Archived B1-B6 runs contain execution errors; completed slots are not necessarily successful experiments. " +
        "**Interpretation:** This table is descriptive, not medical accuracy. "
        "B1–B4 retrieve provisional AI-assisted source summaries; B5–B8 use G0 graph records. "
        "Mixed retrieval-only/end-to-end modes mean generation latency is not comparable across all eight.","",
        "## B. Repaired retrieval-only comparison on the original 100 queries","",
    ]
    if retrieval100:
        md.extend([_table(retrieval100,("model","n","status_error","evidence_returned","mean_retrieval_ms")),"",
                   "B1–B6 are all retrieval-only, with recorded corpus/encoder checksums. B7/B8 are not retrieval-only."])
    else:
        md.append("**Pending:** Run a clean retrieval-only B1-B6 comparison on the original 100 queries before claiming a complete baseline comparison.")
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
        "9. Results A: historical 100-query B1–B8 matrix (include execution errors explicitly).",
        "10. Results A2: clean 100-query B1–B6 retrieval-only comparison (after executed).",
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
        "5. Historical B1–B8 100-query matrix: include actual execution errors; add repaired B1–B6 retrieval-only panel.",
        "6. Core investigation: graph expansion causes target drift; show paired B5/B6 G0→G1→stable counts.",
        "7. Manual evidence audit: 31 cases, 132 occurrences, 58 unique assertions; reviewed relevance, wrong targets.",
        "8. Contributions & limitations: reviewed G1 graph + matched comparison + stable resolver; no medical gold yet.",
        "9. Concrete next phase and supervisor questions: typed sufficiency, adaptive feedback, independent holdout, latency ablation.",
    ]
    (out/"partial_ppt_outline.md").write_text("\n\n".join(ppt)+"\n",encoding="utf-8")
    print(json.dumps({"output":str(out),"files":[p.name for p in out.iterdir()],
                      "historical_models":len(all100),"retrieval100_verified":retrieval100 is not None,
                      "extension_verified":thirtytwo is not None,
                      "reviewed_cases":review_summary["reviewed_cases"]},indent=2))
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    subs=p.add_subparsers(dest="command",required=True)
    subs.add_parser("verify")
    run=subs.add_parser("run32")
    run100=subs.add_parser("run100")
    for command in (run,run100):
        command.add_argument("--encoder",choices=("sentence","lsa"),default="sentence")
        command.add_argument("--models",default="B1,B2,B3,B4,B5,B6")
    build=subs.add_parser("assemble")
    build.add_argument("--run32-dir",type=Path)
    build.add_argument("--retrieval100-dir",type=Path)
    build.add_argument("--stable-dir",type=Path,default=STABLE)
    build.add_argument("--analysis-dir",type=Path,default=ANALYSIS)
    args=p.parse_args()
    if Path.cwd().resolve()!=ROOT:p.error("Run from project root")
    if args.command=="verify":verify()
    elif args.command in {"run32","run100"}:
        models=tuple(x.strip() for x in args.models.split(",") if x.strip())
        if len(models)!=len(set(models)):p.error("Duplicate model IDs")
        run_retrieval(args.encoder,models,"32" if args.command=="run32" else "100")
    else:assemble(args.run32_dir,args.retrieval100_dir,args.stable_dir,args.analysis_dir)


if __name__=="__main__":
    main()
