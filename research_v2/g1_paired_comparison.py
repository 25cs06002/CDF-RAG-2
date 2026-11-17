"""Paired, retrieval-only G0/G1 comparison; no model generation or gold labels.

For each question, evaluate:
* B5/B6 legacy algorithms on frozen G0
* SAME B5/B6 algorithms on G1 causal-format snapshot (contains 37
  UNVERIFIED inherited G0 edges + 15 candidate reviewed CAUSES edges)
* B5/B6 G1 typed-only algorithms (24 manually approved typed edges,
  with only 15 CAUSES links; excludes the inherited unverified G0 edges)

These last conditions have different retrieval behavior, so typed vs legacy
differences are NOT attributable solely to graph expansion.
Direct G1 construction queries PILOT_046..PILOT_060 are flagged;
all questions remain draft, not an independent clinical test set.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from .base_pipeline import PipelineRequest
from .experiment_runner import PROJECT_ROOT, file_sha256
from .g1_typed_retrieval import G1TypedIndex
from .pipelines.baselines import BaselinePipeline, G0Corpus

ROOT = PROJECT_ROOT
G0 = ROOT/"research_artifacts/graph-G0"
PILOT = ROOT/"presentation_20261118/benchmark/questions_only.jsonl"
EXPANSION = ROOT/"data/benchmark/medical_causal_expansion32_draft_v1.jsonl"
OUTPUT = ROOT/"presentation_20261118/g1_paired_comparison"
PHASE2M_SEEDS = {f"PILOT_{i:03d}" for i in range(46,61)}
VERSIONS = (
    ("B5_G0_LEGACY","B5"),
    ("B5_G1_LEGACY","B5"),
    ("B5_G1_TYPED","B5"),
    ("B6_G0_LEGACY","B6"),
    ("B6_G1_LEGACY","B6"),
    ("B6_G1_TYPED","B6"),
)


def questions_from(path: Path):
    with path.open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if not rows or any(not r.get("query_id") or not r.get("query") for r in rows):
        raise ValueError("Empty or invalid question records")
    ids = [r["query_id"] for r in rows]
    if len(ids)!=len(set(ids)):
        raise ValueError("Duplicate query IDs")
    # Deliberately discard all benchmark metadata, draft target labels,
    # answerability flags, ground truth, and annotations at inference.
    return [{"query_id":r["query_id"],"query":r["query"]} for r in rows]


def percentile(values, pct):
    if not values:return None
    ordered=sorted(values)
    rank=(len(ordered)-1)*pct/100
    lo=math.floor(rank)
    hi=math.ceil(rank)
    return ordered[lo]+(rank-lo)*(ordered[hi]-ordered[lo])


def summarize(rows):
    times=[r["retrieval_ms"] for r in rows]
    return {
        "queries":len(rows),
        "evidence_yield_count":sum(r["evidence_returned"] for r in rows),
        "abstentions":sum(not r["evidence_returned"] for r in rows),
        "twohop_query_count":sum(r["causal_paths"]>0 for r in rows),
        "causal_paths_total":sum(r["causal_paths"] for r in rows),
        "mean_retrieval_ms":statistics.mean(times),
        "median_retrieval_ms":statistics.median(times),
        "p95_retrieval_ms":percentile(times,95),
    }


def run(graph_dir: Path, questions: Path, top_k: int = 5,
        output_root: Path = OUTPUT):
    graph_dir=Path(graph_dir).expanduser().resolve()
    questions=Path(questions).expanduser().resolve()
    output_root=Path(output_root).expanduser().resolve()
    for p in (graph_dir,questions,output_root):
        if not p.is_relative_to(ROOT):
            raise ValueError("All inputs and output must be within the repository")
    if top_k<1:raise ValueError("top_k must be positive")
    query_rows=questions_from(questions)

    # Locked graph & provenance checks before making any comparisons.
    manifest=json.loads((graph_dir/"manifest.json").read_text(encoding="utf-8"))
    if manifest.get("stage")!="G1_CANDIDATE_REQUIRES_INDEPENDENT_VERIFICATION":
        raise ValueError("G1 graph is not a versioned review-gated candidate")
    strict=G1TypedIndex(graph_dir)
    g0=G0Corpus(G0)
    g1=G0Corpus(graph_dir)
    if len(g0.nodes)!=35 or len(g0.edges)!=37:
        raise ValueError("Expected immutable frozen 35-node 37-edge G0")
    if len(g1.nodes)!=manifest.get("total_nodes") or len(g1.edges)!=manifest.get("total_causes_edges"):
        raise ValueError("G1 graph files do not match manifest counts")
    if len(strict.edges)!=sum(manifest.get("approved_typed_relations",{}).values()):
        raise ValueError("Typed edge count differs from G1 manifest")

    pipelines={
        "B5_G0_LEGACY":BaselinePipeline("graph_rag",corpus=g0,top_k=top_k,generate=False),
        "B5_G1_LEGACY":BaselinePipeline("graph_rag",corpus=g1,top_k=top_k,generate=False),
        "B6_G0_LEGACY":BaselinePipeline("causal_kg_rag",corpus=g0,top_k=top_k,generate=False),
        "B6_G1_LEGACY":BaselinePipeline("causal_kg_rag",corpus=g1,top_k=top_k,generate=False),
    }
    results=[]
    for q in query_rows:
        query_id=q["query_id"]
        request=PipelineRequest(run_id="G1_PAIRED",query_id=query_id,query=q["query"])
        common={
            "query_id":query_id,
            "question":q["query"],
            "phase2m_graph_construction_seed":query_id in PHASE2M_SEEDS,
        }
        for model,_ in VERSIONS:
            if model.endswith("_TYPED"):
                record,reason=strict.retrieve(
                    q["query"],causal_multihop=model.startswith("B6_"),
                    top_k=top_k,query_id=query_id,run_id="G1_PAIRED")
            else:
                record=pipelines[model].run(request)
                reason="legacy_query_resolver"
            results.append({
                **common,
                "model":model,
                "evidence_provenance":(
                    "candidate_reviewed_typed_only" if model.endswith("_TYPED") else
                    "legacy_mixed_reviewed_and_unverified" if model.endswith("_G1_LEGACY") else
                    "legacy_G0_unverified"
                ),
                "status":record.status,
                "matched_concept":record.matched_concept or "",
                "reason":reason,
                "evidence_returned":bool(record.retrieved_documents or record.causal_paths),
                "evidence_ids":"|".join(d.document_id for d in record.retrieved_documents),
                "causal_edges":len(record.causal_edges),
                "causal_paths":len(record.causal_paths),
                "retrieval_ms":record.timings.retrieval_ms,
            })
    counts=Counter(r["model"] for r in results)
    if any(v!=len(query_rows) for v in counts.values()) or len(counts)!=6:
        raise ValueError("Unpaired model/query results")

    all_by_model={}
    seed_strata={}
    for model,_ in VERSIONS:
        subset=[r for r in results if r["model"]==model]
        all_by_model[model]=summarize(subset)
        seed_strata[model]={
            "phase2m_seed":summarize([r for r in subset if r["phase2m_graph_construction_seed"]])
                if any(r["phase2m_graph_construction_seed"] for r in subset) else None,
            "non_phase2m_seed":summarize([r for r in subset if not r["phase2m_graph_construction_seed"]])
                if any(not r["phase2m_graph_construction_seed"] for r in subset) else None,
        }
    comparisons={}
    for family in ("B5","B6"):
        for other in ("G1_LEGACY","G1_TYPED"):
            a={r["query_id"]:r for r in results if r["model"]==family+"_G0_LEGACY"}
            b={r["query_id"]:r for r in results if r["model"]==family+"_"+other}
            comparisons[family+"_G0_LEGACY_vs_"+family+"_"+other]={
                "recovered_evidence":sum(not a[q]["evidence_returned"] and b[q]["evidence_returned"] for q in a),
                "lost_evidence":sum(a[q]["evidence_returned"] and not b[q]["evidence_returned"] for q in a),
                "both_evidence":sum(a[q]["evidence_returned"] and b[q]["evidence_returned"] for q in a),
                "neither_evidence":sum(not a[q]["evidence_returned"] and not b[q]["evidence_returned"] for q in a),
                "like_for_like_algorithm":other=="G1_LEGACY",
            }
    metadata={
        "mode":"paired_retrieval_only_no_model_generation",
        "graph_dir":str(graph_dir.relative_to(ROOT)),
        "graph_manifest_sha256":file_sha256(graph_dir/"manifest.json"),
        "questions_file":str(questions.relative_to(ROOT)),
        "questions_sha256":file_sha256(questions),
        "query_count":len(query_rows),
        "top_k":top_k,
        "g0_nodes":len(g0.nodes),"g0_unverified_edges":len(g0.edges),
        "g1_nodes":len(g1.nodes),"g1_causal_format_edges":len(g1.edges),
        "g1_inherited_unverified_edges":manifest.get("inherited_G0_edges_unverified"),
        "g1_reviewed_typed_edges":len(strict.edges),
        "approved_aliases":len(strict.aliases),
        "phase2m_seed_query_count":sum(q["query_id"] in PHASE2M_SEEDS for q in query_rows),
        "medical_accuracy":False,
        "independent_holdout":False,
        "reference_labels_used_for_inference":False,
        "warnings":[
            "G0 and G1 legacy use the same algorithms, but G1 legacy includes inherited unverified G0 edges.",
            "G1 typed uses only source-reviewed proposed graph facts; it changes the algorithm/evidence policy, not just graph size.",
            "Phase2M seed questions helped construct G1; do not cite their retrieval yield as out-of-sample generalization.",
            "Other benchmark queries supplied concept inventory inputs, so non-seed is not guaranteed a clean held-out split.",
            "Evidence yield and causal path count are NOT medical accuracy, precision, recall, or proof of causation.",
            "Submillisecond Python retrieval latency is exploratory; report machine/cold-cache/repeats for rigorous benchmarking."
        ],
        "models":all_by_model,
        "model_strata":seed_strata,
        "paired_changes":comparisons,
    }
    out=output_root/("paired_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")+"_"+uuid.uuid4().hex[:6])
    out.mkdir(parents=True,exist_ok=False)
    with (out/"per_query.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    (out/"summary.json").write_text(json.dumps(metadata,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"output":str(out),"query_count":len(query_rows),
        "phase2m_seed_query_count":metadata["phase2m_seed_query_count"],
        "models":all_by_model,"paired_changes":comparisons},indent=2))
    return out,metadata


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-dir",type=Path,required=True)
    parser.add_argument("--queries",type=Path,default=PILOT)
    parser.add_argument("--top-k",type=int,default=5)
    parser.add_argument("--output-root",type=Path,default=OUTPUT)
    a=parser.parse_args()
    if Path.cwd().resolve()!=ROOT:parser.error("Run from repo root")
    run(a.graph_dir,a.queries,a.top_k,a.output_root)


if __name__=="__main__":
    main()
