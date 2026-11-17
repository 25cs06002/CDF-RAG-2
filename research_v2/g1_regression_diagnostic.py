"""Diagnose G0 -> G1 B5/B6 evidence changes WITHOUT changing the retriever.

Reads already-produced paired per_query.csv files. Cheap optional shadow:
replay G1 B5/B6 with the target chosen by the frozen G0 lexical matcher.
This separates new-graph target-vocabulary drift from graph-edge expansion.

Does NOT use draft target_concept, gold answers, generation, Neo4j, PPO,
or medical-accuracy labels. Output is query-level diagnostic, not a new model.
"""
from __future__ import annotations

import argparse
import csv
import json
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from .base_pipeline import PipelineRequest
from .experiment_runner import PROJECT_ROOT, file_sha256
from .pipelines.baselines import G0Corpus, BaselinePipeline

ROOT = PROJECT_ROOT
G0 = ROOT / "research_artifacts/graph-G0"
OUTPUT = ROOT / "presentation_20261118/g1_regression_diagnostic"
PAIRS = (("B5","graph_rag"),("B6","causal_kg_rag"))
STAGES = ("G0_LEGACY","G1_LEGACY","G1_TYPED")


def _read_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as f:
        rows=list(csv.DictReader(f))
    if not rows:return []
    required={"model","query_id","question","matched_concept",
              "evidence_returned","reason","causal_paths"}
    if not required.issubset(rows[0]):
        raise ValueError(f"{path}: missing expected paired comparison columns")
    return rows


def _truth(value):
    if str(value).casefold()=="true":return True
    if str(value).casefold()=="false":return False
    raise ValueError("Expected true/false boolean, got "+repr(value))


class FrozenTargetCorpus(G0Corpus):
    """G1 edges and nodes, but original G0 query-only target selector."""
    def __init__(self, graph_dir: Path, frozen: G0Corpus):
        super().__init__(graph_dir=graph_dir)
        self.frozen=frozen

    def concept(self, query):
        target=self.frozen.concept(query)
        if target is not None and target not in self.nodes:
            raise ValueError("G0 target missing from G1: "+target)
        return target


def diagnose(graph_dir: Path, paired_dirs: list[Path],
             top_k: int = 5, output_root: Path = OUTPUT):
    graph_dir=Path(graph_dir).resolve()
    output_root=Path(output_root).resolve()
    paired_dirs=[Path(p).resolve() for p in paired_dirs]
    for p in (graph_dir,output_root,*paired_dirs):
        if not p.is_relative_to(ROOT):
            raise ValueError("Paths must be inside the project")
    manifest=json.loads((graph_dir/"manifest.json").read_text(encoding="utf-8"))
    if manifest.get("stage")!="G1_CANDIDATE_REQUIRES_INDEPENDENT_VERIFICATION":
        raise ValueError("Versioned G1 candidate graph required")
    if top_k<1:raise ValueError("top_k must be positive")

    frozen=G0Corpus(G0)
    locked=FrozenTargetCorpus(graph_dir,frozen)
    shadow={key:BaselinePipeline(name,corpus=locked,top_k=top_k,generate=False)
            for key,name in PAIRS}
    all_rows=[]
    summaries={}
    for paired_dir in paired_dirs:
        inputs=_read_csv(paired_dir/"per_query.csv")
        if not inputs:raise ValueError("Empty paired CSV: "+str(paired_dir))
        summary=json.loads((paired_dir/"summary.json").read_text(encoding="utf-8"))
        if summary.get("top_k")!=top_k or summary.get("graph_manifest_sha256")!=file_sha256(graph_dir/"manifest.json"):
            raise ValueError("Mismatch in paired top_k or G1 graph snapshot: "+str(paired_dir))
        group=summary["questions_file"]
        expected=summary["query_count"]
        indexed={}
        queries={}
        for row in inputs:
            key=(row["query_id"],row["model"])
            if key in indexed:raise ValueError("Duplicate model/query: "+str(key))
            indexed[key]=row
            qid=row["query_id"]
            if qid in queries and queries[qid]!=row["question"]:
                raise ValueError("Conflicting query text for "+qid)
            queries[qid]=row["question"]
        if len(queries)!=expected:
            raise ValueError("Paired summary and CSV query counts differ")
        stats={}
        for prefix,_ in PAIRS:
            transitions=Counter()
            drift_details=Counter()
            for qid,question in queries.items():
                key0=(qid,prefix+"_G0_LEGACY")
                key1=(qid,prefix+"_G1_LEGACY")
                keyt=(qid,prefix+"_G1_TYPED")
                if not all(k in indexed for k in (key0,key1,keyt)):
                    raise ValueError("Missing paired record: "+qid+" "+prefix)
                original,expanded,typed=(indexed[k] for k in (key0,key1,keyt))
                before=_truth(original["evidence_returned"])
                after=_truth(expanded["evidence_returned"])
                typed_yes=_truth(typed["evidence_returned"])
                transition=("gained" if after and not before else
                            "lost" if before and not after else
                            "both" if before else "neither")
                transitions[transition]+=1
                old_target=original["matched_concept"]
                new_target=expanded["matched_concept"]
                target_drift=old_target!=new_target
                # Target is selected only from query text by the ORIGINAL
                # G0 matcher. Never reference benchmark answers/targets.
                request=PipelineRequest(
                    run_id="G1_TARGET_LOCK_DIAGNOSTIC",
                    query_id=qid,query=question)
                locked_record=shadow[prefix].run(request)
                locked_yes=bool(locked_record.retrieved_documents or locked_record.causal_paths)
                if before and not locked_yes:
                    drift_details["g0_evidence_lost_even_with_locked_target"]+=1
                if transition=="lost":
                    drift_details["loss_with_target_drift" if target_drift
                                  else "loss_without_target_drift"]+=1
                    if locked_yes:
                        drift_details["loss_recovered_by_G0_target_lock"]+=1
                if transition=="gained":
                    drift_details["gains_with_target_drift" if target_drift
                                  else "gains_without_target_drift"]+=1
                all_rows.append({
                    "question_set":group,
                    "query_id":qid,
                    "question":question,
                    "phase2m_seed":original["phase2m_graph_construction_seed"],
                    "family":prefix,
                    "g0_target":old_target,
                    "g1_target":new_target,
                    "g1_typed_target":typed["matched_concept"],
                    "target_drift":target_drift,
                    "transition":transition,
                    "g0_evidence":before,
                    "g1_legacy_evidence":after,
                    "g1_frozen_target_evidence":locked_yes,
                    "g1_typed_evidence":typed_yes,
                    "g1_typed_reason":typed["reason"],
                    "g0_evidence_ids":original["evidence_ids"],
                    "g1_legacy_evidence_ids":expanded["evidence_ids"],
                    "g1_frozen_target_evidence_ids":"|".join(
                        d.document_id for d in locked_record.retrieved_documents),
                    "g1_typed_evidence_ids":typed["evidence_ids"],
                    "g0_causal_paths":original["causal_paths"],
                    "g1_legacy_causal_paths":expanded["causal_paths"],
                    "g1_frozen_target_causal_paths":len(locked_record.causal_paths),
                    "g1_typed_causal_paths":typed["causal_paths"],
                    "triage_priority":(
                        "P0_G1_LOST" if transition=="lost" else
                        "P1_G1_GAIN_NEEDS_RELEVANCE_REVIEW" if transition=="gained" else
                        "P2_TYPED_ABSTAIN_REVIEW" if not typed_yes else
                        "P3"
                    ),
                })
            stats[prefix]={"transitions":dict(transitions),
                           "target_drift_diagnostics":dict(drift_details),
                           "g1_locked_target_yield":sum(
                               r["g1_frozen_target_evidence"] for r in all_rows
                               if r["question_set"]==group and r["family"]==prefix)}
        summaries[group]=stats

    out=output_root/("triage_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")+"_"+uuid.uuid4().hex[:6])
    out.mkdir(parents=True,exist_ok=False)
    with (out/"all_query_diagnostics.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    selected=[r for r in all_rows if r["triage_priority"]!="P3"]
    with (out/"review_queue.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(sorted(selected,key=lambda r:(r["triage_priority"],r["question_set"],r["query_id"])))
    report={
        "graph":str(graph_dir.relative_to(ROOT)),
        "graph_manifest_sha256":file_sha256(graph_dir/"manifest.json"),
        "per_set":summaries,
        "all_diagnostic_rows":len(all_rows),
        "review_queue_rows":len(selected),
        "no_ground_truth_used":True,
        "medical_accuracy":False,
        "g1_legacy_includes_unverified_inherited_g0":True,
        "warning":"Frozen-target is a query-only diagnostic, NOT a new clinical model. Newly returned evidence must be checked for relevance. Do not compare sub-ms latencies without repeated timing trials.",
    }
    (out/"summary.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"output":str(out),**report},indent=2))
    return out,report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--graph-dir",type=Path,required=True)
    p.add_argument("--paired-dirs",type=Path,nargs="+",required=True)
    p.add_argument("--top-k",type=int,default=5)
    p.add_argument("--output-root",type=Path,default=OUTPUT)
    a=p.parse_args()
    if Path.cwd().resolve()!=ROOT:p.error("Run from repository root")
    diagnose(a.graph_dir,a.paired_dirs,a.top_k,a.output_root)


if __name__=="__main__":
    main()
