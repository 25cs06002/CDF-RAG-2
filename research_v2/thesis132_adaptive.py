"""B9 prototype: source-typed evidence gate and bounded adaptive retrieval.

This is an experimental retrieval/decision controller, NOT full
source-verified medical RAG and NOT PPO. It uses query text and
retrieval evidence ONLY. No benchmark answer/target/reviewer labels are
passed to the controller. Typed G1 edges carry source-review assertions,
not independently certified clinical gold. BM25 candidate documents are
contextual fallback, never sufficient to certify a causal claim.

Policies: fixed1/fixed2/fixed3 versus adaptive.
Rows are query-model observations on exactly the frozen 132-question set.
"""
from __future__ import annotations
import argparse
import json
import re
import time
import uuid
import statistics
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path

from .base_pipeline import PipelineRequest
from .g1_typed_retrieval import G1TypedIndex
from .g1_query_interpreter import ABSTAIN_RX
from .ollama_generation import check_ollama
from .thesis132 import _inside,_cohort,_write_json,CORPUS
from .schema import PipelineRecord,StageTimings

ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/"presentation_20261118/thesis132/adaptive"
POLICIES={"fixed1","fixed2","fixed3","adaptive"}
MULTIHOP=re.compile(r"\b(intermediate|pathways?|multi.hop|through|chain|sequence|leads? to)\b",re.I)
COMPARISON=re.compile(r"\b(compare|compared|difference|different|versus|vs\.?|contrast)\b",re.I)


class AdaptiveCandidateController:
    def __init__(self,index,document_pipeline,top_k=5):
        self.index=index
        self.document_pipeline=document_pipeline
        if top_k<1:raise ValueError("top_k must be positive")
        self.top_k=top_k

    def _multi_ambiguous(self,query):
        if not COMPARISON.search(query):return False
        # Exact entity mentions in the question only, no benchmark targets.
        mentioned={n.casefold() for n in self.index.nodes
                   if n.casefold() in query.casefold()}
        return len(mentioned)>=2

    def run(self,request,policy="adaptive"):
        if policy not in POLICIES:raise ValueError("Unknown policy")
        start=time.perf_counter()
        require_path=bool(MULTIHOP.search(request.query))
        if ABSTAIN_RX.search(request.query) or self._multi_ambiguous(request.query):
            return PipelineRecord(
                run_id=request.run_id,query_id=request.query_id,
                pipeline_id="b9_candidate_"+policy,
                original_query=request.query,status="abstained",
                action_history=["abstain_ambiguous_or_unsupported"],
                timings=StageTimings(total_ms=(time.perf_counter()-start)*1000,
                                     retrieval_ms=0),
            )
        evidence_docs=[]
        typed_edges=[]
        typed_paths=[]
        typed_document_ids=set()
        actions=[]
        reason="insufficient_candidate_typed_evidence"
        typed_ms=0.0
        document_ms=0.0
        target=None
        needed=(3 if require_path else 2) if policy=="adaptive" else (
            3 if policy=="fixed3" else 2 if policy=="fixed2" else 1)
        # Query is never modified using its expected target or reviewer labels.
        for stage in range(needed):
            if policy=="adaptive":
                # If a pathway was requested, attempt directed two-hop
                # BEFORE broad document fallback; otherwise ask document
                # retriever only when a typed match was insufficient.
                step=("typed_direct" if stage==0 else
                      "typed_twohop" if stage==1 and require_path else
                      "document_bm25" if stage==1 or stage==2 else "stop")
            else:
                step=("typed_direct","document_bm25","typed_twohop")[stage]
            if step=="stop":break
            actions.append(step)
            if step.startswith("typed_"):
                candidate,explanation=self.index.retrieve(
                    request.query,causal_multihop=(step=="typed_twohop"),
                    top_k=self.top_k,query_id=request.query_id,run_id=request.run_id)
                typed_ms+=(candidate.timings.retrieval_ms or 0.0)
                if candidate.matched_concept:
                    target=candidate.matched_concept
                seen={d.document_id for d in evidence_docs}
                for d in candidate.retrieved_documents:
                    if d.document_id not in seen:
                        evidence_docs.append(d);seen.add(d.document_id)
                    typed_document_ids.add(d.document_id)
                # Restrict PATH sufficiency to typed CAUSES→CAUSES
                # paths returned by the trusted-structure candidate index.
                seen_paths={tuple(p.nodes) for p in typed_paths}
                for path in candidate.causal_paths:
                    if tuple(path.nodes) not in seen_paths:
                        typed_paths.append(path)
                        seen_paths.add(tuple(path.nodes))
                seen_edges={(e.cause,e.relation,e.effect) for e in typed_edges}
                for edge in candidate.causal_edges:
                    if (edge.cause,edge.relation,edge.effect) not in seen_edges:
                        typed_edges.append(edge)
                        seen_edges.add((edge.cause,edge.relation,edge.effect))
                reason=explanation
            elif step=="document_bm25":
                doc=self.document_pipeline.run(request)
                document_ms+=(doc.timings.retrieval_ms or 0.0)
                seen={d.document_id for d in evidence_docs}
                for d in doc.retrieved_documents:
                    if d.document_id not in seen:
                        evidence_docs.append(d);seen.add(d.document_id)
            enough=bool(typed_paths) if require_path else bool(typed_document_ids)
            # Source-reviewed typed relationships can support a
            # CANDIDATE answer, NOT independent clinical gold.
            # Never let provisional BM25 summaries alone certify sufficiency.
            if policy=="adaptive" and enough:
                actions.append("early_stop_candidate_evidence")
                break
        enough=bool(typed_paths) if require_path else bool(typed_document_ids)
        decision="candidate_for_review" if enough else "abstain_unverified"
        total_ms=(time.perf_counter()-start)*1000
        return PipelineRecord(
            run_id=request.run_id,query_id=request.query_id,
            pipeline_id="b9_candidate_"+policy,
            original_query=request.query,matched_concept=target,
            retrieved_documents=evidence_docs,causal_edges=typed_edges,
            causal_paths=typed_paths,final_answer=None,
            feedback_iterations=max(0,sum(a in {"typed_direct","typed_twohop","document_bm25"}
                                           for a in actions)-1),
            action_history=actions+[decision],
            status="ok" if enough else "abstained",
            timings=StageTimings(retrieval_ms=typed_ms+document_ms,
                                 total_ms=total_ms),
            error_message=None,
        )


def run(graph_dir,policy="adaptive",top_k=5,output_root=OUTPUT):
    if policy not in POLICIES:raise ValueError("Choose fixed1, fixed2, fixed3 or adaptive")
    graph_dir=_inside(graph_dir)
    output_root=_inside(output_root)
    graph_manifest=graph_dir/"manifest.json"
    if not graph_manifest.is_file():
        raise FileNotFoundError("No local reviewed-workflow G1 candidate graph")
    index=G1TypedIndex(graph_dir)
    if not CORPUS.is_file():
        raise FileNotFoundError("B1–B4 candidate document corpus missing: "+str(CORPUS))
    from .pipelines import create_pipeline
    from .experiment_runner import file_sha256
    bm25=create_pipeline("B2",document_corpus=CORPUS,top_k=top_k,generate=False)
    controller=AdaptiveCandidateController(index,bm25,top_k)
    _,source,questions=_cohort()
    out=output_root/("b9_"+policy+"_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")+
                     "_"+uuid.uuid4().hex[:6])
    out.mkdir(parents=True,exist_ok=False)
    records=[]
    for q in questions:
        req=PipelineRequest(run_id=out.name,query_id=q.query_id,query=q.query,
                            graph_version="G1_CANDIDATE")
        rec=controller.run(req,policy=policy)
        records.append(rec)
    if len(records)!=132 or len({r.query_id for r in records})!=132:
        raise ValueError("Incomplete 132 experiment")
    (out/"records.jsonl").write_text(
        "".join(r.model_dump_json()+"\n" for r in records),encoding="utf-8")
    metrics={
        "experiment":"B9_RULE_BASED_PROTOTYPE_RETRIEVAL_NOT_FINAL_ADAPTIVE_RAG",
        "policy":policy,"queries":132,"top_k":top_k,
        "input_sha256":file_sha256(source),
        "graph_manifest_sha256":file_sha256(graph_manifest),
        "document_corpus_sha256":file_sha256(CORPUS),
        "status_ok":sum(x.status=="ok" for x in records),
        "status_abstained":sum(x.status=="abstained" for x in records),
        "status_error":sum(x.status=="error" for x in records),
        "candidate_evidence_returned":sum(x.status=="ok" for x in records),
        "average_retrieval_ms":statistics.mean(x.timings.retrieval_ms or 0 for x in records),
        "average_total_ms":statistics.mean(x.timings.total_ms or 0 for x in records),
        "mean_actions":statistics.mean(len(x.action_history) for x in records),
        "early_stop_queries":sum("early_stop_candidate_evidence" in x.action_history for x in records),
        "two_hop_queries":sum(bool(x.causal_paths) for x in records),
        "medical_accuracy":False,
        "unseen_holdout":False,
        "generator_used":False,
        "PPO_used":False,
        "warning":"G1 typed relations and BM25 summaries are source-reviewed CANDIDATES only. This tests a policy mechanism and cost, not medical correctness."
    }
    _write_json(out/"summary.json",metrics)
    print(json.dumps({"output":str(out),**metrics},indent=2))
    return out,metrics


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--graph-dir",type=Path,required=True)
    p.add_argument("--policy",choices=sorted(POLICIES),default="adaptive")
    p.add_argument("--top-k",type=int,default=5)
    p.add_argument("--output-root",type=Path,default=OUTPUT)
    args=p.parse_args()
    run(args.graph_dir,args.policy,args.top_k,args.output_root)

if __name__=="__main__":
    main()
