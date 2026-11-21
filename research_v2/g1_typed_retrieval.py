"""Review-gated G1 typed graph retrieval without treating association as causation.

Experimental B5_G1_TYPED and B6_G1_TYPED; archived B1-B8 untouched.
Reads approved edges from G1_CANDIDATE, ignores unreviewed inherited G0 edges.
No LLM, no Neo4j, no PPO and no benchmark target/gold-label input.
"""
from __future__ import annotations
import argparse
import csv
import json
import re
import statistics
import time
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from .experiment_runner import PROJECT_ROOT, file_sha256
from .g1_query_interpreter import ABSTAIN_RX, exact_mention, load_approved_aliases
from .g1_target_resolution import resolve_target
from .g1_foundation import RELATIONS
from .schema import CausalEdge, CausalPath, EvidenceDocument, PipelineRecord, StageTimings

EXPANSION=PROJECT_ROOT/"data/benchmark/medical_causal_expansion32_draft_v1.jsonl"
PILOT=PROJECT_ROOT/"presentation_20261118/benchmark/questions_only.jsonl"
OUTPUT=PROJECT_ROOT/"presentation_20261118/g1_typed_comparison"
STRONG_CAUSAL=re.compile(r"\b(caus(?:e|es|ed|al)|mechanism|pathway|intermediate|through)\b",re.I)
RISK=re.compile(r"\b(risk|associat|predispos|correlat)\w*\b",re.I)
TRIGGER=re.compile(r"\b(trigger|attack|exacerbat)\w*\b",re.I)


class G1TypedIndex:
    def __init__(self, graph_dir: Path):
        self.graph_dir=Path(graph_dir).resolve()
        if not self.graph_dir.is_relative_to(PROJECT_ROOT):
            raise ValueError("G1 snapshot must remain inside project")
        manifest=json.loads((self.graph_dir/"manifest.json").read_text(encoding="utf-8"))
        if manifest.get("stage")!="G1_CANDIDATE_REQUIRES_INDEPENDENT_VERIFICATION":
            raise ValueError("Not an approved-workflow G1 candidate graph")
        edges_file=self.graph_dir/"typed_relationships.csv"
        alias_file=self.graph_dir/"approved_aliases.csv"
        if file_sha256(edges_file)!=manifest.get("edge_ledger_sha256"):
            raise ValueError("Typed relationships have changed since build")
        if file_sha256(alias_file)!=manifest.get("alias_ledger_sha256"):
            raise ValueError("Aliases have changed since build")
        with (self.graph_dir/"nodes.csv").open(newline="",encoding="utf-8") as f:
            self.nodes=[row["name"] for row in csv.DictReader(f)]
        self.aliases=load_approved_aliases(alias_file,self.nodes)
        with edges_file.open(newline="",encoding="utf-8") as f:
            all_edges=list(csv.DictReader(f))
        self.edges=[]
        for edge in all_edges:
            if edge["review_status"]!="approved":
                raise ValueError("Unreviewed typed relationship in G1 snapshot")
            if edge["relation"] not in RELATIONS:
                raise ValueError("Unknown typed relationship")
            self.edges.append(edge)
        self.by_effect=defaultdict(list)
        for item in self.edges:
            self.by_effect[item["object"].casefold()].append(item)

    def target(self,query: str):
        if ABSTAIN_RX.search(query):
            return None,"under_determined_question"
        # Only explicitly mentioned evidence-eligible EFFECTS are targets.
        # Directional syntax may resolve a downstream endpoint, but
        # comparisons and unresolved multiple effects abstain.
        eligible={n for n in self.nodes if self.by_effect[n.casefold()]}
        return resolve_target(query,self.nodes,self.aliases,eligible)

    @staticmethod
    def allowed_relations(query):
        # A causal question must not be answered using only correlations.
        if STRONG_CAUSAL.search(query):
            return {"CAUSES"}
        if TRIGGER.search(query):
            return {"TRIGGERS_EXACERBATION","CAUSES"}
        if RISK.search(query):
            return {"RISK_FACTOR_FOR","ASSOCIATED_WITH","CAUSES","PROTECTS_AGAINST"}
        # Taxonomy-only and explicitly uncertain edges cannot establish
        # candidate causal/risk evidence for an unspecified question.
        return RELATIONS-{"IS_A","UNCERTAIN"}

    def retrieve(self,query: str, *, causal_multihop: bool = False, top_k: int = 5,
                 query_id: str = "Q", run_id: str = "G1_SHADOW"):
        if top_k<1:raise ValueError("top_k must be positive")
        start=time.perf_counter()
        target,reason=self.target(query)
        chosen=[]
        paths=[]
        if target:
            allowed=self.allowed_relations(query)
            candidates=[r for r in self.by_effect[target.casefold()] if r["relation"] in allowed]
            chosen=candidates[:top_k]
            if causal_multihop:
                # A two-hop path is valid only when both reviewed arrows are CAUSES.
                causal=[r for r in self.by_effect[target.casefold()] if r["relation"]=="CAUSES"]
                for last in causal:
                    for prior in self.by_effect[last["subject"].casefold()]:
                        if prior["relation"]=="CAUSES":
                            paths.append(CausalPath(
                                nodes=[prior["subject"],last["subject"],last["object"]],
                                evidence_ids=[prior["source_id"],last["source_id"]]))
                            if len(paths)>=top_k:break
                    if len(paths)>=top_k:break
            if not chosen and not paths:
                reason="target_found_no_relation_of_requested_type"
        docs=[
            EvidenceDocument(
                document_id=r["edge_id"],source_id=r["source_id"],
                text=f'[{r["relation"]}] {r["subject"]} -> {r["object"]}. '
                     f'Reviewer paraphrase: {r["support_summary"]}')
            for r in chosen
        ]
        causal_edges=[
            CausalEdge(cause=r["subject"],effect=r["object"],
                       relation="CAUSES",evidence_ids=[r["source_id"]])
            for r in chosen if r["relation"]=="CAUSES"
        ]
        ms=(time.perf_counter()-start)*1000
        record=PipelineRecord(
            run_id=run_id,query_id=query_id,
            pipeline_id="g1_typed_twohop" if causal_multihop else "g1_typed_onehop",
            original_query=query,graph_version="G1_CANDIDATE",
            matched_concept=target,
            retrieved_documents=docs,causal_edges=causal_edges,
            causal_paths=paths,
            status="ok" if docs or paths else "abstained",
            timings=StageTimings(retrieval_ms=ms,total_ms=ms))
        return record,reason


def run(graph_dir: Path, questions: Path = PILOT, top_k=5, output_root=OUTPUT):
    index=G1TypedIndex(graph_dir)
    rows=[
        json.loads(x) for x in Path(questions).read_text(encoding="utf-8").splitlines()
        if x.strip()
    ]
    if not rows or len({r["query_id"] for r in rows})!=len(rows):
        raise ValueError("Missing/duplicate input query IDs")
    results=[]
    for method,multi in (("B5_G1_TYPED",False),("B6_G1_TYPED",True)):
        for row in rows:
            rec,reason=index.retrieve(row["query"],causal_multihop=multi,top_k=top_k,
                                      query_id=row["query_id"],run_id="G1_TYPED")
            results.append({
                "model":method,"query_id":row["query_id"],
                "status":rec.status,"matched_concept":rec.matched_concept or "",
                "reason":reason,"evidence_returned":bool(rec.retrieved_documents or rec.causal_paths),
                "typed_relation_ids":"|".join(d.document_id for d in rec.retrieved_documents),
                "causal_edges":len(rec.causal_edges),"causal_paths":len(rec.causal_paths),
                "retrieval_ms":rec.timings.retrieval_ms,
            })
    summary={
        "mode":"G1_reviewed_candidate_typed_retrieval_only",
        "queries":len(rows),"approved_relation_count":len(index.edges),
        "legacy_G0_edges_used_as_verified_evidence":False,
        "medical_accuracy":False,"top_k":top_k,
        "models":{
            method:{
                "evidence_yield":sum(r["evidence_returned"] for r in results if r["model"]==method),
                "abstentions":sum(r["status"]=="abstained" for r in results if r["model"]==method),
                "causal_paths":sum(r["causal_paths"] for r in results if r["model"]==method),
                "mean_retrieval_ms":statistics.mean(r["retrieval_ms"] for r in results if r["model"]==method),
            } for method in ("B5_G1_TYPED","B6_G1_TYPED")
        },
        "warning":"Source review status is a human assertion. Typed association evidence is not causation; evidence yield is not clinical correctness."
    }
    base=Path(output_root).resolve()
    if not base.is_relative_to(PROJECT_ROOT):raise ValueError("Output must stay in repo")
    out=base/("typed_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")+"_"+uuid.uuid4().hex[:6])
    out.mkdir(parents=True,exist_ok=False)
    with (out/"per_query.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(results[0]))
        w.writeheader();w.writerows(results)
    (out/"summary.json").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"output":str(out),**summary},indent=2))
    return out,summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--graph-dir",type=Path,required=True)
    p.add_argument("--queries",type=Path,default=PILOT)
    p.add_argument("--top-k",type=int,default=5)
    a=p.parse_args()
    if Path.cwd().resolve()!=PROJECT_ROOT:p.error("Run from repository root")
    run(a.graph_dir,a.queries,a.top_k)


if __name__=="__main__":main()
