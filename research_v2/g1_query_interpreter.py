"""Query-only graph target selection + shadow retrieval (no benchmark labels).

B5/B6 and G0 remain untouched; optionally run against a review-gated G1
candidate, where only reviewed CAUSES edges are treated as causal edges.
Returns zero evidence when a target is unsupported, ambiguous or absent.
"""
from __future__ import annotations
import argparse
import csv
import json
import re
import statistics
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from .base_pipeline import PipelineRequest
from .experiment_runner import PROJECT_ROOT, file_sha256
from .pipelines.baselines import G0Corpus, BaselinePipeline
from .presentation_matrix import WORKSPACE

G0 = PROJECT_ROOT/"research_artifacts/graph-G0"
EXPANSION = PROJECT_ROOT/"data/benchmark/medical_causal_expansion32_draft_v1.jsonl"
PILOT = WORKSPACE/"benchmark/questions_only.jsonl"
ABSTAIN_RX=re.compile(
    r"\b(exact (?:disease|diagnosis)|definitely (?:have|present)|"
    r"without (?:any )?(?:symptoms|measurements|records)|"
    r"predict the exact disease|diagnose a specific|"
    r"from (?:their occupation|this alone)|"
    r"certain\??)\b",re.I)


def exact_mention(question,phrase):
    return bool(re.search(r"(?<![a-z0-9])"+re.escape(phrase.casefold())+
                          r"(?![a-z0-9])",question.casefold()))


def load_approved_aliases(path, nodes):
    if path is None:return {}
    path=Path(path).resolve()
    with path.open(newline="",encoding="utf-8") as f:
        rows=list(csv.DictReader(f))
    approved={}
    canonical={n.casefold():n for n in nodes}
    for row in rows:
        if row.get("review_status")!="approved":continue
        target=canonical.get(row.get("canonical_concept","").casefold())
        alias=row.get("alias","").strip().casefold()
        if not alias or not target or not row.get("source_id") or not row.get("reviewer"):
            raise ValueError("Approved alias missing canonical/source/reviewer")
        if alias in canonical or alias in approved:
            raise ValueError("Conflicting approved alias")
        approved[alias]=target
    return approved


class InterpretedGraph(G0Corpus):
    def __init__(self, graph_dir=G0, alias_file=None):
        super().__init__(graph_dir=graph_dir)
        self.approved_aliases=load_approved_aliases(alias_file,self.nodes)
        self.interpretation="init"

    def concept(self,question):
        self.interpretation="no_explicit_supported_target"
        if ABSTAIN_RX.search(question):
            self.interpretation="insufficient_information_request"
            return None
        mentions=[]
        for node in self.nodes:
            # A subtype is NOT equivalent to its parent label; avoid silently
            # treating 'type 2 diabetes' as the broad G0 'Diabetes' concept.
            if node.casefold()=="diabetes" and re.search(
                r"\btype\s*[12]\s*diabetes\b",question,re.I):
                continue
            if exact_mention(question,node) and self.reverse[node]:
                mentions.append(node)
        for alias,target in self.approved_aliases.items():
            if exact_mention(question,alias) and self.reverse[target]:
                mentions.append(target)
        mentions=list(dict.fromkeys(mentions))
        if len(mentions)==1:
            self.interpretation="unique_explicit_causal_target"
            return mentions[0]
        if len(mentions)>1:
            self.interpretation="multiple_target_mentions_abstain"
            return None
        return None


def _query_rows(path):
    rows=[json.loads(s) for s in Path(path).read_text(encoding="utf-8").splitlines() if s.strip()]
    ids=[r["query_id"] for r in rows]
    if len(ids)!=len(set(ids)) or not rows:raise ValueError("Empty or duplicate query IDs")
    return rows


def probe(graph_dir=G0, queries=PILOT, alias_file=None, output_root=None, top_k=5):
    graph_dir=Path(graph_dir).resolve()
    if not graph_dir.is_relative_to(PROJECT_ROOT):raise ValueError("Graph must stay in project")
    if graph_dir!=G0:
        manifest=json.loads((graph_dir/"manifest.json").read_text(encoding="utf-8"))
        if manifest.get("stage")!="G1_CANDIDATE_REQUIRES_INDEPENDENT_VERIFICATION":
            raise ValueError("Only explicitly versioned G1 candidate is permitted")
    if top_k<1:raise ValueError("top_k must be positive")
    questions=_query_rows(queries)
    original=G0Corpus(graph_dir=graph_dir)
    improved=InterpretedGraph(graph_dir=graph_dir,alias_file=alias_file)
    rows=[]
    for p in ("graph_rag","causal_kg_rag"):
        b=BaselinePipeline(p,corpus=original,top_k=top_k,generate=False)
        v=BaselinePipeline(p,corpus=improved,top_k=top_k,generate=False)
        for q in questions:
            req=PipelineRequest(run_id="G1_SHADOW",query_id=q["query_id"],query=q["query"])
            old=b.run(req); new=v.run(req)
            rows.append({
                "query_id":q["query_id"],"pipeline":p,
                "old_target":old.matched_concept or "",
                "new_target":new.matched_concept or "",
                "interpretation_reason":improved.interpretation,
                "old_evidence":bool(old.retrieved_documents or old.causal_edges),
                "new_evidence":bool(new.retrieved_documents or new.causal_edges),
                "old_edges":len(old.causal_edges),"new_edges":len(new.causal_edges),
                "old_retrieval_ms":old.timings.retrieval_ms,
                "new_retrieval_ms":new.timings.retrieval_ms
            })
    metrics=[]
    for pipeline in ("graph_rag","causal_kg_rag"):
        subset=[r for r in rows if r["pipeline"]==pipeline]
        metrics.append({
            "pipeline":pipeline,"queries":len(subset),
            "old_evidence":sum(r["old_evidence"] for r in subset),
            "shadow_evidence":sum(r["new_evidence"] for r in subset),
            "recovered":sum(not r["old_evidence"] and r["new_evidence"] for r in subset),
            "lost":sum(r["old_evidence"] and not r["new_evidence"] for r in subset),
            "shadow_abstentions":sum(not r["new_evidence"] for r in subset),
            "mean_shadow_retrieval_ms":statistics.mean(r["new_retrieval_ms"] for r in subset),
        })
    dest=Path(output_root or WORKSPACE/"g1_query_interpretation").resolve()
    if not dest.is_relative_to(PROJECT_ROOT):raise ValueError("Output must stay in project")
    out=dest/("probe_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")+"_"+uuid.uuid4().hex[:6])
    out.mkdir(parents=True,exist_ok=False)
    with (out/"per_query.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summary={
        "mode":"no_ollama_no_neo4j_no_PPO",
        "graph_version":"G0" if graph_dir==G0 else "G1_CANDIDATE",
        "graph_nodes":len(original.nodes),"graph_causal_edges":len(original.edges),
        "query_count":len(questions),"queries_sha256":file_sha256(queries),
        "alias_count":len(improved.approved_aliases),
        "reference_labels_used_for_inference":False,
        "medical_accuracy":False,"pipeline_metrics":metrics,
        "warning":"Evidence yield is not medical accuracy. A conservative target resolver can abstain more often; assess relevance manually."
    }
    (out/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps({"output":str(out),**summary},indent=2))
    return out,summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--graph-dir",type=Path,default=G0)
    p.add_argument("--queries",type=Path,default=PILOT)
    p.add_argument("--alias-file",type=Path)
    p.add_argument("--top-k",type=int,default=5)
    a=p.parse_args()
    if Path.cwd().resolve()!=PROJECT_ROOT:p.error("Run from project root")
    probe(a.graph_dir,a.queries,a.alias_file,top_k=a.top_k)


if __name__=="__main__":main()
