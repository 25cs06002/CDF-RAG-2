"""Query-only, stability-first G1 B5/B6 retrieval experiment.

Preserve G0's lexical target whenever it retrieves a graph edge in G1.
When that anchor has no graph evidence, permit exactly one explicitly
mentioned NEW G1 concept with the appropriate graph connectivity.
Never use benchmark target labels. Do not alter historical B5/B6.

This is a *behavioral* algorithm experiment with mixed legacy/unverified
G0 + candidate G1 causal-format evidence. It is NOT a clinical accuracy
test or the reviewed-only typed retriever. It is intentionally conservative
and will miss paraphrases until manually supported aliases are added.
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
from .g1_query_interpreter import ABSTAIN_RX, exact_mention
from .pipelines.baselines import BaselinePipeline, G0Corpus

ROOT=PROJECT_ROOT
G0=ROOT/"research_artifacts/graph-G0"
OUTPUT=ROOT/"presentation_20261118/g1_stable_retrieval"
FAMILIES={"B5":"graph_rag", "B6":"causal_kg_rag"}


def flag(value):
    if str(value).lower()=="true":return True
    if str(value).lower()=="false":return False
    raise ValueError("Invalid serialized boolean: "+repr(value))


class StableTargetG1(G0Corpus):
    def __init__(self, graph_dir: Path, *, frozen: G0Corpus, pipeline: str):
        super().__init__(graph_dir=graph_dir)
        if pipeline not in FAMILIES.values():
            raise ValueError("Only B5/B6 graph retrieval is supported")
        self.frozen=frozen
        self.pipeline=pipeline
        self.old_nodes=set(frozen.nodes)
        if not self.old_nodes.issubset(set(self.nodes)):
            raise ValueError("G1 must preserve all frozen G0 concepts")
        self.new_nodes=[node for node in self.nodes if node not in self.old_nodes]
        self.last_reason="not_run"

    def connected(self,concept: str):
        if self.pipeline=="graph_rag":
            return bool(self.neighbors[concept])
        return bool(self.reverse[concept])

    def concept(self,query: str):
        old=self.frozen.concept(query)
        if old and self.connected(old):
            self.last_reason="preserved_G0_anchor_with_evidence"
            return old
        # Only try novel G1 nodes on a clear, query-only exact mention.
        # No metadata/answer labels and no fuzzy similarity on new nodes.
        if ABSTAIN_RX.search(query):
            self.last_reason="under_determined_query_no_new_override"
            return old
        candidates=[node for node in self.new_nodes
                    if exact_mention(query,node) and self.connected(node)]
        # If 'heart damage' and 'damage' both match, we do not infer which
        # was intended. Reviewer-curated aliases/ontology can solve later.
        if len(candidates)==1:
            self.last_reason="one_explicit_connected_new_G1_target"
            return candidates[0]
        if len(candidates)>1:
            self.last_reason="ambiguous_multiple_connected_new_G1_targets"
            return old
        self.last_reason="no_explicit_connected_new_G1_target"
        return old


def _csv(path:Path):
    with path.open(newline="",encoding="utf-8") as f:
        return list(csv.DictReader(f))


def compare(graph_dir:Path,paired_dirs:list[Path],top_k:int=5,
            output_root:Path=OUTPUT):
    graph_dir=Path(graph_dir).expanduser().resolve()
    output_root=Path(output_root).expanduser().resolve()
    paired_dirs=[Path(p).expanduser().resolve() for p in paired_dirs]
    for p in (graph_dir,output_root,*paired_dirs):
        if not p.is_relative_to(ROOT):
            raise ValueError("Inputs and outputs must stay inside repo")
    if top_k<1:raise ValueError("top_k must be positive")
    manifest_path=graph_dir/"manifest.json"
    manifest=json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("stage")!="G1_CANDIDATE_REQUIRES_INDEPENDENT_VERIFICATION":
        raise ValueError("Expected versioned G1 candidate graph")
    frozen=G0Corpus(G0)
    pipelines={}
    corpora={}
    for family,name in FAMILIES.items():
        corpus=StableTargetG1(graph_dir,frozen=frozen,pipeline=name)
        corpora[family]=corpus
        pipelines[family]=BaselinePipeline(name,corpus=corpus,top_k=top_k,generate=False)
    all_rows=[]
    summaries={}
    seen_sets=set()
    for paired in paired_dirs:
        source=json.loads((paired/"summary.json").read_text(encoding="utf-8"))
        if source["graph_manifest_sha256"]!=file_sha256(manifest_path) or source["top_k"]!=top_k:
            raise ValueError("Paired experiment has a different G1 graph or top_k")
        setname=source["questions_file"]
        if setname in seen_sets:raise ValueError("Duplicate question set")
        seen_sets.add(setname)
        if file_sha256(ROOT/setname)!=source["questions_sha256"]:
            raise ValueError("Original question set changed: "+setname)
        input_rows=_csv(paired/"per_query.csv")
        keyed={}
        ordered={}
        for r in input_rows:
            k=(r["model"],r["query_id"])
            if k in keyed:raise ValueError("Duplicated model/query record "+str(k))
            keyed[k]=r
            qid=r["query_id"]
            if qid in ordered and ordered[qid]!=r["question"]:
                raise ValueError("Query text mismatch for "+qid)
            ordered[qid]=r["question"]
        if len(ordered)!=source["query_count"]:
            raise ValueError("Paired input missing questions")
        group={}
        for family in FAMILIES:
            changes=Counter()
            modes=Counter()
            subset=[]
            for qid,question in ordered.items():
                base=keyed[(family+"_G0_LEGACY",qid)]
                expanded=keyed[(family+"_G1_LEGACY",qid)]
                request=PipelineRequest(run_id="G1_STABILITY_SHADOW",
                                        query_id=qid,query=question)
                record=pipelines[family].run(request)
                reason=corpora[family].last_reason
                old_yes=flag(base["evidence_returned"])
                g1_yes=flag(expanded["evidence_returned"])
                stable_yes=bool(record.retrieved_documents or record.causal_paths)
                new_row={
                    "question_set":setname,
                    "query_id":qid,
                    "question":question,
                    "phase2m_graph_construction_seed":base["phase2m_graph_construction_seed"],
                    "model":family+"_G1_STABLE_SHADOW",
                    "g0_target":base["matched_concept"],
                    "original_g1_target":expanded["matched_concept"],
                    "stable_g1_target":record.matched_concept or "",
                    "stable_selection_reason":reason,
                    "g0_evidence":old_yes,
                    "original_g1_evidence":g1_yes,
                    "stable_evidence":stable_yes,
                    "g0_to_stable_change":(
                        "gained" if stable_yes and not old_yes else
                        "lost" if old_yes and not stable_yes else
                        "both" if old_yes else "neither"
                    ),
                    "original_G1_loss_recovered":old_yes and not g1_yes and stable_yes,
                    "stable_evidence_ids":"|".join(d.document_id for d in record.retrieved_documents),
                    "stable_paths":len(record.causal_paths),
                    "stable_retrieval_ms":record.timings.retrieval_ms,
                    "needs_relevance_review":stable_yes,
                    "evidence_provenance":"mixed_inherited_G0_unverified_and_new_G1_candidate",
                }
                subset.append(new_row)
                all_rows.append(new_row)
                modes[reason]+=1
                changes[new_row["g0_to_stable_change"]]+=1
            group[family]={
                "queries":len(subset),
                "g0_evidence_yield":sum(r["g0_evidence"] for r in subset),
                "legacy_g1_evidence_yield":sum(r["original_g1_evidence"] for r in subset),
                "stable_g1_evidence_yield":sum(r["stable_evidence"] for r in subset),
                "recovered_original_G1_losses":sum(r["original_G1_loss_recovered"] for r in subset),
                "net_changes_vs_G0":dict(changes),
                "selection_reasons":dict(modes),
                "new_explicit_target_evidence":sum(
                    r["stable_evidence"] and
                    r["stable_selection_reason"]=="one_explicit_connected_new_G1_target"
                    for r in subset),
                "twohop_query_count":sum(int(r["stable_paths"])>0 for r in subset),
            }
        summaries[setname]=group
    out=output_root/("stable_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")+"_"+uuid.uuid4().hex[:6])
    out.mkdir(parents=True,exist_ok=False)
    with (out/"per_query.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(all_rows[0]))
        w.writeheader();w.writerows(all_rows)
    review=[r for r in all_rows if r["g0_to_stable_change"] in {"gained","lost"}]
    with (out/"new_change_review_queue.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(all_rows[0]))
        w.writeheader();w.writerows(review)
    report={
        "mode":"G1_legacy_stable_anchor_plus_explicit_new_targets",
        "graph_manifest_sha256":file_sha256(manifest_path),
        "top_k":top_k,
        "per_set":summaries,
        "query_model_rows":len(all_rows),
        "new_change_review_queue_rows":len(review),
        "medical_accuracy":False,"gold_labels_used_for_inference":False,
        "inherited_G0_edges_unverified":manifest.get("inherited_G0_edges_unverified"),
        "reviewed_only_typed":False,
        "warnings":[
            "Stable retrieval is a separate shadow algorithm, not legacy B5/B6 and not an independently validated model.",
            "Preserving baseline evidence does not prove relevance or clinical correctness.",
            "G1 legacy causal-format edges inherit unverified G0 statements; this does not satisfy a verified-only evidence policy.",
            "Construction-seed and pilot draft questions are not independent held-out evaluation.",
            "New exact mention coverage is deliberately conservative; aliases require source review."
        ],
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
    compare(a.graph_dir,a.paired_dirs,a.top_k,a.output_root)


if __name__=="__main__":
    main()
