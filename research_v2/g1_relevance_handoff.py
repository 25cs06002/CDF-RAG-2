"""Review handoff for changed G1 stable retrieval cases, no automatic gold.

Joins frozen paired and stable runs to map each returned document ID to a
graph edge, provenance and (for typed G1) approved source-ID crosswalk.
Produces *manual* case and evidence review worksheets. The audit merely
checks annotation completeness and never creates medical gold/accuracy.
No Ollama, Neo4j, PPO or offline-sprint tests are run.
"""
from __future__ import annotations

import argparse
import csv
import json
import uuid
from collections import Counter
from datetime import date, datetime, timezone
from pathlib import Path

from .experiment_runner import PROJECT_ROOT, file_sha256

ROOT=PROJECT_ROOT
G0=ROOT/"research_artifacts/graph-G0"
OUT=ROOT/"presentation_20261118/g1_relevance_review"
CASE_FIELDS=(
    "case_id","question_set","query_id","question",
    "phase2m_graph_construction_seed","family",
    "g0_target","g1_legacy_target","g1_stable_target","stable_selection_reason",
    "g0_evidence","g1_legacy_evidence","g1_stable_evidence","g1_typed_evidence",
    "original_G1_change","g0_to_stable_change",
    "review_query_intent","review_stable_target","reviewer","review_date","review_note",
)
EVIDENCE_FIELDS=(
    "case_id","query_id","family","stage","document_id",
    "subject","relation","object","evidence_provenance",
    "original_supporting_source_ids","g1_supporting_source_ids","source_urls",
    "reviewer_paraphrase","review_evidence_relevance","review_relation_direction",
    "review_source_checked","reviewer","review_date","review_note",
)
INTENT={"answerable","ambiguous","unsupported","unknown"}
TARGET={"appropriate","wrong","uncertain","none_expected"}
RELEVANCE={"relevant","irrelevant","uncertain"}
DIRECTION={"correct","incorrect","uncertain"}
CHECKED={"checked","not_checked","not_applicable"}


def load(path):
    with path.open(newline="",encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write(path,columns,rows):
    with path.open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=columns)
        writer.writeheader();writer.writerows(rows)


def make_edge_id(position):
    return f"G0_EDGE_{position:04d}"


def bool_text(x):
    if str(x).lower()=="true":return True
    if str(x).lower()=="false":return False
    raise ValueError("Expected serialized bool: "+repr(x))


def approved_graph(graph_dir):
    manifest=json.loads((graph_dir/"manifest.json").read_text(encoding="utf-8"))
    if manifest.get("stage")!="G1_CANDIDATE_REQUIRES_INDEPENDENT_VERIFICATION":
        raise ValueError("Expected versioned G1 candidate graph")
    g0_edges=load(G0/"edges.csv")
    g1_edges=load(graph_dir/"edges.csv")
    if len(g1_edges)<len(g0_edges):
        raise ValueError("G1 has fewer edges than frozen G0")
    for old,new in zip(g0_edges,g1_edges):
        if old["cause"]!=new["cause"] or old["effect"]!=new["effect"]:
            raise ValueError("G1 does not preserve frozen G0 edge ordering")
    if manifest.get("inherited_G0_edges_unverified")!=len(g0_edges):
        raise ValueError("Frozen inherited edge count differs")
    if manifest.get("total_causes_edges")!=len(g1_edges):
        raise ValueError("G1 edge count differs")
    if file_sha256(graph_dir/"typed_relationships.csv")!=manifest.get("edge_ledger_sha256"):
        raise ValueError("G1 typed relationship checksum differs")
    edge_by_doc={}
    for i,edge in enumerate(g0_edges,1):
        edge_by_doc[("G0",make_edge_id(i))]={
            "subject":edge["cause"],"relation":"G0_UNVERIFIED_CAUSAL_EDGE",
            "object":edge["effect"],"evidence_provenance":"inherited_G0_unverified",
            "original_supporting_source_ids":"","g1_supporting_source_ids":"",
            "source_urls":"","reviewer_paraphrase":"",
        }
    review_root=ROOT/manifest["review_workspace"]
    source_lookup={r["source_id"]:r for r in load(review_root/"source_review.csv")}
    provenance={r["g1_edge_id"]:r for r in
                load(review_root/"phase2m_edge_provenance.csv")}
    typed_edges=load(graph_dir/"typed_relationships.csv")
    by_pair={}
    for edge in typed_edges:
        if edge["review_status"]!="approved":
            raise ValueError("G1 typed snapshot contains unapproved relationship")
        key=(edge["subject"].strip().casefold(),edge["object"].strip().casefold())
        by_pair.setdefault(key,[]).append(edge)
        maprow=provenance.get(edge["edge_id"])
        source_ids=(maprow["g1_supporting_source_ids"].split("|") if maprow else [edge["source_id"]])
        originals=(maprow["original_supporting_source_ids"] if maprow else "")
        urls="|".join(dict.fromkeys(
            source_lookup.get(sid,{}).get("source_url","") for sid in source_ids if sid
        ))
        edge_by_doc[("TYPED",edge["edge_id"])]={
            "subject":edge["subject"],"relation":edge["relation"],"object":edge["object"],
            "evidence_provenance":"G1_approved_typed_candidate",
            "original_supporting_source_ids":originals,
            "g1_supporting_source_ids":"|".join(source_ids),
            "source_urls":urls,"reviewer_paraphrase":edge["support_summary"],
        }
    for i,edge in enumerate(g1_edges,1):
        key=(edge["cause"].strip().casefold(),edge["effect"].strip().casefold())
        if i<=len(g0_edges):
            edge_by_doc[("G1",make_edge_id(i))]=edge_by_doc[("G0",make_edge_id(i))].copy()
            continue
        supported=[r for r in by_pair.get(key,[]) if r["relation"]=="CAUSES"]
        if not supported:
            raise ValueError("New G1 causal graph edge has no approved typed CAUSES provenance")
        approved=supported[0]
        provenance_row=edge_by_doc[("TYPED",approved["edge_id"])]
        edge_by_doc[("G1",make_edge_id(i))]={
            **provenance_row,
            "evidence_provenance":"G1_new_candidate_CAUSES",
        }
    return edge_by_doc,manifest


def prepare(graph_dir,stable_dir,paired_dirs,scope="stable_changes",
            output_root=OUT):
    graph_dir=Path(graph_dir).resolve()
    stable_dir=Path(stable_dir).resolve()
    paired_dirs=[Path(x).resolve() for x in paired_dirs]
    output_root=Path(output_root).resolve()
    for path in (graph_dir,stable_dir,output_root,*paired_dirs):
        if not path.is_relative_to(ROOT):raise ValueError("Paths must stay inside repo")
    if scope not in {"stable_changes","any_changes","all"}:
        raise ValueError("Invalid scope")
    edge_by_doc,graph_manifest=approved_graph(graph_dir)
    stable_summary=json.loads((stable_dir/"summary.json").read_text(encoding="utf-8"))
    graph_hash=file_sha256(graph_dir/"manifest.json")
    if stable_summary["graph_manifest_sha256"]!=graph_hash:
        raise ValueError("Stable run does not use requested graph")
    paired_by_set={}
    for folder in paired_dirs:
        summary=json.loads((folder/"summary.json").read_text(encoding="utf-8"))
        if summary["graph_manifest_sha256"]!=graph_hash:
            raise ValueError("Paired run graph mismatch")
        if summary["top_k"]!=stable_summary["top_k"]:
            raise ValueError("Paired run top_k mismatch")
        name=summary["questions_file"]
        if name in paired_by_set:raise ValueError("Duplicate question set")
        if file_sha256(ROOT/name)!=summary["questions_sha256"]:
            raise ValueError("Question file changed after original paired run")
        rows=load(folder/"per_query.csv")
        indexed={}
        for row in rows:
            key=(row["query_id"],row["model"])
            if key in indexed:raise ValueError("Duplicate query-model in paired run")
            indexed[key]=row
        paired_by_set[name]=indexed
    cases=[]
    evidence=[]
    selected_ids=set()
    for row in load(stable_dir/"per_query.csv"):
        question_set=row["question_set"]
        if question_set not in paired_by_set:
            raise ValueError("Stable question set lacks paired baseline: "+question_set)
        changed_stable=row["g0_to_stable_change"] in {"gained","lost"}
        changed_original=row["original_G1_change"] in {"gained","lost"}
        if scope=="stable_changes" and not changed_stable:continue
        if scope=="any_changes" and not (changed_stable or changed_original):continue
        family=row["model"].split("_",1)[0]
        if family not in {"B5","B6"}:raise ValueError("Invalid stable model")
        case_id=family+"_"+row["query_id"]
        if case_id in selected_ids:raise ValueError("Duplicate case ID")
        selected_ids.add(case_id)
        models=paired_by_set[question_set]
        baseline=models[(row["query_id"],family+"_G0_LEGACY")]
        expanded=models[(row["query_id"],family+"_G1_LEGACY")]
        typed=models[(row["query_id"],family+"_G1_TYPED")]
        for old in (baseline,expanded,typed):
            if old["question"]!=row["question"]:
                raise ValueError("Question mismatch between runs")
        cases.append({
            "case_id":case_id,
            "question_set":question_set,"query_id":row["query_id"],
            "question":row["question"],
            "phase2m_graph_construction_seed":row["phase2m_graph_construction_seed"],
            "family":family,"g0_target":row["g0_target"],
            "g1_legacy_target":row["original_g1_target"],
            "g1_stable_target":row["stable_g1_target"],
            "stable_selection_reason":row["stable_selection_reason"],
            "g0_evidence":row["g0_evidence"],
            "g1_legacy_evidence":row["original_g1_evidence"],
            "g1_stable_evidence":row["stable_evidence"],
            "g1_typed_evidence":typed["evidence_returned"],
            "original_G1_change":row["original_G1_change"],
            "g0_to_stable_change":row["g0_to_stable_change"],
            "review_query_intent":"","review_stable_target":"",
            "reviewer":"","review_date":"","review_note":"",
        })
        stages=(
            ("G0_LEGACY",baseline["evidence_ids"],"G0"),
            ("G1_LEGACY",expanded["evidence_ids"],"G1"),
            ("G1_STABLE",row["stable_evidence_ids"],"G1"),
            ("G1_TYPED",typed["evidence_ids"],"TYPED"),
        )
        for stage,ids_text,which in stages:
            ids=[x for x in ids_text.split("|") if x]
            if len(ids)!=len(set(ids)):
                raise ValueError("Duplicate returned evidence in "+case_id+" "+stage)
            for doc in ids:
                if (which,doc) not in edge_by_doc:
                    raise ValueError("Unrecognized evidence ID "+doc+" in "+case_id+" "+stage)
                info=edge_by_doc[(which,doc)]
                evidence.append({
                    "case_id":case_id,"query_id":row["query_id"],
                    "family":family,"stage":stage,"document_id":doc,
                    **info,"review_evidence_relevance":"",
                    "review_relation_direction":"","review_source_checked":"",
                    "reviewer":"","review_date":"","review_note":"",
                })
    if not cases:raise ValueError("No cases selected")
    dest=output_root/("review_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")+
                     "_"+uuid.uuid4().hex[:6])
    dest.mkdir(parents=True,exist_ok=False)
    write(dest/"review_cases.csv",CASE_FIELDS,cases)
    write(dest/"review_evidence.csv",EVIDENCE_FIELDS,evidence)
    metadata={
        "mode":"manual_relevance_and_target_review_not_medical_gold",
        "graph":str(graph_dir.relative_to(ROOT)),
        "graph_manifest_sha256":graph_hash,
        "stable_dir":str(stable_dir.relative_to(ROOT)),
        "stable_summary_sha256":file_sha256(stable_dir/"summary.json"),
        "stable_per_query_sha256":file_sha256(stable_dir/"per_query.csv"),
        "paired_dirs":[str(p.relative_to(ROOT)) for p in paired_dirs],
        "paired_summary_sha256":[file_sha256(p/"summary.json") for p in paired_dirs],
        "scope":scope,"review_case_count":len(cases),
        "distinct_query_count":len({c["query_id"] for c in cases}),
        "evidence_review_rows":len(evidence),
        "evidence_provenance":dict(Counter(e["evidence_provenance"] for e in evidence)),
        "no_auto_approvals":True,"medical_gold":False,
        "warnings":[
            "Graph records inherited from G0 are unverified even if a new matcher returns them.",
            "Review the actual query intent and cited source passages; do not mark accuracy from yield.",
            "The 15 Phase2M construction-seed questions are not holdout cases.",
            "These worksheets are for reviewed sample errors, not automatically representative accuracy on all 132 queries.",
        ],
    }
    (dest/"manifest.json").write_text(json.dumps(metadata,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"output":str(dest),"review_cases":len(cases),
        "distinct_queries":metadata["distinct_query_count"],
        "evidence_rows":len(evidence),
        "provenance_counts":metadata["evidence_provenance"]},indent=2))
    return dest


def valid_date(value):
    try:
        return bool(value) and date.fromisoformat(value)<=datetime.now(timezone.utc).date()
    except (ValueError,TypeError):
        return False


def audit(folder,require_complete=False):
    folder=Path(folder).resolve()
    if not folder.is_relative_to(ROOT):raise ValueError("Review directory outside repository")
    manifest=json.loads((folder/"manifest.json").read_text(encoding="utf-8"))
    cases=load(folder/"review_cases.csv")
    evidence=load(folder/"review_evidence.csv")
    if len(cases)!=manifest["review_case_count"] or len(evidence)!=manifest["evidence_review_rows"]:
        raise ValueError("Worksheet row counts changed")
    case_ids={c["case_id"] for c in cases}
    if len(case_ids)!=len(cases):raise ValueError("Duplicate case IDs")
    if any(e["case_id"] not in case_ids for e in evidence):
        raise ValueError("Evidence references unknown case")
    warnings=[]
    done_case=0
    done_evidence=0
    for c in cases:
        values=(c["review_query_intent"],c["review_stable_target"])
        any_label=any(values) or c["reviewer"] or c["review_date"] or c["review_note"]
        if not any_label:continue
        if (c["review_query_intent"] not in INTENT or
            c["review_stable_target"] not in TARGET or
            not c["reviewer"].strip() or
            not valid_date(c["review_date"]) or
            not c["review_note"].strip()):
            warnings.append("Incomplete or invalid case review "+c["case_id"])
        else:done_case+=1
    for e in evidence:
        values=(e["review_evidence_relevance"],e["review_relation_direction"],
                e["review_source_checked"])
        any_label=any(values) or e["reviewer"] or e["review_date"] or e["review_note"]
        if not any_label:continue
        if (e["review_evidence_relevance"] not in RELEVANCE or
            e["review_relation_direction"] not in DIRECTION or
            e["review_source_checked"] not in CHECKED or
            not e["reviewer"].strip() or not valid_date(e["review_date"]) or
            not e["review_note"].strip()):
            warnings.append("Incomplete or invalid evidence review "+
                            e["case_id"]+" "+e["stage"]+" "+e["document_id"])
        else:done_evidence+=1
    report={
        "case_rows":len(cases),"reviewed_cases":done_case,
        "evidence_rows":len(evidence),"reviewed_evidence_rows":done_evidence,
        "incomplete_cases":len(cases)-done_case,
        "incomplete_evidence_rows":len(evidence)-done_evidence,
        "review_warnings":warnings,
        "all_reviewed":done_case==len(cases) and done_evidence==len(evidence),
        "medical_gold":False,
        "accuracy_not_computed":"Annotations limited to changed/selected cases; not an independent held-out test.",
    }
    (folder/"review_audit.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(report,indent=2))
    if require_complete and not report["all_reviewed"]:
        raise SystemExit("Manual evidence review remains incomplete")
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    subs=p.add_subparsers(dest="command",required=True)
    prep=subs.add_parser("prepare")
    prep.add_argument("--graph-dir",type=Path,required=True)
    prep.add_argument("--stable-dir",type=Path,required=True)
    prep.add_argument("--paired-dirs",type=Path,nargs="+",required=True)
    prep.add_argument("--scope",choices=("stable_changes","any_changes","all"),
                      default="stable_changes")
    prep.add_argument("--output-root",type=Path,default=OUT)
    check=subs.add_parser("audit")
    check.add_argument("--folder",type=Path,required=True)
    check.add_argument("--require-complete",action="store_true")
    args=p.parse_args()
    if Path.cwd().resolve()!=ROOT:p.error("Run from repository root")
    if args.command=="prepare":
        prepare(args.graph_dir,args.stable_dir,args.paired_dirs,args.scope,args.output_root)
    else:audit(args.folder,args.require_complete)


if __name__=="__main__":
    main()
