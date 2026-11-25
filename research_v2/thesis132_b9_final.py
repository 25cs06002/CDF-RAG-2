"""Final B9 archive audit, semantic proxies, and deduplicated AI-review handoff.
No inference, no clinical gold. Only exact answer/context matches share reviews.
"""
from __future__ import annotations
import argparse,hashlib,json,math,os,statistics,zipfile,shutil
from pathlib import Path
from collections import Counter
from research_v2.thesis132_b9_review import (
    ROOT,RUN,BASE,META,CORPUS,GRAPH,CATALOG,FIELDS,PROTOCOL,STATEMENT,LABELS,CLAIMS,AUDIT,
    readcsv,readjsonl,writecsv,sha,dump,stamp,need,validate,validate_row,aggregate)
from compare_b9_versions_132 import V1,Q
from research_v2.thesis132_b9_v2 import fingerprints
from research_v2.sentence_model_assets import resolve_cached_snapshot,fingerprint_snapshot
T=ROOT/"presentation_20261118/thesis132"
OUT=T/"b9_final_evaluation_v2"
PACKAGE=RUN/"ai_review_package_20261125T184530034418Z"
REVIEW=RUN/"ai_review_v1"
FULL=BASE/"full_metrics_132_20261123T115629452212Z_2879fc"
CONSOLIDATED=REVIEW/"evaluation_b9_132_20261125T191520949435Z"
RUNS={
 "B9_v1":V1,
 "B9_v2":RUN,
 "fixed1":T/"b9_end_to_end_v2/b9_v2_fixed1_20261125T192322027622Z_c03d30",
 "fixed3":T/"b9_end_to_end_v2/b9_v2_fixed3_20261125T193008968650Z_81f3ea",
 "graph_off":T/"b9_component_ablation_v2/graph_off_20261125T194501492831Z_4c5d64",
 "document_off":T/"b9_component_ablation_v2/document_off_20261125T195928994629Z_2e3c5b",
 "original_ids":T/"b9_inference_optimization_v2/original_ids_20261125T200608339047Z_c293b2",
 "compact_ids":T/"b9_inference_optimization_v2/compact_ids_20261125T201909732485Z_07af85",
 "llama3b":T/"b9_generator_comparison_v2/llama3b_20261125T203941977277Z_cc2218",
 "llama1b":T/"b9_generator_comparison_v2/llama1b_20261125T204921199608Z_748c9e",
 "qwen15b":T/"b9_generator_comparison_v2/qwen15b_20261125T210017051361Z_774d20",
}
NEW_LABELS="ai_variant_labels.csv"
NEW_CLAIMS="ai_variant_claims.jsonl"
NEW_AUDIT="ai_variant_audit.json"
def canonical(obj):
    return json.dumps(obj,sort_keys=True,ensure_ascii=False,separators=(",",":"))
def hashobj(obj):return hashlib.sha256(canonical(obj).encode()).hexdigest()
def writejsonl(path,rows):
    path.write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in rows),encoding="utf-8")
def mean(xs):
    xs=[v for v in xs if v is not None];return statistics.mean(xs) if xs else None

def archived():
    questions=readjsonl(Q);expected=[(r["query_id"],r["query"]) for r in questions]
    need(len(expected)==132 and len(set(expected))==132,"Frozen cohort mismatch")
    meta={r["query_id"]:r for r in readcsv(META)}
    base=json.loads((RUN/"manifest.json").read_text())
    need(fingerprints(GRAPH)==base["inputs"],"Frozen graph/corpus/code drift")
    rows=[];hashes={}
    for name,path in RUNS.items():
        m=json.loads((path/"manifest.json").read_text())
        need(m["status"]=="COMPLETE_132" and m["questions"]==132 and m["dataset_sha256"]==sha(Q),"Incomplete or different cohort: "+name)
        need(m.get("reference_labels_used_for_inference",False) is False,"Reference leakage")
        for f,h in m["outputs_sha256"].items():
            need(Path(f).name==f and sha(path/f)==h,"Archive changed: "+name+"/"+f)
        for k,v in m["inputs"].items():
            need(base["inputs"].get(k)==v,"Frozen inputs differ: "+name+"/"+k)
        rr=readjsonl(path/"records.jsonl");tt=readjsonl(path/"round_traces.jsonl")
        need(len(rr)==len(tt)==132 and [(r["query_id"],r["original_query"]) for r in rr]==expected,"Record alignment: "+name)
        summary=json.loads((path/"summary.json").read_text())
        need(summary["errors"]==0,"Errors in "+name)
        hashes[name]=sha(path/"manifest.json")
        for r,t in zip(rr,tt):
            q=r["query_id"];a=r["final_answer"] or ""
            need(q==t["query_id"] and r["status"] in ("ok","abstained") and bool(a)==(r["status"]=="ok"),"Status/trace drift: "+name)
            scope=t.get("answer_scope")
            if scope is None:scope="v1_answer" if a else "none"
            rows.append(dict(run=name,model="B9",query_id=q,query=r["original_query"],
                final_response=a,final_response_sha256=hashlib.sha256(a.encode()).hexdigest(),
                pipeline_status=r["status"],query_type=meta[q]["query_type"],medical_domain=meta[q]["medical_domain"],
                answer_scope=scope,retrieved_documents=r["retrieved_documents"],causal_edges=r["causal_edges"],causal_paths=r["causal_paths"],
                selected_ids=t["selected_ids"],selected_evidence=t["selected_evidence"],
                refined_query=r.get("refined_query"),rewritten_knowledge=r.get("rewritten_knowledge"),
                latency_ms=r["timings"]["total_ms"]))
    need(len(rows)==len(RUNS)*132,"Archive row count")
    return rows,hashes

# Conservative identity: retain ordering, evidence text, source, and scores.
# Exclude run identity and measured latency; these do not change medical content.
KEYS=("query_id","query","final_response","pipeline_status","answer_scope","retrieved_documents","causal_edges","causal_paths","selected_ids","selected_evidence")
def signature(r):return hashobj({k:r[k] for k in KEYS})

def review_plan():
    rows,hashes=archived()
    _,old=validate(RUN,PACKAGE,REVIEW)
    prior={r["query_id"]:r for r in old}
    original=readjsonl(PACKAGE/"B9_review_inputs_132.jsonl")
    oldkeys={signature(r):r["query_id"] for r in original}
    need(len(oldkeys)==132,"Duplicate existing review identity")
    cases={};links=[]
    for r in rows:
        sig=signature(r)
        if sig in oldkeys:
            source="EXACT_PRIOR_AI_REVIEW";case_id=oldkeys[sig]
        else:
            source="NEW_AI_REVIEW";case_id="R_"+sig
            cases.setdefault(case_id,{**{k:r[k] for k in KEYS}, "model":"B9","review_id":case_id,
                "final_response_sha256":r["final_response_sha256"],"query_type":r["query_type"],"medical_domain":r["medical_domain"]})
        links.append(dict(run=r["run"],query_id=r["query_id"],review_id=case_id,review_source=source,identity_sha256=sig,answer_sha256=r["final_response_sha256"]))
    return rows,hashes,prior,cases,links

PROMPT="""# B9 experiment variants: deduplicated external AI review
Review every unique record in review_cases.jsonl using the same AIJ-MED-HALLUC v1
protocol as the previously reviewed B9-v2. This is a development-exposed medical
RAG experiment, not independent clinical validation. Variant names and timing
are withheld; all cases use model=B9. review_id uniquely identifies a case;
query_id can repeat. Do not skip any case. Do not infer correctness from citations.
The original protocol is attached as PRIOR_PROTOCOL.md; follow its claim,
grounding, completeness and uncertainty rules, with the output schema below
instead of its original fixed-132 instructions. Preserve exact query, answer,
IDs and answer SHA256. Prior annotations were single-AI judgments and source
access was limited; use the same sentence-level claim granularity (split
independent assertions when needed) and report actual source access.
Check each medically material claim against authoritative sources actually
inspected; distinguish causes, association, risk, triggers and subtypes.
Do not fabricate evidence, quotations, hashes or human approvals. Empty
answers have no factual claims and hallucination No; separately assess
abstention appropriateness. Partial pathways are not complete by default.
A true but irrelevant answer is not necessarily a hallucination.

Return THREE UTF-8 files:
1. ai_variant_labels.csv, exact header in review_label_template.csv. Fill
every row, preserve review_id, model, query_id, final_response_sha256,
protocol_version and provenance. Use AI_AUTO_ACCEPTED or AI_UNRESOLVED and
NOT_REVIEWED, with Yes/No/Uncertain according to PRIOR_PROTOCOL.md.
2. ai_variant_claims.jsonl, one object per review_id. Fields: review_id,
model, query_id, query, final_response, final_response_sha256, ai_final_label,
ai_review_status, human_review_status, confidence, evidence_grounding_status,
answer_completeness, fabricated_citation (boolean), claims (array). Each
claim: claim, verdict, reasoning, evidence (array of objects: source_url,
source_title, access_mode, passage_or_finding, evidence_kind [exact_quote
or paraphrase], supports_or_contradicts). Follow the previous protocol's
claim counts and evidence requirements.
3. ai_variant_audit.json: protocol_version, evaluator, annotation_date,
human_clinical_review="none", scope={"total_responses":N,"records_processed":N}
where N is the actual number of cases; review_input_sha256 (SHA256 of exact
review_cases.jsonl bytes), output_sha256 (mapping of the two actual output
filenames to SHA256), access_limitations (array), label_provenance_statement
(exact template string). Do not invent hashes; compute with a file tool.
Return partial work honestly if incomplete. Use a batch workflow if needed.
No summary CSV required: the local validator calculates all aggregates.
"""
def prepare(output=None):
    rows,hashes,prior,cases,links=review_plan()
    dest=Path(output).resolve() if output else OUT/("review_package_"+stamp())
    dest.mkdir(parents=True,exist_ok=False)
    writejsonl(dest/"review_cases.jsonl",[cases[k] for k in sorted(cases)])
    fields=["review_id",*FIELDS]
    writecsv(dest/"review_label_template.csv",[dict(review_id=k,model="B9",query_id=r["query_id"],final_response_sha256=r["final_response_sha256"],human_review_status="NOT_REVIEWED",protocol_version=PROTOCOL,label_provenance_statement=STATEMENT) for k,r in sorted(cases.items())],fields)
    writecsv(dest/"local_review_mapping.csv",links)
    (dest/"README_MASTER_PROMPT.md").write_text(PROMPT,encoding="utf-8")
    shutil.copyfile(PACKAGE/"README_MASTER_PROMPT.md",dest/"PRIOR_PROTOCOL.md")
    for p in (CORPUS,CORPUS.parent/"provenance.jsonl",GRAPH/"typed_relationships.csv",GRAPH/"nodes.csv",CATALOG):
        shutil.copyfile(p,dest/p.name)
    reused=sum(r["review_source"]=="EXACT_PRIOR_AI_REVIEW" for r in links)
    manifest=dict(schema="B9_FINAL_REVIEW_V1",run_manifest_sha256=hashes,prior_audit_sha256=sha(REVIEW/AUDIT),
        total_outputs=len(rows),prior_reused_outputs=reused,new_unique_cases=len(cases),new_review_mapped_outputs=len(rows)-reused,
        review_input_sha256=sha(dest/"review_cases.jsonl"),input_sha256={p.name:sha(p) for p in sorted(dest.iterdir())})
    dump(dest/"package_manifest.json",manifest)
    with zipfile.ZipFile(dest.with_suffix(".zip"),"x",zipfile.ZIP_DEFLATED) as z:
        for p in sorted(dest.iterdir()):
            # Mapping contains experiment identities; omit from external package.
            if p.name!="local_review_mapping.csv":z.write(p,p.name)
    print(json.dumps({"output":str(dest),"upload_zip":str(dest.with_suffix(".zip")),**{k:manifest[k] for k in ("total_outputs","prior_reused_outputs","new_unique_cases","new_review_mapped_outputs")}},indent=2))
    return dest

def indexed(rows):
    out={}
    for r in rows:
        k=r["review_id"];need(k not in out,"Duplicate review_id "+k);out[k]=r
    return out
def quality(package,review):
    rows,hashes,prior,cases,links=review_plan()
    m=json.loads((package/"package_manifest.json").read_text())
    need(m["schema"]=="B9_FINAL_REVIEW_V1" and m["run_manifest_sha256"]==hashes and m["prior_audit_sha256"]==sha(REVIEW/AUDIT),"Wrong review package")
    for f,h in m["input_sha256"].items():need(Path(f).name==f and sha(package/f)==h,"Package changed "+f)
    need(indexed(readjsonl(package/"review_cases.jsonl"))==cases and readcsv(package/"local_review_mapping.csv")==links,"Review plan mismatch")
    a=json.loads((review/NEW_AUDIT).read_text())
    need(a["protocol_version"]==PROTOCOL and a["human_clinical_review"]=="none" and a["label_provenance_statement"]==STATEMENT,"Invalid review provenance")
    need(a["scope"]["total_responses"]==a["scope"]["records_processed"]==len(cases),"Incomplete audit")
    need(a["review_input_sha256"]==m["review_input_sha256"]==sha(package/"review_cases.jsonl"),"Wrong review inputs")
    need(bool(a["evaluator"]) and bool(a["annotation_date"]) and isinstance(a["access_limitations"],list),"Incomplete audit details")
    for f in (NEW_LABELS,NEW_CLAIMS):need(a["output_sha256"][f]==sha(review/f),"Review changed "+f)
    labels=indexed(readcsv(review/NEW_LABELS));claims=indexed(readjsonl(review/NEW_CLAIMS))
    need(set(labels)==set(claims)==set(cases),"Missing/extra case IDs")
    judged={}
    for k,r in cases.items():
        l,c=labels[k],claims[k]
        need(l["model"]==c["model"]=="B9" and l["query_id"]==c["query_id"]==r["query_id"],"Case ID mismatch "+k)
        judged[k]=validate_row(l,c,dict(r,latency_ms=0))
    results=[]
    for r,link in zip(rows,links):
        j=prior[link["review_id"]] if link["review_source"]=="EXACT_PRIOR_AI_REVIEW" else judged[link["review_id"]]
        results.append(dict(j,run=r["run"],latency_ms=r["latency_ms"],pipeline_status=r["pipeline_status"],answer_scope=r["answer_scope"],
            review_source=link["review_source"],review_id=link["review_id"],identity_sha256=link["identity_sha256"]))
    summary=[]
    for name in RUNS:
        rs=[r for r in results if r["run"]==name]
        ar=[r for r in rows if r["run"]==name]
        s=json.loads((RUNS[name]/"summary.json").read_text())
        wall=s.get("wall_time_seconds",s.get("wall_seconds"))
        summary.append(dict(run=name,**aggregate(rs),complete_answers=sum(r["answer_completeness"]=="COMPLETE" for r in rs),
            partial_answers=sum(r["answer_completeness"]=="PARTIAL" for r in rs),
            nonresponsive_answers=sum(r["answer_completeness"]=="NONRESPONSIVE" for r in rs),
            appropriate_abstentions=sum(r["answer_completeness"]=="APPROPRIATE_ABSTENTION" for r in rs),
            inappropriate_abstentions=sum(r["answer_completeness"]=="INAPPROPRIATE_ABSTENTION" for r in rs),
            strict_complete_yield=sum(r["label"]=="No" and r["substantive"] and r["grounding"]=="FULLY_GROUNDED" and r["answer_completeness"]=="COMPLETE" and r["SUPPORTED"]>0 and not any(r[v] for v in ("PARTIALLY_SUPPORTED","CONTRADICTED","UNSUPPORTED","UNVERIFIABLE")) for r in rs)/132,
            mean_latency_ms=mean([r["latency_ms"] for r in ar]),throughput_qps=132/wall,
            reused_prior_annotations=sum(r["review_source"]=="EXACT_PRIOR_AI_REVIEW" for r in rs)))
    dest=OUT/("quality_"+stamp());dest.mkdir(parents=True,exist_ok=False)
    writecsv(dest/"variant_quality_summary.csv",summary)
    writecsv(dest/"variant_quality_per_query.csv",results)
    dump(dest/"manifest.json",dict(schema="B9_FINAL_QUALITY_V1",records=len(results),run_manifest_sha256=hashes,
        package_sha256=sha(package/"package_manifest.json"),review_audit_sha256=sha(review/NEW_AUDIT),prior_audit_sha256=sha(REVIEW/AUDIT),
        limitations=a["access_limitations"]+["Exact context identity reuse is AI reference reuse, not independent judgments. Different judge sessions may drift. Development cohort; no clinical gold."],
        outputs_sha256={p.name:sha(p) for p in sorted(dest.iterdir())}))
    print(f"{'Run':<16}{'Subst':>8}{'Yes':>7}{'Uncert':>8}{'Complete':>10}{'Strict %':>10}{'Mean s':>10}")
    for r in summary:print(f"{r['run']:<16}{r['substantive_answers']:>8}{r['yes']:>7}{r['uncertain']:>8}{r['complete_answers']:>10}{100*r['strict_complete_yield']:>10.2f}{r['mean_latency_ms']/1000:>10.3f}")
    print("Output:",dest);return dest

def semantic_pairs(r):
    evidence="\n".join(d["text"] for d in r["retrieved_documents"] if d.get("text"))
    context=r.get("rewritten_knowledge") or evidence
    grounding=r.get("rewritten_knowledge") or r["final_response"]
    return {"srs_cosine_proxy":(r["query"],r.get("refined_query") or ""),
        "context_cosine_proxy":(r["query"],context),"grounding_cosine_proxy":(grounding,evidence)}

def semantic(batch_size=16):
    need(batch_size>0,"Invalid batch size")
    rows,hashes=archived()
    bm=json.loads((FULL/"full_metrics_manifest.json").read_text())
    need(bm["semantic_proxies_enabled"] and bm["input_dataset_sha256"]==sha(Q),"Wrong baseline semantic report")
    for f,h in bm["file_sha256"].items():need(Path(f).name==f and sha(FULL/f)==h,"Baseline semantic drift")
    snap=resolve_cached_snapshot()
    need(snap.name==bm["semantic_proxy_encoder_revision"],"Different sentence model revision")
    asset=fingerprint_snapshot(snap)["snapshot_sha256"]
    need(asset=="42d8d798e4f01e68d9bb10634b9c712de00f7f8495271636fd6311b2db58e506","Unexpected sentence model fingerprint")
    from sentence_transformers import SentenceTransformer
    import importlib.metadata
    model=SentenceTransformer(str(snap),device="cpu",local_files_only=True)
    ps=[semantic_pairs(r) for r in rows]
    texts=sorted({s for p in ps for pair in p.values() if all(pair) for s in pair})
    vec=model.encode(texts,batch_size=batch_size,normalize_embeddings=True,show_progress_bar=True)
    lookup=dict(zip(texts,vec))
    def cos(a,b):
        if not a or not b:return None
        v=float(lookup[a]@lookup[b])
        return max(-1.,min(1.,v)) if math.isfinite(v) else None
    details=[dict(run=r["run"],query_id=r["query_id"],**{k:cos(*pair) for k,pair in p.items()}) for r,p in zip(rows,ps)]
    summary=[]
    keys=("srs_cosine_proxy","context_cosine_proxy","grounding_cosine_proxy")
    for name in RUNS:
        g=[r for r in details if r["run"]==name]
        summary.append(dict(run=name,queries=len(g),**{k:mean([r[k] for r in g]) for k in keys},
            **{k+"_n":sum(r[k] is not None for r in g) for k in keys}))
    dest=OUT/("semantic_"+stamp());dest.mkdir(parents=True,exist_ok=False)
    writecsv(dest/"B9_variant_semantic_summary.csv",summary);writecsv(dest/"B9_variant_semantic_per_query.csv",details)
    # Keep prior B1-B9 medical-quality table intact; enrich the B9 row with
    # the same cosine metrics (only archived B9-v2, not the compact candidate).
    cm=json.loads((CONSOLIDATED/"manifest.json").read_text())
    for f,h in cm["output_sha256"].items():need(Path(f).name==f and sha(CONSOLIDATED/f)==h,"Prior consolidated report changed")
    base=readcsv(CONSOLIDATED/"consolidated_B1_B9_132.csv")
    need([r["model"] for r in base]==[f"B{i}" for i in range(1,10)],"Unexpected consolidated models")
    vals=next(r for r in summary if r["run"]=="B9_v2")
    for k in keys:
        base[-1][k]=vals[k];base[-1][k+"_n"]=vals[k+"_n"]
    writecsv(dest/"consolidated_B1_B9_with_semantic_132.csv",base)
    dump(dest/"manifest.json",dict(schema="B9_FINAL_SEMANTIC_V1",run_manifest_sha256=hashes,
        encoder_revision=snap.name,encoder_snapshot_sha256=asset,sentence_transformers_version=importlib.metadata.version("sentence-transformers"),
        baseline_semantic_manifest_sha256=sha(FULL/"full_metrics_manifest.json"),
        limitations=["Cosine similarity proxies only, not clinical entailment or accuracy. Text bases differ by pipeline. Missing refinement, evidence or answers produce N/A. Uses same pinned encoder; long texts may truncate."],
        outputs_sha256={p.name:sha(p) for p in sorted(dest.iterdir())}))
    print("Output:",dest);return dest
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("command",choices=("prepare","semantic","quality"));p.add_argument("--package-dir",type=Path);p.add_argument("--review-dir",type=Path);p.add_argument("--batch-size",type=int,default=16);a=p.parse_args()
    if a.command=="prepare":prepare(a.package_dir)
    elif a.command=="semantic":semantic(a.batch_size)
    else:
        if not a.package_dir or not a.review_dir:p.error("Both directories required")
        quality(a.package_dir,a.review_dir)
if __name__=="__main__":main()
