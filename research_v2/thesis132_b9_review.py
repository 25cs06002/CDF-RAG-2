"""Prepare and validate external AI reference annotations for the archived B9-v2 run."""
from __future__ import annotations
import argparse,csv,hashlib,json,shutil,zipfile
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
from compare_b9_versions_132 import load
from .thesis132_b9 import ROOT,CORPUS,GRAPH,sha,dump
from .thesis132_ai_evaluation import aggregate,VERDICTS,COUNTS,NO_CLAIMS,truth,PROVENANCE

RUN=ROOT/"presentation_20261118/thesis132/b9_end_to_end_v2/b9_v2_adaptive_20261125T182727462347Z_c92418"
BASE=ROOT/"presentation_20261118/thesis132/csv_reports/batch132_csv_20261121T061653054773Z_bfeb15"
BASELINE=BASE/"consolidated_132_20261125T175313016959Z"
META=ROOT/"data/benchmark/final_132_v1/metadata_review_only.csv"
CATALOG=ROOT/"data/benchmark/v2/source_catalog_screened_20261121.json"
LABELS="ai_verified_hallucination_labels_132.csv"
CLAIMS="medical_claim_verification_132.jsonl"
SUMMARY="ai_hallucination_evaluation_summary.csv"
AUDIT="ai_evaluation_audit.json"
PROTOCOL="AIJ-MED-HALLUC v1"
STATEMENT="Labels were generated through AI-assisted evidence verification and automatically accepted under a predefined evaluation protocol. They have not been independently adjudicated by human clinical reviewers."
GROUNDING={"FULLY_GROUNDED","PARTIALLY_GROUNDED","NOT_GROUNDED","NO_FACTUAL_CLAIMS","NOT_APPLICABLE_EMPTY_ANSWER"}
COMPLETENESS={"COMPLETE","PARTIAL","NONRESPONSIVE","APPROPRIATE_ABSTENTION","INAPPROPRIATE_ABSTENTION","UNCERTAIN"}
FIELDS=["model","query_id","final_response_sha256","ai_candidate_label","ai_final_label","ai_review_status","human_review_status","confidence",*COUNTS.values(),"evidence_grounding_status","source_urls","verified_source_evidence","verification_reasoning","unresolved_issues","answer_completeness","fabricated_citation","citation_issues","protocol_version","label_provenance_statement"]
def need(ok,msg):
    if not ok:raise ValueError(msg)
def stamp():return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
def readcsv(p):
    with p.open(encoding="utf-8-sig",newline="") as f:return list(csv.DictReader(f))
def writecsv(p,rows,fields=None):
    with p.open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields or list(rows[0]));w.writeheader();w.writerows(rows)
def readjsonl(p):return [json.loads(s) for s in p.read_text(encoding="utf-8").splitlines() if s.strip()]
def keyed(rows):
    out={}
    for r in rows:
        k=(r["model"],r["query_id"])
        need(k not in out,"Duplicate "+str(k));out[k]=r
    return out
def inputs(run):
    m,records,traces=load(run,"THESIS132_B9_E2E_V2")
    need(m["policy"]=="adaptive","Require adaptive run")
    meta={r["query_id"]:r for r in readcsv(META)}
    rows=[]
    for r,t in zip(records,traces):
        q=r["query_id"];answer=r["final_answer"] or ""
        rows.append(dict(model="B9",variant="B9_v2",query_id=q,query=r["original_query"],
            final_response=answer,final_response_sha256=hashlib.sha256(answer.encode()).hexdigest(),
            pipeline_status=r["status"],query_type=meta[q]["query_type"],medical_domain=meta[q]["medical_domain"],
            route=t["route"],answer_scope=t["answer_scope"],stop_reason=t["stopping_reason"],
            matched_concept=r.get("matched_concept"),targets=t["targets"],
            retrieved_documents=r["retrieved_documents"],causal_edges=r["causal_edges"],causal_paths=r["causal_paths"],
            selected_ids=t["selected_ids"],selected_evidence=t["selected_evidence"],
            latency_ms=r["timings"]["total_ms"],detector_flag=None))
    need(len(keyed(rows))==132,"Missing records")
    return m,rows
PROMPT="""# External AI reviewer: B9-v2, 132 medical RAG answers

Evaluate every record in B9_review_inputs_132.jsonl. The other files in this
package are supporting context. No access to the user's machine or GitHub is
needed. The thesis is Adaptive Multi-Round RAG Architecture for Reliable
Domain-Specific Question Answering. B9-v2 uses bounded graph/document retrieval,
Ollama evidence-ID selection and deterministic rendering of candidate summaries.
The fixed 132 questions are development-exposed. G1 and document summaries are
provisional, not independently clinically verified. Do not infer medical
correctness from pipeline status, citations or the fact that B9 is proposed.

## Protocol: AIJ-MED-HALLUC v1, matching the B1-B8 AI reference

For ALL 132 records, extract atomic medically meaningful claims. Verify each
against actually inspected authoritative medical references, with source URL,
title, passage or finding, access mode and reasoning. Keep qualifiers and
distinguish causes, risk factors, association, triggers, subtype and direction.
Classify each claim SUPPORTED, PARTIALLY_SUPPORTED, CONTRADICTED, UNSUPPORTED,
UNVERIFIABLE or NON_FACTUAL. UNSUPPORTED requires adequate search, not merely
absence of retrieved support. If unable to verify, use UNVERIFIABLE. Do not
invent quotations or source access; distinguish paraphrases and exact quotes.
Verify citations against the actual retrieved evidence separately.

Answer-level label Yes if any CONTRADICTED or UNSUPPORTED material claim or
fabricated citation; otherwise Uncertain if any UNVERIFIABLE claim; otherwise
No. Partial support alone is not automatically Yes. Empty answer has no
claims, label No for hallucination only, but may be inappropriate abstention.
Use AI_AUTO_ACCEPTED for Yes/No with medium/high confidence; AI_UNRESOLVED for
Uncertain with low confidence. Human status always NOT_REVIEWED.

Grounding: FULLY_GROUNDED, PARTIALLY_GROUNDED, NOT_GROUNDED, NO_FACTUAL_CLAIMS,
NOT_APPLICABLE_EMPTY_ANSWER. Check evidence entailment and citation scope, not
mere ID membership. Completeness: COMPLETE, PARTIAL, NONRESPONSIVE,
APPROPRIATE_ABSTENTION, INAPPROPRIATE_ABSTENTION, UNCERTAIN. An externally
correct but irrelevant answer is not necessarily hallucinated. A partial
pathway is not automatically complete. Review all empty answers separately.
Question category is only a development descriptor, not clinical ground truth.

## Deliver four UTF-8 files

1. ai_verified_hallucination_labels_132.csv: use the exact header and IDs
from B9_ai_label_template_132.csv; fill all labels and counts. Preserve exact
final_response_sha256. Count claims by verdict. Set fabricated_citation true
or false. Document rationale and source evidence; keep uncertainty. Preserve
protocol_version and label_provenance_statement as prefilled.

2. medical_claim_verification_132.jsonl: 132 objects with model, query_id,
query, final_response, final_response_sha256, ai_final_label, ai_review_status,
human_review_status, confidence, evidence_grounding_status, answer_completeness,
fabricated_citation (boolean), claims (array). Every claim object has claim
(exact assertion), verdict, reasoning, evidence (array of objects with
source_url, source_title, access_mode, passage_or_finding, evidence_kind
[exact_quote or paraphrase], supports_or_contradicts). Supported, partial
and contradicted claims require traceable evidence. Unsupported/unverifiable
can have empty evidence only with a reason. No claims for empty answers.

3. ai_hallucination_evaluation_summary.csv: one B9 row with model,
total_responses, auto_accepted_total, auto_accepted_yes, auto_accepted_no,
uncertain_unresolved, substantive_answers, substantive_answers_accepted,
externally_supported_answers, fully_grounded_answers. Substantive means at
least one factual claim (anything except NON_FACTUAL); externally_supported
means substantive accepted No, per historical metric. Compute, do not guess.

4. ai_evaluation_audit.json: protocol_version (AIJ-MED-HALLUC v1), evaluator
(model/version), annotation_date, human_clinical_review ("none"),
scope ({"total_responses":132,"records_processed":132}), review_input_sha256
(SHA256 of B9_review_inputs_132.jsonl bytes), run_manifest_sha256 (from
review_package_manifest.json), output_sha256 (mapping of exact filenames of
the other three outputs to actual SHA256 digests), access_limitations (array),
label_provenance_statement (exact prefilled statement in label template).

No human approval. No detector predictions for B9. Check every exact answer,
ID, hash, claim count and aggregate. If you cannot finish or compute hashes,
report partial results and limitations; do not fabricate. Return the four
files, not merely a text summary. Process in batches if necessary.
"""
def prepare(run=RUN,output=None):
    run=Path(run).resolve();m,rows=inputs(run)
    dest=Path(output).resolve() if output else run/("ai_review_package_"+stamp())
    dest.mkdir(parents=True,exist_ok=False)
    (dest/"B9_review_inputs_132.jsonl").write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in rows),encoding="utf-8")
    fields=("model","variant","query_id","query","final_response","final_response_sha256","pipeline_status","query_type","medical_domain","route","answer_scope","stop_reason")
    writecsv(dest/"B9_review_worksheet_132.csv",[{k:r[k] for k in fields} for r in rows])
    writecsv(dest/"B9_ai_label_template_132.csv",[dict(model="B9",query_id=r["query_id"],final_response_sha256=r["final_response_sha256"],human_review_status="NOT_REVIEWED",protocol_version=PROTOCOL,label_provenance_statement=STATEMENT) for r in rows],FIELDS)
    (dest/"README_MASTER_PROMPT.md").write_text(PROMPT,encoding="utf-8")
    for source in (run/"B9_queries_132.csv",run/"manifest.json",CORPUS,CORPUS.parent/"provenance.jsonl",GRAPH/"typed_relationships.csv",GRAPH/"nodes.csv",CATALOG):
        key=str(source.relative_to(ROOT))
        if key in m["inputs"]:need(sha(source)==m["inputs"][key],"Evidence changed: "+key)
        shutil.copyfile(source,dest/("b9_run_manifest.json" if source.name=="manifest.json" else source.name))
    dump(dest/"review_package_manifest.json",dict(schema="B9_V2_AI_REVIEW_PACKAGE_V1",records=132,run_manifest_sha256=sha(run/"manifest.json"),review_input_sha256=sha(dest/"B9_review_inputs_132.jsonl"),input_sha256={p.name:sha(p) for p in sorted(dest.iterdir())},medical_gold=False))
    archive=dest.with_suffix(".zip")
    with zipfile.ZipFile(archive,"x",compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(dest.iterdir()):z.write(p,p.name)
    print("Review folder:",dest,"\nUpload ZIP:",archive,"\nNo labels generated.")
    return dest

def validate_row(l,c,r):
    key=(r["model"],r["query_id"])
    need(c["query"]==r["query"] and c["final_response"]==r["final_response"],f"{key}: original mismatch")
    need(l["final_response_sha256"]==c["final_response_sha256"]==r["final_response_sha256"],f"{key}: answer hash mismatch")
    for k in ("ai_final_label","ai_review_status","human_review_status","confidence","evidence_grounding_status","answer_completeness"):
        need(l[k]==c[k],f"{key}: inconsistent {k}")
    need(l["protocol_version"]==PROTOCOL and l["label_provenance_statement"]==STATEMENT and l["human_review_status"]=="NOT_REVIEWED",f"{key}: provenance")
    label=l["ai_final_label"]
    need(label in {"Yes","No","Uncertain"} and l["ai_candidate_label"]==label,f"{key}: invalid label")
    need(l["ai_review_status"]==("AI_UNRESOLVED" if label=="Uncertain" else "AI_AUTO_ACCEPTED"),f"{key}: status")
    need(l["confidence"] in ({"low"} if label=="Uncertain" else {"medium","high"}),f"{key}: confidence")
    need(bool(l["verification_reasoning"].strip()) and isinstance(c["claims"],list),f"{key}: no reasoning/claims")
    counts=Counter(z["verdict"] for z in c["claims"])
    need(set(counts)<=set(VERDICTS),f"{key}: verdict")
    for v,k in COUNTS.items():need(int(l[k])==counts[v],f"{key}: count {k}")
    for z in c["claims"]:
        need(bool(z["claim"].strip() and z["reasoning"].strip()) and isinstance(z["evidence"],list),f"{key}: claim schema")
        if z["verdict"] in {"SUPPORTED","PARTIALLY_SUPPORTED","CONTRADICTED"}:need(bool(z["evidence"]),f"{key}: missing source support")
        for e in z["evidence"]:
            need(e["source_url"].startswith(("https://","http://")) and all(str(e[k]).strip() for k in ("source_title","access_mode","passage_or_finding","supports_or_contradicts")) and e["evidence_kind"] in {"exact_quote","paraphrase"},f"{key}: malformed evidence")
    fabricated=truth(l["fabricated_citation"])
    need(fabricated is not None and isinstance(c["fabricated_citation"],bool) and fabricated==c["fabricated_citation"],f"{key}: citation flag")
    derived="Yes" if counts["CONTRADICTED"]+counts["UNSUPPORTED"] or fabricated else "Uncertain" if counts["UNVERIFIABLE"] else "No"
    need(label==derived,f"{key}: protocol label")
    substantive=any(counts[v] for v in VERDICTS[:-1])
    g=l["evidence_grounding_status"];complete=l["answer_completeness"]
    need(g in GROUNDING and substantive==(g not in NO_CLAIMS) and complete in COMPLETENESS,f"{key}: grounding/completeness")
    if not r["final_response"]:
        need(not c["claims"] and not fabricated and label=="No" and g=="NOT_APPLICABLE_EMPTY_ANSWER" and complete in {"APPROPRIATE_ABSTENTION","INAPPROPRIATE_ABSTENTION","UNCERTAIN"},f"{key}: empty answer")
    return dict(model="B9",query_id=r["query_id"],query_type=r["query_type"],medical_domain=r["medical_domain"],pipeline_status=r["pipeline_status"],label=label,confidence=l["confidence"],substantive=substantive,grounding=g,answer_completeness=complete,fabricated_citation=fabricated,detector_flag=None,latency_ms=r["latency_ms"],answer_scope=r["answer_scope"],**{v:counts[v] for v in VERDICTS},claims_without_evidence_ids=sum(not z["evidence"] and z["verdict"]!="NON_FACTUAL" for z in c["claims"]))
def validate(run,package,review):
    _,rows=inputs(run)
    pm=json.loads((package/"review_package_manifest.json").read_text())
    need(pm["schema"]=="B9_V2_AI_REVIEW_PACKAGE_V1" and pm["records"]==132 and pm["run_manifest_sha256"]==sha(run/"manifest.json"),"Wrong review package/run")
    for k,v in pm["input_sha256"].items():need(Path(k).name==k and sha(package/k)==v,"Package changed: "+k)
    need(readjsonl(package/"B9_review_inputs_132.jsonl")==rows,"Package answers differ")
    a=json.loads((review/AUDIT).read_text())
    need(a["protocol_version"]==PROTOCOL and a["human_clinical_review"]=="none" and a["label_provenance_statement"]==STATEMENT,"Audit provenance")
    need(a["scope"]["total_responses"]==a["scope"]["records_processed"]==132,"Audit coverage")
    need(a["review_input_sha256"]==pm["review_input_sha256"]==sha(package/"B9_review_inputs_132.jsonl") and a["run_manifest_sha256"]==pm["run_manifest_sha256"],"Audit fingerprint")
    need(bool(a["evaluator"]) and bool(a["annotation_date"]) and isinstance(a["access_limitations"],list),"Audit details")
    for name in (LABELS,CLAIMS,SUMMARY):need(a["output_sha256"][name]==sha(review/name),"Changed review output: "+name)
    labels=keyed(readcsv(review/LABELS));claims=keyed(readjsonl(review/CLAIMS));expected=keyed(rows)
    need(set(labels)==set(claims)==set(expected),"Missing/extra review rows")
    result=[validate_row(labels[k],claims[k],r) for k,r in expected.items()]
    m=aggregate(result);s=readcsv(review/SUMMARY)
    need(len(s)==1 and s[0]["model"]=="B9","External summary must contain B9 only")
    for ext,local in (("total_responses","total_responses"),("auto_accepted_total","accepted"),("auto_accepted_yes","yes"),("auto_accepted_no","no"),("uncertain_unresolved","uncertain"),("substantive_answers","substantive_answers"),("substantive_answers_accepted","substantive_accepted"),("externally_supported_answers","externally_supported_substantive_answers"),("fully_grounded_answers","fully_grounded_substantive_answers")):
        need(int(s[0][ext])==m[local],"Summary mismatch: "+ext)
    return a,result
def baseline_rows(folder):
    m=json.loads((folder/"consolidated_manifest.json").read_text())
    need(m["schema"]=="THESIS132_CONSOLIDATED_V1" and m["records"]==1056 and m["source_manifest_sha256"]==sha(BASE/"export_manifest.json"),"Wrong baseline")
    for k,v in m["output_sha256"].items():need(Path(k).name==k and sha(folder/k)==v,"Changed baseline: "+k)
    rows=readcsv(folder/"consolidated_B1_B8_132.csv")
    need([r["model"] for r in rows]==[f"B{i}" for i in range(1,9)] and all(int(r["questions"])==132 for r in rows),"Baseline alignment")
    return rows
def evaluate(run,package,review,baseline=BASELINE):
    audit,rows=validate(run,package,review)
    baselines=baseline_rows(baseline);ops=json.loads((run/"summary.json").read_text());aq=aggregate(rows)
    b9=dict(model="B9",model_name="Adaptive multi-round RAG v2",questions=132,ok=ops["ok"],abstained=ops["abstained"],errors=ops["errors"],candidate_evidence_count=ops["candidate_evidence_count"],candidate_evidence_rate=ops["candidate_evidence_count"]/132,answer_present_count=ops["answer_present_count"],mean_latency_ms=ops["mean_latency_ms"],median_latency_ms=ops["median_latency_ms"],p95_latency_ms=ops["p95_latency_ms"],mean_latency_ok_ms=ops["mean_latency_ok_ms"],wall_time_seconds=ops["wall_time_seconds"],run_throughput_qps=ops["actual_run_throughput_qps"],evidence_source_type="G1 typed candidates + provisional summaries",**aq,classification_support=0,detector_observations=0,ai_supported_substantive_answers_per_wall_second=aq["externally_supported_substantive_answers"]/ops["wall_time_seconds"])
    out=review/("evaluation_b9_132_"+stamp());out.mkdir(exist_ok=False)
    writecsv(out/"B9_ai_quality_132.csv",[dict(**b9,complete_answer_count=sum(r["answer_completeness"]=="COMPLETE" for r in rows),strict_supported_grounded_complete_yield=sum(r["substantive"] and r["label"]=="No" and r["grounding"]=="FULLY_GROUNDED" and r["SUPPORTED"]>0 and not any(r[v] for v in VERDICTS[1:-1]) and r["answer_completeness"]=="COMPLETE" for r in rows)/132,completeness_counts=json.dumps(dict(Counter(r["answer_completeness"] for r in rows))))])
    writecsv(out/"B9_ai_per_query_132.csv",rows)
    fields=list(baselines[0]);writecsv(out/"consolidated_B1_B9_132.csv",baselines+[{k:b9.get(k) for k in fields}],fields)
    bytype=[]
    for k in sorted({r["query_type"] for r in rows}):
        g=[r for r in rows if r["query_type"]==k];bytype.append(dict(query_type=k,**aggregate(g),complete_answers=sum(r["answer_completeness"]=="COMPLETE" for r in g)))
    writecsv(out/"B9_ai_by_query_type_132.csv",bytype)
    dump(out/"manifest.json",dict(schema="THESIS132_B9_AI_REFERENCE_V1",provenance=PROVENANCE,run_manifest_sha256=sha(run/"manifest.json"),review_audit_sha256=sha(review/AUDIT),baseline_manifest_sha256=sha(baseline/"consolidated_manifest.json"),limitations=audit["access_limitations"]+["AI reference only; B9 judge session may differ from B1-B8.","B9 semantic proxies not calculated; no detector predictions.","Completeness labels B9-only; development-exposed questions and different evidence/generation, descriptive comparison."],output_sha256={p.name:sha(p) for p in sorted(out.iterdir())}))
    print("B1-B9 AI-reference comparison; N/A means unavailable")
    print(f"{'Model':<6}{'Evidence':>10}{'Substantive':>13}{'AI hall.':>11}{'Strict yield':>14}{'Latency s':>12}{'QPS':>10}")
    def pct(v):return "N/A" if v is None or v=="" else f"{100*float(v):.2f}%"
    for r in baselines+[b9]:
        print(f"{r['model']:<6}{pct(r['candidate_evidence_rate']):>10}{pct(r['substantive_coverage']):>13}{pct(r['hallucination_rate_ai']):>11}{pct(r['strict_supported_and_fully_grounded_yield']):>14}{float(r['mean_latency_ms'])/1000:>12.3f}{float(r['run_throughput_qps']):>10.4f}")
    print("Output:",out);return out
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("command",choices=("prepare","validate","evaluate"));p.add_argument("--run-dir",type=Path,default=RUN);p.add_argument("--package-dir",type=Path);p.add_argument("--review-dir",type=Path);p.add_argument("--baseline-dir",type=Path,default=BASELINE);a=p.parse_args()
    if a.command=="prepare":prepare(a.run_dir,a.package_dir)
    else:
        if not a.package_dir or not a.review_dir:p.error("--package-dir and --review-dir required")
        if a.command=="validate":
            _,rows=validate(a.run_dir,a.package_dir,a.review_dir);print(json.dumps(aggregate(rows),indent=2));print("132 exact answer-bound AI review records structurally validated. Not clinical certification.")
        else:evaluate(a.run_dir,a.package_dir,a.review_dir,a.baseline_dir)
if __name__=="__main__":main()
