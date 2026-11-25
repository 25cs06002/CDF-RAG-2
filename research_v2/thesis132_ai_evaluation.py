"""Evaluate AI-judge annotations against the archived B1-B8 answers.

No inference, no human-gold promotion. Fail closed on source/hash/schema drift.
Outputs use distinct AI-reference names; no original input is overwritten.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from datetime import datetime, timezone

from .thesis132_evaluation import ROOT, MODELS, _load, _resolve_dir

BASE=ROOT/"presentation_20261118/thesis132/csv_reports/batch132_csv_20261121T061653054773Z_bfeb15"
REVIEW=BASE/"hallucination_review_132_20261123T121224403405Z_d28834/ai_review_v1"
LABELS="ai_verified_hallucination_labels_1056.csv"
CLAIMS="medical_claim_verification_1056.jsonl"
SUMMARY="ai_hallucination_evaluation_summary.csv"
AUDIT="ai_evaluation_audit.json"
VERDICTS=("SUPPORTED","PARTIALLY_SUPPORTED","CONTRADICTED","UNSUPPORTED","UNVERIFIABLE","NON_FACTUAL")
COUNTS={v:v.lower()+"_claim_count" for v in VERDICTS[:-1]}
NO_CLAIMS={"NO_FACTUAL_CLAIMS","NOT_APPLICABLE_EMPTY_ANSWER"}
PROVENANCE="AI judge reference only; no independent human clinical adjudication"

def require(condition,message):
    if not condition: raise ValueError(message)

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def readcsv(path):
    with path.open(encoding="utf-8-sig",newline="") as f:
        return list(csv.DictReader(f))

def indexed(rows):
    out={}
    for r in rows:
        key=(r["model"],r["query_id"])
        require(key not in out,f"Duplicate record: {key}")
        out[key]=r
    return out

def truth(value):
    if value is None or str(value).strip()=="": return None
    v=str(value).lower()
    require(v in ("true","false","yes","no","1","0"),f"Invalid boolean {value}")
    return v in ("true","yes","1")

def ratio(a,b):
    return a/b if b else None

def classify(rows):
    paired=[r for r in rows if r["label"] in ("Yes","No") and r["detector_flag"] is not None]
    tp=sum(r["label"]=="Yes" and r["detector_flag"] for r in paired)
    tn=sum(r["label"]=="No" and not r["detector_flag"] for r in paired)
    fp=sum(r["label"]=="No" and r["detector_flag"] for r in paired)
    fn=sum(r["label"]=="Yes" and not r["detector_flag"] for r in paired)
    n=len(paired)
    return dict(classification_support=n,detector_observations=sum(r["detector_flag"] is not None for r in rows),
                uncertain_with_detector=sum(r["label"]=="Uncertain" and r["detector_flag"] is not None for r in rows),
                tp=tp if n else None,tn=tn if n else None,fp=fp if n else None,fn=fn if n else None,
                accuracy_vs_ai=ratio(tp+tn,n),precision_vs_ai=ratio(tp,tp+fp),
                recall_vs_ai=ratio(tp,tp+fn),f1_vs_ai=ratio(2*tp,2*tp+fp+fn))

def aggregate(rows):
    n=len(rows)
    accepted=[r for r in rows if r["label"]!="Uncertain"]
    substantive=[r for r in rows if r["substantive"]]
    sa=[r for r in substantive if r["label"]!="Uncertain"]
    yes=sum(r["label"]=="Yes" for r in accepted)
    sy=sum(r["label"]=="Yes" for r in sa)
    counts={v:sum(r[v] for r in rows) for v in VERDICTS}
    total=sum(counts[v] for v in VERDICTS[:-1])
    fully=sum(r["grounding"]=="FULLY_GROUNDED" for r in substantive)
    high=[r for r in accepted if r["confidence"]=="high"]
    supported=sum(r["label"]=="No" for r in sa)
    return dict(total_responses=n,accepted=len(accepted),yes=yes,no=len(accepted)-yes,
                uncertain=n-len(accepted),label_coverage=ratio(len(accepted),n),
                hallucination_rate_ai=ratio(yes,len(accepted)),
                substantive_answers=len(substantive),substantive_accepted=len(sa),
                substantive_coverage=ratio(len(substantive),n),
                substantive_yes=sy,hallucination_rate_substantive_ai=ratio(sy,len(sa)),
                no_factual_claims=n-len(substantive),
                externally_supported_substantive_answers=supported,
                supported_substantive_yield=ratio(supported,n),
                fully_grounded_substantive_answers=fully,
                fully_grounded_substantive_rate=ratio(fully,len(substantive)),
                strict_supported_and_fully_grounded_yield=ratio(sum(r["label"]=="No" and
                    r["grounding"]=="FULLY_GROUNDED" and r["SUPPORTED"]>0 and
                    not any(r[v] for v in ("PARTIALLY_SUPPORTED","CONTRADICTED","UNSUPPORTED","UNVERIFIABLE"))
                    for r in substantive),n),
                high_confidence_accepted=len(high),
                high_confidence_hallucination_rate_ai=ratio(sum(r["label"]=="Yes" for r in high),len(high)),
                factual_claim_instances=total,**{v.lower()+"_claims":counts[v] for v in VERDICTS},
                supported_claim_rate=ratio(counts["SUPPORTED"],total),
                partial_claim_rate=ratio(counts["PARTIALLY_SUPPORTED"],total),
                contradicted_claim_rate=ratio(counts["CONTRADICTED"],total),
                unsupported_claim_rate=ratio(counts["UNSUPPORTED"],total),
                unverifiable_claim_rate=ratio(counts["UNVERIFIABLE"],total),
                claim_instances_without_evidence_ids=sum(r["claims_without_evidence_ids"] for r in rows),
                fabricated_citations=sum(r["fabricated_citation"] for r in rows),
                provenance=PROVENANCE)

def validate(source,review):
    source,review=_resolve_dir(source),_resolve_dir(review)
    base,manifest,_,raw,_=_load(source)
    audit=json.loads((review/AUDIT).read_text(encoding="utf-8"))
    require(audit["human_involvement"]["human_clinical_review"]=="none","Unexpected human-review provenance")
    require(audit["protocol_version"]=="AIJ-MED-HALLUC v1","Unexpected annotation protocol")
    require(audit["scope"]["total_responses"]==1056 and audit["scope"]["records_processed"]==1056,"Audit scope mismatch")
    output_hashes={r["file"]:r["sha256"] for r in audit["output_files"]}
    for name in (LABELS,CLAIMS,SUMMARY):
        require(sha(review/name)==output_hashes.get(name),"Audit output hash mismatch: "+name)
    # Check recorded source hashes against exact archived export bytes, not just IDs.
    input_hashes={r["file"]:r["sha256"] for r in audit["input_files"]}
    for m in MODELS:
        name=f"{m}_queries_132.csv"
        require(sha(base/name)==input_hashes.get(name),"Audit original input hash mismatch: "+name)
    cfg=manifest["session_configuration"]
    for name,k in (("documents.jsonl","candidate_corpus_sha256"),("edges.csv","G0_edges_sha256"),("nodes.csv","G0_nodes_sha256")):
        require(input_hashes.get(name)==cfg[k],"Audit evidence snapshot mismatch: "+name)
    labels=indexed(readcsv(review/LABELS))
    with (review/CLAIMS).open(encoding="utf-8") as f:
        claims=indexed(json.loads(line) for line in f if line.strip())
    expected={(m,r["query_id"]):r for m in MODELS for r in raw[m]}
    require(len(expected)==1056 and set(labels)==set(claims)==set(expected),"Incomplete/misaligned 132x8 review")
    evidence={e["evidence_id"]:e for e in audit["evidence_library"]}
    catalogue={e["key"]:e for e in audit["claim_catalogue"]["entries"]}
    provenance=audit["label_provenance_statement"]
    rows=[]
    for key,original in expected.items():
        l,c=labels[key],claims[key]
        require(c["query"]==original["query"] and c["final_response"]==original["final_response"],f"Original question/answer mismatch: {key}")
        require(hashlib.sha256(c["final_response"].encode()).hexdigest()==c["final_response_sha256"],f"Answer hash mismatch: {key}")
        for field in ("ai_candidate_label","ai_final_label","ai_review_status","human_review_status",
                      "confidence","evidence_grounding_status","answer_completeness","grounded_factual_claims",
                      "protocol_version","label_provenance_statement"):
            require(l[field]==c[field],f"{key}: inconsistent {field}")
        require(l["human_review_status"]=="NOT_REVIEWED" and l["label_provenance_statement"]==provenance,
                f"Invalid provenance: {key}")
        require(l["protocol_version"]==audit["protocol_version"],f"Protocol mismatch: {key}")
        require(l["verification_reasoning"].strip(),f"Missing reasoning: {key}")
        label=l["ai_final_label"]
        require(label in ("Yes","No","Uncertain") and l["ai_candidate_label"]==label,f"Invalid label: {key}")
        require(l["ai_review_status"]==("AI_UNRESOLVED" if label=="Uncertain" else "AI_AUTO_ACCEPTED"),
                f"Label acceptance conflict: {key}")
        require(l["confidence"] in (("low",) if label=="Uncertain" else ("medium","high")),f"Confidence mismatch: {key}")
        counts=Counter(z["verdict"] for z in c["claims"])
        require(set(counts)<=set(VERDICTS),f"Invalid verdict: {key}")
        for v,field in COUNTS.items():
            require(int(l[field])==counts[v],f"Claim count mismatch: {key} {field}")
        for z in c["claims"]:
            require(z["key"] in catalogue,f"Unknown claim key: {key}")
            require(set(z["evidence_ids"])<=set(evidence),f"Unknown evidence ID: {key}")
        fabricated=truth(l["fabricated_citation"])
        require(fabricated is not None,f"Missing citation flag: {key}")
        derived="Yes" if counts["CONTRADICTED"]+counts["UNSUPPORTED"] or fabricated else "Uncertain" if counts["UNVERIFIABLE"] else "No"
        require(label==derived,f"Protocol label conflict: {key}")
        detector=truth(original["is_hallucination"])
        require(detector==c["detector_flag_not_used_for_labeling"],f"Detector prediction mismatch: {key}")
        substantive=any(counts[v] for v in VERDICTS[:-1])
        grounding=l["evidence_grounding_status"]
        require(grounding in audit["criteria"]["grounding_status_values"],f"Unknown grounding status: {key}")
        require(substantive==(grounding not in NO_CLAIMS),f"Substantive/grounding mismatch: {key}")
        rows.append(dict(model=key[0],query_id=key[1],query_type=original["query_type"],
                         medical_domain=original["medical_domain"],semantic_group=original["semantic_group"],
                         pipeline_status=original["status"],label=label,confidence=l["confidence"],
                         substantive=substantive,grounding=grounding,answer_completeness=l["answer_completeness"],
                         fabricated_citation=fabricated,detector_flag=detector,
                         evidence_returned=truth(original["retrieval_success"]),latency_ms=float(original["latency_ms"]),
                         **{v:counts[v] for v in VERDICTS},
                         claims_without_evidence_ids=sum(not z["evidence_ids"] and z["verdict"]!="NON_FACTUAL" for z in c["claims"])))
    external={r["model"]:r for r in readcsv(review/SUMMARY)}
    require(set(external)==set(MODELS)|{"ALL"},"Invalid external summary model set")
    for m in (*MODELS,"ALL"):
        current=aggregate([r for r in rows if m=="ALL" or r["model"]==m])
        for ext,local in (("total_responses","total_responses"),("auto_accepted_total","accepted"),
                          ("auto_accepted_yes","yes"),("auto_accepted_no","no"),("uncertain_unresolved","uncertain"),
                          ("substantive_answers","substantive_answers"),("substantive_answers_accepted","substantive_accepted"),
                          ("externally_supported_answers","externally_supported_substantive_answers"),
                          ("fully_grounded_answers","fully_grounded_substantive_answers")):
            require(int(external[m][ext])==current[local],f"External summary mismatch: {m} {ext}")
    require(dict(Counter(r["label"] for r in rows))==audit["counts"]["labels"],"Audit label totals disagree")
    return base,audit,rows

def writecsv(path,rows):
    require(bool(rows),"Cannot infer CSV schema from empty data")
    with path.open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def run(source,review,output=None):
    base,audit,rows=validate(source,review)
    out=_resolve_dir(output) if output else _resolve_dir(review)/("evaluation_ai_132_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    out.mkdir(parents=True,exist_ok=False)
    groups={m:[r for r in rows if r["model"]==m] for m in MODELS}
    metrics=[dict(model=m,**aggregate(g)) for m,g in groups.items()]
    writecsv(out/"ai_quality_comparison_B1_B8_132.csv",metrics)
    writecsv(out/"ai_detector_classification_B1_B8_132.csv",[dict(model=m,**classify(g),provenance=PROVENANCE) for m,g in groups.items()])
    writecsv(out/"ai_per_query_metrics_132.csv",rows)
    for field in ("query_type","medical_domain"):
        result=[]
        for m,g in groups.items():
            for value in sorted({r[field] for r in g}):
                result.append(dict(model=m,**{field:value},**aggregate([r for r in g if r[field]==value])))
        writecsv(out/f"ai_comparison_by_{field}_132.csv",result)
    by7={r["query_id"]:r for r in groups["B7"]}
    pairs=[]
    for r8 in groups["B8"]:
        r7=by7[r8["query_id"]]
        pairs.append(dict(query_id=r8["query_id"],B7_label=r7["label"],B8_label=r8["label"],
                          B7_substantive=r7["substantive"],B8_substantive=r8["substantive"],
                          B7_grounding=r7["grounding"],B8_grounding=r8["grounding"],
                          B7_detector=r7["detector_flag"],B8_detector=r8["detector_flag"],
                          latency_delta_ms=r8["latency_ms"]-r7["latency_ms"],
                          both_binary=r7["label"]!="Uncertain" and r8["label"]!="Uncertain"))
    writecsv(out/"ai_paired_B7_B8_132.csv",pairs)
    writecsv(out/"ai_quality_performance_comparison_132.csv",[
        dict(model=m,**aggregate(g),mean_latency_ms=sum(r["latency_ms"] for r in g)/len(g),
             candidate_evidence_rate=ratio(sum(r["evidence_returned"] is True for r in g),len(g)),
             status_abstained=sum(r["pipeline_status"]=="abstained" for r in g))
        for m,g in groups.items()])
    uncertain=[r for r in rows if r["label"]=="Uncertain"]
    if uncertain: writecsv(out/"ai_unresolved_cases_132.csv",uncertain)
    report=dict(schema="THESIS132_AI_REFERENCE_V1",provenance=PROVENANCE,records=len(rows),
                source_manifest_sha256=sha(base/"export_manifest.json"),
                review_inputs_sha256={name:sha(_resolve_dir(review)/name) for name in (AUDIT,LABELS,CLAIMS,SUMMARY)},
                evaluator_reported=audit["evaluator"],limitations=audit["access_limitations"]+[
                  "Checks establish structural consistency and original-answer alignment, not medical correctness.",
                  "Unsupported claims without evidence IDs are retained under the original AI protocol.",
                  "Missing detector predictions are never imputed; classification is agreement with AI reference.",
                  "Non-substantive responses counted as No by original protocol; separate substantive rates reported."],
                output_sha256={p.name:sha(p) for p in sorted(out.glob("*.csv"))})
    (out/"ai_analysis_manifest.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    return out

def pct(x):
    return "N/A" if x is None or x=="" else f"{100*float(x):.2f}%"

def show(out):
    out=_resolve_dir(out)
    manifest=json.loads((out/"ai_analysis_manifest.json").read_text())
    require(manifest["schema"]=="THESIS132_AI_REFERENCE_V1","Unexpected report schema")
    for name,digest in manifest["output_sha256"].items():
        require(sha(out/name)==digest,"Report hash mismatch: "+name)
    print("\nAI-JUDGE REFERENCE — NOT HUMAN GOLD")
    print(f"{'Model':<6}{'Yes':>6}{'No':>6}{'Uncert':>8}{'All rate':>12}{'Subst n':>9}{'Subst rate':>13}{'Supported yield':>18}")
    for r in readcsv(out/"ai_quality_comparison_B1_B8_132.csv"):
        print(f"{r['model']:<6}{r['yes']:>6}{r['no']:>6}{r['uncertain']:>8}{pct(r['hallucination_rate_ai']):>12}{r['substantive_answers']:>9}{pct(r['hallucination_rate_substantive_ai']):>13}{pct(r['supported_substantive_yield']):>18}")
    print("\nDETECTOR AGREEMENT WITH AI REFERENCE")
    print(f"{'Model':<6}{'n':>6}{'Accuracy':>12}{'Precision':>12}{'Recall':>12}{'F1':>12}{'TN':>6}{'FP':>6}{'FN':>6}{'TP':>6}")
    for r in readcsv(out/"ai_detector_classification_B1_B8_132.csv"):
        print(f"{r['model']:<6}{r['classification_support']:>6}"+ "".join(f"{pct(r[k]):>12}" for k in ("accuracy_vs_ai","precision_vs_ai","recall_vs_ai","f1_vs_ai"))+"".join(f"{r[k] or 'N/A':>6}" for k in ("tn","fp","fn","tp")))
    print("\nOutput folder:",out)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir",type=Path,default=BASE)
    p.add_argument("--review-dir",type=Path,default=REVIEW)
    p.add_argument("--output-dir",type=Path)
    p.add_argument("--report-dir",type=Path,help="Display existing validated report")
    p.add_argument("--validate-only",action="store_true")
    a=p.parse_args()
    if a.report_dir:
        show(a.report_dir)
    elif a.validate_only:
        _,_,rows=validate(a.input_dir,a.review_dir)
        print(f"Validated {len(rows)} AI-judge records against exact source answers and audit hashes. Not clinical certification.")
    else:
        show(run(a.input_dir,a.review_dir,a.output_dir))

if __name__=="__main__": main()
