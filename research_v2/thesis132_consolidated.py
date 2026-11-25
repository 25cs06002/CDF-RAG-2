"""Consolidate the pinned B1-B8 experiment, semantic proxies and AI-reference review.
No inference. Do not replace missing clinical gold with AI labels. All output files
are new, versioned analysis artifacts. The historical source files remain unchanged.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from .thesis132_evaluation import ROOT, MODELS, _load, _derive, _aggregate, _resolve_dir
from .thesis132_ai_evaluation import BASE, REVIEW, validate, aggregate, classify

FULL = BASE / "full_metrics_132_20261123T115629452212Z_2879fc"
AI = REVIEW / "evaluation_ai_132_20261125T174933785470Z"
SEMANTIC = {
    "SRS_Cosine_Proxy": "srs_cosine_proxy",
    "Context_Relevance_Cosine_Proxy": "context_cosine_proxy",
    "Groundedness_Cosine_Proxy": "grounding_cosine_proxy",
}
LIMITATIONS = [
    "Fixed development-exposed 132 questions; no independent clinical holdout.",
    "AI-reference labels are single-judge estimates, not independent human gold.",
    "B1-B4 use provisional documents; B5-B8 use unverified G0 graph.",
    "Similarity scores are cosine proxies and have different text bases across pipelines.",
    "Conditional rates exclude unavailable labels; substantive yield uses all 132 questions.",
    "Non-substantive responses count as No in the original AI protocol; report coverage separately.",
    "B5/B6 abstentions affect all-query latency; compare OK-only latency as well.",
    "B1-B6 have no detector predictions, not zero detector performance.",
    "No attribution of observed differences to PPO alone; no measured memory or energy claims.",
]
def need(ok, message):
    if not ok:
        raise ValueError(message)

def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def csvread(p):
    with p.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

def csvwrite(p, rows):
    need(bool(rows), "Empty report: " + str(p))
    with p.open("w", encoding="utf-8", newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

def checked_manifest(folder,name,schema,hash_key,source):
    m=json.loads((folder/name).read_text(encoding="utf-8"))
    need(m["schema"]==schema,"Wrong schema: "+name)
    need(m["source_manifest_sha256"]==digest(source/"export_manifest.json"),
         "Different source experiment: "+name)
    for file, expected in m[hash_key].items():
        need(Path(file).name==file,"Unexpected manifest path")
        need(digest(folder/file)==expected,"Report integrity mismatch: "+file)
    return m

def num(x):
    if x is None or x=="": return None
    n=float(x)
    need(math.isfinite(n),"Invalid number")
    return n

def mean(values):
    values=[v for v in values if v is not None]
    return sum(values)/len(values) if values else None

def fraction(n,d):
    return n/d if d else None

def same(a,b):
    if a is None or b is None: return a is b
    return math.isclose(float(a),float(b),rel_tol=1e-9,abs_tol=1e-10)

def ids(rows,model):
    need(len(rows)==132 and all(r["model"]==model for r in rows),model+": row count/model mismatch")
    out={r["query_id"]:r for r in rows}
    need(len(out)==132,model+": duplicate IDs")
    return out

def load(source,full,ai):
    base,manifest,original,raw,_=_load(source)
    full,ai=_resolve_dir(full),_resolve_dir(ai)
    fm=checked_manifest(full,"full_metrics_manifest.json","THESIS132_FULL_METRICS_V1","file_sha256",base)
    am=checked_manifest(ai,"ai_analysis_manifest.json","THESIS132_AI_REFERENCE_V1","output_sha256",base)
    need(fm["models"]==list(MODELS) and fm["questions"]==132 and fm["semantic_proxies_enabled"],
         "Select a complete semantic-proxy report")
    need(fm["input_dataset_sha256"]==manifest["dataset_sha256"] and am["records"]==1056,
         "Dataset mismatch")
    need(not fm["gold_labels_supplied"],"Expected no-human-gold semantic report")
    review=ai.parent
    for name,sha in am["review_inputs_sha256"].items():
        need(Path(name).name==name and digest(review/name)==sha,"AI review input drift: "+name)
    _,audit,reviewed=validate(base,review)
    # Recompute quality and detector summaries from validated original annotations.
    # Do not blindly trust a joined summary or the externally generated counts.
    quality={r["model"]:r for r in csvread(ai/"ai_quality_comparison_B1_B8_132.csv")}
    detector={r["model"]:r for r in csvread(ai/"ai_detector_classification_B1_B8_132.csv")}
    need(set(quality)==set(detector)==set(MODELS),"AI report model mismatch")
    combined=[]
    per=[]
    for model in MODELS:
        original_by_id=ids(raw[model],model)
        f=ids(csvread(full/f"evaluation_metrics_{model}_132_full.csv"),model)
        q=ids([r for r in reviewed if r["model"]==model],model)
        need(set(f)==set(q)==set(original_by_id),"Question mismatch: "+model)
        ops=_aggregate(model,_derive(model,raw[model]),original[model])
        aq=aggregate(list(q.values()))
        det=classify(list(q.values()))
        for record,recomputed in ((quality[model],aq),(detector[model],det)):
            for key,value in recomputed.items():
                if key=="provenance": continue
                need(same(num(record[key]),value),f"AI summary disagrees: {model}.{key}")
        joined=dict(model=model,model_name=ops["model_name"],
                    questions=132,ok=ops["ok"],abstained=ops["abstained"],errors=ops["errors"],
                    candidate_evidence_count=ops["candidate_evidence_count"],
                    candidate_evidence_rate=ops["candidate_evidence_rate"],
                    answer_present_count=ops["answer_present_count"],
                    causal_edge_count=ops["total_CAUSES_edges"],causal_path_count=ops["total_causal_paths"],
                    mean_path_hops_if_any=ops["mean_path_hops_for_questions_with_paths"],
                    mean_latency_ms=ops["mean_latency_ms"],median_latency_ms=ops["median_latency_ms"],
                    p95_latency_ms=ops["p95_latency_ms"],mean_latency_ok_ms=ops["mean_latency_ok_ms"],
                    mean_latency_abstained_ms=ops["mean_latency_abstained_ms"],
                    wall_time_seconds=ops["wall_time_seconds"],run_throughput_qps=ops["run_throughput_qps"],
                    mean_retrieval_ms=ops["mean_retrieval_ms"],mean_generation_ms=ops["mean_generation_ms"],
                    mean_answer_words=ops["mean_answer_words_if_present"],
                    detector_flag_rate_not_gold=ops["detector_flag_rate_NOT_gold"],
                    evidence_source_type=ops["source_type"])
        for col,name in SEMANTIC.items():
            values=[num(r[col]) for r in f.values()]
            need(all(v is None or -1.000001<=v<=1.000001 for v in values),"Invalid cosine")
            joined[name]=mean(values)
            joined[name+"_n"]=sum(v is not None for v in values)
        joined.update(aq)
        joined.update(det)
        joined["ai_supported_substantive_answers_per_wall_second"]=fraction(
            aq["externally_supported_substantive_answers"],ops["wall_time_seconds"])
        joined["human_gold_accuracy"]=None
        joined["human_gold_hallucination_rate"]=None
        combined.append(joined)
        for key,r in q.items():
            s=f[key]
            o=original_by_id[key]
            need(s["query"]==o["query"] and s["final_response"]==o["final_response"],
                 "Semantic report answer mismatch: "+model+"/"+key)
            need(same(num(s["latency_ms"]),num(o["latency_ms"])),"Semantic latency mismatch")
            per.append(dict(**r,**{name:num(s[col]) for col,name in SEMANTIC.items()},
                            context_basis=s["context_basis"],grounding_basis=s["grounding_basis"]))
    return base,fm,am,audit,combined,per

def paired(rows):
    a={r["query_id"]:r for r in rows if r["model"]=="B7"}
    b={r["query_id"]:r for r in rows if r["model"]=="B8"}
    need(len(a)==len(b)==132 and set(a)==set(b),"Invalid B7/B8 pair set")
    changes=Counter()
    latency=[]
    for key in a:
        x,y=a[key],b[key]
        changes["label_"+x["label"]+"_to_"+y["label"]]+=1
        changes["substantive_both"]+=int(x["substantive"] and y["substantive"])
        changes["substantive_B7_only"]+=int(x["substantive"] and not y["substantive"])
        changes["substantive_B8_only"]+=int(y["substantive"] and not x["substantive"])
        changes["substantive_neither"]+=int(not x["substantive"] and not y["substantive"])
        if x["substantive"] and y["substantive"] and "Uncertain" not in (x["label"],y["label"]):
            changes["both_substantive_binary"]+=1
            changes["B7_yes_common_substantive"]+=int(x["label"]=="Yes")
            changes["B8_yes_common_substantive"]+=int(y["label"]=="Yes")
        latency.append(y["latency_ms"]-x["latency_ms"])
    return dict(question_pairs=132,**dict(sorted(changes.items())),
                mean_paired_latency_delta_ms=mean(latency),
                note="Descriptive same-query comparison. Common-substantive subset is selected by model behavior.")

# Explicit metric dictionary: status and denominator are part of the output.
METRICS=(
 ("candidate_evidence_rate","Candidate evidence yield","fraction","operational","all 132"),
 ("substantive_coverage","Substantive answer coverage","fraction","AI reference","all 132"),
 ("hallucination_rate_ai","AI hallucination estimate (all accepted)","fraction","AI reference","accepted Yes/No"),
 ("hallucination_rate_substantive_ai","AI hallucination estimate (substantive)","fraction","AI reference","accepted substantive Yes/No"),
 ("supported_substantive_yield","AI-supported substantive yield","fraction","AI reference","all 132; No label includes partial claims"),
 ("strict_supported_and_fully_grounded_yield","Strict supported+grounded yield","fraction","AI reference","all 132; no partial/unsupported/contradicted/unverifiable claims"),
 ("supported_claim_rate","AI supported claim rate","fraction","AI reference","factual claim instances"),
 ("fully_grounded_substantive_rate","AI full grounding","fraction","AI reference","substantive answers"),
 ("srs_cosine_proxy","SRS cosine proxy","cosine","embedding proxy","available refined queries"),
 ("context_cosine_proxy","Context cosine proxy","cosine","embedding proxy","available query/context pairs"),
 ("grounding_cosine_proxy","Grounding cosine proxy","cosine","embedding proxy","available context/evidence pairs"),
 ("mean_latency_ms","Mean latency","ms","measured","all 132"),
 ("p95_latency_ms","P95 latency","ms","measured","all 132"),
 ("mean_latency_ok_ms","Mean OK latency","ms","measured","status OK"),
 ("run_throughput_qps","Run throughput","QPS","measured","132 / measured wall seconds"),
 ("ai_supported_substantive_answers_per_wall_second","AI-supported substantive yield per second","answers/s","derived AI reference","AI-supported substantive answers / wall seconds"),
 ("accuracy_vs_ai","Detector accuracy vs AI","fraction","AI reference","observed detector + accepted binary label"),
 ("precision_vs_ai","Detector precision vs AI","fraction","AI reference","predicted positive with accepted binary label"),
 ("recall_vs_ai","Detector recall vs AI","fraction","AI reference","AI positive with observed detector"),
 ("f1_vs_ai","Detector F1 vs AI","fraction","AI reference","paired observed binary predictions"),
 ("human_gold_accuracy","Human-gold accuracy","fraction","unavailable","no independent adjudication"),
)
def report(source=BASE,full=FULL,ai=AI,output=None):
    base,fm,am,audit,summary,rows=load(source,full,ai)
    folder=_resolve_dir(output) if output else base/("consolidated_132_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    folder.mkdir(parents=True,exist_ok=False)
    csvwrite(folder/"consolidated_B1_B8_132.csv",summary)
    csvwrite(folder/"consolidated_per_query_132.csv",rows)
    wide=[]
    for key,label,unit,kind,denom in METRICS:
        vals={r["model"]:r[key] for r in summary}
        a,b=vals["B7"],vals["B8"]
        wide.append(dict(metric=label,unit=unit,provenance=kind,denominator=denom,
                         **vals,B8_minus_B7=(b-a if a is not None and b is not None else None)))
    csvwrite(folder/"comparison_thesis_metrics_132.csv",wide)
    # Same-query groups, not assumed independent clinical samples.
    for field in ("query_type","medical_domain"):
        grouped=[]
        for model in MODELS:
            for value in sorted({r[field] for r in rows if r["model"]==model}):
                g=[r for r in rows if r["model"]==model and r[field]==value]
                grouped.append(dict(model=model,**{field:value},**aggregate(g),
                                    mean_latency_ms=mean(r["latency_ms"] for r in g),
                                    candidate_evidence_rate=fraction(sum(r["evidence_returned"] is True for r in g),len(g))))
        csvwrite(folder/f"consolidated_by_{field}_132.csv",grouped)
    (folder/"paired_B7_B8_analysis.json").write_text(json.dumps(paired(rows),indent=2)+"\n")
    metadata=dict(schema="THESIS132_CONSOLIDATED_V1",models=list(MODELS),records=len(rows),
                  source_manifest_sha256=digest(base/"export_manifest.json"),
                  full_metrics_manifest_sha256=digest(_resolve_dir(full)/"full_metrics_manifest.json"),
                  ai_analysis_manifest_sha256=digest(_resolve_dir(ai)/"ai_analysis_manifest.json"),
                  semantic_encoder_revision=fm["semantic_proxy_encoder_revision"],
                  limitations=LIMITATIONS+am["limitations"],
                  output_sha256={p.name:digest(p) for p in sorted(folder.iterdir())})
    (folder/"consolidated_manifest.json").write_text(json.dumps(metadata,indent=2)+"\n")
    return folder

def display(folder):
    folder=_resolve_dir(folder)
    m=json.loads((folder/"consolidated_manifest.json").read_text())
    need(m["schema"]=="THESIS132_CONSOLIDATED_V1","Unexpected schema")
    for name,sha in m["output_sha256"].items():
        need(Path(name).name==name and digest(folder/name)==sha,"Output changed: "+name)
    rows=csvread(folder/"comparison_thesis_metrics_132.csv")
    print("\nB1-B8 CONSOLIDATED EXPERIMENT — 132 DEVELOPMENT QUESTIONS")
    print("AI reference is not human gold. N/A is unavailable, not zero.")
    print(f"{'Metric':<46}"+"".join(f"{m:>11}" for m in MODELS))
    for row in rows:
        def fmt(m):
            v=num(row[m])
            if v is None:return "N/A"
            if row["unit"]=="fraction":return f"{v*100:.2f}%"
            if row["unit"]=="ms":return f"{v:.2f}"
            return f"{v:.4f}"
        print(f"{row['metric']:<46}"+"".join(f"{fmt(m):>11}" for m in MODELS))
    print("\nPAIRED B7/B8:",(folder/"paired_B7_B8_analysis.json").read_text())
    print("Results:",folder)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir",type=Path,default=BASE)
    p.add_argument("--full-dir",type=Path,default=FULL)
    p.add_argument("--ai-dir",type=Path,default=AI)
    p.add_argument("--output-dir",type=Path)
    p.add_argument("--report-dir",type=Path)
    a=p.parse_args()
    display(a.report_dir or report(a.input_dir,a.full_dir,a.ai_dir,a.output_dir))
if __name__=="__main__":main()
