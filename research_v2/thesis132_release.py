"""Final, reproducible thesis export. No model calls or new labels."""
from __future__ import annotations
import argparse,json,math,shutil,statistics
from pathlib import Path
from research_v2.thesis132_b9_final import (
    ROOT,T,OUT,RUNS,RUN,REVIEW,FULL,Q,archived,readcsv,writecsv,sha,dump,stamp,need,aggregate,NEW_AUDIT)
from research_v2.thesis132_ai_evaluation import VERDICTS,truth
from research_v2.thesis132_b9_review import AUDIT,BASE
from research_v2.thesis132_b9_v2 import generation_settings

QUALITY=OUT/"quality_20261125T212505161710Z"
SEMANTIC=OUT/"semantic_20261125T211108434735Z"
PACKAGE=OUT/"review_package_20261125T211125453041Z"
ROUND=T/"b9_ablation_v2/ablation_132_20261125T194115545393Z"
COMPONENT=T/"b9_component_ablation_v2/comparison_20261125T200056094771Z_97903e"
PROFILE=T/"b9_inference_optimization_v2/comparison_20261125T202923195711Z_b9bb44"
GEN=T/"b9_generator_comparison_v2/comparison_20261125T210538858356Z_2cf5b2"
CACHE=T/"b9_component_ablation_v2/diagnostics_20261125T200034038364Z_bf71d3"
FINAL=T/"final_thesis_results_132"
PRIMARY="B9_v2"
FAST="llama3b"
LIMITS=[
 "Fixed development-exposed cohort; no independent held-out clinical evaluation or independently human-adjudicated gold.",
 "AI-reference judgments are single-judge, with limited source access and reused annotations; zero observed Yes labels do not prove zero hallucinations.",
 "Strict supported-and-grounded yield includes incomplete or irrelevant answers. Completeness is a distinct B9-only AI judgment.",
 "Graph/corpus and answer generation differ between B1-B8 and B9; descriptive comparisons, not causal attribution to any one component.",
 "Semantic similarities are cosine proxies, with varying text bases and observation counts; they are not medical entailment or retrieval recall.",
 "All-query latency includes abstentions; compare answered-only and common-answered subsets too. Single runs have uncontrolled OS/cache/thermal conditions.",
 "Adaptive reduces retrieval rounds, not necessarily LLM calls or end-to-end latency. Selector-driven retry was not exercised in the reference run.",
 "Resource samples describe system usage and sampled RSS, not energy or exclusive model memory. Cross-model tokens depend on tokenization.",
 "Primary configuration selected after observing development results. Compact-ID variant is faster but has fewer complete answers; graph-off has more complete answers."
]
def manifest(folder,key="outputs_sha256"):
    m=json.loads((folder/"manifest.json").read_text())
    for f,h in m[key].items():need(Path(f).name==f and sha(folder/f)==h,"Changed output "+str(folder/f))
    return m
def close(a,b):
    if a is None:return b==""
    if isinstance(a,(int,float)):return math.isclose(float(b),float(a),rel_tol=1e-9,abs_tol=1e-10)
    return str(a)==b
def typed(r):
    return dict(r,substantive=truth(r["substantive"]),fabricated_citation=truth(r["fabricated_citation"]),
        detector_flag=truth(r["detector_flag"]),latency_ms=float(r["latency_ms"]),
        claims_without_evidence_ids=int(r["claims_without_evidence_ids"]),**{v:int(r[v]) for v in VERDICTS})
def strict_complete(r):
    return (r["label"]=="No" and r["substantive"] and r["grounding"]=="FULLY_GROUNDED" and
        r["answer_completeness"]=="COMPLETE" and r["SUPPORTED"]>0 and
        not any(r[v] for v in ("PARTIALLY_SUPPORTED","CONTRADICTED","UNSUPPORTED","UNVERIFIABLE")))
def audited():
    rows,hashes=archived()
    qm=manifest(QUALITY);sm=manifest(SEMANTIC)
    need(qm["run_manifest_sha256"]==sm["run_manifest_sha256"]==hashes,"Run links differ")
    need(qm["package_sha256"]==sha(PACKAGE/"package_manifest.json") and
        qm["review_audit_sha256"]==sha(PACKAGE/"returned_review"/NEW_AUDIT) and
        qm["prior_audit_sha256"]==sha(REVIEW/AUDIT),"Review provenance changed")
    need(sm["baseline_semantic_manifest_sha256"]==sha(FULL/"full_metrics_manifest.json"),"Baseline semantic reference changed")
    source={(r["run"],r["query_id"]):r for r in rows}
    qrows=[typed(r) for r in readcsv(QUALITY/"variant_quality_per_query.csv")]
    qmap={(r["run"],r["query_id"]):r for r in qrows}
    need(len(qrows)==len(qmap)==len(source)==1452 and set(qmap)==set(source),"Quality alignment")
    for k,r in qmap.items():
        s=source[k]
        need(r["pipeline_status"]==s["pipeline_status"] and r["answer_scope"]==s["answer_scope"] and
             math.isclose(r["latency_ms"],s["latency_ms"],rel_tol=1e-9),"Quality record mismatch")
    qs=readcsv(QUALITY/"variant_quality_summary.csv")
    ss=readcsv(SEMANTIC/"B9_variant_semantic_summary.csv")
    need([r["run"] for r in qs]==[r["run"] for r in ss]==list(RUNS),"Summary model alignment")
    semmap={r["run"]:r for r in ss}
    variants=[]
    for q in qs:
        name=q["run"];g=[r for r in qrows if r["run"]==name];s=semmap[name]
        for k,v in aggregate(g).items():need(close(v,q[k]),"Quality summary mismatch "+name+"/"+k)
        for col,label in (("complete_answers","COMPLETE"),("partial_answers","PARTIAL"),
            ("nonresponsive_answers","NONRESPONSIVE"),("appropriate_abstentions","APPROPRIATE_ABSTENTION"),
            ("inappropriate_abstentions","INAPPROPRIATE_ABSTENTION")):
            need(int(q[col])==sum(r["answer_completeness"]==label for r in g),"Completeness summary mismatch")
        need(close(sum(strict_complete(r) for r in g)/132,q["strict_complete_yield"]),"Strict complete mismatch")
        run=json.loads((RUNS[name]/"summary.json").read_text())
        wall=run.get("wall_time_seconds",run.get("wall_seconds"))
        need(wall and close(132/wall,q["throughput_qps"]),"Throughput mismatch")
        ar=[r for r in rows if r["run"]==name]
        variants.append(dict(**q,**{k:v for k,v in s.items() if k not in ("run","queries")},
            evidence_count=sum(bool(r["retrieved_documents"] or r["causal_paths"]) for r in ar),
            answers_present=sum(bool(r["final_response"]) for r in ar),
            mean_answered_latency_ms=statistics.mean([r["latency_ms"] for r in ar if r["final_response"]]),
            complete_answers_per_second=int(q["complete_answers"])/wall,
            strict_complete_answers_per_second=float(q["strict_complete_yield"])*132/wall))
    models=readcsv(SEMANTIC/"consolidated_B1_B9_with_semantic_132.csv")
    need([r["model"] for r in models]==[f"B{i}" for i in range(1,10)],"Baseline model order")
    v=next(r for r in variants if r["run"]==PRIMARY)
    for k in ("substantive_answers","strict_supported_and_fully_grounded_yield","hallucination_rate_ai","mean_latency_ms","context_cosine_proxy"):
        need(close(float(v[k]),models[-1][k]),"B9 mismatch: "+k)
    return models,variants,qrows,hashes,qm,sm
def charts(dest,models,variants):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False,"svg.fonttype":"none"})
    folder=dest/"figures";folder.mkdir()
    def save(fig,name):
        fig.tight_layout()
        for ext in ("png","svg"):fig.savefig(folder/(name+"."+ext),dpi=300,bbox_inches="tight")
        plt.close(fig)
    def bar(rows,x,k,title,ylabel,name,mult=1):
        fig,ax=plt.subplots(figsize=(9,4.6))
        ax.bar([r[x] for r in rows],[float(r[k])*mult for r in rows])
        ax.set(title=title,ylabel=ylabel);ax.tick_params(axis="x",rotation=35);ax.grid(axis="y",alpha=.2)
        save(fig,name)
    bar(models,"model","mean_latency_ms","B1-B9: mean latency (all 132 queries)","Seconds","01_baseline_latency",.001)
    bar(models,"model","strict_supported_and_fully_grounded_yield","B1-B9: strict supported and grounded yield (AI reference)","Percent of all 132 queries","02_baseline_strict_yield",100)
    bar(models,"model","hallucination_rate_ai","B1-B9: AI-reference hallucination estimate","Percent of accepted binary labels","03_baseline_ai_hallucination",100)
    bar(variants,"run","strict_complete_yield","B9 variants: strict supported, grounded and complete yield","Percent of all 132 queries","04_variant_strict_complete",100)
    fig,ax=plt.subplots(figsize=(10,5.2))
    for r in variants:
        x=float(r["mean_latency_ms"])/1000;y=100*float(r["strict_complete_yield"])
        ax.scatter(x,y);ax.annotate(r["run"],(x,y),xytext=(4,4),textcoords="offset points",fontsize=8)
    ax.set(xlabel="Mean latency (seconds; all queries)",ylabel="Strict supported, grounded and complete yield (%)",title="B9 quality-efficiency trade-off (AI reference)")
    ax.grid(alpha=.2);save(fig,"05_variant_tradeoff")
    gen=[r for r in variants if r["run"] in ("llama3b","llama1b","qwen15b")]
    bar(gen,"run","answers_present","Generator comparison: answer availability","Queries answered (of 132)","06_generator_coverage")
def export(output=None,plots=True):
    models,variants,qrows,hashes,qm,sm=audited()
    manifests={str(p.relative_to(ROOT)):sha(p/"manifest.json") for p in (QUALITY,SEMANTIC)}
    for p in (ROUND,COMPONENT,PROFILE,GEN):
        manifest(p);manifests[str(p.relative_to(ROOT))]=sha(p/"manifest.json")
    dest=Path(output).resolve() if output else FINAL/("release_"+stamp())
    dest.mkdir(parents=True,exist_ok=False)
    writecsv(dest/"table_B1_B9_132.csv",models)
    writecsv(dest/"table_B9_variants_132.csv",variants)
    for src,name in ((ROUND/"ablation_summary_132.csv","table_round_policy_operational.csv"),
        (COMPONENT/"component_summary_132.csv","table_component_operational.csv"),
        (PROFILE/"inference_comparison_132.csv","table_identifier_resource_profile.csv"),
        (GEN/"generator_summary_132.csv","table_generator_resource_profile.csv"),
        (CACHE/"diagnostics.json","cache_feedback_diagnostics.json")):
        shutil.copyfile(src,dest/name)
    bytype=[]
    for name in RUNS:
        for cat in sorted({r["query_type"] for r in qrows}):
            rs=[r for r in qrows if r["run"]==name and r["query_type"]==cat]
            bytype.append(dict(run=name,query_type=cat,questions=len(rs),
                answers=sum(r["pipeline_status"]=="ok" for r in rs),
                complete=sum(r["answer_completeness"]=="COMPLETE" for r in rs),
                strict_complete=sum(strict_complete(r) for r in rs),ai_yes=sum(r["label"]=="Yes" for r in rs),
                ai_uncertain=sum(r["label"]=="Uncertain" for r in rs)))
    writecsv(dest/"table_B9_by_query_type.csv",bytype)
    ref=json.loads((RUN/"manifest.json").read_text())
    dump(dest/"final_architecture_lock.json",dict(schema="THESIS132_FINAL_B9_SELECTION_V1",
        primary=PRIMARY,optional_speed_variant=FAST,selection_basis="Quality-oriented development-set decision; not universally dominant",
        primary_run=str(RUN.relative_to(ROOT)),primary_manifest_sha256=hashes[PRIMARY],
        optional_run=str(RUNS[FAST].relative_to(ROOT)),optional_manifest_sha256=hashes[FAST],
        dataset_sha256=ref["dataset_sha256"],inputs=ref["inputs"],model_digest=ref["model_digest"],
        generation_settings=ref["generation_settings"],policy=ref["policy"],top_k=ref["top_k"],
        max_retrieval_rounds=3,selector="original_ids",medical_gold=False,
        reproduction_command="python run_b9_v2_132.py --allow-live --policy adaptive --top-k 5",
        limitations=LIMITS))
    (dest/"README.md").write_text("# Final thesis experiment export\n\nPrimary: B9-v2 adaptive, original evidence IDs. Optional speed variant: Llama 3B compact IDs.\n\n"
        "All tables derive from archived development experiments. No model inference or new labels. Empty metric cells mean unavailable, not zero. Operational ablation tables are historical exports; use table_B9_variants_132.csv for completed variant quality labels. Strict complete yield includes AI completeness and has no B1-B8 equivalent.\n\n## Limitations\n\n"+
        "\n".join("- "+s for s in LIMITS)+"\n",encoding="utf-8")
    if plots:charts(dest,models,variants)
    dump(dest/"manifest.json",dict(schema="THESIS132_FINAL_EXPORT_V1",primary=PRIMARY,optional_speed_variant=FAST,
        dataset_sha256=sha(Q),source_manifests=manifests,run_manifest_sha256=hashes,
        cache_diagnostics_sha256=sha(CACHE/"diagnostics.json"),limitations=LIMITS,
        output_sha256={str(p.relative_to(dest)):sha(p) for p in sorted(dest.rglob("*")) if p.is_file()}))
    print("\nFinal B1-B9 experiment (AI reference; 132 development questions)")
    print(f"{'Model':<7}{'Substantive':>13}{'AI hall.':>11}{'Strict yield':>14}{'Mean s':>11}{'QPS':>10}")
    for r in models:
        print(f"{r['model']:<7}{int(r['substantive_answers']):>13}{100*float(r['hallucination_rate_ai']):>10.2f}%{100*float(r['strict_supported_and_fully_grounded_yield']):>13.2f}%{float(r['mean_latency_ms'])/1000:>11.3f}{float(r['run_throughput_qps']):>10.4f}")
    print("Output:",dest);return dest
def verify(folder):
    folder=Path(folder).resolve();m=json.loads((folder/"manifest.json").read_text())
    need(m["schema"]=="THESIS132_FINAL_EXPORT_V1","Wrong final export")
    for f,h in m["output_sha256"].items():
        p=(folder/f).resolve();need(p.is_relative_to(folder) and sha(p)==h,"Output integrity mismatch "+f)
    print("Verified final export:",folder)
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output-dir",type=Path);p.add_argument("--no-plots",action="store_true");p.add_argument("--verify",type=Path);a=p.parse_args()
    if a.verify:verify(a.verify)
    else:export(a.output_dir,not a.no_plots)
if __name__=="__main__":main()
