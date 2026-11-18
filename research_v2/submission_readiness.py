"""Independent, read-only consistency audit of an assembled 48-hour MTP submission.

Verifies exported tables against the preserved frozen 100/32 datasets, the
reviewed G1 evidence and shadow G0/G1 results. Creates NEW versioned audit
and presentation factsheets; NEVER overwrites original runs or reviews.

No generation, Neo4j, PPO, network access or medical accuracy grading.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from .experiment_runner import PROJECT_ROOT, file_sha256
from .partial_submission_sprint import (
    ROOT, MODEL100, MODEL32, Q100, Q32, _expansion, _safe,
    _read_rows, _read_json, _write_rows, _cohort,
)
from .presentation_matrix import validate_inputs

WORKSPACE=ROOT/"presentation_20261118"
DEFAULT_OUTPUT=WORKSPACE/"submission_readiness"


def _int(row,key):
    return int(row[key])


def _float(row,key):
    return float(row[key])


def _require(cond,message):
    if not cond:
        raise ValueError("SUBMISSION AUDIT FAILED: "+message)


def _eq(actual,expected,message):
    _require(actual==expected,message+": "+repr(actual)+" != "+repr(expected))


def _model_map(rows,field="model"):
    values={row[field]:row for row in rows}
    _eq(len(values),len(rows),"Duplicate model entries")
    return values


def _validate_exported_rows(root,summary,details,category):
    questions100,_=validate_inputs(Q100)
    questions32,_=_expansion()
    expected_ids={q.query_id for q in [*questions100,*questions32]}
    _eq(len(expected_ids),132,"Frozen 100+32 query IDs must be distinct")
    _eq(len(details),792,"Combined 132-question raw records")
    _eq(len(summary),6,"Combined B1-B6 summary model count")
    _eq({r["model"] for r in summary},set(MODEL32),"Combined summary model set")
    _eq(set(summary[0]),{
        "model","questions","evidence_returned","abstained","errors",
        "mean_retrieval_ms","evidence_basis","mode",
    },"Summary fields")
    seen=set()
    by_model=defaultdict(list)
    by_cat=defaultdict(list)
    for r in details:
        model=r["model_id"];qid=r["query_id"]
        _require(model in MODEL32 and qid in expected_ids,"Unknown model/query "+str((model,qid)))
        _require((model,qid) not in seen,"Duplicate model/query "+str((model,qid)))
        seen.add((model,qid))
        _require(r["status"] in {"ok","abstained"},"Unexpected query status "+r["status"])
        _require(r["returned_evidence"] in {"True","False"},"Invalid evidence flag")
        _require(
            (r["status"]=="ok")==(r["returned_evidence"]=="True"),
            "Successful/abstention status and evidence flag mismatch for "+str((model,qid)))
        _require(float(r["retrieval_ms"])>=0,"Missing/negative retrieval milliseconds")
        by_model[model].append(r)
        by_cat[(model,r["query_type"])].append(r)
    for m in MODEL32:
        _eq({r["query_id"] for r in by_model[m]},expected_ids,"Missing questions for "+m)
    for m,r in _model_map(summary).items():
        items=by_model[m]
        _eq(_int(r,"questions"),132,"Question count for "+m)
        _eq(_int(r,"evidence_returned"),sum(x["returned_evidence"]=="True" for x in items),"Evidence yield for "+m)
        _eq(_int(r,"abstained"),sum(x["status"]=="abstained" for x in items),"Abstentions for "+m)
        _eq(_int(r,"errors"),0,"Errors for "+m)
        _eq(r["mode"],"retrieval_only","Retrieval mode for "+m)
        basis=("provisional_AI_assisted_source_summaries"
               if m in MODEL32[:4] else "G0_unverified_graph")
        _eq(r["evidence_basis"],basis,"Evidence basis for "+m)
        mean=sum(float(x["retrieval_ms"]) for x in items)/132
        _require(abs(mean-_float(r,"mean_retrieval_ms"))<0.00002,
                 "Per-query mean differs from summary for "+m)
    _eq(len(category),len(by_cat),"By-category summary row count")
    for row in category:
        key=(row["model"],row["query_type"])
        _require(key in by_cat,"Unknown model/category: "+str(key))
        items=by_cat[key]
        _eq(_int(row,"questions"),len(items),"Category question count "+str(key))
        _eq(_int(row,"evidence_returned"),
            sum(x["returned_evidence"]=="True" for x in items),"Category yield "+str(key))
        _eq(_int(row,"abstained"),
            sum(x["status"]=="abstained" for x in items),"Category abstained "+str(key))


def _validate_100_32(bundle,combined):
    first=_model_map(_read_rows(bundle/"clean100_retrieval_only_B1_B6.csv"))
    ext=_model_map(_read_rows(bundle/"new32_B1_B6.csv"))
    _eq(set(first),set(MODEL32),"100-query models")
    _eq(set(ext),set(MODEL32),"32-query models")
    for model in MODEL32:
        a=first[model];b=ext[model];both=combined[model]
        _eq(_int(a,"n"),100,"Pilot n "+model)
        _eq(_int(b,"n"),32,"Expansion n "+model)
        _eq(_int(a,"status_error"),0,"Pilot execution errors "+model)
        _eq(_int(b,"status_error"),0,"Expansion execution errors "+model)
        _eq(_int(both,"evidence_returned"),
            _int(a,"evidence_returned")+_int(b,"evidence_returned"),
            "100+32 evidence yield for "+model)
        _eq(_int(both,"abstained"),
            _int(a,"status_abstained")+_int(b,"status_abstained"),
            "100+32 abstentions for "+model)
        weighted=(
            100*_float(a,"mean_retrieval_ms")+32*_float(b,"mean_retrieval_ms")
        )/132
        _require(abs(weighted-_float(both,"mean_retrieval_ms"))<0.0001,
                 "100/32 weighted latency mismatch for "+model)


def _validate_history(bundle,manifest):
    history=_model_map(_read_rows(bundle/"historical100_B1_B8.csv"))
    _eq(set(history),set(MODEL100),"Historical B1–B8 model set")
    _eq(manifest["historical_800_model_query_records"],800,"Historical model-query slots")
    for model,row in history.items():
        _eq(_int(row,"n"),100,"Historical query slots "+model)
        _eq(_int(row,"n"),sum(_int(row,metric) for metric in
                             ("status_ok","status_abstained","status_error")),
            "Historical execution statuses "+model)
        _eq(row["mode"],"end_to_end","Historical execution mode "+model)
    _eq(sum(_int(r,"status_error") for r in history.values()),
        manifest["historical_archive_execution_errors"],
        "Archived error-count consistency")
    # Selected archival runs are allowed to contain errors, but cannot be
    # falsely characterized as error-free.
    good=sorted(m for m,r in history.items() if _int(r,"status_error")==0)
    _eq(sorted(manifest["historical_error_free_models"]),good,
        "Historically successful model list")
    return history


def _validate_graph_and_reviews(bundle,manifest,combined):
    graph=_model_map(_read_rows(bundle/"G0_G1_132_shadow.csv"))
    _eq(set(graph),{"B5","B6"},"G0/G1 paired graph families")
    for model,row in graph.items():
        _eq(_int(row,"G0_yield_132"),_int(combined[model],"evidence_returned"),
            "B5/B6 frozen G0 evidence agreement across independent runs: "+model)
        for field in ("G0_yield_132","G1_legacy_yield_132","G1_stable_yield_132"):
            _require(0<=_int(row,field)<=132,"Graph yield out of bounds "+model+" "+field)
    review=_model_map(_read_rows(bundle/"reviewed_changed_case_B5_B6.csv"),"family")
    _eq(set(review),{"B5","B6"},"Reviewed family set")
    _eq(sum(_int(r,"selected_changed_cases") for r in review.values()),31,
        "Reviewed model-query cases")
    _eq(manifest["reviewed_case_count"],31,"Reviewed case count in manifest")
    _eq(manifest["reviewed_evidence_occurrences"],132,"Reviewed evidence occurrences")
    _eq(manifest["reviewed_distinct_assertions"],58,"Distinct question-evidence assertions")
    _eq(manifest["attention_cases"],17,"Flagged attention cases")
    for model,row in review.items():
        _require(
            0<=_int(row,"cases_with_relevant_direction_correct_checked_candidate")
            <=_int(row,"cases_with_relevant_direction_correct_evidence")
            <=_int(row,"G1_stable_cases_with_evidence")
            <=_int(row,"selected_changed_cases"),
            "Inconsistent reviewed evidence nested counts for "+model)
    return graph,review


def _numeric_rows(rows,key):
    return {r[key]:r for r in rows}


def _slide_notes(bundle,manifest,historical,combined,graph,review):
    ordered=lambda group:[group[m] for m in sorted(group)]
    lines=[
        "# CDF-RAG-2 — audited slide factsheet",
        "",
        "**Scope:** existing 100 + 32 draft questions; no independently "
        "adjudicated medical gold or clinical accuracy.",
        "",
        "## Panel 1: B1–B8 archived end-to-end (100 questions)",
        "",
        "| Model | OK | Abstained | Errors | Evidence | Mean end-to-end (s) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for m,r in ordered(historical):
        lines.append(
            "| "+m+" | "+r["status_ok"]+" | "+r["status_abstained"]+
            " | "+r["status_error"]+" | "+r["evidence_returned"]+
            " | "+f'{float(r["mean_total_ms"])/1000:.3f}'+" |"
        )
    lines.extend([
        "",
        "**Caveat:** different evidence bases; B7/B8 use legacy graph/PPO, "
        "not the new typed G1. Execution OK is not answer correctness.",
        "",
        "## Panel 2: clean B1–B6 retrieval-only (132 questions)",
        "",
        "| Model | Evidence / 132 | Abstained | Mean retrieval (ms) |",
        "|---|---:|---:|---:|",
    ])
    for m,r in ordered(combined):
        lines.append("| "+m+" | "+r["evidence_returned"]+
                     " | "+r["abstained"]+" | "+
                     f'{float(r["mean_retrieval_ms"]):.3f}'+" |")
    lines.extend([
        "",
        "**Caveat:** scores reflect returned evidence, not relevance. "
        "Pretrained embedding initialization and LLM generation are excluded.",
        "",
        "## Panel 3: paired graph retrieval on 132 queries",
        "",
        "| Model | Frozen G0 | Legacy G1 | Stabilized G1 |",
        "|---|---:|---:|---:|",
    ])
    for m,r in ordered(graph):
        lines.append("| "+m+" | "+r["G0_yield_132"]+" | "+
                     r["G1_legacy_yield_132"]+" | "+r["G1_stable_yield_132"]+" |")
    lines.extend([
        "",
        "**Finding:** Graph expansion can change target matching and reduce "
        "retrieval coverage. The stability-aware shadow resolver recovers "
        "some coverage. It remains a separate algorithm, not a verified "
        "clinical improvement.",
        "",
        "## Panel 4: human-reviewed changed-case error analysis",
        "",
        "| Family | Cases selected | Stable evidence returned | "
        "Reviewed-relevant & direction-correct cases |",
        "|---|---:|---:|---:|",
    ])
    for m,r in ordered(review):
        lines.append("| "+m+" | "+r["selected_changed_cases"]+
            " | "+r["G1_stable_cases_with_evidence"]+
            " | "+r["cases_with_relevant_direction_correct_evidence"]+" |")
    flags=manifest["review_flags"]
    lines.extend([
        "",
        f'**Quality inspection:** {manifest["reviewed_evidence_occurrences"]} evidence '
        f'occurrences; {manifest["reviewed_distinct_assertions"]} distinct '
        f'question-evidence assertions; {manifest["attention_cases"]} flagged '
        f'cases. Wrong targets: {flags.get("wrong_stable_target",0)}; '
        f'uncertain targets: {flags.get("stable_target_uncertain",0)}; '
        f'unsupported gains: {flags.get("gain_without_relevant_direction_correct_evidence",0)}.',
        "",
        "**Caveat:** cases were selected for changed retrieval behavior; "
        "review judgments are not representative held-out accuracy.",
        "",
        "## Defensible preliminary claim",
        "",
        "We identified and quantified graph-expansion-induced target drift, "
        "constructed a candidate typed G1 graph with source-review records, "
        "and implemented a stability-aware query resolver. Our 132-question "
        "draft-pilot comparison shows altered and partially recovered evidence "
        "coverage. Manual review isolates unresolved target and relevance "
        "errors that motivate evidence-sufficiency-guided multi-round retrieval.",
        "",
        "## Still not established",
        "",
        "Independent clinical correctness, recall/precision against gold, "
        "hallucination reduction, statistical generalization, or superiority "
        "of an adaptive multi-round system that has not yet been completed.",
    ])
    return "\n".join(lines)+"\n"


def audit(submission_dir:Path,output_root:Path=DEFAULT_OUTPUT):
    submission=_safe(submission_dir)
    output_root=_safe(output_root)
    _require(submission.is_dir(),"Submission directory not found")
    manifest=_read_json(submission/"submission_manifest.json")
    _eq(manifest["frozen_100_sha256"],file_sha256(Q100),"Frozen 100 checksum")
    _eq(manifest["frozen_32_sha256"],file_sha256(Q32),"Frozen 32 checksum")
    _require(manifest["historical_B1_B8_all_eight"],"Historical B1-B8 not complete")
    _require(manifest["clean132_retrieval_B1_B6_complete"],"Combined 132 B1-B6 not present")
    _eq(manifest["clean132_model_query_records"],792,"Combined record count")
    _require(manifest["new32_B1_B6_complete"] and
             manifest["clean100_retrieval_B1_B6_complete"],
             "100/32 retrieval cohort not complete")
    combined_rows=_read_rows(submission/"combined132_retrieval_only_B1_B6.csv")
    combined=_model_map(combined_rows)
    details=_read_rows(submission/"all132_retrieval_only_B1_B6.csv")
    cats=_read_rows(submission/"by_query_type132_B1_B6.csv")
    _validate_exported_rows(submission,combined_rows,details,cats)
    _validate_100_32(submission,combined)
    historical=_validate_history(submission,manifest)
    graph,review=_validate_graph_and_reviews(submission,manifest,combined)
    for fname in ("submission_results.md","partial_report_outline.md",
                  "partial_ppt_outline.md","figure_manifest.json"):
        _require((submission/fname).is_file() and (submission/fname).stat().st_size>0,
                 "Missing/empty report output: "+fname)
    figures=_read_json(submission/"figure_manifest.json")
    for item in figures.get("generated",[]):
        p=Path(item)
        _require(p.is_file() and p.stat().st_size>100,"Missing/empty generated chart: "+str(p))
    checks={
        "frozen_inputs":True,
        "archived_B1_B8_100_slots":True,
        "retrieval_only_B1_B6_100_plus_32":True,
        "792_distinct_query_model_records":True,
        "measured_retrieval_latency_consistency":True,
        "category_aggregates":True,
        "G0_shadow_independent_yield_agreement":True,
        "reviewed_31_case_counts":True,
        "figures_and_outlines_present":True,
    }
    notes=[
        "This is technical/provenance validation, NOT clinical or thesis acceptance.",
        "B1-B4 and B5-B6 are on different evidence bases.",
        "G0 graph edges inherited by G1 remain medically unverified.",
        "The reviewed 31 cases are selected behavior-change cases, not a holdout.",
        "Do not compare legacy end-to-end latency to retrieval-only latency.",
    ]
    out=output_root/(
        "audit_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")+
        "_"+uuid.uuid4().hex[:6]
    )
    out.mkdir(parents=True,exist_ok=False)
    report={
        "status":"PASS_TECHNICAL_INTEGRITY",
        "submission_dir":str(submission.relative_to(ROOT)),
        "submission_manifest_sha256":file_sha256(submission/"submission_manifest.json"),
        "checked":checks,
        "new_retrieval_records":len(details),
        "baseline_models":list(MODEL32),
        "selected_historical_models":list(MODEL100),
        "medical_gold":False,
        "claims":notes,
        "figures_present":len(figures.get("generated",[])),
        "files":["audit.json","slide_factsheet.md"],
    }
    (out/"audit.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    (out/"slide_factsheet.md").write_text(
        _slide_notes(submission,manifest,historical,combined,graph,review),
        encoding="utf-8",
    )
    print(json.dumps({"output":str(out),**report},indent=2))
    return out,report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--submission-dir",type=Path,required=True)
    p.add_argument("--output-root",type=Path,default=DEFAULT_OUTPUT)
    args=p.parse_args()
    if Path.cwd().resolve()!=ROOT:
        p.error("Run from CDF-RAG-2 repository root")
    audit(args.submission_dir,args.output_root)


if __name__=="__main__":
    main()
