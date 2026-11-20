"""Canonical MTP provenance audit. No model runs, external services, or writes.

Refuses invented clinical gold, mismatched source bytes, stale run selections,
or accidental pooling of legacy 22, 100-question end-to-end and 132-query
retrieval-only results. Local-only observations remain explicitly labeled.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
REGISTRY=ROOT/"research_artifacts/unified/experiment_registry_v1.json"
EXPECTED_MODELS={f"B{i}" for i in range(1,9)}


def check(cond,msg):
    if not cond:
        raise ValueError("MTP_UNIFIED_INTEGRITY_FAILED: "+msg)


def blob_sha(path):
    raw=path.read_bytes()
    return hashlib.sha1(b"blob "+str(len(raw)).encode()+b"\0"+raw).hexdigest()


def load_csv(path):
    with path.open(newline="",encoding="utf-8") as f:
        return list(csv.DictReader(f))


def close(a,b,msg,tol=1e-7):
    check(math.isclose(float(a),float(b),rel_tol=tol,abs_tol=tol),msg)


def verify():
    registry=json.loads(REGISTRY.read_text(encoding="utf-8"))
    check(registry["registry_version"]=="1.0","Schema version mismatch")
    check(registry["canonical_branch"]=="master","Expected default branch master")
    source22=registry["study_22_deterministic"]
    path22=ROOT/source22["source"]
    check(blob_sha(path22)==source22["source_git_blob_sha"],"Legacy22 CSV modified")
    rows22=load_csv(path22)
    check(len(rows22)==16 and source22["queries"]==22,
          "22-query metric inventory changed")
    for row in rows22:
        observed=source22["metrics"][row["Metric"]]
        close(row["No-RL"],observed["no_rl"],"No-RL metric altered: "+row["Metric"])
        close(row["RL"],observed["rl"],"RL metric altered: "+row["Metric"])
        close(row["RL - No-RL"],observed["delta"],"Delta altered: "+row["Metric"])
    check(source22["status"]=="TRACKED_VERIFIABLE","Legacy22 status lost")
    alt=registry["study_22_alternate_gold_named_exports"]
    for item in alt["sources"]:
        check(blob_sha(ROOT/item["path"])==item["git_blob_sha"],
              "Historical alternate output modified: "+item["path"])
    check(alt["gold_in_filename_does_not_imply_independent_gold"],
          "Alternate export falsely certified as medical gold")
    e100=registry["study_100_end_to_end"]
    check(blob_sha(ROOT/e100["source"])==e100["source_git_blob_sha"],
          "Selected 100-query result CSV changed")
    csv100=load_csv(ROOT/e100["source"])
    check(len(csv100)==len(e100["selected_models"])==8,
          "Expected eight selected B1-B8 results")
    by_id={r["model_id"]:r for r in csv100}
    check(set(by_id)==EXPECTED_MODELS,"Missing/duplicate B1-B8 ID")
    for model in e100["selected_models"]:
        mid=model["model"]
        item=by_id[mid]
        n=int(item["queries"])
        success=int(item["success"])
        abstained=int(item["abstained"])
        errors=int(item["errors"])
        check(n==100 and success+abstained+errors==100 and errors==0,
              "Selected archive not error-free for "+mid)
        check((n,success,abstained,errors)==(
              model["queries"],model["status_ok"],
              model["status_abstained"],model["status_error"]),
              "Selected baseline status differs: "+mid)
        check(item["run_id"]==model["run_id"],"Run ID drift: "+mid)
        check(item["execution_mode"]=="end_to_end"==model["mode"],
              "End-to-end run mode mismatch: "+mid)
        check(item["evidence_basis"]==model["evidence_basis"],
              "Evidence basis mismatch: "+mid)
        close(item["mean_latency_s"],model["mean_total_latency_s"],"Latency changed: "+mid)
        close(item["p50_latency_s"],model["p50_total_latency_s"],"P50 changed: "+mid)
        close(item["p95_latency_s"],model["p95_total_latency_s"],"P95 changed: "+mid)
        close(float(item["retrieval_yield_rate"])*100,model["evidence_yield_pct"],
              "Evidence yield changed: "+mid)
        recorded=ROOT/"presentation_20261118/runs"/item["run_id"]
        summary=json.loads((recorded/"summary.json").read_text(encoding="utf-8"))
        check(summary["run_id"]==item["run_id"] and
              summary["total_queries"]==100 and
              summary["errors"]==0 and
              summary["successful"]==success and
              summary["abstained"]==abstained,
              "Archived run summary mismatch: "+mid)
        check((recorded/"records.jsonl").exists(),"Selected raw run absent: "+mid)
    pair=registry["paired_b7_b8"]
    check(blob_sha(ROOT/pair["source"])==pair["source_git_blob_sha"],
          "B7/B8 paired comparison modified")
    p=json.loads((ROOT/pair["source"]).read_text(encoding="utf-8"))
    check(p["provenance_verified"] and not p["clinical_accuracy_measured"],
          "B7/B8 paired result validity changed")
    pm=p["paired_operational_metrics"]
    check(pm["B7_evidence_yield"]==pair["observed_b7_graph_evidence"]==49 and
          pm["B8_evidence_yield"]==pair["observed_b8_graph_evidence"]==65,
          "B7/B8 yield disagrees")
    close(pm["B8_minus_B7_mean_paired_latency_ms"],
          pair["paired_mean_latency_delta_ms"],"Paired B7/B8 delta drift")
    for mid in ("B7","B8"):
        key=mid.lower()+"_micro_f1"
        close(p["g0_mapped_structural_metrics"][mid]["micro_f1"],
              pair["structural_reference_20"][key],
              "G0 structural score changed: "+mid)
    q132=registry["study_132_retrieval_only"]
    check(q132["questions"]==132 and q132["query_model_records"]==792 and
          not q132["raw_records_committed"],
          "Local-only 132 provenance status is wrong")
    check(len(q132["models"])==6 and
          {r["model"] for r in q132["models"]}==
          {f"B{i}" for i in range(1,7)},
          "Expected 6 locally reported B1-B6 retrieval runs")
    for r in q132["models"]:
        check(r["evidence_100"]+r["evidence_32"]==r["evidence_132"] and
              r["query_count"]==132 and r["errors"]==0 and
              r["mode"]=="retrieval_only",
              "Local 132 summary inconsistent for "+r["model"])
    graphs=registry["graph_expansion_132"]
    for mid in ("B5","B6"):
        corresponding=next(row for row in q132["models"] if row["model"]==mid)
        check(corresponding["evidence_132"]==graphs[mid]["g0_evidence"],
              "G0 132 graph yield inconsistent: "+mid)
    review=registry["manually_reviewed_changes"]
    check(review["reviewed_model_query_cases"]==31 and
          review["evidence_occurrences"]==132 and
          review["flagged_model_query_cases"]==17 and
          review["local_outputs_not_in_git"],
          "Local G1 review status inconsistent")
    data=registry["dataset_132_v2"]
    check(blob_sha(ROOT/data["manifest"])==data["manifest_git_blob_sha"],
          "Benchmark v2 manifest modified")
    manifest=json.loads((ROOT/data["manifest"]).read_text(encoding="utf-8"))
    check(manifest["count"]==132 and
          manifest["clinical_gold_question_count"]==0 and
          manifest["independent_clinician_reviewed_question_count"]==0,
          "Bench v2 expert-review status falsified")
    check(data["clinically_verified_gold_questions"]==0 and
          data["independent_clinician_reviewed_questions"]==0,
          "Registry falsely claims clinical certification")
    root_readme=(ROOT/"readme.md").read_text(encoding="utf-8")
    check("Canonical branch: master" in root_readme,
          "Canonical project readme not reconciled")
    check("100-query run pending" not in root_readme,
          "Root readme has stale B7/B8 pending claim")
    result={
        "status":"PASS_MTP_UNIFIED_PROVENANCE",
        "legacy22_metrics":len(rows22),
        "archived_end_to_end_models":len(csv100),
        "archived_100_model_query_slots":800,
        "selected_archive_errors":sum(int(r["errors"]) for r in csv100),
        "locally_audited_unimported_retrieval_records":792,
        "locally_reported_G1_changed_cases":31,
        "dataset_source_screened_questions":132,
        "independently_certified_medical_gold_questions":0,
        "cross_cohort_accuracy_pooling_permitted":False,
        "local_output_import_pending":True,
        "legacy_branches_retained_for_history":True,
    }
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json",action="store_true")
    args=parser.parse_args()
    result=verify()
    print(json.dumps(result,indent=2) if args.json else
          "PASS_MTP_UNIFIED_PROVENANCE: "+json.dumps(result,sort_keys=True))


if __name__=="__main__":
    main()
