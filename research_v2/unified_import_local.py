"""Stage hash-checked, allowlisted local MTP results for optional Git archival.

This is NOT an automatic push or a medical gold-certification step. Preserves
original run folders and source reviews untouched. Requires an existing
PASS_TECHNICAL_INTEGRITY audit referring to the exact submission manifest
hash; copies only selected reproducible aggregate files to a new directory.
To include query-level records or reviewer examples requires explicit flags,
after checking for personal/private/copyrighted material.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
REG=ROOT/"research_artifacts/unified/experiment_registry_v1.json"
STAGING=ROOT/"research_artifacts/unified/imported_local"

AGGREGATES=[
 "submission_manifest.json","historical100_B1_B8.csv",
 "clean100_retrieval_only_B1_B6.csv","new32_B1_B6.csv",
 "combined132_retrieval_only_B1_B6.csv","by_query_type132_B1_B6.csv",
 "G0_G1_132_shadow.csv","reviewed_changed_case_B5_B6.csv",
 "G0_structural20_B7_B8.csv","legacy22_RL_vs_NoRL.csv",
 "submission_results.md","figure_manifest.json",
]
DETAILS=["all132_retrieval_only_B1_B6.csv"]
FIGURES=[
 "historical100_execution_status.png",
 "clean100_evidence_yield.png",
 "G0_G1_stable_132_evidence_yield.png",
]


def _inside(path,root=ROOT):
    """Resolve input relative to selected (possibly separate) checkout."""
    p=Path(path).expanduser()
    if not p.is_absolute():
        p=Path(root)/p
    p=p.resolve()
    if not p.is_relative_to(root):
        raise ValueError("Path escapes the selected CDF-RAG-2 checkout")
    return p


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _csv(path):
    with path.open(encoding="utf-8",newline="") as stream:
        return list(csv.DictReader(stream))


def stage(submission, audit, output_root=STAGING, per_query=False,
          triage=None, reviewed_case_examples=False, case_studies=None,
          source_repo=None):
    reg=_read(REG)
    source_root=Path(source_repo).expanduser().resolve() if source_repo else ROOT
    if not (source_root/"data/benchmark/medical_causal_pilot100_draft_v1.jsonl").is_file():
        raise ValueError("Source repo does not have frozen MTP pilot100")
    origin=_inside(submission,source_root)
    audit_folder=_inside(audit,source_root)
    dest_root=_inside(output_root,ROOT)
    audit_json=_read(audit_folder/"audit.json")
    manifest=origin/"submission_manifest.json"
    if audit_json.get("status")!="PASS_TECHNICAL_INTEGRITY":
        raise ValueError("Missing PASS_TECHNICAL_INTEGRITY")
    if audit_json.get("submission_dir")!=str(origin.relative_to(source_root)):
        raise ValueError("Audit belongs to a different submission directory")
    if audit_json.get("submission_manifest_sha256")!=_sha(manifest):
        raise ValueError("Submission manifest was modified after audit")
    if _sha(manifest)!=reg["study_132_retrieval_only"][
            "local_audit_submission_manifest_sha256"]:
        raise ValueError("Submission hash differs from canonical experiment registry")
    if _read(manifest).get("clean132_model_query_records")!=792:
        raise ValueError("Expected 792 verified B1-B6 retrieval records")
    if len(_csv(origin/"combined132_retrieval_only_B1_B6.csv"))!=6:
        raise ValueError("Six-model combined summary missing")
    if len(_csv(origin/"G0_G1_132_shadow.csv"))!=2:
        raise ValueError("Two-model G0/G1 shadow summary missing")
    if per_query:
        rows=_csv(origin/"all132_retrieval_only_B1_B6.csv")
        if len(rows)!=792:
            raise ValueError("Expected exactly 792 per-query results")
    if reviewed_case_examples and not case_studies:
        raise ValueError("--include-reviewed-examples requires --case-studies-dir")
    out=dest_root/(
        "import_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")+
        "_"+uuid.uuid4().hex[:6])
    if out.exists():
        raise FileExistsError("Import folder exists; no overwrite")
    chosen=list(AGGREGATES)+(DETAILS if per_query else [])
    for name in chosen:
        if not (origin/name).is_file():
            raise FileNotFoundError("Missing verified aggregate: "+name)
    triage_files=[]
    if triage:
        tri=_inside(triage,source_root)
        summary=tri/"summary.json"
        if not summary.is_file():
            raise FileNotFoundError("Triage summary unavailable")
        info=_read(summary)
        if info.get("attention_cases")!=17 or info.get("distinct_attention_queries")!=12:
            raise ValueError("Triage is not the reviewed 17-case snapshot")
        triage_files.append((summary,"g1_triage_summary.json"))
    case_files=[]
    if reviewed_case_examples:
        case=_inside(case_studies,source_root)
        m=case/"case_studies_manifest.json"
        if not m.is_file() or _read(m).get("cases")!=17:
            raise ValueError("Case-study manifest doesn't match reviewed cohort")
        case_files.append((m,"case_studies_manifest.json"))
        example=case/"representative_case_studies.md"
        if not example.is_file():
            raise FileNotFoundError("No reviewed example file")
        case_files.append((example,"representative_case_studies.md"))
    out.mkdir(parents=True,exist_ok=False)
    copied=[]
    for name in chosen:
        src=origin/name;dst=out/name
        shutil.copy2(src,dst)
        copied.append({"source":str(src.relative_to(source_root)),
                       "path":name,"sha256":_sha(dst),"bytes":dst.stat().st_size})
    fig_root=origin/"figures"
    for name in FIGURES:
        src=fig_root/name
        if src.is_file():
            (out/"figures").mkdir(exist_ok=True)
            shutil.copy2(src,out/"figures"/name)
            copied.append({"source":str(src.relative_to(source_root)),
                           "path":"figures/"+name,
                           "sha256":_sha(out/"figures"/name),
                           "bytes":(out/"figures"/name).stat().st_size})
    for src,name in triage_files+case_files:
        shutil.copy2(src,out/name)
        copied.append({"source":str(src.relative_to(source_root)),
                       "path":name,"sha256":_sha(out/name),
                       "bytes":(out/name).stat().st_size})
    ready={
        "status":"LOCAL_RESULTS_STAGED_NOT_AUTOMATICALLY_PUBLISHED",
        "source_audit_status":"PASS_TECHNICAL_INTEGRITY",
        "source_audit_file_sha256":_sha(audit_folder/"audit.json"),
        "source_manifest_sha256":_sha(manifest),
        "source_checkout_role":"separate_original_checkout" if source_root!=ROOT else "same_checkout",
        "destination_checkout_role":"canonical_project_checkout",
        "include_query_level_data":per_query,
        "include_reviewer_examples":reviewed_case_examples,
        "medical_gold":False,
        "files":copied,
        "warnings":[
            "The 132 dataset remains development-only, not independently medical gold.",
            "Verify outputs contain no private data or third-party copyrighted passages before committing to public GitHub.",
            "Run git status and inspect each staged file; this tool does not git add/commit/push.",
        ],
    }
    (out/"import_manifest.json").write_text(
        json.dumps(ready,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print(json.dumps({"output":str(out),"status":ready["status"],
                     "files":len(copied),"manifest_sha256":_sha(out/"import_manifest.json")},indent=2))
    return out,ready


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--submission-dir",type=Path,required=True)
    p.add_argument("--audit-dir",type=Path,required=True)
    p.add_argument("--source-repo",type=Path,
                   help="Read local outputs from another CDF-RAG-2 checkout/worktree")
    p.add_argument("--output-root",type=Path,default=STAGING)
    p.add_argument("--include-per-query",action="store_true")
    p.add_argument("--triage-dir",type=Path)
    p.add_argument("--include-reviewed-examples",action="store_true")
    p.add_argument("--case-studies-dir",type=Path)
    args=p.parse_args()
    stage(args.submission_dir,args.audit_dir,args.output_root,
          args.include_per_query,args.triage_dir,
          args.include_reviewed_examples,args.case_studies_dir,
          args.source_repo)


if __name__=="__main__":
    main()
