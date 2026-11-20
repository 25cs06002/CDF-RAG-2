"""Export safe, query-only 132-question development input from benchmark v2.

Runs benchmark_v2_validation first, then exports {query_id, query} without
draft answers, review metadata, target labels or gold fields. This prevents
query/target leakage into retrievers. It is a *development* cohort, never
an independent test split.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .benchmark_v2_validation import DATASET, ROOT, validate

OUTPUT=ROOT/"presentation_20261118/benchmark_v2_export"


def export(output_root=OUTPUT):
    audit,rows,_=validate()
    if audit["clinical_gold_questions"] or audit["g1_independent_holdout"]:
        raise ValueError("Unexpected clinical gold/holdout status")
    output_root=Path(output_root).expanduser().resolve()
    if not output_root.is_relative_to(ROOT):
        raise ValueError("Export must remain within CDF-RAG-2 repository")
    out=output_root/("dev132_"+datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%S%fZ")+"_"+uuid.uuid4().hex[:6])
    out.mkdir(parents=True,exist_ok=False)
    query_file=out/"questions_only_development132.jsonl"
    query_file.write_text(
        "".join(json.dumps({
            "query_id":r["query_id"],"query":r["query"]
        },ensure_ascii=False)+"\n" for r in rows),
        encoding="utf-8")
    with (out/"metadata_development132.csv").open(
            "w",newline="",encoding="utf-8") as f:
        keys=["query_id","cohort","query_type","medical_domain",
              "difficulty_legacy","task_objective","expected_answerability",
              "semantic_group","split","annotation_review_status"]
        writer=csv.DictWriter(f,fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k:row[k] for k in keys})
    manifest={
        "status":"SAFE_DEVELOPMENT_QUERY_EXPORT",
        "query_count":132,
        "source_v2_sha256":hashlib.sha256(DATASET.read_bytes()).hexdigest(),
        "query_only_sha256":hashlib.sha256(query_file.read_bytes()).hexdigest(),
        "no_benchmark_gold_exposed_to_retriever":True,
        "independent_holdout":False,
        "medical_gold":False,
        "warning":"Benchmark v2 annotations are screened candidate data only.",
    }
    (out/"export_manifest.json").write_text(
        json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"output":str(out),**manifest},indent=2))
    return out,manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-root",type=Path,default=OUTPUT)
    args=p.parse_args()
    export(args.output_root)


if __name__=="__main__":
    main()
