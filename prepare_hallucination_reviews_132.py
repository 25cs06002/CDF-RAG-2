"""Create UNFILLED output-specific clinical hallucination adjudication worksheets.

Does NOT generate gold medical judgments. One row per model/query, for
independent review of the exact final answer and retrieved-evidence basis.
"""
import argparse
import csv
import json
import uuid
from datetime import datetime,timezone
from pathlib import Path
from research_v2.thesis132_full_metrics import (
    DEFAULT_SOURCE,MODELS,GOLD_COLUMNS,_read_report,
)
from research_v2.thesis132_evaluation import _load, _resolve_dir

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir",type=Path,default=DEFAULT_SOURCE)
    p.add_argument("--output-dir",type=Path)
    a=p.parse_args()
    base,manifest,summary,raw,_=_load(a.input_dir)
    path=_resolve_dir(a.output_dir) if a.output_dir else base/(
        "hallucination_review_132_"+datetime.now(timezone.utc).strftime(
            "%Y%m%dT%H%M%S%fZ")+"_"+uuid.uuid4().hex[:6])
    path.mkdir(parents=True,exist_ok=False)
    headers=(*GOLD_COLUMNS,"query","final_answer","detector_flag","matched_concept",
             "retrieved_document_ids_json","direct_causes","multi_hop_paths")
    result=path/"hallucination_gold_review_template_132.csv"
    with result.open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=headers)
        writer.writeheader()
        for model in MODELS:
            for row in raw[model]:
                writer.writerow({
                    "model":model,"query_id":row["query_id"],
                    "is_hallucination_gold":"","review_status":"pending",
                    "reviewer_id":"","reviewer_qualification":"",
                    "independence_declaration":"","verified_source_evidence":"",
                    "review_notes":"",
                    "query":row["query"],"final_answer":row["final_response"],
                    "detector_flag":row["is_hallucination"],
                    "matched_concept":row["matched_concept"],
                    "retrieved_document_ids_json":row["retrieved_document_ids_json"],
                    "direct_causes":row["direct_causes"],
                    "multi_hop_paths":row["multi_hop_paths"],
                })
    print("Created 1056 UNREVIEWED answer-specific worksheet rows:")
    print(result)
    print("No medically reviewed labels created.")
    print("For approved rows fill Yes/No, reviewer identification, independence and")
    print("exact verified source evidence; DO NOT mark unreviewed rows approved.")

if __name__=="__main__":main()
