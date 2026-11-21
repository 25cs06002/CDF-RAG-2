"""Generate screenshot-style metric reports from existing fixed 132 B1–B8 CSVs.

NO MODEL INFERENCE. Optional --semantic-proxies loads the pinned local
all-MiniLM-L6-v2 only for cosine SIMILARITY (not factual accuracy).

Example:
python evaluate_all_metrics_132.py --semantic-proxies
python evaluate_all_metrics_132.py --semantic-proxies --gold-csv PATH_TO_APPROVED_GOLD.csv
"""
import argparse
from pathlib import Path
from research_v2.thesis132_full_metrics import DEFAULT_SOURCE, evaluate

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir",type=Path,default=DEFAULT_SOURCE)
    p.add_argument("--output-dir",type=Path,
                   help="Fresh, nonexistent directory; default timestamped next to source")
    p.add_argument("--semantic-proxies",action="store_true",
                   help="Compute SRS, context relevance, groundedness as cosine proxies")
    p.add_argument("--gold-csv",type=Path,
                   help="Independent answer-specific approved review records, optional")
    p.add_argument("--batch-size",type=int,default=32)
    a=p.parse_args()
    if a.batch_size<=0:p.error("--batch-size must be positive")
    out,manifest=evaluate(a.input_dir,a.output_dir,semantic_proxies=a.semantic_proxies,
                          gold_csv=a.gold_csv,batch_size=a.batch_size)
    print("\n"+"="*72)
    print("FIXED 132-QUERY EVALUATION COMPLETE")
    print("="*72)
    print("Models: B1 through B8 | Question-model records: 1056")
    print("Semantic similarity proxies:",a.semantic_proxies)
    print("Independently approved hallucination labels:",
          manifest["labels_reviewed_by_model"])
    print("Results directory:",out)
    print("Comparison CSV:",out/"comparison_all_models_132_full.csv")
    print("Hallucination summary:",out/"hallucination_classification_summary_132.csv")
    print("\nNo medical accuracy or clinical groundedness has been inferred.")

if __name__=="__main__":main()
