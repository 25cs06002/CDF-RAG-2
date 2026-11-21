"""Print hallucination detector evaluation like original 22-query script.

True accuracy, precision, recall, F1, and confusion matrices are computed
ONLY for query-specific independently approved reference labels. Without
labels, print detector observations and N/A (never invent gold).

python compare_hallucination_132.py --report-dir FULL_METRICS_FOLDER --model B7
python compare_hallucination_132.py --report-dir FULL_METRICS_FOLDER --model all
"""
import argparse
from pathlib import Path
from research_v2.thesis132_full_metrics import MODELS, display_hallucination

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--report-dir",required=True,type=Path)
    p.add_argument("--model",default="all",choices=("all",*MODELS))
    a=p.parse_args()
    for model in MODELS if a.model=="all" else [a.model]:
        display_hallucination(a.report_dir,model)

if __name__=="__main__":main()
