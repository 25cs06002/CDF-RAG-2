"""Print screenshot-style per-pipeline quality/retrieval/performance/refinement report.

python show_evaluation_132.py --report-dir FULL_METRICS_FOLDER --model B7
python show_evaluation_132.py --report-dir FULL_METRICS_FOLDER --model all
"""
import argparse
from pathlib import Path
from research_v2.thesis132_full_metrics import MODELS, display_model

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--report-dir",required=True,type=Path)
    p.add_argument("--model",default="all",choices=("all",*MODELS))
    a=p.parse_args()
    for m in MODELS if a.model=="all" else [a.model]:
        display_model(a.report_dir,m)

if __name__=="__main__":main()
