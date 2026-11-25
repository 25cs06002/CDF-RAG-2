"""Print B1–B8 full metric comparison, plus B8-B7 column.

python compare_all_models_132.py --report-dir FULL_METRICS_FOLDER
"""
import argparse
from pathlib import Path
from research_v2.thesis132_full_metrics import display_comparison

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--report-dir",required=True,type=Path)
    a=p.parse_args()
    display_comparison(a.report_dir)

if __name__=="__main__":main()
