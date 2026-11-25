"""Print and validate the all-model and paired B7/B8 comparison.

Existing analysis (no recomputation):
  python compare_evaluation_132.py --analysis-dir path/to/evaluation_132_*

Or generate a fresh analysis from original 132 CSVs then print comparison:
  python compare_evaluation_132.py --input-dir path/to/batch132_csv_*
"""
import argparse
import csv
import json
from pathlib import Path

from research_v2.thesis132_evaluation import ROOT, _hash, _resolve_dir, analyze, MODELS


def review(analysis_dir: Path):
    location = _resolve_dir(analysis_dir)
    manifest = json.loads((location / "analysis_manifest.json").read_text(
        encoding="utf-8"))
    if manifest.get("schema") != "THESIS132_EVALUATION_V1":
        raise ValueError("Not a thesis132 evaluation analysis")
    for name, digest in manifest["generated_sha256"].items():
        if _hash(location / name) != digest:
            raise ValueError("Analysis file modified: " + name)

    with (location / "evaluation_summary_all_models_132.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        records = list(csv.DictReader(handle))
    if [row["model"] for row in records] != list(MODELS):
        raise ValueError("Expected exactly eight pipeline summary records")

    print("\nEXPERIMENT SUMMARY (132 queries per pipeline)")
    print(f"{'Model':<6} {'OK':>5} {'Abstain':>8} {'Evidence %':>12} "
          f"{'Mean ms':>11} {'P95 ms':>11} {'QPS':>10}")
    for r in records:
        print(f"{r['model']:<6} {r['ok']:>5} {r['abstained']:>8} "
              f"{100*float(r['candidate_evidence_rate']):>11.2f}% "
              f"{float(r['mean_latency_ms']):>11.2f} "
              f"{float(r['p95_latency_ms']):>11.2f} "
              f"{float(r['run_throughput_qps']):>10.4f}")

    paired = manifest["paired_B7_B8"]
    print("\nB7 vs B8 paired candidate-evidence yield (NOT medically verified accuracy):")
    for label, count in paired["candidate_evidence_pair_counts"].items():
        print(f"  {label:<10} {count}")
    print("  B8 minus B7 net covered questions:",
          paired["B8_minus_B7_net_evidence_questions"])
    print("  B8 minus B7 evidence percentage points:",
          f"{paired['B8_minus_B7_evidence_percentage_points']:.2f}")
    print("  Mean paired latency increase (ms):",
          f"{paired['B8_minus_B7_mean_paired_latency_ms']:.2f}")
    print("\nCAVEATS: Evidence yield is not medical accuracy; detector flags are not"
          " validated hallucinations. B1–B4 and B5–B8 use different evidence bases.")
    print("\nAnalysis files:", location)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    opts = parser.add_mutually_exclusive_group(required=True)
    opts.add_argument("--analysis-dir", type=Path)
    opts.add_argument("--input-dir", type=Path)
    args = parser.parse_args()
    folder = args.analysis_dir
    if args.input_dir:
        folder, _ = analyze(args.input_dir)
    review(folder)


if __name__ == "__main__":
    main()
