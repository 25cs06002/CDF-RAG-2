"""Generate B1–B8 per-query metrics, legacy-style per-model summaries, and comparisons.

No LLM, Neo4j or model inference. Requires an existing verified full
132-query export, e.g.:

python evaluate_metrics_132.py \
  --input-dir presentation_20261118/thesis132/csv_reports/batch132_csv_20261121T061653054773Z_bfeb15

Creates a new evaluation_132_* folder, never overwrites historical
evaluation_summary_*.csv or evaluation_metrics_*.csv.
"""
from research_v2.thesis132_evaluation import main

if __name__ == "__main__":
    main()
