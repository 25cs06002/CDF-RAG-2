#!/usr/bin/env bash
# Creates a source-review backlog ONLY; not a verified graph or medical gold.
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON_CMD="${PYTHON:-python3}"
"$PYTHON_CMD" evaluate_all_baselines.py prepare
"$PYTHON_CMD" evaluate_all_baselines.py validate
# Generate diagnostics from the actual run files if needed.
"$PYTHON_CMD" -m research_v2.g1_review queue
echo "Review the generated review_queue.csv. Enter *real reviewed sources*"
echo "in evidence_candidates.csv before requesting a new G1 candidate."
