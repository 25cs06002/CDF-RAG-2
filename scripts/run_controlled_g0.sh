#!/usr/bin/env bash
# Fast, controlled retrieval-only B1-B6 G0 experiment (no Ollama/Neo4j).
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON_CMD="${PYTHON:-python3}"
ENCODER="${1:-lsa}"
"$PYTHON_CMD" evaluate_all_baselines.py prepare
"$PYTHON_CMD" evaluate_all_baselines.py validate
"$PYTHON_CMD" -m research_v2.controlled_g0 --encoder "$ENCODER" --top-k "1,3,5,10"
echo "Results: presentation_20261118/controlled_g0/"
