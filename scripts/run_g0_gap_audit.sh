#!/usr/bin/env bash
# No Ollama/Neo4j; versioned and nonmutating audit of 64 graph gaps.
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON_CMD="${PYTHON:-python3}"
"$PYTHON_CMD" evaluate_all_baselines.py validate
"$PYTHON_CMD" -m research_v2.g0_gap_audit
