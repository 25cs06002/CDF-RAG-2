#!/usr/bin/env bash
# Checks saved B7/B8 runs, graph integrity journals and identical shared runtime.
# Offline: does not start Neo4j, Ollama or PPO, or rerun either pipeline.
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON_CMD="${PYTHON:-python3}"
"$PYTHON_CMD" evaluate_all_baselines.py validate
"$PYTHON_CMD" -m research_v2.legacy_pair_comparison
