#!/usr/bin/env bash
# Offline 64-issue triage and conservative B6 shadow probe, no LLM reruns.
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON_CMD="${PYTHON:-python3}"
"$PYTHON_CMD" evaluate_all_baselines.py validate
"$PYTHON_CMD" -m research_v2.graph_gap_triage "$@"
