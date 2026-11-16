#!/usr/bin/env bash
# No offline sprint CI, no expensive pipelines. Run from repository checkout.
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON_CMD="${PYTHON:-python3}"
"$PYTHON_CMD" -m research_v2.g1_foundation validate-queries
"$PYTHON_CMD" -m research_v2.g1_foundation prepare
"$PYTHON_CMD" -m research_v2.g1_query_interpreter --top-k 5
