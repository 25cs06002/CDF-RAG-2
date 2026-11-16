#!/usr/bin/env bash
# Deliberate label-leakage diagnostic, not a production retriever.
set -euo pipefail
cd "$(dirname "$0")/.."
"${PYTHON:-python3}" -m research_v2.g0_target_ceiling "$@"
