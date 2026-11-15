#!/usr/bin/env bash
# Execute the M.Tech presentation experiment from the existing Ubuntu checkout.
# This script deliberately does not install dependencies, reset Git, delete
# files, modify Neo4j or publish unreviewed medical outputs.
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON_CMD="${PYTHON:-python3}"
WORKSPACE="presentation_20261118"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$WORKSPACE/logs"

MODE="${1:-help}"
ENCODER="${2:-sentence}"
if [[ "$ENCODER" != "lsa" && "$ENCODER" != "sentence" ]]; then
  echo "Encoder must be 'lsa' or 'sentence'" >&2
  exit 2
fi

run_logged() {
  local tag="$1"
  shift
  echo "START $tag $(date -u +%FT%TZ)"
  "$@" 2>&1 | tee "$WORKSPACE/logs/${STAMP}_${tag}.log"
  echo "FINISH $tag $(date -u +%FT%TZ)"
}

case "$MODE" in
  prepare)
    run_logged prepare "$PYTHON_CMD" evaluate_all_baselines.py prepare
    ;;
  test)
    run_logged tests "$PYTHON_CMD" -m unittest \
      tests.test_presentation_sprint \
      tests.test_presentation_baselines \
      tests.test_presentation_corpus \
      tests.test_presentation_matrix \
      tests.test_model_registry -v
    ;;
  smoke)
    run_logged prepare "$PYTHON_CMD" evaluate_all_baselines.py prepare
    run_logged smoke "$PYTHON_CMD" evaluate_all_baselines.py smoke --encoder "$ENCODER"
    ;;
  full)
    run_logged prepare "$PYTHON_CMD" evaluate_all_baselines.py prepare
    run_logged verify "$PYTHON_CMD" evaluate_all_baselines.py validate
    # B1-B6 use the identical local generator; B7-B8 use original legacy graphs.
    # The latter are validated against the G0 runtime lock and PPO checkpoint.
    run_logged end_to_end_100 "$PYTHON_CMD" evaluate_all_baselines.py run --models all --encoder "$ENCODER"
    run_logged report "$PYTHON_CMD" evaluate_all_baselines.py report
    run_logged score_g0 "$PYTHON_CMD" evaluate_all_baselines.py score-g0
    ;;
  report)
    run_logged report "$PYTHON_CMD" evaluate_all_baselines.py report --allow-partial
    ;;
  *)
    cat <<'HELP'
Run from the existing repository on Ubuntu:
  bash scripts/run_presentation.sh prepare
  bash scripts/run_presentation.sh test
  bash scripts/run_presentation.sh smoke lsa
  bash scripts/run_presentation.sh full sentence
  bash scripts/run_presentation.sh report

FULL MODE requires running Ollama, Neo4j, the stored PPO checkpoint, and the
cached sentence-transformer; tests and smoke mode do not require them.
All files remain inside ~/Folder/CDF-RAG-2/presentation_20261118/.
No auto-push: inspect outputs for confidential patient details before committing.
HELP
    ;;
esac
