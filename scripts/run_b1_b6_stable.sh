#!/usr/bin/env bash
# Low-memory Ubuntu run for B1-B6 ONLY. Never reruns B7/B8 or overwrites evidence.
set -euo pipefail
cd "$(dirname "$0")/.."

# 8 GiB CPU systems: periodic unloading limits Ollama's accumulating prompt
# cache. These are recorded per run; override them deliberately if necessary.
export OLLAMA_MODEL="${OLLAMA_MODEL:-llama3.2:latest}"
export OLLAMA_NUM_CTX="${OLLAMA_NUM_CTX:-2048}"
export OLLAMA_NUM_PREDICT="${OLLAMA_NUM_PREDICT:-256}"
export OLLAMA_UNLOAD_EVERY_N="${OLLAMA_UNLOAD_EVERY_N:-5}"
export OLLAMA_GENERATION_ATTEMPTS="${OLLAMA_GENERATION_ATTEMPTS:-3}"
export OLLAMA_KEEP_ALIVE_REQUEST="${OLLAMA_KEEP_ALIVE_REQUEST:-5m}"

PYTHON_CMD="${PYTHON:-python3}"
ENCODER="${1:-sentence}"
if [[ "$ENCODER" != "sentence" && "$ENCODER" != "lsa" ]]; then
  echo "Encoder must be sentence or lsa" >&2
  exit 2
fi

"$PYTHON_CMD" evaluate_all_baselines.py prepare
"$PYTHON_CMD" evaluate_all_baselines.py validate
"$PYTHON_CMD" -m research_v2.ollama_generation check
"$PYTHON_CMD" -m research_v2.ollama_generation settings

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p presentation_20261118/logs
"$PYTHON_CMD" evaluate_all_baselines.py run \
  --models B1,B2,B3,B4,B5,B6 \
  --encoder "$ENCODER" --skip-complete --fail-fast-errors \
  2>&1 | tee "presentation_20261118/logs/${STAMP}_stable_B1_B6.log"

# No success is claimed merely because a run contains 100 records.
"$PYTHON_CMD" - <<'PY'
import json
from pathlib import Path
journal = Path("presentation_20261118/matrix_runs.jsonl")
latest = {}
for line in journal.read_text(encoding="utf-8").splitlines():
    if line.strip():
        row = json.loads(line)
        if row.get("execution_mode") == "end_to_end":
            latest[row["model_id"]] = row
for model in ("B1","B2","B3","B4","B5","B6"):
    row = latest.get(model)
    if row is None:
        raise SystemExit(f"No end-to-end run for {model}")
    summary = row["summary"]
    if (summary["total_queries"] != 100 or summary["errors"] != 0 or
            summary.get("aborted") or
            summary["successful"] + summary["abstained"] != 100):
        raise SystemExit(f"{model} incomplete or errored: {summary}")
    print(model, "success", summary["successful"],
          "abstained", summary["abstained"],
          "run", row["run_id"])
PY

"$PYTHON_CMD" evaluate_all_baselines.py report
"$PYTHON_CMD" evaluate_all_baselines.py score-g0
"$PYTHON_CMD" evaluate_all_baselines.py export
"$PYTHON_CMD" evaluate_all_baselines.py diagnose
echo "B1-B6 completed without runtime errors. B7/B8 historical runs preserved."
