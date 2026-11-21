"""Run all 132 frozen queries on B1-B8 and export per-model CSV files.

Uses the exact versioned research_v2.thesis132 experiment engine, not the old
hard-coded 22-query scripts. No live work runs on import.

All eight pipelines (end-to-end, with Neo4j+Ollama+PPO ready):
    python evaluate_batch_queries_132.py --models all --allow-live

B1-B6 retrieval-only (no live LLM needed):
    python evaluate_batch_queries_132.py --models B1,B2,B3,B4,B5,B6 --mode retrieval_only

A failed later model does not erase completed earlier runs: any recorded
132-query model is exported from the session with its true error status.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from research_v2.experiment_runner import PROJECT_ROOT
from research_v2.thesis132 import BASE, _inside, _selected, run
from research_v2.thesis132_csv_export import DEFAULT_OUTPUT, export


def _new_session(before: set[Path]) -> Path | None:
    folder = BASE / "runs"
    candidates = sorted(
        (path for path in folder.glob("batch_*")
         if path.is_dir() and path.resolve() not in before and
         (path / "session.json").is_file()),
        key=lambda p: p.name,
    )
    if len(candidates) > 1:
        raise RuntimeError("Multiple new sessions detected; select explicitly")
    return candidates[0] if candidates else None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", default="all", help="all or B1,B2,...,B8")
    p.add_argument("--mode", choices=["retrieval_only", "end_to_end"],
                   default="end_to_end")
    p.add_argument("--encoder", choices=["sentence", "lsa"], default="sentence")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--allow-live", action="store_true",
                   help="Required for end-to-end Ollama/Neo4j/PPO experiments")
    p.add_argument("--session-dir", type=Path,
                   help="Resume an existing matching session, never overwrite results")
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT,
                   help="Parent of new timestamped CSV export directory")
    p.add_argument("--require-all", action="store_true",
                   help="Require all 8 models to be in the CSV export")
    p.add_argument("--fail-on-errors", action="store_true",
                   help="Reject the exported set if any model records an error")
    a = p.parse_args()
    if Path.cwd().resolve() != PROJECT_ROOT:
        p.error("Run from ~/Folder/CDF-RAG-2-thesis132 repository root")
    selected = _selected(a.models)
    if a.mode == "retrieval_only" and any(m in {"B7", "B8"} for m in selected):
        p.error("B7/B8 require --mode end_to_end --allow-live")
    if a.mode == "end_to_end" and not a.allow_live:
        p.error("End-to-end execution needs --allow-live and working services")
    if a.top_k < 1:
        p.error("--top-k must be >=1")
    if a.require_all and set(selected) != {f"B{i}" for i in range(1, 9)}:
        p.error("--require-all requires --models all")

    runs_dir = BASE / "runs"
    before = {path.resolve() for path in runs_dir.glob("batch_*") if path.is_dir()}
    session = _inside(a.session_dir) if a.session_dir else None
    failure = None
    try:
        session = run(
            ",".join(selected), a.mode, a.encoder, None, a.top_k,
            a.allow_live, session,
        )
    except Exception as exc:
        failure = exc
        if session is None:
            session = _new_session(before)
        print("Pipeline execution stopped: " + type(exc).__name__ + ": " + str(exc),
              file=sys.stderr)
    finally:
        if session is not None and (session / "session.json").is_file():
            meta = json.loads((session / "session.json").read_text(encoding="utf-8"))
            if meta.get("runs"):
                try:
                    out, report = export([session], a.output_root,
                                         require_all=a.require_all if failure is None else False,
                                         fail_on_errors=a.fail_on_errors if failure is None else False)
                    print("CSV files:", out)
                    print("Models exported:", ", ".join(report["models_present"]))
                    if report["models_missing"] or report["models_with_errors"]:
                        print("INCOMPLETE_OR_INVALID:", report["models_missing"],
                              report["models_with_errors"], file=sys.stderr)
                except Exception as export_exc:
                    print("CSV export failed: " + repr(export_exc), file=sys.stderr)
                    if failure is None:
                        raise
    if failure is not None:
        raise SystemExit(
            "Run failed; see original error and any completed per-model CSV exports above"
        )


if __name__ == "__main__":
    main()
