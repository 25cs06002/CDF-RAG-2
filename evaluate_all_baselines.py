"""One entry point for the November 18 B1-B8 presentation experiment.

Run from ~/Folder/CDF-RAG-2:

  python3 evaluate_all_baselines.py prepare
  python3 evaluate_all_baselines.py validate
  python3 evaluate_all_baselines.py smoke --encoder lsa
  python3 evaluate_all_baselines.py run --models all
  python3 evaluate_all_baselines.py report

The old evaluate_batch_queries_rl.py / _no_rl.py files are preserved.
These presentation experiments use research_v2 and never overwrite legacy CSVs.
"""
import argparse
from pathlib import Path

from research_v2.experiment_runner import PROJECT_ROOT
from research_v2 import (
    presentation_corpus,
    presentation_g0_reference,
    presentation_matrix,
    presentation_sprint,
)


def prepare():
    root = presentation_sprint.DEFAULT_WORKSPACE
    if (root / "benchmark/manifest.json").is_file():
        presentation_sprint.status()
    else:
        presentation_sprint.prepare()
    if (presentation_corpus.OUTPUT / "manifest.json").is_file():
        print("Provisional corpus already prepared: preserving existing version")
    else:
        presentation_corpus.prepare()
    if (presentation_g0_reference.REFERENCE_DIR / "scope.json").is_file():
        print("G0 structural reference already prepared: preserving version")
    else:
        presentation_g0_reference.prepare()
    presentation_matrix.validate_inputs()
    print("PRESENTATION INPUTS READY. Human-reviewed medical gold: NO.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("prepare")
    commands.add_parser("validate")
    smoke = commands.add_parser("smoke")
    smoke.add_argument("--models", default="B1,B2,B3,B4,B5,B6")
    smoke.add_argument("--encoder", choices=["sentence", "lsa"], default="lsa")
    run = commands.add_parser("run")
    run.add_argument("--models", default="all")
    run.add_argument("--limit", type=int)
    run.add_argument("--encoder", choices=["sentence", "lsa"], default="sentence")
    run.add_argument("--retrieval-only", action="store_true")
    run.add_argument("--top-k", type=int, default=5)
    report = commands.add_parser("report")
    report.add_argument("--allow-partial", action="store_true")
    commands.add_parser("score-g0")
    args = parser.parse_args()

    if Path.cwd().resolve() != PROJECT_ROOT:
        parser.error("Run inside ~/Folder/CDF-RAG-2")
    if args.command == "prepare":
        prepare()
    elif args.command == "validate":
        questions, _ = presentation_matrix.validate_inputs()
        print(f"VALIDATED {len(questions)} frozen pilot queries")
    elif args.command == "smoke":
        presentation_matrix.run_models(
            selected=args.models,
            limit=3,
            retrieval_only=True,
            encoder=args.encoder,
        )
    elif args.command == "run":
        presentation_matrix.run_models(
            selected=args.models,
            limit=args.limit,
            retrieval_only=args.retrieval_only,
            top_k=args.top_k,
            encoder=args.encoder,
        )
    elif args.command == "report":
        presentation_matrix.report(allow_partial=args.allow_partial)
    else:
        presentation_g0_reference.score()


if __name__ == "__main__":
    main()
