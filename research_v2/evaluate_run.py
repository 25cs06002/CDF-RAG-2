"""
Offline evaluation CLI.

Does not execute Ollama, PPO, or Neo4j.
"""

import argparse
import json

from pathlib import Path

from .evaluation import (
    GoldAnnotation,
    evaluate,
    load_jsonl,
)

from .experiment_runner import (
    file_sha256,
    git_commit,
)

from .schema import PipelineRecord


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--records",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--gold",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )

    args = parser.parse_args()

    records = load_jsonl(
        args.records,
        PipelineRecord,
    )

    references = load_jsonl(
        args.gold,
        GoldAnnotation,
    )

    summary, rows = evaluate(
        records,
        references,
    )

    # No output directory is created if
    # loading or evaluation fails.
    args.output_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    (
        args.output_dir / "summary.json"
    ).write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    (
        args.output_dir / "per_query.jsonl"
    ).write_text(
        "".join(
            json.dumps(row) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )

    provenance = {
        "records_sha256": file_sha256(
            args.records
        ),
        "gold_sha256": file_sha256(
            args.gold
        ),
        "evaluator_git_commit": git_commit(),
        "annotation_warning": (
            "Synthetic and graph_snapshot labels "
            "do not establish medical correctness."
        ),
    }

    (
        args.output_dir / "provenance.json"
    ).write_text(
        json.dumps(
            provenance,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(
        summary,
        indent=2,
    ))

    print(
        "Evaluation saved:",
        args.output_dir,
    )


if __name__ == "__main__":
    main()
