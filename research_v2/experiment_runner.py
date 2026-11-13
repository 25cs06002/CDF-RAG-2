"""
Shared experiment runner for CDF-RAG research-v2.

Run from the CDF-RAG-2 project root.

One pipeline and one question are executed at a time.
"""

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
import uuid

from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .base_pipeline import BaseRAGPipeline, PipelineRequest
from .schema import PipelineRecord, StageTimings


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT.parent
    / "CDF-RAG-2-experiment-artifacts"
    / "step11"
)


class BenchmarkQuestion(BaseModel):
    """
    Only inference inputs are passed to the pipeline.

    Gold labels, if present in the dataset, are ignored
    here and must be evaluated separately.
    """

    model_config = ConfigDict(
        extra="ignore",
        frozen=True,
    )

    query_id: str = Field(min_length=1)
    query: str = Field(min_length=1)

    @field_validator("query_id", "query")
    @classmethod
    def reject_blank(cls, value):
        if not value.strip():
            raise ValueError("Blank value is not allowed")
        return value


def load_questions(path: Path):
    if not path.is_file():
        raise FileNotFoundError(path)

    questions = []
    seen = set()

    with path.open(encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            if not line.strip():
                continue

            data = json.loads(line)
            question = BenchmarkQuestion.model_validate(data)

            if question.query_id in seen:
                raise ValueError(
                    f"Duplicate query_id at line "
                    f"{line_number}: {question.query_id}"
                )

            seen.add(question.query_id)
            questions.append(question)

    if not questions:
        raise ValueError("Dataset contains no questions")

    return questions


def file_sha256(path: Path):
    digest = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024), b""
        ):
            digest.update(chunk)

    return digest.hexdigest()


def git_commit():
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def run_experiment(
    pipeline: BaseRAGPipeline,
    questions: list[BenchmarkQuestion],
    run_id: str,
    source_file: Path,
    output_root: Path,
    graph_version: str = "G0",
    random_seed: int | None = None,
):
    """
    Run one selected pipeline sequentially.

    Every question produces one PipelineRecord,
    including when that question raises an exception.
    """

    if not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}",
        run_id,
    ) or run_id in {".", ".."}:
        raise ValueError("Invalid run_id")

    if not questions:
        raise ValueError("No questions supplied")

    output_dir = output_root / run_id

    # Never overwrite a previous experiment.
    output_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    manifest = {
        "run_id": run_id,
        "pipeline_id": pipeline.pipeline_id,
        "graph_version_label": graph_version,
        "random_seed_label": random_seed,
        "git_commit": git_commit(),
        "input_file": str(source_file.resolve()),
        "input_sha256": file_sha256(source_file),
        "question_count": len(questions),
        "python_version": sys.version,
        "started_utc": utc_now(),
    }

    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    success = 0
    abstained = 0
    errors = 0

    records_path = output_dir / "records.jsonl"

    started = time.perf_counter()

    with records_path.open(
        "w",
        encoding="utf-8",
    ) as output:

        for index, question in enumerate(
            questions, 1
        ):
            print(
                f"\n[{index}/{len(questions)}] "
                f"{pipeline.pipeline_id} | "
                f"{question.query_id}",
                flush=True,
            )

            request = PipelineRequest(
                run_id=run_id,
                query_id=question.query_id,
                query=question.query,
                graph_version=graph_version,
                random_seed=random_seed,
            )

            query_start = time.perf_counter()

            try:
                record = pipeline.run(request)

            except Exception as exc:
                # Store the exception class without
                # exposing credentials in a raw trace.
                record = PipelineRecord(
                    run_id=run_id,
                    query_id=question.query_id,
                    pipeline_id=pipeline.pipeline_id,
                    graph_version=graph_version,
                    original_query=question.query,
                    random_seed=random_seed,
                    status="error",
                    error_message=type(exc).__name__,
                    timings=StageTimings(
                        total_ms=(
                            time.perf_counter()
                            - query_start
                        ) * 1000
                    ),
                )

            output.write(
                record.model_dump_json() + "\n"
            )
            output.flush()

            if record.status == "ok":
                success += 1
            elif record.status == "abstained":
                abstained += 1
            else:
                errors += 1

            print(
                f"Status: {record.status} | "
                f"Latency: "
                f"{(record.timings.total_ms or 0)/1000:.2f}s",
                flush=True,
            )

    summary = {
        "run_id": run_id,
        "pipeline_id": pipeline.pipeline_id,
        "total_queries": len(questions),
        "successful": success,
        "abstained": abstained,
        "errors": errors,
        "wall_time_seconds": round(
            time.perf_counter() - started, 3
        ),
        "finished_utc": utc_now(),
    }

    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    return summary, output_dir


def create_pipeline(name: str):
    # Lazy imports prevent loading both large
    # legacy pipelines unnecessarily.

    if name == "no_rl":
        from .legacy_no_rl import LegacyNoRLPipeline
        return LegacyNoRLPipeline()

    if name == "rl":
        from .legacy_rl import LegacyRLPipeline
        return LegacyRLPipeline()

    raise ValueError(f"Unknown pipeline: {name}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--pipeline",
        choices=["no_rl", "rl"],
        required=True,
    )

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--graph-version",
        default="G0",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--run-id",
        default=None,
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )

    parser.add_argument(
        "--validate-only",
        action="store_true",
    )

    args = parser.parse_args()

    questions = load_questions(args.input)

    if args.limit is not None:
        if args.limit < 1:
            parser.error("--limit must be positive")
        questions = questions[:args.limit]

    if args.validate_only:
        print("DATASET VALIDATION PASSED")
        print("Questions:", len(questions))
        return

    if Path.cwd().resolve() != PROJECT_ROOT:
        parser.error(
            "Run from the CDF-RAG-2 project root"
        )

    run_id = args.run_id or (
        datetime.now(timezone.utc)
        .strftime("%Y%m%dT%H%M%SZ")
        + "_"
        + uuid.uuid4().hex[:8]
    )

    pipeline = create_pipeline(args.pipeline)

    summary, output_dir = run_experiment(
        pipeline=pipeline,
        questions=questions,
        run_id=run_id,
        source_file=args.input,
        output_root=args.output_root,
        graph_version=args.graph_version,
        random_seed=args.seed,
    )

    print("\nEXPERIMENT SUMMARY")
    print(json.dumps(summary, indent=2))
    print("Output directory:", output_dir)


if __name__ == "__main__":
    main()
