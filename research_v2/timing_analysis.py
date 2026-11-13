"""
Offline timing analysis for CDF-RAG experiments.

Separates pipeline-reported latency from the measured
graph-fingerprint read time. Does not run Neo4j or Ollama.
"""

import argparse
import json
import math
import statistics

from collections import Counter, defaultdict
from pathlib import Path

from .evaluation import percentile
from .schema import PipelineRecord


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [
            json.loads(line)
            for line in stream
            if line.strip()
        ]


def describe(values):
    if not values:
        return {"count": 0}

    return {
        "count": len(values),
        "total_ms": sum(values),
        "mean_ms": statistics.mean(values),
        "p50_ms": percentile(values, 0.50),
        "p95_ms": percentile(values, 0.95),
        "max_ms": max(values),
    }


def analyze_run(run_dir: Path):
    root = Path(run_dir)

    run_summary = json.loads(
        (root / "summary.json").read_text(
            encoding="utf-8"
        )
    )

    records = [
        PipelineRecord.model_validate(row)
        for row in read_jsonl(
            root / "records.jsonl"
        )
    ]

    if len(records) != run_summary["total_queries"]:
        raise ValueError(
            "Record count does not match run summary"
        )

    provenance_file = (
        root / "config_provenance.json"
    )

    provenance = (
        json.loads(
            provenance_file.read_text(
                encoding="utf-8"
            )
        )
        if provenance_file.is_file()
        else {}
    )

    checks_file = root / "graph_checks.jsonl"

    monitoring_expected = provenance.get(
        "graph_checked_per_query",
        False,
    )

    if monitoring_expected and not checks_file.is_file():
        raise ValueError(
            "Graph monitoring was enabled, "
            "but its journal is missing"
        )

    monitored = checks_file.is_file()
    events = (
        read_jsonl(checks_file)
        if monitored else []
    )

    by_index = defaultdict(dict)
    status_counts = Counter()
    phases = {"before", "after"}

    for event in events:
        index = event.get("query_index")
        phase = event.get("phase")
        elapsed = event.get("check_elapsed_ms")

        if (
            not isinstance(index, int)
            or isinstance(index, bool)
            or index < 1
            or index > len(records)
        ):
            raise ValueError(
                "Invalid graph-check query index"
            )

        if phase not in phases:
            raise ValueError(
                "Invalid graph-check phase"
            )

        if phase in by_index[index]:
            raise ValueError(
                "Duplicate graph-check phase"
            )

        if (
            not isinstance(elapsed, (int, float))
            or not math.isfinite(elapsed)
            or elapsed < 0
        ):
            raise ValueError(
                "Invalid graph-check elapsed time"
            )

        if event.get("status") not in {
            "pass", "drift", "error"
        }:
            raise ValueError(
                "Invalid graph-check status"
            )

        by_index[index][phase] = event
        status_counts[event["status"]] += 1

    per_query = []
    check_times = []
    successful_pipeline_times = []

    for index, record in enumerate(records, 1):
        checks = by_index.get(index, {})

        if monitored:
            before = checks.get("before")
            after = checks.get("after")

            if before is None:
                raise ValueError(
                    f"Missing pre-query graph check: {index}"
                )

            needs_after = (
                record.error_message
                != "GraphIntegrityFailedBeforeQuery"
            )

            if needs_after and after is None:
                raise ValueError(
                    f"Missing post-query graph check: {index}"
                )

            if not needs_after and after is not None:
                raise ValueError(
                    "Unexpected post-query check "
                    "after a pre-query integrity failure"
                )

            times = [
                event["check_elapsed_ms"]
                for event in checks.values()
            ]

            check_times.extend(times)
            graph_ms = sum(times)

        else:
            graph_ms = None

        pipeline_ms = record.timings.total_ms

        if record.status == "ok" and pipeline_ms is not None:
            successful_pipeline_times.append(
                pipeline_ms
            )

        per_query.append({
            "query_index": index,
            "query_id": record.query_id,
            "status": record.status,
            "pipeline_reported_ms": pipeline_ms,
            "graph_check_read_ms_lower_bound": graph_ms,
            "graph_checks": {
                phase: event["status"]
                for phase, event in checks.items()
            },
        })

    wall_seconds = run_summary.get(
        "wall_time_seconds"
    )

    graph_total = (
        sum(check_times)
        if monitored else None
    )

    summary = {
        "run_id": run_summary["run_id"],
        "pipeline_id": run_summary["pipeline_id"],
        "processed_queries": len(records),
        "graph_monitoring_present": monitored,
        "graph_check_status_counts": (
            dict(status_counts)
        ),
        "graph_check_read_ms": (
            describe(check_times)
            if monitored else None
        ),
        "successful_pipeline_ms": describe(
            successful_pipeline_times
        ),
        "run_wall_time_seconds": wall_seconds,
        "graph_read_fraction_of_run_wall_lower_bound": (
            graph_total / (wall_seconds * 1000)
            if (
                monitored
                and wall_seconds is not None
                and wall_seconds > 0
            )
            else None
        ),
        "notes": [
            "Pipeline latency is taken from PipelineRecord.",
            "Graph-check read timing excludes journal writing.",
            "Run wall time includes other runner overhead.",
            "Preflight and pipeline initialization are excluded "
            "from the runner's wall-time measurement.",
            "Do not subtract these values to infer exact "
            "LLM generation or retrieval latency.",
        ],
    }

    return summary, per_query


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )

    args = parser.parse_args()

    summary, rows = analyze_run(
        args.run_dir
    )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    (
        args.output_dir / "timing_summary.json"
    ).write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    (
        args.output_dir / "per_query_timing.jsonl"
    ).write_text(
        "".join(
            json.dumps(row) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(summary, indent=2)
    )
    print(
        "Timing report saved:",
        args.output_dir,
    )


if __name__ == "__main__":
    main()
