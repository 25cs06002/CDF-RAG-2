"""
Offline, reference-based evaluation of standardized RAG results.

Synthetic and G0 graph-snapshot references do NOT establish
medical correctness.
"""

import statistics

from collections import Counter
from pathlib import Path
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)

from .schema import (
    CausalEdge,
    CausalPath,
    PipelineRecord,
)


class GoldAnnotation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query_id: str = Field(min_length=1)

    reference_basis: Literal[
        "synthetic",
        "graph_snapshot",
        "expert_reviewed",
    ]

    document_ids: list[str] | None = None
    causal_edges: list[CausalEdge] | None = None
    causal_paths: list[CausalPath] | None = None

    @model_validator(mode="after")
    def require_annotation(self):
        fields = (
            self.document_ids,
            self.causal_edges,
            self.causal_paths,
        )

        if all(value is None for value in fields):
            raise ValueError(
                "At least one reference field "
                "must be annotated"
            )

        if not self.query_id.strip():
            raise ValueError(
                "query_id cannot be blank"
            )

        return self


def load_jsonl(path: Path, model):
    result = []
    seen = set()

    with Path(path).open(
        encoding="utf-8"
    ) as stream:

        for number, line in enumerate(
            stream, 1
        ):
            if not line.strip():
                continue

            item = model.model_validate_json(line)

            if item.query_id in seen:
                raise ValueError(
                    f"Duplicate query_id at line "
                    f"{number}: {item.query_id}"
                )

            seen.add(item.query_id)
            result.append(item)

    if not result:
        raise ValueError(
            f"Empty JSONL input: {path}"
        )

    return result


def norm(value: str) -> str:
    return " ".join(
        value.casefold().split()
    )


def keys(kind, values):
    """
    Convert evidence into normalized, unique identities.
    """

    if kind == "documents":
        return {
            norm(
                item if isinstance(item, str)
                else item.document_id
            )
            for item in values
        }

    if kind == "edges":
        return {
            (
                norm(item.cause),
                norm(item.effect),
                norm(item.relation),
            )
            for item in values
        }

    if kind == "paths":
        return {
            tuple(
                norm(node)
                for node in item.nodes
            )
            for item in values
        }

    raise ValueError(
        f"Unknown metric: {kind}"
    )


def scores(predicted, gold, *, success: bool):
    """
    Exact-set Precision, Recall, F1.

    Non-successful executions receive zero per-query
    scores for every annotated metric.
    """

    if not success:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "tp": 0,
            "fp": 0,
            "fn": len(gold),
            "predicted_count": 0,
            "gold_count": len(gold),
        }

    tp = len(predicted & gold)
    fp = len(predicted - gold)
    fn = len(gold - predicted)

    if not predicted and not gold:
        precision = recall = f1 = 1.0

    else:
        precision = (
            tp / len(predicted)
            if predicted else 0.0
        )

        recall = (
            tp / len(gold)
            if gold else 0.0
        )

        f1 = (
            2 * precision * recall
            / (precision + recall)
            if precision + recall
            else 0.0
        )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "predicted_count": len(predicted),
        "gold_count": len(gold),
    }


def percentile(values, fraction):
    values = sorted(values)

    position = (
        (len(values) - 1) * fraction
    )

    lower = int(position)

    upper = min(
        lower + 1,
        len(values) - 1,
    )

    weight = position - lower

    return (
        values[lower] * (1 - weight)
        + values[upper] * weight
    )


def evaluate(
    records: list[PipelineRecord],
    gold: list[GoldAnnotation],
):
    if not records:
        raise ValueError(
            "No prediction records"
        )

    if len({
        r.query_id for r in records
    }) != len(records):
        raise ValueError(
            "Duplicate prediction query_id"
        )

    if len({
        g.query_id for g in gold
    }) != len(gold):
        raise ValueError(
            "Duplicate gold query_id"
        )

    if (
        len({r.run_id for r in records}) != 1
        or len({
            r.pipeline_id for r in records
        }) != 1
    ):
        raise ValueError(
            "Mixed run IDs or pipeline IDs"
        )

    labels = {
        g.query_id: g
        for g in gold
    }

    missing = [
        record.query_id
        for record in records
        if record.query_id not in labels
    ]

    if missing:
        raise ValueError(
            f"Missing reference labels: {missing}"
        )

    kinds = {
        "documents": (
            "retrieved_documents",
            "document_ids",
        ),
        "edges": (
            "causal_edges",
            "causal_edges",
        ),
        "paths": (
            "causal_paths",
            "causal_paths",
        ),
    }

    per_query = []
    status_counts = Counter()
    basis_counts = Counter()
    latencies = []

    for record in records:
        annotation = labels[record.query_id]

        status_counts[record.status] += 1
        basis_counts[
            annotation.reference_basis
        ] += 1

        success = record.status == "ok"

        if (
            success
            and record.timings.total_ms is not None
        ):
            latencies.append(
                record.timings.total_ms / 1000
            )

        row = {
            "query_id": record.query_id,
            "status": record.status,
            "reference_basis": (
                annotation.reference_basis
            ),
            "metrics": {},
        }

        for kind, (
            record_field,
            gold_field,
        ) in kinds.items():

            target = getattr(
                annotation, gold_field
            )

            # None means unannotated.
            # [] means annotated and empty.
            if target is None:
                continue

            predicted = (
                keys(
                    kind,
                    getattr(
                        record,
                        record_field,
                    ),
                )
                if success
                else set()
            )

            row["metrics"][kind] = scores(
                predicted,
                keys(kind, target),
                success=success,
            )

        per_query.append(row)

    aggregate = {}

    for kind in kinds:
        items = [
            row["metrics"][kind]
            for row in per_query
            if kind in row["metrics"]
        ]

        if not items:
            aggregate[kind] = {
                "evaluated_queries": 0
            }
            continue

        tp, fp, fn = (
            sum(
                item[key]
                for item in items
            )
            for key in (
                "tp",
                "fp",
                "fn",
            )
        )

        if tp + fp + fn:
            precision = (
                tp / (tp + fp)
                if tp + fp else 0.0
            )

            recall = (
                tp / (tp + fn)
                if tp + fn else 0.0
            )

            f1 = (
                2 * precision * recall
                / (precision + recall)
                if precision + recall
                else 0.0
            )

        else:
            # No positive or predicted items:
            # micro metrics are undefined.
            precision = recall = f1 = None

        aggregate[kind] = {
            "evaluated_queries": len(items),
            "macro_precision": statistics.mean(
                x["precision"] for x in items
            ),
            "macro_recall": statistics.mean(
                x["recall"] for x in items
            ),
            "macro_f1": statistics.mean(
                x["f1"] for x in items
            ),
            "micro_precision": precision,
            "micro_recall": recall,
            "micro_f1": f1,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }

    summary = {
        "run_id": records[0].run_id,
        "pipeline_id": records[0].pipeline_id,
        "total_records": len(records),

        "status_counts": dict(
            status_counts
        ),

        "reference_basis_counts": dict(
            basis_counts
        ),

        "metrics": aggregate,

        "successful_latency_seconds": (
            {
                "count": len(latencies),
                "mean": statistics.mean(
                    latencies
                ),
                "p50": percentile(
                    latencies, 0.50
                ),
                "p95": percentile(
                    latencies, 0.95
                ),
            }
            if latencies
            else {"count": 0}
        ),
    }

    return summary, per_query
