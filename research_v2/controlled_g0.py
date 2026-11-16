"""Controlled B1-B6 G0 structural retrieval benchmark (NO Ollama).

All six retrieval methods receive the same fixed G0 35-node/37-edge snapshot.
B1-B4 search graph-edge verbalizations, B5-B6 traverse the corresponding
relationships. Scores are agreement with G0 direct edges on the mapped
subset, NOT clinical accuracy or independent reference-gold performance.

Examples:
  python -m research_v2.controlled_g0 --encoder lsa --top-k 1,3,5,10
  python -m research_v2.controlled_g0 --encoder sentence --top-k 5
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .evaluation import GoldAnnotation, evaluate, load_jsonl
from .experiment_runner import PROJECT_ROOT, file_sha256, run_experiment
from .pipelines.baselines import BaselinePipeline, G0Corpus, LSAEncoder
from .presentation_matrix import INPUT, WORKSPACE, validate_inputs
from .presentation_g0_reference import G0, REFERENCE_DIR
from .schema import PipelineRecord

MODEL_NAMES = {
    "B1": "naive_rag",
    "B2": "bm25_rag",
    "B3": "dense_rag",
    "B4": "hybrid_rag",
    "B5": "graph_rag",
    "B6": "causal_kg_rag",
}


def parse_models(value: str) -> list[str]:
    models = value.split(",")
    if not models or len(models) != len(set(models)) or any(m not in MODEL_NAMES for m in models):
        raise ValueError("Use unique IDs from B1,B2,B3,B4,B5,B6")
    return models


def parse_top_k(value: str) -> list[int]:
    try:
        values = [int(part) for part in value.split(",")]
    except ValueError as exc:
        raise ValueError("top-k must be comma-separated positive integers") from exc
    if not values or len(values) != len(set(values)) or any(k < 1 for k in values):
        raise ValueError("top-k values must be unique positive integers")
    return values


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("No rows to export")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def describe(records: list[PipelineRecord], gold: list[GoldAnnotation]) -> tuple[dict, list[dict]]:
    """Compute exact-match structural edge metrics on mapped pilot queries only."""
    if not records:
        raise ValueError("No records")
    by_id = {r.query_id: r for r in records}
    if len(by_id) != len(records):
        raise ValueError("Duplicate query IDs")
    selected_gold = [annotation for annotation in gold if annotation.query_id in by_id]
    if not selected_gold:
        edge_metrics = {
            "evaluated_queries": 0, "macro_f1": None, "micro_f1": None,
            "micro_precision": None, "micro_recall": None,
        }
        per_query = []
    else:
        selected = [by_id[g.query_id] for g in selected_gold]
        result, per_query = evaluate(selected, selected_gold)
        edge_metrics = result["metrics"]["edges"]
    latencies = [r.timings.total_ms for r in records if r.timings.total_ms is not None]
    retrieval = [r.timings.retrieval_ms for r in records if r.timings.retrieval_ms is not None]
    failures = sum(r.status == "error" for r in records)
    summary = {
        "evaluated": len(records),
        "ok": sum(r.status == "ok" for r in records),
        "abstained": sum(r.status == "abstained" for r in records),
        "errors": failures,
        "evidence_returned": sum(bool(r.retrieved_documents or r.causal_edges or r.causal_paths) for r in records),
        "mean_query_latency_ms": statistics.mean(latencies) if latencies else None,
        "mean_retrieval_ms": statistics.mean(retrieval) if retrieval else None,
        "mapped_questions": edge_metrics["evaluated_queries"],
        "g0_edge_macro_f1": edge_metrics["macro_f1"],
        "g0_edge_micro_f1": edge_metrics["micro_f1"],
        "g0_edge_micro_precision": edge_metrics["micro_precision"],
        "g0_edge_micro_recall": edge_metrics["micro_recall"],
    }
    return summary, per_query


def run_controlled(*, encoder: str = "lsa", top_k: list[int] | None = None,
                   models: list[str] | None = None, limit: int = 100,
                   output_root: Path | None = None):
    if encoder not in {"lsa", "sentence"}:
        raise ValueError("encoder must be 'lsa' or 'sentence'")
    if not 1 <= limit <= 100:
        raise ValueError("limit must be from 1 to 100")
    k_values = top_k if top_k is not None else [1, 3, 5, 10]
    model_ids = models if models is not None else list(MODEL_NAMES)
    if not k_values or len(set(k_values)) != len(k_values) or any(k < 1 for k in k_values):
        raise ValueError("top-k must contain unique positive integers")
    if not model_ids or len(set(model_ids)) != len(model_ids) or any(m not in MODEL_NAMES for m in model_ids):
        raise ValueError("models must contain unique B1-B6 IDs")
    questions, metadata = validate_inputs()
    questions = questions[:limit]
    gold_file = REFERENCE_DIR / "g0_gold.jsonl"
    if not gold_file.is_file():
        raise FileNotFoundError("Prepare frozen G0 structural reference first")
    gold = load_jsonl(gold_file, GoldAnnotation)
    corpus = G0Corpus()
    if len(corpus.nodes) != 35 or len(corpus.edges) != 37:
        raise ValueError("G0 snapshot does not match the frozen benchmark")

    base = (output_root or WORKSPACE / "controlled_g0").expanduser().resolve()
    if not base.is_relative_to(PROJECT_ROOT):
        raise ValueError("Output must remain within repository")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = base / f"run_{stamp}_{uuid.uuid4().hex[:6]}"
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "purpose": "Controlled, retrieval-only B1-B6 structural G0 experiment",
        "reference_basis": "graph_snapshot",
        "medical_gold_verified": False,
        "clinical_accuracy_measured": False,
        "document_representation": "G0_EDGE verbalizations of the same G0 relationship IDs",
        "graph_nodes": len(corpus.nodes),
        "graph_edges": len(corpus.edges),
        "nodes_sha256": file_sha256(G0 / "nodes.csv"),
        "edges_sha256": file_sha256(G0 / "edges.csv"),
        "questions_sha256": file_sha256(INPUT),
        "g0_gold_sha256": file_sha256(gold_file),
        "encoder": encoder,
        "top_k": k_values,
        "models": model_ids,
        "limit": limit,
        "warning": (
            "This reuses G0 graph-derived texts and G0 structural reference edges; "
            "results measure within-snapshot retrieval mechanics, NOT clinical accuracy. "
            "B7/B8 and provisional external document summaries are not part of this controlled condition."
        ),
    }
    (output / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    summaries, details = [], []
    for k in k_values:
        for model_id in model_ids:
            dense_encoder = LSAEncoder() if encoder == "lsa" and model_id in {"B3", "B4"} else None
            pipeline = BaselinePipeline(
                MODEL_NAMES[model_id], corpus=corpus, top_k=k,
                generate=False, dense_encoder=dense_encoder,
            )
            run_id = f"G0_{model_id}_K{k}_{uuid.uuid4().hex[:8]}"
            result, directory = run_experiment(
                pipeline=pipeline, questions=questions,
                run_id=run_id, source_file=INPUT,
                output_root=output / "runs", graph_version="G0",
            )
            records = load_jsonl(directory / "records.jsonl", PipelineRecord)
            report, query_rows = describe(records, gold)
            row = {"model_id": model_id, "top_k": k, "encoder": encoder, "run_id": run_id,
                   **report}
            summaries.append(row)
            for item in query_rows:
                metric = item["metrics"]["edges"]
                details.append({
                    "model_id": model_id, "top_k": k,
                    "query_id": item["query_id"], "status": item["status"],
                    "edge_tp": metric["tp"], "edge_fp": metric["fp"], "edge_fn": metric["fn"],
                    "edge_f1": metric["f1"],
                })
            if result["errors"]:
                raise RuntimeError(f"Error in {model_id} at k={k}; see {directory}")
    _write_csv(output / "controlled_summary.csv", summaries)
    if details:
        _write_csv(output / "mapped_query_edge_scores.csv", details)
    print(json.dumps({"output": str(output), "configurations": len(summaries),
                      "questions_per_configuration": limit,
                      "mapped_queries_in_pilot": len(gold),
                      "medical_accuracy": False}, indent=2))
    return output, summaries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder", choices=("lsa", "sentence"), default="lsa")
    parser.add_argument("--top-k", default="1,3,5,10")
    parser.add_argument("--models", default="B1,B2,B3,B4,B5,B6")
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()
    if Path.cwd().resolve() != PROJECT_ROOT:
        parser.error("Run from CDF-RAG-2 repository root")
    run_controlled(
        encoder=args.encoder, top_k=parse_top_k(args.top_k),
        models=parse_models(args.models), limit=args.limit
    )


if __name__ == "__main__":
    main()
