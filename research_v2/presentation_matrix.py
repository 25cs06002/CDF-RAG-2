"""Execute and compare B1-B8 on the same 100-question pilot.

The pilot is unreviewed. Without independent reference labels this module
reports *operational/behavioral/performance* metrics, NOT medical accuracy.
The six new baselines use graph-derived G0 evidence by default.

Quick tests:
  python3 -m research_v2.presentation_matrix validate
  python3 -m research_v2.presentation_matrix run --models B1,B2,B5,B6 --limit 3 --retrieval-only
  python3 -m research_v2.presentation_matrix report --allow-partial
Full run on Ubuntu (requires Ollama and Neo4j for legacy pipelines):
  python3 -m research_v2.presentation_matrix run --models all
  python3 -m research_v2.presentation_matrix report
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from .experiment_runner import PROJECT_ROOT, file_sha256, load_questions, run_experiment
from .presentation_sprint import PIPELINES
from .schema import PipelineRecord

WORKSPACE = PROJECT_ROOT / "presentation_20261118"
INPUT = WORKSPACE / "benchmark/questions_only.jsonl"
PILOT = PROJECT_ROOT / "data/benchmark/medical_causal_pilot100_draft_v1.jsonl"
MODEL_IDS = tuple(PIPELINES)
LEGACY = frozenset({"B7", "B8"})
EXPECTED_PIPELINE_IDS = {
    **{key: val for key, val in PIPELINES.items() if key not in LEGACY},
    "B7": "cdf_rag_no_rl_legacy",
    "B8": "cdf_rag_rl_legacy",
}


def utc_timestamp():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def inside_repo(path):
    value = Path(path).expanduser().resolve()
    if not value.is_relative_to(PROJECT_ROOT):
        raise ValueError("Output path must remain inside CDF-RAG-2")
    return value


def load_metadata():
    metadata = {}
    with PILOT.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            qid = row["query_id"]
            if qid in metadata:
                raise ValueError(f"Duplicate pilot query ID: {qid}")
            metadata[qid] = row
    if len(metadata) != 100:
        raise ValueError("Expected exactly 100 pilot metadata entries")
    return metadata


def validate_inputs(input_path=INPUT):
    questions = load_questions(input_path)
    metadata = load_metadata()
    if len(questions) != 100:
        raise ValueError(f"Expected exactly 100 queries; found {len(questions)}")
    if set(q.query_id for q in questions) != set(metadata):
        raise ValueError("Question-only and metadata query IDs differ")
    for q in questions:
        if q.query != metadata[q.query_id]["query"]:
            raise ValueError(f"Question text mismatch: {q.query_id}")
    if any(row.get("review_status") != "draft" for row in metadata.values()):
        raise ValueError("Unexpected pilot review status")
    manifest = INPUT.parent / "manifest.json"
    if manifest.is_file() and input_path.resolve() == INPUT.resolve():
        stored = json.loads(manifest.read_text(encoding="utf-8"))
        if file_sha256(input_path) != stored["inference_input_sha256"]:
            raise ValueError("Frozen presentation inputs have changed")
    return questions, metadata


def choose_models(selected):
    if selected == "all":
        return list(MODEL_IDS)
    names = selected.split(",")
    if not names or len(names) != len(set(names)) or any(n not in MODEL_IDS for n in names):
        raise ValueError("Choose unique model IDs from B1,B2,B3,B4,B5,B6,B7,B8")
    return names


def find_completed_run(journal, model_id, query_count, *, dataset_sha256,
                       execution_mode, top_k, encoder, current_git_commit,
                       corpus_sha256=None):
    """Return a verified prior completed run at the *same code revision*.

    This prevents a long Ubuntu run from unintentionally repeating completed
    100-query models, but never reuses results after code/data/model-setting
    changes. Use only when --skip-complete is explicitly requested.
    """
    if not journal.is_file():
        return None
    for raw in reversed(journal.read_text(encoding="utf-8").splitlines()):
        if not raw.strip():
            continue
        row = json.loads(raw)
        if row.get("model_id") != model_id:
            continue
        if row.get("query_count") != query_count:
            continue
        if row.get("dataset_sha256") != dataset_sha256:
            continue
        if row.get("execution_mode") != execution_mode:
            continue
        if model_id not in LEGACY and row.get("top_k") != top_k:
            continue
        if model_id in {"B3", "B4"} and row.get("embedding_backend") != encoder:
            continue
        if model_id in {"B1", "B2", "B3", "B4"} and row.get("corpus_sha256") != corpus_sha256:
            continue
        summary = row.get("summary", {})
        if summary.get("aborted") or summary.get("errors") != 0:
            continue
        if summary.get("total_queries") != query_count:
            continue
        if summary.get("successful", 0) + summary.get("abstained", 0) != query_count:
            continue
        base = (PROJECT_ROOT / row.get("run_dir", "")).resolve()
        if not base.is_relative_to(PROJECT_ROOT):
            continue
        manifest_path, records_path = base / "manifest.json", base / "records.jsonl"
        if not manifest_path.is_file() or not records_path.is_file():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("git_commit") != current_git_commit:
            continue
        if manifest.get("input_sha256") != dataset_sha256:
            continue
        if manifest.get("question_count") != query_count:
            continue
        if len(records_path.read_text(encoding="utf-8").splitlines()) != query_count:
            continue
        return row
    return None


def run_models(selected="all", limit=None, retrieval_only=False, top_k=5, encoder="sentence", skip_complete=False):
    questions, metadata = validate_inputs()
    names = choose_models(selected)
    if retrieval_only and any(name in LEGACY for name in names):
        raise ValueError("Legacy B7/B8 always perform generation; retrieval-only would be incomparable")
    if limit is not None:
        if limit < 1 or limit > 100:
            raise ValueError("--limit must be 1..100")
        questions = questions[:limit]
    if top_k < 1:
        raise ValueError("top-k must be >= 1")
    if encoder not in {"sentence", "lsa"}:
        raise ValueError("encoder must be sentence or lsa")
    runs_dir = inside_repo(WORKSPACE / "runs")
    runs_dir.mkdir(parents=True, exist_ok=True)
    configs_dir = inside_repo(WORKSPACE / "configs")
    configs_dir.mkdir(parents=True, exist_ok=True)
    journal = WORKSPACE / "matrix_runs.jsonl"
    source_hash = file_sha256(INPUT)
    # Older experiments may have a different code revision or evidence
    # corpus. Never silently reuse them as new results.
    from .experiment_runner import git_commit
    current_commit = git_commit()
    current_corpus = WORKSPACE / "corpus/provisional_source_summaries_v1/documents.jsonl"
    current_corpus_hash = file_sha256(current_corpus) if current_corpus.is_file() else None

    for model_id in names:
        if skip_complete:
            existing = find_completed_run(
                journal, model_id, len(questions),
                dataset_sha256=source_hash,
                execution_mode="retrieval_only" if retrieval_only else "end_to_end",
                top_k=top_k,
                encoder=encoder,
                current_git_commit=current_commit,
                corpus_sha256=current_corpus_hash,
            )
            if existing is not None:
                print(json.dumps({
                    "skipped_completed_model": model_id,
                    "verified_previous_run": existing["run_id"],
                    "reason": "Same input, code, corpus, encoder, mode and complete successful record count",
                }, indent=2))
                continue
        name = PIPELINES[model_id]
        run_id = f"{model_id}_{utc_timestamp()}_{uuid.uuid4().hex[:6]}"
        wall_start = time.perf_counter()
        if model_id in LEGACY:
            # The existing runtime lock, Neo4j G0 fingerprint and local
            # Ollama/PPO preflight are enforced by this runner.
            from .configured_experiment import run_configured_experiment
            config = {
                "schema_version": 1,
                "experiment_name": f"presentation_{model_id}_{len(questions)}",
                "pipeline": name,
                "dataset_path": INPUT.relative_to(PROJECT_ROOT).as_posix(),
                "dataset_sha256": file_sha256(INPUT),
                "graph_version": "G0",
                "random_seed": 2026,
                "limit": len(questions),
                "purpose": "Exploratory, non-gold 100-question presentation pilot.",
            }
            config_path = configs_dir / f"{run_id}.json"
            config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
            summary, output_dir = run_configured_experiment(
                config_path=config_path, output_root=runs_dir, run_id=run_id
            )
        else:
            from .pipelines import create_pipeline
            documents = WORKSPACE / "corpus/provisional_source_summaries_v1/documents.jsonl"
            if model_id in {"B1", "B2", "B3", "B4"} and not documents.is_file():
                raise FileNotFoundError(
                    "Build the candidate document corpus first: "
                    "python3 -m research_v2.presentation_corpus prepare"
                )
            pipeline = create_pipeline(
                model_id,
                document_corpus=(documents if model_id in {"B1","B2","B3","B4"} else None),
                top_k=top_k,
                generate=not retrieval_only,
                encoder=encoder,
            )
            summary, output_dir = run_experiment(
                pipeline=pipeline,
                questions=questions,
                run_id=run_id,
                source_file=INPUT,
                output_root=runs_dir,
                graph_version="G0",
                # B1-B6 are deterministic retrieval-only methods with a fixed
                # LSA SVD random_state=2026; do not require global PyTorch RNG
                # initialization when testing on a lightweight CI machine.
                random_seed=None,
            )

        evidence_basis = (
            "AI-assisted candidate-source summaries" if model_id in {"B1","B2","B3","B4"}
            else "G0 graph snapshot"
        )
        row = {
            "evidence_basis": evidence_basis,
            "corpus_sha256": (
                file_sha256(WORKSPACE / "corpus/provisional_source_summaries_v1/documents.jsonl")
                if model_id in {"B1","B2","B3","B4"} else None
            ),
            "model_id": model_id,
            "pipeline": name,
            "run_id": run_id,
            "run_dir": output_dir.relative_to(PROJECT_ROOT).as_posix(),
            "query_count": len(questions),
            "dataset_sha256": file_sha256(INPUT),
            "execution_mode": "retrieval_only" if retrieval_only else "end_to_end",
            "top_k": top_k if model_id not in LEGACY else None,
            "embedding_backend": (encoder if model_id in {"B3","B4"} else None),
            "elapsed_including_initialization_seconds": time.perf_counter() - wall_start,
            "summary": summary,
            "logged_utc": datetime.now(timezone.utc).isoformat(),
        }
        with journal.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(json.dumps({
            "completed_model": model_id,
            "run_dir": str(output_dir),
            "successful": summary["successful"],
            "errors": summary["errors"],
            "abstained": summary["abstained"],
            "aborted": summary["aborted"],
        }, indent=2))
        if summary["aborted"]:
            raise RuntimeError(f"Graph-integrity abort in {model_id}; stop matrix")


def percentile(values, fraction):
    ordered = sorted(values)
    if not ordered:
        return None
    position = (len(ordered) - 1) * fraction
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    weight = position - low
    return ordered[low] * (1 - weight) + ordered[high] * weight


def summarize(records, metadata, elapsed):
    n = len(records)
    statuses = Counter(r.status for r in records)
    latencies = [r.timings.total_ms / 1000.0 for r in records
                 if r.timings.total_ms is not None]
    retrieval_times = [r.timings.retrieval_ms / 1000.0 for r in records
                       if r.timings.retrieval_ms is not None]
    generation_times = [r.timings.generation_ms / 1000.0 for r in records
                        if r.timings.generation_ms is not None]
    categories = defaultdict(list)
    for record in records:
        categories[metadata[record.query_id]["query_type"]].append(record)
    return {
        "queries": n,
        "success": statuses["ok"],
        "abstained": statuses["abstained"],
        "errors": statuses["error"],
        "success_rate": statuses["ok"] / n,
        "retrieval_yield_rate": sum(bool(r.retrieved_documents or r.causal_edges or r.causal_paths) for r in records) / n,
        "answer_present_rate": sum(bool(r.final_answer and r.final_answer.strip()) for r in records) / n,
        "avg_documents": sum(len(r.retrieved_documents) for r in records) / n,
        "avg_causal_edges": sum(len(r.causal_edges) for r in records) / n,
        "avg_causal_paths": sum(len(r.causal_paths) for r in records) / n,
        "mean_latency_s": statistics.mean(latencies) if latencies else None,
        "p50_latency_s": percentile(latencies, 0.50),
        "p95_latency_s": percentile(latencies, 0.95),
        "mean_retrieval_s": statistics.mean(retrieval_times) if retrieval_times else None,
        "mean_generation_s": statistics.mean(generation_times) if generation_times else None,
        "throughput_queries_per_s_including_init": n / elapsed if elapsed > 0 else None,
        "by_category": {
            name: {
                "total": len(group),
                "ok": sum(r.status == "ok" for r in group),
                "abstained": sum(r.status == "abstained" for r in group),
                "errors": sum(r.status == "error" for r in group),
                "evidence_yield": sum(bool(r.causal_edges or r.causal_paths or r.retrieved_documents) for r in group),
            }
            for name, group in sorted(categories.items())
        },
    }


def report(allow_partial=False, workspace=WORKSPACE):
    workspace = inside_repo(workspace)
    questions, metadata = validate_inputs()
    journal = workspace / "matrix_runs.jsonl"
    if not journal.is_file():
        raise FileNotFoundError(f"Run experiments before reporting: {journal}")
    selected = {}
    for line in journal.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        run = json.loads(line)
        model_id = run["model_id"]
        if run["dataset_sha256"] != file_sha256(INPUT):
            continue
        if not allow_partial and (run["query_count"] != 100 or run["summary"].get("aborted")):
            continue
        if run["query_count"] == 100 or model_id not in selected:
            selected[model_id] = run

    missing = [model_id for model_id in MODEL_IDS if model_id not in selected]
    if missing and not allow_partial:
        raise ValueError(f"No completed 100-query run for: {', '.join(missing)}")
    if not selected:
        raise ValueError("No eligible runs to report")

    summaries, per_query = [], []
    modes = set()
    for model_id in MODEL_IDS:
        if model_id not in selected:
            continue
        run = selected[model_id]
        modes.add(run["execution_mode"])
        root = PROJECT_ROOT / run["run_dir"]
        record_path = root / "records.jsonl"
        records = [
            PipelineRecord.model_validate_json(line)
            for line in record_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if len(records) != run["query_count"]:
            raise ValueError(f"Run record count mismatch: {model_id}")
        expected_ids = [q.query_id for q in questions[:run["query_count"]]]
        if [r.query_id for r in records] != expected_ids:
            raise ValueError(f"Query order and identity mismatch: {model_id}")
        if {r.pipeline_id for r in records} != {EXPECTED_PIPELINE_IDS[model_id]}:
            raise ValueError(f"Unexpected pipeline identity for {model_id}")
        metrics = summarize(records, metadata, run["elapsed_including_initialization_seconds"])
        summaries.append({
            "model_id": model_id,
            "pipeline": PIPELINES[model_id],
            "run_id": run["run_id"],
            "execution_mode": run["execution_mode"],
            "evidence_basis": run.get("evidence_basis", "unrecorded"),
            "corpus_sha256": run.get("corpus_sha256"),
            "embedding_backend": run.get("embedding_backend"),
            **{key: value for key, value in metrics.items() if key != "by_category"},
        })
        for category, result in metrics["by_category"].items():
            per_query.append({"model_id": model_id, "category": category, **result})

    if len(modes) > 1 and not allow_partial:
        raise ValueError("Mixed retrieval-only and end-to-end runs: not a fair final comparison")

    output = workspace / "metrics" / f"matrix_{utc_timestamp()}"
    output.mkdir(parents=True, exist_ok=False)
    fields = list(summaries[0])
    with (output / "pipeline_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summaries)
    with (output / "category_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_query[0]))
        writer.writeheader()
        writer.writerows(per_query)
    note = {
        "model_count": len(selected),
        "missing_models": missing,
        "modes": sorted(modes),
        "100_query_complete": not missing and all(s["queries"] == 100 for s in summaries),
        "medical_accuracy_measured": False,
        "medical_gold_labels_available": False,
        "evidence_bases": sorted(set(run.get("evidence_basis", "unrecorded") for run in selected.values())),
        "document_graph_corpus_equivalence_established": False,
        "warning": (
            "B1-B4 use provisional AI-assisted source-summary texts; "
            "B5-B8 use G0 graph relationships. The evidence bases are not "
            "identical and cannot support a controlled architecture-only "
            "medical accuracy claim."
        ),
        "latency_notes": [
            "Total latency includes query execution; model initialization/preflight tracked by separate elapsed field.",
            "Different evidence types and model-generation modes should not be compared as equivalent quality metrics.",
            "Missing stage timings are null, not zero.",
            "These operational metrics are not medical answer correctness, faithfulness or hallucination rates.",
        ],
        "run_ids": {key: value["run_id"] for key, value in selected.items()},
    }
    (output / "report_manifest.json").write_text(
        json.dumps(note, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(output), **note}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("validate")
    run = commands.add_parser("run")
    run.add_argument("--models", default="all", help="all or comma-separated B1..B8")
    run.add_argument("--limit", type=int)
    run.add_argument("--top-k", type=int, default=5)
    run.add_argument("--encoder", choices=["sentence", "lsa"], default="sentence")
    run.add_argument("--retrieval-only", action="store_true")
    run.add_argument("--skip-complete", action="store_true", help="Reuse only fully verified runs from identical code, data and settings")
    rep = commands.add_parser("report")
    rep.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    if Path.cwd().resolve() != PROJECT_ROOT:
        parser.error("Run from CDF-RAG-2 project root")
    if args.command == "validate":
        questions, metadata = validate_inputs()
        print(json.dumps({
            "status": "VALIDATED",
            "queries": len(questions),
            "dataset_sha256": file_sha256(INPUT),
            "categories": dict(Counter(row["query_type"] for row in metadata.values())),
            "reference_basis": "unreviewed_pilot",
        }, indent=2))
    elif args.command == "run":
        run_models(args.models, args.limit, args.retrieval_only, args.top_k, args.encoder, args.skip_complete)
    else:
        report(args.allow_partial)


if __name__ == "__main__":
    main()
