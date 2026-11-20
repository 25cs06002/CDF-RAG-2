"""Strict 132-query adjudicated-reference scoring for versioned run artifacts.

NO clinical gold is generated here. An externally approved gold JSONL is
required and must contain exactly the original 132 questions. The same
questions influenced G1 development: this is retrospective, NOT an unseen
holdout. Answer/groundedness quality requires separate human judgments.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from .benchmark_v2_validation import validate as validate_screened
from .experiment_runner import file_sha256
from .schema import PipelineRecord
from .thesis132 import _inside, _write_json, BASE

GOLD_FIELDS = {
    "query_id", "query", "answerability", "verified_answer",
    "relevant_source_ids", "gold_causal_edges", "gold_paths",
    "evidence_spans", "independent_reviewer_ids", "adjudicator_id",
    "review_status",
}
BOOL = {"yes": True, "no": False}


def _jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _normalize(t: str) -> str:
    return " ".join(str(t).casefold().split())


def _edges(rows: list[dict]) -> set[tuple[str, str, str]]:
    return {(_normalize(e["cause"]), _normalize(e["relation"]), _normalize(e["effect"]))
            for e in rows}


def validate_gold(path: Path, query_rows: list[dict]) -> dict[str, dict]:
    gold = _jsonl(path)
    expected = {q["query_id"]: q["query"] for q in query_rows}
    if len(gold) != 132 or len(expected) != 132:
        raise ValueError("Medical gold scoring requires exactly 132 references")
    if {x.get("query_id") for x in gold} != set(expected):
        raise ValueError("Independent references do not match all frozen query IDs")
    by_id = {}
    for r in gold:
        qid = r["query_id"]
        if qid in by_id or set(r) != GOLD_FIELDS or r["query"] != expected[qid]:
            raise ValueError("Missing/extra gold fields, duplicate ID or changed query: " + qid)
        if r["answerability"] not in ("answerable", "unanswerable"):
            raise ValueError("Invalid answerability: " + qid)
        if r["review_status"] != "externally_adjudicated":
            raise ValueError("Unreviewed reference: " + qid)
        reviewers = r["independent_reviewer_ids"]
        if not isinstance(reviewers, list) or len(reviewers) < 2 or any(
            not isinstance(x, str) or not x.strip() for x in reviewers
        ) or len(set(x.casefold() for x in reviewers)) != len(reviewers):
            raise ValueError("Two distinct external reviewers are required: " + qid)
        if not r["adjudicator_id"] or r["adjudicator_id"].casefold() in {
            x.casefold() for x in reviewers
        }:
            raise ValueError("Independent adjudicator required: " + qid)
        if not isinstance(r["relevant_source_ids"], list) or any(
            not isinstance(s, str) or not s.strip() for s in r["relevant_source_ids"]
        ):
            raise ValueError("Malformed relevant source IDs: " + qid)
        if not isinstance(r["gold_causal_edges"], list) or not isinstance(r["gold_paths"], list):
            raise ValueError("Gold edges/paths must be lists: " + qid)
        for edge in r["gold_causal_edges"]:
            if not isinstance(edge, dict) or not {"cause", "effect", "relation"}.issubset(edge):
                raise ValueError("Gold causal edge malformed: " + qid)
        for nodes in r["gold_paths"]:
            if not isinstance(nodes, list) or len(nodes) < 2 or any(
                not isinstance(n, str) or not n.strip() for n in nodes
            ):
                raise ValueError("Gold path malformed: " + qid)
        if not isinstance(r["evidence_spans"], list):
            raise ValueError("Evidence spans not a list: " + qid)
        for span in r["evidence_spans"]:
            if not isinstance(span, dict) or not all(
                isinstance(span.get(k), str) and span[k].strip()
                for k in ("source_id", "passage_locator", "exact_text")
            ):
                raise ValueError("A reviewed evidence span lacks exact text or locator: " + qid)
        if r["answerability"] == "answerable":
            if not isinstance(r["verified_answer"], str) or not r["verified_answer"].strip():
                raise ValueError("Missing independently verified reference answer: " + qid)
            if not r["relevant_source_ids"] or not r["evidence_spans"]:
                raise ValueError("Answerable reference requires verified source IDs/spans: " + qid)
            if not set(r["relevant_source_ids"]).issubset({
                x["source_id"] for x in r["evidence_spans"]
            }):
                raise ValueError("Source ID not supported by an evidence span: " + qid)
        else:
            if r["gold_causal_edges"] or r["gold_paths"]:
                raise ValueError("Individual unsupported query must not invent gold paths: " + qid)
        by_id[qid] = r
    return by_id


def score_record(record: PipelineRecord, gold: dict, top_k: int = 5) -> dict:
    if top_k < 1:
        raise ValueError("top_k must be >=1")
    answerable = gold["answerability"] == "answerable"
    abstained = record.status == "abstained"
    completed = record.status == "ok"
    # Precision/Recall measure provenance/source IDs, not text-level relevance
    # or medical truth by themselves.
    ids = list(dict.fromkeys(
        doc.source_id for doc in record.retrieved_documents[:top_k]
        if doc.source_id
    ))
    relevant = set(gold["relevant_source_ids"])
    source_hits = len(set(ids) & relevant)
    source_precision = source_hits / len(ids) if ids else None
    source_recall = source_hits / len(relevant) if relevant else None
    claimed_edges = _edges([{
        "cause": e.cause, "relation": e.relation, "effect": e.effect
    } for e in record.causal_edges])
    gold_edges = _edges(gold["gold_causal_edges"])
    edge_precision = len(claimed_edges & gold_edges) / len(claimed_edges) if claimed_edges else None
    edge_recall = len(claimed_edges & gold_edges) / len(gold_edges) if gold_edges else None
    gold_paths = {tuple(_normalize(x) for x in nodes) for nodes in gold["gold_paths"]}
    pred_paths = {tuple(_normalize(x) for x in path.nodes) for path in record.causal_paths}
    path_recall = len(gold_paths & pred_paths) / len(gold_paths) if gold_paths else None
    return {
        "query_id": record.query_id,
        "status": record.status,
        "answerability": gold["answerability"],
        "abstention_decision_correct": (not answerable and abstained) or (answerable and completed),
        "unsafe_nonabstention": not answerable and completed,
        "false_abstention": answerable and abstained,
        "execution_error": record.status == "error",
        "source_precision_at_k": source_precision,
        "source_recall_at_k": source_recall,
        "gold_edge_precision": edge_precision,
        "gold_edge_recall": edge_recall,
        "gold_path_recall": path_recall,
        "latency_ms": record.timings.total_ms,
        "retrieval_ms": record.timings.retrieval_ms,
        "rounds_recorded": record.feedback_iterations + 1,
        "answer_correct_human": None,
        "grounded_human": None,
        "clinically_safe_human": None,
    }


def _mean(values):
    finite = [v for v in values if v is not None]
    return statistics.mean(finite) if finite else None


def _summary(rows: list[dict]) -> dict:
    return {
        "queries": len(rows),
        "abstention_decision_accuracy": _mean([r["abstention_decision_correct"] for r in rows]),
        "unsafe_nonabstention": sum(r["unsafe_nonabstention"] for r in rows),
        "execution_errors": sum(r["execution_error"] for r in rows),
        "false_abstention": sum(r["false_abstention"] for r in rows),
        "source_precision_at_k": _mean([r["source_precision_at_k"] for r in rows]),
        "source_recall_at_k": _mean([r["source_recall_at_k"] for r in rows]),
        "gold_edge_precision": _mean([r["gold_edge_precision"] for r in rows]),
        "gold_edge_recall": _mean([r["gold_edge_recall"] for r in rows]),
        "gold_path_recall": _mean([r["gold_path_recall"] for r in rows]),
        "answer_correct_human": _mean([r["answer_correct_human"] for r in rows]),
        "grounded_human": _mean([r["grounded_human"] for r in rows]),
        "clinically_safe_human": _mean([r["clinically_safe_human"] for r in rows]),
        "mean_latency_ms": _mean([r["latency_ms"] for r in rows]),
    }


def _grouped_ci(rows: list[dict], folds: dict[str, str],
                field: str, *, seed=2026, iterations=1000) -> dict:
    groups = defaultdict(list)
    for r in rows:
        if r[field] is not None:
            groups[folds[r["query_id"]]].append(float(r[field]))
    if not groups:
        return {"observed": None, "lower_95": None, "upper_95": None}
    rng = random.Random(seed)
    keys = sorted(groups)
    draws = []
    for _ in range(iterations):
        sampled = [v for _ in keys for v in groups[rng.choice(keys)]]
        draws.append(statistics.mean(sampled))
    draws.sort()
    return {
        "observed": _mean([v for group in groups.values() for v in group]),
        "lower_95": draws[int(.025 * (iterations - 1))],
        "upper_95": draws[int(.975 * (iterations - 1))],
        "bootstrap_unit": "semantic_family",
        "note": "Within-development-cohort uncertainty, NOT external generalization CI",
    }


def _load_human_judgments(path: Path) -> dict:
    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    judgments = {}
    for r in rows:
        key = (r["model"], r["query_id"])
        if key in judgments:
            raise ValueError("Duplicate human judgment " + repr(key))
        if not r["reviewer_id"].strip():
            raise ValueError("Answer judgment missing real human reviewer declaration")
        for col in ("answer_correct", "grounded", "clinically_safe"):
            if r[col].strip().lower() not in BOOL:
                raise ValueError("Answer judgment must be yes/no for " + col)
        judgments[key] = {
            "answer_correct_human": BOOL[r["answer_correct"].strip().lower()],
            "grounded_human": BOOL[r["grounded"].strip().lower()],
            "clinically_safe_human": BOOL[r["clinically_safe"].strip().lower()],
        }
    return judgments


def evaluate(gold_file: Path, session_dir: Path,
             answer_judgments: Path | None = None, top_k: int = 5,
             output_root: Path | None = None) -> tuple[Path, dict]:
    gold_file = _inside(gold_file)
    session_dir = _inside(session_dir)
    audit, benchmark, _ = validate_screened()
    gold = validate_gold(gold_file, benchmark)
    session = json.loads((session_dir / "session.json").read_text(encoding="utf-8"))
    if session["query_count"] != 132 or not session.get("runs"):
        raise ValueError("Full 132-query model session required; not a smoke run")
    if session["dataset_sha256"] != file_sha256(
        _inside(session["dataset_path"])
    ):
        # Session dataset_path is repo relative, never a user-provided URL.
        raise ValueError("Session dataset SHA changed")
    judgments = _load_human_judgments(_inside(answer_judgments)) if answer_judgments else None
    groups = {r["query_id"]: r["semantic_group"] for r in benchmark}
    all_rows, reports = [], {}
    for model, details in sorted(session["runs"].items()):
        path = session_dir / details["folder"] / "records.jsonl"
        if file_sha256(path) != details["record_sha256"]:
            raise ValueError("Model output changed: " + model)
        records = [PipelineRecord.model_validate_json(x) for x in
                   path.read_text(encoding="utf-8").splitlines() if x.strip()]
        if len(records) != 132 or {r.query_id for r in records} != set(gold):
            raise ValueError("Incomplete model run: " + model)
        rows = []
        for rec in records:
            scored = score_record(rec, gold[rec.query_id], top_k)
            scored["model"] = model
            if judgments is not None:
                key = (model, rec.query_id)
                if key not in judgments:
                    raise ValueError("Human judgment missing: " + repr(key))
                scored.update(judgments[key])
            rows.append(scored)
        all_rows.extend(rows)
        reports[model] = {
            **_summary(rows),
            "group_bootstrap_95": {
                key: _grouped_ci(rows, groups, key)
                for key in ("abstention_decision_correct", "source_recall_at_k",
                            "answer_correct_human")
            },
        }
    if judgments is not None and set(judgments) != {
        (m, q) for m in session["runs"] for q in gold
    }:
        raise ValueError("Unexpected or extra human judgments")
    root = _inside(output_root) if output_root else session_dir.parent
    out = root / ("gold_scoring_" + datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%S%fZ") + "_" + uuid.uuid4().hex[:6])
    out.mkdir(parents=True, exist_ok=False)
    with (out / "per_query.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    result = {
        "mode": "EXTERNALLY_DECLARED_GOLD_132_RETROSPECTIVE",
        "gold_sha256": file_sha256(gold_file),
        "session_sha256": file_sha256(session_dir / "session.json"),
        "answer_judgments_available": judgments is not None,
        "unseen_holdout": False,
        "score_caveats": [
            "Review status, reviewer identity, clinical correctness and source authenticity require independent external authentication.",
            "Provenance-ID precision and recall are not medical answer accuracy.",
            "Do not compare B1-B8 retrieval relevance causally if session evidence_bases_differ is true.",
            "Automatic generation of medical correctness/groundedness labels is prohibited.",
        ],
        "evidence_bases_differ": session.get("evidence_bases_differ", True),
        "models": reports,
    }
    _write_json(out / "summary.json", result)
    return out, result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gold", required=True, type=Path)
    p.add_argument("--session-dir", required=True, type=Path)
    p.add_argument("--answer-judgments", type=Path)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--output-root", type=Path)
    args = p.parse_args()
    out, data = evaluate(args.gold, args.session_dir, args.answer_judgments,
                         args.top_k, args.output_root)
    print(json.dumps({"output": str(out), **data}, indent=2))


if __name__ == "__main__":
    main()
