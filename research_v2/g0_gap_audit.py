"""Offline G0 graph coverage audit using the existing 100-query diagnostic CSV.

Does not modify G0, infer medical synonym equivalence, or generate new
causal edges. All proposed fixes require human source/concept review.
"""
from __future__ import annotations

import argparse
import csv
import json
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from .experiment_runner import PROJECT_ROOT, file_sha256
from .pipelines.baselines import G0Corpus, similarity
from .presentation_matrix import INPUT, WORKSPACE, validate_inputs

GAPS = frozenset({"G0_SNAPSHOT_COVERAGE_GAP", "DIRECTED_RETRIEVAL_GAP"})
COLUMNS = (
    "query_id", "bucket", "query_type", "draft_target",
    "target_in_g0_exact", "matcher_concept", "matcher_equals_target",
    "matched_node_in_degree", "matched_node_out_degree",
    "target_in_degree", "target_out_degree",
    "graph_select_B5_edges", "graph_select_B6_edges",
    "B5_evidence", "B6_evidence", "B7_evidence", "B8_evidence",
    "review_category", "investigation_priority", "concept_candidates_review_only",
    "reviewer_decision", "review_notes",
)


def normalize(value: str) -> str:
    return " ".join(value.strip().casefold().split())


def read_diagnostics(path: Path) -> list[dict]:
    with Path(path).open(newline="", encoding="utf-8") as src:
        reader = csv.DictReader(src)
        required = {"query_id", "question", "query_type", "target_concept_draft",
                    "retrieval_review_bucket", "B5_evidence", "B6_evidence",
                    "B7_evidence", "B8_evidence"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("Unsupported query diagnostics CSV schema")
        rows = list(reader)
    if len(rows) != 100 or len({r["query_id"] for r in rows}) != 100:
        raise ValueError("Expected 100 unique diagnostic queries")
    return rows


def latest_input(workspace: Path = WORKSPACE) -> Path:
    candidates = sorted((workspace / "metrics").glob(
        "query_diagnostics_*/graph_gap_review_queue.csv"
    ), reverse=True)
    for path in candidates:
        manifest = path.parent / "diagnostics_manifest.json"
        if not manifest.is_file():
            continue
        info = json.loads(manifest.read_text(encoding="utf-8"))
        if (info.get("input_sha256") == file_sha256(INPUT)
                and info.get("question_count") == 100
                and {"B5","B6","B7","B8"}.issubset(info.get("measured_models", []))):
            return path
    raise FileNotFoundError("No complete diagnostics. Run: python evaluate_all_baselines.py diagnose")


def classify(row: dict, corpus: G0Corpus) -> dict | None:
    bucket = row["retrieval_review_bucket"]
    if bucket not in GAPS:
        return None
    target = (row["target_concept_draft"] or "").strip()
    q = row["question"]
    graph_node = {normalize(n): n for n in corpus.nodes}
    exact_target = graph_node.get(normalize(target)) if target else None
    matched = corpus.concept(q)
    indegree = Counter(effect for cause, effect in corpus.edges)
    outdegree = Counter(cause for cause, effect in corpus.edges)
    b5, _ = _select(corpus, "graph_rag", q)
    b6, _ = _select(corpus, "causal_kg_rag", q)

    if not target:
        category = "NO_DRAFT_TARGET_REQUIRES_REVIEW"
        priority = 1
    elif exact_target is None:
        category = "TARGET_NOT_PRESENT_IN_G0_REQUIRES_ENTITY_REVIEW"
        priority = 2
    elif not matched:
        category = "TARGET_EXISTS_BUT_MATCHER_FOUND_NO_CONCEPT"
        priority = 1
    elif normalize(matched) != normalize(exact_target):
        category = "MATCHED_DIFFERENT_G0_CONCEPT"
        priority = 1
    elif bucket == "DIRECTED_RETRIEVAL_GAP":
        category = "MATCHED_TARGET_BUT_NO_DIRECTED_CAUSAL_EVIDENCE"
        priority = 2
    else:
        category = "TARGET_MATCHED_BUT_NO_GRAPH_NEIGHBOR_EVIDENCE"
        priority = 2
    candidates = [
        (similarity(target, n), n) for n in corpus.nodes
        if target and normalize(n) != normalize(target)
    ]
    ranked = sorted(candidates, key=lambda x: (-x[0], x[1]))
    # Similarity is a lexical triage hint, not medical synonym approval.
    notes = "; ".join(f"{name} ({score:.2f})" for score,name in ranked[:3]
                      if score > 0)
    return {
        "query_id": row["query_id"], "bucket": bucket,
        "query_type": row["query_type"], "draft_target": target,
        "target_in_g0_exact": bool(exact_target),
        "matcher_concept": matched or "",
        "matcher_equals_target": bool(exact_target and matched
                                      and normalize(matched) == normalize(exact_target)),
        "matched_node_in_degree": indegree.get(matched, 0),
        "matched_node_out_degree": outdegree.get(matched, 0),
        "target_in_degree": indegree.get(exact_target, 0),
        "target_out_degree": outdegree.get(exact_target, 0),
        "graph_select_B5_edges": len(b5), "graph_select_B6_edges": len(b6),
        "B5_evidence": row["B5_evidence"], "B6_evidence": row["B6_evidence"],
        "B7_evidence": row["B7_evidence"], "B8_evidence": row["B8_evidence"],
        "review_category": category, "investigation_priority": priority,
        "concept_candidates_review_only": notes,
        "reviewer_decision": "", "review_notes": "",
    }


def _select(corpus: G0Corpus, strategy: str, query: str) -> tuple[list[int], str | None]:
    # Pure index-level selection, no Ollama/Neo4j/generation.
    from .pipelines.baselines import BaselinePipeline
    return BaselinePipeline(strategy, corpus=corpus, generate=False).select(query)


def audit(diagnostics: Path | None = None, output_root: Path | None = None):
    validate_inputs()
    input_path = Path(diagnostics).expanduser().resolve() if diagnostics else latest_input()
    if not input_path.is_relative_to(PROJECT_ROOT):
        raise ValueError("Diagnostics must remain inside CDF-RAG-2 repository")
    rows = read_diagnostics(input_path)
    corpus = G0Corpus()
    findings = [item for row in rows if (item := classify(row, corpus)) is not None]
    if not findings:
        raise ValueError("No graph gap rows to audit")
    for row in findings:
        if row["bucket"] == "G0_SNAPSHOT_COVERAGE_GAP":
            assert row["B5_evidence"] == row["B6_evidence"] == "False"
        else:
            assert row["B5_evidence"] == "True" and row["B6_evidence"] == "False"
    findings.sort(key=lambda item: (
        item["investigation_priority"], item["review_category"], item["query_id"]
    ))
    root = Path(output_root or WORKSPACE / "g1_review").expanduser().resolve()
    if not root.is_relative_to(PROJECT_ROOT):
        raise ValueError("Output must remain in repository")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    out = root / f"gap_audit_{stamp}_{uuid.uuid4().hex[:6]}"
    out.mkdir(parents=True, exist_ok=False)
    with (out / "gap_classification.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(findings)
    summary = {
        "input": input_path.relative_to(PROJECT_ROOT).as_posix(),
        "input_sha256": file_sha256(input_path),
        "g0_nodes": len(corpus.nodes), "g0_edges": len(corpus.edges),
        "g0_nodes_sha256": file_sha256(PROJECT_ROOT / "research_artifacts/graph-G0/nodes.csv"),
        "g0_edges_sha256": file_sha256(PROJECT_ROOT / "research_artifacts/graph-G0/edges.csv"),
        "review_rows": len(findings),
        "bucket_counts": dict(Counter(x["bucket"] for x in findings)),
        "classification_counts": dict(Counter(x["review_category"] for x in findings)),
        "b8_evidence_in_gap_rows": sum(x["B8_evidence"] == "True" for x in findings),
        "requires_review": True, "medical_gold": False,
        "warning": (
            "Concept candidates are lexical hints only, not approved synonyms. "
            "G0 includes unreviewed causal claims; do not fill missing edges "
            "or treat evidence yield as relevance/clinical accuracy."
        ),
    }
    (out / "audit_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"output": str(out), **summary}, indent=2))
    return out, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostics", type=Path)
    args = parser.parse_args()
    if Path.cwd().resolve() != PROJECT_ROOT:
        parser.error("Run from repository root")
    audit(args.diagnostics)


if __name__ == "__main__":
    main()
