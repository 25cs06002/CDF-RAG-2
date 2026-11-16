"""Offline G0 graph-gap root-cause triage + conservative, SHADOW B6 resolver.

Inputs: frozen 100 pilot queries and saved query diagnostics.
No LLM/Neo4j, changes to B1-B8, or creation of medical claims.
Draft target labels are used ONLY for auditing gaps, never as inputs to
the proposed query-time concept selector. The shadow output is not medical
accuracy and must not replace independently adjudicated evaluation data.

  python -m research_v2.graph_gap_triage
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from .base_pipeline import PipelineRequest
from .experiment_runner import PROJECT_ROOT, file_sha256, git_commit
from .g1_review import latest_diagnostics
from .pipelines.baselines import BaselinePipeline, G0Corpus
from .presentation_matrix import WORKSPACE, validate_inputs

G0_DIR = PROJECT_ROOT / "research_artifacts/graph-G0"
OUTPUT_ROOT = WORKSPACE / "gap_triage"
GAP_BUCKETS = {"G0_SNAPSHOT_COVERAGE_GAP", "DIRECTED_RETRIEVAL_GAP"}
UNDERDETERMINED_TYPES = {"unsupported", "ambiguous"}


def csv_rows(path: Path) -> list[dict]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("CSV headers missing")
        return list(reader)


def save_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("Refuse empty CSV")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def exact_mention(query: str, concept: str) -> bool:
    """Match a graph label as a full lexical phrase, not substring fragments."""
    pattern = r"(?<![a-z0-9])" + re.escape(concept.casefold()) + r"(?![a-z0-9])"
    return bool(re.search(pattern, query.casefold()))


def effect_mentions(query: str, graph: G0Corpus) -> list[str]:
    """Exact graph targets only; never invent or use pilot target annotations."""
    mentioned = []
    for concept in graph.nodes:
        if not graph.reverse[concept] or not exact_mention(query, concept):
            continue
        # A general disease class is not an equivalent entity to its subtype.
        # Such mappings require separately reviewed concept normalization.
        if concept.casefold() == "diabetes" and re.search(
            r"\btype\s+[12]\s+diabetes\b", query, flags=re.I
        ):
            continue
        mentioned.append(concept)
    return mentioned


class TargetOrientedG0Corpus(G0Corpus):
    """Read-only exploratory resolver; baseline G0Corpus is untouched.

    Never replace an already answerable B6 target. If the baseline selected
    a cause-only concept, prefer a SINGLE explicitly mentioned graph effect.
    This deliberately does not infer synonyms (CAD->heart disease, T2DM->Diabetes).
    """

    def concept(self, query: str) -> str | None:
        original = super().concept(query)
        if original is not None and self.reverse[original]:
            return original
        targets = effect_mentions(query, self)
        return targets[0] if len(targets) == 1 else original


def triage(rows: list[dict], graph: G0Corpus) -> tuple[list[dict], dict]:
    if len(rows) != 100 or len({r.get("query_id") for r in rows}) != 100:
        raise ValueError("Expected 100 unique diagnostic queries")
    output = []
    breakdown = Counter()
    for item in rows:
        bucket = item.get("retrieval_review_bucket", "")
        if bucket not in GAP_BUCKETS:
            continue
        b5, b6 = item.get("B5_status"), item.get("B6_status")
        if b5 not in {"ok", "abstained"} or b6 not in {"ok", "abstained"}:
            raise ValueError("Diagnostics contain graph runtime errors or missing runs")
        question = item["question"]
        query_type = item["query_type"]
        target = (item.get("target_concept_draft") or "").strip()
        observed = graph.concept(question)
        mentioned = effect_mentions(question, graph)
        graph_target = next(
            (n for n in graph.nodes if n.casefold() == target.casefold()), None
        ) if target else None
        oriented = TargetOrientedG0Corpus.concept(graph, question) if isinstance(
            graph, TargetOrientedG0Corpus
        ) else None
        # The shadow matching decision is reconstructed without draft labels.
        if oriented is None:
            oriented = (
                observed if observed and graph.reverse[observed] else
                mentioned[0] if len(mentioned) == 1 else observed
            )
        if query_type in UNDERDETERMINED_TYPES:
            classification = "UNDERDETERMINED_REVIEW_NO_AUTOMATIC_GRAPH_EXPANSION"
            priority = 4
        elif bucket == "DIRECTED_RETRIEVAL_GAP":
            classification = "DIRECTED_TARGET_SELECTION_OR_MISSING_PATH"
            priority = 1
        elif not graph_target:
            classification = "DRAFT_TARGET_NOT_EXACTLY_IN_G0"
            priority = 2
        else:
            classification = "EXISTING_G0_TARGET_QUERY_MATCHING_OR_EVIDENCE_GAP"
            priority = 3
        breakdown[classification] += 1
        output.append({
            "query_id": item["query_id"],
            "priority": priority,
            "classification": classification,
            "original_bucket": bucket,
            "query_type": query_type,
            "medical_domain": item.get("medical_domain", ""),
            "question": question,
            "draft_target_hint_NOT_MODEL_INPUT": target,
            "draft_target_exact_in_g0": bool(graph_target),
            "baseline_concept_from_query": observed or "",
            "unique_known_effect_mentions": " | ".join(mentioned),
            "shadow_oriented_concept": oriented or "",
            "shadow_concept_changed": oriented != observed,
            "B5_evidence": item["B5_evidence"],
            "B6_evidence": item["B6_evidence"],
            "B7_evidence": item.get("B7_evidence", ""),
            "B8_evidence": item.get("B8_evidence", ""),
            "reviewer_decision": "",
            "source_or_concept_review_notes": "",
        })
    output.sort(key=lambda r: (r["priority"], r["query_id"]))
    return output, dict(breakdown)


def shadow_probe(questions) -> tuple[list[dict], dict]:
    """Independent inference using query strings; no pilot target labels."""
    original_graph, oriented_graph = G0Corpus(), TargetOrientedG0Corpus()
    original = BaselinePipeline("causal_kg_rag", corpus=original_graph, generate=False)
    oriented = BaselinePipeline("causal_kg_rag", corpus=oriented_graph, generate=False)
    paired = []
    for question in questions:
        before = original.run(PipelineRequest(
            run_id="G0_original_shadow", query_id=question.query_id,
            query=question.query, graph_version="G0"
        ))
        after = oriented.run(PipelineRequest(
            run_id="G0_target_oriented_shadow", query_id=question.query_id,
            query=question.query, graph_version="G0"
        ))
        before_yield = bool(before.retrieved_documents or before.causal_edges or before.causal_paths)
        after_yield = bool(after.retrieved_documents or after.causal_edges or after.causal_paths)
        paired.append({
            "query_id": question.query_id,
            "original_concept": before.matched_concept or "",
            "shadow_concept": after.matched_concept or "",
            "changed_concept": before.matched_concept != after.matched_concept,
            "original_evidence": before_yield,
            "shadow_evidence": after_yield,
            "original_edge_count": len(before.causal_edges),
            "shadow_edge_count": len(after.causal_edges),
            "original_path_count": len(before.causal_paths),
            "shadow_path_count": len(after.causal_paths),
            "original_retrieval_ms": before.timings.retrieval_ms,
            "shadow_retrieval_ms": after.timings.retrieval_ms,
        })
    summary = {
        "queries": len(paired),
        "original_evidence": sum(r["original_evidence"] for r in paired),
        "shadow_evidence": sum(r["shadow_evidence"] for r in paired),
        "changed_concept": sum(r["changed_concept"] for r in paired),
        "newly_recovered": sum(not r["original_evidence"] and r["shadow_evidence"] for r in paired),
        "regressed_yield": sum(r["original_evidence"] and not r["shadow_evidence"] for r in paired),
        "not_clinical_accuracy": True,
        "only_exact_effect_mentions_no_automatic_aliases": True,
    }
    return paired, summary


def run(*, diagnostics: Path | None = None, output_root: Path = OUTPUT_ROOT):
    questions, metadata = validate_inputs()
    diag = Path(diagnostics).expanduser().resolve() if diagnostics else latest_diagnostics()
    if not diag.is_relative_to(PROJECT_ROOT):
        raise ValueError("Diagnostics must remain inside repository")
    source = csv_rows(diag)
    if {q.query_id for q in questions} != {r["query_id"] for r in source}:
        raise ValueError("Diagnostic query IDs differ from frozen pilot")
    if any(r["question"] != metadata[r["query_id"]]["query"] for r in source):
        raise ValueError("Diagnostic question text mismatch")
    graph = G0Corpus()
    issues, groups = triage(source, graph)
    probe, effects = shadow_probe(questions)
    if effects["regressed_yield"]:
        raise AssertionError("Conservative shadow matcher reduced G0 retrieval yield")
    output_root = Path(output_root).expanduser().resolve()
    if not output_root.is_relative_to(PROJECT_ROOT):
        raise ValueError("Output must remain inside repository")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = output_root / f"triage_{stamp}_{uuid.uuid4().hex[:6]}"
    output.mkdir(parents=True, exist_ok=False)
    save_csv(output / "graph_gap_root_causes.csv", issues)
    save_csv(output / "b6_target_oriented_shadow.csv", probe)
    manifest = {
        "stage": "offline_shadow_research_not_production",
        "source_diagnostics": diag.relative_to(PROJECT_ROOT).as_posix(),
        "source_diagnostics_sha256": file_sha256(diag),
        "dataset_sha256": file_sha256(WORKSPACE / "benchmark/questions_only.jsonl"),
        "nodes_sha256": file_sha256(G0_DIR / "nodes.csv"),
        "edges_sha256": file_sha256(G0_DIR / "edges.csv"),
        "git_commit": git_commit(),
        "gaps": len(issues),
        "classifications": groups,
        "B6_shadow": effects,
        "warnings": [
            "Draft target is used only for audit; shadow model sees only question and G0.",
            "G0 graph relations are unreviewed and may contain medically unsupported causal assertions.",
            "Evidence yield does not demonstrate relevance, medical correctness or safety.",
            "An ambiguous or unsupported question must not become an asserted causal fact.",
            "Never replace frozen B6 or clinical ground truth with these shadow results.",
        ],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), **{
        k: manifest[k] for k in ("gaps", "classifications", "B6_shadow")
    }}, indent=2))
    return output, manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostics", type=Path)
    args = parser.parse_args()
    if Path.cwd().resolve() != PROJECT_ROOT:
        parser.error("Run from repository root")
    run(diagnostics=args.diagnostics)


if __name__ == "__main__":
    main()
