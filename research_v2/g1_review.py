"""Review-gated G1 CANDIDATE graph expansion; never modify frozen G0.

queue: rank diagnostic graph coverage gaps and create an EMPTY evidence ledger.
check: validate approved source-attributed CAUSES edges before building anything.
build: merge human-reviewed candidates into a new versioned graph snapshot.
probe: inspect changed B5/B6 retrieval yield, NOT answer or clinical accuracy.

No causal relationships are fabricated from draft query metadata.
Approved means a reviewer attests to a source; URLs are NOT verified online.
G0 inherited edges remain unreviewed even in a candidate G1 graph.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import uuid
from collections import Counter
from datetime import date, datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

from .base_pipeline import PipelineRequest
from .experiment_runner import PROJECT_ROOT, file_sha256, load_questions
from .pipelines.baselines import BaselinePipeline, G0Corpus
from .presentation_matrix import INPUT, WORKSPACE, validate_inputs

G0_DIR = PROJECT_ROOT / "research_artifacts/graph-G0"
G1_WORKSPACE = WORKSPACE / "g1_review"
PRIORITY_BUCKETS = ("G0_SNAPSHOT_COVERAGE_GAP", "DIRECTED_RETRIEVAL_GAP")
QUEUE_COLUMNS = (
    "query_id", "review_bucket", "query_type", "medical_domain",
    "draft_target_concept", "question", "B5_status", "B6_status",
    "B5_evidence", "B6_evidence", "review_status", "research_notes"
)
EVIDENCE_COLUMNS = (
    "evidence_id", "query_id", "cause", "effect", "relation",
    "source_url", "source_title", "source_publisher", "evidence_note",
    "review_status", "reviewer", "reviewed_date"
)
ATTRIBUTION_COLUMNS = EVIDENCE_COLUMNS + ("edge_id", "edge_origin")
BLOCKED = {"tbd", "none", "unknown", "pending", "na", "n/a", "-", "example"}
WORD = re.compile(r"\s+")


def normalize(value: str) -> str:
    return WORD.sub(" ", value.strip()).casefold()


def allowed_path(path: Path) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_relative_to(PROJECT_ROOT):
        raise ValueError("G1 workspace must remain within repository")
    return resolved


def read_csv(path: Path) -> list[dict]:
    with Path(path).open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        if not reader.fieldnames:
            raise ValueError(f"No CSV headers: {path}")
        return list(reader)


def write_csv(path: Path, cols: tuple[str, ...], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as dest:
        writer = csv.DictWriter(dest, fieldnames=cols, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def select_review_gaps(rows: list[dict]) -> list[dict]:
    if len({r.get("query_id") for r in rows}) != len(rows):
        raise ValueError("Duplicate query IDs in diagnostics")
    if not rows:
        raise ValueError("No diagnostic rows")
    found = []
    for row in rows:
        bucket = row.get("retrieval_review_bucket")
        if bucket not in PRIORITY_BUCKETS:
            continue
        if row.get("B5_status") == "NOT_RUN" or row.get("B6_status") == "NOT_RUN":
            raise ValueError("Graph retrieval baselines must run before G1 planning")
        found.append({
            "query_id": row["query_id"],
            "review_bucket": bucket,
            "query_type": row.get("query_type", ""),
            "medical_domain": row.get("medical_domain", ""),
            "draft_target_concept": row.get("target_concept_draft", ""),
            "question": row.get("question", ""),
            "B5_status": row.get("B5_status", ""),
            "B6_status": row.get("B6_status", ""),
            "B5_evidence": row.get("B5_evidence", ""),
            "B6_evidence": row.get("B6_evidence", ""),
            "review_status": "unreviewed",
            "research_notes": "",
        })
    return sorted(found, key=lambda r: (
        PRIORITY_BUCKETS.index(r["review_bucket"]), r["query_id"]
    ))


def latest_diagnostics(workspace: Path = WORKSPACE) -> Path:
    candidates = sorted((workspace / "metrics").glob(
        "query_diagnostics_*/graph_gap_review_queue.csv"
    ), reverse=True)
    for path in candidates:
        manifest = path.parent / "diagnostics_manifest.json"
        if not manifest.exists():
            continue
        metadata = json.loads(manifest.read_text(encoding="utf-8"))
        if (metadata.get("input_sha256") == file_sha256(INPUT)
                and {"B5", "B6"}.issubset(metadata.get("measured_models", []))
                and metadata.get("question_count") == 100):
            return path
    raise FileNotFoundError(
        "No complete G0 diagnostics for the frozen pilot. "
        "Run: python evaluate_all_baselines.py diagnose"
    )


def queue(diagnostics: Path | None = None, output_root: Path = G1_WORKSPACE) -> Path:
    validate_inputs()
    source = allowed_path(diagnostics) if diagnostics else latest_diagnostics()
    gaps = select_review_gaps(read_csv(source))
    if not gaps:
        raise ValueError("No graph gaps identified; do not create an empty review project")
    root = allowed_path(output_root)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    path = root / f"queue_{stamp}_{uuid.uuid4().hex[:6]}"
    path.mkdir(parents=True, exist_ok=False)
    write_csv(path / "review_queue.csv", QUEUE_COLUMNS, gaps)
    write_csv(path / "evidence_candidates.csv", EVIDENCE_COLUMNS, [])
    manifest = {
        "stage": "review_queue_not_evidence",
        "diagnostics_file": source.relative_to(PROJECT_ROOT).as_posix(),
        "diagnostics_sha256": file_sha256(source),
        "query_count": len(gaps),
        "bucket_counts": dict(Counter(r["review_bucket"] for r in gaps)),
        "input_sha256": file_sha256(INPUT),
        "g0_nodes_sha256": file_sha256(G0_DIR / "nodes.csv"),
        "g0_edges_sha256": file_sha256(G0_DIR / "edges.csv"),
        "approval_count": 0,
        "medical_gold": False,
        "notes": (
            "Draft concepts are search hints, not facts. Enter only independently "
            "reviewed, source-supported directional CAUSES edges. "
            "Never auto-convert risk factors or correlations into causation."
        ),
    }
    (path / "queue_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"review_queue": str(path), **{
        k: manifest[k] for k in ("query_count", "bucket_counts", "approval_count")
    }}, indent=2))
    return path


def _provided(value: str) -> bool:
    return bool(value.strip()) and normalize(value) not in BLOCKED


def validate_candidates(file: Path, allowed_ids: set[str]) -> tuple[list[dict], dict]:
    with Path(file).open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if set(reader.fieldnames or []) != set(EVIDENCE_COLUMNS):
            raise ValueError("Evidence ledger must use exact EVIDENCE_COLUMNS")
        rows = list(reader)
    approved, seen = [], set()
    counts = Counter()
    for index, row in enumerate(rows, start=2):
        status = normalize(row.get("review_status") or "")
        if status not in {"proposed", "approved", "rejected"}:
            raise ValueError(f"Row {index}: review_status must be proposed/approved/rejected")
        counts[status] += 1
        if status != "approved":
            continue
        eid = row["evidence_id"].strip()
        if not _provided(eid) or eid in seen:
            raise ValueError(f"Row {index}: approved evidence ID missing/duplicate")
        seen.add(eid)
        if row["query_id"].strip() not in allowed_ids:
            raise ValueError(f"Row {index}: not a queued gap query")
        for field in ("cause", "effect", "source_url", "source_title",
                      "source_publisher", "evidence_note", "reviewer", "reviewed_date"):
            if not _provided(row[field]):
                raise ValueError(f"Row {index}: missing approved field: {field}")
        if normalize(row["cause"]) == normalize(row["effect"]):
            raise ValueError(f"Row {index}: self-causal edge not permitted")
        if normalize(row["relation"]) != "causes":
            raise ValueError(
                f"Row {index}: only CAUSES can enter a causal graph; "
                "association/risk-factor relationships require separately typed evidence"
            )
        if len(row["evidence_note"].strip()) < 20:
            raise ValueError(f"Row {index}: evidence note too short to review")
        parsed = urlsplit(row["source_url"].strip())
        if parsed.scheme != "https" or not parsed.netloc or parsed.username or parsed.password:
            raise ValueError(f"Row {index}: a public HTTPS source URL is required")
        if normalize(parsed.hostname or "") in {"example.com", "localhost", "127.0.0.1"}:
            raise ValueError(f"Row {index}: placeholder or local source URL")
        try:
            reviewed = date.fromisoformat(row["reviewed_date"].strip())
        except ValueError as exc:
            raise ValueError(f"Row {index}: reviewed_date must be ISO YYYY-MM-DD") from exc
        if reviewed > datetime.now(timezone.utc).date():
            raise ValueError(f"Row {index}: reviewed_date cannot be in future")
        approved.append({k: str(row[k]).strip() for k in EVIDENCE_COLUMNS})
    return approved, {
        "total_candidates": len(rows),
        "review_counts": dict(counts),
        "approved_evidence_rows": len(approved),
        "source_urls_checked_online": False,
        "medical_accuracy_gold_available": False,
    }


def review_file(queue_dir: Path) -> tuple[Path, list[dict], dict]:
    root = allowed_path(queue_dir)
    manifest = json.loads((root / "queue_manifest.json").read_text(encoding="utf-8"))
    if (file_sha256(INPUT) != manifest["input_sha256"]
            or file_sha256(G0_DIR / "nodes.csv") != manifest["g0_nodes_sha256"]
            or file_sha256(G0_DIR / "edges.csv") != manifest["g0_edges_sha256"]):
        raise ValueError("Frozen G0 or input changed; cannot reuse this queue")
    gaps = read_csv(root / "review_queue.csv")
    ids = {r["query_id"] for r in gaps}
    if len(ids) != len(gaps) or len(ids) != manifest["query_count"]:
        raise ValueError("Review queue query IDs changed")
    ledger = root / "evidence_candidates.csv"
    approved, audit = validate_candidates(ledger, ids)
    audit["queue_dir"] = root.relative_to(PROJECT_ROOT).as_posix()
    audit["evidence_ledger_sha256"] = file_sha256(ledger)
    return ledger, approved, audit


def _stable_id(prefix: str, value: str) -> str:
    return f"{prefix}_{hashlib.sha256(value.encode('utf-8')).hexdigest()[:16]}"


def build_graph(base: Path, approved: list[dict], target: Path) -> dict:
    if not approved:
        raise ValueError("No approved, source-attributed causal evidence; G1 not built")
    base = Path(base).resolve()
    target = allowed_path(target)
    if target.exists():
        raise FileExistsError(f"Never overwrite a candidate graph: {target}")
    nodes = read_csv(base / "nodes.csv")
    edges = read_csv(base / "edges.csv")
    if not all({"node_id", "name"} <= set(n) for n in nodes):
        raise ValueError("Invalid G0 node schema")
    if not all({"edge_id", "cause", "effect"} <= set(e) for e in edges):
        raise ValueError("Invalid G0 edge schema")
    names = {normalize(n["name"]): n["name"] for n in nodes}
    if len(names) != len(nodes):
        raise ValueError("Duplicate G0 concepts")
    existing_edges = {(normalize(e["cause"]), normalize(e["effect"])): e["edge_id"] for e in edges}
    if len(existing_edges) != len(edges):
        raise ValueError("Duplicate G0 causal links")
    base_count = len(edges)
    attributions = []
    added = []
    for row in approved:
        pair = (normalize(row["cause"]), normalize(row["effect"]))
        for concept in (row["cause"], row["effect"]):
            key = normalize(concept)
            if key not in names:
                name = concept.strip()
                names[key] = name
                nodes.append({"node_id": _stable_id("G1N", key), "name": name})
        if pair in existing_edges:
            edge_id = existing_edges[pair]
            origin = "reviewed_existing_g0" if edge_id in {e["edge_id"] for e in edges[:base_count]} else "reviewed_new_candidate"
        else:
            edge_id = _stable_id("G1E", "\x1f".join(pair))
            edges.append({"edge_id": edge_id, "cause": names[pair[0]], "effect": names[pair[1]]})
            existing_edges[pair] = edge_id
            added.append(edge_id)
            origin = "reviewed_new_candidate"
        attributions.append({**row, "edge_id": edge_id, "edge_origin": origin})
    # Legacy G0 links retain their unverified provenance, even when
    # source attribution for a subset is later reviewed.
    target.mkdir(parents=True, exist_ok=False)
    write_csv(target / "nodes.csv", ("node_id", "name"), nodes)
    write_csv(target / "edges.csv", ("edge_id", "cause", "effect"), edges)
    write_csv(target / "approved_edge_attributions.csv", ATTRIBUTION_COLUMNS, attributions)
    manifest = {
        "stage": "G1_CANDIDATE_REQUIRES_FULL_SOURCE_AUDIT",
        "source_g0_nodes_sha256": file_sha256(base / "nodes.csv"),
        "source_g0_edges_sha256": file_sha256(base / "edges.csv"),
        "source_g0_unreviewed_edges_inherited": base_count,
        "approved_evidence_rows": len(approved),
        "new_nodes": len(nodes) - len(read_csv(base / "nodes.csv")),
        "new_edges": len(added),
        "total_nodes": len(nodes),
        "total_edges": len(edges),
        "clinical_accuracy_verified": False,
        "source_urls_checked_online": False,
        "note": "Reviewer approval is an assertion, not independent source or clinical validation. G0 inherited relationships remain unreviewed.",
    }
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def build(queue_dir: Path) -> Path:
    ledger, approved, audit = review_file(queue_dir)
    if not approved:
        raise ValueError("No approved causal evidence. Fill and review evidence_candidates.csv first.")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    target = allowed_path(queue_dir) / "graphs" / f"G1_CANDIDATE_{stamp}_{uuid.uuid4().hex[:6]}"
    manifest = build_graph(G0_DIR, approved, target)
    manifest.update(audit)
    manifest["evidence_ledger_sha256"] = file_sha256(ledger)
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"graph_dir": str(target), **{
        k: manifest[k] for k in ("new_nodes", "new_edges", "total_nodes", "total_edges")
    }}, indent=2))
    return target


def probe(graph_dir: Path, top_k: int = 5) -> None:
    validate_inputs()
    if top_k < 1:
        raise ValueError("top-k must be positive")
    folder = allowed_path(graph_dir)
    info = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    if info.get("stage") != "G1_CANDIDATE_REQUIRES_FULL_SOURCE_AUDIT":
        raise ValueError("Only a versioned G1 candidate graph can be probed")
    questions = load_questions(INPUT)
    findings = []
    for label, source in (("G0", G0_DIR), ("G1_CANDIDATE", folder)):
        graph = G0Corpus(graph_dir=source)
        for model in ("graph_rag", "causal_kg_rag"):
            pipeline = BaselinePipeline(model, corpus=graph, top_k=top_k)
            records = [pipeline.run(PipelineRequest(
                run_id=label, query_id=q.query_id, query=q.query, graph_version=label
            )) for q in questions]
            findings.append({
                "graph": label, "model": model, "top_k": top_k,
                "queries": len(records),
                "retrieval_yield": sum(r.status == "ok" for r in records),
                "abstained": sum(r.status == "abstained" for r in records),
                "errors": sum(r.status == "error" for r in records),
            })
    target = folder / f"probe_k{top_k}.json"
    if target.exists():
        raise FileExistsError(f"Preserve previous probe: {target}")
    target.write_text(json.dumps({
        "results": findings, "reference_basis": "graph_snapshot",
        "clinical_accuracy": False, "new_graph_fully_verified": False,
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"probe": str(target), "results": findings}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="action", required=True)
    subs.add_parser("queue")
    check = subs.add_parser("check")
    check.add_argument("--queue-dir", required=True, type=Path)
    create = subs.add_parser("build")
    create.add_argument("--queue-dir", required=True, type=Path)
    assess = subs.add_parser("probe")
    assess.add_argument("--graph-dir", required=True, type=Path)
    assess.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    if Path.cwd().resolve() != PROJECT_ROOT:
        parser.error("Run from the CDF-RAG-2 root")
    if args.action == "queue":
        queue()
    elif args.action == "check":
        _, approved, audit = review_file(args.queue_dir)
        print(json.dumps(audit, indent=2))
    elif args.action == "build":
        build(args.queue_dir)
    else:
        probe(args.graph_dir, args.top_k)


if __name__ == "__main__":
    main()
