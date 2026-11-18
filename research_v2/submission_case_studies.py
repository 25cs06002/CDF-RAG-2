"""Build reviewer-grounded G1 error case studies for partial MTP submission.

Read-only inputs: an independently audited submission and the G1 triage CSVs.
Outputs: new versioned case-study Markdown, supervisor brief, and index JSON.
All examples retain the original reviewer assessments. No new clinical labels,
graph changes, model runs, LLM calls or external-source verification.
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

ROOT = PROJECT_ROOT
OUT = ROOT / "presentation_20261118/submission_case_studies"
PRIORITIES = (
    "P0_WRONG_TARGET", "P0_UNSAFE_RETURN",
    "P1_UNSUPPORTED_GAIN", "P1_INCORRECT_EDGE_DIRECTION",
    "P2_TARGET_UNCERTAIN", "P3_OTHER",
)
STAGES = ("G0_LEGACY", "G1_LEGACY", "G1_STABLE", "G1_TYPED")


def _path(value):
    result = Path(value).expanduser().resolve()
    if not result.is_relative_to(ROOT):
        raise ValueError("Input/output must remain inside CDF-RAG-2")
    return result


def _data(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _csv(path):
    with path.open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def _require(condition, message):
    if not condition:
        raise ValueError("Case-study integrity error: " + message)


def _quote(value):
    """Escape Markdown table markup; all text is local reviewer data."""
    return " ".join(str(value or "").split()).replace("|", "\\|").replace("`","'")


def _bool(value):
    return value in ("True", "true", "1", True)


def _table(rows, keys):
    if not rows:
        return "_No rows recorded._"
    head = "| " + " | ".join(keys) + " |"
    sep = "| " + " | ".join("---" for _ in keys) + " |"
    body = [
        "| " + " | ".join(_quote(r.get(k, "")) for k in keys) + " |"
        for r in rows
    ]
    return "\n".join([head, sep, *body])


def validate(submission_dir, audit_dir, triage_dir):
    submission = _path(submission_dir)
    audit = _path(audit_dir)
    triage = _path(triage_dir)
    manifest = _data(submission / "submission_manifest.json")
    audit_record = _data(audit / "audit.json")
    triage_record = _data(triage / "summary.json")
    _require(audit_record["status"] == "PASS_TECHNICAL_INTEGRITY",
             "Independent submission audit is not passed")
    _require(
        audit_record["submission_dir"] ==
        str(submission.relative_to(ROOT)),
        "Audit points to a different submission")
    _require(
        audit_record["submission_manifest_sha256"] ==
        file_sha256(submission / "submission_manifest.json"),
        "Submission manifest changed after technical audit")
    _require(triage_record["mode"] ==
             "reviewed_changed_case_root_cause_triage_only",
             "Not an original reviewed G1 triage")
    _require(
        triage_record["source_analysis"] ==
        manifest["review_analysis_dir"],
        "Triage and submission reference different review snapshots")
    cases = _csv(triage / "priority_cases.csv")
    evidence = _csv(triage / "reviewed_edge_details.csv")
    _require(len(cases) == triage_record["attention_cases"],
             "Different prioritized case count")
    _require(len(evidence) == triage_record["evidence_occurrences_on_attention_cases"],
             "Different evidence occurrence count")
    _require(
        len({r["case_id"] for r in cases}) == len(cases),
        "Duplicate model-query case")
    _require(
        len({r["query_id"] for r in cases}) ==
        triage_record["distinct_attention_queries"],
        "Question count differs from frozen triage")
    actual_priorities = Counter(r["priority"] for r in cases)
    _require(
        dict(actual_priorities) == triage_record["case_priority_counts"],
        "Priority counts differ from frozen triage")
    _require(set(actual_priorities).issubset(PRIORITIES),
             "Unrecognized priority label")
    by_id = {r["case_id"]: r for r in cases}
    details = defaultdict(list)
    for row in evidence:
        cid = row["case_id"]
        _require(cid in by_id, "Evidence references unknown case " + cid)
        _require(row["query_id"] == by_id[cid]["query_id"],
                 "Case/evidence query identity differs")
        _require(row["family"] == by_id[cid]["family"],
                 "Case/evidence model family differs")
        _require(row["stage"] in STAGES,
                 "Evidence stage is unexpected: " + row["stage"])
        details[cid].append(row)
    for cid, item in by_id.items():
        for stage in STAGES:
            actual = sum(r["stage"] == stage for r in details[cid])
            _require(
                actual == int(item[stage + "_evidence"]),
                cid + " has inconsistent evidence occurrences in " + stage)
    return manifest, audit_record, triage_record, cases, details


def choose_examples(cases):
    """Choose one P0 wrong-target and one distinct P1 unsupported-gain case."""
    selected = []
    used_queries = set()
    for priority in ("P0_WRONG_TARGET", "P1_UNSUPPORTED_GAIN"):
        options = sorted(
            (c for c in cases if c["priority"] == priority),
            key=lambda r: (
                r["query_id"] in used_queries,
                not bool(r["review_note"].strip()),
                r["query_id"],
                r["family"],
                r["case_id"],
            ))
        if options:
            selected.append(options[0])
            used_queries.add(options[0]["query_id"])
    return selected


def _example_block(c, evidence, ordinal):
    cid = c["case_id"]
    title = (
        "Wrong target selection" if c["priority"] == "P0_WRONG_TARGET"
        else "Unsupported evidence gain" if c["priority"] == "P1_UNSUPPORTED_GAIN"
        else c["priority"]
    )
    lines = [
        "### Case " + str(ordinal) + " — " + title,
        "",
        "**Identity:** `" + _quote(cid) + "` · " +
        _quote(c["family"]) + " · `" + _quote(c["query_id"]) + "`",
        "",
        "**Original question:** " + _quote(c["question"]),
        "",
        "**Observed target selection:**",
        "",
        "- G0 target: `" + _quote(c["g0_target"]) + "`",
        "- G1 legacy target: `" + _quote(c["g1_legacy_target"]) + "`",
        "- G1 stabilized target: `" + _quote(c["stable_target"]) + "`",
        "- Selection reason: `" + _quote(c["selection_reason"]) + "`",
        "- Reviewer-assessed stable target: **" +
          _quote(c["review_stable_target"]) + "**",
        "- Review intent: " + _quote(c["review_query_intent"]),
        "- Reviewed flags: `" + _quote(c["issue_flags"]) + "`",
        "- Original G1 / stabilized retrieval transitions: " +
          _quote(c["original_G1_change"]) + " / " +
          _quote(c["g0_to_stable_change"]),
        "",
        "**Retrieved evidence occurrences by stage:**",
        "",
    ]
    counts = [
        {"stage": stage,
         "occurrences": sum(r["stage"] == stage for r in evidence)}
        for stage in STAGES
    ]
    lines.extend([_table(counts, ("stage", "occurrences")), ""])
    stable = [r for r in evidence if r["stage"] == "G1_STABLE"]
    if stable:
        lines.extend([
            "**Sample stabilized retrieved relationships (up to 4):**",
            "",
        ])
        edge_rows = [
            {
                "relationship": r["subject"] + " " + r["relation"] + " " + r["object"],
                "provenance": r["provenance"],
                "relevance_review": r["evidence_relevance"],
                "direction_review": r["relation_direction"],
                "source_checked": r["source_checked"],
            }
            for r in stable[:4]
        ]
        lines.extend([
            _table(
                edge_rows,
                ("relationship", "provenance", "relevance_review",
                 "direction_review", "source_checked")),
            "",
        ])
    else:
        lines.extend(["**Stabilized evidence:** no graph evidence returned.", ""])
    if c["review_note"].strip():
        lines.extend([
            "**Reviewer note:** " + _quote(c["review_note"]),
            "",
        ])
    lines.extend([
        "**Interpretation:** The recorded target/evidence was assessed by "
        "the reviewer; this example does not establish a medical ground-truth "
        "target or independent clinical correctness.",
        "",
        "**Candidate engineering response (not yet evaluated):** " +
        (
            "introduce source-backed subtype/alias discrimination and abstain "
            "on unresolved target ambiguity."
            if c["priority"] == "P0_WRONG_TARGET" else
            "apply type-aware evidence sufficiency checks and withhold "
            "answers where only irrelevant, unverified or incorrectly "
            "directed relationships are available."
        ),
        "",
    ])
    return "\n".join(lines)


def create(submission_dir, audit_dir, triage_dir, output_root=OUT):
    manifest, audited, triage, cases, details = validate(
        submission_dir, audit_dir, triage_dir
    )
    selected = choose_examples(cases)
    _require(len(selected) == 2 and
             {r["priority"] for r in selected} ==
             {"P0_WRONG_TARGET", "P1_UNSUPPORTED_GAIN"},
             "Cannot select two distinct error classes")
    priority_rows = [
        {"priority": priority,
         "cases": sum(r["priority"] == priority for r in cases),
         "questions": len({r["query_id"] for r in cases
                          if r["priority"] == priority})}
        for priority in PRIORITIES
        if any(r["priority"] == priority for r in cases)
    ]
    reasons = Counter(c["selection_reason"] for c in cases)
    reason_rows = [
        {"selection_reason": key, "cases": value}
        for key, value in sorted(reasons.items())
    ]
    reason_cross = [
        {"selection_reason": reason, "priority": priority, "cases":
         sum(c["selection_reason"] == reason and
             c["priority"] == priority for c in cases)}
        for reason in sorted(reasons)
        for priority in PRIORITIES
        if any(c["selection_reason"] == reason and
               c["priority"] == priority for c in cases)
    ]
    md = [
        "# G1 target and evidence failures — reviewer-grounded case studies",
        "",
        "## Scope",
        "",
        f"{triage['attention_cases']} prioritized model-query cases, " +
        f"{triage['distinct_attention_queries']} distinct questions and " +
        f"{triage['evidence_occurrences_on_attention_cases']} evidence occurrences.",
        "",
        "These are selected changed cases from draft questions. "
        "No independent clinical gold standard or held-out medical accuracy "
        "is established. A reviewer source-check attestation is not "
        "independent clinical certification.",
        "",
        "## Priority distribution",
        "",
        _table(priority_rows, ("priority", "cases", "questions")),
        "",
        "## Selection-reason diagnostics",
        "",
        _table(reason_rows, ("selection_reason", "cases")),
        "",
        "**Priority × selection-reason intersections:**",
        "",
        _table(reason_cross, ("selection_reason", "priority", "cases")),
        "",
        "The classification is descriptive. The 17 cases are not "
        "17 independent questions; they cover 12 distinct questions.",
        "",
        "## Representative cases (retrospective)",
        "",
    ]
    for i, c in enumerate(selected, 1):
        md.append(_example_block(c, details[c["case_id"]], i))
    md.extend([
        "## Proposed implementation sequence",
        "",
        "1. Record the failure mechanism and maintain abstention for "
        "unresolved target ambiguities; do not invent synonym mappings.",
        "2. Add provenance/type-aware evidence sufficiency and explicit "
        "fallback/abstention rules without changing historical labels.",
        "3. Treat these 17 cases as exploratory development diagnostics; "
        "evaluate any fix separately on independently reviewed examples.",
        "4. Only then study dynamic multi-round refinement and latency.",
        "",
        "## What this does NOT prove",
        "",
        "No clinical answer accuracy, causal-effect validity, model "
        "generalization, or adaptive multi-round improvement is claimed.",
        "",
    ])
    short = [
        "# Supervisor briefing — audited partial MTP work",
        "",
        "**Research topic:** Performance-aware, evidence-grounded adaptive "
        "multi-round causal RAG for domain-specific question answering.",
        "",
        "**Completed reproducible work:** B1–B8 end-to-end historical runs "
        "on 100 draft questions; B1–B6 retrieval-only evaluation on 100+32 "
        "questions (792 model-query records); G0/G1 paired B5/B6 retrieval "
        "comparison; 31 reviewed change cases comprising 132 evidence "
        "occurrences; technical audit passed.",
        "",
        "**Graph expansion investigation:** A review-gated G1 candidate "
        "graph expands G0, but the legacy query-to-graph resolver can "
        "lose retrieval coverage after expansion. Stabilized matching "
        "recovers some coverage; more coverage alone is not evidence quality.",
        "",
        "**Failure analysis:** " +
        str(triage["attention_cases"]) + " model-query cases over " +
        str(triage["distinct_attention_queries"]) + " questions require "
        "attention. Wrong target and unsupported evidence gains are "
        "the two primary prioritization categories.",
        "",
        "**Demonstrable individual contribution:** structured baseline "
        "benchmarking, graph/source-review workflow, G0/G1 regression "
        "diagnosis, stabilized retrieval evaluation, reviewer-based "
        "failure classification and reproducible experiment artifacts.",
        "",
        "**Next scientific step:** an evidence-aware concept-selection "
        "and sufficiency gate, followed by matched ablations and an "
        "independently adjudicated evaluation.",
        "",
        "**Bounded preliminary claim:** Graph expansion changes retrieval "
        "coverage partly via query-target selection; stability-aware "
        "retrieval can mitigate some regressions. The current results "
        "are not independently validated medical accuracy.",
        "",
        "## Two cases to discuss",
        "",
    ]
    for c in selected:
        short.append(
            "- `" + _quote(c["case_id"]) + "` (" +
            _quote(c["priority"]) + "): " + _quote(c["question"]) +
            "; stable target `" + _quote(c["stable_target"]) +
            "`; review " + _quote(c["review_stable_target"]) + "."
        )
    short.extend([
        "",
        "See `representative_case_studies.md` for exact provenance and "
        "reviewed relationship details. See the independently generated "
        "`slide_factsheet.md` for all measured model/latency values.",
        "",
        "**Supervisor questions:** Which reviewer assessments require "
        "expert adjudication? What is the acceptable abstention policy? "
        "Which sufficiency signals and matched ablations should be "
        "prioritized before PPO?",
        "",
    ])
    out = _path(output_root) / (
        "case_studies_" +
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") +
        "_" + uuid.uuid4().hex[:6]
    )
    out.mkdir(parents=True, exist_ok=False)
    (out / "representative_case_studies.md").write_text(
        "\n".join(md) + "\n", encoding="utf-8")
    (out / "supervisor_brief.md").write_text(
        "\n".join(short) + "\n", encoding="utf-8")
    record = {
        "status": "REVIEWER_GROUNDED_CASE_STUDIES_READY",
        "submission_manifest_sha256": audited["submission_manifest_sha256"],
        "triage_summary_sha256": file_sha256(_path(triage_dir) / "summary.json"),
        "triage_priority_csv_sha256": file_sha256(_path(triage_dir) /
                                                "priority_cases.csv"),
        "triage_evidence_csv_sha256": file_sha256(_path(triage_dir) /
                                                "reviewed_edge_details.csv"),
        "cases": triage["attention_cases"],
        "distinct_questions": triage["distinct_attention_queries"],
        "evidence_occurrences": triage["evidence_occurrences_on_attention_cases"],
        "selection_reason_by_priority": reason_cross,
        "selected_case_ids": [c["case_id"] for c in selected],
        "selected_query_ids": [c["query_id"] for c in selected],
        "medical_gold": False,
        "independent_holdout": False,
        "outputs": ["representative_case_studies.md", "supervisor_brief.md"],
    }
    (out / "case_studies_manifest.json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(out), **record}, indent=2))
    return out, record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission-dir", required=True, type=Path)
    parser.add_argument("--audit-dir", required=True, type=Path)
    parser.add_argument("--triage-dir", required=True, type=Path)
    parser.add_argument("--output-root", type=Path, default=OUT)
    args = parser.parse_args()
    if Path.cwd().resolve() != ROOT:
        parser.error("Run from CDF-RAG-2 project root")
    create(args.submission_dir, args.audit_dir, args.triage_dir, args.output_root)


if __name__ == "__main__":
    main()
