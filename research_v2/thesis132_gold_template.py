"""Create an EMPTY 132-question external gold-adjudication template.

This is a structure-only handoff; it never carries AI candidate answer
rubrics, benchmark original targets or G1-derived 'gold' into the reference.
The resulting file MUST fail independent-gold scoring until qualified human
reviewers and an adjudicator complete real work.
"""
from __future__ import annotations

import argparse
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .benchmark_v2_validation import validate
from .experiment_runner import file_sha256
from .thesis132 import _inside, _write_json

OUTPUT = Path(__file__).resolve().parents[1] / "presentation_20261118/thesis132/gold_templates"


def prepare(output_root: Path = OUTPUT):
    audit, rows, _ = validate()
    if audit["questions"] != 132 or audit["clinical_gold_questions"] != 0:
        raise ValueError("Expected 132 non-gold development questions")
    output_root = _inside(output_root)
    out = output_root / ("gold_template_" + datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%S%fZ") + "_" + uuid.uuid4().hex[:6])
    out.mkdir(parents=True, exist_ok=False)
    file = out / "gold_132_PENDING_REVIEW.jsonl"
    data = [{
        "query_id": q["query_id"],
        "query": q["query"],
        "answerability": None,
        "verified_answer": None,
        "relevant_source_ids": [],
        "gold_causal_edges": [],
        "gold_paths": [],
        "evidence_spans": [],
        "independent_reviewer_ids": [],
        "adjudicator_id": None,
        "review_status": "PENDING_EXTERNAL_ADJUDICATION",
    } for q in rows]
    file.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n"
                            for row in data), encoding="utf-8")
    manifest = {
        "schema": "132_GOLD_TEMPLATE_ONLY_V1",
        "query_count": 132,
        "file_sha256": file_sha256(file),
        "all_references_unreviewed": True,
        "no_references_certified": True,
        "development_exposed": True,
        "instructions": [
            "Use the independent 264-slot worksheet for real double-review before any gold release.",
            "Have an external adjudicator resolve disagreements and verify precise evidence passages.",
            "Create a NEW versioned final file; NEVER overwrite the pending template.",
            "The externally adjudicated release must pass research_v2.thesis132_gold_scoring.validate_gold.",
            "Original 132 were previously exposed to G1 construction; never call the cohort an external unseen holdout.",
        ],
    }
    _write_json(out / "manifest.json", manifest)
    print(json.dumps({"status": "EMPTY_GOLD_TEMPLATE_NOT_VALIDATED_GOLD",
                      "output": str(out), "questions": 132}, indent=2))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-root", type=Path, default=OUTPUT)
    args = p.parse_args()
    prepare(args.output_root)


if __name__ == "__main__":
    main()
