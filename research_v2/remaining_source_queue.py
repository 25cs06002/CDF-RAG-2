"""Phase 2F: candidate reading queue for remaining 70 pilot questions.

All sources are unreviewed reading candidates, not evidence that any
particular answer, causal relation, or multi-hop path is correct.
"""

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator

from .medical_benchmark import file_sha256
from .pilot_preparation import validate_pilot
from .source_review_queue import load_candidate_sources


PILOT = Path("data/benchmark/medical_causal_pilot100_draft_v1.jsonl")
BASE_SOURCES = Path("data/benchmark/source_candidates15_v2.jsonl")
EXTRA_SOURCES = Path("data/benchmark/additional_source_candidates8_v1.jsonl")
SOURCE_MAP = Path("data/benchmark/remaining70_source_map_v1.jsonl")

ROUTES = {
    "indirect": "concept_resolution",
    "multi_hop": "multi_hop_link_evidence",
    "comparative": "two_topic_comparison",
    "risk_factor": "population_risk",
    "ambiguous": "uncertain_symptom",
    "unsupported": "unsupported_individual_prediction",
}
EXPECTED_COUNTS = {
    "indirect": 15, "multi_hop": 15, "comparative": 12,
    "risk_factor": 12, "ambiguous": 8, "unsupported": 8,
}


class SupplementalSource(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    url: HttpUrl
    topic: str = Field(min_length=1)
    coverage_purpose: str = Field(min_length=1)
    scope_warning: str = Field(min_length=1)
    candidate_status: Literal["external_page_found_needs_question_review"]
    source_checked_date: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")
    human_review_completed: Literal[False]
    medical_gold_created: Literal[False]


class SourceMapping(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query_id: str = Field(min_length=1)
    candidate_source_ids: list[str] = Field(default_factory=list)
    review_route: Literal[
        "concept_resolution", "multi_hop_link_evidence",
        "two_topic_comparison", "population_risk",
        "uncertain_symptom", "unsupported_individual_prediction",
    ]
    needs_additional_sources: bool
    scope_note: str = Field(min_length=20)
    human_review_completed: Literal[False]
    candidate_sources_verify_question_answer: Literal[False]
    gold_annotation_created: Literal[False]

    @model_validator(mode="after")
    def check_refs(self):
        if len(self.candidate_source_ids) != len(set(self.candidate_source_ids)):
            raise ValueError("Duplicate candidate source IDs in mapping")
        if not self.candidate_source_ids and (
            self.review_route != "unsupported_individual_prediction"
            or not self.needs_additional_sources
        ):
            raise ValueError(
                "Empty source list allowed only for unsupported questions "
                "with an explicit evidence gap"
            )
        if self.review_route == "multi_hop_link_evidence" and (
            not self.needs_additional_sources
        ):
            raise ValueError("Multi-hop requires independent link verification")
        if self.review_route == "two_topic_comparison" and (
            len(self.candidate_source_ids) < 2
            and not self.needs_additional_sources
        ):
            raise ValueError("Comparison requires two sources or explicit gap")
        return self


def read_jsonl(path, model):
    rows = []
    with Path(path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                rows.append(model.model_validate_json(line))
            except Exception as exc:
                raise ValueError(f"Invalid source queue entry at line {number}: {exc}") from exc
    if not rows:
        raise ValueError(f"Empty source queue input: {path}")
    return rows


def build_remaining_queue(
    pilot_path=PILOT,
    base_sources_path=BASE_SOURCES,
    extra_sources_path=EXTRA_SOURCES,
    source_map_path=SOURCE_MAP,
):
    pilot, pilot_profile = validate_pilot(pilot_path)
    pending = {
        item.query_id: item for item in pilot
        if item.query_type not in {"direct", "paraphrased"}
    }
    if len(pending) != 70:
        raise ValueError("Pilot must contain 70 remaining questions")

    base_sources = load_candidate_sources(base_sources_path)
    extra_sources = read_jsonl(extra_sources_path, SupplementalSource)
    ids, urls = set(), set()
    catalog = {}

    for item in [*base_sources, *extra_sources]:
        if item.source_id in ids:
            raise ValueError(f"Duplicate cross-registry source ID: {item.source_id}")
        ids.add(item.source_id)
        url = str(item.url).rstrip("/").casefold()
        if url in urls:
            raise ValueError(f"Duplicate cross-registry source URL: {url}")
        urls.add(url)
        catalog[item.source_id] = item

    mappings = read_jsonl(source_map_path, SourceMapping)
    if len(mappings) != 70:
        raise ValueError(f"Expected 70 source mappings; got {len(mappings)}")

    output = []
    seen = set()
    for row in mappings:
        if row.query_id in seen:
            raise ValueError(f"Duplicate source mapping query: {row.query_id}")
        seen.add(row.query_id)
        question = pending.get(row.query_id)
        if question is None:
            raise ValueError(f"Unknown or already-covered question: {row.query_id}")
        if row.review_route != ROUTES[question.query_type]:
            raise ValueError(f"Review route does not match question type: {row.query_id}")
        unknown = set(row.candidate_source_ids) - ids
        if unknown:
            raise ValueError(
                f"Unknown supplemental source ID for {row.query_id}: "
                + ", ".join(sorted(unknown))
            )

        sources = []
        for source_id in row.candidate_source_ids:
            source = catalog[source_id]
            sources.append({
                "source_id": source_id,
                "title": source.title,
                "url": str(source.url),
                "candidate_origin": (
                    "supplemental_link" if source_id.startswith("EXT")
                    else "original_topic_registry"
                ),
            })
        output.append({
            "query_id": row.query_id,
            "query": question.query,
            "query_type": question.query_type,
            "medical_domain": question.medical_domain,
            "target_concept": question.target_concept,
            "review_route": row.review_route,
            "candidate_sources": sources,
            "source_bundle_status": (
                "no_candidate_source" if not sources else
                "incomplete_candidate_bundle" if row.needs_additional_sources
                else "topic_background_not_claim_verified"
            ),
            "needs_additional_sources": row.needs_additional_sources,
            "scope_note": row.scope_note,
            "review_tasks": [
                "Check source accessibility, date, evidence scope and version",
                "Verify the question against actual source passages",
                "Require separate evidence for each multi-hop arrow",
                "Distinguish disease onset from triggers and risk association",
                "Identify evidence gaps; do not invent a gold causal path",
                "Complete actual human review separately",
            ],
            "human_review_completed": False,
            "gold_annotation_created": False,
        })

    if seen != set(pending):
        raise ValueError("Incomplete remaining 70-question source coverage")

    profile = {
        "schema_version": 1,
        "data_role": "unreviewed_source_discovery_queue_not_medical_gold",
        "pilot_questions": pilot_profile["total_questions"],
        "remaining_questions": len(output),
        "previous_30_question_worklist_not_modified": True,
        "query_type_counts": dict(sorted(Counter(
            pending[row["query_id"]].query_type for row in output
        ).items())),
        "candidate_source_links": sum(
            len(row["candidate_sources"]) for row in output
        ),
        "unique_candidate_sources_used": len({
            src["source_id"]
            for row in output for src in row["candidate_sources"]
        }),
        "questions_with_candidate_links": sum(
            bool(row["candidate_sources"]) for row in output
        ),
        "questions_without_candidate_links": sum(
            not row["candidate_sources"] for row in output
        ),
        "questions_needing_additional_sources": sum(
            row["needs_additional_sources"] for row in output
        ),
        "human_review_completed": False,
        "causal_paths_verified": False,
        "medical_gold_created": False,
        "final_splits_assigned": False,
        "pilot_sha256": file_sha256(pilot_path),
        "base_sources_sha256": file_sha256(base_sources_path),
        "extra_sources_sha256": file_sha256(extra_sources_path),
        "source_map_sha256": file_sha256(source_map_path),
    }
    return output, profile


def export_remaining_queue(
    output_dir, pilot_path=PILOT, base_sources_path=BASE_SOURCES,
    extra_sources_path=EXTRA_SOURCES, source_map_path=SOURCE_MAP,
):
    rows, profile = build_remaining_queue(
        pilot_path, base_sources_path, extra_sources_path, source_map_path
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    (output / "remaining70_source_queue.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
        encoding="utf-8",
    )
    (output / "remaining70_source_profile.json").write_text(
        json.dumps(profile, indent=2) + "\n", encoding="utf-8"
    )
    return profile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", type=Path, default=PILOT)
    parser.add_argument("--base-sources", type=Path, default=BASE_SOURCES)
    parser.add_argument("--extra-sources", type=Path, default=EXTRA_SOURCES)
    parser.add_argument("--source-map", type=Path, default=SOURCE_MAP)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.output_dir:
        _, profile = None, export_remaining_queue(
            args.output_dir, args.pilot, args.base_sources,
            args.extra_sources, args.source_map
        )
    else:
        _, profile = build_remaining_queue(
            args.pilot, args.base_sources, args.extra_sources, args.source_map
        )
    print(json.dumps(profile, indent=2))


if __name__ == "__main__":
    main()
