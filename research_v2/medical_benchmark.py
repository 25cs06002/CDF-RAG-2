"""Phase 2 medical *causal* QA benchmark validation and provenance audit.

Draft annotations are not medical ground truth. This module verifies
schema/provenance consistency, not the clinical correctness of citations.
"""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator


QueryType = Literal[
    "direct", "paraphrased", "indirect", "multi_hop",
    "comparative", "risk_factor", "ambiguous", "unsupported",
]
RelationType = Literal[
    "CAUSES", "RISK_FACTOR_FOR", "ASSOCIATED_WITH",
    "PROTECTS_AGAINST", "UNCERTAIN",
]
Split = Literal["train", "validation", "test"]
ReviewStatus = Literal["draft", "source_reviewed", "expert_reviewed"]


class EvidenceSource(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    url: HttpUrl
    source_type: Literal[
        "clinical_guideline", "systematic_review", "research_article",
        "public_health", "textbook", "other",
    ]


class GoldRelationship(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cause: str = Field(min_length=1)
    effect: str = Field(min_length=1)
    relation: RelationType
    evidence_source_ids: list[str] = Field(default_factory=list)


class GoldPath(BaseModel):
    model_config = ConfigDict(extra="forbid")

    nodes: list[str] = Field(min_length=2)
    evidence_source_ids: list[str] = Field(default_factory=list)


class MedicalBenchmarkRecord(BaseModel):
    """One question, with independently *reviewable* reference annotations."""

    model_config = ConfigDict(extra="forbid")

    query_id: str = Field(min_length=1)
    query: str = Field(min_length=1)
    query_type: QueryType
    medical_domain: Literal[
        "cardiovascular", "respiratory", "metabolic", "renal",
        "neurological", "infectious_disease", "mental_health",
        "oncology", "endocrine", "gastroenterology",
        "environmental_health", "general_medicine",
    ] | None = None
    difficulty: Literal["easy", "medium", "hard"]
    target_concept: str | None = None
    answerable: bool | None = None
    gold_answer: str | None = None
    gold_entities: list[str] = Field(default_factory=list)
    gold_causal_edges: list[GoldRelationship] = Field(default_factory=list)
    gold_paths: list[GoldPath] = Field(default_factory=list)
    evidence_sources: list[EvidenceSource] = Field(default_factory=list)
    abstention_reason: str | None = None
    review_status: ReviewStatus = "draft"
    reviewer_id: str | None = None
    split: Split | None = None
    # Used later for concept/source-aware and paraphrase-safe splits.
    question_family: str | None = None
    concept_family: str | None = None
    source_family: str | None = None

    @model_validator(mode="after")
    def check_reference_integrity(self):
        if not self.query.strip() or not self.query_id.strip():
            raise ValueError("Blank query or query_id")

        ids = [s.source_id for s in self.evidence_sources]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate evidence source ID")

        available = set(ids)
        for relation in [*self.gold_causal_edges, *self.gold_paths]:
            references = relation.evidence_source_ids
            if not set(references).issubset(available):
                raise ValueError("Unknown evidence source ID")

        if self.review_status == "draft":
            if self.split is not None:
                raise ValueError("Draft cannot be assigned to a final split")
            return self

        # These are integrity requirements, NOT proof of medical validity.
        if not self.reviewer_id or not self.reviewer_id.strip():
            raise ValueError("Reviewed annotation requires reviewer_id")
        if self.answerable is None:
            raise ValueError("Reviewed annotation requires answerable")
        if not all(
            value is not None and value.strip()
            for value in (
                self.question_family, self.concept_family, self.source_family,
            )
        ):
            raise ValueError("Reviewed annotation requires all split families")

        if self.answerable:
            if not self.gold_answer or not self.gold_answer.strip():
                raise ValueError("Answerable question requires gold_answer")
            if not self.evidence_sources:
                raise ValueError("Answerable question requires source evidence")
            if not (
                self.gold_entities or self.gold_causal_edges or self.gold_paths
            ):
                raise ValueError("Answerable question needs structured gold evidence")
        elif not self.abstention_reason or not self.abstention_reason.strip():
            raise ValueError("Unsupported question requires abstention_reason")

        for relation in [*self.gold_causal_edges, *self.gold_paths]:
            if not relation.evidence_source_ids:
                raise ValueError("Reviewed relationship/path needs evidence IDs")

        return self


def load_benchmark(path: Path) -> list[MedicalBenchmarkRecord]:
    """Read JSONL; reject malformed rows with useful line numbers."""

    records = []
    with Path(path).open(encoding="utf-8") as source:
        for number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                records.append(MedicalBenchmarkRecord.model_validate_json(line))
            except Exception as exc:
                raise ValueError(
                    f"Invalid benchmark entry at line {number}: {exc}"
                ) from exc
    if not records:
        raise ValueError("Empty benchmark")
    return records


def audit_benchmark(records: list[MedicalBenchmarkRecord], *, strict=False) -> dict:
    """Reject duplicate queries and cross-split family leakage."""

    if not records:
        raise ValueError("Empty benchmark")

    seen_ids = set()
    seen_questions = set()
    family_splits = {
        name: {} for name in (
            "question_family", "concept_family", "source_family"
        )
    }

    for record in records:
        if record.query_id in seen_ids:
            raise ValueError(f"Duplicate query_id: {record.query_id}")
        seen_ids.add(record.query_id)

        normalized = " ".join(record.query.casefold().split())
        if normalized in seen_questions:
            raise ValueError(f"Duplicate normalized question: {record.query_id}")
        seen_questions.add(normalized)

        if strict and (record.review_status == "draft" or record.split is None):
            raise ValueError(
                f"Strict benchmark needs reviewed and split labels: {record.query_id}"
            )

        if record.split is None:
            continue
        for name, groups in family_splits.items():
            value = getattr(record, name)
            if not value:
                continue
            if value in groups and groups[value] != record.split:
                raise ValueError(
                    f"Cross-split {name} leakage for {value}: "
                    f"{groups[value]} versus {record.split}"
                )
            groups[value] = record.split

    return {
        "schema_version": 1,
        "total_questions": len(records),
        "query_type_counts": dict(sorted(Counter(
            r.query_type for r in records
        ).items())),
        "medical_domain_counts": dict(sorted(Counter(
            r.medical_domain or "unspecified" for r in records
        ).items())),
        "review_status_counts": dict(sorted(Counter(
            r.review_status for r in records
        ).items())),
        "split_counts": dict(sorted(Counter(
            r.split or "unassigned" for r in records
        ).items())),
        "source_evidence_count": sum(len(r.evidence_sources) for r in records),
        "strict_review_and_split_validation": bool(strict),
        "clinical_correctness_independently_verified_by_code": False,
        "target_final_questions": 650,
        "target_split_counts": {"train": 400, "validation": 100, "test": 150},
    }


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    result = audit_benchmark(load_benchmark(args.input), strict=args.strict)
    result["dataset_sha256"] = file_sha256(args.input)
    print(json.dumps(result, indent=2))

    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
