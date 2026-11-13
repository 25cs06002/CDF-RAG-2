"""Validate candidate relationships from checked medical webpages.

Evidence proposals are *not* medical gold and are *not* human-reviewed.
"""

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator

from .medical_benchmark import load_benchmark
from .source_review_queue import load_candidate_sources


Relation = Literal["CAUSES", "RISK_FACTOR_FOR", "ASSOCIATED_WITH", "UNCERTAIN"]


class EvidenceProposal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    proposal_id: str = Field(min_length=1)
    target_concept: str = Field(min_length=1)
    query_ids: list[str] = Field(min_length=2, max_length=2)
    candidate_source_id: str = Field(min_length=1)
    source_section: str = Field(min_length=1)
    source_checked_date: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")
    proposed_relationships: list["ProposedRelationship"] = Field(min_length=1)
    unresolved_scope_issue: str | None = None
    provenance_status: Literal["ai_source_triage_needs_independent_review"]
    human_review_completed: Literal[False]
    gold_annotation_created: Literal[False]

    @model_validator(mode="after")
    def no_duplicate_query_ids(self):
        if len(set(self.query_ids)) != len(self.query_ids):
            raise ValueError("Duplicate query ID within proposal")
        return self


class ProposedRelationship(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cause: str = Field(min_length=1)
    effect: str = Field(min_length=1)
    relation: Relation
    source_support_summary: str = Field(min_length=15)
    source_section: str = Field(min_length=1)
    review_question: str = Field(min_length=15)


def load_proposals(path: Path) -> list[EvidenceProposal]:
    proposals = []
    with Path(path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                proposals.append(EvidenceProposal.model_validate_json(line))
            except Exception as exc:
                raise ValueError(f"Invalid evidence proposal at line {number}: {exc}") from exc
    if not proposals:
        raise ValueError("No evidence proposals")
    return proposals


def build_evidence_review_pack(pilot_file: Path, sources_file: Path, proposals_file: Path):
    pilot = load_benchmark(pilot_file)
    sources = load_candidate_sources(sources_file)
    proposals = load_proposals(proposals_file)
    queries = {item.query_id: item for item in pilot}
    source_map = {src.source_id: src for src in sources}
    if len(queries) != len(pilot):
        raise ValueError("Duplicate query IDs in pilot")

    seen_proposals = set()
    seen_questions = set()
    reviewed_rows = []
    for proposal in proposals:
        if proposal.proposal_id in seen_proposals:
            raise ValueError(f"Duplicate proposal ID: {proposal.proposal_id}")
        seen_proposals.add(proposal.proposal_id)

        source = source_map.get(proposal.candidate_source_id)
        if source is None:
            raise ValueError(f"Unknown candidate source: {proposal.candidate_source_id}")
        if source.target_concept.casefold() != proposal.target_concept.casefold():
            raise ValueError(f"Source topic mismatch: {proposal.proposal_id}")

        selected = []
        for query_id in proposal.query_ids:
            if query_id in seen_questions:
                raise ValueError(f"Query appears in multiple proposals: {query_id}")
            item = queries.get(query_id)
            if item is None:
                raise ValueError(f"Unknown query ID: {query_id}")
            if item.target_concept != proposal.target_concept:
                raise ValueError(f"Query topic mismatch: {query_id}")
            if item.medical_domain != source.medical_domain:
                raise ValueError(f"Query domain mismatch: {query_id}")
            if item.review_status != "draft" or item.split is not None:
                raise ValueError(f"Proposal query must be unsplit draft: {query_id}")
            selected.append(item)
            seen_questions.add(query_id)
        if {item.query_type for item in selected} != {"direct", "paraphrased"}:
            raise ValueError(f"Need direct and paraphrased pair: {proposal.proposal_id}")
        if len({item.question_family for item in selected}) != 1:
            raise ValueError(f"Mismatched paraphrase family: {proposal.proposal_id}")

        reviewed_rows.append({
            "proposal_id": proposal.proposal_id,
            "query_ids": proposal.query_ids,
            "candidate_source": source.model_dump(mode="json"),
            "checked_date": proposal.source_checked_date,
            "source_section": proposal.source_section,
            "proposed_relationships": [
                item.model_dump() for item in proposal.proposed_relationships
            ],
            "unresolved_scope_issue": proposal.unresolved_scope_issue,
            "review_steps": [
                "Verify the cited page and its medical scope",
                "Verify each relation type and each claimed direction",
                "Seek additional supporting references where necessary",
                "Mark actual reviewer and source details in a NEW annotation version",
                "Do not promote proposals to gold without independent review",
            ],
            "human_review_completed": False,
            "gold_annotation_created": False,
        })

    return reviewed_rows, {
        "data_role": "source_checked_ai_proposals_for_independent_review",
        "proposal_count": len(reviewed_rows),
        "question_count": len(seen_questions),
        "unique_candidate_sources": len({p.candidate_source_id for p in proposals}),
        "relationship_type_counts": dict(sorted(Counter(
            rel.relation for p in proposals for rel in p.proposed_relationships
        ).items())),
        "human_review_completed": False,
        "gold_annotation_created": False,
        "clinical_accuracy_established": False,
    }


def write_evidence_review_pack(
    pilot_file: Path, sources_file: Path, proposals_file: Path, output_dir: Path
):
    rows, profile = build_evidence_review_pack(pilot_file, sources_file, proposals_file)
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=False)
    (root / "evidence_review_worklist.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    (root / "evidence_review_profile.json").write_text(
        json.dumps(profile, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return profile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pilot", type=Path,
        default=Path("data/benchmark/medical_causal_pilot100_draft_v1.jsonl"),
    )
    parser.add_argument(
        "--sources", type=Path,
        default=Path("data/benchmark/source_candidates15_v1.jsonl"),
    )
    parser.add_argument(
        "--proposals", type=Path,
        default=Path("data/benchmark/evidence_proposals3_v1.jsonl"),
    )
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.output_dir is None:
        _, report = build_evidence_review_pack(args.pilot, args.sources, args.proposals)
    else:
        report = write_evidence_review_pack(
            args.pilot, args.sources, args.proposals, args.output_dir
        )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
