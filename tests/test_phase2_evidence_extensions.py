"""Tests for source-scope handling in Phase 2 evidence proposals.

These do not establish medical truth or reviewer approval.
"""

import json
import tempfile
import unittest
from pathlib import Path

from pydantic import ValidationError

from research_v2.evidence_proposal_review import (
    EvidenceProposal,
    build_evidence_review_pack,
)
from research_v2.source_review_queue import load_candidate_sources


ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / "data/benchmark/medical_causal_pilot100_draft_v1.jsonl"
SOURCES_V1 = ROOT / "data/benchmark/source_candidates15_v1.jsonl"
PROPOSALS_V1 = ROOT / "data/benchmark/evidence_proposals3_v1.jsonl"
SOURCES_V2 = ROOT / "data/benchmark/source_candidates15_v2.jsonl"
PROPOSALS_V2 = ROOT / "data/benchmark/evidence_proposals15_v2.jsonl"


def old_proposal(**overrides):
    data = json.loads(PROPOSALS_V1.read_text(
        encoding="utf-8"
    ).splitlines()[0])
    data.update(overrides)
    return data


class TestPhase2EvidenceExtensions(unittest.TestCase):
    def test_legacy_source_proposals_still_pass(self):
        rows, profile = build_evidence_review_pack(
            PILOT, SOURCES_V1, PROPOSALS_V1
        )
        self.assertEqual(len(rows), 3)
        self.assertEqual(profile["questions_with_candidate_relations"], 6)
        self.assertEqual(profile["questions_needing_additional_sources"], 0)

    def test_insufficient_source_proposal_is_allowed(self):
        record = EvidenceProposal.model_validate(old_proposal(
            proposed_relationships=[],
            proposal_outcome="insufficient_for_question",
            unresolved_scope_issue=(
                "A page discussing asthma attacks does not establish "
                "causes of initial asthma disease onset."
            ),
        ))
        self.assertEqual(record.proposal_outcome, "insufficient_for_question")
        self.assertFalse(record.proposed_relationships)

    def test_cannot_claim_candidate_outcome_without_relationships(self):
        with self.assertRaisesRegex(
            ValidationError, "requires relationships"
        ):
            EvidenceProposal.model_validate(old_proposal(
                proposed_relationships=[]
            ))

    def test_insufficient_outcome_cannot_contain_relationships(self):
        with self.assertRaisesRegex(
            ValidationError, "cannot include relationships"
        ):
            EvidenceProposal.model_validate(old_proposal(
                proposal_outcome="insufficient_for_question"
            ))

    def test_insufficient_source_requires_explicit_scope_reason(self):
        with self.assertRaisesRegex(
            ValidationError, "requires unresolved_scope_issue"
        ):
            EvidenceProposal.model_validate(old_proposal(
                proposed_relationships=[],
                proposal_outcome="insufficient_for_question",
                unresolved_scope_issue=None,
            ))

    def test_pack_reports_unresolved_question_counts(self):
        records = [
            json.loads(line) for line in PROPOSALS_V1.read_text(
                encoding="utf-8"
            ).splitlines()
        ]
        records[0]["proposal_outcome"] = "insufficient_for_question"
        records[0]["proposed_relationships"] = []
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "proposals.jsonl"
            path.write_text(
                "".join(json.dumps(item) + "\n" for item in records),
                encoding="utf-8",
            )
            rows, profile = build_evidence_review_pack(
                PILOT, SOURCES_V1, path
            )
        self.assertEqual(len(rows), 3)
        self.assertEqual(profile["questions_needing_additional_sources"], 2)
        self.assertEqual(profile["questions_with_candidate_relations"], 4)
        self.assertEqual(profile["proposal_outcome_counts"], {
            "candidate_relations": 2,
            "insufficient_for_question": 1,
        })

    def test_expanded_15_topic_review_pack(self):
        rows, profile = build_evidence_review_pack(
            PILOT, SOURCES_V2, PROPOSALS_V2
        )
        self.assertEqual(len(rows), 15)
        self.assertEqual(profile["question_count"], 30)
        self.assertEqual(profile["unique_candidate_sources"], 15)
        self.assertEqual(profile["questions_with_candidate_relations"], 28)
        self.assertEqual(profile["questions_needing_additional_sources"], 2)
        self.assertEqual(profile["relationship_type_counts"], {
            "ASSOCIATED_WITH": 1,
            "CAUSES": 12,
            "RISK_FACTOR_FOR": 11,
        })
        self.assertEqual(profile["proposal_outcome_counts"], {
            "candidate_relations": 14,
            "insufficient_for_question": 1,
        })
        self.assertTrue(all(
            not item["human_review_completed"]
            and not item["gold_annotation_created"]
            for item in rows
        ))

    def test_unavailable_pages_replaced_in_v2_without_editing_v1(self):
        first = {
            item.source_id: str(item.url)
            for item in load_candidate_sources(SOURCES_V1)
        }
        second = {
            item.source_id: str(item.url)
            for item in load_candidate_sources(SOURCES_V2)
        }
        changed = {
            key for key in first
            if first[key] != second[key]
        }
        self.assertEqual(changed, {"SRC010", "SRC013"})
        self.assertEqual(
            second["SRC010"], "https://medlineplus.gov/epilepsy.html"
        )
        self.assertEqual(
            second["SRC013"], "https://medlineplus.gov/depression.html"
        )


if __name__ == "__main__":
    unittest.main()
