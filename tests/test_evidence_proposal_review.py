"""Evidence proposals must remain separate from verified medical gold."""

import json
import tempfile
import unittest
from pathlib import Path

from research_v2.evidence_proposal_review import (
    build_evidence_review_pack,
    load_proposals,
    write_evidence_review_pack,
)

ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / "data/benchmark/medical_causal_pilot100_draft_v1.jsonl"
SOURCES = ROOT / "data/benchmark/source_candidates15_v1.jsonl"
PROPOSALS = ROOT / "data/benchmark/evidence_proposals3_v1.jsonl"


class TestEvidenceProposalReview(unittest.TestCase):
    def rows(self):
        return [
            json.loads(line) for line in PROPOSALS.read_text(
                encoding="utf-8"
            ).splitlines() if line.strip()
        ]

    def replacement_file(self, tmp, rows):
        path = Path(tmp) / "proposals.jsonl"
        path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )
        return path

    def test_three_source_checked_topics_six_queries(self):
        rows, profile = build_evidence_review_pack(PILOT, SOURCES, PROPOSALS)
        self.assertEqual(len(rows), 3)
        self.assertEqual(profile["question_count"], 6)
        self.assertEqual(profile["unique_candidate_sources"], 3)
        self.assertEqual(profile["relationship_type_counts"], {
            "CAUSES": 2, "RISK_FACTOR_FOR": 4,
        })

    def test_no_gold_or_human_review_claim(self):
        rows, profile = build_evidence_review_pack(PILOT, SOURCES, PROPOSALS)
        self.assertFalse(profile["human_review_completed"])
        self.assertFalse(profile["gold_annotation_created"])
        self.assertFalse(profile["clinical_accuracy_established"])
        for item in rows:
            self.assertFalse(item["human_review_completed"])
            self.assertFalse(item["gold_annotation_created"])
            self.assertTrue(item["review_steps"])

    def test_unknown_candidate_source_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = self.rows()
            rows[0]["candidate_source_id"] = "MISSING"
            with self.assertRaisesRegex(ValueError, "Unknown candidate source"):
                build_evidence_review_pack(
                    PILOT, SOURCES, self.replacement_file(tmp, rows)
                )

    def test_mismatched_source_topic_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = self.rows()
            rows[0]["candidate_source_id"] = "SRC004"
            with self.assertRaisesRegex(ValueError, "Source topic mismatch"):
                build_evidence_review_pack(
                    PILOT, SOURCES, self.replacement_file(tmp, rows)
                )

    def test_unknown_question_id_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = self.rows()
            rows[0]["query_ids"][1] = "UNKNOWN"
            with self.assertRaisesRegex(ValueError, "Unknown query ID"):
                build_evidence_review_pack(
                    PILOT, SOURCES, self.replacement_file(tmp, rows)
                )

    def test_pair_must_include_direct_and_paraphrase(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = self.rows()
            rows[0]["query_ids"][1] = "PILOT_002"
            with self.assertRaisesRegex(ValueError, "Query topic mismatch"):
                build_evidence_review_pack(
                    PILOT, SOURCES, self.replacement_file(tmp, rows)
                )

    def test_duplicate_proposal_ids_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = self.rows()
            rows[1]["proposal_id"] = rows[0]["proposal_id"]
            with self.assertRaisesRegex(ValueError, "Duplicate proposal ID"):
                build_evidence_review_pack(
                    PILOT, SOURCES, self.replacement_file(tmp, rows)
                )

    def test_invalid_medical_relation_type_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = self.rows()
            rows[0]["proposed_relationships"][0]["relation"] = "PROVEN"
            with self.assertRaisesRegex(ValueError, "line 1"):
                load_proposals(self.replacement_file(tmp, rows))

    def test_cannot_claim_human_review_completed(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = self.rows()
            rows[0]["human_review_completed"] = True
            with self.assertRaisesRegex(ValueError, "line 1"):
                load_proposals(self.replacement_file(tmp, rows))

    def test_pack_export_and_non_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "review"
            profile = write_evidence_review_pack(
                PILOT, SOURCES, PROPOSALS, dest
            )
            self.assertEqual(profile["proposal_count"], 3)
            self.assertTrue((dest / "evidence_review_worklist.jsonl").exists())
            self.assertTrue((dest / "evidence_review_profile.json").exists())
            with self.assertRaises(FileExistsError):
                write_evidence_review_pack(PILOT, SOURCES, PROPOSALS, dest)


if __name__ == "__main__":
    unittest.main()
