"""Phase 2I remaining70 review workspace tests. Synthetic review is not gold."""

import csv
import json
import tempfile
import unittest
from pathlib import Path

from research_v2.remaining_review_workspace import (
    build_remaining_review_tables,
    create_remaining_review_workspace,
    audit_remaining_review_workspace,
)


class TestRemainingReviewWorkspace(unittest.TestCase):
    def workspace(self, temp_dir):
        path = Path(temp_dir) / "remaining70_reviews"
        create_remaining_review_workspace(path)
        return path

    @staticmethod
    def update_row(path, key, identity, **changes):
        with path.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            fields = reader.fieldnames
            rows = list(reader)
        selected = [r for r in rows if r[key] == identity]
        if len(selected) != 1:
            raise AssertionError(f"Missing or duplicate fixture row {identity}")
        selected[0].update(changes)
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    @staticmethod
    def reviewed(**fields):
        return {
            "reviewer_id": "SYNTHETIC_TEST_NOT_ACTUAL_REVIEWER",
            "reviewed_date": "2026-11-14",
            "review_note": "Synthetic structural validation only, not medical verification",
            **fields,
        }

    def test_expected_table_counts_and_no_made_up_links(self):
        questions, sources, links = build_remaining_review_tables()
        self.assertEqual(len(questions), 70)
        self.assertEqual(len(sources), 89)
        self.assertEqual(len(links), 30)
        self.assertEqual(len({r["query_id"] for r in questions}), 70)
        self.assertTrue(all(not r["answerability_decision"] for r in questions))
        self.assertTrue(all(not r["review_decision"] for r in sources))
        self.assertTrue(all(not r["review_decision"] and not r["cause_node"] for r in links))
        self.assertEqual(len([r for r in questions if r["needs_additional_sources"] == "true"]), 18)

    def test_manifest_and_pending_audit(self):
        with tempfile.TemporaryDirectory() as temp:
            root = self.workspace(temp)
            manifest = json.loads((root / "review_manifest.json").read_text())
            self.assertFalse(manifest["medical_gold_created"])
            self.assertFalse(manifest["link_slots_are_proposals"])
            profile = audit_remaining_review_workspace(root)
            self.assertEqual(profile["pending_decisions"], 189)
            self.assertEqual(profile["question_decision_counts"]["pending"], 70)
            self.assertEqual(profile["source_decision_counts"]["pending"], 89)
            self.assertEqual(profile["link_decision_counts"]["pending"], 30)
            self.assertFalse(profile["causal_paths_verified_by_code"])

    def test_strict_rejects_unreviewed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = self.workspace(temp)
            with self.assertRaisesRegex(ValueError, "Incomplete remaining70"):
                audit_remaining_review_workspace(root, strict=True)

    def test_existing_output_refused(self):
        with tempfile.TemporaryDirectory() as temp:
            root = self.workspace(temp)
            with self.assertRaises(FileExistsError):
                create_remaining_review_workspace(root)

    def test_immutable_question_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = self.workspace(temp)
            self.update_row(
                root / "question_review.csv", "query_id", "PILOT_031",
                query="altered question"
            )
            with self.assertRaisesRegex(ValueError, "Original candidate field modified"):
                audit_remaining_review_workspace(root)

    def test_immutable_source_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = self.workspace(temp)
            self.update_row(
                root / "source_review.csv", "source_task_id", "PILOT_031:SRC001",
                candidate_source_url="https://example.com/changed"
            )
            with self.assertRaisesRegex(ValueError, "Original candidate field modified"):
                audit_remaining_review_workspace(root)

    def test_source_registry_hash_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = self.workspace(temp)
            manifest = root / "review_manifest.json"
            data = json.loads(manifest.read_text())
            data["source_map_sha256"] = "0" * 64
            manifest.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "provenance hash"):
                audit_remaining_review_workspace(root)

    def test_reviewed_source_and_indirect_answer_without_gold(self):
        with tempfile.TemporaryDirectory() as temp:
            root = self.workspace(temp)
            self.update_row(
                root / "source_review.csv", "source_task_id", "PILOT_031:SRC001",
                **self.reviewed(
                    review_decision="supports",
                    source_passage_locator="SYNTHETIC passage location",
                    supported_claim_summary="SYNTHETIC source summary"
                )
            )
            self.update_row(
                root / "question_review.csv", "query_id", "PILOT_031",
                **self.reviewed(
                    answerability_decision="answerable",
                    draft_reference_answer="SYNTHETIC QA reference, not gold",
                    supporting_source_ids="SRC001"
                )
            )
            profile = audit_remaining_review_workspace(root)
            self.assertEqual(profile["pending_decisions"], 187)
            self.assertEqual(profile["question_decision_counts"]["answerable"], 1)
            self.assertEqual(profile["source_decision_counts"]["supports"], 1)
            self.assertFalse(profile["gold_labels_created"])

    def test_answerable_question_rejects_unreviewed_source(self):
        with tempfile.TemporaryDirectory() as temp:
            root = self.workspace(temp)
            self.update_row(
                root / "question_review.csv", "query_id", "PILOT_031",
                **self.reviewed(
                    answerability_decision="answerable",
                    draft_reference_answer="SYNTHETIC QA reference",
                    supporting_source_ids="SRC001"
                )
            )
            with self.assertRaisesRegex(ValueError, "unreviewed/partial source"):
                audit_remaining_review_workspace(root)

    def test_unanswerable_no_source_supported(self):
        with tempfile.TemporaryDirectory() as temp:
            root = self.workspace(temp)
            self.update_row(
                root / "question_review.csv", "query_id", "PILOT_093",
                **self.reviewed(
                    answerability_decision="unanswerable",
                    abstention_reason="Cannot predict a future medical diagnosis without any patient data"
                )
            )
            profile = audit_remaining_review_workspace(root)
            self.assertEqual(profile["question_decision_counts"]["unanswerable"], 1)

    def test_unsupported_exact_person_prediction_cannot_be_answerable(self):
        with tempfile.TemporaryDirectory() as temp:
            root = self.workspace(temp)
            self.update_row(
                root / "question_review.csv", "query_id", "PILOT_093",
                **self.reviewed(
                    answerability_decision="answerable",
                    supporting_source_ids="SRC001",
                    draft_reference_answer="SYNTHETIC"
                )
            )
            with self.assertRaisesRegex(ValueError, "Unsupported exact-person prediction"):
                audit_remaining_review_workspace(root)

    def test_multi_hop_cannot_claim_complete_path_from_one_link(self):
        with tempfile.TemporaryDirectory() as temp:
            root = self.workspace(temp)
            self.update_row(
                root / "source_review.csv", "source_task_id", "PILOT_046:SRC001",
                **self.reviewed(
                    review_decision="supports",
                    source_passage_locator="SYNTHETIC",
                    supported_claim_summary="SYNTHETIC summary"
                )
            )
            self.update_row(
                root / "link_review.csv", "link_task_id", "PILOT_046:LINK1",
                **self.reviewed(
                    review_decision="supports", cause_node="smoking",
                    effect_node="plaque", relation="CAUSES",
                    supporting_source_ids="SRC001", source_passage_locator="SYNTHETIC"
                )
            )
            self.update_row(
                root / "link_review.csv", "link_task_id", "PILOT_046:LINK2",
                **self.reviewed(
                    review_decision="needs_more_evidence"
                )
            )
            self.update_row(
                root / "question_review.csv", "query_id", "PILOT_046",
                **self.reviewed(
                    answerability_decision="answerable",
                    draft_reference_answer="SYNTHETIC tentative chain",
                    supporting_source_ids="SRC001",
                    scope_resolution_note="SYNTHETIC source gap note",
                    candidate_path_status="two_links_source_supported"
                )
            )
            with self.assertRaisesRegex(ValueError, "Two-link status needs two supported"):
                audit_remaining_review_workspace(root)

    def test_multi_hop_source_supported_chain_requires_adjacency(self):
        with tempfile.TemporaryDirectory() as temp:
            root = self.workspace(temp)
            self.update_row(
                root / "source_review.csv", "source_task_id", "PILOT_046:SRC001",
                **self.reviewed(
                    review_decision="supports",
                    source_passage_locator="SYNTHETIC",
                    supported_claim_summary="SYNTHETIC summary"
                )
            )
            for slot, cause, effect in [(1, "smoking", "plaque"), (2, "plaque", "CAD")]:
                self.update_row(
                    root / "link_review.csv", "link_task_id", f"PILOT_046:LINK{slot}",
                    **self.reviewed(
                        review_decision="supports", cause_node=cause,
                        effect_node=effect, relation="CAUSES",
                        supporting_source_ids="SRC001",
                        source_passage_locator="SYNTHETIC"
                    )
                )
            self.update_row(
                root / "question_review.csv", "query_id", "PILOT_046",
                **self.reviewed(
                    answerability_decision="answerable",
                    draft_reference_answer="SYNTHETIC, not clinically verified",
                    supporting_source_ids="SRC001",
                    scope_resolution_note="SYNTHETIC source gap resolution",
                    candidate_path_status="two_links_source_supported"
                )
            )
            profile = audit_remaining_review_workspace(root)
            self.assertEqual(profile["question_decision_counts"]["answerable"], 1)
            self.assertIn("PILOT_046", profile["answerability_with_original_source_gap_flag"])
            self.assertFalse(profile["causal_paths_verified_by_code"])
            self.update_row(
                root / "link_review.csv", "link_task_id", "PILOT_046:LINK2",
                cause_node="unconnected node"
            )
            with self.assertRaisesRegex(ValueError, "do not connect"):
                audit_remaining_review_workspace(root)

    def test_comparison_needs_two_evidence_sides(self):
        with tempfile.TemporaryDirectory() as temp:
            root = self.workspace(temp)
            for source in ("SRC002", "SRC001"):
                self.update_row(
                    root / "source_review.csv", "source_task_id", f"PILOT_061:{source}",
                    **self.reviewed(
                        review_decision="supports", source_passage_locator="SYNTHETIC",
                        supported_claim_summary="SYNTHETIC"
                    )
                )
            self.update_row(
                root / "question_review.csv", "query_id", "PILOT_061",
                **self.reviewed(
                    answerability_decision="answerable",
                    supporting_source_ids="SRC002;SRC001",
                    draft_reference_answer="SYNTHETIC comparison"
                )
            )
            with self.assertRaisesRegex(ValueError, "Comparison needs both sides"):
                audit_remaining_review_workspace(root)
            self.update_row(
                root / "question_review.csv", "query_id", "PILOT_061",
                comparison_topic_a="stroke", comparison_topic_b="coronary artery disease",
                comparison_evidence_note="SYNTHETIC comparison only",
                comparison_source_ids_a="SRC002", comparison_source_ids_b="SRC001"
            )
            profile = audit_remaining_review_workspace(root)
            self.assertEqual(profile["question_decision_counts"]["answerable"], 1)

    def test_source_support_requires_passage(self):
        with tempfile.TemporaryDirectory() as temp:
            root = self.workspace(temp)
            self.update_row(
                root / "source_review.csv", "source_task_id", "PILOT_031:SRC001",
                **self.reviewed(
                    review_decision="supports", supported_claim_summary="SYNTHETIC"
                )
            )
            with self.assertRaisesRegex(ValueError, "requires passage"):
                audit_remaining_review_workspace(root)


if __name__ == "__main__":
    unittest.main()
