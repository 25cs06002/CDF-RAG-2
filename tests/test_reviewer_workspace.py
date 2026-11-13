"""Review worksheet tests. Synthetic decisions are NOT medical verification."""

import csv
import json
import tempfile
import unittest
from pathlib import Path

from research_v2.reviewer_workspace import (
    audit_review_workspace,
    build_review_tables,
    create_review_workspace,
)


class TestReviewerWorkspace(unittest.TestCase):
    def workspace(self, tmp):
        output = Path(tmp) / "review"
        create_review_workspace(output)
        return output

    def modify_csv(self, path, mutate):
        with path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            fields = reader.fieldnames
            records = list(reader)
        mutate(records)
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(records)

    def test_task_counts_and_source_gap(self):
        claims, questions = build_review_tables()
        self.assertEqual(len(claims), 25)
        self.assertEqual(len(questions), 30)
        gaps = [r for r in claims if r["task_type"] == "source_gap"]
        self.assertEqual(len(gaps), 1)
        self.assertEqual(gaps[0]["task_id"], "EVP007:GAP")

    def test_all_decision_fields_start_empty(self):
        claims, questions = build_review_tables()
        self.assertTrue(all(not r["review_decision"] for r in claims))
        self.assertTrue(all(not r["answerability_decision"] for r in questions))

    def test_workspace_outputs_and_reports_pending(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = self.workspace(tmp)
            profile = audit_review_workspace(workspace)
            self.assertEqual(profile["claim_tasks"], 25)
            self.assertEqual(profile["question_tasks"], 30)
            self.assertEqual(profile["pending_decisions"], 55)
            self.assertFalse(profile["gold_labels_created"])
            self.assertFalse(profile["reviewer_identity_verified_by_code"])

    def test_strict_audit_fails_on_pending(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = self.workspace(tmp)
            with self.assertRaisesRegex(ValueError, "Incomplete review decisions"):
                audit_review_workspace(workspace, strict=True)

    def test_cannot_overwrite_workspace(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = self.workspace(tmp)
            with self.assertRaises(FileExistsError):
                create_review_workspace(workspace)

    def test_immutable_claim_fields_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = self.workspace(tmp)
            self.modify_csv(
                workspace / "claim_review.csv",
                lambda rows: rows[0].update(proposed_relation="CAUSES_CHANGED"),
            )
            with self.assertRaisesRegex(ValueError, "Original candidate field modified"):
                audit_review_workspace(workspace)

    def test_immutable_question_fields_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = self.workspace(tmp)
            self.modify_csv(
                workspace / "question_review.csv",
                lambda rows: rows[0].update(query="tampered query"),
            )
            with self.assertRaisesRegex(ValueError, "Original candidate field modified"):
                audit_review_workspace(workspace)

    def test_duplicate_task_id_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = self.workspace(tmp)
            self.modify_csv(
                workspace / "claim_review.csv",
                lambda rows: rows[1].update(task_id=rows[0]["task_id"]),
            )
            with self.assertRaisesRegex(ValueError, "Duplicate or unknown worksheet ID"):
                audit_review_workspace(workspace)

    def test_source_gap_cannot_be_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = self.workspace(tmp)

            def decide(rows):
                gap = next(x for x in rows if x["task_type"] == "source_gap")
                gap.update(
                    review_decision="accept",
                    reviewer_id="synthetic_id",
                    reviewed_date="2026-11-13",
                    review_note="Synthetic test decision only",
                )

            self.modify_csv(workspace / "claim_review.csv", decide)
            with self.assertRaisesRegex(ValueError, "Cannot accept a source gap"):
                audit_review_workspace(workspace)

    def test_valid_synthetic_rejection_does_not_create_gold(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = self.workspace(tmp)

            def decide(rows):
                rows[0].update(
                    review_decision="reject",
                    reviewer_id="synthetic_id",
                    reviewed_date="2026-11-13",
                    review_note="Synthetic test: the claim is rejected.",
                )

            self.modify_csv(workspace / "claim_review.csv", decide)
            profile = audit_review_workspace(workspace)
            self.assertEqual(profile["claim_decision_counts"]["reject"], 1)
            self.assertEqual(profile["pending_decisions"], 54)
            self.assertFalse(profile["gold_labels_created"])

    def test_claim_accept_requires_source_locator(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = self.workspace(tmp)

            def decide(rows):
                rows[0].update(
                    review_decision="accept",
                    reviewer_id="synthetic_id",
                    reviewed_date="2026-11-13",
                    review_note="Synthetic test only",
                )

            self.modify_csv(workspace / "claim_review.csv", decide)
            with self.assertRaisesRegex(ValueError, "Missing evidence passage locator"):
                audit_review_workspace(workspace)

    def test_question_answerable_requires_reference_answer(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = self.workspace(tmp)

            def decide(rows):
                rows[0].update(
                    answerability_decision="answerable",
                    reviewer_id="synthetic_id",
                    reviewed_date="2026-11-13",
                    review_note="Synthetic test only",
                    supporting_source_ids=rows[0]["candidate_source_id"],
                )

            self.modify_csv(workspace / "question_review.csv", decide)
            with self.assertRaisesRegex(ValueError, "lacks reference answer"):
                audit_review_workspace(workspace)

    def test_question_cannot_cite_another_topic_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = self.workspace(tmp)

            def decide(rows):
                rows[0].update(
                    answerability_decision="answerable",
                    draft_reference_answer="Synthetic reference response.",
                    reviewer_id="synthetic_id",
                    reviewed_date="2026-11-13",
                    review_note="Synthetic test only",
                    supporting_source_ids="SRC004",
                )

            self.modify_csv(workspace / "question_review.csv", decide)
            with self.assertRaisesRegex(ValueError, "unrelated reference source IDs"):
                audit_review_workspace(workspace)

    def test_changed_input_manifest_sha_blocks_review(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = self.workspace(tmp)
            manifest_path = workspace / "review_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["pilot_sha256"] = "0" * 64
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Input provenance hash changed"):
                audit_review_workspace(workspace)

    def test_revamped_claim_requires_valid_relation(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = self.workspace(tmp)

            def decide(rows):
                rows[0].update(
                    review_decision="revise",
                    corrected_cause="synthetic cause",
                    corrected_effect="synthetic effect",
                    corrected_relation="UNRECOGNIZED_RELATION",
                    source_passage_locator="Section 1, synthetic locator",
                    reviewer_id="synthetic_id",
                    reviewed_date="2026-11-13",
                    review_note="Synthetic revision only",
                )

            self.modify_csv(workspace / "claim_review.csv", decide)
            with self.assertRaisesRegex(ValueError, "valid relation"):
                audit_review_workspace(workspace)


if __name__ == "__main__":
    unittest.main()
