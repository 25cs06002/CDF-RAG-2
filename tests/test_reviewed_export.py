"""Phase 2H export regression tests. All decisions here are synthetic."""

import csv
import json
import tempfile
import unittest
from pathlib import Path

from research_v2.reviewer_workspace import create_review_workspace
from research_v2.reviewed_export import (
    OUTPUT_FILES,
    build_reviewed_export,
    export_reviewed_batch,
    verify_checksums,
    sha256,
)


class TestReviewedExport(unittest.TestCase):
    def _rewrite(self, path, modify):
        with path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            fields = reader.fieldnames
            rows = list(reader)
        modify(rows)
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    def _fixture(self, tmp):
        root = Path(tmp) / "reviews"
        create_review_workspace(root)

        def review_claims(rows):
            for row in rows:
                row.update(
                    reviewer_id="SYNTHETIC_NOT_REAL",
                    reviewed_date="2026-11-14",
                    review_note="Synthetic test only, not medical evidence",
                    source_passage_locator="Synthetic test passage",
                )
                if row["task_type"] == "source_gap":
                    row["review_decision"] = "needs_more_evidence"
                elif row["task_id"] in {"EVP006:R01", "EVP006:R02"}:
                    row["review_decision"] = "needs_more_evidence"
                elif row["task_id"] in {"EVP003:R02", "EVP015:R02"}:
                    row.update(
                        review_decision="revise",
                        corrected_cause="Synthetic corrected cause " + row["task_id"],
                        corrected_effect=row["proposed_effect"],
                        corrected_relation=row["proposed_relation"],
                    )
                else:
                    row["review_decision"] = "accept"

        def review_questions(rows):
            for row in rows:
                row.update(
                    reviewer_id="SYNTHETIC_NOT_REAL",
                    reviewed_date="2026-11-14",
                    review_note="Synthetic question test only",
                )
                if row["query_id"] in {"PILOT_006", "PILOT_021"}:
                    row["answerability_decision"] = "needs_more_evidence"
                else:
                    row.update(
                        answerability_decision="answerable",
                        draft_reference_answer="Synthetic answer, not real medical QA",
                        supporting_source_ids=row["candidate_source_id"],
                    )

        self._rewrite(root / "claim_review.csv", review_claims)
        self._rewrite(root / "question_review.csv", review_questions)
        sums = root / "SHA256SUMS.txt"
        sums.write_text("".join(f"{sha256(root / name)}  {name}\n" for name in (
            "claim_review.csv", "question_review.csv", "review_manifest.json"
        )), encoding="utf-8")
        return root, sums

    def test_all_four_exports_and_warning(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, checksums = self._fixture(tmp)
            out = Path(tmp) / "export"
            profile = export_reviewed_batch(root, out, checksums)
            counts = profile["counts"]
            self.assertEqual(counts["retained_reviewed_candidate_relationships"], 22)
            self.assertEqual(counts["unresolved_or_rejected_claim_tasks"], 3)
            self.assertEqual(counts["answerable_question_drafts"], 28)
            self.assertEqual(counts["unresolved_or_unanswerable_questions"], 2)
            self.assertFalse(profile["provenance_limitations"]["medical_gold_created"])
            self.assertTrue(profile["strict_audit"]["strict_decisions_required"])
            for name in OUTPUT_FILES:
                self.assertTrue((out / name).exists())
                self.assertEqual(sha256(out / name), profile["export_sha256"][name])
            rels = [json.loads(x) for x in (out / OUTPUT_FILES[0]).read_text().splitlines()]
            r = next(x for x in rels if x["task_id"] == "EVP003:R02")
            self.assertTrue(r["revision_applied"])
            self.assertNotEqual(r["original_candidate"]["cause"], r["effective_candidate"]["cause"])
            self.assertIn("PILOT_005", profile["warnings"][0]["query_ids"])
            self.assertIn("PILOT_020", profile["warnings"][0]["query_ids"])
            answers = [json.loads(x) for x in (out / OUTPUT_FILES[2]).read_text().splitlines()]
            self.assertTrue(all(x["split"] is None for x in answers))
            self.assertTrue(all(x["data_role"] != "medical_gold" for x in answers))

    def test_unreviewed_workspace_cannot_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "review"
            create_review_workspace(root)
            out = Path(tmp) / "export"
            with self.assertRaisesRegex(ValueError, "Incomplete review decisions"):
                export_reviewed_batch(root, out)
            self.assertFalse(out.exists())

    def test_existing_output_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, _ = self._fixture(tmp)
            out = Path(tmp) / "export"
            out.mkdir()
            with self.assertRaises(FileExistsError):
                export_reviewed_batch(root, out)

    def test_checksum_mismatch_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, checksums = self._fixture(tmp)
            with (root / "question_review.csv").open("a", encoding="utf-8") as f:
                f.write("\n")
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                verify_checksums(root, checksums)
            self.assertFalse((Path(tmp) / "export").exists())

    def test_rejects_not_strict(self):
        with self.assertRaisesRegex(ValueError, "strict"):
            build_reviewed_export([], [], {}, {}, {
                "claim_tasks": 0, "question_tasks": 0,
                "pending_decisions": 0, "strict_decisions_required": False,
            })


if __name__ == "__main__":
    unittest.main()
