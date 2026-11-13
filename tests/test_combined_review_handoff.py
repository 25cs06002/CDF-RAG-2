"""Phase 2I-C2 combination tests; synthetic reviewer records, not medical gold."""
import csv
import tempfile
import unittest
from pathlib import Path

from research_v2.remaining_review_workspace import create_remaining_review_workspace
from research_v2.indirect_source_extension import create as create_supplement
from research_v2.combined_review_handoff import create, audit

class TestCombinedReviewHandoff(unittest.TestCase):
    def setup(self, temp):
        base, supplement, combined = (Path(temp) / n for n in ("base", "supp", "combined"))
        create_remaining_review_workspace(base)
        create_supplement(supplement, base)
        path = supplement / "supplemental_source_review.csv"
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f); fields=reader.fieldnames; rows=list(reader)
        for row in rows:
            row.update(
                review_decision="partial",
                source_passage_locator="SYNTHETIC LOCATION",
                supported_claim_summary="SYNTHETIC CLAIM",
                reviewer_id="SYNTHETIC_TEST",
                reviewed_date="2026-11-14",
                review_note="Only tests structural integrity; not a real medical review",
            )
        with path.open("w", newline="", encoding="utf-8") as f:
            writer=csv.DictWriter(f, fieldnames=fields);writer.writeheader();writer.writerows(rows)
        return base, supplement, combined

    def test_combined_count_and_decisions(self):
        with tempfile.TemporaryDirectory() as tmp:
            base, supplement, combined = self.setup(tmp)
            profile=create(combined, base, supplement)
            self.assertEqual(profile["source_tasks"], 94)
            self.assertEqual(profile["new_source_decisions"]["partial"], 5)
            self.assertFalse(profile["medical_gold_created"])
            self.assertFalse(profile["combined_snapshot_editable"])
            self.assertEqual(audit(combined,base,supplement)["status"],"audit_passed")

    def test_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            base, supplement, combined = self.setup(tmp)
            create(combined,base,supplement)
            with self.assertRaises(FileExistsError):
                create(combined,base,supplement)

    def test_rejects_mutated_combined_answer(self):
        with tempfile.TemporaryDirectory() as tmp:
            base, supplement, combined = self.setup(tmp)
            create(combined,base,supplement)
            path=combined/"question_review.csv"
            content=path.read_text()
            path.write_text(content.replace("PILOT_031","PILOT_999",1))
            with self.assertRaisesRegex(ValueError,"differs"):
                audit(combined,base,supplement)

    def test_rejects_changed_supplement(self):
        with tempfile.TemporaryDirectory() as tmp:
            base, supplement, combined = self.setup(tmp)
            create(combined,base,supplement)
            path=supplement/"supplemental_source_review.csv"
            path.write_text(path.read_text()+"\n")
            with self.assertRaises(ValueError):
                audit(combined,base,supplement)

if __name__ == "__main__":
    unittest.main()
