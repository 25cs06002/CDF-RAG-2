"""Synthetic-only validation tests for editable 94-source review workspace."""
import csv
import tempfile
import unittest
from pathlib import Path

from research_v2.remaining_review_workspace import create_remaining_review_workspace
from research_v2.indirect_source_extension import create as create_supp
from research_v2.combined_review_handoff import create as create_handoff
from research_v2.editable_combined_review import create, audit

class TestEditableCombinedReview(unittest.TestCase):
    def setup(self, root):
        base, supp, combined, editable = (root/n for n in ("base","supp","combined","editable"))
        create_remaining_review_workspace(base)
        create_supp(supp, base)
        path=supp/"supplemental_source_review.csv"
        with path.open(newline="",encoding="utf-8") as fh:
            reader=csv.DictReader(fh); fields=reader.fieldnames; rows=list(reader)
        for row in rows:
            row.update(review_decision="partial", source_passage_locator="SYNTHETIC",
                       supported_claim_summary="SYNTHETIC", reviewer_id="TEST_ONLY",
                       reviewed_date="2026-11-14", review_note="Synthetic test, not medical review")
        self.write(path,fields,rows)
        create_handoff(combined,base,supp)
        return base,supp,combined,editable

    @staticmethod
    def write(path, fields, rows):
        with path.open("w",newline="",encoding="utf-8") as fh:
            writer=csv.DictWriter(fh,fieldnames=fields);writer.writeheader();writer.writerows(rows)

    def modify(self, path, key, value, **fields):
        with path.open(newline="", encoding="utf-8") as fh:
            reader=csv.DictReader(fh); names=reader.fieldnames; rows=list(reader)
        selected=[row for row in rows if row[key]==value]
        self.assertEqual(len(selected),1)
        selected[0].update(fields)
        self.write(path,names,rows)

    def test_create_audit_copies_and_preserves_94_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            base,supp,combined,editable=self.setup(Path(tmp))
            profile=create(editable,combined,base,supp)
            self.assertEqual(profile["source_tasks"],94)
            self.assertEqual(profile["question_tasks"],70)
            self.assertEqual(profile["pending_decisions"],189)
            self.assertFalse(profile["gold_labels_created"])
            self.assertTrue(profile["original_handoff_unchanged"])

    def test_promote_only_with_valid_support(self):
        with tempfile.TemporaryDirectory() as tmp:
            base,supp,combined,editable=self.setup(Path(tmp))
            create(editable,combined,base,supp)
            self.modify(editable/"question_review.csv","query_id","PILOT_034",
                answerability_decision="answerable",draft_reference_answer="SYNTHETIC",
                supporting_source_ids="ADD001",reviewer_id="TEST_ONLY",reviewed_date="2026-11-14",
                review_note="Synthetic validation, not medical gold")
            with self.assertRaisesRegex(ValueError, "partial/non-supporting"):
                audit(editable,combined,base,supp)
            self.modify(editable/"source_review.csv","source_task_id","PILOT_034:ADD001",
                review_decision="supports")
            profile=audit(editable,combined,base,supp)
            self.assertEqual(profile["question_decisions"]["answerable"],1)
            self.assertFalse(profile["medical_correctness_verified_by_code"])

    def test_rejects_protected_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            base,supp,combined,editable=self.setup(Path(tmp))
            create(editable,combined,base,supp)
            self.modify(editable/"question_review.csv","query_id","PILOT_034",query="changed")
            with self.assertRaisesRegex(ValueError,"Original candidate field modified"):
                audit(editable,combined,base,supp)

    def test_rejects_original_handoff_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            base,supp,combined,editable=self.setup(Path(tmp))
            create(editable,combined,base,supp)
            with (combined/"question_review.csv").open("a") as fh: fh.write("\n")
            with self.assertRaises(ValueError):
                audit(editable,combined,base,supp)

    def test_strict_rejects_pending(self):
        with tempfile.TemporaryDirectory() as tmp:
            base,supp,combined,editable=self.setup(Path(tmp))
            create(editable,combined,base,supp)
            with self.assertRaisesRegex(ValueError,"Incomplete editable reviews"):
                audit(editable,combined,base,supp,strict=True)

if __name__ == "__main__":
    unittest.main()
