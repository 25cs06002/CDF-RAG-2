"""Focused, in-memory tests for compact task reconciliation; no sprint jobs."""
import unittest

from research_v2.g1_compact_reconcile import repair_rows, identity


FIELDS=("task_id","query_id","question","case_ids","review_query_intent",
        "review_stable_target","reviewer","review_date","review_note")
REVIEW=("review_query_intent","review_stable_target","reviewer","review_date","review_note")


def canonical():
    return [
        {"task_id":"C_a1","query_id":"PILOT_001","question":"Question one",
         "case_ids":"B5_PILOT_001",**{name:"" for name in REVIEW}},
        {"task_id":"C_b2","query_id":"PILOT_002","question":"Question two",
         "case_ids":"B6_PILOT_002",**{name:"" for name in REVIEW}},
    ]


def reviewed(row):
    return {**row,"review_query_intent":"answerable",
            "review_stable_target":"appropriate","reviewer":"STU_001",
            "review_date":"2026-11-17","review_note":"Evidence reviewer attestation."}


class ReconcileTests(unittest.TestCase):
    def test_restores_changed_id_with_immutable_identity(self):
        source=canonical()
        edited=[reviewed({**source[0],"task_id":"manually_changed"}),source[1]]
        out,repairs,missing=repair_rows(FIELDS,REVIEW,source,edited,False,"cases")
        self.assertEqual(len(repairs),1)
        self.assertEqual(missing,[])
        self.assertEqual(out[0]["task_id"],"C_a1")
        self.assertEqual(out[0]["review_note"],"Evidence reviewer attestation.")

    def test_refuses_missing_task_without_opt_in(self):
        source=canonical()
        with self.assertRaisesRegex(ValueError,"tasks missing"):
            repair_rows(FIELDS,REVIEW,source,[reviewed(source[0])],False,"cases")

    def test_restores_missing_as_blank_only_when_requested(self):
        source=canonical()
        out,_,missing=repair_rows(FIELDS,REVIEW,source,[reviewed(source[0])],True,"cases")
        self.assertEqual(missing,["C_b2"])
        self.assertEqual(out[0]["review_query_intent"],"answerable")
        self.assertEqual(out[1]["review_query_intent"],"")

    def test_rejects_edited_immutable_question_and_preserves_review(self):
        source=canonical()
        edited=[reviewed({**source[0],"question":"CHANGED"}),source[1]]
        with self.assertRaisesRegex(ValueError,"immutable fields changed"):
            repair_rows(FIELDS,REVIEW,source,edited,False,"cases")

    def test_rejects_extra_unmatchable_row(self):
        source=canonical()
        edited=[*source,{"task_id":"X","query_id":"UNKNOWN","question":"Other",
                         "case_ids":"X",**{field:"" for field in REVIEW}}]
        with self.assertRaisesRegex(ValueError,"unknown task_id"):
            repair_rows(FIELDS,REVIEW,source,edited,False,"cases")

    def test_rejects_duplicate_rows(self):
        source=canonical()
        with self.assertRaisesRegex(ValueError,"duplicate or conflicting"):
            repair_rows(FIELDS,REVIEW,source,[source[0],source[0],source[1]],False,"cases")


if __name__=="__main__":
    unittest.main()
