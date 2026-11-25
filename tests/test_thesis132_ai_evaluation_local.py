import unittest
from research_v2.thesis132_ai_evaluation import classify, aggregate, VERDICTS, indexed

class AIReferenceTests(unittest.TestCase):
    def row(self,label,flag,substantive=True):
        return dict(label=label,detector_flag=flag,substantive=substantive,
                    grounding="FULLY_GROUNDED" if substantive else "NO_FACTUAL_CLAIMS",
                    confidence="high",fabricated_citation=False,claims_without_evidence_ids=0,
                    **{v:int(v=="SUPPORTED" and substantive) for v in VERDICTS})
    def test_classifier_excludes_uncertain_and_missing(self):
        r=classify([self.row("Yes",True),self.row("No",False),self.row("No",True),
                    self.row("Yes",False),self.row("Uncertain",True),self.row("No",None)])
        self.assertEqual(r["classification_support"],4)
        self.assertEqual([r[k] for k in ("tn","fp","fn","tp")],[1,1,1,1])
        self.assertEqual(r["f1_vs_ai"],.5)
    def test_no_predictions_is_na(self):
        self.assertIsNone(classify([self.row("No",None)])["accuracy_vs_ai"])
    def test_abstention_denominator(self):
        r=aggregate([self.row("Yes",True),self.row("No",False,False)])
        self.assertEqual(r["hallucination_rate_ai"],.5)
        self.assertEqual(r["hallucination_rate_substantive_ai"],1)
    def test_duplicate_rejected(self):
        with self.assertRaises(ValueError): indexed([{"model":"B1","query_id":"a"}]*2)
