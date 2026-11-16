"""Offline regression tests: G0 gap classification never creates evidence."""
import unittest

from research_v2.g0_gap_audit import classify, normalize
from research_v2.pipelines.baselines import G0Corpus


class AuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.graph = G0Corpus()

    def row(self, **kwargs):
        row = {
            "query_id": "PILOT_001", "retrieval_review_bucket": "G0_SNAPSHOT_COVERAGE_GAP",
            "target_concept_draft": "coronary artery disease",
            "question": "What causes coronary artery disease?",
            "query_type": "direct",
            "B5_evidence": "False", "B6_evidence": "False",
            "B7_evidence": "False", "B8_evidence": "True",
        }
        row.update(kwargs)
        return row

    def test_absent_target_requires_review_not_synonym_inference(self):
        finding = classify(self.row(), self.graph)
        self.assertFalse(finding["target_in_g0_exact"])
        self.assertEqual(finding["review_category"],
                         "TARGET_NOT_PRESENT_IN_G0_REQUIRES_ENTITY_REVIEW")
        self.assertEqual(finding["reviewer_decision"], "")
        self.assertIn("concept_candidates_review_only", finding)

    def test_exact_target_and_missing_direction(self):
        finding = classify(self.row(
            query_id="PILOT_047", target_concept_draft="stroke",
            question="How might high blood pressure cause stroke?",
            retrieval_review_bucket="DIRECTED_RETRIEVAL_GAP",
            B5_evidence="True",
        ), self.graph)
        self.assertTrue(finding["target_in_g0_exact"])
        self.assertGreater(finding["target_in_degree"], 0)
        self.assertIn(finding["review_category"],
                      {"MATCHED_DIFFERENT_G0_CONCEPT",
                       "MATCHED_TARGET_BUT_NO_DIRECTED_CAUSAL_EVIDENCE"})

    def test_no_bucket_excluded(self):
        self.assertIsNone(classify(self.row(
            retrieval_review_bucket="BOTH_GRAPH_METHODS_RETURN_EVIDENCE"
        ), self.graph))

    def test_normalizer_does_not_equate_related_diseases(self):
        self.assertNotEqual(normalize("coronary artery disease"),
                            normalize("heart disease"))


if __name__ == "__main__":
    unittest.main()
