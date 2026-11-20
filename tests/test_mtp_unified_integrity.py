"""Canonical MTP cross-branch provenance regression tests.

These tests check archived scientific measurements and honesty of provenance,
not medical correctness.
"""
import unittest
from pathlib import Path
from research_v2.unified_integrity import ROOT, REGISTRY, verify


class UnifiedExperimentRegistryTests(unittest.TestCase):
    def test_22_100_132_scopes_and_provenance_pass(self):
        out=verify()
        self.assertEqual(out["status"],"PASS_MTP_UNIFIED_PROVENANCE")
        self.assertEqual(out["legacy22_metrics"],16)
        self.assertEqual(out["archived_end_to_end_models"],8)
        self.assertEqual(out["selected_archive_errors"],0)
        self.assertEqual(out["locally_audited_unimported_retrieval_records"],792)
        self.assertEqual(out["independently_certified_medical_gold_questions"],0)
        self.assertFalse(out["cross_cohort_accuracy_pooling_permitted"])

    def test_original_upstream_credit_is_kept(self):
        upstream=(ROOT/"docs/ORIGINAL_CDF_RAG_README.md").read_text(encoding="utf-8")
        self.assertIn("Elahe Khatibi",upstream)
        self.assertIn("Ziyu Wang",upstream)
        self.assertIn("Amir M. Rahmani",upstream)
        self.assertTrue((ROOT/"research_artifacts/legacy22/scripts/temp_hallucination_report.py").is_file())

    def test_readme_is_current_and_gold_claims_are_not_inflated(self):
        source=(ROOT/"readme.md").read_text(encoding="utf-8")
        self.assertIn("Canonical branch: master",source)
        self.assertIn("132-query",source)
        self.assertIn("NOT yet",source)
        self.assertNotIn("100-query run pending",source)


if __name__=="__main__":
    unittest.main()
