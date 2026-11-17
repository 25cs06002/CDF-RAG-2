"""Focused tests for 48h partial-submission helper; no expensive experiments."""
import unittest

from research_v2.partial_submission_sprint import _measured, _table, MODEL32
from research_v2.schema import PipelineRecord, StageTimings, EvidenceDocument


def record(model, query, status, has_evidence):
    return PipelineRecord(
        run_id="TEST",
        query_id=query,
        pipeline_id="bm25_rag",
        original_query="Synthetic query for testing",
        retrieved_documents=(
            [EvidenceDocument(document_id="TEST_DOC",text="Testing",source_id="TEST")]
            if has_evidence else []
        ),
        status=status,
        timings=StageTimings(retrieval_ms=2.0,total_ms=3.0),
    )


class SubmissionHelperTests(unittest.TestCase):
    def test_archival_errors_are_not_successes(self):
        records=[record("B2","Q1","error",False),
                 record("B2","Q2","ok",True),
                 record("B2","Q3","abstained",False)]
        metrics=_measured(records,"B2","retrieval_only","test_corpus")
        self.assertEqual(metrics["status_error"],1)
        self.assertEqual(metrics["status_ok"],1)
        self.assertEqual(metrics["status_abstained"],1)
        self.assertEqual(metrics["evidence_returned"],1)
        self.assertEqual(metrics["mean_retrieval_ms"],2.0)
        self.assertEqual(metrics["n"],3)

    def test_report_table_keeps_zero_errors_explicit(self):
        rendered=_table([{"model":"B2","status_error":100}],
                        ("model","status_error"))
        self.assertIn("B2",rendered)
        self.assertIn("100",rendered)
        self.assertIn("status_error",rendered)

    def test_only_six_new_baselines(self):
        self.assertEqual(MODEL32,("B1","B2","B3","B4","B5","B6"))


if __name__=="__main__":
    unittest.main()
