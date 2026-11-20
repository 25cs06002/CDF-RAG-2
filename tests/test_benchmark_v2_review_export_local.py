"""Offline review-gate and nonleaking 132-query export tests."""
import json
import tempfile
import unittest
from pathlib import Path

from research_v2.benchmark_v2_export import export
from research_v2.benchmark_v2_review_gate import analyze
from research_v2.benchmark_v2_validation import (
    ROOT,validate,write_review_queue,
)


class BenchmarkV2ReviewExportTests(unittest.TestCase):
    def test_blank_264_review_slots_cannot_be_called_complete(self):
        _,rows,sources=validate()
        with tempfile.TemporaryDirectory() as tmp:
            directory=write_review_queue(rows,sources,Path(tmp))
            report=analyze(directory/"independent_adjudication_worksheet.csv")
            self.assertEqual(report["submitted_review_rows"],264)
            self.assertEqual(report["questions_pending"],132)
            self.assertEqual(report["questions_with_formally_complete_double_review"],0)
            self.assertEqual(report["status"],"INCOMPLETE_EXTERNAL_REVIEW")
            self.assertFalse(report["clinical_gold_certified"])

    def test_pipeline_input_contains_no_target_or_draft_answers(self):
        base=ROOT/"presentation_20261118"
        base.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=base) as tmp:
            directory,manifest=export(Path(tmp))
            rows=[json.loads(line) for line in (
                directory/"questions_only_development132.jsonl").read_text(
                    encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual(len(rows),132)
            self.assertTrue(all(set(r)=={"query_id","query"} for r in rows))
            self.assertFalse(manifest["independent_holdout"])
            self.assertTrue(manifest["no_benchmark_gold_exposed_to_retriever"])


if __name__=="__main__":
    unittest.main()
