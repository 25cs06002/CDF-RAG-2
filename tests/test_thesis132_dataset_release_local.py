"""B1–B9 pinned 132-query release; no medical correctness claims."""
import json
import tempfile
import unittest
from pathlib import Path

from research_v2.thesis132_dataset_release import (
    RELEASE, verify_release, verify_model_records,
)


class Release132Tests(unittest.TestCase):
    def test_complete_and_no_answer_leakage(self):
        report = verify_release()
        self.assertEqual(report["query_count"], 132)
        self.assertEqual(report["query_sha256"],
                         "a45fcddfc153c0f9c2211c8167e001b03513216bf9191f2f08a4444d52df6c76")
        self.assertEqual(report["fold_counts"], [27,27,26,26,26])
        self.assertFalse(report["independent_medical_gold"])
        self.assertFalse(report["historically_unseen_holdout"])
        self.assertEqual(len(report["pipelines"]), 9)

    def _mock_records(self):
        rows=[json.loads(x) for x in (RELEASE/"questions_only.jsonl").read_text(
            encoding="utf-8").splitlines() if x.strip()]
        return [{"query_id":x["query_id"], "original_query":x["query"],
                 "status":"ok"} for x in rows]

    def _validate_records(self, records, rejected=None):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"records.jsonl"
            path.write_text("".join(json.dumps(r)+"\n" for r in records),
                            encoding="utf-8")
            if rejected:
                with self.assertRaisesRegex(ValueError, rejected):
                    verify_model_records(path, "B9")
            else:
                self.assertEqual(verify_model_records(path,"B9")["records"],132)

    def test_synthetic_complete_records_pass(self):
        self._validate_records(self._mock_records())

    def test_reject_missing_or_reordered_question(self):
        rows=self._mock_records()
        self._validate_records(rows[:-1], "Missing or extra")
        rows=self._mock_records()
        rows[0],rows[1]=rows[1],rows[0]
        self._validate_records(rows, "ID/order mismatch")

    def test_reject_changed_query_or_execution_error(self):
        rows=self._mock_records()
        rows[0]["original_query"]="Altered text"
        self._validate_records(rows,"question text mismatch")
        rows=self._mock_records()
        rows[0]["status"]="error"
        self._validate_records(rows,"execution error")


if __name__ == "__main__":
    unittest.main()
