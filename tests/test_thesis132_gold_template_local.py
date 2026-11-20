"""Gold template must never be mistaken for reviewed gold."""
import json
import tempfile
import unittest
from pathlib import Path

from research_v2.experiment_runner import PROJECT_ROOT as ROOT
from research_v2.thesis132_gold_template import prepare
from research_v2.thesis132_gold_scoring import validate_gold
from research_v2.benchmark_v2_validation import validate


class GoldTemplateTests(unittest.TestCase):
    def test_132_all_pending_no_labels_and_scoring_refuses_it(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            out = prepare(Path(temporary))
            path = out / "gold_132_PENDING_REVIEW.jsonl"
            rows = [json.loads(s) for s in path.read_text().splitlines() if s.strip()]
            self.assertEqual(len(rows), 132)
            self.assertEqual(len({r["query_id"] for r in rows}), 132)
            self.assertTrue(all(r["review_status"] == "PENDING_EXTERNAL_ADJUDICATION"
                                and r["verified_answer"] is None
                                and r["answerability"] is None
                                and not r["gold_paths"]
                                for r in rows))
            _, questions, _ = validate()
            with self.assertRaises(ValueError):
                validate_gold(path, questions)


if __name__ == "__main__":
    unittest.main()
