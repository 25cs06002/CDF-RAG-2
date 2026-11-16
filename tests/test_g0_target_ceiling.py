"""Offline tests ensure draft labels are never represented as model performance."""
import unittest
from research_v2.g0_target_ceiling import assess
from research_v2.pipelines.baselines import G0Corpus


class TargetCeilingTests(unittest.TestCase):
    def test_counts_and_unsupported_are_not_forced(self):
        rows = [
            {"query_id": f"Q{i}", "query_type": "indirect",
             "draft_target": "not in G0", "B6_evidence": "False"}
            for i in range(64)
        ]
        rows[0]["draft_target"] = "stroke"
        rows[1].update(draft_target="depression", query_type="unsupported")
        rows[2].update(draft_target="migraine", query_type="ambiguous")
        data, meta = assess(rows, G0Corpus())
        self.assertEqual(meta["counts"]["SNAPSHOT_REACHABLE_ONLY_WITH_LEAKED_DRAFT_TARGET"], 1)
        self.assertEqual(meta["counts"]["UNDERDETERMINED_NO_FORCED_DIAGNOSIS"], 2)
        self.assertEqual(meta["counts"]["TARGET_ABSENT_G0"], 61)
        self.assertTrue(meta["draft_target_leakage"])
        self.assertFalse(meta["is_model_result"])
        self.assertTrue(all(x["requires_manual_review"] for x in data))

    def test_rejects_incomplete_and_non_gap(self):
        rows = [
            {"query_id": f"Q{i}", "query_type": "direct",
             "draft_target": "stroke", "B6_evidence": "False"}
            for i in range(64)
        ]
        with self.assertRaisesRegex(ValueError, "64 unique"):
            assess(rows[:3], G0Corpus())
        rows[0]["B6_evidence"] = "True"
        with self.assertRaisesRegex(ValueError, "not an original"):
            assess(rows, G0Corpus())
