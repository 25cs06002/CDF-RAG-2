"""Synthetic 132 gold-scoring gate and retrieved-evidence tests; NOT medical gold."""
import tempfile
import json
import unittest
from pathlib import Path

from research_v2.benchmark_v2_validation import validate
from research_v2.schema import (
    PipelineRecord, EvidenceDocument, CausalEdge, CausalPath, StageTimings
)
from research_v2.thesis132_gold_scoring import (
    validate_gold, score_record, _summary, _grouped_ci,
)


def sample_gold(qid="Q", question="Synthetic question?", answerable=True):
    return {
        "query_id": qid,
        "query": question,
        "answerability": "answerable" if answerable else "unanswerable",
        "verified_answer": "Synthetic reference" if answerable else "",
        "relevant_source_ids": ["SRC"] if answerable else [],
        "gold_causal_edges": [
            {"cause": "A", "relation": "CAUSES", "effect": "B"}
        ] if answerable else [],
        "gold_paths": [["A", "B", "C"]] if answerable else [],
        "evidence_spans": [
            {"source_id": "SRC", "passage_locator": "Section 1",
             "exact_text": "synthetic evidence"}
        ] if answerable else [],
        "independent_reviewer_ids": ["reviewer-one", "reviewer-two"],
        "adjudicator_id": "independent-adjudicator",
        "review_status": "externally_adjudicated",
    }


class GoldValidationTests(unittest.TestCase):
    def test_entire_132_requires_externally_reviewed_rows(self):
        _, questions, _ = validate()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "gold.jsonl"
            synthetic = [sample_gold(q["query_id"], q["query"]) for q in questions]
            path.write_text("".join(json.dumps(x) + "\n" for x in synthetic),
                            encoding="utf-8")
            by_id = validate_gold(path, questions)
            self.assertEqual(len(by_id), 132)
            synthetic[0]["review_status"] = "candidate"
            path.write_text("".join(json.dumps(x) + "\n" for x in synthetic),
                            encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Unreviewed reference"):
                validate_gold(path, questions)
            synthetic[0]["review_status"] = "externally_adjudicated"
            synthetic[0]["independent_reviewer_ids"] = ["one", "one"]
            path.write_text("".join(json.dumps(x) + "\n" for x in synthetic),
                            encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Two distinct"):
                validate_gold(path, questions)

    def test_gold_cannot_change_question_or_drop_query(self):
        _, questions, _ = validate()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "gold.jsonl"
            synthetic = [sample_gold(q["query_id"], q["query"]) for q in questions]
            path.write_text("".join(json.dumps(x) + "\n" for x in synthetic[:-1]),
                            encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "exactly 132"):
                validate_gold(path, questions)
            synthetic[0]["query"] = "Changed question"
            path.write_text("".join(json.dumps(x) + "\n" for x in synthetic),
                            encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "changed query"):
                validate_gold(path, questions)


class GoldScoreTests(unittest.TestCase):
    def test_risk_and_direction_do_not_auto_match_causes(self):
        gold = sample_gold()
        record = PipelineRecord(
            run_id="synthetic", query_id="Q", pipeline_id="G1",
            original_query="Synthetic question?", status="ok",
            retrieved_documents=[EvidenceDocument(document_id="D",
                                                  source_id="SRC", text="synthetic")],
            causal_edges=[CausalEdge(cause="A", effect="B", relation="RISK_FACTOR_FOR")],
            causal_paths=[CausalPath(nodes=["A", "B", "C"])],
            timings=StageTimings(total_ms=20, retrieval_ms=5),
        )
        score = score_record(record, gold)
        self.assertEqual(score["source_precision_at_k"], 1)
        self.assertEqual(score["source_recall_at_k"], 1)
        self.assertEqual(score["gold_edge_precision"], 0)
        self.assertEqual(score["gold_edge_recall"], 0)
        self.assertEqual(score["gold_path_recall"], 1)
        self.assertIsNone(score["answer_correct_human"])

    def test_unanswerable_requires_abstention(self):
        gold = sample_gold(answerable=False)
        good = PipelineRecord(
            run_id="test", query_id="Q", pipeline_id="X",
            original_query="Synthetic question?", status="abstained")
        bad = PipelineRecord(
            run_id="test", query_id="Q", pipeline_id="X",
            original_query="Synthetic question?", status="ok")
        self.assertTrue(score_record(good, gold)["abstention_decision_correct"])
        self.assertFalse(score_record(bad, gold)["abstention_decision_correct"])
        self.assertTrue(score_record(bad, gold)["unsafe_nonabstention"])

    def test_group_bootstrap_uses_family_units(self):
        scores = [
            {"query_id": "a", "source_recall_at_k": 1},
            {"query_id": "b", "source_recall_at_k": 1},
            {"query_id": "c", "source_recall_at_k": 0},
        ]
        groups = {"a": "family1", "b": "family1", "c": "family2"}
        result = _grouped_ci(scores, groups, "source_recall_at_k",
                             seed=2026, iterations=100)
        self.assertAlmostEqual(result["observed"], 2 / 3)
        self.assertTrue(0 <= result["lower_95"] <= result["upper_95"] <= 1)
        self.assertEqual(result["bootstrap_unit"], "semantic_family")


if __name__ == "__main__":
    unittest.main()
