"""Focused case-study integrity checks without graph, LLM or PPO execution."""
import unittest

from research_v2.submission_case_studies import (
    _quote, _table, _example_block, choose_examples,
)


def case(cid, priority, qid):
    return {
        "case_id": cid, "priority": priority, "query_id": qid,
        "family": "B6", "question": "Synthetic target question",
        "issue_flags": "wrong_stable_target" if priority=="P0_WRONG_TARGET"
            else "gain_without_relevant_direction_correct_evidence",
        "g0_target": "old concept", "g1_legacy_target": "new concept",
        "stable_target": "other concept",
        "selection_reason": "test_rule", "review_stable_target": "wrong",
        "review_query_intent": "answerable", "original_G1_change": "gained",
        "g0_to_stable_change": "gained", "review_note": "Synthetic reviewer note",
    }


class CaseStudiesTests(unittest.TestCase):
    def test_selects_two_independent_question_examples(self):
        cases=[
            case("B6_A","P0_WRONG_TARGET","Q1"),
            case("B6_B","P1_UNSUPPORTED_GAIN","Q1"),
            case("B5_C","P1_UNSUPPORTED_GAIN","Q2"),
        ]
        selected=choose_examples(cases)
        self.assertEqual(len(selected),2)
        self.assertEqual({x["priority"] for x in selected},
                         {"P0_WRONG_TARGET","P1_UNSUPPORTED_GAIN"})
        self.assertEqual({x["query_id"] for x in selected},{"Q1","Q2"})

    def test_markdown_tables_escape_question_metadata(self):
        self.assertEqual(_quote("foo|bar"),r"foo\|bar")
        result=_table([{"column":"foo|bar"}],("column",))
        self.assertIn(r"foo\|bar",result)

    def test_case_example_uses_real_review_judgment_and_not_ground_truth(self):
        item=case("B6_A","P0_WRONG_TARGET","Q1")
        evidence=[
            {
                "stage":"G1_STABLE","subject":"synthetic factor",
                "relation":"RISK_FACTOR_FOR","object":"synthetic condition",
                "provenance":"G1_approved_typed_candidate",
                "evidence_relevance":"irrelevant",
                "relation_direction":"incorrect","source_checked":"not_checked",
            }
        ]
        text=_example_block(item,evidence,1)
        self.assertIn("Synthetic target question",text)
        self.assertIn("G1 stabilized target",text)
        self.assertIn("RISK_FACTOR_FOR",text)
        self.assertIn("independent clinical correctness",text)


if __name__=="__main__":
    unittest.main()
