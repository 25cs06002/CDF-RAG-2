"""Offline verification that the new 132-question source-screened set is honest.

Tests *metadata integrity and anti-gold safeguards*, not medical truth.
"""
import json
import tempfile
import unittest
from pathlib import Path

from research_v2.benchmark_v2_validation import (
    validate, DATASET, PILOT, EXT, SOURCE, MANIFEST, write_review_queue,
)


class BenchmarkV2Tests(unittest.TestCase):
    def test_complete_v2_metadata_and_sources(self):
        report,rows,sources=validate()
        self.assertEqual(report["status"],"PASS_SOURCE_SCREENED_METADATA_INTEGRITY")
        self.assertEqual(report["questions"],132)
        self.assertEqual(report["source_catalog_entries"],24)
        self.assertEqual(report["clinical_gold_questions"],0)
        self.assertFalse(report["g1_independent_holdout"])
        self.assertEqual(len(rows),132)
        self.assertEqual({r["query_type"] for r in rows},{
            "direct","paraphrased","indirect","multi_hop","comparative",
            "risk_factor","ambiguous","unsupported",
        })
        self.assertTrue(all(r["gold_answer"] is None for r in rows))
        self.assertEqual(report["by_type"]["multi_hop"],19)
        self.assertGreater(len(report["cross_cohort_semantic_groups"]),0)

    def test_independent_review_form_is_unfilled(self):
        _,rows,sources=validate()
        with tempfile.TemporaryDirectory() as temporary:
            out=write_review_queue(rows,sources,Path(temporary))
            content=(out/"independent_adjudication_worksheet.csv").read_text(encoding="utf-8")
            self.assertEqual(len(content.splitlines()),265)
            self.assertIn("reviewer_independence_declaration",content)
            self.assertIn("verified_source_spans",content)
            self.assertIn("review_slot",content)
            self.assertIn("EXTQ_021",content)
            self.assertIn("PILOT_050",content)

    def _mutate_and_reject(self,alter,expected_message):
        with tempfile.TemporaryDirectory() as temp:
            mutated=[json.loads(x) for x in DATASET.read_text(encoding="utf-8").splitlines() if x.strip()]
            alter(mutated)
            path=Path(temp)/"invalid_v2.jsonl"
            path.write_text("\n".join(json.dumps(x) for x in mutated)+"\n",encoding="utf-8")
            with self.assertRaisesRegex(ValueError,expected_message):
                validate(pilot=PILOT,ext=EXT,dataset=path,
                         catalog=SOURCE,manifest=MANIFEST)

    def test_reject_fabricated_gold(self):
        self._mutate_and_reject(
            lambda rows:rows[0].__setitem__("gold_answer","fabricated answer"),
            "Unadjudicated gold_answer")

    def test_reject_false_independent_reviewer(self):
        self._mutate_and_reject(
            lambda rows:rows[0].__setitem__("independent_review_count",1),
            "Invented independent reviewer")

    def test_reject_broken_source_reference(self):
        self._mutate_and_reject(
            lambda rows:rows[0].__setitem__("source_reference_ids",["FAKE_SOURCE"]),
            "Unknown source")

    def test_preserves_target_role_pilot050(self):
        _,rows,_=validate()
        q={r["query_id"]:r for r in rows}
        self.assertEqual(q["PILOT_050"]["original_target_concept"],"type 2 diabetes")
        self.assertIn("kidney complications",q["PILOT_050"]["focus_entities"])
        self.assertEqual(
            q["EXTQ_021"]["expected_response_type"],
            "qualified_source_cited_explanation",
        )


if __name__=="__main__":
    unittest.main()
