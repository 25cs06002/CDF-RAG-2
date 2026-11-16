"""Unit tests: G0-only query target selection; no medical knowledge injected."""
import unittest

from research_v2.graph_gap_triage import (
    TargetOrientedG0Corpus, effect_mentions, exact_mention, shadow_probe, triage,
)
from research_v2.pipelines.baselines import G0Corpus, BaselinePipeline
from research_v2.base_pipeline import PipelineRequest
from research_v2.experiment_runner import BenchmarkQuestion


class GapTriageTests(unittest.TestCase):
    def setUp(self):
        self.g0 = G0Corpus()

    def test_word_boundaries(self):
        self.assertTrue(exact_mention("Risk of STROKE?", "stroke"))
        self.assertFalse(exact_mention("Stroke-like words", "stroke999"))
        self.assertFalse(exact_mention("depressive symptoms", "depression"))

    def test_query_target_only_no_draft_label(self):
        question = ("How might persistent high blood pressure affect vascular injury "
                    "and subsequent stroke risk?")
        self.assertEqual(self.g0.concept(question), "High blood pressure")
        alternative = TargetOrientedG0Corpus()
        self.assertEqual(alternative.concept(question), "stroke")
        old = BaselinePipeline("causal_kg_rag", corpus=self.g0).run(
            PipelineRequest(run_id="x",query_id="1",query=question)
        )
        new = BaselinePipeline("causal_kg_rag", corpus=alternative).run(
            PipelineRequest(run_id="x",query_id="1",query=question)
        )
        self.assertEqual(old.status, "abstained")
        self.assertEqual(new.status, "ok")
        self.assertTrue(new.causal_edges)

    def test_no_subtype_to_broad_diabetes_alias(self):
        q = "Does insulin resistance cause type 2 diabetes?"
        self.assertNotIn("Diabetes", effect_mentions(q, self.g0))
        self.assertNotEqual(TargetOrientedG0Corpus().concept(q), "Diabetes")

    def test_keep_existing_answerable_concept(self):
        alternative = TargetOrientedG0Corpus()
        q = "What factors cause stroke?"
        self.assertEqual(alternative.concept(q), self.g0.concept(q))

    def test_triage_flags_unsupported_as_review_not_add(self):
        rows = []
        for i in range(1, 101):
            rows.append({
                "query_id": f"PILOT_{i:03d}", "retrieval_review_bucket": (
                    "G0_SNAPSHOT_COVERAGE_GAP" if i == 1 else
                    "DIRECTED_RETRIEVAL_GAP" if i == 2 else
                    "BOTH_GRAPH_METHODS_RETURN_EVIDENCE"),
                "question": "What causes stroke?", "query_type": (
                    "unsupported" if i == 1 else "multi_hop"),
                "target_concept_draft": "stroke",
                "B5_status": "abstained" if i == 1 else "ok",
                "B6_status": "abstained" if i <= 2 else "ok",
                "B5_evidence": "False", "B6_evidence": "False",
            })
        result, counts = triage(rows, self.g0)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["query_id"], "PILOT_002")
        self.assertEqual(result[1]["classification"],
                         "UNDERDETERMINED_REVIEW_NO_AUTOMATIC_GRAPH_EXPANSION")
        self.assertEqual(counts["DIRECTED_TARGET_SELECTION_OR_MISSING_PATH"], 1)
        self.assertFalse(result[1]["shadow_concept_changed"])

    def test_shadow_produces_no_retrieval_regression(self):
        qs = [
            BenchmarkQuestion(query_id="Q1",query="How might high blood pressure affect stroke risk?"),
            BenchmarkQuestion(query_id="Q2",query="What causes migraine?"),
            BenchmarkQuestion(query_id="Q3",query="What is an unnamed person's exact disease?"),
        ]
        _, summary = shadow_probe(qs)
        self.assertEqual(summary["queries"], 3)
        self.assertEqual(summary["regressed_yield"], 0)
        self.assertGreaterEqual(summary["newly_recovered"], 1)


if __name__ == "__main__":
    unittest.main()
