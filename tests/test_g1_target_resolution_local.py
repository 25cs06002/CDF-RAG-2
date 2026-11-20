"""Pure synthetic G1 target and sufficiency regression tests.

Do not rely on real clinical graph edges, gold answers or local G1 artifacts.
"""
from collections import defaultdict
import unittest

from research_v2.g1_target_resolution import mentioned_concepts, resolve_target
from research_v2.g1_typed_retrieval import G1TypedIndex
from research_v2.thesis132_g1 import _safe_gate
from research_v2.schema import PipelineRecord, EvidenceDocument, CausalPath, StageTimings


class TargetResolutionTests(unittest.TestCase):
    def setUp(self):
        self.nodes = [
            "A", "B", "Stroke", "Diabetes", "Type 2 diabetes",
            "Hypertension", "Severe hypertension",
        ]

    def test_direction_to_downstream_endpoint(self):
        target, reason = resolve_target(
            "What pathway leads from A through B to Stroke?",
            self.nodes, eligible={"A", "B", "Stroke"},
        )
        # 'to Stroke' must be an explicit directional cue to select it.
        self.assertEqual(target, "Stroke")
        self.assertEqual(reason, "explicit_directional_endpoint")

    def test_cause_verb_and_risk_of_disambiguation(self):
        target, reason = resolve_target(
            "Does Hypertension cause Stroke?",
            self.nodes, eligible={"Hypertension", "Stroke"},
        )
        self.assertEqual((target, reason), ("Stroke", "explicit_directional_endpoint"))
        target, _ = resolve_target(
            "What is the risk of Stroke in people with Hypertension?",
            self.nodes, eligible={"Hypertension", "Stroke"},
        )
        self.assertEqual(target, "Stroke")

    def test_comparison_and_conjunction_do_not_guess(self):
        for query in (
            "Compare Stroke and Hypertension.",
            "What causes Stroke and Hypertension?",
        ):
            self.assertIsNone(resolve_target(
                query, self.nodes, eligible={"Stroke", "Hypertension"})[0])

    def test_specific_nested_term_wins_only_at_same_span(self):
        q = "What contributes to Severe hypertension?"
        hits = mentioned_concepts(
            q, self.nodes, eligible={"Hypertension", "Severe hypertension"})
        self.assertEqual([name for _, _, name in hits], ["Severe hypertension"])
        q2 = "Compare Severe hypertension with Hypertension"
        self.assertEqual(len(mentioned_concepts(
            q2, self.nodes, eligible={"Hypertension", "Severe hypertension"})), 2)

    def test_unreviewed_diabetes_alias_does_not_promote_generic(self):
        target, reason = resolve_target(
            "What causes type 2 diabetes?", self.nodes,
            eligible={"Diabetes"},
        )
        self.assertIsNone(target)
        self.assertEqual(reason, "no_explicit_reviewed_effect")

    def test_approved_alias_may_resolve_to_exact_target(self):
        target, _ = resolve_target(
            "What causes CVA?", self.nodes,
            aliases={"cva": "Stroke"}, eligible={"Stroke"},
        )
        self.assertEqual(target, "Stroke")

    def test_ambiguous_directional_endpoints_withheld(self):
        target, reason = resolve_target(
            "Does A cause Stroke and Hypertension?", self.nodes,
            eligible={"A", "Stroke", "Hypertension"},
        )
        self.assertIsNone(target)
        self.assertEqual(reason, "ambiguous_directional_endpoints")

    def test_index_integrates_directional_resolver(self):
        idx = G1TypedIndex.__new__(G1TypedIndex)
        idx.nodes = ["A", "Stroke"]
        idx.aliases = {}
        idx.by_effect = defaultdict(list, {
            "a": [{"edge_id": "E1"}], "stroke": [{"edge_id": "E2"}],
        })
        self.assertEqual(
            idx.target("Does A cause Stroke?"),
            ("Stroke", "explicit_directional_endpoint"),
        )
        self.assertEqual(
            idx.target("Compare A and Stroke"),
            (None, "comparative_multiple_effects"),
        )

    def test_uncertain_taxonomic_relations_not_sufficient(self):
        self.assertNotIn("IS_A", G1TypedIndex.allowed_relations("What contributes to Stroke?"))
        self.assertNotIn("UNCERTAIN", G1TypedIndex.allowed_relations("What contributes to Stroke?"))
        self.assertNotIn("UNCERTAIN", G1TypedIndex.allowed_relations("What risk factors predict Stroke?"))


class FakeIndex:
    nodes = ["Stroke"]
    aliases = {}
    by_effect = defaultdict(list, {"stroke": [{"edge_id": "E"}]})

    def retrieve(self, question, *, causal_multihop, top_k, query_id, run_id):
        kw = dict(run_id=run_id, query_id=query_id, pipeline_id="test",
                  original_query=question, matched_concept="Stroke",
                  timings=StageTimings(retrieval_ms=1.0))
        if causal_multihop:
            paths = [CausalPath(nodes=["A", "B", "Stroke"])] if "supported path" in question else []
            return PipelineRecord(**kw, causal_paths=paths), "path_or_no_path"
        docs = [EvidenceDocument(document_id="EDGE_1", source_id="SRC", text="candidate")]
        return PipelineRecord(**kw, retrieved_documents=docs), "onehop"


class MultiHopGateTests(unittest.TestCase):
    def test_onehop_does_not_satisfy_explicit_path(self):
        result = _safe_gate(FakeIndex(), "What pathway leads to Stroke?")
        self.assertEqual(result["status"], "abstained")
        self.assertEqual(result["rounds"], 2)
        self.assertEqual(result["evidence_ids"], [])

    def test_twohop_can_supply_candidate_for_review(self):
        result = _safe_gate(FakeIndex(), "What supported path leads to Stroke?")
        self.assertEqual(result["status"], "candidate_evidence")
        self.assertEqual(result["rounds"], 2)
        self.assertTrue(any(x.startswith("PATH:") for x in result["evidence_ids"]))


if __name__ == "__main__":
    unittest.main()
