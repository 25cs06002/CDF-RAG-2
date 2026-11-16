"""Focused local-only checks; NOT the offline sprint test suite."""
import tempfile
import unittest
from pathlib import Path

from research_v2.g1_foundation import (
    ROOT, candidate_registry, validate_expansion, prepare, audit_and_build,
    SOURCE_COLS, EDGE_COLS, CONCEPT_COLS, ALIAS_COLS,
)
from research_v2.g1_query_interpreter import InterpretedGraph, exact_mention
from research_v2.pipelines.baselines import G0Corpus


class G1FoundationLocalTests(unittest.TestCase):
    def test_100_plus_32_are_frozen_separate_and_draft(self):
        pilot,ext=validate_expansion()
        self.assertEqual((len(pilot),len(ext)),(100,32))
        self.assertTrue(all(x.review_status=="draft" and x.answerable is None for x in ext))
        self.assertEqual(len({q.query_id for q in pilot+ext}),132)

    def test_candidate_registry_reuses_existing_23(self):
        src=candidate_registry()
        self.assertEqual(len(src),25)
        self.assertIn("SRC001",src)
        self.assertIn("EXT008",src)
        self.assertIn("G1SRC_CDC_STROKE",src)

    def test_no_auto_approved_graph_or_aliases(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            p=prepare(Path(tmp)/"review")
            self.assertTrue((p/"source_review.csv").exists())
            self.assertTrue((p/"typed_relationship_review.csv").exists())
            self.assertTrue((p/"alias_review.csv").exists())
            report=audit_and_build(p)
            self.assertEqual(report["approved_source_count"],0)
            self.assertEqual(report["approved_new_node_count"],0)
            with self.assertRaisesRegex(ValueError,"No approved typed relationships"):
                audit_and_build(p,build=True)

    def test_conservative_interpreter_does_not_infer_subtypes(self):
        graph=InterpretedGraph()
        self.assertEqual(graph.concept("What factors cause stroke?"),"stroke")
        self.assertEqual(graph.concept("How does hypertension affect type 2 diabetes?"),None)
        self.assertEqual(graph.concept("What causes coronary artery disease?"),None)
        self.assertIsNone(graph.concept("Diagnose a specific psychiatric disorder from their occupation alone."))
        self.assertEqual(graph.interpretation,"insufficient_information_request")

    def test_multiple_targets_abstain_not_assumed_causality(self):
        graph=InterpretedGraph()
        self.assertIsNone(graph.concept("Compare stroke and asthma mechanisms."))
        self.assertEqual(graph.interpretation,"multiple_target_mentions_abstain")
        self.assertTrue(exact_mention("risk of stroke?", "stroke"))
        self.assertFalse(exact_mention("risk of stroke-like illness?", "stroke999"))


if __name__=="__main__":
    unittest.main()
