"""No fabricated medical evidence in G1: validation and provenance checks."""
import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from research_v2 import g1_review as g1
from research_v2.experiment_runner import PROJECT_ROOT


class ReviewGateTests(unittest.TestCase):
    def sample(self, **changes):
        row = {
            "evidence_id": "SYNTHETIC_1", "query_id": "Q1", "cause": "A",
            "effect": "B", "relation": "CAUSES",
            "source_url": "https://www.cdc.gov/synthetic-test-reference",
            "source_title": "Synthetic test title",
            "source_publisher": "Synthetic test publisher",
            "evidence_note": "SYNTHETIC UNIT TEST ONLY. A hypothetical causal claim.",
            "review_status": "approved", "reviewer": "unit-test",
            "reviewed_date": "2026-01-01",
        }
        row.update(changes)
        return row

    def ledger(self, path, rows):
        g1.write_csv(path, g1.EVIDENCE_COLUMNS, rows)

    def test_diagnostics_gap_selection_never_invents_facts(self):
        rows = [
            {"query_id": "Q1", "retrieval_review_bucket": "G0_SNAPSHOT_COVERAGE_GAP",
             "query_type": "direct", "question": "What causes X?", "target_concept_draft": "X",
             "B5_status": "abstained", "B6_status": "abstained"},
            {"query_id": "Q2", "retrieval_review_bucket": "DIRECTED_RETRIEVAL_GAP",
             "query_type": "indirect", "B5_status": "ok", "B6_status": "abstained"},
            {"query_id": "Q3", "retrieval_review_bucket": "BOTH_GRAPH_METHODS_RETURN_EVIDENCE",
             "B5_status": "ok", "B6_status": "ok"},
        ]
        gaps = g1.select_review_gaps(rows)
        self.assertEqual(len(gaps), 2)
        self.assertEqual({r["query_id"] for r in gaps}, {"Q1", "Q2"})
        self.assertNotIn("cause", gaps[0])
        self.assertEqual(gaps[0]["review_status"], "unreviewed")

    def test_blank_evidence_ledger_no_approved(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "evidence.csv"
            self.ledger(path, [])
            approved, audit = g1.validate_candidates(path, {"Q1"})
            self.assertEqual(approved, [])
            self.assertEqual(audit["approved_evidence_rows"], 0)

    def test_refuse_unreviewed_and_association_relations(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "evidence.csv"
            for row, message in [
                (self.sample(source_url="http://example.org"), "HTTPS"),
                (self.sample(relation="ASSOCIATED_WITH"), "only CAUSES"),
                (self.sample(reviewed_date="2037-01-01"), "future"),
                (self.sample(query_id="not_queued"), "not a queued"),
                (self.sample(evidence_note="too short"), "too short"),
            ]:
                with self.subTest(message=message):
                    self.ledger(path, [row])
                    with self.assertRaisesRegex(ValueError, message):
                        g1.validate_candidates(path, {"Q1"})

    def test_rejected_rows_not_accepted_as_approved(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "evidence.csv"
            self.ledger(path, [self.sample(review_status="rejected", source_url="")])
            approved, audit = g1.validate_candidates(path, {"Q1"})
            self.assertFalse(approved)
            self.assertEqual(audit["review_counts"]["rejected"], 1)

    def test_build_keeps_g0_unchanged_and_sources_not_verified(self):
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as tmp:
            root = Path(tmp)
            base, out = root / "baseline", root / "candidate"
            base.mkdir()
            g1.write_csv(base / "nodes.csv", ("node_id", "name"), [
                {"node_id":"OLDN_A", "name":"A"}, {"node_id":"OLDN_B", "name":"B"}
            ])
            g1.write_csv(base / "edges.csv", ("edge_id","cause","effect"), [
                {"edge_id":"OLDE_AB","cause":"A","effect":"B"}
            ])
            old_sha = g1.file_sha256(base / "edges.csv")
            manifest = g1.build_graph(base, [self.sample(
                evidence_id="SYNTHETIC_2", cause="B", effect="C"
            )], out)
            self.assertEqual(manifest["new_edges"], 1)
            self.assertEqual(manifest["total_nodes"], 3)
            self.assertEqual(manifest["total_edges"], 2)
            self.assertEqual(manifest["source_g0_unreviewed_edges_inherited"], 1)
            self.assertFalse(manifest["clinical_accuracy_verified"])
            self.assertEqual(g1.file_sha256(base / "edges.csv"), old_sha)
            edges = g1.read_csv(out / "edges.csv")
            self.assertEqual(edges[0]["edge_id"], "OLDE_AB")
            self.assertEqual((edges[1]["cause"], edges[1]["effect"]), ("B", "C"))

    def test_no_graph_without_approvals(self):
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as tmp:
            with self.assertRaisesRegex(ValueError, "No approved"):
                g1.build_graph(g1.G0_DIR, [], Path(tmp) / "G1")


if __name__ == "__main__":
    unittest.main()
