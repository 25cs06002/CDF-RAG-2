"""Synthetic, offline evidence-review gate tests. Never assert clinical gold."""
import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from research_v2.experiment_runner import PROJECT_ROOT as ROOT
from research_v2.thesis132_evidence import (
    FIELDS, prepare, audit, freeze,
)


class StubIndex:
    def __init__(self, graph_dir):
        self.edges = [{
            "edge_id": "E1", "subject": "Concept A", "relation": "CAUSES",
            "object": "Concept B", "source_id": "SRC1",
        }]


class EvidenceReviewTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.graph = self.root / "g1"
        self.graph.mkdir()
        (self.graph / "manifest.json").write_text("{}\n", encoding="utf-8")
        (self.graph / "typed_relationships.csv").write_text(
            "edge_id,subject,relation,object,source_id\nE1,Concept A,CAUSES,Concept B,SRC1\n",
            encoding="utf-8",
        )
        self.patcher = patch("research_v2.thesis132_evidence.G1TypedIndex", StubIndex)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)
        self.workspace = prepare(self.graph, self.root / "review")

    def _read(self):
        with (self.workspace / "edge_adjudication.csv").open(newline="") as f:
            return list(csv.DictReader(f))

    def _write(self, rows):
        with (self.workspace / "edge_adjudication.csv").open(
            "w", encoding="utf-8", newline=""
        ) as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)

    def _reviewed(self):
        rows = self._read()
        row = rows[0]
        row.update({
            "decision": "verified",
            "source_url": "https://example.org/source",
            "source_snapshot_path": "sources/source.txt",
            "exact_source_passage": "Synthetic study supports A leading to B.",
            "passage_locator": "section 2",
            "relation_direction_checked": "yes",
            "reviewer_1": "reviewer-one",
            "reviewer_2": "reviewer-two",
            "independence_declared": "yes",
            "adjudicator": "third-party",
            "review_date": "2026-11-21",
            "review_notes": "Synthetic fixture; not a medical assertion",
        })
        (self.workspace / "sources").mkdir()
        (self.workspace / "sources" / "source.txt").write_text(
            "Synthetic study supports A leading to B.\n", encoding="utf-8"
        )
        self._write(rows)

    def test_new_graph_is_not_gold(self):
        rows, summary = audit(self.workspace, self.graph)
        self.assertEqual(len(rows), 1)
        self.assertEqual(summary["status"], "INCOMPLETE_REVIEW")
        self.assertEqual(summary["decisions"], {"pending": 1})
        self.assertFalse(summary["clinical_gold_certified_by_code"])

    def test_exact_source_span_and_review_signoff(self):
        self._reviewed()
        rows, summary = audit(self.workspace, self.graph)
        self.assertEqual(summary["status"], "READY_FOR_EXTERNAL_SIGNOFF")
        self.assertEqual(summary["decisions"], {"verified": 1})
        signed = self.workspace / "supervisor_signoff.json"
        with self.assertRaises(FileNotFoundError):
            freeze(self.workspace, self.graph, signed)
        signed.write_text(json.dumps({
            "supervisor_name": "synthetic-supervisor",
            "approval_date": "2026-11-21",
            "reviewer_credentials_checked": True,
            "source_authenticity_checked": True,
            "causal_directions_checked": True,
            "independent_external_adjudication_completed": True,
        }), encoding="utf-8")
        output = freeze(self.workspace, self.graph, signed)
        manifest = json.loads((output / "manifest.json").read_text())
        self.assertEqual(manifest["counts"]["verified"], 1)
        self.assertIn("REQUIRES_HUMAN_AUTHENTICATION", manifest["status"])
        with self.assertRaises(FileExistsError):
            freeze(self.workspace, self.graph, signed)

    def test_verified_edge_must_have_exact_located_source(self):
        self._reviewed()
        rows = self._read()
        rows[0]["exact_source_passage"] = "Not present in the source snapshot"
        self._write(rows)
        _, report = audit(self.workspace, self.graph)
        self.assertIn("exact_source_passage_not_found_in_local_snapshot",
                      report["failed_edges"]["E1"])

    def test_candidate_identity_cannot_be_edited(self):
        rows = self._read()
        rows[0]["object"] = "Concept C"
        self._write(rows)
        _, report = audit(self.workspace, self.graph)
        self.assertIn("candidate_identity_modified", report["failed_edges"]["E1"])

    def test_new_candidate_graph_drift_blocks_review(self):
        (self.graph / "typed_relationships.csv").write_text("changed", encoding="utf-8")
        with self.assertRaises(ValueError):
            audit(self.workspace, self.graph)


if __name__ == "__main__":
    unittest.main()
