"""Validate versioned reconciled G1 without altering the historical candidate graph."""
import csv
import json
import unittest
from collections import Counter
from pathlib import Path

from research_v2.experiment_runner import PROJECT_ROOT, file_sha256
from research_v2.g1_typed_retrieval import G1TypedIndex

GRAPH = PROJECT_ROOT / "research_artifacts/graph-G1-reconciled-20261121"
CROSSWALK = PROJECT_ROOT / "research_artifacts/g1_reconciliation_20261121/phase2m_source_crosswalk.csv"


def csv_rows(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


class ReconciledG1Tests(unittest.TestCase):
    def test_counts_and_embedded_file_hashes(self):
        manifest = json.loads((GRAPH / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["stage"], "G1_CANDIDATE_REQUIRES_INDEPENDENT_VERIFICATION")
        self.assertFalse(manifest["medical_gold"])
        self.assertEqual(manifest["independent_verification_documentation_verified_by_this_package"], False)
        self.assertEqual(len(csv_rows(GRAPH / "nodes.csv")), 78)
        self.assertEqual(len(csv_rows(GRAPH / "edges.csv")), 52)
        self.assertEqual(len(csv_rows(GRAPH / "typed_relationships.csv")), 24)
        self.assertEqual(file_sha256(GRAPH / "typed_relationships.csv"), manifest["edge_ledger_sha256"])
        self.assertEqual(file_sha256(GRAPH / "approved_aliases.csv"), manifest["alias_ledger_sha256"])
        self.assertEqual(file_sha256(CROSSWALK), manifest["source_crosswalk_sha256"])

    def test_exact_canonical_node_reference_and_typed_index(self):
        nodes = {row["name"] for row in csv_rows(GRAPH / "nodes.csv")}
        edges = csv_rows(GRAPH / "edges.csv")
        typed = csv_rows(GRAPH / "typed_relationships.csv")
        self.assertTrue(all(r["cause"] in nodes and r["effect"] in nodes for r in edges))
        self.assertTrue(all(r["subject"] in nodes and r["object"] in nodes for r in typed))
        self.assertEqual(Counter(x["relation"] for x in typed),
                         {"CAUSES": 15, "RISK_FACTOR_FOR": 8, "ASSOCIATED_WITH": 1})
        index = G1TypedIndex(GRAPH)
        self.assertEqual(len(index.edges), 24)
        self.assertTrue(all(x["source_url"].startswith("https://") for x in typed))

    def test_internal_source_mapping_is_consistent_not_clinical_gold(self):
        typed = csv_rows(GRAPH / "typed_relationships.csv")
        xwalk = csv_rows(CROSSWALK)
        ids = {x["g1_source_id"]: x for x in xwalk}
        self.assertEqual(len(ids), len(xwalk))
        self.assertEqual(len(xwalk), 52)
        for row in typed:
            self.assertIn(row["source_id"], ids)
            source = ids[row["source_id"]]
            self.assertEqual(source["source_url"], row["source_url"])
            self.assertEqual(row["crosswalk_g1_approval"], "proposed")
            self.assertEqual(row["independent_reviewer_record_status"],
                             "independent_reviewer_identity_not_documented_in_inputs")
            self.assertEqual(row["independent_verification_attestation"],
                             "user_attested_completed_2026-11-21")


if __name__ == "__main__":
    unittest.main()
