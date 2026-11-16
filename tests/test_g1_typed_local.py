"""Focused typed-relation checks. No network, model, Neo4j or sprint suite."""
import csv
import json
import tempfile
import unittest
from pathlib import Path

from research_v2.experiment_runner import PROJECT_ROOT, file_sha256
from research_v2.g1_typed_retrieval import G1TypedIndex
from research_v2.g1_foundation import EDGE_COLS, ALIAS_COLS


class TypedReviewTests(unittest.TestCase):
    def synthetic(self,root):
        graph=root/"G1"
        graph.mkdir()
        (graph/"nodes.csv").write_text(
            "node_id,name\n1,A\n2,Stroke\n3,B\n",encoding="utf-8")
        with (graph/"typed_relationships.csv").open("w",newline="",encoding="utf-8") as f:
            w=csv.DictWriter(f,fieldnames=EDGE_COLS)
            w.writeheader()
            w.writerow({
                "edge_id":"RISK_001","subject":"A","relation":"RISK_FACTOR_FOR",
                "object":"Stroke","source_id":"TEST_SRC",
                "passage_locator":"Synthetic section","support_summary":"Unit-test-only risk relationship, not a medical assertion.",
                "review_status":"approved","reviewer":"synthetic","review_date":"2026-01-01",
            })
            w.writerow({
                "edge_id":"CAUSE_001","subject":"B","relation":"CAUSES",
                "object":"Stroke","source_id":"TEST_SRC",
                "passage_locator":"Synthetic section","support_summary":"Unit-test-only causal relationship, not a medical assertion.",
                "review_status":"approved","reviewer":"synthetic","review_date":"2026-01-01",
            })
        with (graph/"approved_aliases.csv").open("w",newline="",encoding="utf-8") as f:
            csv.DictWriter(f,fieldnames=ALIAS_COLS).writeheader()
        (graph/"manifest.json").write_text(json.dumps({
            "stage":"G1_CANDIDATE_REQUIRES_INDEPENDENT_VERIFICATION",
            "edge_ledger_sha256":file_sha256(graph/"typed_relationships.csv"),
            "alias_ledger_sha256":file_sha256(graph/"approved_aliases.csv"),
        }),encoding="utf-8")
        return graph

    def test_risk_not_promoted_to_cause(self):
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as tmp:
            index=G1TypedIndex(self.synthetic(Path(tmp)))
            record,reason=index.retrieve("Which risk factors are linked to Stroke?")
            self.assertEqual(record.status,"ok")
            self.assertEqual(len(record.retrieved_documents),2)
            self.assertEqual(len(record.causal_edges),1)
            self.assertEqual(record.causal_edges[0].cause,"B")
            self.assertIn("RISK_FACTOR_FOR",record.retrieved_documents[0].text)
            causal,_=index.retrieve("What causes Stroke?")
            self.assertTrue(all("RISK_FACTOR_FOR" not in d.text for d in causal.retrieved_documents))
            self.assertEqual(len(causal.causal_edges),1)

    def test_abstains_without_explicit_reviewed_target(self):
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as tmp:
            index=G1TypedIndex(self.synthetic(Path(tmp)))
            record,reason=index.retrieve("Predict an exact disease without any symptoms")
            self.assertEqual(record.status,"abstained")
            self.assertEqual(reason,"under_determined_question")
            self.assertFalse(record.retrieved_documents)


if __name__=="__main__":
    unittest.main()
