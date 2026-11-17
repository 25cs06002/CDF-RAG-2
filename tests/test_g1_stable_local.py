"""Focused synthetic G1 stable resolver checks; never launches offline sprint."""
import csv
import tempfile
import unittest
from pathlib import Path

from research_v2.g1_stable_retrieval import StableTargetG1
from research_v2.pipelines.baselines import G0Corpus


def make_graph(path, nodes, edges):
    path.mkdir(parents=True,exist_ok=True)
    with (path/"nodes.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=("node_id","name"))
        writer.writeheader()
        writer.writerows([{"node_id":str(i),"name":n} for i,n in enumerate(nodes)])
    with (path/"edges.csv").open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=("edge_id","cause","effect"))
        writer.writeheader()
        writer.writerows([{"edge_id":str(i),"cause":c,"effect":e} for i,(c,e) in enumerate(edges)])


class StableTargetTests(unittest.TestCase):
    def test_preserves_evidence_even_when_new_phrase_is_longer(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            g0=root/"g0"
            g1=root/"g1"
            make_graph(g0,["Hypertension","Smoking"],
                       [("Smoking","Hypertension")])
            make_graph(g1,["Hypertension","Smoking","Severe hypertension","Vascular damage"],
                       [("Smoking","Hypertension"),("Vascular damage","Severe hypertension")])
            original=G0Corpus(g0)
            for mode in ("graph_rag","causal_kg_rag"):
                stable=StableTargetG1(g1,frozen=original,pipeline=mode)
                self.assertEqual(
                    stable.concept("What contributes to severe hypertension?"),
                    "Hypertension")
                self.assertEqual(stable.last_reason,"preserved_G0_anchor_with_evidence")

    def test_new_target_requires_explicit_unique_mention(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            g0=root/"g0"
            g1=root/"g1"
            make_graph(g0,["Diabetes","Fatigue"],
                       [("Fatigue","Diabetes")])
            make_graph(g1,["Diabetes","Fatigue","Heart failure","Cardiac injury"],
                       [("Fatigue","Diabetes"),("Cardiac injury","Heart failure")])
            original=G0Corpus(g0)
            stable=StableTargetG1(g1,frozen=original,pipeline="causal_kg_rag")
            self.assertEqual(stable.concept("What causes heart failure?"),"Heart failure")
            self.assertEqual(stable.last_reason,"one_explicit_connected_new_G1_target")
            self.assertNotEqual(stable.concept("What causes unknown illness?"),"Heart failure")


if __name__=="__main__":
    unittest.main()
