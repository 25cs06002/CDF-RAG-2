import csv
import json
import unittest
from pathlib import Path
from research_v2.thesis132_b9_v2 import B9V2Controller, personal_guard, ROOT, GRAPH, CORPUS, G1TypedIndex, PipelineRequest, sha
from research_v2.pipelines import create_pipeline

BASE=ROOT/"presentation_20261118/thesis132/b9_end_to_end/b9_adaptive_20261125T180912711211Z_e6efb5"
QUESTIONS=ROOT/"data/benchmark/final_132_v1/questions_only.jsonl"

def stub(prompt):
    obj=json.loads(prompt.split("\n",1)[1])
    selected=list(dict.fromkeys(g[0] for g in obj["groups"].values() if g))
    return json.dumps({"selected_ids":selected or [obj["items"][0]["id"]]})

class B9V2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.idx=G1TypedIndex(GRAPH)
        cls.docs=create_pipeline("B2",document_corpus=CORPUS,generate=False)
    def ctrl(self,selector=stub):
        return B9V2Controller(self.idx,self.docs,selector)
    def test_evidence_certainty_not_personal(self):
        self.assertFalse(personal_guard("Which environmental exposures are associated with asthma onset, and how certain is the evidence?"))
        self.assertTrue(personal_guard("Is a particular chronic kidney diagnosis certain without context?"))
    def test_open_descriptive(self):
        r,t=self.ctrl().run(PipelineRequest(run_id="T",query_id="Q",query="Which cardiac condition is discussed when the heart cannot pump blood effectively?"))
        self.assertEqual(t["route"],"descriptive")
        self.assertEqual(r.status,"ok")
    def test_comparison_requires_both(self):
        q=PipelineRequest(run_id="T",query_id="Q",query="How do risk factors for stroke compare with those for coronary artery disease?")
        r,t=self.ctrl().run(q)
        self.assertEqual(t["route"],"comparison")
        self.assertEqual(r.status,"ok")
        bad=lambda p:json.dumps({"selected_ids":[json.loads(p.split("\n",1)[1])["items"][0]["id"]]})
        # An item mentioning both targets may legitimately cover both; use
        # a query where evidence groups have distinct records instead.
        q2=PipelineRequest(run_id="T",query_id="Q",query="How do migraine and pneumonia compare in their causes?")
        r,t=self.ctrl(bad).run(q2)
        self.assertEqual(r.status,"abstained")
    def test_partial_path_is_not_causal_chain(self):
        q=PipelineRequest(run_id="T",query_id="Q",query="Does evidence support a pathway from excess adiposity through insulin resistance to type 2 diabetes, and what are its limitations?")
        r,t=self.ctrl().run(q)
        self.assertEqual(t["answer_scope"],"partial_path_evidence")
        self.assertIn("do not establish a complete causal chain",r.final_answer)
        self.assertFalse(r.causal_paths)
    def test_132_and_v1_alignment(self):
        manifest=json.loads((BASE/"manifest.json").read_text())
        self.assertEqual(manifest["status"],"COMPLETE_132")
        self.assertEqual(manifest["dataset_sha256"],sha(QUESTIONS))
        for name,expected in manifest["outputs_sha256"].items():
            self.assertEqual(sha(BASE/name),expected)
        qs=[json.loads(s) for s in QUESTIONS.read_text().splitlines()]
        prior=[json.loads(s) for s in (BASE/"records.jsonl").read_text().splitlines()]
        self.assertEqual([(x["query_id"],x["query"]) for x in qs],[(x["query_id"],x["original_query"]) for x in prior])
        with (ROOT/"data/benchmark/final_132_v1/metadata_review_only.csv").open(newline="") as f:
            meta={r["query_id"]:r for r in csv.DictReader(f)}
        c=self.ctrl();counts={};routes={};regressions=[];negative=[]
        for q,p in zip(qs,prior):
            r,t=c.run(PipelineRequest(run_id="T132",query_id=q["query_id"],query=q["query"]))
            self.assertNotEqual(r.status,"error",(q["query_id"],r.error_message))
            self.assertLessEqual(len(t["rounds"]),3)
            self.assertLessEqual(t["selector_calls"],3)
            self.assertIsNone(r.hallucination_detected)
            if meta[q["query_id"]]["query_type"] in {"ambiguous","unsupported"}:
                negative.append(q["query_id"])
                self.assertEqual(r.status,"abstained",q["query_id"])
            if p["status"]=="ok" and r.status!="ok":regressions.append(q["query_id"])
            counts[r.status]=counts.get(r.status,0)+1
            routes[t["route"]]=routes.get(t["route"],0)+1
        self.assertEqual(len(qs),132)
        self.assertEqual(len(negative),24)
        self.assertGreater(counts.get("ok",0),54)
        print("STUB_ONLY_V2",counts,"routes",routes,"v1_ok_to_stub_abstain",regressions)

if __name__=="__main__":unittest.main()
