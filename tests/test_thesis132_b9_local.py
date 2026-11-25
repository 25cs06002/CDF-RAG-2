import json
import unittest
from research_v2.thesis132_b9 import B9Controller, parse_selection, GRAPH
from research_v2.base_pipeline import PipelineRequest
from research_v2.g1_typed_retrieval import G1TypedIndex
from research_v2.thesis132 import CORPUS
from research_v2.pipelines import create_pipeline

def req(q): return PipelineRequest(run_id="test",query_id="Q",query=q,graph_version="G1_CANDIDATE")

class B9Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.index=G1TypedIndex(GRAPH)
        cls.docs=create_pipeline("B2",document_corpus=CORPUS,generate=False)
    def ctrl(self,selector,policy="adaptive"):
        return B9Controller(self.index,self.docs,selector,policy=policy)
    def first(self,prompt):
        obj=json.loads(prompt.split("\n",1)[1])
        return json.dumps({"selected_ids":[obj["items"][0]["id"]]})
    def test_json_gate(self):
        self.assertIsNone(parse_selection('{"selected_ids":["invented"]}',{"x"},5))
        self.assertIsNone(parse_selection('{"selected_ids":["x","x"]}',{"x"},5))
        self.assertIsNone(parse_selection('{"selected_ids":["x"],"answer":"new text"}',{"x"},5))
        self.assertEqual(parse_selection('{"selected_ids":[]}',{"x"},5),[])
    def test_direct_selection_and_early_stop(self):
        r,t=self.ctrl(self.first).run(req("What factors contribute to heart failure?"))
        self.assertEqual(r.status,"ok")
        self.assertEqual(len(t["rounds"]),1)
        self.assertTrue(r.validator_passed)
        self.assertIn(t["selected_ids"][0],r.final_answer)
        self.assertIsNone(r.hallucination_detected)
    def test_fixed3_same_question_runs_budget(self):
        r,t=self.ctrl(self.first,"fixed3").run(req("What factors contribute to heart failure?"))
        self.assertEqual(len(t["rounds"]),3)
        self.assertEqual(r.status,"ok")
    def test_unknown_target_no_generator(self):
        r,t=self.ctrl(lambda _:self.fail("Generator should not run")).run(req("What causes unknown disorder xyz?"))
        self.assertEqual(r.status,"abstained")
        self.assertEqual(t["selector_calls"],0)
    def test_patient_specific_no_generator(self):
        r,t=self.ctrl(lambda _:self.fail("Generator should not run")).run(req("My chest feels strange; is heart disease the cause?"))
        self.assertEqual(r.status,"abstained")
        self.assertEqual(t["selector_calls"],0)
    def test_invalid_selection_abstains(self):
        r,t=self.ctrl(lambda _:'{"selected_ids":["unknown"]}').run(req("What factors contribute to heart failure?"))
        self.assertEqual(r.status,"abstained")
        self.assertFalse(r.validator_passed)
        self.assertIsNone(r.final_answer)
    def test_transport_error_not_abstention(self):
        def fail(_):raise RuntimeError("transport")
        r,t=self.ctrl(fail).run(req("What factors contribute to heart failure?"))
        self.assertEqual(r.status,"error")
        self.assertIn("transport",r.error_message)
    def test_path_source_ids_are_retained(self):
        r,t=self.ctrl(self.first).run(req("What pathway from coronary artery disease through heart muscle damage or weakening leads to heart failure?"))
        self.assertEqual(r.status,"ok")
        self.assertTrue(r.causal_paths)
        self.assertEqual(len(t["rounds"]),2)
        for eid in r.causal_paths[0].evidence_ids:
            self.assertIn(eid,r.final_answer)
    def test_document_fallback(self):
        r,t=self.ctrl(self.first).run(req("What factors contribute to chronic obstructive pulmonary disease?"))
        self.assertEqual(r.status,"ok")
        self.assertEqual(t["selected_evidence"][0]["kind"],"candidate_document")
        self.assertEqual(len(t["rounds"]),2)

if __name__=="__main__":unittest.main()
