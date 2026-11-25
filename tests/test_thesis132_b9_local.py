import json
import unittest
from research_v2.thesis132_b9 import B9Controller, parse_selection, GRAPH
from research_v2.base_pipeline import PipelineRequest
from research_v2.g1_typed_retrieval import G1TypedIndex
from research_v2.thesis132 import CORPUS, ROOT
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
    def test_decline_triggers_new_evidence_round(self):
        seen=[]
        def decline_then_select(prompt):
            items=json.loads(prompt.split("\n",1)[1])["items"]
            seen.append({x["id"] for x in items})
            return '{"selected_ids":[]}' if len(seen)==1 else json.dumps({"selected_ids":[items[0]["id"]]})
        r,t=self.ctrl(decline_then_select).run(req("What factors contribute to heart failure?"))
        self.assertEqual(r.status,"ok")
        self.assertEqual(t["selector_calls"],2)
        self.assertTrue(seen[1]-seen[0])

    def test_full_frozen_cohort_stub_selector(self):
        questions=[json.loads(x) for x in (ROOT/"data/benchmark/final_132_v1/questions_only.jsonl").read_text().splitlines()]
        self.assertEqual(len(questions),132)
        controller=self.ctrl(self.first)
        records=[]
        for q in questions:
            r,t=controller.run(PipelineRequest(run_id="test132",query_id=q["query_id"],query=q["query"],graph_version="G1_CANDIDATE"))
            self.assertNotEqual(r.status,"error",(q["query_id"],r.error_message))
            self.assertLessEqual(len(t["rounds"]),3)
            self.assertLessEqual(t["selector_calls"],3)
            self.assertEqual(r.query_id,q["query_id"])
            self.assertIsNone(r.hallucination_detected)
            if r.status=="ok":
                self.assertTrue(r.final_answer)
                self.assertTrue(r.validator_passed)
            else:self.assertIsNone(r.final_answer)
            records.append(r)
        self.assertTrue(any(r.status=="ok" for r in records))
        self.assertTrue(any(r.status=="abstained" for r in records))
        print("Stub-only full-cohort behavior:",{s:sum(r.status==s for r in records) for s in ("ok","abstained","error")})

    def test_document_fallback(self):
        r,t=self.ctrl(self.first).run(req("What factors contribute to chronic obstructive pulmonary disease?"))
        self.assertEqual(r.status,"ok")
        self.assertEqual(t["selected_evidence"][0]["kind"],"candidate_document")
        self.assertEqual(len(t["rounds"]),2)

if __name__=="__main__":unittest.main()
