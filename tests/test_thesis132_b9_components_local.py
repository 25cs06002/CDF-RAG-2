import json,unittest
from research_v2.thesis132_b9_components import controller,frozen,evidence_check
from research_v2.thesis132_b9_v2 import _cohort,PipelineRequest

def stub(prompt):
    obj=json.loads(prompt.split("\n",1)[1])
    ids=list(dict.fromkeys(g[0] for g in obj["groups"].values() if g))
    return json.dumps({"selected_ids":ids or [obj["items"][0]["id"]]})

class ComponentTests(unittest.TestCase):
    def test_frozen_reference(self):
        self.assertEqual(frozen()["questions"],132)
    def test_full_cohort(self):
        _,_,qs=_cohort()
        self.assertEqual(len(qs),132)
        counts={}
        for variant in ("full","graph_off","document_off"):
            c=controller(variant,stub)
            counts[variant]=0
            for q in qs:
                r,t=c.run(PipelineRequest(run_id="stub",query_id=q.query_id,query=q.query,graph_version="G1_CANDIDATE"))
                self.assertNotEqual(r.status,"error",(variant,q.query_id,r.error_message))
                evidence_check(variant,r,t)
                self.assertLessEqual(len(t["rounds"]),3)
                if t["route"]=="guard":self.assertEqual(r.status,"abstained")
                counts[variant]+=r.status=="ok"
        self.assertGreater(counts["full"],counts["document_off"])
        print("STUB ONLY component answers",counts)
    def test_cache_ids(self):
        c=controller("full",stub)
        a,h=c.retrieve_docs("What causes coronary artery disease?")
        b,k=c.retrieve_docs("What causes coronary artery disease?")
        self.assertFalse(h);self.assertTrue(k);self.assertEqual(a,b)
