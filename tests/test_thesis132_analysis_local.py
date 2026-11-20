"""Thesis132 descriptive statistics tests; never infer clinical gold."""
import unittest

from research_v2.thesis132_analysis import _bootstrap_by_group,_paired_b7_b8,_summarize

def record(qid,evidence,ms,group):
    return {"query_id":qid,"semantic_group":group,
        "evidence":evidence,"total_ms":ms,
        "status":"ok","answer":True,
        "retrieval_ms":1.0,"generation_ms":None,"paths":0,
        "causal_edges":0,"mean_path_depth":None,"feedback_iterations":0,
        "hallucination_detected":None,"validator_passed":None}

class Thesis132AnalysisTests(unittest.TestCase):
    def test_operational_summary_leaves_medical_accuracy_empty(self):
        a=record("Q1",True,10.0,"A")
        b=record("Q2",False,20.0,"B")
        b["status"]="abstained"
        b["answer"]=False
        result=_summarize([a,b])
        self.assertEqual(result["n"],2)
        self.assertEqual(result["evidence_yield"],1)
        self.assertEqual(result["status_abstained"],1)
        self.assertIsNone(result["clinical_correctness"])
        self.assertEqual(result["p50_total_ms"],15)

    def test_family_group_bootstrap_is_a_descriptive_proxy(self):
        rows=[record("Q1",True,10,"A"),record("Q2",False,15,"A"),
              record("Q3",True,25,"B"),record("Q4",True,30,"C")]
        result=_bootstrap_by_group(rows,iterations=100,seed=1337)
        self.assertEqual(result["observed"],.75)
        self.assertLessEqual(result["lower_95"],result["upper_95"])
        self.assertIn("not_external_generalization",result["interpretation"])

    def test_paired_rl_comparison_has_no_ppo_causal_claim(self):
        qids=["Q1","Q2","Q3","Q4"]
        meta={q:{"semantic_group":q} for q in qids}
        b7=[record(q,i%2==0,20+i,q) for i,q in enumerate(qids)]
        b8=[record(q,i!=2,50+i,q) for i,q in enumerate(qids)]
        paired=_paired_b7_b8({"B7":b7,"B8":b8},meta)
        self.assertEqual(paired["n"],4)
        self.assertTrue(paired["not_a_PPO_causal_estimate"])
        self.assertFalse(paired["clinical_accuracy"])
        self.assertEqual(paired["B8_minus_B7_mean_latency_ms"],30)


if __name__=="__main__":
    unittest.main()
