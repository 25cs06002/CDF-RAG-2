"""B9 deterministic rule-gate ablations; no graph files or generators required."""
import unittest
from research_v2.base_pipeline import PipelineRequest
from research_v2.schema import PipelineRecord,EvidenceDocument,CausalPath,StageTimings
from research_v2.thesis132_adaptive import AdaptiveCandidateController


class StubIndex:
    nodes=["heart failure","coronary artery disease"]
    def retrieve(self,question,*,causal_multihop,top_k,query_id,run_id):
        if "typed" in question.lower():
            return PipelineRecord(run_id=run_id,query_id=query_id,original_query=question,
                pipeline_id="typed",
                matched_concept="heart failure",
                retrieved_documents=[EvidenceDocument(
                    document_id="EDGE_CAND_0001",source_id="APPROVED_CANDIDATE_SOURCE",text="risk factor")],
                status="ok",timings=StageTimings(retrieval_ms=1)),"reviewed_typed_candidate"
        if "pathway" in question.lower() and causal_multihop:
            return PipelineRecord(run_id=run_id,query_id=query_id,original_query=question,
                pipeline_id="typed",matched_concept="heart failure",
                causal_paths=[CausalPath(nodes=["A","B","heart failure"])],
                status="ok",timings=StageTimings(retrieval_ms=2)),"reviewed_candidate_path"
        return PipelineRecord(run_id=run_id,query_id=query_id,original_query=question,
            pipeline_id="typed",status="abstained",
            timings=StageTimings(retrieval_ms=1)),"no_typed_candidate"


class StubBm25:
    def run(self,request):
        return PipelineRecord(run_id=request.run_id,query_id=request.query_id,
            original_query=request.query,pipeline_id="bm25_rag",status="ok",
            retrieved_documents=[EvidenceDocument(document_id="DOC_CAND",source_id="CANDIDATE",text="maybe")],
            timings=StageTimings(retrieval_ms=3))


def req(q):
    return PipelineRequest(run_id="TEST",query_id="Q",query=q)


class AdaptiveCandidateTests(unittest.TestCase):
    def setUp(self):
        self.ctrl=AdaptiveCandidateController(StubIndex(),StubBm25())

    def test_adaptive_direct_early_stop_no_extra_rounds(self):
        out=self.ctrl.run(req("Which typed factors contribute to heart failure?"),"adaptive")
        self.assertEqual(out.status,"ok")
        self.assertEqual(out.feedback_iterations,0)
        self.assertEqual(out.action_history[:2],["typed_direct","early_stop_candidate_evidence"])
        self.assertIsNone(out.final_answer)

    def test_document_summaries_alone_cannot_certify_typed_evidence(self):
        out=self.ctrl.run(req("What explains heart failure?"),"adaptive")
        self.assertEqual(out.status,"abstained")
        self.assertIn("document_bm25",out.action_history)
        self.assertEqual(out.feedback_iterations,1)
        self.assertTrue(out.retrieved_documents)
        self.assertFalse(out.causal_paths)

    def test_path_query_uses_second_round(self):
        out=self.ctrl.run(req("What pathway from A through B leads to heart failure?"),"adaptive")
        self.assertEqual(out.status,"ok")
        self.assertTrue(out.causal_paths)
        self.assertEqual(out.feedback_iterations,1)
        self.assertNotIn("document_bm25",out.action_history)

    def test_fixed3_has_two_extra_retrieval_rounds(self):
        out=self.ctrl.run(req("What typed factors contribute to heart failure?"),"fixed3")
        self.assertEqual(out.feedback_iterations,2)
        self.assertEqual(out.status,"ok")

    def test_comparative_requires_multi_target_handling(self):
        out=self.ctrl.run(req("Compare heart failure versus coronary artery disease"),"adaptive")
        self.assertEqual(out.status,"abstained")
        self.assertEqual(out.action_history,["abstain_ambiguous_or_unsupported"])


if __name__=="__main__":
    unittest.main()
