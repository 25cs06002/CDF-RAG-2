import json,tempfile,unittest,zipfile
from pathlib import Path
from research_v2.thesis132_b9_review import RUN,ROOT,prepare,inputs,baseline_rows,BASELINE,validate_row,PROTOCOL,STATEMENT,COUNTS,sha
class ReviewTests(unittest.TestCase):
    def test_real_package(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            d=prepare(RUN,Path(tmp)/"package")
            m=json.loads((d/"review_package_manifest.json").read_text())
            self.assertEqual(m["records"],132)
            self.assertEqual(m["run_manifest_sha256"],sha(RUN/"manifest.json"))
            with zipfile.ZipFile(d.with_suffix(".zip")) as z:
                self.assertIn("README_MASTER_PROMPT.md",z.namelist())
                self.assertIsNone(z.testzip())
        _,rows=inputs(RUN)
        self.assertEqual(sum(bool(r["final_response"]) for r in rows),99)
        self.assertEqual(sum(r["answer_scope"]=="partial_path_evidence" for r in rows),11)
        self.assertEqual(len(baseline_rows(BASELINE)),8)
    def test_synthetic_empty_answer(self):
        r=dict(model="B9",query_id="synthetic",query="synthetic",final_response="",final_response_sha256="test",pipeline_status="abstained",query_type="ambiguous",medical_domain="test",latency_ms=1,answer_scope="none")
        common=dict(ai_final_label="No",ai_review_status="AI_AUTO_ACCEPTED",human_review_status="NOT_REVIEWED",confidence="high",evidence_grounding_status="NOT_APPLICABLE_EMPTY_ANSWER",answer_completeness="APPROPRIATE_ABSTENTION",final_response_sha256="test")
        l=dict(**common,ai_candidate_label="No",protocol_version=PROTOCOL,label_provenance_statement=STATEMENT,verification_reasoning="synthetic only",fabricated_citation="false",**{k:0 for k in COUNTS.values()})
        c=dict(**common,query=r["query"],final_response="",fabricated_citation=False,claims=[])
        self.assertFalse(validate_row(l,c,r)["substantive"])
        c["final_response"]="changed"
        with self.assertRaises(ValueError):validate_row(l,c,r)
