"""Offline 132-question protocol, actual B1 smoke and guarded G1 logic."""
import json
import tempfile
import unittest
from pathlib import Path

from research_v2.schema import PipelineRecord,StageTimings,EvidenceDocument,CausalPath
from research_v2.thesis132 import freeze,_cohort,_grouped_folds,run,summarize
from research_v2.thesis132_g1 import _safe_gate


class FakeIndex:
    nodes=["Coronary artery disease","Heart failure"]
    by_effect={"coronary artery disease":[1],"heart failure":[1]}
    def retrieve(self,question,causal_multihop=False,top_k=5,query_id="Q",run_id="R"):
        if "through" not in question.lower():
            return (PipelineRecord(run_id=run_id,query_id=query_id,
                    pipeline_id="g1_typed_onehop",original_query=question,
                    status="abstained",timings=StageTimings(retrieval_ms=.01)),
                    "no_direct")
        if not causal_multihop:
            return (PipelineRecord(run_id=run_id,query_id=query_id,
                    pipeline_id="g1_typed_onehop",original_query=question,
                    status="abstained"),"no_direct")
        return (PipelineRecord(run_id=run_id,query_id=query_id,
                pipeline_id="g1_typed_twohop",original_query=question,
                matched_concept="Heart failure",status="ok",
                causal_paths=[CausalPath(nodes=["A","B","Heart failure"])]),
                "typed_causal_path")


class Thesis132UnitTests(unittest.TestCase):
    def test_freeze_132_questions_and_semantic_folds(self):
        directory=freeze()
        _,data,items=_cohort()
        self.assertEqual(len(items),132)
        self.assertEqual(len({x.query_id for x in items}),132)
        self.assertTrue(all(set(json.loads(line))=={"query_id","query"}
                            for line in data.read_text(encoding="utf-8").splitlines()))
        import csv
        with (directory/"fold_metadata.csv").open(newline="",encoding="utf-8") as f:
            meta=list(csv.DictReader(f))
        self.assertEqual(len(meta),132)
        by_group={}
        for row in meta:
            by_group.setdefault(row["semantic_group"],set()).add(row["fold"])
        self.assertTrue(all(len(folds)==1 for folds in by_group.values()))
        self.assertEqual(set(r["fold"] for r in meta),set("01234"))
        self.assertTrue((directory/"manifest.json").is_file())
        self.assertFalse(json.loads((directory/"manifest.json").read_text())["independent_clinical_gold"])

    def test_reject_legacy_retrieval_only_or_unapproved_llm(self):
        with self.assertRaisesRegex(ValueError,"only supports"):
            run("B7","retrieval_only","sentence",1,5,False)
        with self.assertRaisesRegex(ValueError,"--allow-live"):
            run("B1","end_to_end","sentence",1,5,False)

    def test_real_b1_single_query_smoke_retrieval_only(self):
        directory=run("B1","retrieval_only","sentence",1,5,False)
        result=summarize(directory)
        self.assertEqual(result["question_count"],1)
        self.assertEqual(len(result["models"]),1)
        self.assertEqual(result["models"][0]["model"],"B1")
        self.assertEqual(result["models"][0]["errors"],0)
        self.assertFalse(result["independent_clinical_gold"])

    def test_safe_g1_gate_abstains_on_comparison_or_unsupported(self):
        fake=FakeIndex()
        unclear=_safe_gate(fake,"Find the exact disease without any measurements")
        self.assertEqual(unclear["decision"],"withhold")
        self.assertEqual(unclear["rounds"],0)
        comparative=_safe_gate(fake,"Compare heart failure versus coronary artery disease")
        self.assertEqual(comparative["reason"],"comparative_multi_entity_not_single_target")
        self.assertEqual(comparative["status"],"abstained")

    def test_safe_gate_only_retries_for_explicit_multihop(self):
        fake=FakeIndex()
        result=_safe_gate(fake,"How does A contribute through B to heart failure?")
        self.assertEqual(result["rounds"],2)
        self.assertEqual(result["decision"],"candidate_for_review")
        self.assertTrue(any(x.startswith("PATH:") for x in result["evidence_ids"]))
        self.assertEqual(_safe_gate(fake,"What is heart failure?")["rounds"],1)


if __name__=="__main__":
    unittest.main()
