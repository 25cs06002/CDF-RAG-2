"""Paired B9 policy ablation comparison must reject mismatched G1/corpus."""
import json
import tempfile
import unittest
from pathlib import Path

from research_v2.schema import PipelineRecord,StageTimings
from research_v2.thesis132 import BASE
from research_v2.thesis132_adaptive_compare import compare


class AdaptiveComparisonTests(unittest.TestCase):
    def _runs(self,root):
        dirs=[]
        for policy,success in (("fixed1",50),("fixed2",65),("fixed3",75),("adaptive",70)):
            folder=root/policy
            folder.mkdir()
            summary={
                "policy":policy,"queries":132,"medical_accuracy":False,
                "status_ok":success,"status_abstained":132-success,"status_error":0,
                "input_sha256":"same-data","graph_manifest_sha256":"same-graph",
                "document_corpus_sha256":"same-corpus","top_k":5,
            }
            (folder/"summary.json").write_text(json.dumps(summary),encoding="utf-8")
            with (folder/"records.jsonl").open("w",encoding="utf-8") as f:
                for i in range(132):
                    r=PipelineRecord(
                        run_id="Test",query_id=f"Q{i:03d}",
                        pipeline_id="b9_candidate_"+policy,original_query="Test question",
                        status="ok" if i<success else "abstained",
                        feedback_iterations=0 if policy=="fixed1" else 1,
                        timings=StageTimings(total_ms=i+1,retrieval_ms=i+1))
                    f.write(r.model_dump_json()+"\n")
            dirs.append(folder)
        return dirs

    def test_full_paired_policy_provenance(self):
        BASE.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=BASE) as tmp:
            dirs=self._runs(Path(tmp))
            output,report=compare(dirs)
            self.assertEqual(len(report["per_policy"]),4)
            self.assertIsNone(report["clinical_accuracy"])
            self.assertFalse(report["learned_PPO"])
            self.assertEqual(report["paired_adaptive_vs_fixed3"]["both_candidate"],70)
            self.assertTrue((output/"ablation.json").is_file())

    def test_reject_incomparable_graph_snapshots(self):
        BASE.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=BASE) as tmp:
            dirs=self._runs(Path(tmp))
            path=dirs[1]/"summary.json"
            summary=json.loads(path.read_text(encoding="utf-8"))
            summary["graph_manifest_sha256"]="other-graph"
            path.write_text(json.dumps(summary),encoding="utf-8")
            with self.assertRaisesRegex(ValueError,"Incomparable"):
                compare(dirs)


if __name__=="__main__":
    unittest.main()
