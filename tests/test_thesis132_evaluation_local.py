"""Real-data 132-query evaluation tests: require the already committed B1–B8 CSVs.

No models, network or medical-label assertions. CI exercises each source CSV,
the exported summary counts, all derived metrics, paired rows and integrity.
"""
import csv
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from research_v2.thesis132_evaluation import (
    ROOT, MODELS, _load, analyze, _hash, _read_csv,
)

SOURCE = (
    ROOT / "presentation_20261118/thesis132/csv_reports" /
    "batch132_csv_20261121T061653054773Z_bfeb15"
)


class Thesis132EvaluationTests(unittest.TestCase):
    def test_actual_source_export_complete_and_integrity_checked(self):
        base, manifest, summaries, records, _ = _load(SOURCE)
        self.assertEqual(base, SOURCE)
        self.assertEqual(len(records), 8)
        self.assertEqual(set(records), set(MODELS))
        self.assertEqual([len(records[m]) for m in MODELS], [132]*8)
        self.assertEqual(int(summaries["B5"]["status_abstained"]), 83)
        self.assertEqual(int(summaries["B6"]["status_abstained"]), 91)
        self.assertEqual(int(summaries["B7"]["candidate_evidence_count"]), 61)
        self.assertEqual(int(summaries["B8"]["candidate_evidence_count"]), 86)

    def test_generate_all_real_outputs_and_preserve_missing_quality_fields(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            output, report = analyze(SOURCE, Path(temp)/"analysis")
            self.assertEqual(report["total_model_query_records"], 1056)
            self.assertEqual(report["paired_B7_B8"][
                "B8_minus_B7_net_evidence_questions"],25)
            self.assertEqual(len(report["per_model_summary"]), 8)
            self.assertEqual(report["per_model_summary"]["B5"]["abstained"],83)
            self.assertEqual(report["per_model_summary"]["B6"]["abstained"],91)
            self.assertGreater(report["per_model_summary"]["B8"]["mean_latency_ms"],
                               report["per_model_summary"]["B7"]["mean_latency_ms"])

            for model in MODELS:
                path = output/f"evaluation_metrics_{model}_132.csv"
                rows,_ = _read_csv(path)
                self.assertEqual(len(rows),132)
                summary, _ = _read_csv(output/f"evaluation_summary_{model}_132.csv")
                metrics = {row["Metric"]:row["Value"] for row in summary}
                self.assertEqual(metrics["Queries Evaluated"], "132")
                for field in ("Groundedness", "Hallucination Rate",
                              "Context Relevance", "Semantic Refinement Score"):
                    self.assertEqual(metrics[field], "",
                                     "Must not fabricate reference labels")
            all_models,_=_read_csv(output/"evaluation_summary_all_models_132.csv")
            self.assertEqual([r["model"] for r in all_models],list(MODELS))
            self.assertEqual(len(all_models),8)
            pairs,_=_read_csv(output/"comparison_B7_vs_B8_paired_132.csv")
            self.assertEqual(len(pairs),132)
            self.assertEqual(len({r["query_id"] for r in pairs}),132)
            changed,_=_read_csv(output/"B7_B8_evidence_changed_questions_132.csv")
            counts=report["paired_B7_B8"]["candidate_evidence_pair_counts"]
            self.assertEqual(len(changed),counts["B7_only"]+counts["B8_only"])
            self.assertEqual(counts["B8_only"]-counts["B7_only"],25)
            by_kind,_=_read_csv(output/"comparison_by_query_type_132.csv")
            by_domain,_=_read_csv(output/"comparison_by_medical_domain_132.csv")
            for grouping in (by_kind,by_domain):
                for model in MODELS:
                    self.assertEqual(sum(int(r["questions"]) for r in grouping
                                         if r["model"] == model),132)
            for filename, sha in report["generated_sha256"].items():
                self.assertEqual(_hash(output/filename),sha)

    def test_modified_csv_is_rejected_not_reported_as_accuracy(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            source = Path(temp)/"tampered"
            source.mkdir()
            files = ["export_manifest.json", "pipeline_summary_132.csv"] + [
                f"{model}_queries_132.csv" for model in MODELS
            ]
            for filename in files:
                shutil.copyfile(SOURCE/filename,source/filename)
            with (source/"B8_queries_132.csv").open("a",encoding="utf-8") as f:
                f.write("\n")
            with self.assertRaisesRegex(ValueError,"modified model CSV"):
                analyze(source,Path(temp)/"reports")


if __name__=="__main__":
    unittest.main()
