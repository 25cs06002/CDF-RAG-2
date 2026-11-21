"""Offline verification of screenshot-style metrics on authentic frozen 132 CSVs."""
import contextlib
import csv
import io
import tempfile
import unittest
from pathlib import Path

from research_v2.thesis132_full_metrics import (
    DEFAULT_SOURCE, MODELS, METRIC_NAMES, _classification, _read_gold,
    evaluate, display_model, display_comparison, display_hallucination,
)
from research_v2.thesis132_evaluation import ROOT

class FullMetric132Tests(unittest.TestCase):
    def test_actual_132x8_metrics_without_embedding_or_gold(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            folder, meta=evaluate(DEFAULT_SOURCE,Path(temp)/"report",semantic_proxies=False)
            self.assertEqual(meta["questions"],132)
            self.assertEqual(meta["models"],list(MODELS))
            self.assertFalse(meta["gold_labels_supplied"])
            for model in MODELS:
                with (folder/f"evaluation_metrics_{model}_132_full.csv").open(
                    newline="",encoding="utf-8") as file:
                    detailed=list(csv.DictReader(file))
                self.assertEqual(len(detailed),132)
                with (folder/f"evaluation_summary_{model}_132_full.csv").open(
                    newline="",encoding="utf-8") as file:
                    summaries=list(csv.DictReader(file))
                self.assertEqual(tuple(r["Metric"] for r in summaries),METRIC_NAMES)
                self.assertEqual(summaries[3]["Value"],"")
                self.assertEqual(summaries[4]["Value"],"")
                self.assertEqual(summaries[5]["Value"],"")
                self.assertEqual(summaries[7]["Value"],"")
            with (folder/"hallucination_classification_summary_132.csv").open(
                newline="",encoding="utf-8") as file:
                hall={r["model"]:r for r in csv.DictReader(file)}
            self.assertEqual(hall["B7"]["detector_flags"],"33")
            self.assertEqual(hall["B8"]["detector_flags"],"43")
            self.assertEqual(hall["B1"]["classification_status"],
                             "unavailable_no_approved_gold")
            with (folder/"comparison_B7_vs_B8_paired_132_full.csv").open(
                newline="",encoding="utf-8") as file:
                matched=list(csv.DictReader(file))
            self.assertEqual(len(matched),132)
            b8_only=sum(r["evidence_category"]=="B8_only" for r in matched)
            b7_only=sum(r["evidence_category"]=="B7_only" for r in matched)
            self.assertEqual((b7_only,b8_only),(8,33))
            with contextlib.redirect_stdout(io.StringIO()) as f:
                display_model(folder,"B7")
                display_comparison(folder)
                display_hallucination(folder,"B8")
            output=f.getvalue()
            self.assertIn("QUALITY / SIMILARITY PROXIES",output)
            self.assertIn("B1–B8 / 132-QUESTION EVALUATION COMPARISON",output)
            self.assertIn("HALLUCINATION DETECTION EVALUATION",output)
            self.assertIn("N/A",output)

    def test_classification_uses_only_approved_gold_and_observed_predictions(self):
        entries=[
            {"is_hallucination_gold":False,"hallucination_detector_flag":False},
            {"is_hallucination_gold":False,"hallucination_detector_flag":True},
            {"is_hallucination_gold":True,"hallucination_detector_flag":False},
            {"is_hallucination_gold":True,"hallucination_detector_flag":True},
            {"is_hallucination_gold":None,"hallucination_detector_flag":True},
        ]
        c=_classification("B7",entries)
        self.assertEqual([c[k] for k in ("tn","fp","fn","tp")],[1,1,1,1])
        self.assertEqual(c["classification_support"],4)
        self.assertEqual(c["precision"],0.5)
        self.assertEqual(c["recall"],0.5)
        self.assertEqual(c["f1"],0.5)
        self.assertTrue(c["classification_status"].startswith("partial"))

    def test_gold_rejects_fabricated_approved_labels(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            file=Path(temp)/"fake_gold.csv"
            file.write_text(
                "model,query_id,is_hallucination_gold,review_status,reviewer_id,"
                "reviewer_qualification,independence_declaration,"
                "verified_source_evidence,review_notes\n"
                "B7,PILOT_001,Yes,approved,,,,,\n",encoding="utf-8")
            with self.assertRaisesRegex(ValueError,"missing reviewer_id"):
                _read_gold(file,{"PILOT_001"})

if __name__=="__main__":
    unittest.main()
