import csv
import json
import tempfile
import unittest
from pathlib import Path
from research_v2.thesis132_consolidated import ROOT, BASE, FULL, AI, report, csvread

class ConsolidatedTests(unittest.TestCase):
    def test_actual_archived_experiment(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            out=report(BASE,FULL,AI,Path(temp)/"consolidated")
            rows={r["model"]:r for r in csvread(out/"consolidated_B1_B8_132.csv")}
            self.assertEqual(len(rows),8)
            self.assertEqual(rows["B7"]["accepted"],"131")
            self.assertEqual(rows["B8"]["substantive_answers"],"68")
            self.assertEqual(rows["B7"]["tp"],"11")
            self.assertEqual(rows["B8"]["tp"],"14")
            self.assertEqual(rows["B1"]["accuracy_vs_ai"],"")
            self.assertEqual(rows["B1"]["human_gold_accuracy"],"")
            self.assertEqual(rows["B7"]["strict_supported_and_fully_grounded_yield"],
                             rows["B8"]["strict_supported_and_fully_grounded_yield"])
            self.assertEqual(len(csvread(out/"consolidated_per_query_132.csv")),1056)
            self.assertEqual(json.loads((out/"paired_B7_B8_analysis.json").read_text())["question_pairs"],132)
