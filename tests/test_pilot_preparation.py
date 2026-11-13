"""Tests for the unreviewed 100-question causal medical QA pilot."""

import json
import tempfile
import unittest
from pathlib import Path

from research_v2.medical_benchmark import load_benchmark
from research_v2.pilot_preparation import (
    PILOT_COUNTS, validate_pilot, write_pilot_inputs,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE = (
    PROJECT_ROOT / "data/benchmark/medical_causal_pilot100_draft_v1.jsonl"
)


class TestPilotPreparation(unittest.TestCase):

    def rows(self):
        return [
            json.loads(line)
            for line in SOURCE.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def temporary_source(self, directory, rows):
        path = Path(directory) / "draft.jsonl"
        path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )
        return path

    def test_draft_pilot_has_expected_distribution(self):
        records, profile = validate_pilot(SOURCE)
        self.assertEqual(len(records), 100)
        self.assertEqual(profile["query_type_counts"], PILOT_COUNTS)
        self.assertEqual(profile["review_status_counts"], {"draft": 100})
        self.assertEqual(profile["split_counts"], {"unassigned": 100})
        self.assertEqual(profile["question_family_pair_count"], 15)

    def test_pilot_covers_multiple_medical_domains(self):
        _, profile = validate_pilot(SOURCE)
        self.assertGreaterEqual(len(profile["medical_domain_counts"]), 8)
        self.assertEqual(sum(profile["medical_domain_counts"].values()), 100)

    def test_pilot_does_not_claim_independent_gold(self):
        records = load_benchmark(SOURCE)
        self.assertTrue(all(record.answerable is None for record in records))
        self.assertTrue(all(record.review_status == "draft" for record in records))
        self.assertTrue(all(not record.evidence_sources for record in records))
        self.assertTrue(all(not record.gold_causal_edges for record in records))

    def test_inference_export_contains_questions_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "export"
            profile = write_pilot_inputs(SOURCE, out)
            questions = [
                json.loads(line)
                for line in (out / "questions_only.jsonl").read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
            self.assertEqual(len(questions), 100)
            self.assertTrue(all(set(q) == {"query_id", "query"} for q in questions))
            self.assertFalse(profile["gold_labels_created"])
            self.assertTrue((out / "pilot_profile.json").is_file())

    def test_rejects_wrong_size(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = self.temporary_source(tmp, self.rows()[:-1])
            with self.assertRaisesRegex(ValueError, "requires 100"):
                validate_pilot(source)

    def test_rejects_wrong_type_distribution(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = self.rows()
            rows[0]["query_type"] = "ambiguous"
            source = self.temporary_source(tmp, rows)
            with self.assertRaisesRegex(ValueError, "distribution"):
                validate_pilot(source)

    def test_rejects_premature_answer_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = self.rows()
            rows[0]["answerable"] = True
            source = self.temporary_source(tmp, rows)
            with self.assertRaisesRegex(ValueError, "premature gold"):
                validate_pilot(source)

    def test_rejects_missing_medical_domain(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = self.rows()
            rows[0]["medical_domain"] = None
            source = self.temporary_source(tmp, rows)
            with self.assertRaisesRegex(ValueError, "Missing medical_domain"):
                validate_pilot(source)

    def test_rejects_broken_paraphrase_family(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = self.rows()
            rows[15]["question_family"] = "unpaired_family"
            source = self.temporary_source(tmp, rows)
            with self.assertRaisesRegex(ValueError, "15 paired"):
                validate_pilot(source)

    def test_does_not_overwrite_existing_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "export"
            write_pilot_inputs(SOURCE, out)
            with self.assertRaises(FileExistsError):
                write_pilot_inputs(SOURCE, out)


if __name__ == "__main__":
    unittest.main()
