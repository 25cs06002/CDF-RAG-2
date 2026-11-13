"""Offline tests for the candidate-source review worklist."""

import json
import tempfile
import unittest
from pathlib import Path

from research_v2.source_review_queue import (
    build_source_review_queue, load_candidate_sources,
    write_source_review_queue,
)


ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / "data/benchmark/medical_causal_pilot100_draft_v1.jsonl"
CANDIDATES = ROOT / "data/benchmark/source_candidates15_v1.jsonl"


class TestSourceReviewQueue(unittest.TestCase):
    def candidate_rows(self):
        return [
            json.loads(line)
            for line in CANDIDATES.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def as_file(self, folder, rows):
        path = Path(folder) / "candidates.jsonl"
        path.write_text(
            "".join(json.dumps(r) + "\n" for r in rows),
            encoding="utf-8",
        )
        return path

    def test_real_queue_is_30_questions_from_15_topics(self):
        queue, profile = build_source_review_queue(PILOT, CANDIDATES)
        self.assertEqual(len(queue), 30)
        self.assertEqual(profile["distinct_source_candidates"], 15)
        self.assertEqual(
            profile["query_type_counts"],
            {"direct": 15, "paraphrased": 15},
        )

    def test_candidate_urls_use_authoritative_sources(self):
        sources = load_candidate_sources(CANDIDATES)
        self.assertTrue(all(
            str(s.url).startswith("https://") for s in sources
        ))
        self.assertTrue(all(
            str(s.url).split("/")[2].endswith(
                (".nih.gov", ".cdc.gov", ".cancer.gov", ".gov")
            )
            for s in sources
        ))

    def test_no_evidence_or_medical_gold_is_created(self):
        queue, profile = build_source_review_queue(PILOT, CANDIDATES)
        self.assertFalse(profile["source_review_completed"])
        self.assertFalse(profile["medical_gold_created"])
        self.assertTrue(all(
            not item["medical_gold_created"]
            and not item["source_review_completed"]
            and item["candidate_source"]["candidate_status"]
                == "awaiting_source_review"
            for item in queue
        ))
        self.assertTrue(all(
            "gold_answer" not in item
            and "gold_causal_edges" not in item
            for item in queue
        ))

    def test_review_queue_export_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "review"
            profile = write_source_review_queue(PILOT, CANDIDATES, dest)
            rows = [
                json.loads(line)
                for line in (dest / "source_review_queue.jsonl")
                .read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(rows), 30)
            self.assertEqual(profile["review_questions"], 30)
            self.assertTrue((dest / "source_review_profile.json").exists())
            with self.assertRaises(FileExistsError):
                write_source_review_queue(PILOT, CANDIDATES, dest)

    def test_missing_topic_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = self.candidate_rows()
            rows[0]["target_concept"] = "missing-topic"
            source = self.as_file(tmp, rows)
            with self.assertRaisesRegex(ValueError, "Missing candidate source"):
                build_source_review_queue(PILOT, source)

    def test_domain_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = self.candidate_rows()
            rows[0]["medical_domain"] = "renal"
            source = self.as_file(tmp, rows)
            with self.assertRaisesRegex(ValueError, "Medical domain mismatch"):
                build_source_review_queue(PILOT, source)

    def test_duplicate_source_ids_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = self.candidate_rows()
            rows[1]["source_id"] = rows[0]["source_id"]
            with self.assertRaisesRegex(ValueError, "Duplicate candidate source ID"):
                load_candidate_sources(self.as_file(tmp, rows))

    def test_duplicate_urls_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = self.candidate_rows()
            rows[1]["url"] = rows[0]["url"]
            with self.assertRaisesRegex(ValueError, "Duplicate candidate source URL"):
                load_candidate_sources(self.as_file(tmp, rows))

    def test_cannot_claim_candidates_are_reviewed(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = self.candidate_rows()
            rows[0]["candidate_status"] = "source_reviewed"
            with self.assertRaisesRegex(ValueError, "line 1"):
                load_candidate_sources(self.as_file(tmp, rows))

    def test_cannot_claim_medical_gold(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = self.candidate_rows()
            rows[0]["verified_medical_gold"] = True
            with self.assertRaisesRegex(ValueError, "line 1"):
                load_candidate_sources(self.as_file(tmp, rows))


if __name__ == "__main__":
    unittest.main()
