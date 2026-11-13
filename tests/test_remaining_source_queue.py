"""Phase 2F candidate source queue checks (not medical truth tests)."""

import json
import tempfile
import unittest
from pathlib import Path

from research_v2.remaining_source_queue import (
    BASE_SOURCES, EXTRA_SOURCES, PILOT, SOURCE_MAP,
    build_remaining_queue, export_remaining_queue,
)


class TestRemainingSourceQueue(unittest.TestCase):
    def rows(self, path):
        return [
            json.loads(line) for line in Path(path).read_text(
                encoding="utf-8"
            ).splitlines() if line.strip()
        ]

    def write(self, tmp, name, records):
        path = Path(tmp) / name
        path.write_text(
            "".join(json.dumps(item) + "\n" for item in records),
            encoding="utf-8",
        )
        return path

    def test_70_questions_all_six_query_types(self):
        rows, profile = build_remaining_queue()
        self.assertEqual(len(rows), 70)
        self.assertEqual(profile["remaining_questions"], 70)
        self.assertEqual(profile["pilot_questions"], 100)
        self.assertEqual(profile["query_type_counts"], {
            "ambiguous": 8, "comparative": 12, "indirect": 15,
            "multi_hop": 15, "risk_factor": 12, "unsupported": 8,
        })

    def test_candidate_coverage_does_not_imply_gold(self):
        rows, profile = build_remaining_queue()
        self.assertEqual(profile["questions_with_candidate_links"], 69)
        self.assertEqual(profile["questions_without_candidate_links"], 1)
        self.assertEqual(profile["questions_needing_additional_sources"], 18)
        self.assertEqual(profile["unique_candidate_sources_used"], 23)
        self.assertEqual(profile["candidate_source_links"], 89)
        self.assertFalse(profile["human_review_completed"])
        self.assertFalse(profile["causal_paths_verified"])
        self.assertFalse(profile["medical_gold_created"])
        for item in rows:
            self.assertFalse(item["human_review_completed"])
            self.assertFalse(item["gold_annotation_created"])
            self.assertNotIn("gold_answer", item)
            self.assertNotIn("gold_causal_edges", item)

    def test_unsupported_question_with_no_source_explicit(self):
        rows, _ = build_remaining_queue()
        q = next(item for item in rows if item["query_id"] == "PILOT_093")
        self.assertEqual(q["candidate_sources"], [])
        self.assertTrue(q["needs_additional_sources"])
        self.assertEqual(q["source_bundle_status"], "no_candidate_source")

    def test_multi_hop_requires_independent_link_checks(self):
        rows, _ = build_remaining_queue()
        paths = [item for item in rows if item["query_type"] == "multi_hop"]
        self.assertEqual(len(paths), 15)
        self.assertTrue(all(item["needs_additional_sources"] for item in paths))
        self.assertTrue(all(
            item["source_bundle_status"] == "incomplete_candidate_bundle"
            for item in paths
        ))

    def test_source_urls_are_present_but_not_verified_claims(self):
        rows, _ = build_remaining_queue()
        for item in rows:
            for source in item["candidate_sources"]:
                self.assertTrue(source["url"].startswith("https://"))
                self.assertIn(source["candidate_origin"], {
                    "supplemental_link", "original_topic_registry"
                })

    def test_export_files_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "worklist"
            report = export_remaining_queue(out)
            self.assertEqual(report["remaining_questions"], 70)
            self.assertTrue((out / "remaining70_source_queue.jsonl").is_file())
            self.assertTrue((out / "remaining70_source_profile.json").is_file())
            self.assertEqual(
                len(self.rows(out / "remaining70_source_queue.jsonl")), 70
            )
            with self.assertRaises(FileExistsError):
                export_remaining_queue(out)

    def test_unknown_candidate_source_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = self.rows(SOURCE_MAP)
            rows[0]["candidate_source_ids"] = ["FAKE_SOURCE"]
            path = self.write(tmp, "mapping.jsonl", rows)
            with self.assertRaisesRegex(ValueError, "Unknown supplemental source ID"):
                build_remaining_queue(source_map_path=path)

    def test_wrong_question_route_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = self.rows(SOURCE_MAP)
            rows[0]["review_route"] = "population_risk"
            path = self.write(tmp, "mapping.jsonl", rows)
            with self.assertRaisesRegex(ValueError, "Review route does not match"):
                build_remaining_queue(source_map_path=path)

    def test_duplicate_question_mapping_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = self.rows(SOURCE_MAP)
            rows[1]["query_id"] = rows[0]["query_id"]
            path = self.write(tmp, "mapping.jsonl", rows)
            with self.assertRaisesRegex(ValueError, "Duplicate source mapping query"):
                build_remaining_queue(source_map_path=path)

    def test_missing_question_mapping_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = self.rows(SOURCE_MAP)[:-1]
            path = self.write(tmp, "mapping.jsonl", rows)
            with self.assertRaisesRegex(ValueError, "Expected 70"):
                build_remaining_queue(source_map_path=path)

    def test_empty_source_for_indirect_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = self.rows(SOURCE_MAP)
            rows[0]["candidate_source_ids"] = []
            path = self.write(tmp, "mapping.jsonl", rows)
            with self.assertRaisesRegex(ValueError, "line 1"):
                build_remaining_queue(source_map_path=path)

    def test_multi_hop_cannot_claim_no_more_evidence_needed(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = self.rows(SOURCE_MAP)
            item = next(item for item in rows if item["query_id"] == "PILOT_046")
            item["needs_additional_sources"] = False
            path = self.write(tmp, "mapping.jsonl", rows)
            with self.assertRaisesRegex(ValueError, "Multi-hop requires"):
                build_remaining_queue(source_map_path=path)

    def test_comparison_without_two_sources_or_gap_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = self.rows(SOURCE_MAP)
            item = next(item for item in rows if item["query_id"] == "PILOT_069")
            item["needs_additional_sources"] = False
            path = self.write(tmp, "mapping.jsonl", rows)
            with self.assertRaisesRegex(ValueError, "Comparison requires"):
                build_remaining_queue(source_map_path=path)

    def test_cannot_falsely_claim_human_review(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = self.rows(SOURCE_MAP)
            rows[0]["human_review_completed"] = True
            path = self.write(tmp, "mapping.jsonl", rows)
            with self.assertRaisesRegex(ValueError, "line 1"):
                build_remaining_queue(source_map_path=path)

    def test_duplicate_url_across_registries_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            extras = self.rows(EXTRA_SOURCES)
            bases = self.rows(BASE_SOURCES)
            extras[0]["url"] = bases[0]["url"]
            path = self.write(tmp, "extra.jsonl", extras)
            with self.assertRaisesRegex(ValueError, "Duplicate cross-registry source URL"):
                build_remaining_queue(extra_sources_path=path)


if __name__ == "__main__":
    unittest.main()
