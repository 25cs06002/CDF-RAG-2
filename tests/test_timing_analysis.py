import json
import tempfile
import unittest

from pathlib import Path

from research_v2.schema import (
    PipelineRecord,
    StageTimings,
)

from research_v2.timing_analysis import (
    analyze_run,
)


class TestTimingAnalysis(unittest.TestCase):

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)

        self.root = Path(self.temp.name)

        records = [
            PipelineRecord(
                run_id="test_run",
                query_id=f"Q{i}",
                pipeline_id="mock",
                original_query="Test",
                status="ok",
                timings=StageTimings(
                    total_ms=i * 1000
                ),
            )
            for i in (1, 2)
        ]

        (
            self.root / "records.jsonl"
        ).write_text(
            "".join(
                r.model_dump_json() + "\n"
                for r in records
            ),
            encoding="utf-8",
        )

        self.write_json(
            "summary.json",
            {
                "run_id": "test_run",
                "pipeline_id": "mock",
                "total_queries": 2,
                "wall_time_seconds": 5.0,
            },
        )

        self.write_json(
            "config_provenance.json",
            {
                "graph_checked_per_query": True
            },
        )

        self.events = [
            {
                "query_index": 1,
                "phase": "before",
                "status": "pass",
                "check_elapsed_ms": 100.0,
            },
            {
                "query_index": 1,
                "phase": "after",
                "status": "pass",
                "check_elapsed_ms": 200.0,
            },
            {
                "query_index": 2,
                "phase": "before",
                "status": "pass",
                "check_elapsed_ms": 300.0,
            },
            {
                "query_index": 2,
                "phase": "after",
                "status": "pass",
                "check_elapsed_ms": 400.0,
            },
        ]

        self.write_events()

    def write_json(self, name, obj):
        (self.root / name).write_text(
            json.dumps(obj) + "\n",
            encoding="utf-8",
        )

    def write_events(self):
        (
            self.root / "graph_checks.jsonl"
        ).write_text(
            "".join(
                json.dumps(event) + "\n"
                for event in self.events
            ),
            encoding="utf-8",
        )

    def test_separates_pipeline_and_graph_timing(self):
        summary, rows = analyze_run(
            self.root
        )

        self.assertEqual(
            rows[0]["pipeline_reported_ms"],
            1000,
        )

        self.assertEqual(
            rows[0][
                "graph_check_read_ms_lower_bound"
            ],
            300,
        )

        self.assertEqual(
            summary["graph_check_read_ms"][
                "total_ms"
            ],
            1000,
        )

        self.assertEqual(
            summary["successful_pipeline_ms"][
                "total_ms"
            ],
            3000,
        )

    def test_graph_monitoring_fraction(self):
        summary, _ = analyze_run(
            self.root
        )

        self.assertAlmostEqual(
            summary[
                "graph_read_fraction_of_run_wall_lower_bound"
            ],
            0.2,
        )

    def test_missing_post_check_rejected(self):
        self.events.pop(1)
        self.write_events()

        with self.assertRaisesRegex(
            ValueError,
            "Missing post-query",
        ):
            analyze_run(self.root)

    def test_duplicate_phase_rejected(self):
        self.events.append(
            dict(self.events[0])
        )
        self.write_events()

        with self.assertRaisesRegex(
            ValueError,
            "Duplicate graph-check phase",
        ):
            analyze_run(self.root)

    def test_missing_expected_journal_rejected(self):
        (
            self.root / "graph_checks.jsonl"
        ).unlink()

        with self.assertRaisesRegex(
            ValueError,
            "journal is missing",
        ):
            analyze_run(self.root)

    def test_old_unmonitored_run_supported(self):
        (
            self.root / "graph_checks.jsonl"
        ).unlink()

        self.write_json(
            "config_provenance.json",
            {
                "graph_checked_per_query": False
            },
        )

        summary, rows = analyze_run(
            self.root
        )

        self.assertFalse(
            summary["graph_monitoring_present"]
        )

        self.assertIsNone(
            rows[0][
                "graph_check_read_ms_lower_bound"
            ]
        )


if __name__ == "__main__":
    unittest.main()
