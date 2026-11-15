"""Tests for the presentation workspace (no Neo4j/Ollama required)."""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from research_v2 import presentation_sprint as ps


class TestPresentationSprint(unittest.TestCase):
    def test_plan_contains_all_eight_distinct_pipelines(self):
        self.assertEqual(len(ps.PIPELINES), 8)
        self.assertEqual(len(set(ps.PIPELINES.values())), 8)

    def test_rejects_workspace_outside_repository(self):
        with self.assertRaisesRegex(ValueError, "inside CDF-RAG-2"):
            ps.workspace_path(ps.ROOT.parent / "not_allowed")

    def test_preparation_and_status(self):
        # Temporary workspace must remain below the repository root.
        with tempfile.TemporaryDirectory(prefix="presentation_test_", dir=ps.ROOT) as temp:
            path = Path(temp) / "presentation"
            with contextlib.redirect_stdout(io.StringIO()):
                ps.prepare(path)
            manifest = json.loads(
                (path / "benchmark/manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["questions"], 100)
            self.assertFalse(manifest["medical_gold_verified"])
            self.assertEqual(manifest["graph_counts"], {"nodes": 35, "edges": 37})
            self.assertEqual(len(ps.load_questions(path / "benchmark/questions_only.jsonl")), 100)
            with contextlib.redirect_stdout(io.StringIO()):
                ps.status(path)
            with self.assertRaises(FileExistsError):
                ps.prepare(path)

    def test_preparation_uses_query_only_export(self):
        with tempfile.TemporaryDirectory(prefix="presentation_test_", dir=ps.ROOT) as temp:
            path = Path(temp) / "presentation"
            with contextlib.redirect_stdout(io.StringIO()):
                ps.prepare(path)
            with (path / "benchmark/questions_only.jsonl").open(encoding="utf-8") as handle:
                for line in handle:
                    self.assertEqual(set(json.loads(line)), {"query_id", "query"})


if __name__ == "__main__":
    unittest.main()
