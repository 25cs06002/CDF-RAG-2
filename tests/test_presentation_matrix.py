"""Offline checks for the complete B1-B8 presentation orchestration.

Run: python3 -m unittest tests.test_presentation_matrix -v
"""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from research_v2.experiment_runner import BenchmarkQuestion
from research_v2.presentation_matrix import (
    EXPECTED_PIPELINE_IDS, MODEL_IDS, PILOT,
    choose_models, percentile, summarize, validate_inputs,
)
from research_v2.schema import (
    CausalEdge, EvidenceDocument, PipelineRecord, StageTimings
)
from research_v2 import presentation_g0_reference as ref


class PresentationMatrixTests(unittest.TestCase):
    def test_exact_eight_ids(self):
        self.assertEqual(MODEL_IDS, tuple(f"B{i}" for i in range(1, 9)))
        self.assertEqual(len(set(EXPECTED_PIPELINE_IDS.values())), 8)
        self.assertEqual(choose_models("all"), list(MODEL_IDS))
        self.assertEqual(choose_models("B1,B4,B8"), ["B1", "B4", "B8"])

    def test_invalid_model_selection(self):
        for value in ("B1,B1", "B9", "B1,", ""):
            with self.subTest(value=value), self.assertRaises(ValueError):
                choose_models(value)

    def test_percentile_and_empty(self):
        self.assertIsNone(percentile([], 0.95))
        self.assertEqual(percentile([1, 2, 3], 0.5), 2)
        self.assertAlmostEqual(percentile([1, 2, 3], 0.95), 2.9)

    def test_report_handles_errors_abstentions_and_unmeasured_stages(self):
        inputs = [
            PipelineRecord(
                run_id="t", query_id="A", pipeline_id="bm25_rag",
                original_query="What causes X?", status="ok",
                retrieved_documents=[EvidenceDocument(document_id="D", text="evidence")],
                causal_edges=[CausalEdge(cause="A", effect="X")],
                timings=StageTimings(retrieval_ms=10, total_ms=100),
            ),
            PipelineRecord(
                run_id="t", query_id="B", pipeline_id="bm25_rag",
                original_query="What causes Y?", status="abstained",
                timings=StageTimings(retrieval_ms=12, total_ms=200),
            ),
            PipelineRecord(
                run_id="t", query_id="C", pipeline_id="bm25_rag",
                original_query="What causes Z?", status="error",
                timings=StageTimings(total_ms=300),
            ),
        ]
        metadata = {
            "A": {"query_type": "direct"},
            "B": {"query_type": "direct"},
            "C": {"query_type": "multi_hop"},
        }
        result = summarize(inputs, metadata, elapsed=2.0)
        self.assertEqual((result["success"], result["abstained"], result["errors"]), (1, 1, 1))
        self.assertAlmostEqual(result["success_rate"], 1 / 3)
        self.assertAlmostEqual(result["mean_latency_s"], 0.2)
        self.assertAlmostEqual(result["throughput_queries_per_s_including_init"], 1.5)
        self.assertEqual(result["by_category"]["direct"]["total"], 2)
        self.assertEqual(result["by_category"]["multi_hop"]["errors"], 1)
        self.assertIsNone(result["mean_generation_s"])

    def test_original_100_metadata_can_be_exported_without_gold_leakage(self):
        rows = [json.loads(line) for line in PILOT.read_text(encoding="utf-8").splitlines() if line.strip()]
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory) / "queries.jsonl"
            file.write_text(
                "".join(json.dumps({"query_id": row["query_id"], "query": row["query"]}) + "\n" for row in rows),
                encoding="utf-8",
            )
            questions, metadata = validate_inputs(file)
            self.assertEqual(len(questions), 100)
            self.assertEqual(len(metadata), 100)
            rows[0]["query"] = "This query differs from frozen metadata"
            file.write_text(
                "".join(json.dumps({"query_id": row["query_id"], "query": row["query"]}) + "\n" for row in rows),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "Question text mismatch"):
                validate_inputs(file)

    def test_g0_reference_subset_is_structural_not_medical(self):
        metadata = {
            row["query_id"]: row
            for row in (
                json.loads(line) for line in PILOT.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
        }
        questions = [BenchmarkQuestion(query_id=i, query=v["query"]) for i, v in metadata.items()]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "pilot"
            ref_dir = root / "metrics/g0_structural_pilot_v1"
            with (
                patch.object(ref, "validate_inputs", return_value=(questions, metadata)),
                patch.object(ref, "WORKSPACE", root),
                patch.object(ref, "REFERENCE_DIR", ref_dir),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                ref.prepare()
            scope = json.loads((ref_dir / "scope.json").read_text(encoding="utf-8"))
            self.assertEqual(scope["reference_basis"], "graph_snapshot")
            self.assertFalse(scope["clinical_accuracy_verified"])
            self.assertGreater(scope["structurally_mapped_subset"], 0)
            self.assertLess(scope["structurally_mapped_subset"], 100)


if __name__ == "__main__":
    unittest.main()
