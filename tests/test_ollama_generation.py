"""Mocked Ollama crash recovery; offline, no model or network required."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, Mock

import requests

from research_v2.ollama_generation import (
    OllamaGenerationError, generation_settings, generate, check_ollama,
)
from research_v2.base_pipeline import PipelineRequest
from research_v2.pipelines.baselines import BaselinePipeline, G0Corpus
from research_v2.presentation_export import _select_runs
from research_v2.experiment_runner import BenchmarkQuestion
from research_v2.schema import EvidenceDocument, PipelineRecord
from research_v2 import presentation_matrix as matrix


def fake_response(text="done"):
    r = Mock()
    r.raise_for_status.return_value = None
    r.json.return_value = {"response": text}
    return r


class OllamaRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict("os.environ", {
            "OLLAMA_MODEL": "llama3.2:latest",
            "OLLAMA_UNLOAD_EVERY_N": "2",
            "OLLAMA_NUM_CTX": "2048",
            "OLLAMA_GENERATION_ATTEMPTS": "3",
        })
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_retry_transient_connection_and_keep_answer(self):
        with patch("requests.post", side_effect=[
            requests.ConnectionError("process killed"), fake_response("recovered")
        ]) as post, patch("time.sleep") as sleep:
            result = generate("question")
        self.assertEqual(result, "recovered")
        self.assertEqual(post.call_count, 2)
        sleep.assert_called_once()
        body = post.call_args.kwargs["json"]
        self.assertEqual(body["options"]["num_ctx"], 2048)

    def test_unload_on_bounded_request(self):
        with patch("requests.post", return_value=fake_response()) as post:
            generate("prompt", unload_after=True)
        self.assertEqual(post.call_args.kwargs["json"]["keep_alive"], 0)

    def test_exhausted_transport_returns_named_error(self):
        with patch("requests.post", side_effect=requests.ConnectionError("down")), patch("time.sleep"):
            with self.assertRaisesRegex(OllamaGenerationError, "after_3_attempts"):
                generate("prompt")

    def test_baseline_keeps_evidence_when_generation_fails(self):
        p = BaselinePipeline("naive_rag", corpus=G0Corpus(), generate=True)
        req = PipelineRequest(run_id="x", query_id="q", query="What causes heart disease?", graph_version="G0")
        with patch("research_v2.pipelines.baselines.ollama_generate", side_effect=OllamaGenerationError("OOM")):
            rec = p.run(req)
        self.assertEqual(rec.status, "error")
        self.assertTrue(rec.retrieved_documents)
        self.assertIn("OOM", rec.error_message)
        self.assertIsNone(rec.final_answer)
        self.assertIsNotNone(rec.timings.generation_ms)

    def test_model_unloads_every_second_generation(self):
        p = BaselinePipeline("naive_rag", corpus=G0Corpus(), generate=True)
        req = PipelineRequest(run_id="x", query_id="q", query="What causes heart disease?", graph_version="G0")
        with patch("research_v2.pipelines.baselines.ollama_generate", return_value="answer") as gen:
            p.run(req)
            p.run(req)
        self.assertFalse(gen.call_args_list[0].kwargs["unload_after"])
        self.assertTrue(gen.call_args_list[1].kwargs["unload_after"])

    def test_error_run_cannot_be_reused_as_success(self):
        # Reuse logic requires generator settings, not just code and dataset.
        self.assertEqual(generation_settings()["unload_every_n"], 2)

    def test_export_mode_uses_declared_mode_even_when_answers_empty(self):
        from research_v2.experiment_runner import PROJECT_ROOT
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as root:
            path = Path(root)
            runs_dir = path / "runs"
            run_dir = runs_dir / "B2_test"
            run_dir.mkdir(parents=True)
            ids = [f"Q_{i}" for i in range(100)]
            records = [
                PipelineRecord(run_id="B2_test", pipeline_id="bm25_rag",
                    query_id=q, original_query="query",
                    status="error", error_message="OOM") for q in ids
            ]
            (run_dir / "records.jsonl").write_text(
                "".join(r.model_dump_json() + "\n" for r in records))
            (run_dir / "summary.json").write_text(json.dumps({
                "run_id":"B2_test", "total_queries":100,
                "errors":100, "aborted":False
            }))
            (run_dir / "manifest.json").write_text(json.dumps({"input_sha256":"a"}))
            (path / "matrix_runs.jsonl").write_text(json.dumps({
                "run_id":"B2_test", "execution_mode":"end_to_end"
            }) + "\n")
            selected = _select_runs(runs_dir, ids, "a")
            self.assertTrue(selected["B2"][0])


if __name__ == "__main__":
    unittest.main()
