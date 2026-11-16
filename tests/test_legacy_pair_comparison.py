"""Offline tests for B7/B8 pairing without loading Neo4j/PPO/Ollama."""
import json
import tempfile
import unittest
from pathlib import Path

from research_v2.legacy_pair_comparison import audit_run, pair_audit, paired_rows
from research_v2.schema import PipelineRecord, StageTimings, CausalEdge


def record(label, qid, evidence, latency):
    return PipelineRecord(
        run_id=label, query_id=qid,
        pipeline_id=("cdf_rag_no_rl_legacy" if label == "B7" else "cdf_rag_rl_legacy"),
        original_query=qid, status="ok",
        causal_edges=[CausalEdge(cause="X", effect="Y")] if evidence else [],
        final_answer="synthetic unit-test response",
        timings=StageTimings(total_ms=latency),
    )


class LegacyPairTests(unittest.TestCase):
    def test_pair_counts_and_paired_latency(self):
        left = [record("B7", f"Q{i}", i % 2 == 0, 100.0) for i in range(100)]
        right = [record("B8", f"Q{i}", i % 3 == 0, 120.0) for i in range(100)]
        rows, metrics = paired_rows(left, right, {
            f"Q{i}": "direct" for i in range(100)
        })
        self.assertEqual(len(rows), 100)
        self.assertEqual(metrics["B7_evidence_yield"], 50)
        self.assertEqual(metrics["B8_evidence_yield"], 34)
        self.assertEqual(metrics["B8_minus_B7_mean_paired_latency_ms"], 20)
        self.assertEqual(metrics["evidence_outcomes"]["both"], 17)
        self.assertEqual(metrics["evidence_outcomes"]["B7_only"], 33)
        self.assertEqual(metrics["evidence_outcomes"]["B8_only"], 17)
        self.assertEqual(metrics["evidence_outcomes"]["neither"], 33)
        self.assertFalse("final_answer" in rows[0])

    def test_pair_rejects_mismatched_query_order(self):
        left = [record("B7", f"Q{i}", True, 1) for i in range(100)]
        right = [record("B8", f"Q{i}", True, 1) for i in range(100)]
        right[0], right[1] = right[1], right[0]
        with self.assertRaisesRegex(ValueError, "order"):
            paired_rows(left, right, {f"Q{i}":"direct" for i in range(100)})

    def test_runtime_shared_checks_required(self):
        checks = {"graph_version": {"expected":"G0","actual":"G0","passed":True},
                  "generation_model_name": {"expected":"local","actual":"local","passed":True}}
        base = {"manifest":{"git_commit":"sha", "input_sha256":"x",
                            "graph_version_label":"G0"},
                "config":{"dataset_sha256":"x","graph_version":"G0",
                          "random_seed":2026,"limit":100,"pipeline":"no_rl"},
                "graph_fingerprint":"graphhash", "runtime_lock_sha256":"lockhash",
                "preflight":{"checks":checks, "ppo_checkpoint_checked":False}}
        other = {**base, "config":{**base["config"],"pipeline":"rl"},
                 "preflight":{"checks":checks,"ppo_checkpoint_checked":True}}
        self.assertEqual(pair_audit(base, other), [])
        self.assertIn("mismatched_graph_fingerprint",
                      pair_audit(base, {**other,"graph_fingerprint":"other"}))
        self.assertIn("rl_checkpoint_not_checked", pair_audit(base, {
            **other, "preflight": {**other["preflight"],"ppo_checkpoint_checked":False}
        }))

    def test_missing_preflight_is_reported_not_silently_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            info, failures = audit_run(
                Path(tmp), [record("B7", f"Q{i}", True, 1) for i in range(100)],
                expected_sha="abc"
            )
            self.assertEqual(info, {})
            self.assertIn("missing_or_invalid_run_provenance:FileNotFoundError", failures)

    def test_audit_all_graph_checks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sha = "a" * 64
            files = {
                "manifest.json": {"input_sha256":"test", "question_count":100,
                                  "graph_version_label":"G0"},
                "summary.json": {"total_queries":100,"errors":0,
                                 "aborted":False,"successful":100},
                "experiment_config.json":{"dataset_sha256":"test", "graph_version":"G0",
                                          "limit":100,"random_seed":2026},
                "runtime_preflight.json":{
                    "status":"pass", "failed_checks":[], "runtime_lock_sha256":"b"*64,
                    "checks":{**{key:{"expected":"x","actual":"x","passed":True}
                                  for key in (
                                    "archived_graph_fingerprint", "export_matches_live",
                                    "nodes_csv_sha256", "edges_csv_sha256",
                                    "generation_model_name", "generation_model_digest",
                                    "refinement_model_name", "refinement_model_digest",
                                    "embedding_model_name", "embedding_model_digest")},
                              "live_graph_fingerprint":{
                                  "expected":sha,"actual":sha,"passed":True}}
                },
            }
            for name, data in files.items():
                (root / name).write_text(json.dumps(data))
            checks = [{"query_index":i,"phase":phase,"status":"pass",
                       "expected_sha256":sha,"observed_sha256":sha}
                      for i in range(1,101) for phase in ("before","after")]
            (root / "graph_checks.jsonl").write_text(
                "".join(json.dumps(item)+"\n" for item in checks)
            )
            records = [record("B7",f"Q{i}",True,100) for i in range(100)]
            info, failures = audit_run(root, records, expected_sha="test")
            self.assertEqual(failures, [])
            self.assertEqual(info["graph_check_count"], 200)
            checks[50]["status"] = "drift"
            (root / "graph_checks.jsonl").write_text(
                "".join(json.dumps(item)+"\n" for item in checks)
            )
            _, failures = audit_run(root, records, expected_sha="test")
            self.assertIn("graph_drift_or_failed_graph_check", failures)


if __name__ == "__main__":
    unittest.main()
