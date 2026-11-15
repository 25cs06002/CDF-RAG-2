"""Offline checks for non-gold all-model evidence-yield diagnostics."""
import unittest

from research_v2.experiment_runner import BenchmarkQuestion
from research_v2.query_diagnostics import (
    build_query_diagnostics, build_pairwise, pair_comparability,
)
from research_v2.schema import EvidenceDocument, PipelineRecord, StageTimings


class QueryDiagnosticsTests(unittest.TestCase):
    def questions(self):
        return [
            BenchmarkQuestion(query_id="Q1", query="Causes of heart disease?"),
            BenchmarkQuestion(query_id="Q2", query="Causes of unknown disease?"),
        ]

    def make(self, model, qid, evidence, latency=1.0):
        ids = {
            "B1": "naive_rag", "B2": "bm25_rag",
            "B5": "graph_rag", "B6": "causal_kg_rag",
        }
        return PipelineRecord(
            run_id="test", query_id=qid,
            pipeline_id=ids[model],
            original_query=qid,
            status="ok" if evidence else "abstained",
            retrieved_documents=(
                [EvidenceDocument(document_id="D", text="unverified")]
                if evidence else []
            ),
            timings=StageTimings(total_ms=latency),
        )

    def test_absent_models_not_assumed_to_fail(self):
        qs = self.questions()
        meta = {q.query_id: {"query_type": "direct"} for q in qs}
        records = {
            "B1": {q.query_id: self.make("B1", q.query_id, True) for q in qs},
            "B5": {"Q1": self.make("B5", "Q1", True),
                   "Q2": self.make("B5", "Q2", False)},
            "B6": {"Q1": self.make("B6", "Q1", False),
                   "Q2": self.make("B6", "Q2", False)},
        }
        rows = build_query_diagnostics(qs, meta, records)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["retrieval_review_bucket"], "DIRECTED_RETRIEVAL_GAP")
        self.assertEqual(rows[1]["retrieval_review_bucket"], "G0_SNAPSHOT_COVERAGE_GAP")
        self.assertEqual(rows[0]["B8_status"], "NOT_RUN")
        self.assertIsNone(rows[0]["B8_evidence"])
        self.assertEqual(rows[1]["document_models_returning_evidence"], 1)

    def test_rejects_incomplete_run(self):
        qs = self.questions()
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            build_query_diagnostics(qs, {"Q1": {"query_type": "direct"}},
                                    {"B1": {"Q1": self.make("B1", "Q1", True)}})

    def test_pairs_guard_runtime_and_corpus_equivalence(self):
        base = {
            "has_matrix_journal": True, "code_sha": "git1",
            "dataset_sha256": "input1", "execution_mode": "retrieval_only",
            "query_count": 2, "graph_version": "G0", "aborted": False,
            "corpus_sha256": "doc1", "top_k": 5, "embedding_backend": None,
        }
        settings = {
            "B1": dict(base),
            "B2": dict(base),
            "B5": {**base, "corpus_sha256": None},
            "B6": {**base, "corpus_sha256": None},
        }
        self.assertEqual(pair_comparability("B1", "B2", settings), (True, "document_candidate"))
        self.assertEqual(pair_comparability("B5", "B6", settings), (True, "graph_snapshot"))
        self.assertEqual(pair_comparability("B1", "B5", settings)[1],
                         "different_evidence_families")
        settings["B2"]["corpus_sha256"] = "other"
        self.assertEqual(pair_comparability("B1", "B2", settings)[1],
                         "mismatched_document_corpus")

    def test_paired_outcome_is_not_accuracy(self):
        qs = self.questions()
        meta = {q.query_id: {"query_type": "direct"} for q in qs}
        records = {
            "B5": {"Q1": self.make("B5", "Q1", True, 5),
                   "Q2": self.make("B5", "Q2", False, 10)},
            "B6": {"Q1": self.make("B6", "Q1", False, 2),
                   "Q2": self.make("B6", "Q2", False, 4)},
        }
        rows = build_query_diagnostics(qs, meta, records)
        base = {
            "has_matrix_journal": True, "code_sha": "same",
            "dataset_sha256": "same", "execution_mode": "retrieval_only",
            "query_count": 2, "graph_version": "G0", "aborted": False,
            "top_k": 5,
        }
        summaries, per_query = build_pairwise(rows, {"B5": dict(base), "B6": dict(base)})
        paired = next(x for x in summaries if x["left_model"] == "B5")
        self.assertTrue(paired["comparable"])
        self.assertEqual(paired["left_only"], 1)
        self.assertEqual(paired["neither"], 1)
        self.assertEqual(paired["paired_queries"], 2)
        self.assertEqual(len(per_query), 2)


if __name__ == "__main__":
    unittest.main()
