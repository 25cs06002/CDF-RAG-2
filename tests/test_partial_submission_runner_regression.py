"""Focused regression tests for the repaired 100/32 query sprint.

No generation, Ollama, Neo4j, PPO, or graph modifications. Uses the
existing frozen candidate corpus and G0 snapshot only.
"""
import unittest
from unittest.mock import patch

from research_v2.base_pipeline import PipelineRequest
from research_v2.partial_submission_sprint import (
    _cohort, MODEL100, MODEL32, Q100, CORPUS,
)
from research_v2.pipelines import create_pipeline
from research_v2.pipelines.baselines import BaselinePipeline, G0Corpus
from research_v2.ollama_generation import OllamaGenerationError
from research_v2.experiment_runner import load_questions


class ArchivedRunsRegression(unittest.TestCase):
    def test_archived_summary_paths_resolve_to_directories(self):
        rows,_=_cohort()
        self.assertEqual({r["model"] for r in rows},set(MODEL100))
        for row in rows:
            self.assertEqual(row["n"],100)
            self.assertEqual(row["n"],(
                row["status_ok"]+row["status_abstained"]+row["status_error"]
            ))
            self.assertFalse(row["run_dir"].endswith("summary.json"))
            self.assertEqual(len(row["source_file_sha256"]),64)


class BaselineRetrievalSmoke(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.query=load_questions(Q100)[0]

    def test_each_B1_to_B6_can_run_retrieval_only_with_LSA(self):
        for model in MODEL32:
            with self.subTest(model=model):
                pipeline=create_pipeline(
                    model,
                    document_corpus=CORPUS if model in MODEL32[:4] else None,
                    top_k=5,generate=False,encoder="lsa",
                )
                record=pipeline.run(PipelineRequest(
                    run_id="REGRESSION",query_id=self.query.query_id,
                    query=self.query.query,graph_version="G0",
                ))
                self.assertIn(record.status,{"ok","abstained"})
                self.assertIsNone(record.error_message)
                self.assertIsNone(record.generator_model)
                self.assertIsNone(record.final_answer)
                self.assertIsNotNone(record.timings.retrieval_ms)

    def test_transport_error_keeps_retrieved_evidence(self):
        graph=G0Corpus()
        # Choose a graph target known to have at least one incoming edge.
        target=graph.edges[0][1]
        pipeline=BaselinePipeline("causal_kg_rag",graph,generate=True)
        with patch(
            "research_v2.pipelines.baselines.ollama_generate",
            side_effect=OllamaGenerationError("synthetic_ollama_unavailable"),
        ):
            record=pipeline.run(PipelineRequest(
                run_id="REGRESSION",query_id="SYNTHETIC",
                query="What causes "+target+"?",graph_version="G0",
            ))
        self.assertEqual(record.status,"error")
        self.assertGreater(len(record.retrieved_documents),0)
        self.assertIn("synthetic_ollama_unavailable",record.error_message)
        self.assertIsNone(record.final_answer)


if __name__=="__main__":
    unittest.main()
