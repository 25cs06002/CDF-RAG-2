import unittest

from research_v2.schema import (
    CausalEdge,
    CausalPath,
    EvidenceDocument,
    PipelineRecord,
    StageTimings,
)

from research_v2.evaluation import (
    GoldAnnotation,
    evaluate,
    keys,
    scores,
)


class TestEvaluation(unittest.TestCase):

    def setUp(self):
        self.records = [
            PipelineRecord(
                run_id="demo",
                query_id="Q1",
                pipeline_id="mock",
                original_query="Question 1?",
                status="ok",

                retrieved_documents=[
                    EvidenceDocument(
                        document_id="D1",
                        text="Synthetic",
                    ),
                    EvidenceDocument(
                        document_id="D3",
                        text="Synthetic",
                    ),
                ],

                causal_edges=[
                    CausalEdge(
                        cause="A", effect="X"
                    ),
                    CausalEdge(
                        cause="B", effect="X"
                    ),
                ],

                causal_paths=[
                    CausalPath(nodes=["A", "X"])
                ],

                timings=StageTimings(
                    total_ms=1000
                ),
            ),

            PipelineRecord(
                run_id="demo",
                query_id="Q2",
                pipeline_id="mock",
                original_query="Question 2?",
                status="ok",
                timings=StageTimings(
                    total_ms=800
                ),
            ),
        ]

        self.gold = [
            GoldAnnotation(
                query_id="Q1",
                reference_basis="synthetic",
                document_ids=["D1", "D2"],

                causal_edges=[
                    CausalEdge(
                        cause="A", effect="X"
                    ),
                    CausalEdge(
                        cause="C", effect="X"
                    ),
                ],

                causal_paths=[
                    CausalPath(nodes=["A", "X"]),
                    CausalPath(nodes=["C", "X"]),
                ],
            ),

            GoldAnnotation(
                query_id="Q2",
                reference_basis="synthetic",
                document_ids=[],
                causal_edges=[],
                causal_paths=[],
            ),
        ]

    def test_precision_recall_f1(self):
        summary, rows = evaluate(
            self.records,
            self.gold,
        )

        edge_scores = rows[0]["metrics"]["edges"]

        self.assertEqual(edge_scores["tp"], 1)
        self.assertEqual(edge_scores["fp"], 1)
        self.assertEqual(edge_scores["fn"], 1)

        self.assertAlmostEqual(
            edge_scores["f1"], 0.5
        )

        self.assertAlmostEqual(
            summary["metrics"]["edges"]["macro_f1"],
            0.75,
        )

        self.assertAlmostEqual(
            summary["metrics"]["edges"]["micro_f1"],
            0.5,
        )

    def test_empty_gold_and_prediction(self):
        _, rows = evaluate(
            self.records,
            self.gold,
        )

        self.assertEqual(
            rows[1]["metrics"]["documents"]["f1"],
            1.0,
        )

    def test_paths_and_latency(self):
        summary, rows = evaluate(
            self.records,
            self.gold,
        )

        self.assertAlmostEqual(
            rows[0]["metrics"]["paths"]["f1"],
            2 / 3,
        )

        self.assertAlmostEqual(
            summary["successful_latency_seconds"]["mean"],
            0.9,
        )

    def test_missing_gold_rejected(self):
        with self.assertRaisesRegex(
            ValueError,
            "Missing reference",
        ):
            evaluate(
                self.records,
                self.gold[:1],
            )

    def test_duplicate_record_rejected(self):
        with self.assertRaisesRegex(
            ValueError,
            "Duplicate prediction",
        ):
            evaluate(
                [
                    self.records[0],
                    self.records[0],
                ],
                self.gold,
            )

    def test_unannotated_metric_excluded(self):
        self.gold[0] = self.gold[0].model_copy(
            update={"causal_paths": None}
        )

        summary, rows = evaluate(
            self.records,
            self.gold,
        )

        self.assertNotIn(
            "paths",
            rows[0]["metrics"],
        )

        self.assertEqual(
            summary["metrics"]["paths"]["evaluated_queries"],
            1,
        )

    def test_error_does_not_count_as_success(self):
        self.records[1] = (
            self.records[1].model_copy(
                update={"status": "error"}
            )
        )

        _, rows = evaluate(
            self.records,
            self.gold,
        )

        self.assertEqual(
            rows[1]["metrics"]["edges"]["f1"],
            0,
        )

        self.assertEqual(
            rows[1]["metrics"]["documents"]["f1"],
            0,
        )

    def test_normalization_and_duplicates(self):
        predicted = keys(
            "edges",
            [
                CausalEdge(
                    cause=" Factor A ",
                    effect="Condition X",
                ),
                CausalEdge(
                    cause="factor a",
                    effect="condition x",
                ),
            ],
        )

        gold = keys(
            "edges",
            [
                CausalEdge(
                    cause="factor a",
                    effect="CONDITION X",
                )
            ],
        )

        result = scores(
            predicted,
            gold,
            success=True,
        )

        self.assertEqual(
            result["f1"],
            1.0,
        )


if __name__ == "__main__":
    unittest.main()
