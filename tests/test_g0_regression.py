import tempfile
import unittest

from pathlib import Path

from research_v2.g0_regression import build_g0_gold


class TestG0Regression(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

        self.root = Path(self.tmp.name)
        self.graph = self.root / "graph"
        self.graph.mkdir()

        (self.graph / "nodes.csv").write_text(
            "node_id,name\n"
            "1,Factor A\n"
            "2,Factor B\n"
            "3,Condition X\n",
            encoding="utf-8",
        )

        (self.graph / "edges.csv").write_text(
            "edge_id,cause,effect\n"
            "1,Factor A,Condition X\n"
            "2,Factor B,Condition X\n",
            encoding="utf-8",
        )

        self.mapping = self.root / "mapping.jsonl"

        self.mapping.write_text(
            '{"query_id":"Q001",'
            '"target_concept":"Condition X"}\n',
            encoding="utf-8",
        )

    def build(self, name="result"):
        return build_g0_gold(
            self.graph,
            self.mapping,
            self.root / name,
            expected_nodes=3,
            expected_edges=2,
        )

    def test_generates_correct_edges(self):
        annotations, provenance = self.build()

        self.assertEqual(len(annotations), 1)

        gold = annotations[0]

        self.assertEqual(
            gold.reference_basis,
            "graph_snapshot",
        )

        self.assertEqual(
            len(gold.causal_edges),
            2,
        )

        self.assertEqual(
            {
                x.cause
                for x in gold.causal_edges
            },
            {"Factor A", "Factor B"},
        )

        self.assertIsNone(
            gold.causal_paths
        )

        self.assertFalse(
            provenance[
                "medical_correctness_verified"
            ]
        )

    def test_unexpected_graph_size_rejected(self):
        with self.assertRaisesRegex(
            ValueError,
            "node count",
        ):
            build_g0_gold(
                self.graph,
                self.mapping,
                self.root / "bad",
            )

    def test_unknown_target_rejected(self):
        self.mapping.write_text(
            '{"query_id":"Q001",'
            '"target_concept":"Missing"}\n',
            encoding="utf-8",
        )

        with self.assertRaisesRegex(
            ValueError,
            "Unknown target",
        ):
            self.build()

    def test_duplicate_query_rejected(self):
        self.mapping.write_text(
            '{"query_id":"Q001",'
            '"target_concept":"Condition X"}\n'
            '{"query_id":"Q001",'
            '"target_concept":"Condition X"}\n',
            encoding="utf-8",
        )

        with self.assertRaisesRegex(
            ValueError,
            "Duplicate query",
        ):
            self.build()


if __name__ == "__main__":
    unittest.main()
