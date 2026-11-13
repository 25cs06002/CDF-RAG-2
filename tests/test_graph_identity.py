import tempfile
import unittest

from pathlib import Path

from research_v2.graph_identity import (
    canonical_graph,
    compare_graphs,
    read_export,
    validate_live_schema,
)


class TestGraphIdentity(unittest.TestCase):

    def setUp(self):
        self.nodes = [
            "Factor A",
            "Factor B",
            "Condition X",
        ]

        self.edges = [
            ("Factor A", "Condition X"),
            ("Factor B", "Condition X"),
        ]

    def test_order_does_not_change_fingerprint(self):
        a = canonical_graph(
            self.nodes,
            self.edges,
        )

        b = canonical_graph(
            list(reversed(self.nodes)),
            list(reversed(self.edges)),
        )

        self.assertEqual(
            a["sha256"],
            b["sha256"],
        )

    def test_identical_graph_matches(self):
        a = canonical_graph(
            self.nodes,
            self.edges,
        )

        result = compare_graphs(a, a)

        self.assertTrue(result["match"])
        self.assertEqual(
            result["missing_edges"], []
        )

    def test_changed_edge_detected(self):
        expected = canonical_graph(
            self.nodes,
            self.edges,
        )

        changed = canonical_graph(
            self.nodes,
            [
                ("Factor A", "Condition X"),
                ("Condition X", "Factor B"),
            ],
        )

        result = compare_graphs(
            expected,
            changed,
        )

        self.assertFalse(result["match"])
        self.assertEqual(
            len(result["missing_edges"]), 1
        )
        self.assertEqual(
            len(result["extra_edges"]), 1
        )

        # Total counts are identical, despite
        # the graph being structurally different.
        self.assertEqual(
            result["export_edge_count"],
            result["live_edge_count"],
        )

    def test_changed_node_detected(self):
        expected = canonical_graph(
            self.nodes,
            self.edges,
        )

        changed = canonical_graph(
            [
                "Factor A",
                "Factor B",
                "Condition Y",
            ],
            [
                ("Factor A", "Condition Y"),
                ("Factor B", "Condition Y"),
            ],
        )

        result = compare_graphs(
            expected,
            changed,
        )

        self.assertFalse(result["match"])
        self.assertIn(
            "Condition X",
            result["missing_nodes"],
        )

    def test_unexpected_properties_rejected(self):
        nodes = [{
            "labels": ["Concept"],
            "property_keys": [
                "name",
                "unverified_property",
            ],
            "name": "Factor A",
        }]

        with self.assertRaisesRegex(
            ValueError,
            "node properties",
        ):
            validate_live_schema(nodes, [])

    def test_export_csv_loading(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)

            (folder / "nodes.csv").write_text(
                "node_id,name\n"
                "1,Factor A\n"
                "2,Condition X\n",
                encoding="utf-8",
            )

            (folder / "edges.csv").write_text(
                "edge_id,cause,effect\n"
                "1,Factor A,Condition X\n",
                encoding="utf-8",
            )

            graph = read_export(
                folder,
                expected_nodes=2,
                expected_edges=1,
            )

            self.assertEqual(
                len(graph["nodes"]), 2
            )
            self.assertEqual(
                len(graph["edges"]), 1
            )


if __name__ == "__main__":
    unittest.main()
