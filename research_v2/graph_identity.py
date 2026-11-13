"""
Read-only structural identity verification for G0.

Compares a live Neo4j graph against the archived CSV export.
Does not establish medical correctness or database backup integrity.
"""

import argparse
import csv
import hashlib
import json
import os
import sys

from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from .experiment_runner import file_sha256


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def read_csv(path, required_columns):
    with Path(path).open(
        newline="",
        encoding="utf-8",
    ) as stream:
        reader = csv.DictReader(stream)

        if not required_columns.issubset(
            set(reader.fieldnames or [])
        ):
            raise ValueError(
                f"Invalid CSV columns: {path}"
            )

        return list(reader)


def canonical_graph(nodes, edges):
    """
    Canonical graph representation.

    Preserves spelling, case, edge direction and
    duplicate relationships. Ignores Neo4j element IDs.
    """

    nodes = [str(x).strip() for x in nodes]
    edges = [
        (str(a).strip(), str(b).strip())
        for a, b in edges
    ]

    if any(not x for x in nodes):
        raise ValueError("Blank node name")

    if len(set(nodes)) != len(nodes):
        raise ValueError("Duplicate node names")

    names = set(nodes)

    for cause, effect in edges:
        if cause not in names or effect not in names:
            raise ValueError(
                "Edge references an unknown node"
            )

    payload = {
        "nodes": sorted(nodes),
        "edges": sorted(
            [list(edge) for edge in edges]
        ),
    }

    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    return {
        "nodes": nodes,
        "edges": edges,
        "sha256": hashlib.sha256(
            encoded
        ).hexdigest(),
    }


def read_export(
    graph_dir,
    expected_nodes=35,
    expected_edges=37,
):
    graph_dir = Path(graph_dir)

    node_file = graph_dir / "nodes.csv"
    edge_file = graph_dir / "edges.csv"

    node_rows = read_csv(
        node_file,
        {"node_id", "name"},
    )

    edge_rows = read_csv(
        edge_file,
        {"edge_id", "cause", "effect"},
    )

    if len(node_rows) != expected_nodes:
        raise ValueError(
            "Unexpected exported node count"
        )

    if len(edge_rows) != expected_edges:
        raise ValueError(
            "Unexpected exported edge count"
        )

    return canonical_graph(
        [row["name"] for row in node_rows],
        [
            (row["cause"], row["effect"])
            for row in edge_rows
        ],
    )


def validate_live_schema(
    node_rows,
    edge_rows,
):
    """
    G0 was audited as:
    - Every node: :Concept {name}
    - Every relationship: :CAUSES
    - No relationship properties
    """

    for row in node_rows:
        if sorted(row["labels"]) != ["Concept"]:
            raise ValueError(
                "Unexpected live node labels"
            )

        if sorted(
            row["property_keys"]
        ) != ["name"]:
            raise ValueError(
                "Unexpected live node properties"
            )

    for row in edge_rows:
        if row["relation_type"] != "CAUSES":
            raise ValueError(
                "Unexpected relationship type"
            )

        if row["property_keys"]:
            raise ValueError(
                "Unexpected relationship properties"
            )


def read_live_graph():
    """
    Read the live Neo4j graph in one read transaction.
    """

    load_dotenv(
        dotenv_path=PROJECT_ROOT / ".env",
        override=False,
    )

    password = os.getenv("NEO4J_PASSWORD")

    if not password:
        raise RuntimeError(
            "NEO4J_PASSWORD is not configured"
        )

    from neo4j import GraphDatabase, READ_ACCESS

    driver = GraphDatabase.driver(
        os.getenv(
            "NEO4J_URI",
            "bolt://localhost:7687",
        ),
        auth=(
            os.getenv("NEO4J_USER", "neo4j"),
            password,
        ),
    )

    def read_transaction(tx):
        nodes = [
            dict(row)
            for row in tx.run("""
                MATCH (n)
                RETURN
                    labels(n) AS labels,
                    keys(n) AS property_keys,
                    n.name AS name
            """)
        ]

        edges = [
            dict(row)
            for row in tx.run("""
                MATCH (a)-[r]->(b)
                RETURN
                    a.name AS cause,
                    b.name AS effect,
                    type(r) AS relation_type,
                    keys(r) AS property_keys
            """)
        ]

        return nodes, edges

    try:
        with driver.session(
            default_access_mode=READ_ACCESS
        ) as session:
            node_rows, edge_rows = (
                session.execute_read(
                    read_transaction
                )
            )
    finally:
        driver.close()

    validate_live_schema(
        node_rows,
        edge_rows,
    )

    return canonical_graph(
        [row["name"] for row in node_rows],
        [
            (row["cause"], row["effect"])
            for row in edge_rows
        ],
    )


def compare_graphs(expected, actual):
    expected_nodes = Counter(
        expected["nodes"]
    )
    actual_nodes = Counter(
        actual["nodes"]
    )

    expected_edges = Counter(
        map(tuple, expected["edges"])
    )
    actual_edges = Counter(
        map(tuple, actual["edges"])
    )

    def edge_list(counter):
        return [
            list(edge)
            for edge in sorted(
                counter.elements()
            )
        ]

    return {
        "match": (
            expected["sha256"]
            == actual["sha256"]
        ),
        "export_structural_sha256": (
            expected["sha256"]
        ),
        "live_structural_sha256": (
            actual["sha256"]
        ),
        "export_node_count": len(
            expected["nodes"]
        ),
        "live_node_count": len(
            actual["nodes"]
        ),
        "export_edge_count": len(
            expected["edges"]
        ),
        "live_edge_count": len(
            actual["edges"]
        ),
        "missing_nodes": sorted(
            (expected_nodes - actual_nodes)
            .elements()
        ),
        "extra_nodes": sorted(
            (actual_nodes - expected_nodes)
            .elements()
        ),
        "missing_edges": edge_list(
            expected_edges - actual_edges
        ),
        "extra_edges": edge_list(
            actual_edges - expected_edges
        ),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--graph-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )

    args = parser.parse_args()

    exported = read_export(
        args.graph_dir
    )

    live = read_live_graph()

    report = compare_graphs(
        exported,
        live,
    )

    report.update({
        "checked_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "reference": "G0",
        "scope": "Structural graph identity",
        "full_database_backup_verified": False,
        "nodes_csv_sha256": file_sha256(
            args.graph_dir / "nodes.csv"
        ),
        "edges_csv_sha256": file_sha256(
            args.graph_dir / "edges.csv"
        ),
    })

    args.output_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    report_path = (
        args.output_dir
        / "graph_identity_report.json"
    )

    report_path.write_text(
        json.dumps(
            report,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )

    print("=" * 60)
    print("G0 LIVE GRAPH IDENTITY CHECK")
    print("=" * 60)

    print("Export nodes:", report["export_node_count"])
    print("Live nodes:", report["live_node_count"])
    print("Export edges:", report["export_edge_count"])
    print("Live edges:", report["live_edge_count"])

    print(
        "Export fingerprint:",
        report["export_structural_sha256"],
    )

    print(
        "Live fingerprint:",
        report["live_structural_sha256"],
    )

    print(
        "Identity:",
        "MATCH" if report["match"] else "MISMATCH",
    )

    print("Report:", report_path)

    if not report["match"]:
        print(
            "WARNING: Live graph differs from G0."
        )
        sys.exit(1)

    print("\nG0 STRUCTURAL IDENTITY VERIFIED")


if __name__ == "__main__":
    main()
