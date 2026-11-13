"""
Build structural regression labels from the exported G0 graph.

Labels measure agreement with G0, NOT medical correctness.
No Neo4j or Ollama connection is required.
"""

import argparse
import csv
import json

from pathlib import Path

from pydantic import BaseModel, Field, ConfigDict

from .evaluation import GoldAnnotation, norm
from .experiment_runner import file_sha256
from .schema import CausalEdge


class TargetMapping(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query_id: str = Field(min_length=1)
    target_concept: str = Field(min_length=1)


def read_csv(path: Path, required: set[str]):
    with path.open(
        newline="",
        encoding="utf-8",
    ) as stream:
        reader = csv.DictReader(stream)

        if not required.issubset(
            set(reader.fieldnames or [])
        ):
            raise ValueError(
                f"Invalid CSV columns: {path}"
            )

        return list(reader)


def build_g0_gold(
    graph_dir: Path,
    mapping_file: Path,
    output_dir: Path,
    expected_nodes: int = 35,
    expected_edges: int = 37,
):
    graph_dir = Path(graph_dir).resolve()
    mapping_file = Path(mapping_file).resolve()
    output_dir = Path(output_dir).resolve()

    node_file = graph_dir / "nodes.csv"
    edge_file = graph_dir / "edges.csv"

    nodes = read_csv(
        node_file,
        {"node_id", "name"},
    )

    edges = read_csv(
        edge_file,
        {"edge_id", "cause", "effect"},
    )

    if len(nodes) != expected_nodes:
        raise ValueError(
            f"Unexpected G0 node count: {len(nodes)}"
        )

    if len(edges) != expected_edges:
        raise ValueError(
            f"Unexpected G0 edge count: {len(edges)}"
        )

    concept_names = {}

    for node in nodes:
        name = (node["name"] or "").strip()

        if not name:
            raise ValueError("Blank concept name")

        key = norm(name)

        if key in concept_names:
            raise ValueError(
                f"Duplicate normalized concept: {name}"
            )

        concept_names[key] = name

    for edge in edges:
        cause = norm(edge["cause"] or "")
        effect = norm(edge["effect"] or "")

        if (
            cause not in concept_names
            or effect not in concept_names
        ):
            raise ValueError(
                "An edge references an unknown concept"
            )

    mappings = []
    seen_queries = set()

    with mapping_file.open(
        encoding="utf-8"
    ) as stream:
        for line in stream:
            if not line.strip():
                continue

            item = TargetMapping.model_validate_json(
                line
            )

            if (
                not item.query_id.strip()
                or not item.target_concept.strip()
            ):
                raise ValueError("Blank mapping value")

            if item.query_id in seen_queries:
                raise ValueError(
                    f"Duplicate query ID: {item.query_id}"
                )

            seen_queries.add(item.query_id)
            mappings.append(item)

    if not mappings:
        raise ValueError("No target mappings")

    annotations = []

    for item in mappings:
        target_key = norm(item.target_concept)

        if target_key not in concept_names:
            raise ValueError(
                f"Unknown target: {item.target_concept}"
            )

        target_name = concept_names[target_key]

        # Deduplicate physical relationships with the
        # same semantic cause/effect identity.
        causes = {}

        for edge in edges:
            if norm(edge["effect"]) == target_key:
                cause_key = norm(edge["cause"])
                causes[cause_key] = (
                    concept_names[cause_key]
                )

        gold_edges = [
            CausalEdge(
                cause=causes[key],
                effect=target_name,
                relation="CAUSES",
            )
            for key in sorted(causes)
        ]

        annotations.append(
            GoldAnnotation(
                query_id=item.query_id,
                reference_basis="graph_snapshot",
                document_ids=None,
                causal_edges=gold_edges,
                causal_paths=None,
            )
        )

    # Only create output after all checks succeed.
    output_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    gold_file = output_dir / "g0_gold.jsonl"

    gold_file.write_text(
        "".join(
            item.model_dump_json() + "\n"
            for item in annotations
        ),
        encoding="utf-8",
    )

    provenance = {
        "reference_basis": "graph_snapshot",
        "medical_correctness_verified": False,
        "graph_version_label": "G0",
        "node_count": len(nodes),
        "edge_count": len(edges),
        "nodes_sha256": file_sha256(node_file),
        "edges_sha256": file_sha256(edge_file),
        "target_mapping_sha256": file_sha256(
            mapping_file
        ),
        "gold_sha256": file_sha256(gold_file),
        "annotated_queries": len(annotations),
        "evaluated_evidence_types": [
            "direct_causal_edges"
        ],
    }

    (
        output_dir / "graph_provenance.json"
    ).write_text(
        json.dumps(
            provenance,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )

    return annotations, provenance


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--graph-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--mapping",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )

    args = parser.parse_args()

    annotations, provenance = build_g0_gold(
        graph_dir=args.graph_dir,
        mapping_file=args.mapping,
        output_dir=args.output_dir,
    )

    print("G0 REGRESSION GOLD GENERATED")
    print("Nodes:", provenance["node_count"])
    print("Edges:", provenance["edge_count"])
    print("Questions:", len(annotations))

    for item in annotations:
        print(
            item.query_id,
            "direct gold edges:",
            len(item.causal_edges or []),
        )

    print("Output:", args.output_dir)


if __name__ == "__main__":
    main()
