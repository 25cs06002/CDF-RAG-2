"""Post-hoc G0 target-edge reachability diagnostic. NOT a deployable model.
Uses draft benchmark labels; never feed this result into B5/B6 inference.
"""
import argparse
import csv
import json
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from .experiment_runner import PROJECT_ROOT, file_sha256
from .pipelines.baselines import G0Corpus
from .presentation_matrix import WORKSPACE

UNSAFE_TYPES = {"unsupported", "ambiguous"}


def assess(rows, graph):
    if len(rows) != 64 or len({r["query_id"] for r in rows}) != 64:
        raise ValueError("Expected 64 unique graph-gap rows")
    nodes = {" ".join(n.lower().split()): n for n in graph.nodes}
    output = []
    for row in rows:
        target = (row["draft_target"] or "").strip()
        actual = nodes.get(" ".join(target.lower().split()))
        incoming = len(graph.reverse[actual]) if actual else 0
        uncertain = row["query_type"] in UNSAFE_TYPES or not target
        if str(row["B6_evidence"]).lower() != "false":
            raise ValueError("This input is not an original B6 retrieval gap")
        status = (
            "UNDERDETERMINED_NO_FORCED_DIAGNOSIS" if uncertain else
            "TARGET_ABSENT_G0" if actual is None else
            "TARGET_WITHOUT_INCOMING_EDGE" if not incoming else
            "SNAPSHOT_REACHABLE_ONLY_WITH_LEAKED_DRAFT_TARGET"
        )
        output.append({
            "query_id": row["query_id"], "query_type": row["query_type"],
            "draft_target": target, "target_exact_in_G0": bool(actual),
            "incoming_snapshot_edges": incoming, "diagnostic_status": status,
            "source_label_leakage": True, "requires_manual_review": True
        })
    counts = dict(Counter(r["diagnostic_status"] for r in output))
    return output, {
        "queries": len(output), "counts": counts,
        "draft_target_leakage": True, "is_model_result": False,
        "medical_accuracy": False, "verified_medical_evidence": False,
        "warning": "Forced target is draft query metadata. Reachable G0 edges are not proof of answerability."
    }


def run(path=None):
    if path is None:
        choices = sorted((WORKSPACE / "g1_review").glob("gap_audit_*/gap_classification.csv"))
        if not choices:
            raise FileNotFoundError("Run: bash scripts/run_g0_gap_audit.sh")
        path = choices[-1]
    path = Path(path).resolve()
    if not path.is_relative_to(PROJECT_ROOT):
        raise ValueError("Input must remain inside repository")
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    result, summary = assess(rows, G0Corpus())
    name = "target_ceiling_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "_" + uuid.uuid4().hex[:6]
    out = WORKSPACE / "g1_review" / name
    out.mkdir(parents=True, exist_ok=False)
    with (out / "target_reachability.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(result[0]))
        writer.writeheader()
        writer.writerows(result)
    summary["source_sha256"] = file_sha256(path)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(out), **summary}, indent=2))
    return out, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path)
    args = parser.parse_args()
    if Path.cwd().resolve() != PROJECT_ROOT:
        parser.error("Run from CDF-RAG-2 root")
    run(args.audit)


if __name__ == "__main__":
    main()
