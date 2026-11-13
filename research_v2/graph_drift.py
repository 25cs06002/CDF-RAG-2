"""
Per-query structural graph drift monitoring.

Writes a journal before and after RAG queries.
A mismatch or read failure returns False, causing
the experiment runner to abort.
"""

import json
import re
import time

from datetime import datetime, timezone
from pathlib import Path

from .graph_identity import read_live_graph


class GraphDriftMonitor:

    def __init__(
        self,
        expected_sha256: str,
        reader=None,
    ):
        if not re.fullmatch(
            r"[0-9a-f]{64}",
            expected_sha256,
        ):
            raise ValueError(
                "Invalid expected graph SHA-256"
            )

        self.expected_sha256 = expected_sha256

        self.reader = (
            reader
            if reader is not None
            else read_live_graph
        )

    def __call__(
        self,
        phase: str,
        query_index: int,
        output_dir: Path,
    ) -> bool:

        if phase not in {"before", "after"}:
            raise ValueError(
                f"Invalid graph-check phase: {phase}"
            )

        started = time.perf_counter()

        entry = {
            "checked_utc": datetime.now(
                timezone.utc
            ).isoformat(),
            "query_index": query_index,
            "phase": phase,
            "expected_sha256": (
                self.expected_sha256
            ),
            "observed_sha256": None,
            "status": None,
        }

        try:
            actual = self.reader()
            observed = actual["sha256"]

            entry["observed_sha256"] = observed

            entry["status"] = (
                "pass"
                if observed == self.expected_sha256
                else "drift"
            )

        except Exception as exc:
            # Do not record exception messages that
            # could include connection credentials.
            entry["status"] = "error"
            entry["error_type"] = (
                type(exc).__name__
            )

        entry["check_elapsed_ms"] = (
            time.perf_counter() - started
        ) * 1000

        journal = (
            Path(output_dir)
            / "graph_checks.jsonl"
        )

        with journal.open(
            "a",
            encoding="utf-8",
        ) as output:
            output.write(
                json.dumps(entry) + "\n"
            )

        return entry["status"] == "pass"
