"""
Compatibility adapter for the original PPO-based CDF-RAG.

Reuses the No-RL result conversion logic.
Does not modify or retrain the legacy PPO model.
"""

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from .legacy_no_rl import LegacyNoRLPipeline


def execute_legacy_rl(query: str) -> Mapping[str, Any]:
    """
    Execute the original PPO + Neo4j + Ollama pipeline.

    Imports are delayed until the live experiment runs.
    """

    root = Path(__file__).resolve().parents[1]

    # The original PPO loader uses a relative model path.
    if Path.cwd().resolve() != root:
        raise RuntimeError(
            "Run the legacy RL pipeline from the "
            "CDF-RAG-2 project root."
        )

    checkpoint = root / "ppo_query_refiner_2.zip"

    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"PPO checkpoint not found: {checkpoint}"
        )

    load_dotenv(
        dotenv_path=root / ".env",
        override=False,
    )

    from config import CONFIG

    if not CONFIG.get("NEO4J_PASSWORD"):
        raise RuntimeError(
            "NEO4J_PASSWORD is not configured."
        )

    from agents.document_causal_retriever_rl import (
        CausalGraphRetriever,
        graph as legacy_graph,
    )

    graph_store = CausalGraphRetriever(
        CONFIG["NEO4J_URI"],
        CONFIG["NEO4J_USER"],
        CONFIG["NEO4J_PASSWORD"],
    )

    try:
        result = legacy_graph.invoke({
            "query": query,
            "graph": graph_store,
        })

        return result

    finally:
        graph_store.close()


class LegacyRLPipeline(LegacyNoRLPipeline):
    """
    PPO-based legacy CDF-RAG implementation.

    Uses the same output converter as LegacyNoRLPipeline.
    """

    def __init__(
        self,
        executor: Callable[
            [str], Mapping[str, Any]
        ] | None = None,
    ):
        live_execution = executor is None

        selected_executor = (
            execute_legacy_rl
            if live_execution
            else executor
        )

        super().__init__(
            executor=selected_executor
        )

        self.pipeline_id = "cdf_rag_rl_legacy"
        self._live_execution = live_execution
