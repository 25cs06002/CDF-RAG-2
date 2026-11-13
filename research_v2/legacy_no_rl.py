"""
Compatibility adapter for the original CDF-RAG No-RL pipeline.

The legacy algorithm is not modified.
"""

import json
import re
import time

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from .base_pipeline import BaseRAGPipeline, PipelineRequest
from .schema import (
    PipelineRecord,
    CausalEdge,
    CausalPath,
    StageTimings,
)


def execute_legacy_no_rl(query: str) -> Mapping[str, Any]:
    """
    Execute the original No-RL LangGraph.

    Imports are deliberately delayed until execution.
    Neo4j connections are closed even when execution fails.
    """

    project_root = Path(__file__).resolve().parents[1]
    load_dotenv(
        dotenv_path=project_root / ".env",
        override=False,
    )

    from config import CONFIG
    from agents.document_causal_retriever_no_rl import (
        CausalGraphRetriever,
        graph as legacy_graph,
    )

    if not CONFIG.get("NEO4J_PASSWORD"):
        raise RuntimeError(
            "NEO4J_PASSWORD is not configured."
        )

    graph_store = CausalGraphRetriever(
        CONFIG["NEO4J_URI"],
        CONFIG["NEO4J_USER"],
        CONFIG["NEO4J_PASSWORD"],
    )

    try:
        final_state = legacy_graph.invoke({
            "query": query,
            "graph": graph_store,
        })

        return final_state

    finally:
        graph_store.close()


def claim_to_string(claim: Any) -> str:
    """
    Preserve legacy claim dictionaries as JSON strings.
    """
    if isinstance(claim, str):
        return claim

    return json.dumps(
        claim,
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    )


def parse_legacy_path(value: str) -> CausalPath:
    """
    Convert a legacy causal path string into a node list.
    """

    nodes = [
        node.strip()
        for node in re.split(r"\s*(?:→|->)\s*", value)
        if node.strip()
    ]

    if len(nodes) < 2:
        raise ValueError(
            f"Invalid legacy causal path: {value!r}"
        )

    return CausalPath(nodes=nodes)


class LegacyNoRLPipeline(BaseRAGPipeline):
    """
    Runs the existing No-RL implementation and maps
    its output into the common PipelineRecord format.

    An optional executor permits isolated unit tests
    without loading Neo4j, Ollama, or large ML models.
    """

    def __init__(
        self,
        executor: Callable[
            [str], Mapping[str, Any]
        ] | None = None,
    ):
        super().__init__(
            pipeline_id="cdf_rag_no_rl_legacy"
        )

        self._live_execution = executor is None
        self._executor = (
            executor
            if executor is not None
            else execute_legacy_no_rl
        )

    def _run(
        self,
        request: PipelineRequest,
    ) -> PipelineRecord:

        start = time.perf_counter()
        state = self._executor(request.query)

        if not isinstance(state, Mapping):
            raise TypeError(
                "Legacy pipeline must return a mapping."
            )

        target = str(
            state.get("matched_concept") or ""
        ).strip()

        direct_causes = (
            state.get("retrieved_docs") or []
        )

        if direct_causes and not target:
            raise ValueError(
                "Direct causes were retrieved, but "
                "matched_concept is missing."
            )

        edges = [
            CausalEdge(
                cause=str(cause),
                effect=target,
                relation="CAUSES",
            )
            for cause in direct_causes
        ]

        paths = [
            parse_legacy_path(str(path))
            for path in (
                state.get("causal_docs") or []
            )
        ]

        supported = [
            claim_to_string(claim)
            for claim in (
                state.get("supported_claims") or []
            )
        ]

        unsupported = [
            claim_to_string(claim)
            for claim in (
                state.get("unsupported_claims") or []
            )
        ]

        # The original No-RL implementation leaves
        # the query text unchanged, but resolves a
        # separate concept for graph retrieval.
        refined_text = (
            state.get("refined_query_text")
            or request.query
        )

        generator_model = None
        embedding_model = None

        if self._live_execution:
            from config import CONFIG

            generator_model = CONFIG.get("LLM_MODEL")
            embedding_model = CONFIG.get("ENCODER_MODEL")

        total_ms = (
            time.perf_counter() - start
        ) * 1000.0

        return PipelineRecord(
            run_id=request.run_id,
            query_id=request.query_id,
            pipeline_id=self.pipeline_id,
            graph_version=request.graph_version,

            original_query=request.query,
            refined_query=str(refined_text),
            refinement_action=(
                state.get("refinement_type") or None
            ),
            matched_concept=target or None,

            # Legacy retrieved_docs contains cause names,
            # not source documents.
            retrieved_documents=[],
            causal_edges=edges,
            causal_paths=paths,

            rewritten_knowledge=(
                state.get("rewritten_knowledge") or None
            ),
            final_answer=(
                state.get("final_response") or None
            ),

            hallucination_detected=(
                bool(state["is_hallucination"])
                if "is_hallucination" in state
                else None
            ),
            validator_passed=(
                bool(state["validator_passed"])
                if "validator_passed" in state
                else None
            ),
            supported_claims=supported,
            unsupported_claims=unsupported,

            feedback_iterations=0,
            action_history=[],

            timings=StageTimings(
                total_ms=total_ms
            ),

            generator_model=generator_model,
            embedding_model=embedding_model,
            random_seed=request.random_seed,

            status="ok",
        )
