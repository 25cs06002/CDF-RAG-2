from abc import ABC, abstractmethod

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
)

from .schema import PipelineRecord


class PipelineRequest(BaseModel):
    """
    Standard input for every RAG pipeline.

    Gold answers and evaluation labels are deliberately
    excluded to prevent test-label leakage.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    run_id: str = Field(min_length=1)
    query_id: str = Field(min_length=1)
    query: str = Field(min_length=1)

    graph_version: str | None = None
    random_seed: int | None = None

    @field_validator(
        "run_id",
        "query_id",
        "query",
    )
    @classmethod
    def reject_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError(
                "Value cannot be blank or whitespace-only"
            )
        return value


class BaseRAGPipeline(ABC):
    """
    Abstract base class for all experimental pipelines.

    Subclasses implement _run().

    The public run() method checks that the result
    belongs to the correct query and experiment.
    """

    def __init__(self, pipeline_id: str) -> None:

        if (
            not isinstance(pipeline_id, str)
            or not pipeline_id.strip()
        ):
            raise ValueError(
                "pipeline_id must be a nonempty string"
            )

        self.pipeline_id = pipeline_id

    def run(
        self,
        request: PipelineRequest,
    ) -> PipelineRecord:

        if not isinstance(request, PipelineRequest):
            raise TypeError(
                "Expected PipelineRequest"
            )

        result = self._run(request)

        if not isinstance(result, PipelineRecord):
            raise TypeError(
                "Pipeline must return PipelineRecord"
            )

        expected = {
            "run_id": request.run_id,
            "query_id": request.query_id,
            "original_query": request.query,
            "pipeline_id": self.pipeline_id,
        }

        for field, value in expected.items():
            if getattr(result, field) != value:
                raise ValueError(
                    f"Pipeline result mismatch: {field}"
                )

        return result

    @abstractmethod
    def _run(
        self,
        request: PipelineRequest,
    ) -> PipelineRecord:
        """
        Implement the actual pipeline execution.

        Examples:
        - BM25 retrieval
        - Dense vector retrieval
        - Neo4j causal retrieval
        - Hybrid retrieval
        - PPO-based CDF-RAG
        """
        raise NotImplementedError
