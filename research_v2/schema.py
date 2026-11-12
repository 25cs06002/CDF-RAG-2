from typing import Literal
from pydantic import BaseModel, Field, ConfigDict


class EvidenceDocument(BaseModel):
    document_id: str
    text: str
    source_id: str | None = None
    score: float | None = None


class CausalEdge(BaseModel):
    cause: str
    effect: str
    relation: str = "CAUSES"
    evidence_ids: list[str] = Field(default_factory=list)


class CausalPath(BaseModel):
    nodes: list[str] = Field(min_length=2)
    evidence_ids: list[str] = Field(default_factory=list)


class StageTimings(BaseModel):
    refinement_ms: float | None = Field(default=None, ge=0)
    retrieval_ms: float | None = Field(default=None, ge=0)
    rewriting_ms: float | None = Field(default=None, ge=0)
    generation_ms: float | None = Field(default=None, ge=0)
    validation_ms: float | None = Field(default=None, ge=0)
    total_ms: float | None = Field(default=None, ge=0)


class PipelineRecord(BaseModel):

    model_config = ConfigDict(extra="forbid")

    # Experiment identity
    run_id: str
    query_id: str
    pipeline_id: str
    graph_version: str | None = None

    # Query processing
    original_query: str
    refined_query: str | None = None
    refinement_action: str | None = None
    matched_concept: str | None = None

    # Retrieved evidence
    retrieved_documents: list[EvidenceDocument] = Field(
        default_factory=list
    )
    causal_edges: list[CausalEdge] = Field(
        default_factory=list
    )
    causal_paths: list[CausalPath] = Field(
        default_factory=list
    )

    # Generation
    rewritten_knowledge: str | None = None
    final_answer: str | None = None

    # Verification
    hallucination_detected: bool | None = None
    validator_passed: bool | None = None
    supported_claims: list[str] = Field(
        default_factory=list
    )
    unsupported_claims: list[str] = Field(
        default_factory=list
    )

    # Adaptive feedback
    feedback_iterations: int = Field(default=0, ge=0)
    action_history: list[str] = Field(default_factory=list)

    # Performance
    timings: StageTimings = Field(
        default_factory=StageTimings
    )

    # Reproducibility
    generator_model: str | None = None
    embedding_model: str | None = None
    random_seed: int | None = None

    # Execution
    status: Literal["ok", "abstained", "error"] = "ok"
    error_message: str | None = None
