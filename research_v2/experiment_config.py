"""
Versioned experiment configuration for CDF-RAG research-v2.

Uses JSON and Pydantic. No additional dependencies required.
"""

import json

from pathlib import Path
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
)

from .experiment_runner import (
    PROJECT_ROOT,
    file_sha256,
    load_questions,
)


class ExperimentConfig(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    schema_version: Literal[1]

    experiment_name: str = Field(min_length=1)
    pipeline: Literal["no_rl", "rl"]

    dataset_path: str = Field(min_length=1)
    dataset_sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$"
    )

    graph_version: str = Field(min_length=1)
    random_seed: int | None = None

    limit: int | None = Field(
        default=None,
        ge=1,
    )

    purpose: str = ""

    @field_validator(
        "experiment_name",
        "dataset_path",
        "graph_version",
    )
    @classmethod
    def reject_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError(
                "Configuration value cannot be blank"
            )
        return value


def prepare_experiment(
    config_path: Path,
    project_root: Path = PROJECT_ROOT,
):
    """
    Load and validate a configuration.

    Returns:
        (config, dataset_path, questions)

    Does not execute Neo4j, PPO, or Ollama.
    """

    config_path = Path(config_path).resolve()
    project_root = Path(project_root).resolve()

    data = json.loads(
        config_path.read_text(encoding="utf-8")
    )

    config = ExperimentConfig.model_validate(data)

    relative_dataset = Path(config.dataset_path)

    if relative_dataset.is_absolute():
        raise ValueError(
            "Dataset path must be project-relative"
        )

    dataset_path = (
        project_root / relative_dataset
    ).resolve()

    if not dataset_path.is_relative_to(project_root):
        raise ValueError(
            "Dataset path escapes the project directory"
        )

    actual_hash = file_sha256(dataset_path)

    if actual_hash != config.dataset_sha256:
        raise ValueError(
            "Dataset SHA256 mismatch. "
            "The dataset differs from the "
            "version recorded in the configuration."
        )

    questions = load_questions(dataset_path)

    if config.limit is not None:
        questions = questions[:config.limit]

    return config, dataset_path, questions
