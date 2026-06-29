"""Validated, immutable experiment protocol loading."""

from pathlib import Path
from typing import Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class Preprocessing(BaseModel):
    """Locked preprocessing choices for the v0 student."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    feature_scaling: Literal["scalar_rms"]
    student_fit: Literal["no_intercept"]
    label_ddof: Literal[0]


class Protocol(BaseModel):
    """Validated protocol shared by all stages of an experiment run."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    data_release: str = Field(min_length=1)
    substitutions_url: str = Field(min_length=1)
    zero_shot_scores_url: str = Field(min_length=1)
    metadata_url: str = Field(min_length=1)
    teacher_column: str = Field(min_length=1)
    model: str = Field(min_length=1)

    working_size: int = Field(gt=0)
    n_labeled: int = Field(gt=0)
    n_unlabeled: int = Field(gt=0)
    n_test: int = Field(gt=0)
    q: int = Field(gt=0)

    pseudo_weight: float = Field(gt=0)
    ridge_lambda: float = Field(gt=0)
    damping: float = Field(gt=0)

    seeds: tuple[int, ...] = Field(min_length=1)
    assay_count: int = Field(gt=0)
    max_length: int = Field(gt=0)
    preprocessing: Preprocessing
    analysis_seed: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_cardinalities(self) -> Self:
        """Reject split and selection sizes that cannot fit the working set."""
        active_size = self.n_labeled + self.n_unlabeled + self.n_test
        if active_size > self.working_size:
            raise ValueError(
                "n_labeled + n_unlabeled + n_test must not exceed working_size"
            )
        if self.q > self.n_unlabeled:
            raise ValueError("q must not exceed n_unlabeled")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be distinct")
        return self


def load_protocol(path: Path | str) -> Protocol:
    """Load a YAML protocol and validate it as an immutable model."""
    with Path(path).open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError("protocol YAML must contain a mapping")
    return Protocol.model_validate(payload)
