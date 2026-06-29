"""Validated, immutable experiment protocol loading."""

from pathlib import Path
from typing import Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator


class _UniqueKeySafeLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects duplicate keys at every mapping level."""

    def construct_mapping(
        self,
        node: yaml.MappingNode,
        deep: bool = False,
    ) -> dict[object, object]:
        self.flatten_mapping(node)
        mapping: dict[object, object] = {}
        for key_node, value_node in node.value:
            key: object = self.construct_object(key_node, deep=deep)
            try:
                hash(key)
            except TypeError as error:
                raise yaml.constructor.ConstructorError(
                    "while constructing a mapping",
                    node.start_mark,
                    "found an unhashable key",
                    key_node.start_mark,
                ) from error
            if key in mapping:
                line = key_node.start_mark.line + 1
                raise ValueError(f"duplicate key {key!r} at line {line}")
            mapping[key] = self.construct_object(value_node, deep=deep)
        return mapping


class Preprocessing(BaseModel):
    """Locked preprocessing choices for the v0 student."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        allow_inf_nan=False,
    )

    feature_scaling: Literal["scalar_rms"]
    student_fit: Literal["no_intercept"]
    label_ddof: int = Field(ge=0, le=0, strict=True)


class Protocol(BaseModel):
    """Validated protocol shared by all stages of an experiment run."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        allow_inf_nan=False,
    )

    data_release: str = Field(min_length=1)
    substitutions_url: str = Field(min_length=1)
    zero_shot_scores_url: str = Field(min_length=1)
    metadata_url: str = Field(min_length=1)
    teacher_column: str = Field(min_length=1)
    model: str = Field(min_length=1)

    working_size: int = Field(gt=0, strict=True)
    n_labeled: int = Field(gt=0, strict=True)
    n_unlabeled: int = Field(gt=0, strict=True)
    n_test: int = Field(gt=0, strict=True)
    q: int = Field(gt=0, strict=True)

    pseudo_weight: float = Field(gt=0, strict=True)
    ridge_lambda: float = Field(gt=0, strict=True)
    damping: float = Field(gt=0, strict=True)

    seeds: tuple[StrictInt, ...] = Field(min_length=1)
    assay_count: int = Field(gt=0, strict=True)
    max_length: int = Field(gt=0, strict=True)
    preprocessing: Preprocessing
    analysis_seed: int = Field(ge=0, strict=True)

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
        payload = yaml.load(handle, Loader=_UniqueKeySafeLoader)
    if not isinstance(payload, dict):
        raise ValueError("protocol YAML must contain a mapping")
    return Protocol.model_validate(payload)
