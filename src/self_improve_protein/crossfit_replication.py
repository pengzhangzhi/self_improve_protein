"""Pure contracts for the preregistered untouched crossfit replication."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal, Self, cast

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr, model_validator

from self_improve_protein.analysis import (
    PairwiseSummary,
    exact_sign_flip_pvalue,
    pairwise_summary,
    validate_result_table,
)
from self_improve_protein.config import GitCommit, Protocol
from self_improve_protein.crossfit import CARD_ID as SCREEN_CARD_ID
from self_improve_protein.crossfit import CARD_SHA as SCREEN_CARD_SHA
from self_improve_protein.crossfit_data import CrossfitPoolManifest
from self_improve_protein.data import Sha256Hex
from self_improve_protein.experiment import (
    METHOD_NAMES,
    NUMERICAL_POLICY,
    canonical_protocol_digest,
)
from self_improve_protein.provenance import atomic_write_json

CROSSFIT_PROMOTION_GATE_SCHEMA_ID: Final = (
    "self-improve-protein.crossfit-screen-promotion.v1"
)
_AGGREGATE_TOP_LEVEL_KEYS: Final = frozenset(
    {
        "analysis",
        "card",
        "development_diagnostics",
        "development_long_results",
        "effects",
        "execution",
        "grid",
        "kind",
        "method_table",
        "primary_long_results",
        "promotion",
        "provenance",
        "schema_version",
        "task_diagnostics",
    }
)
_PAIRWISE_KEYS: Final = frozenset(
    {
        "first",
        "second",
        "metric",
        "mean_gain",
        "standard_error",
        "task_wins",
        "task_total",
        "task_win_rate",
        "assay_wins",
        "assay_total",
        "assay_win_rate",
        "exact_sign_flip_pvalue",
        "assay_deltas",
    }
)
_RESULT_KEYS: Final = frozenset(
    {"assay_id", "method", "mse", "ndcg_10pct", "seed", "spearman"}
)


class _FrozenContractModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)


class ScreenPrimarySummary(_FrozenContractModel):
    """Exact immutable crossfit-minus-random summary from the 40-task screen."""

    first: Literal["crossfit"]
    second: Literal["random"]
    metric: Literal["spearman"]
    mean_gain: float
    standard_error: float = Field(ge=0.0)
    task_wins: int = Field(ge=0, le=40, strict=True)
    task_total: Literal[40]
    task_win_rate: float = Field(ge=0.0, le=1.0)
    assay_wins: int = Field(ge=0, le=8, strict=True)
    assay_total: Literal[8]
    assay_win_rate: float = Field(ge=0.0, le=1.0)
    exact_sign_flip_pvalue: float = Field(ge=0.0, le=1.0)
    assay_deltas: tuple[float, ...] = Field(min_length=8, max_length=8)

    @model_validator(mode="after")
    def validate_exact_summary(self) -> Self:
        summary = self.to_pairwise_summary()
        values = np.asarray(summary.assay_deltas, dtype=np.float64)
        expected_standard_error = float(np.std(values, ddof=1) / np.sqrt(8))
        checks = (
            summary.mean_gain == float(np.mean(values)),
            summary.standard_error == expected_standard_error,
            summary.assay_wins == int(np.count_nonzero(values > 0.0)),
            summary.exact_sign_flip_pvalue == exact_sign_flip_pvalue(values),
        )
        if not all(checks):
            raise ValueError("screen primary summary is not internally exact")
        return self

    def to_pairwise_summary(self) -> PairwiseSummary:
        """Return the shared immutable analysis representation."""
        return PairwiseSummary(
            first=self.first,
            second=self.second,
            metric=self.metric,
            mean_gain=self.mean_gain,
            standard_error=self.standard_error,
            task_wins=self.task_wins,
            task_total=self.task_total,
            task_win_rate=self.task_win_rate,
            assay_wins=self.assay_wins,
            assay_total=self.assay_total,
            assay_win_rate=self.assay_win_rate,
            exact_sign_flip_pvalue=self.exact_sign_flip_pvalue,
            assay_deltas=self.assay_deltas,
        )

    @classmethod
    def from_pairwise_summary(cls, summary: PairwiseSummary) -> ScreenPrimarySummary:
        """Validate and freeze one shared paired summary."""
        if not isinstance(summary, PairwiseSummary):
            raise TypeError("summary must be a PairwiseSummary")
        return cls.model_validate(
            {
                "first": summary.first,
                "second": summary.second,
                "metric": summary.metric,
                "mean_gain": summary.mean_gain,
                "standard_error": summary.standard_error,
                "task_wins": summary.task_wins,
                "task_total": summary.task_total,
                "task_win_rate": summary.task_win_rate,
                "assay_wins": summary.assay_wins,
                "assay_total": summary.assay_total,
                "assay_win_rate": summary.assay_win_rate,
                "exact_sign_flip_pvalue": summary.exact_sign_flip_pvalue,
                "assay_deltas": summary.assay_deltas,
            }
        )


class ScreenPromotionGate(_FrozenContractModel):
    """Canonical proof that the locked screen authorized untouched replication."""

    schema_id: Literal["self-improve-protein.crossfit-screen-promotion.v1"]
    schema_version: Literal[1]
    card_id: Literal["crossfit_outer_gradient_v1"]
    card_sha256: Literal[
        "383afd7a5bae9c2ebd6768a112a82980236540fc0f66e3a294ef298961b8596f"
    ]
    base_manifest_sha256: Sha256Hex
    pool_manifest_sha256: Sha256Hex
    screen_aggregate_sha256: Sha256Hex
    protocol_digest: Sha256Hex
    git_commit: GitCommit
    screen_primary_summary: ScreenPrimarySummary
    untouched_assay_ids: tuple[StrictStr, ...] = Field(min_length=26, max_length=26)
    seeds: tuple[StrictInt, ...] = Field(min_length=5, max_length=5)
    promote_to_untouched_replication: Literal[True]

    @model_validator(mode="after")
    def validate_replication_authorization(self) -> Self:
        if (
            len(set(self.untouched_assay_ids)) != 26
            or self.untouched_assay_ids != tuple(sorted(self.untouched_assay_ids))
        ):
            raise ValueError(
                "untouched_assay_ids must be 26 unique lexically ordered IDs"
            )
        if len(set(self.seeds)) != 5:
            raise ValueError("seeds must contain five unique integers")
        summary = self.screen_primary_summary
        if not (
            summary.mean_gain > 0.0
            and summary.task_wins >= 25
            and summary.assay_wins >= 5
        ):
            raise ValueError("screen primary summary does not satisfy promotion rule")
        return self


def _aggregate_mapping(
    value: object,
    *,
    name: str,
    keys: frozenset[str],
) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError(f"screen aggregate {name} schema is invalid")
    return cast(dict[str, object], value)


def _aggregate_list(value: object, *, name: str, length: int) -> list[object]:
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"screen aggregate {name} schema is invalid")
    return value


def _is_lower_hex(value: object, *, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value)
    )


def _pairwise_from_aggregate(value: object, *, name: str) -> PairwiseSummary:
    payload = _aggregate_mapping(value, name=name, keys=_PAIRWISE_KEYS)
    try:
        return PairwiseSummary(**cast(dict[str, Any], payload))
    except (TypeError, ValueError) as error:
        raise ValueError(f"screen aggregate {name} summary is invalid") from error


def _validate_current_screen_aggregate(
    aggregate: dict[str, object],
    *,
    pool_manifest: CrossfitPoolManifest,
    protocol: Protocol,
    pool_manifest_sha256: str,
    git_commit: str,
) -> PairwiseSummary:
    if set(aggregate) != _AGGREGATE_TOP_LEVEL_KEYS:
        raise ValueError("screen aggregate top-level schema is invalid")
    if (
        type(aggregate.get("schema_version")) is not int
        or aggregate.get("schema_version") != 1
        or aggregate.get("kind") != "crossfit_aggregate_result"
    ):
        raise ValueError("screen aggregate identity is invalid")

    analysis = _aggregate_mapping(
        aggregate.get("analysis"),
        name="analysis",
        keys=frozenset({"inference_unit", "metric", "sign_flip"}),
    )
    if analysis != {
        "inference_unit": "assay",
        "metric": "spearman",
        "sign_flip": "exact",
    }:
        raise ValueError("screen aggregate analysis contract is invalid")
    card = _aggregate_mapping(
        aggregate.get("card"),
        name="card",
        keys=frozenset({"id", "sha256"}),
    )
    if card != {"id": SCREEN_CARD_ID, "sha256": SCREEN_CARD_SHA}:
        raise ValueError("screen aggregate card binding is invalid")

    execution = _aggregate_mapping(
        aggregate.get("execution"),
        name="execution",
        keys=frozenset(
            {"bypass", "numerical_policy", "numerical_runtime", "official"}
        ),
    )
    if (
        execution["bypass"] is not None
        or execution["numerical_policy"] != NUMERICAL_POLICY
        or not isinstance(execution["numerical_runtime"], dict)
        or execution["official"] is not True
    ):
        raise ValueError("screen aggregate execution is not official")

    screen_ids = pool_manifest.screen_ids
    seeds = protocol.seeds
    screen_grid = tuple(
        (assay_id, seed) for assay_id in screen_ids for seed in protocol.seeds
    )
    primary_ids = screen_ids[1:]
    primary_grid = tuple(
        (assay_id, seed) for assay_id in primary_ids for seed in protocol.seeds
    )
    if len(screen_ids) != 9 or len(seeds) != 5:
        raise ValueError("screen aggregate requires the exact 9-by-5 screen")
    grid = _aggregate_mapping(
        aggregate.get("grid"),
        name="grid",
        keys=frozenset(
            {
                "development_assay_id",
                "primary_assay_ids",
                "primary_tasks",
                "screen_assay_ids",
                "screen_tasks",
                "seeds",
            }
        ),
    )
    expected_grid: dict[str, object] = {
        "development_assay_id": screen_ids[0],
        "primary_assay_ids": list(primary_ids),
        "primary_tasks": [list(task) for task in primary_grid],
        "screen_assay_ids": list(screen_ids),
        "screen_tasks": [list(task) for task in screen_grid],
        "seeds": list(seeds),
    }
    if grid != expected_grid:
        raise ValueError("screen aggregate grid does not match the locked pool")

    promotion = _aggregate_mapping(
        aggregate.get("promotion"),
        name="promotion",
        keys=frozenset(
            {
                "assay_win_threshold",
                "development_assay_excluded",
                "mean_gain_strictly_positive",
                "promote_to_untouched_replication",
                "task_win_threshold",
                "untouched_assay_ids",
            }
        ),
    )
    if promotion["promote_to_untouched_replication"] is not True:
        raise ValueError("screen aggregate did not promote to untouched replication")
    if (
        type(promotion["assay_win_threshold"]) is not int
        or promotion["assay_win_threshold"] != 5
        or promotion["development_assay_excluded"] is not True
        or promotion["mean_gain_strictly_positive"] is not True
        or type(promotion["task_win_threshold"]) is not int
        or promotion["task_win_threshold"] != 25
        or promotion["untouched_assay_ids"] != list(pool_manifest.untouched_ids)
    ):
        raise ValueError("screen aggregate promotion contract is invalid")

    provenance = _aggregate_mapping(
        aggregate.get("provenance"),
        name="provenance",
        keys=frozenset(
            {
                "base_manifest_sha256",
                "git_commit",
                "pool_manifest_sha256",
                "protocol_digest",
                "task_count",
                "task_manifest",
            }
        ),
    )
    expected_protocol_digest = canonical_protocol_digest(protocol)
    if (
        provenance["base_manifest_sha256"] != pool_manifest.base_manifest_sha256
        or provenance["pool_manifest_sha256"] != pool_manifest_sha256
        or provenance["protocol_digest"] != expected_protocol_digest
        or provenance["git_commit"] != git_commit
        or type(provenance["task_count"]) is not int
        or provenance["task_count"] != 45
        or not _is_lower_hex(provenance["base_manifest_sha256"], length=64)
        or not _is_lower_hex(provenance["pool_manifest_sha256"], length=64)
        or not _is_lower_hex(provenance["protocol_digest"], length=64)
        or not _is_lower_hex(provenance["git_commit"], length=40)
    ):
        raise ValueError("screen aggregate provenance binding is invalid")
    task_manifest = _aggregate_list(
        provenance["task_manifest"],
        name="task manifest",
        length=45,
    )
    task_identities: list[tuple[object, object]] = []
    for raw_task in task_manifest:
        task = _aggregate_mapping(
            raw_task,
            name="task manifest row",
            keys=frozenset({"assay_id", "seed", "sha256"}),
        )
        if not _is_lower_hex(task["sha256"], length=64):
            raise ValueError("screen aggregate task manifest digest is invalid")
        task_identities.append((task["assay_id"], task["seed"]))
    if tuple(task_identities) != screen_grid:
        raise ValueError("screen aggregate task manifest grid is invalid")

    for name, expected_length in (
        ("development_diagnostics", 5),
        ("development_long_results", 30),
        ("method_table", 6),
        ("task_diagnostics", 45),
    ):
        rows = _aggregate_list(aggregate.get(name), name=name, length=expected_length)
        if any(not isinstance(row, dict) for row in rows):
            raise ValueError(f"screen aggregate {name} rows are invalid")

    raw_primary_rows = _aggregate_list(
        aggregate.get("primary_long_results"),
        name="primary results",
        length=240,
    )
    if any(
        not isinstance(row, dict) or set(row) != _RESULT_KEYS
        for row in raw_primary_rows
    ):
        raise ValueError("screen aggregate primary result rows are invalid")
    try:
        primary_results = validate_result_table(
            pd.DataFrame(raw_primary_rows),
            assay_ids=primary_ids,
            seeds=seeds,
            required_methods=(*METHOD_NAMES, "crossfit"),
        )
    except (TypeError, ValueError) as error:
        raise ValueError("screen aggregate primary results are invalid") from error

    effects = _aggregate_mapping(
        aggregate.get("effects"),
        name="effects",
        keys=frozenset(f"crossfit_minus_{method}" for method in METHOD_NAMES),
    )
    summaries: dict[str, PairwiseSummary] = {}
    for second in METHOD_NAMES:
        name = f"crossfit_minus_{second}"
        reported = _pairwise_from_aggregate(effects[name], name=name)
        recomputed = pairwise_summary(
            primary_results,
            first="crossfit",
            second=second,
            metric="spearman",
        )
        if reported != recomputed:
            raise ValueError(f"screen aggregate {name} is not exact")
        summaries[second] = recomputed
    primary = summaries["random"]
    promotes = (
        primary.mean_gain > 0.0
        and primary.task_wins >= 25
        and primary.assay_wins >= 5
    )
    if not promotes:
        raise ValueError("screen aggregate promotion is inconsistent with results")
    return primary


def build_screen_promotion_gate(
    aggregate: dict[str, object],
    *,
    pool_manifest: CrossfitPoolManifest,
    protocol: Protocol,
    pool_manifest_sha256: str,
    screen_aggregate_sha256: str,
    git_commit: str,
) -> ScreenPromotionGate:
    """Validate the official screen aggregate and freeze replication authority."""
    if not isinstance(aggregate, dict):
        raise TypeError("aggregate must be a parsed JSON object")
    if not isinstance(pool_manifest, CrossfitPoolManifest):
        raise TypeError("pool_manifest must be a CrossfitPoolManifest")
    if not isinstance(protocol, Protocol):
        raise TypeError("protocol must be a Protocol")
    replication_grid(pool_manifest, protocol)
    primary = _validate_current_screen_aggregate(
        aggregate,
        pool_manifest=pool_manifest,
        protocol=protocol,
        pool_manifest_sha256=pool_manifest_sha256,
        git_commit=git_commit,
    )
    return ScreenPromotionGate(
        schema_id=CROSSFIT_PROMOTION_GATE_SCHEMA_ID,
        schema_version=1,
        card_id=SCREEN_CARD_ID,
        card_sha256=SCREEN_CARD_SHA,
        base_manifest_sha256=pool_manifest.base_manifest_sha256,
        pool_manifest_sha256=pool_manifest_sha256,
        screen_aggregate_sha256=screen_aggregate_sha256,
        protocol_digest=canonical_protocol_digest(protocol),
        git_commit=git_commit,
        screen_primary_summary=ScreenPrimarySummary.from_pairwise_summary(primary),
        untouched_assay_ids=pool_manifest.untouched_ids,
        seeds=protocol.seeds,
        promote_to_untouched_replication=True,
    )


def _canonical_json_bytes(payload: object) -> bytes:
    return (
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def load_screen_promotion_gate(path: Path | str) -> ScreenPromotionGate:
    """Load one canonical gate while rejecting duplicate keys and byte drift."""
    source = Path(path)
    with source.open(encoding="utf-8") as handle:
        payload = json.load(handle, object_pairs_hook=_unique_json_object)
    gate = ScreenPromotionGate.model_validate(payload)
    canonical = _canonical_json_bytes(gate.model_dump(mode="json"))
    if source.read_bytes() != canonical:
        raise ValueError("screen promotion gate must use canonical JSON bytes")
    return gate


def write_screen_promotion_gate(
    path: Path | str,
    gate: ScreenPromotionGate,
) -> None:
    """Atomically write one validated promotion gate in canonical JSON form."""
    if not isinstance(gate, ScreenPromotionGate):
        raise TypeError("gate must be a ScreenPromotionGate")
    atomic_write_json(path, gate.model_dump(mode="json"))


def replication_grid(
    pool_manifest: CrossfitPoolManifest,
    protocol: Protocol,
) -> tuple[tuple[str, int], ...]:
    """Return the locked assay-major 26-assay by five-seed replication grid."""
    if not isinstance(pool_manifest, CrossfitPoolManifest):
        raise TypeError("pool_manifest must be a CrossfitPoolManifest")
    if not isinstance(protocol, Protocol):
        raise TypeError("protocol must be a Protocol")
    if len(pool_manifest.untouched_ids) != 26 or len(protocol.seeds) != 5:
        raise ValueError("replication requires exactly 26 assays by 5 seeds")
    grid = tuple(
        (assay_id, seed)
        for assay_id in pool_manifest.untouched_ids
        for seed in protocol.seeds
    )
    if len(grid) != 130:
        raise ValueError("replication requires exactly 26 assays by 5 seeds")
    return grid


def resolve_replication_task_index(
    pool_manifest: CrossfitPoolManifest,
    protocol: Protocol,
    task_index: int,
) -> tuple[str, int]:
    """Resolve one strict zero-based index in the locked replication grid."""
    grid = replication_grid(pool_manifest, protocol)
    if type(task_index) is not int or task_index < 0 or task_index >= len(grid):
        raise ValueError("task index is outside the exact 130-task replication grid")
    return grid[task_index]


@dataclass(frozen=True, slots=True)
class CrossfitReplicationVerdict:
    """Immutable decision under the preregistered untouched-replication rule."""

    replication_success: bool
    crossfit_minus_random: PairwiseSummary

    def __post_init__(self) -> None:
        summary = self.crossfit_minus_random
        if not isinstance(summary, PairwiseSummary) or (
            summary.first != "crossfit"
            or summary.second != "random"
            or summary.metric != "spearman"
            or summary.task_total != 130
            or summary.assay_total != 26
        ):
            raise ValueError(
                "replication verdict requires the locked crossfit-minus-random "
                "Spearman comparison over 130 tasks and 26 assays"
            )
        expected = (
            summary.mean_gain > 0.0
            and summary.task_wins >= 78
            and summary.assay_wins >= 16
        )
        if type(self.replication_success) is not bool or (
            self.replication_success is not expected
        ):
            raise ValueError("replication_success must use the locked replication rule")

    @property
    def exact_sign_flip_pvalue(self) -> float:
        """Report, but deliberately do not gate on, the exact sign-flip p-value."""
        return self.crossfit_minus_random.exact_sign_flip_pvalue


def crossfit_replication_verdict(
    crossfit_minus_random: PairwiseSummary,
) -> CrossfitReplicationVerdict:
    """Apply the exact preregistered replication rule to one paired summary."""
    if not isinstance(crossfit_minus_random, PairwiseSummary):
        raise TypeError("crossfit_minus_random must be a PairwiseSummary")
    success = (
        crossfit_minus_random.mean_gain > 0.0
        and crossfit_minus_random.task_wins >= 78
        and crossfit_minus_random.assay_wins >= 16
    )
    return CrossfitReplicationVerdict(
        replication_success=success,
        crossfit_minus_random=crossfit_minus_random,
    )
