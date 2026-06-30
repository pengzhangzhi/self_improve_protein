from __future__ import annotations

import copy
import importlib.util
import json
from dataclasses import FrozenInstanceError, asdict
from hashlib import sha256
from pathlib import Path
from typing import cast

import pandas as pd  # type: ignore[import-untyped]
import pytest
from pydantic import ValidationError

from self_improve_protein import crossfit_replication
from self_improve_protein.analysis import (
    PairwiseSummary,
    pairwise_summary,
    validate_result_table,
)
from self_improve_protein.config import Protocol, load_protocol
from self_improve_protein.crossfit import CARD_ID as SCREEN_CARD_ID
from self_improve_protein.crossfit import CARD_SHA as SCREEN_CARD_SHA
from self_improve_protein.crossfit_data import (
    CROSSFIT_CARD_ID,
    CROSSFIT_CARD_SHA256,
    CROSSFIT_POOL_SCHEMA_ID,
    CrossfitPoolManifest,
)
from self_improve_protein.crossfit_replication import ScreenPromotionGate
from self_improve_protein.data import (
    ManifestSource,
    ManifestSources,
    SelectedAssayManifest,
)
from self_improve_protein.experiment import (
    METHOD_NAMES,
    NUMERICAL_POLICY,
    canonical_protocol_digest,
)


def test_crossfit_replication_module_exists() -> None:
    assert importlib.util.find_spec(
        "self_improve_protein.crossfit_replication"
    ) is not None


def _protocol(*, seeds: tuple[int, ...] = (0, 1, 2, 3, 4)) -> Protocol:
    payload = load_protocol("configs/v0.yaml").model_dump(mode="python")
    payload.update(
        working_size=3,
        n_labeled=1,
        n_unlabeled=1,
        n_test=1,
        q=1,
        seeds=seeds,
    )
    return Protocol.model_validate(payload)


def _pool(protocol: Protocol) -> CrossfitPoolManifest:
    eligible = tuple(f"ASSAY_{index:02d}" for index in range(35))
    sources = ManifestSources(
        substitutions=ManifestSource(
            url=protocol.substitutions_url,
            sha256=protocol.substitutions_sha256,
        ),
        scores=ManifestSource(
            url=protocol.zero_shot_scores_url,
            sha256=protocol.zero_shot_scores_sha256,
        ),
        metadata=ManifestSource(
            url=protocol.metadata_url,
            sha256=protocol.metadata_sha256,
        ),
    )
    records = tuple(
        SelectedAssayManifest(
            dms_id=dms_id,
            usable_count=protocol.working_size,
            sequence_length=4,
            row_hashes=tuple(
                sorted(
                    sha256(f"{dms_id}-{row}".encode()).hexdigest()
                    for row in range(protocol.working_size)
                )
            ),
        )
        for dms_id in eligible
    )
    return CrossfitPoolManifest(
        schema_id=CROSSFIT_POOL_SCHEMA_ID,
        schema_version=1,
        card_id=CROSSFIT_CARD_ID,
        card_sha256=CROSSFIT_CARD_SHA256,
        base_manifest_sha256="0" * 64,
        protocol_sha256="1" * 64,
        data_release=protocol.data_release,
        teacher_column=protocol.teacher_column,
        sources=sources,
        upstream_revision=protocol.proteingym_upstream_commit,
        max_length=protocol.max_length,
        working_size=protocol.working_size,
        eligible_assay_ids=eligible,
        screen_ids=(eligible[8], *eligible[:8]),
        untouched_ids=eligible[9:],
        selected_assays=records,
    )


def test_replication_grid_is_exact_assay_major_26_by_5() -> None:
    protocol = _protocol()
    pool = _pool(protocol)
    function = getattr(crossfit_replication, "replication_grid", None)

    assert callable(function)
    grid = function(pool, protocol)

    assert len(grid) == 130
    assert grid[:5] == tuple((pool.untouched_ids[0], seed) for seed in protocol.seeds)
    assert grid[-5:] == tuple(
        (pool.untouched_ids[-1], seed) for seed in protocol.seeds
    )
    assert tuple(dict.fromkeys(assay_id for assay_id, _ in grid)) == (
        pool.untouched_ids
    )


def test_replication_grid_requires_exactly_five_protocol_seeds() -> None:
    protocol = _protocol(seeds=(0, 1, 2, 3))
    pool = _pool(protocol)
    function = getattr(crossfit_replication, "replication_grid", None)

    assert callable(function)
    with pytest.raises(ValueError, match="exactly 26 assays by 5 seeds"):
        function(pool, protocol)


@pytest.mark.parametrize("task_index", [True, -1, 130])
def test_resolve_replication_task_index_rejects_non_grid_indices(
    task_index: object,
) -> None:
    protocol = _protocol()
    pool = _pool(protocol)
    function = getattr(
        crossfit_replication,
        "resolve_replication_task_index",
        None,
    )

    assert callable(function)
    with pytest.raises(ValueError, match="outside the exact 130-task replication grid"):
        function(pool, protocol, task_index)


def test_resolve_replication_task_index_maps_exact_boundaries() -> None:
    protocol = _protocol()
    pool = _pool(protocol)
    function = getattr(
        crossfit_replication,
        "resolve_replication_task_index",
        None,
    )

    assert callable(function)
    assert function(pool, protocol, 0) == (pool.untouched_ids[0], 0)
    assert function(pool, protocol, 129) == (pool.untouched_ids[-1], 4)


def _replication_summary(
    *,
    first: str = "crossfit",
    second: str = "random",
    metric: str = "spearman",
    mean_gain: float = 0.01,
    task_wins: int = 78,
    task_total: int = 130,
    assay_wins: int = 16,
    assay_total: int = 26,
    exact_sign_flip_pvalue: float = 1.0,
) -> PairwiseSummary:
    return PairwiseSummary(
        first=first,
        second=second,
        metric=metric,
        mean_gain=mean_gain,
        standard_error=0.02,
        task_wins=task_wins,
        task_total=task_total,
        task_win_rate=task_wins / task_total,
        assay_wins=assay_wins,
        assay_total=assay_total,
        assay_win_rate=assay_wins / assay_total,
        exact_sign_flip_pvalue=exact_sign_flip_pvalue,
        assay_deltas=tuple(0.01 for _ in range(assay_total)),
    )


def test_replication_verdict_uses_locked_thresholds_but_not_significance() -> None:
    function = getattr(
        crossfit_replication,
        "crossfit_replication_verdict",
        None,
    )
    summary = _replication_summary(exact_sign_flip_pvalue=1.0)

    assert callable(function)
    verdict = function(summary)

    assert verdict.replication_success is True
    assert verdict.crossfit_minus_random is summary
    assert verdict.exact_sign_flip_pvalue == 1.0
    with pytest.raises(FrozenInstanceError):
        verdict.replication_success = False


@pytest.mark.parametrize(
    "summary",
    [
        _replication_summary(mean_gain=0.0),
        _replication_summary(task_wins=77),
        _replication_summary(assay_wins=15),
    ],
    ids=("nonpositive-mean", "77-task-wins", "15-assay-wins"),
)
def test_replication_verdict_rejects_each_failed_gate(
    summary: PairwiseSummary,
) -> None:
    function = getattr(
        crossfit_replication,
        "crossfit_replication_verdict",
        None,
    )

    assert callable(function)
    assert function(summary).replication_success is False


@pytest.mark.parametrize(
    "summary",
    [
        _replication_summary(first="ours"),
        _replication_summary(second="supervised"),
        _replication_summary(metric="mse"),
        _replication_summary(task_total=129, task_wins=78),
        _replication_summary(assay_total=25, assay_wins=16),
    ],
    ids=("first", "second", "metric", "task-total", "assay-total"),
)
def test_replication_verdict_requires_exact_comparison_and_grid(
    summary: PairwiseSummary,
) -> None:
    function = getattr(
        crossfit_replication,
        "crossfit_replication_verdict",
        None,
    )

    assert callable(function)
    with pytest.raises(ValueError, match="locked crossfit-minus-random Spearman"):
        function(summary)


def _screen_primary_results(
    pool: CrossfitPoolManifest,
    protocol: Protocol,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    primary_ids = pool.screen_ids[1:]
    methods = (*METHOD_NAMES, "crossfit")
    for assay_index, assay_id in enumerate(primary_ids):
        for seed in protocol.seeds:
            for method in methods:
                spearman = 0.0
                if method == "crossfit":
                    spearman = 0.1 if assay_index < 5 else -0.01
                rows.append(
                    {
                        "assay_id": assay_id,
                        "method": method,
                        "mse": 1.0,
                        "ndcg_10pct": 0.5,
                        "seed": seed,
                        "spearman": spearman,
                    }
                )
    return validate_result_table(
        pd.DataFrame(rows),
        assay_ids=primary_ids,
        seeds=protocol.seeds,
        required_methods=methods,
    )


def _screen_aggregate(
    pool: CrossfitPoolManifest,
    protocol: Protocol,
    *,
    pool_sha256: str = "2" * 64,
    git_commit: str = "4" * 40,
) -> dict[str, object]:
    primary = _screen_primary_results(pool, protocol)
    effects = {
        f"crossfit_minus_{second}": asdict(
            pairwise_summary(
                primary,
                first="crossfit",
                second=second,
                metric="spearman",
            )
        )
        for second in METHOD_NAMES
    }
    screen_grid = tuple(
        (assay_id, seed)
        for assay_id in pool.screen_ids
        for seed in protocol.seeds
    )
    primary_grid = tuple(
        (assay_id, seed)
        for assay_id in pool.screen_ids[1:]
        for seed in protocol.seeds
    )
    payload: dict[str, object] = {
        "analysis": {
            "inference_unit": "assay",
            "metric": "spearman",
            "sign_flip": "exact",
        },
        "card": {"id": SCREEN_CARD_ID, "sha256": SCREEN_CARD_SHA},
        "development_diagnostics": [{} for _ in protocol.seeds],
        "development_long_results": [
            {"row": index} for index in range(len(protocol.seeds) * 6)
        ],
        "effects": effects,
        "execution": {
            "bypass": None,
            "numerical_policy": NUMERICAL_POLICY,
            "numerical_runtime": {"blas_threads": 1},
            "official": True,
        },
        "grid": {
            "development_assay_id": pool.screen_ids[0],
            "primary_assay_ids": list(pool.screen_ids[1:]),
            "primary_tasks": [list(task) for task in primary_grid],
            "screen_assay_ids": list(pool.screen_ids),
            "screen_tasks": [list(task) for task in screen_grid],
            "seeds": list(protocol.seeds),
        },
        "kind": "crossfit_aggregate_result",
        "method_table": [{"method": method} for method in (*METHOD_NAMES, "crossfit")],
        "primary_long_results": primary.to_dict(orient="records"),
        "promotion": {
            "assay_win_threshold": 5,
            "development_assay_excluded": True,
            "mean_gain_strictly_positive": True,
            "promote_to_untouched_replication": True,
            "task_win_threshold": 25,
            "untouched_assay_ids": list(pool.untouched_ids),
        },
        "provenance": {
            "base_manifest_sha256": pool.base_manifest_sha256,
            "git_commit": git_commit,
            "pool_manifest_sha256": pool_sha256,
            "protocol_digest": canonical_protocol_digest(protocol),
            "task_count": 45,
            "task_manifest": [
                {
                    "assay_id": assay_id,
                    "seed": seed,
                    "sha256": sha256(f"{assay_id}-{seed}".encode()).hexdigest(),
                }
                for assay_id, seed in screen_grid
            ],
        },
        "schema_version": 1,
        "task_diagnostics": [{} for _ in screen_grid],
    }
    return cast(dict[str, object], json.loads(json.dumps(payload)))


def _build_gate() -> tuple[
    ScreenPromotionGate,
    dict[str, object],
    CrossfitPoolManifest,
    Protocol,
]:
    protocol = _protocol()
    pool = _pool(protocol)
    aggregate = _screen_aggregate(pool, protocol)
    function = getattr(crossfit_replication, "build_screen_promotion_gate", None)
    assert callable(function)
    gate = function(
        aggregate,
        pool_manifest=pool,
        protocol=protocol,
        pool_manifest_sha256="2" * 64,
        screen_aggregate_sha256="3" * 64,
        git_commit="4" * 40,
    )
    assert isinstance(gate, ScreenPromotionGate)
    return gate, aggregate, pool, protocol


def test_promotion_gate_binds_exact_preregistered_evidence() -> None:
    gate, _, pool, protocol = _build_gate()

    assert gate.schema_id == "self-improve-protein.crossfit-screen-promotion.v1"
    assert gate.schema_version == 1
    assert gate.card_id == SCREEN_CARD_ID
    assert gate.card_sha256 == SCREEN_CARD_SHA
    assert gate.base_manifest_sha256 == pool.base_manifest_sha256
    assert gate.pool_manifest_sha256 == "2" * 64
    assert gate.screen_aggregate_sha256 == "3" * 64
    assert gate.protocol_digest == canonical_protocol_digest(protocol)
    assert gate.git_commit == "4" * 40
    assert gate.screen_primary_summary.first == "crossfit"
    assert gate.screen_primary_summary.second == "random"
    assert gate.screen_primary_summary.task_total == 40
    assert gate.screen_primary_summary.assay_total == 8
    assert gate.screen_primary_summary.task_wins == 25
    assert gate.screen_primary_summary.assay_wins == 5
    assert gate.untouched_assay_ids == pool.untouched_ids
    assert gate.seeds == protocol.seeds
    assert gate.promote_to_untouched_replication is True
    with pytest.raises(ValidationError, match="frozen"):
        gate.__setattr__("git_commit", "5" * 40)


def test_promotion_gate_builder_rejects_nonpromoting_screen() -> None:
    protocol = _protocol()
    pool = _pool(protocol)
    aggregate = _screen_aggregate(pool, protocol)
    promotion = aggregate["promotion"]
    assert isinstance(promotion, dict)
    promotion["promote_to_untouched_replication"] = False
    function = getattr(crossfit_replication, "build_screen_promotion_gate", None)

    assert callable(function)
    with pytest.raises(ValueError, match="screen aggregate did not promote"):
        function(
            aggregate,
            pool_manifest=pool,
            protocol=protocol,
            pool_manifest_sha256="2" * 64,
            screen_aggregate_sha256="3" * 64,
            git_commit="4" * 40,
        )


@pytest.mark.parametrize(
    "tamper_kind",
    [
        "top_level",
        "card",
        "grid",
        "promotion_rule",
        "summary",
        "base_hash",
        "pool_hash",
        "protocol_digest",
        "git_commit",
        "official",
    ],
)
def test_promotion_gate_builder_fails_closed_on_aggregate_tamper(
    tamper_kind: str,
) -> None:
    protocol = _protocol()
    pool = _pool(protocol)
    aggregate = _screen_aggregate(pool, protocol)
    tampered = copy.deepcopy(aggregate)
    if tamper_kind == "top_level":
        tampered["unexpected"] = True
    elif tamper_kind == "card":
        card = tampered["card"]
        assert isinstance(card, dict)
        card["sha256"] = "f" * 64
    elif tamper_kind == "grid":
        grid = tampered["grid"]
        assert isinstance(grid, dict)
        grid["primary_tasks"] = list(reversed(grid["primary_tasks"]))
    elif tamper_kind == "promotion_rule":
        promotion = tampered["promotion"]
        assert isinstance(promotion, dict)
        promotion["task_win_threshold"] = 24
    elif tamper_kind == "summary":
        effects = tampered["effects"]
        assert isinstance(effects, dict)
        summary = effects["crossfit_minus_random"]
        assert isinstance(summary, dict)
        summary["mean_gain"] = 0.9
    elif tamper_kind == "official":
        execution = tampered["execution"]
        assert isinstance(execution, dict)
        execution["official"] = False
    else:
        provenance = tampered["provenance"]
        assert isinstance(provenance, dict)
        key = {
            "base_hash": "base_manifest_sha256",
            "pool_hash": "pool_manifest_sha256",
            "protocol_digest": "protocol_digest",
            "git_commit": "git_commit",
        }[tamper_kind]
        provenance[key] = "f" * (40 if key == "git_commit" else 64)
    function = getattr(crossfit_replication, "build_screen_promotion_gate", None)

    assert callable(function)
    with pytest.raises(ValueError, match="screen aggregate"):
        function(
            tampered,
            pool_manifest=pool,
            protocol=protocol,
            pool_manifest_sha256="2" * 64,
            screen_aggregate_sha256="3" * 64,
            git_commit="4" * 40,
        )


def test_promotion_gate_canonical_roundtrip_and_duplicate_key_rejection(
    tmp_path: Path,
) -> None:
    gate, _, _, _ = _build_gate()
    writer = getattr(crossfit_replication, "write_screen_promotion_gate", None)
    loader = getattr(crossfit_replication, "load_screen_promotion_gate", None)
    assert callable(writer)
    assert callable(loader)
    first = tmp_path / "gate.json"
    second = tmp_path / "nested" / "gate.json"

    writer(first, gate)
    writer(second, gate)

    assert first.read_bytes() == second.read_bytes()
    assert loader(first) == gate
    compact = tmp_path / "compact.json"
    compact.write_text(json.dumps(gate.model_dump(mode="json")), encoding="utf-8")
    with pytest.raises(ValueError, match="canonical JSON bytes"):
        loader(compact)
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text(
        '{"schema_id":"self-improve-protein.crossfit-screen-promotion.v1",'
        '"schema_id":"duplicate"}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r"duplicate JSON key.*schema_id"):
        loader(duplicate)
