"""Restart-safe CLI for the predeclared pseudo-perturbation locality screen."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, cast

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import typer

from self_improve_protein.analysis import pairwise_summary, validate_result_table
from self_improve_protein.cli import (
    _ESM2_35M_EMBEDDING_DIM,
    _cache_width,
    _canonical_json_bytes,
    _embedding_sources,
    _git_commit,
    _jsonable,
    _load_identity_frame,
    _load_unique_json,
    _ordered_outcomes,
    _require_exact_payload,
    _run_command,
    _write_json_once,
    load_evaluation_labels,
)
from self_improve_protein.config import DEFAULT_CONFIG_PATH, Protocol, load_protocol
from self_improve_protein.crossfit_cli import (
    _load_manifests,
    _method_row,
    _record,
    _screen_grid,
    _validate_execution_policy,
)
from self_improve_protein.crossfit_data import CrossfitPoolManifest
from self_improve_protein.data import DataManifest, make_split
from self_improve_protein.embeddings import load_embedding_cache
from self_improve_protein.experiment import (
    NUMERICAL_POLICY,
    FitInputs,
    canonical_evaluation_digest,
    canonical_protocol_digest,
    canonical_source_digest,
)
from self_improve_protein.locality import (
    CARD_ID,
    CARD_SHA,
    Q_VALUES,
    SELECTORS,
    W_VALUES,
    LocalityCellArtifact,
    LocalityCellEvaluation,
    LocalityEvaluationResult,
    LocalityFitResult,
    canonical_locality_fit_digest,
    evaluate_locality_task,
    fit_locality_task,
)
from self_improve_protein.provenance import sha256_file

app = typer.Typer(
    name="self-improve-protein-locality",
    help="Run the predeclared nine-assay pseudo-perturbation locality screen.",
    invoke_without_command=True,
    no_args_is_help=False,
    pretty_exceptions_enable=False,
)


@dataclass(frozen=True, slots=True)
class _RuntimeOptions:
    config: Path
    dry_run: bool


@app.callback()
def main(
    context: typer.Context,
    config: Annotated[Path, typer.Option("--config")] = DEFAULT_CONFIG_PATH,
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
) -> None:
    """Set the frozen v0 protocol and mutation policy."""
    context.obj = _RuntimeOptions(config=config, dry_run=dry_run)
    if context.invoked_subcommand is None:
        typer.echo(context.get_help())
        raise typer.Exit()


def _options(context: typer.Context) -> _RuntimeOptions:
    if not isinstance(context.obj, _RuntimeOptions):
        raise RuntimeError("locality CLI runtime options were not initialized")
    return context.obj


def _exact_grid(
    pool: CrossfitPoolManifest,
    protocol: Protocol,
) -> tuple[tuple[str, int], ...]:
    grid = _screen_grid(
        development_id=pool.screen_ids[0],
        confirmatory_ids=pool.screen_ids[1:],
        seeds=protocol.seeds,
    )
    if len(pool.screen_ids) != 9 or len(protocol.seeds) != 5 or len(grid) != 45:
        raise ValueError("locality screen requires exactly 9 assays by 5 seeds")
    return grid


def _resolve_task_index(
    pool: CrossfitPoolManifest,
    protocol: Protocol,
    task_index: int,
) -> tuple[str, int]:
    grid = _exact_grid(pool, protocol)
    if type(task_index) is not int or task_index < 0 or task_index >= len(grid):
        raise ValueError("task index is outside the exact 45-task locality grid")
    return grid[task_index]


def _require_screen_roots(
    processed_root: Path,
    embedding_root: Path,
    *,
    non_official_bypass: bool,
) -> None:
    if non_official_bypass:
        return
    if processed_root.parts[-2:] != ("processed", "v0"):
        raise ValueError("official locality screen requires processed/v0")
    if embedding_root.parts[-2:] != ("embeddings", "v0"):
        raise ValueError("official locality screen requires embeddings/v0")


def _load_context(
    *,
    config: Path,
    base_manifest_path: Path,
    pool_manifest_path: Path,
    non_official_bypass: bool,
) -> tuple[Protocol, DataManifest, CrossfitPoolManifest, str]:
    protocol = load_protocol(config)
    _validate_execution_policy(
        protocol,
        non_official_bypass=non_official_bypass,
    )
    base, pool = _load_manifests(
        protocol=protocol,
        base_manifest_path=base_manifest_path,
        pool_manifest_path=pool_manifest_path,
    )
    _exact_grid(pool, protocol)
    commit = _git_commit(require_clean=not non_official_bypass)
    return protocol, base, pool, commit


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_task_payload(payload: dict[str, object]) -> None:
    required = {
        "card",
        "cells",
        "digests",
        "execution",
        "kind",
        "orderings",
        "provenance",
        "reference_methods",
        "schema_version",
        "task",
    }
    if (
        set(payload) != required
        or payload.get("schema_version") != 1
        or payload.get("kind") != "locality_task_result"
    ):
        raise ValueError("locality task schema is invalid")
    card = payload.get("card")
    task = payload.get("task")
    digests = payload.get("digests")
    if (
        card != {"id": CARD_ID, "sha256": CARD_SHA}
        or not isinstance(task, dict)
        or task.get("phase") != "locality_screen"
        or not isinstance(task.get("assay_id"), str)
        or type(task.get("seed")) is not int
        or not isinstance(digests, dict)
        or any(
            not _is_sha256(digests.get(name))
            for name in ("evaluation", "fit", "protocol", "source")
        )
    ):
        raise ValueError("locality task identity is invalid")
    orderings = payload.get("orderings")
    cells = payload.get("cells")
    if (
        not isinstance(orderings, list)
        or tuple(
            row.get("selector") if isinstance(row, dict) else None
            for row in orderings
        )
        != SELECTORS
        or not isinstance(cells, list)
        or len(cells) != 60
    ):
        raise ValueError("locality task factorial grid is invalid")
    keys = tuple(
        (
            row.get("selector"),
            row.get("q"),
            row.get("pseudo_weight"),
        )
        for row in cells
        if isinstance(row, dict)
    )
    expected = tuple(
        (selector, q, weight)
        for selector in SELECTORS
        for q in Q_VALUES
        for weight in W_VALUES
    )
    if keys != expected:
        raise ValueError("locality task cells do not match frozen grid")


def _ordering_payload(fit: LocalityFitResult) -> list[dict[str, object]]:
    return [
        {
            "ordered_hashes": list(ordering.ordered_hashes),
            "ordered_indices": list(ordering.ordered_indices),
            "selector": ordering.selector,
        }
        for ordering in fit.orderings
    ]


def _cell_payload(
    artifact: LocalityCellArtifact,
    evaluation: LocalityCellEvaluation,
) -> dict[str, object]:
    if (
        artifact.selector,
        artifact.q,
        artifact.pseudo_weight,
    ) != (
        evaluation.selector,
        evaluation.q,
        evaluation.pseudo_weight,
    ):
        raise ValueError("locality fit/evaluation cell identity mismatch")
    return {
        "coefficients": artifact.coefficients.tolist(),
        **cast(dict[str, object], _jsonable(dataclasses.asdict(evaluation))),
        "selected_hashes": list(artifact.selected_hashes),
        "selected_indices": list(artifact.selected_indices),
        "test_predictions": artifact.test_predictions.tolist(),
        "training_weight_sum": artifact.training_weight_sum,
    }


def _build_task_payload(
    *,
    protocol: Protocol,
    pool: CrossfitPoolManifest,
    git_commit: str,
    base_manifest_path: Path,
    pool_manifest_path: Path,
    processed_root: Path,
    embedding_root: Path,
    assay_id: str,
    seed: int,
    non_official_bypass: bool,
) -> dict[str, object]:
    if (assay_id, seed) not in _exact_grid(pool, protocol):
        raise ValueError("task is outside the exact locality screen grid")
    _require_screen_roots(
        processed_root,
        embedding_root,
        non_official_bypass=non_official_bypass,
    )
    record = _record(pool, assay_id)
    parquet_path = processed_root / f"{assay_id}.parquet"
    identity = _load_identity_frame(
        parquet_path,
        protocol=protocol,
        record=record,
    )
    split = make_split(
        identity,
        assay_id,
        seed,
        protocol.n_labeled,
        protocol.n_unlabeled,
        protocol.n_test,
    )
    metadata_path = embedding_root / f"{assay_id}.json"
    npy_path = embedding_root / f"{assay_id}.npy"
    embeddings = load_embedding_cache(
        npy_path,
        metadata_path,
        dms_id=assay_id,
        row_hashes=record.row_hashes,
        model_id=protocol.model,
        model_revision=protocol.model_revision,
        sources=_embedding_sources(protocol),
        expected_embedding_dim=(
            _cache_width(metadata_path)
            if non_official_bypass
            else _ESM2_35M_EMBEDDING_DIM
        ),
    )
    y_l = _ordered_outcomes(
        parquet_path,
        requested_hashes=split.labeled_hashes,
        expected_all_hashes=record.row_hashes,
    )
    teacher = identity[protocol.teacher_column].to_numpy(dtype=np.float64)
    source_digest = canonical_source_digest(protocol)
    inputs = FitInputs(
        assay_id=assay_id,
        seed=seed,
        source_digest=source_digest,
        labeled_hashes=split.labeled_hashes,
        unlabeled_hashes=split.unlabeled_hashes,
        test_hashes=split.test_hashes,
        x_l=embeddings[np.asarray(split.labeled, dtype=np.int64)],
        y_l=y_l,
        z_l=teacher[np.asarray(split.labeled, dtype=np.int64)],
        x_u=embeddings[np.asarray(split.unlabeled, dtype=np.int64)],
        z_u=teacher[np.asarray(split.unlabeled, dtype=np.int64)],
        x_test=embeddings[np.asarray(split.test, dtype=np.int64)],
        z_test=teacher[np.asarray(split.test, dtype=np.int64)],
    )

    # Leakage boundary: all 60 fits and their canonical digest exist before
    # hidden unlabeled/test outcomes can be loaded.
    fit = fit_locality_task(inputs, protocol)
    fit_digest = canonical_locality_fit_digest(fit)
    labels = load_evaluation_labels(
        parquet_path,
        assay_id=assay_id,
        seed=seed,
        source_digest=source_digest,
        labeled_hashes=split.labeled_hashes,
        unlabeled_hashes=split.unlabeled_hashes,
        test_hashes=split.test_hashes,
        expected_all_hashes=record.row_hashes,
    )
    evaluation_digest = canonical_evaluation_digest(labels)
    evaluation = evaluate_locality_task(
        fit,
        labels,
        protocol=protocol,
        expected_fit_digest=fit_digest,
        expected_evaluation_digest=evaluation_digest,
    )
    if not isinstance(evaluation, LocalityEvaluationResult):
        raise ValueError("locality evaluation result is invalid")
    measured = {item.name: item for item in evaluation.reference.methods}
    references = [
        _method_row(
            artifact=method,
            evaluation=measured[method.name],
            name=method.name,
        )
        for method in fit.base_fit.methods
    ]
    cells = [
        _cell_payload(artifact, result)
        for artifact, result in zip(fit.cells, evaluation.cells, strict=True)
    ]
    payload: dict[str, object] = {
        "card": {"id": CARD_ID, "sha256": CARD_SHA},
        "cells": cells,
        "digests": {
            "evaluation": evaluation_digest,
            "fit": fit_digest,
            "protocol": canonical_protocol_digest(protocol),
            "source": source_digest,
        },
        "execution": {
            "bypass": (
                "non_official_test_or_development" if non_official_bypass else None
            ),
            "numerical_policy": fit.base_fit.numerical_policy,
            "numerical_runtime": fit.base_fit.numerical_runtime,
            "official": not non_official_bypass,
        },
        "kind": "locality_task_result",
        "orderings": _ordering_payload(fit),
        "provenance": {
            "base_manifest_sha256": sha256_file(base_manifest_path),
            "embedding_metadata_sha256": sha256_file(metadata_path),
            "embedding_npy_sha256": sha256_file(npy_path),
            "git_commit": git_commit,
            "pool_manifest_sha256": sha256_file(pool_manifest_path),
            "processed_sha256": sha256_file(parquet_path),
        },
        "reference_methods": references,
        "schema_version": 1,
        "task": {
            "assay_id": assay_id,
            "labeled_hashes": list(split.labeled_hashes),
            "phase": "locality_screen",
            "seed": seed,
            "test_hashes": list(split.test_hashes),
            "unlabeled_hashes": list(split.unlabeled_hashes),
        },
    }
    normalized = cast(dict[str, object], _jsonable(payload))
    _validate_task_payload(normalized)
    return normalized


def _task_path(results_root: Path, assay_id: str, seed: int) -> Path:
    return results_root / "tasks" / assay_id / f"seed_{seed}.json"


@app.command("run-task")
def run_task(
    context: typer.Context,
    base_manifest_path: Annotated[Path, typer.Option("--base-manifest")],
    pool_manifest_path: Annotated[Path, typer.Option("--pool-manifest")],
    processed_root: Annotated[Path, typer.Option("--processed-root")],
    embedding_root: Annotated[Path, typer.Option("--embedding-root")],
    results_root: Annotated[Path, typer.Option("--results-root")],
    task_index: Annotated[int, typer.Option("--task-index", min=0)],
    non_official_bypass: Annotated[
        bool,
        typer.Option("--non-official-bypass"),
    ] = False,
) -> None:
    """Run one immutable member of the exact 45-task locality screen."""
    options = _options(context)

    def action() -> dict[str, object]:
        protocol, _, pool, commit = _load_context(
            config=options.config,
            base_manifest_path=base_manifest_path,
            pool_manifest_path=pool_manifest_path,
            non_official_bypass=non_official_bypass,
        )
        assay_id, seed = _resolve_task_index(pool, protocol, task_index)
        destination = _task_path(results_root, assay_id, seed)
        if options.dry_run:
            return {"plan": {"assay_id": assay_id, "seed": seed}}
        payload = _build_task_payload(
            protocol=protocol,
            pool=pool,
            git_commit=commit,
            base_manifest_path=base_manifest_path,
            pool_manifest_path=pool_manifest_path,
            processed_root=processed_root,
            embedding_root=embedding_root,
            assay_id=assay_id,
            seed=seed,
            non_official_bypass=non_official_bypass,
        )
        created = _write_json_once(destination, payload)
        return {
            "artifact": str(destination),
            "artifact_sha256": sha256_file(destination),
            "assay_id": assay_id,
            "created": created,
            "seed": seed,
        }

    _run_command("locality-run-task", dry_run=options.dry_run, action=action)


def _long_row(
    assay_id: str,
    seed: int,
    cell: dict[str, object],
) -> dict[str, object]:
    keep = {
        "displacement_cosine",
        "displacement_cosine_defined",
        "displacement_relative_error",
        "effective_pseudo_fraction",
        "labeled_loss_prediction_error",
        "labeled_loss_sign_agreement",
        "locality_index",
        "mse",
        "ndcg_10pct",
        "predicted_labeled_loss_change",
        "predicted_test_loss_change",
        "pseudo_weight",
        "q",
        "realized_labeled_loss_change",
        "realized_test_loss_change",
        "selected_pseudo_label_mae",
        "selector",
        "spearman",
        "test_loss_prediction_error",
        "test_loss_sign_agreement",
        "test_oracle_score_alignment",
        "test_oracle_score_vs_absolute_error",
    }
    return {
        "assay_id": assay_id,
        "seed": seed,
        **{key: cell[key] for key in keep},
    }


def _effect_payload(
    rows: pd.DataFrame,
    *,
    selector: str,
    q: int,
    pseudo_weight: float,
) -> dict[str, object]:
    subset = rows.loc[
        (rows["q"] == q) & (rows["pseudo_weight"] == pseudo_weight),
        ["assay_id", "seed", "selector", "spearman", "mse", "ndcg_10pct"],
    ].rename(columns={"selector": "method"})
    validated = validate_result_table(
        subset,
        assay_ids=tuple(sorted(str(value) for value in rows["assay_id"].unique())),
        seeds=tuple(sorted(int(value) for value in rows["seed"].unique())),
        required_methods=SELECTORS,
    )
    summary = pairwise_summary(
        validated,
        first=selector,
        second="random",
        metric="spearman",
    )
    return {
        **cast(dict[str, object], _jsonable(dataclasses.asdict(summary))),
        "q": q,
        "pseudo_weight": pseudo_weight,
    }


def _correlation_value(value: object) -> float | None:
    if not isinstance(value, dict) or value.get("defined") is not True:
        return None
    raw = value.get("value")
    return float(raw) if isinstance(raw, (int, float)) else None


def _cell_trends(primary: pd.DataFrame) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for selector in SELECTORS:
        for q in Q_VALUES:
            for weight in W_VALUES:
                group = primary.loc[
                    (primary["selector"] == selector)
                    & (primary["q"] == q)
                    & (primary["pseudo_weight"] == weight)
                ]
                if len(group) != 40:
                    raise ValueError("locality mechanism cell is incomplete")
                cosines = [
                    float(value)
                    for value in group["displacement_cosine"]
                    if value is not None
                ]
                alignments = [
                    value
                    for value in (
                        _correlation_value(item)
                        for item in group["test_oracle_score_alignment"]
                    )
                    if value is not None
                ]
                output.append(
                    {
                        "selector": selector,
                        "q": q,
                        "pseudo_weight": weight,
                        "effective_pseudo_fraction": float(
                            group["effective_pseudo_fraction"].iloc[0]
                        ),
                        "mean_spearman": float(group["spearman"].mean()),
                        "mean_mse": float(group["mse"].mean()),
                        "mean_selected_pseudo_label_mae": float(
                            group["selected_pseudo_label_mae"].mean()
                        ),
                        "mean_displacement_cosine": (
                            float(np.mean(cosines)) if cosines else None
                        ),
                        "mean_displacement_relative_error": float(
                            group["displacement_relative_error"].mean()
                        ),
                        "mean_locality_index": float(
                            group["locality_index"].mean()
                        ),
                        "labeled_loss_sign_agreement_rate": float(
                            group["labeled_loss_sign_agreement"].mean()
                        ),
                        "test_loss_sign_agreement_rate": float(
                            group["test_loss_sign_agreement"].mean()
                        ),
                        "mean_labeled_loss_prediction_error": float(
                            group["labeled_loss_prediction_error"].mean()
                        ),
                        "mean_test_loss_prediction_error": float(
                            group["test_loss_prediction_error"].mean()
                        ),
                        "mean_test_oracle_score_alignment": (
                            float(np.mean(alignments)) if alignments else None
                        ),
                    }
                )
    return output


def _selector_trends(cell_trends: list[dict[str, object]]) -> list[dict[str, object]]:
    return [
        {
            "selector": selector,
            "cells": [
                row for row in cell_trends if row["selector"] == selector
            ],
        }
        for selector in SELECTORS
    ]


def _validate_aggregate_payload(payload: dict[str, object]) -> None:
    if (
        payload.get("schema_version") != 1
        or payload.get("kind") != "locality_aggregate_result"
    ):
        raise ValueError("locality aggregate schema is invalid")
    analysis = payload.get("analysis")
    grid = payload.get("grid")
    provenance = payload.get("provenance")
    if (
        not isinstance(analysis, dict)
        or analysis.get("confirmatory_claim") is not False
        or analysis.get("scope") != "exploratory_mechanism_screen"
        or not isinstance(grid, dict)
        or len(cast(list[object], grid.get("screen_tasks", []))) != 45
        or len(cast(list[object], grid.get("primary_tasks", []))) != 40
        or not isinstance(provenance, dict)
        or provenance.get("task_count") != 45
        or "promotion" in payload
    ):
        raise ValueError("locality aggregate contract is invalid")


def _build_aggregate_payload(
    *,
    protocol: Protocol,
    pool: CrossfitPoolManifest,
    git_commit: str,
    base_manifest_path: Path,
    pool_manifest_path: Path,
    processed_root: Path,
    embedding_root: Path,
    results_root: Path,
    non_official_bypass: bool,
) -> dict[str, object]:
    grid = _exact_grid(pool, protocol)
    development_id = pool.screen_ids[0]
    primary_ids = pool.screen_ids[1:]
    primary_grid = tuple(
        (assay_id, seed)
        for assay_id in primary_ids
        for seed in protocol.seeds
    )
    rows: list[dict[str, object]] = []
    task_manifest: list[dict[str, object]] = []
    commits: set[str] = set()
    runtimes: set[bytes] = set()
    runtime: object = None
    for assay_id, seed in grid:
        path = _task_path(results_root, assay_id, seed)
        if not path.is_file():
            raise ValueError(f"missing locality task artifact: {path}")
        actual = _load_unique_json(path)
        _validate_task_payload(actual)
        expected = _build_task_payload(
            protocol=protocol,
            pool=pool,
            git_commit=git_commit,
            base_manifest_path=base_manifest_path,
            pool_manifest_path=pool_manifest_path,
            processed_root=processed_root,
            embedding_root=embedding_root,
            assay_id=assay_id,
            seed=seed,
            non_official_bypass=non_official_bypass,
        )
        _require_exact_payload(actual, expected, artifact_kind="locality task")
        execution = cast(dict[str, object], actual["execution"])
        provenance = cast(dict[str, object], actual["provenance"])
        commits.add(cast(str, provenance["git_commit"]))
        runtime = execution["numerical_runtime"]
        runtimes.add(_canonical_json_bytes(runtime))
        for cell in cast(list[dict[str, object]], actual["cells"]):
            rows.append(_long_row(assay_id, seed, cell))
        task_manifest.append(
            {"assay_id": assay_id, "seed": seed, "sha256": sha256_file(path)}
        )
    if commits != {git_commit} or len(runtimes) != 1:
        raise ValueError("locality tasks mix implementations or runtimes")
    table = pd.DataFrame(rows).sort_values(
        ["assay_id", "seed", "selector", "q", "pseudo_weight"],
        kind="stable",
    ).reset_index(drop=True)
    primary = table.loc[table["assay_id"].isin(primary_ids)].copy()
    development = table.loc[table["assay_id"] == development_id].copy()
    if len(table) != 45 * 60 or len(primary) != 40 * 60 or len(development) != 5 * 60:
        raise ValueError("locality result grid is incomplete")
    effects = [
        _effect_payload(
            primary,
            selector=selector,
            q=q,
            pseudo_weight=weight,
        )
        for selector in SELECTORS
        if selector != "random"
        for q in Q_VALUES
        for weight in W_VALUES
    ]
    cell_trends = _cell_trends(primary)
    payload: dict[str, object] = {
        "analysis": {
            "confirmatory_claim": False,
            "inference_unit": "assay",
            "scope": "exploratory_mechanism_screen",
        },
        "card": {"id": CARD_ID, "sha256": CARD_SHA},
        "development_long_results": development.to_dict(orient="records"),
        "effects": effects,
        "execution": {
            "bypass": (
                "non_official_test_or_development" if non_official_bypass else None
            ),
            "numerical_policy": NUMERICAL_POLICY,
            "numerical_runtime": runtime,
            "official": not non_official_bypass,
        },
        "grid": {
            "development_assay_id": development_id,
            "primary_assay_ids": list(primary_ids),
            "primary_tasks": [list(item) for item in primary_grid],
            "q_values": list(Q_VALUES),
            "screen_assay_ids": list(pool.screen_ids),
            "screen_tasks": [list(item) for item in grid],
            "seeds": list(protocol.seeds),
            "selectors": list(SELECTORS),
            "w_values": list(W_VALUES),
        },
        "kind": "locality_aggregate_result",
        "mechanism": {
            "cell_trends": cell_trends,
            "selector_trends": _selector_trends(cell_trends),
        },
        "primary_long_results": primary.to_dict(orient="records"),
        "provenance": {
            "base_manifest_sha256": sha256_file(base_manifest_path),
            "git_commit": git_commit,
            "pool_manifest_sha256": sha256_file(pool_manifest_path),
            "protocol_digest": canonical_protocol_digest(protocol),
            "task_count": len(task_manifest),
            "task_manifest": task_manifest,
        },
        "schema_version": 1,
    }
    normalized = cast(dict[str, object], _jsonable(payload))
    _validate_aggregate_payload(normalized)
    return normalized


@app.command("aggregate")
def aggregate(
    context: typer.Context,
    base_manifest_path: Annotated[Path, typer.Option("--base-manifest")],
    pool_manifest_path: Annotated[Path, typer.Option("--pool-manifest")],
    processed_root: Annotated[Path, typer.Option("--processed-root")],
    embedding_root: Annotated[Path, typer.Option("--embedding-root")],
    results_root: Annotated[Path, typer.Option("--results-root")],
    output: Annotated[Path, typer.Option("--output")],
    non_official_bypass: Annotated[
        bool,
        typer.Option("--non-official-bypass"),
    ] = False,
) -> None:
    """Exact-rebuild all 45 tasks and summarize the mechanism screen."""
    options = _options(context)

    def action() -> dict[str, object]:
        protocol, _, pool, commit = _load_context(
            config=options.config,
            base_manifest_path=base_manifest_path,
            pool_manifest_path=pool_manifest_path,
            non_official_bypass=non_official_bypass,
        )
        if options.dry_run:
            return {"plan": {"output": str(output), "task_count": 45}}
        payload = _build_aggregate_payload(
            protocol=protocol,
            pool=pool,
            git_commit=commit,
            base_manifest_path=base_manifest_path,
            pool_manifest_path=pool_manifest_path,
            processed_root=processed_root,
            embedding_root=embedding_root,
            results_root=results_root,
            non_official_bypass=non_official_bypass,
        )
        created = _write_json_once(output, payload)
        return {
            "artifact": str(output),
            "artifact_sha256": sha256_file(output),
            "created": created,
            "task_count": 45,
        }

    _run_command("locality-aggregate", dry_run=options.dry_run, action=action)


def _verify_inputs(
    protocol: Protocol,
    pool: CrossfitPoolManifest,
    *,
    processed_root: Path,
    embedding_root: Path,
    non_official_bypass: bool,
) -> None:
    _require_screen_roots(
        processed_root,
        embedding_root,
        non_official_bypass=non_official_bypass,
    )
    for assay_id in pool.screen_ids:
        record = _record(pool, assay_id)
        _load_identity_frame(
            processed_root / f"{assay_id}.parquet",
            protocol=protocol,
            record=record,
        )
        metadata = embedding_root / f"{assay_id}.json"
        load_embedding_cache(
            embedding_root / f"{assay_id}.npy",
            metadata,
            dms_id=assay_id,
            row_hashes=record.row_hashes,
            model_id=protocol.model,
            model_revision=protocol.model_revision,
            sources=_embedding_sources(protocol),
            expected_embedding_dim=(
                _cache_width(metadata)
                if non_official_bypass
                else _ESM2_35M_EMBEDDING_DIM
            ),
        )


@app.command("verify")
def verify(
    context: typer.Context,
    base_manifest_path: Annotated[Path, typer.Option("--base-manifest")],
    pool_manifest_path: Annotated[Path, typer.Option("--pool-manifest")],
    processed_root: Annotated[Path, typer.Option("--processed-root")],
    embedding_root: Annotated[Path, typer.Option("--embedding-root")],
    results_root: Annotated[Path | None, typer.Option("--results-root")] = None,
    task_artifact: Annotated[Path | None, typer.Option("--task-artifact")] = None,
    aggregate_artifact: Annotated[
        Path | None,
        typer.Option("--aggregate-artifact"),
    ] = None,
    non_official_bypass: Annotated[
        bool,
        typer.Option("--non-official-bypass"),
    ] = False,
) -> None:
    """Verify inputs and optionally exact locality task/aggregate artifacts."""
    options = _options(context)

    def action() -> dict[str, object]:
        protocol, _, pool, commit = _load_context(
            config=options.config,
            base_manifest_path=base_manifest_path,
            pool_manifest_path=pool_manifest_path,
            non_official_bypass=non_official_bypass,
        )
        if options.dry_run:
            return {"plan": {"task_count": 45}}
        _verify_inputs(
            protocol,
            pool,
            processed_root=processed_root,
            embedding_root=embedding_root,
            non_official_bypass=non_official_bypass,
        )
        verified = ["base_manifest", "pool", "processed", "embeddings"]
        if task_artifact is not None:
            actual = _load_unique_json(task_artifact)
            _validate_task_payload(actual)
            task = cast(dict[str, object], actual["task"])
            expected = _build_task_payload(
                protocol=protocol,
                pool=pool,
                git_commit=commit,
                base_manifest_path=base_manifest_path,
                pool_manifest_path=pool_manifest_path,
                processed_root=processed_root,
                embedding_root=embedding_root,
                assay_id=cast(str, task["assay_id"]),
                seed=cast(int, task["seed"]),
                non_official_bypass=non_official_bypass,
            )
            _require_exact_payload(actual, expected, artifact_kind="locality task")
            verified.append("task")
        if results_root is not None:
            expected_aggregate = _build_aggregate_payload(
                protocol=protocol,
                pool=pool,
                git_commit=commit,
                base_manifest_path=base_manifest_path,
                pool_manifest_path=pool_manifest_path,
                processed_root=processed_root,
                embedding_root=embedding_root,
                results_root=results_root,
                non_official_bypass=non_official_bypass,
            )
            verified.append("exact_45_task_grid")
            if aggregate_artifact is not None:
                actual_aggregate = _load_unique_json(aggregate_artifact)
                _validate_aggregate_payload(actual_aggregate)
                _require_exact_payload(
                    actual_aggregate,
                    expected_aggregate,
                    artifact_kind="locality aggregate",
                )
                verified.append("aggregate")
        elif aggregate_artifact is not None:
            raise ValueError("aggregate verification requires --results-root")
        return {"verified": verified}

    _run_command("locality-verify", dry_run=options.dry_run, action=action)


if __name__ == "__main__":
    app()
