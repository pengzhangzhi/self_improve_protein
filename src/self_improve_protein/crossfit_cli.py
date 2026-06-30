"""Restart-safe CLI for the separately carded cross-fitted exploration."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, cast

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import typer

from self_improve_protein.analysis import (
    method_summary_table,
    pairwise_summary,
    validate_result_table,
)
from self_improve_protein.cli import (
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
    _validate_manifest,
    _write_json_once,
    load_evaluation_labels,
)
from self_improve_protein.config import Protocol, load_protocol
from self_improve_protein.crossfit import (
    CARD_ID,
    CARD_SHA,
    canonical_crossfit_fit_digest,
    evaluate_crossfit_task,
    fit_crossfit_task,
)
from self_improve_protein.crossfit_data import (
    CROSSFIT_POOL_SCHEMA_ID,
    CrossfitPoolManifest,
    load_crossfit_pool_manifest,
    validate_crossfit_pool_provenance,
)
from self_improve_protein.data import (
    DataManifest,
    SelectedAssayManifest,
    load_data_manifest,
    make_split,
)
from self_improve_protein.embeddings import load_embedding_cache
from self_improve_protein.experiment import (
    METHOD_NAMES,
    NUMERICAL_POLICY,
    FitInputs,
    canonical_evaluation_digest,
    canonical_protocol_digest,
    canonical_source_digest,
    require_openblas_coretype,
)
from self_improve_protein.provenance import sha256_file

app = typer.Typer(
    name="self-improve-protein-crossfit",
    help="Run the preregistered cross-fitted outer-gradient exploration.",
    invoke_without_command=True,
    no_args_is_help=False,
    pretty_exceptions_enable=False,
)

_SCHEMA_VERSION = 1
_LOCKED_V0_PROTOCOL_DIGEST = (
    "0b2a74ff76b8c7c508ceea16b004a1c128ba15704138138d49b2c153bcbfa49a"
)
_ESM2_35M_EMBEDDING_DIM = 480
_CROSSFIT_METHOD = "crossfit"


@dataclass(frozen=True, slots=True)
class _RuntimeOptions:
    config: Path
    dry_run: bool


@app.callback()
def main(
    context: typer.Context,
    config: Annotated[
        Path,
        typer.Option("--config", help="Validated frozen v0 protocol YAML."),
    ] = Path("configs/v0.yaml"),
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Plan without writing artifacts."),
    ] = False,
) -> None:
    """Set the protocol and mutation policy for every crossfit command."""
    context.obj = _RuntimeOptions(config=config, dry_run=dry_run)
    if context.invoked_subcommand is None:
        typer.echo(context.get_help())
        raise typer.Exit()


def _options(context: typer.Context) -> _RuntimeOptions:
    if not isinstance(context.obj, _RuntimeOptions):
        raise RuntimeError("CLI runtime options were not initialized")
    return context.obj


def _screen_grid(
    *,
    development_id: str,
    confirmatory_ids: tuple[str, ...],
    seeds: tuple[int, ...],
) -> tuple[tuple[str, int], ...]:
    """Return the preregistered development-first screen task order."""
    return tuple(
        (assay_id, seed)
        for assay_id in (development_id, *confirmatory_ids)
        for seed in seeds
    )


def _exact_screen_grid(
    pool: CrossfitPoolManifest,
    protocol: Protocol,
) -> tuple[tuple[str, int], ...]:
    grid = _screen_grid(
        development_id=pool.screen_ids[0],
        confirmatory_ids=pool.screen_ids[1:],
        seeds=protocol.seeds,
    )
    if len(pool.screen_ids) != 9 or len(protocol.seeds) != 5 or len(grid) != 45:
        raise ValueError("crossfit screen requires exactly 9 assays by 5 seeds")
    return grid


def _resolve_task_index(
    pool: CrossfitPoolManifest,
    protocol: Protocol,
    task_index: int,
) -> tuple[str, int]:
    grid = _exact_screen_grid(pool, protocol)
    if type(task_index) is not int or task_index < 0 or task_index >= len(grid):
        raise ValueError("task index is outside the exact 45-task screen grid")
    return grid[task_index]


def _record(
    pool: CrossfitPoolManifest,
    assay_id: str,
) -> SelectedAssayManifest:
    matches = [
        record for record in pool.selected_assays if record.dms_id == assay_id
    ]
    if len(matches) != 1:
        raise ValueError(f"assay {assay_id!r} is not unique in crossfit pool")
    return matches[0]


def _load_manifests(
    *,
    protocol: Protocol,
    base_manifest_path: Path,
    pool_manifest_path: Path,
) -> tuple[DataManifest, CrossfitPoolManifest]:
    base = load_data_manifest(base_manifest_path)
    _validate_manifest(protocol, base)
    pool = load_crossfit_pool_manifest(pool_manifest_path)
    validate_crossfit_pool_provenance(
        pool,
        protocol=protocol,
        base_manifest_path=base_manifest_path,
    )
    if pool_manifest_path.read_bytes() != _canonical_json_bytes(
        pool.model_dump(mode="json")
    ):
        raise ValueError("crossfit pool manifest must use canonical JSON bytes")
    _exact_screen_grid(pool, protocol)
    return base, pool


def _validate_execution_policy(
    protocol: Protocol,
    *,
    non_official_bypass: bool,
) -> None:
    if not non_official_bypass:
        if canonical_protocol_digest(protocol) != _LOCKED_V0_PROTOCOL_DIGEST:
            raise ValueError("official crossfit execution requires locked v0 protocol")
        require_openblas_coretype("Haswell")


def _finite_metric(row: dict[str, object], metric: str) -> None:
    value = row.get(metric)
    if (
        not isinstance(value, (int, float, np.integer, np.floating))
        or isinstance(value, (bool, np.bool_))
        or not np.isfinite(float(value))
    ):
        raise ValueError(f"crossfit task {metric} is invalid")


def _validate_crossfit_task_payload(payload: dict[str, object]) -> None:
    """Validate the separate task schema without widening locked v0 schemas."""
    required = {
        "card",
        "diagnostics",
        "digests",
        "execution",
        "kind",
        "provenance",
        "reference_methods",
        "schema_version",
        "task",
        "variant",
    }
    if (
        set(payload) != required
        or payload.get("schema_version") != 1
        or payload.get("kind") != "crossfit_task_result"
    ):
        raise ValueError("crossfit task artifact schema is invalid")
    card = payload.get("card")
    task = payload.get("task")
    digests = payload.get("digests")
    if (
        not isinstance(card, dict)
        or card.get("id") != CARD_ID
        or card.get("sha256") != CARD_SHA
        or not isinstance(task, dict)
        or task.get("phase") != "screen"
        or not isinstance(task.get("assay_id"), str)
        or type(task.get("seed")) is not int
        or not isinstance(digests, dict)
    ):
        raise ValueError("crossfit task identity is invalid")
    for name in ("base_fit", "crossfit_fit", "evaluation", "protocol", "source"):
        value = digests.get(name)
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError("crossfit task digest block is invalid")
    references = payload.get("reference_methods")
    if (
        not isinstance(references, list)
        or tuple(
            row.get("name") if isinstance(row, dict) else None for row in references
        )
        != METHOD_NAMES
    ):
        raise ValueError("crossfit task must preserve five locked references")
    for row in cast(list[dict[str, object]], references):
        for metric in ("spearman", "mse", "ndcg_10pct"):
            _finite_metric(row, metric)
    variant = payload.get("variant")
    if not isinstance(variant, dict) or variant.get("name") != _CROSSFIT_METHOD:
        raise ValueError("crossfit variant block is invalid")
    for metric in ("spearman", "mse", "ndcg_10pct"):
        _finite_metric(variant, metric)


def _method_row(
    *,
    artifact: Any,
    evaluation: Any,
    name: str,
) -> dict[str, object]:
    return {
        "mse": evaluation.mse,
        "name": name,
        "ndcg_10pct": evaluation.ndcg_10pct,
        "selected_hashes": list(artifact.selected_hashes),
        "selected_indices": list(artifact.selected_indices),
        "selected_pseudo_label_mae": evaluation.selected_pseudo_label_mae,
        "spearman": evaluation.spearman,
        "test_predictions": artifact.test_predictions.tolist(),
    }


def _build_crossfit_task_payload(
    *,
    protocol: Protocol,
    base_manifest_path: Path,
    pool_manifest: CrossfitPoolManifest,
    pool_manifest_path: Path,
    processed_root: Path,
    embedding_root: Path,
    assay_id: str,
    seed: int,
    non_official_bypass: bool,
) -> dict[str, object]:
    """Rebuild one task, freezing both fits before hidden outcomes are loaded."""
    _validate_execution_policy(
        protocol,
        non_official_bypass=non_official_bypass,
    )
    if (assay_id, seed) not in _exact_screen_grid(pool_manifest, protocol):
        raise ValueError("task identity is outside the exact screen grid")
    git_commit = _git_commit(require_clean=not non_official_bypass)
    record = _record(pool_manifest, assay_id)
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
    embedding_dim = (
        _cache_width(metadata_path)
        if non_official_bypass
        else _ESM2_35M_EMBEDDING_DIM
    )
    embeddings = load_embedding_cache(
        npy_path,
        metadata_path,
        dms_id=assay_id,
        row_hashes=record.row_hashes,
        model_id=protocol.model,
        model_revision=protocol.model_revision,
        sources=_embedding_sources(protocol),
        expected_embedding_dim=embedding_dim,
    )
    y_l = _ordered_outcomes(
        parquet_path,
        requested_hashes=split.labeled_hashes,
        expected_all_hashes=record.row_hashes,
    )
    source_digest = canonical_source_digest(protocol)
    teacher = identity[protocol.teacher_column].to_numpy(dtype=np.float64)
    fit_inputs = FitInputs(
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

    # Leakage boundary: both the locked reference fit and crossfit fit, including
    # their canonical digests, exist before this function can load hidden labels.
    fit = fit_crossfit_task(fit_inputs, protocol)
    base_fit_digest = fit.base_fit_digest
    crossfit_fit_digest = canonical_crossfit_fit_digest(fit)
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
    evaluation = evaluate_crossfit_task(
        fit,
        labels,
        protocol=protocol,
        expected_crossfit_fit_digest=crossfit_fit_digest,
        expected_evaluation_digest=evaluation_digest,
    )
    measured = {item.name: item for item in evaluation.reference.methods}
    reference_methods = [
        _method_row(
            artifact=method,
            evaluation=measured[method.name],
            name=method.name,
        )
        for method in fit.base_fit.methods
    ]
    variant = _method_row(
        artifact=fit.method,
        evaluation=evaluation.crossfit,
        name=_CROSSFIT_METHOD,
    )
    payload: dict[str, object] = {
        "card": {"id": CARD_ID, "sha256": CARD_SHA},
        "diagnostics": {
            "evaluation": {
                "crossfit_test_risk_oracle": evaluation.test_risk_oracle,
                "full_test_risk_oracle": (
                    evaluation.reference.full_test_risk_oracle
                ),
                "no_hessian_test_risk_oracle": (
                    evaluation.reference.no_hessian_test_risk_oracle
                ),
                "pool_pseudo_label_mae": (
                    evaluation.reference.pool_pseudo_label_mae
                ),
                "random_error_reference": (
                    evaluation.reference.random_error_reference
                ),
                "teacher_test_spearman": (
                    evaluation.reference.teacher_test_spearman
                ),
            },
            "fit": {
                "crossfit": fit.diagnostics,
                "reference": fit.base_fit.diagnostics,
            },
        },
        "digests": {
            "base_fit": base_fit_digest,
            "crossfit_fit": crossfit_fit_digest,
            "evaluation": evaluation_digest,
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
        "kind": "crossfit_task_result",
        "provenance": {
            "base_manifest_sha256": sha256_file(base_manifest_path),
            "embedding_metadata_sha256": sha256_file(metadata_path),
            "embedding_npy_sha256": sha256_file(npy_path),
            "git_commit": git_commit,
            "pool_manifest_sha256": sha256_file(pool_manifest_path),
            "pool_schema_id": CROSSFIT_POOL_SCHEMA_ID,
            "processed_sha256": sha256_file(parquet_path),
            "sources": {
                "metadata": protocol.metadata_sha256,
                "scores": protocol.zero_shot_scores_sha256,
                "substitutions": protocol.substitutions_sha256,
                "upstream_commit": protocol.proteingym_upstream_commit,
            },
        },
        "reference_methods": reference_methods,
        "schema_version": 1,
        "task": {
            "assay_id": assay_id,
            "labeled_hashes": list(split.labeled_hashes),
            "phase": "screen",
            "seed": seed,
            "test_hashes": list(split.test_hashes),
            "unlabeled_hashes": list(split.unlabeled_hashes),
        },
        "variant": variant,
    }
    normalized = cast(dict[str, object], _jsonable(payload))
    _validate_crossfit_task_payload(normalized)
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
    """Run one immutable member of the exact 45-task screen."""
    options = _options(context)

    def action() -> dict[str, object]:
        protocol = load_protocol(options.config)
        _, pool = _load_manifests(
            protocol=protocol,
            base_manifest_path=base_manifest_path,
            pool_manifest_path=pool_manifest_path,
        )
        assay_id, seed = _resolve_task_index(pool, protocol, task_index)
        destination = _task_path(results_root, assay_id, seed)
        if options.dry_run:
            return {
                "plan": {
                    "assay_id": assay_id,
                    "output": str(destination),
                    "seed": seed,
                    "stages": [
                        "validate_provenance",
                        "load_labeled_only",
                        "fit_and_freeze_base_and_crossfit_digests",
                        "load_hidden_labels",
                        "evaluate_and_write_once",
                    ],
                }
            }
        payload = _build_crossfit_task_payload(
            protocol=protocol,
            base_manifest_path=base_manifest_path,
            pool_manifest=pool,
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

    _run_command("crossfit-run-task", dry_run=options.dry_run, action=action)


def _summary_payload(summary: Any) -> dict[str, object]:
    return cast(dict[str, object], _jsonable(dataclasses.asdict(summary)))


def _validate_crossfit_aggregate_payload(payload: dict[str, object]) -> None:
    if (
        payload.get("schema_version") != 1
        or payload.get("kind") != "crossfit_aggregate_result"
    ):
        raise ValueError("crossfit aggregate artifact schema is invalid")
    grid = payload.get("grid")
    provenance = payload.get("provenance")
    promotion = payload.get("promotion")
    if (
        not isinstance(grid, dict)
        or len(cast(list[object], grid.get("screen_tasks", []))) != 45
        or len(cast(list[object], grid.get("primary_tasks", []))) != 40
        or not isinstance(provenance, dict)
        or provenance.get("task_count") != 45
        or not isinstance(promotion, dict)
    ):
        raise ValueError("crossfit aggregate grid is invalid")


def _build_crossfit_aggregate_payload(
    *,
    protocol: Protocol,
    base_manifest_path: Path,
    pool_manifest: CrossfitPoolManifest,
    pool_manifest_path: Path,
    processed_root: Path,
    embedding_root: Path,
    results_root: Path,
    non_official_bypass: bool,
) -> dict[str, object]:
    """Independently rebuild all 45 tasks and derive the frozen 40-task gate."""
    _validate_execution_policy(
        protocol,
        non_official_bypass=non_official_bypass,
    )
    screen_grid = _exact_screen_grid(pool_manifest, protocol)
    development_id = pool_manifest.screen_ids[0]
    primary_ids = pool_manifest.screen_ids[1:]
    primary_grid = tuple(
        (assay_id, seed)
        for assay_id in primary_ids
        for seed in protocol.seeds
    )
    long_rows: list[dict[str, object]] = []
    task_manifest: list[dict[str, object]] = []
    diagnostic_rows: list[dict[str, object]] = []
    git_commits: set[str] = set()
    runtimes: set[bytes] = set()
    numerical_runtime: object = None
    for assay_id, seed in screen_grid:
        path = _task_path(results_root, assay_id, seed)
        if not path.is_file():
            raise ValueError(f"missing crossfit task artifact: {path}")
        actual = _load_unique_json(path)
        _validate_crossfit_task_payload(actual)
        expected = _build_crossfit_task_payload(
            protocol=protocol,
            base_manifest_path=base_manifest_path,
            pool_manifest=pool_manifest,
            pool_manifest_path=pool_manifest_path,
            processed_root=processed_root,
            embedding_root=embedding_root,
            assay_id=assay_id,
            seed=seed,
            non_official_bypass=non_official_bypass,
        )
        _require_exact_payload(actual, expected, artifact_kind="crossfit task")
        provenance = cast(dict[str, object], actual["provenance"])
        execution = cast(dict[str, object], actual["execution"])
        raw_git_commit = provenance.get("git_commit")
        if not isinstance(raw_git_commit, str):
            raise ValueError("crossfit task git commit is invalid")
        git_commits.add(raw_git_commit)
        numerical_runtime = execution.get("numerical_runtime")
        runtimes.add(_canonical_json_bytes(numerical_runtime))
        rows = [
            *cast(list[dict[str, object]], actual["reference_methods"]),
            cast(dict[str, object], actual["variant"]),
        ]
        for row in rows:
            long_rows.append(
                {
                    "assay_id": assay_id,
                    "method": row["name"],
                    "mse": row["mse"],
                    "ndcg_10pct": row["ndcg_10pct"],
                    "seed": seed,
                    "spearman": row["spearman"],
                }
            )
        digest = sha256_file(path)
        task_manifest.append(
            {"assay_id": assay_id, "seed": seed, "sha256": digest}
        )
        diagnostic_rows.append(
            {
                "assay_id": assay_id,
                "diagnostics": actual["diagnostics"],
                "seed": seed,
                "task_artifact_sha256": digest,
            }
        )
    if len(git_commits) != 1 or len(runtimes) != 1:
        raise ValueError("crossfit task grid mixes implementations or runtimes")
    required_methods = (*METHOD_NAMES, _CROSSFIT_METHOD)
    all_results = validate_result_table(
        pd.DataFrame(long_rows),
        assay_ids=pool_manifest.screen_ids,
        seeds=protocol.seeds,
        required_methods=required_methods,
    )
    primary_results = validate_result_table(
        all_results.loc[all_results["assay_id"].isin(primary_ids)].copy(),
        assay_ids=primary_ids,
        seeds=protocol.seeds,
        required_methods=required_methods,
    )
    development_results = validate_result_table(
        all_results.loc[all_results["assay_id"] == development_id].copy(),
        assay_ids=(development_id,),
        seeds=protocol.seeds,
        required_methods=required_methods,
    )
    comparisons = {
        second: pairwise_summary(
            primary_results,
            first=_CROSSFIT_METHOD,
            second=second,
            metric="spearman",
        )
        for second in METHOD_NAMES
    }
    primary = comparisons["random"]
    promotes = (
        primary.mean_gain > 0.0
        and primary.task_wins >= 25
        and primary.assay_wins >= 5
    )
    payload: dict[str, object] = {
        "analysis": {
            "inference_unit": "assay",
            "metric": "spearman",
            "sign_flip": "exact",
        },
        "card": {"id": CARD_ID, "sha256": CARD_SHA},
        "development_diagnostics": diagnostic_rows[: len(protocol.seeds)],
        "development_long_results": development_results.to_dict(orient="records"),
        "effects": {
            f"crossfit_minus_{second}": _summary_payload(summary)
            for second, summary in comparisons.items()
        },
        "execution": {
            "bypass": (
                "non_official_test_or_development" if non_official_bypass else None
            ),
            "numerical_policy": NUMERICAL_POLICY,
            "numerical_runtime": numerical_runtime,
            "official": not non_official_bypass,
        },
        "grid": {
            "development_assay_id": development_id,
            "primary_assay_ids": list(primary_ids),
            "primary_tasks": [list(task) for task in primary_grid],
            "screen_assay_ids": list(pool_manifest.screen_ids),
            "screen_tasks": [list(task) for task in screen_grid],
            "seeds": list(protocol.seeds),
        },
        "kind": "crossfit_aggregate_result",
        "method_table": method_summary_table(primary_results).to_dict(
            orient="records"
        ),
        "primary_long_results": primary_results.to_dict(orient="records"),
        "promotion": {
            "assay_win_threshold": 5,
            "development_assay_excluded": True,
            "mean_gain_strictly_positive": True,
            "promote_to_untouched_replication": promotes,
            "task_win_threshold": 25,
            "untouched_assay_ids": list(pool_manifest.untouched_ids),
        },
        "provenance": {
            "base_manifest_sha256": sha256_file(base_manifest_path),
            "git_commit": git_commits.pop(),
            "pool_manifest_sha256": sha256_file(pool_manifest_path),
            "protocol_digest": canonical_protocol_digest(protocol),
            "task_count": len(task_manifest),
            "task_manifest": task_manifest,
        },
        "schema_version": 1,
        "task_diagnostics": diagnostic_rows,
    }
    normalized = cast(dict[str, object], _jsonable(payload))
    _validate_crossfit_aggregate_payload(normalized)
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
    """Rebuild the 45-task screen and evaluate the 40-task frozen gate."""
    options = _options(context)

    def action() -> dict[str, object]:
        protocol = load_protocol(options.config)
        _, pool = _load_manifests(
            protocol=protocol,
            base_manifest_path=base_manifest_path,
            pool_manifest_path=pool_manifest_path,
        )
        grid = _exact_screen_grid(pool, protocol)
        if options.dry_run:
            return {
                "plan": {
                    "output": str(output),
                    "task_artifacts": [
                        str(_task_path(results_root, assay_id, seed))
                        for assay_id, seed in grid
                    ],
                }
            }
        payload = _build_crossfit_aggregate_payload(
            protocol=protocol,
            base_manifest_path=base_manifest_path,
            pool_manifest=pool,
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
            "promote_to_untouched_replication": cast(
                dict[str, object], payload["promotion"]
            )["promote_to_untouched_replication"],
            "task_count": 45,
        }

    _run_command("crossfit-aggregate", dry_run=options.dry_run, action=action)


def _verify_screen_inputs(
    *,
    protocol: Protocol,
    pool: CrossfitPoolManifest,
    processed_root: Path,
    embedding_root: Path,
    non_official_bypass: bool,
) -> None:
    for assay_id in pool.screen_ids:
        record = _record(pool, assay_id)
        _load_identity_frame(
            processed_root / f"{assay_id}.parquet",
            protocol=protocol,
            record=record,
        )
        metadata_path = embedding_root / f"{assay_id}.json"
        load_embedding_cache(
            embedding_root / f"{assay_id}.npy",
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


@app.command("verify")
def verify(
    context: typer.Context,
    base_manifest_path: Annotated[Path, typer.Option("--base-manifest")],
    pool_manifest_path: Annotated[Path, typer.Option("--pool-manifest")],
    processed_root: Annotated[Path | None, typer.Option("--processed-root")] = None,
    embedding_root: Annotated[Path | None, typer.Option("--embedding-root")] = None,
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
    """Fail closed on pool, cache, task, and aggregate tampering."""
    options = _options(context)

    def action() -> dict[str, object]:
        protocol = load_protocol(options.config)
        _, pool = _load_manifests(
            protocol=protocol,
            base_manifest_path=base_manifest_path,
            pool_manifest_path=pool_manifest_path,
        )
        _validate_execution_policy(
            protocol,
            non_official_bypass=non_official_bypass,
        )
        _git_commit(require_clean=not non_official_bypass)
        if (processed_root is None) != (embedding_root is None):
            raise ValueError("processed and embedding roots must be supplied together")
        if results_root is not None and (
            processed_root is None or embedding_root is None
        ):
            raise ValueError("results verification requires processed and cache roots")
        if (task_artifact is not None or aggregate_artifact is not None) and (
            processed_root is None or embedding_root is None
        ):
            raise ValueError("artifact verification requires processed and cache roots")
        if options.dry_run:
            return {"plan": {"stages": ["verify_exact_crossfit_provenance"]}}
        verified = ["base_manifest", "crossfit_pool"]
        if processed_root is not None and embedding_root is not None:
            _verify_screen_inputs(
                protocol=protocol,
                pool=pool,
                processed_root=processed_root,
                embedding_root=embedding_root,
                non_official_bypass=non_official_bypass,
            )
            verified.extend(("processed_screen", "embedding_screen"))
        if task_artifact is not None:
            actual = _load_unique_json(task_artifact)
            _validate_crossfit_task_payload(actual)
            task = cast(dict[str, object], actual["task"])
            assay_id = cast(str, task["assay_id"])
            seed = cast(int, task["seed"])
            assert processed_root is not None and embedding_root is not None
            expected = _build_crossfit_task_payload(
                protocol=protocol,
                base_manifest_path=base_manifest_path,
                pool_manifest=pool,
                pool_manifest_path=pool_manifest_path,
                processed_root=processed_root,
                embedding_root=embedding_root,
                assay_id=assay_id,
                seed=seed,
                non_official_bypass=non_official_bypass,
            )
            _require_exact_payload(
                actual,
                expected,
                artifact_kind="crossfit task",
            )
            verified.append("task_artifact")
        if results_root is not None:
            assert processed_root is not None and embedding_root is not None
            expected_aggregate = _build_crossfit_aggregate_payload(
                protocol=protocol,
                base_manifest_path=base_manifest_path,
                pool_manifest=pool,
                pool_manifest_path=pool_manifest_path,
                processed_root=processed_root,
                embedding_root=embedding_root,
                results_root=results_root,
                non_official_bypass=non_official_bypass,
            )
            verified.append("exact_45_task_grid")
            if aggregate_artifact is not None:
                actual_aggregate = _load_unique_json(aggregate_artifact)
                _validate_crossfit_aggregate_payload(actual_aggregate)
                _require_exact_payload(
                    actual_aggregate,
                    expected_aggregate,
                    artifact_kind="crossfit aggregate",
                )
                verified.append("aggregate_artifact")
        elif aggregate_artifact is not None:
            raise ValueError("aggregate verification requires --results-root")
        return {"verified": verified}

    _run_command("crossfit-verify", dry_run=options.dry_run, action=action)


if __name__ == "__main__":
    app()
