"""Restart-safe CLI for the predeclared exact-CV greedy screen."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, cast

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import typer
from numpy.typing import NDArray

from self_improve_protein.cli import (
    _ESM2_35M_EMBEDDING_DIM,
    _cache_width,
    _canonical_json_bytes,
    _embedding_sources,
    _git_commit,
    _jsonable,
    _load_identity_frame,
    _load_unique_json,
    _require_exact_payload,
    _run_command,
    _write_json_once,
)
from self_improve_protein.cli import (
    _ordered_outcomes as _ordered_outcomes,
)
from self_improve_protein.cli import (
    load_evaluation_labels as load_evaluation_labels,
)
from self_improve_protein.config import Protocol, load_protocol
from self_improve_protein.crossfit_cli import (
    _load_manifests as _load_manifests,
)
from self_improve_protein.crossfit_cli import (
    _method_row,
    _record,
    _screen_grid,
    _validate_execution_policy,
)
from self_improve_protein.crossfit_data import CrossfitPoolManifest
from self_improve_protein.data import DataManifest, make_split
from self_improve_protein.embeddings import load_embedding_cache
from self_improve_protein.exact_cv import (
    CARD_ID as CARD_ID,
)
from self_improve_protein.exact_cv import (
    CARD_SHA as CARD_SHA,
)
from self_improve_protein.exact_cv import (
    FOLD_COUNT,
    FOLD_PURPOSE,
    PREFIX_COUNTS,
    REANCHOR_STEPS,
    ExactCVEvaluationResult,
    ExactCVFitResult,
    ExactCVPrefixArtifact,
    ExactCVPrefixEvaluation,
    evaluate_exact_cv_task,
)
from self_improve_protein.exact_cv import (
    canonical_exact_cv_fit_digest as canonical_exact_cv_fit_digest,
)
from self_improve_protein.exact_cv import (
    fit_exact_cv_task as fit_exact_cv_task,
)
from self_improve_protein.experiment import (
    METHOD_NAMES,
    NUMERICAL_POLICY,
    FitInputs,
    canonical_evaluation_digest,
    canonical_protocol_digest,
    canonical_source_digest,
)
from self_improve_protein.provenance import sha256_bytes
from self_improve_protein.provenance import sha256_file as sha256_file

app = typer.Typer(
    name="self-improve-protein-exact-cv",
    help="Run and verify the predeclared nine-assay exact-CV screen.",
    invoke_without_command=True,
    no_args_is_help=False,
    pretty_exceptions_enable=False,
)

_SCHEMA_VERSION = 1
_METHOD_NAME = "exact_cv"


@dataclass(frozen=True, slots=True)
class _RuntimeOptions:
    config: Path
    dry_run: bool


@app.callback()
def main(
    context: typer.Context,
    config: Annotated[Path, typer.Option("--config")] = Path("configs/v0.yaml"),
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
) -> None:
    """Set the frozen protocol and mutation policy."""
    context.obj = _RuntimeOptions(config=config, dry_run=dry_run)
    if context.invoked_subcommand is None:
        typer.echo(context.get_help())
        raise typer.Exit()


def _options(context: typer.Context) -> _RuntimeOptions:
    if not isinstance(context.obj, _RuntimeOptions):
        raise RuntimeError("exact-CV CLI runtime options were not initialized")
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
        raise ValueError("exact-CV screen requires exactly 9 assays by 5 seeds")
    return grid


def _resolve_task_index(
    pool: CrossfitPoolManifest,
    protocol: Protocol,
    task_index: int,
) -> tuple[str, int]:
    grid = _exact_grid(pool, protocol)
    if type(task_index) is not int or not 0 <= task_index < len(grid):
        raise ValueError("task index is outside the exact 45-task exact-CV grid")
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
        raise ValueError("official exact-CV screen requires processed/v0")
    if embedding_root.parts[-2:] != ("embeddings", "v0"):
        raise ValueError("official exact-CV screen requires embeddings/v0")


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


def _task_path(results_root: Path, assay_id: str, seed: int) -> Path:
    return results_root / "tasks" / assay_id / f"seed_{seed}.json"


def _receipt_path(results_root: Path, assay_id: str, seed: int) -> Path:
    return results_root / "receipts" / assay_id / f"seed_{seed}.json"


def _prefix_payload(
    artifact: ExactCVPrefixArtifact,
    evaluation: ExactCVPrefixEvaluation,
    *,
    overlaps: dict[str, float],
) -> dict[str, object]:
    if artifact.q != evaluation.q:
        raise ValueError("exact-CV prefix fit/evaluation identities differ")
    return {
        **cast(dict[str, object], _jsonable(dataclasses.asdict(evaluation))),
        "coefficients": artifact.coefficients.tolist(),
        **overlaps,
        "selected_hashes": list(artifact.selected_hashes),
        "selected_indices": list(artifact.selected_indices),
        "test_predictions": artifact.test_predictions.tolist(),
        "training_weight_sum": artifact.training_weight_sum,
    }


def _prefix_overlaps(
    fit: ExactCVFitResult,
    artifact: ExactCVPrefixArtifact,
) -> dict[str, float]:
    selected = set(artifact.selected_indices)
    locked: dict[str, set[int]] = {
        method.name: set(method.selected_indices) for method in fit.base_fit.methods
    }
    mapping = {
        "overlap_full": "ours",
        "overlap_no_hessian": "no_hessian",
        "overlap_random": "random",
        "overlap_top_teacher": "top_teacher",
    }
    return {
        field: len(selected & locked[method]) / artifact.q
        for field, method in mapping.items()
    }


def _fold_payload(fit: ExactCVFitResult) -> list[dict[str, object]]:
    return [
        {
            "coefficient_absolute_drift": fold.coefficient_absolute_drift,
            "coefficient_relative_drift": fold.coefficient_relative_drift,
            "direct_normal_equation_residual": (
                fold.direct_normal_equation_residual
            ),
            "feature_scale": fold.feature_transform.scale,
            "final_validation_mse": fold.final_validation_mse,
            "fold_id": fold.fold_id,
            "initial_validation_mse": fold.initial_validation_mse,
            "label_mean": fold.label_transform.mean,
            "label_scale": fold.label_transform.scale,
            "regularizer_mass": fold.regularizer_mass,
            "teacher_intercept": fold.teacher_calibration.intercept,
            "teacher_slope": fold.teacher_calibration.slope,
            "training_indices": list(fold.training_indices),
            "validation_indices": list(fold.validation_indices),
            "validation_prediction_max_absolute_drift": (
                fold.validation_prediction_max_absolute_drift
            ),
        }
        for fold in fit.folds
    ]


def _validate_task_payload(payload: dict[str, object]) -> None:
    required = {
        "card",
        "digests",
        "execution",
        "folds",
        "greedy",
        "kind",
        "prefixes",
        "provenance",
        "reference_methods",
        "schema_version",
        "task",
    }
    if (
        set(payload) != required
        or payload.get("schema_version") != _SCHEMA_VERSION
        or payload.get("kind") != "exact_cv_task_result"
    ):
        raise ValueError("exact-CV task schema is invalid")
    card = payload.get("card")
    task = payload.get("task")
    digests = payload.get("digests")
    if (
        card != {"id": CARD_ID, "sha256": CARD_SHA}
        or not isinstance(task, dict)
        or task.get("phase") != "exact_cv_screen"
        or not isinstance(task.get("assay_id"), str)
        or type(task.get("seed")) is not int
        or not isinstance(digests, dict)
        or any(
            not _is_sha256(digests.get(name))
            for name in ("base_fit", "evaluation", "fit", "protocol", "source")
        )
    ):
        raise ValueError("exact-CV task identity is invalid")
    references = payload.get("reference_methods")
    if (
        not isinstance(references, list)
        or tuple(
            row.get("name") if isinstance(row, dict) else None
            for row in references
        )
        != METHOD_NAMES
    ):
        raise ValueError("exact-CV reference methods are invalid")
    folds = payload.get("folds")
    if (
        not isinstance(folds, list)
        or len(folds) != FOLD_COUNT
        or tuple(
            row.get("fold_id") if isinstance(row, dict) else None for row in folds
        )
        != tuple(range(FOLD_COUNT))
    ):
        raise ValueError("exact-CV fold payload is invalid")
    greedy = payload.get("greedy")
    if (
        not isinstance(greedy, dict)
        or greedy.get("fold_purpose") != FOLD_PURPOSE
        or greedy.get("reanchor_steps") != list(REANCHOR_STEPS)
        or not isinstance(greedy.get("ordered_indices"), list)
        or len(cast(list[object], greedy["ordered_indices"])) != 192
        or not isinstance(greedy.get("steps"), list)
        or len(cast(list[object], greedy["steps"])) != 192
    ):
        raise ValueError("exact-CV greedy payload is invalid")
    prefixes = payload.get("prefixes")
    if (
        not isinstance(prefixes, list)
        or tuple(
            row.get("q") if isinstance(row, dict) else None for row in prefixes
        )
        != PREFIX_COUNTS
    ):
        raise ValueError("exact-CV prefix payload is invalid")
    ordering = cast(list[int], greedy["ordered_indices"])
    locked = {
        cast(str, row["name"]): set(cast(list[int], row["selected_indices"]))
        for row in cast(list[dict[str, object]], references)
    }
    overlap_mapping = {
        "overlap_full": "ours",
        "overlap_no_hessian": "no_hessian",
        "overlap_random": "random",
        "overlap_top_teacher": "top_teacher",
    }
    for row in cast(list[dict[str, object]], prefixes):
        q = cast(int, row["q"])
        if row.get("selected_indices") != ordering[:q]:
            raise ValueError("exact-CV prefixes are not nested ordering prefixes")
        selected = set(cast(list[int], row["selected_indices"]))
        for field, method in overlap_mapping.items():
            value = row.get(field)
            expected = len(selected & locked[method]) / q
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not np.isfinite(float(value))
                or float(value) != expected
            ):
                raise ValueError("exact-CV prefix overlap diagnostic is invalid")


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
        raise ValueError("task is outside the exact exact-CV screen grid")
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

    # Leakage boundary: the full ordering, prefix fits/predictions, and its
    # reconstructive digest are frozen before hidden pool/test labels load.
    fit = fit_exact_cv_task(inputs, protocol)
    fit_digest = canonical_exact_cv_fit_digest(fit)
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
    evaluation = evaluate_exact_cv_task(
        fit,
        labels,
        protocol=protocol,
        expected_fit_digest=fit_digest,
        expected_evaluation_digest=evaluation_digest,
    )
    if not isinstance(evaluation, ExactCVEvaluationResult):
        raise ValueError("exact-CV evaluation result is invalid")
    measured = {item.name: item for item in evaluation.reference.methods}
    references = [
        _method_row(
            artifact=method,
            evaluation=measured[method.name],
            name=method.name,
        )
        for method in fit.base_fit.methods
    ]
    prefixes = [
        _prefix_payload(
            artifact,
            result,
            overlaps=_prefix_overlaps(fit, artifact),
        )
        for artifact, result in zip(
            fit.prefixes,
            evaluation.prefixes,
            strict=True,
        )
    ]
    payload: dict[str, object] = {
        "card": {"id": CARD_ID, "sha256": CARD_SHA},
        "digests": {
            "base_fit": fit.base_fit_digest,
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
        "folds": _fold_payload(fit),
        "greedy": {
            "fold_assignment": fit.fold_assignment.tolist(),
            "fold_purpose": FOLD_PURPOSE,
            "ordered_hashes": list(fit.greedy.ordered_hashes),
            "ordered_indices": list(fit.greedy.ordered_indices),
            "reanchor_steps": list(fit.greedy.reanchor_steps),
            "regularizer_mass": fit.greedy.regularizer_mass,
            "steps": [dataclasses.asdict(step) for step in fit.greedy.steps],
        },
        "kind": "exact_cv_task_result",
        "prefixes": prefixes,
        "provenance": {
            "base_manifest_sha256": sha256_file(base_manifest_path),
            "embedding_metadata_sha256": sha256_file(metadata_path),
            "embedding_npy_sha256": sha256_file(npy_path),
            "git_commit": git_commit,
            "pool_manifest_sha256": sha256_file(pool_manifest_path),
            "processed_sha256": sha256_file(parquet_path),
        },
        "reference_methods": references,
        "schema_version": _SCHEMA_VERSION,
        "task": {
            "assay_id": assay_id,
            "labeled_hashes": list(split.labeled_hashes),
            "phase": "exact_cv_screen",
            "seed": seed,
            "test_hashes": list(split.test_hashes),
            "unlabeled_hashes": list(split.unlabeled_hashes),
        },
    }
    normalized = cast(dict[str, object], _jsonable(payload))
    _validate_task_payload(normalized)
    return normalized


@dataclass(frozen=True, slots=True)
class _ProbeFitInputs:
    inputs: FitInputs
    parquet_path: Path
    metadata_path: Path
    npy_path: Path
    identity_teacher_projection_digest: str
    labeled_projection_digest: str


def _identity_teacher_projection_digest(
    identity: pd.DataFrame,
    *,
    teacher_column: str,
) -> str:
    rows = [
        [
            cast(str, dms_id),
            cast(str, mutant),
            cast(str, sequence),
            cast(str, sequence_hash),
            float(teacher),
        ]
        for dms_id, mutant, sequence, sequence_hash, teacher in identity.loc[
            :,
            [
                "dms_id",
                "mutant",
                "mutated_sequence",
                "sequence_hash",
                teacher_column,
            ],
        ].itertuples(index=False, name=None)
    ]
    return sha256_bytes(
        _canonical_json_bytes(
            {
                "columns": [
                    "dms_id",
                    "mutant",
                    "mutated_sequence",
                    "sequence_hash",
                    teacher_column,
                ],
                "rows": rows,
            }
        )
    )


def _labeled_projection_digest(
    labeled_hashes: tuple[str, ...],
    y_l: NDArray[np.float64],
) -> str:
    if len(labeled_hashes) != 96 or y_l.shape != (96,):
        raise ValueError("fit probe labeled projection must contain exactly 96 rows")
    return sha256_bytes(
        _canonical_json_bytes(
            {
                "labeled_hashes": list(labeled_hashes),
                "labeled_outcomes": [float(value) for value in y_l],
            }
        )
    )


def _load_probe_fit_inputs(
    *,
    protocol: Protocol,
    pool: CrossfitPoolManifest,
    processed_root: Path,
    embedding_root: Path,
    assay_id: str,
    seed: int,
    non_official_bypass: bool,
) -> _ProbeFitInputs:
    """Load outcome-free covariates plus exactly the 96 labeled outcomes."""
    if (assay_id, seed) not in _exact_grid(pool, protocol):
        raise ValueError("fit probe task is outside the exact screen grid")
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
        expected_embedding_dim=_ESM2_35M_EMBEDDING_DIM,
    )
    y_l = _ordered_outcomes(
        parquet_path,
        requested_hashes=split.labeled_hashes,
        expected_all_hashes=record.row_hashes,
    )
    identity_digest = _identity_teacher_projection_digest(
        identity,
        teacher_column=protocol.teacher_column,
    )
    labeled_digest = _labeled_projection_digest(split.labeled_hashes, y_l)
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
    return _ProbeFitInputs(
        inputs=inputs,
        parquet_path=parquet_path,
        metadata_path=metadata_path,
        npy_path=npy_path,
        identity_teacher_projection_digest=identity_digest,
        labeled_projection_digest=labeled_digest,
    )


def _validate_probe_payload(payload: dict[str, object]) -> None:
    required = {
        "card",
        "diagnostics",
        "digests",
        "dimensions",
        "execution",
        "greedy",
        "hidden_outcomes_loaded",
        "kind",
        "provenance",
        "schema_version",
        "task",
    }
    dimensions = payload.get("dimensions")
    digests = payload.get("digests")
    diagnostics = payload.get("diagnostics")
    greedy = payload.get("greedy")
    execution = payload.get("execution")
    provenance = payload.get("provenance")
    if (
        set(payload) != required
        or payload.get("schema_version") != _SCHEMA_VERSION
        or payload.get("kind") != "exact_cv_real_shape_fit_probe"
        or payload.get("card") != {"id": CARD_ID, "sha256": CARD_SHA}
        or payload.get("hidden_outcomes_loaded") is not False
        or dimensions
        != {
            "embedding_width": 480,
            "n_labeled": 96,
            "n_test": 1000,
            "n_unlabeled": 2000,
        }
        or not isinstance(digests, dict)
        or set(digests) != {"base_fit", "fit", "protocol", "source"}
        or any(not _is_sha256(value) for value in digests.values())
        or not isinstance(diagnostics, dict)
        or not isinstance(greedy, dict)
        or not isinstance(execution, dict)
        or not isinstance(provenance, dict)
        or set(provenance)
        != {
            "base_manifest_sha256",
            "embedding_metadata_sha256",
            "embedding_npy_sha256",
            "git_commit",
            "identity_teacher_projection_digest",
            "labeled_projection_digest",
            "pool_manifest_sha256",
        }
        or any(
            not _is_sha256(provenance.get(name))
            for name in (
                "base_manifest_sha256",
                "embedding_metadata_sha256",
                "embedding_npy_sha256",
                "identity_teacher_projection_digest",
                "labeled_projection_digest",
                "pool_manifest_sha256",
            )
        )
    ):
        raise ValueError("exact-CV real-shape fit probe schema is invalid")
    for name in (
        "coefficient_absolute_drift_max",
        "coefficient_relative_drift_max",
        "direct_normal_equation_residual_max",
        "validation_prediction_absolute_drift_max",
    ):
        value = diagnostics.get(name)
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not np.isfinite(float(value))
            or float(value) < 0.0
        ):
            raise ValueError("exact-CV fit probe diagnostics are invalid")
    minimum_denominator = greedy.get("minimum_sherman_morrison_denominator")
    final_cv_mse = greedy.get("final_cv_mse")
    if (
        greedy.get("q") != 192
        or greedy.get("reanchor_steps") != list(REANCHOR_STEPS)
        or not isinstance(minimum_denominator, (int, float))
        or isinstance(minimum_denominator, bool)
        or not np.isfinite(float(minimum_denominator))
        or float(minimum_denominator) < 1.0
        or not isinstance(final_cv_mse, (int, float))
        or isinstance(final_cv_mse, bool)
        or not np.isfinite(float(final_cv_mse))
        or float(final_cv_mse) < 0.0
    ):
        raise ValueError("exact-CV fit probe greedy state is invalid")
    if (
        execution.get("numerical_policy") != NUMERICAL_POLICY
        or not isinstance(execution.get("numerical_runtime"), dict)
        or type(execution.get("official")) is not bool
        or execution.get("bypass")
        != (
            None
            if execution.get("official") is True
            else "non_official_test_or_development"
        )
    ):
        raise ValueError("exact-CV fit probe execution state is invalid")


def _validate_probe_against_roots(
    payload: dict[str, object],
    *,
    protocol: Protocol,
    pool: CrossfitPoolManifest,
    git_commit: str,
    base_manifest_path: Path,
    pool_manifest_path: Path,
    processed_root: Path,
    embedding_root: Path,
) -> None:
    """Bind one official probe to current task-0 files and trust roots."""
    _validate_probe_payload(payload)
    assay_id, seed = _resolve_task_index(pool, protocol, 0)
    execution = cast(dict[str, object], payload["execution"])
    task = cast(dict[str, object], payload["task"])
    provenance = cast(dict[str, object], payload["provenance"])
    digests = cast(dict[str, object], payload["digests"])
    loaded = _load_probe_fit_inputs(
        protocol=protocol,
        pool=pool,
        processed_root=processed_root,
        embedding_root=embedding_root,
        assay_id=assay_id,
        seed=seed,
        non_official_bypass=False,
    )
    expected_provenance = {
        "base_manifest_sha256": sha256_file(base_manifest_path),
        "embedding_metadata_sha256": sha256_file(
            embedding_root / f"{assay_id}.json"
        ),
        "embedding_npy_sha256": sha256_file(
            embedding_root / f"{assay_id}.npy"
        ),
        "git_commit": git_commit,
        "identity_teacher_projection_digest": (
            loaded.identity_teacher_projection_digest
        ),
        "labeled_projection_digest": loaded.labeled_projection_digest,
        "pool_manifest_sha256": sha256_file(pool_manifest_path),
    }
    if (
        execution.get("official") is not True
        or execution.get("bypass") is not None
        or task != {"assay_id": assay_id, "index": 0, "seed": seed}
        or provenance != expected_provenance
        or digests.get("protocol") != canonical_protocol_digest(protocol)
        or digests.get("source") != canonical_source_digest(protocol)
    ):
        raise ValueError("exact-CV fit probe does not match current roots")


def _build_probe_payload(
    *,
    protocol: Protocol,
    pool: CrossfitPoolManifest,
    git_commit: str,
    base_manifest_path: Path,
    pool_manifest_path: Path,
    processed_root: Path,
    embedding_root: Path,
    task_index: int,
    non_official_bypass: bool,
) -> dict[str, object]:
    assay_id, seed = _resolve_task_index(pool, protocol, task_index)
    loaded = _load_probe_fit_inputs(
        protocol=protocol,
        pool=pool,
        processed_root=processed_root,
        embedding_root=embedding_root,
        assay_id=assay_id,
        seed=seed,
        non_official_bypass=non_official_bypass,
    )
    fit = fit_exact_cv_task(loaded.inputs, protocol)
    fit_digest = canonical_exact_cv_fit_digest(fit)
    minimum_denominator = min(
        min(step.sherman_morrison_denominators) for step in fit.greedy.steps
    )
    payload: dict[str, object] = {
        "card": {"id": CARD_ID, "sha256": CARD_SHA},
        "diagnostics": {
            "coefficient_absolute_drift_max": max(
                fold.coefficient_absolute_drift for fold in fit.folds
            ),
            "coefficient_relative_drift_max": max(
                fold.coefficient_relative_drift for fold in fit.folds
            ),
            "direct_normal_equation_residual_max": max(
                fold.direct_normal_equation_residual for fold in fit.folds
            ),
            "validation_prediction_absolute_drift_max": max(
                fold.validation_prediction_max_absolute_drift for fold in fit.folds
            ),
        },
        "digests": {
            "base_fit": fit.base_fit_digest,
            "fit": fit_digest,
            "protocol": canonical_protocol_digest(protocol),
            "source": canonical_source_digest(protocol),
        },
        "dimensions": {
            "embedding_width": int(loaded.inputs.x_l.shape[1]),
            "n_labeled": int(loaded.inputs.x_l.shape[0]),
            "n_test": int(loaded.inputs.x_test.shape[0]),
            "n_unlabeled": int(loaded.inputs.x_u.shape[0]),
        },
        "execution": {
            "bypass": (
                "non_official_test_or_development"
                if non_official_bypass
                else None
            ),
            "numerical_policy": fit.base_fit.numerical_policy,
            "numerical_runtime": fit.base_fit.numerical_runtime,
            "official": not non_official_bypass,
        },
        "greedy": {
            "final_cv_mse": fit.greedy.steps[-1].mean_mse_after,
            "minimum_sherman_morrison_denominator": minimum_denominator,
            "q": len(fit.greedy.ordered_indices),
            "reanchor_steps": list(fit.greedy.reanchor_steps),
        },
        "hidden_outcomes_loaded": False,
        "kind": "exact_cv_real_shape_fit_probe",
        "provenance": {
            "base_manifest_sha256": sha256_file(base_manifest_path),
            "embedding_metadata_sha256": sha256_file(loaded.metadata_path),
            "embedding_npy_sha256": sha256_file(loaded.npy_path),
            "git_commit": git_commit,
            "identity_teacher_projection_digest": (
                loaded.identity_teacher_projection_digest
            ),
            "labeled_projection_digest": loaded.labeled_projection_digest,
            "pool_manifest_sha256": sha256_file(pool_manifest_path),
        },
        "schema_version": _SCHEMA_VERSION,
        "task": {"assay_id": assay_id, "index": task_index, "seed": seed},
    }
    normalized = cast(dict[str, object], _jsonable(payload))
    _validate_probe_payload(normalized)
    return normalized


@app.command("probe-fit")
def probe_fit(
    context: typer.Context,
    base_manifest_path: Annotated[Path, typer.Option("--base-manifest")],
    pool_manifest_path: Annotated[Path, typer.Option("--pool-manifest")],
    processed_root: Annotated[Path, typer.Option("--processed-root")],
    embedding_root: Annotated[Path, typer.Option("--embedding-root")],
    task_index: Annotated[int, typer.Option("--task-index", min=0)],
    output: Annotated[Path, typer.Option("--output")],
    non_official_bypass: Annotated[
        bool,
        typer.Option("--non-official-bypass"),
    ] = False,
) -> None:
    """Run one labeled-only real-shape fit and reconstructive digest probe."""
    options = _options(context)

    def action() -> dict[str, object]:
        protocol, _, pool, commit = _load_context(
            config=options.config,
            base_manifest_path=base_manifest_path,
            pool_manifest_path=pool_manifest_path,
            non_official_bypass=non_official_bypass,
        )
        assay_id, seed = _resolve_task_index(pool, protocol, task_index)
        if options.dry_run:
            return {
                "plan": {
                    "assay_id": assay_id,
                    "hidden_outcomes_loaded": False,
                    "seed": seed,
                }
            }
        payload = _build_probe_payload(
            protocol=protocol,
            pool=pool,
            git_commit=commit,
            base_manifest_path=base_manifest_path,
            pool_manifest_path=pool_manifest_path,
            processed_root=processed_root,
            embedding_root=embedding_root,
            task_index=task_index,
            non_official_bypass=non_official_bypass,
        )
        created = _write_json_once(output, payload)
        return {
            "artifact": str(output),
            "artifact_sha256": sha256_file(output),
            "created": created,
            "hidden_outcomes_loaded": False,
        }

    _run_command("exact-cv-probe-fit", dry_run=options.dry_run, action=action)


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

    _run_command("exact-cv-run-task", dry_run=options.dry_run, action=action)


def _validate_receipt_payload(payload: dict[str, object]) -> None:
    required = {
        "card",
        "digests",
        "execution",
        "kind",
        "provenance",
        "schema_version",
        "task",
        "task_artifact_sha256",
    }
    task = payload.get("task")
    if (
        set(payload) != required
        or payload.get("schema_version") != _SCHEMA_VERSION
        or payload.get("kind") != "exact_cv_verification_receipt"
        or payload.get("card") != {"id": CARD_ID, "sha256": CARD_SHA}
        or not isinstance(task, dict)
        or set(task) != {"assay_id", "index", "seed"}
        or not isinstance(task.get("assay_id"), str)
        or type(task.get("index")) is not int
        or type(task.get("seed")) is not int
        or not _is_sha256(payload.get("task_artifact_sha256"))
    ):
        raise ValueError("exact-CV verification receipt schema is invalid")
    digests = payload.get("digests")
    provenance = payload.get("provenance")
    if (
        not isinstance(digests, dict)
        or any(not _is_sha256(value) for value in digests.values())
        or not isinstance(provenance, dict)
        or not isinstance(provenance.get("git_commit"), str)
    ):
        raise ValueError("exact-CV verification receipt provenance is invalid")


def _receipt_from_verified_task(
    task_payload: dict[str, object],
    *,
    task_index: int,
    task_artifact_sha256: str,
    git_commit: str,
) -> dict[str, object]:
    _validate_task_payload(task_payload)
    task = cast(dict[str, object], task_payload["task"])
    receipt: dict[str, object] = {
        "card": {"id": CARD_ID, "sha256": CARD_SHA},
        "digests": cast(dict[str, object], task_payload["digests"]),
        "execution": cast(dict[str, object], task_payload["execution"]),
        "kind": "exact_cv_verification_receipt",
        "provenance": {
            **cast(dict[str, object], task_payload["provenance"]),
            "git_commit": git_commit,
        },
        "schema_version": _SCHEMA_VERSION,
        "task": {
            "assay_id": task["assay_id"],
            "index": task_index,
            "seed": task["seed"],
        },
        "task_artifact_sha256": task_artifact_sha256,
    }
    normalized = cast(dict[str, object], _jsonable(receipt))
    _validate_receipt_payload(normalized)
    return normalized


def _build_receipt_payload(
    *,
    protocol: Protocol,
    pool: CrossfitPoolManifest,
    git_commit: str,
    base_manifest_path: Path,
    pool_manifest_path: Path,
    processed_root: Path,
    embedding_root: Path,
    results_root: Path,
    task_index: int,
    non_official_bypass: bool,
) -> dict[str, object]:
    assay_id, seed = _resolve_task_index(pool, protocol, task_index)
    task_path = _task_path(results_root, assay_id, seed)
    if not task_path.is_file():
        raise ValueError(f"missing exact-CV task artifact: {task_path}")
    actual = _load_unique_json(task_path)
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
    _require_exact_payload(actual, expected, artifact_kind="exact-CV task")
    return _receipt_from_verified_task(
        actual,
        task_index=task_index,
        task_artifact_sha256=sha256_file(task_path),
        git_commit=git_commit,
    )


@app.command("verify-task")
def verify_task(
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
    """Exact-rebuild one task and write an immutable verification receipt."""
    options = _options(context)

    def action() -> dict[str, object]:
        protocol, _, pool, commit = _load_context(
            config=options.config,
            base_manifest_path=base_manifest_path,
            pool_manifest_path=pool_manifest_path,
            non_official_bypass=non_official_bypass,
        )
        assay_id, seed = _resolve_task_index(pool, protocol, task_index)
        destination = _receipt_path(results_root, assay_id, seed)
        if options.dry_run:
            return {"plan": {"assay_id": assay_id, "seed": seed}}
        receipt = _build_receipt_payload(
            protocol=protocol,
            pool=pool,
            git_commit=commit,
            base_manifest_path=base_manifest_path,
            pool_manifest_path=pool_manifest_path,
            processed_root=processed_root,
            embedding_root=embedding_root,
            results_root=results_root,
            task_index=task_index,
            non_official_bypass=non_official_bypass,
        )
        created = _write_json_once(destination, receipt)
        return {
            "artifact": str(destination),
            "artifact_sha256": sha256_file(destination),
            "assay_id": assay_id,
            "created": created,
            "seed": seed,
        }

    _run_command("exact-cv-verify-task", dry_run=options.dry_run, action=action)


def _load_verified_screen(
    *,
    protocol: Protocol,
    pool: CrossfitPoolManifest,
    git_commit: str,
    base_manifest_path: Path,
    pool_manifest_path: Path,
    results_root: Path,
    non_official_bypass: bool,
) -> tuple[
    list[tuple[dict[str, object], dict[str, object]]],
    list[dict[str, object]],
]:
    pairs: list[tuple[dict[str, object], dict[str, object]]] = []
    manifest: list[dict[str, object]] = []
    expected_base_sha = sha256_file(base_manifest_path)
    expected_pool_sha = sha256_file(pool_manifest_path)
    expected_protocol = canonical_protocol_digest(protocol)
    expected_source = canonical_source_digest(protocol)
    for index, (assay_id, seed) in enumerate(_exact_grid(pool, protocol)):
        task_path = _task_path(results_root, assay_id, seed)
        receipt_path = _receipt_path(results_root, assay_id, seed)
        if not task_path.is_file() or not receipt_path.is_file():
            raise ValueError("exact-CV task/receipt grid is incomplete")
        task = _load_unique_json(task_path)
        receipt = _load_unique_json(receipt_path)
        _validate_task_payload(task)
        _validate_receipt_payload(receipt)
        identity = cast(dict[str, object], task["task"])
        if (identity.get("assay_id"), identity.get("seed")) != (assay_id, seed):
            raise ValueError("exact-CV task identity is outside its grid position")
        task_sha = sha256_file(task_path)
        expected_receipt = _receipt_from_verified_task(
            task,
            task_index=index,
            task_artifact_sha256=task_sha,
            git_commit=git_commit,
        )
        _require_exact_payload(
            receipt,
            expected_receipt,
            artifact_kind="exact-CV verification receipt",
        )
        provenance = cast(dict[str, object], task["provenance"])
        digests = cast(dict[str, object], task["digests"])
        execution = cast(dict[str, object], task["execution"])
        if (
            provenance.get("git_commit") != git_commit
            or provenance.get("base_manifest_sha256") != expected_base_sha
            or provenance.get("pool_manifest_sha256") != expected_pool_sha
            or digests.get("protocol") != expected_protocol
            or digests.get("source") != expected_source
            or execution.get("official") is not (not non_official_bypass)
            or execution.get("numerical_policy") != NUMERICAL_POLICY
        ):
            raise ValueError("exact-CV task provenance does not match current roots")
        pairs.append((task, receipt))
        manifest.append(
            {
                "assay_id": assay_id,
                "receipt_sha256": sha256_file(receipt_path),
                "seed": seed,
                "task_sha256": task_sha,
            }
        )
    return pairs, manifest


def _metric_gate(endpoint: pd.DataFrame, *, metric: str) -> dict[str, object]:
    pivot = endpoint.pivot(
        index=["assay_id", "seed"],
        columns="method",
        values=metric,
    )
    if set(pivot.columns) != {"random", _METHOD_NAME} or len(pivot) != 40:
        raise ValueError("exact-CV endpoint table is incomplete")
    if metric == "mse":
        gains = pivot["random"] - pivot[_METHOD_NAME]
        gain_name = "random_mse_minus_exact_cv_mse"
    elif metric == "spearman":
        gains = pivot[_METHOD_NAME] - pivot["random"]
        gain_name = "exact_cv_spearman_minus_random_spearman"
    else:
        raise ValueError("exact-CV promotion metric is invalid")
    assay_gains = gains.groupby(level="assay_id").mean()
    if len(assay_gains) != 8:
        raise ValueError("exact-CV promotion requires exactly eight assays")
    macro = float(assay_gains.mean())
    task_wins = int(np.count_nonzero(gains.to_numpy() > 0.0))
    assay_wins = int(np.count_nonzero(assay_gains.to_numpy() > 0.0))
    return {
        "assay_macro_mean_gain": macro,
        "assay_standard_error": float(
            assay_gains.std(ddof=1) / np.sqrt(len(assay_gains))
        ),
        "assay_total": 8,
        "assay_win_threshold": 5,
        "assay_wins": assay_wins,
        "gain": gain_name,
        "passed": bool(macro > 0.0 and task_wins >= 25 and assay_wins >= 5),
        "task_total": 40,
        "task_win_threshold": 25,
        "task_wins": task_wins,
    }


def _validate_aggregate_payload(payload: dict[str, object]) -> None:
    if (
        payload.get("schema_version") != _SCHEMA_VERSION
        or payload.get("kind") != "exact_cv_aggregate_result"
        or payload.get("card") != {"id": CARD_ID, "sha256": CARD_SHA}
    ):
        raise ValueError("exact-CV aggregate schema is invalid")
    analysis = payload.get("analysis")
    promotion = payload.get("promotion")
    provenance = payload.get("provenance")
    execution = payload.get("execution")
    if (
        not isinstance(execution, dict)
        or execution.get("numerical_policy") != NUMERICAL_POLICY
        or not isinstance(execution.get("numerical_runtime"), dict)
        or type(execution.get("official")) is not bool
        or execution.get("bypass")
        != (
            None
            if execution.get("official") is True
            else "non_official_test_or_development"
        )
    ):
        raise ValueError("exact-CV aggregate contract is invalid")
    expected_trust_scope = (
        "trusted_cluster_execution_clean_git_and_filesystem"
        if execution["official"] is True
        else "nonofficial_bypass_filesystem_only"
    )
    if (
        analysis
        != {
            "confirmatory_claim": False,
            "inference_unit": "assay",
            "scope": "exploratory_exact_cv_screen",
            "verification_model": (
                "independent_exact_rebuild_receipts_noncryptographic"
            ),
            "verification_trust_scope": expected_trust_scope,
        }
        or not isinstance(promotion, dict)
        or not isinstance(promotion.get("mse"), dict)
        or not isinstance(promotion.get("spearman"), dict)
        or type(promotion.get("both_passed")) is not bool
        or not isinstance(provenance, dict)
        or provenance.get("task_count") != 45
        or provenance.get("receipt_count") != 45
    ):
        raise ValueError("exact-CV aggregate contract is invalid")


def _build_aggregate_payload(
    *,
    protocol: Protocol,
    pool: CrossfitPoolManifest,
    git_commit: str,
    base_manifest_path: Path,
    pool_manifest_path: Path,
    results_root: Path,
    non_official_bypass: bool,
) -> dict[str, object]:
    pairs, artifact_manifest = _load_verified_screen(
        protocol=protocol,
        pool=pool,
        git_commit=git_commit,
        base_manifest_path=base_manifest_path,
        pool_manifest_path=pool_manifest_path,
        results_root=results_root,
        non_official_bypass=non_official_bypass,
    )
    development_id = pool.screen_ids[0]
    primary_ids = pool.screen_ids[1:]
    endpoint_rows: list[dict[str, object]] = []
    prefix_rows: list[dict[str, object]] = []
    development_rows: list[dict[str, object]] = []
    runtimes: set[bytes] = set()
    runtime: object = None
    for task, _ in pairs:
        identity = cast(dict[str, object], task["task"])
        assay_id = cast(str, identity["assay_id"])
        seed = cast(int, identity["seed"])
        execution = cast(dict[str, object], task["execution"])
        runtime = execution["numerical_runtime"]
        runtimes.add(_canonical_json_bytes(runtime))
        random_row = next(
            row
            for row in cast(list[dict[str, object]], task["reference_methods"])
            if row["name"] == "random"
        )
        endpoint = cast(list[dict[str, object]], task["prefixes"])[-1]
        rows = (
            {
                "assay_id": assay_id,
                "method": "random",
                "mse": random_row["mse"],
                "ndcg_10pct": random_row["ndcg_10pct"],
                "seed": seed,
                "spearman": random_row["spearman"],
            },
            {
                "assay_id": assay_id,
                "method": _METHOD_NAME,
                "mse": endpoint["mse"],
                "ndcg_10pct": endpoint["ndcg_10pct"],
                "seed": seed,
                "spearman": endpoint["spearman"],
            },
        )
        if assay_id == development_id:
            development_rows.extend(rows)
        else:
            endpoint_rows.extend(rows)
            for prefix in cast(list[dict[str, object]], task["prefixes"]):
                prefix_rows.append(
                    {
                        "assay_id": assay_id,
                        "fold_cv_mse": prefix["fold_cv_mse"],
                        "marginal_validation_mse_reduction": prefix[
                            "marginal_validation_mse_reduction"
                        ],
                        "mse": prefix["mse"],
                        "ndcg_10pct": prefix["ndcg_10pct"],
                        "overlap_full": prefix["overlap_full"],
                        "overlap_no_hessian": prefix["overlap_no_hessian"],
                        "overlap_random": prefix["overlap_random"],
                        "overlap_top_teacher": prefix["overlap_top_teacher"],
                        "q": prefix["q"],
                        "seed": seed,
                        "selected_pseudo_label_mae": prefix[
                            "selected_pseudo_label_mae"
                        ],
                        "spearman": prefix["spearman"],
                    }
                )
    if len(runtimes) != 1:
        raise ValueError("exact-CV tasks mix numerical runtimes")
    endpoint = pd.DataFrame(endpoint_rows).sort_values(
        ["assay_id", "seed", "method"],
        kind="stable",
    )
    prefix_table = pd.DataFrame(prefix_rows).sort_values(
        ["q", "assay_id", "seed"],
        kind="stable",
    )
    if len(endpoint) != 80 or len(prefix_table) != 200:
        raise ValueError("exact-CV primary grid is incomplete")
    mse_gate = _metric_gate(endpoint, metric="mse")
    spearman_gate = _metric_gate(endpoint, metric="spearman")
    prefix_descriptive = [
        {
            "mean_fold_cv_mse": float(group["fold_cv_mse"].mean()),
            "mean_marginal_validation_mse_reduction": float(
                group["marginal_validation_mse_reduction"].mean()
            ),
            "mean_mse": float(group["mse"].mean()),
            "mean_ndcg_10pct": float(group["ndcg_10pct"].mean()),
            "mean_overlap_full": float(group["overlap_full"].mean()),
            "mean_overlap_no_hessian": float(
                group["overlap_no_hessian"].mean()
            ),
            "mean_overlap_random": float(group["overlap_random"].mean()),
            "mean_overlap_top_teacher": float(
                group["overlap_top_teacher"].mean()
            ),
            "mean_selected_pseudo_label_mae": float(
                group["selected_pseudo_label_mae"].mean()
            ),
            "mean_spearman": float(group["spearman"].mean()),
            "q": int(q),
            "task_count": len(group),
        }
        for q, group in prefix_table.groupby("q", sort=True)
    ]
    grid = _exact_grid(pool, protocol)
    primary_grid = tuple(
        (assay_id, seed) for assay_id in primary_ids for seed in protocol.seeds
    )
    payload: dict[str, object] = {
        "analysis": {
            "confirmatory_claim": False,
            "inference_unit": "assay",
            "scope": "exploratory_exact_cv_screen",
            "verification_model": (
                "independent_exact_rebuild_receipts_noncryptographic"
            ),
            "verification_trust_scope": (
                "nonofficial_bypass_filesystem_only"
                if non_official_bypass
                else "trusted_cluster_execution_clean_git_and_filesystem"
            ),
        },
        "card": {"id": CARD_ID, "sha256": CARD_SHA},
        "development_endpoint_results": development_rows,
        "execution": {
            "bypass": (
                "non_official_test_or_development"
                if non_official_bypass
                else None
            ),
            "numerical_policy": NUMERICAL_POLICY,
            "numerical_runtime": runtime,
            "official": not non_official_bypass,
        },
        "grid": {
            "development_assay_id": development_id,
            "prefix_counts": list(PREFIX_COUNTS),
            "primary_assay_ids": list(primary_ids),
            "primary_tasks": [list(item) for item in primary_grid],
            "screen_assay_ids": list(pool.screen_ids),
            "screen_tasks": [list(item) for item in grid],
            "seeds": list(protocol.seeds),
        },
        "kind": "exact_cv_aggregate_result",
        "prefix_descriptive": prefix_descriptive,
        "primary_endpoint_results": endpoint.to_dict(orient="records"),
        "promotion": {
            "both_passed": bool(mse_gate["passed"] and spearman_gate["passed"]),
            "mse": mse_gate,
            "spearman": spearman_gate,
        },
        "provenance": {
            "artifact_manifest": artifact_manifest,
            "base_manifest_sha256": sha256_file(base_manifest_path),
            "git_commit": git_commit,
            "pool_manifest_sha256": sha256_file(pool_manifest_path),
            "protocol_digest": canonical_protocol_digest(protocol),
            "receipt_count": len(artifact_manifest),
            "task_count": len(artifact_manifest),
        },
        "schema_version": _SCHEMA_VERSION,
    }
    normalized = cast(dict[str, object], _jsonable(payload))
    _validate_aggregate_payload(normalized)
    return normalized


@app.command("aggregate")
def aggregate(
    context: typer.Context,
    base_manifest_path: Annotated[Path, typer.Option("--base-manifest")],
    pool_manifest_path: Annotated[Path, typer.Option("--pool-manifest")],
    results_root: Annotated[Path, typer.Option("--results-root")],
    output: Annotated[Path, typer.Option("--output")],
    non_official_bypass: Annotated[
        bool,
        typer.Option("--non-official-bypass"),
    ] = False,
) -> None:
    """Summarize only the 45 receipt-verified screen tasks."""
    options = _options(context)

    def action() -> dict[str, object]:
        protocol, _, pool, commit = _load_context(
            config=options.config,
            base_manifest_path=base_manifest_path,
            pool_manifest_path=pool_manifest_path,
            non_official_bypass=non_official_bypass,
        )
        if options.dry_run:
            return {"plan": {"output": str(output), "receipt_count": 45}}
        payload = _build_aggregate_payload(
            protocol=protocol,
            pool=pool,
            git_commit=commit,
            base_manifest_path=base_manifest_path,
            pool_manifest_path=pool_manifest_path,
            results_root=results_root,
            non_official_bypass=non_official_bypass,
        )
        created = _write_json_once(output, payload)
        return {
            "artifact": str(output),
            "artifact_sha256": sha256_file(output),
            "created": created,
            "receipt_count": 45,
        }

    _run_command("exact-cv-aggregate", dry_run=options.dry_run, action=action)


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
    probe_artifact: Annotated[
        Path | None,
        typer.Option("--probe-artifact"),
    ] = None,
    results_root: Annotated[Path | None, typer.Option("--results-root")] = None,
    aggregate_artifact: Annotated[
        Path | None,
        typer.Option("--aggregate-artifact"),
    ] = None,
    non_official_bypass: Annotated[
        bool,
        typer.Option("--non-official-bypass"),
    ] = False,
) -> None:
    """Verify pinned inputs and optionally all receipts and aggregate bytes."""
    options = _options(context)

    def action() -> dict[str, object]:
        protocol, _, pool, commit = _load_context(
            config=options.config,
            base_manifest_path=base_manifest_path,
            pool_manifest_path=pool_manifest_path,
            non_official_bypass=non_official_bypass,
        )
        if probe_artifact is None:
            if not non_official_bypass:
                raise ValueError(
                    "official exact-CV verification requires --probe-artifact"
                )
        else:
            probe = _load_unique_json(probe_artifact)
            _validate_probe_against_roots(
                probe,
                protocol=protocol,
                pool=pool,
                git_commit=commit,
                base_manifest_path=base_manifest_path,
                pool_manifest_path=pool_manifest_path,
                processed_root=processed_root,
                embedding_root=embedding_root,
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
        verified = ["base_manifest", "pool", "screen_grid", "inputs"]
        if probe_artifact is not None:
            verified.append("probe_fit")
        if results_root is not None:
            expected = _build_aggregate_payload(
                protocol=protocol,
                pool=pool,
                git_commit=commit,
                base_manifest_path=base_manifest_path,
                pool_manifest_path=pool_manifest_path,
                results_root=results_root,
                non_official_bypass=non_official_bypass,
            )
            verified.extend(["tasks", "receipts"])
            if aggregate_artifact is not None:
                actual = _load_unique_json(aggregate_artifact)
                _validate_aggregate_payload(actual)
                _require_exact_payload(
                    actual,
                    expected,
                    artifact_kind="exact-CV aggregate",
                )
                verified.append("aggregate")
        elif aggregate_artifact is not None:
            raise ValueError("aggregate verification requires --results-root")
        return {"verified": verified}

    _run_command("exact-cv-verify", dry_run=options.dry_run, action=action)


if __name__ == "__main__":
    app()
