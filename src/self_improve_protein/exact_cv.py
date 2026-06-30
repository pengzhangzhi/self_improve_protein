"""Hidden-label-safe exact cross-validated greedy pseudo-sample selection."""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final, TypeAlias

import numpy as np
from numpy.typing import NDArray

from self_improve_protein.config import Protocol
from self_improve_protein.experiment import (
    EvaluationLabels,
    EvaluationResult,
    FitInputs,
    FitResult,
    _array_identity,
    _canonical_payload_digest,
    _diagnostic_values_close,
    _float_array,
    _locked_blas_scope,
    _scaled_algebra_close,
    canonical_fit_digest,
    evaluate_task,
    fit_task,
)
from self_improve_protein.metrics import (
    ndcg_at_10_percent,
    spearman_correlation,
    standardized_mse,
)
from self_improve_protein.ridge import (
    FeatureTransform,
    LabelTransform,
    TeacherCalibration,
    fit_feature_transform,
    fit_label_transform,
    fit_teacher_calibration,
    fit_weighted_ridge,
)
from self_improve_protein.selection import balanced_fold_assignment

CARD_ID: Final = "exact_cv_greedy_v1"
CARD_SHA: Final = "90f35965bd9a36320bc3d5553deb8ea241961cf35f1a887b714cba417e6a4c3a"
FOLD_COUNT: Final = 4
FOLD_PURPOSE: Final = "exact_cv_folds_v1"
PREFIX_COUNTS: Final[tuple[int, ...]] = (24, 48, 72, 96, 192)
REANCHOR_STEPS: Final[tuple[int, ...]] = (0, 1, 24, 48, 72, 96, 192)
PARITY_EPS_MULTIPLIER: Final = 8192.0
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")

FloatArray: TypeAlias = NDArray[np.float64]


def _positive_finite(value: object, *, name: str) -> float:
    if isinstance(value, (bool, str, bytes)):
        raise ValueError(f"{name} must be finite and positive")
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be finite and positive") from error
    if not np.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return number


def _parity_close(first: object, second: object, *, width: int) -> bool:
    """Compare cached/direct algebra under the declared width-scaled tolerance."""
    try:
        left = np.asarray(first, dtype=np.float64)
        right = np.asarray(second, dtype=np.float64)
    except (TypeError, ValueError):
        return False
    if left.shape != right.shape or not np.all(np.isfinite(left)):
        return False
    if not np.all(np.isfinite(right)):
        return False
    scale = np.maximum(1.0, np.maximum(np.abs(left), np.abs(right)))
    tolerance = (
        PARITY_EPS_MULTIPLIER
        * np.finfo(np.float64).eps
        * max(1, width)
        * scale
    )
    return bool(np.all(np.abs(left - right) <= tolerance))


@dataclass(frozen=True)
class ExactCVFoldInputs:
    """One already transformed fold used by the exact greedy solver."""

    fold_id: int
    x_train: FloatArray
    y_train: FloatArray
    x_validation: FloatArray
    y_validation: FloatArray
    x_u: FloatArray
    pseudo_labels_u: FloatArray

    def __post_init__(self) -> None:
        if type(self.fold_id) is not int or self.fold_id < 0:
            raise ValueError("fold_id must be a non-negative integer")
        x_train = _float_array(self.x_train, name="x_train", ndim=2)
        y_train = _float_array(self.y_train, name="y_train", ndim=1)
        x_validation = _float_array(
            self.x_validation,
            name="x_validation",
            ndim=2,
        )
        y_validation = _float_array(
            self.y_validation,
            name="y_validation",
            ndim=1,
        )
        x_u = _float_array(self.x_u, name="x_u", ndim=2)
        pseudo_labels_u = _float_array(
            self.pseudo_labels_u,
            name="pseudo_labels_u",
            ndim=1,
        )
        if x_train.shape[0] != y_train.size:
            raise ValueError("x_train and y_train cardinalities must match")
        if x_validation.shape[0] != y_validation.size:
            raise ValueError("validation cardinalities must match")
        if x_u.shape[0] != pseudo_labels_u.size:
            raise ValueError("candidate cardinalities must match")
        if len({x_train.shape[1], x_validation.shape[1], x_u.shape[1]}) != 1:
            raise ValueError("all fold feature widths must match")
        for name, value in (
            ("x_train", x_train),
            ("y_train", y_train),
            ("x_validation", x_validation),
            ("y_validation", y_validation),
            ("x_u", x_u),
            ("pseudo_labels_u", pseudo_labels_u),
        ):
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class ExactCVStep:
    """One common exact-CV greedy decision and its four-fold utilities."""

    step: int
    selected_index: int
    selected_hash: str
    fold_mse_before: tuple[float, ...]
    fold_mse_after: tuple[float, ...]
    fold_mse_reduction: tuple[float, ...]
    mean_mse_before: float
    mean_mse_after: float
    mean_mse_reduction: float
    runner_up_mean_mse_gap: float
    sherman_morrison_denominators: tuple[float, ...]

    def __post_init__(self) -> None:
        if type(self.step) is not int or self.step <= 0:
            raise ValueError("step must be a positive integer")
        if type(self.selected_index) is not int or self.selected_index < 0:
            raise ValueError("selected_index must be a non-negative integer")
        if not isinstance(self.selected_hash, str) or not self.selected_hash:
            raise ValueError("selected_hash must be non-empty")
        vectors = (
            self.fold_mse_before,
            self.fold_mse_after,
            self.fold_mse_reduction,
            self.sherman_morrison_denominators,
        )
        if any(len(values) != FOLD_COUNT for values in vectors):
            raise ValueError("step diagnostics must contain exactly four folds")
        scalars = (
            *self.fold_mse_before,
            *self.fold_mse_after,
            *self.fold_mse_reduction,
            self.mean_mse_before,
            self.mean_mse_after,
            self.mean_mse_reduction,
            self.runner_up_mean_mse_gap,
            *self.sherman_morrison_denominators,
        )
        if any(not np.isfinite(value) for value in scalars):
            raise ValueError("step diagnostics must be finite")
        if any(value < 0.0 for value in self.fold_mse_before + self.fold_mse_after):
            raise ValueError("validation MSE values must be non-negative")
        if self.runner_up_mean_mse_gap < 0.0:
            raise ValueError("runner-up MSE gap must be non-negative")
        tolerance = 4096.0 * np.finfo(np.float64).eps
        if any(value < 1.0 - tolerance for value in self.sherman_morrison_denominators):
            raise ValueError("Sherman-Morrison denominators must be at least one")


@dataclass(frozen=True)
class ExactCVGreedyResult:
    """Complete common ordering plus direct-solve parity state."""

    ordered_indices: tuple[int, ...]
    ordered_hashes: tuple[str, ...]
    steps: tuple[ExactCVStep, ...]
    regularizer_mass: float
    reanchor_steps: tuple[int, ...]
    final_recursive_coefficients: tuple[FloatArray, ...]
    final_direct_coefficients: tuple[FloatArray, ...]

    def __post_init__(self) -> None:
        indices = tuple(self.ordered_indices)
        hashes = tuple(self.ordered_hashes)
        steps = tuple(self.steps)
        recursive = tuple(
            _float_array(value, name="final_recursive_coefficients", ndim=1)
            for value in self.final_recursive_coefficients
        )
        direct = tuple(
            _float_array(value, name="final_direct_coefficients", ndim=1)
            for value in self.final_direct_coefficients
        )
        if len(indices) != len(hashes) or len(indices) != len(steps):
            raise ValueError("greedy ordering lengths must match")
        if len(set(indices)) != len(indices):
            raise ValueError("greedy ordering indices must be unique")
        if tuple(step.selected_index for step in steps) != indices:
            raise ValueError("greedy steps must match ordered_indices")
        if tuple(step.selected_hash for step in steps) != hashes:
            raise ValueError("greedy steps must match ordered_hashes")
        if len(recursive) != FOLD_COUNT or len(direct) != FOLD_COUNT:
            raise ValueError("final coefficients must contain four folds")
        object.__setattr__(self, "ordered_indices", indices)
        object.__setattr__(self, "ordered_hashes", hashes)
        object.__setattr__(self, "steps", steps)
        object.__setattr__(
            self,
            "regularizer_mass",
            _positive_finite(self.regularizer_mass, name="regularizer_mass"),
        )
        object.__setattr__(self, "reanchor_steps", tuple(self.reanchor_steps))
        object.__setattr__(self, "final_recursive_coefficients", recursive)
        object.__setattr__(self, "final_direct_coefficients", direct)


@dataclass
class _RecursiveFoldState:
    inverse: FloatArray
    coefficients: FloatArray
    candidate_inverse_rows: FloatArray
    candidate_validation_directions: FloatArray
    candidate_predictions: FloatArray
    validation_predictions: FloatArray
    candidate_leverages: FloatArray


def _direct_state(
    fold: ExactCVFoldInputs,
    selected: tuple[int, ...],
    *,
    regularizer_mass: float,
    pseudo_weight: float,
    ridge_lambda: float,
) -> tuple[FloatArray, FloatArray]:
    width = fold.x_train.shape[1]
    normal = (
        fold.x_train.T @ fold.x_train
        + regularizer_mass * ridge_lambda * np.eye(width, dtype=np.float64)
    )
    right_hand_side = fold.x_train.T @ fold.y_train
    if selected:
        indices = np.asarray(selected, dtype=np.int64)
        x_selected = fold.x_u[indices]
        normal = normal + pseudo_weight * (x_selected.T @ x_selected)
        right_hand_side = right_hand_side + pseudo_weight * (
            x_selected.T @ fold.pseudo_labels_u[indices]
        )
    try:
        inverse = np.linalg.solve(normal, np.eye(width, dtype=np.float64))
        coefficients = np.linalg.solve(normal, right_hand_side)
    except np.linalg.LinAlgError as error:
        raise np.linalg.LinAlgError(
            "exact-CV fold normal matrix is singular"
        ) from error
    if not np.all(np.isfinite(inverse)) or not np.all(np.isfinite(coefficients)):
        raise ValueError("exact-CV direct fold state must be finite")
    inverse_float: FloatArray = np.asarray(
        (inverse + inverse.T) / 2.0,
        dtype=np.float64,
    )
    return inverse_float, np.asarray(
        coefficients,
        dtype=np.float64,
    )


def _recursive_state(
    fold: ExactCVFoldInputs,
    inverse: FloatArray,
    coefficients: FloatArray,
) -> _RecursiveFoldState:
    candidate_inverse_rows = np.asarray(fold.x_u @ inverse, dtype=np.float64)
    directions = np.asarray(
        candidate_inverse_rows @ fold.x_validation.T,
        dtype=np.float64,
    )
    candidate_predictions = np.asarray(fold.x_u @ coefficients, dtype=np.float64)
    validation_predictions = np.asarray(
        fold.x_validation @ coefficients,
        dtype=np.float64,
    )
    leverages = np.asarray(
        np.einsum("ij,ij->i", candidate_inverse_rows, fold.x_u),
        dtype=np.float64,
    )
    arrays = (
        candidate_inverse_rows,
        directions,
        candidate_predictions,
        validation_predictions,
        leverages,
    )
    if any(not np.all(np.isfinite(array)) for array in arrays):
        raise ValueError("exact-CV recursive fold state must be finite")
    return _RecursiveFoldState(
        inverse=np.asarray(inverse, dtype=np.float64),
        coefficients=np.asarray(coefficients, dtype=np.float64),
        candidate_inverse_rows=candidate_inverse_rows,
        candidate_validation_directions=directions,
        candidate_predictions=candidate_predictions,
        validation_predictions=validation_predictions,
        candidate_leverages=leverages,
    )


def _validated_greedy_inputs(
    folds: Sequence[ExactCVFoldInputs],
    stable_hashes: Sequence[str],
    *,
    q: int,
) -> tuple[tuple[ExactCVFoldInputs, ...], tuple[str, ...]]:
    fold_tuple = tuple(folds)
    if len(fold_tuple) != FOLD_COUNT or any(
        not isinstance(fold, ExactCVFoldInputs) for fold in fold_tuple
    ):
        raise ValueError("folds must contain exactly four ExactCVFoldInputs")
    if tuple(fold.fold_id for fold in fold_tuple) != tuple(range(FOLD_COUNT)):
        raise ValueError("fold IDs must be ordered from zero through three")
    hashes = tuple(stable_hashes)
    pool_sizes = {fold.x_u.shape[0] for fold in fold_tuple}
    train_sizes = {fold.x_train.shape[0] for fold in fold_tuple}
    if len(pool_sizes) != 1 or len(train_sizes) != 1:
        raise ValueError("all folds must share candidate and training counts")
    pool_size = next(iter(pool_sizes))
    if len(hashes) != pool_size or any(
        not isinstance(value, str) or not value for value in hashes
    ):
        raise ValueError("stable_hashes must match the candidate pool")
    if len(set(hashes)) != len(hashes):
        raise ValueError("stable_hashes must be unique")
    if type(q) is not int or not 0 < q <= pool_size:
        raise ValueError("q must be a valid positive selection count")
    return fold_tuple, hashes


def greedy_exact_cv_order(
    folds: Sequence[ExactCVFoldInputs],
    stable_hashes: Sequence[str],
    *,
    q: int,
    pseudo_weight: float,
    ridge_lambda: float,
) -> ExactCVGreedyResult:
    """Build one common ordering by exact four-fold post-addition MSE greed."""
    fold_tuple, hashes = _validated_greedy_inputs(folds, stable_hashes, q=q)
    weight = _positive_finite(pseudo_weight, name="pseudo_weight")
    regularization = _positive_finite(ridge_lambda, name="ridge_lambda")
    training_count = fold_tuple[0].x_train.shape[0]
    regularizer_mass = float(training_count + weight * q)
    states: list[_RecursiveFoldState] = []
    for fold in fold_tuple:
        inverse, coefficients = _direct_state(
            fold,
            (),
            regularizer_mass=regularizer_mass,
            pseudo_weight=weight,
            ridge_lambda=regularization,
        )
        states.append(_recursive_state(fold, inverse, coefficients))

    selected: tuple[int, ...] = ()
    steps: list[ExactCVStep] = []
    active = np.ones(len(hashes), dtype=np.bool_)
    hash_array = np.asarray(hashes, dtype=np.str_)
    reanchors = tuple(
        step for step in REANCHOR_STEPS if step <= q
    ) + (() if q in REANCHOR_STEPS else (q,))
    final_recursive_before_reanchor: tuple[FloatArray, ...] | None = None
    for step_number in range(1, q + 1):
        fold_before: list[float] = []
        fold_candidate_after: list[FloatArray] = []
        fold_denominators: list[FloatArray] = []
        for fold, state in zip(fold_tuple, states, strict=True):
            residual = state.validation_predictions - fold.y_validation
            before = float(np.mean(residual * residual))
            candidate_denominators = 1.0 + weight * state.candidate_leverages
            tolerance = (
                4096.0
                * np.finfo(np.float64).eps
                * np.maximum(1.0, np.abs(candidate_denominators))
            )
            if (
                not np.all(np.isfinite(candidate_denominators))
                or np.any(candidate_denominators <= 0.0)
                or np.any(candidate_denominators < 1.0 - tolerance)
            ):
                raise ValueError(
                    "Sherman-Morrison denominators must be finite and at least one"
                )
            alpha = (
                weight
                * (fold.pseudo_labels_u - state.candidate_predictions)
                / candidate_denominators
            )
            delta = state.candidate_validation_directions * alpha[:, None]
            after = np.mean((residual[None, :] + delta) ** 2, axis=1)
            if not np.all(np.isfinite(after)):
                raise ValueError("candidate validation MSE must be finite")
            fold_before.append(before)
            fold_candidate_after.append(np.asarray(after, dtype=np.float64))
            fold_denominators.append(
                np.asarray(candidate_denominators, dtype=np.float64)
            )
        mean_after = np.mean(np.stack(fold_candidate_after, axis=0), axis=0)
        active_indices = np.flatnonzero(active)
        active_order = np.lexsort(
            (hash_array[active_indices], mean_after[active_indices])
        )
        chosen = int(active_indices[int(active_order[0])])
        runner_up_gap = 0.0
        if active_indices.size > 1:
            runner_up = int(active_indices[int(active_order[1])])
            runner_up_gap = float(mean_after[runner_up] - mean_after[chosen])
            if runner_up_gap < 0.0 or not np.isfinite(runner_up_gap):
                raise ValueError("runner-up validation MSE gap must be non-negative")
        selected += (chosen,)
        active[chosen] = False

        chosen_denominators: list[float] = []
        for fold, state, denominators in zip(
            fold_tuple,
            states,
            fold_denominators,
            strict=True,
        ):
            inverse_direction = np.asarray(
                state.inverse @ fold.x_u[chosen],
                dtype=np.float64,
            )
            candidate_cross = np.asarray(
                state.candidate_inverse_rows @ fold.x_u[chosen],
                dtype=np.float64,
            )
            validation_direction = np.asarray(
                fold.x_validation @ inverse_direction,
                dtype=np.float64,
            )
            cached_inverse_direction = state.candidate_inverse_rows[chosen]
            cached_validation_direction = (
                state.candidate_validation_directions[chosen]
            )
            cached_leverage = state.candidate_leverages[chosen]
            direct_leverage = float(fold.x_u[chosen] @ inverse_direction)
            if (
                not _parity_close(
                    cached_inverse_direction,
                    inverse_direction,
                    width=fold.x_train.shape[1],
                )
                or not _parity_close(
                    cached_validation_direction,
                    validation_direction,
                    width=fold.x_train.shape[1],
                )
                or not _parity_close(
                    cached_leverage,
                    direct_leverage,
                    width=fold.x_train.shape[1],
                )
            ):
                raise ValueError("exact-CV cached selected-candidate state drifted")
            chosen_denominator = float(denominators[chosen])
            chosen_alpha = float(
                weight
                * (
                    fold.pseudo_labels_u[chosen]
                    - state.candidate_predictions[chosen]
                )
                / chosen_denominator
            )
            state.coefficients = (
                state.coefficients + chosen_alpha * inverse_direction
            )
            state.inverse = state.inverse - (
                weight
                * np.outer(inverse_direction, inverse_direction)
                / chosen_denominator
            )
            state.candidate_inverse_rows = state.candidate_inverse_rows - (
                weight
                * np.outer(candidate_cross, inverse_direction)
                / chosen_denominator
            )
            state.candidate_validation_directions = (
                state.candidate_validation_directions
                - weight
                * np.outer(candidate_cross, validation_direction)
                / chosen_denominator
            )
            state.candidate_predictions = (
                state.candidate_predictions + chosen_alpha * candidate_cross
            )
            state.validation_predictions = (
                state.validation_predictions
                + chosen_alpha * validation_direction
            )
            state.candidate_leverages = state.candidate_leverages - (
                weight
                * candidate_cross
                * candidate_cross
                / chosen_denominator
            )
            chosen_denominators.append(chosen_denominator)

        if step_number == q:
            final_recursive_before_reanchor = tuple(
                np.array(state.coefficients, dtype=np.float64, copy=True)
                for state in states
            )
        if step_number in reanchors:
            for index, fold in enumerate(fold_tuple):
                inverse, coefficients = _direct_state(
                    fold,
                    selected,
                    regularizer_mass=regularizer_mass,
                    pseudo_weight=weight,
                    ridge_lambda=regularization,
                )
                states[index] = _recursive_state(fold, inverse, coefficients)

        fold_after = tuple(
            float(
                np.mean(
                    (state.validation_predictions - fold.y_validation) ** 2
                )
            )
            for fold, state in zip(fold_tuple, states, strict=True)
        )
        selected_post_mse = tuple(
            float(values[chosen]) for values in fold_candidate_after
        )
        if not _parity_close(
            fold_after,
            selected_post_mse,
            width=fold_tuple[0].x_train.shape[1],
        ):
            raise ValueError(
                "selected candidate post-MSE does not match updated fold state"
            )
        fold_before_tuple = tuple(fold_before)
        fold_reduction = tuple(
            before - after
            for before, after in zip(fold_before_tuple, fold_after, strict=True)
        )
        steps.append(
            ExactCVStep(
                step=step_number,
                selected_index=chosen,
                selected_hash=hashes[chosen],
                fold_mse_before=fold_before_tuple,
                fold_mse_after=fold_after,
                fold_mse_reduction=fold_reduction,
                mean_mse_before=float(np.mean(fold_before_tuple)),
                mean_mse_after=float(np.mean(fold_after)),
                mean_mse_reduction=float(np.mean(fold_reduction)),
                runner_up_mean_mse_gap=runner_up_gap,
                sherman_morrison_denominators=tuple(chosen_denominators),
            )
        )

    final_direct: list[FloatArray] = []
    for fold in fold_tuple:
        _, coefficients = _direct_state(
            fold,
            selected,
            regularizer_mass=regularizer_mass,
            pseudo_weight=weight,
            ridge_lambda=regularization,
        )
        final_direct.append(coefficients)
    if final_recursive_before_reanchor is None:
        raise RuntimeError("exact-CV greedy path did not reach q")
    return ExactCVGreedyResult(
        ordered_indices=selected,
        ordered_hashes=tuple(hashes[index] for index in selected),
        steps=tuple(steps),
        regularizer_mass=regularizer_mass,
        reanchor_steps=reanchors,
        final_recursive_coefficients=final_recursive_before_reanchor,
        final_direct_coefficients=tuple(final_direct),
    )


@dataclass(frozen=True)
class ExactCVFoldArtifact:
    """Training-only transforms and final direct-solve parity for one fold."""

    fold_id: int
    training_indices: tuple[int, ...]
    validation_indices: tuple[int, ...]
    feature_transform: FeatureTransform
    label_transform: LabelTransform
    teacher_calibration: TeacherCalibration
    regularizer_mass: float
    initial_validation_mse: float
    final_validation_mse: float
    final_recursive_coefficients: FloatArray
    final_direct_coefficients: FloatArray
    coefficient_absolute_drift: float
    coefficient_relative_drift: float
    validation_prediction_max_absolute_drift: float
    direct_normal_equation_residual: float

    def __post_init__(self) -> None:
        if type(self.fold_id) is not int or not 0 <= self.fold_id < FOLD_COUNT:
            raise ValueError("fold_id must be in [0, 4)")
        training = tuple(self.training_indices)
        validation = tuple(self.validation_indices)
        if (
            not training
            or not validation
            or len(set(training + validation)) != len(training) + len(validation)
            or any(
                type(index) is not int or index < 0
                for index in training + validation
            )
        ):
            raise ValueError("fold indices must be non-empty and disjoint")
        if not isinstance(self.feature_transform, FeatureTransform):
            raise ValueError("feature_transform must be FeatureTransform")
        if not isinstance(self.label_transform, LabelTransform):
            raise ValueError("label_transform must be LabelTransform")
        if not isinstance(self.teacher_calibration, TeacherCalibration):
            raise ValueError("teacher_calibration must be TeacherCalibration")
        recursive = _float_array(
            self.final_recursive_coefficients,
            name="final_recursive_coefficients",
            ndim=1,
        )
        direct = _float_array(
            self.final_direct_coefficients,
            name="final_direct_coefficients",
            ndim=1,
        )
        if recursive.shape != direct.shape:
            raise ValueError("final fold coefficient widths must match")
        for name in (
            "regularizer_mass",
            "initial_validation_mse",
            "final_validation_mse",
            "coefficient_absolute_drift",
            "coefficient_relative_drift",
            "validation_prediction_max_absolute_drift",
            "direct_normal_equation_residual",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
            object.__setattr__(self, name, value)
        if self.regularizer_mass <= 0.0:
            raise ValueError("regularizer_mass must be positive")
        object.__setattr__(self, "training_indices", training)
        object.__setattr__(self, "validation_indices", validation)
        object.__setattr__(self, "final_recursive_coefficients", recursive)
        object.__setattr__(self, "final_direct_coefficients", direct)


@dataclass(frozen=True)
class ExactCVPrefixArtifact:
    """Full-96 deployed-objective refit for one frozen ordering prefix."""

    q: int
    selected_indices: tuple[int, ...]
    selected_hashes: tuple[str, ...]
    selected_pseudo_labels: FloatArray
    pseudo_weight: float
    ridge_lambda: float
    training_weight_sum: float
    fold_cv_mse: float
    coefficients: FloatArray
    test_predictions: FloatArray

    def __post_init__(self) -> None:
        if type(self.q) is not int or self.q not in PREFIX_COUNTS:
            raise ValueError("q must be a frozen exact-CV prefix count")
        indices = tuple(self.selected_indices)
        hashes = tuple(self.selected_hashes)
        if (
            len(indices) != self.q
            or len(set(indices)) != self.q
            or any(type(index) is not int or index < 0 for index in indices)
        ):
            raise ValueError("selected_indices must contain q unique indices")
        if len(hashes) != self.q or any(
            not isinstance(value, str) or not value for value in hashes
        ):
            raise ValueError("selected_hashes must contain q strings")
        labels = _float_array(
            self.selected_pseudo_labels,
            name="selected_pseudo_labels",
            ndim=1,
        )
        coefficients = _float_array(self.coefficients, name="coefficients", ndim=1)
        predictions = _float_array(
            self.test_predictions,
            name="test_predictions",
            ndim=1,
        )
        if labels.size != self.q:
            raise ValueError("selected_pseudo_labels must contain q values")
        for name in (
            "pseudo_weight",
            "ridge_lambda",
            "training_weight_sum",
        ):
            value = _positive_finite(getattr(self, name), name=name)
            object.__setattr__(self, name, value)
        fold_cv_mse = float(self.fold_cv_mse)
        if not np.isfinite(fold_cv_mse) or fold_cv_mse < 0.0:
            raise ValueError("fold_cv_mse must be finite and non-negative")
        object.__setattr__(self, "fold_cv_mse", fold_cv_mse)
        object.__setattr__(self, "selected_indices", indices)
        object.__setattr__(self, "selected_hashes", hashes)
        object.__setattr__(self, "selected_pseudo_labels", labels)
        object.__setattr__(self, "coefficients", coefficients)
        object.__setattr__(self, "test_predictions", predictions)


@dataclass(frozen=True)
class ExactCVFitResult:
    """Canonical hidden-label-free exact-CV selection and prefix fits."""

    base_fit: FitResult
    base_fit_digest: str
    card_id: str
    card_sha: str
    fold_count: int
    fold_purpose: str
    prefix_counts: tuple[int, ...]
    reanchor_steps: tuple[int, ...]
    raw_x_l: FloatArray
    raw_x_u: FloatArray
    raw_x_test: FloatArray
    fold_assignment: NDArray[np.int64]
    folds: tuple[ExactCVFoldArtifact, ...]
    greedy: ExactCVGreedyResult
    prefixes: tuple[ExactCVPrefixArtifact, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.base_fit, FitResult):
            raise ValueError("base_fit must be FitResult")
        if not isinstance(self.base_fit_digest, str):
            raise ValueError("base_fit_digest must be a string")
        raw_x_l = _float_array(self.raw_x_l, name="raw_x_l", ndim=2)
        raw_x_u = _float_array(self.raw_x_u, name="raw_x_u", ndim=2)
        raw_x_test = _float_array(self.raw_x_test, name="raw_x_test", ndim=2)
        try:
            assignment_raw = np.asarray(self.fold_assignment)
        except (TypeError, ValueError) as error:
            raise ValueError("fold_assignment must be an integer vector") from error
        if assignment_raw.ndim != 1 or assignment_raw.dtype.kind not in {"i", "u"}:
            raise ValueError("fold_assignment must be an integer vector")
        assignment = np.array(assignment_raw, dtype=np.int64, copy=True)
        assignment.setflags(write=False)
        folds = tuple(self.folds)
        prefixes = tuple(self.prefixes)
        if tuple(fold.fold_id for fold in folds) != tuple(range(FOLD_COUNT)):
            raise ValueError("fold artifacts must be ordered from zero through three")
        if tuple(prefix.q for prefix in prefixes) != PREFIX_COUNTS:
            raise ValueError("prefix artifacts must match the frozen counts")
        if not isinstance(self.greedy, ExactCVGreedyResult):
            raise ValueError("greedy must be ExactCVGreedyResult")
        object.__setattr__(self, "raw_x_l", raw_x_l)
        object.__setattr__(self, "raw_x_u", raw_x_u)
        object.__setattr__(self, "raw_x_test", raw_x_test)
        object.__setattr__(self, "fold_assignment", assignment)
        object.__setattr__(self, "folds", folds)
        object.__setattr__(self, "prefixes", prefixes)

    @property
    def assay_id(self) -> str:
        return self.base_fit.assay_id

    @property
    def seed(self) -> int:
        return self.base_fit.seed

    @property
    def source_digest(self) -> str:
        return self.base_fit.source_digest

    @property
    def protocol_digest(self) -> str:
        return self.base_fit.protocol_digest

    def to_payload(self) -> dict[str, object]:
        """Return a compact canonical payload containing no hidden outcomes."""
        return {
            "assay_id": self.assay_id,
            "seed": self.seed,
            "source_digest": self.source_digest,
            "protocol_digest": self.protocol_digest,
            "base_fit_digest": self.base_fit_digest,
            "card_id": self.card_id,
            "card_sha": self.card_sha,
            "fold_count": self.fold_count,
            "fold_purpose": self.fold_purpose,
            "prefix_counts": list(self.prefix_counts),
            "reanchor_steps": list(self.reanchor_steps),
            "raw_features": {
                "x_l": _array_identity(self.raw_x_l),
                "x_u": _array_identity(self.raw_x_u),
                "x_test": _array_identity(self.raw_x_test),
            },
            "fold_assignment": _array_identity(
                self.fold_assignment,  # type: ignore[arg-type]
            ),
            "folds": [
                {
                    "fold_id": fold.fold_id,
                    "training_indices": list(fold.training_indices),
                    "validation_indices": list(fold.validation_indices),
                    "feature_transform": {
                        "mean": _array_identity(fold.feature_transform.mean),
                        "scale": fold.feature_transform.scale,
                    },
                    "label_transform": dataclasses.asdict(fold.label_transform),
                    "teacher_calibration": dataclasses.asdict(
                        fold.teacher_calibration
                    ),
                    "regularizer_mass": fold.regularizer_mass,
                    "initial_validation_mse": fold.initial_validation_mse,
                    "final_validation_mse": fold.final_validation_mse,
                    "final_recursive_coefficients": _array_identity(
                        fold.final_recursive_coefficients
                    ),
                    "final_direct_coefficients": _array_identity(
                        fold.final_direct_coefficients
                    ),
                    "coefficient_absolute_drift": fold.coefficient_absolute_drift,
                    "coefficient_relative_drift": fold.coefficient_relative_drift,
                    "validation_prediction_max_absolute_drift": (
                        fold.validation_prediction_max_absolute_drift
                    ),
                    "direct_normal_equation_residual": (
                        fold.direct_normal_equation_residual
                    ),
                }
                for fold in self.folds
            ],
            "greedy": {
                "ordered_indices": list(self.greedy.ordered_indices),
                "ordered_hashes": list(self.greedy.ordered_hashes),
                "regularizer_mass": self.greedy.regularizer_mass,
                "reanchor_steps": list(self.greedy.reanchor_steps),
                "steps": [dataclasses.asdict(step) for step in self.greedy.steps],
                "final_recursive_coefficients": [
                    _array_identity(value)
                    for value in self.greedy.final_recursive_coefficients
                ],
                "final_direct_coefficients": [
                    _array_identity(value)
                    for value in self.greedy.final_direct_coefficients
                ],
            },
            "prefixes": [
                {
                    "q": prefix.q,
                    "selected_indices": list(prefix.selected_indices),
                    "selected_hashes": list(prefix.selected_hashes),
                    "selected_pseudo_labels": _array_identity(
                        prefix.selected_pseudo_labels
                    ),
                    "pseudo_weight": prefix.pseudo_weight,
                    "ridge_lambda": prefix.ridge_lambda,
                    "training_weight_sum": prefix.training_weight_sum,
                    "fold_cv_mse": prefix.fold_cv_mse,
                    "coefficients": _array_identity(prefix.coefficients),
                    "test_predictions": _array_identity(prefix.test_predictions),
                }
                for prefix in self.prefixes
            ],
        }


@dataclass(frozen=True)
class _PreparedFold:
    inputs: ExactCVFoldInputs
    feature_transform: FeatureTransform
    label_transform: LabelTransform
    teacher_calibration: TeacherCalibration
    training_indices: tuple[int, ...]
    validation_indices: tuple[int, ...]


def _validate_exact_cv_protocol(protocol: Protocol) -> None:
    if not isinstance(protocol, Protocol):
        raise ValueError("protocol must be Protocol")
    frozen_values = (
        ("working_size", protocol.working_size, 6000),
        ("n_labeled", protocol.n_labeled, 96),
        ("n_unlabeled", protocol.n_unlabeled, 2000),
        ("n_test", protocol.n_test, 1000),
        ("q", protocol.q, 192),
        ("pseudo_weight", protocol.pseudo_weight, 0.1),
        ("ridge_lambda", protocol.ridge_lambda, 0.01),
        ("damping", protocol.damping, 0.0001),
        ("teacher_column", protocol.teacher_column, "ESM1v_ensemble"),
        ("max_length", protocol.max_length, 512),
        ("feature_scaling", protocol.preprocessing.feature_scaling, "scalar_rms"),
        ("student_fit", protocol.preprocessing.student_fit, "no_intercept"),
        ("label_ddof", protocol.preprocessing.label_ddof, 0),
    )
    for name, actual, expected in frozen_values:
        if actual != expected:
            raise ValueError(f"protocol {name} does not match the exact-CV card")


def _prepare_folds(
    *,
    raw_x_l: FloatArray,
    raw_x_u: FloatArray,
    y_l_raw: FloatArray,
    z_l: FloatArray,
    z_u: FloatArray,
    assignment: NDArray[np.int64],
) -> tuple[_PreparedFold, ...]:
    prepared: list[_PreparedFold] = []
    for fold_id in range(FOLD_COUNT):
        validation_mask = assignment == fold_id
        training_mask = ~validation_mask
        training_indices = tuple(
            int(index) for index in np.flatnonzero(training_mask)
        )
        validation_indices = tuple(
            int(index) for index in np.flatnonzero(validation_mask)
        )
        feature_transform = fit_feature_transform(raw_x_l[training_mask])
        label_transform = fit_label_transform(y_l_raw[training_mask], ddof=0)
        x_train = feature_transform.transform(raw_x_l[training_mask])
        x_validation = feature_transform.transform(raw_x_l[validation_mask])
        x_u = feature_transform.transform(raw_x_u)
        y_train = label_transform.transform(y_l_raw[training_mask])
        y_validation = label_transform.transform(y_l_raw[validation_mask])
        teacher_calibration = fit_teacher_calibration(
            z_l[training_mask],
            y_train,
        )
        pseudo_labels_u = teacher_calibration.predict(z_u)
        prepared.append(
            _PreparedFold(
                inputs=ExactCVFoldInputs(
                    fold_id=fold_id,
                    x_train=x_train,
                    y_train=y_train,
                    x_validation=x_validation,
                    y_validation=y_validation,
                    x_u=x_u,
                    pseudo_labels_u=pseudo_labels_u,
                ),
                feature_transform=feature_transform,
                label_transform=label_transform,
                teacher_calibration=teacher_calibration,
                training_indices=training_indices,
                validation_indices=validation_indices,
            )
        )
    return tuple(prepared)


def _fold_artifacts(
    prepared: tuple[_PreparedFold, ...],
    greedy: ExactCVGreedyResult,
    *,
    pseudo_weight: float,
    ridge_lambda: float,
) -> tuple[ExactCVFoldArtifact, ...]:
    selected = greedy.ordered_indices
    artifacts: list[ExactCVFoldArtifact] = []
    for fold_index, fold in enumerate(prepared):
        direct = greedy.final_direct_coefficients[fold_index]
        recursive = greedy.final_recursive_coefficients[fold_index]
        absolute_drift = float(np.linalg.norm(recursive - direct))
        relative_drift = absolute_drift / max(
            float(np.linalg.norm(direct)),
            float(np.finfo(np.float64).tiny),
        )
        validation_prediction_drift = float(
            np.max(
                np.abs(
                    fold.inputs.x_validation @ recursive
                    - fold.inputs.x_validation @ direct
                )
            )
        )
        indices = np.asarray(selected, dtype=np.int64)
        x_selected = fold.inputs.x_u[indices]
        normal = (
            fold.inputs.x_train.T @ fold.inputs.x_train
            + pseudo_weight * (x_selected.T @ x_selected)
            + greedy.regularizer_mass
            * ridge_lambda
            * np.eye(fold.inputs.x_train.shape[1])
        )
        rhs = (
            fold.inputs.x_train.T @ fold.inputs.y_train
            + pseudo_weight
            * (x_selected.T @ fold.inputs.pseudo_labels_u[indices])
        )
        direct_residual = float(np.linalg.norm(normal @ direct - rhs))
        width = fold.inputs.x_train.shape[1]
        if (
            not _parity_close(recursive, direct, width=width)
            or not _parity_close(
                fold.inputs.x_validation @ recursive,
                fold.inputs.x_validation @ direct,
                width=width,
            )
            or direct_residual
            > (
                PARITY_EPS_MULTIPLIER
                * np.finfo(np.float64).eps
                * max(1, width)
                * max(1.0, float(np.linalg.norm(rhs)))
            )
        ):
            raise ValueError("exact-CV final direct-solve parity check failed")
        artifacts.append(
            ExactCVFoldArtifact(
                fold_id=fold_index,
                training_indices=fold.training_indices,
                validation_indices=fold.validation_indices,
                feature_transform=fold.feature_transform,
                label_transform=fold.label_transform,
                teacher_calibration=fold.teacher_calibration,
                regularizer_mass=greedy.regularizer_mass,
                initial_validation_mse=greedy.steps[0].fold_mse_before[fold_index],
                final_validation_mse=greedy.steps[-1].fold_mse_after[fold_index],
                final_recursive_coefficients=recursive,
                final_direct_coefficients=direct,
                coefficient_absolute_drift=absolute_drift,
                coefficient_relative_drift=relative_drift,
                validation_prediction_max_absolute_drift=(
                    validation_prediction_drift
                ),
                direct_normal_equation_residual=direct_residual,
            )
        )
    return tuple(artifacts)


def _prefix_artifacts(
    base_fit: FitResult,
    greedy: ExactCVGreedyResult,
) -> tuple[ExactCVPrefixArtifact, ...]:
    prefixes: list[ExactCVPrefixArtifact] = []
    for q in PREFIX_COUNTS:
        selected_indices = greedy.ordered_indices[:q]
        selected = np.asarray(selected_indices, dtype=np.int64)
        selected_labels = base_fit.pseudo_labels_u[selected]
        combined_x = np.concatenate([base_fit.x_l, base_fit.x_u[selected]], axis=0)
        combined_y = np.concatenate(
            [base_fit.y_l_standardized, selected_labels]
        )
        weights = np.concatenate(
            [
                np.ones(base_fit.x_l.shape[0], dtype=np.float64),
                np.full(q, base_fit.pseudo_weight, dtype=np.float64),
            ]
        )
        coefficients = fit_weighted_ridge(
            combined_x,
            combined_y,
            base_fit.ridge_lambda,
            sample_weight=weights,
        )
        prefixes.append(
            ExactCVPrefixArtifact(
                q=q,
                selected_indices=selected_indices,
                selected_hashes=greedy.ordered_hashes[:q],
                selected_pseudo_labels=selected_labels,
                pseudo_weight=base_fit.pseudo_weight,
                ridge_lambda=base_fit.ridge_lambda,
                training_weight_sum=float(
                    base_fit.x_l.shape[0] + base_fit.pseudo_weight * q
                ),
                fold_cv_mse=greedy.steps[q - 1].mean_mse_after,
                coefficients=coefficients,
                test_predictions=np.asarray(
                    base_fit.x_test @ coefficients,
                    dtype=np.float64,
                ),
            )
        )
    return tuple(prefixes)


def _fit_exact_cv_task_single_thread(
    inputs: FitInputs,
    protocol: Protocol,
) -> ExactCVFitResult:
    if not isinstance(inputs, FitInputs):
        raise ValueError("inputs must be FitInputs")
    _validate_exact_cv_protocol(protocol)
    if inputs.seed not in protocol.seeds:
        raise ValueError("inputs seed is not a member of the exact-CV protocol")
    base_fit = fit_task(inputs, protocol)
    base_fit_digest = canonical_fit_digest(base_fit)
    assignment = balanced_fold_assignment(
        protocol.n_labeled,
        inputs.assay_id,
        inputs.seed,
        fold_count=FOLD_COUNT,
        purpose=FOLD_PURPOSE,
    )
    prepared = _prepare_folds(
        raw_x_l=inputs.x_l,
        raw_x_u=inputs.x_u,
        y_l_raw=inputs.y_l,
        z_l=inputs.z_l,
        z_u=inputs.z_u,
        assignment=assignment,
    )
    greedy = greedy_exact_cv_order(
        tuple(fold.inputs for fold in prepared),
        inputs.unlabeled_hashes,
        q=protocol.q,
        pseudo_weight=protocol.pseudo_weight,
        ridge_lambda=protocol.ridge_lambda,
    )
    folds = _fold_artifacts(
        prepared,
        greedy,
        pseudo_weight=protocol.pseudo_weight,
        ridge_lambda=protocol.ridge_lambda,
    )
    prefixes = _prefix_artifacts(base_fit, greedy)
    return ExactCVFitResult(
        base_fit=base_fit,
        base_fit_digest=base_fit_digest,
        card_id=CARD_ID,
        card_sha=CARD_SHA,
        fold_count=FOLD_COUNT,
        fold_purpose=FOLD_PURPOSE,
        prefix_counts=PREFIX_COUNTS,
        reanchor_steps=REANCHOR_STEPS,
        raw_x_l=inputs.x_l,
        raw_x_u=inputs.x_u,
        raw_x_test=inputs.x_test,
        fold_assignment=assignment,
        folds=folds,
        greedy=greedy,
        prefixes=prefixes,
    )


def fit_exact_cv_task(inputs: FitInputs, protocol: Protocol) -> ExactCVFitResult:
    """Fit and freeze the complete exact-CV ordering before unblinding."""
    with _locked_blas_scope():
        return _fit_exact_cv_task_single_thread(inputs, protocol)


def _greedy_results_close(
    actual: ExactCVGreedyResult,
    expected: ExactCVGreedyResult,
) -> bool:
    if (
        actual.ordered_indices != expected.ordered_indices
        or actual.ordered_hashes != expected.ordered_hashes
        or actual.reanchor_steps != expected.reanchor_steps
        or actual.regularizer_mass != expected.regularizer_mass
        or not _diagnostic_values_close(
            [dataclasses.asdict(step) for step in actual.steps],
            [dataclasses.asdict(step) for step in expected.steps],
        )
    ):
        return False
    return all(
        _scaled_algebra_close(left, right)
        for left, right in zip(
            actual.final_recursive_coefficients
            + actual.final_direct_coefficients,
            expected.final_recursive_coefficients
            + expected.final_direct_coefficients,
            strict=True,
        )
    )


def _validate_exact_cv_fit(result: ExactCVFitResult) -> None:
    if (
        result.card_id != CARD_ID
        or result.card_sha != CARD_SHA
        or result.fold_count != FOLD_COUNT
        or result.fold_purpose != FOLD_PURPOSE
        or result.prefix_counts != PREFIX_COUNTS
        or result.reanchor_steps != REANCHOR_STEPS
    ):
        raise ValueError("exact-CV card identity mismatch")
    actual_base_digest = canonical_fit_digest(result.base_fit)
    if result.base_fit_digest != actual_base_digest:
        raise ValueError("exact-CV base fit digest mismatch")
    base = result.base_fit
    available_card_values = (
        ("n_labeled", base.x_l.shape[0], 96),
        ("n_unlabeled", base.x_u.shape[0], 2000),
        ("n_test", base.x_test.shape[0], 1000),
        ("q", base.q, 192),
        ("pseudo_weight", base.pseudo_weight, 0.1),
        ("ridge_lambda", base.ridge_lambda, 0.01),
        ("damping", base.damping, 0.0001),
    )
    for name, card_actual, card_expected in available_card_values:
        if card_actual != card_expected:
            raise ValueError(f"exact-CV base fit {name} mismatches the card")
    expected_shapes = (
        (result.raw_x_l.shape, base.x_l.shape),
        (result.raw_x_u.shape, base.x_u.shape),
        (result.raw_x_test.shape, base.x_test.shape),
    )
    if any(left != right for left, right in expected_shapes):
        raise ValueError("exact-CV raw feature shapes mismatch")
    for raw, transformed in (
        (result.raw_x_l, base.x_l),
        (result.raw_x_u, base.x_u),
        (result.raw_x_test, base.x_test),
    ):
        if not _scaled_algebra_close(
            base.feature_transform.transform(raw),
            transformed,
        ):
            raise ValueError("exact-CV raw features do not reconstruct base fit")
    expected_assignment = balanced_fold_assignment(
        base.x_l.shape[0],
        base.assay_id,
        base.seed,
        fold_count=FOLD_COUNT,
        purpose=FOLD_PURPOSE,
    )
    if not np.array_equal(result.fold_assignment, expected_assignment):
        raise ValueError("exact-CV fold assignment mismatch")
    prepared = _prepare_folds(
        raw_x_l=result.raw_x_l,
        raw_x_u=result.raw_x_u,
        y_l_raw=base.y_l_raw,
        z_l=base.z_l,
        z_u=base.z_u,
        assignment=expected_assignment,
    )
    expected_greedy = greedy_exact_cv_order(
        tuple(fold.inputs for fold in prepared),
        base.unlabeled_hashes,
        q=base.q,
        pseudo_weight=base.pseudo_weight,
        ridge_lambda=base.ridge_lambda,
    )
    if not _greedy_results_close(result.greedy, expected_greedy):
        raise ValueError("exact-CV greedy state mismatch")
    expected_folds = _fold_artifacts(
        prepared,
        expected_greedy,
        pseudo_weight=base.pseudo_weight,
        ridge_lambda=base.ridge_lambda,
    )
    if len(result.folds) != len(expected_folds):
        raise ValueError("exact-CV fold artifact count mismatch")
    for actual_fold, expected_fold in zip(
        result.folds,
        expected_folds,
        strict=True,
    ):
        discrete_match = (
            actual_fold.fold_id == expected_fold.fold_id
            and actual_fold.training_indices == expected_fold.training_indices
            and actual_fold.validation_indices == expected_fold.validation_indices
            and actual_fold.label_transform.ddof
            == expected_fold.label_transform.ddof
        )
        numerical_pairs = (
            (
                actual_fold.feature_transform.mean,
                expected_fold.feature_transform.mean,
            ),
            (
                actual_fold.feature_transform.scale,
                expected_fold.feature_transform.scale,
            ),
            (actual_fold.label_transform.mean, expected_fold.label_transform.mean),
            (actual_fold.label_transform.scale, expected_fold.label_transform.scale),
            (
                actual_fold.teacher_calibration.slope,
                expected_fold.teacher_calibration.slope,
            ),
            (
                actual_fold.teacher_calibration.intercept,
                expected_fold.teacher_calibration.intercept,
            ),
            (actual_fold.regularizer_mass, expected_fold.regularizer_mass),
            (
                actual_fold.initial_validation_mse,
                expected_fold.initial_validation_mse,
            ),
            (
                actual_fold.final_validation_mse,
                expected_fold.final_validation_mse,
            ),
            (
                actual_fold.final_recursive_coefficients,
                expected_fold.final_recursive_coefficients,
            ),
            (
                actual_fold.final_direct_coefficients,
                expected_fold.final_direct_coefficients,
            ),
            (
                actual_fold.coefficient_absolute_drift,
                expected_fold.coefficient_absolute_drift,
            ),
            (
                actual_fold.coefficient_relative_drift,
                expected_fold.coefficient_relative_drift,
            ),
            (
                actual_fold.validation_prediction_max_absolute_drift,
                expected_fold.validation_prediction_max_absolute_drift,
            ),
            (
                actual_fold.direct_normal_equation_residual,
                expected_fold.direct_normal_equation_residual,
            ),
        )
        if not discrete_match or any(
            not _scaled_algebra_close(left, right)
            for left, right in numerical_pairs
        ):
            raise ValueError("exact-CV fold artifact mismatch")
    expected_prefixes = _prefix_artifacts(base, expected_greedy)
    if len(result.prefixes) != len(expected_prefixes):
        raise ValueError("exact-CV prefix artifact count mismatch")
    for actual_prefix, expected_prefix in zip(
        result.prefixes,
        expected_prefixes,
        strict=True,
    ):
        if (
            actual_prefix.q != expected_prefix.q
            or actual_prefix.selected_indices != expected_prefix.selected_indices
            or actual_prefix.selected_hashes != expected_prefix.selected_hashes
            or actual_prefix.pseudo_weight != expected_prefix.pseudo_weight
            or actual_prefix.ridge_lambda != expected_prefix.ridge_lambda
            or actual_prefix.training_weight_sum
            != expected_prefix.training_weight_sum
            or actual_prefix.fold_cv_mse != expected_prefix.fold_cv_mse
            or not _scaled_algebra_close(
                actual_prefix.selected_pseudo_labels,
                expected_prefix.selected_pseudo_labels,
            )
            or not _scaled_algebra_close(
                actual_prefix.coefficients,
                expected_prefix.coefficients,
            )
            or not _scaled_algebra_close(
                actual_prefix.test_predictions,
                expected_prefix.test_predictions,
            )
        ):
            raise ValueError("exact-CV prefix artifact mismatch")
    endpoint = result.prefixes[-1]
    random_method = next(
        method for method in base.methods if method.name == "random"
    )
    if not _scaled_algebra_close(
        random_method.training_weight_sum,
        endpoint.training_weight_sum,
    ):
        raise ValueError("exact-CV endpoint is not weight-matched to random")


def canonical_exact_cv_fit_digest(result: object) -> str:
    """Reconstruct, validate, and hash one hidden-label-free exact-CV fit."""
    if not isinstance(result, ExactCVFitResult):
        raise ValueError("result must be ExactCVFitResult")
    with _locked_blas_scope():
        _validate_exact_cv_fit(result)
    return _canonical_payload_digest(result.to_payload())


@dataclass(frozen=True)
class ExactCVPrefixEvaluation:
    """Hidden-label metrics for one descriptive full-96 prefix fit."""

    q: int
    fold_cv_mse: float
    marginal_validation_mse_reduction: float
    spearman: float
    mse: float
    ndcg_10pct: float
    selected_pseudo_label_mae: float

    def __post_init__(self) -> None:
        if type(self.q) is not int or self.q not in PREFIX_COUNTS:
            raise ValueError("q must be a frozen exact-CV prefix count")
        for name in (
            "fold_cv_mse",
            "marginal_validation_mse_reduction",
            "spearman",
            "mse",
            "ndcg_10pct",
            "selected_pseudo_label_mae",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class ExactCVEvaluationResult:
    """Reference methods and exact-CV prefix metrics after unblinding."""

    reference: EvaluationResult
    prefixes: tuple[ExactCVPrefixEvaluation, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.reference, EvaluationResult):
            raise ValueError("reference must be EvaluationResult")
        prefixes = tuple(self.prefixes)
        if tuple(prefix.q for prefix in prefixes) != PREFIX_COUNTS:
            raise ValueError("prefix evaluations must match frozen counts")
        object.__setattr__(self, "prefixes", prefixes)

    @property
    def assay_id(self) -> str:
        return self.reference.assay_id

    @property
    def seed(self) -> int:
        return self.reference.seed

    def to_payload(self) -> dict[str, object]:
        return {
            "assay_id": self.assay_id,
            "seed": self.seed,
            "reference": dataclasses.asdict(self.reference),
            "prefixes": [dataclasses.asdict(prefix) for prefix in self.prefixes],
        }


def _evaluate_exact_cv_task_single_thread(
    fit: ExactCVFitResult,
    labels: EvaluationLabels,
    *,
    protocol: Protocol,
    expected_fit_digest: str,
    expected_evaluation_digest: str,
) -> ExactCVEvaluationResult:
    if not isinstance(fit, ExactCVFitResult):
        raise ValueError("fit must be ExactCVFitResult")
    if not isinstance(expected_fit_digest, str) or not _SHA256_PATTERN.fullmatch(
        expected_fit_digest
    ):
        raise ValueError("expected_fit_digest must be a lowercase SHA-256 digest")
    actual_digest = canonical_exact_cv_fit_digest(fit)
    if actual_digest != expected_fit_digest:
        raise ValueError(
            "current exact-CV fit digest does not match expected fit digest"
        )
    reference = evaluate_task(
        fit.base_fit,
        labels,
        protocol=protocol,
        expected_fit_digest=fit.base_fit_digest,
        expected_evaluation_digest=expected_evaluation_digest,
    )
    y_u = fit.base_fit.label_transform.transform(labels.y_u)
    y_test = fit.base_fit.label_transform.transform(labels.y_test)
    absolute_errors = np.abs(fit.base_fit.pseudo_labels_u - y_u)
    evaluations: list[ExactCVPrefixEvaluation] = []
    for prefix in fit.prefixes:
        selected = np.asarray(prefix.selected_indices, dtype=np.int64)
        step = fit.greedy.steps[prefix.q - 1]
        evaluations.append(
            ExactCVPrefixEvaluation(
                q=prefix.q,
                fold_cv_mse=prefix.fold_cv_mse,
                marginal_validation_mse_reduction=step.mean_mse_reduction,
                spearman=spearman_correlation(y_test, prefix.test_predictions),
                mse=standardized_mse(y_test, prefix.test_predictions),
                ndcg_10pct=ndcg_at_10_percent(y_test, prefix.test_predictions),
                selected_pseudo_label_mae=float(np.mean(absolute_errors[selected])),
            )
        )
    return ExactCVEvaluationResult(
        reference=reference,
        prefixes=tuple(evaluations),
    )


def evaluate_exact_cv_task(
    fit: ExactCVFitResult,
    labels: EvaluationLabels,
    *,
    protocol: Protocol,
    expected_fit_digest: str,
    expected_evaluation_digest: str,
) -> ExactCVEvaluationResult:
    """Evaluate a reconstructively frozen exact-CV fit against hidden labels."""
    with _locked_blas_scope():
        return _evaluate_exact_cv_task_single_thread(
            fit,
            labels,
            protocol=protocol,
            expected_fit_digest=expected_fit_digest,
            expected_evaluation_digest=expected_evaluation_digest,
        )
