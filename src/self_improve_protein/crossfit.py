"""Pure hidden-label-safe cross-fitted outer-gradient experiment."""

from __future__ import annotations

import dataclasses
import re
from dataclasses import dataclass
from typing import Final

import numpy as np

from self_improve_protein.config import Protocol
from self_improve_protein.experiment import (
    CosineDiagnostic,
    EvaluationLabels,
    EvaluationResult,
    FitInputs,
    FitResult,
    MethodArtifact,
    NormalMatrixDiagnostics,
    OracleDiagnostics,
    ScoreDiagnostics,
    _array_identity,
    _canonical_payload_digest,
    _constant,
    _correlation,
    _cosine_diagnostic,
    _diagnostic_values_close,
    _float_array,
    _locality_index,
    _locked_blas_scope,
    _matrix_diagnostics,
    _metric_cosine,
    _scaled_algebra_close,
    _score_diagnostics,
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
    fit_weighted_ridge,
    labeled_gradient_hessian,
    squared_loss,
)
from self_improve_protein.selection import (
    balanced_fold_assignment,
    cross_fitted_influence_scores,
    out_of_fold_ridge_gradient,
    stable_top_k,
)

CARD_ID: Final = "crossfit_outer_gradient_v1"
CARD_SHA: Final = "383afd7a5bae9c2ebd6768a112a82980236540fc0f66e3a294ef298961b8596f"
FOLD_PURPOSE: Final = "crossfit_outer_folds_v1"
FOLD_COUNT: Final = 4

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class CrossfitMethodArtifact:
    """Frozen cross-fitted selection and exact weighted-ridge fit."""

    selected_indices: tuple[int, ...]
    selected_hashes: tuple[str, ...]
    selected_pseudo_labels: np.ndarray
    pseudo_weight: float
    ridge_lambda: float
    training_weight_sum: float
    coefficients: np.ndarray
    test_predictions: np.ndarray

    def __post_init__(self) -> None:
        indices = tuple(self.selected_indices)
        hashes = tuple(self.selected_hashes)
        if any(type(index) is not int or index < 0 for index in indices):
            raise ValueError("selected_indices must contain non-negative integers")
        if len(indices) != len(set(indices)):
            raise ValueError("selected_indices must be unique")
        if any(not isinstance(value, str) for value in hashes):
            raise ValueError("selected_hashes must contain strings")
        selected_labels = _float_array(
            self.selected_pseudo_labels,
            name="selected_pseudo_labels",
            ndim=1,
        )
        coefficients = _float_array(
            self.coefficients,
            name="coefficients",
            ndim=1,
        )
        predictions = _float_array(
            self.test_predictions,
            name="test_predictions",
            ndim=1,
        )
        if len(indices) != len(hashes) or len(indices) != selected_labels.size:
            raise ValueError("selected artifact lengths must match")
        for value, name in (
            (self.pseudo_weight, "pseudo_weight"),
            (self.ridge_lambda, "ridge_lambda"),
            (self.training_weight_sum, "training_weight_sum"),
        ):
            if isinstance(value, (bool, str, bytes)) or not np.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        object.__setattr__(self, "selected_indices", indices)
        object.__setattr__(self, "selected_hashes", hashes)
        object.__setattr__(self, "selected_pseudo_labels", selected_labels)
        object.__setattr__(self, "coefficients", coefficients)
        object.__setattr__(self, "test_predictions", predictions)


@dataclass(frozen=True)
class CrossfitMethodDiagnostics:
    stationarity_residual: float
    normal_matrix: NormalMatrixDiagnostics
    first_order_outer_loss_change: float
    realized_labeled_loss_change: float
    displacement_cosine: float | None
    displacement_cosine_defined: bool
    displacement_relative_error: float
    locality_index: float


@dataclass(frozen=True)
class CrossfitFitDiagnostics:
    score: ScoreDiagnostics
    overlap_random: float
    overlap_top_teacher: float
    overlap_full: float
    overlap_no_hessian: float
    outer_full_gradient_cosine: CosineDiagnostic
    method: CrossfitMethodDiagnostics


def _int_array(value: object, *, name: str) -> np.ndarray:
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be an integer 1D array") from error
    if raw.ndim != 1 or raw.size == 0 or raw.dtype.kind not in {"i", "u"}:
        raise ValueError(f"{name} must be a non-empty integer 1D array")
    if raw.dtype.kind == "u" and np.any(raw > np.iinfo(np.int64).max):
        raise ValueError(f"{name} values exceed int64")
    array = np.array(raw, dtype=np.int64, copy=True, order="C")
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class CrossfitFitResult:
    """Hidden-label-free crossfit fit bound to one validated v0 base fit."""

    base_fit: FitResult
    base_fit_digest: str
    card_id: str
    card_sha: str
    fold_purpose: str
    fold_count: int
    fold_assignment: np.ndarray
    outer_gradient: np.ndarray
    scores: np.ndarray
    method: CrossfitMethodArtifact
    diagnostics: CrossfitFitDiagnostics

    def __post_init__(self) -> None:
        if not isinstance(self.base_fit, FitResult):
            raise ValueError("base_fit must be FitResult")
        if not isinstance(self.base_fit_digest, str):
            raise ValueError("base_fit_digest must be a string")
        for value, name in (
            (self.card_id, "card_id"),
            (self.card_sha, "card_sha"),
            (self.fold_purpose, "fold_purpose"),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string")
        if type(self.fold_count) is not int or self.fold_count < 2:
            raise ValueError("fold_count must be an integer of at least two")
        assignment = _int_array(self.fold_assignment, name="fold_assignment")
        outer_gradient = _float_array(
            self.outer_gradient,
            name="outer_gradient",
            ndim=1,
        )
        scores = _float_array(self.scores, name="scores", ndim=1)
        if not isinstance(self.method, CrossfitMethodArtifact):
            raise ValueError("method must be CrossfitMethodArtifact")
        if not isinstance(self.diagnostics, CrossfitFitDiagnostics):
            raise ValueError("diagnostics must be CrossfitFitDiagnostics")
        object.__setattr__(self, "fold_assignment", assignment)
        object.__setattr__(self, "outer_gradient", outer_gradient)
        object.__setattr__(self, "scores", scores)

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
        """Return the canonical hidden-label-free crossfit fit identity."""
        return {
            "assay_id": self.assay_id,
            "seed": self.seed,
            "source_digest": self.source_digest,
            "protocol_digest": self.protocol_digest,
            "base_fit_digest": self.base_fit_digest,
            "card_id": self.card_id,
            "card_sha": self.card_sha,
            "fold_purpose": self.fold_purpose,
            "fold_count": self.fold_count,
            "fold_assignment": self.fold_assignment.tolist(),
            "outer_gradient": _array_identity(self.outer_gradient),
            "scores": _array_identity(self.scores),
            "method": {
                "selected_indices": list(self.method.selected_indices),
                "selected_hashes": list(self.method.selected_hashes),
                "selected_pseudo_labels": _array_identity(
                    self.method.selected_pseudo_labels
                ),
                "pseudo_weight": self.method.pseudo_weight,
                "ridge_lambda": self.method.ridge_lambda,
                "training_weight_sum": self.method.training_weight_sum,
                "coefficients": _array_identity(self.method.coefficients),
                "test_predictions": _array_identity(self.method.test_predictions),
            },
            "diagnostics": dataclasses.asdict(self.diagnostics),
        }


@dataclass(frozen=True)
class CrossfitMethodEvaluation:
    spearman: float
    mse: float
    ndcg_10pct: float
    selected_pseudo_label_mae: float


@dataclass(frozen=True)
class CrossfitEvaluationResult:
    reference: EvaluationResult
    crossfit: CrossfitMethodEvaluation
    test_risk_oracle: OracleDiagnostics


def _reference_methods(base_fit: FitResult) -> dict[str, MethodArtifact]:
    return {method.name: method for method in base_fit.methods}


def _build_crossfit_method_diagnostics(
    base_fit: FitResult,
    outer_gradient: np.ndarray,
    method: CrossfitMethodArtifact,
) -> CrossfitMethodDiagnostics:
    selected = np.asarray(method.selected_indices, dtype=np.int64)
    x_selected = base_fit.x_u[selected]
    y_selected = base_fit.pseudo_labels_u[selected]
    combined_x = np.concatenate([base_fit.x_l, x_selected], axis=0)
    combined_y = np.concatenate([base_fit.y_l_standardized, y_selected])
    weights = np.concatenate(
        [
            np.ones(base_fit.x_l.shape[0], dtype=np.float64),
            np.full(base_fit.q, base_fit.pseudo_weight, dtype=np.float64),
        ]
    )
    denominator = float(np.sum(weights))
    stationarity = (
        combined_x.T
        @ (weights * (combined_x @ method.coefficients - combined_y))
        / denominator
        + base_fit.ridge_lambda * method.coefficients
    )
    theta_zero = base_fit.methods[0].coefficients
    full_gradient, hessian = labeled_gradient_hessian(
        base_fit.x_l,
        base_fit.y_l_standardized,
        theta_zero,
        base_fit.ridge_lambda,
    )
    selected_gradient = (
        x_selected.T @ (x_selected @ theta_zero - y_selected) / base_fit.q
    )
    t = (
        base_fit.pseudo_weight
        * base_fit.q
        / (base_fit.x_l.shape[0] + base_fit.pseudo_weight * base_fit.q)
    )
    predicted_displacement = -t * np.linalg.solve(
        hessian,
        selected_gradient - full_gradient,
    )
    realized_displacement = method.coefficients - theta_zero
    realized_norm = float(np.linalg.norm(realized_displacement))
    displacement_cosine = _cosine_diagnostic(
        predicted_displacement,
        realized_displacement,
    )
    return CrossfitMethodDiagnostics(
        stationarity_residual=float(np.linalg.norm(stationarity)),
        normal_matrix=_matrix_diagnostics(
            combined_x,
            weights,
            base_fit.ridge_lambda,
        ),
        first_order_outer_loss_change=float(outer_gradient @ predicted_displacement),
        realized_labeled_loss_change=float(
            squared_loss(
                base_fit.x_l,
                base_fit.y_l_standardized,
                method.coefficients,
            )
            - squared_loss(
                base_fit.x_l,
                base_fit.y_l_standardized,
                theta_zero,
            )
        ),
        displacement_cosine=displacement_cosine.value,
        displacement_cosine_defined=displacement_cosine.defined,
        displacement_relative_error=float(
            np.linalg.norm(realized_displacement - predicted_displacement)
            / max(realized_norm, float(np.finfo(np.float64).tiny))
        ),
        locality_index=_locality_index(
            base_fit.x_l,
            x_selected,
            hessian,
            t,
        ),
    )


def _build_crossfit_diagnostics(
    base_fit: FitResult,
    outer_gradient: np.ndarray,
    scores: np.ndarray,
    method: CrossfitMethodArtifact,
) -> CrossfitFitDiagnostics:
    references = _reference_methods(base_fit)
    random_method = references["random"]
    top_teacher = references["top_teacher"]
    full = references["ours"]
    no_hessian = references["no_hessian"]
    selected = method.selected_indices
    random_selected = random_method.selected_indices
    full_gradient, _ = labeled_gradient_hessian(
        base_fit.x_l,
        base_fit.y_l_standardized,
        base_fit.methods[0].coefficients,
        base_fit.ridge_lambda,
    )
    selected_set = set(selected)
    return CrossfitFitDiagnostics(
        score=_score_diagnostics(scores, selected, random_selected),
        overlap_random=len(selected_set & set(random_selected)) / base_fit.q,
        overlap_top_teacher=(
            len(selected_set & set(top_teacher.selected_indices)) / base_fit.q
        ),
        overlap_full=len(selected_set & set(full.selected_indices)) / base_fit.q,
        overlap_no_hessian=(
            len(selected_set & set(no_hessian.selected_indices)) / base_fit.q
        ),
        outer_full_gradient_cosine=_cosine_diagnostic(
            outer_gradient,
            full_gradient,
        ),
        method=_build_crossfit_method_diagnostics(
            base_fit,
            outer_gradient,
            method,
        ),
    )


def _fit_crossfit_task_single_thread(
    inputs: FitInputs,
    protocol: Protocol,
) -> CrossfitFitResult:
    base_fit = fit_task(inputs, protocol)
    base_fit_digest = canonical_fit_digest(base_fit)
    fold_assignment = balanced_fold_assignment(
        base_fit.x_l.shape[0],
        base_fit.assay_id,
        base_fit.seed,
        fold_count=FOLD_COUNT,
        purpose=FOLD_PURPOSE,
    )
    outer_gradient = out_of_fold_ridge_gradient(
        base_fit.x_l,
        base_fit.y_l_standardized,
        fold_assignment,
        base_fit.ridge_lambda,
    )
    theta_zero = base_fit.methods[0].coefficients
    scores = cross_fitted_influence_scores(
        base_fit.x_l,
        base_fit.y_l_standardized,
        base_fit.x_u,
        base_fit.pseudo_labels_u,
        theta_zero,
        base_fit.ridge_lambda,
        base_fit.damping,
        base_fit.assay_id,
        base_fit.seed,
        fold_count=FOLD_COUNT,
        purpose=FOLD_PURPOSE,
    )
    selected = tuple(
        int(index)
        for index in stable_top_k(
            scores,
            base_fit.unlabeled_hashes,
            base_fit.q,
        )
    )
    selected_array = np.asarray(selected, dtype=np.int64)
    selected_labels = base_fit.pseudo_labels_u[selected_array]
    combined_x = np.concatenate(
        [base_fit.x_l, base_fit.x_u[selected_array]],
        axis=0,
    )
    combined_y = np.concatenate([base_fit.y_l_standardized, selected_labels])
    weights = np.concatenate(
        [
            np.ones(base_fit.x_l.shape[0], dtype=np.float64),
            np.full(base_fit.q, base_fit.pseudo_weight, dtype=np.float64),
        ]
    )
    coefficients = fit_weighted_ridge(
        combined_x,
        combined_y,
        base_fit.ridge_lambda,
        sample_weight=weights,
    )
    predictions = np.asarray(base_fit.x_test @ coefficients, dtype=np.float64)
    if not np.all(np.isfinite(predictions)) or _constant(predictions):
        raise ValueError("crossfit produced a constant primary prediction")
    method = CrossfitMethodArtifact(
        selected_indices=selected,
        selected_hashes=tuple(base_fit.unlabeled_hashes[index] for index in selected),
        selected_pseudo_labels=selected_labels,
        pseudo_weight=base_fit.pseudo_weight,
        ridge_lambda=base_fit.ridge_lambda,
        training_weight_sum=float(np.sum(weights)),
        coefficients=coefficients,
        test_predictions=predictions,
    )
    diagnostics = _build_crossfit_diagnostics(
        base_fit,
        outer_gradient,
        scores,
        method,
    )
    return CrossfitFitResult(
        base_fit=base_fit,
        base_fit_digest=base_fit_digest,
        card_id=CARD_ID,
        card_sha=CARD_SHA,
        fold_purpose=FOLD_PURPOSE,
        fold_count=FOLD_COUNT,
        fold_assignment=fold_assignment,
        outer_gradient=outer_gradient,
        scores=scores,
        method=method,
        diagnostics=diagnostics,
    )


def fit_crossfit_task(inputs: FitInputs, protocol: Protocol) -> CrossfitFitResult:
    """Fit the frozen reference task and crossfit repair before unblinding."""
    with _locked_blas_scope():
        return _fit_crossfit_task_single_thread(inputs, protocol)


def _validate_crossfit_fit(result: CrossfitFitResult) -> None:
    if not isinstance(result, CrossfitFitResult):
        raise ValueError("result must be CrossfitFitResult")
    if result.card_id != CARD_ID or result.card_sha != CARD_SHA:
        raise ValueError("crossfit card identity mismatch")
    if result.fold_purpose != FOLD_PURPOSE or result.fold_count != FOLD_COUNT:
        raise ValueError("crossfit fold policy mismatch")
    if not _SHA256_PATTERN.fullmatch(result.base_fit_digest):
        raise ValueError("base_fit_digest must be a lowercase SHA-256 digest")
    actual_base_digest = canonical_fit_digest(result.base_fit)
    if result.base_fit_digest != actual_base_digest:
        raise ValueError("base fit digest mismatch")
    base_fit = result.base_fit
    expected_folds = balanced_fold_assignment(
        base_fit.x_l.shape[0],
        base_fit.assay_id,
        base_fit.seed,
        fold_count=FOLD_COUNT,
        purpose=FOLD_PURPOSE,
    )
    if not np.array_equal(result.fold_assignment, expected_folds):
        raise ValueError("crossfit fold assignment mismatch")
    expected_outer = out_of_fold_ridge_gradient(
        base_fit.x_l,
        base_fit.y_l_standardized,
        expected_folds,
        base_fit.ridge_lambda,
    )
    if not _scaled_algebra_close(result.outer_gradient, expected_outer):
        raise ValueError("crossfit outer gradient mismatch")
    expected_scores = cross_fitted_influence_scores(
        base_fit.x_l,
        base_fit.y_l_standardized,
        base_fit.x_u,
        base_fit.pseudo_labels_u,
        base_fit.methods[0].coefficients,
        base_fit.ridge_lambda,
        base_fit.damping,
        base_fit.assay_id,
        base_fit.seed,
        fold_count=FOLD_COUNT,
        purpose=FOLD_PURPOSE,
    )
    if not _scaled_algebra_close(result.scores, expected_scores):
        raise ValueError("crossfit score mismatch")
    expected_selected = tuple(
        int(index)
        for index in stable_top_k(
            expected_scores,
            base_fit.unlabeled_hashes,
            base_fit.q,
        )
    )
    method = result.method
    if method.selected_indices != expected_selected:
        raise ValueError("crossfit selected indices mismatch")
    expected_hashes = tuple(
        base_fit.unlabeled_hashes[index] for index in expected_selected
    )
    if method.selected_hashes != expected_hashes:
        raise ValueError("crossfit selected hashes mismatch")
    selected_array = np.asarray(expected_selected, dtype=np.int64)
    expected_labels = base_fit.pseudo_labels_u[selected_array]
    if not np.array_equal(method.selected_pseudo_labels, expected_labels):
        raise ValueError("crossfit selected pseudo-labels mismatch")
    if (
        method.pseudo_weight != base_fit.pseudo_weight
        or method.ridge_lambda != base_fit.ridge_lambda
    ):
        raise ValueError("crossfit method hyperparameter mismatch")
    combined_x = np.concatenate(
        [base_fit.x_l, base_fit.x_u[selected_array]],
        axis=0,
    )
    combined_y = np.concatenate([base_fit.y_l_standardized, expected_labels])
    weights = np.concatenate(
        [
            np.ones(base_fit.x_l.shape[0], dtype=np.float64),
            np.full(base_fit.q, base_fit.pseudo_weight, dtype=np.float64),
        ]
    )
    denominator = float(np.sum(weights))
    if method.training_weight_sum != denominator:
        raise ValueError("crossfit training weight sum mismatch")
    stationarity = (
        combined_x.T
        @ (weights * (combined_x @ method.coefficients - combined_y))
        / denominator
        + base_fit.ridge_lambda * method.coefficients
    )
    right_hand_scale = float(
        np.linalg.norm(combined_x.T @ (weights * combined_y) / denominator)
    )
    tolerance = (
        256.0
        * np.finfo(np.float64).eps
        * max(1.0, right_hand_scale, float(np.linalg.norm(method.coefficients)))
        * max(1, combined_x.shape[1])
    )
    if float(np.linalg.norm(stationarity)) > tolerance:
        raise ValueError("crossfit coefficients violate normalized normal equations")
    expected_predictions = base_fit.x_test @ method.coefficients
    if not _scaled_algebra_close(method.test_predictions, expected_predictions):
        raise ValueError("crossfit test predictions mismatch")
    if _constant(method.test_predictions):
        raise ValueError("crossfit produced a constant primary prediction")
    expected_diagnostics = _build_crossfit_diagnostics(
        base_fit,
        expected_outer,
        expected_scores,
        method,
    )
    if not _diagnostic_values_close(
        dataclasses.asdict(result.diagnostics),
        dataclasses.asdict(expected_diagnostics),
    ):
        raise ValueError("crossfit diagnostics mismatch")


def _canonical_crossfit_fit_digest_unchecked(result: CrossfitFitResult) -> str:
    return _canonical_payload_digest(result.to_payload())


def canonical_crossfit_fit_digest(result: CrossfitFitResult) -> str:
    """Reconstruct, validate, and hash every hidden-label-free crossfit claim."""
    with _locked_blas_scope():
        _validate_crossfit_fit(result)
    return _canonical_crossfit_fit_digest_unchecked(result)


def _evaluate_crossfit_task_single_thread(
    fit: CrossfitFitResult,
    labels: EvaluationLabels,
    *,
    protocol: Protocol,
    expected_crossfit_fit_digest: str,
    expected_evaluation_digest: str,
) -> CrossfitEvaluationResult:
    if not isinstance(fit, CrossfitFitResult):
        raise ValueError("fit must be CrossfitFitResult")
    if not isinstance(
        expected_crossfit_fit_digest, str
    ) or not _SHA256_PATTERN.fullmatch(expected_crossfit_fit_digest):
        raise ValueError("expected_crossfit_fit_digest must be a SHA-256 digest")
    actual_digest = canonical_crossfit_fit_digest(fit)
    if actual_digest != expected_crossfit_fit_digest:
        raise ValueError("current crossfit fit digest does not match expected digest")
    reference = evaluate_task(
        fit.base_fit,
        labels,
        protocol=protocol,
        expected_fit_digest=fit.base_fit_digest,
        expected_evaluation_digest=expected_evaluation_digest,
    )
    y_u = fit.base_fit.label_transform.transform(labels.y_u)
    y_test = fit.base_fit.label_transform.transform(labels.y_test)
    selected = np.asarray(fit.method.selected_indices, dtype=np.int64)
    absolute_errors = np.abs(fit.base_fit.pseudo_labels_u - y_u)
    crossfit_evaluation = CrossfitMethodEvaluation(
        spearman=spearman_correlation(y_test, fit.method.test_predictions),
        mse=standardized_mse(y_test, fit.method.test_predictions),
        ndcg_10pct=ndcg_at_10_percent(y_test, fit.method.test_predictions),
        selected_pseudo_label_mae=float(np.mean(absolute_errors[selected])),
    )
    theta_zero = fit.base_fit.methods[0].coefficients
    full_gradient, hessian = labeled_gradient_hessian(
        fit.base_fit.x_l,
        fit.base_fit.y_l_standardized,
        theta_zero,
        fit.base_fit.ridge_lambda,
    )
    test_gradient = (
        fit.base_fit.x_test.T
        @ (fit.base_fit.x_test @ theta_zero - y_test)
        / y_test.size
    )
    candidate_difference = (
        fit.base_fit.x_u @ theta_zero - fit.base_fit.pseudo_labels_u
    )[:, None] * fit.base_fit.x_u - full_gradient
    damped_hessian = hessian + fit.base_fit.damping * np.eye(hessian.shape[0])
    oracle_scores = candidate_difference @ np.linalg.solve(
        damped_hessian,
        test_gradient,
    )
    gradient_cosine, cosine_defined = _metric_cosine(
        fit.outer_gradient,
        test_gradient,
        damped_hessian,
    )
    oracle = OracleDiagnostics(
        score_alignment=_correlation(oracle_scores, fit.scores),
        score_vs_absolute_error=_correlation(absolute_errors, fit.scores),
        gradient_cosine=gradient_cosine,
        gradient_cosine_defined=cosine_defined,
    )
    return CrossfitEvaluationResult(
        reference=reference,
        crossfit=crossfit_evaluation,
        test_risk_oracle=oracle,
    )


def evaluate_crossfit_task(
    fit: CrossfitFitResult,
    labels: EvaluationLabels,
    *,
    protocol: Protocol,
    expected_crossfit_fit_digest: str,
    expected_evaluation_digest: str,
) -> CrossfitEvaluationResult:
    """Validate frozen digests before crossfit metrics access hidden outcomes."""
    with _locked_blas_scope():
        return _evaluate_crossfit_task_single_thread(
            fit,
            labels,
            protocol=protocol,
            expected_crossfit_fit_digest=expected_crossfit_fit_digest,
            expected_evaluation_digest=expected_evaluation_digest,
        )
