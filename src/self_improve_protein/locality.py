"""Pure hidden-label-safe pseudo-perturbation locality experiment."""

from __future__ import annotations

import dataclasses
import re
from dataclasses import dataclass
from typing import Final, Literal, TypeAlias, cast

import numpy as np

from self_improve_protein.config import Protocol
from self_improve_protein.crossfit import FOLD_COUNT, FOLD_PURPOSE
from self_improve_protein.experiment import (
    CorrelationDiagnostic,
    EvaluationLabels,
    EvaluationResult,
    FitInputs,
    FitResult,
    _array_identity,
    _canonical_payload_digest,
    _correlation,
    _cosine_diagnostic,
    _diagnostic_values_close,
    _float_array,
    _locality_index,
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
from self_improve_protein.provenance import derive_seed
from self_improve_protein.ridge import (
    fit_weighted_ridge,
    labeled_gradient_hessian,
    squared_loss,
)
from self_improve_protein.selection import (
    cross_fitted_influence_scores,
    stable_top_k,
)

CARD_ID: Final = "pseudo_perturbation_locality_v1"
CARD_SHA: Final = "e99aba9fe582499d9b4244281b99340a2150f37583cd438b0855b8afe2e7a613"
Q_VALUES: Final[tuple[int, ...]] = (24, 48, 72, 96, 192)
W_VALUES: Final[tuple[float, ...]] = (0.01, 0.03, 0.10)
SelectorName: TypeAlias = Literal["random", "full", "crossfit", "no_hessian"]
SELECTORS: Final[tuple[SelectorName, ...]] = (
    "random",
    "full",
    "crossfit",
    "no_hessian",
)
RANDOM_PURPOSE: Final = "locality_random_order_v1"
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _selector(value: object) -> SelectorName:
    if value not in SELECTORS:
        raise ValueError("unknown locality selector")
    return value


def _finite(value: object, *, name: str) -> float:
    if isinstance(value, (bool, str, bytes)):
        raise ValueError(f"{name} must be finite")
    try:
        number = float(cast(float, value))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be finite") from error
    if not np.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


@dataclass(frozen=True)
class LocalityOrdering:
    """One complete hidden-label-free candidate preference ordering."""

    selector: SelectorName
    ordered_indices: tuple[int, ...]
    ordered_hashes: tuple[str, ...]
    scores: np.ndarray | None

    def __post_init__(self) -> None:
        selector = _selector(self.selector)
        indices = tuple(self.ordered_indices)
        hashes = tuple(self.ordered_hashes)
        if not indices or any(type(index) is not int or index < 0 for index in indices):
            raise ValueError("ordered_indices must contain non-negative integers")
        if len(set(indices)) != len(indices):
            raise ValueError("ordered_indices must be unique")
        if len(hashes) != len(indices) or any(
            not isinstance(value, str) or not value for value in hashes
        ):
            raise ValueError("ordered_hashes must match ordered_indices")
        scores = self.scores
        if selector == "random":
            if scores is not None:
                raise ValueError("random ordering must not contain scores")
        else:
            scores = _float_array(scores, name="scores", ndim=1)
            if scores.size != len(indices):
                raise ValueError("ordering scores must match ordered_indices")
        object.__setattr__(self, "selector", selector)
        object.__setattr__(self, "ordered_indices", indices)
        object.__setattr__(self, "ordered_hashes", hashes)
        object.__setattr__(self, "scores", scores)


@dataclass(frozen=True)
class LocalityCellDiagnostics:
    """Finite-step fidelity diagnostics for one selector/count/weight cell."""

    stationarity_residual: float
    predicted_labeled_loss_change: float
    realized_labeled_loss_change: float
    labeled_loss_prediction_error: float
    labeled_loss_sign_agreement: bool
    displacement_cosine: float | None
    displacement_cosine_defined: bool
    displacement_relative_error: float
    locality_index: float

    def __post_init__(self) -> None:
        for field in (
            "stationarity_residual",
            "predicted_labeled_loss_change",
            "realized_labeled_loss_change",
            "labeled_loss_prediction_error",
            "displacement_relative_error",
            "locality_index",
        ):
            object.__setattr__(self, field, _finite(getattr(self, field), name=field))
        if type(self.labeled_loss_sign_agreement) is not bool:
            raise ValueError("labeled_loss_sign_agreement must be a boolean")
        if type(self.displacement_cosine_defined) is not bool:
            raise ValueError("displacement_cosine_defined must be a boolean")
        if self.displacement_cosine_defined:
            cosine = _finite(self.displacement_cosine, name="displacement_cosine")
            if not -1.0 <= cosine <= 1.0:
                raise ValueError("displacement_cosine must be in [-1, 1]")
            object.__setattr__(self, "displacement_cosine", cosine)
        elif self.displacement_cosine is not None:
            raise ValueError("undefined displacement cosine must be None")


@dataclass(frozen=True)
class LocalityCellArtifact:
    """One exact weighted-ridge fit on a selector-order prefix."""

    selector: SelectorName
    q: int
    pseudo_weight: float
    effective_pseudo_fraction: float
    selected_indices: tuple[int, ...]
    selected_hashes: tuple[str, ...]
    selected_pseudo_labels: np.ndarray
    training_weight_sum: float
    coefficients: np.ndarray
    test_predictions: np.ndarray
    diagnostics: LocalityCellDiagnostics

    def __post_init__(self) -> None:
        selector = _selector(self.selector)
        if type(self.q) is not int or self.q not in Q_VALUES:
            raise ValueError("q must be a frozen locality count")
        weight = _finite(self.pseudo_weight, name="pseudo_weight")
        if weight not in W_VALUES:
            raise ValueError("pseudo_weight must be a frozen locality weight")
        effective = _finite(
            self.effective_pseudo_fraction,
            name="effective_pseudo_fraction",
        )
        if not 0.0 < effective < 1.0:
            raise ValueError("effective_pseudo_fraction must be in (0, 1)")
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
        training_weight_sum = _finite(
            self.training_weight_sum,
            name="training_weight_sum",
        )
        if not isinstance(self.diagnostics, LocalityCellDiagnostics):
            raise ValueError("diagnostics must be LocalityCellDiagnostics")
        object.__setattr__(self, "selector", selector)
        object.__setattr__(self, "pseudo_weight", weight)
        object.__setattr__(self, "effective_pseudo_fraction", effective)
        object.__setattr__(self, "selected_indices", indices)
        object.__setattr__(self, "selected_hashes", hashes)
        object.__setattr__(self, "selected_pseudo_labels", labels)
        object.__setattr__(self, "training_weight_sum", training_weight_sum)
        object.__setattr__(self, "coefficients", coefficients)
        object.__setattr__(self, "test_predictions", predictions)


@dataclass(frozen=True)
class LocalityFitResult:
    """Canonical hidden-label-free state for all 60 locality cells."""

    base_fit: FitResult
    base_fit_digest: str
    card_id: str
    card_sha: str
    random_purpose: str
    q_values: tuple[int, ...]
    w_values: tuple[float, ...]
    selectors: tuple[SelectorName, ...]
    crossfit_scores: np.ndarray
    orderings: tuple[LocalityOrdering, ...]
    cells: tuple[LocalityCellArtifact, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.base_fit, FitResult):
            raise ValueError("base_fit must be FitResult")
        if not isinstance(self.base_fit_digest, str):
            raise ValueError("base_fit_digest must be a string")
        scores = _float_array(self.crossfit_scores, name="crossfit_scores", ndim=1)
        orderings = tuple(self.orderings)
        cells = tuple(self.cells)
        if any(not isinstance(item, LocalityOrdering) for item in orderings):
            raise ValueError("orderings must contain LocalityOrdering values")
        if any(not isinstance(item, LocalityCellArtifact) for item in cells):
            raise ValueError("cells must contain LocalityCellArtifact values")
        if tuple(item.selector for item in orderings) != SELECTORS:
            raise ValueError("orderings must match every frozen selector exactly once")
        expected_cell_keys = tuple(
            (selector, q, weight)
            for selector in SELECTORS
            for q in Q_VALUES
            for weight in W_VALUES
        )
        if (
            tuple((cell.selector, cell.q, cell.pseudo_weight) for cell in cells)
            != expected_cell_keys
        ):
            raise ValueError("cells must match the complete frozen factorial grid")
        if scores.size != self.base_fit.x_u.shape[0]:
            raise ValueError("crossfit_scores must match the unlabeled pool")
        object.__setattr__(self, "crossfit_scores", scores)
        object.__setattr__(self, "orderings", orderings)
        object.__setattr__(self, "cells", cells)

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
            "random_purpose": self.random_purpose,
            "q_values": list(self.q_values),
            "w_values": list(self.w_values),
            "selectors": list(self.selectors),
            "crossfit_scores": _array_identity(self.crossfit_scores),
            "orderings": [
                {
                    "selector": ordering.selector,
                    "ordered_indices": list(ordering.ordered_indices),
                    "ordered_hashes": list(ordering.ordered_hashes),
                    "scores": (
                        None
                        if ordering.scores is None
                        else _array_identity(ordering.scores)
                    ),
                }
                for ordering in self.orderings
            ],
            "cells": [
                {
                    "selector": cell.selector,
                    "q": cell.q,
                    "pseudo_weight": cell.pseudo_weight,
                    "effective_pseudo_fraction": cell.effective_pseudo_fraction,
                    "selected_indices": list(cell.selected_indices),
                    "selected_hashes": list(cell.selected_hashes),
                    "selected_pseudo_labels": _array_identity(
                        cell.selected_pseudo_labels
                    ),
                    "training_weight_sum": cell.training_weight_sum,
                    "coefficients": _array_identity(cell.coefficients),
                    "test_predictions": _array_identity(cell.test_predictions),
                    "diagnostics": dataclasses.asdict(cell.diagnostics),
                }
                for cell in self.cells
            ],
        }


def _cell_diagnostics(
    base_fit: FitResult,
    selected: np.ndarray,
    *,
    q: int,
    pseudo_weight: float,
    coefficients: np.ndarray,
) -> LocalityCellDiagnostics:
    theta_zero = base_fit.methods[0].coefficients
    x_selected = base_fit.x_u[selected]
    y_selected = base_fit.pseudo_labels_u[selected]
    gradient, hessian = labeled_gradient_hessian(
        base_fit.x_l,
        base_fit.y_l_standardized,
        theta_zero,
        base_fit.ridge_lambda,
    )
    selected_gradient = x_selected.T @ (x_selected @ theta_zero - y_selected) / q
    effective_fraction = pseudo_weight * q / (base_fit.x_l.shape[0] + pseudo_weight * q)
    predicted_displacement = -effective_fraction * np.linalg.solve(
        hessian,
        selected_gradient - gradient,
    )
    realized_displacement = coefficients - theta_zero
    cosine = _cosine_diagnostic(predicted_displacement, realized_displacement)
    predicted_loss = float(gradient @ predicted_displacement)
    realized_loss = float(
        squared_loss(base_fit.x_l, base_fit.y_l_standardized, coefficients)
        - squared_loss(base_fit.x_l, base_fit.y_l_standardized, theta_zero)
    )
    combined_x = np.concatenate([base_fit.x_l, x_selected], axis=0)
    combined_y = np.concatenate([base_fit.y_l_standardized, y_selected])
    weights = np.concatenate(
        [
            np.ones(base_fit.x_l.shape[0], dtype=np.float64),
            np.full(q, pseudo_weight, dtype=np.float64),
        ]
    )
    denominator = float(np.sum(weights))
    stationarity = (
        combined_x.T
        @ (weights * (combined_x @ coefficients - combined_y))
        / denominator
        + base_fit.ridge_lambda * coefficients
    )
    return LocalityCellDiagnostics(
        stationarity_residual=float(np.linalg.norm(stationarity)),
        predicted_labeled_loss_change=predicted_loss,
        realized_labeled_loss_change=realized_loss,
        labeled_loss_prediction_error=realized_loss - predicted_loss,
        labeled_loss_sign_agreement=bool(
            np.signbit(predicted_loss) == np.signbit(realized_loss)
        ),
        displacement_cosine=cosine.value,
        displacement_cosine_defined=cosine.defined,
        displacement_relative_error=float(
            np.linalg.norm(realized_displacement - predicted_displacement)
            / max(
                float(np.linalg.norm(realized_displacement)),
                float(np.finfo(np.float64).tiny),
            )
        ),
        locality_index=_locality_index(
            base_fit.x_l,
            x_selected,
            hessian,
            effective_fraction,
        ),
    )


def _fit_locality_task_single_thread(
    inputs: FitInputs,
    protocol: Protocol,
) -> LocalityFitResult:
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
        (
            "feature_scaling",
            protocol.preprocessing.feature_scaling,
            "scalar_rms",
        ),
        ("student_fit", protocol.preprocessing.student_fit, "no_intercept"),
        ("label_ddof", protocol.preprocessing.label_ddof, 0),
    )
    for name, actual, expected in frozen_values:
        if actual != expected:
            raise ValueError(f"protocol {name} does not match the locality card")
    if protocol.n_unlabeled < max(Q_VALUES):
        raise ValueError("Protocol n_unlabeled is smaller than the locality q grid")
    base_fit = fit_task(inputs, protocol)
    base_fit_digest = canonical_fit_digest(base_fit)
    theta_zero = base_fit.methods[0].coefficients
    crossfit_scores = cross_fitted_influence_scores(
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
    random_indices = np.random.Generator(
        np.random.PCG64(derive_seed(base_fit.assay_id, base_fit.seed, RANDOM_PURPOSE))
    ).permutation(base_fit.x_u.shape[0])
    score_arrays: dict[SelectorName, np.ndarray | None] = {
        "random": None,
        "full": base_fit.full_scores,
        "crossfit": crossfit_scores,
        "no_hessian": base_fit.no_hessian_scores,
    }
    ordered: dict[SelectorName, tuple[int, ...]] = {
        "random": tuple(int(index) for index in random_indices),
        "full": tuple(
            int(index)
            for index in stable_top_k(
                base_fit.full_scores,
                base_fit.unlabeled_hashes,
                base_fit.x_u.shape[0],
            )
        ),
        "crossfit": tuple(
            int(index)
            for index in stable_top_k(
                crossfit_scores,
                base_fit.unlabeled_hashes,
                base_fit.x_u.shape[0],
            )
        ),
        "no_hessian": tuple(
            int(index)
            for index in stable_top_k(
                base_fit.no_hessian_scores,
                base_fit.unlabeled_hashes,
                base_fit.x_u.shape[0],
            )
        ),
    }
    orderings = tuple(
        LocalityOrdering(
            selector=selector,
            ordered_indices=ordered[selector],
            ordered_hashes=tuple(
                base_fit.unlabeled_hashes[index] for index in ordered[selector]
            ),
            scores=score_arrays[selector],
        )
        for selector in SELECTORS
    )
    cells: list[LocalityCellArtifact] = []
    for selector in SELECTORS:
        for q in Q_VALUES:
            selected_indices = ordered[selector][:q]
            selected = np.asarray(selected_indices, dtype=np.int64)
            selected_labels = base_fit.pseudo_labels_u[selected]
            combined_x = np.concatenate(
                [base_fit.x_l, base_fit.x_u[selected]],
                axis=0,
            )
            combined_y = np.concatenate([base_fit.y_l_standardized, selected_labels])
            for pseudo_weight in W_VALUES:
                weights = np.concatenate(
                    [
                        np.ones(base_fit.x_l.shape[0], dtype=np.float64),
                        np.full(q, pseudo_weight, dtype=np.float64),
                    ]
                )
                coefficients = fit_weighted_ridge(
                    combined_x,
                    combined_y,
                    base_fit.ridge_lambda,
                    sample_weight=weights,
                )
                predictions = np.asarray(
                    base_fit.x_test @ coefficients,
                    dtype=np.float64,
                )
                effective_fraction = (
                    pseudo_weight * q / (base_fit.x_l.shape[0] + pseudo_weight * q)
                )
                cells.append(
                    LocalityCellArtifact(
                        selector=selector,
                        q=q,
                        pseudo_weight=pseudo_weight,
                        effective_pseudo_fraction=effective_fraction,
                        selected_indices=selected_indices,
                        selected_hashes=tuple(
                            base_fit.unlabeled_hashes[index]
                            for index in selected_indices
                        ),
                        selected_pseudo_labels=selected_labels,
                        training_weight_sum=float(np.sum(weights)),
                        coefficients=coefficients,
                        test_predictions=predictions,
                        diagnostics=_cell_diagnostics(
                            base_fit,
                            selected,
                            q=q,
                            pseudo_weight=pseudo_weight,
                            coefficients=coefficients,
                        ),
                    )
                )
    return LocalityFitResult(
        base_fit=base_fit,
        base_fit_digest=base_fit_digest,
        card_id=CARD_ID,
        card_sha=CARD_SHA,
        random_purpose=RANDOM_PURPOSE,
        q_values=Q_VALUES,
        w_values=W_VALUES,
        selectors=SELECTORS,
        crossfit_scores=crossfit_scores,
        orderings=orderings,
        cells=tuple(cells),
    )


def fit_locality_task(inputs: FitInputs, protocol: Protocol) -> LocalityFitResult:
    """Fit the frozen locality grid before unblinding."""
    with _locked_blas_scope():
        return _fit_locality_task_single_thread(inputs, protocol)


def canonical_locality_fit_digest(result: object) -> str:
    """Validate and hash one hidden-label-free locality fit."""
    if not isinstance(result, LocalityFitResult):
        raise ValueError("result must be LocalityFitResult")
    with _locked_blas_scope():
        if (
            result.card_id != CARD_ID
            or result.card_sha != CARD_SHA
            or result.random_purpose != RANDOM_PURPOSE
        ):
            raise ValueError("locality card identity mismatch")
        if (
            result.q_values != Q_VALUES
            or result.w_values != W_VALUES
            or result.selectors != SELECTORS
        ):
            raise ValueError("locality factorial grid mismatch")
        actual_base_digest = canonical_fit_digest(result.base_fit)
        if result.base_fit_digest != actual_base_digest:
            raise ValueError("locality base fit digest mismatch")
        base_fit = result.base_fit
        if base_fit.x_u.shape[0] < max(Q_VALUES):
            raise ValueError("locality base fit cannot support the q grid")
        theta_zero = base_fit.methods[0].coefficients
        expected_crossfit_scores = cross_fitted_influence_scores(
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
        if not _scaled_algebra_close(
            result.crossfit_scores,
            expected_crossfit_scores,
        ):
            raise ValueError("locality crossfit scores mismatch")
        random_indices = np.random.Generator(
            np.random.PCG64(
                derive_seed(base_fit.assay_id, base_fit.seed, RANDOM_PURPOSE)
            )
        ).permutation(base_fit.x_u.shape[0])
        expected_scores: dict[SelectorName, np.ndarray | None] = {
            "random": None,
            "full": base_fit.full_scores,
            "crossfit": expected_crossfit_scores,
            "no_hessian": base_fit.no_hessian_scores,
        }
        expected_ordered: dict[SelectorName, tuple[int, ...]] = {
            "random": tuple(int(index) for index in random_indices),
            "full": tuple(
                int(index)
                for index in stable_top_k(
                    base_fit.full_scores,
                    base_fit.unlabeled_hashes,
                    base_fit.x_u.shape[0],
                )
            ),
            "crossfit": tuple(
                int(index)
                for index in stable_top_k(
                    expected_crossfit_scores,
                    base_fit.unlabeled_hashes,
                    base_fit.x_u.shape[0],
                )
            ),
            "no_hessian": tuple(
                int(index)
                for index in stable_top_k(
                    base_fit.no_hessian_scores,
                    base_fit.unlabeled_hashes,
                    base_fit.x_u.shape[0],
                )
            ),
        }
        if len(result.orderings) != len(SELECTORS):
            raise ValueError("locality ordering count mismatch")
        for ordering, selector in zip(result.orderings, SELECTORS, strict=True):
            if ordering.selector != selector:
                raise ValueError("locality ordering selector mismatch")
            indices = expected_ordered[selector]
            hashes = tuple(base_fit.unlabeled_hashes[index] for index in indices)
            if ordering.ordered_indices != indices or ordering.ordered_hashes != hashes:
                raise ValueError("locality complete ordering mismatch")
            scores = expected_scores[selector]
            if scores is None:
                if ordering.scores is not None:
                    raise ValueError("random locality ordering contains scores")
            elif ordering.scores is None or not _scaled_algebra_close(
                ordering.scores,
                scores,
            ):
                raise ValueError("locality ordering scores mismatch")

        expected_keys = tuple(
            (selector, q, weight)
            for selector in SELECTORS
            for q in Q_VALUES
            for weight in W_VALUES
        )
        actual_keys = tuple(
            (cell.selector, cell.q, cell.pseudo_weight) for cell in result.cells
        )
        if actual_keys != expected_keys:
            raise ValueError("locality cell grid mismatch")
        for cell in result.cells:
            indices = expected_ordered[cell.selector][: cell.q]
            if cell.selected_indices != indices:
                raise ValueError("locality cell is not an ordering prefix")
            hashes = tuple(base_fit.unlabeled_hashes[index] for index in indices)
            if cell.selected_hashes != hashes:
                raise ValueError("locality cell selected hashes mismatch")
            selected = np.asarray(indices, dtype=np.int64)
            selected_labels = base_fit.pseudo_labels_u[selected]
            if not np.array_equal(cell.selected_pseudo_labels, selected_labels):
                raise ValueError("locality cell selected pseudo-labels mismatch")
            combined_x = np.concatenate(
                [base_fit.x_l, base_fit.x_u[selected]],
                axis=0,
            )
            combined_y = np.concatenate([base_fit.y_l_standardized, selected_labels])
            weights = np.concatenate(
                [
                    np.ones(base_fit.x_l.shape[0], dtype=np.float64),
                    np.full(cell.q, cell.pseudo_weight, dtype=np.float64),
                ]
            )
            expected_training_weight_sum = float(np.sum(weights))
            expected_effective_fraction = (
                cell.pseudo_weight
                * cell.q
                / (base_fit.x_l.shape[0] + cell.pseudo_weight * cell.q)
            )
            if (
                cell.training_weight_sum != expected_training_weight_sum
                or cell.effective_pseudo_fraction != expected_effective_fraction
            ):
                raise ValueError("locality cell weight normalization mismatch")
            expected_coefficients = fit_weighted_ridge(
                combined_x,
                combined_y,
                base_fit.ridge_lambda,
                sample_weight=weights,
            )
            if not _scaled_algebra_close(cell.coefficients, expected_coefficients):
                raise ValueError("locality cell coefficients mismatch")
            expected_predictions = base_fit.x_test @ expected_coefficients
            if not _scaled_algebra_close(
                cell.test_predictions,
                expected_predictions,
            ):
                raise ValueError("locality cell test predictions mismatch")
            expected_diagnostics = _cell_diagnostics(
                base_fit,
                selected,
                q=cell.q,
                pseudo_weight=cell.pseudo_weight,
                coefficients=expected_coefficients,
            )
            if not _diagnostic_values_close(
                dataclasses.asdict(cell.diagnostics),
                dataclasses.asdict(expected_diagnostics),
            ):
                raise ValueError("locality cell diagnostics mismatch")
    return _canonical_payload_digest(result.to_payload())


@dataclass(frozen=True)
class LocalityCellEvaluation:
    """Hidden-label evaluation and same-functional finite-step diagnostics."""

    selector: SelectorName
    q: int
    pseudo_weight: float
    effective_pseudo_fraction: float
    spearman: float
    mse: float
    ndcg_10pct: float
    selected_pseudo_label_mae: float
    predicted_labeled_loss_change: float
    realized_labeled_loss_change: float
    labeled_loss_prediction_error: float
    labeled_loss_sign_agreement: bool
    predicted_test_loss_change: float
    realized_test_loss_change: float
    test_loss_prediction_error: float
    test_loss_sign_agreement: bool
    displacement_cosine: float | None
    displacement_cosine_defined: bool
    displacement_relative_error: float
    locality_index: float
    test_oracle_score_alignment: CorrelationDiagnostic
    test_oracle_score_vs_absolute_error: CorrelationDiagnostic

    def __post_init__(self) -> None:
        _selector(self.selector)
        if type(self.q) is not int or self.q not in Q_VALUES:
            raise ValueError("q must be a frozen locality count")
        if self.pseudo_weight not in W_VALUES:
            raise ValueError("pseudo_weight must be a frozen locality weight")
        for field in (
            "effective_pseudo_fraction",
            "spearman",
            "mse",
            "ndcg_10pct",
            "selected_pseudo_label_mae",
            "predicted_labeled_loss_change",
            "realized_labeled_loss_change",
            "labeled_loss_prediction_error",
            "predicted_test_loss_change",
            "realized_test_loss_change",
            "test_loss_prediction_error",
            "displacement_relative_error",
            "locality_index",
        ):
            object.__setattr__(self, field, _finite(getattr(self, field), name=field))
        for field in (
            "labeled_loss_sign_agreement",
            "test_loss_sign_agreement",
            "displacement_cosine_defined",
        ):
            if type(getattr(self, field)) is not bool:
                raise ValueError(f"{field} must be a boolean")
        if self.displacement_cosine_defined:
            cosine = _finite(self.displacement_cosine, name="displacement_cosine")
            if not -1.0 <= cosine <= 1.0:
                raise ValueError("displacement_cosine must be in [-1, 1]")
            object.__setattr__(self, "displacement_cosine", cosine)
        elif self.displacement_cosine is not None:
            raise ValueError("undefined displacement cosine must be None")
        for field in (
            "test_oracle_score_alignment",
            "test_oracle_score_vs_absolute_error",
        ):
            if not isinstance(getattr(self, field), CorrelationDiagnostic):
                raise ValueError(f"{field} must be CorrelationDiagnostic")


@dataclass(frozen=True)
class LocalityEvaluationResult:
    """Post-digest hidden-label evaluation of all locality cells."""

    reference: EvaluationResult
    cells: tuple[LocalityCellEvaluation, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.reference, EvaluationResult):
            raise ValueError("reference must be EvaluationResult")
        cells = tuple(self.cells)
        if any(not isinstance(cell, LocalityCellEvaluation) for cell in cells):
            raise ValueError("cells must contain LocalityCellEvaluation values")
        expected_cell_keys = tuple(
            (selector, q, weight)
            for selector in SELECTORS
            for q in Q_VALUES
            for weight in W_VALUES
        )
        if (
            tuple((cell.selector, cell.q, cell.pseudo_weight) for cell in cells)
            != expected_cell_keys
        ):
            raise ValueError("cells must match the complete frozen factorial grid")
        object.__setattr__(self, "cells", cells)

    @property
    def assay_id(self) -> str:
        return self.reference.assay_id

    @property
    def seed(self) -> int:
        return self.reference.seed

    def to_payload(self) -> dict[str, object]:
        """Return JSON-compatible post-fit metrics and diagnostics."""
        return {
            "assay_id": self.assay_id,
            "seed": self.seed,
            "reference": dataclasses.asdict(self.reference),
            "cells": [dataclasses.asdict(cell) for cell in self.cells],
        }


def _preference_values(ordering: LocalityOrdering) -> np.ndarray:
    if ordering.scores is not None:
        return ordering.scores
    preference = np.empty(len(ordering.ordered_indices), dtype=np.float64)
    preference[np.asarray(ordering.ordered_indices, dtype=np.int64)] = np.arange(
        len(ordering.ordered_indices),
        0,
        -1,
        dtype=np.float64,
    )
    return preference


def _evaluate_locality_task_single_thread(
    fit: LocalityFitResult,
    labels: EvaluationLabels,
    *,
    protocol: Protocol,
    expected_fit_digest: str,
    expected_evaluation_digest: str,
) -> LocalityEvaluationResult:
    if not isinstance(fit, LocalityFitResult):
        raise ValueError("fit must be LocalityFitResult")
    if not isinstance(expected_fit_digest, str) or not _SHA256_PATTERN.fullmatch(
        expected_fit_digest
    ):
        raise ValueError("expected_fit_digest must be a SHA-256 digest")
    actual_fit_digest = canonical_locality_fit_digest(fit)
    if actual_fit_digest != expected_fit_digest:
        raise ValueError(
            "current locality fit digest does not match expected_fit_digest"
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
    theta_zero = fit.base_fit.methods[0].coefficients
    gradient_l, hessian = labeled_gradient_hessian(
        fit.base_fit.x_l,
        fit.base_fit.y_l_standardized,
        theta_zero,
        fit.base_fit.ridge_lambda,
    )
    gradient_test = (
        fit.base_fit.x_test.T
        @ (fit.base_fit.x_test @ theta_zero - y_test)
        / y_test.size
    )
    candidate_difference = (
        fit.base_fit.x_u @ theta_zero - fit.base_fit.pseudo_labels_u
    )[:, None] * fit.base_fit.x_u - gradient_l
    damped_hessian = hessian + fit.base_fit.damping * np.eye(hessian.shape[0])
    full_oracle_scores = candidate_difference @ np.linalg.solve(
        damped_hessian,
        gradient_test,
    )
    no_hessian_oracle_scores = candidate_difference @ gradient_test
    ordering_by_selector = {ordering.selector: ordering for ordering in fit.orderings}
    oracle_alignment: dict[
        SelectorName,
        tuple[CorrelationDiagnostic, CorrelationDiagnostic],
    ] = {}
    for selector in SELECTORS:
        preference = _preference_values(ordering_by_selector[selector])
        oracle_scores = (
            no_hessian_oracle_scores if selector == "no_hessian" else full_oracle_scores
        )
        oracle_alignment[selector] = (
            _correlation(oracle_scores, preference),
            _correlation(absolute_errors, preference),
        )

    evaluations: list[LocalityCellEvaluation] = []
    base_test_loss = squared_loss(fit.base_fit.x_test, y_test, theta_zero)
    for cell in fit.cells:
        selected = np.asarray(cell.selected_indices, dtype=np.int64)
        x_selected = fit.base_fit.x_u[selected]
        selected_gradient = (
            x_selected.T
            @ (x_selected @ theta_zero - fit.base_fit.pseudo_labels_u[selected])
            / cell.q
        )
        predicted_displacement = -cell.effective_pseudo_fraction * np.linalg.solve(
            hessian,
            selected_gradient - gradient_l,
        )
        predicted_test_loss = float(gradient_test @ predicted_displacement)
        realized_test_loss = float(
            squared_loss(
                fit.base_fit.x_test,
                y_test,
                cell.coefficients,
            )
            - base_test_loss
        )
        alignment, score_vs_error = oracle_alignment[cell.selector]
        diagnostics = cell.diagnostics
        evaluations.append(
            LocalityCellEvaluation(
                selector=cell.selector,
                q=cell.q,
                pseudo_weight=cell.pseudo_weight,
                effective_pseudo_fraction=cell.effective_pseudo_fraction,
                spearman=spearman_correlation(y_test, cell.test_predictions),
                mse=standardized_mse(y_test, cell.test_predictions),
                ndcg_10pct=ndcg_at_10_percent(y_test, cell.test_predictions),
                selected_pseudo_label_mae=float(np.mean(absolute_errors[selected])),
                predicted_labeled_loss_change=(
                    diagnostics.predicted_labeled_loss_change
                ),
                realized_labeled_loss_change=(diagnostics.realized_labeled_loss_change),
                labeled_loss_prediction_error=(
                    diagnostics.labeled_loss_prediction_error
                ),
                labeled_loss_sign_agreement=(diagnostics.labeled_loss_sign_agreement),
                predicted_test_loss_change=predicted_test_loss,
                realized_test_loss_change=realized_test_loss,
                test_loss_prediction_error=realized_test_loss - predicted_test_loss,
                test_loss_sign_agreement=bool(
                    np.signbit(predicted_test_loss) == np.signbit(realized_test_loss)
                ),
                displacement_cosine=diagnostics.displacement_cosine,
                displacement_cosine_defined=(diagnostics.displacement_cosine_defined),
                displacement_relative_error=(diagnostics.displacement_relative_error),
                locality_index=diagnostics.locality_index,
                test_oracle_score_alignment=alignment,
                test_oracle_score_vs_absolute_error=score_vs_error,
            )
        )
    return LocalityEvaluationResult(reference=reference, cells=tuple(evaluations))


def evaluate_locality_task(
    fit: LocalityFitResult,
    labels: EvaluationLabels,
    *,
    protocol: Protocol,
    expected_fit_digest: str,
    expected_evaluation_digest: str,
) -> LocalityEvaluationResult:
    """Evaluate a digest-frozen locality fit against hidden outcomes."""
    with _locked_blas_scope():
        return _evaluate_locality_task_single_thread(
            fit,
            labels,
            protocol=protocol,
            expected_fit_digest=expected_fit_digest,
            expected_evaluation_digest=expected_evaluation_digest,
        )
