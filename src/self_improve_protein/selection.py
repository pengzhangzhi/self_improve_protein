"""Hidden-label-safe influence and baseline selection policies."""

from collections.abc import Sequence
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from self_improve_protein.provenance import derive_seed
from self_improve_protein.ridge import (
    _as_float64_matrix,
    _as_float64_vector,
    _validated_ridge_lambda,
    fit_weighted_ridge,
)

FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int64]


def _validated_score_inputs(
    x_l: FloatArray,
    y_l: FloatArray,
    x_u: FloatArray,
    yhat_u: FloatArray,
    theta: FloatArray,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]:
    """Validate and normalize the arrays shared by both score functions."""
    labeled_features = _as_float64_matrix(x_l, name="x_l")
    labeled_response = _as_float64_vector(y_l, name="y_l")
    unlabeled_features = _as_float64_matrix(x_u, name="x_u")
    pseudo_response = _as_float64_vector(yhat_u, name="yhat_u")
    parameters = _as_float64_vector(theta, name="theta")

    if labeled_features.shape[0] != labeled_response.shape[0]:
        raise ValueError("x_l and y_l must have the same number of rows")
    if unlabeled_features.shape[0] != pseudo_response.shape[0]:
        raise ValueError("x_u and yhat_u must have the same number of rows")
    if labeled_features.shape[1] != unlabeled_features.shape[1]:
        raise ValueError("x_l and x_u must have the same feature width")
    if parameters.shape[0] != labeled_features.shape[1]:
        raise ValueError("theta length must match the feature width")
    return (
        labeled_features,
        labeled_response,
        unlabeled_features,
        pseudo_response,
        parameters,
    )


def _validated_damping(damping: float) -> float:
    """Return a strict finite, non-negative damping coefficient."""
    if isinstance(damping, (bool, str, bytes)):
        raise ValueError("damping must be a finite non-negative number")
    try:
        value = float(damping)
    except (TypeError, ValueError) as error:
        raise ValueError("damping must be a finite non-negative number") from error
    if not np.isfinite(value) or value < 0.0:
        raise ValueError("damping must be a finite non-negative number")
    return value


def _labeled_gradient(
    x_l: FloatArray,
    y_l: FloatArray,
    theta: FloatArray,
) -> FloatArray:
    """Compute the unregularized labeled loss gradient."""
    sample_count = x_l.shape[0]
    gradient = x_l.T @ (x_l @ theta - y_l) / sample_count
    if not np.all(np.isfinite(gradient)):
        raise ValueError("labeled gradient must be finite")
    return np.asarray(gradient, dtype=np.float64)


def _regularized_hessian(x_l: FloatArray, ridge_lambda: float) -> FloatArray:
    """Compute the regularized labeled Hessian."""
    hessian = x_l.T @ x_l / x_l.shape[0] + ridge_lambda * np.eye(
        x_l.shape[1], dtype=np.float64
    )
    if not np.all(np.isfinite(hessian)):
        raise ValueError("labeled Hessian must be finite")
    return np.asarray(hessian, dtype=np.float64)


def influence_scores(
    x_l: FloatArray,
    y_l: FloatArray,
    x_u: FloatArray,
    yhat_u: FloatArray,
    theta: FloatArray,
    ridge_lambda: float,
    damping: float,
) -> FloatArray:
    """Return full inverse-Hessian influence scores using one linear solve.

    With zero damping, a positive score is the negative derivative at zero of
    unregularized labeled loss under infinitesimal pseudo-sample upweighting.
    Simultaneously scaling all features by ``c``, parameters by ``1 / c``, and
    both regularizers by ``c**2`` leaves these scores unchanged.
    """
    regularization = _validated_ridge_lambda(ridge_lambda)
    damping_value = _validated_damping(damping)
    (
        labeled_features,
        labeled_response,
        unlabeled_features,
        pseudo_response,
        parameters,
    ) = _validated_score_inputs(x_l, y_l, x_u, yhat_u, theta)
    gradient = _labeled_gradient(
        labeled_features,
        labeled_response,
        parameters,
    )
    hessian = _regularized_hessian(labeled_features, regularization)
    with np.errstate(over="ignore"):
        system = hessian + damping_value * np.eye(
            labeled_features.shape[1],
            dtype=np.float64,
        )
    if not np.all(np.isfinite(system)):
        raise ValueError("damped Hessian system must be finite")
    try:
        direction = np.linalg.solve(system, gradient)
    except np.linalg.LinAlgError as error:
        raise np.linalg.LinAlgError("damped Hessian system is singular") from error
    residual = unlabeled_features @ parameters - pseudo_response
    scores = residual * (unlabeled_features @ direction) - gradient @ direction
    if not np.all(np.isfinite(scores)):
        raise ValueError("influence scores must be finite")
    return np.asarray(scores, dtype=np.float64)


def no_hessian_scores(
    x_l: FloatArray,
    y_l: FloatArray,
    x_u: FloatArray,
    yhat_u: FloatArray,
    theta: FloatArray,
    ridge_lambda: float,
) -> FloatArray:
    """Return the identity-geometry ablation ``g_L.T @ (g_j - g_L)``.

    Under common feature scaling by ``c`` and parameter scaling by ``1 / c``,
    these scores scale by the positive factor ``c**2`` and preserve ordering.
    ``ridge_lambda`` is validated for API consistency with the full selector.
    """
    _validated_ridge_lambda(ridge_lambda)
    (
        labeled_features,
        labeled_response,
        unlabeled_features,
        pseudo_response,
        parameters,
    ) = _validated_score_inputs(x_l, y_l, x_u, yhat_u, theta)
    gradient = _labeled_gradient(
        labeled_features,
        labeled_response,
        parameters,
    )
    residual = unlabeled_features @ parameters - pseudo_response
    scores = residual * (unlabeled_features @ gradient) - gradient @ gradient
    if not np.all(np.isfinite(scores)):
        raise ValueError("no-Hessian scores must be finite")
    return np.asarray(scores, dtype=np.float64)


def _strict_integer(value: int, *, name: str) -> int:
    """Return a Python integer while rejecting bools and coercible values."""
    if type(value) is not int:
        raise ValueError(f"{name} must be an integer")
    return value


def _nonempty_string(value: str, *, name: str) -> str:
    """Return a non-empty string without coercing identifiers."""
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def balanced_fold_assignment(
    sample_count: int,
    assay_id: str,
    seed: int,
    *,
    fold_count: int = 4,
    purpose: str = "crossfit_outer_folds_v1",
) -> IntArray:
    """Assign samples to exactly balanced, purpose-separated PCG64 folds."""
    size = _strict_integer(sample_count, name="sample_count")
    folds = _strict_integer(fold_count, name="fold_count")
    seed_value = _strict_integer(seed, name="seed")
    assay = _nonempty_string(assay_id, name="assay_id")
    stream_purpose = _nonempty_string(purpose, name="purpose")
    if size <= 0:
        raise ValueError("sample_count must be positive")
    if folds < 2 or folds > size:
        raise ValueError("fold_count must satisfy 2 <= fold_count <= sample_count")
    if size % folds != 0:
        raise ValueError("sample_count must be divisible by fold_count")
    if seed_value < 0:
        raise ValueError("seed must be non-negative")

    generator = np.random.Generator(
        np.random.PCG64(derive_seed(assay, seed_value, stream_purpose))
    )
    permutation = generator.permutation(size)
    assignment = np.empty(size, dtype=np.int64)
    assignment[permutation] = np.repeat(
        np.arange(folds, dtype=np.int64),
        size // folds,
    )
    return assignment


def _validated_fold_assignment(
    fold_assignment: IntArray,
    *,
    sample_count: int,
) -> IntArray:
    """Validate contiguous, exactly balanced integer fold identifiers."""
    try:
        raw = np.asarray(fold_assignment)
    except (TypeError, ValueError) as error:
        raise ValueError("fold_assignment must be an integer 1D array") from error
    if raw.ndim != 1:
        raise ValueError("fold_assignment must be a 1D array")
    if raw.shape[0] != sample_count:
        raise ValueError("fold_assignment length must match x_l rows")
    if raw.dtype.kind not in {"i", "u"}:
        raise ValueError("fold_assignment must contain integers")
    if raw.dtype.kind == "u" and np.any(raw > np.iinfo(np.int64).max):
        raise ValueError("fold_assignment values exceed int64")
    assignment = np.asarray(raw, dtype=np.int64)
    unique, counts = np.unique(assignment, return_counts=True)
    if unique.size < 2 or not np.array_equal(
        unique,
        np.arange(unique.size, dtype=np.int64),
    ):
        raise ValueError("fold_assignment must use contiguous IDs starting at zero")
    if np.any(counts != counts[0]):
        raise ValueError("fold_assignment must be exactly balanced")
    return assignment


def out_of_fold_ridge_gradient(
    x_l: FloatArray,
    y_l: FloatArray,
    fold_assignment: IntArray,
    ridge_lambda: float,
) -> FloatArray:
    """Return the mean labeled gradient from held-out ridge residuals."""
    labeled_features = _as_float64_matrix(x_l, name="x_l")
    labeled_response = _as_float64_vector(y_l, name="y_l")
    if labeled_features.shape[0] != labeled_response.shape[0]:
        raise ValueError("x_l and y_l must have the same number of rows")
    regularization = _validated_ridge_lambda(ridge_lambda)
    assignment = _validated_fold_assignment(
        fold_assignment,
        sample_count=labeled_features.shape[0],
    )
    residuals = np.empty(labeled_features.shape[0], dtype=np.float64)
    for fold in range(int(np.max(assignment)) + 1):
        held_out = assignment == fold
        theta_minus_fold = fit_weighted_ridge(
            labeled_features[~held_out],
            labeled_response[~held_out],
            regularization,
        )
        with np.errstate(over="ignore", invalid="ignore"):
            residuals[held_out] = (
                labeled_features[held_out] @ theta_minus_fold
                - labeled_response[held_out]
            )
    with np.errstate(over="ignore", invalid="ignore"):
        gradient = labeled_features.T @ residuals / labeled_features.shape[0]
    if not np.all(np.isfinite(residuals)) or not np.all(np.isfinite(gradient)):
        raise ValueError("out-of-fold residual gradient must be finite")
    return np.asarray(gradient, dtype=np.float64)


def cross_fitted_influence_scores(
    x_l: FloatArray,
    y_l: FloatArray,
    x_u: FloatArray,
    yhat_u: FloatArray,
    theta: FloatArray,
    ridge_lambda: float,
    damping: float,
    assay_id: str,
    seed: int,
    *,
    fold_count: int = 4,
    purpose: str = "crossfit_outer_folds_v1",
) -> FloatArray:
    """Score full-fit pseudo-gradients using a cross-fitted outer gradient."""
    regularization = _validated_ridge_lambda(ridge_lambda)
    damping_value = _validated_damping(damping)
    (
        labeled_features,
        labeled_response,
        unlabeled_features,
        pseudo_response,
        parameters,
    ) = _validated_score_inputs(x_l, y_l, x_u, yhat_u, theta)
    assignment = balanced_fold_assignment(
        labeled_features.shape[0],
        assay_id,
        seed,
        fold_count=fold_count,
        purpose=purpose,
    )
    cross_fitted_gradient = out_of_fold_ridge_gradient(
        labeled_features,
        labeled_response,
        assignment,
        regularization,
    )
    full_gradient = _labeled_gradient(
        labeled_features,
        labeled_response,
        parameters,
    )
    hessian = _regularized_hessian(labeled_features, regularization)
    with np.errstate(over="ignore"):
        system = hessian + damping_value * np.eye(
            labeled_features.shape[1],
            dtype=np.float64,
        )
    if not np.all(np.isfinite(system)):
        raise ValueError("damped Hessian system must be finite")
    try:
        direction = np.linalg.solve(system, cross_fitted_gradient)
    except np.linalg.LinAlgError as error:
        raise np.linalg.LinAlgError("damped Hessian system is singular") from error
    residual = unlabeled_features @ parameters - pseudo_response
    scores = residual * (unlabeled_features @ direction) - full_gradient @ direction
    if not np.all(np.isfinite(scores)):
        raise ValueError("cross-fitted influence scores must be finite")
    return np.asarray(scores, dtype=np.float64)


def stable_top_k(
    scores: FloatArray,
    stable_hashes: Sequence[str],
    k: int,
    largest: bool = True,
) -> IntArray:
    """Select by score, breaking exact ties by ascending stable hash."""
    score_array = _as_float64_vector(scores, name="scores")
    if isinstance(stable_hashes, (str, bytes)):
        raise ValueError("stable_hashes must be a sequence of strings")
    try:
        hashes = tuple(stable_hashes)
    except TypeError as error:
        raise ValueError("stable_hashes must be a sequence of strings") from error
    if len(hashes) != score_array.shape[0]:
        raise ValueError("stable_hashes length must match scores length")
    if any(not isinstance(stable_hash, str) for stable_hash in hashes):
        raise ValueError("stable_hashes must contain only strings")
    if len(set(hashes)) != len(hashes):
        raise ValueError("stable_hashes must be unique")
    count = _strict_integer(k, name="k")
    if not 0 < count <= score_array.shape[0]:
        raise ValueError("k must satisfy 0 < k <= number of scores")
    if type(largest) is not bool:
        raise ValueError("largest must be a boolean")

    primary_key = -score_array if largest else score_array
    hash_key = np.asarray(hashes, dtype=np.str_)
    order = np.lexsort((hash_key, primary_key))
    return np.asarray(order[:count], dtype=np.int64)


def random_indices(pool_size: int, k: int, seed: int) -> IntArray:
    """Return sorted, unique PCG64 sample indices without replacement."""
    size = _strict_integer(pool_size, name="pool_size")
    count = _strict_integer(k, name="k")
    seed_value = _strict_integer(seed, name="seed")
    if size < 0:
        raise ValueError("pool_size must be non-negative")
    if count < 0:
        raise ValueError("k must be non-negative")
    if seed_value < 0:
        raise ValueError("seed must be non-negative")
    if not 0 < count <= size:
        raise ValueError("k must satisfy 0 < k <= pool_size")

    generator = np.random.Generator(np.random.PCG64(seed_value))
    indices = generator.choice(size, size=count, replace=False)
    return np.asarray(np.sort(indices), dtype=np.int64)


def top_teacher_indices(
    yhat: FloatArray,
    hashes: Sequence[str],
    k: int,
) -> IntArray:
    """Select the largest teacher predictions with stable hash tie-breaking."""
    return stable_top_k(yhat, hashes, k, largest=True)
