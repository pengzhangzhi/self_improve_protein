"""Hidden-label-safe influence and baseline selection policies."""

from collections.abc import Sequence
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from self_improve_protein.ridge import (
    _as_float64_matrix,
    _as_float64_vector,
    _validated_ridge_lambda,
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
    system = hessian + damping_value * np.eye(
        labeled_features.shape[1],
        dtype=np.float64,
    )
    direction = np.linalg.solve(system, gradient)
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
