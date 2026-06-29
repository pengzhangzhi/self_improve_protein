"""Strict regression metrics matching the locked ProteinGym protocol."""

from typing import TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import spearmanr  # type: ignore[import-untyped]

FloatArray: TypeAlias = NDArray[np.float64]


def _validated_pair(
    y_true: ArrayLike,
    y_pred: ArrayLike,
) -> tuple[FloatArray, FloatArray]:
    """Return finite, non-empty, matched one-dimensional float64 vectors."""
    arrays: list[FloatArray] = []
    for value, name in ((y_true, "y_true"), (y_pred, "y_pred")):
        try:
            raw_array = np.asarray(value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{name} must be a numeric real 1D array") from error
        is_real_numeric = np.issubdtype(
            raw_array.dtype,
            np.integer,
        ) or np.issubdtype(raw_array.dtype, np.floating)
        if not is_real_numeric:
            raise ValueError(f"{name} must be a numeric real 1D array")
        array = np.asarray(raw_array, dtype=np.float64)
        if array.ndim != 1:
            raise ValueError(f"{name} must be a 1D array")
        if array.size == 0:
            raise ValueError(f"{name} must be non-empty")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must contain only finite values")
        arrays.append(np.asarray(array, dtype=np.float64))
    truth, prediction = arrays
    if truth.size != prediction.size:
        raise ValueError("y_true and y_pred must have the same length")
    return truth, prediction


def spearman_correlation(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Return Spearman correlation, rejecting inputs where it is undefined."""
    truth, prediction = _validated_pair(y_true, y_pred)
    if np.all(truth == truth[0]) or np.all(prediction == prediction[0]):
        raise ValueError("Spearman correlation is undefined for constant input")
    statistic = float(spearmanr(truth, prediction).statistic)
    if not np.isfinite(statistic):
        raise ValueError("Spearman correlation is undefined for these inputs")
    return statistic


def standardized_mse(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Return mean squared error in labeled-standardized response space."""
    truth, prediction = _validated_pair(y_true, y_pred)
    with np.errstate(over="ignore", invalid="ignore"):
        result = float(np.mean(np.square(truth - prediction)))
    if not np.isfinite(result):
        raise ValueError("MSE result must be finite")
    return result


def ndcg_at_10_percent(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Return ProteinGym-style continuous-gain NDCG over the top decile.

    The locked behavior follows ProteinGym v1.3's continuous min-max gains and
    floor-sized top decile, while making tied predicted ranks deterministic via
    stable input order.
    """
    truth, prediction = _validated_pair(y_true, y_pred)
    k = int(np.floor(0.10 * truth.size))
    if k < 1:
        raise ValueError("top 10% must contain at least one observation")

    truth_min = float(np.min(truth))
    truth_max = float(np.max(truth))
    if truth_max == truth_min:
        return 0.0
    with np.errstate(over="ignore", invalid="ignore"):
        truth_range = truth_max - truth_min
    if np.isfinite(truth_range):
        gains = np.asarray((truth - truth_min) / truth_range, dtype=np.float64)
    else:
        scale = max(abs(truth_min), abs(truth_max))
        scaled_truth = truth / scale
        scaled_min = truth_min / scale
        scaled_range = truth_max / scale - scaled_min
        gains = np.asarray(
            (scaled_truth - scaled_min) / scaled_range,
            dtype=np.float64,
        )
    discounts = np.log2(np.arange(2, k + 2, dtype=np.float64))

    predicted_order = np.argsort(-prediction, kind="stable")[:k]
    ideal_order = np.argsort(-gains, kind="stable")[:k]
    dcg = float(np.sum(gains[predicted_order] / discounts))
    ideal_dcg = float(np.sum(gains[ideal_order] / discounts))
    if ideal_dcg == 0.0:
        return 0.0
    result = dcg / ideal_dcg
    if not np.isfinite(result):
        raise ValueError("NDCG result must be finite")
    return float(result)
