import warnings
from inspect import signature

import numpy as np
import pytest

from self_improve_protein.metrics import (
    ndcg_at_10_percent,
    spearman_correlation,
    standardized_mse,
)


def test_spearman_matches_rank_correlation_with_ties() -> None:
    y_true = np.array([-2.0, 1.0, 1.0, 4.0, 8.0], dtype=np.float64)
    y_pred = np.array([10.0, 2.0, 2.0, -1.0, 5.0], dtype=np.float64)

    assert spearman_correlation(y_true, y_pred) == pytest.approx(-0.368421052631579)


@pytest.mark.parametrize(
    ("y_true", "y_pred"),
    [
        (np.ones(4), np.arange(4.0)),
        (np.arange(4.0), np.ones(4)),
        (np.ones(1), np.ones(1)),
    ],
)
def test_spearman_explicitly_rejects_undefined_constant_inputs(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> None:
    with pytest.raises(ValueError, match=r"undefined.*constant"):
        spearman_correlation(y_true, y_pred)


def test_standardized_mse_has_explicit_true_then_prediction_argument_order() -> None:
    y_true = np.array([-2.0, 0.0, 3.0], dtype=np.float64)
    y_pred = np.array([-1.0, 2.0, -1.0], dtype=np.float64)

    assert tuple(signature(standardized_mse).parameters) == ("y_true", "y_pred")
    assert standardized_mse(y_true, y_pred) == pytest.approx(7.0)


def test_ndcg_matches_pinned_proteingym_v13_reference_fixture() -> None:
    # Expected value derived from calc_ndcg in the official ProteinGym source:
    # commit 1f8de974dead8ff7501eff087b725d14a965e9f9,
    # proteingym/performance_DMS_benchmarks.py:14-70.  This fixture keeps only
    # the numerical behavior needed for parity; it does not copy upstream code.
    y_true = np.arange(-5.0, 15.0, dtype=np.float64)
    y_pred = -np.arange(20.0, dtype=np.float64)
    y_pred[19] = 100.0
    y_pred[10] = 90.0

    assert ndcg_at_10_percent(y_true, y_pred) == pytest.approx(
        0.8337292223703456,
        abs=1e-15,
    )


def test_ndcg_uses_minmax_gains_for_negative_truth_and_zero_for_zero_dcg() -> None:
    y_true = np.arange(-4.0, 6.0, dtype=np.float64)

    assert ndcg_at_10_percent(y_true, y_true) == pytest.approx(1.0)
    assert ndcg_at_10_percent(y_true, -y_true) == pytest.approx(0.0)


def test_ndcg_stably_breaks_prediction_ties_by_input_order() -> None:
    y_true = np.arange(10.0, dtype=np.float64)
    tied_predictions = np.zeros(10, dtype=np.float64)
    reordered_true = y_true.copy()
    reordered_true[[0, 9]] = reordered_true[[9, 0]]

    assert ndcg_at_10_percent(y_true, tied_predictions) == pytest.approx(0.0)
    assert ndcg_at_10_percent(reordered_true, tied_predictions) == pytest.approx(1.0)


def test_ndcg_constant_truth_returns_zero_because_ideal_dcg_is_zero() -> None:
    y_true = np.full(10, -7.0, dtype=np.float64)
    y_pred = np.arange(10.0, dtype=np.float64)

    assert ndcg_at_10_percent(y_true, y_pred) == 0.0


@pytest.mark.parametrize(
    "metric",
    [spearman_correlation, standardized_mse, ndcg_at_10_percent],
)
@pytest.mark.parametrize(
    ("y_true", "y_pred", "message"),
    [
        (np.array([[1.0, 2.0]]), np.array([1.0, 2.0]), "1D"),
        (np.array([1.0, 2.0]), np.array([[1.0, 2.0]]), "1D"),
        (np.array([], dtype=np.float64), np.array([], dtype=np.float64), "non-empty"),
        (np.array([1.0, 2.0]), np.array([1.0]), "same length"),
        (np.array([1.0, np.nan]), np.array([1.0, 2.0]), "finite"),
        (np.array([1.0, 2.0]), np.array([1.0, np.inf]), "finite"),
    ],
)
def test_metrics_reject_bad_shapes_lengths_and_nonfinite_values(
    metric: object,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        metric(y_true, y_pred)  # type: ignore[operator]


def test_ndcg_rejects_a_sample_too_small_for_a_nonempty_top_decile() -> None:
    with pytest.raises(ValueError, match=r"top 10%.*at least one"):
        ndcg_at_10_percent(np.arange(9.0), np.arange(9.0))


def test_metric_functions_do_not_mutate_inputs() -> None:
    y_true = np.linspace(-2.0, 2.0, 20)
    y_pred = np.roll(y_true, 3)
    true_before = y_true.copy()
    pred_before = y_pred.copy()

    spearman_correlation(y_true, y_pred)
    standardized_mse(y_true, y_pred)
    ndcg_at_10_percent(y_true, y_pred)

    np.testing.assert_array_equal(y_true, true_before)
    np.testing.assert_array_equal(y_pred, pred_before)


@pytest.mark.parametrize(
    "bad_values",
    [
        np.array([True, False]),
        np.array(["1.0", "2.0"]),
        np.array([1.0 + 2.0j, 3.0 + 0.0j]),
    ],
)
def test_metrics_reject_non_real_numeric_inputs(bad_values: np.ndarray) -> None:
    with pytest.raises(ValueError, match="numeric real"):
        standardized_mse(bad_values, np.arange(2.0))


def test_rank_metrics_handle_extreme_finite_ranges_without_warnings() -> None:
    values = np.array(
        [-1e308, -1e200, -1e100, -1.0, 0.0, 1.0, 1e100, 1e200, 5e307, 1e308]
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        spearman = spearman_correlation(values, values)
        ndcg = ndcg_at_10_percent(values, values)

    assert caught == []
    assert spearman == pytest.approx(1.0)
    assert ndcg == pytest.approx(1.0)


def test_mse_rejects_nonfinite_result_from_finite_overflowing_residuals() -> None:
    with pytest.raises(ValueError, match=r"MSE.*finite"):
        standardized_mse(np.array([1e308, -1e308]), np.array([-1e308, 1e308]))
