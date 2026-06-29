from dataclasses import FrozenInstanceError

import numpy as np
import pytest
from sklearn.linear_model import Ridge

from self_improve_protein.ridge import (
    FeatureTransform,
    LabelTransform,
    TeacherCalibration,
    fit_feature_transform,
    fit_label_transform,
    fit_teacher_calibration,
    fit_weighted_ridge,
    labeled_gradient_hessian,
    squared_loss,
)


def test_feature_transform_uses_labeled_center_and_one_global_rms() -> None:
    x_l = np.array(
        [
            [1.0, 10.0, -4.0],
            [3.0, 14.0, 2.0],
            [8.0, 18.0, 5.0],
            [12.0, 22.0, 9.0],
        ],
        dtype=np.float64,
    )

    transform = fit_feature_transform(x_l)
    transformed = transform.transform(x_l)
    expected_mean = x_l.mean(axis=0)
    expected_scale = float(np.sqrt(np.mean((x_l - expected_mean) ** 2)))

    assert transformed.dtype == np.float64
    np.testing.assert_allclose(transform.mean, expected_mean, atol=0.0, rtol=0.0)
    assert transform.scale == pytest.approx(expected_scale)
    np.testing.assert_allclose(transformed.mean(axis=0), 0.0, atol=1e-15)
    assert np.sqrt(np.mean(transformed**2)) == pytest.approx(1.0)


def test_feature_transform_reuses_labeled_statistics_on_same_width_matrix() -> None:
    x_l = np.array([[1.0, 5.0], [3.0, 9.0]], dtype=np.float64)
    x_u = np.array([[7.0, 13.0], [11.0, 17.0]], dtype=np.float32)

    transform = fit_feature_transform(x_l)

    np.testing.assert_allclose(
        transform.transform(x_u),
        (x_u.astype(np.float64) - transform.mean) / transform.scale,
    )


@pytest.mark.parametrize(
    ("bad_x", "message"),
    [
        (np.array([1.0, 2.0]), "2D"),
        (np.array([[[1.0]]]), "2D"),
        (np.empty((0, 2)), "non-empty"),
        (np.empty((2, 0)), "non-empty"),
        (np.array([[1.0, np.nan]]), "finite"),
        (np.array([[1.0, np.inf]]), "finite"),
    ],
)
def test_fit_feature_transform_rejects_invalid_matrices(
    bad_x: np.ndarray, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        fit_feature_transform(bad_x)


def test_feature_transform_rejects_constant_labeled_matrix() -> None:
    with pytest.raises(ValueError, match="scale"):
        fit_feature_transform(np.ones((4, 3), dtype=np.float64))


def test_feature_transform_rejects_effectively_zero_scale_but_accepts_control() -> (
    None
):
    tiny_delta = 1e-10
    tiny_variation = np.array(
        [
            [1.0 - tiny_delta, 1.0 + tiny_delta],
            [1.0 + tiny_delta, 1.0 - tiny_delta],
        ],
        dtype=np.float64,
    )
    control_delta = 1e-6
    resolvable_variation = np.array(
        [
            [1.0 - control_delta, 1.0 + control_delta],
            [1.0 + control_delta, 1.0 - control_delta],
        ],
        dtype=np.float64,
    )

    with pytest.raises(ValueError, match="effectively zero"):
        fit_feature_transform(tiny_variation)
    with pytest.raises(ValueError, match="effectively zero"):
        FeatureTransform(mean=np.array([1.0, 1.0]), scale=tiny_delta)

    transform = fit_feature_transform(resolvable_variation)
    assert transform.scale == pytest.approx(control_delta)


def test_feature_transform_rejects_wrong_width_rank_and_nonfinite_inputs() -> None:
    transform = fit_feature_transform(
        np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float64)
    )

    with pytest.raises(ValueError, match="width"):
        transform.transform(np.ones((3, 3), dtype=np.float64))
    with pytest.raises(ValueError, match="2D"):
        transform.transform(np.ones(2, dtype=np.float64))
    with pytest.raises(ValueError, match="finite"):
        transform.transform(np.array([[0.0, np.nan]], dtype=np.float64))


def test_feature_transform_validates_parameters_and_is_frozen() -> None:
    with pytest.raises(ValueError, match=r"mean.*1D"):
        FeatureTransform(mean=np.ones((1, 2)), scale=1.0)
    with pytest.raises(ValueError, match=r"mean.*finite"):
        FeatureTransform(mean=np.array([np.inf]), scale=1.0)
    with pytest.raises(ValueError, match="scale"):
        FeatureTransform(mean=np.array([0.0]), scale=0.0)
    with pytest.raises(ValueError, match="scale"):
        FeatureTransform(mean=np.array([0.0]), scale=np.nan)

    transform = FeatureTransform(mean=np.array([0.0]), scale=1.0)
    with pytest.raises(FrozenInstanceError):
        transform.scale = 2.0
    with pytest.raises(ValueError, match="read-only"):
        transform.mean[0] = 2.0


def test_label_transform_uses_population_scale_and_round_trips() -> None:
    y_l = np.array([-3.0, 0.0, 4.0, 10.0], dtype=np.float64)

    transform = fit_label_transform(y_l)
    standardized = transform.transform(y_l)

    assert transform.ddof == 0
    assert transform.mean == pytest.approx(float(y_l.mean()))
    assert transform.scale == pytest.approx(float(y_l.std(ddof=0)))
    assert standardized.mean() == pytest.approx(0.0, abs=1e-15)
    assert standardized.std(ddof=0) == pytest.approx(1.0)
    assert standardized.dtype == np.float64
    np.testing.assert_allclose(transform.inverse(standardized), y_l, atol=1e-15)


@pytest.mark.parametrize(
    "bad_y",
    [
        np.array([[1.0, 2.0]]),
        np.array([], dtype=np.float64),
        np.array([1.0, np.nan]),
        np.array([1.0, np.inf]),
    ],
)
def test_fit_label_transform_rejects_invalid_labels(bad_y: np.ndarray) -> None:
    with pytest.raises(ValueError):
        fit_label_transform(bad_y)


def test_label_transform_rejects_zero_scale_invalid_ddof_and_bad_values() -> None:
    with pytest.raises(ValueError, match="scale"):
        fit_label_transform(np.ones(3, dtype=np.float64))
    with pytest.raises(ValueError, match="ddof"):
        fit_label_transform(np.array([1.0, 2.0]), ddof=1)
    with pytest.raises(ValueError, match="mean"):
        LabelTransform(mean=np.inf, scale=1.0)
    with pytest.raises(ValueError, match="scale"):
        LabelTransform(mean=0.0, scale=0.0)
    with pytest.raises(ValueError, match="ddof"):
        LabelTransform(mean=0.0, scale=1.0, ddof=1)

    transform = LabelTransform(mean=0.0, scale=1.0)
    with pytest.raises(ValueError, match="1D"):
        transform.transform(np.ones((1, 2), dtype=np.float64))
    with pytest.raises(ValueError, match="finite"):
        transform.inverse(np.array([np.nan]))
    with pytest.raises(FrozenInstanceError):
        transform.mean = 3.0


def test_label_transform_rejects_effectively_zero_scale_but_accepts_control() -> (
    None
):
    tiny_delta = 1e-10
    tiny_variation = np.array(
        [1.0 - tiny_delta, 1.0 + tiny_delta],
        dtype=np.float64,
    )
    control_delta = 1e-6
    resolvable_variation = np.array(
        [1.0 - control_delta, 1.0 + control_delta],
        dtype=np.float64,
    )

    with pytest.raises(ValueError, match="effectively zero"):
        fit_label_transform(tiny_variation)
    with pytest.raises(ValueError, match="effectively zero"):
        LabelTransform(mean=1.0, scale=tiny_delta)

    transform = fit_label_transform(resolvable_variation)
    assert transform.scale == pytest.approx(control_delta)


def test_teacher_calibration_exactly_recovers_affine_teacher_mapping() -> None:
    z_l = np.array([-4.0, -1.0, 0.5, 3.0, 8.0], dtype=np.float64)
    y_l_std = 2.75 * z_l - 1.25

    calibration = fit_teacher_calibration(z_l, y_l_std)

    assert calibration.slope == pytest.approx(2.75, abs=1e-13)
    assert calibration.intercept == pytest.approx(-1.25, abs=1e-13)
    np.testing.assert_allclose(
        calibration.predict(z_l), y_l_std, atol=1e-13, rtol=0.0
    )


def test_constant_teacher_uses_zero_slope_and_label_mean_intercept() -> None:
    z_l = np.full(4, 7.0, dtype=np.float64)
    y_l_std = np.array([-2.0, 0.0, 1.0, 5.0], dtype=np.float64)

    calibration = fit_teacher_calibration(z_l, y_l_std)

    assert calibration.slope == 0.0
    assert calibration.intercept == pytest.approx(float(y_l_std.mean()))
    np.testing.assert_allclose(
        calibration.predict(np.array([-10.0, 0.0, 10.0])),
        y_l_std.mean(),
    )


def test_teacher_calibration_recovers_affine_map_at_large_offset() -> None:
    offset = 1e8
    z_l = offset + np.linspace(-0.25, 0.25, 9, dtype=np.float64)
    y_l_std = 2.75 * z_l - 1.25

    calibration = fit_teacher_calibration(z_l, y_l_std)

    assert np.ptp(z_l) < np.sqrt(np.finfo(np.float64).eps) * offset
    assert calibration.slope == pytest.approx(2.75, abs=1e-13)
    assert calibration.intercept == pytest.approx(-1.25, abs=1e-5)
    np.testing.assert_allclose(
        calibration.predict(z_l),
        y_l_std,
        atol=1e-12,
        rtol=0.0,
    )


def test_near_constant_teacher_uses_intercept_only_calibration() -> None:
    offset = 1e8
    spacing = np.spacing(offset)
    z_l = offset + np.array([-spacing, 0.0, spacing, 0.0])
    y_l_std = np.array([-2.0, 0.0, 1.0, 5.0], dtype=np.float64)

    calibration = fit_teacher_calibration(z_l, y_l_std)

    assert np.ptp(z_l) > 0.0
    assert calibration.slope == 0.0
    assert calibration.intercept == pytest.approx(float(y_l_std.mean()))
    np.testing.assert_allclose(calibration.predict(z_l), y_l_std.mean())


def test_teacher_calibration_rejects_invalid_inputs_and_parameters() -> None:
    with pytest.raises(ValueError, match="same number"):
        fit_teacher_calibration(np.ones(2), np.ones(3))
    with pytest.raises(ValueError, match="1D"):
        fit_teacher_calibration(np.ones((2, 1)), np.ones(2))
    with pytest.raises(ValueError, match="finite"):
        fit_teacher_calibration(np.array([0.0, np.nan]), np.ones(2))
    with pytest.raises(ValueError, match="non-empty"):
        fit_teacher_calibration(np.array([]), np.array([]))
    with pytest.raises(ValueError, match="slope"):
        TeacherCalibration(slope=np.nan, intercept=0.0)
    with pytest.raises(ValueError, match="intercept"):
        TeacherCalibration(slope=1.0, intercept=np.inf)

    calibration = TeacherCalibration(slope=1.0, intercept=0.0)
    with pytest.raises(ValueError, match="1D"):
        calibration.predict(np.ones((2, 1)))
    with pytest.raises(ValueError, match="finite"):
        calibration.predict(np.array([np.inf]))
    with pytest.raises(FrozenInstanceError):
        calibration.slope = 2.0


def test_supervised_ridge_satisfies_exact_normal_equation_and_stationarity() -> None:
    x = np.array(
        [
            [1.0, -2.0, 0.5],
            [0.0, 3.0, 1.0],
            [2.0, 1.0, -1.0],
            [-1.0, 0.5, 2.0],
            [3.0, -1.0, 1.5],
        ],
        dtype=np.float64,
    )
    y = np.array([1.0, -2.0, 0.5, 4.0, -1.5], dtype=np.float64)
    ridge_lambda = 0.17

    theta = fit_weighted_ridge(x, y, ridge_lambda)
    expected = np.linalg.solve(
        x.T @ x + x.shape[0] * ridge_lambda * np.eye(x.shape[1]),
        x.T @ y,
    )
    gradient, hessian = labeled_gradient_hessian(x, y, theta, ridge_lambda)

    assert theta.dtype == np.float64
    np.testing.assert_allclose(theta, expected, atol=1e-13, rtol=1e-13)
    np.testing.assert_allclose(
        (x.T @ x + x.shape[0] * ridge_lambda * np.eye(x.shape[1]))
        @ theta,
        x.T @ y,
        atol=1e-13,
        rtol=1e-13,
    )
    assert np.linalg.norm(gradient + ridge_lambda * theta) < 1e-13
    np.testing.assert_allclose(
        hessian,
        x.T @ x / x.shape[0] + ridge_lambda * np.eye(x.shape[1]),
    )


def test_pseudo_weighted_ridge_uses_normalization_and_sklearn_oracle() -> None:
    rng = np.random.default_rng(20260629)
    x_true = rng.normal(size=(7, 4))
    x_pseudo = rng.normal(size=(5, 4))
    y_true = rng.normal(size=7)
    y_pseudo = rng.normal(size=5)
    x = np.vstack((x_true, x_pseudo)).astype(np.float64)
    y = np.concatenate((y_true, y_pseudo)).astype(np.float64)
    weights = np.concatenate((np.ones(7), np.full(5, 0.1))).astype(np.float64)
    ridge_lambda = 0.23
    denominator = float(weights.sum())

    theta = fit_weighted_ridge(x, y, ridge_lambda, sample_weight=weights)
    gram = x.T @ (weights[:, None] * x)
    rhs = x.T @ (weights * y)
    expected = np.linalg.solve(
        gram + denominator * ridge_lambda * np.eye(x.shape[1]), rhs
    )
    sklearn_theta = Ridge(
        alpha=denominator * ridge_lambda,
        fit_intercept=False,
        solver="cholesky",
    ).fit(x, y, sample_weight=weights).coef_
    wrong_alpha_theta = Ridge(
        alpha=ridge_lambda,
        fit_intercept=False,
        solver="cholesky",
    ).fit(x, y, sample_weight=weights).coef_

    np.testing.assert_allclose(theta, expected, atol=1e-13, rtol=1e-13)
    np.testing.assert_allclose(theta, sklearn_theta, atol=1e-12, rtol=1e-12)
    assert np.linalg.norm(theta - wrong_alpha_theta) > 1e-3


def test_weighted_ridge_predictions_are_feature_scale_equivariant() -> None:
    rng = np.random.default_rng(7)
    x = rng.normal(size=(11, 5)).astype(np.float64)
    y = rng.normal(size=11).astype(np.float64)
    weights = rng.uniform(0.1, 2.0, size=11).astype(np.float64)
    ridge_lambda = 0.031
    feature_scale = 4.25

    theta = fit_weighted_ridge(x, y, ridge_lambda, sample_weight=weights)
    scaled_theta = fit_weighted_ridge(
        feature_scale * x,
        y,
        feature_scale**2 * ridge_lambda,
        sample_weight=weights,
    )

    np.testing.assert_allclose(
        scaled_theta, theta / feature_scale, atol=1e-13, rtol=1e-12
    )
    np.testing.assert_allclose(
        feature_scale * x @ scaled_theta,
        x @ theta,
        atol=1e-13,
        rtol=1e-12,
    )


def test_unregularized_fit_matches_lstsq_on_ill_conditioned_full_rank_design() -> (
    None
):
    t = np.linspace(-1.0, 1.0, 8, dtype=np.float64)
    x = np.column_stack((np.ones(8), 1.0 + 3e-8 * t))
    theta_true = np.array([1.25, -0.75], dtype=np.float64)
    y = x @ theta_true

    theta = fit_weighted_ridge(x, y, ridge_lambda=0.0)
    lstsq_theta = np.linalg.lstsq(x, y, rcond=None)[0]

    assert 5e7 < np.linalg.cond(x) < 2e8
    assert np.linalg.norm(lstsq_theta - theta_true) < 1e-7
    np.testing.assert_allclose(theta, lstsq_theta, atol=1e-12, rtol=1e-12)
    assert np.linalg.norm(theta - theta_true) < 1e-7


@pytest.mark.parametrize("ridge_lambda", [0.0, 0.19])
def test_weighted_ridge_is_invariant_to_common_positive_weight_rescaling(
    ridge_lambda: float,
) -> None:
    rng = np.random.default_rng(193)
    x = rng.normal(size=(9, 4)).astype(np.float64)
    y = rng.normal(size=9).astype(np.float64)
    weights = rng.uniform(0.2, 1.7, size=9).astype(np.float64)

    theta = fit_weighted_ridge(x, y, ridge_lambda, sample_weight=weights)
    rescaled_theta = fit_weighted_ridge(
        x,
        y,
        ridge_lambda,
        sample_weight=37.0 * weights,
    )

    np.testing.assert_allclose(rescaled_theta, theta, atol=1e-13, rtol=1e-12)


def test_squared_loss_is_half_the_mean_squared_residual() -> None:
    x = np.array([[1.0, 2.0], [-1.0, 3.0], [2.0, 0.0]], dtype=np.float64)
    y = np.array([0.5, -2.0, 3.0], dtype=np.float64)
    theta = np.array([1.5, -0.25], dtype=np.float64)
    residual = x @ theta - y

    assert squared_loss(x, y, theta) == pytest.approx(
        float(np.mean(residual**2) / 2.0)
    )


@pytest.mark.parametrize("bad_lambda", [-1.0, np.nan, np.inf, -np.inf, True, "0.1"])
def test_weighted_ridge_rejects_invalid_regularization(bad_lambda: object) -> None:
    x = np.eye(2, dtype=np.float64)
    y = np.ones(2, dtype=np.float64)

    with pytest.raises(ValueError, match="ridge_lambda"):
        fit_weighted_ridge(x, y, bad_lambda)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "bad_weight",
    [
        np.array([1.0]),
        np.array([[1.0, 1.0]]),
        np.array([1.0, 0.0]),
        np.array([1.0, -0.1]),
        np.array([1.0, np.nan]),
        np.array([1.0, np.inf]),
    ],
)
def test_weighted_ridge_rejects_invalid_weights(bad_weight: np.ndarray) -> None:
    with pytest.raises(ValueError, match="sample_weight"):
        fit_weighted_ridge(
            np.eye(2, dtype=np.float64),
            np.ones(2, dtype=np.float64),
            0.1,
            sample_weight=bad_weight,
        )


@pytest.mark.parametrize(
    ("bad_x", "bad_y", "message"),
    [
        (np.ones(2), np.ones(2), "2D"),
        (np.ones((2, 2)), np.ones((2, 1)), "1D"),
        (np.ones((2, 2)), np.ones(3), "same number"),
        (np.array([[1.0, np.nan], [0.0, 1.0]]), np.ones(2), "finite"),
        (np.eye(2), np.array([1.0, np.inf]), "finite"),
        (np.empty((0, 2)), np.empty(0), "non-empty"),
    ],
)
def test_weighted_ridge_rejects_invalid_training_arrays(
    bad_x: np.ndarray, bad_y: np.ndarray, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        fit_weighted_ridge(bad_x, bad_y, 0.1)


def test_unregularized_singular_ridge_raises_clear_linear_algebra_error() -> None:
    x = np.array([[1.0, 2.0], [2.0, 4.0]], dtype=np.float64)
    y = np.array([1.0, 2.0], dtype=np.float64)

    with pytest.raises(np.linalg.LinAlgError, match="singular"):
        fit_weighted_ridge(x, y, ridge_lambda=0.0)


def test_unregularized_rank_deficiency_is_rejected_before_silent_lapack_solve() -> (
    None
):
    rng = np.random.default_rng(4)
    column = rng.normal(size=7)
    x = np.column_stack((column, column)).astype(np.float64)
    y = rng.normal(size=7).astype(np.float64)

    assert np.linalg.matrix_rank(x) == 1
    with pytest.raises(np.linalg.LinAlgError, match="singular"):
        fit_weighted_ridge(x, y, ridge_lambda=0.0)


def test_gradient_hessian_and_loss_reject_bad_theta_and_parameters() -> None:
    x = np.eye(2, dtype=np.float64)
    y = np.ones(2, dtype=np.float64)

    with pytest.raises(ValueError, match=r"theta.*length"):
        labeled_gradient_hessian(x, y, np.ones(3), 0.1)
    with pytest.raises(ValueError, match=r"theta.*1D"):
        labeled_gradient_hessian(x, y, np.ones((2, 1)), 0.1)
    with pytest.raises(ValueError, match="ridge_lambda"):
        labeled_gradient_hessian(x, y, np.ones(2), -0.1)
    with pytest.raises(ValueError, match=r"theta.*finite"):
        squared_loss(x, y, np.array([1.0, np.nan]))
    with pytest.raises(ValueError, match="same number"):
        squared_loss(x, np.ones(3), np.ones(2))
