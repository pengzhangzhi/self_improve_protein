import warnings
from collections.abc import Callable
from inspect import signature
from typing import NoReturn, TypeAlias

import numpy as np
import pytest
from numpy.typing import NDArray

import self_improve_protein.selection as selection_module
from self_improve_protein.provenance import derive_seed
from self_improve_protein.ridge import fit_weighted_ridge, squared_loss
from self_improve_protein.selection import (
    influence_scores,
    no_hessian_scores,
    random_indices,
    stable_top_k,
    top_teacher_indices,
)

FloatArray: TypeAlias = NDArray[np.float64]


def _ridge_case() -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, float]:
    rng = np.random.default_rng(20260629)
    x_l = rng.normal(size=(9, 4)).astype(np.float64)
    y_l = rng.normal(size=9).astype(np.float64)
    x_u = rng.normal(size=(7, 4)).astype(np.float64)
    teacher_theta = rng.normal(size=4)
    yhat_u = (x_u @ teacher_theta + rng.normal(scale=0.4, size=7)).astype(np.float64)
    return x_l, y_l, x_u, yhat_u, 0.23


def test_balanced_fold_assignment_has_exact_disjoint_4_by_24_coverage() -> None:
    assignment = selection_module.balanced_fold_assignment(
        96,
        "ADRB2_HUMAN_Jones_2020",
        7,
    )

    assert assignment.dtype == np.int64
    assert assignment.shape == (96,)
    fold_members = [set(np.flatnonzero(assignment == fold)) for fold in range(4)]
    assert [len(members) for members in fold_members] == [24, 24, 24, 24]
    assert set.union(*fold_members) == set(range(96))
    overlaps = (
        first & second
        for first in fold_members
        for second in fold_members
        if first is not second
    )
    assert all(not overlap for overlap in overlaps)


def test_balanced_fold_assignment_matches_purpose_separated_pcg64_and_repeats() -> None:
    assay_id = "ADRB2_HUMAN_Jones_2020"
    seed = 7
    purpose = "crossfit_outer_folds_v1"
    permutation = np.random.Generator(
        np.random.PCG64(derive_seed(assay_id, seed, purpose))
    ).permutation(96)
    expected = np.empty(96, dtype=np.int64)
    expected[permutation] = np.repeat(np.arange(4, dtype=np.int64), 24)

    default = selection_module.balanced_fold_assignment(96, assay_id, seed)
    explicit = selection_module.balanced_fold_assignment(
        96,
        assay_id,
        seed,
        purpose=purpose,
    )
    repeated = selection_module.balanced_fold_assignment(96, assay_id, seed)

    np.testing.assert_array_equal(default, expected)
    np.testing.assert_array_equal(explicit, expected)
    np.testing.assert_array_equal(repeated, expected)


def test_balanced_fold_assignment_changes_with_purpose() -> None:
    default = selection_module.balanced_fold_assignment(
        96,
        "ADRB2_HUMAN_Jones_2020",
        7,
        purpose="crossfit_outer_folds_v1",
    )
    alternate = selection_module.balanced_fold_assignment(
        96,
        "ADRB2_HUMAN_Jones_2020",
        7,
        purpose="crossfit_outer_folds_sensitivity_v1",
    )

    assert not np.array_equal(default, alternate)


def test_out_of_fold_ridge_gradient_matches_explicit_fold_fits() -> None:
    rng = np.random.default_rng(8128)
    x_l = rng.normal(size=(12, 3)).astype(np.float64)
    y_l = rng.normal(size=12).astype(np.float64)
    ridge_lambda = 0.17
    folds = np.tile(np.arange(4, dtype=np.int64), 3)
    expected_residuals = np.empty(12, dtype=np.float64)
    for fold in range(4):
        held_out = folds == fold
        theta_minus_fold = fit_weighted_ridge(
            x_l[~held_out],
            y_l[~held_out],
            ridge_lambda,
        )
        expected_residuals[held_out] = x_l[held_out] @ theta_minus_fold - y_l[held_out]
    expected = x_l.T @ expected_residuals / x_l.shape[0]

    actual = selection_module.out_of_fold_ridge_gradient(
        x_l,
        y_l,
        folds,
        ridge_lambda,
    )

    assert actual.dtype == np.float64
    np.testing.assert_allclose(actual, expected, atol=1e-14, rtol=1e-13)


def test_cross_fitted_scores_match_hand_computed_v0_candidate_and_hessian_pieces() -> (
    None
):
    rng = np.random.default_rng(417)
    x_l = rng.normal(size=(12, 3)).astype(np.float64)
    y_l = rng.normal(size=12).astype(np.float64)
    x_u = rng.normal(size=(7, 3)).astype(np.float64)
    yhat_u = rng.normal(size=7).astype(np.float64)
    ridge_lambda = 0.19
    damping = 0.03
    assay_id = "TINY_ASSAY"
    seed = 11
    purpose = "crossfit_outer_folds_v1"
    theta = fit_weighted_ridge(x_l, y_l, ridge_lambda)
    folds = selection_module.balanced_fold_assignment(
        x_l.shape[0],
        assay_id,
        seed,
        purpose=purpose,
    )
    oof_residuals = np.empty(x_l.shape[0], dtype=np.float64)
    for fold in range(4):
        held_out = folds == fold
        theta_minus_fold = fit_weighted_ridge(
            x_l[~held_out],
            y_l[~held_out],
            ridge_lambda,
        )
        oof_residuals[held_out] = x_l[held_out] @ theta_minus_fold - y_l[held_out]
    g_cf = x_l.T @ oof_residuals / x_l.shape[0]
    g_l = x_l.T @ (x_l @ theta - y_l) / x_l.shape[0]
    hessian = x_l.T @ x_l / x_l.shape[0] + ridge_lambda * np.eye(3)
    system = hessian + damping * np.eye(3)
    candidate_gradients = (x_u @ theta - yhat_u)[:, None] * x_u
    expected = np.array(
        [
            g_cf @ np.linalg.solve(system, candidate_gradient - g_l)
            for candidate_gradient in candidate_gradients
        ],
        dtype=np.float64,
    )

    actual = selection_module.cross_fitted_influence_scores(
        x_l,
        y_l,
        x_u,
        yhat_u,
        theta,
        ridge_lambda,
        damping,
        assay_id,
        seed,
        purpose=purpose,
    )

    assert actual.dtype == np.float64
    np.testing.assert_allclose(actual, expected, atol=1e-13, rtol=1e-12)


@pytest.mark.parametrize(
    ("sample_count", "assay_id", "seed", "fold_count", "purpose"),
    [
        (95, "ASSAY", 0, 4, "crossfit_outer_folds_v1"),
        (96, "", 0, 4, "crossfit_outer_folds_v1"),
        (96, "ASSAY", -1, 4, "crossfit_outer_folds_v1"),
        (96, "ASSAY", 0, 1, "crossfit_outer_folds_v1"),
        (96, "ASSAY", 0, 5, "crossfit_outer_folds_v1"),
        (96, "ASSAY", 0, 4, ""),
        (True, "ASSAY", 0, 4, "crossfit_outer_folds_v1"),
        (96, "ASSAY", True, 4, "crossfit_outer_folds_v1"),
    ],
)
def test_balanced_fold_assignment_rejects_invalid_inputs(
    sample_count: object,
    assay_id: object,
    seed: object,
    fold_count: object,
    purpose: object,
) -> None:
    with pytest.raises(ValueError):
        selection_module.balanced_fold_assignment(
            sample_count,  # type: ignore[arg-type]
            assay_id,  # type: ignore[arg-type]
            seed,  # type: ignore[arg-type]
            fold_count=fold_count,  # type: ignore[arg-type]
            purpose=purpose,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("x_l", "y_l", "folds"),
    [
        (np.array([[np.nan], [1.0], [2.0], [3.0]]), np.ones(4), np.array([0, 0, 1, 1])),
        (np.ones((4, 1)), np.array([0.0, 1.0, np.inf, 3.0]), np.array([0, 0, 1, 1])),
        (np.ones((4, 1)), np.ones(3), np.array([0, 0, 1, 1])),
        (np.ones((4, 1)), np.ones(4), np.array([0.0, 0.0, 1.0, 1.0])),
        (np.ones((4, 1)), np.ones(4), np.array([0, 0, 0, 1])),
        (np.ones((4, 1)), np.ones(4), np.array([0, 0, 2, 2])),
    ],
)
def test_out_of_fold_ridge_gradient_rejects_nonfinite_or_invalid_fold_inputs(
    x_l: FloatArray,
    y_l: FloatArray,
    folds: NDArray[np.generic],
) -> None:
    with pytest.raises(ValueError):
        selection_module.out_of_fold_ridge_gradient(
            x_l,
            y_l,
            folds,  # type: ignore[arg-type]
            ridge_lambda=0.1,
        )


def test_cross_fitted_scores_reject_nonfinite_candidate_inputs() -> None:
    x_l, y_l, x_u, yhat_u, ridge_lambda = _ridge_case()
    theta = fit_weighted_ridge(x_l, y_l, ridge_lambda)
    bad_yhat = yhat_u.copy()
    bad_yhat[0] = np.nan

    with pytest.raises(ValueError):
        selection_module.cross_fitted_influence_scores(
            x_l,
            y_l,
            x_u,
            bad_yhat,
            theta,
            ridge_lambda,
            damping=0.01,
            assay_id="ASSAY",
            seed=0,
        )


def test_vectorized_influence_matches_explicit_candidate_gradients_and_one_solve(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    x_l, y_l, x_u, yhat_u, ridge_lambda = _ridge_case()
    damping = 0.07
    theta = fit_weighted_ridge(x_l, y_l, ridge_lambda)
    g_l = x_l.T @ (x_l @ theta - y_l) / x_l.shape[0]
    hessian = x_l.T @ x_l / x_l.shape[0] + ridge_lambda * np.eye(
        x_l.shape[1], dtype=np.float64
    )
    system = hessian + damping * np.eye(x_l.shape[1], dtype=np.float64)
    residual = x_u @ theta - yhat_u
    candidate_gradients = residual[:, None] * x_u
    expected = np.array(
        [
            g_l @ np.linalg.solve(system, candidate_gradient - g_l)
            for candidate_gradient in candidate_gradients
        ],
        dtype=np.float64,
    )
    original_solve = np.linalg.solve
    solve_calls = 0

    def counting_solve(a: FloatArray, b: FloatArray) -> FloatArray:
        nonlocal solve_calls
        solve_calls += 1
        return np.asarray(original_solve(a, b), dtype=np.float64)

    monkeypatch.setattr(np.linalg, "solve", counting_solve)

    actual = influence_scores(
        x_l,
        y_l,
        x_u,
        yhat_u,
        theta,
        ridge_lambda,
        damping,
    )

    assert actual.dtype == np.float64
    assert actual.shape == (x_u.shape[0],)
    assert solve_calls == 1
    np.testing.assert_allclose(actual, expected, atol=1e-13, rtol=1e-12)


def test_influence_is_negative_labeled_loss_derivative_for_multiple_candidates() -> (
    None
):
    x_l, y_l, x_u, yhat_u, ridge_lambda = _ridge_case()
    theta = fit_weighted_ridge(x_l, y_l, ridge_lambda)
    scores = influence_scores(
        x_l,
        y_l,
        x_u,
        yhat_u,
        theta,
        ridge_lambda,
        damping=0.0,
    )
    sample_count = x_l.shape[0]
    labeled_gram = x_l.T @ x_l / sample_count
    labeled_rhs = x_l.T @ y_l / sample_count
    identity = np.eye(x_l.shape[1], dtype=np.float64)
    base_loss = squared_loss(x_l, y_l, theta)
    epsilon = 1e-6

    for candidate_index in (0, 2, 5):
        x_j = x_u[candidate_index]
        mixed_system = (
            (1.0 - epsilon) * labeled_gram
            + epsilon * np.outer(x_j, x_j)
            + ridge_lambda * identity
        )
        mixed_rhs = (1.0 - epsilon) * labeled_rhs + epsilon * x_j * yhat_u[
            candidate_index
        ]
        theta_t = np.linalg.solve(mixed_system, mixed_rhs)
        finite_difference = (squared_loss(x_l, y_l, theta_t) - base_loss) / epsilon

        assert finite_difference == pytest.approx(
            -scores[candidate_index],
            rel=3e-5,
            abs=3e-7,
        )


def test_influence_rejects_overflowed_damped_system_without_warning() -> None:
    x_l = np.array([[1e154]], dtype=np.float64)
    y_l = np.array([0.0], dtype=np.float64)
    x_u = np.array([[1.0]], dtype=np.float64)
    yhat_u = np.array([0.0], dtype=np.float64)
    theta = np.array([0.0], dtype=np.float64)

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        with pytest.raises(ValueError, match="damped Hessian system must be finite"):
            influence_scores(
                x_l,
                y_l,
                x_u,
                yhat_u,
                theta,
                ridge_lambda=0.0,
                damping=1e308,
            )

    assert caught_warnings == []


def test_influence_reports_singular_damped_hessian_with_context() -> None:
    x_l = np.array([[1.0, 2.0]], dtype=np.float64)
    y_l = np.array([1.0], dtype=np.float64)
    x_u = np.array([[3.0, 4.0]], dtype=np.float64)
    yhat_u = np.array([2.0], dtype=np.float64)
    theta = np.zeros(2, dtype=np.float64)

    with pytest.raises(
        np.linalg.LinAlgError,
        match="damped Hessian system is singular",
    ):
        influence_scores(
            x_l,
            y_l,
            x_u,
            yhat_u,
            theta,
            ridge_lambda=0.0,
            damping=0.0,
        )


def test_self_teacher_scores_are_identical_and_nonpositive() -> None:
    x_l, y_l, x_u, _, ridge_lambda = _ridge_case()
    theta = fit_weighted_ridge(x_l, y_l, ridge_lambda)
    yhat_u = x_u @ theta

    scores = influence_scores(
        x_l,
        y_l,
        x_u,
        yhat_u,
        theta,
        ridge_lambda,
        damping=1e-4,
    )

    np.testing.assert_allclose(scores, scores[0], atol=1e-14, rtol=0.0)
    assert scores[0] <= 0.0


def test_full_rank_ols_scores_are_numerically_zero_and_tied() -> None:
    rng = np.random.default_rng(817)
    x_l = rng.normal(size=(11, 4)).astype(np.float64)
    y_l = rng.normal(size=11).astype(np.float64)
    x_u = rng.normal(size=(8, 4)).astype(np.float64)
    yhat_u = rng.normal(size=8).astype(np.float64)
    theta = fit_weighted_ridge(x_l, y_l, ridge_lambda=0.0)

    scores = influence_scores(
        x_l,
        y_l,
        x_u,
        yhat_u,
        theta,
        ridge_lambda=0.0,
        damping=0.0,
    )

    np.testing.assert_allclose(scores, 0.0, atol=2e-14, rtol=0.0)
    np.testing.assert_allclose(scores, scores[0], atol=3e-14, rtol=0.0)


def test_no_hessian_scores_match_explicit_gradient_inner_products() -> None:
    x_l, y_l, x_u, yhat_u, ridge_lambda = _ridge_case()
    theta = fit_weighted_ridge(x_l, y_l, ridge_lambda)
    g_l = x_l.T @ (x_l @ theta - y_l) / x_l.shape[0]
    residual = x_u @ theta - yhat_u
    expected = np.array(
        [
            g_l @ (residual_j * x_j - g_l)
            for residual_j, x_j in zip(residual, x_u, strict=True)
        ],
        dtype=np.float64,
    )

    actual = no_hessian_scores(
        x_l,
        y_l,
        x_u,
        yhat_u,
        theta,
        ridge_lambda,
    )

    assert actual.dtype == np.float64
    np.testing.assert_allclose(actual, expected, atol=1e-14, rtol=1e-13)


def test_no_hessian_scores_do_not_construct_a_hessian(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    x_l, y_l, x_u, yhat_u, ridge_lambda = _ridge_case()
    theta = fit_weighted_ridge(x_l, y_l, ridge_lambda)

    def unexpected_eye(*args: object, **kwargs: object) -> NoReturn:
        raise AssertionError("no-Hessian scoring must not construct a Hessian")

    monkeypatch.setattr(np, "eye", unexpected_eye)

    scores = no_hessian_scores(
        x_l,
        y_l,
        x_u,
        yhat_u,
        theta,
        ridge_lambda,
    )

    assert scores.shape == (x_u.shape[0],)


def test_feature_rescaling_preserves_full_scores_and_both_rankings() -> None:
    x_l, y_l, x_u, yhat_u, ridge_lambda = _ridge_case()
    damping = 0.013
    theta = fit_weighted_ridge(x_l, y_l, ridge_lambda)
    scale = 6.5
    hashes = [f"hash-{index:02d}" for index in range(x_u.shape[0])]

    full = influence_scores(
        x_l,
        y_l,
        x_u,
        yhat_u,
        theta,
        ridge_lambda,
        damping,
    )
    scaled_full = influence_scores(
        scale * x_l,
        y_l,
        scale * x_u,
        yhat_u,
        theta / scale,
        scale**2 * ridge_lambda,
        scale**2 * damping,
    )
    no_hessian = no_hessian_scores(
        x_l,
        y_l,
        x_u,
        yhat_u,
        theta,
        ridge_lambda,
    )
    scaled_no_hessian = no_hessian_scores(
        scale * x_l,
        y_l,
        scale * x_u,
        yhat_u,
        theta / scale,
        scale**2 * ridge_lambda,
    )

    np.testing.assert_allclose(scaled_full, full, atol=2e-13, rtol=2e-12)
    np.testing.assert_allclose(
        scaled_no_hessian,
        scale**2 * no_hessian,
        atol=2e-13,
        rtol=2e-12,
    )
    np.testing.assert_array_equal(
        stable_top_k(scaled_full, hashes, len(hashes)),
        stable_top_k(full, hashes, len(hashes)),
    )
    np.testing.assert_array_equal(
        stable_top_k(scaled_no_hessian, hashes, len(hashes)),
        stable_top_k(no_hessian, hashes, len(hashes)),
    )


def test_stable_top_k_orders_exact_ties_by_ascending_hash() -> None:
    scores = np.array([1.0, 2.0, 2.0, -1.0, 1.0], dtype=np.float64)
    hashes = ["e", "c", "a", "d", "b"]

    largest = stable_top_k(scores, hashes, k=4, largest=True)
    smallest = stable_top_k(scores, hashes, k=4, largest=False)

    assert largest.dtype == np.int64
    assert smallest.dtype == np.int64
    np.testing.assert_array_equal(largest, np.array([2, 1, 4, 0]))
    np.testing.assert_array_equal(smallest, np.array([3, 4, 0, 2]))


@pytest.mark.parametrize(
    ("scores", "hashes", "k", "largest"),
    [
        (np.array([0.0, np.nan]), ["a", "b"], 1, True),
        (np.array([0.0, np.inf]), ["a", "b"], 1, True),
        (np.ones((2, 1)), ["a", "b"], 1, True),
        (np.array([], dtype=np.float64), [], 1, True),
        (np.array([1.0, 2.0]), ["a"], 1, True),
        (np.array([1.0, 2.0]), ["a", 2], 1, True),
        (np.array([1.0, 2.0]), ["a", "a"], 1, True),
        (np.array([1.0, 2.0]), "ab", 1, True),
        (np.array([1.0, 2.0]), ["a", "b"], 0, True),
        (np.array([1.0, 2.0]), ["a", "b"], 3, True),
        (np.array([1.0, 2.0]), ["a", "b"], True, True),
        (np.array([1.0, 2.0]), ["a", "b"], np.int64(1), True),
        (np.array([1.0, 2.0]), ["a", "b"], 1, 1),
    ],
)
def test_stable_top_k_rejects_invalid_inputs(
    scores: FloatArray,
    hashes: object,
    k: object,
    largest: object,
) -> None:
    with pytest.raises(ValueError):
        stable_top_k(scores, hashes, k, largest=largest)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("pool_size", "k", "seed", "expected_values"),
    [
        (23, 9, 987654321, [3, 4, 5, 7, 12, 13, 17, 19, 22]),
        (7, 3, 123, [0, 4, 6]),
    ],
)
def test_random_indices_match_frozen_numpy_2_3_5_vectors(
    pool_size: int,
    k: int,
    seed: int,
    expected_values: list[int],
) -> None:
    expected = np.array(expected_values, dtype=np.int64)
    first = random_indices(pool_size, k, seed)
    second = random_indices(pool_size, k, seed)

    assert first.dtype == np.int64
    np.testing.assert_array_equal(first, expected)
    np.testing.assert_array_equal(second, expected)
    np.testing.assert_array_equal(first, np.sort(first))
    assert np.unique(first).size == k
    assert np.all((first >= 0) & (first < pool_size))


@pytest.mark.parametrize(
    ("pool_size", "k", "seed"),
    [
        (-1, 1, 0),
        (0, 1, 0),
        (3, 0, 0),
        (3, 4, 0),
        (True, 1, 0),
        (3.0, 1, 0),
        (np.int64(3), 1, 0),
        (3, True, 0),
        (3, 1.0, 0),
        (3, np.int64(1), 0),
        (3, 1, -1),
        (3, 1, True),
        (3, 1, 1.0),
        (3, 1, np.int64(1)),
    ],
)
def test_random_indices_rejects_invalid_integer_inputs(
    pool_size: object,
    k: object,
    seed: object,
) -> None:
    with pytest.raises(ValueError):
        random_indices(pool_size, k, seed)  # type: ignore[arg-type]


def test_top_teacher_indices_use_descending_score_and_stable_hash_ties() -> None:
    yhat = np.array([0.2, 1.7, 1.7, -0.5, 0.2], dtype=np.float64)
    hashes = ["z", "q", "a", "x", "b"]

    indices = top_teacher_indices(yhat, hashes, k=4)

    np.testing.assert_array_equal(indices, np.array([2, 1, 4, 0]))


def test_selector_signatures_cannot_receive_hidden_unlabeled_or_test_labels() -> None:
    expected_parameters: dict[Callable[..., object], tuple[str, ...]] = {
        influence_scores: (
            "x_l",
            "y_l",
            "x_u",
            "yhat_u",
            "theta",
            "ridge_lambda",
            "damping",
        ),
        no_hessian_scores: (
            "x_l",
            "y_l",
            "x_u",
            "yhat_u",
            "theta",
            "ridge_lambda",
        ),
        selection_module.balanced_fold_assignment: (
            "sample_count",
            "assay_id",
            "seed",
            "fold_count",
            "purpose",
        ),
        selection_module.out_of_fold_ridge_gradient: (
            "x_l",
            "y_l",
            "fold_assignment",
            "ridge_lambda",
        ),
        selection_module.cross_fitted_influence_scores: (
            "x_l",
            "y_l",
            "x_u",
            "yhat_u",
            "theta",
            "ridge_lambda",
            "damping",
            "assay_id",
            "seed",
            "fold_count",
            "purpose",
        ),
        stable_top_k: ("scores", "stable_hashes", "k", "largest"),
        random_indices: ("pool_size", "k", "seed"),
        top_teacher_indices: ("yhat", "hashes", "k"),
    }

    for function, parameters in expected_parameters.items():
        assert tuple(signature(function).parameters) == parameters


@pytest.mark.parametrize(
    "score_function",
    [influence_scores, no_hessian_scores],
)
def test_score_functions_accept_float_inputs_and_return_float64(
    score_function: Callable[..., FloatArray],
) -> None:
    x_l, y_l, x_u, yhat_u, ridge_lambda = _ridge_case()
    theta = fit_weighted_ridge(x_l, y_l, ridge_lambda)
    arguments: list[object] = [
        x_l.astype(np.float32),
        y_l.astype(np.float32),
        x_u.astype(np.float32),
        yhat_u.astype(np.float32),
        theta.astype(np.float32),
        ridge_lambda,
    ]
    if score_function is influence_scores:
        arguments.append(0.01)

    scores = score_function(*arguments)

    assert scores.dtype == np.float64
    assert scores.shape == (x_u.shape[0],)


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("x_l", np.ones(3)),
        ("x_l", np.empty((0, 3))),
        ("x_l", np.array([[1.0, np.nan, 2.0]])),
        ("y_l", np.ones((3, 1))),
        ("y_l", np.ones(2)),
        ("x_u", np.ones(3)),
        ("x_u", np.ones((2, 4))),
        ("x_u", np.array([[1.0, np.inf, 2.0]])),
        ("yhat_u", np.ones((3, 1))),
        ("yhat_u", np.ones(2)),
        ("theta", np.ones((1, 3))),
        ("theta", np.ones(2)),
    ],
)
def test_score_functions_reject_invalid_shapes_widths_and_values(
    field: str,
    bad_value: FloatArray,
) -> None:
    x_l = np.arange(15, dtype=np.float64).reshape(5, 3)
    y_l = np.arange(5, dtype=np.float64)
    x_u = np.arange(12, dtype=np.float64).reshape(4, 3)
    yhat_u = np.arange(4, dtype=np.float64)
    theta = np.ones(3, dtype=np.float64)
    values = {
        "x_l": x_l,
        "y_l": y_l,
        "x_u": x_u,
        "yhat_u": yhat_u,
        "theta": theta,
    }
    values[field] = bad_value

    with pytest.raises(ValueError):
        no_hessian_scores(
            values["x_l"],
            values["y_l"],
            values["x_u"],
            values["yhat_u"],
            values["theta"],
            ridge_lambda=0.1,
        )
    with pytest.raises(ValueError):
        influence_scores(
            values["x_l"],
            values["y_l"],
            values["x_u"],
            values["yhat_u"],
            values["theta"],
            ridge_lambda=0.1,
            damping=0.01,
        )


@pytest.mark.parametrize(
    "bad_ridge_lambda",
    [True, "0.1", -0.1, np.nan, np.inf, -np.inf],
)
def test_score_functions_reject_invalid_ridge_lambda(
    bad_ridge_lambda: object,
) -> None:
    x_l, y_l, x_u, yhat_u, _ = _ridge_case()
    theta = np.ones(x_l.shape[1], dtype=np.float64)

    with pytest.raises(ValueError):
        no_hessian_scores(
            x_l,
            y_l,
            x_u,
            yhat_u,
            theta,
            bad_ridge_lambda,  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError):
        influence_scores(
            x_l,
            y_l,
            x_u,
            yhat_u,
            theta,
            bad_ridge_lambda,  # type: ignore[arg-type]
            damping=0.0,
        )


@pytest.mark.parametrize(
    "bad_damping",
    [True, "0.1", -0.1, np.nan, np.inf, -np.inf],
)
def test_influence_scores_reject_invalid_damping(bad_damping: object) -> None:
    x_l, y_l, x_u, yhat_u, ridge_lambda = _ridge_case()
    theta = np.ones(x_l.shape[1], dtype=np.float64)

    with pytest.raises(ValueError):
        influence_scores(
            x_l,
            y_l,
            x_u,
            yhat_u,
            theta,
            ridge_lambda,
            bad_damping,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("score_function", [influence_scores, no_hessian_scores])
def test_score_functions_reject_nonfinite_pseudo_labels_and_parameters(
    score_function: Callable[..., FloatArray],
) -> None:
    x_l, y_l, x_u, yhat_u, ridge_lambda = _ridge_case()
    theta = np.ones(x_l.shape[1], dtype=np.float64)
    bad_yhat = yhat_u.copy()
    bad_yhat[0] = np.nan
    bad_theta = theta.copy()
    bad_theta[0] = np.inf

    for candidate_yhat, candidate_theta in (
        (bad_yhat, theta),
        (yhat_u, bad_theta),
    ):
        arguments: list[object] = [
            x_l,
            y_l,
            x_u,
            candidate_yhat,
            candidate_theta,
            ridge_lambda,
        ]
        if score_function is influence_scores:
            arguments.append(0.0)
        with pytest.raises(ValueError):
            score_function(*arguments)
