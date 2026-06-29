from collections.abc import Callable
from inspect import signature
from typing import NoReturn, TypeAlias

import numpy as np
import pytest
from numpy.typing import NDArray

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


def test_random_indices_match_direct_pcg64_choice_and_are_sorted_unique() -> None:
    pool_size = 23
    k = 9
    seed = 987654321
    expected = np.sort(
        np.random.Generator(np.random.PCG64(seed)).choice(
            pool_size,
            size=k,
            replace=False,
        )
    ).astype(np.int64)

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
