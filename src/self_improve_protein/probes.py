"""Deterministic, outcome-blind numerical probes for verification rungs R2-R3."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

from self_improve_protein.provenance import (
    atomic_write_json,
    sha256_bytes,
    sha256_file,
)
from self_improve_protein.ridge import (
    fit_weighted_ridge,
    labeled_gradient_hessian,
    squared_loss,
)
from self_improve_protein.selection import (
    influence_scores,
    no_hessian_scores,
    stable_top_k,
)

FloatArray: TypeAlias = NDArray[np.float64]

PROBE_SCHEMA_VERSION = 1
PROBE_SEED = 20260629
LEARNABILITY_RIDGE_LAMBDA = 1e-10
LEARNABILITY_TRAIN_MSE_TOLERANCE = 1e-18
LEARNABILITY_PARAMETER_ERROR_TOLERANCE = 1e-8
LEARNABILITY_NORMAL_EQUATION_TOLERANCE = 1e-12
CAUSAL_EPSILON = 1e-6
CAUSAL_DERIVATIVE_ERROR_TOLERANCE = 2e-5
CAUSAL_ALGEBRA_TOLERANCE = 1e-12

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_LEARNABILITY_SAMPLE_COUNT = 128
_LEARNABILITY_FEATURE_COUNT = 8
_CAUSAL_SAMPLE_COUNT = 80
_CAUSAL_FEATURE_COUNT = 5
_CAUSAL_RIDGE_LAMBDA = 0.4
_SELECTION_COUNT = 3
_VERIFICATION_RUNG_DIRECTORIES = ("r1", "r2", "r3")
_VERIFICATION_RELATIVE_FILES = (
    "r1/report.json",
    "r1/fresh-environment-resolution.json",
    "r2/pytest.txt",
    "r2/algebra_probe.json",
    "r3/synthetic_probe.json",
)
_VERIFICATION_REPORT_OUTPUT_FILES = _VERIFICATION_RELATIVE_FILES[1:]
_TRUST_ROOT_HASH_FIELDS = (
    "pyproject_sha256",
    "uv_lock_sha256",
    "config_sha256",
    "verification_script_sha256",
    "python_executable_sha256",
    "uv_executable_sha256",
    "pytest_executable_sha256",
    "ruff_executable_sha256",
    "mypy_executable_sha256",
)


def _canonical_digest(payload: object) -> str:
    serialized = json.dumps(
        payload,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return sha256_bytes(serialized)


def _float_list(values: FloatArray) -> list[float]:
    return [float(value) for value in np.ravel(values)]


def _score_statistics(scores: FloatArray) -> dict[str, object]:
    quantiles = np.asarray(
        np.quantile(scores, [0.0, 0.25, 0.5, 0.75, 1.0]),
        dtype=np.float64,
    )
    return {
        "minimum": float(np.min(scores)),
        "maximum": float(np.max(scores)),
        "mean": float(np.mean(scores)),
        "standard_deviation": float(np.std(scores, ddof=0)),
        "positive_fraction": float(np.mean(scores > 0.0)),
        "unique_count": int(np.unique(scores).size),
        "quantiles_0_25_50_75_100": _float_list(quantiles),
    }


def _learnability_probe(seed: int) -> dict[str, object]:
    generator = np.random.Generator(np.random.PCG64(seed))
    x = generator.normal(
        size=(_LEARNABILITY_SAMPLE_COUNT, _LEARNABILITY_FEATURE_COUNT)
    ).astype(np.float64)
    theta_true = generator.normal(size=_LEARNABILITY_FEATURE_COUNT).astype(np.float64)
    y = np.asarray(x @ theta_true, dtype=np.float64)
    theta = fit_weighted_ridge(x, y, LEARNABILITY_RIDGE_LAMBDA)
    residual = np.asarray(x @ theta - y, dtype=np.float64)
    normal_residual = x.T @ residual + x.shape[0] * LEARNABILITY_RIDGE_LAMBDA * theta

    return {
        "seed": seed,
        "sample_count": int(x.shape[0]),
        "feature_count": int(x.shape[1]),
        "matrix_rank": int(np.linalg.matrix_rank(x)),
        "condition_number": float(np.linalg.cond(x)),
        "ridge_lambda": LEARNABILITY_RIDGE_LAMBDA,
        "train_mse": float(np.mean(residual * residual)),
        "parameter_error_norm": float(np.linalg.norm(theta - theta_true)),
        "normal_equation_residual_norm": float(np.linalg.norm(normal_residual)),
        "x_dtype": str(x.dtype),
        "y_dtype": str(y.dtype),
        "theta_dtype": str(theta.dtype),
        "tolerances": {
            "train_mse": LEARNABILITY_TRAIN_MSE_TOLERANCE,
            "parameter_error_norm": LEARNABILITY_PARAMETER_ERROR_TOLERANCE,
            "normal_equation_residual_norm": (LEARNABILITY_NORMAL_EQUATION_TOLERANCE),
        },
    }


def _candidate_hashes(seed: int, count: int) -> list[str]:
    return [
        sha256_bytes(f"r3-candidate\0{seed}\0{index}".encode())
        for index in range(count)
    ]


def _exact_perturbed_fit(
    x_l: FloatArray,
    y_l: FloatArray,
    x_candidate: FloatArray,
    yhat_candidate: float,
    *,
    ridge_lambda: float,
    epsilon: float,
) -> tuple[FloatArray, float]:
    x_augmented = np.vstack((x_l, x_candidate[None, :])).astype(np.float64)
    y_augmented = np.concatenate((y_l, np.asarray([yhat_candidate], dtype=np.float64)))
    weights = np.concatenate(
        (
            np.full(
                x_l.shape[0],
                (1.0 - epsilon) / x_l.shape[0],
                dtype=np.float64,
            ),
            np.asarray([epsilon], dtype=np.float64),
        )
    )
    theta = fit_weighted_ridge(
        x_augmented,
        y_augmented,
        ridge_lambda,
        sample_weight=weights,
    )
    weighted_residual = x_augmented @ theta - y_augmented
    normal_residual = (
        x_augmented.T @ (weights * weighted_residual) + ridge_lambda * theta
    )
    return theta, float(np.linalg.norm(normal_residual))


def _causal_score_probe(seed: int, epsilon: float) -> dict[str, object]:
    generator = np.random.Generator(np.random.PCG64(seed))
    x_l = generator.normal(size=(_CAUSAL_SAMPLE_COUNT, _CAUSAL_FEATURE_COUNT)).astype(
        np.float64
    )
    theta_truth = generator.normal(size=_CAUSAL_FEATURE_COUNT).astype(np.float64)
    noise = 0.2 * generator.normal(size=_CAUSAL_SAMPLE_COUNT)
    y_l = np.asarray(x_l @ theta_truth + noise, dtype=np.float64)
    theta = fit_weighted_ridge(x_l, y_l, _CAUSAL_RIDGE_LAMBDA)
    gradient, hessian = labeled_gradient_hessian(
        x_l,
        y_l,
        theta,
        _CAUSAL_RIDGE_LAMBDA,
    )
    gradient_identity_residual = gradient + _CAUSAL_RIDGE_LAMBDA * theta
    influence_direction = np.linalg.solve(hessian, gradient)
    direction_squared_norm = float(influence_direction @ influence_direction)
    if direction_squared_norm <= 0.0:
        raise RuntimeError("causal probe produced a zero influence direction")

    x_u = generator.normal(size=(6, _CAUSAL_FEATURE_COUNT)).astype(np.float64)
    teacher_offset = 2.0 * generator.normal(size=_CAUSAL_FEATURE_COUNT)
    teacher_theta = np.asarray(theta + teacher_offset, dtype=np.float64)
    yhat_u = np.asarray(x_u @ teacher_theta, dtype=np.float64)
    gradient_geometry = float(gradient @ influence_direction)
    pseudo_residual = np.asarray(x_u @ theta - yhat_u, dtype=np.float64)

    scores = influence_scores(
        x_l,
        y_l,
        x_u,
        yhat_u,
        theta,
        _CAUSAL_RIDGE_LAMBDA,
        damping=0.0,
    )
    no_h_scores = no_hessian_scores(
        x_l,
        y_l,
        x_u,
        yhat_u,
        theta,
        _CAUSAL_RIDGE_LAMBDA,
    )
    manual_no_h = pseudo_residual * (x_u @ gradient) - gradient @ gradient

    baseline_loss = squared_loss(x_l, y_l, theta)
    realized_changes: list[float] = []
    perturbed_normal_residuals: list[float] = []
    for candidate, pseudo_label in zip(x_u, yhat_u, strict=True):
        perturbed_theta, normal_residual = _exact_perturbed_fit(
            x_l,
            y_l,
            candidate,
            float(pseudo_label),
            ridge_lambda=_CAUSAL_RIDGE_LAMBDA,
            epsilon=epsilon,
        )
        realized_changes.append(squared_loss(x_l, y_l, perturbed_theta) - baseline_loss)
        perturbed_normal_residuals.append(normal_residual)

    realized = np.asarray(realized_changes, dtype=np.float64)
    predicted = np.asarray(-epsilon * scores, dtype=np.float64)
    finite_difference_derivatives = realized / epsilon
    derivative_errors = finite_difference_derivatives + scores
    predicted_order = np.argsort(-scores, kind="stable")
    realized_order = np.argsort(realized, kind="stable")

    self_yhat = np.asarray(x_u @ theta, dtype=np.float64)
    self_pseudo_gradients = (x_u @ theta - self_yhat)[:, None] * x_u
    self_scores = influence_scores(
        x_l,
        y_l,
        x_u,
        self_yhat,
        theta,
        _CAUSAL_RIDGE_LAMBDA,
        damping=0.0,
    )
    self_no_h_scores = no_hessian_scores(
        x_l,
        y_l,
        x_u,
        self_yhat,
        theta,
        _CAUSAL_RIDGE_LAMBDA,
    )
    expected_self_score = -gradient_geometry
    expected_self_no_h = -float(gradient @ gradient)

    hashes = _candidate_hashes(seed, x_u.shape[0])
    full_h_indices = stable_top_k(scores, hashes, _SELECTION_COUNT)
    no_h_indices = stable_top_k(no_h_scores, hashes, _SELECTION_COUNT)
    full_h_hashes = [hashes[int(index)] for index in full_h_indices]
    no_h_hashes = [hashes[int(index)] for index in no_h_indices]

    return {
        "seed": seed,
        "epsilon": epsilon,
        "sample_count": int(x_l.shape[0]),
        "feature_count": int(x_l.shape[1]),
        "candidate_count": int(x_u.shape[0]),
        "ridge_lambda": _CAUSAL_RIDGE_LAMBDA,
        "damping": 0.0,
        "x_l_dtype": str(x_l.dtype),
        "y_l_dtype": str(y_l.dtype),
        "x_u_dtype": str(x_u.dtype),
        "theta_dtype": str(theta.dtype),
        "raw_normal_equation_residual_norm": float(
            np.linalg.norm(gradient_identity_residual)
        ),
        "perturbed_normal_equation_max_residual": float(
            np.max(perturbed_normal_residuals)
        ),
        "baseline_labeled_half_mse": baseline_loss,
        "external_teacher": {
            "kind": "fixed_nonidentical_linear_teacher",
            "parameter_distance_norm": float(np.linalg.norm(teacher_offset)),
            "pseudo_label_dtype": str(yhat_u.dtype),
            "pseudo_residual_minimum": float(np.min(pseudo_residual)),
            "pseudo_residual_maximum": float(np.max(pseudo_residual)),
            "pseudo_residual_norm": float(np.linalg.norm(pseudo_residual)),
        },
        "full_h_scores": _float_list(scores),
        "full_h_score_statistics": _score_statistics(scores),
        "no_h_scores": _float_list(no_h_scores),
        "no_h_score_statistics": _score_statistics(no_h_scores),
        "no_h_formula_max_abs_error": float(np.max(np.abs(no_h_scores - manual_no_h))),
        "predicted_loss_changes": _float_list(predicted),
        "realized_loss_changes": _float_list(realized),
        "finite_difference_derivatives": _float_list(finite_difference_derivatives),
        "derivative_errors": _float_list(derivative_errors),
        "derivative_max_abs_error": float(np.max(np.abs(derivative_errors))),
        "first_order_signs_match": bool(
            np.array_equal(np.sign(predicted), np.sign(realized))
        ),
        "first_order_order_matches": bool(
            np.array_equal(predicted_order, realized_order)
        ),
        "predicted_order": [int(index) for index in predicted_order],
        "realized_order": [int(index) for index in realized_order],
        "derivative_error_tolerance": CAUSAL_DERIVATIVE_ERROR_TOLERANCE,
        "self_teacher_control": {
            "full_h_scores": _float_list(self_scores),
            "no_h_scores": _float_list(self_no_h_scores),
            "expected_full_h_constant": expected_self_score,
            "expected_no_h_constant": expected_self_no_h,
            "score_range": float(np.ptp(self_scores)),
            "expected_constant_max_abs_error": float(
                np.max(np.abs(self_scores - expected_self_score))
            ),
            "no_h_expected_constant_max_abs_error": float(
                np.max(np.abs(self_no_h_scores - expected_self_no_h))
            ),
            "pseudo_gradient_max_norm": float(
                np.max(np.linalg.norm(self_pseudo_gradients, axis=1))
            ),
        },
        "selection": {
            "selected_count": _SELECTION_COUNT,
            "candidate_hashes": hashes,
            "full_h_selected_indices": [int(index) for index in full_h_indices],
            "full_h_selected_hashes": full_h_hashes,
            "full_h_selection_digest": _canonical_digest(full_h_hashes),
            "no_h_selected_indices": [int(index) for index in no_h_indices],
            "no_h_selected_hashes": no_h_hashes,
            "no_h_selection_digest": _canonical_digest(no_h_hashes),
        },
    }


def _single_probe(seed: int, epsilon: float) -> dict[str, object]:
    learnability = _learnability_probe(seed)
    causal = _causal_score_probe(seed + 1, epsilon)
    learnability_normal_residual = float(
        cast(float, learnability["normal_equation_residual_norm"])
    )
    causal_normal_residual = float(
        cast(float, causal["raw_normal_equation_residual_norm"])
    )
    perturbed_normal_residual = float(
        cast(float, causal["perturbed_normal_equation_max_residual"])
    )
    score_stats = cast(dict[str, object], causal["full_h_score_statistics"])
    algebra: dict[str, object] = {
        "dimensions": {
            "learnability_x": [
                cast(int, learnability["sample_count"]),
                cast(int, learnability["feature_count"]),
            ],
            "causal_labeled_x": [
                cast(int, causal["sample_count"]),
                cast(int, causal["feature_count"]),
            ],
            "causal_unlabeled_x": [
                cast(int, causal["candidate_count"]),
                cast(int, causal["feature_count"]),
            ],
        },
        "dtypes": {
            "learnability_x": learnability["x_dtype"],
            "learnability_y": learnability["y_dtype"],
            "learnability_theta": learnability["theta_dtype"],
            "causal_labeled_x": causal["x_l_dtype"],
            "causal_labeled_y": causal["y_l_dtype"],
            "causal_unlabeled_x": causal["x_u_dtype"],
            "causal_theta": causal["theta_dtype"],
        },
        "finite_checks": {
            "learnability": True,
            "causal_scores": True,
            "finite_differences": True,
            "all": True,
        },
        "gradient_identity_residual_norm": causal_normal_residual,
        "normal_equation_residual_norm": max(
            learnability_normal_residual,
            causal_normal_residual,
            perturbed_normal_residual,
        ),
        "score_statistics": score_stats,
    }
    return {
        "schema_version": PROBE_SCHEMA_VERSION,
        "seed": seed,
        "outcome_blind": True,
        "learnability": learnability,
        "causal_score": causal,
        "algebra": algebra,
    }


def run_synthetic_probe(
    *,
    seed: int = PROBE_SEED,
    epsilon: float = CAUSAL_EPSILON,
) -> dict[str, object]:
    """Run the R2-R3 probe twice and return one validated deterministic payload."""
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if seed != PROBE_SEED:
        raise ValueError(f"seed must equal the locked value {PROBE_SEED}")
    if isinstance(epsilon, (bool, str, bytes)):
        raise ValueError(
            "epsilon must be a finite number strictly between zero and one"
        )
    epsilon_value = float(epsilon)
    if not math.isfinite(epsilon_value) or not 0.0 < epsilon_value < 1.0:
        raise ValueError(
            "epsilon must be a finite number strictly between zero and one"
        )
    if epsilon_value != CAUSAL_EPSILON:
        raise ValueError(f"epsilon must equal the locked value {CAUSAL_EPSILON}")

    first = _single_probe(seed, epsilon_value)
    second = _single_probe(seed, epsilon_value)
    first_digest = _canonical_digest(first)
    second_digest = _canonical_digest(second)
    if first != second or first_digest != second_digest:
        raise RuntimeError("synthetic probe repeated execution was not deterministic")
    payload = dict(first)
    payload["repeat_verification"] = {
        "run_count": 2,
        "run_digests": [first_digest, second_digest],
        "digests_match": True,
    }
    payload["deterministic_digest"] = _canonical_digest(payload)
    validate_synthetic_probe(payload)
    return payload


def _mapping(payload: object, *, name: str) -> dict[str, object]:
    if not isinstance(payload, dict) or any(
        not isinstance(key, str) for key in payload
    ):
        raise ValueError(f"{name} must be a JSON object")
    return cast(dict[str, object], payload)


def _finite_number(payload: dict[str, object], field: str) -> float:
    value = payload.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be a finite number")
    return result


def _finite_number_list(payload: dict[str, object], field: str) -> list[float]:
    values = payload.get(field)
    if not isinstance(values, list) or not values:
        raise ValueError(f"{field} must be a non-empty finite-number list")
    result: list[float] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{field} must be a non-empty finite-number list")
        converted = float(value)
        if not math.isfinite(converted):
            raise ValueError(f"{field} must be a non-empty finite-number list")
        result.append(converted)
    return result


def validate_synthetic_probe(
    payload: object,
    *,
    verify_digest: bool = True,
) -> None:
    """Validate JSON safety, deterministic identity, and every R2-R3 threshold."""
    root = _mapping(payload, name="synthetic probe")
    try:
        json.dumps(root, allow_nan=False, sort_keys=True)
    except (TypeError, ValueError) as error:
        raise ValueError("synthetic probe must be finite JSON") from error
    if root.get("schema_version") != PROBE_SCHEMA_VERSION:
        raise ValueError("synthetic probe schema version is invalid")
    if root.get("outcome_blind") is not True:
        raise ValueError("synthetic probe must be marked outcome-blind")
    if verify_digest:
        provided_digest = root.get("deterministic_digest")
        if not isinstance(provided_digest, str) or not _SHA256_PATTERN.fullmatch(
            provided_digest
        ):
            raise ValueError("deterministic digest is invalid")
        unsigned = dict(root)
        del unsigned["deterministic_digest"]
        if _canonical_digest(unsigned) != provided_digest:
            raise ValueError("deterministic digest does not match payload")
    if root.get("seed") != PROBE_SEED:
        raise ValueError("synthetic probe seed is not locked")

    learnability = _mapping(root.get("learnability"), name="learnability")
    if learnability.get("seed") != PROBE_SEED:
        raise ValueError("learnability seed is not locked")
    if (
        learnability.get("sample_count") != _LEARNABILITY_SAMPLE_COUNT
        or learnability.get("feature_count") != _LEARNABILITY_FEATURE_COUNT
    ):
        raise ValueError("learnability dimensions are not locked")
    if any(
        learnability.get(field) != "float64"
        for field in ("x_dtype", "y_dtype", "theta_dtype")
    ):
        raise ValueError("learnability dtypes must be float64")
    if _finite_number(learnability, "ridge_lambda") != (LEARNABILITY_RIDGE_LAMBDA):
        raise ValueError("learnability ridge lambda is not locked")
    if _finite_number(learnability, "train_mse") >= (LEARNABILITY_TRAIN_MSE_TOLERANCE):
        raise ValueError("learnability train MSE exceeds tolerance")
    if _finite_number(learnability, "parameter_error_norm") >= (
        LEARNABILITY_PARAMETER_ERROR_TOLERANCE
    ):
        raise ValueError("learnability parameter error exceeds tolerance")
    if _finite_number(learnability, "normal_equation_residual_norm") >= (
        LEARNABILITY_NORMAL_EQUATION_TOLERANCE
    ):
        raise ValueError("learnability normal equation residual exceeds tolerance")
    if learnability.get("matrix_rank") != learnability.get("feature_count"):
        raise ValueError("learnability design must have full column rank")
    sample_count = _finite_number(learnability, "sample_count")
    feature_count = _finite_number(learnability, "feature_count")
    if sample_count <= feature_count:
        raise ValueError("learnability design must have more samples than features")

    causal = _mapping(root.get("causal_score"), name="causal score")
    if causal.get("seed") != PROBE_SEED + 1:
        raise ValueError("causal seed is not locked")
    if (
        causal.get("sample_count") != _CAUSAL_SAMPLE_COUNT
        or causal.get("feature_count") != _CAUSAL_FEATURE_COUNT
        or causal.get("candidate_count") != 6
    ):
        raise ValueError("causal dimensions are not locked")
    if any(
        causal.get(field) != "float64"
        for field in ("x_l_dtype", "y_l_dtype", "x_u_dtype", "theta_dtype")
    ):
        raise ValueError("causal dtypes must be float64")
    if _finite_number(causal, "ridge_lambda") != _CAUSAL_RIDGE_LAMBDA:
        raise ValueError("causal ridge lambda is not locked")
    if _finite_number(causal, "damping") != 0.0:
        raise ValueError("causal derivative probe must use zero damping")
    if _finite_number(causal, "epsilon") != CAUSAL_EPSILON:
        raise ValueError("causal epsilon is not locked")
    if causal.get("first_order_signs_match") is not True:
        raise ValueError("causal first-order signs do not match")
    if causal.get("first_order_order_matches") is not True:
        raise ValueError("causal first-order order does not match")
    if _finite_number(causal, "derivative_max_abs_error") > (
        CAUSAL_DERIVATIVE_ERROR_TOLERANCE
    ):
        raise ValueError("causal derivative error exceeds tolerance")
    if _finite_number(causal, "no_h_formula_max_abs_error") >= (
        CAUSAL_ALGEBRA_TOLERANCE
    ):
        raise ValueError("no-Hessian formula residual exceeds tolerance")
    if _finite_number(causal, "raw_normal_equation_residual_norm") >= (
        CAUSAL_ALGEBRA_TOLERANCE
    ):
        raise ValueError("causal normal equation residual exceeds tolerance")
    if _finite_number(causal, "perturbed_normal_equation_max_residual") >= (
        CAUSAL_ALGEBRA_TOLERANCE
    ):
        raise ValueError("perturbed normal equation residual exceeds tolerance")
    teacher = _mapping(causal.get("external_teacher"), name="external teacher")
    if teacher.get("kind") != "fixed_nonidentical_linear_teacher":
        raise ValueError("causal probe must use the fixed external teacher")
    if _finite_number(teacher, "parameter_distance_norm") <= 0.0:
        raise ValueError("external teacher must differ from the student")
    if _finite_number(teacher, "pseudo_residual_norm") <= 0.0:
        raise ValueError("external teacher must produce nonzero pseudo-gradients")

    scores = _finite_number_list(causal, "full_h_scores")
    no_h_scores = _finite_number_list(causal, "no_h_scores")
    predicted = _finite_number_list(causal, "predicted_loss_changes")
    realized = _finite_number_list(causal, "realized_loss_changes")
    finite_difference = _finite_number_list(
        causal,
        "finite_difference_derivatives",
    )
    derivative_errors = _finite_number_list(causal, "derivative_errors")
    vector_lengths = {
        len(scores),
        len(no_h_scores),
        len(predicted),
        len(realized),
        len(finite_difference),
        len(derivative_errors),
    }
    if len(vector_lengths) != 1 or len(set(scores)) != len(scores):
        raise ValueError("causal score vectors must align and full-H scores differ")
    if not min(scores) < 0.0 < max(scores):
        raise ValueError("causal full-H scores must contain both signs")
    expected_predicted = [-CAUSAL_EPSILON * score for score in scores]
    expected_finite_difference = [change / CAUSAL_EPSILON for change in realized]
    expected_errors = [
        derivative + score
        for derivative, score in zip(
            expected_finite_difference,
            scores,
            strict=True,
        )
    ]
    if not np.allclose(predicted, expected_predicted, rtol=0.0, atol=1e-15):
        raise ValueError("predicted first-order changes are inconsistent")
    if not np.allclose(
        finite_difference,
        expected_finite_difference,
        rtol=0.0,
        atol=1e-15,
    ):
        raise ValueError("finite-difference derivatives are inconsistent")
    if not np.allclose(
        derivative_errors,
        expected_errors,
        rtol=0.0,
        atol=1e-15,
    ):
        raise ValueError("finite-difference errors are inconsistent")
    actual_max_error = max(abs(error) for error in expected_errors)
    if actual_max_error > CAUSAL_DERIVATIVE_ERROR_TOLERANCE:
        raise ValueError("causal derivative error exceeds tolerance")
    if not math.isclose(
        _finite_number(causal, "derivative_max_abs_error"),
        actual_max_error,
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise ValueError("causal derivative maximum is inconsistent")
    if any(
        prediction * observation <= 0.0
        for prediction, observation in zip(predicted, realized, strict=True)
    ):
        raise ValueError("causal first-order signs do not match")
    expected_order = sorted(
        range(len(scores)),
        key=scores.__getitem__,
        reverse=True,
    )
    realized_order = sorted(range(len(realized)), key=realized.__getitem__)
    if expected_order != realized_order:
        raise ValueError("causal first-order order does not match")
    if causal.get("predicted_order") != expected_order:
        raise ValueError("stored predicted order is inconsistent")
    if causal.get("realized_order") != realized_order:
        raise ValueError("stored realized order is inconsistent")
    if causal.get("full_h_score_statistics") != _score_statistics(
        np.asarray(scores, dtype=np.float64)
    ):
        raise ValueError("full-H score statistics are inconsistent")
    if causal.get("no_h_score_statistics") != _score_statistics(
        np.asarray(no_h_scores, dtype=np.float64)
    ):
        raise ValueError("no-H score statistics are inconsistent")

    control = _mapping(
        causal.get("self_teacher_control"),
        name="self-teacher control",
    )
    self_scores = _finite_number_list(control, "full_h_scores")
    self_no_h_scores = _finite_number_list(control, "no_h_scores")
    if any(value >= 0.0 for value in self_scores):
        raise ValueError("self-teacher scores must be finite and strictly negative")
    if len(self_scores) != len(scores) or len(self_no_h_scores) != len(scores):
        raise ValueError("self-teacher control vectors must match candidates")
    if np.ptp(self_scores) >= CAUSAL_ALGEBRA_TOLERANCE:
        raise ValueError("self-teacher full-H scores must be constant")
    if np.ptp(self_no_h_scores) >= CAUSAL_ALGEBRA_TOLERANCE:
        raise ValueError("self-teacher no-H scores must be constant")
    for field in (
        "score_range",
        "expected_constant_max_abs_error",
        "no_h_expected_constant_max_abs_error",
        "pseudo_gradient_max_norm",
    ):
        if _finite_number(control, field) >= CAUSAL_ALGEBRA_TOLERANCE:
            raise ValueError("self-teacher degeneracy control exceeds tolerance")

    selection = _mapping(causal.get("selection"), name="selection")
    if selection.get("selected_count") != _SELECTION_COUNT:
        raise ValueError("selection count is not locked")
    candidate_hashes = selection.get("candidate_hashes")
    if (
        not isinstance(candidate_hashes, list)
        or not candidate_hashes
        or any(
            not isinstance(item, str) or not _SHA256_PATTERN.fullmatch(item)
            for item in candidate_hashes
        )
        or len(set(candidate_hashes)) != len(candidate_hashes)
    ):
        raise ValueError("candidate hashes are invalid")
    for field in ("full_h_selected_hashes", "no_h_selected_hashes"):
        selected = selection.get(field)
        if (
            not isinstance(selected, list)
            or len(selected) != _SELECTION_COUNT
            or any(item not in candidate_hashes for item in selected)
        ):
            raise ValueError(f"{field} is invalid")
    selection_contracts = (
        (
            "full_h_selected_indices",
            "full_h_selected_hashes",
            "full_h_selection_digest",
            scores,
        ),
        (
            "no_h_selected_indices",
            "no_h_selected_hashes",
            "no_h_selection_digest",
            no_h_scores,
        ),
    )
    for indices_field, hashes_field, digest_field, method_scores in selection_contracts:
        indices = selection.get(indices_field)
        selected_hashes = selection.get(hashes_field)
        digest = selection.get(digest_field)
        if (
            not isinstance(indices, list)
            or any(type(index) is not int for index in indices)
            or not isinstance(selected_hashes, list)
            or not isinstance(digest, str)
            or not _SHA256_PATTERN.fullmatch(digest)
        ):
            raise ValueError(f"{indices_field} contract is invalid")
        expected_indices = stable_top_k(
            np.asarray(method_scores, dtype=np.float64),
            candidate_hashes,
            _SELECTION_COUNT,
        ).tolist()
        expected_hashes = [candidate_hashes[index] for index in expected_indices]
        if indices != expected_indices or selected_hashes != expected_hashes:
            raise ValueError(f"{indices_field} does not match score ordering")
        if digest != _canonical_digest(expected_hashes):
            raise ValueError(f"{digest_field} does not match selected hashes")

    repeat = _mapping(root.get("repeat_verification"), name="repeat verification")
    run_digests = repeat.get("run_digests")
    if (
        repeat.get("run_count") != 2
        or repeat.get("digests_match") is not True
        or not isinstance(run_digests, list)
        or len(run_digests) != 2
        or len(set(run_digests)) != 1
        or any(
            not isinstance(digest, str) or not _SHA256_PATTERN.fullmatch(digest)
            for digest in run_digests
        )
    ):
        raise ValueError("repeat verification is invalid")
    base_payload = dict(root)
    base_payload.pop("deterministic_digest", None)
    base_payload.pop("repeat_verification", None)
    if run_digests[0] != _canonical_digest(base_payload):
        raise ValueError("repeat digest does not match the repeated payload")

    algebra = _mapping(root.get("algebra"), name="algebra")
    if set(algebra) != {
        "dimensions",
        "dtypes",
        "finite_checks",
        "gradient_identity_residual_norm",
        "normal_equation_residual_norm",
        "score_statistics",
    }:
        raise ValueError("algebra artifact schema is invalid")
    finite_checks = _mapping(
        algebra.get("finite_checks"),
        name="algebra finite checks",
    )
    if not finite_checks or any(value is not True for value in finite_checks.values()):
        raise ValueError("algebra finite checks did not pass")
    if algebra.get("score_statistics") != causal.get("full_h_score_statistics"):
        raise ValueError("algebra score statistics are inconsistent")


def write_synthetic_probe(
    output: Path | str,
    *,
    algebra_output: Path | str | None = None,
) -> dict[str, object]:
    """Run, validate, and atomically write synthetic R3 and optional R2 data."""
    payload = run_synthetic_probe()
    validate_synthetic_probe(payload)
    atomic_write_json(output, payload)
    if algebra_output is not None:
        atomic_write_json(algebra_output, payload["algebra"])
    return payload


def require_clean_verification_git_state(
    repository: Path | str,
    *,
    expected_head: str | None = None,
) -> str:
    """Return HEAD only when all tracked/untracked files are clean and stable."""
    root = Path(repository).resolve()
    if expected_head is not None and (
        not isinstance(expected_head, str)
        or not re.fullmatch(r"[0-9a-f]{40}", expected_head)
    ):
        raise ValueError("expected_head must be a lowercase 40-character git hash")
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            [
                "git",
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
            ],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError("repository must be an accessible git worktree") from error
    if not re.fullmatch(r"[0-9a-f]{40}", head):
        raise ValueError("git HEAD must be a lowercase 40-character hash")
    if status:
        raise ValueError("verification requires a clean git worktree")
    if expected_head is not None and head != expected_head:
        raise ValueError(f"verification HEAD changed from {expected_head} to {head}")
    return head


def _json_mapping_file(path: Path, *, name: str) -> dict[str, object]:
    try:
        payload: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{name} must be readable JSON") from error
    return _mapping(payload, name=name)


def _verification_files(root: Path) -> set[str]:
    files: set[str] = set()
    for directory_name in _VERIFICATION_RUNG_DIRECTORIES:
        directory = root / directory_name
        if not directory.is_dir() or directory.is_symlink():
            raise ValueError(f"verification {directory_name} must be a directory")
        for path in directory.rglob("*"):
            if path.is_symlink():
                raise ValueError("verification artifacts must not contain symlinks")
            if path.is_file():
                files.add(path.relative_to(root).as_posix())
    return files


def _validated_trust_root(
    payload: object,
    *,
    git_head: str,
) -> dict[str, object]:
    trust_root = _mapping(payload, name="verification trust root")
    if set(trust_root) != {"git_head", *_TRUST_ROOT_HASH_FIELDS}:
        raise ValueError("verification trust root fields are incomplete")
    if trust_root.get("git_head") != git_head:
        raise ValueError("verification trust root git HEAD is inconsistent")
    for field in _TRUST_ROOT_HASH_FIELDS:
        value = trust_root.get(field)
        if not isinstance(value, str) or not _SHA256_PATTERN.fullmatch(value):
            raise ValueError(f"verification trust root {field} is invalid")
    return trust_root


def _validate_fresh_environment_resolution(payload: object) -> None:
    resolution = _mapping(payload, name="fresh environment resolution")
    sync = _mapping(resolution.get("sync"), name="fresh environment sync")
    if sync.get("action") != "create":
        raise ValueError("fresh environment resolution must create a new environment")
    environment = _mapping(
        sync.get("environment"),
        name="fresh environment target",
    )
    environment_path = environment.get("path")
    if not isinstance(environment_path, str) or not environment_path.strip():
        raise ValueError("fresh environment target path is invalid")
    if Path(environment_path).name == ".venv":
        raise ValueError("fresh environment proof must not target ambient .venv")
    changes = sync.get("changes")
    if not isinstance(changes, list):
        raise ValueError("fresh environment changes must be a list")
    versions: dict[str, str] = {}
    for raw_change in changes:
        change = _mapping(raw_change, name="fresh environment change")
        name = change.get("name")
        version = change.get("version")
        action = change.get("action")
        if isinstance(name, str) and isinstance(version, str) and action == "installed":
            versions[name] = version
    expected = {"torch": "2.10.0", "transformers": "4.57.6"}
    if any(versions.get(name) != version for name, version in expected.items()):
        raise ValueError("fresh environment resolution omits pinned embed dependencies")


def _expected_artifact_paths(artifact_root: Path) -> dict[str, str]:
    return {
        "r1_report": str(artifact_root / "r1" / "report.json"),
        "fresh_environment_resolution": str(
            artifact_root / "r1" / "fresh-environment-resolution.json"
        ),
        "r2_pytest": str(artifact_root / "r2" / "pytest.txt"),
        "r2_algebra": str(artifact_root / "r2" / "algebra_probe.json"),
        "r3_probe": str(artifact_root / "r3" / "synthetic_probe.json"),
        "completion": str(artifact_root / "completion.json"),
    }


def _validate_verification_content(
    content_root: Path,
    *,
    artifact_root: Path,
    completion: dict[str, object],
    expected_git_head: str | None,
) -> None:
    if _verification_files(content_root) != set(_VERIFICATION_RELATIVE_FILES):
        raise ValueError("verification bundle has missing or stale artifact files")
    if completion.get("schema_version") != 1 or completion.get("status") != "passed":
        raise ValueError("verification completion schema or status is invalid")
    git_head = completion.get("git_head")
    if not isinstance(git_head, str) or not re.fullmatch(r"[0-9a-f]{40}", git_head):
        raise ValueError("verification completion git HEAD is invalid")
    if expected_git_head is not None and git_head != expected_git_head:
        raise ValueError(
            "verification completion git HEAD does not match expected HEAD"
        )
    published_at = completion.get("published_at")
    if not isinstance(published_at, str) or not published_at.strip():
        raise ValueError("verification completion timestamp is invalid")
    if completion.get("artifact_root") != str(artifact_root):
        raise ValueError("verification completion artifact root is inconsistent")
    expected_paths = {
        relative_path: str(artifact_root / relative_path)
        for relative_path in _VERIFICATION_RELATIVE_FILES
    }
    if completion.get("artifact_paths") != expected_paths:
        raise ValueError("verification completion artifact paths are inconsistent")
    trust_root = _validated_trust_root(completion.get("trust_root"), git_head=git_head)

    output_hashes = _mapping(
        completion.get("output_sha256"),
        name="verification output hashes",
    )
    if set(output_hashes) != set(_VERIFICATION_RELATIVE_FILES):
        raise ValueError("verification completion output hashes are incomplete")
    for relative_path in _VERIFICATION_RELATIVE_FILES:
        digest = output_hashes.get(relative_path)
        if not isinstance(digest, str) or not _SHA256_PATTERN.fullmatch(digest):
            raise ValueError(f"verification SHA-256 for {relative_path} is invalid")
        if sha256_file(content_root / relative_path) != digest:
            raise ValueError(f"verification SHA-256 mismatch for {relative_path}")

    report = _json_mapping_file(
        content_root / "r1" / "report.json",
        name="R1 report",
    )
    if (
        report.get("schema_version") != 2
        or report.get("rung") != "R1"
        or report.get("status") != "passed"
        or report.get("git_head") != git_head
    ):
        raise ValueError("R1 report schema, status, or git HEAD is invalid")
    if report.get("artifacts") != _expected_artifact_paths(artifact_root):
        raise ValueError("R1 report does not honor the configured artifact root")
    if report.get("trust_root") != trust_root:
        raise ValueError("R1 report trust root differs from completion")
    repository_state = _mapping(
        report.get("repository_state"),
        name="R1 repository state",
    )
    if repository_state != {
        "start_head": git_head,
        "end_head": git_head,
        "start_clean": True,
        "end_clean": True,
    }:
        raise ValueError("R1 report is not bound to one clean stable git HEAD")
    expected_command_names = {
        "r1_fresh_environment_resolution",
        "r1_package_import",
        "r1_locked_config",
        "r1_cli_help",
        "r1_ruff",
        "r1_mypy",
        "r2_targeted_pytest",
        "r2_full_pytest",
        "r3_synthetic_probe",
    }
    commands = report.get("commands")
    if not isinstance(commands, list):
        raise ValueError("R1 commands must be a list")
    command_mappings = [_mapping(command, name="R1 command") for command in commands]
    if (
        len(command_mappings) != len(expected_command_names)
        or {command.get("name") for command in command_mappings}
        != expected_command_names
        or any(command.get("exit_code") != 0 for command in command_mappings)
    ):
        raise ValueError("R1 commands are missing, duplicated, or failed")
    config = _mapping(report.get("config"), name="R1 config")
    if config.get("sha256") != trust_root.get("config_sha256"):
        raise ValueError("R1 config hash differs from trust root")
    toolchain = _mapping(report.get("toolchain"), name="R1 toolchain")
    for name in ("python", "uv", "pytest", "ruff", "mypy"):
        tool = _mapping(toolchain.get(name), name=f"R1 {name} toolchain")
        if tool.get("sha256") != trust_root.get(f"{name}_executable_sha256"):
            raise ValueError(f"R1 {name} hash differs from trust root")
    report_hashes = _mapping(
        report.get("output_sha256"),
        name="R1 output hashes",
    )
    if set(report_hashes) != set(_VERIFICATION_REPORT_OUTPUT_FILES):
        raise ValueError("R1 report output hashes are incomplete")
    for relative_path in _VERIFICATION_REPORT_OUTPUT_FILES:
        if report_hashes.get(relative_path) != output_hashes.get(relative_path):
            raise ValueError(f"R1 output SHA-256 for {relative_path} is inconsistent")
    report_content_digest = report.get("report_content_sha256")
    if not isinstance(report_content_digest, str) or not _SHA256_PATTERN.fullmatch(
        report_content_digest
    ):
        raise ValueError("R1 report content digest is invalid")
    unsigned_report = dict(report)
    del unsigned_report["report_content_sha256"]
    if _canonical_digest(unsigned_report) != report_content_digest:
        raise ValueError("R1 report content digest does not match report")
    package_versions = _mapping(
        report.get("package_versions"),
        name="R1 package versions",
    )
    if package_versions.get("torch") != "2.10.0":
        raise ValueError("R1 report must record torch 2.10.0")
    if package_versions.get("transformers") != "4.57.6":
        raise ValueError("R1 report must record transformers 4.57.6")

    resolution = _json_mapping_file(
        content_root / "r1" / "fresh-environment-resolution.json",
        name="fresh environment resolution",
    )
    _validate_fresh_environment_resolution(resolution)
    probe = _json_mapping_file(
        content_root / "r3" / "synthetic_probe.json",
        name="R3 synthetic probe",
    )
    validate_synthetic_probe(probe)
    algebra = _json_mapping_file(
        content_root / "r2" / "algebra_probe.json",
        name="R2 algebra probe",
    )
    if algebra != probe.get("algebra"):
        raise ValueError("R2 algebra probe differs from R3 algebra payload")
    try:
        pytest_output = (content_root / "r2" / "pytest.txt").read_text(encoding="utf-8")
    except OSError as error:
        raise ValueError("R2 pytest output must be readable") from error
    if pytest_output.count("exit_code=0") != 2:
        raise ValueError("R2 pytest output must record two passing commands")


def build_verification_completion(
    staging_root: Path | str,
    *,
    artifact_root: Path | str,
    git_head: str,
    trust_root: dict[str, object],
    published_at: str,
) -> dict[str, object]:
    """Build and validate the completion marker for one staged R1-R3 bundle."""
    staging = Path(staging_root).resolve()
    destination = Path(artifact_root).resolve()
    if not re.fullmatch(r"[0-9a-f]{40}", git_head):
        raise ValueError("git_head must be a lowercase 40-character hash")
    if not isinstance(published_at, str) or not published_at.strip():
        raise ValueError("published_at must be a non-empty timestamp")
    validated_trust_root = _validated_trust_root(trust_root, git_head=git_head)
    if _verification_files(staging) != set(_VERIFICATION_RELATIVE_FILES):
        raise ValueError("staged verification bundle has missing or stale files")
    output_hashes = {
        relative_path: sha256_file(staging / relative_path)
        for relative_path in _VERIFICATION_RELATIVE_FILES
    }
    completion: dict[str, object] = {
        "schema_version": 1,
        "status": "passed",
        "git_head": git_head,
        "published_at": published_at,
        "artifact_root": str(destination),
        "trust_root": validated_trust_root,
        "artifact_paths": {
            relative_path: str(destination / relative_path)
            for relative_path in _VERIFICATION_RELATIVE_FILES
        },
        "output_sha256": output_hashes,
    }
    _validate_verification_content(
        staging,
        artifact_root=destination,
        completion=completion,
        expected_git_head=git_head,
    )
    return completion


def _fsync_directory_path(directory: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(directory, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def publish_verification_bundle(
    staging_root: Path | str,
    artifact_root: Path | str,
    completion: dict[str, object],
) -> None:
    """Publish exact R1-R3 directories, making completion authoritative last."""
    staging = Path(staging_root).resolve()
    destination = Path(artifact_root).resolve()
    if staging.parent != destination.parent:
        raise ValueError("verification staging directory must be a sibling of output")
    _validate_verification_content(
        staging,
        artifact_root=destination,
        completion=completion,
        expected_git_head=cast(str, completion.get("git_head")),
    )
    destination.mkdir(parents=True, exist_ok=True)
    completion_path = destination / "completion.json"
    completion_path.unlink(missing_ok=True)
    _fsync_directory_path(destination)
    for directory_name in _VERIFICATION_RUNG_DIRECTORIES:
        source = staging / directory_name
        target = destination / directory_name
        if target.is_symlink() or target.is_file():
            target.unlink()
        elif target.is_dir():
            shutil.rmtree(target)
        os.replace(source, target)
        _fsync_directory_path(destination)
    atomic_write_json(completion_path, completion)
    validate_verification_bundle(
        destination,
        expected_git_head=cast(str, completion.get("git_head")),
    )


def validate_verification_bundle(
    artifact_root: Path | str,
    *,
    expected_git_head: str | None = None,
) -> dict[str, object]:
    """Validate the authoritative completion marker and every published byte."""
    destination = Path(artifact_root).resolve()
    completion_path = destination / "completion.json"
    if not completion_path.is_file() or completion_path.is_symlink():
        raise ValueError("verification completion marker is missing")
    completion = _json_mapping_file(
        completion_path,
        name="verification completion",
    )
    _validate_verification_content(
        destination,
        artifact_root=destination,
        completion=completion,
        expected_git_head=expected_git_head,
    )
    return completion


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write deterministic outcome-blind R2-R3 probe artifacts."
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--algebra-output", type=Path)
    return parser.parse_args()


def main() -> None:
    """Command-line entry point used by the offline verification script."""
    arguments = _parse_arguments()
    payload = write_synthetic_probe(
        arguments.output,
        algebra_output=arguments.algebra_output,
    )
    print(
        json.dumps(
            {
                "deterministic_digest": payload["deterministic_digest"],
                "output": str(arguments.output),
                "status": "ok",
            },
            allow_nan=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
