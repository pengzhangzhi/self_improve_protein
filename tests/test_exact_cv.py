from __future__ import annotations

import dataclasses
import importlib
from functools import lru_cache
from inspect import signature
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest

from self_improve_protein.config import Protocol, load_protocol
from self_improve_protein.experiment import (
    EvaluationLabels,
    FitInputs,
    canonical_evaluation_digest,
    canonical_source_digest,
)
from self_improve_protein.metrics import (
    ndcg_at_10_percent,
    spearman_correlation,
    standardized_mse,
)
from self_improve_protein.provenance import sha256_bytes
from self_improve_protein.ridge import (
    fit_feature_transform,
    fit_label_transform,
    fit_teacher_calibration,
)
from self_improve_protein.selection import balanced_fold_assignment


def test_exact_cv_card_constants_and_public_signatures_are_frozen() -> None:
    module = importlib.import_module("self_improve_protein.exact_cv")

    assert module.CARD_ID == "exact_cv_greedy_v1"
    assert module.CARD_SHA == (
        "90f35965bd9a36320bc3d5553deb8ea241961cf35f1a887b714cba417e6a4c3a"
    )
    assert module.FOLD_COUNT == 4
    assert module.FOLD_PURPOSE == "exact_cv_folds_v1"
    assert module.PREFIX_COUNTS == (24, 48, 72, 96, 192)
    assert tuple(signature(module.fit_exact_cv_task).parameters) == (
        "inputs",
        "protocol",
    )
    assert tuple(signature(module.canonical_exact_cv_fit_digest).parameters) == (
        "result",
    )
    assert tuple(signature(module.evaluate_exact_cv_task).parameters) == (
        "fit",
        "labels",
        "protocol",
        "expected_fit_digest",
        "expected_evaluation_digest",
    )


def _tiny_folds(module: ModuleType) -> tuple[Any, ...]:
    rng = np.random.Generator(np.random.PCG64(8821))
    folds = []
    for fold_id in range(4):
        x_train = rng.normal(size=(5, 3))
        y_train = rng.normal(size=5)
        x_validation = rng.normal(size=(3, 3))
        y_validation = rng.normal(size=3)
        x_u = rng.normal(size=(6, 3))
        pseudo_labels_u = rng.normal(size=6)
        folds.append(
            module.ExactCVFoldInputs(
                fold_id=fold_id,
                x_train=x_train,
                y_train=y_train,
                x_validation=x_validation,
                y_validation=y_validation,
                x_u=x_u,
                pseudo_labels_u=pseudo_labels_u,
            )
        )
    return tuple(folds)


def _hash(prefix: str, index: int) -> str:
    return sha256_bytes(f"{prefix}:{index}".encode())


@lru_cache(maxsize=1)
def _case() -> tuple[Protocol, FitInputs, EvaluationLabels]:
    protocol = load_protocol(Path("configs/v0.yaml"))
    rng = np.random.Generator(np.random.PCG64(20260630))
    width = 3
    x_l = rng.normal(size=(protocol.n_labeled, width))
    x_u = rng.normal(size=(protocol.n_unlabeled, width))
    x_test = rng.normal(size=(protocol.n_test, width))
    beta = np.asarray([0.8, -0.5, 0.25])
    teacher_beta = beta + np.asarray([0.2, -0.1, 0.08])
    y_l = x_l @ beta + 0.12 * rng.normal(size=protocol.n_labeled)
    y_u = x_u @ beta + 0.12 * rng.normal(size=protocol.n_unlabeled)
    y_test = x_test @ beta + 0.12 * rng.normal(size=protocol.n_test)
    z_l = x_l @ teacher_beta + 0.18 * rng.normal(size=protocol.n_labeled)
    z_u = x_u @ teacher_beta + 0.18 * rng.normal(size=protocol.n_unlabeled)
    z_test = x_test @ teacher_beta + 0.18 * rng.normal(size=protocol.n_test)
    common: dict[str, Any] = {
        "assay_id": "SYNTHETIC_EXACT_CV",
        "seed": 3,
        "source_digest": canonical_source_digest(protocol),
        "labeled_hashes": tuple(
            _hash("exact-labeled", index) for index in range(protocol.n_labeled)
        ),
        "unlabeled_hashes": tuple(
            _hash("exact-unlabeled", index)
            for index in range(protocol.n_unlabeled)
        ),
        "test_hashes": tuple(
            _hash("exact-test", index) for index in range(protocol.n_test)
        ),
    }
    inputs = FitInputs(
        **common,
        x_l=x_l,
        y_l=y_l,
        z_l=z_l,
        x_u=x_u,
        z_u=z_u,
        x_test=x_test,
        z_test=z_test,
    )
    labels = EvaluationLabels(**common, y_u=y_u, y_test=y_test)
    return protocol, inputs, labels


def _direct_fold_state(
    fold: Any,
    selected: tuple[int, ...],
    *,
    q: int,
    pseudo_weight: float,
    ridge_lambda: float,
) -> tuple[np.ndarray, float]:
    denominator = fold.x_train.shape[0] + pseudo_weight * q
    gram = (
        fold.x_train.T @ fold.x_train
        + denominator * ridge_lambda * np.eye(fold.x_train.shape[1])
    )
    rhs = fold.x_train.T @ fold.y_train
    if selected:
        indices = np.asarray(selected, dtype=np.int64)
        gram = gram + pseudo_weight * (
            fold.x_u[indices].T @ fold.x_u[indices]
        )
        rhs = rhs + pseudo_weight * (
            fold.x_u[indices].T @ fold.pseudo_labels_u[indices]
        )
    coefficients = np.linalg.solve(gram, rhs)
    residual = fold.x_validation @ coefficients - fold.y_validation
    return coefficients, float(np.mean(residual * residual))


def test_greedy_utilities_and_order_match_brute_force_direct_refits() -> None:
    module = importlib.import_module("self_improve_protein.exact_cv")
    folds = _tiny_folds(module)
    hashes = tuple(sha256_bytes(f"candidate:{index}".encode()) for index in range(6))
    q = 4
    pseudo_weight = 0.2
    ridge_lambda = 0.3

    result = module.greedy_exact_cv_order(
        folds,
        hashes,
        q=q,
        pseudo_weight=pseudo_weight,
        ridge_lambda=ridge_lambda,
    )

    selected: tuple[int, ...] = ()
    for step in result.steps:
        before = tuple(
            _direct_fold_state(
                fold,
                selected,
                q=q,
                pseudo_weight=pseudo_weight,
                ridge_lambda=ridge_lambda,
            )[1]
            for fold in folds
        )
        candidates: list[tuple[float, str, int, tuple[float, ...]]] = []
        for candidate in range(6):
            if candidate in selected:
                continue
            after = tuple(
                _direct_fold_state(
                    fold,
                    (*selected, candidate),
                    q=q,
                    pseudo_weight=pseudo_weight,
                    ridge_lambda=ridge_lambda,
                )[1]
                for fold in folds
            )
            candidates.append(
                (float(np.mean(after)), hashes[candidate], candidate, after)
            )
        candidates.sort()
        expected_mean_after, _, expected_index, expected_after = candidates[0]
        expected_runner_up_gap = candidates[1][0] - expected_mean_after
        expected_reductions = tuple(
            left - right for left, right in zip(before, expected_after, strict=True)
        )
        assert step.selected_index == expected_index
        assert step.selected_hash == hashes[expected_index]
        np.testing.assert_allclose(step.fold_mse_before, before, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(
            step.fold_mse_after,
            expected_after,
            rtol=1e-12,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            step.fold_mse_reduction,
            expected_reductions,
            rtol=1e-12,
            atol=1e-12,
        )
        assert np.isclose(step.mean_mse_reduction, np.mean(expected_reductions))
        assert np.isclose(step.runner_up_mean_mse_gap, expected_runner_up_gap)
        selected += (expected_index,)

    assert result.ordered_indices == selected
    for fold, recursive, direct in zip(
        folds,
        result.final_recursive_coefficients,
        result.final_direct_coefficients,
        strict=True,
    ):
        expected, _ = _direct_fold_state(
            fold,
            selected,
            q=q,
            pseudo_weight=pseudo_weight,
            ridge_lambda=ridge_lambda,
        )
        np.testing.assert_allclose(recursive, expected, rtol=2e-12, atol=2e-12)
        np.testing.assert_allclose(direct, expected, rtol=2e-12, atol=2e-12)


def test_exact_ties_use_ascending_stable_hash_without_tolerance() -> None:
    module = importlib.import_module("self_improve_protein.exact_cv")
    x_u = np.asarray([[1.0, -0.5]] * 5)
    folds = tuple(
        module.ExactCVFoldInputs(
            fold_id=fold,
            x_train=np.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),
            y_train=np.asarray([0.5, -0.2, 0.3]),
            x_validation=np.asarray([[0.2, 0.4], [-0.3, 0.8]]),
            y_validation=np.asarray([0.1, -0.4]),
            x_u=x_u,
            pseudo_labels_u=np.full(5, 0.25),
        )
        for fold in range(4)
    )
    hashes = tuple(_hash("tie", index) for index in range(5))

    result = module.greedy_exact_cv_order(
        folds,
        hashes,
        q=4,
        pseudo_weight=0.1,
        ridge_lambda=0.01,
    )

    assert result.ordered_hashes == tuple(sorted(hashes)[:4])


def test_fit_uses_only_each_folds_72_training_labels_and_freezes_prefixes() -> None:
    module = importlib.import_module("self_improve_protein.exact_cv")
    protocol, inputs, _ = _case()

    result = module.fit_exact_cv_task(inputs, protocol)

    expected_assignment = balanced_fold_assignment(
        protocol.n_labeled,
        inputs.assay_id,
        inputs.seed,
        fold_count=module.FOLD_COUNT,
        purpose=module.FOLD_PURPOSE,
    )
    np.testing.assert_array_equal(result.fold_assignment, expected_assignment)
    assert result.base_fit_digest == module.canonical_fit_digest(result.base_fit)
    assert len(result.greedy.steps) == protocol.q
    assert len(set(result.greedy.ordered_indices)) == protocol.q
    assert result.greedy.ordered_hashes == tuple(
        inputs.unlabeled_hashes[index] for index in result.greedy.ordered_indices
    )
    assert result.greedy.regularizer_mass == 72 + protocol.pseudo_weight * protocol.q
    assert result.greedy.reanchor_steps == module.REANCHOR_STEPS
    assert tuple(prefix.q for prefix in result.prefixes) == module.PREFIX_COUNTS
    for prefix in result.prefixes:
        assert prefix.selected_indices == result.greedy.ordered_indices[: prefix.q]
        assert prefix.selected_hashes == result.greedy.ordered_hashes[: prefix.q]
        assert prefix.training_weight_sum == (
            protocol.n_labeled + protocol.pseudo_weight * prefix.q
        )

    for fold_artifact in result.folds:
        train = expected_assignment != fold_artifact.fold_id
        validation = ~train
        assert fold_artifact.training_indices == tuple(
            int(i) for i in np.flatnonzero(train)
        )
        assert fold_artifact.validation_indices == tuple(
            int(i) for i in np.flatnonzero(validation)
        )
        assert len(fold_artifact.training_indices) == 72
        assert len(fold_artifact.validation_indices) == 24
        expected_feature = fit_feature_transform(inputs.x_l[train])
        expected_label = fit_label_transform(inputs.y_l[train], ddof=0)
        expected_calibration = fit_teacher_calibration(
            inputs.z_l[train],
            expected_label.transform(inputs.y_l[train]),
        )
        np.testing.assert_allclose(
            fold_artifact.feature_transform.mean,
            expected_feature.mean,
        )
        assert fold_artifact.feature_transform.scale == expected_feature.scale
        assert fold_artifact.label_transform == expected_label
        assert fold_artifact.teacher_calibration == expected_calibration
        np.testing.assert_allclose(
            fold_artifact.final_recursive_coefficients,
            fold_artifact.final_direct_coefficients,
            rtol=1e-10,
            atol=1e-10,
        )
        assert fold_artifact.coefficient_relative_drift < 1e-10


def test_fold_zero_transforms_and_calibration_ignore_its_validation_rows() -> None:
    module = importlib.import_module("self_improve_protein.exact_cv")
    protocol, inputs, _ = _case()
    assignment = balanced_fold_assignment(
        protocol.n_labeled,
        inputs.assay_id,
        inputs.seed,
        fold_count=module.FOLD_COUNT,
        purpose=module.FOLD_PURPOSE,
    )
    original = module._prepare_folds(
        raw_x_l=inputs.x_l,
        raw_x_u=inputs.x_u,
        y_l_raw=inputs.y_l,
        z_l=inputs.z_l,
        z_u=inputs.z_u,
        assignment=assignment,
    )[0]
    held_out = assignment == 0
    changed_x_l = np.array(inputs.x_l, copy=True)
    changed_y_l = np.array(inputs.y_l, copy=True)
    changed_z_l = np.array(inputs.z_l, copy=True)
    changed_x_l[held_out] += 1000.0
    changed_y_l[held_out] -= 500.0
    changed_z_l[held_out] += 700.0
    changed = module._prepare_folds(
        raw_x_l=changed_x_l,
        raw_x_u=inputs.x_u,
        y_l_raw=changed_y_l,
        z_l=changed_z_l,
        z_u=inputs.z_u,
        assignment=assignment,
    )[0]

    np.testing.assert_array_equal(
        original.feature_transform.mean,
        changed.feature_transform.mean,
    )
    assert original.feature_transform.scale == changed.feature_transform.scale
    assert original.label_transform == changed.label_transform
    assert original.teacher_calibration == changed.teacher_calibration
    np.testing.assert_array_equal(
        original.inputs.x_train,
        changed.inputs.x_train,
    )
    np.testing.assert_array_equal(
        original.inputs.y_train,
        changed.inputs.y_train,
    )
    np.testing.assert_array_equal(
        original.inputs.pseudo_labels_u,
        changed.inputs.pseudo_labels_u,
    )
    assert not np.array_equal(
        original.inputs.x_validation,
        changed.inputs.x_validation,
    )
    assert not np.array_equal(
        original.inputs.y_validation,
        changed.inputs.y_validation,
    )


def test_every_prefix_satisfies_the_actual_full96_normal_equations() -> None:
    module = importlib.import_module("self_improve_protein.exact_cv")
    protocol, inputs, _ = _case()
    result = module.fit_exact_cv_task(inputs, protocol)

    for prefix in result.prefixes:
        selected = np.asarray(prefix.selected_indices, dtype=np.int64)
        x = np.concatenate([result.base_fit.x_l, result.base_fit.x_u[selected]])
        y = np.concatenate(
            [
                result.base_fit.y_l_standardized,
                result.base_fit.pseudo_labels_u[selected],
            ]
        )
        weights = np.concatenate(
            [
                np.ones(protocol.n_labeled),
                np.full(prefix.q, protocol.pseudo_weight),
            ]
        )
        denominator = float(np.sum(weights))
        residual = (
            x.T @ (weights * (x @ prefix.coefficients - y)) / denominator
            + protocol.ridge_lambda * prefix.coefficients
        )
        assert np.linalg.norm(residual) < 1e-10
        np.testing.assert_allclose(
            prefix.test_predictions,
            result.base_fit.x_test @ prefix.coefficients,
        )
    endpoint = result.prefixes[-1]
    assert np.array_equal(
        endpoint.selected_pseudo_labels,
        result.base_fit.pseudo_labels_u[
            np.asarray(endpoint.selected_indices, dtype=np.int64)
        ],
    )


def test_digest_reconstructs_fit_and_rejects_mutation() -> None:
    module = importlib.import_module("self_improve_protein.exact_cv")
    protocol, inputs, _ = _case()
    fit = module.fit_exact_cv_task(inputs, protocol)

    digest = module.canonical_exact_cv_fit_digest(fit)
    assert len(digest) == 64
    changed_step = dataclasses.replace(
        fit.greedy.steps[0],
        mean_mse_reduction=fit.greedy.steps[0].mean_mse_reduction + 0.01,
    )
    changed_greedy = dataclasses.replace(
        fit.greedy,
        steps=(changed_step, *fit.greedy.steps[1:]),
    )
    changed = dataclasses.replace(fit, greedy=changed_greedy)
    with pytest.raises(ValueError, match="greedy"):
        module.canonical_exact_cv_fit_digest(changed)


def test_evaluation_requires_frozen_digest_and_reports_exact_prefix_metrics() -> None:
    module = importlib.import_module("self_improve_protein.exact_cv")
    protocol, inputs, labels = _case()
    fit = module.fit_exact_cv_task(inputs, protocol)
    fit_digest = module.canonical_exact_cv_fit_digest(fit)
    evaluation_digest = canonical_evaluation_digest(labels)

    with pytest.raises(ValueError, match="fit digest"):
        module.evaluate_exact_cv_task(
            fit,
            labels,
            protocol=protocol,
            expected_fit_digest="f" * 64,
            expected_evaluation_digest=evaluation_digest,
        )
    evaluation = module.evaluate_exact_cv_task(
        fit,
        labels,
        protocol=protocol,
        expected_fit_digest=fit_digest,
        expected_evaluation_digest=evaluation_digest,
    )

    y_u = fit.base_fit.label_transform.transform(labels.y_u)
    y_test = fit.base_fit.label_transform.transform(labels.y_test)
    absolute_error = np.abs(fit.base_fit.pseudo_labels_u - y_u)
    assert tuple(item.q for item in evaluation.prefixes) == module.PREFIX_COUNTS
    for prefix, metrics in zip(fit.prefixes, evaluation.prefixes, strict=True):
        selected = np.asarray(prefix.selected_indices, dtype=np.int64)
        assert metrics.spearman == spearman_correlation(
            y_test,
            prefix.test_predictions,
        )
        assert metrics.mse == standardized_mse(y_test, prefix.test_predictions)
        assert metrics.ndcg_10pct == ndcg_at_10_percent(
            y_test,
            prefix.test_predictions,
        )
        assert metrics.selected_pseudo_label_mae == float(
            np.mean(absolute_error[selected])
        )
        assert metrics.fold_cv_mse == fit.greedy.steps[prefix.q - 1].mean_mse_after
    assert module.canonical_exact_cv_fit_digest(fit) == fit_digest


def test_fit_api_and_state_cannot_receive_hidden_candidate_or_test_outcomes() -> None:
    module = importlib.import_module("self_improve_protein.exact_cv")

    assert set(signature(module.fit_exact_cv_task).parameters) == {
        "inputs",
        "protocol",
    }
    fit_fields = {
        field.name for field in dataclasses.fields(module.ExactCVFitResult)
    }
    assert "y_u" not in fit_fields
    assert "y_test" not in {
        field.name for field in dataclasses.fields(module.ExactCVFitResult)
    }
    assert "y_u" not in {
        field.name for field in dataclasses.fields(module.ExactCVFoldInputs)
    }
    assert "y_test" not in {
        field.name for field in dataclasses.fields(module.ExactCVFoldInputs)
    }
