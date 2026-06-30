import dataclasses
import importlib
from collections.abc import Iterator
from contextlib import contextmanager
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
    canonical_fit_digest,
    canonical_source_digest,
    evaluate_task,
    fit_task,
)
from self_improve_protein.metrics import (
    ndcg_at_10_percent,
    spearman_correlation,
    standardized_mse,
)
from self_improve_protein.provenance import sha256_bytes
from self_improve_protein.ridge import fit_weighted_ridge, labeled_gradient_hessian
from self_improve_protein.selection import (
    balanced_fold_assignment,
    cross_fitted_influence_scores,
    out_of_fold_ridge_gradient,
    stable_top_k,
)


def _module() -> ModuleType:
    return importlib.import_module("self_improve_protein.crossfit")


def _hash(prefix: str, index: int) -> str:
    return sha256_bytes(f"{prefix}:{index}".encode())


def _case() -> tuple[Protocol, FitInputs, EvaluationLabels]:
    payload = load_protocol(Path("configs/v0.yaml")).model_dump(mode="python")
    payload.update(
        working_size=62,
        n_labeled=16,
        n_unlabeled=28,
        n_test=18,
        q=6,
        random_diagnostic_replicates=5,
    )
    protocol = Protocol.model_validate(payload)
    rng = np.random.Generator(np.random.PCG64(7319))
    width = 5
    x_l = rng.normal(size=(protocol.n_labeled, width))
    x_u = rng.normal(size=(protocol.n_unlabeled, width))
    x_test = rng.normal(size=(protocol.n_test, width))
    beta = np.array([0.8, -0.5, 0.25, 0.6, -0.2])
    y_l = x_l @ beta + 0.12 * rng.normal(size=protocol.n_labeled)
    y_u = x_u @ beta + 0.12 * rng.normal(size=protocol.n_unlabeled)
    y_test = x_test @ beta + 0.12 * rng.normal(size=protocol.n_test)
    teacher_beta = beta + np.array([0.2, -0.1, 0.08, 0.04, -0.05])
    z_l = x_l @ teacher_beta + 0.18 * rng.normal(size=protocol.n_labeled)
    z_u = x_u @ teacher_beta + 0.18 * rng.normal(size=protocol.n_unlabeled)
    z_test = x_test @ teacher_beta + 0.18 * rng.normal(size=protocol.n_test)
    common: dict[str, Any] = {
        "assay_id": "SYNTHETIC_CROSSFIT",
        "seed": 3,
        "source_digest": canonical_source_digest(protocol),
        "labeled_hashes": tuple(
            _hash("labeled", index) for index in range(protocol.n_labeled)
        ),
        "unlabeled_hashes": tuple(
            _hash("unlabeled", index) for index in range(protocol.n_unlabeled)
        ),
        "test_hashes": tuple(_hash("test", index) for index in range(protocol.n_test)),
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


def test_crossfit_card_constants_and_public_signatures_are_frozen() -> None:
    module = _module()

    assert module.CARD_ID == "crossfit_outer_gradient_v1"
    assert module.CARD_SHA == (
        "383afd7a5bae9c2ebd6768a112a82980236540fc0f66e3a294ef298961b8596f"
    )
    assert module.FOLD_PURPOSE == "crossfit_outer_folds_v1"
    assert module.FOLD_COUNT == 4
    assert tuple(signature(module.fit_crossfit_task).parameters) == (
        "inputs",
        "protocol",
    )
    assert tuple(signature(module.canonical_crossfit_fit_digest).parameters) == (
        "result",
    )
    assert tuple(signature(module.evaluate_crossfit_task).parameters) == (
        "fit",
        "labels",
        "protocol",
        "expected_crossfit_fit_digest",
        "expected_evaluation_digest",
    )


def test_fit_crossfit_reuses_locked_base_fit_and_exact_single_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    protocol, inputs, _ = _case()
    active = False
    calls = {"base": 0, "score": 0, "ridge": 0}
    real_base_fit = module.fit_task
    real_score = module.cross_fitted_influence_scores
    real_ridge = module.fit_weighted_ridge

    @contextmanager
    def tracked_scope() -> Iterator[None]:
        nonlocal active
        assert not active
        active = True
        try:
            yield
        finally:
            active = False

    def tracked_base_fit(*args: object, **kwargs: object) -> object:
        assert active
        calls["base"] += 1
        return real_base_fit(*args, **kwargs)

    def tracked_score(*args: object, **kwargs: object) -> object:
        assert active
        calls["score"] += 1
        return real_score(*args, **kwargs)

    def tracked_ridge(*args: object, **kwargs: object) -> object:
        assert active
        calls["ridge"] += 1
        return real_ridge(*args, **kwargs)

    monkeypatch.setattr(module, "_locked_blas_scope", tracked_scope)
    monkeypatch.setattr(module, "fit_task", tracked_base_fit)
    monkeypatch.setattr(module, "cross_fitted_influence_scores", tracked_score)
    monkeypatch.setattr(module, "fit_weighted_ridge", tracked_ridge)

    result = module.fit_crossfit_task(inputs, protocol)

    assert not active
    assert calls == {"base": 1, "score": 1, "ridge": 1}
    assert result.base_fit_digest == canonical_fit_digest(result.base_fit)
    expected_folds = balanced_fold_assignment(
        protocol.n_labeled,
        inputs.assay_id,
        inputs.seed,
        fold_count=module.FOLD_COUNT,
        purpose=module.FOLD_PURPOSE,
    )
    np.testing.assert_array_equal(result.fold_assignment, expected_folds)
    expected_outer = out_of_fold_ridge_gradient(
        result.base_fit.x_l,
        result.base_fit.y_l_standardized,
        expected_folds,
        protocol.ridge_lambda,
    )
    np.testing.assert_allclose(
        result.outer_gradient,
        expected_outer,
        atol=0.0,
        rtol=0.0,
    )
    supervised = result.base_fit.methods[0]
    expected_scores = cross_fitted_influence_scores(
        result.base_fit.x_l,
        result.base_fit.y_l_standardized,
        result.base_fit.x_u,
        result.base_fit.pseudo_labels_u,
        supervised.coefficients,
        protocol.ridge_lambda,
        protocol.damping,
        inputs.assay_id,
        inputs.seed,
        fold_count=module.FOLD_COUNT,
        purpose=module.FOLD_PURPOSE,
    )
    np.testing.assert_allclose(result.scores, expected_scores, atol=0.0, rtol=0.0)
    expected_selected = tuple(
        int(index)
        for index in stable_top_k(
            expected_scores,
            inputs.unlabeled_hashes,
            protocol.q,
        )
    )
    assert result.method.selected_indices == expected_selected
    selected = np.asarray(expected_selected, dtype=np.int64)
    expected_x = np.concatenate(
        [result.base_fit.x_l, result.base_fit.x_u[selected]], axis=0
    )
    expected_y = np.concatenate(
        [
            result.base_fit.y_l_standardized,
            result.base_fit.pseudo_labels_u[selected],
        ]
    )
    weights = np.concatenate(
        [
            np.ones(protocol.n_labeled),
            np.full(protocol.q, protocol.pseudo_weight),
        ]
    )
    expected_coefficients = fit_weighted_ridge(
        expected_x,
        expected_y,
        protocol.ridge_lambda,
        sample_weight=weights,
    )
    np.testing.assert_allclose(
        result.method.coefficients,
        expected_coefficients,
        atol=1e-14,
        rtol=1e-13,
    )
    np.testing.assert_allclose(
        result.method.test_predictions,
        result.base_fit.x_test @ expected_coefficients,
        atol=1e-14,
        rtol=1e-13,
    )
    denominator = float(np.sum(weights))
    stationarity = (
        expected_x.T
        @ (weights * (expected_x @ result.method.coefficients - expected_y))
        / denominator
        + protocol.ridge_lambda * result.method.coefficients
    )
    np.testing.assert_allclose(stationarity, 0.0, atol=2e-13, rtol=0.0)


def test_crossfit_fit_is_deterministic_hidden_label_free_and_read_only() -> None:
    module = _module()
    protocol, inputs, _ = _case()

    first = module.fit_crossfit_task(inputs, protocol)
    second = module.fit_crossfit_task(inputs, protocol)
    first_digest = module.canonical_crossfit_fit_digest(first)

    assert first_digest == module.canonical_crossfit_fit_digest(second)
    assert "y_u" not in signature(module.fit_crossfit_task).parameters
    assert "y_test" not in signature(module.fit_crossfit_task).parameters
    payload_text = str(first.to_payload())
    assert "y_u" not in payload_text
    assert "y_test" not in payload_text
    for array in (
        first.fold_assignment,
        first.outer_gradient,
        first.scores,
        first.method.selected_pseudo_labels,
        first.method.coefficients,
        first.method.test_predictions,
    ):
        assert not array.flags.writeable


def test_crossfit_digest_recomputes_and_rejects_every_derived_fit_claim() -> None:
    module = _module()
    protocol, inputs, _ = _case()
    result = module.fit_crossfit_task(inputs, protocol)
    module.canonical_crossfit_fit_digest(result)
    reversed_indices = tuple(reversed(result.method.selected_indices))
    reversed_method = dataclasses.replace(
        result.method,
        selected_indices=reversed_indices,
        selected_hashes=tuple(reversed(result.method.selected_hashes)),
        selected_pseudo_labels=result.method.selected_pseudo_labels[::-1],
    )
    changed_score_diagnostics = dataclasses.replace(
        result.diagnostics.score,
        mean=result.diagnostics.score.mean + 0.25,
    )
    mutations = (
        dataclasses.replace(result, base_fit_digest="f" * 64),
        dataclasses.replace(result, fold_assignment=result.fold_assignment[::-1]),
        dataclasses.replace(result, outer_gradient=result.outer_gradient + 0.01),
        dataclasses.replace(result, scores=np.roll(result.scores, 1)),
        dataclasses.replace(result, method=reversed_method),
        dataclasses.replace(
            result,
            method=dataclasses.replace(
                result.method,
                coefficients=result.method.coefficients + 0.01,
            ),
        ),
        dataclasses.replace(
            result,
            method=dataclasses.replace(
                result.method,
                test_predictions=result.method.test_predictions + 0.01,
            ),
        ),
        dataclasses.replace(
            result,
            diagnostics=dataclasses.replace(
                result.diagnostics,
                score=changed_score_diagnostics,
            ),
        ),
    )

    for mutation in mutations:
        with pytest.raises(ValueError):
            module.canonical_crossfit_fit_digest(mutation)


def test_crossfit_fit_diagnostics_cover_scores_overlaps_gradient_and_locality() -> None:
    module = _module()
    protocol, inputs, _ = _case()
    result = module.fit_crossfit_task(inputs, protocol)
    diagnostics = result.diagnostics

    assert diagnostics.score.unique_count > 1
    assert 0.0 <= diagnostics.score.positive_fraction <= 1.0
    for overlap in (
        diagnostics.overlap_random,
        diagnostics.overlap_top_teacher,
        diagnostics.overlap_full,
        diagnostics.overlap_no_hessian,
    ):
        assert 0.0 <= overlap <= 1.0
    assert diagnostics.outer_full_gradient_cosine.defined
    assert np.isfinite(diagnostics.outer_full_gradient_cosine.value)
    method = diagnostics.method
    for value in (
        method.stationarity_residual,
        method.first_order_outer_loss_change,
        method.realized_labeled_loss_change,
        method.displacement_cosine,
        method.displacement_relative_error,
        method.locality_index,
        method.normal_matrix.condition_number,
    ):
        assert np.isfinite(value)
    assert method.displacement_cosine_defined


def test_evaluate_crossfit_calls_reference_evaluation_and_matches_hidden_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    protocol, inputs, labels = _case()
    fit = module.fit_crossfit_task(inputs, protocol)
    crossfit_digest = module.canonical_crossfit_fit_digest(fit)
    evaluation_digest = canonical_evaluation_digest(labels)
    calls = 0
    real_evaluate = module.evaluate_task

    def tracked_evaluate(*args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        return real_evaluate(*args, **kwargs)

    monkeypatch.setattr(module, "evaluate_task", tracked_evaluate)
    before = module.canonical_crossfit_fit_digest(fit)

    result = module.evaluate_crossfit_task(
        fit,
        labels,
        protocol=protocol,
        expected_crossfit_fit_digest=crossfit_digest,
        expected_evaluation_digest=evaluation_digest,
    )

    assert calls == 1
    assert module.canonical_crossfit_fit_digest(fit) == before
    y_u = fit.base_fit.label_transform.transform(labels.y_u)
    y_test = fit.base_fit.label_transform.transform(labels.y_test)
    selected = np.asarray(fit.method.selected_indices, dtype=np.int64)
    expected_mae = float(
        np.mean(np.abs(fit.base_fit.pseudo_labels_u[selected] - y_u[selected]))
    )
    assert result.crossfit.spearman == pytest.approx(
        spearman_correlation(y_test, fit.method.test_predictions)
    )
    assert result.crossfit.mse == pytest.approx(
        standardized_mse(y_test, fit.method.test_predictions)
    )
    assert result.crossfit.ndcg_10pct == pytest.approx(
        ndcg_at_10_percent(y_test, fit.method.test_predictions)
    )
    assert result.crossfit.selected_pseudo_label_mae == pytest.approx(expected_mae)
    theta_zero = fit.base_fit.methods[0].coefficients
    g_l, hessian = labeled_gradient_hessian(
        fit.base_fit.x_l,
        fit.base_fit.y_l_standardized,
        theta_zero,
        fit.base_fit.ridge_lambda,
    )
    g_test = (
        fit.base_fit.x_test.T
        @ (fit.base_fit.x_test @ theta_zero - y_test)
        / y_test.size
    )
    candidate_difference = (
        fit.base_fit.x_u @ theta_zero - fit.base_fit.pseudo_labels_u
    )[:, None] * fit.base_fit.x_u - g_l
    oracle_scores = candidate_difference @ np.linalg.solve(
        hessian + fit.base_fit.damping * np.eye(hessian.shape[0]),
        g_test,
    )
    assert result.test_risk_oracle.score_alignment.defined
    assert result.test_risk_oracle.score_alignment.value == pytest.approx(
        spearman_correlation(oracle_scores, fit.scores)
    )
    assert result.test_risk_oracle.score_vs_absolute_error.defined
    assert result.test_risk_oracle.gradient_cosine_defined
    assert np.isfinite(result.test_risk_oracle.gradient_cosine)
    assert len(result.reference.methods) == 5


def test_evaluate_crossfit_requires_external_fit_and_evaluation_digests() -> None:
    module = _module()
    protocol, inputs, labels = _case()
    fit = module.fit_crossfit_task(inputs, protocol)
    crossfit_digest = module.canonical_crossfit_fit_digest(fit)
    evaluation_digest = canonical_evaluation_digest(labels)

    with pytest.raises(ValueError, match="crossfit fit digest"):
        module.evaluate_crossfit_task(
            fit,
            labels,
            protocol=protocol,
            expected_crossfit_fit_digest="f" * 64,
            expected_evaluation_digest=evaluation_digest,
        )
    with pytest.raises(ValueError, match="evaluation digest"):
        module.evaluate_crossfit_task(
            fit,
            labels,
            protocol=protocol,
            expected_crossfit_fit_digest=crossfit_digest,
            expected_evaluation_digest="f" * 64,
        )
    permuted = dataclasses.replace(labels, y_u=labels.y_u[::-1])
    with pytest.raises(ValueError, match="evaluation digest"):
        module.evaluate_crossfit_task(
            fit,
            permuted,
            protocol=protocol,
            expected_crossfit_fit_digest=crossfit_digest,
            expected_evaluation_digest=evaluation_digest,
        )


def test_crossfit_module_does_not_shadow_locked_reference_implementations() -> None:
    module = _module()

    assert module.fit_task is fit_task
    assert module.evaluate_task is evaluate_task
    assert not hasattr(module, "load_evaluation_labels")
    assert not hasattr(module, "load_data_manifest")
    assert all(
        name not in signature(module.fit_crossfit_task).parameters
        for name in ("y_u", "y_test", "labels")
    )
