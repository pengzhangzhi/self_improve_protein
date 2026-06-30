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
from self_improve_protein.provenance import derive_seed, sha256_bytes
from self_improve_protein.ridge import labeled_gradient_hessian, squared_loss
from self_improve_protein.selection import stable_top_k


def _module() -> ModuleType:
    return importlib.import_module("self_improve_protein.locality")


def _hash(prefix: str, index: int) -> str:
    return sha256_bytes(f"{prefix}:{index}".encode())


@lru_cache(maxsize=1)
def _case() -> tuple[Protocol, FitInputs, EvaluationLabels]:
    protocol = load_protocol(Path("configs/v0.yaml"))
    rng = np.random.Generator(np.random.PCG64(20260630))
    width = 5
    x_l = rng.normal(size=(protocol.n_labeled, width))
    x_u = rng.normal(size=(protocol.n_unlabeled, width))
    x_test = rng.normal(size=(protocol.n_test, width))
    beta = np.asarray([0.8, -0.5, 0.25, 0.6, -0.2])
    teacher_beta = beta + np.asarray([0.2, -0.1, 0.08, 0.04, -0.05])
    y_l = x_l @ beta + 0.12 * rng.normal(size=protocol.n_labeled)
    y_u = x_u @ beta + 0.12 * rng.normal(size=protocol.n_unlabeled)
    y_test = x_test @ beta + 0.12 * rng.normal(size=protocol.n_test)
    z_l = x_l @ teacher_beta + 0.18 * rng.normal(size=protocol.n_labeled)
    z_u = x_u @ teacher_beta + 0.18 * rng.normal(size=protocol.n_unlabeled)
    z_test = x_test @ teacher_beta + 0.18 * rng.normal(size=protocol.n_test)
    common: dict[str, Any] = {
        "assay_id": "SYNTHETIC_LOCALITY",
        "seed": 3,
        "source_digest": canonical_source_digest(protocol),
        "labeled_hashes": tuple(
            _hash("locality-labeled", index) for index in range(protocol.n_labeled)
        ),
        "unlabeled_hashes": tuple(
            _hash("locality-unlabeled", index) for index in range(protocol.n_unlabeled)
        ),
        "test_hashes": tuple(
            _hash("locality-test", index) for index in range(protocol.n_test)
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


def test_locality_card_constants_and_public_signatures_are_frozen() -> None:
    module = _module()

    assert module.CARD_ID == "pseudo_perturbation_locality_v1"
    assert module.CARD_SHA == (
        "e99aba9fe582499d9b4244281b99340a2150f37583cd438b0855b8afe2e7a613"
    )
    assert module.Q_VALUES == (24, 48, 72, 96, 192)
    assert module.W_VALUES == (0.01, 0.03, 0.10)
    assert module.SELECTORS == ("random", "full", "crossfit", "no_hessian")
    assert module.RANDOM_PURPOSE == "locality_random_order_v1"
    assert tuple(signature(module.fit_locality_task).parameters) == (
        "inputs",
        "protocol",
    )
    assert tuple(signature(module.canonical_locality_fit_digest).parameters) == (
        "result",
    )
    assert tuple(signature(module.evaluate_locality_task).parameters) == (
        "fit",
        "labels",
        "protocol",
        "expected_fit_digest",
        "expected_evaluation_digest",
    )


def test_fit_builds_one_complete_ordering_and_nested_prefix_grid() -> None:
    module = _module()
    protocol, inputs, _ = _case()

    result = module.fit_locality_task(inputs, protocol)

    assert result.assay_id == inputs.assay_id
    assert result.seed == inputs.seed
    assert result.source_digest == inputs.source_digest
    assert result.protocol_digest == result.base_fit.protocol_digest
    assert tuple(ordering.selector for ordering in result.orderings) == module.SELECTORS
    for ordering in result.orderings:
        assert len(ordering.ordered_indices) == protocol.n_unlabeled
        assert set(ordering.ordered_indices) == set(range(protocol.n_unlabeled))
        assert ordering.ordered_hashes == tuple(
            inputs.unlabeled_hashes[index] for index in ordering.ordered_indices
        )
    random_order = next(
        ordering for ordering in result.orderings if ordering.selector == "random"
    )
    expected_random = np.random.Generator(
        np.random.PCG64(
            derive_seed(inputs.assay_id, inputs.seed, module.RANDOM_PURPOSE)
        )
    ).permutation(protocol.n_unlabeled)
    assert random_order.ordered_indices == tuple(
        int(index) for index in expected_random
    )
    score_arrays = {
        "full": result.base_fit.full_scores,
        "crossfit": result.crossfit_scores,
        "no_hessian": result.base_fit.no_hessian_scores,
    }
    for selector, scores in score_arrays.items():
        ordering = next(item for item in result.orderings if item.selector == selector)
        expected = stable_top_k(
            scores,
            inputs.unlabeled_hashes,
            protocol.n_unlabeled,
        )
        assert ordering.ordered_indices == tuple(int(index) for index in expected)

    assert len(result.cells) == 60
    assert tuple(
        (cell.selector, cell.q, cell.pseudo_weight) for cell in result.cells
    ) == tuple(
        (selector, q, weight)
        for selector in module.SELECTORS
        for q in module.Q_VALUES
        for weight in module.W_VALUES
    )
    by_selector = {
        selector: next(
            ordering for ordering in result.orderings if ordering.selector == selector
        )
        for selector in module.SELECTORS
    }
    for cell in result.cells:
        assert (
            cell.selected_indices
            == by_selector[cell.selector].ordered_indices[: cell.q]
        )
        assert (
            cell.selected_hashes == by_selector[cell.selector].ordered_hashes[: cell.q]
        )
        assert cell.effective_pseudo_fraction == (
            cell.pseudo_weight
            * cell.q
            / (protocol.n_labeled + cell.pseudo_weight * cell.q)
        )
        for value in (
            cell.diagnostics.stationarity_residual,
            cell.diagnostics.predicted_labeled_loss_change,
            cell.diagnostics.realized_labeled_loss_change,
            cell.diagnostics.labeled_loss_prediction_error,
            cell.diagnostics.displacement_relative_error,
            cell.diagnostics.locality_index,
        ):
            assert np.isfinite(value)
    payload = result.to_payload()
    assert payload["card_id"] == module.CARD_ID
    assert "y_u" not in str(payload)
    assert "y_test" not in str(payload)


def test_locality_digest_reconstructs_all_hidden_label_free_claims() -> None:
    module = _module()
    protocol, inputs, _ = _case()
    first = module.fit_locality_task(inputs, protocol)
    second = module.fit_locality_task(inputs, protocol)

    digest = module.canonical_locality_fit_digest(first)

    assert len(digest) == 64
    assert digest == module.canonical_locality_fit_digest(second)
    assert "y_u" not in signature(module.fit_locality_task).parameters
    assert "y_test" not in signature(module.fit_locality_task).parameters
    for ordering in first.orderings:
        if ordering.scores is not None:
            assert not ordering.scores.flags.writeable
    for cell in first.cells:
        for array in (
            cell.selected_pseudo_labels,
            cell.coefficients,
            cell.test_predictions,
        ):
            assert not array.flags.writeable

    reversed_ordering = dataclasses.replace(
        first.orderings[0],
        ordered_indices=tuple(reversed(first.orderings[0].ordered_indices)),
        ordered_hashes=tuple(reversed(first.orderings[0].ordered_hashes)),
    )
    changed_cell = dataclasses.replace(
        first.cells[0],
        coefficients=first.cells[0].coefficients + 0.01,
    )
    changed_diagnostics = dataclasses.replace(
        first.cells[0].diagnostics,
        predicted_labeled_loss_change=(
            first.cells[0].diagnostics.predicted_labeled_loss_change + 0.1
        ),
    )
    mutations = (
        dataclasses.replace(first, base_fit_digest="f" * 64),
        dataclasses.replace(first, card_sha="f" * 64),
        dataclasses.replace(first, crossfit_scores=np.roll(first.crossfit_scores, 1)),
        dataclasses.replace(
            first,
            orderings=(reversed_ordering, *first.orderings[1:]),
        ),
        dataclasses.replace(first, cells=(changed_cell, *first.cells[1:])),
        dataclasses.replace(
            first,
            cells=(
                dataclasses.replace(first.cells[0], diagnostics=changed_diagnostics),
                *first.cells[1:],
            ),
        ),
    )
    for mutation in mutations:
        with pytest.raises(ValueError):
            module.canonical_locality_fit_digest(mutation)


def test_fit_rejects_protocols_that_change_the_frozen_v0_base() -> None:
    module = _module()
    protocol, inputs, _ = _case()
    payload = protocol.model_dump(mode="python")

    for changed in (
        {"q": 96},
        {"pseudo_weight": 0.03},
        {"ridge_lambda": 0.02},
        {"damping": 0.001},
        {"teacher_column": "another_teacher"},
    ):
        altered = Protocol.model_validate({**payload, **changed})
        with pytest.raises(ValueError, match="locality card"):
            module.fit_locality_task(inputs, altered)


def test_locality_result_dataclasses_reject_incomplete_factorial_state() -> None:
    module = _module()
    protocol, inputs, labels = _case()
    fit = module.fit_locality_task(inputs, protocol)
    evaluated = module.evaluate_locality_task(
        fit,
        labels,
        protocol=protocol,
        expected_fit_digest=module.canonical_locality_fit_digest(fit),
        expected_evaluation_digest=canonical_evaluation_digest(labels),
    )

    with pytest.raises(ValueError, match="orderings"):
        dataclasses.replace(fit, orderings=())
    with pytest.raises(ValueError, match="cells"):
        dataclasses.replace(fit, cells=fit.cells[:-1])
    with pytest.raises(ValueError, match="cells"):
        dataclasses.replace(evaluated, cells=())


def test_evaluate_locality_validates_digest_before_all_hidden_label_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    protocol, inputs, labels = _case()
    fit = module.fit_locality_task(inputs, protocol)
    fit_digest = module.canonical_locality_fit_digest(fit)
    evaluation_digest = canonical_evaluation_digest(labels)
    events: list[str] = []
    real_digest = module.canonical_locality_fit_digest
    real_evaluate = module.evaluate_task

    def tracked_digest(result: object) -> str:
        events.append("fit_digest")
        return str(real_digest(result))

    def tracked_evaluate(*args: object, **kwargs: object) -> object:
        events.append("hidden_evaluation")
        return real_evaluate(*args, **kwargs)

    monkeypatch.setattr(module, "canonical_locality_fit_digest", tracked_digest)
    monkeypatch.setattr(module, "evaluate_task", tracked_evaluate)

    result = module.evaluate_locality_task(
        fit,
        labels,
        protocol=protocol,
        expected_fit_digest=fit_digest,
        expected_evaluation_digest=evaluation_digest,
    )

    assert events[:2] == ["fit_digest", "hidden_evaluation"]
    assert result.assay_id == fit.assay_id
    assert result.seed == fit.seed
    assert len(result.reference.methods) == 5
    assert len(result.cells) == 60
    y_u = fit.base_fit.label_transform.transform(labels.y_u)
    y_test = fit.base_fit.label_transform.transform(labels.y_test)
    absolute_errors = np.abs(fit.base_fit.pseudo_labels_u - y_u)
    theta_zero = fit.base_fit.methods[0].coefficients
    gradient_l, hessian = labeled_gradient_hessian(
        fit.base_fit.x_l,
        fit.base_fit.y_l_standardized,
        theta_zero,
        fit.base_fit.ridge_lambda,
    )
    for artifact, evaluation in zip(fit.cells, result.cells, strict=True):
        assert (evaluation.selector, evaluation.q, evaluation.pseudo_weight) == (
            artifact.selector,
            artifact.q,
            artifact.pseudo_weight,
        )
        selected = np.asarray(artifact.selected_indices, dtype=np.int64)
        assert evaluation.spearman == pytest.approx(
            spearman_correlation(y_test, artifact.test_predictions)
        )
        assert evaluation.mse == pytest.approx(
            standardized_mse(y_test, artifact.test_predictions)
        )
        assert evaluation.ndcg_10pct == pytest.approx(
            ndcg_at_10_percent(y_test, artifact.test_predictions)
        )
        assert evaluation.selected_pseudo_label_mae == pytest.approx(
            float(np.mean(absolute_errors[selected]))
        )
        x_selected = fit.base_fit.x_u[selected]
        selected_gradient = (
            x_selected.T
            @ (x_selected @ theta_zero - fit.base_fit.pseudo_labels_u[selected])
            / artifact.q
        )
        predicted_displacement = -artifact.effective_pseudo_fraction * np.linalg.solve(
            hessian,
            selected_gradient - gradient_l,
        )
        gradient_test = (
            fit.base_fit.x_test.T
            @ (fit.base_fit.x_test @ theta_zero - y_test)
            / y_test.size
        )
        predicted_test_change = float(gradient_test @ predicted_displacement)
        realized_test_change = float(
            squared_loss(fit.base_fit.x_test, y_test, artifact.coefficients)
            - squared_loss(fit.base_fit.x_test, y_test, theta_zero)
        )
        assert evaluation.predicted_test_loss_change == pytest.approx(
            predicted_test_change
        )
        assert evaluation.realized_test_loss_change == pytest.approx(
            realized_test_change
        )
        assert evaluation.test_loss_prediction_error == pytest.approx(
            realized_test_change - predicted_test_change
        )
        assert evaluation.predicted_labeled_loss_change == (
            artifact.diagnostics.predicted_labeled_loss_change
        )
        assert evaluation.realized_labeled_loss_change == (
            artifact.diagnostics.realized_labeled_loss_change
        )

    full = next(cell for cell in result.cells if cell.selector == "full")
    candidate_difference = (
        fit.base_fit.x_u @ theta_zero - fit.base_fit.pseudo_labels_u
    )[:, None] * fit.base_fit.x_u - gradient_l
    test_gradient = (
        fit.base_fit.x_test.T
        @ (fit.base_fit.x_test @ theta_zero - y_test)
        / y_test.size
    )
    oracle_scores = candidate_difference @ np.linalg.solve(
        hessian + fit.base_fit.damping * np.eye(hessian.shape[0]),
        test_gradient,
    )
    assert full.test_oracle_score_alignment.defined
    assert full.test_oracle_score_alignment.value == pytest.approx(
        spearman_correlation(oracle_scores, fit.base_fit.full_scores)
    )
    assert full.test_oracle_score_vs_absolute_error.defined
    assert result.to_payload()["assay_id"] == fit.assay_id
