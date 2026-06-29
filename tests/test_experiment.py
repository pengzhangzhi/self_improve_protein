import dataclasses
import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from self_improve_protein.config import Protocol, load_protocol
from self_improve_protein.experiment import (
    EvaluationLabels,
    FitInputs,
    FitResult,
    MethodArtifact,
    canonical_fit_digest,
    evaluate_task,
    fit_task,
)
from self_improve_protein.provenance import derive_seed, sha256_bytes

FloatArray = NDArray[np.float64]


def _hash(prefix: str, index: int) -> str:
    return sha256_bytes(f"{prefix}:{index}".encode())


def _protocol(*, random_replicates: int = 7) -> Protocol:
    data = load_protocol(Path("configs/v0.yaml")).model_dump(mode="python")
    data.update(
        working_size=62,
        n_labeled=16,
        n_unlabeled=28,
        n_test=18,
        q=6,
        random_diagnostic_replicates=random_replicates,
    )
    return Protocol.model_validate(data)


def _case(
    *,
    random_replicates: int = 7,
) -> tuple[Protocol, FitInputs, EvaluationLabels]:
    protocol = _protocol(random_replicates=random_replicates)
    rng = np.random.Generator(np.random.PCG64(20260629))
    width = 5
    x_l = rng.normal(size=(protocol.n_labeled, width))
    x_u = rng.normal(size=(protocol.n_unlabeled, width))
    x_test = rng.normal(size=(protocol.n_test, width))
    beta = np.array([1.1, -0.7, 0.3, 0.8, -0.4])
    y_l = x_l @ beta + 0.15 * rng.normal(size=protocol.n_labeled)
    y_u = x_u @ beta + 0.15 * rng.normal(size=protocol.n_unlabeled)
    y_test = x_test @ beta + 0.15 * rng.normal(size=protocol.n_test)
    teacher_beta = beta + np.array([0.25, -0.15, 0.1, 0.05, -0.1])
    z_l = x_l @ teacher_beta + 0.2 * rng.normal(size=protocol.n_labeled)
    z_u = x_u @ teacher_beta + 0.2 * rng.normal(size=protocol.n_unlabeled)
    z_test = x_test @ teacher_beta + 0.2 * rng.normal(size=protocol.n_test)
    labeled_hashes = tuple(_hash("l", index) for index in range(protocol.n_labeled))
    unlabeled_hashes = tuple(_hash("u", index) for index in range(protocol.n_unlabeled))
    test_hashes = tuple(_hash("t", index) for index in range(protocol.n_test))
    common = dict(
        assay_id="SYNTHETIC",
        seed=3,
        source_digest="a" * 64,
        labeled_hashes=labeled_hashes,
        unlabeled_hashes=unlabeled_hashes,
        test_hashes=test_hashes,
    )
    fit_inputs = FitInputs(
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
    return protocol, fit_inputs, labels


def _methods(fit: FitResult) -> dict[str, MethodArtifact]:
    return {method.name: method for method in fit.methods}


def test_fit_inputs_excludes_hidden_labels_and_copies_arrays_read_only() -> None:
    _, inputs, _ = _case()

    assert "y_u" not in {field.name for field in dataclasses.fields(FitInputs)}
    assert "y_test" not in {field.name for field in dataclasses.fields(FitInputs)}
    assert hasattr(inputs, "z_test")
    for name in ("x_l", "y_l", "z_l", "x_u", "z_u", "x_test", "z_test"):
        assert not getattr(inputs, name).flags.writeable

    original = np.arange(80, dtype=np.float64).reshape(16, 5)
    _, base, _ = _case()
    copied = dataclasses.replace(base, x_l=original)
    original[0, 0] = -999.0
    assert copied.x_l[0, 0] != -999.0
    with pytest.raises(ValueError, match="read-only"):
        copied.x_l[0, 0] = 0.0


def test_fit_task_produces_five_deterministic_exact_budget_methods() -> None:
    protocol, inputs, _ = _case()

    first = fit_task(inputs, protocol)
    second = fit_task(inputs, protocol)
    methods = _methods(first)

    assert tuple(methods) == (
        "supervised",
        "random",
        "top_teacher",
        "ours",
        "no_hessian",
    )
    assert methods["supervised"].selected_indices == ()
    for name in ("random", "top_teacher", "ours", "no_hessian"):
        method = methods[name]
        assert len(method.selected_indices) == protocol.q
        assert method.pseudo_weight == protocol.pseudo_weight
        np.testing.assert_array_equal(
            method.selected_pseudo_labels,
            first.pseudo_labels_u[np.asarray(method.selected_indices)],
        )
    assert canonical_fit_digest(first) == canonical_fit_digest(second)


def test_confirmatory_random_selection_uses_locked_stream() -> None:
    protocol, inputs, _ = _case()
    fit = fit_task(inputs, protocol)
    random_method = _methods(fit)["random"]
    expected = np.sort(
        np.random.Generator(
            np.random.PCG64(
                derive_seed(inputs.assay_id, inputs.seed, "random_selection")
            )
        ).choice(protocol.n_unlabeled, size=protocol.q, replace=False)
    )
    np.testing.assert_array_equal(random_method.selected_indices, expected)


def test_pseudo_methods_satisfy_exact_weighted_normal_equations() -> None:
    protocol, inputs, _ = _case()
    fit = fit_task(inputs, protocol)
    identity = np.eye(fit.x_l.shape[1])

    for method in fit.methods:
        if method.name == "supervised":
            x = fit.x_l
            y = fit.y_l_standardized
            weights = np.ones(protocol.n_labeled)
        else:
            selected = np.asarray(method.selected_indices)
            x = np.concatenate([fit.x_l, fit.x_u[selected]], axis=0)
            y = np.concatenate([fit.y_l_standardized, fit.pseudo_labels_u[selected]])
            weights = np.concatenate(
                [
                    np.ones(protocol.n_labeled),
                    np.full(protocol.q, protocol.pseudo_weight),
                ]
            )
        denominator = float(weights.sum())
        residual = (
            x.T @ (weights * (x @ method.coefficients - y))
            + denominator * protocol.ridge_lambda * identity @ method.coefficients
        )
        np.testing.assert_allclose(residual, 0.0, atol=2e-11, rtol=0.0)
        assert method.training_weight_sum == pytest.approx(denominator)


def test_hidden_label_permutation_cannot_change_canonical_fit_digest() -> None:
    protocol, inputs, labels = _case()
    fit = fit_task(inputs, protocol)
    permuted = dataclasses.replace(
        labels,
        y_u=labels.y_u[::-1],
        y_test=np.roll(labels.y_test, 5),
    )

    assert canonical_fit_digest(fit) == canonical_fit_digest(fit_task(inputs, protocol))
    assert canonical_fit_digest(fit) == canonical_fit_digest(fit)
    evaluate_task(fit, labels)
    evaluate_task(fit, permuted)
    assert canonical_fit_digest(fit) == canonical_fit_digest(fit)
    assert not np.array_equal(labels.y_u, permuted.y_u)
    assert not np.array_equal(labels.y_test, permuted.y_test)


def test_random_diagnostic_draws_are_counted_and_purpose_separated() -> None:
    protocol, inputs, _ = _case(random_replicates=11)
    fit = fit_task(inputs, protocol)
    confirmatory = _methods(fit)["random"].selected_indices

    assert len(fit.random_diagnostic_indices) == 11
    for replicate, indices in enumerate(fit.random_diagnostic_indices):
        expected_seed = derive_seed(
            inputs.assay_id,
            inputs.seed,
            f"random_diagnostic:{replicate}",
        )
        expected = np.sort(
            np.random.Generator(np.random.PCG64(expected_seed)).choice(
                protocol.n_unlabeled,
                size=protocol.q,
                replace=False,
            )
        )
        np.testing.assert_array_equal(indices, expected)
    assert all(
        derive_seed(
            inputs.assay_id,
            inputs.seed,
            f"random_diagnostic:{replicate}",
        )
        != derive_seed(inputs.assay_id, inputs.seed, "random_selection")
        for replicate in range(protocol.random_diagnostic_replicates)
    )
    assert _methods(fit)["random"].selected_indices == confirmatory


def test_fit_diagnostics_are_finite_and_cover_scores_geometry_and_locality() -> None:
    protocol, inputs, _ = _case()
    diagnostics = fit_task(inputs, protocol).diagnostics

    assert diagnostics.full_score.unique_count > 1
    assert 0.0 <= diagnostics.full_score.positive_fraction <= 1.0
    assert np.isfinite(diagnostics.full_score.mean_selected_ours)
    assert np.isfinite(diagnostics.full_score.mean_selected_random)
    assert 0.0 <= diagnostics.ours_top_teacher_overlap <= 1.0
    assert len(diagnostics.methods) == 5
    for method in diagnostics.methods:
        assert np.isfinite(method.stationarity_residual)
        assert np.isfinite(method.normal_matrix.condition_number)
        assert 0.0 <= method.normal_matrix.effective_df <= inputs.x_l.shape[1]
        if method.name != "supervised":
            assert np.isfinite(method.first_order_labeled_loss_change)
            assert np.isfinite(method.realized_labeled_loss_change)
            assert np.isfinite(method.displacement_cosine)
            assert np.isfinite(method.displacement_relative_error)
            assert np.isfinite(method.locality_index)


def test_evaluate_task_hidden_diagnostics_do_not_mutate_fit() -> None:
    protocol, inputs, labels = _case(random_replicates=9)
    fit = fit_task(inputs, protocol)
    before = canonical_fit_digest(fit)

    evaluation = evaluate_task(fit, labels)

    assert canonical_fit_digest(fit) == before
    assert tuple(metric.name for metric in evaluation.methods) == tuple(
        method.name for method in fit.methods
    )
    for metric in evaluation.methods:
        assert np.isfinite(metric.spearman)
        assert np.isfinite(metric.mse)
        assert np.isfinite(metric.ndcg_10pct)
        assert metric.name == "supervised" or np.isfinite(
            metric.selected_pseudo_label_mae
        )
    assert evaluation.pool_pseudo_label_mae > 0.0
    assert len(evaluation.random_error_reference) == 9
    assert evaluation.teacher_test_spearman.defined
    assert evaluation.teacher_test_spearman.value is not None
    assert evaluation.full_test_risk_oracle.score_alignment.defined
    assert evaluation.no_hessian_test_risk_oracle.score_alignment.defined
    assert np.isfinite(evaluation.full_test_risk_oracle.gradient_cosine)
    assert np.isfinite(evaluation.no_hessian_test_risk_oracle.gradient_cosine)
    assert evaluation.full_test_risk_oracle.score_vs_absolute_error.defined


def test_constant_teacher_test_prediction_is_explicitly_undefined() -> None:
    protocol, inputs, labels = _case()
    inputs = dataclasses.replace(inputs, z_test=np.ones(protocol.n_test))
    evaluation = evaluate_task(fit_task(inputs, protocol), labels)

    assert not evaluation.teacher_test_spearman.defined
    assert evaluation.teacher_test_spearman.value is None
    assert evaluation.teacher_test_spearman.reason == "constant_prediction"


@pytest.mark.parametrize("field", ["assay_id", "seed", "source_digest"])
def test_evaluation_provenance_mismatch_fails_closed(field: str) -> None:
    protocol, inputs, labels = _case()
    fit = fit_task(inputs, protocol)
    replacements: dict[str, object] = {
        "assay_id": "OTHER",
        "seed": 4,
        "source_digest": "b" * 64,
    }

    with pytest.raises(ValueError, match=field):
        evaluate_task(fit, dataclasses.replace(labels, **{field: replacements[field]}))


@pytest.mark.parametrize(
    "hash_field",
    ["labeled_hashes", "unlabeled_hashes", "test_hashes"],
)
def test_evaluation_split_hash_mismatch_fails_closed(hash_field: str) -> None:
    protocol, inputs, labels = _case()
    fit = fit_task(inputs, protocol)
    hashes = list(getattr(labels, hash_field))
    hashes[0] = "f" * 64

    with pytest.raises(ValueError, match=hash_field):
        evaluate_task(fit, dataclasses.replace(labels, **{hash_field: tuple(hashes)}))


@pytest.mark.parametrize(
    "tamper",
    [
        "selected_count",
        "selected_hash_correspondence",
        "pseudo_weight",
        "ridge_lambda",
        "training_weight_sum",
        "method_order",
    ],
)
def test_evaluation_rejects_tampered_method_card_fields(tamper: str) -> None:
    protocol, inputs, labels = _case()
    fit = fit_task(inputs, protocol)
    methods = list(fit.methods)
    random_method = methods[1]
    if tamper == "selected_count":
        methods[1] = dataclasses.replace(
            random_method,
            selected_indices=random_method.selected_indices[:-1],
            selected_hashes=random_method.selected_hashes[:-1],
            selected_pseudo_labels=random_method.selected_pseudo_labels[:-1],
        )
    elif tamper == "selected_hash_correspondence":
        hashes = list(random_method.selected_hashes)
        unselected = next(
            item for item in fit.unlabeled_hashes if item not in set(hashes)
        )
        hashes[0] = unselected
        methods[1] = dataclasses.replace(random_method, selected_hashes=tuple(hashes))
    elif tamper == "pseudo_weight":
        methods[1] = dataclasses.replace(
            random_method,
            pseudo_weight=random_method.pseudo_weight + 0.01,
        )
    elif tamper == "ridge_lambda":
        methods[1] = dataclasses.replace(
            random_method,
            ridge_lambda=random_method.ridge_lambda + 0.01,
        )
    elif tamper == "training_weight_sum":
        methods[1] = dataclasses.replace(
            random_method,
            training_weight_sum=random_method.training_weight_sum + 1.0,
        )
    elif tamper == "method_order":
        methods[1], methods[2] = methods[2], methods[1]
    else:  # pragma: no cover - exhaustive parameter guard
        raise AssertionError(tamper)

    with pytest.raises(ValueError, match=tamper.replace("_", "|")):
        evaluate_task(dataclasses.replace(fit, methods=tuple(methods)), labels)


def test_evaluation_rejects_tampered_teacher_calibration() -> None:
    protocol, inputs, labels = _case()
    fit = fit_task(inputs, protocol)
    changed_calibration = dataclasses.replace(
        fit.teacher_calibration,
        slope=fit.teacher_calibration.slope + 0.1,
    )

    with pytest.raises(ValueError, match="teacher calibration"):
        evaluate_task(
            dataclasses.replace(fit, teacher_calibration=changed_calibration),
            labels,
        )


def test_evaluation_rejects_missing_random_diagnostic_replicate() -> None:
    protocol, inputs, labels = _case(random_replicates=5)
    fit = fit_task(inputs, protocol)
    changed = dataclasses.replace(
        fit,
        random_diagnostic_indices=fit.random_diagnostic_indices[:-1],
    )

    with pytest.raises(ValueError, match="random diagnostic replicate count"):
        evaluate_task(changed, labels)


@pytest.mark.parametrize(
    ("field", "mutate", "message"),
    [
        ("x_l", lambda value: value[:, :-1], "feature width"),
        ("z_test", lambda value: value[:-1], "z_test"),
        ("y_l", lambda value: np.full_like(value, np.nan), "finite"),
        ("unlabeled_hashes", lambda value: value[:-1], "unlabeled_hashes"),
    ],
)
def test_fit_inputs_reject_invalid_shapes_values_and_hash_counts(
    field: str,
    mutate: Callable[[object], object],
    message: str,
) -> None:
    _, inputs, _ = _case()

    with pytest.raises(ValueError, match=message):
        dataclasses.replace(inputs, **{field: mutate(getattr(inputs, field))})


def test_fit_inputs_reject_duplicate_or_cross_split_hashes() -> None:
    _, inputs, _ = _case()
    duplicated = list(inputs.unlabeled_hashes)
    duplicated[1] = duplicated[0]
    with pytest.raises(ValueError, match="unique"):
        dataclasses.replace(inputs, unlabeled_hashes=tuple(duplicated))

    overlapping = list(inputs.test_hashes)
    overlapping[0] = inputs.labeled_hashes[0]
    with pytest.raises(ValueError, match="disjoint"):
        dataclasses.replace(inputs, test_hashes=tuple(overlapping))


def test_constant_primary_prediction_fails_fit() -> None:
    protocol, inputs, _ = _case()
    constant_test = np.ones_like(inputs.x_test)

    with pytest.raises(ValueError, match="constant primary prediction"):
        fit_task(dataclasses.replace(inputs, x_test=constant_test), protocol)


def test_evaluation_labels_are_separate_immutable_defensive_copies() -> None:
    _, _, labels = _case()
    original = np.arange(labels.y_u.size, dtype=np.float64)
    copied = dataclasses.replace(labels, y_u=original)
    original[0] = -99.0
    assert copied.y_u[0] != -99.0
    assert not copied.y_u.flags.writeable
    assert {field.name for field in dataclasses.fields(EvaluationLabels)} >= {
        "y_u",
        "y_test",
    }


def test_canonical_fit_payload_contains_no_hidden_or_evaluation_fields() -> None:
    protocol, inputs, _ = _case()
    fit = fit_task(inputs, protocol)
    serialized = json.dumps(fit.to_payload(), sort_keys=True)

    for forbidden in (
        '"y_u"',
        '"y_test"',
        '"spearman"',
        '"ndcg"',
        '"pseudo_label_mae"',
        '"oracle"',
    ):
        assert forbidden not in serialized
