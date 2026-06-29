import dataclasses
import json
import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from self_improve_protein.config import Protocol, load_protocol
from self_improve_protein.experiment import (
    NUMERICAL_POLICY,
    EvaluationLabels,
    EvaluationResult,
    FitInputs,
    FitResult,
    MethodArtifact,
    _cosine_diagnostic,
    _matrix_diagnostics,
    canonical_evaluation_digest,
    canonical_fit_digest,
    canonical_protocol_digest,
    canonical_source_digest,
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
        source_digest=canonical_source_digest(protocol),
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


def _evaluate_frozen(
    fit: FitResult,
    labels: EvaluationLabels,
    protocol: Protocol,
    *,
    expected_fit_digest: str,
    expected_evaluation_digest: str,
) -> EvaluationResult:
    return evaluate_task(
        fit,
        labels,
        protocol=protocol,
        expected_fit_digest=expected_fit_digest,
        expected_evaluation_digest=expected_evaluation_digest,
    )


def test_protocol_and_source_digests_are_canonical_and_purpose_scoped() -> None:
    protocol = _protocol()
    same = Protocol.model_validate(protocol.model_dump(mode="python"))
    analysis_only = Protocol.model_validate(
        {
            **protocol.model_dump(mode="python"),
            "random_diagnostic_replicates": 8,
        }
    )
    changed_source = Protocol.model_validate(
        {
            **protocol.model_dump(mode="python"),
            "teacher_column": "OTHER_TEACHER",
        }
    )

    assert canonical_protocol_digest(protocol) == canonical_protocol_digest(same)
    assert canonical_source_digest(protocol) == canonical_source_digest(same)
    assert canonical_protocol_digest(protocol) != canonical_protocol_digest(
        analysis_only
    )
    assert canonical_source_digest(protocol) == canonical_source_digest(analysis_only)
    assert canonical_source_digest(protocol) != canonical_source_digest(changed_source)
    assert len(canonical_protocol_digest(protocol)) == 64
    assert len(canonical_source_digest(protocol)) == 64


def test_evaluation_digest_binds_identity_hash_order_and_hidden_label_bytes() -> None:
    _, _, labels = _case()
    original = canonical_evaluation_digest(labels)

    assert original == canonical_evaluation_digest(labels)
    assert original != canonical_evaluation_digest(
        dataclasses.replace(labels, y_u=labels.y_u[::-1])
    )
    reversed_hashes = tuple(reversed(labels.test_hashes))
    assert original != canonical_evaluation_digest(
        dataclasses.replace(labels, test_hashes=reversed_hashes)
    )


def test_locked_blas_policy_is_cross_thread_digest_and_restart_portable(
    tmp_path: Path,
) -> None:
    artifact_path = tmp_path / "fit.pkl"
    script = r"""
import json
import os
import pickle
from pathlib import Path

import numpy as np

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
from self_improve_protein.provenance import sha256_bytes

path = Path(os.environ["ARTIFACT_PATH"])
action = os.environ["ACTION"]
if action == "validate":
    with path.open("rb") as handle:
        fit, labels, protocol, fit_digest, evaluation_digest = pickle.load(handle)
else:
    data = load_protocol("configs/v0.yaml").model_dump(mode="python")
    data.update(
        working_size=388,
        n_labeled=96,
        n_unlabeled=192,
        n_test=100,
        q=32,
        random_diagnostic_replicates=3,
    )
    protocol = Protocol.model_validate(data)
    rng = np.random.Generator(np.random.PCG64(86420))
    width = 480
    x_l = rng.normal(size=(protocol.n_labeled, width))
    x_u = rng.normal(size=(protocol.n_unlabeled, width))
    x_test = rng.normal(size=(protocol.n_test, width))
    y_l = rng.normal(size=protocol.n_labeled)
    y_u = rng.normal(size=protocol.n_unlabeled)
    y_test = rng.normal(size=protocol.n_test)
    z_l = rng.normal(size=protocol.n_labeled)
    z_u = rng.normal(size=protocol.n_unlabeled)
    z_test = rng.normal(size=protocol.n_test)
    hashes = lambda prefix, count: tuple(
        sha256_bytes(f"{prefix}:{index}".encode()) for index in range(count)
    )
    common = dict(
        assay_id="THREAD_PORTABLE",
        seed=2,
        source_digest=canonical_source_digest(protocol),
        labeled_hashes=hashes("l", protocol.n_labeled),
        unlabeled_hashes=hashes("u", protocol.n_unlabeled),
        test_hashes=hashes("t", protocol.n_test),
    )
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
    fit = fit_task(inputs, protocol)
    fit_digest = canonical_fit_digest(fit)
    evaluation_digest = canonical_evaluation_digest(labels)
    if action == "fit":
        with path.open("wb") as handle:
            pickle.dump(
                (fit, labels, protocol, fit_digest, evaluation_digest),
                handle,
            )

evaluation = evaluate_task(
    fit,
    labels,
    protocol=protocol,
    expected_fit_digest=fit_digest,
    expected_evaluation_digest=evaluation_digest,
)
print(json.dumps({
    "digest": canonical_fit_digest(fit),
    "selections": [list(method.selected_indices) for method in fit.methods],
    "spearman": [method.spearman for method in evaluation.methods],
}, sort_keys=True))
"""

    def run(action: str, threads: int) -> dict[str, object]:
        environment = os.environ.copy()
        environment.update(
            ACTION=action,
            ARTIFACT_PATH=str(artifact_path),
            OPENBLAS_NUM_THREADS=str(threads),
            OMP_NUM_THREADS=str(threads),
            MKL_NUM_THREADS=str(threads),
            PYTHONPATH=str(Path("src").resolve()),
        )
        completed = subprocess.run(
            [sys.executable, "-c", script],
            check=True,
            capture_output=True,
            text=True,
            env=environment,
        )
        return json.loads(completed.stdout.strip().splitlines()[-1])

    fitted_under_eight = run("fit", 8)
    independently_fitted_under_one = run("fresh", 1)
    loaded_and_validated_under_one = run("validate", 1)

    assert fitted_under_eight == independently_fitted_under_one
    assert fitted_under_eight == loaded_and_validated_under_one


def test_fit_anchors_protocol_and_rejects_mismatched_source_identity() -> None:
    protocol, inputs, _ = _case()
    fit = fit_task(inputs, protocol)

    assert fit.protocol_digest == canonical_protocol_digest(protocol)
    assert fit.source_digest == canonical_source_digest(protocol)
    assert fit.numerical_policy == NUMERICAL_POLICY
    assert fit.to_payload()["protocol_digest"] == fit.protocol_digest
    assert fit.to_payload()["numerical_policy"] == NUMERICAL_POLICY
    with pytest.raises(ValueError, match="source_digest"):
        fit_task(dataclasses.replace(inputs, source_digest="b" * 64), protocol)


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
    expected_fit_digest = canonical_fit_digest(fit)
    expected_evaluation_digest = canonical_evaluation_digest(labels)
    permuted = dataclasses.replace(
        labels,
        y_u=labels.y_u[::-1],
        y_test=np.roll(labels.y_test, 5),
    )

    assert canonical_fit_digest(fit) == canonical_fit_digest(fit_task(inputs, protocol))
    assert canonical_fit_digest(fit) == canonical_fit_digest(fit)
    _evaluate_frozen(
        fit,
        labels,
        protocol,
        expected_fit_digest=expected_fit_digest,
        expected_evaluation_digest=expected_evaluation_digest,
    )
    with pytest.raises(ValueError, match="evaluation digest"):
        _evaluate_frozen(
            fit,
            permuted,
            protocol,
            expected_fit_digest=expected_fit_digest,
            expected_evaluation_digest=expected_evaluation_digest,
        )
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
    assert np.isfinite(diagnostics.full_score.mean_selected_method)
    assert np.isfinite(diagnostics.full_score.maximum_selected_method)
    assert np.isfinite(diagnostics.full_score.mean_selected_random)
    assert not hasattr(diagnostics.full_score, "mean_selected_ours")
    assert not hasattr(diagnostics.full_score, "maximum_selected_ours")
    assert diagnostics.calibration_labeled_spearman.defined
    assert diagnostics.calibration_labeled_spearman.value is not None
    for summary, raw_scores in (
        (diagnostics.teacher_scores_labeled, inputs.z_l),
        (diagnostics.teacher_scores_unlabeled, inputs.z_u),
        (diagnostics.teacher_scores_test, inputs.z_test),
    ):
        assert summary.finite_count == summary.count
        assert summary.finite_fraction == 1.0
        assert summary.variance == pytest.approx(float(np.var(raw_scores)))
    residual = diagnostics.teacher_student_unlabeled_residual
    assert residual.count == inputs.x_u.shape[0]
    assert residual.minimum <= residual.mean <= residual.maximum
    assert len(residual.quantiles) == 5
    assert 0.0 <= diagnostics.ours_top_teacher_overlap <= 1.0
    assert len(diagnostics.methods) == 5
    for method in diagnostics.methods:
        assert np.isfinite(method.stationarity_residual)
        assert np.isfinite(method.normal_matrix.condition_number)
        assert method.normal_matrix.data_numerical_rank <= inputs.x_l.shape[1]
        assert not hasattr(method.normal_matrix, "numerical_rank")
        assert 0.0 <= method.normal_matrix.effective_df <= inputs.x_l.shape[1]
        if method.name != "supervised":
            assert np.isfinite(method.first_order_labeled_loss_change)
            assert np.isfinite(method.realized_labeled_loss_change)
            assert method.displacement_cosine_defined
            assert np.isfinite(method.displacement_cosine)
            assert np.isfinite(method.displacement_relative_error)
            assert np.isfinite(method.locality_index)


def test_data_rank_uses_unregularized_centered_design_when_p_exceeds_n() -> None:
    rng = np.random.Generator(np.random.PCG64(48096))
    x = rng.normal(size=(96, 480))
    x -= x.mean(axis=0)
    diagnostics = _matrix_diagnostics(x, np.ones(96), ridge_lambda=0.01)

    assert diagnostics.data_numerical_rank <= 95
    assert diagnostics.minimum_eigenvalue > 0.0
    assert diagnostics.effective_df <= diagnostics.data_numerical_rank


def test_evaluate_task_hidden_diagnostics_do_not_mutate_fit() -> None:
    protocol, inputs, labels = _case(random_replicates=9)
    fit = fit_task(inputs, protocol)
    before = canonical_fit_digest(fit)
    expected_evaluation_digest = canonical_evaluation_digest(labels)

    evaluation = _evaluate_frozen(
        fit,
        labels,
        protocol,
        expected_fit_digest=before,
        expected_evaluation_digest=expected_evaluation_digest,
    )

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
    fit = fit_task(inputs, protocol)
    expected_fit_digest = canonical_fit_digest(fit)
    expected_evaluation_digest = canonical_evaluation_digest(labels)
    evaluation = _evaluate_frozen(
        fit,
        labels,
        protocol,
        expected_fit_digest=expected_fit_digest,
        expected_evaluation_digest=expected_evaluation_digest,
    )

    assert not evaluation.teacher_test_spearman.defined
    assert evaluation.teacher_test_spearman.value is None
    assert evaluation.teacher_test_spearman.reason == "constant_prediction"


def test_expected_fit_digest_is_strict_and_required_before_unblinding() -> None:
    protocol, inputs, labels = _case()
    fit = fit_task(inputs, protocol)
    expected_evaluation_digest = canonical_evaluation_digest(labels)

    with pytest.raises(ValueError, match="expected_fit_digest"):
        evaluate_task(
            fit,
            labels,
            protocol=protocol,
            expected_fit_digest="not-a-digest",
            expected_evaluation_digest=expected_evaluation_digest,
        )
    with pytest.raises(ValueError, match="fit digest"):
        evaluate_task(
            fit,
            labels,
            protocol=protocol,
            expected_fit_digest="f" * 64,
            expected_evaluation_digest=expected_evaluation_digest,
        )


def test_expected_evaluation_digest_is_strict_and_required_before_metrics() -> None:
    protocol, inputs, labels = _case()
    fit = fit_task(inputs, protocol)
    expected_fit_digest = canonical_fit_digest(fit)

    with pytest.raises(ValueError, match="expected_evaluation_digest"):
        evaluate_task(
            fit,
            labels,
            protocol=protocol,
            expected_fit_digest=expected_fit_digest,
            expected_evaluation_digest="not-a-digest",
        )
    with pytest.raises(ValueError, match="evaluation digest"):
        evaluate_task(
            fit,
            labels,
            protocol=protocol,
            expected_fit_digest=expected_fit_digest,
            expected_evaluation_digest="f" * 64,
        )


def test_external_digest_rejects_reordered_test_state_and_predictions() -> None:
    protocol, inputs, labels = _case()
    fit = fit_task(inputs, protocol)
    expected_fit_digest = canonical_fit_digest(fit)
    expected_evaluation_digest = canonical_evaluation_digest(labels)
    reversed_methods = tuple(
        dataclasses.replace(
            method,
            test_predictions=method.test_predictions[::-1],
        )
        for method in fit.methods
    )
    changed = dataclasses.replace(
        fit,
        x_test=fit.x_test[::-1],
        z_test=fit.z_test[::-1],
        teacher_predictions_test=fit.teacher_predictions_test[::-1],
        methods=reversed_methods,
    )

    with pytest.raises(ValueError, match="fit digest"):
        evaluate_task(
            changed,
            labels,
            protocol=protocol,
            expected_fit_digest=expected_fit_digest,
            expected_evaluation_digest=expected_evaluation_digest,
        )


def test_external_digest_rejects_global_card_and_joint_source_rewrites() -> None:
    protocol, inputs, labels = _case()
    fit = fit_task(inputs, protocol)
    expected_fit_digest = canonical_fit_digest(fit)
    expected_evaluation_digest = canonical_evaluation_digest(labels)
    changed_weight = fit.pseudo_weight + 0.05
    changed_methods = tuple(
        dataclasses.replace(
            method,
            pseudo_weight=0.0 if method.name == "supervised" else changed_weight,
            training_weight_sum=(
                float(fit.x_l.shape[0])
                if method.name == "supervised"
                else float(fit.x_l.shape[0] + changed_weight * fit.q)
            ),
        )
        for method in fit.methods
    )
    changed_card = dataclasses.replace(
        fit,
        pseudo_weight=changed_weight,
        methods=changed_methods,
    )
    rewritten_source = "b" * 64
    changed_source_fit = dataclasses.replace(fit, source_digest=rewritten_source)
    changed_source_labels = dataclasses.replace(
        labels,
        source_digest=rewritten_source,
    )
    changed_source_evaluation_digest = canonical_evaluation_digest(
        changed_source_labels
    )

    with pytest.raises(ValueError, match="fit digest"):
        evaluate_task(
            changed_card,
            labels,
            protocol=protocol,
            expected_fit_digest=expected_fit_digest,
            expected_evaluation_digest=expected_evaluation_digest,
        )
    with pytest.raises(ValueError, match="fit digest"):
        evaluate_task(
            changed_source_fit,
            changed_source_labels,
            protocol=protocol,
            expected_fit_digest=expected_fit_digest,
            expected_evaluation_digest=changed_source_evaluation_digest,
        )


def test_protocol_root_rejects_fake_protocol_digest_with_fresh_fit_digest() -> None:
    protocol, inputs, labels = _case()
    fit = fit_task(inputs, protocol)
    fake_protocol_digest = "f" * 64
    assert fake_protocol_digest != canonical_protocol_digest(protocol)
    changed_fit = dataclasses.replace(
        fit,
        protocol_digest=fake_protocol_digest,
    )
    changed_fit_digest = canonical_fit_digest(changed_fit)
    expected_evaluation_digest = canonical_evaluation_digest(labels)

    with pytest.raises(ValueError, match="protocol_digest"):
        evaluate_task(
            changed_fit,
            labels,
            protocol=protocol,
            expected_fit_digest=changed_fit_digest,
            expected_evaluation_digest=expected_evaluation_digest,
        )


def test_protocol_root_rejects_joint_source_rewrite_with_fresh_digests() -> None:
    protocol, inputs, labels = _case()
    fit = fit_task(inputs, protocol)
    rewritten_source = "b" * 64
    changed_fit = dataclasses.replace(fit, source_digest=rewritten_source)
    changed_labels = dataclasses.replace(labels, source_digest=rewritten_source)
    changed_fit_digest = canonical_fit_digest(changed_fit)
    changed_evaluation_digest = canonical_evaluation_digest(changed_labels)

    with pytest.raises(ValueError, match="source_digest"):
        evaluate_task(
            changed_fit,
            changed_labels,
            protocol=protocol,
            expected_fit_digest=changed_fit_digest,
            expected_evaluation_digest=changed_evaluation_digest,
        )


def test_protocol_root_rejects_relabelled_alternate_weight_fit() -> None:
    protocol, inputs, labels = _case()
    alternate_data = protocol.model_dump(mode="python")
    alternate_data["pseudo_weight"] = 0.2
    alternate_protocol = Protocol.model_validate(alternate_data)
    alternate_fit = fit_task(inputs, alternate_protocol)
    relabelled_fit = dataclasses.replace(
        alternate_fit,
        protocol_digest=canonical_protocol_digest(protocol),
    )
    relabelled_fit_digest = canonical_fit_digest(relabelled_fit)
    expected_evaluation_digest = canonical_evaluation_digest(labels)

    with pytest.raises(ValueError, match="pseudo_weight"):
        evaluate_task(
            relabelled_fit,
            labels,
            protocol=protocol,
            expected_fit_digest=relabelled_fit_digest,
            expected_evaluation_digest=expected_evaluation_digest,
        )


def test_fresh_digest_rejects_coefficient_and_matching_prediction_tamper() -> None:
    protocol, inputs, _ = _case()
    fit = fit_task(inputs, protocol)
    methods = list(fit.methods)
    changed_coefficients = methods[1].coefficients.copy()
    changed_coefficients[0] += 0.75
    methods[1] = dataclasses.replace(
        methods[1],
        coefficients=changed_coefficients,
        test_predictions=fit.x_test @ changed_coefficients,
    )
    changed = dataclasses.replace(fit, methods=tuple(methods))

    with pytest.raises(ValueError, match="normal equations"):
        canonical_fit_digest(changed)


@pytest.mark.parametrize("diagnostic_value", [float("nan"), 12345.0])
def test_fresh_digest_rejects_false_or_nonfinite_fit_diagnostics(
    diagnostic_value: float,
) -> None:
    protocol, inputs, _ = _case()
    fit = fit_task(inputs, protocol)
    changed_diagnostics = dataclasses.replace(
        fit.diagnostics,
        calibration_labeled_rmse=diagnostic_value,
    )

    with pytest.raises(ValueError, match="fit diagnostics"):
        canonical_fit_digest(dataclasses.replace(fit, diagnostics=changed_diagnostics))


def test_constant_labeled_teacher_has_explicit_undefined_calibration_rank() -> None:
    protocol, inputs, _ = _case()
    fit = fit_task(
        dataclasses.replace(inputs, z_l=np.ones_like(inputs.z_l)),
        protocol,
    )

    diagnostic = fit.diagnostics.calibration_labeled_spearman
    assert not diagnostic.defined
    assert diagnostic.value is None
    assert diagnostic.reason == "constant_prediction"
    assert fit.diagnostics.teacher_scores_labeled.variance == 0.0


def test_teacher_student_residual_distribution_matches_frozen_fit_state() -> None:
    protocol, inputs, _ = _case()
    fit = fit_task(inputs, protocol)
    residual = fit.x_u @ fit.methods[0].coefficients - fit.pseudo_labels_u
    summary = fit.diagnostics.teacher_student_unlabeled_residual

    assert summary.count == residual.size
    assert summary.minimum == pytest.approx(float(np.min(residual)))
    assert summary.maximum == pytest.approx(float(np.max(residual)))
    assert summary.mean == pytest.approx(float(np.mean(residual)))
    assert summary.standard_deviation == pytest.approx(float(np.std(residual)))
    np.testing.assert_allclose(
        summary.quantiles,
        np.quantile(residual, [0.0, 0.25, 0.5, 0.75, 1.0]),
    )


def test_zero_norm_displacement_cosine_is_undefined_not_zero() -> None:
    diagnostic = _cosine_diagnostic(np.zeros(3), np.ones(3))

    assert not diagnostic.defined
    assert diagnostic.value is None
    assert diagnostic.reason == "zero_norm"


@pytest.mark.parametrize("field", ["assay_id", "seed", "source_digest"])
def test_evaluation_provenance_mismatch_fails_closed(field: str) -> None:
    protocol, inputs, labels = _case()
    fit = fit_task(inputs, protocol)
    replacements: dict[str, object] = {
        "assay_id": "OTHER",
        "seed": 4,
        "source_digest": "b" * 64,
    }
    changed_labels = dataclasses.replace(
        labels,
        **{field: replacements[field]},
    )
    expected_fit_digest = canonical_fit_digest(fit)
    expected_evaluation_digest = canonical_evaluation_digest(changed_labels)

    with pytest.raises(ValueError, match=field):
        evaluate_task(
            fit,
            changed_labels,
            protocol=protocol,
            expected_fit_digest=expected_fit_digest,
            expected_evaluation_digest=expected_evaluation_digest,
        )


@pytest.mark.parametrize(
    "hash_field",
    ["labeled_hashes", "unlabeled_hashes", "test_hashes"],
)
def test_evaluation_split_hash_mismatch_fails_closed(hash_field: str) -> None:
    protocol, inputs, labels = _case()
    fit = fit_task(inputs, protocol)
    hashes = list(getattr(labels, hash_field))
    hashes[0] = "f" * 64
    changed_labels = dataclasses.replace(labels, **{hash_field: tuple(hashes)})
    expected_fit_digest = canonical_fit_digest(fit)
    expected_evaluation_digest = canonical_evaluation_digest(changed_labels)

    with pytest.raises(ValueError, match=hash_field):
        evaluate_task(
            fit,
            changed_labels,
            protocol=protocol,
            expected_fit_digest=expected_fit_digest,
            expected_evaluation_digest=expected_evaluation_digest,
        )


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
    expected_fit_digest = canonical_fit_digest(fit)
    expected_evaluation_digest = canonical_evaluation_digest(labels)
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

    with pytest.raises(ValueError, match="fit digest"):
        evaluate_task(
            dataclasses.replace(fit, methods=tuple(methods)),
            labels,
            protocol=protocol,
            expected_fit_digest=expected_fit_digest,
            expected_evaluation_digest=expected_evaluation_digest,
        )


def test_evaluation_rejects_tampered_teacher_calibration() -> None:
    protocol, inputs, labels = _case()
    fit = fit_task(inputs, protocol)
    expected_fit_digest = canonical_fit_digest(fit)
    expected_evaluation_digest = canonical_evaluation_digest(labels)
    changed_calibration = dataclasses.replace(
        fit.teacher_calibration,
        slope=fit.teacher_calibration.slope + 0.1,
    )

    with pytest.raises(ValueError, match="teacher calibration"):
        canonical_fit_digest(
            dataclasses.replace(fit, teacher_calibration=changed_calibration)
        )

    with pytest.raises(ValueError, match="fit digest"):
        evaluate_task(
            dataclasses.replace(fit, teacher_calibration=changed_calibration),
            labels,
            protocol=protocol,
            expected_fit_digest=expected_fit_digest,
            expected_evaluation_digest=expected_evaluation_digest,
        )


def test_evaluation_rejects_missing_random_diagnostic_replicate() -> None:
    protocol, inputs, labels = _case(random_replicates=5)
    fit = fit_task(inputs, protocol)
    expected_fit_digest = canonical_fit_digest(fit)
    expected_evaluation_digest = canonical_evaluation_digest(labels)
    changed = dataclasses.replace(
        fit,
        random_diagnostic_indices=fit.random_diagnostic_indices[:-1],
    )

    with pytest.raises(ValueError, match="fit digest"):
        evaluate_task(
            changed,
            labels,
            protocol=protocol,
            expected_fit_digest=expected_fit_digest,
            expected_evaluation_digest=expected_evaluation_digest,
        )


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
