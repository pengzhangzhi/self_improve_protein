import copy
import json
import re
from pathlib import Path
from typing import cast

import pytest

from self_improve_protein.probes import (
    CAUSAL_DERIVATIVE_ERROR_TOLERANCE,
    CAUSAL_EPSILON,
    LEARNABILITY_PARAMETER_ERROR_TOLERANCE,
    LEARNABILITY_RIDGE_LAMBDA,
    LEARNABILITY_TRAIN_MSE_TOLERANCE,
    run_synthetic_probe,
    validate_synthetic_probe,
    write_synthetic_probe,
)

JsonObject = dict[str, object]


def _object(value: object) -> JsonObject:
    assert isinstance(value, dict)
    return cast(JsonObject, value)


def _float_list(value: object) -> list[float]:
    assert isinstance(value, list)
    assert all(isinstance(item, float) for item in value)
    return cast(list[float], value)


def _string_list(value: object) -> list[str]:
    assert isinstance(value, list)
    assert all(isinstance(item, str) for item in value)
    return cast(list[str], value)


def test_seeded_noiseless_ridge_probe_is_learnable_at_declared_tolerances() -> None:
    payload = run_synthetic_probe()
    learnability = _object(payload["learnability"])

    assert learnability["ridge_lambda"] == LEARNABILITY_RIDGE_LAMBDA
    assert learnability["sample_count"] > learnability["feature_count"]
    assert learnability["matrix_rank"] == learnability["feature_count"]
    assert learnability["x_dtype"] == "float64"
    assert learnability["y_dtype"] == "float64"
    assert learnability["theta_dtype"] == "float64"
    assert cast(float, learnability["train_mse"]) < (
        LEARNABILITY_TRAIN_MSE_TOLERANCE
    )
    assert cast(float, learnability["parameter_error_norm"]) < (
        LEARNABILITY_PARAMETER_ERROR_TOLERANCE
    )
    assert cast(float, learnability["normal_equation_residual_norm"]) < 1e-12


def test_external_teacher_probe_matches_first_order_sign_and_order() -> None:
    payload = run_synthetic_probe()
    causal = _object(payload["causal_score"])
    scores = _float_list(causal["full_h_scores"])
    predicted = _float_list(causal["predicted_loss_changes"])
    realized = _float_list(causal["realized_loss_changes"])
    derivative_errors = _float_list(causal["derivative_errors"])
    teacher = _object(causal["external_teacher"])

    assert causal["epsilon"] == CAUSAL_EPSILON
    assert teacher["kind"] == "fixed_nonidentical_linear_teacher"
    assert cast(float, teacher["parameter_distance_norm"]) > 1.0
    assert cast(float, teacher["pseudo_residual_norm"]) > 1.0
    assert len(set(scores)) == len(scores)
    assert min(scores) < 0.0 < max(scores)
    assert all(
        abs(error) <= CAUSAL_DERIVATIVE_ERROR_TOLERANCE
        for error in derivative_errors
    )
    assert cast(bool, causal["first_order_signs_match"])
    assert cast(bool, causal["first_order_order_matches"])
    assert sorted(range(len(scores)), key=scores.__getitem__, reverse=True) == sorted(
        range(len(realized)), key=realized.__getitem__
    )
    assert all(
        prediction * observation > 0.0
        for prediction, observation in zip(predicted, realized, strict=True)
    )
    assert cast(float, causal["no_h_formula_max_abs_error"]) < 1e-12
    assert cast(float, causal["perturbed_normal_equation_max_residual"]) < 1e-12


def test_literal_self_teacher_is_a_strict_negative_constant_control() -> None:
    payload = run_synthetic_probe()
    control = _object(_object(payload["causal_score"])["self_teacher_control"])
    scores = _float_list(control["full_h_scores"])

    assert max(scores) < 0.0
    assert cast(float, control["score_range"]) < 1e-12
    assert cast(float, control["expected_constant_max_abs_error"]) < 1e-12
    assert cast(float, control["pseudo_gradient_max_norm"]) < 1e-12


def test_probe_is_json_safe_deterministic_and_saves_selection_hashes() -> None:
    first = run_synthetic_probe()
    second = run_synthetic_probe()
    validate_synthetic_probe(first)

    assert first == second
    assert first["deterministic_digest"] == second["deterministic_digest"]
    assert json.loads(json.dumps(first, allow_nan=False, sort_keys=True)) == first

    repeat = _object(first["repeat_verification"])
    assert repeat["run_count"] == 2
    assert repeat["digests_match"] is True
    assert len(set(_string_list(repeat["run_digests"]))) == 1

    selection = _object(_object(first["causal_score"])["selection"])
    all_hashes = _string_list(selection["candidate_hashes"])
    assert all(re.fullmatch(r"[0-9a-f]{64}", item) for item in all_hashes)
    assert len(set(all_hashes)) == len(all_hashes)
    for field in ("full_h_selected_hashes", "no_h_selected_hashes"):
        selected = _string_list(selection[field])
        assert len(selected) == 3
        assert set(selected) <= set(all_hashes)
    for field in ("full_h_selection_digest", "no_h_selection_digest"):
        assert re.fullmatch(r"[0-9a-f]{64}", cast(str, selection[field]))


def test_probe_writer_atomically_emits_valid_r3_and_r2_payloads(
    tmp_path: Path,
) -> None:
    r3_path = tmp_path / "r3" / "synthetic_probe.json"
    r2_path = tmp_path / "r2" / "algebra_probe.json"

    payload = write_synthetic_probe(r3_path, algebra_output=r2_path)

    loaded = json.loads(r3_path.read_text(encoding="utf-8"))
    algebra = json.loads(r2_path.read_text(encoding="utf-8"))
    assert loaded == payload
    validate_synthetic_probe(loaded)
    assert algebra == loaded["algebra"]
    assert set(algebra) == {
        "dimensions",
        "dtypes",
        "finite_checks",
        "gradient_identity_residual_norm",
        "normal_equation_residual_norm",
        "score_statistics",
    }
    assert not list(tmp_path.rglob("*.tmp"))


def test_probe_validator_rejects_digest_or_numerical_contract_tampering() -> None:
    digest_tampered = copy.deepcopy(run_synthetic_probe())
    digest_tampered["seed"] = 1

    with pytest.raises(ValueError, match="digest"):
        validate_synthetic_probe(digest_tampered)

    contract_tampered = copy.deepcopy(run_synthetic_probe())
    causal = _object(contract_tampered["causal_score"])
    causal["first_order_signs_match"] = False

    with pytest.raises(ValueError, match="first-order"):
        validate_synthetic_probe(contract_tampered, verify_digest=False)


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"seed": -1}, "seed"),
        ({"seed": True}, "seed"),
        ({"seed": 1}, "locked"),
        ({"epsilon": float("nan")}, "epsilon"),
        ({"epsilon": 1e-5}, "locked"),
    ],
)
def test_probe_rejects_invalid_or_unlocked_parameters(
    arguments: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        run_synthetic_probe(**arguments)  # type: ignore[arg-type]


def test_r1_r3_script_is_fail_fast_atomic_and_offline() -> None:
    script = Path("scripts/verify_r1_r3.sh").read_text(encoding="utf-8")

    assert "set -euo pipefail" in script
    assert "git rev-parse --show-toplevel" in script
    assert "mktemp -d" in script
    assert "os.replace" in script
    assert "self-improve-protein --help" in script
    assert "ruff check ." in script
    assert "mypy src" in script
    assert "tests/test_ridge.py" in script
    assert "tests/test_selection.py" in script
    assert "tests/test_data.py" in script
    assert "tests/test_embeddings.py" in script
    assert "tests/test_metrics.py" in script
    assert "tests/test_experiment.py" in script
    assert "pytest -q" in script
    assert "self_improve_protein.probes" in script
    assert "allow_nan=False" in script
    assert "curl " not in script
    assert "wget " not in script
    assert "sbatch " not in script
