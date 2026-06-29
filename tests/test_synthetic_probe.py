import copy
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import cast

import pytest

from self_improve_protein import probes as probe_module
from self_improve_protein.probes import (
    CAUSAL_DERIVATIVE_ERROR_TOLERANCE,
    CAUSAL_EPSILON,
    LEARNABILITY_PARAMETER_ERROR_TOLERANCE,
    LEARNABILITY_RIDGE_LAMBDA,
    LEARNABILITY_TRAIN_MSE_TOLERANCE,
    build_verification_completion,
    publish_verification_bundle,
    require_clean_verification_git_state,
    run_synthetic_probe,
    validate_synthetic_probe,
    validate_verification_bundle,
    write_synthetic_probe,
)
from self_improve_protein.provenance import atomic_write_json, sha256_file

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
    assert cast(float, learnability["train_mse"]) < (LEARNABILITY_TRAIN_MSE_TOLERANCE)
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
        abs(error) <= CAUSAL_DERIVATIVE_ERROR_TOLERANCE for error in derivative_errors
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
    assert "self-improve-protein --help" in script
    assert "UV_BASE=(uv run --frozen --offline --extra dev --extra embed)" in script
    assert "UV_PROJECT_ENVIRONMENT" in script
    assert "uv sync --dry-run" in script
    assert "r1_fresh_environment_resolution" in script
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
    assert "git status --porcelain=v1 --untracked-files=all" in script
    assert "START_HEAD" in script
    assert "require_clean_verification_git_state" in script
    assert "publish_verification_bundle" in script
    assert "completion.json" in script
    assert '"torch"' in script
    assert '"transformers"' in script
    assert "pyproject.toml" in script
    assert "uv.lock" in script
    assert "os.replace" in Path("src/self_improve_protein/probes.py").read_text(
        encoding="utf-8"
    )
    assert "curl " not in script
    assert "wget " not in script
    assert "sbatch " not in script


def _staged_verification_bundle(
    tmp_path: Path,
    artifact_root: Path,
    *,
    include_transformers: bool = True,
) -> tuple[Path, dict[str, object]]:
    staging = artifact_root.parent / ".verification-staging"
    (staging / "r1").mkdir(parents=True)
    (staging / "r2").mkdir()
    (staging / "r3").mkdir()
    probe = run_synthetic_probe()
    atomic_write_json(staging / "r3" / "synthetic_probe.json", probe)
    atomic_write_json(staging / "r2" / "algebra_probe.json", probe["algebra"])
    changes = [{"name": "torch", "version": "2.10.0", "action": "installed"}]
    if include_transformers:
        changes.append(
            {
                "name": "transformers",
                "version": "4.57.6",
                "action": "installed",
            }
        )
    atomic_write_json(
        staging / "r1" / "fresh-environment-resolution.json",
        {
            "sync": {
                "action": "create",
                "environment": {"path": str(staging / "fresh-environment")},
                "changes": changes,
            }
        },
    )
    (staging / "r2" / "pytest.txt").write_text(
        "targeted exit_code=0: 1 passed\nfull exit_code=0: 2 passed\n",
        encoding="utf-8",
    )
    output_hashes = {
        "r1/fresh-environment-resolution.json": sha256_file(
            staging / "r1" / "fresh-environment-resolution.json"
        ),
        "r2/pytest.txt": sha256_file(staging / "r2" / "pytest.txt"),
        "r2/algebra_probe.json": sha256_file(staging / "r2" / "algebra_probe.json"),
        "r3/synthetic_probe.json": sha256_file(staging / "r3" / "synthetic_probe.json"),
    }
    trust_root = {
        "git_head": "a" * 40,
        "pyproject_sha256": "1" * 64,
        "uv_lock_sha256": "2" * 64,
        "config_sha256": "3" * 64,
        "verification_script_sha256": "4" * 64,
        "python_executable_sha256": "5" * 64,
        "uv_executable_sha256": "6" * 64,
        "pytest_executable_sha256": "7" * 64,
        "ruff_executable_sha256": "8" * 64,
        "mypy_executable_sha256": "9" * 64,
    }
    report = {
        "schema_version": 2,
        "rung": "R1",
        "status": "passed",
        "git_head": "a" * 40,
        "repository_state": {
            "start_head": "a" * 40,
            "end_head": "a" * 40,
            "start_clean": True,
            "end_clean": True,
        },
        "trust_root": trust_root,
        "toolchain": {
            name: {"sha256": trust_root[f"{name}_executable_sha256"]}
            for name in ("python", "uv", "pytest", "ruff", "mypy")
        },
        "config": {"sha256": trust_root["config_sha256"]},
        "package_versions": {
            "torch": "2.10.0",
            "transformers": "4.57.6",
        },
        "commands": [
            {"name": name, "exit_code": 0}
            for name in (
                "r1_fresh_environment_resolution",
                "r1_package_import",
                "r1_locked_config",
                "r1_cli_help",
                "r1_ruff",
                "r1_mypy",
                "r2_targeted_pytest",
                "r2_full_pytest",
                "r3_synthetic_probe",
            )
        ],
        "artifacts": {
            "r1_report": str(artifact_root / "r1" / "report.json"),
            "fresh_environment_resolution": str(
                artifact_root / "r1" / "fresh-environment-resolution.json"
            ),
            "r2_pytest": str(artifact_root / "r2" / "pytest.txt"),
            "r2_algebra": str(artifact_root / "r2" / "algebra_probe.json"),
            "r3_probe": str(artifact_root / "r3" / "synthetic_probe.json"),
            "completion": str(artifact_root / "completion.json"),
        },
        "output_sha256": output_hashes,
    }
    report["report_content_sha256"] = hashlib.sha256(
        json.dumps(
            report,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
    ).hexdigest()
    atomic_write_json(staging / "r1" / "report.json", report)
    completion = build_verification_completion(
        staging,
        artifact_root=artifact_root,
        git_head="a" * 40,
        trust_root=trust_root,
        published_at="2026-06-29T00:00:00Z",
    )
    return staging, completion


def test_verification_bundle_custom_root_is_exact_and_completion_is_last(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact_root = tmp_path / "custom-output" / "verification"
    staging, completion = _staged_verification_bundle(tmp_path, artifact_root)
    events: list[tuple[Path, Path]] = []
    real_replace = probe_module.os.replace

    def recording_replace(source: Path, destination: Path) -> None:
        events.append((Path(source), Path(destination)))
        real_replace(source, destination)

    monkeypatch.setattr(probe_module.os, "replace", recording_replace)

    publish_verification_bundle(staging, artifact_root, completion)

    loaded = validate_verification_bundle(
        artifact_root,
        expected_git_head="a" * 40,
    )
    assert loaded == completion
    assert events[-1][1] == artifact_root / "completion.json"
    assert json.loads(
        (artifact_root / "r1" / "report.json").read_text(encoding="utf-8")
    )["artifacts"] == {
        "r1_report": str(artifact_root / "r1" / "report.json"),
        "fresh_environment_resolution": str(
            artifact_root / "r1" / "fresh-environment-resolution.json"
        ),
        "r2_pytest": str(artifact_root / "r2" / "pytest.txt"),
        "r2_algebra": str(artifact_root / "r2" / "algebra_probe.json"),
        "r3_probe": str(artifact_root / "r3" / "synthetic_probe.json"),
        "completion": str(artifact_root / "completion.json"),
    }
    assert set(completion["output_sha256"]) == {
        "r1/report.json",
        "r1/fresh-environment-resolution.json",
        "r2/pytest.txt",
        "r2/algebra_probe.json",
        "r3/synthetic_probe.json",
    }


def test_interrupted_publication_removes_authoritative_success_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact_root = tmp_path / "verification"
    artifact_root.mkdir()
    atomic_write_json(artifact_root / "completion.json", {"status": "passed"})
    staging, completion = _staged_verification_bundle(tmp_path, artifact_root)
    real_replace = probe_module.os.replace

    def fail_before_r2(source: Path, destination: Path) -> None:
        if Path(source).name == "r2":
            raise OSError("simulated interruption before R2 publication")
        real_replace(source, destination)

    monkeypatch.setattr(probe_module.os, "replace", fail_before_r2)

    with pytest.raises(OSError, match="simulated interruption"):
        publish_verification_bundle(staging, artifact_root, completion)

    assert not (artifact_root / "completion.json").exists()
    with pytest.raises(ValueError, match="completion"):
        validate_verification_bundle(artifact_root)


def test_successful_publication_removes_stale_files_and_detects_tampering(
    tmp_path: Path,
) -> None:
    artifact_root = tmp_path / "verification"
    stale = artifact_root / "r2" / "stale-success.txt"
    stale.parent.mkdir(parents=True)
    stale.write_text("obsolete\n", encoding="utf-8")
    staging, completion = _staged_verification_bundle(tmp_path, artifact_root)

    publish_verification_bundle(staging, artifact_root, completion)

    assert not stale.exists()
    validate_verification_bundle(artifact_root)
    with (artifact_root / "r2" / "pytest.txt").open("a", encoding="utf-8") as handle:
        handle.write("tampered\n")
    with pytest.raises(ValueError, match="SHA-256"):
        validate_verification_bundle(artifact_root)


def test_fresh_environment_proof_requires_both_pinned_embed_dependencies(
    tmp_path: Path,
) -> None:
    artifact_root = tmp_path / "verification"

    with pytest.raises(ValueError, match="embed dependencies"):
        _staged_verification_bundle(
            tmp_path,
            artifact_root,
            include_transformers=False,
        )


def test_clean_git_state_rejects_dirty_or_changed_head(tmp_path: Path) -> None:
    repository = tmp_path / "repo"
    repository.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
    subprocess.run(
        ["git", "config", "user.email", "probe@example.invalid"],
        cwd=repository,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Probe"],
        cwd=repository,
        check=True,
    )
    tracked = repository / "tracked.txt"
    tracked.write_text("first\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.txt"], cwd=repository, check=True)
    subprocess.run(["git", "commit", "-qm", "first"], cwd=repository, check=True)
    first = require_clean_verification_git_state(repository)

    untracked = repository / "untracked.txt"
    untracked.write_text("dirty\n", encoding="utf-8")
    with pytest.raises(ValueError, match="clean"):
        require_clean_verification_git_state(repository, expected_head=first)
    untracked.unlink()

    tracked.write_text("second\n", encoding="utf-8")
    with pytest.raises(ValueError, match="clean"):
        require_clean_verification_git_state(repository, expected_head=first)
    subprocess.run(["git", "add", "tracked.txt"], cwd=repository, check=True)
    subprocess.run(["git", "commit", "-qm", "second"], cwd=repository, check=True)
    with pytest.raises(ValueError, match="HEAD changed"):
        require_clean_verification_git_state(repository, expected_head=first)


def test_audit_distinguishes_supervised_from_three_pseudo_methods() -> None:
    audit = Path("docs/research/protocol-audit.md").read_text(encoding="utf-8")

    assert "Three pseudo-label methods" in audit
    assert "supervised-only method uses no pseudo-samples" in audit
    assert "Four confirmatory methods differ only in selection" not in audit
