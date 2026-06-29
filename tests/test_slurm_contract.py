import hashlib
import json
import os
import re
import subprocess
import tomllib
from pathlib import Path

SLURM = Path("slurm")
SBATCH_FILES = (
    "prepare.sbatch",
    "embed_array.sbatch",
    "task_array.sbatch",
    "aggregate.sbatch",
)


def test_console_script_is_registered() -> None:
    project = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

    assert project["project"]["scripts"]["self-improve-protein"] == (
        "self_improve_protein.cli:cli_main"
    )


def test_batch_scripts_are_strict_requeueable_env_driven_and_core_pinned() -> None:
    forbidden_private_literals = ("/lustre/", "nvr_lpr_compgenai", "fredp")
    for filename in SBATCH_FILES:
        text = (SLURM / filename).read_text(encoding="utf-8")
        lines = text.splitlines()
        first_executable = next(
            index
            for index, line in enumerate(lines)
            if line and not line.startswith("#!") and not line.startswith("#SBATCH")
        )
        last_directive = max(
            index for index, line in enumerate(lines) if line.startswith("#SBATCH")
        )

        assert "set -euo pipefail" in text
        assert "#SBATCH --requeue" in text
        assert "OccupiedIdleGPUsJobReaper" in text
        assert "export OPENBLAS_CORETYPE=Haswell" in text
        assert "require_openblas_coretype" in text
        assert "SLURM_CONF" in text
        assert last_directive < first_executable
        assert text.index("SLURM_CONF") > text.rindex("#SBATCH")
        assert all(literal not in text for literal in forbidden_private_literals)
        assert "SI_REPO_ROOT" in text
        assert "SI_DATA_ROOT" in text
        assert "SI_ARTIFACT_ROOT" in text


def test_embedding_has_one_gpu_and_task_array_is_cpu_only() -> None:
    embedding = (SLURM / "embed_array.sbatch").read_text(encoding="utf-8")
    task = (SLURM / "task_array.sbatch").read_text(encoding="utf-8")

    assert re.search(r"#SBATCH --gpus(?:-per-node)?=1", embedding)
    assert "--gpus" not in task
    assert "SLURM_ARRAY_TASK_ID" in embedding
    assert "SLURM_ARRAY_TASK_ID" in task
    assert "--assay-index" in embedding
    assert "--task-index" in task


def test_submit_pipeline_derives_exact_arrays_and_afterok_dependencies() -> None:
    text = (SLURM / "submit_pipeline.sh").read_text(encoding="utf-8")

    assert "set -euo pipefail" in text
    assert "--parsable" in text
    assert "--array=0-$((embed_count - 1))" in text
    assert "--array=0-$((task_count - 1))" in text
    assert text.count("--dependency=afterok:") >= 3
    assert "local/" in text
    assert "job_ids.json" in text
    assert 'mkdir -p "$run_root"' in text
    assert 'mkdir "$run_dir"' in text
    assert 'mkdir -p "$run_dir"' not in text
    assert text.index('mkdir "$run_dir"') < text.index('prepare_job="$(submit_job')
    assert 'SI_MODE="${SI_MODE:-development}"' in text
    assert 'export SLURM_CONF="$SI_SLURM_CONF"' in text
    assert "SI_R5_GATE" in text
    assert text.count("write_job_manifest") == 5
    for variable in (
        "SI_ACCOUNT",
        "SI_CPU_PARTITION",
        "SI_GPU_PARTITION",
        "SI_REPO_ROOT",
        "SI_DATA_ROOT",
        "SI_ARTIFACT_ROOT",
    ):
        assert variable in text
    assert "/lustre/" not in text
    assert "nvr_lpr_compgenai" not in text


def _mock_submit_environment(
    tmp_path: Path,
    *,
    fail_at: int | None = None,
) -> tuple[dict[str, str], Path, Path]:
    repo = tmp_path / "repo"
    python_path = repo / ".venv" / "bin" / "python"
    python_path.parent.mkdir(parents=True)
    active_python = (Path(".venv") / "bin" / "python").absolute()
    python_path.write_text(
        f'#!/usr/bin/env bash\nexec "{active_python}" "$@"\n',
        encoding="utf-8",
    )
    python_path.chmod(0o755)
    cli_path = repo / ".venv" / "bin" / "self-improve-protein"
    source_root = (Path.cwd() / "src").absolute()
    cli_path.write_text(
        "#!/usr/bin/env bash\n"
        f'export PYTHONPATH="{source_root}${{PYTHONPATH:+:${{PYTHONPATH}}}}"\n'
        f'exec "{active_python}" -m self_improve_protein.cli "$@"\n',
        encoding="utf-8",
    )
    cli_path.chmod(0o755)
    config_dir = repo / "configs"
    config_dir.mkdir()
    (config_dir / "v0.yaml").write_bytes(Path("configs/v0.yaml").read_bytes())
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    calls = tmp_path / "sbatch.calls"
    count = tmp_path / "sbatch.count"
    sbatch = fake_bin / "sbatch"
    sbatch.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
current=0
if [[ -f "$MOCK_SBATCH_COUNT" ]]; then current="$(cat "$MOCK_SBATCH_COUNT")"; fi
current=$((current + 1))
printf '%s' "$current" >"$MOCK_SBATCH_COUNT"
printf '%s|%s\n' "${SLURM_CONF:-}" "$*" >>"$MOCK_SBATCH_CALLS"
if [[ -n "${MOCK_SBATCH_FAIL_AT:-}" && "$current" == "$MOCK_SBATCH_FAIL_AT" ]]; then
    exit 42
fi
printf '%s\n' "$((7000 + current))"
""",
        encoding="utf-8",
    )
    sbatch.chmod(0o755)
    env = os.environ.copy()
    env.update(
        PATH=f"{fake_bin}:{env['PATH']}",
        SI_ACCOUNT="test-account",
        SI_CPU_PARTITION="cpu",
        SI_GPU_PARTITION="gpu",
        SI_REPO_ROOT=str(repo),
        SI_DATA_ROOT=str(tmp_path / "data"),
        SI_ARTIFACT_ROOT=str(tmp_path / "artifacts"),
        SI_SLURM_CONF=str(tmp_path / "slurm.conf"),
        SI_RUN_ID="test-run",
        MOCK_SBATCH_CALLS=str(calls),
        MOCK_SBATCH_COUNT=str(count),
    )
    if fail_at is not None:
        env["MOCK_SBATCH_FAIL_AT"] = str(fail_at)
    return env, repo, calls


def test_fresh_default_development_submit_needs_no_preexisting_manifest(
    tmp_path: Path,
) -> None:
    env, repo, calls_path = _mock_submit_environment(tmp_path)
    script = (SLURM / "submit_pipeline.sh").resolve()

    completed = subprocess.run(
        ["bash", str(script)],
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    calls = calls_path.read_text(encoding="utf-8").splitlines()
    assert len(calls) == 4
    assert all(call.startswith(f"{env['SI_SLURM_CONF']}|") for call in calls)
    assert "--array=0-8" in calls[1]
    assert "--array=0-1" in calls[2]
    default_manifest = tmp_path / "artifacts" / "studies" / "v0" / "data_manifest.json"
    assert not default_manifest.exists()
    manifest = json.loads(
        (repo / "local" / "slurm" / "test-run" / "job_ids.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["jobs"] == {
        "aggregate": "7004",
        "embed": "7002",
        "prepare": "7001",
        "task": "7003",
    }


def test_partial_sbatch_failure_preserves_each_prior_job_id_atomically(
    tmp_path: Path,
) -> None:
    env, repo, calls_path = _mock_submit_environment(tmp_path, fail_at=3)
    script = (SLURM / "submit_pipeline.sh").resolve()

    completed = subprocess.run(
        ["bash", str(script)],
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert len(calls_path.read_text(encoding="utf-8").splitlines()) == 3
    run_dir = repo / "local" / "slurm" / "test-run"
    manifest = json.loads((run_dir / "job_ids.json").read_text(encoding="utf-8"))
    assert manifest["jobs"] == {
        "aggregate": None,
        "embed": "7002",
        "prepare": "7001",
        "task": None,
    }
    assert not list(run_dir.glob("*.tmp"))


def test_confirmatory_submit_requires_passed_r5_gate_before_any_sbatch(
    tmp_path: Path,
) -> None:
    env, repo, calls_path = _mock_submit_environment(tmp_path)
    env["SI_MODE"] = "confirmatory"
    script = (SLURM / "submit_pipeline.sh").resolve()

    completed = subprocess.run(
        ["bash", str(script)],
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "SI_R5_GATE" in completed.stderr
    assert not calls_path.exists()
    assert not (repo / "local" / "slurm" / "test-run").exists()


def test_forged_field_only_r5_gate_without_evidence_submits_no_jobs(
    tmp_path: Path,
) -> None:
    env, repo, calls_path = _mock_submit_environment(tmp_path)
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=repo,
        check=True,
    )
    subprocess.run(["git", "add", "configs/v0.yaml"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "test config"], cwd=repo, check=True)
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}\n", encoding="utf-8")
    gate = tmp_path / "forged-r5.json"
    git_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    gate.write_text(
        json.dumps(
            {
                "git_commit": git_commit,
                "kind": "r5_gate",
                "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
                "protocol_digest": (
                    "0b2a74ff76b8c7c508ceea16b004a1c128ba15704138138d49b2c153bcbfa49a"
                ),
                "schema_version": 1,
                "status": "passed",
            }
        ),
        encoding="utf-8",
    )
    env.update(
        SI_MODE="confirmatory",
        SI_MANIFEST=str(manifest),
        SI_R5_GATE=str(gate),
    )
    script = (SLURM / "submit_pipeline.sh").resolve()

    completed = subprocess.run(
        ["bash", str(script)],
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert not calls_path.exists()
    assert not (repo / "local" / "slurm" / "test-run").exists()


def test_all_shell_files_pass_bash_syntax() -> None:
    for path in (
        *[SLURM / name for name in SBATCH_FILES],
        SLURM / "submit_pipeline.sh",
    ):
        completed = subprocess.run(
            ["bash", "-n", str(path)],
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, completed.stderr
