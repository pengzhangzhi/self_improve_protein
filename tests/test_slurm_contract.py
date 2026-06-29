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
        "self_improve_protein.cli:app"
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
    assert text.index('mkdir "$run_dir"') < text.index('prepare_job="$(sbatch')
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
