from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pytest
import yaml
from typer.testing import CliRunner

import self_improve_protein.crossfit_cli as crossfit_cli
import self_improve_protein.crossfit_data as crossfit_data
from self_improve_protein.config import Protocol, load_protocol
from self_improve_protein.crossfit_data import (
    CROSSFIT_CARD_ID,
    CROSSFIT_CARD_SHA256,
    CROSSFIT_POOL_SCHEMA_ID,
    CrossfitPoolManifest,
    write_crossfit_pool_manifest,
)
from self_improve_protein.data import (
    DataManifest,
    ManifestSource,
    ManifestSources,
    SelectedAssayManifest,
    row_hash,
    write_data_manifest,
)
from self_improve_protein.embeddings import (
    EmbeddingCacheSources,
    write_embedding_cache,
)
from self_improve_protein.experiment import METHOD_NAMES
from self_improve_protein.provenance import sha256_file

RUNNER = CliRunner()
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


def _sequence(index: int) -> str:
    return (
        "A"
        + AMINO_ACIDS[(index // (20 * 20)) % 20]
        + AMINO_ACIDS[(index // 20) % 20]
        + AMINO_ACIDS[index % 20]
    )


def _synthetic_crossfit_workspace(tmp_path: Path) -> dict[str, Path]:
    protocol_payload = load_protocol("configs/v0.yaml").model_dump(mode="python")
    protocol_payload.update(
        working_size=30,
        n_labeled=8,
        n_unlabeled=10,
        n_test=12,
        q=4,
        seeds=(0, 1, 2, 3, 4),
        assay_count=8,
        random_diagnostic_replicates=2,
    )
    protocol = Protocol.model_validate(protocol_payload)
    config = tmp_path / "protocol.yaml"
    config.write_text(
        yaml.safe_dump(protocol.model_dump(mode="json"), sort_keys=False),
        encoding="utf-8",
    )
    processed_root = tmp_path / "processed"
    embedding_root = tmp_path / "embeddings"
    results_root = tmp_path / "results"
    processed_root.mkdir()
    embedding_root.mkdir()

    assay_ids = tuple(f"ASSAY_{index:02d}" for index in range(35))
    selected: list[SelectedAssayManifest] = []
    sources = EmbeddingCacheSources(
        proteingym_upstream_commit=protocol.proteingym_upstream_commit,
        substitutions_sha256=protocol.substitutions_sha256,
        zero_shot_scores_sha256=protocol.zero_shot_scores_sha256,
        metadata_sha256=protocol.metadata_sha256,
    )
    for assay_offset, dms_id in enumerate(assay_ids):
        frame = pd.DataFrame(
            {
                "dms_id": [dms_id] * protocol.working_size,
                "mutant": [
                    f"A{index + 1}C" for index in range(protocol.working_size)
                ],
                "mutated_sequence": [
                    _sequence(index + assay_offset * protocol.working_size)
                    for index in range(protocol.working_size)
                ],
            }
        )
        frame["sequence_hash"] = [
            row_hash(dms_id, mutant, sequence)
            for mutant, sequence in zip(
                frame["mutant"],
                frame["mutated_sequence"],
                strict=True,
            )
        ]
        frame = frame.sort_values("sequence_hash", kind="stable").reset_index(
            drop=True
        )
        row_hashes = tuple(str(value) for value in frame["sequence_hash"])
        selected.append(
            SelectedAssayManifest(
                dms_id=dms_id,
                usable_count=protocol.working_size,
                sequence_length=4,
                row_hashes=row_hashes,
            )
        )
        if assay_offset >= 9:
            continue
        generator = np.random.Generator(np.random.PCG64(800 + assay_offset))
        embeddings = generator.normal(
            size=(protocol.working_size, 5)
        ).astype(np.float32)
        beta = np.array([0.8, -0.5, 0.3, 0.45, -0.2])
        frame["DMS_score"] = embeddings.astype(np.float64) @ beta + np.linspace(
            -0.2,
            0.2,
            protocol.working_size,
        )
        frame[protocol.teacher_column] = (
            embeddings.astype(np.float64) @ (beta + 0.12) + 0.03
        )
        frame.to_parquet(processed_root / f"{dms_id}.parquet", index=False)
        write_embedding_cache(
            embedding_root / f"{dms_id}.npy",
            embedding_root / f"{dms_id}.json",
            embeddings,
            dms_id=dms_id,
            row_hashes=row_hashes,
            model_id=protocol.model,
            model_revision=protocol.model_revision,
            sources=sources,
        )

    manifest_sources = ManifestSources(
        substitutions=ManifestSource(
            url=protocol.substitutions_url,
            sha256=protocol.substitutions_sha256,
        ),
        scores=ManifestSource(
            url=protocol.zero_shot_scores_url,
            sha256=protocol.zero_shot_scores_sha256,
        ),
        metadata=ManifestSource(
            url=protocol.metadata_url,
            sha256=protocol.metadata_sha256,
        ),
    )
    base = DataManifest(
        schema_version=1,
        data_release=protocol.data_release,
        teacher_column=protocol.teacher_column,
        sources=manifest_sources,
        upstream_revision=protocol.proteingym_upstream_commit,
        eligible_assay_ids=assay_ids,
        confirmatory_ids=assay_ids[:8],
        development_id=assay_ids[8],
        max_length=protocol.max_length,
        working_size=protocol.working_size,
        selected_assays=tuple(selected[:9]),
    )
    base_path = tmp_path / "base_manifest.json"
    write_data_manifest(base_path, base)
    pool = CrossfitPoolManifest(
        schema_id=CROSSFIT_POOL_SCHEMA_ID,
        schema_version=1,
        card_id=CROSSFIT_CARD_ID,
        card_sha256=CROSSFIT_CARD_SHA256,
        base_manifest_sha256=sha256_file(base_path),
        protocol_sha256=crossfit_data._protocol_sha256(protocol),
        data_release=protocol.data_release,
        teacher_column=protocol.teacher_column,
        sources=manifest_sources,
        upstream_revision=protocol.proteingym_upstream_commit,
        max_length=protocol.max_length,
        working_size=protocol.working_size,
        eligible_assay_ids=assay_ids,
        screen_ids=(assay_ids[8], *assay_ids[:8]),
        untouched_ids=assay_ids[9:],
        selected_assays=tuple(selected),
    )
    pool_path = tmp_path / "pool_manifest.json"
    write_crossfit_pool_manifest(pool_path, pool)
    return {
        "base": base_path,
        "config": config,
        "embeddings": embedding_root,
        "pool": pool_path,
        "processed": processed_root,
        "results": results_root,
    }


def _run_task_args(paths: dict[str, Path], index: int) -> list[str]:
    return [
        "--config",
        str(paths["config"]),
        "run-task",
        "--base-manifest",
        str(paths["base"]),
        "--pool-manifest",
        str(paths["pool"]),
        "--processed-root",
        str(paths["processed"]),
        "--embedding-root",
        str(paths["embeddings"]),
        "--results-root",
        str(paths["results"]),
        "--task-index",
        str(index),
        "--non-official-bypass",
    ]


def _aggregate_args(paths: dict[str, Path], output: Path) -> list[str]:
    return [
        "--config",
        str(paths["config"]),
        "aggregate",
        "--base-manifest",
        str(paths["base"]),
        "--pool-manifest",
        str(paths["pool"]),
        "--processed-root",
        str(paths["processed"]),
        "--embedding-root",
        str(paths["embeddings"]),
        "--results-root",
        str(paths["results"]),
        "--output",
        str(output),
        "--non-official-bypass",
    ]


def test_crossfit_module_exposes_separate_screen_commands() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "self_improve_protein.crossfit_cli", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "run-task" in completed.stdout
    assert "aggregate" in completed.stdout
    assert "verify" in completed.stdout


def test_screen_grid_is_development_then_eight_confirmatory_by_five_seeds() -> None:
    development_id = "GFP_AEQVI_Sarkisyan_2016"
    confirmatory_ids = tuple(f"ASSAY_{index}" for index in range(8))
    seeds = (0, 1, 2, 3, 4)

    grid = crossfit_cli._screen_grid(
        development_id=development_id,
        confirmatory_ids=confirmatory_ids,
        seeds=seeds,
    )

    assert len(grid) == 45
    assert grid[:5] == tuple((development_id, seed) for seed in seeds)
    assert grid[5:10] == tuple((confirmatory_ids[0], seed) for seed in seeds)
    assert grid[-5:] == tuple((confirmatory_ids[-1], seed) for seed in seeds)


def test_crossfit_screen_slurm_scripts_are_cpu_only_strict_and_exact_45_array() -> None:
    task = Path("slurm/crossfit_task_array.sbatch")
    aggregate = Path("slurm/crossfit_aggregate.sbatch")
    submit = Path("slurm/submit_crossfit_screen.sh")

    for path in (task, aggregate, submit):
        completed = subprocess.run(
            ["bash", "-n", str(path)],
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, completed.stderr
    for path in (task, aggregate):
        text = path.read_text(encoding="utf-8")
        assert "set -euo pipefail" in text
        assert "#SBATCH --requeue" in text
        assert "export OPENBLAS_CORETYPE=Haswell" in text
        assert "--gpus" not in text
        assert "python -m self_improve_protein.crossfit_cli" in text
    task_text = task.read_text(encoding="utf-8")
    assert "SLURM_ARRAY_TASK_ID" in task_text
    assert "--task-index" in task_text
    submit_text = submit.read_text(encoding="utf-8")
    assert "--array=0-44" in submit_text
    assert "--dependency=afterok:" in submit_text
    assert "job_ids.json" in submit_text
    assert '${SI_DATA_ROOT}/processed/v0' in submit_text
    assert '${SI_DATA_ROOT}/embeddings/v0' in submit_text
    assert '${SI_DATA_ROOT}/processed/crossfit_v1' not in submit_text
    assert '${SI_DATA_ROOT}/embeddings/crossfit_v1' not in submit_text
    assert submit.stat().st_mode & 0o111


def test_crossfit_submitter_preflights_and_writes_dependent_job_manifest(
    tmp_path: Path,
) -> None:
    fake_repo = tmp_path / "repo"
    fake_bin = tmp_path / "bin"
    (fake_repo / ".venv" / "bin").mkdir(parents=True)
    fake_bin.mkdir()
    fake_python = fake_repo / ".venv" / "bin" / "python"
    fake_python.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    fake_python.chmod(0o755)
    sbatch_log = tmp_path / "sbatch.log"
    fake_sbatch = fake_bin / "sbatch"
    fake_sbatch.write_text(
        "#!/usr/bin/env bash\n"
        "printf '%s\\n' \"$*\" >> \"$FAKE_SBATCH_LOG\"\n"
        "count=$(wc -l < \"$FAKE_SBATCH_LOG\")\n"
        "if [[ $count -eq 1 ]]; then printf '101\\n'; else printf '102\\n'; fi\n",
        encoding="utf-8",
    )
    fake_sbatch.chmod(0o755)
    environment = {
        **os.environ,
        "FAKE_SBATCH_LOG": str(sbatch_log),
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "SI_ACCOUNT": "acct",
        "SI_ARTIFACT_ROOT": str(tmp_path / "artifacts"),
        "SI_CPU_PARTITION": "cpu",
        "SI_DATA_ROOT": str(tmp_path / "data"),
        "SI_REPO_ROOT": str(fake_repo),
        "SI_RUN_ID": "fixed-run",
        "SI_SLURM_CONF": str(tmp_path / "slurm.conf"),
    }

    completed = subprocess.run(
        [str(Path("slurm/submit_crossfit_screen.sh").resolve())],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert completed.returncode == 0, completed.stderr
    manifest_path = fake_repo / "local" / "slurm" / "fixed-run" / "job_ids.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest == {
        "jobs": {"aggregate": "102", "task": "101"},
        "kind": "crossfit_screen_jobs",
        "schema_version": 1,
    }
    calls = sbatch_log.read_text(encoding="utf-8").splitlines()
    assert "--array=0-44" in calls[0]
    assert "--dependency=afterok:101" in calls[1]


def test_crossfit_submitter_does_not_submit_when_preflight_fails(
    tmp_path: Path,
) -> None:
    fake_repo = tmp_path / "repo"
    fake_bin = tmp_path / "bin"
    (fake_repo / ".venv" / "bin").mkdir(parents=True)
    fake_bin.mkdir()
    fake_python = fake_repo / ".venv" / "bin" / "python"
    fake_python.write_text("#!/usr/bin/env bash\nexit 7\n", encoding="utf-8")
    fake_python.chmod(0o755)
    sbatch_log = tmp_path / "sbatch.log"
    fake_sbatch = fake_bin / "sbatch"
    fake_sbatch.write_text(
        "#!/usr/bin/env bash\nprintf 'called\\n' >> \"$FAKE_SBATCH_LOG\"\n",
        encoding="utf-8",
    )
    fake_sbatch.chmod(0o755)
    environment = {
        **os.environ,
        "FAKE_SBATCH_LOG": str(sbatch_log),
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "SI_ACCOUNT": "acct",
        "SI_ARTIFACT_ROOT": str(tmp_path / "artifacts"),
        "SI_CPU_PARTITION": "cpu",
        "SI_DATA_ROOT": str(tmp_path / "data"),
        "SI_REPO_ROOT": str(fake_repo),
        "SI_RUN_ID": "fixed-run",
        "SI_SLURM_CONF": str(tmp_path / "slurm.conf"),
    }

    completed = subprocess.run(
        [str(Path("slurm/submit_crossfit_screen.sh").resolve())],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert completed.returncode != 0
    assert not sbatch_log.exists()


def test_crossfit_task_schema_is_distinct_and_keeps_locked_references() -> None:
    payload = {
        "schema_version": 1,
        "kind": "crossfit_task_result",
        "card": {
            "id": crossfit_cli.CARD_ID,
            "sha256": crossfit_cli.CARD_SHA,
        },
        "task": {"assay_id": "A", "seed": 0, "phase": "screen"},
        "digests": {
            "base_fit": "0" * 64,
            "crossfit_fit": "1" * 64,
            "evaluation": "2" * 64,
            "protocol": "3" * 64,
            "source": "4" * 64,
        },
        "execution": {},
        "provenance": {},
        "reference_methods": [
            {
                "name": name,
                "spearman": 0.0,
                "mse": 1.0,
                "ndcg_10pct": 0.5,
            }
            for name in METHOD_NAMES
        ],
        "variant": {
            "name": "crossfit",
            "spearman": 0.1,
            "mse": 0.9,
            "ndcg_10pct": 0.6,
        },
        "diagnostics": {},
    }

    crossfit_cli._validate_crossfit_task_payload(payload)

    payload["kind"] = "task_result"
    try:
        crossfit_cli._validate_crossfit_task_payload(payload)
    except ValueError as error:
        assert "schema" in str(error)
    else:  # pragma: no cover - assertion branch
        raise AssertionError("locked v0 task schema must not be accepted")


def test_crossfit_cli_has_no_fold_purpose_or_model_hyperparameter_overrides() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "self_improve_protein.crossfit_cli",
            "run-task",
            "--help",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    for forbidden in (
        "--fold-count",
        "--fold-purpose",
        "--ridge-lambda",
        "--pseudo-weight",
        "--damping",
        "--q",
    ):
        assert forbidden not in completed.stdout


def test_crossfit_task_freezes_both_fit_digests_before_hidden_label_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _synthetic_crossfit_workspace(tmp_path)
    events: list[str] = []
    real_fit = crossfit_cli.fit_crossfit_task
    real_digest = crossfit_cli.canonical_crossfit_fit_digest
    real_load = crossfit_cli.load_evaluation_labels

    def fit_then_record(*args: object, **kwargs: object) -> object:
        fitted = real_fit(*args, **kwargs)  # type: ignore[arg-type]
        assert len(fitted.base_fit_digest) == 64
        events.append("base_and_crossfit_fit")
        return fitted

    def digest_then_record(*args: object, **kwargs: object) -> str:
        digest = real_digest(*args, **kwargs)  # type: ignore[arg-type]
        events.append("crossfit_digest")
        return digest

    def hidden_load(*args: object, **kwargs: object) -> object:
        assert events == ["base_and_crossfit_fit", "crossfit_digest"]
        events.append("hidden_labels")
        return real_load(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(crossfit_cli, "fit_crossfit_task", fit_then_record)
    monkeypatch.setattr(
        crossfit_cli,
        "canonical_crossfit_fit_digest",
        digest_then_record,
    )
    monkeypatch.setattr(crossfit_cli, "load_evaluation_labels", hidden_load)

    result = RUNNER.invoke(crossfit_cli.app, _run_task_args(paths, 0))

    assert result.exit_code == 0, result.output
    assert events[0:3] == [
        "base_and_crossfit_fit",
        "crossfit_digest",
        "hidden_labels",
    ]


def test_crossfit_screen_is_exact_restart_safe_and_excludes_gfp_from_gate(
    tmp_path: Path,
) -> None:
    paths = _synthetic_crossfit_workspace(tmp_path)

    first = RUNNER.invoke(crossfit_cli.app, _run_task_args(paths, 0))
    assert first.exit_code == 0, first.output
    task_path = paths["results"] / "tasks" / "ASSAY_08" / "seed_0.json"
    original_task = task_path.read_bytes()
    task_payload = json.loads(original_task)
    assert task_payload["kind"] == "crossfit_task_result"
    assert [row["name"] for row in task_payload["reference_methods"]] == list(
        METHOD_NAMES
    )
    assert task_payload["variant"]["name"] == "crossfit"

    again = RUNNER.invoke(crossfit_cli.app, _run_task_args(paths, 0))
    assert again.exit_code == 0, again.output
    assert json.loads(again.output.splitlines()[-1])["created"] is False
    assert task_path.read_bytes() == original_task

    tampered = dict(task_payload)
    tampered["variant"] = dict(task_payload["variant"])
    tampered["variant"]["spearman"] = 0.123456789
    task_path.write_text(json.dumps(tampered) + "\n", encoding="utf-8")
    rejected = RUNNER.invoke(crossfit_cli.app, _run_task_args(paths, 0))
    assert rejected.exit_code != 0
    assert "mismatched existing artifact" in rejected.output
    task_path.write_bytes(original_task)

    for index in range(1, 45):
        result = RUNNER.invoke(crossfit_cli.app, _run_task_args(paths, index))
        assert result.exit_code == 0, f"task {index}: {result.output}"

    output = tmp_path / "crossfit_aggregate.json"
    aggregated = RUNNER.invoke(crossfit_cli.app, _aggregate_args(paths, output))
    assert aggregated.exit_code == 0, aggregated.output
    original_aggregate = output.read_bytes()
    payload = json.loads(original_aggregate)
    assert payload["kind"] == "crossfit_aggregate_result"
    assert payload["provenance"]["task_count"] == 45
    assert len(payload["grid"]["screen_tasks"]) == 45
    assert len(payload["grid"]["primary_tasks"]) == 40
    assert payload["grid"]["development_assay_id"] == "ASSAY_08"
    assert "ASSAY_08" not in payload["grid"]["primary_assay_ids"]
    assert len(payload["primary_long_results"]) == 40 * 6
    primary_effect = payload["effects"]["crossfit_minus_random"]
    assert primary_effect["task_total"] == 40
    assert primary_effect["assay_total"] == 8
    expected_promotion = (
        primary_effect["mean_gain"] > 0.0
        and primary_effect["task_wins"] >= 25
        and primary_effect["assay_wins"] >= 5
    )
    assert (
        payload["promotion"]["promote_to_untouched_replication"]
        is expected_promotion
    )

    aggregate_again = RUNNER.invoke(
        crossfit_cli.app,
        _aggregate_args(paths, output),
    )
    assert aggregate_again.exit_code == 0, aggregate_again.output
    assert json.loads(aggregate_again.output.splitlines()[-1])["created"] is False
    assert output.read_bytes() == original_aggregate

    aggregate_payload = json.loads(original_aggregate)
    aggregate_payload["promotion"]["task_win_threshold"] = 24
    output.write_text(json.dumps(aggregate_payload) + "\n", encoding="utf-8")
    aggregate_rejected = RUNNER.invoke(
        crossfit_cli.app,
        _aggregate_args(paths, output),
    )
    assert aggregate_rejected.exit_code != 0
    assert "mismatched existing artifact" in aggregate_rejected.output
