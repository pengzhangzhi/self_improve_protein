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

import self_improve_protein.crossfit_data as crossfit_data
import self_improve_protein.locality_cli as locality_cli
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
from self_improve_protein.provenance import sha256_file

RUNNER = CliRunner()
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


def _sequence(index: int) -> str:
    digits: list[str] = []
    value = index
    for _ in range(4):
        digits.append(AMINO_ACIDS[value % 20])
        value //= 20
    return "A" + "".join(reversed(digits))


def _workspace(tmp_path: Path) -> dict[str, Path]:
    payload = load_protocol("configs/v0.yaml").model_dump(mode="python")
    payload.update(
        seeds=(0, 1, 2, 3, 4),
        assay_count=8,
        random_diagnostic_replicates=2,
    )
    protocol = Protocol.model_validate(payload)
    config = tmp_path / "protocol.yaml"
    config.write_text(
        yaml.safe_dump(protocol.model_dump(mode="json"), sort_keys=False),
        encoding="utf-8",
    )
    processed = tmp_path / "processed"
    embeddings_root = tmp_path / "embeddings"
    results = tmp_path / "results"
    processed.mkdir()
    embeddings_root.mkdir()
    assay_ids = tuple(f"ASSAY_{index:02d}" for index in range(35))
    records: list[SelectedAssayManifest] = []
    cache_sources = EmbeddingCacheSources(
        proteingym_upstream_commit=protocol.proteingym_upstream_commit,
        substitutions_sha256=protocol.substitutions_sha256,
        zero_shot_scores_sha256=protocol.zero_shot_scores_sha256,
        metadata_sha256=protocol.metadata_sha256,
    )
    for assay_offset, assay_id in enumerate(assay_ids):
        frame = pd.DataFrame(
            {
                "dms_id": [assay_id] * protocol.working_size,
                "mutant": [f"A{row + 1}C" for row in range(protocol.working_size)],
                "mutated_sequence": [
                    _sequence(row + assay_offset * protocol.working_size)
                    for row in range(protocol.working_size)
                ],
            }
        )
        frame["sequence_hash"] = [
            row_hash(assay_id, mutant, sequence)
            for mutant, sequence in zip(
                frame["mutant"], frame["mutated_sequence"], strict=True
            )
        ]
        frame = frame.sort_values("sequence_hash", kind="stable").reset_index(
            drop=True
        )
        row_hashes = tuple(str(value) for value in frame["sequence_hash"])
        records.append(
            SelectedAssayManifest(
                dms_id=assay_id,
                usable_count=protocol.working_size,
                sequence_length=5,
                row_hashes=row_hashes,
            )
        )
        if assay_offset >= 9:
            continue
        generator = np.random.Generator(np.random.PCG64(900 + assay_offset))
        embedding = generator.normal(size=(protocol.working_size, 5)).astype(
            np.float32
        )
        beta = np.array([0.8, -0.5, 0.3, 0.45, -0.2])
        frame["DMS_score"] = embedding.astype(np.float64) @ beta + np.linspace(
            -0.2, 0.2, protocol.working_size
        )
        frame[protocol.teacher_column] = (
            embedding.astype(np.float64) @ (beta + 0.12) + 0.03
        )
        frame.to_parquet(processed / f"{assay_id}.parquet", index=False)
        write_embedding_cache(
            embeddings_root / f"{assay_id}.npy",
            embeddings_root / f"{assay_id}.json",
            embedding,
            dms_id=assay_id,
            row_hashes=row_hashes,
            model_id=protocol.model,
            model_revision=protocol.model_revision,
            sources=cache_sources,
        )
    sources = ManifestSources(
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
        sources=sources,
        upstream_revision=protocol.proteingym_upstream_commit,
        eligible_assay_ids=assay_ids,
        confirmatory_ids=assay_ids[:8],
        development_id=assay_ids[8],
        max_length=protocol.max_length,
        working_size=protocol.working_size,
        selected_assays=tuple(records[:9]),
    )
    base_path = tmp_path / "base.json"
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
        sources=sources,
        upstream_revision=protocol.proteingym_upstream_commit,
        max_length=protocol.max_length,
        working_size=protocol.working_size,
        eligible_assay_ids=assay_ids,
        screen_ids=(assay_ids[8], *assay_ids[:8]),
        untouched_ids=assay_ids[9:],
        selected_assays=tuple(records),
    )
    pool_path = tmp_path / "pool.json"
    write_crossfit_pool_manifest(pool_path, pool)
    return {
        "base": base_path,
        "config": config,
        "embeddings": embeddings_root,
        "pool": pool_path,
        "processed": processed,
        "results": results,
    }


def _task_args(paths: dict[str, Path], index: int) -> list[str]:
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

def test_locality_module_exposes_task_aggregate_and_verify() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "self_improve_protein.locality_cli", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    for command in ("run-task", "aggregate", "verify"):
        assert command in completed.stdout


def test_locality_cli_has_no_factorial_or_model_overrides() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "self_improve_protein.locality_cli",
            "run-task",
            "--help",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    for forbidden in (
        "--selector",
        "--q",
        "--pseudo-weight",
        "--ridge-lambda",
        "--damping",
        "--fold-count",
        "--fold-purpose",
    ):
        assert forbidden not in completed.stdout


def test_locality_slurm_is_exact_cpu_45_array_with_dependent_aggregate() -> None:
    task = Path("slurm/locality_task_array.sbatch")
    aggregate = Path("slurm/locality_aggregate.sbatch")
    submit = Path("slurm/submit_locality_screen.sh")
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
        assert "#SBATCH --requeue" in text
        assert "--gpus" not in text
        assert "export OPENBLAS_CORETYPE=Haswell" in text
        assert "python -m self_improve_protein.locality_cli" in text
        time_line = next(line for line in text.splitlines() if "--time=" in line)
        assert time_line <= "#SBATCH --time=04:00:00"
    submit_text = submit.read_text(encoding="utf-8")
    assert "--array=0-44" in submit_text
    assert "--dependency=afterok:" in submit_text
    assert "processed/v0" in submit_text
    assert "embeddings/v0" in submit_text
    assert submit_text.index(" verify \\") < submit_text.index('raw="$(sbatch')
    assert submit.stat().st_mode & 0o111


def test_locality_submitter_preflights_and_writes_dependent_job_manifest(
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
        "if [[ $count -eq 1 ]]; then printf '201\\n'; else printf '202\\n'; fi\n",
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
        "SI_RUN_ID": "locality-fixed-run",
        "SI_SLURM_CONF": str(tmp_path / "slurm.conf"),
    }

    completed = subprocess.run(
        [str(Path("slurm/submit_locality_screen.sh").resolve())],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert completed.returncode == 0, completed.stderr
    manifest_path = (
        fake_repo / "local" / "slurm" / "locality-fixed-run" / "job_ids.json"
    )
    assert json.loads(manifest_path.read_text(encoding="utf-8")) == {
        "jobs": {"aggregate": "202", "task": "201"},
        "kind": "locality_screen_jobs",
        "schema_version": 1,
    }
    calls = sbatch_log.read_text(encoding="utf-8").splitlines()
    assert "--array=0-44" in calls[0]
    assert "--dependency=afterok:201" in calls[1]


def test_locality_submitter_does_not_submit_when_preflight_fails(
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
        "SI_RUN_ID": "locality-fixed-run",
        "SI_SLURM_CONF": str(tmp_path / "slurm.conf"),
    }

    completed = subprocess.run(
        [str(Path("slurm/submit_locality_screen.sh").resolve())],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert completed.returncode != 0
    assert not sbatch_log.exists()


def test_locality_task_freezes_complete_fit_before_hidden_label_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _workspace(tmp_path)
    events: list[str] = []
    real_fit = locality_cli.fit_locality_task
    real_digest = locality_cli.canonical_locality_fit_digest
    real_load = locality_cli.load_evaluation_labels

    def fit_then_record(*args: object, **kwargs: object) -> object:
        fitted = real_fit(*args, **kwargs)  # type: ignore[arg-type]
        assert len(fitted.orderings) == 4
        assert len(fitted.cells) == 60
        events.append("fit_60_cells")
        return fitted

    def digest_then_record(*args: object, **kwargs: object) -> str:
        digest = real_digest(*args, **kwargs)  # type: ignore[arg-type]
        events.append("fit_digest")
        return digest

    def hidden_load(*args: object, **kwargs: object) -> object:
        assert events == ["fit_60_cells", "fit_digest"]
        events.append("hidden_labels")
        return real_load(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(locality_cli, "fit_locality_task", fit_then_record)
    monkeypatch.setattr(
        locality_cli,
        "canonical_locality_fit_digest",
        digest_then_record,
    )
    monkeypatch.setattr(locality_cli, "load_evaluation_labels", hidden_load)

    result = RUNNER.invoke(locality_cli.app, _task_args(paths, 0))

    assert result.exit_code == 0, result.output
    assert events == ["fit_60_cells", "fit_digest", "hidden_labels"]


def test_locality_screen_is_exact_restart_safe_and_exploratory(
    tmp_path: Path,
) -> None:
    paths = _workspace(tmp_path)

    first = RUNNER.invoke(locality_cli.app, _task_args(paths, 0))
    assert first.exit_code == 0, first.output
    task_path = paths["results"] / "tasks" / "ASSAY_08" / "seed_0.json"
    original_task = task_path.read_bytes()
    task_payload = json.loads(original_task)
    assert task_payload["kind"] == "locality_task_result"
    assert task_payload["task"]["phase"] == "locality_screen"
    assert task_payload["card"] == {
        "id": locality_cli.CARD_ID,
        "sha256": locality_cli.CARD_SHA,
    }
    assert len(task_payload["orderings"]) == 4
    assert len(task_payload["cells"]) == 60
    assert {
        (row["selector"], row["q"], row["pseudo_weight"])
        for row in task_payload["cells"]
    } == {
        (selector, q, weight)
        for selector in locality_cli.SELECTORS
        for q in locality_cli.Q_VALUES
        for weight in locality_cli.W_VALUES
    }
    orderings = {
        row["selector"]: row["ordered_indices"]
        for row in task_payload["orderings"]
    }
    for row in task_payload["cells"]:
        assert row["selected_indices"] == orderings[row["selector"]][: row["q"]]

    again = RUNNER.invoke(locality_cli.app, _task_args(paths, 0))
    assert again.exit_code == 0, again.output
    assert json.loads(again.output.splitlines()[-1])["created"] is False
    assert task_path.read_bytes() == original_task

    tampered = json.loads(original_task)
    tampered["cells"][0]["spearman"] = 0.123456789
    task_path.write_text(json.dumps(tampered) + "\n", encoding="utf-8")
    rejected = RUNNER.invoke(locality_cli.app, _task_args(paths, 0))
    assert rejected.exit_code != 0
    assert "mismatched existing artifact" in rejected.output
    task_path.write_bytes(original_task)

    for index in range(1, 45):
        result = RUNNER.invoke(locality_cli.app, _task_args(paths, index))
        assert result.exit_code == 0, f"task {index}: {result.output}"

    output = tmp_path / "locality_aggregate.json"
    aggregated = RUNNER.invoke(locality_cli.app, _aggregate_args(paths, output))
    assert aggregated.exit_code == 0, aggregated.output
    original_aggregate = output.read_bytes()
    payload = json.loads(original_aggregate)
    assert payload["kind"] == "locality_aggregate_result"
    assert payload["analysis"]["scope"] == "exploratory_mechanism_screen"
    assert payload["analysis"]["confirmatory_claim"] is False
    assert "promotion" not in payload
    assert payload["provenance"]["task_count"] == 45
    assert len(payload["grid"]["screen_tasks"]) == 45
    assert len(payload["grid"]["primary_tasks"]) == 40
    assert payload["grid"]["development_assay_id"] == "ASSAY_08"
    assert "ASSAY_08" not in payload["grid"]["primary_assay_ids"]
    assert len(payload["primary_long_results"]) == 40 * 60
    assert len(payload["effects"]) == 3 * 5 * 3
    for effect in payload["effects"]:
        assert effect["second"] == "random"
        assert effect["task_total"] == 40
        assert effect["assay_total"] == 8
    assert len(payload["mechanism"]["cell_trends"]) == 60
    assert len(payload["mechanism"]["selector_trends"]) == 4
    serialized = json.dumps(payload, sort_keys=True)
    for untouched in (f"ASSAY_{index:02d}" for index in range(9, 35)):
        assert untouched not in serialized

    aggregate_again = RUNNER.invoke(
        locality_cli.app,
        _aggregate_args(paths, output),
    )
    assert aggregate_again.exit_code == 0, aggregate_again.output
    assert json.loads(aggregate_again.output.splitlines()[-1])["created"] is False
    assert output.read_bytes() == original_aggregate

    tampered_aggregate = json.loads(original_aggregate)
    tampered_aggregate["analysis"]["confirmatory_claim"] = True
    output.write_text(json.dumps(tampered_aggregate) + "\n", encoding="utf-8")
    aggregate_rejected = RUNNER.invoke(
        locality_cli.app,
        _aggregate_args(paths, output),
    )
    assert aggregate_rejected.exit_code != 0
    assert "mismatched existing artifact" in aggregate_rejected.output
