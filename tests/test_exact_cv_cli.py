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
import self_improve_protein.exact_cv_cli as exact_cv_cli
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
from self_improve_protein.exact_cv import CARD_ID, CARD_SHA
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


def _workspace(
    tmp_path: Path,
    *,
    embedding_width: int = 5,
    materialized_offsets: tuple[int, ...] = tuple(range(9)),
) -> dict[str, Path]:
    payload = load_protocol("configs/v0.yaml").model_dump(mode="python")
    payload.update(random_diagnostic_replicates=2)
    protocol = Protocol.model_validate(payload)
    config = tmp_path / "protocol.yaml"
    config.write_text(
        yaml.safe_dump(protocol.model_dump(mode="json"), sort_keys=False),
        encoding="utf-8",
    )
    processed = tmp_path / "processed" / "v0"
    embeddings_root = tmp_path / "embeddings" / "v0"
    results = tmp_path / "results"
    processed.mkdir(parents=True)
    embeddings_root.mkdir(parents=True)
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
                "mutant": [
                    f"A{row + 1}C" for row in range(protocol.working_size)
                ],
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
        if assay_offset >= 9 or assay_offset not in materialized_offsets:
            continue
        generator = np.random.Generator(np.random.PCG64(1900 + assay_offset))
        embedding = generator.normal(
            size=(protocol.working_size, embedding_width)
        ).astype(np.float32)
        beta = np.linspace(-0.5, 0.8, embedding_width)
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


def _common_args(paths: dict[str, Path]) -> list[str]:
    return [
        "--config",
        str(paths["config"]),
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
    ]


def _task_args(paths: dict[str, Path], index: int) -> list[str]:
    common = _common_args(paths)
    return [
        common[0],
        common[1],
        "run-task",
        *common[2:],
        "--task-index",
        str(index),
        "--non-official-bypass",
    ]


def _verify_task_args(paths: dict[str, Path], index: int) -> list[str]:
    common = _common_args(paths)
    return [
        common[0],
        common[1],
        "verify-task",
        *common[2:],
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
        "--results-root",
        str(paths["results"]),
        "--output",
        str(output),
        "--non-official-bypass",
    ]


def _verify_args(
    paths: dict[str, Path],
    *,
    aggregate: Path | None = None,
) -> list[str]:
    args = [
        "--config",
        str(paths["config"]),
        "verify",
        "--base-manifest",
        str(paths["base"]),
        "--pool-manifest",
        str(paths["pool"]),
        "--processed-root",
        str(paths["processed"]),
        "--embedding-root",
        str(paths["embeddings"]),
        "--non-official-bypass",
    ]
    if aggregate is not None:
        args.extend(
            [
                "--results-root",
                str(paths["results"]),
                "--aggregate-artifact",
                str(aggregate),
            ]
        )
    return args


def _probe_args(
    paths: dict[str, Path],
    output: Path,
    *,
    task_index: int = 0,
) -> list[str]:
    return [
        "--config",
        str(paths["config"]),
        "probe-fit",
        "--base-manifest",
        str(paths["base"]),
        "--pool-manifest",
        str(paths["pool"]),
        "--processed-root",
        str(paths["processed"]),
        "--embedding-root",
        str(paths["embeddings"]),
        "--task-index",
        str(task_index),
        "--output",
        str(output),
        "--non-official-bypass",
    ]

def test_exact_cv_module_exposes_three_stage_commands() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "self_improve_protein.exact_cv_cli", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    for command in (
        "probe-fit",
        "run-task",
        "verify-task",
        "aggregate",
        "verify",
    ):
        assert command in completed.stdout


def test_exact_cv_slurm_is_cpu_45_by_45_then_aggregate_chain() -> None:
    task = Path("slurm/exact_cv_task_array.sbatch")
    verify = Path("slurm/exact_cv_verify_array.sbatch")
    aggregate = Path("slurm/exact_cv_aggregate.sbatch")
    submit = Path("slurm/submit_exact_cv_screen.sh")
    for path in (task, verify, aggregate, submit):
        completed = subprocess.run(
            ["bash", "-n", str(path)],
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, completed.stderr
    for path in (task, verify, aggregate):
        text = path.read_text(encoding="utf-8")
        assert "#SBATCH --requeue" in text
        assert "--gpus" not in text
        assert "export OPENBLAS_CORETYPE=Haswell" in text
        assert "python -m self_improve_protein.exact_cv_cli" in text
    submit_text = submit.read_text(encoding="utf-8")
    assert submit_text.count("--array=0-44") == 2
    assert submit_text.count("--dependency=afterok:") == 2
    assert "processed/v0" in submit_text
    assert "embeddings/v0" in submit_text
    assert "SI_EXACT_CV_PROBE" in submit_text
    assert "--probe-artifact" in submit_text
    assert submit_text.index(" verify \\") < submit_text.index('raw="$(sbatch')
    assert submit.stat().st_mode & 0o111


def test_submitter_never_calls_sbatch_when_probe_preflight_is_missing(
    tmp_path: Path,
) -> None:
    fake_repo = tmp_path / "repo"
    fake_bin = tmp_path / "bin"
    (fake_repo / ".venv" / "bin").mkdir(parents=True)
    fake_bin.mkdir()
    fake_python = fake_repo / ".venv" / "bin" / "python"
    fake_python.write_text(
        "#!/usr/bin/env bash\n"
        "probe=''\n"
        "while [[ $# -gt 0 ]]; do\n"
        "  if [[ $1 == --probe-artifact ]]; then probe=$2; shift 2; else shift; fi\n"
        "done\n"
        "[[ -n $probe && -f $probe ]]\n",
        encoding="utf-8",
    )
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
        "SI_RUN_ID": "exact-cv-missing-probe",
        "SI_SLURM_CONF": str(tmp_path / "slurm.conf"),
    }

    completed = subprocess.run(
        [str(Path("slurm/submit_exact_cv_screen.sh").resolve())],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert completed.returncode != 0
    assert not sbatch_log.exists()


def test_submitter_preflights_three_stage_chain_and_writes_job_manifest(
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
        "case $count in 1) printf '301\\n';; "
        "2) printf '302\\n';; *) printf '303\\n';; esac\n",
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
        "SI_RUN_ID": "exact-cv-fixed-run",
        "SI_SLURM_CONF": str(tmp_path / "slurm.conf"),
    }

    completed = subprocess.run(
        [str(Path("slurm/submit_exact_cv_screen.sh").resolve())],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert completed.returncode == 0, completed.stderr
    manifest_path = (
        fake_repo / "local" / "slurm" / "exact-cv-fixed-run" / "job_ids.json"
    )
    assert json.loads(manifest_path.read_text(encoding="utf-8")) == {
        "jobs": {"aggregate": "303", "task": "301", "verify": "302"},
        "kind": "exact_cv_screen_jobs",
        "schema_version": 1,
    }
    calls = sbatch_log.read_text(encoding="utf-8").splitlines()
    assert "--array=0-44" in calls[0]
    assert "--array=0-44" in calls[1]
    assert "--dependency=afterok:301" in calls[1]
    assert "--dependency=afterok:302" in calls[2]


def test_exact_cv_cli_has_no_method_or_outcome_overrides() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "self_improve_protein.exact_cv_cli",
            "run-task",
            "--help",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    for forbidden in (
        "--q",
        "--pseudo-weight",
        "--ridge-lambda",
        "--fold-count",
        "--fold-purpose",
        "--teacher-column",
        "--untouched",
    ):
        assert forbidden not in completed.stdout
    verify_help = subprocess.run(
        [
            sys.executable,
            "-m",
            "self_improve_protein.exact_cv_cli",
            "verify",
            "--help",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert verify_help.returncode == 0, verify_help.stderr
    assert "--probe-artifact" in verify_help.stdout


def test_probe_fit_loads_only_96_labeled_outcomes_before_fitting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _workspace(
        tmp_path,
        embedding_width=480,
        materialized_offsets=(8,),
    )
    output = tmp_path / "real-shape-fit-probe.json"
    calls: list[int] = []
    real_ordered = exact_cv_cli._ordered_outcomes
    real_sha256_file = exact_cv_cli.sha256_file

    def labeled_projection(*args: object, **kwargs: object) -> object:
        requested = kwargs.get("requested_hashes")
        assert isinstance(requested, tuple)
        calls.append(len(requested))
        return real_ordered(*args, **kwargs)  # type: ignore[arg-type]

    def forbidden_hidden_loader(*args: object, **kwargs: object) -> object:
        raise AssertionError("fit probe must never load hidden outcomes")

    def forbid_full_processed_hash(path: Path) -> str:
        if path.suffix == ".parquet":
            raise AssertionError("fit probe must not hash the full outcome file")
        return real_sha256_file(path)

    monkeypatch.setattr(exact_cv_cli, "_ordered_outcomes", labeled_projection)
    monkeypatch.setattr(
        exact_cv_cli,
        "load_evaluation_labels",
        forbidden_hidden_loader,
    )
    monkeypatch.setattr(exact_cv_cli, "sha256_file", forbid_full_processed_hash)

    result = RUNNER.invoke(exact_cv_cli.app, _probe_args(paths, output))

    assert result.exit_code == 0, result.output
    assert calls == [96]
    payload = json.loads(output.read_bytes())
    assert payload["kind"] == "exact_cv_real_shape_fit_probe"
    assert payload["hidden_outcomes_loaded"] is False
    assert payload["dimensions"] == {
        "embedding_width": 480,
        "n_labeled": 96,
        "n_test": 1000,
        "n_unlabeled": 2000,
    }
    assert payload["greedy"]["q"] == 192
    assert payload["greedy"]["reanchor_steps"] == [0, 1, 24, 48, 72, 96, 192]
    assert payload["greedy"]["minimum_sherman_morrison_denominator"] >= 1.0
    assert payload["execution"]["official"] is False
    assert output.stat().st_size < 100_000
    protocol = load_protocol(paths["config"])
    _, pool = exact_cv_cli._load_manifests(
        protocol=protocol,
        base_manifest_path=paths["base"],
        pool_manifest_path=paths["pool"],
    )
    official_payload = json.loads(json.dumps(payload))
    official_payload["execution"]["official"] = True
    official_payload["execution"]["bypass"] = None
    exact_cv_cli._validate_probe_against_roots(
        official_payload,
        protocol=protocol,
        pool=pool,
        git_commit=official_payload["provenance"]["git_commit"],
        base_manifest_path=paths["base"],
        pool_manifest_path=paths["pool"],
        processed_root=paths["processed"],
        embedding_root=paths["embeddings"],
    )
    assert calls == [96, 96]
    assert calls == [96, 96]
    tampered_probe = json.loads(json.dumps(official_payload))
    tampered_probe["provenance"]["embedding_npy_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="current roots"):
        exact_cv_cli._validate_probe_against_roots(
            tampered_probe,
            protocol=protocol,
            pool=pool,
            git_commit=official_payload["provenance"]["git_commit"],
            base_manifest_path=paths["base"],
            pool_manifest_path=paths["pool"],
            processed_root=paths["processed"],
            embedding_root=paths["embeddings"],
        )


def test_probe_fit_is_write_once_and_rejects_tampered_existing_probe(
    tmp_path: Path,
) -> None:
    paths = _workspace(
        tmp_path,
        embedding_width=480,
        materialized_offsets=(8,),
    )
    output = tmp_path / "real-shape-fit-probe.json"

    first = RUNNER.invoke(exact_cv_cli.app, _probe_args(paths, output))
    assert first.exit_code == 0, first.output
    original = output.read_bytes()
    repeated = RUNNER.invoke(exact_cv_cli.app, _probe_args(paths, output))
    assert repeated.exit_code == 0, repeated.output
    assert json.loads(repeated.output.splitlines()[-1])["created"] is False
    assert output.read_bytes() == original

    changed = json.loads(original)
    changed["greedy"]["final_cv_mse"] += 1.0
    output.write_text(json.dumps(changed) + "\n", encoding="utf-8")
    rejected = RUNNER.invoke(exact_cv_cli.app, _probe_args(paths, output))
    assert rejected.exit_code != 0
    assert "mismatched existing artifact" in rejected.output


def test_task_freezes_exact_fit_digest_before_hidden_label_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _workspace(tmp_path)
    events: list[str] = []
    real_fit = exact_cv_cli.fit_exact_cv_task
    real_digest = exact_cv_cli.canonical_exact_cv_fit_digest
    real_load = exact_cv_cli.load_evaluation_labels

    def fit_then_record(*args: object, **kwargs: object) -> object:
        fitted = real_fit(*args, **kwargs)  # type: ignore[arg-type]
        assert len(fitted.greedy.steps) == 192
        assert len(fitted.prefixes) == 5
        events.append("fit")
        return fitted

    def digest_then_record(*args: object, **kwargs: object) -> str:
        digest = real_digest(*args, **kwargs)
        events.append("fit_digest")
        return digest

    def hidden_load(*args: object, **kwargs: object) -> object:
        assert events == ["fit", "fit_digest"]
        events.append("hidden_labels")
        return real_load(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(exact_cv_cli, "fit_exact_cv_task", fit_then_record)
    monkeypatch.setattr(
        exact_cv_cli,
        "canonical_exact_cv_fit_digest",
        digest_then_record,
    )
    monkeypatch.setattr(exact_cv_cli, "load_evaluation_labels", hidden_load)

    result = RUNNER.invoke(exact_cv_cli.app, _task_args(paths, 0))

    assert result.exit_code == 0, result.output
    assert events == ["fit", "fit_digest", "hidden_labels"]


def test_task_exact_verification_receipt_is_write_once_and_binds_artifact(
    tmp_path: Path,
) -> None:
    paths = _workspace(tmp_path)

    first = RUNNER.invoke(exact_cv_cli.app, _task_args(paths, 0))
    assert first.exit_code == 0, first.output
    task_path = paths["results"] / "tasks" / "ASSAY_08" / "seed_0.json"
    original_task = task_path.read_bytes()
    task = json.loads(original_task)
    assert task["kind"] == "exact_cv_task_result"
    assert task["card"] == {
        "id": CARD_ID,
        "sha256": CARD_SHA,
    }
    assert len(task["greedy"]["steps"]) == 192
    assert [item["q"] for item in task["prefixes"]] == [24, 48, 72, 96, 192]
    assert task["task"]["phase"] == "exact_cv_screen"
    locked_indices = {
        row["name"]: set(row["selected_indices"])
        for row in task["reference_methods"]
    }
    overlap_methods = {
        "overlap_full": "ours",
        "overlap_no_hessian": "no_hessian",
        "overlap_random": "random",
        "overlap_top_teacher": "top_teacher",
    }
    for prefix in task["prefixes"]:
        selected = set(prefix["selected_indices"])
        for field, method in overlap_methods.items():
            assert prefix[field] == (
                len(selected & locked_indices[method]) / prefix["q"]
            )

    repeated = RUNNER.invoke(exact_cv_cli.app, _task_args(paths, 0))
    assert repeated.exit_code == 0, repeated.output
    assert json.loads(repeated.output.splitlines()[-1])["created"] is False
    assert task_path.read_bytes() == original_task

    verified = RUNNER.invoke(exact_cv_cli.app, _verify_task_args(paths, 0))
    assert verified.exit_code == 0, verified.output
    receipt_path = (
        paths["results"] / "receipts" / "ASSAY_08" / "seed_0.json"
    )
    original_receipt = receipt_path.read_bytes()
    receipt = json.loads(original_receipt)
    assert receipt["kind"] == "exact_cv_verification_receipt"
    assert receipt["task_artifact_sha256"] == sha256_file(task_path)
    assert receipt["task"] == {"assay_id": "ASSAY_08", "index": 0, "seed": 0}

    verified_again = RUNNER.invoke(
        exact_cv_cli.app,
        _verify_task_args(paths, 0),
    )
    assert verified_again.exit_code == 0, verified_again.output
    assert receipt_path.read_bytes() == original_receipt

    changed = json.loads(original_task)
    changed["prefixes"][-1]["mse"] += 1.0
    task_path.write_text(json.dumps(changed) + "\n", encoding="utf-8")
    rejected = RUNNER.invoke(exact_cv_cli.app, _verify_task_args(paths, 0))
    assert rejected.exit_code != 0
    task_path.write_bytes(original_task)


def test_aggregate_consumes_receipts_without_refitting_and_applies_dual_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _workspace(tmp_path)
    for index in range(45):
        task = RUNNER.invoke(exact_cv_cli.app, _task_args(paths, index))
        assert task.exit_code == 0, f"task {index}: {task.output}"
        receipt = RUNNER.invoke(
            exact_cv_cli.app,
            _verify_task_args(paths, index),
        )
        assert receipt.exit_code == 0, f"receipt {index}: {receipt.output}"

    def forbidden_rebuild(*args: object, **kwargs: object) -> object:
        raise AssertionError("aggregate must not recompute fits")

    monkeypatch.setattr(exact_cv_cli, "_build_task_payload", forbidden_rebuild)
    output = tmp_path / "aggregate.json"
    aggregated = RUNNER.invoke(
        exact_cv_cli.app,
        _aggregate_args(paths, output),
    )
    assert aggregated.exit_code == 0, aggregated.output
    original = output.read_bytes()
    payload = json.loads(original)
    assert payload["kind"] == "exact_cv_aggregate_result"
    assert payload["analysis"] == {
        "confirmatory_claim": False,
        "inference_unit": "assay",
        "scope": "exploratory_exact_cv_screen",
        "verification_model": (
            "independent_exact_rebuild_receipts_noncryptographic"
        ),
        "verification_trust_scope": (
            "nonofficial_bypass_filesystem_only"
        ),
    }
    assert payload["execution"]["official"] is False
    assert payload["execution"]["bypass"] == (
        "non_official_test_or_development"
    )
    assert payload["execution"]["numerical_policy"] == (
        "threadpoolctl:blas_threads=1"
    )
    assert isinstance(payload["execution"]["numerical_runtime"], dict)
    invalid_execution = json.loads(json.dumps(payload))
    invalid_execution["execution"]["official"] = True
    with pytest.raises(ValueError, match="aggregate contract"):
        exact_cv_cli._validate_aggregate_payload(invalid_execution)
    valid_official_execution = json.loads(json.dumps(payload))
    valid_official_execution["execution"]["official"] = True
    valid_official_execution["execution"]["bypass"] = None
    valid_official_execution["analysis"]["verification_trust_scope"] = (
        "trusted_cluster_execution_clean_git_and_filesystem"
    )
    exact_cv_cli._validate_aggregate_payload(valid_official_execution)
    assert len(payload["grid"]["screen_tasks"]) == 45
    assert len(payload["grid"]["primary_tasks"]) == 40
    assert payload["grid"]["development_assay_id"] == "ASSAY_08"
    assert "ASSAY_08" not in payload["grid"]["primary_assay_ids"]
    assert len(payload["primary_endpoint_results"]) == 40 * 2
    assert len(payload["prefix_descriptive"]) == 5
    expected_overlaps: dict[int, dict[str, list[float]]] = {
        q: {
            field: []
            for field in (
                "overlap_full",
                "overlap_no_hessian",
                "overlap_random",
                "overlap_top_teacher",
            )
        }
        for q in (24, 48, 72, 96, 192)
    }
    for assay_id in (f"ASSAY_{index:02d}" for index in range(8)):
        for seed in range(5):
            task_path = paths["results"] / "tasks" / assay_id / f"seed_{seed}.json"
            task_payload = json.loads(task_path.read_bytes())
            for prefix in task_payload["prefixes"]:
                for field in expected_overlaps[prefix["q"]]:
                    expected_overlaps[prefix["q"]][field].append(prefix[field])
    for row in payload["prefix_descriptive"]:
        q = row["q"]
        for field, values in expected_overlaps[q].items():
            assert row[f"mean_{field}"] == float(np.mean(values))
    promotion = payload["promotion"]
    assert promotion["mse"]["gain"] == "random_mse_minus_exact_cv_mse"
    assert promotion["spearman"]["gain"] == (
        "exact_cv_spearman_minus_random_spearman"
    )
    for gate in (promotion["mse"], promotion["spearman"]):
        assert gate["task_win_threshold"] == 25
        assert gate["assay_win_threshold"] == 5
        assert gate["task_total"] == 40
        assert gate["assay_total"] == 8
        assert gate["passed"] == (
            gate["assay_macro_mean_gain"] > 0.0
            and gate["task_wins"] >= 25
            and gate["assay_wins"] >= 5
        )
    assert promotion["both_passed"] == (
        promotion["mse"]["passed"] and promotion["spearman"]["passed"]
    )
    assert payload["provenance"]["task_count"] == 45
    assert payload["provenance"]["receipt_count"] == 45
    serialized = json.dumps(payload, sort_keys=True)
    for untouched in (f"ASSAY_{index:02d}" for index in range(9, 35)):
        assert untouched not in serialized

    repeated = RUNNER.invoke(
        exact_cv_cli.app,
        _aggregate_args(paths, output),
    )
    assert repeated.exit_code == 0, repeated.output
    assert output.read_bytes() == original

    verified = RUNNER.invoke(
        exact_cv_cli.app,
        _verify_args(paths, aggregate=output),
    )
    assert verified.exit_code == 0, verified.output
    verified_payload = json.loads(verified.output.splitlines()[-1])
    assert verified_payload["verified"] == [
        "base_manifest",
        "pool",
        "screen_grid",
        "inputs",
        "tasks",
        "receipts",
        "aggregate",
    ]

    # Receipts are intentionally non-cryptographic attestations. A coordinated
    # rewrite of both task and receipt inside the trusted filesystem is not
    # claimed to be detectable by the receipt-only aggregate.
    forged_task_path = paths["results"] / "tasks" / "ASSAY_00" / "seed_0.json"
    forged_receipt_path = (
        paths["results"] / "receipts" / "ASSAY_00" / "seed_0.json"
    )
    original_forged_task = forged_task_path.read_bytes()
    original_forged_receipt = forged_receipt_path.read_bytes()
    forged_task = json.loads(original_forged_task)
    forged_task["prefixes"][0]["mse"] += 0.25
    forged_task_path.write_text(json.dumps(forged_task) + "\n", encoding="utf-8")
    forged_receipt = exact_cv_cli._receipt_from_verified_task(
        forged_task,
        task_index=5,
        task_artifact_sha256=sha256_file(forged_task_path),
        git_commit=forged_task["provenance"]["git_commit"],
    )
    forged_receipt_path.write_text(
        json.dumps(forged_receipt) + "\n",
        encoding="utf-8",
    )
    coordinated_output = tmp_path / "coordinated-noncryptographic.json"
    coordinated = RUNNER.invoke(
        exact_cv_cli.app,
        _aggregate_args(paths, coordinated_output),
    )
    assert coordinated.exit_code == 0, coordinated.output
    coordinated_payload = json.loads(coordinated_output.read_bytes())
    assert coordinated_payload["analysis"]["verification_model"] == (
        "independent_exact_rebuild_receipts_noncryptographic"
    )
    assert coordinated_payload["analysis"]["verification_trust_scope"] == (
        "nonofficial_bypass_filesystem_only"
    )
    forged_task_path.write_bytes(original_forged_task)
    forged_receipt_path.write_bytes(original_forged_receipt)

    receipt_path = (
        paths["results"] / "receipts" / "ASSAY_08" / "seed_0.json"
    )
    changed_receipt = json.loads(receipt_path.read_bytes())
    changed_receipt["task_artifact_sha256"] = "f" * 64
    receipt_path.write_text(json.dumps(changed_receipt) + "\n", encoding="utf-8")
    rejected = RUNNER.invoke(
        exact_cv_cli.app,
        _aggregate_args(paths, tmp_path / "rejected.json"),
    )
    assert rejected.exit_code != 0
    assert not (tmp_path / "rejected.json").exists()
