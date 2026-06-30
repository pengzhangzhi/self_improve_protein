from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pytest
import yaml
from typer.testing import CliRunner

import self_improve_protein.crossfit_pool_cli as pool_cli
from self_improve_protein.config import Protocol
from self_improve_protein.crossfit_data import (
    CROSSFIT_CARD_ID,
    CROSSFIT_CARD_SHA256,
    CROSSFIT_POOL_SCHEMA_ID,
    CrossfitPoolManifest,
    write_crossfit_pool_manifest,
)
from self_improve_protein.data import (
    ManifestSource,
    ManifestSources,
    SelectedAssayManifest,
    row_hash,
)
from self_improve_protein.embeddings import (
    EmbeddingCacheSources,
    write_embedding_cache,
)

RUNNER = CliRunner()


@dataclass(frozen=True, slots=True)
class PoolCliCase:
    protocol: Protocol
    config: Path
    base_manifest: Path
    pool_manifest: Path
    dms_zip: Path
    scores_zip: Path
    metadata_csv: Path
    processed_root: Path
    embedding_root: Path
    manifest: CrossfitPoolManifest
    frames: dict[str, pd.DataFrame]


def _protocol() -> Protocol:
    return Protocol(
        data_release="v1.3",
        substitutions_url="https://example.test/dms.zip",
        zero_shot_scores_url="https://example.test/scores.zip",
        metadata_url="https://example.test/metadata.csv",
        proteingym_upstream_commit="1" * 40,
        substitutions_sha256="2" * 64,
        zero_shot_scores_sha256="3" * 64,
        metadata_sha256="4" * 64,
        teacher_column="ESM1v_ensemble",
        model="facebook/esm2_t12_35M_UR50D",
        model_revision="6fbf070e65b0b7291e7bbcd451118c216cff79d8",
        working_size=3,
        n_labeled=1,
        n_unlabeled=1,
        n_test=1,
        q=1,
        pseudo_weight=0.1,
        ridge_lambda=0.01,
        damping=0.0001,
        seeds=(0,),
        assay_count=8,
        max_length=512,
        preprocessing={
            "feature_scaling": "scalar_rms",
            "student_fit": "no_intercept",
            "label_ddof": 0,
        },
        analysis_seed=20260629,
        random_diagnostic_replicates=100,
    )


def _frame(dms_id: str, teacher_column: str) -> pd.DataFrame:
    rows = pd.DataFrame(
        {
            "dms_id": [dms_id] * 3,
            "mutant": ["A1C", "A2D", "A3E"],
            "mutated_sequence": ["ACDE", "ACDF", "ACDG"],
            "DMS_score": [101.0, 202.0, 303.0],
            "source_row": [0, 1, 2],
            teacher_column: [0.1, 0.2, 0.3],
        }
    )
    rows["sequence_hash"] = [
        row_hash(dms_id, mutant, sequence)
        for mutant, sequence in rows.loc[
            :, ["mutant", "mutated_sequence"]
        ].itertuples(index=False, name=None)
    ]
    return rows.sort_values("sequence_hash", kind="stable").reset_index(drop=True)


def _case(tmp_path: Path) -> PoolCliCase:
    protocol = _protocol()
    config = tmp_path / "config.yaml"
    config.write_text(
        yaml.safe_dump(protocol.model_dump(mode="json"), sort_keys=False),
        encoding="utf-8",
    )
    eligible_ids = tuple(f"ASSAY_{index:02d}" for index in range(35))
    all_frames = {
        dms_id: _frame(dms_id, protocol.teacher_column) for dms_id in eligible_ids
    }
    records = tuple(
        SelectedAssayManifest(
            dms_id=dms_id,
            usable_count=len(all_frames[dms_id]),
            sequence_length=4,
            row_hashes=tuple(all_frames[dms_id]["sequence_hash"]),
        )
        for dms_id in eligible_ids
    )
    screen_ids = (eligible_ids[8], *eligible_ids[:8])
    untouched_ids = eligible_ids[9:]
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
    manifest = CrossfitPoolManifest(
        schema_id=CROSSFIT_POOL_SCHEMA_ID,
        schema_version=1,
        card_id=CROSSFIT_CARD_ID,
        card_sha256=CROSSFIT_CARD_SHA256,
        base_manifest_sha256="5" * 64,
        protocol_sha256="6" * 64,
        data_release=protocol.data_release,
        teacher_column=protocol.teacher_column,
        sources=sources,
        upstream_revision=protocol.proteingym_upstream_commit,
        max_length=protocol.max_length,
        working_size=protocol.working_size,
        eligible_assay_ids=eligible_ids,
        screen_ids=screen_ids,
        untouched_ids=untouched_ids,
        selected_assays=records,
    )
    base_manifest = tmp_path / "base-manifest.json"
    base_manifest.write_text("{}\n", encoding="utf-8")
    dms_zip = tmp_path / "dms.zip"
    scores_zip = tmp_path / "scores.zip"
    metadata_csv = tmp_path / "metadata.csv"
    for path in (dms_zip, scores_zip, metadata_csv):
        path.write_bytes(b"fixture")
    return PoolCliCase(
        protocol=protocol,
        config=config,
        base_manifest=base_manifest,
        pool_manifest=tmp_path / "pool-manifest.json",
        dms_zip=dms_zip,
        scores_zip=scores_zip,
        metadata_csv=metadata_csv,
        processed_root=tmp_path / "processed",
        embedding_root=tmp_path / "embeddings",
        manifest=manifest,
        frames={dms_id: all_frames[dms_id] for dms_id in untouched_ids},
    )


def _base_args(case: PoolCliCase) -> list[str]:
    return ["--config", str(case.config)]


def _prepare_args(case: PoolCliCase) -> list[str]:
    return [
        *_base_args(case),
        "prepare-pool",
        "--base-manifest",
        str(case.base_manifest),
        "--pool-manifest",
        str(case.pool_manifest),
        "--dms-zip",
        str(case.dms_zip),
        "--scores-zip",
        str(case.scores_zip),
        "--metadata-csv",
        str(case.metadata_csv),
        "--processed-root",
        str(case.processed_root),
        "--non-official-bypass",
    ]


def _terminal_payload(result: Any) -> dict[str, object]:
    lines = [json.loads(line) for line in result.stdout.splitlines() if line.strip()]
    return lines[-1]


def _install_builder_stub(
    monkeypatch: pytest.MonkeyPatch,
    case: PoolCliCase,
) -> list[dict[str, object]]:
    calls: list[dict[str, object]] = []

    def fake_build(
        **kwargs: object,
    ) -> tuple[CrossfitPoolManifest, dict[str, pd.DataFrame]]:
        calls.append(kwargs)
        callback = kwargs["on_untouched_frame"]
        assert callable(callback)
        for dms_id, frame in case.frames.items():
            callback(dms_id, frame.copy(deep=True))
        return case.manifest, {
            key: value.copy(deep=True) for key, value in case.frames.items()
        }

    monkeypatch.setattr(pool_cli, "build_crossfit_pool", fake_build)
    monkeypatch.setattr(
        pool_cli,
        "validate_crossfit_pool_provenance",
        lambda *args, **kwargs: None,
    )
    return calls


def _write_pool_artifacts(case: PoolCliCase) -> None:
    write_crossfit_pool_manifest(case.pool_manifest, case.manifest)
    case.processed_root.mkdir(parents=True, exist_ok=True)
    for dms_id, frame in case.frames.items():
        frame.to_parquet(case.processed_root / f"{dms_id}.parquet", index=False)


def _embedding_sources(protocol: Protocol) -> EmbeddingCacheSources:
    return EmbeddingCacheSources(
        proteingym_upstream_commit=protocol.proteingym_upstream_commit,
        substitutions_sha256=protocol.substitutions_sha256,
        zero_shot_scores_sha256=protocol.zero_shot_scores_sha256,
        metadata_sha256=protocol.metadata_sha256,
    )


def test_module_exposes_separate_pool_commands() -> None:
    result = RUNNER.invoke(pool_cli.app, ["--help"])

    assert result.exit_code == 0, result.output
    assert "prepare-pool" in result.output
    assert "embed-assay" in result.output
    assert "verify" in result.output


def test_prepare_pool_writes_26_frames_then_canonical_manifest_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _case(tmp_path)
    calls = _install_builder_stub(monkeypatch, case)

    first = RUNNER.invoke(pool_cli.app, _prepare_args(case))

    assert first.exit_code == 0, first.output
    assert len(calls) == 1
    written_ids = tuple(
        sorted(path.stem for path in case.processed_root.glob("*.parquet"))
    )
    assert written_ids == tuple(sorted(case.manifest.untouched_ids))
    assert case.pool_manifest.is_file()
    payload = _terminal_payload(first)
    assert payload["processed_created"] == 26
    assert payload["manifest_created"] is True
    sample_id = case.manifest.untouched_ids[0]
    written = pd.read_parquet(case.processed_root / f"{sample_id}.parquet")
    assert written["DMS_score"].tolist() == case.frames[sample_id]["DMS_score"].tolist()

    original_manifest = case.pool_manifest.read_bytes()
    second = RUNNER.invoke(pool_cli.app, _prepare_args(case))

    assert second.exit_code == 0, second.output
    assert case.pool_manifest.read_bytes() == original_manifest
    second_payload = _terminal_payload(second)
    assert second_payload["processed_created"] == 0
    assert second_payload["manifest_created"] is False


def test_prepare_pool_rejects_mismatched_existing_processed_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _case(tmp_path)
    _install_builder_stub(monkeypatch, case)
    first = RUNNER.invoke(pool_cli.app, _prepare_args(case))
    assert first.exit_code == 0, first.output
    dms_id = case.manifest.untouched_ids[0]
    path = case.processed_root / f"{dms_id}.parquet"
    changed = pd.read_parquet(path)
    changed.loc[0, "DMS_score"] += 1.0
    changed.to_parquet(path, index=False)

    repeated = RUNNER.invoke(pool_cli.app, _prepare_args(case))

    assert repeated.exit_code == 1
    assert "mismatched existing processed artifact" in repeated.output


def test_prepare_pool_dry_run_never_calls_builder_or_writes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _case(tmp_path)

    def forbidden(**kwargs: object) -> object:
        raise AssertionError(f"builder called during dry-run: {kwargs}")

    monkeypatch.setattr(pool_cli, "build_crossfit_pool", forbidden)
    args = _prepare_args(case)
    args.insert(2, "--dry-run")

    result = RUNNER.invoke(pool_cli.app, args)

    assert result.exit_code == 0, result.output
    assert _terminal_payload(result)["status"] == "planned"
    assert not case.pool_manifest.exists()
    assert not case.processed_root.exists()


def test_embed_assay_resolves_untouched_index_and_uses_only_identity_frame(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _case(tmp_path)
    _write_pool_artifacts(case)
    monkeypatch.setattr(
        pool_cli,
        "validate_crossfit_pool_provenance",
        lambda *args, **kwargs: None,
    )
    captured: dict[str, object] = {}

    def fake_cache(npy_path: Path, metadata_path: Path, **kwargs: object) -> np.ndarray:
        captured.update(kwargs)
        npy_path.parent.mkdir(parents=True, exist_ok=True)
        array = np.zeros((case.protocol.working_size, 480), dtype=np.float32)
        np.save(npy_path, array, allow_pickle=False)
        metadata_path.write_text("{}\n", encoding="utf-8")
        return array

    monkeypatch.setattr(pool_cli, "get_or_create_embedding_cache", fake_cache)
    result = RUNNER.invoke(
        pool_cli.app,
        [
            *_base_args(case),
            "embed-assay",
            "--base-manifest",
            str(case.base_manifest),
            "--pool-manifest",
            str(case.pool_manifest),
            "--processed-root",
            str(case.processed_root),
            "--embedding-root",
            str(case.embedding_root),
            "--assay-index",
            "0",
            "--device",
            "cpu",
            "--non-official-bypass",
        ],
    )

    assert result.exit_code == 0, result.output
    dms_id = case.manifest.untouched_ids[0]
    assert captured["dms_id"] == dms_id
    assert captured["row_hashes"] == next(
        record.row_hashes
        for record in case.manifest.selected_assays
        if record.dms_id == dms_id
    )
    assert tuple(captured["sequences"]) == tuple(
        case.frames[dms_id]["mutated_sequence"]
    )
    assert "DMS_score" not in captured


@pytest.mark.parametrize(
    "selector_args",
    [
        (),
        ("--assay-id", "ASSAY_09", "--assay-index", "0"),
        ("--assay-index", "26"),
        ("--assay-id", "ASSAY_00"),
    ],
)
def test_embed_assay_requires_exactly_one_untouched_selector(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    selector_args: tuple[str, ...],
) -> None:
    case = _case(tmp_path)
    write_crossfit_pool_manifest(case.pool_manifest, case.manifest)
    monkeypatch.setattr(
        pool_cli,
        "validate_crossfit_pool_provenance",
        lambda *args, **kwargs: None,
    )

    result = RUNNER.invoke(
        pool_cli.app,
        [
            *_base_args(case),
            "embed-assay",
            "--base-manifest",
            str(case.base_manifest),
            "--pool-manifest",
            str(case.pool_manifest),
            "--processed-root",
            str(case.processed_root),
            "--embedding-root",
            str(case.embedding_root),
            *selector_args,
            "--non-official-bypass",
        ],
    )

    assert result.exit_code == 1
    assert "untouched" in result.output or "exactly one" in result.output


def test_verify_requires_canonical_manifest_and_all_processed_frames(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _case(tmp_path)
    _write_pool_artifacts(case)
    monkeypatch.setattr(
        pool_cli,
        "validate_crossfit_pool_provenance",
        lambda *args, **kwargs: None,
    )
    args = [
        *_base_args(case),
        "verify",
        "--base-manifest",
        str(case.base_manifest),
        "--pool-manifest",
        str(case.pool_manifest),
        "--processed-root",
        str(case.processed_root),
        "--non-official-bypass",
    ]

    complete = RUNNER.invoke(pool_cli.app, args)

    assert complete.exit_code == 0, complete.output
    assert _terminal_payload(complete)["processed_verified"] == 26
    missing_id = case.manifest.untouched_ids[-1]
    (case.processed_root / f"{missing_id}.parquet").unlink()
    missing = RUNNER.invoke(pool_cli.app, args)
    assert missing.exit_code == 1
    assert "outcome-free working set" in missing.output

    case.frames[missing_id].to_parquet(
        case.processed_root / f"{missing_id}.parquet",
        index=False,
    )
    case.pool_manifest.write_bytes(case.pool_manifest.read_bytes() + b"\n")
    noncanonical = RUNNER.invoke(pool_cli.app, args)
    assert noncanonical.exit_code == 1
    assert "canonical" in noncanonical.output


def test_verify_optionally_validates_all_26_embedding_caches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _case(tmp_path)
    _write_pool_artifacts(case)
    monkeypatch.setattr(
        pool_cli,
        "validate_crossfit_pool_provenance",
        lambda *args, **kwargs: None,
    )
    sources = _embedding_sources(case.protocol)
    for record in case.manifest.selected_assays:
        if record.dms_id not in case.manifest.untouched_ids:
            continue
        write_embedding_cache(
            case.embedding_root / f"{record.dms_id}.npy",
            case.embedding_root / f"{record.dms_id}.json",
            np.zeros((case.protocol.working_size, 4), dtype=np.float32),
            dms_id=record.dms_id,
            row_hashes=record.row_hashes,
            model_id=case.protocol.model,
            model_revision=case.protocol.model_revision,
            sources=sources,
        )
    args = [
        *_base_args(case),
        "verify",
        "--base-manifest",
        str(case.base_manifest),
        "--pool-manifest",
        str(case.pool_manifest),
        "--processed-root",
        str(case.processed_root),
        "--embedding-root",
        str(case.embedding_root),
        "--non-official-bypass",
    ]

    complete = RUNNER.invoke(pool_cli.app, args)

    assert complete.exit_code == 0, complete.output
    assert _terminal_payload(complete)["embeddings_verified"] == 26
    missing_id = case.manifest.untouched_ids[0]
    (case.embedding_root / f"{missing_id}.npy").unlink()
    missing = RUNNER.invoke(pool_cli.app, args)
    assert missing.exit_code == 1
    assert "missing or incomplete" in missing.output


def test_pool_slurm_stages_are_strict_and_use_exact_26_gpu_array() -> None:
    prepare = Path("slurm/crossfit_prepare_pool.sbatch")
    embed = Path("slurm/crossfit_embed_pool_array.sbatch")

    for path in (prepare, embed):
        completed = subprocess.run(
            ["bash", "-n", str(path)],
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, completed.stderr
        text = path.read_text(encoding="utf-8")
        assert "set -euo pipefail" in text
        assert "python -m self_improve_protein.crossfit_pool_cli" in text
        assert "--requeue" in text
        assert "require_openblas_coretype" in text
        assert "DMS_score" not in text
        assert "make_split" not in text
    prepare_text = prepare.read_text(encoding="utf-8")
    assert "prepare-pool" in prepare_text
    assert "--gpus" not in prepare_text
    assert "#SBATCH --time=04:00:00" in prepare_text
    assert "${SI_PROCESSED_ROOT:-" in prepare_text
    embed_text = embed.read_text(encoding="utf-8")
    assert "#SBATCH --array=0-25" in embed_text
    assert "--gpus-per-node=1" in embed_text
    assert "embed-assay" in embed_text
    assert "${SI_PROCESSED_ROOT:-" in embed_text
    assert "${SI_EMBEDDING_ROOT:-" in embed_text
    assert '--assay-index "$SLURM_ARRAY_TASK_ID"' in embed_text
