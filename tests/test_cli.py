import json
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pytest
import yaml
from typer.testing import CliRunner

from self_improve_protein.cli import app, load_evaluation_labels
from self_improve_protein.config import Protocol, load_protocol
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

RUNNER = CliRunner()
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


def _sequence(index: int) -> str:
    return "A" + AMINO_ACIDS[(index // 20) % 20] + AMINO_ACIDS[index % 20]


def _synthetic_workspace(tmp_path: Path) -> dict[str, Path]:
    protocol_payload = load_protocol("configs/v0.yaml").model_dump(mode="python")
    protocol_payload.update(
        working_size=32,
        n_labeled=8,
        n_unlabeled=12,
        n_test=10,
        q=4,
        seeds=(0, 1),
        assay_count=1,
        random_diagnostic_replicates=3,
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

    selected: list[SelectedAssayManifest] = []
    for assay_offset, dms_id in enumerate(("AAA", "BBB")):
        frame = pd.DataFrame(
            {
                "dms_id": [dms_id] * protocol.working_size,
                "mutant": [f"A{index + 1}C" for index in range(protocol.working_size)],
                "mutated_sequence": [
                    _sequence(index + assay_offset * protocol.working_size)
                    for index in range(protocol.working_size)
                ],
            }
        )
        frame["sequence_hash"] = [
            row_hash(dms_id, mutant, sequence)
            for mutant, sequence in zip(
                frame["mutant"], frame["mutated_sequence"], strict=True
            )
        ]
        frame = frame.sort_values("sequence_hash", kind="stable").reset_index(drop=True)
        rng = np.random.Generator(np.random.PCG64(40 + assay_offset))
        embeddings = rng.normal(size=(protocol.working_size, 5)).astype(np.float32)
        beta = np.array([0.7, -0.4, 0.2, 0.6, -0.1])
        frame["DMS_score"] = embeddings.astype(np.float64) @ beta + np.linspace(
            -0.1, 0.1, protocol.working_size
        )
        frame[protocol.teacher_column] = (
            embeddings.astype(np.float64) @ (beta + 0.15) + 0.05
        )
        frame.to_parquet(processed_root / f"{dms_id}.parquet", index=False)
        row_hashes = tuple(str(value) for value in frame["sequence_hash"])
        selected.append(
            SelectedAssayManifest(
                dms_id=dms_id,
                usable_count=protocol.working_size,
                sequence_length=3,
                row_hashes=row_hashes,
            )
        )
        sources = EmbeddingCacheSources(
            proteingym_upstream_commit=protocol.proteingym_upstream_commit,
            substitutions_sha256=protocol.substitutions_sha256,
            zero_shot_scores_sha256=protocol.zero_shot_scores_sha256,
            metadata_sha256=protocol.metadata_sha256,
        )
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

    manifest = DataManifest(
        schema_version=1,
        data_release=protocol.data_release,
        teacher_column=protocol.teacher_column,
        sources=ManifestSources(
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
        ),
        upstream_revision=protocol.proteingym_upstream_commit,
        eligible_assay_ids=("AAA", "BBB"),
        confirmatory_ids=("AAA",),
        development_id="BBB",
        max_length=protocol.max_length,
        working_size=protocol.working_size,
        selected_assays=tuple(selected),
    )
    manifest_path = tmp_path / "manifest.json"
    write_data_manifest(manifest_path, manifest)
    return {
        "config": config,
        "manifest": manifest_path,
        "processed": processed_root,
        "embeddings": embedding_root,
        "results": results_root,
    }


def _run_task_args(paths: dict[str, Path]) -> list[str]:
    return [
        "--config",
        str(paths["config"]),
        "run-task",
        "--manifest",
        str(paths["manifest"]),
        "--processed-root",
        str(paths["processed"]),
        "--embedding-root",
        str(paths["embeddings"]),
        "--results-root",
        str(paths["results"]),
        "--assay-id",
        "BBB",
        "--seed",
        "0",
        "--mode",
        "development",
        "--non-official-bypass",
    ]


def test_cli_help_exposes_five_commands_and_global_options() -> None:
    result = RUNNER.invoke(app, ["--help"])

    assert result.exit_code == 0, result.output
    for command in ("prepare-data", "embed-assay", "run-task", "aggregate", "verify"):
        assert command in result.output
    assert "--config" in result.output
    assert "--dry-run" in result.output


def test_dry_run_emits_normalized_records_without_mutation(tmp_path: Path) -> None:
    paths = _synthetic_workspace(tmp_path)
    planned_results = tmp_path / "planned-results"
    args = _run_task_args(paths)
    args[args.index(str(paths["results"]))] = str(planned_results)
    args.insert(2, "--dry-run")

    result = RUNNER.invoke(app, args)

    assert result.exit_code == 0, result.output
    records = [json.loads(line) for line in result.output.splitlines()]
    assert [record["event"] for record in records] == ["start", "terminal"]
    assert records[-1]["status"] == "planned"
    assert records[-1]["plan"]["assay_id"] == "BBB"
    assert not planned_results.exists()


def test_synthetic_task_is_leakage_staged_idempotent_and_serializable(
    tmp_path: Path,
) -> None:
    paths = _synthetic_workspace(tmp_path)

    first = RUNNER.invoke(app, _run_task_args(paths))
    assert first.exit_code == 0, first.output
    artifact = paths["results"] / "tasks" / "BBB" / "seed_0.json"
    first_bytes = artifact.read_bytes()
    payload = json.loads(first_bytes)

    assert payload["schema_version"] == 1
    assert payload["execution"]["official"] is False
    assert payload["execution"]["bypass"] == "non_official_test_or_development"
    assert len(payload["methods"]) == 5
    assert {row["name"] for row in payload["methods"]} == {
        "supervised",
        "random",
        "top_teacher",
        "ours",
        "no_hessian",
    }
    assert len(payload["digests"]["fit"]) == 64
    assert len(payload["digests"]["evaluation"]) == 64
    assert payload["provenance"]["manifest_sha256"]
    assert all("test_predictions" in row for row in payload["methods"])

    second = RUNNER.invoke(app, _run_task_args(paths))
    assert second.exit_code == 0, second.output
    assert artifact.read_bytes() == first_bytes


def test_task_refuses_mismatched_existing_artifact(tmp_path: Path) -> None:
    paths = _synthetic_workspace(tmp_path)
    first = RUNNER.invoke(app, _run_task_args(paths))
    assert first.exit_code == 0, first.output
    artifact = paths["results"] / "tasks" / "BBB" / "seed_0.json"
    artifact.write_text('{"schema_version": 999}\n', encoding="utf-8")

    second = RUNNER.invoke(app, _run_task_args(paths))

    assert second.exit_code != 0
    assert "mismatched existing artifact" in second.output


@pytest.mark.parametrize(
    ("assay_id", "mode"),
    (("AAA", "development"), ("BBB", "confirmatory")),
)
def test_explicit_task_refuses_assay_outside_requested_mode(
    tmp_path: Path,
    assay_id: str,
    mode: str,
) -> None:
    paths = _synthetic_workspace(tmp_path)

    result = RUNNER.invoke(
        app,
        [
            "--config",
            str(paths["config"]),
            "run-task",
            "--manifest",
            str(paths["manifest"]),
            "--processed-root",
            str(paths["processed"]),
            "--embedding-root",
            str(paths["embeddings"]),
            "--results-root",
            str(paths["results"]),
            "--assay-id",
            assay_id,
            "--seed",
            "0",
            "--mode",
            mode,
            "--non-official-bypass",
        ],
    )

    assert result.exit_code != 0
    assert "outside the requested mode" in result.output
    assert not (paths["results"] / "tasks" / assay_id / "seed_0.json").exists()


def test_hidden_label_loader_requires_exact_ordered_hashes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _synthetic_workspace(tmp_path)
    frame = pd.read_parquet(paths["processed"] / "BBB.parquet")
    requested = tuple(str(value) for value in frame["sequence_hash"].iloc[[4, 1, 7]])
    read_parquet = Mock(wraps=pd.read_parquet)
    monkeypatch.setattr("self_improve_protein.cli.pd.read_parquet", read_parquet)

    labels = load_evaluation_labels(
        paths["processed"] / "BBB.parquet",
        assay_id="BBB",
        seed=0,
        source_digest="0" * 64,
        labeled_hashes=(str(frame["sequence_hash"].iloc[0]),),
        unlabeled_hashes=requested[:2],
        test_hashes=requested[2:],
    )

    expected = frame.set_index("sequence_hash").loc[list(requested), "DMS_score"]
    np.testing.assert_array_equal(
        np.concatenate([labels.y_u, labels.y_test]), expected.to_numpy()
    )
    assert read_parquet.call_args_list[0].kwargs["filters"] == [
        ("sequence_hash", "in", list(requested))
    ]
    with pytest.raises(ValueError, match=r"missing|duplicate|ordered"):
        load_evaluation_labels(
            paths["processed"] / "BBB.parquet",
            assay_id="BBB",
            seed=0,
            source_digest="0" * 64,
            labeled_hashes=(str(frame["sequence_hash"].iloc[0]),),
            unlabeled_hashes=("f" * 64,),
            test_hashes=requested[2:],
        )


def test_development_aggregate_is_explicit_and_cannot_emit_v0_verdict(
    tmp_path: Path,
) -> None:
    paths = _synthetic_workspace(tmp_path)
    task = RUNNER.invoke(app, _run_task_args(paths))
    assert task.exit_code == 0, task.output
    aggregate_path = tmp_path / "aggregate.json"

    result = RUNNER.invoke(
        app,
        [
            "--config",
            str(paths["config"]),
            "aggregate",
            "--manifest",
            str(paths["manifest"]),
            "--results-root",
            str(paths["results"]),
            "--output",
            str(aggregate_path),
            "--mode",
            "development",
            "--seed",
            "0",
            "--bootstrap-resamples",
            "20",
        ],
    )

    assert result.exit_code == 0, result.output
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    assert aggregate["mode"] == "development"
    assert aggregate["v0_verdict"] is None
    assert len(aggregate["long_results"]) == 5

    confirmatory = RUNNER.invoke(
        app,
        [
            "--config",
            str(paths["config"]),
            "aggregate",
            "--manifest",
            str(paths["manifest"]),
            "--results-root",
            str(paths["results"]),
            "--output",
            str(tmp_path / "bad.json"),
            "--mode",
            "confirmatory",
            "--seed",
            "0",
        ],
    )
    assert confirmatory.exit_code != 0
    assert not (tmp_path / "bad.json").exists()


def test_aggregate_rejects_task_source_digest_outside_protocol_root(
    tmp_path: Path,
) -> None:
    paths = _synthetic_workspace(tmp_path)
    task = RUNNER.invoke(app, _run_task_args(paths))
    assert task.exit_code == 0, task.output
    task_path = paths["results"] / "tasks" / "BBB" / "seed_0.json"
    payload = json.loads(task_path.read_text(encoding="utf-8"))
    payload["digests"]["source"] = "f" * 64
    task_path.write_text(json.dumps(payload), encoding="utf-8")

    aggregate = RUNNER.invoke(
        app,
        [
            "--config",
            str(paths["config"]),
            "aggregate",
            "--manifest",
            str(paths["manifest"]),
            "--results-root",
            str(paths["results"]),
            "--output",
            str(tmp_path / "aggregate.json"),
            "--mode",
            "development",
            "--seed",
            "0",
            "--bootstrap-resamples",
            "20",
        ],
    )

    assert aggregate.exit_code != 0
    assert "source digest mismatch" in aggregate.output
    assert not (tmp_path / "aggregate.json").exists()


def test_verify_fails_closed_after_cache_corruption(tmp_path: Path) -> None:
    paths = _synthetic_workspace(tmp_path)
    valid = RUNNER.invoke(
        app,
        [
            "--config",
            str(paths["config"]),
            "verify",
            "--manifest",
            str(paths["manifest"]),
            "--processed-root",
            str(paths["processed"]),
            "--embedding-root",
            str(paths["embeddings"]),
        ],
    )
    assert valid.exit_code == 0, valid.output
    with (paths["embeddings"] / "BBB.npy").open("ab") as handle:
        handle.write(b"corrupt")

    invalid = RUNNER.invoke(
        app,
        [
            "--config",
            str(paths["config"]),
            "verify",
            "--manifest",
            str(paths["manifest"]),
            "--processed-root",
            str(paths["processed"]),
            "--embedding-root",
            str(paths["embeddings"]),
        ],
    )
    assert invalid.exit_code != 0
    assert "checksum" in invalid.output or "corrupt" in invalid.output
