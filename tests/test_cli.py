import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pytest
import yaml
from typer.testing import CliRunner

import self_improve_protein.cli as cli_module
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


def _aggregate_args(
    paths: dict[str, Path],
    output: Path,
    *,
    seeds: tuple[int, ...] = (0,),
) -> list[str]:
    args = [
        "--config",
        str(paths["config"]),
        "aggregate",
        "--manifest",
        str(paths["manifest"]),
        "--processed-root",
        str(paths["processed"]),
        "--embedding-root",
        str(paths["embeddings"]),
        "--results-root",
        str(paths["results"]),
        "--output",
        str(output),
        "--mode",
        "development",
        "--bootstrap-resamples",
        "20",
        "--non-official-bypass",
    ]
    for seed in seeds:
        args.extend(("--seed", str(seed)))
    return args


def test_cli_help_exposes_five_commands_and_global_options() -> None:
    result = RUNNER.invoke(app, ["--help"])

    assert result.exit_code == 0, result.output
    for command in ("prepare-data", "embed-assay", "run-task", "aggregate", "verify"):
        assert command in result.output
    assert "--config" in result.output
    assert "--dry-run" in result.output


def test_console_usage_error_is_one_terminal_json_record_without_false_start() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "self_improve_protein.cli", "prepare-data"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    records = [json.loads(line) for line in completed.stdout.splitlines()]
    assert len(records) == 1
    assert records[0]["event"] == "terminal"
    assert records[0]["status"] == "error"


def test_show_config_exits_zero_without_dummy_subcommand() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "self_improve_protein.cli", "--show-config"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["event"] == "config"
    assert len(payload["protocol_digest"]) == 64


def test_console_usage_error_extracts_command_after_global_option_value() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "self_improve_protein.cli",
            "--config",
            "configs/v0.yaml",
            "prepare-data",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    payload = json.loads(completed.stdout)
    assert payload["command"] == "prepare-data"
    assert payload["event"] == "terminal"
    assert payload["status"] == "error"


def test_console_action_error_propagates_nonzero_exit(tmp_path: Path) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "self_improve_protein.cli",
            "verify",
            "--manifest",
            str(tmp_path / "missing.json"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    records = [json.loads(line) for line in completed.stdout.splitlines()]
    assert [record["event"] for record in records] == ["start", "terminal"]
    assert records[-1]["status"] == "error"


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
    assert all(record["numerical_runtime"]["libraries"] for record in records)
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

    result = RUNNER.invoke(app, _aggregate_args(paths, aggregate_path))

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
            "--processed-root",
            str(paths["processed"]),
            "--embedding-root",
            str(paths["embeddings"]),
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
        _aggregate_args(paths, tmp_path / "aggregate.json"),
    )

    assert aggregate.exit_code != 0
    assert "content mismatch" in aggregate.output
    assert not (tmp_path / "aggregate.json").exists()


@pytest.mark.parametrize(
    "tamper",
    ("metric", "selection", "prediction", "git", "fingerprint"),
)
def test_aggregate_reconstructs_tasks_instead_of_trusting_stored_content(
    tmp_path: Path,
    tamper: str,
) -> None:
    paths = _synthetic_workspace(tmp_path)
    task = RUNNER.invoke(app, _run_task_args(paths))
    assert task.exit_code == 0, task.output
    task_path = paths["results"] / "tasks" / "BBB" / "seed_0.json"
    payload = json.loads(task_path.read_text(encoding="utf-8"))
    if tamper == "metric":
        payload["methods"][0]["spearman"] += 0.25
    elif tamper == "selection":
        payload["methods"][1]["selected_indices"] = []
        payload["methods"][1]["selected_hashes"] = []
    elif tamper == "prediction":
        payload["methods"][0]["test_predictions"] = [0.0] * 10
    elif tamper == "git":
        payload["provenance"]["git_commit"] = "f" * 40
    elif tamper == "fingerprint":
        payload["execution"]["numerical_runtime"]["libraries"][0]["architecture"] = (
            "Fakewell"
        )
    else:
        raise AssertionError(tamper)
    task_path.write_text(json.dumps(payload), encoding="utf-8")
    output = tmp_path / "aggregate.json"

    aggregate = RUNNER.invoke(app, _aggregate_args(paths, output))

    assert aggregate.exit_code != 0
    assert not output.exists()


@pytest.mark.parametrize(
    "tamper",
    ("long_results", "method_table", "effect_table", "verdict"),
)
def test_verify_recomputes_aggregate_instead_of_trusting_tables(
    tmp_path: Path,
    tamper: str,
) -> None:
    paths = _synthetic_workspace(tmp_path)
    task = RUNNER.invoke(app, _run_task_args(paths))
    assert task.exit_code == 0, task.output
    aggregate_path = tmp_path / "aggregate.json"
    aggregate = RUNNER.invoke(app, _aggregate_args(paths, aggregate_path))
    assert aggregate.exit_code == 0, aggregate.output
    payload = json.loads(aggregate_path.read_text(encoding="utf-8"))
    if tamper == "long_results":
        payload["long_results"][0]["mse"] += 0.5
    elif tamper == "method_table":
        payload["method_table"][0]["mean_spearman"] += 0.5
    elif tamper == "effect_table":
        payload["effect_table"][0]["mean_spearman_gain"] += 0.5
    elif tamper == "verdict":
        payload["v0_verdict"] = {"fabricated": True}
    else:
        raise AssertionError(tamper)
    aggregate_path.write_text(json.dumps(payload), encoding="utf-8")

    verified = RUNNER.invoke(
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
            "--results-root",
            str(paths["results"]),
            "--aggregate-artifact",
            str(aggregate_path),
            "--non-official-bypass",
        ],
    )

    assert verified.exit_code != 0


@pytest.mark.parametrize("tamper", ("git", "fingerprint"))
def test_aggregate_rejects_mixed_task_implementations_and_runtimes(
    tmp_path: Path,
    tamper: str,
) -> None:
    paths = _synthetic_workspace(tmp_path)
    first = RUNNER.invoke(app, _run_task_args(paths))
    assert first.exit_code == 0, first.output
    second_args = _run_task_args(paths)
    second_args[second_args.index("0")] = "1"
    second = RUNNER.invoke(app, second_args)
    assert second.exit_code == 0, second.output
    second_path = paths["results"] / "tasks" / "BBB" / "seed_1.json"
    payload = json.loads(second_path.read_text(encoding="utf-8"))
    if tamper == "git":
        payload["provenance"]["git_commit"] = "f" * 40
    else:
        payload["execution"]["numerical_runtime"]["libraries"][0]["architecture"] = (
            "Fakewell"
        )
    second_path.write_text(json.dumps(payload), encoding="utf-8")

    aggregate = RUNNER.invoke(
        app,
        _aggregate_args(paths, tmp_path / "mixed.json", seeds=(0, 1)),
    )

    assert aggregate.exit_code != 0
    assert not (tmp_path / "mixed.json").exists()


def test_aggregate_rejects_fabricated_schema_only_task(tmp_path: Path) -> None:
    paths = _synthetic_workspace(tmp_path)
    task = RUNNER.invoke(app, _run_task_args(paths))
    assert task.exit_code == 0, task.output
    task_path = paths["results"] / "tasks" / "BBB" / "seed_0.json"
    payload = json.loads(task_path.read_text(encoding="utf-8"))
    payload["methods"] = [
        {"name": name, "spearman": 0.1, "mse": 0.1, "ndcg_10pct": 0.1}
        for name in ("supervised", "random", "top_teacher", "ours", "no_hessian")
    ]
    task_path.write_text(json.dumps(payload), encoding="utf-8")

    aggregate = RUNNER.invoke(
        app,
        _aggregate_args(paths, tmp_path / "fabricated.json"),
    )

    assert aggregate.exit_code != 0
    assert not (tmp_path / "fabricated.json").exists()


def test_confirmatory_refuses_non_official_bypass_and_alternate_protocol(
    tmp_path: Path,
) -> None:
    paths = _synthetic_workspace(tmp_path)
    args = _run_task_args(paths)
    args[args.index("BBB")] = "AAA"
    args[args.index("development")] = "confirmatory"

    bypass = RUNNER.invoke(app, args)

    assert bypass.exit_code != 0
    assert "confirmatory" in bypass.output

    args.remove("--non-official-bypass")
    args.insert(2, "--dry-run")
    alternate = RUNNER.invoke(app, args)
    assert alternate.exit_code != 0
    assert "locked v0 protocol" in alternate.output


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
            "--non-official-bypass",
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
            "--non-official-bypass",
        ],
    )
    assert invalid.exit_code != 0
    assert "checksum" in invalid.output or "corrupt" in invalid.output


def test_task_verify_rebuilds_current_processed_and_embedding_inputs(
    tmp_path: Path,
) -> None:
    paths = _synthetic_workspace(tmp_path)
    task = RUNNER.invoke(app, _run_task_args(paths))
    assert task.exit_code == 0, task.output
    artifact = paths["results"] / "tasks" / "BBB" / "seed_0.json"
    args = [
        "--config",
        str(paths["config"]),
        "verify",
        "--manifest",
        str(paths["manifest"]),
        "--processed-root",
        str(paths["processed"]),
        "--embedding-root",
        str(paths["embeddings"]),
        "--task-artifact",
        str(artifact),
        "--non-official-bypass",
    ]
    valid = RUNNER.invoke(app, args)
    assert valid.exit_code == 0, valid.output
    with (paths["processed"] / "BBB.parquet").open("ab") as handle:
        handle.write(b"tampered")

    invalid = RUNNER.invoke(app, args)

    assert invalid.exit_code != 0


def test_five_wide_cache_is_not_accepted_as_official_without_bypass(
    tmp_path: Path,
) -> None:
    paths = _synthetic_workspace(tmp_path)

    result = RUNNER.invoke(
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

    assert result.exit_code != 0
    assert "locked v0 protocol" in result.output


def test_r5_gate_is_written_only_from_reconstructed_two_seed_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _synthetic_workspace(tmp_path)
    protocol = load_protocol(paths["config"])
    monkeypatch.setattr(
        cli_module,
        "_LOCKED_V0_PROTOCOL_DIGEST",
        cli_module.canonical_protocol_digest(protocol),
    )
    monkeypatch.setattr(cli_module, "_ESM2_35M_EMBEDDING_DIM", 5)
    monkeypatch.setattr(
        cli_module,
        "require_openblas_coretype",
        lambda expected_core: expected_core,
    )
    monkeypatch.setattr(
        cli_module,
        "_git_commit",
        lambda *, require_clean=False: "a" * 40,
    )
    first_args = _run_task_args(paths)
    first_args.remove("--non-official-bypass")
    first = RUNNER.invoke(app, first_args)
    assert first.exit_code == 0, first.output
    second_args = first_args.copy()
    second_args[second_args.index("0")] = "1"
    second = RUNNER.invoke(app, second_args)
    assert second.exit_code == 0, second.output
    aggregate_path = tmp_path / "r5-aggregate.json"
    aggregate_args = _aggregate_args(paths, aggregate_path, seeds=(0, 1))
    aggregate_args.remove("--non-official-bypass")
    aggregate_args[aggregate_args.index("20")] = "10000"
    aggregate = RUNNER.invoke(app, aggregate_args)
    assert aggregate.exit_code == 0, aggregate.output
    gate_path = tmp_path / "r5-gate.json"
    verify_args = [
        "--config",
        str(paths["config"]),
        "verify",
        "--manifest",
        str(paths["manifest"]),
        "--processed-root",
        str(paths["processed"]),
        "--embedding-root",
        str(paths["embeddings"]),
        "--results-root",
        str(paths["results"]),
        "--aggregate-artifact",
        str(aggregate_path),
        "--write-r5-gate",
        str(gate_path),
    ]

    written = RUNNER.invoke(app, verify_args)

    assert written.exit_code == 0, written.output
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    assert gate["status"] == "passed"
    assert gate["task_count"] == 2
    assert gate["method_row_count"] == 10
    assert [entry["seed"] for entry in gate["task_manifest"]] == [0, 1]
    confirmatory_args = first_args.copy()
    confirmatory_args[confirmatory_args.index("BBB")] = "AAA"
    confirmatory_args[confirmatory_args.index("development")] = "confirmatory"
    confirmatory_args.extend(("--r5-gate", str(gate_path)))
    confirmatory = RUNNER.invoke(app, confirmatory_args)
    assert confirmatory.exit_code == 0, confirmatory.output
    confirmatory_task = json.loads(
        (paths["results"] / "tasks" / "AAA" / "seed_0.json").read_text(encoding="utf-8")
    )
    assert confirmatory_task["provenance"]["r5_gate_sha256"] == (
        cli_module.sha256_file(gate_path)
    )
    task_path = paths["results"] / "tasks" / "BBB" / "seed_1.json"
    task = json.loads(task_path.read_text(encoding="utf-8"))
    task["methods"][0]["mse"] += 1.0
    task_path.write_text(json.dumps(task), encoding="utf-8")

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
            "--results-root",
            str(paths["results"]),
            "--r5-gate",
            str(gate_path),
        ],
    )

    assert invalid.exit_code != 0
