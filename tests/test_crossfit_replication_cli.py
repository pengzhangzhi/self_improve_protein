from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pytest
from test_crossfit_cli import (
    AMINO_ACIDS,
    _synthetic_crossfit_workspace,
)
from typer.testing import CliRunner

import self_improve_protein.crossfit_replication_cli as replication_cli
from self_improve_protein.analysis import PairwiseSummary, exact_sign_flip_pvalue
from self_improve_protein.config import load_protocol
from self_improve_protein.crossfit_data import load_crossfit_pool_manifest
from self_improve_protein.crossfit_replication import (
    ScreenPrimarySummary,
    ScreenPromotionGate,
    load_screen_promotion_gate,
    write_screen_promotion_gate,
)
from self_improve_protein.data import row_hash
from self_improve_protein.embeddings import (
    EmbeddingCacheSources,
    write_embedding_cache,
)
from self_improve_protein.experiment import canonical_protocol_digest
from self_improve_protein.provenance import sha256_file

RUNNER = CliRunner()


def _sequence(index: int) -> str:
    return (
        "A"
        + AMINO_ACIDS[(index // (20 * 20)) % 20]
        + AMINO_ACIDS[(index // 20) % 20]
        + AMINO_ACIDS[index % 20]
    )


def _replication_workspace(tmp_path: Path) -> dict[str, Path]:
    paths = _synthetic_crossfit_workspace(tmp_path)
    protocol = load_protocol(paths["config"])
    pool = load_crossfit_pool_manifest(paths["pool"])
    sources = EmbeddingCacheSources(
        proteingym_upstream_commit=protocol.proteingym_upstream_commit,
        substitutions_sha256=protocol.substitutions_sha256,
        zero_shot_scores_sha256=protocol.zero_shot_scores_sha256,
        metadata_sha256=protocol.metadata_sha256,
    )
    records = {record.dms_id: record for record in pool.selected_assays}
    for assay_offset, dms_id in enumerate(pool.eligible_assay_ids[9:], start=9):
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
                frame["mutant"], frame["mutated_sequence"], strict=True
            )
        ]
        frame = frame.sort_values("sequence_hash", kind="stable").reset_index(
            drop=True
        )
        assert tuple(frame["sequence_hash"]) == records[dms_id].row_hashes
        generator = np.random.Generator(np.random.PCG64(800 + assay_offset))
        embeddings = generator.normal(
            size=(protocol.working_size, 5)
        ).astype(np.float32)
        beta = np.array([0.8, -0.5, 0.3, 0.45, -0.2])
        frame["DMS_score"] = embeddings.astype(np.float64) @ beta + np.linspace(
            -0.2, 0.2, protocol.working_size
        )
        frame[protocol.teacher_column] = (
            embeddings.astype(np.float64) @ (beta + 0.12) + 0.03
        )
        frame.to_parquet(paths["processed"] / f"{dms_id}.parquet", index=False)
        write_embedding_cache(
            paths["embeddings"] / f"{dms_id}.npy",
            paths["embeddings"] / f"{dms_id}.json",
            embeddings,
            dms_id=dms_id,
            row_hashes=records[dms_id].row_hashes,
            model_id=protocol.model,
            model_revision=protocol.model_revision,
            sources=sources,
        )
    paths["results"] = tmp_path / "replication-results"
    paths["gate"] = tmp_path / "promotion-gate.json"
    values = tuple(0.1 for _ in range(8))
    summary = PairwiseSummary(
        first="crossfit",
        second="random",
        metric="spearman",
        mean_gain=float(np.mean(values)),
        standard_error=float(np.std(values, ddof=1) / np.sqrt(8)),
        task_wins=40,
        task_total=40,
        task_win_rate=1.0,
        assay_wins=8,
        assay_total=8,
        assay_win_rate=1.0,
        exact_sign_flip_pvalue=exact_sign_flip_pvalue(values),
        assay_deltas=values,
    )
    gate = ScreenPromotionGate(
        schema_id="self-improve-protein.crossfit-screen-promotion.v1",
        schema_version=1,
        card_id="crossfit_outer_gradient_v1",
        card_sha256=(
            "383afd7a5bae9c2ebd6768a112a82980236540fc0f66e3a294ef298961b8596f"
        ),
        base_manifest_sha256=sha256_file(paths["base"]),
        pool_manifest_sha256=sha256_file(paths["pool"]),
        screen_aggregate_sha256="a" * 64,
        protocol_digest=canonical_protocol_digest(protocol),
        git_commit=replication_cli._git_commit(),
        screen_primary_summary=ScreenPrimarySummary.from_pairwise_summary(summary),
        untouched_assay_ids=pool.untouched_ids,
        seeds=protocol.seeds,
        promote_to_untouched_replication=True,
    )
    write_screen_promotion_gate(paths["gate"], gate)
    return paths


def _task_args(paths: dict[str, Path], index: int) -> list[str]:
    return [
        "--config",
        str(paths["config"]),
        "run-task",
        "--base-manifest",
        str(paths["base"]),
        "--pool-manifest",
        str(paths["pool"]),
        "--promotion-gate",
        str(paths["gate"]),
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


def _create_gate_args(
    paths: dict[str, Path],
    screen_aggregate: Path,
    output: Path,
) -> list[str]:
    return [
        "--config",
        str(paths["config"]),
        "create-gate",
        "--base-manifest",
        str(paths["base"]),
        "--pool-manifest",
        str(paths["pool"]),
        "--processed-root",
        str(paths["processed"]),
        "--embedding-root",
        str(paths["embeddings"]),
        "--screen-results-root",
        str(paths["results"]),
        "--screen-aggregate",
        str(screen_aggregate),
        "--output",
        str(output),
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
        "--promotion-gate",
        str(paths["gate"]),
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


def _verify_args(paths: dict[str, Path]) -> list[str]:
    return [
        "--config",
        str(paths["config"]),
        "verify",
        "--base-manifest",
        str(paths["base"]),
        "--pool-manifest",
        str(paths["pool"]),
        "--promotion-gate",
        str(paths["gate"]),
        "--processed-root",
        str(paths["processed"]),
        "--embedding-root",
        str(paths["embeddings"]),
        "--non-official-bypass",
    ]

def test_replication_module_exposes_gate_task_aggregate_and_verify() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "self_improve_protein.crossfit_replication_cli",
            "--help",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    for command in ("create-gate", "run-task", "aggregate", "verify"):
        assert command in completed.stdout


def test_replication_slurm_is_cpu_only_exact_130_and_dependent() -> None:
    task = Path("slurm/crossfit_replication_task_array.sbatch")
    aggregate = Path("slurm/crossfit_replication_aggregate.sbatch")
    submit = Path("slurm/submit_crossfit_replication.sh")

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
        assert "export OPENBLAS_CORETYPE=Haswell" in text
        assert "--gpus" not in text
        assert "python -m self_improve_protein.crossfit_replication_cli" in text
        assert "SI_EXPECTED_CROSSFIT_PROMOTION_GATE_SHA256" in text
        assert "--expected-promotion-gate-sha256" in text
        time_line = next(line for line in text.splitlines() if "--time=" in line)
        assert time_line <= "#SBATCH --time=04:00:00"
    submit_text = submit.read_text(encoding="utf-8")
    assert "--array=0-129" in submit_text
    assert "--dependency=afterok:" in submit_text
    assert "crossfit_v1" in submit_text
    assert "create-gate" in submit_text
    assert "SI_CROSSFIT_SCREEN_RESULTS_ROOT" in submit_text
    assert "SI_CROSSFIT_SCREEN_AGGREGATE" in submit_text
    assert "processed/v0" in submit_text
    assert "embeddings/v0" in submit_text
    assert "sha256sum" in submit_text
    assert "export SI_EXPECTED_CROSSFIT_PROMOTION_GATE_SHA256" in submit_text
    assert submit_text.index("create-gate") < submit_text.index("sha256sum")
    assert submit_text.index("sha256sum") < submit_text.index(" verify \\")
    assert submit_text.index(" verify \\") < submit_text.index('raw="$(sbatch')
    assert "job_ids.json" in submit_text
    assert submit.stat().st_mode & 0o111


def test_replication_task_fits_before_hidden_labels_and_binds_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _replication_workspace(tmp_path)
    events: list[str] = []
    real_fit = replication_cli.fit_crossfit_task
    real_digest = replication_cli.canonical_crossfit_fit_digest
    real_labels = replication_cli.load_evaluation_labels

    def fit(*args: object, **kwargs: object) -> object:
        result = real_fit(*args, **kwargs)  # type: ignore[arg-type]
        events.append("fit")
        return result

    def digest(*args: object, **kwargs: object) -> str:
        result = real_digest(*args, **kwargs)  # type: ignore[arg-type]
        events.append("digest")
        return result

    def labels(*args: object, **kwargs: object) -> object:
        assert events == ["fit", "digest"]
        events.append("labels")
        return real_labels(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(replication_cli, "fit_crossfit_task", fit)
    monkeypatch.setattr(
        replication_cli,
        "canonical_crossfit_fit_digest",
        digest,
    )
    monkeypatch.setattr(replication_cli, "load_evaluation_labels", labels)

    result = RUNNER.invoke(replication_cli.app, _task_args(paths, 0))

    assert result.exit_code == 0, result.output
    assert events[:3] == ["fit", "digest", "labels"]
    pool = load_crossfit_pool_manifest(paths["pool"])
    artifact = (
        paths["results"]
        / "tasks"
        / pool.untouched_ids[0]
        / "seed_0.json"
    )
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    assert payload["kind"] == "crossfit_replication_task_result"
    assert payload["task"]["phase"] == "replication"
    assert payload["provenance"]["promotion_gate_sha256"] == sha256_file(
        paths["gate"]
    )


def test_replication_gate_context_rejects_wrong_current_head(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _replication_workspace(tmp_path)
    monkeypatch.setattr(
        replication_cli,
        "_validate_execution_policy",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(replication_cli, "_git_commit", lambda **kwargs: "f" * 40)

    with pytest.raises(ValueError, match="gate Git commit"):
        replication_cli._load_replication_context(
            config=paths["config"],
            base_manifest_path=paths["base"],
            pool_manifest_path=paths["pool"],
            gate_path=paths["gate"],
            expected_gate_sha256=sha256_file(paths["gate"]),
            non_official_bypass=False,
        )


def test_fabricated_canonical_gate_is_rejected_by_captured_sha(
    tmp_path: Path,
) -> None:
    paths = _replication_workspace(tmp_path)
    captured_sha = sha256_file(paths["gate"])
    original = load_screen_promotion_gate(paths["gate"])
    fabricated = original.model_copy(
        update={"screen_aggregate_sha256": "f" * 64}
    )
    write_screen_promotion_gate(paths["gate"], fabricated)
    assert load_screen_promotion_gate(paths["gate"]) == fabricated

    with pytest.raises(ValueError, match="captured expected SHA"):
        replication_cli._load_replication_context(
            config=paths["config"],
            base_manifest_path=paths["base"],
            pool_manifest_path=paths["pool"],
            gate_path=paths["gate"],
            expected_gate_sha256=captured_sha,
            non_official_bypass=True,
        )


def test_gate_replacement_after_preflight_is_rejected_by_task_context(
    tmp_path: Path,
) -> None:
    paths = _replication_workspace(tmp_path)
    captured_sha = sha256_file(paths["gate"])
    preflight = replication_cli._load_replication_context(
        config=paths["config"],
        base_manifest_path=paths["base"],
        pool_manifest_path=paths["pool"],
        gate_path=paths["gate"],
        expected_gate_sha256=captured_sha,
        non_official_bypass=True,
    )
    replacement = preflight.gate.model_copy(
        update={"screen_aggregate_sha256": "e" * 64}
    )
    write_screen_promotion_gate(paths["gate"], replacement)

    with pytest.raises(ValueError, match="captured expected SHA"):
        replication_cli._load_replication_context(
            config=paths["config"],
            base_manifest_path=paths["base"],
            pool_manifest_path=paths["pool"],
            gate_path=paths["gate"],
            expected_gate_sha256=captured_sha,
            non_official_bypass=True,
        )


def test_create_gate_exact_rebuilds_screen_and_is_write_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _replication_workspace(tmp_path)
    expected_gate = load_screen_promotion_gate(paths["gate"])
    screen_aggregate = tmp_path / "screen-aggregate.json"
    screen_payload = {"kind": "synthetic-screen", "value": 7}
    screen_aggregate.write_text(
        json.dumps(screen_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "created-gate.json"
    calls: list[dict[str, object]] = []

    monkeypatch.setattr(
        replication_cli,
        "_validate_execution_policy",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        replication_cli,
        "_git_commit",
        lambda **kwargs: expected_gate.git_commit,
    )
    monkeypatch.setattr(
        replication_cli,
        "_validate_crossfit_aggregate_payload",
        lambda payload: None,
    )

    def rebuild(**kwargs: object) -> dict[str, object]:
        calls.append(kwargs)
        return dict(screen_payload)

    def build_gate(aggregate: object, **kwargs: object) -> ScreenPromotionGate:
        assert aggregate == screen_payload
        assert kwargs["screen_aggregate_sha256"] == sha256_file(screen_aggregate)
        return expected_gate

    monkeypatch.setattr(replication_cli, "_build_crossfit_aggregate_payload", rebuild)
    monkeypatch.setattr(replication_cli, "build_screen_promotion_gate", build_gate)

    first = RUNNER.invoke(
        replication_cli.app,
        _create_gate_args(paths, screen_aggregate, output),
    )
    second = RUNNER.invoke(
        replication_cli.app,
        _create_gate_args(paths, screen_aggregate, output),
    )

    assert first.exit_code == 0, first.output
    assert second.exit_code == 0, second.output
    assert len(calls) == 2
    assert load_screen_promotion_gate(output) == expected_gate
    assert json.loads(second.output.splitlines()[-1])["created"] is False

    screen_aggregate.write_text('{"kind":"tampered"}\n', encoding="utf-8")
    tampered = RUNNER.invoke(
        replication_cli.app,
        _create_gate_args(paths, screen_aggregate, output),
    )
    assert tampered.exit_code != 0
    assert "content mismatch" in tampered.output


def test_replication_exact_130_aggregate_verdict_and_tamper_rejection(
    tmp_path: Path,
) -> None:
    paths = _replication_workspace(tmp_path)
    pool = load_crossfit_pool_manifest(paths["pool"])

    first = RUNNER.invoke(replication_cli.app, _task_args(paths, 0))
    assert first.exit_code == 0, first.output
    task_path = paths["results"] / "tasks" / pool.untouched_ids[0] / "seed_0.json"
    task_bytes = task_path.read_bytes()
    second = RUNNER.invoke(replication_cli.app, _task_args(paths, 0))
    assert second.exit_code == 0, second.output
    assert json.loads(second.output.splitlines()[-1])["created"] is False

    task_payload = json.loads(task_bytes)
    task_payload["variant"]["spearman"] = 0.1234
    task_path.write_text(json.dumps(task_payload) + "\n", encoding="utf-8")
    rejected = RUNNER.invoke(replication_cli.app, _task_args(paths, 0))
    assert rejected.exit_code != 0
    assert "mismatched existing artifact" in rejected.output
    task_path.write_bytes(task_bytes)

    for index in range(1, 130):
        result = RUNNER.invoke(replication_cli.app, _task_args(paths, index))
        assert result.exit_code == 0, f"task {index}: {result.output}"

    output = tmp_path / "replication-aggregate.json"
    aggregate = RUNNER.invoke(replication_cli.app, _aggregate_args(paths, output))
    assert aggregate.exit_code == 0, aggregate.output
    aggregate_bytes = output.read_bytes()
    payload = json.loads(aggregate_bytes)
    assert payload["kind"] == "crossfit_replication_aggregate_result"
    assert payload["provenance"]["task_count"] == 130
    assert len(payload["grid"]["tasks"]) == 130
    assert len(payload["long_results"]) == 130 * 6
    effect = payload["effects"]["crossfit_minus_random"]
    assert effect["task_total"] == 130
    assert effect["assay_total"] == 26
    expected = (
        effect["mean_gain"] > 0.0
        and effect["task_wins"] >= 78
        and effect["assay_wins"] >= 16
    )
    assert payload["verdict"]["replication_success"] is expected

    repeated = RUNNER.invoke(replication_cli.app, _aggregate_args(paths, output))
    assert repeated.exit_code == 0, repeated.output
    assert json.loads(repeated.output.splitlines()[-1])["created"] is False
    assert output.read_bytes() == aggregate_bytes

    changed = json.loads(aggregate_bytes)
    changed["verdict"]["rule"]["task_win_threshold"] = 77
    output.write_text(json.dumps(changed) + "\n", encoding="utf-8")
    aggregate_rejected = RUNNER.invoke(
        replication_cli.app,
        _aggregate_args(paths, output),
    )
    assert aggregate_rejected.exit_code != 0
    assert "mismatched existing artifact" in aggregate_rejected.output


def test_replication_verify_checks_all_inputs_task_and_canonical_gate(
    tmp_path: Path,
) -> None:
    paths = _replication_workspace(tmp_path)
    pool = load_crossfit_pool_manifest(paths["pool"])
    task = RUNNER.invoke(replication_cli.app, _task_args(paths, 0))
    assert task.exit_code == 0, task.output
    task_path = paths["results"] / "tasks" / pool.untouched_ids[0] / "seed_0.json"

    verified = RUNNER.invoke(
        replication_cli.app,
        [*_verify_args(paths), "--task-artifact", str(task_path)],
    )

    assert verified.exit_code == 0, verified.output
    terminal = json.loads(verified.output.splitlines()[-1])
    assert terminal["verified"] == ["gate", "processed", "embeddings", "task"]

    paths["gate"].write_bytes(paths["gate"].read_bytes() + b"\n")
    rejected = RUNNER.invoke(replication_cli.app, _verify_args(paths))
    assert rejected.exit_code != 0
    assert "canonical" in rejected.output
