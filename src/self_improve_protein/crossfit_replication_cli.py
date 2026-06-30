"""Restart-safe CLI for the gated untouched crossfit replication."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, cast

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import typer

from self_improve_protein.analysis import (
    method_summary_table,
    pairwise_summary,
    validate_result_table,
)
from self_improve_protein.cli import (
    _ESM2_35M_EMBEDDING_DIM,
    _cache_width,
    _canonical_json_bytes,
    _embedding_sources,
    _git_commit,
    _jsonable,
    _load_identity_frame,
    _load_unique_json,
    _ordered_outcomes,
    _require_exact_payload,
    _run_command,
    _write_json_once,
    load_evaluation_labels,
)
from self_improve_protein.config import DEFAULT_CONFIG_PATH, Protocol, load_protocol
from self_improve_protein.crossfit import (
    CARD_ID,
    CARD_SHA,
    canonical_crossfit_fit_digest,
    evaluate_crossfit_task,
    fit_crossfit_task,
)
from self_improve_protein.crossfit_cli import (
    _build_crossfit_aggregate_payload,
    _load_manifests,
    _method_row,
    _record,
    _validate_crossfit_aggregate_payload,
    _validate_execution_policy,
)
from self_improve_protein.crossfit_data import CrossfitPoolManifest
from self_improve_protein.crossfit_replication import (
    CrossfitReplicationVerdict,
    ScreenPromotionGate,
    build_screen_promotion_gate,
    crossfit_replication_verdict,
    load_screen_promotion_gate,
    replication_grid,
    resolve_replication_task_index,
)
from self_improve_protein.data import DataManifest, make_split
from self_improve_protein.embeddings import load_embedding_cache
from self_improve_protein.experiment import (
    METHOD_NAMES,
    NUMERICAL_POLICY,
    FitInputs,
    canonical_evaluation_digest,
    canonical_protocol_digest,
    canonical_source_digest,
)
from self_improve_protein.provenance import sha256_file

app = typer.Typer(
    name="self-improve-protein-crossfit-replication",
    help="Create the promotion gate and run the untouched replication.",
    invoke_without_command=True,
    no_args_is_help=False,
    pretty_exceptions_enable=False,
)

_METHOD = "crossfit"


@dataclass(frozen=True, slots=True)
class _RuntimeOptions:
    config: Path
    dry_run: bool


@dataclass(frozen=True, slots=True)
class ReplicationContext:
    protocol: Protocol
    base_manifest: DataManifest
    pool_manifest: CrossfitPoolManifest
    gate: ScreenPromotionGate
    gate_sha256: str
    git_commit: str


@app.callback()
def main(
    context: typer.Context,
    config: Annotated[Path, typer.Option("--config")] = DEFAULT_CONFIG_PATH,
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
) -> None:
    """Set the frozen protocol and dry-run policy."""
    context.obj = _RuntimeOptions(config=config, dry_run=dry_run)
    if context.invoked_subcommand is None:
        typer.echo(context.get_help())
        raise typer.Exit()


def _options(context: typer.Context) -> _RuntimeOptions:
    if not isinstance(context.obj, _RuntimeOptions):
        raise RuntimeError("CLI runtime options were not initialized")
    return context.obj


def _load_replication_context(
    *,
    config: Path,
    base_manifest_path: Path,
    pool_manifest_path: Path,
    gate_path: Path,
    expected_gate_sha256: str | None,
    non_official_bypass: bool,
) -> ReplicationContext:
    protocol = load_protocol(config)
    _validate_execution_policy(
        protocol,
        non_official_bypass=non_official_bypass,
    )
    base, pool = _load_manifests(
        protocol=protocol,
        base_manifest_path=base_manifest_path,
        pool_manifest_path=pool_manifest_path,
    )
    gate = load_screen_promotion_gate(gate_path)
    gate_sha = sha256_file(gate_path)
    if expected_gate_sha256 is None:
        if not non_official_bypass:
            raise ValueError(
                "official replication requires the captured expected gate SHA"
            )
    elif not _is_sha256(expected_gate_sha256):
        raise ValueError("captured expected gate SHA must be lowercase SHA-256")
    elif gate_sha != expected_gate_sha256:
        raise ValueError("promotion gate does not match captured expected SHA")
    head = _git_commit(require_clean=not non_official_bypass)
    expected = (
        ("base manifest", gate.base_manifest_sha256, sha256_file(base_manifest_path)),
        ("pool manifest", gate.pool_manifest_sha256, sha256_file(pool_manifest_path)),
        ("protocol", gate.protocol_digest, canonical_protocol_digest(protocol)),
        ("untouched assays", gate.untouched_assay_ids, pool.untouched_ids),
        ("seeds", gate.seeds, protocol.seeds),
    )
    mismatches = [name for name, actual, wanted in expected if actual != wanted]
    if mismatches:
        raise ValueError("promotion gate provenance mismatch: " + ", ".join(mismatches))
    replication_grid(pool, protocol)
    if not non_official_bypass and head != gate.git_commit:
        raise ValueError("current HEAD does not equal the promotion gate Git commit")
    return ReplicationContext(
        protocol=protocol,
        base_manifest=base,
        pool_manifest=pool,
        gate=gate,
        gate_sha256=gate_sha,
        git_commit=head,
    )


def _require_replication_roots(
    processed_root: Path,
    embedding_root: Path,
    *,
    non_official_bypass: bool,
) -> None:
    if non_official_bypass:
        return
    if processed_root.parts[-2:] != ("processed", "crossfit_v1"):
        raise ValueError("official replication requires processed/crossfit_v1")
    if embedding_root.parts[-2:] != ("embeddings", "crossfit_v1"):
        raise ValueError("official replication requires embeddings/crossfit_v1")


def _validate_replication_task_payload(payload: dict[str, object]) -> None:
    required = {
        "card",
        "diagnostics",
        "digests",
        "execution",
        "kind",
        "provenance",
        "reference_methods",
        "schema_version",
        "task",
        "variant",
    }
    if (
        set(payload) != required
        or payload.get("schema_version") != 1
        or payload.get("kind") != "crossfit_replication_task_result"
    ):
        raise ValueError("crossfit replication task schema is invalid")
    card = payload.get("card")
    task = payload.get("task")
    provenance = payload.get("provenance")
    if (
        not isinstance(card, dict)
        or card != {"id": CARD_ID, "sha256": CARD_SHA}
        or not isinstance(task, dict)
        or task.get("phase") != "replication"
        or not isinstance(task.get("assay_id"), str)
        or type(task.get("seed")) is not int
        or not isinstance(provenance, dict)
        or not _is_sha256(provenance.get("promotion_gate_sha256"))
    ):
        raise ValueError("crossfit replication task identity is invalid")
    digests = payload.get("digests")
    if not isinstance(digests, dict) or any(
        not _is_sha256(digests.get(name))
        for name in ("base_fit", "crossfit_fit", "evaluation", "protocol", "source")
    ):
        raise ValueError("crossfit replication task digests are invalid")
    references = payload.get("reference_methods")
    if (
        not isinstance(references, list)
        or tuple(
            row.get("name") if isinstance(row, dict) else None for row in references
        )
        != METHOD_NAMES
    ):
        raise ValueError("replication task reference methods are invalid")
    variant = payload.get("variant")
    rows = [*cast(list[dict[str, object]], references), variant]
    if not isinstance(variant, dict) or variant.get("name") != _METHOD:
        raise ValueError("replication task variant is invalid")
    for row in cast(list[dict[str, object]], rows):
        for metric in ("spearman", "mse", "ndcg_10pct"):
            value = row.get(metric)
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, float, np.integer, np.floating))
                or not np.isfinite(float(value))
            ):
                raise ValueError("replication task metrics are invalid")


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _build_replication_task_payload(
    *,
    context: ReplicationContext,
    base_manifest_path: Path,
    pool_manifest_path: Path,
    gate_path: Path,
    processed_root: Path,
    embedding_root: Path,
    assay_id: str,
    seed: int,
    non_official_bypass: bool,
) -> dict[str, object]:
    protocol = context.protocol
    pool = context.pool_manifest
    if (assay_id, seed) not in replication_grid(pool, protocol):
        raise ValueError("task is outside the exact replication grid")
    _require_replication_roots(
        processed_root,
        embedding_root,
        non_official_bypass=non_official_bypass,
    )
    record = _record(pool, assay_id)
    parquet_path = processed_root / f"{assay_id}.parquet"
    identity = _load_identity_frame(
        parquet_path,
        protocol=protocol,
        record=record,
    )
    split = make_split(
        identity,
        assay_id,
        seed,
        protocol.n_labeled,
        protocol.n_unlabeled,
        protocol.n_test,
    )
    metadata_path = embedding_root / f"{assay_id}.json"
    npy_path = embedding_root / f"{assay_id}.npy"
    embedding_dim = (
        _cache_width(metadata_path)
        if non_official_bypass
        else _ESM2_35M_EMBEDDING_DIM
    )
    embeddings = load_embedding_cache(
        npy_path,
        metadata_path,
        dms_id=assay_id,
        row_hashes=record.row_hashes,
        model_id=protocol.model,
        model_revision=protocol.model_revision,
        sources=_embedding_sources(protocol),
        expected_embedding_dim=embedding_dim,
    )
    y_l = _ordered_outcomes(
        parquet_path,
        requested_hashes=split.labeled_hashes,
        expected_all_hashes=record.row_hashes,
    )
    teacher = identity[protocol.teacher_column].to_numpy(dtype=np.float64)
    source_digest = canonical_source_digest(protocol)
    inputs = FitInputs(
        assay_id=assay_id,
        seed=seed,
        source_digest=source_digest,
        labeled_hashes=split.labeled_hashes,
        unlabeled_hashes=split.unlabeled_hashes,
        test_hashes=split.test_hashes,
        x_l=embeddings[np.asarray(split.labeled, dtype=np.int64)],
        y_l=y_l,
        z_l=teacher[np.asarray(split.labeled, dtype=np.int64)],
        x_u=embeddings[np.asarray(split.unlabeled, dtype=np.int64)],
        z_u=teacher[np.asarray(split.unlabeled, dtype=np.int64)],
        x_test=embeddings[np.asarray(split.test, dtype=np.int64)],
        z_test=teacher[np.asarray(split.test, dtype=np.int64)],
    )
    fit = fit_crossfit_task(inputs, protocol)
    base_digest = fit.base_fit_digest
    crossfit_digest = canonical_crossfit_fit_digest(fit)
    labels = load_evaluation_labels(
        parquet_path,
        assay_id=assay_id,
        seed=seed,
        source_digest=source_digest,
        labeled_hashes=split.labeled_hashes,
        unlabeled_hashes=split.unlabeled_hashes,
        test_hashes=split.test_hashes,
        expected_all_hashes=record.row_hashes,
    )
    evaluation_digest = canonical_evaluation_digest(labels)
    evaluation = evaluate_crossfit_task(
        fit,
        labels,
        protocol=protocol,
        expected_crossfit_fit_digest=crossfit_digest,
        expected_evaluation_digest=evaluation_digest,
    )
    measured = {item.name: item for item in evaluation.reference.methods}
    references = [
        _method_row(
            artifact=method,
            evaluation=measured[method.name],
            name=method.name,
        )
        for method in fit.base_fit.methods
    ]
    variant = _method_row(
        artifact=fit.method,
        evaluation=evaluation.crossfit,
        name=_METHOD,
    )
    payload: dict[str, object] = {
        "card": {"id": CARD_ID, "sha256": CARD_SHA},
        "diagnostics": {
            "evaluation": {
                "crossfit_test_risk_oracle": evaluation.test_risk_oracle,
                "full_test_risk_oracle": evaluation.reference.full_test_risk_oracle,
                "no_hessian_test_risk_oracle": (
                    evaluation.reference.no_hessian_test_risk_oracle
                ),
                "pool_pseudo_label_mae": evaluation.reference.pool_pseudo_label_mae,
                "random_error_reference": evaluation.reference.random_error_reference,
                "teacher_test_spearman": evaluation.reference.teacher_test_spearman,
            },
            "fit": {
                "crossfit": fit.diagnostics,
                "reference": fit.base_fit.diagnostics,
            },
        },
        "digests": {
            "base_fit": base_digest,
            "crossfit_fit": crossfit_digest,
            "evaluation": evaluation_digest,
            "protocol": canonical_protocol_digest(protocol),
            "source": source_digest,
        },
        "execution": {
            "bypass": (
                "non_official_test_or_development" if non_official_bypass else None
            ),
            "numerical_policy": fit.base_fit.numerical_policy,
            "numerical_runtime": fit.base_fit.numerical_runtime,
            "official": not non_official_bypass,
        },
        "kind": "crossfit_replication_task_result",
        "provenance": {
            "base_manifest_sha256": sha256_file(base_manifest_path),
            "embedding_metadata_sha256": sha256_file(metadata_path),
            "embedding_npy_sha256": sha256_file(npy_path),
            "git_commit": context.git_commit,
            "pool_manifest_sha256": sha256_file(pool_manifest_path),
            "processed_sha256": sha256_file(parquet_path),
            "promotion_gate_sha256": sha256_file(gate_path),
            "sources": {
                "metadata": protocol.metadata_sha256,
                "scores": protocol.zero_shot_scores_sha256,
                "substitutions": protocol.substitutions_sha256,
                "upstream_commit": protocol.proteingym_upstream_commit,
            },
        },
        "reference_methods": references,
        "schema_version": 1,
        "task": {
            "assay_id": assay_id,
            "labeled_hashes": list(split.labeled_hashes),
            "phase": "replication",
            "seed": seed,
            "test_hashes": list(split.test_hashes),
            "unlabeled_hashes": list(split.unlabeled_hashes),
        },
        "variant": variant,
    }
    normalized = cast(dict[str, object], _jsonable(payload))
    _validate_replication_task_payload(normalized)
    return normalized


def _task_path(results_root: Path, assay_id: str, seed: int) -> Path:
    return results_root / "tasks" / assay_id / f"seed_{seed}.json"


@app.command("create-gate")
def create_gate(
    context: typer.Context,
    base_manifest_path: Annotated[Path, typer.Option("--base-manifest")],
    pool_manifest_path: Annotated[Path, typer.Option("--pool-manifest")],
    processed_root: Annotated[Path, typer.Option("--processed-root")],
    embedding_root: Annotated[Path, typer.Option("--embedding-root")],
    screen_results_root: Annotated[Path, typer.Option("--screen-results-root")],
    screen_aggregate: Annotated[Path, typer.Option("--screen-aggregate")],
    output: Annotated[Path, typer.Option("--output")],
) -> None:
    """Exact-rebuild the official screen and write its one-way promotion gate."""
    options = _options(context)

    def action() -> dict[str, object]:
        protocol = load_protocol(options.config)
        _validate_execution_policy(protocol, non_official_bypass=False)
        _, pool = _load_manifests(
            protocol=protocol,
            base_manifest_path=base_manifest_path,
            pool_manifest_path=pool_manifest_path,
        )
        if options.dry_run:
            return {"plan": {"output": str(output), "task_count": 45}}
        git_commit = _git_commit(require_clean=True)
        actual = _load_unique_json(screen_aggregate)
        _validate_crossfit_aggregate_payload(actual)
        expected = _build_crossfit_aggregate_payload(
            protocol=protocol,
            base_manifest_path=base_manifest_path,
            pool_manifest=pool,
            pool_manifest_path=pool_manifest_path,
            processed_root=processed_root,
            embedding_root=embedding_root,
            results_root=screen_results_root,
            non_official_bypass=False,
        )
        _require_exact_payload(actual, expected, artifact_kind="crossfit screen")
        gate = build_screen_promotion_gate(
            expected,
            pool_manifest=pool,
            protocol=protocol,
            pool_manifest_sha256=sha256_file(pool_manifest_path),
            screen_aggregate_sha256=sha256_file(screen_aggregate),
            git_commit=git_commit,
        )
        created = _write_json_once(output, gate.model_dump(mode="json"))
        load_screen_promotion_gate(output)
        return {
            "artifact": str(output),
            "artifact_sha256": sha256_file(output),
            "created": created,
        }

    _run_command("crossfit-create-gate", dry_run=options.dry_run, action=action)


@app.command("run-task")
def run_task(
    context: typer.Context,
    base_manifest_path: Annotated[Path, typer.Option("--base-manifest")],
    pool_manifest_path: Annotated[Path, typer.Option("--pool-manifest")],
    promotion_gate: Annotated[Path, typer.Option("--promotion-gate")],
    processed_root: Annotated[Path, typer.Option("--processed-root")],
    embedding_root: Annotated[Path, typer.Option("--embedding-root")],
    results_root: Annotated[Path, typer.Option("--results-root")],
    task_index: Annotated[int, typer.Option("--task-index", min=0)],
    expected_gate_sha256: Annotated[
        str | None,
        typer.Option("--expected-promotion-gate-sha256"),
    ] = None,
    non_official_bypass: Annotated[
        bool,
        typer.Option("--non-official-bypass"),
    ] = False,
) -> None:
    """Run one exact member of the gated 130-task replication."""
    options = _options(context)

    def action() -> dict[str, object]:
        loaded = _load_replication_context(
            config=options.config,
            base_manifest_path=base_manifest_path,
            pool_manifest_path=pool_manifest_path,
            gate_path=promotion_gate,
            expected_gate_sha256=expected_gate_sha256,
            non_official_bypass=non_official_bypass,
        )
        assay_id, seed = resolve_replication_task_index(
            loaded.pool_manifest,
            loaded.protocol,
            task_index,
        )
        destination = _task_path(results_root, assay_id, seed)
        if options.dry_run:
            return {"plan": {"assay_id": assay_id, "seed": seed}}
        payload = _build_replication_task_payload(
            context=loaded,
            base_manifest_path=base_manifest_path,
            pool_manifest_path=pool_manifest_path,
            gate_path=promotion_gate,
            processed_root=processed_root,
            embedding_root=embedding_root,
            assay_id=assay_id,
            seed=seed,
            non_official_bypass=non_official_bypass,
        )
        created = _write_json_once(destination, payload)
        return {
            "artifact": str(destination),
            "artifact_sha256": sha256_file(destination),
            "assay_id": assay_id,
            "created": created,
            "seed": seed,
        }

    _run_command("crossfit-replication-task", dry_run=options.dry_run, action=action)


def _summary_payload(summary: Any) -> dict[str, object]:
    return cast(dict[str, object], _jsonable(dataclasses.asdict(summary)))


def _validate_replication_aggregate_payload(payload: dict[str, object]) -> None:
    if (
        payload.get("schema_version") != 1
        or payload.get("kind") != "crossfit_replication_aggregate_result"
    ):
        raise ValueError("crossfit replication aggregate schema is invalid")
    grid = payload.get("grid")
    provenance = payload.get("provenance")
    verdict = payload.get("verdict")
    if (
        not isinstance(grid, dict)
        or len(cast(list[object], grid.get("tasks", []))) != 130
        or not isinstance(provenance, dict)
        or provenance.get("task_count") != 130
        or not isinstance(verdict, dict)
    ):
        raise ValueError("crossfit replication aggregate grid is invalid")


def _build_replication_aggregate_payload(
    *,
    context: ReplicationContext,
    base_manifest_path: Path,
    pool_manifest_path: Path,
    gate_path: Path,
    processed_root: Path,
    embedding_root: Path,
    results_root: Path,
    non_official_bypass: bool,
) -> dict[str, object]:
    grid = replication_grid(context.pool_manifest, context.protocol)
    rows: list[dict[str, object]] = []
    diagnostics: list[dict[str, object]] = []
    task_manifest: list[dict[str, object]] = []
    git_commits: set[str] = set()
    runtimes: set[bytes] = set()
    runtime: object = None
    for assay_id, seed in grid:
        path = _task_path(results_root, assay_id, seed)
        if not path.is_file():
            raise ValueError(f"missing replication task artifact: {path}")
        actual = _load_unique_json(path)
        _validate_replication_task_payload(actual)
        expected = _build_replication_task_payload(
            context=context,
            base_manifest_path=base_manifest_path,
            pool_manifest_path=pool_manifest_path,
            gate_path=gate_path,
            processed_root=processed_root,
            embedding_root=embedding_root,
            assay_id=assay_id,
            seed=seed,
            non_official_bypass=non_official_bypass,
        )
        _require_exact_payload(actual, expected, artifact_kind="replication task")
        provenance = cast(dict[str, object], actual["provenance"])
        execution = cast(dict[str, object], actual["execution"])
        git_commits.add(cast(str, provenance["git_commit"]))
        runtime = execution["numerical_runtime"]
        runtimes.add(_canonical_json_bytes(runtime))
        method_rows = [
            *cast(list[dict[str, object]], actual["reference_methods"]),
            cast(dict[str, object], actual["variant"]),
        ]
        for method in method_rows:
            rows.append(
                {
                    "assay_id": assay_id,
                    "method": method["name"],
                    "mse": method["mse"],
                    "ndcg_10pct": method["ndcg_10pct"],
                    "seed": seed,
                    "spearman": method["spearman"],
                }
            )
        task_sha = sha256_file(path)
        diagnostics.append(
            {
                "assay_id": assay_id,
                "diagnostics": actual["diagnostics"],
                "seed": seed,
                "task_artifact_sha256": task_sha,
            }
        )
        task_manifest.append(
            {"assay_id": assay_id, "seed": seed, "sha256": task_sha}
        )
    if git_commits != {context.git_commit} or len(runtimes) != 1:
        raise ValueError("replication tasks mix implementations or runtimes")
    methods = (*METHOD_NAMES, _METHOD)
    results = validate_result_table(
        pd.DataFrame(rows),
        assay_ids=context.pool_manifest.untouched_ids,
        seeds=context.protocol.seeds,
        required_methods=methods,
    )
    comparisons = {
        second: pairwise_summary(
            results,
            first=_METHOD,
            second=second,
            metric="spearman",
        )
        for second in METHOD_NAMES
    }
    verdict: CrossfitReplicationVerdict = crossfit_replication_verdict(
        comparisons["random"]
    )
    payload: dict[str, object] = {
        "analysis": {
            "inference_unit": "assay",
            "metric": "spearman",
            "sign_flip": "exact",
        },
        "card": {"id": CARD_ID, "sha256": CARD_SHA},
        "diagnostics": diagnostics,
        "effects": {
            f"crossfit_minus_{second}": _summary_payload(summary)
            for second, summary in comparisons.items()
        },
        "execution": {
            "bypass": (
                "non_official_test_or_development" if non_official_bypass else None
            ),
            "numerical_policy": NUMERICAL_POLICY,
            "numerical_runtime": runtime,
            "official": not non_official_bypass,
        },
        "grid": {
            "assay_ids": list(context.pool_manifest.untouched_ids),
            "seeds": list(context.protocol.seeds),
            "tasks": [list(task) for task in grid],
        },
        "kind": "crossfit_replication_aggregate_result",
        "long_results": results.to_dict(orient="records"),
        "method_table": method_summary_table(results).to_dict(orient="records"),
        "provenance": {
            "base_manifest_sha256": sha256_file(base_manifest_path),
            "git_commit": context.git_commit,
            "pool_manifest_sha256": sha256_file(pool_manifest_path),
            "promotion_gate_sha256": sha256_file(gate_path),
            "protocol_digest": canonical_protocol_digest(context.protocol),
            "task_count": len(task_manifest),
            "task_manifest": task_manifest,
        },
        "schema_version": 1,
        "verdict": {
            "exact_sign_flip_pvalue": verdict.exact_sign_flip_pvalue,
            "replication_success": verdict.replication_success,
            "rule": {
                "assay_win_threshold": 16,
                "mean_gain_strictly_positive": True,
                "task_win_threshold": 78,
            },
        },
    }
    normalized = cast(dict[str, object], _jsonable(payload))
    _validate_replication_aggregate_payload(normalized)
    return normalized


@app.command("aggregate")
def aggregate(
    context: typer.Context,
    base_manifest_path: Annotated[Path, typer.Option("--base-manifest")],
    pool_manifest_path: Annotated[Path, typer.Option("--pool-manifest")],
    promotion_gate: Annotated[Path, typer.Option("--promotion-gate")],
    processed_root: Annotated[Path, typer.Option("--processed-root")],
    embedding_root: Annotated[Path, typer.Option("--embedding-root")],
    results_root: Annotated[Path, typer.Option("--results-root")],
    output: Annotated[Path, typer.Option("--output")],
    expected_gate_sha256: Annotated[
        str | None,
        typer.Option("--expected-promotion-gate-sha256"),
    ] = None,
    non_official_bypass: Annotated[
        bool,
        typer.Option("--non-official-bypass"),
    ] = False,
) -> None:
    """Exact-rebuild all 130 tasks and apply the replication verdict."""
    options = _options(context)

    def action() -> dict[str, object]:
        loaded = _load_replication_context(
            config=options.config,
            base_manifest_path=base_manifest_path,
            pool_manifest_path=pool_manifest_path,
            gate_path=promotion_gate,
            expected_gate_sha256=expected_gate_sha256,
            non_official_bypass=non_official_bypass,
        )
        if options.dry_run:
            return {"plan": {"output": str(output), "task_count": 130}}
        payload = _build_replication_aggregate_payload(
            context=loaded,
            base_manifest_path=base_manifest_path,
            pool_manifest_path=pool_manifest_path,
            gate_path=promotion_gate,
            processed_root=processed_root,
            embedding_root=embedding_root,
            results_root=results_root,
            non_official_bypass=non_official_bypass,
        )
        created = _write_json_once(output, payload)
        return {
            "artifact": str(output),
            "artifact_sha256": sha256_file(output),
            "created": created,
            "replication_success": cast(dict[str, object], payload["verdict"])[
                "replication_success"
            ],
            "task_count": 130,
        }

    _run_command(
        "crossfit-replication-aggregate",
        dry_run=options.dry_run,
        action=action,
    )


def _verify_inputs(
    context: ReplicationContext,
    *,
    processed_root: Path,
    embedding_root: Path,
    non_official_bypass: bool,
) -> None:
    _require_replication_roots(
        processed_root,
        embedding_root,
        non_official_bypass=non_official_bypass,
    )
    for assay_id in context.pool_manifest.untouched_ids:
        record = _record(context.pool_manifest, assay_id)
        _load_identity_frame(
            processed_root / f"{assay_id}.parquet",
            protocol=context.protocol,
            record=record,
        )
        metadata = embedding_root / f"{assay_id}.json"
        load_embedding_cache(
            embedding_root / f"{assay_id}.npy",
            metadata,
            dms_id=assay_id,
            row_hashes=record.row_hashes,
            model_id=context.protocol.model,
            model_revision=context.protocol.model_revision,
            sources=_embedding_sources(context.protocol),
            expected_embedding_dim=(
                _cache_width(metadata)
                if non_official_bypass
                else _ESM2_35M_EMBEDDING_DIM
            ),
        )


@app.command("verify")
def verify(
    context: typer.Context,
    base_manifest_path: Annotated[Path, typer.Option("--base-manifest")],
    pool_manifest_path: Annotated[Path, typer.Option("--pool-manifest")],
    promotion_gate: Annotated[Path, typer.Option("--promotion-gate")],
    processed_root: Annotated[Path, typer.Option("--processed-root")],
    embedding_root: Annotated[Path, typer.Option("--embedding-root")],
    results_root: Annotated[Path | None, typer.Option("--results-root")] = None,
    task_artifact: Annotated[Path | None, typer.Option("--task-artifact")] = None,
    aggregate_artifact: Annotated[
        Path | None,
        typer.Option("--aggregate-artifact"),
    ] = None,
    expected_gate_sha256: Annotated[
        str | None,
        typer.Option("--expected-promotion-gate-sha256"),
    ] = None,
    non_official_bypass: Annotated[
        bool,
        typer.Option("--non-official-bypass"),
    ] = False,
) -> None:
    """Verify the gate, all inputs, and optionally exact task outputs."""
    options = _options(context)

    def action() -> dict[str, object]:
        loaded = _load_replication_context(
            config=options.config,
            base_manifest_path=base_manifest_path,
            pool_manifest_path=pool_manifest_path,
            gate_path=promotion_gate,
            expected_gate_sha256=expected_gate_sha256,
            non_official_bypass=non_official_bypass,
        )
        if options.dry_run:
            return {"plan": {"task_count": 130}}
        _verify_inputs(
            loaded,
            processed_root=processed_root,
            embedding_root=embedding_root,
            non_official_bypass=non_official_bypass,
        )
        verified = ["gate", "processed", "embeddings"]
        if task_artifact is not None:
            actual = _load_unique_json(task_artifact)
            _validate_replication_task_payload(actual)
            identity = cast(dict[str, object], actual["task"])
            expected = _build_replication_task_payload(
                context=loaded,
                base_manifest_path=base_manifest_path,
                pool_manifest_path=pool_manifest_path,
                gate_path=promotion_gate,
                processed_root=processed_root,
                embedding_root=embedding_root,
                assay_id=cast(str, identity["assay_id"]),
                seed=cast(int, identity["seed"]),
                non_official_bypass=non_official_bypass,
            )
            _require_exact_payload(actual, expected, artifact_kind="replication task")
            verified.append("task")
        if results_root is not None:
            expected_aggregate = _build_replication_aggregate_payload(
                context=loaded,
                base_manifest_path=base_manifest_path,
                pool_manifest_path=pool_manifest_path,
                gate_path=promotion_gate,
                processed_root=processed_root,
                embedding_root=embedding_root,
                results_root=results_root,
                non_official_bypass=non_official_bypass,
            )
            verified.append("exact_130_task_grid")
            if aggregate_artifact is not None:
                actual_aggregate = _load_unique_json(aggregate_artifact)
                _validate_replication_aggregate_payload(actual_aggregate)
                _require_exact_payload(
                    actual_aggregate,
                    expected_aggregate,
                    artifact_kind="replication aggregate",
                )
                verified.append("aggregate")
        elif aggregate_artifact is not None:
            raise ValueError("aggregate verification requires --results-root")
        return {"verified": verified}

    _run_command("crossfit-replication-verify", dry_run=options.dry_run, action=action)


if __name__ == "__main__":
    app()
