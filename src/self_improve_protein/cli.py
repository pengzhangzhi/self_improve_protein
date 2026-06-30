"""Restart-safe command line interface for the frozen ProteinGym study."""

from __future__ import annotations

import dataclasses
import fcntl
import json
import os
import subprocess
import sys
import tempfile
import zipfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal, cast

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import typer
from numpy.typing import NDArray
from typer import _click

from self_improve_protein.analysis import (
    comparison_summary_table,
    method_summary_table,
    v0_analysis_verdict,
    validate_result_table,
    validate_v0_result_table,
)
from self_improve_protein.config import DEFAULT_CONFIG_PATH, Protocol, load_protocol
from self_improve_protein.data import (
    AssayEligibility,
    DataManifest,
    ManifestSource,
    ManifestSources,
    SelectedAssayManifest,
    build_working_set,
    filter_usable_variants,
    load_assay_from_archives,
    load_data_manifest,
    make_split,
    row_hash,
    select_eligible_assays,
)
from self_improve_protein.embeddings import (
    EmbeddingCacheSources,
    get_or_create_embedding_cache,
    load_embedding_cache,
)
from self_improve_protein.experiment import (
    METHOD_NAMES,
    NUMERICAL_POLICY,
    EvaluationLabels,
    FitInputs,
    canonical_evaluation_digest,
    canonical_fit_digest,
    canonical_protocol_digest,
    canonical_source_digest,
    current_numerical_runtime_fingerprint,
    evaluate_task,
    fit_task,
    require_openblas_coretype,
)
from self_improve_protein.provenance import (
    atomic_write_json,
    sha256_file,
)

app = typer.Typer(
    name="self-improve-protein",
    help="Run the provenance-locked ProteinGym low-label experiment.",
    invoke_without_command=True,
    no_args_is_help=False,
    pretty_exceptions_enable=False,
)

_SCHEMA_VERSION = 1
_ESM2_35M_EMBEDDING_DIM = 480
_LOCKED_V0_PROTOCOL_DIGEST = (
    "0b2a74ff76b8c7c508ceea16b004a1c128ba15704138138d49b2c153bcbfa49a"
)
_SHA256_CHARS = frozenset("0123456789abcdef")
FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class _RuntimeOptions:
    config: Path
    dry_run: bool


def _jsonable(value: object) -> object:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _jsonable(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _emit(payload: dict[str, object]) -> None:
    typer.echo(
        json.dumps(
            _jsonable(payload),
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    )


def _run_command(
    name: str,
    *,
    dry_run: bool,
    action: Callable[[], dict[str, object]],
) -> None:
    numerical_runtime = _jsonable(current_numerical_runtime_fingerprint())
    _emit(
        {
            "command": name,
            "event": "start",
            "numerical_runtime": numerical_runtime,
            "schema_version": _SCHEMA_VERSION,
        }
    )
    try:
        terminal = action()
    except Exception as error:
        _emit(
            {
                "command": name,
                "error": str(error),
                "error_type": type(error).__name__,
                "event": "terminal",
                "numerical_runtime": numerical_runtime,
                "schema_version": _SCHEMA_VERSION,
                "status": "error",
            }
        )
        raise typer.Exit(code=1) from error
    _emit(
        {
            "command": name,
            "event": "terminal",
            "numerical_runtime": numerical_runtime,
            "schema_version": _SCHEMA_VERSION,
            "status": "planned" if dry_run else "complete",
            **terminal,
        }
    )


def _options(context: typer.Context) -> _RuntimeOptions:
    if not isinstance(context.obj, _RuntimeOptions):
        raise RuntimeError("CLI runtime options were not initialized")
    return context.obj


@app.callback()
def main(
    context: typer.Context,
    config: Annotated[
        Path,
        typer.Option(
            "--config",
            help="Validated protocol YAML used by every stage.",
        ),
    ] = DEFAULT_CONFIG_PATH,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help="Render the exact stage plan without writing artifacts.",
        ),
    ] = False,
    show_config: Annotated[
        bool,
        typer.Option(
            "--show-config",
            help="Print the validated canonical protocol and exit.",
        ),
    ] = False,
) -> None:
    """Set the shared protocol and mutation policy for all subcommands."""
    context.obj = _RuntimeOptions(config=config, dry_run=dry_run)
    if show_config:
        protocol = load_protocol(config)
        _emit(
            {
                "event": "config",
                "protocol": protocol.model_dump(mode="json"),
                "protocol_digest": canonical_protocol_digest(protocol),
                "schema_version": _SCHEMA_VERSION,
            }
        )
        raise typer.Exit()
    if context.invoked_subcommand is None:
        typer.echo(context.get_help())
        raise typer.Exit()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and set(value).issubset(_SHA256_CHARS)
    )


def _validate_execution_policy(
    protocol: Protocol,
    *,
    mode: Literal["confirmatory", "development"],
    non_official_bypass: bool,
) -> None:
    protocol_digest = canonical_protocol_digest(protocol)
    if mode == "confirmatory" and non_official_bypass:
        raise ValueError("confirmatory execution forbids --non-official-bypass")
    if not non_official_bypass and protocol_digest != _LOCKED_V0_PROTOCOL_DIGEST:
        raise ValueError("official execution requires the locked v0 protocol digest")
    if mode == "confirmatory" and protocol_digest != _LOCKED_V0_PROTOCOL_DIGEST:
        raise ValueError(
            "confirmatory execution requires the locked v0 protocol digest"
        )


def _load_unique_json(path: Path) -> dict[str, object]:
    def unique(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r} in {path}")
            result[key] = value
        return result

    try:
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle, object_pairs_hook=unique)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON artifact: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact must be an object: {path}")
    return payload


def _canonical_json_bytes(payload: object) -> bytes:
    try:
        return (
            json.dumps(
                _jsonable(payload),
                allow_nan=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode()
    except (TypeError, ValueError) as error:
        raise ValueError(
            "artifact payload must be finite and JSON serializable"
        ) from error


def _write_json_once(path: Path, payload: object) -> bool:
    """Atomically create an artifact or accept only an exact existing payload."""
    expected = _canonical_json_bytes(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(f".{path.name}.lock")
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            if path.exists():
                try:
                    actual_payload = _load_unique_json(path)
                    actual = _canonical_json_bytes(actual_payload)
                except ValueError as error:
                    raise ValueError(f"mismatched existing artifact: {path}") from error
                if actual != expected:
                    raise ValueError(f"mismatched existing artifact: {path}")
                return False
            atomic_write_json(path, _jsonable(payload))
            return True
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _write_parquet_once(path: Path, frame: pd.DataFrame) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(f".{path.name}.lock")
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            if path.exists():
                existing = pd.read_parquet(path)
                try:
                    pd.testing.assert_frame_equal(
                        existing,
                        frame,
                        check_exact=True,
                        check_dtype=True,
                        check_like=False,
                    )
                except AssertionError as error:
                    raise ValueError(
                        f"mismatched existing processed artifact: {path}"
                    ) from error
                return False
            temporary: Path | None = None
            try:
                with tempfile.NamedTemporaryFile(
                    mode="w+b",
                    dir=path.parent,
                    prefix=f".{path.name}.",
                    suffix=".tmp",
                    delete=False,
                ) as handle:
                    temporary = Path(handle.name)
                frame.to_parquet(temporary, index=False)
                with temporary.open("rb") as handle:
                    os.fsync(handle.fileno())
                os.replace(temporary, path)
                temporary = None
                directory_fd = os.open(
                    path.parent,
                    os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
                )
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
            return True
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _validate_manifest(protocol: Protocol, manifest: DataManifest) -> None:
    checks: tuple[tuple[str, object, object], ...] = (
        ("data_release", manifest.data_release, protocol.data_release),
        ("teacher_column", manifest.teacher_column, protocol.teacher_column),
        (
            "upstream_revision",
            manifest.upstream_revision,
            protocol.proteingym_upstream_commit,
        ),
        ("working_size", manifest.working_size, protocol.working_size),
        ("max_length", manifest.max_length, protocol.max_length),
        ("assay_count", len(manifest.confirmatory_ids), protocol.assay_count),
        (
            "substitutions URL",
            manifest.sources.substitutions.url,
            protocol.substitutions_url,
        ),
        (
            "substitutions SHA",
            manifest.sources.substitutions.sha256,
            protocol.substitutions_sha256,
        ),
        ("scores URL", manifest.sources.scores.url, protocol.zero_shot_scores_url),
        (
            "scores SHA",
            manifest.sources.scores.sha256,
            protocol.zero_shot_scores_sha256,
        ),
        ("metadata URL", manifest.sources.metadata.url, protocol.metadata_url),
        ("metadata SHA", manifest.sources.metadata.sha256, protocol.metadata_sha256),
    )
    mismatches = [name for name, actual, expected in checks if actual != expected]
    if mismatches:
        raise ValueError("manifest does not match protocol: " + ", ".join(mismatches))


def _embedding_sources(protocol: Protocol) -> EmbeddingCacheSources:
    return EmbeddingCacheSources(
        proteingym_upstream_commit=protocol.proteingym_upstream_commit,
        substitutions_sha256=protocol.substitutions_sha256,
        zero_shot_scores_sha256=protocol.zero_shot_scores_sha256,
        metadata_sha256=protocol.metadata_sha256,
    )


def _record(manifest: DataManifest, dms_id: str) -> SelectedAssayManifest:
    matches = [record for record in manifest.selected_assays if record.dms_id == dms_id]
    if len(matches) != 1:
        raise ValueError(f"assay {dms_id!r} is not uniquely selected in manifest")
    return matches[0]


def _resolve_assay(
    manifest: DataManifest,
    *,
    assay_id: str | None,
    assay_index: int | None,
) -> str:
    if (assay_id is None) == (assay_index is None):
        raise ValueError("provide exactly one of --assay-id or --assay-index")
    if assay_id is not None:
        _record(manifest, assay_id)
        return assay_id
    assert assay_index is not None
    if assay_index < 0 or assay_index >= len(manifest.selected_assays):
        raise ValueError("assay index is outside the selected manifest")
    return manifest.selected_assays[assay_index].dms_id


def _load_identity_frame(
    parquet_path: Path,
    *,
    protocol: Protocol,
    record: SelectedAssayManifest,
) -> pd.DataFrame:
    columns = [
        "dms_id",
        "mutant",
        "mutated_sequence",
        "sequence_hash",
        protocol.teacher_column,
    ]
    try:
        frame = pd.read_parquet(parquet_path, columns=columns)
    except (OSError, ValueError) as error:
        raise ValueError(
            f"cannot load outcome-free working set: {parquet_path}"
        ) from error
    if len(frame) != protocol.working_size:
        raise ValueError("processed working-set row count does not match protocol")
    actual_hashes = tuple(str(value) for value in frame["sequence_hash"])
    if actual_hashes != record.row_hashes:
        raise ValueError("processed working-set row hashes do not match manifest")
    for dms_id, mutant, sequence, expected_hash in frame.loc[
        :, ["dms_id", "mutant", "mutated_sequence", "sequence_hash"]
    ].itertuples(index=False, name=None):
        if (
            not isinstance(dms_id, str)
            or not isinstance(mutant, str)
            or not isinstance(sequence, str)
            or dms_id != record.dms_id
            or row_hash(dms_id, mutant, sequence) != expected_hash
        ):
            raise ValueError("processed working-set identity is corrupt")
    teacher = pd.to_numeric(frame[protocol.teacher_column], errors="raise").to_numpy(
        dtype=np.float64
    )
    if not np.all(np.isfinite(teacher)):
        raise ValueError("processed teacher scores must be finite")
    return frame


def _ordered_outcomes(
    parquet_path: Path,
    *,
    requested_hashes: Sequence[str],
    expected_all_hashes: Sequence[str] | None,
) -> FloatArray:
    requested = tuple(requested_hashes)
    if not requested or len(set(requested)) != len(requested):
        raise ValueError("requested outcome hashes must be non-empty and unique")
    try:
        if expected_all_hashes is not None:
            identities = pd.read_parquet(
                parquet_path,
                columns=["sequence_hash"],
            )
            actual_all = tuple(str(value) for value in identities["sequence_hash"])
            if actual_all != tuple(expected_all_hashes):
                raise ValueError(
                    "outcome projection ordered hashes do not match manifest"
                )
        outcomes = pd.read_parquet(
            parquet_path,
            columns=["sequence_hash", "DMS_score"],
            filters=[("sequence_hash", "in", list(requested))],
        )
    except (OSError, ValueError) as error:
        raise ValueError(f"cannot load outcome projection: {parquet_path}") from error
    if outcomes["sequence_hash"].duplicated().any():
        raise ValueError("outcome projection has duplicate sequence hashes")
    keyed = outcomes.set_index("sequence_hash")
    missing = [value for value in requested if value not in keyed.index]
    if missing:
        raise ValueError("requested outcome hashes are missing from the working set")
    values = pd.to_numeric(
        keyed.loc[list(requested), "DMS_score"],
        errors="raise",
    ).to_numpy(dtype=np.float64)
    if values.shape != (len(requested),) or not np.all(np.isfinite(values)):
        raise ValueError("ordered outcomes must be one-dimensional and finite")
    return cast(FloatArray, values)


def load_evaluation_labels(
    parquet_path: Path | str,
    *,
    assay_id: str,
    seed: int,
    source_digest: str,
    labeled_hashes: Sequence[str],
    unlabeled_hashes: Sequence[str],
    test_hashes: Sequence[str],
    expected_all_hashes: Sequence[str] | None = None,
) -> EvaluationLabels:
    """Load hidden pool/test outcomes in an independently verified hash order."""
    hidden_hashes = (*tuple(unlabeled_hashes), *tuple(test_hashes))
    values = _ordered_outcomes(
        Path(parquet_path),
        requested_hashes=hidden_hashes,
        expected_all_hashes=expected_all_hashes,
    )
    unlabeled_count = len(tuple(unlabeled_hashes))
    return EvaluationLabels(
        assay_id=assay_id,
        seed=seed,
        source_digest=source_digest,
        labeled_hashes=tuple(labeled_hashes),
        unlabeled_hashes=tuple(unlabeled_hashes),
        test_hashes=tuple(test_hashes),
        y_u=values[:unlabeled_count],
        y_test=values[unlabeled_count:],
    )


def _cache_width(metadata_path: Path) -> int:
    payload = _load_unique_json(metadata_path)
    shape = payload.get("shape")
    if (
        not isinstance(shape, list)
        or len(shape) != 2
        or any(type(value) is not int or value <= 0 for value in shape)
    ):
        raise ValueError("embedding metadata has invalid shape")
    return cast(int, shape[1])


def _git_commit(*, require_clean: bool = False) -> str:
    if require_clean:
        clean = subprocess.run(
            [
                "git",
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if clean.returncode != 0 or clean.stdout:
            raise ValueError("official execution requires a clean Git worktree")
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    value = completed.stdout.strip()
    if completed.returncode != 0 or len(value) != 40:
        raise ValueError("cannot determine git commit")
    return value


@app.command("prepare-data")
def prepare_data(
    context: typer.Context,
    dms_zip: Annotated[Path, typer.Option("--dms-zip")],
    scores_zip: Annotated[Path, typer.Option("--scores-zip")],
    metadata_csv: Annotated[Path, typer.Option("--metadata-csv")],
    processed_root: Annotated[Path, typer.Option("--processed-root")],
    manifest_path: Annotated[Path, typer.Option("--manifest")],
) -> None:
    """Verify pinned sources, select assays, and freeze processed working sets."""
    options = _options(context)

    def action() -> dict[str, object]:
        protocol = load_protocol(options.config)
        plan = {
            "manifest": str(manifest_path),
            "processed_root": str(processed_root),
            "source_paths": [str(dms_zip), str(scores_zip), str(metadata_csv)],
            "stages": ["verify_sources", "scan_eligibility", "freeze_working_sets"],
        }
        if options.dry_run:
            return {"plan": plan}
        expected_hashes = (
            (dms_zip, protocol.substitutions_sha256),
            (scores_zip, protocol.zero_shot_scores_sha256),
            (metadata_csv, protocol.metadata_sha256),
        )
        for path, expected in expected_hashes:
            if not path.is_file() or sha256_file(path) != expected:
                raise ValueError(f"source checksum mismatch: {path}")
        metadata = pd.read_csv(metadata_csv, usecols=["DMS_id", "seq_len"])
        if metadata["DMS_id"].duplicated().any():
            raise ValueError("metadata has duplicate DMS_id values")
        sequence_lengths = {
            str(dms_id): int(sequence_length)
            for dms_id, sequence_length in metadata.itertuples(index=False, name=None)
            if int(sequence_length) <= protocol.max_length
        }
        with (
            zipfile.ZipFile(dms_zip) as dms_archive,
            zipfile.ZipFile(scores_zip) as score_archive,
        ):
            dms_ids = {
                Path(name).stem
                for name in dms_archive.namelist()
                if name.startswith("DMS_ProteinGym_substitutions/")
                and name.endswith(".csv")
            }
            score_ids = {
                Path(name).stem
                for name in score_archive.namelist()
                if "/" not in name.rstrip("/") and name.endswith(".csv")
            }
        candidates = sorted(set(sequence_lengths) & dms_ids & score_ids)
        eligibility: list[AssayEligibility] = []
        working_frames: dict[str, pd.DataFrame] = {}
        for dms_id in candidates:
            merged = load_assay_from_archives(
                dms_zip,
                scores_zip,
                dms_id,
                protocol.teacher_column,
            )
            usable = filter_usable_variants(
                merged,
                protocol.teacher_column,
                protocol.max_length,
            )
            eligibility.append(
                AssayEligibility(
                    dms_id=dms_id,
                    usable_count=len(usable),
                    sequence_length=sequence_lengths[dms_id],
                )
            )
            if len(usable) >= protocol.working_size:
                working_frames[dms_id] = build_working_set(
                    usable,
                    protocol.working_size,
                )
        confirmatory, development = select_eligible_assays(
            eligibility,
            protocol.working_size,
            protocol.assay_count,
        )
        selected_ids = (*confirmatory, development)
        selected_records: list[SelectedAssayManifest] = []
        for dms_id in selected_ids:
            frame = working_frames[dms_id]
            _write_parquet_once(processed_root / f"{dms_id}.parquet", frame)
            eligible = next(item for item in eligibility if item.dms_id == dms_id)
            selected_records.append(
                SelectedAssayManifest(
                    dms_id=dms_id,
                    usable_count=eligible.usable_count,
                    sequence_length=eligible.sequence_length,
                    row_hashes=tuple(str(value) for value in frame["sequence_hash"]),
                )
            )
        eligible_ids = tuple(
            sorted(
                item.dms_id
                for item in eligibility
                if item.usable_count >= protocol.working_size
                and item.sequence_length <= protocol.max_length
            )
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
            eligible_assay_ids=eligible_ids,
            confirmatory_ids=confirmatory,
            development_id=development,
            max_length=protocol.max_length,
            working_size=protocol.working_size,
            selected_assays=tuple(selected_records),
        )
        created = _write_json_once(
            manifest_path,
            manifest.model_dump(mode="json"),
        )
        return {
            "created": created,
            "eligible_count": len(eligible_ids),
            "manifest_sha256": sha256_file(manifest_path),
            "selected_assays": list(selected_ids),
        }

    _run_command("prepare-data", dry_run=options.dry_run, action=action)


@app.command("embed-assay")
def embed_assay(
    context: typer.Context,
    manifest_path: Annotated[Path, typer.Option("--manifest")],
    processed_root: Annotated[Path, typer.Option("--processed-root")],
    embedding_root: Annotated[Path, typer.Option("--embedding-root")],
    assay_id: Annotated[str | None, typer.Option("--assay-id")] = None,
    assay_index: Annotated[int | None, typer.Option("--assay-index")] = None,
    batch_size: Annotated[int, typer.Option("--batch-size", min=1)] = 128,
    device: Annotated[str, typer.Option("--device")] = "cuda",
) -> None:
    """Create or validate one exact revision-pinned ESM-2 embedding cache."""
    options = _options(context)

    def action() -> dict[str, object]:
        protocol = load_protocol(options.config)
        manifest = load_data_manifest(manifest_path)
        _validate_manifest(protocol, manifest)
        resolved = _resolve_assay(
            manifest,
            assay_id=assay_id,
            assay_index=assay_index,
        )
        plan = {
            "assay_id": resolved,
            "batch_size": batch_size,
            "device": device,
            "stages": ["validate_working_set", "embed_or_validate_cache"],
        }
        if options.dry_run:
            return {"plan": plan}
        record = _record(manifest, resolved)
        frame = _load_identity_frame(
            processed_root / f"{resolved}.parquet",
            protocol=protocol,
            record=record,
        )
        npy_path = embedding_root / f"{resolved}.npy"
        metadata_path = embedding_root / f"{resolved}.json"
        cache_hit = npy_path.exists() and metadata_path.exists()
        array = get_or_create_embedding_cache(
            npy_path,
            metadata_path,
            dms_id=resolved,
            row_hashes=record.row_hashes,
            model_id=protocol.model,
            model_revision=protocol.model_revision,
            sources=_embedding_sources(protocol),
            expected_embedding_dim=_ESM2_35M_EMBEDDING_DIM,
            sequences=tuple(str(value) for value in frame["mutated_sequence"]),
            batch_size=batch_size,
            device=device,
        )
        return {
            "assay_id": resolved,
            "cache_hit": cache_hit,
            "metadata_sha256": sha256_file(metadata_path),
            "npy_sha256": sha256_file(npy_path),
            "shape": list(array.shape),
        }

    _run_command("embed-assay", dry_run=options.dry_run, action=action)


def _resolve_task(
    manifest: DataManifest,
    protocol: Protocol,
    *,
    assay_id: str | None,
    seed: int | None,
    task_index: int | None,
    mode: Literal["confirmatory", "development"],
) -> tuple[str, int]:
    assays = (
        manifest.confirmatory_ids
        if mode == "confirmatory"
        else (manifest.development_id,)
    )
    if task_index is not None:
        if assay_id is not None or seed is not None:
            raise ValueError("--task-index cannot be combined with --assay-id/--seed")
        task_count = len(assays) * len(protocol.seeds)
        if task_index < 0 or task_index >= task_count:
            raise ValueError("task index is outside the declared task grid")
        assay_offset, seed_offset = divmod(task_index, len(protocol.seeds))
        return assays[assay_offset], protocol.seeds[seed_offset]
    if assay_id is None or seed is None:
        raise ValueError("provide --task-index or both --assay-id and --seed")
    _record(manifest, assay_id)
    if assay_id not in assays:
        raise ValueError("assay is outside the requested mode")
    if seed not in protocol.seeds:
        raise ValueError("seed is not declared by the protocol")
    return assay_id, seed


def _validate_task_payload(payload: dict[str, object]) -> None:
    if payload.get("schema_version") != 1 or payload.get("kind") != "task_result":
        raise ValueError("task artifact schema is invalid")
    task = payload.get("task")
    digests = payload.get("digests")
    methods = payload.get("methods")
    if not isinstance(task, dict) or not isinstance(digests, dict):
        raise ValueError("task artifact identity or digest block is invalid")
    if any(
        not _is_sha256(digests.get(name))
        for name in ("fit", "evaluation", "protocol", "source")
    ):
        raise ValueError("task artifact digests are invalid")
    if (
        not isinstance(methods, list)
        or tuple(row.get("name") if isinstance(row, dict) else None for row in methods)
        != METHOD_NAMES
    ):
        raise ValueError("task artifact must contain the five locked methods in order")
    method_rows = cast(list[dict[str, object]], methods)
    for row in method_rows:
        for metric in ("spearman", "mse", "ndcg_10pct"):
            value = row.get(metric)
            if not isinstance(value, (int, float, np.integer, np.floating)) or (
                isinstance(value, (bool, np.bool_)) or not np.isfinite(float(value))
            ):
                raise ValueError(f"task artifact method {metric} is invalid")


def _build_task_payload(
    *,
    protocol: Protocol,
    manifest: DataManifest,
    manifest_path: Path,
    processed_root: Path,
    embedding_root: Path,
    assay_id: str,
    seed: int,
    mode: Literal["confirmatory", "development"],
    non_official_bypass: bool,
    r5_gate_sha256: str | None,
) -> dict[str, object]:
    """Recompute one complete task artifact from its current pinned inputs."""
    _validate_execution_policy(
        protocol,
        mode=mode,
        non_official_bypass=non_official_bypass,
    )
    if mode == "confirmatory" and not _is_sha256(r5_gate_sha256):
        raise ValueError("confirmatory execution requires a validated R5 gate")
    if mode == "development" and r5_gate_sha256 is not None:
        raise ValueError("development execution cannot consume an R5 gate")
    git_commit = _git_commit(require_clean=not non_official_bypass)
    expected_assays = (
        manifest.confirmatory_ids
        if mode == "confirmatory"
        else (manifest.development_id,)
    )
    if assay_id not in expected_assays or seed not in protocol.seeds:
        raise ValueError("task identity is outside the requested exact grid")
    if not non_official_bypass:
        require_openblas_coretype("Haswell")
    record = _record(manifest, assay_id)
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
    embedding_dim = (
        _cache_width(metadata_path) if non_official_bypass else _ESM2_35M_EMBEDDING_DIM
    )
    npy_path = embedding_root / f"{assay_id}.npy"
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
    source_digest = canonical_source_digest(protocol)
    teacher = identity[protocol.teacher_column].to_numpy(dtype=np.float64)
    fit_inputs = FitInputs(
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
    fit = fit_task(fit_inputs, protocol)
    fit_digest = canonical_fit_digest(fit)
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
    evaluation = evaluate_task(
        fit,
        labels,
        protocol=protocol,
        expected_fit_digest=fit_digest,
        expected_evaluation_digest=evaluation_digest,
    )
    evaluations = {item.name: item for item in evaluation.methods}
    methods: list[dict[str, object]] = []
    for method in fit.methods:
        measured = evaluations[method.name]
        methods.append(
            {
                "mse": measured.mse,
                "name": method.name,
                "ndcg_10pct": measured.ndcg_10pct,
                "selected_hashes": list(method.selected_hashes),
                "selected_indices": list(method.selected_indices),
                "selected_pseudo_label_mae": measured.selected_pseudo_label_mae,
                "spearman": measured.spearman,
                "test_predictions": method.test_predictions.tolist(),
            }
        )
    payload: dict[str, object] = {
        "diagnostics": {
            "evaluation": {
                "full_test_risk_oracle": evaluation.full_test_risk_oracle,
                "no_hessian_test_risk_oracle": (evaluation.no_hessian_test_risk_oracle),
                "pool_pseudo_label_mae": evaluation.pool_pseudo_label_mae,
                "random_error_reference": evaluation.random_error_reference,
                "teacher_test_spearman": evaluation.teacher_test_spearman,
            },
            "fit": fit.diagnostics,
        },
        "digests": {
            "evaluation": evaluation_digest,
            "fit": fit_digest,
            "protocol": canonical_protocol_digest(protocol),
            "source": source_digest,
        },
        "execution": {
            "bypass": (
                "non_official_test_or_development" if non_official_bypass else None
            ),
            "numerical_policy": fit.numerical_policy,
            "numerical_runtime": fit.numerical_runtime,
            "official": not non_official_bypass,
        },
        "kind": "task_result",
        "methods": methods,
        "provenance": {
            "embedding_metadata_sha256": sha256_file(metadata_path),
            "embedding_npy_sha256": sha256_file(npy_path),
            "git_commit": git_commit,
            "manifest_sha256": sha256_file(manifest_path),
            "processed_sha256": sha256_file(parquet_path),
            "r5_gate_sha256": r5_gate_sha256,
            "sources": {
                "metadata": protocol.metadata_sha256,
                "scores": protocol.zero_shot_scores_sha256,
                "substitutions": protocol.substitutions_sha256,
                "upstream_commit": protocol.proteingym_upstream_commit,
            },
        },
        "schema_version": 1,
        "task": {
            "assay_id": assay_id,
            "labeled_hashes": list(split.labeled_hashes),
            "mode": mode,
            "seed": seed,
            "test_hashes": list(split.test_hashes),
            "unlabeled_hashes": list(split.unlabeled_hashes),
        },
    }
    normalized = cast(dict[str, object], _jsonable(payload))
    _validate_task_payload(normalized)
    return normalized


def _require_exact_payload(
    actual: dict[str, object],
    expected: dict[str, object],
    *,
    artifact_kind: str,
) -> None:
    if _canonical_json_bytes(actual) != _canonical_json_bytes(expected):
        raise ValueError(f"{artifact_kind} artifact content mismatch")


@app.command("run-task")
def run_task(
    context: typer.Context,
    manifest_path: Annotated[Path, typer.Option("--manifest")],
    processed_root: Annotated[Path, typer.Option("--processed-root")],
    embedding_root: Annotated[Path, typer.Option("--embedding-root")],
    results_root: Annotated[Path, typer.Option("--results-root")],
    assay_id: Annotated[str | None, typer.Option("--assay-id")] = None,
    seed: Annotated[int | None, typer.Option("--seed")] = None,
    task_index: Annotated[int | None, typer.Option("--task-index")] = None,
    mode: Annotated[
        Literal["confirmatory", "development"],
        typer.Option("--mode"),
    ] = "confirmatory",
    non_official_bypass: Annotated[
        bool,
        typer.Option(
            "--non-official-bypass",
            help="Development/tests only: skip the pre-start OpenBLAS core assertion.",
        ),
    ] = False,
    r5_gate: Annotated[Path | None, typer.Option("--r5-gate")] = None,
) -> None:
    """Fit and evaluate one leakage-staged assay-seed task."""
    options = _options(context)

    def action() -> dict[str, object]:
        protocol = load_protocol(options.config)
        manifest = load_data_manifest(manifest_path)
        _validate_manifest(protocol, manifest)
        resolved_assay, resolved_seed = _resolve_task(
            manifest,
            protocol,
            assay_id=assay_id,
            seed=seed,
            task_index=task_index,
            mode=mode,
        )
        _validate_execution_policy(
            protocol,
            mode=mode,
            non_official_bypass=non_official_bypass,
        )
        if mode == "confirmatory":
            if r5_gate is None:
                raise ValueError("confirmatory execution requires --r5-gate")
            gate_digest = _validated_r5_gate_digest(
                r5_gate,
                protocol=protocol,
                manifest=manifest,
                manifest_path=manifest_path,
                processed_root=processed_root,
                embedding_root=embedding_root,
                results_root=results_root,
            )
        else:
            if r5_gate is not None:
                raise ValueError("development execution cannot consume --r5-gate")
            gate_digest = None
        destination = (
            results_root / "tasks" / resolved_assay / f"seed_{resolved_seed}.json"
        )
        plan = {
            "assay_id": resolved_assay,
            "output": str(destination),
            "seed": resolved_seed,
            "r5_gate": str(r5_gate) if r5_gate is not None else None,
            "stages": [
                "validate_inputs",
                "load_labeled_only",
                "fit_and_freeze_digest",
                "load_hidden_labels",
                "evaluate_and_write",
            ],
            "task_mode": mode,
        }
        if options.dry_run:
            return {"plan": plan}
        payload = _build_task_payload(
            protocol=protocol,
            manifest=manifest,
            manifest_path=manifest_path,
            processed_root=processed_root,
            embedding_root=embedding_root,
            assay_id=resolved_assay,
            seed=resolved_seed,
            mode=mode,
            non_official_bypass=non_official_bypass,
            r5_gate_sha256=gate_digest,
        )
        created = _write_json_once(destination, payload)
        return {
            "artifact": str(destination),
            "artifact_sha256": sha256_file(destination),
            "assay_id": resolved_assay,
            "created": created,
            "seed": resolved_seed,
        }

    _run_command("run-task", dry_run=options.dry_run, action=action)


def _task_identity(payload: dict[str, object]) -> tuple[str, int, str]:
    task = payload.get("task")
    if not isinstance(task, dict):
        raise ValueError("task artifact identity block is missing")
    assay_id = task.get("assay_id")
    seed = task.get("seed")
    mode = task.get("mode")
    if (
        not isinstance(assay_id, str)
        or type(seed) is not int
        or not isinstance(mode, str)
    ):
        raise ValueError("task artifact identity values are invalid")
    return assay_id, seed, mode


def _validated_aggregate_seeds(
    protocol: Protocol,
    seeds: Sequence[int],
    *,
    mode: Literal["confirmatory", "development"],
) -> tuple[int, ...]:
    requested = tuple(seeds)
    if (
        not requested
        or len(set(requested)) != len(requested)
        or any(
            type(seed) is not int or seed not in protocol.seeds for seed in requested
        )
    ):
        raise ValueError("aggregate seeds must be a distinct protocol subset")
    if mode == "confirmatory" and requested != protocol.seeds:
        raise ValueError("confirmatory aggregation requires the exact protocol seeds")
    return requested


def _build_aggregate_payload(
    *,
    protocol: Protocol,
    manifest: DataManifest,
    manifest_path: Path,
    processed_root: Path,
    embedding_root: Path,
    results_root: Path,
    mode: Literal["confirmatory", "development"],
    seeds: Sequence[int],
    bootstrap_resamples: int,
    non_official_bypass: bool,
    r5_gate_sha256: str | None,
) -> dict[str, object]:
    """Independently rebuild every task and all aggregate-derived content."""
    _validate_execution_policy(
        protocol,
        mode=mode,
        non_official_bypass=non_official_bypass,
    )
    if mode == "confirmatory" and not _is_sha256(r5_gate_sha256):
        raise ValueError("confirmatory aggregation requires a validated R5 gate")
    if mode == "development" and r5_gate_sha256 is not None:
        raise ValueError("development aggregation cannot consume an R5 gate")
    if type(bootstrap_resamples) is not int or bootstrap_resamples <= 0:
        raise ValueError("bootstrap_resamples must be a positive integer")
    if mode == "confirmatory" and bootstrap_resamples != 10_000:
        raise ValueError("confirmatory aggregation requires 10000 bootstrap resamples")
    requested_seeds = _validated_aggregate_seeds(protocol, seeds, mode=mode)
    assay_ids = (
        manifest.confirmatory_ids
        if mode == "confirmatory"
        else (manifest.development_id,)
    )
    long_rows: list[dict[str, object]] = []
    diagnostic_rows: list[dict[str, object]] = []
    task_manifest: list[dict[str, object]] = []
    git_commits: set[str] = set()
    runtime_payloads: set[bytes] = set()
    numerical_runtime: object | None = None
    for assay_id in assay_ids:
        for seed in requested_seeds:
            path = results_root / "tasks" / assay_id / f"seed_{seed}.json"
            if not path.is_file():
                raise ValueError(f"missing task artifact: {path}")
            actual = _load_unique_json(path)
            _validate_task_payload(actual)
            expected = _build_task_payload(
                protocol=protocol,
                manifest=manifest,
                manifest_path=manifest_path,
                processed_root=processed_root,
                embedding_root=embedding_root,
                assay_id=assay_id,
                seed=seed,
                mode=mode,
                non_official_bypass=non_official_bypass,
                r5_gate_sha256=r5_gate_sha256,
            )
            _require_exact_payload(actual, expected, artifact_kind="task")
            provenance = cast(dict[str, object], actual["provenance"])
            execution = cast(dict[str, object], actual["execution"])
            git_commit = provenance.get("git_commit")
            if not isinstance(git_commit, str):
                raise ValueError("task git commit is invalid")
            git_commits.add(git_commit)
            numerical_runtime = execution.get("numerical_runtime")
            runtime_payloads.add(_canonical_json_bytes(numerical_runtime))
            methods = cast(list[dict[str, object]], actual["methods"])
            for method in methods:
                long_rows.append(
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
            diagnostic_rows.append(
                {
                    "assay_id": assay_id,
                    "diagnostics": actual["diagnostics"],
                    "seed": seed,
                    "task_artifact_sha256": task_sha,
                }
            )
            task_manifest.append(
                {
                    "assay_id": assay_id,
                    "seed": seed,
                    "sha256": task_sha,
                }
            )
    if len(git_commits) != 1 or len(runtime_payloads) != 1:
        raise ValueError("task grid mixes implementations or numerical runtimes")
    results = pd.DataFrame(long_rows)
    if mode == "confirmatory":
        validated = validate_v0_result_table(
            results,
            assay_ids=assay_ids,
            protocol=protocol,
            require_no_hessian=True,
        )
    else:
        validated = validate_result_table(
            results,
            assay_ids=assay_ids,
            seeds=requested_seeds,
            required_methods=METHOD_NAMES,
        )
    comparisons = (
        ("ours", "supervised"),
        ("ours", "random"),
        ("ours", "top_teacher"),
        ("ours", "no_hessian"),
    )
    methods_table = method_summary_table(validated)
    effects_table = comparison_summary_table(
        validated,
        comparisons=comparisons,
        analysis_seed=protocol.analysis_seed,
        n_resamples=bootstrap_resamples,
    )
    verdict = v0_analysis_verdict(validated) if mode == "confirmatory" else None
    payload: dict[str, object] = {
        "analysis": {
            "analysis_seed": protocol.analysis_seed,
            "bootstrap_resamples": bootstrap_resamples,
            "inference_unit": "assay",
            "sign_flip": "exact",
        },
        "diagnostics": diagnostic_rows,
        "effect_table": effects_table.to_dict(orient="records"),
        "execution": {
            "bypass": (
                "non_official_test_or_development" if non_official_bypass else None
            ),
            "numerical_policy": NUMERICAL_POLICY,
            "numerical_runtime": numerical_runtime,
            "official": not non_official_bypass,
        },
        "grid": {
            "assay_ids": list(assay_ids),
            "seeds": list(requested_seeds),
        },
        "kind": "aggregate_result",
        "long_results": validated.to_dict(orient="records"),
        "method_table": methods_table.to_dict(orient="records"),
        "mode": mode,
        "provenance": {
            "git_commit": git_commits.pop(),
            "manifest_sha256": sha256_file(manifest_path),
            "protocol_digest": canonical_protocol_digest(protocol),
            "r5_gate_sha256": r5_gate_sha256,
            "task_count": len(task_manifest),
            "task_manifest": task_manifest,
        },
        "schema_version": 1,
        "v0_verdict": verdict,
    }
    return cast(dict[str, object], _jsonable(payload))


def _validate_r5_teacher_diagnostics(
    protocol: Protocol,
    aggregate_payload: dict[str, object],
) -> None:
    diagnostics = aggregate_payload.get("diagnostics")
    if not isinstance(diagnostics, list) or len(diagnostics) != 2:
        raise ValueError("R5 evidence must contain two task diagnostics")
    expected_counts = {
        "teacher_scores_labeled": protocol.n_labeled,
        "teacher_scores_unlabeled": protocol.n_unlabeled,
        "teacher_scores_test": protocol.n_test,
    }
    for task_row in diagnostics:
        if not isinstance(task_row, dict) or not isinstance(
            task_row.get("diagnostics"), dict
        ):
            raise ValueError("R5 task diagnostics are invalid")
        task_diagnostics = cast(dict[str, object], task_row["diagnostics"])
        fit = task_diagnostics.get("fit")
        evaluation = task_diagnostics.get("evaluation")
        if not isinstance(fit, dict) or not isinstance(evaluation, dict):
            raise ValueError("R5 fit or evaluation diagnostics are missing")
        for field, expected_count in expected_counts.items():
            scores = fit.get(field)
            if not isinstance(scores, dict):
                raise ValueError(f"R5 {field} diagnostics are missing")
            variance = scores.get("variance")
            if (
                scores.get("count") != expected_count
                or scores.get("finite_count") != expected_count
                or scores.get("finite_fraction") != 1.0
                or isinstance(variance, bool)
                or not isinstance(variance, (int, float))
                or not np.isfinite(float(variance))
                or float(variance) <= 0.0
            ):
                raise ValueError(
                    f"R5 {field} must have full finite coverage and positive variance"
                )
        slope = fit.get("calibration_slope")
        teacher_spearman = evaluation.get("teacher_test_spearman")
        varying_slope = (
            not isinstance(slope, bool)
            and isinstance(slope, (int, float))
            and np.isfinite(float(slope))
            and float(slope) != 0.0
        )
        defined_spearman = (
            isinstance(teacher_spearman, dict)
            and teacher_spearman.get("defined") is True
            and isinstance(teacher_spearman.get("value"), (int, float))
            and not isinstance(teacher_spearman.get("value"), bool)
            and np.isfinite(float(cast(float, teacher_spearman["value"])))
        )
        if not (varying_slope or defined_spearman):
            raise ValueError("R5 calibrated teacher must have nonzero variation")


def _validated_pilot_note(
    pilot_note_path: Path,
    *,
    protocol_digest: object,
    manifest_sha256: object,
    git_commit: object,
) -> dict[str, object]:
    payload = _load_unique_json(pilot_note_path)
    required_keys = {
        "caveats",
        "classification",
        "development_only",
        "direction",
        "git_commit",
        "kind",
        "manifest_sha256",
        "protocol_digest",
        "schema_version",
        "status",
    }
    caveats = payload.get("caveats")
    if (
        set(payload) != required_keys
        or payload.get("schema_version") != 1
        or payload.get("kind") != "r5_pilot_note"
        or payload.get("development_only") is not True
        or payload.get("status") != "reviewed"
        or payload.get("direction") != "continue"
        or payload.get("classification") not in {"positive", "negative", "ambiguous"}
        or not isinstance(caveats, list)
        or not caveats
        or any(not isinstance(value, str) or not value.strip() for value in caveats)
        or payload.get("protocol_digest") != protocol_digest
        or payload.get("manifest_sha256") != manifest_sha256
        or payload.get("git_commit") != git_commit
    ):
        raise ValueError("pilot note schema or bound identifiers are invalid")
    return {
        "path": str(pilot_note_path.resolve()),
        "sha256": sha256_file(pilot_note_path),
    }


def _build_r5_gate_payload(
    *,
    protocol: Protocol,
    manifest: DataManifest,
    manifest_path: Path,
    processed_root: Path,
    embedding_root: Path,
    results_root: Path,
    aggregate_artifact: Path,
    pilot_note_path: Path,
) -> dict[str, object]:
    """Reconstruct the exact official R5 evidence and derive its promotion gate."""
    actual = _load_unique_json(aggregate_artifact)
    if (
        actual.get("schema_version") != 1
        or actual.get("kind") != "aggregate_result"
        or actual.get("mode") != "development"
    ):
        raise ValueError("R5 aggregate must be a development aggregate artifact")
    analysis = actual.get("analysis")
    grid = actual.get("grid")
    execution = actual.get("execution")
    provenance = actual.get("provenance")
    if (
        not isinstance(analysis, dict)
        or analysis.get("bootstrap_resamples") != 10_000
        or not isinstance(grid, dict)
        or grid.get("assay_ids") != [manifest.development_id]
        or grid.get("seeds") != [0, 1]
        or not isinstance(execution, dict)
        or execution.get("official") is not True
        or execution.get("bypass") is not None
        or not isinstance(provenance, dict)
    ):
        raise ValueError("R5 aggregate is not the exact official two-seed grid")
    expected = _build_aggregate_payload(
        protocol=protocol,
        manifest=manifest,
        manifest_path=manifest_path,
        processed_root=processed_root,
        embedding_root=embedding_root,
        results_root=results_root,
        mode="development",
        seeds=(0, 1),
        bootstrap_resamples=10_000,
        non_official_bypass=False,
        r5_gate_sha256=None,
    )
    _require_exact_payload(actual, expected, artifact_kind="R5 aggregate")
    _validate_r5_teacher_diagnostics(protocol, expected)
    long_results = expected.get("long_results")
    task_manifest = provenance.get("task_manifest")
    if (
        not isinstance(long_results, list)
        or len(long_results) != 10
        or provenance.get("task_count") != 2
        or not isinstance(task_manifest, list)
        or len(task_manifest) != 2
        or [entry.get("seed") for entry in task_manifest if isinstance(entry, dict)]
        != [0, 1]
    ):
        raise ValueError("R5 evidence must contain two tasks and ten method rows")
    pilot_note = _validated_pilot_note(
        pilot_note_path,
        protocol_digest=provenance["protocol_digest"],
        manifest_sha256=provenance["manifest_sha256"],
        git_commit=provenance["git_commit"],
    )
    payload: dict[str, object] = {
        "aggregate": {
            "path": str(aggregate_artifact.resolve()),
            "sha256": sha256_file(aggregate_artifact),
        },
        "git_commit": provenance["git_commit"],
        "kind": "r5_gate",
        "manifest_sha256": provenance["manifest_sha256"],
        "method_row_count": 10,
        "numerical_runtime": execution["numerical_runtime"],
        "pilot_note": pilot_note,
        "pilot_results_root": str(results_root.resolve()),
        "protocol_digest": provenance["protocol_digest"],
        "schema_version": 1,
        "status": "passed",
        "task_count": 2,
        "task_manifest": task_manifest,
    }
    return cast(dict[str, object], _jsonable(payload))


def _validated_r5_gate_digest(
    gate_path: Path,
    *,
    protocol: Protocol,
    manifest: DataManifest,
    manifest_path: Path,
    processed_root: Path,
    embedding_root: Path,
    results_root: Path,
) -> str:
    """Reload and independently reconstruct every artifact bound by an R5 gate."""
    actual = _load_unique_json(gate_path)
    aggregate = actual.get("aggregate")
    pilot_note = actual.get("pilot_note")
    pilot_results_root_raw = actual.get("pilot_results_root")
    if (
        not isinstance(aggregate, dict)
        or not isinstance(aggregate.get("path"), str)
        or not isinstance(pilot_note, dict)
        or not isinstance(pilot_note.get("path"), str)
        or not isinstance(pilot_results_root_raw, str)
    ):
        raise ValueError("R5 gate does not identify its aggregate evidence")
    aggregate_path = Path(cast(str, aggregate["path"]))
    if not aggregate_path.is_absolute():
        raise ValueError("R5 aggregate evidence path must be absolute")
    pilot_note_path = Path(cast(str, pilot_note["path"]))
    if not pilot_note_path.is_absolute():
        raise ValueError("R5 pilot note path must be absolute")
    pilot_results_root = Path(pilot_results_root_raw)
    if (
        not pilot_results_root.is_absolute()
        or pilot_results_root.resolve() != pilot_results_root
    ):
        raise ValueError("R5 pilot results root must be canonical and absolute")
    expected = _build_r5_gate_payload(
        protocol=protocol,
        manifest=manifest,
        manifest_path=manifest_path,
        processed_root=processed_root,
        embedding_root=embedding_root,
        results_root=pilot_results_root,
        aggregate_artifact=aggregate_path,
        pilot_note_path=pilot_note_path,
    )
    _require_exact_payload(actual, expected, artifact_kind="R5 gate")
    return sha256_file(gate_path)


@app.command("aggregate")
def aggregate(
    context: typer.Context,
    manifest_path: Annotated[Path, typer.Option("--manifest")],
    processed_root: Annotated[Path, typer.Option("--processed-root")],
    embedding_root: Annotated[Path, typer.Option("--embedding-root")],
    results_root: Annotated[Path, typer.Option("--results-root")],
    output: Annotated[Path, typer.Option("--output")],
    mode: Annotated[
        Literal["confirmatory", "development"],
        typer.Option("--mode"),
    ],
    seeds: Annotated[list[int] | None, typer.Option("--seed")] = None,
    bootstrap_resamples: Annotated[
        int,
        typer.Option("--bootstrap-resamples", min=1),
    ] = 10_000,
    non_official_bypass: Annotated[
        bool,
        typer.Option("--non-official-bypass"),
    ] = False,
    r5_gate: Annotated[Path | None, typer.Option("--r5-gate")] = None,
) -> None:
    """Rebuild an exact task grid and emit clustered tables and diagnostics."""
    options = _options(context)

    def action() -> dict[str, object]:
        protocol = load_protocol(options.config)
        manifest = load_data_manifest(manifest_path)
        _validate_manifest(protocol, manifest)
        requested_seeds = tuple(protocol.seeds if seeds is None else seeds)
        _validate_execution_policy(
            protocol,
            mode=mode,
            non_official_bypass=non_official_bypass,
        )
        if mode == "confirmatory":
            if r5_gate is None:
                raise ValueError("confirmatory aggregation requires --r5-gate")
            gate_digest = _validated_r5_gate_digest(
                r5_gate,
                protocol=protocol,
                manifest=manifest,
                manifest_path=manifest_path,
                processed_root=processed_root,
                embedding_root=embedding_root,
                results_root=results_root,
            )
        else:
            if r5_gate is not None:
                raise ValueError("development aggregation cannot consume --r5-gate")
            gate_digest = None
        _validated_aggregate_seeds(protocol, requested_seeds, mode=mode)
        assay_ids = (
            manifest.confirmatory_ids
            if mode == "confirmatory"
            else (manifest.development_id,)
        )
        expected_paths = [
            results_root / "tasks" / assay / f"seed_{seed}.json"
            for assay in assay_ids
            for seed in requested_seeds
        ]
        if options.dry_run:
            return {
                "plan": {
                    "assay_ids": list(assay_ids),
                    "mode": mode,
                    "output": str(output),
                    "r5_gate": str(r5_gate) if r5_gate is not None else None,
                    "seeds": list(requested_seeds),
                    "task_artifacts": [str(path) for path in expected_paths],
                }
            }
        payload = _build_aggregate_payload(
            protocol=protocol,
            manifest=manifest,
            manifest_path=manifest_path,
            processed_root=processed_root,
            embedding_root=embedding_root,
            results_root=results_root,
            mode=mode,
            seeds=requested_seeds,
            bootstrap_resamples=bootstrap_resamples,
            non_official_bypass=non_official_bypass,
            r5_gate_sha256=gate_digest,
        )
        created = _write_json_once(output, payload)
        long_results = cast(list[object], payload["long_results"])
        return {
            "artifact": str(output),
            "artifact_sha256": sha256_file(output),
            "created": created,
            "mode": mode,
            "row_count": len(long_results),
        }

    _run_command("aggregate", dry_run=options.dry_run, action=action)


def _verify_task_context(
    path: Path,
    *,
    protocol: Protocol,
    manifest: DataManifest,
    manifest_path: Path,
    processed_root: Path,
    embedding_root: Path,
    non_official_bypass: bool,
    r5_gate_sha256: str | None,
) -> None:
    actual = _load_unique_json(path)
    _validate_task_payload(actual)
    assay_id, seed, raw_mode = _task_identity(actual)
    if raw_mode not in ("confirmatory", "development"):
        raise ValueError("task mode is invalid")
    mode = cast(Literal["confirmatory", "development"], raw_mode)
    expected = _build_task_payload(
        protocol=protocol,
        manifest=manifest,
        manifest_path=manifest_path,
        processed_root=processed_root,
        embedding_root=embedding_root,
        assay_id=assay_id,
        seed=seed,
        mode=mode,
        non_official_bypass=non_official_bypass,
        r5_gate_sha256=r5_gate_sha256,
    )
    _require_exact_payload(actual, expected, artifact_kind="task")


@app.command("verify")
def verify(
    context: typer.Context,
    manifest_path: Annotated[Path | None, typer.Option("--manifest")] = None,
    processed_root: Annotated[Path | None, typer.Option("--processed-root")] = None,
    embedding_root: Annotated[Path | None, typer.Option("--embedding-root")] = None,
    results_root: Annotated[Path | None, typer.Option("--results-root")] = None,
    task_artifact: Annotated[Path | None, typer.Option("--task-artifact")] = None,
    aggregate_artifact: Annotated[
        Path | None,
        typer.Option("--aggregate-artifact"),
    ] = None,
    r5_gate: Annotated[Path | None, typer.Option("--r5-gate")] = None,
    write_r5_gate: Annotated[
        Path | None,
        typer.Option("--write-r5-gate"),
    ] = None,
    pilot_note: Annotated[Path | None, typer.Option("--pilot-note")] = None,
    runtime_only: Annotated[bool, typer.Option("--runtime-only")] = False,
    non_official_bypass: Annotated[
        bool,
        typer.Option("--non-official-bypass"),
    ] = False,
) -> None:
    """Fail closed on protocol, manifest, cache, task, or aggregate corruption."""
    options = _options(context)

    def action() -> dict[str, object]:
        if runtime_only:
            if any(
                value is not None
                for value in (
                    manifest_path,
                    processed_root,
                    embedding_root,
                    results_root,
                    task_artifact,
                    aggregate_artifact,
                    r5_gate,
                    write_r5_gate,
                    pilot_note,
                )
            ):
                raise ValueError(
                    "--runtime-only cannot be combined with artifact inputs"
                )
            if options.dry_run:
                return {"plan": {"stages": ["require_openblas_coretype"]}}
            active = require_openblas_coretype("Haswell")
            return {"openblas_coretype": active, "verified": ["runtime"]}
        if manifest_path is None:
            raise ValueError("--manifest is required unless --runtime-only is used")
        plan = {
            "aggregate_artifact": str(aggregate_artifact)
            if aggregate_artifact
            else None,
            "embedding_root": str(embedding_root) if embedding_root else None,
            "manifest": str(manifest_path),
            "pilot_note": str(pilot_note) if pilot_note else None,
            "processed_root": str(processed_root) if processed_root else None,
            "r5_gate": str(r5_gate) if r5_gate else None,
            "results_root": str(results_root) if results_root else None,
            "task_artifact": str(task_artifact) if task_artifact else None,
            "write_r5_gate": str(write_r5_gate) if write_r5_gate else None,
        }
        if options.dry_run:
            return {"plan": plan}
        protocol = load_protocol(options.config)
        manifest = load_data_manifest(manifest_path)
        _validate_manifest(protocol, manifest)
        _validate_execution_policy(
            protocol,
            mode="development",
            non_official_bypass=non_official_bypass,
        )
        if not non_official_bypass:
            require_openblas_coretype("Haswell")
        verified = ["config", "manifest"]
        if r5_gate is not None and write_r5_gate is not None:
            raise ValueError("--r5-gate and --write-r5-gate are mutually exclusive")
        if write_r5_gate is not None and pilot_note is None:
            raise ValueError("R5 gate generation requires --pilot-note")
        if write_r5_gate is None and pilot_note is not None:
            raise ValueError("--pilot-note is only valid with --write-r5-gate")
        if r5_gate is not None:
            if processed_root is None or embedding_root is None or results_root is None:
                raise ValueError(
                    "R5 gate verification requires --processed-root, "
                    "--embedding-root, and --results-root"
                )
            gate_digest = _validated_r5_gate_digest(
                r5_gate,
                protocol=protocol,
                manifest=manifest,
                manifest_path=manifest_path,
                processed_root=processed_root,
                embedding_root=embedding_root,
                results_root=results_root,
            )
            verified.append("r5_gate")
        else:
            gate_digest = None
        for record in manifest.selected_assays:
            if processed_root is not None:
                _load_identity_frame(
                    processed_root / f"{record.dms_id}.parquet",
                    protocol=protocol,
                    record=record,
                )
            if embedding_root is not None:
                metadata_path = embedding_root / f"{record.dms_id}.json"
                load_embedding_cache(
                    embedding_root / f"{record.dms_id}.npy",
                    metadata_path,
                    dms_id=record.dms_id,
                    row_hashes=record.row_hashes,
                    model_id=protocol.model,
                    model_revision=protocol.model_revision,
                    sources=_embedding_sources(protocol),
                    expected_embedding_dim=(
                        _cache_width(metadata_path)
                        if non_official_bypass
                        else _ESM2_35M_EMBEDDING_DIM
                    ),
                )
        if processed_root is not None:
            verified.append("processed")
        if embedding_root is not None:
            verified.append("embeddings")
        if task_artifact is not None:
            if processed_root is None or embedding_root is None:
                raise ValueError(
                    "task verification requires --processed-root and --embedding-root"
                )
            _verify_task_context(
                task_artifact,
                protocol=protocol,
                manifest=manifest,
                manifest_path=manifest_path,
                processed_root=processed_root,
                embedding_root=embedding_root,
                non_official_bypass=non_official_bypass,
                r5_gate_sha256=gate_digest,
            )
            verified.append("task")
        if aggregate_artifact is not None:
            if processed_root is None or embedding_root is None or results_root is None:
                raise ValueError(
                    "aggregate verification requires --processed-root, "
                    "--embedding-root, and --results-root"
                )
            aggregate_payload = _load_unique_json(aggregate_artifact)
            if (
                aggregate_payload.get("schema_version") != 1
                or aggregate_payload.get("kind") != "aggregate_result"
            ):
                raise ValueError("aggregate artifact schema is invalid")
            raw_mode = aggregate_payload.get("mode")
            grid = aggregate_payload.get("grid")
            analysis = aggregate_payload.get("analysis")
            if (
                raw_mode not in ("confirmatory", "development")
                or not isinstance(grid, dict)
                or not isinstance(grid.get("seeds"), list)
                or not isinstance(analysis, dict)
                or type(analysis.get("bootstrap_resamples")) is not int
            ):
                raise ValueError("aggregate reconstruction metadata is invalid")
            aggregate_mode: Literal["confirmatory", "development"] = (
                "confirmatory" if raw_mode == "confirmatory" else "development"
            )
            aggregate_seeds = cast(list[object], grid["seeds"])
            if any(type(seed) is not int for seed in aggregate_seeds):
                raise ValueError("aggregate seed grid is invalid")
            expected_aggregate = _build_aggregate_payload(
                protocol=protocol,
                manifest=manifest,
                manifest_path=manifest_path,
                processed_root=processed_root,
                embedding_root=embedding_root,
                results_root=results_root,
                mode=aggregate_mode,
                seeds=cast(list[int], aggregate_seeds),
                bootstrap_resamples=cast(int, analysis["bootstrap_resamples"]),
                non_official_bypass=non_official_bypass,
                r5_gate_sha256=gate_digest,
            )
            _require_exact_payload(
                aggregate_payload,
                expected_aggregate,
                artifact_kind="aggregate",
            )
            verified.append("aggregate")
        terminal: dict[str, object] = {"verified": verified}
        if write_r5_gate is not None:
            if non_official_bypass:
                raise ValueError("R5 gate generation forbids --non-official-bypass")
            if (
                aggregate_artifact is None
                or processed_root is None
                or embedding_root is None
                or results_root is None
            ):
                raise ValueError(
                    "R5 gate generation requires --aggregate-artifact, "
                    "--processed-root, --embedding-root, and --results-root"
                )
            gate_payload = _build_r5_gate_payload(
                protocol=protocol,
                manifest=manifest,
                manifest_path=manifest_path,
                processed_root=processed_root,
                embedding_root=embedding_root,
                results_root=results_root,
                aggregate_artifact=aggregate_artifact,
                pilot_note_path=cast(Path, pilot_note),
            )
            created = _write_json_once(write_r5_gate, gate_payload)
            terminal.update(
                r5_gate_artifact=str(write_r5_gate),
                r5_gate_created=created,
                r5_gate_sha256=sha256_file(write_r5_gate),
            )
        return terminal

    _run_command("verify", dry_run=options.dry_run, action=action)


def _command_from_argv(arguments: Sequence[str]) -> str | None:
    commands = {"prepare-data", "embed-assay", "run-task", "aggregate", "verify"}
    index = 0
    while index < len(arguments):
        argument = arguments[index]
        if argument == "--config":
            index += 2
            continue
        if argument.startswith("--config=") or argument in {
            "--dry-run",
            "--show-config",
            "--help",
        }:
            index += 1
            continue
        return argument if argument in commands else None
    return None


def cli_main() -> None:
    """Run Typer while normalizing parse-time usage failures to JSON."""
    try:
        exit_code = app(standalone_mode=False)
        if type(exit_code) is int:
            raise SystemExit(exit_code)
    except _click.ClickException as error:
        _emit(
            {
                "command": _command_from_argv(sys.argv[1:]),
                "error": error.format_message(),
                "error_type": type(error).__name__,
                "event": "terminal",
                "numerical_runtime": current_numerical_runtime_fingerprint(),
                "schema_version": _SCHEMA_VERSION,
                "status": "error",
            }
        )
        raise SystemExit(error.exit_code) from error
    except _click.exceptions.Exit as error:
        raise SystemExit(error.exit_code) from error


if __name__ == "__main__":
    cli_main()
