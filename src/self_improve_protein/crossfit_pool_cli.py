"""Outcome-blind preparation and embedding CLI for untouched crossfit pools."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import typer

from self_improve_protein.cli import (
    _ESM2_35M_EMBEDDING_DIM,
    _LOCKED_V0_PROTOCOL_DIGEST,
    _cache_width,
    _embedding_sources,
    _emit,
    _load_identity_frame,
    _run_command,
    _write_json_once,
    _write_parquet_once,
)
from self_improve_protein.config import Protocol, load_protocol
from self_improve_protein.crossfit_data import (
    CrossfitPoolManifest,
    build_crossfit_pool,
    load_crossfit_pool_manifest,
    validate_crossfit_pool_provenance,
)
from self_improve_protein.data import SelectedAssayManifest
from self_improve_protein.embeddings import (
    get_or_create_embedding_cache,
    load_embedding_cache,
)
from self_improve_protein.experiment import canonical_protocol_digest
from self_improve_protein.provenance import sha256_file

app = typer.Typer(
    name="self-improve-protein-crossfit-pool",
    help="Prepare and verify the outcome-blind untouched crossfit assay pool.",
    invoke_without_command=True,
    no_args_is_help=False,
    pretty_exceptions_enable=False,
)


@dataclass(frozen=True, slots=True)
class _RuntimeOptions:
    config: Path
    dry_run: bool


@app.callback()
def main(
    context: typer.Context,
    config: Annotated[
        Path,
        typer.Option("--config", help="Validated protocol YAML for every stage."),
    ] = Path("configs/v0.yaml"),
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Render the exact plan without writes."),
    ] = False,
    show_config: Annotated[
        bool,
        typer.Option("--show-config", help="Print the validated protocol and exit."),
    ] = False,
) -> None:
    """Set the shared protocol and dry-run policy for all pool commands."""
    context.obj = _RuntimeOptions(config=config, dry_run=dry_run)
    if show_config:
        protocol = load_protocol(config)
        _emit(
            {
                "event": "config",
                "protocol": protocol.model_dump(mode="json"),
                "protocol_digest": canonical_protocol_digest(protocol),
                "schema_version": 1,
            }
        )
        raise typer.Exit()
    if context.invoked_subcommand is None:
        typer.echo(context.get_help())
        raise typer.Exit()


def _options(context: typer.Context) -> _RuntimeOptions:
    if not isinstance(context.obj, _RuntimeOptions):
        raise RuntimeError("CLI runtime options were not initialized")
    return context.obj


def _validate_protocol_policy(
    protocol: Protocol,
    *,
    non_official_bypass: bool,
) -> None:
    if (
        not non_official_bypass
        and canonical_protocol_digest(protocol) != _LOCKED_V0_PROTOCOL_DIGEST
    ):
        raise ValueError("official execution requires the locked v0 protocol digest")


def _canonical_manifest_bytes(manifest: CrossfitPoolManifest) -> bytes:
    return (
        json.dumps(
            manifest.model_dump(mode="json"),
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode()


def _load_validated_manifest(
    path: Path,
    *,
    protocol: Protocol,
    base_manifest_path: Path,
) -> CrossfitPoolManifest:
    manifest = load_crossfit_pool_manifest(path)
    if path.read_bytes() != _canonical_manifest_bytes(manifest):
        raise ValueError("crossfit pool manifest must use canonical JSON bytes")
    validate_crossfit_pool_provenance(
        manifest,
        protocol=protocol,
        base_manifest_path=base_manifest_path,
    )
    return manifest


def _records_by_id(
    manifest: CrossfitPoolManifest,
) -> dict[str, SelectedAssayManifest]:
    records = {record.dms_id: record for record in manifest.selected_assays}
    if len(records) != len(manifest.selected_assays):
        raise ValueError("crossfit pool selected-assay records are not unique")
    return records


def _resolve_untouched_assay(
    manifest: CrossfitPoolManifest,
    *,
    assay_id: str | None,
    assay_index: int | None,
) -> str:
    if (assay_id is None) == (assay_index is None):
        raise ValueError("provide exactly one of --assay-id or --assay-index")
    if assay_id is not None:
        if assay_id not in manifest.untouched_ids:
            raise ValueError("assay ID is not a member of the untouched pool")
        return assay_id
    assert assay_index is not None
    if assay_index < 0 or assay_index >= len(manifest.untouched_ids):
        raise ValueError("assay index is outside the 26-assay untouched pool")
    return manifest.untouched_ids[assay_index]


@app.command("prepare-pool")
def prepare_pool(
    context: typer.Context,
    base_manifest: Annotated[Path, typer.Option("--base-manifest")],
    pool_manifest: Annotated[Path, typer.Option("--pool-manifest")],
    dms_zip: Annotated[Path, typer.Option("--dms-zip")],
    scores_zip: Annotated[Path, typer.Option("--scores-zip")],
    metadata_csv: Annotated[Path, typer.Option("--metadata-csv")],
    processed_root: Annotated[Path, typer.Option("--processed-root")],
    non_official_bypass: Annotated[
        bool,
        typer.Option("--non-official-bypass"),
    ] = False,
) -> None:
    """Rebuild all pools, write the 26 untouched frames, then freeze the manifest."""
    options = _options(context)

    def action() -> dict[str, object]:
        protocol = load_protocol(options.config)
        _validate_protocol_policy(
            protocol,
            non_official_bypass=non_official_bypass,
        )
        plan = {
            "base_manifest": str(base_manifest),
            "pool_manifest": str(pool_manifest),
            "processed_root": str(processed_root),
            "source_paths": [str(dms_zip), str(scores_zip), str(metadata_csv)],
            "stages": [
                "verify_sources_and_base",
                "rebuild_35_working_sets",
                "write_26_untouched_frames",
                "freeze_canonical_pool_manifest",
            ],
        }
        if options.dry_run:
            return {"official": not non_official_bypass, "plan": plan}

        emitted_ids: list[str] = []
        processed_created = 0

        def write_frame(dms_id: str, frame: object) -> None:
            nonlocal processed_created
            if not hasattr(frame, "to_parquet"):
                raise TypeError("pool builder emitted a non-DataFrame value")
            emitted_ids.append(dms_id)
            processed_created += int(
                _write_parquet_once(
                    processed_root / f"{dms_id}.parquet",
                    frame,
                )
            )

        manifest, frames = build_crossfit_pool(
            protocol=protocol,
            base_manifest_path=base_manifest,
            dms_zip=dms_zip,
            scores_zip=scores_zip,
            metadata_csv=metadata_csv,
            on_untouched_frame=write_frame,
        )
        if tuple(emitted_ids) != manifest.untouched_ids:
            raise ValueError(
                "pool builder did not emit the exact untouched assay order"
            )
        if tuple(frames) != manifest.untouched_ids:
            raise ValueError(
                "pool builder did not return the exact untouched frame map"
            )
        manifest_created = _write_json_once(
            pool_manifest,
            manifest.model_dump(mode="json"),
        )
        reloaded = _load_validated_manifest(
            pool_manifest,
            protocol=protocol,
            base_manifest_path=base_manifest,
        )
        if reloaded != manifest:
            raise ValueError("reloaded crossfit pool manifest does not match builder")
        return {
            "manifest_created": manifest_created,
            "manifest_sha256": sha256_file(pool_manifest),
            "official": not non_official_bypass,
            "processed_created": processed_created,
            "untouched_count": len(manifest.untouched_ids),
        }

    _run_command("crossfit-prepare-pool", dry_run=options.dry_run, action=action)


@app.command("embed-assay")
def embed_assay(
    context: typer.Context,
    base_manifest: Annotated[Path, typer.Option("--base-manifest")],
    pool_manifest: Annotated[Path, typer.Option("--pool-manifest")],
    processed_root: Annotated[Path, typer.Option("--processed-root")],
    embedding_root: Annotated[Path, typer.Option("--embedding-root")],
    assay_id: Annotated[str | None, typer.Option("--assay-id")] = None,
    assay_index: Annotated[int | None, typer.Option("--assay-index")] = None,
    batch_size: Annotated[int, typer.Option("--batch-size", min=1)] = 128,
    device: Annotated[str, typer.Option("--device")] = "cuda",
    non_official_bypass: Annotated[
        bool,
        typer.Option("--non-official-bypass"),
    ] = False,
) -> None:
    """Create or validate one pinned ESM-2 cache for an untouched assay."""
    options = _options(context)

    def action() -> dict[str, object]:
        protocol = load_protocol(options.config)
        _validate_protocol_policy(
            protocol,
            non_official_bypass=non_official_bypass,
        )
        manifest = _load_validated_manifest(
            pool_manifest,
            protocol=protocol,
            base_manifest_path=base_manifest,
        )
        resolved = _resolve_untouched_assay(
            manifest,
            assay_id=assay_id,
            assay_index=assay_index,
        )
        plan = {
            "assay_id": resolved,
            "embedding_root": str(embedding_root),
            "processed_root": str(processed_root),
            "stages": [
                "validate_outcome_free_identity",
                "embed_pinned_esm2",
                "validate_cache",
            ],
        }
        if options.dry_run:
            return {"official": not non_official_bypass, "plan": plan}

        record = _records_by_id(manifest)[resolved]
        frame = _load_identity_frame(
            processed_root / f"{resolved}.parquet",
            protocol=protocol,
            record=record,
        )
        npy_path = embedding_root / f"{resolved}.npy"
        metadata_path = embedding_root / f"{resolved}.json"
        cache_hit = npy_path.exists() and metadata_path.exists()
        embeddings = get_or_create_embedding_cache(
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
            "official": not non_official_bypass,
            "shape": list(embeddings.shape),
        }

    _run_command("crossfit-embed-assay", dry_run=options.dry_run, action=action)


@app.command("verify")
def verify(
    context: typer.Context,
    base_manifest: Annotated[Path, typer.Option("--base-manifest")],
    pool_manifest: Annotated[Path, typer.Option("--pool-manifest")],
    processed_root: Annotated[Path, typer.Option("--processed-root")],
    embedding_root: Annotated[Path | None, typer.Option("--embedding-root")] = None,
    non_official_bypass: Annotated[
        bool,
        typer.Option("--non-official-bypass"),
    ] = False,
) -> None:
    """Fail closed unless every requested untouched-pool artifact validates."""
    options = _options(context)

    def action() -> dict[str, object]:
        protocol = load_protocol(options.config)
        _validate_protocol_policy(
            protocol,
            non_official_bypass=non_official_bypass,
        )
        manifest = _load_validated_manifest(
            pool_manifest,
            protocol=protocol,
            base_manifest_path=base_manifest,
        )
        plan = {
            "embedding_root": str(embedding_root) if embedding_root else None,
            "pool_manifest": str(pool_manifest),
            "processed_root": str(processed_root),
            "untouched_count": len(manifest.untouched_ids),
        }
        if options.dry_run:
            return {"official": not non_official_bypass, "plan": plan}

        records = _records_by_id(manifest)
        processed_verified = 0
        embeddings_verified = 0
        for dms_id in manifest.untouched_ids:
            record = records[dms_id]
            _load_identity_frame(
                processed_root / f"{dms_id}.parquet",
                protocol=protocol,
                record=record,
            )
            processed_verified += 1
            if embedding_root is not None:
                metadata_path = embedding_root / f"{dms_id}.json"
                expected_dim = (
                    _cache_width(metadata_path)
                    if non_official_bypass
                    else _ESM2_35M_EMBEDDING_DIM
                )
                load_embedding_cache(
                    embedding_root / f"{dms_id}.npy",
                    metadata_path,
                    dms_id=dms_id,
                    row_hashes=record.row_hashes,
                    model_id=protocol.model,
                    model_revision=protocol.model_revision,
                    sources=_embedding_sources(protocol),
                    expected_embedding_dim=expected_dim,
                )
                embeddings_verified += 1
        return {
            "embeddings_verified": embeddings_verified,
            "manifest_sha256": sha256_file(pool_manifest),
            "official": not non_official_bypass,
            "processed_verified": processed_verified,
            "untouched_count": len(manifest.untouched_ids),
        }

    _run_command("crossfit-pool-verify", dry_run=options.dry_run, action=action)


if __name__ == "__main__":
    app()
