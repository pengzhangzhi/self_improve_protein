"""Outcome-blind data-pool provenance for the cross-fitted repair card."""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from pathlib import Path
from typing import Final, Literal, Self

import pandas as pd  # type: ignore[import-untyped]
from pydantic import BaseModel, ConfigDict, Field, StrictStr, model_validator

from self_improve_protein.config import Protocol
from self_improve_protein.data import (
    DataManifest,
    ManifestSource,
    ManifestSources,
    SelectedAssayManifest,
    Sha256Hex,
    build_working_set,
    filter_usable_variants,
    load_assay_from_archives,
    load_data_manifest,
)
from self_improve_protein.provenance import (
    atomic_write_json,
    sha256_bytes,
    sha256_file,
)

CROSSFIT_POOL_SCHEMA_ID: Final = "self-improve-protein.crossfit-pool.v1"
CROSSFIT_CARD_ID: Final = "docs/research/experiment-card-crossfit.md"
CROSSFIT_CARD_SHA256: Final = (
    "383afd7a5bae9c2ebd6768a112a82980236540fc0f66e3a294ef298961b8596f"
)
_ELIGIBLE_ASSAY_COUNT = 35
_SCREEN_ASSAY_COUNT = 9
_UNTOUCHED_ASSAY_COUNT = 26

UntouchedFrameCallback = Callable[[str, pd.DataFrame], None]


def _canonical_json_bytes(payload: object) -> bytes:
    serialized = json.dumps(
        payload,
        allow_nan=False,
        indent=2,
        sort_keys=True,
    ) + "\n"
    return serialized.encode("utf-8")


def _protocol_sha256(protocol: Protocol) -> str:
    return sha256_bytes(_canonical_json_bytes(protocol.model_dump(mode="json")))


class _FrozenManifestModel(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        allow_inf_nan=False,
    )


class CrossfitPoolManifest(_FrozenManifestModel):
    """Exact provenance and row identities for the 35-assay crossfit pool."""

    schema_id: Literal["self-improve-protein.crossfit-pool.v1"]
    schema_version: int = Field(ge=1, le=1, strict=True)
    card_id: Literal["docs/research/experiment-card-crossfit.md"]
    card_sha256: Literal[
        "383afd7a5bae9c2ebd6768a112a82980236540fc0f66e3a294ef298961b8596f"
    ]
    base_manifest_sha256: Sha256Hex
    protocol_sha256: Sha256Hex
    data_release: str = Field(min_length=1, strict=True)
    teacher_column: str = Field(min_length=1, strict=True)
    sources: ManifestSources
    upstream_revision: str = Field(min_length=1, strict=True)
    max_length: int = Field(gt=0, le=512, strict=True)
    working_size: int = Field(gt=0, strict=True)
    eligible_assay_ids: tuple[StrictStr, ...] = Field(
        min_length=_ELIGIBLE_ASSAY_COUNT,
        max_length=_ELIGIBLE_ASSAY_COUNT,
    )
    screen_ids: tuple[StrictStr, ...] = Field(
        min_length=_SCREEN_ASSAY_COUNT,
        max_length=_SCREEN_ASSAY_COUNT,
    )
    untouched_ids: tuple[StrictStr, ...] = Field(
        min_length=_UNTOUCHED_ASSAY_COUNT,
        max_length=_UNTOUCHED_ASSAY_COUNT,
    )
    selected_assays: tuple[SelectedAssayManifest, ...] = Field(
        min_length=_ELIGIBLE_ASSAY_COUNT,
        max_length=_ELIGIBLE_ASSAY_COUNT,
    )

    @model_validator(mode="after")
    def validate_partition_and_records(self) -> Self:
        """Require the locked screen/untouched partition and all row records."""
        eligible = self.eligible_assay_ids
        if len(set(eligible)) != len(eligible) or eligible != tuple(sorted(eligible)):
            raise ValueError("eligible_assay_ids must be unique and lexically sorted")

        expected_screen = (eligible[8], *eligible[:8])
        if self.screen_ids != expected_screen:
            raise ValueError(
                "screen_ids must be development followed by the eight "
                "confirmatory IDs"
            )
        screen = set(self.screen_ids)
        expected_untouched = tuple(
            dms_id for dms_id in eligible if dms_id not in screen
        )
        if self.untouched_ids != expected_untouched:
            raise ValueError(
                "untouched_ids must preserve eligible_assay_ids order after "
                "subtracting screen_ids"
            )

        record_ids = tuple(record.dms_id for record in self.selected_assays)
        if record_ids != eligible:
            raise ValueError(
                "selected_assays coverage and order must match eligible_assay_ids"
            )
        all_hashes: list[str] = []
        for record in self.selected_assays:
            if record.usable_count < self.working_size:
                raise ValueError(
                    f"usable_count for {record.dms_id} must be at least working_size"
                )
            if record.sequence_length > self.max_length:
                raise ValueError(
                    f"sequence_length for {record.dms_id} exceeds max_length"
                )
            if len(record.row_hashes) != self.working_size:
                raise ValueError(
                    f"row-hash count for {record.dms_id} must equal working_size"
                )
            all_hashes.extend(record.row_hashes)
        if len(set(all_hashes)) != len(all_hashes):
            raise ValueError("row_hashes must be unique across selected_assays")
        return self


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def load_crossfit_pool_manifest(path: Path | str) -> CrossfitPoolManifest:
    """Load a crossfit pool manifest while rejecting duplicate JSON keys."""
    with Path(path).open(encoding="utf-8") as source:
        payload = json.load(source, object_pairs_hook=_unique_json_object)
    return CrossfitPoolManifest.model_validate(payload)


def write_crossfit_pool_manifest(
    path: Path | str,
    manifest: CrossfitPoolManifest,
) -> None:
    """Atomically write a validated manifest in canonical JSON form."""
    if not isinstance(manifest, CrossfitPoolManifest):
        raise TypeError("manifest must be a CrossfitPoolManifest")
    atomic_write_json(path, manifest.model_dump(mode="json"))


def _expected_sources(protocol: Protocol) -> ManifestSources:
    return ManifestSources(
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


def _load_canonical_base_manifest(path: Path | str) -> DataManifest:
    manifest_path = Path(path)
    manifest = load_data_manifest(manifest_path)
    canonical = _canonical_json_bytes(manifest.model_dump(mode="json"))
    if manifest_path.read_bytes() != canonical:
        raise ValueError("base manifest must use canonical JSON bytes")
    return manifest


def _validate_base_protocol_identity(
    base_manifest: DataManifest,
    protocol: Protocol,
) -> None:
    if base_manifest.sources != _expected_sources(protocol):
        raise ValueError("base manifest source identity does not match protocol")
    checks = (
        ("data release", base_manifest.data_release, protocol.data_release),
        ("teacher column", base_manifest.teacher_column, protocol.teacher_column),
        (
            "upstream revision",
            base_manifest.upstream_revision,
            protocol.proteingym_upstream_commit,
        ),
        ("max length", base_manifest.max_length, protocol.max_length),
        ("working size", base_manifest.working_size, protocol.working_size),
        (
            "confirmatory count",
            len(base_manifest.confirmatory_ids),
            protocol.assay_count,
        ),
    )
    mismatches = [name for name, actual, expected in checks if actual != expected]
    if mismatches:
        raise ValueError(
            "base manifest does not match protocol: " + ", ".join(mismatches)
        )
    if len(base_manifest.eligible_assay_ids) != _ELIGIBLE_ASSAY_COUNT:
        raise ValueError("base manifest must contain exactly 35 eligible assays")


def _validate_base_records(
    selected_assays: tuple[SelectedAssayManifest, ...],
    base_manifest: DataManifest,
) -> None:
    rebuilt = {record.dms_id: record for record in selected_assays}
    for expected in base_manifest.selected_assays:
        actual = rebuilt[expected.dms_id]
        if actual.row_hashes != expected.row_hashes:
            raise ValueError(
                f"rebuilt row hashes for {expected.dms_id} do not match base manifest"
            )
        if _canonical_json_bytes(actual.model_dump(mode="json")) != (
            _canonical_json_bytes(expected.model_dump(mode="json"))
        ):
            raise ValueError(
                f"rebuilt record bytes for {expected.dms_id} do not match base manifest"
            )


def validate_crossfit_pool_provenance(
    manifest: CrossfitPoolManifest,
    *,
    protocol: Protocol,
    base_manifest_path: Path | str,
) -> None:
    """Validate a manifest against the exact protocol and canonical v0 manifest."""
    if not isinstance(manifest, CrossfitPoolManifest):
        raise TypeError("manifest must be a CrossfitPoolManifest")
    if not isinstance(protocol, Protocol):
        raise TypeError("protocol must be a Protocol")
    base_path = Path(base_manifest_path)
    if manifest.base_manifest_sha256 != sha256_file(base_path):
        raise ValueError("base manifest SHA does not match crossfit provenance")
    base_manifest = _load_canonical_base_manifest(base_path)
    _validate_base_protocol_identity(base_manifest, protocol)
    if manifest.protocol_sha256 != _protocol_sha256(protocol):
        raise ValueError("protocol SHA does not match crossfit provenance")
    if manifest.sources != _expected_sources(protocol):
        raise ValueError("source identity does not match pinned protocol")

    checks = (
        ("data release", manifest.data_release, protocol.data_release),
        ("teacher column", manifest.teacher_column, protocol.teacher_column),
        (
            "upstream revision",
            manifest.upstream_revision,
            protocol.proteingym_upstream_commit,
        ),
        ("max length", manifest.max_length, protocol.max_length),
        ("working size", manifest.working_size, protocol.working_size),
        (
            "eligible assay IDs",
            manifest.eligible_assay_ids,
            base_manifest.eligible_assay_ids,
        ),
        (
            "screen IDs",
            manifest.screen_ids,
            (base_manifest.development_id, *base_manifest.confirmatory_ids),
        ),
    )
    mismatches = [name for name, actual, expected in checks if actual != expected]
    if mismatches:
        raise ValueError(
            "crossfit manifest provenance mismatch: " + ", ".join(mismatches)
        )
    _validate_base_records(manifest.selected_assays, base_manifest)


def _verify_source_files(
    *,
    dms_zip: Path | str,
    scores_zip: Path | str,
    metadata_csv: Path | str,
    protocol: Protocol,
) -> None:
    sources = (
        ("substitutions", Path(dms_zip), protocol.substitutions_sha256),
        ("scores", Path(scores_zip), protocol.zero_shot_scores_sha256),
        ("metadata", Path(metadata_csv), protocol.metadata_sha256),
    )
    for name, path, expected_sha256 in sources:
        if not path.is_file() or sha256_file(path) != expected_sha256:
            raise ValueError(f"source checksum mismatch for {name}: {path}")


def _load_sequence_lengths(metadata_csv: Path | str) -> dict[str, int]:
    metadata = pd.read_csv(metadata_csv, usecols=["DMS_id", "seq_len"])
    if metadata["DMS_id"].duplicated().any():
        raise ValueError("metadata has duplicate DMS_id values")
    result: dict[str, int] = {}
    for raw_dms_id, raw_length in metadata.itertuples(index=False, name=None):
        if not isinstance(raw_dms_id, str) or not raw_dms_id:
            raise ValueError("metadata DMS_id values must be non-empty strings")
        try:
            numeric_length = float(raw_length)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"metadata seq_len for {raw_dms_id} must be an integer"
            ) from error
        if (
            not math.isfinite(numeric_length)
            or not numeric_length.is_integer()
            or numeric_length <= 0
        ):
            raise ValueError(
                f"metadata seq_len for {raw_dms_id} must be a positive integer"
            )
        result[raw_dms_id] = int(numeric_length)
    return result


def build_crossfit_pool(
    *,
    protocol: Protocol,
    base_manifest_path: Path | str,
    dms_zip: Path | str,
    scores_zip: Path | str,
    metadata_csv: Path | str,
    on_untouched_frame: UntouchedFrameCallback | None = None,
) -> tuple[CrossfitPoolManifest, dict[str, pd.DataFrame]]:
    """Rebuild and bind all 35 pools without constructing or consuming a split."""
    if not isinstance(protocol, Protocol):
        raise TypeError("protocol must be a Protocol")
    if on_untouched_frame is not None and not callable(on_untouched_frame):
        raise TypeError("on_untouched_frame must be callable")
    _verify_source_files(
        dms_zip=dms_zip,
        scores_zip=scores_zip,
        metadata_csv=metadata_csv,
        protocol=protocol,
    )
    base_path = Path(base_manifest_path)
    base_manifest = _load_canonical_base_manifest(base_path)
    _validate_base_protocol_identity(base_manifest, protocol)

    sequence_lengths = _load_sequence_lengths(metadata_csv)
    eligible_ids = base_manifest.eligible_assay_ids
    missing_metadata = [
        dms_id for dms_id in eligible_ids if dms_id not in sequence_lengths
    ]
    if missing_metadata:
        raise ValueError(
            "metadata missing eligible DMS_id values: " + ", ".join(missing_metadata)
        )

    selected_records: list[SelectedAssayManifest] = []
    rebuilt_frames: dict[str, pd.DataFrame] = {}
    screen_ids = (base_manifest.development_id, *base_manifest.confirmatory_ids)
    screen = set(screen_ids)
    for dms_id in eligible_ids:
        sequence_length = sequence_lengths[dms_id]
        if sequence_length > protocol.max_length:
            raise ValueError(
                f"metadata sequence length for {dms_id} exceeds max_length"
            )
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
        working = build_working_set(usable, protocol.working_size)
        selected_records.append(
            SelectedAssayManifest(
                dms_id=dms_id,
                usable_count=len(usable),
                sequence_length=sequence_length,
                row_hashes=tuple(str(value) for value in working["sequence_hash"]),
            )
        )
        if dms_id not in screen:
            rebuilt_frames[dms_id] = working

    records = tuple(selected_records)
    _validate_base_records(records, base_manifest)
    untouched_ids = tuple(dms_id for dms_id in eligible_ids if dms_id not in screen)
    manifest = CrossfitPoolManifest(
        schema_id=CROSSFIT_POOL_SCHEMA_ID,
        schema_version=1,
        card_id=CROSSFIT_CARD_ID,
        card_sha256=CROSSFIT_CARD_SHA256,
        base_manifest_sha256=sha256_file(base_path),
        protocol_sha256=_protocol_sha256(protocol),
        data_release=protocol.data_release,
        teacher_column=protocol.teacher_column,
        sources=_expected_sources(protocol),
        upstream_revision=protocol.proteingym_upstream_commit,
        max_length=protocol.max_length,
        working_size=protocol.working_size,
        eligible_assay_ids=eligible_ids,
        screen_ids=screen_ids,
        untouched_ids=untouched_ids,
        selected_assays=records,
    )
    validate_crossfit_pool_provenance(
        manifest,
        protocol=protocol,
        base_manifest_path=base_path,
    )
    if on_untouched_frame is not None:
        for dms_id, frame in rebuilt_frames.items():
            on_untouched_frame(dms_id, frame.copy(deep=True))
    return manifest, rebuilt_frames
