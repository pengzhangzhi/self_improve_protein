"""Deterministic ProteinGym assay joins, filtering, and data splits."""

from __future__ import annotations

import hashlib
import json
import zipfile
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Self, cast

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictStr,
    StringConstraints,
    model_validator,
)

from self_improve_protein.provenance import atomic_write_json, derive_seed

_CANONICAL_AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWY")
_DMS_REQUIRED_COLUMNS = ("mutant", "mutated_sequence", "DMS_score")
_MAX_PROTEINGYM_SEQUENCE_LENGTH = 512
_NONFINITE_STRINGS = frozenset(
    {
        "nan",
        "+nan",
        "-nan",
        "inf",
        "+inf",
        "-inf",
        "infinity",
        "+infinity",
        "-infinity",
    }
)
_SHA256_PATTERN = r"^[0-9a-f]{64}$"
Sha256Hex = Annotated[
    str,
    StringConstraints(strict=True, pattern=_SHA256_PATTERN),
]


def row_hash(dms_id: str, mutant: str, mutated_sequence: str) -> str:
    """Hash a ProteinGym row using the locked UTF-8, NUL-separated formula."""
    payload = f"{dms_id}\0{mutant}\0{mutated_sequence}".encode()
    return hashlib.sha256(payload).hexdigest()


def _require_columns(
    frame: pd.DataFrame,
    required: Sequence[str],
    frame_name: str,
) -> None:
    missing = [column for column in required if column not in frame.columns]
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"{frame_name} missing required columns: {joined}")


def _reject_duplicate_mutants(frame: pd.DataFrame, frame_name: str) -> None:
    duplicate_mask = frame["mutant"].duplicated(keep=False)
    if duplicate_mask.any():
        duplicate_values = sorted(
            str(value) for value in frame.loc[duplicate_mask, "mutant"].unique()
        )
        raise ValueError(
            f"{frame_name} has duplicate mutant keys: {', '.join(duplicate_values)}"
        )


def _reject_invalid_mutant_keys(frame: pd.DataFrame, frame_name: str) -> None:
    invalid = frame["mutant"].map(lambda value: not isinstance(value, str) or not value)
    if invalid.any():
        raise ValueError(f"{frame_name} has empty or non-string mutant keys")


def merge_assay_frames(
    dms_df: pd.DataFrame,
    scores_df: pd.DataFrame,
    dms_id: str,
    teacher_column: str,
) -> pd.DataFrame:
    """One-to-one inner join one assay with one literal teacher score column."""
    if not isinstance(dms_id, str) or not dms_id:
        raise ValueError("dms_id must be a non-empty string")
    if not isinstance(teacher_column, str) or not teacher_column:
        raise ValueError("teacher_column must be a non-empty string")
    _require_columns(dms_df, _DMS_REQUIRED_COLUMNS, "DMS frame")
    _require_columns(scores_df, ("mutant", teacher_column), "scores frame")
    if teacher_column in dms_df.columns:
        raise ValueError(
            f"DMS frame unexpectedly contains teacher column {teacher_column}"
        )
    if "source_row" in dms_df.columns or "dms_id" in dms_df.columns:
        raise ValueError("DMS frame contains reserved source_row or dms_id column")
    _reject_invalid_mutant_keys(dms_df, "DMS frame")
    _reject_invalid_mutant_keys(scores_df, "scores frame")
    _reject_duplicate_mutants(dms_df, "DMS frame")
    _reject_duplicate_mutants(scores_df, "scores frame")

    dms = dms_df.copy()
    dms["source_row"] = np.arange(len(dms), dtype=np.int64)
    dms["dms_id"] = dms_id
    scores = scores_df.loc[:, ["mutant", teacher_column]].copy()
    return dms.merge(
        scores,
        on="mutant",
        how="inner",
        sort=False,
        validate="one_to_one",
    )


def _coerce_numeric_column(frame: pd.DataFrame, column: str) -> pd.Series:
    source = frame[column]
    normalized = source.astype("string").str.strip().str.lower()
    missing = source.isna() | normalized.eq("")
    explicit_nonfinite = normalized.isin(_NONFINITE_STRINGS)
    bool_values = source.map(lambda value: isinstance(value, (bool, np.bool_)))
    converted = pd.to_numeric(source, errors="coerce")
    non_numeric = (~missing) & (~explicit_nonfinite) & converted.isna()
    if bool_values.any() or non_numeric.any() or np.iscomplexobj(converted.to_numpy()):
        raise ValueError(f"non-numeric values in {column}")
    return converted.astype(np.float64)


def _is_canonical_sequence(value: object, max_length: int) -> bool:
    return (
        isinstance(value, str)
        and 0 < len(value) <= max_length
        and all(residue in _CANONICAL_AMINO_ACIDS for residue in value)
    )


def _validate_exact_sequence_hashes(frame: pd.DataFrame) -> None:
    _require_columns(
        frame,
        ("dms_id", "mutant", "mutated_sequence", "sequence_hash"),
        "working set",
    )
    rows = frame.loc[
        :, ["dms_id", "mutant", "mutated_sequence", "sequence_hash"]
    ].itertuples(index=False, name=None)
    for dms_id, mutant, sequence, sequence_hash in rows:
        if not all(
            isinstance(value, str)
            for value in (dms_id, mutant, sequence, sequence_hash)
        ) or sequence_hash != row_hash(dms_id, mutant, sequence):
            raise ValueError(
                "sequence_hash values must match the exact row hash formula"
            )


def filter_usable_variants(
    frame: pd.DataFrame,
    teacher_column: str,
    max_length: int,
) -> pd.DataFrame:
    """Return finite, canonical, deterministically de-duplicated assay variants."""
    if type(max_length) is not int:
        raise TypeError("max_length must be an int")
    if max_length <= 0:
        raise ValueError("max_length must be positive")
    if max_length > _MAX_PROTEINGYM_SEQUENCE_LENGTH:
        raise ValueError(
            "max_length cannot exceed the locked ProteinGym limit of "
            f"{_MAX_PROTEINGYM_SEQUENCE_LENGTH}"
        )
    _require_columns(
        frame,
        (*_DMS_REQUIRED_COLUMNS, teacher_column, "dms_id"),
        "merged assay frame",
    )

    dms_ids = frame["dms_id"].drop_duplicates().tolist()
    if len(dms_ids) != 1 or not isinstance(dms_ids[0], str) or not dms_ids[0]:
        raise ValueError("merged assay frame must contain exactly one non-empty dms_id")
    dms_id = dms_ids[0]
    dms_scores = _coerce_numeric_column(frame, "DMS_score")
    teacher_scores = _coerce_numeric_column(frame, teacher_column)
    sequences = frame["mutated_sequence"]
    valid_sequences = sequences.map(
        lambda value: _is_canonical_sequence(value, max_length)
    )
    valid_mutants = frame["mutant"].map(
        lambda value: isinstance(value, str) and bool(value)
    )
    usable_mask = (
        np.isfinite(dms_scores.to_numpy())
        & np.isfinite(teacher_scores.to_numpy())
        & valid_sequences.to_numpy()
        & valid_mutants.to_numpy()
    )

    usable = frame.loc[usable_mask].copy()
    usable["DMS_score"] = dms_scores.loc[usable_mask].to_numpy()
    usable[teacher_column] = teacher_scores.loc[usable_mask].to_numpy()
    usable["dms_id"] = dms_id
    usable["sequence_hash"] = [
        row_hash(dms_id, mutant, sequence)
        for mutant, sequence in zip(
            usable["mutant"], usable["mutated_sequence"], strict=True
        )
    ]
    usable = usable.sort_values("sequence_hash", kind="stable")
    usable = usable.drop_duplicates("mutated_sequence", keep="first")
    return usable.reset_index(drop=True)


def _require_strict_positive_int(value: object, name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an int")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def build_working_set(usable: pd.DataFrame, size: int) -> pd.DataFrame:
    """Take an exact-size, SHA-256-ordered working set."""
    checked_size = _require_strict_positive_int(size, "size")
    _require_columns(
        usable,
        ("dms_id", "mutant", "mutated_sequence", "sequence_hash"),
        "usable assay frame",
    )
    for column in ("sequence_hash", "mutated_sequence"):
        if usable[column].duplicated().any():
            raise ValueError(f"usable assay frame has duplicate {column} values")
    _validate_exact_sequence_hashes(usable)
    if len(usable) < checked_size:
        raise ValueError(
            f"insufficient usable rows: need {checked_size}, found {len(usable)}"
        )
    ordered = usable.sort_values("sequence_hash", kind="stable")
    return ordered.iloc[:checked_size].reset_index(drop=True).copy()


def _read_exact_csv_member(
    archive_path: Path | str,
    member_name: str,
) -> pd.DataFrame:
    with zipfile.ZipFile(archive_path) as archive:
        matches = [
            member for member in archive.infolist() if member.filename == member_name
        ]
        if len(matches) != 1:
            raise ValueError(
                f"archive must contain exactly one member named {member_name}; "
                f"found {len(matches)}"
            )
        with archive.open(matches[0]) as source:
            return pd.read_csv(source)


def load_assay_from_archives(
    dms_zip: Path | str,
    scores_zip: Path | str,
    dms_id: str,
    teacher_column: str,
) -> pd.DataFrame:
    """Load and join one assay directly from the pinned v1.3 ZIP layouts."""
    dms_member = f"DMS_ProteinGym_substitutions/{dms_id}.csv"
    score_member = f"{dms_id}.csv"
    dms_frame = _read_exact_csv_member(dms_zip, dms_member)
    score_frame = _read_exact_csv_member(scores_zip, score_member)
    return merge_assay_frames(dms_frame, score_frame, dms_id, teacher_column)


@dataclass(frozen=True, slots=True)
class AssayEligibility:
    """Immutable pre-outcome assay eligibility summary."""

    dms_id: str
    usable_count: int
    sequence_length: int

    def __post_init__(self) -> None:
        if not isinstance(self.dms_id, str) or not self.dms_id:
            raise ValueError("dms_id must be a non-empty string")
        if type(self.usable_count) is not int:
            raise TypeError("usable_count must be an int")
        if self.usable_count < 0:
            raise ValueError("usable_count must be non-negative")
        if type(self.sequence_length) is not int:
            raise TypeError("sequence_length must be an int")
        if self.sequence_length <= 0:
            raise ValueError("sequence_length must be positive")


def select_eligible_assays(
    records: Iterable[AssayEligibility],
    minimum: int,
    assay_count: int,
) -> tuple[tuple[str, ...], str]:
    """Select lexical confirmatory assays and the next development assay."""
    checked_minimum = _require_strict_positive_int(minimum, "minimum")
    checked_assay_count = _require_strict_positive_int(assay_count, "assay_count")
    materialized = tuple(records)
    seen: dict[str, AssayEligibility] = {}
    for record in materialized:
        if not isinstance(record, AssayEligibility):
            raise TypeError("records must contain AssayEligibility values")
        previous = seen.get(record.dms_id)
        if previous is not None:
            if previous.sequence_length != record.sequence_length:
                raise ValueError(
                    f"inconsistent sequence_length records for {record.dms_id}"
                )
            raise ValueError(f"duplicate assay record for {record.dms_id}")
        seen[record.dms_id] = record

    eligible_ids = sorted(
        record.dms_id
        for record in materialized
        if record.usable_count >= checked_minimum
        and record.sequence_length <= _MAX_PROTEINGYM_SEQUENCE_LENGTH
    )
    required = checked_assay_count + 1
    if len(eligible_ids) < required:
        raise ValueError(
            f"need at least {required} eligible assays for confirmatory plus dev; "
            f"found {len(eligible_ids)}"
        )
    return tuple(eligible_ids[:checked_assay_count]), eligible_ids[checked_assay_count]


class _FrozenManifestModel(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        allow_inf_nan=False,
    )


class ManifestSource(_FrozenManifestModel):
    """Immutable URL and content digest for one source artifact."""

    url: str = Field(min_length=1, strict=True)
    sha256: Sha256Hex


class ManifestSources(_FrozenManifestModel):
    """The three source artifacts required by a ProteinGym data manifest."""

    substitutions: ManifestSource
    scores: ManifestSource
    metadata: ManifestSource


class SelectedAssayManifest(_FrozenManifestModel):
    """Immutable selected-assay counts and ordered working-set hashes."""

    dms_id: str = Field(min_length=1, strict=True)
    usable_count: int = Field(gt=0, strict=True)
    sequence_length: int = Field(gt=0, strict=True)
    row_hashes: tuple[Sha256Hex, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_row_hash_order(self) -> Self:
        """Require a unique SHA-ordered working set for this assay."""
        if len(set(self.row_hashes)) != len(self.row_hashes):
            raise ValueError("row_hashes must be unique within each selected assay")
        if self.row_hashes != tuple(sorted(self.row_hashes)):
            raise ValueError("row_hashes must be sorted in ascending hash order")
        return self


class DataManifest(_FrozenManifestModel):
    """Immutable, outcome-free provenance for frozen ProteinGym working sets."""

    schema_version: int = Field(ge=1, le=1, strict=True)
    data_release: str = Field(min_length=1, strict=True)
    teacher_column: str = Field(min_length=1, strict=True)
    sources: ManifestSources
    upstream_revision: str = Field(min_length=1, strict=True)
    eligible_assay_ids: tuple[StrictStr, ...] = Field(min_length=2)
    confirmatory_ids: tuple[StrictStr, ...] = Field(min_length=1)
    development_id: str = Field(min_length=1, strict=True)
    max_length: int = Field(
        gt=0,
        le=_MAX_PROTEINGYM_SEQUENCE_LENGTH,
        strict=True,
    )
    working_size: int = Field(gt=0, strict=True)
    selected_assays: tuple[SelectedAssayManifest, ...] = Field(min_length=2)

    @model_validator(mode="after")
    def validate_selection_and_working_sets(self) -> Self:
        """Validate lexical assay selection and exact working-set coverage."""
        eligible = self.eligible_assay_ids
        if len(set(eligible)) != len(eligible) or eligible != tuple(sorted(eligible)):
            raise ValueError("eligible_assay_ids must be unique and lexically sorted")

        confirmatory_count = len(self.confirmatory_ids)
        if len(eligible) <= confirmatory_count:
            raise ValueError(
                "eligible assay prefix must include confirmatory and development IDs"
            )
        if (
            self.confirmatory_ids != eligible[:confirmatory_count]
            or self.development_id != eligible[confirmatory_count]
        ):
            raise ValueError(
                "confirmatory_ids and development_id must match the "
                "eligible assay prefix"
            )

        expected_selected_ids = (*self.confirmatory_ids, self.development_id)
        actual_selected_ids = tuple(record.dms_id for record in self.selected_assays)
        if actual_selected_ids != expected_selected_ids:
            raise ValueError(
                "selected_assays coverage and order must match "
                "confirmatory plus dev IDs"
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
                    f"row_hash count for {record.dms_id} must equal working_size"
                )
            all_hashes.extend(record.row_hashes)
        if len(set(all_hashes)) != len(all_hashes):
            raise ValueError("row_hashes must be unique across selected assays")
        return self


def write_data_manifest(path: Path | str, manifest: DataManifest) -> None:
    """Atomically serialize a validated data manifest in canonical JSON form."""
    if not isinstance(manifest, DataManifest):
        raise TypeError("manifest must be a DataManifest")
    atomic_write_json(path, manifest.model_dump(mode="json"))


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def load_data_manifest(path: Path | str) -> DataManifest:
    """Load and strictly validate a data manifest JSON artifact."""
    with Path(path).open(encoding="utf-8") as source:
        payload = json.load(source, object_pairs_hook=_unique_json_object)
    return DataManifest.model_validate(payload)


@dataclass(frozen=True, slots=True)
class SplitIndices:
    """Immutable row indices and matching sequence hashes for all split partitions."""

    labeled: tuple[int, ...]
    unlabeled: tuple[int, ...]
    test: tuple[int, ...]
    buffer: tuple[int, ...]
    labeled_sequence_hashes: tuple[str, ...]
    unlabeled_sequence_hashes: tuple[str, ...]
    test_sequence_hashes: tuple[str, ...]
    buffer_sequence_hashes: tuple[str, ...]

    def __post_init__(self) -> None:
        index_names = ("labeled", "unlabeled", "test", "buffer")
        hash_names = (
            "labeled_sequence_hashes",
            "unlabeled_sequence_hashes",
            "test_sequence_hashes",
            "buffer_sequence_hashes",
        )
        for name in (*index_names, *hash_names):
            object.__setattr__(self, name, tuple(getattr(self, name)))

        index_groups = tuple(getattr(self, name) for name in index_names)
        hash_groups = tuple(getattr(self, name) for name in hash_names)
        for index_group, hash_group in zip(index_groups, hash_groups, strict=True):
            if len(index_group) != len(hash_group):
                raise ValueError("split index and sequence-hash counts must match")
            if any(type(index) is not int or index < 0 for index in index_group):
                raise ValueError("split indices must be non-negative ints")
            if any(not isinstance(value, str) or not value for value in hash_group):
                raise ValueError("split sequence hashes must be non-empty strings")

        all_indices = tuple(index for group in index_groups for index in group)
        all_hashes = tuple(value for group in hash_groups for value in group)
        if len(set(all_indices)) != len(all_indices):
            raise ValueError("split indices must be pairwise disjoint")
        if len(set(all_hashes)) != len(all_hashes):
            raise ValueError("split sequence hashes must be pairwise disjoint")

    @property
    def labeled_hashes(self) -> tuple[str, ...]:
        """Alias for the labeled sequence hashes."""
        return self.labeled_sequence_hashes

    @property
    def unlabeled_hashes(self) -> tuple[str, ...]:
        """Alias for the unlabeled sequence hashes."""
        return self.unlabeled_sequence_hashes

    @property
    def test_hashes(self) -> tuple[str, ...]:
        """Alias for the test sequence hashes."""
        return self.test_sequence_hashes

    @property
    def buffer_hashes(self) -> tuple[str, ...]:
        """Alias for the buffer sequence hashes."""
        return self.buffer_sequence_hashes

    def validate_against(self, working_set: pd.DataFrame) -> None:
        """Check stored hashes and complete coverage against a working set."""
        _validate_exact_sequence_hashes(working_set)
        index_groups = (self.labeled, self.unlabeled, self.test, self.buffer)
        hash_groups = (
            self.labeled_sequence_hashes,
            self.unlabeled_sequence_hashes,
            self.test_sequence_hashes,
            self.buffer_sequence_hashes,
        )
        all_indices = tuple(index for group in index_groups for index in group)
        if set(all_indices) != set(range(len(working_set))):
            raise ValueError("split indices do not cover the working set exactly once")
        for indices, expected_hashes in zip(index_groups, hash_groups, strict=True):
            actual_hashes = tuple(
                cast(str, value)
                for value in working_set.iloc[list(indices)]["sequence_hash"]
            )
            if actual_hashes != expected_hashes:
                raise ValueError("split sequence hashes do not match the working set")


def _strict_split_int(value: object, name: str, *, allow_zero: bool) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an int")
    minimum = 0 if allow_zero else 1
    if value < minimum:
        comparison = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be {comparison}")
    return value


def _hashes_at(working_set: pd.DataFrame, indices: tuple[int, ...]) -> tuple[str, ...]:
    return tuple(
        cast(str, value) for value in working_set.iloc[list(indices)]["sequence_hash"]
    )


def make_split(
    working_set: pd.DataFrame,
    dms_id: str,
    seed: int,
    n_labeled: int,
    n_unlabeled: int,
    n_test: int,
) -> SplitIndices:
    """Partition every working row with a purpose-separated PCG64 permutation."""
    if not isinstance(dms_id, str) or not dms_id:
        raise ValueError("dms_id must be a non-empty string")
    checked_seed = _strict_split_int(seed, "seed", allow_zero=True)
    checked_labeled = _strict_split_int(n_labeled, "n_labeled", allow_zero=False)
    checked_unlabeled = _strict_split_int(n_unlabeled, "n_unlabeled", allow_zero=False)
    checked_test = _strict_split_int(n_test, "n_test", allow_zero=False)
    _require_columns(
        working_set,
        ("dms_id", "mutant", "sequence_hash", "mutated_sequence"),
        "working set",
    )
    _validate_exact_sequence_hashes(working_set)
    if working_set["sequence_hash"].duplicated().any():
        raise ValueError("working set has duplicate sequence_hash values")
    if working_set["mutated_sequence"].duplicated().any():
        raise ValueError("working set has duplicate mutated_sequence values")
    if not working_set["dms_id"].map(lambda value: value == dms_id).all():
        raise ValueError("working set dms_id values do not match dms_id")

    row_count = len(working_set)
    assigned_count = checked_labeled + checked_unlabeled + checked_test
    if assigned_count > row_count:
        raise ValueError(
            f"requested split size {assigned_count} exceeds {row_count} working rows"
        )
    generator = np.random.Generator(
        np.random.PCG64(derive_seed(dms_id, checked_seed, "split"))
    )
    permutation = generator.permutation(row_count)
    labeled_end = checked_labeled
    unlabeled_end = labeled_end + checked_unlabeled
    test_end = unlabeled_end + checked_test
    labeled = tuple(int(value) for value in permutation[:labeled_end])
    unlabeled = tuple(int(value) for value in permutation[labeled_end:unlabeled_end])
    test = tuple(int(value) for value in permutation[unlabeled_end:test_end])
    buffer = tuple(int(value) for value in permutation[test_end:])
    split = SplitIndices(
        labeled=labeled,
        unlabeled=unlabeled,
        test=test,
        buffer=buffer,
        labeled_sequence_hashes=_hashes_at(working_set, labeled),
        unlabeled_sequence_hashes=_hashes_at(working_set, unlabeled),
        test_sequence_hashes=_hashes_at(working_set, test),
        buffer_sequence_hashes=_hashes_at(working_set, buffer),
    )
    split.validate_against(working_set)
    return split
