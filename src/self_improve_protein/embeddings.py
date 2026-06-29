"""Residue-only ESM-2 embeddings with provenance-validated atomic caches."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Annotated, Any, Final, Literal, Self, cast

import numpy as np
import torch
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    StringConstraints,
    model_validator,
)

from self_improve_protein.provenance import atomic_write_json, sha256_file

DEFAULT_MODEL_ID: Final = "facebook/esm2_t12_35M_UR50D"
DEFAULT_MODEL_REVISION: Final = "6fbf070e65b0b7291e7bbcd451118c216cff79d8"
POOLING_DEFINITION: Final[
    Literal["mean_last_hidden_state_attention_non_special_v1"]
] = "mean_last_hidden_state_attention_non_special_v1"

_SHA256_PATTERN = r"^[0-9a-f]{64}$"
_GIT_COMMIT_PATTERN = r"^[0-9a-f]{40}$"
_INTEGER_MASK_DTYPES = {
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.bool,
}

Sha256Hex = Annotated[
    str,
    StringConstraints(strict=True, pattern=_SHA256_PATTERN),
]
GitCommit = Annotated[
    str,
    StringConstraints(strict=True, pattern=_GIT_COMMIT_PATTERN),
]


class _FrozenCacheModel(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        allow_inf_nan=False,
    )


class EmbeddingCacheSources(_FrozenCacheModel):
    """Exact upstream inputs that determine an embedding cache identity."""

    proteingym_upstream_commit: GitCommit
    substitutions_sha256: Sha256Hex
    zero_shot_scores_sha256: Sha256Hex
    metadata_sha256: Sha256Hex


class EmbeddingCacheMetadata(_FrozenCacheModel):
    """Immutable content and provenance contract for one assay cache."""

    schema_version: int = Field(ge=1, le=1, strict=True)
    dms_id: str = Field(min_length=1, strict=True)
    row_hash_digest: Sha256Hex
    row_count: int = Field(gt=0, strict=True)
    model_id: str = Field(min_length=1, strict=True)
    model_revision: GitCommit
    pooling: Literal["mean_last_hidden_state_attention_non_special_v1"]
    shape: tuple[StrictInt, StrictInt]
    dtype: Literal["float32"]
    sources: EmbeddingCacheSources
    npy_sha256: Sha256Hex

    @model_validator(mode="after")
    def validate_shape(self) -> Self:
        """Tie the serialized matrix shape to the frozen working-set count."""
        if self.shape[0] <= 0 or self.shape[1] <= 0:
            raise ValueError("embedding shape dimensions must be positive")
        if self.shape[0] != self.row_count:
            raise ValueError("embedding shape row count must match row_count")
        return self


def _require_tensor(value: object, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    return value


def _validate_mask(mask: torch.Tensor, name: str) -> None:
    if mask.dtype not in _INTEGER_MASK_DTYPES:
        raise TypeError(f"{name} mask must have boolean or integer dtype")
    if not bool(torch.all((mask == 0) | (mask == 1)).item()):
        raise ValueError(f"{name} mask values must be zero or one")


def mean_pool_residues(
    last_hidden_state: torch.Tensor,
    attention_mask: torch.Tensor,
    special_tokens_mask: torch.Tensor,
) -> torch.Tensor:
    """Mean-pool finite last-layer states over attended non-special tokens."""
    hidden = _require_tensor(last_hidden_state, "last_hidden_state")
    attention = _require_tensor(attention_mask, "attention_mask")
    special = _require_tensor(special_tokens_mask, "special_tokens_mask")
    if hidden.ndim != 3:
        raise ValueError("last_hidden_state must be 3-D")
    if attention.ndim != 2 or special.ndim != 2:
        raise ValueError("attention and special-token masks must be 2-D")
    expected_mask_shape = hidden.shape[:2]
    if attention.shape != expected_mask_shape or special.shape != expected_mask_shape:
        raise ValueError("mask shape must match the hidden batch and token shape")
    if hidden.shape[0] == 0 or hidden.shape[1] == 0 or hidden.shape[2] == 0:
        raise ValueError("last_hidden_state dimensions must be nonzero")
    if hidden.device != attention.device or hidden.device != special.device:
        raise ValueError("hidden state and masks must be on the same device")
    if not hidden.is_floating_point():
        raise TypeError("last_hidden_state must have a floating dtype")
    _validate_mask(attention, "attention")
    _validate_mask(special, "special_tokens")
    if not bool(torch.isfinite(hidden).all().item()):
        raise ValueError("last_hidden_state contains non-finite values")

    residue_mask = attention.to(torch.bool) & ~special.to(torch.bool)
    residue_counts = residue_mask.sum(dim=1)
    if bool((residue_counts == 0).any().item()):
        raise ValueError("each sequence must contain at least one; zero residue row")
    hidden_float = hidden.to(torch.float32)
    pooled = (hidden_float * residue_mask.unsqueeze(-1)).sum(dim=1)
    pooled = pooled / residue_counts.to(torch.float32).unsqueeze(-1)
    if not bool(torch.isfinite(pooled).all().item()):
        raise ValueError("mean-pooled embeddings contain non-finite values")
    return pooled.to(torch.float32)


def _validate_sequences(sequences: Sequence[str]) -> tuple[str, ...]:
    if isinstance(sequences, (str, bytes)) or not isinstance(sequences, Sequence):
        raise TypeError("sequences must be a non-string sequence")
    materialized = tuple(sequences)
    if not materialized:
        raise ValueError("sequences must not be empty")
    if any(type(sequence) is not str or not sequence for sequence in materialized):
        raise ValueError("each sequence must be a non-empty string")
    return materialized


def _validate_batch_size(batch_size: int) -> int:
    if type(batch_size) is not int:
        raise TypeError("batch_size must be an int")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    return batch_size


def _tokenize_batch(tokenizer: Any, batch: tuple[str, ...]) -> dict[str, torch.Tensor]:
    encoded = tokenizer(
        batch,
        add_special_tokens=True,
        padding=True,
        return_attention_mask=True,
        return_special_tokens_mask=True,
        return_tensors="pt",
        truncation=False,
    )
    if not isinstance(encoded, Mapping):
        raise TypeError("tokenizer output must be a mapping")
    required = {"input_ids", "attention_mask", "special_tokens_mask"}
    if not required.issubset(encoded):
        missing = ", ".join(sorted(required.difference(encoded)))
        raise ValueError(f"tokenizer output is missing required tensors: {missing}")
    tensors: dict[str, torch.Tensor] = {}
    for key, value in encoded.items():
        if not isinstance(key, str) or not isinstance(value, torch.Tensor):
            raise TypeError("tokenizer output must map strings to tensors")
        tensors[key] = value
    return tensors


def embed_sequences_with_model(
    sequences: Sequence[str],
    *,
    tokenizer: Any,
    model: Any,
    batch_size: int,
    device: str | torch.device,
) -> np.ndarray:
    """Embed sequences in exact input order using already-loaded components."""
    ordered_sequences = _validate_sequences(sequences)
    checked_batch_size = _validate_batch_size(batch_size)
    checked_device = torch.device(device)
    model = model.to(checked_device)
    model = model.eval()
    batches: list[np.ndarray] = []
    embedding_width: int | None = None
    for start in range(0, len(ordered_sequences), checked_batch_size):
        batch = ordered_sequences[start : start + checked_batch_size]
        encoded = _tokenize_batch(tokenizer, batch)
        special = encoded.pop("special_tokens_mask").to(checked_device)
        model_inputs = {
            key: value.to(checked_device) for key, value in encoded.items()
        }
        attention = model_inputs["attention_mask"]
        with (
            torch.inference_mode(),
            torch.autocast(device_type=checked_device.type, enabled=False),
        ):
            output = model(**model_inputs, return_dict=True)
            hidden = getattr(output, "last_hidden_state", None)
            if not isinstance(hidden, torch.Tensor):
                raise TypeError("model output must contain tensor last_hidden_state")
            pooled = mean_pool_residues(hidden, attention, special)
        if pooled.shape[0] != len(batch):
            raise ValueError("embedding batch row count does not match input order")
        if embedding_width is None:
            embedding_width = pooled.shape[1]
        elif pooled.shape[1] != embedding_width:
            raise ValueError("embedding width changed between batches")
        batches.append(pooled.detach().cpu().numpy().astype(np.float32, copy=False))
    result = np.concatenate(batches, axis=0)
    if result.shape[0] != len(ordered_sequences) or result.dtype != np.float32:
        raise RuntimeError("embedding output violated row-order or dtype contract")
    if not np.isfinite(result).all():
        raise ValueError("embedding output contains non-finite values")
    return np.ascontiguousarray(result)


def _validate_float32_model_state(model: Any) -> None:
    """Reject floating model state that would change embedding cache bytes."""
    for state_kind, values in (
        ("parameter", model.named_parameters()),
        ("buffer", model.named_buffers()),
    ):
        for name, value in values:
            if not isinstance(name, str) or not isinstance(value, torch.Tensor):
                raise TypeError(f"model {state_kind} entries must be named tensors")
            if value.is_floating_point() and value.dtype != torch.float32:
                raise ValueError(
                    f"model {state_kind} {name!r} must have dtype float32"
                )


def _load_hf_components(model_id: str, model_revision: str) -> tuple[Any, Any]:
    """Load one explicit revision in float32 only on the real command path."""
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(  # type: ignore[no-untyped-call]
        model_id,
        revision=model_revision,
    )
    model = AutoModel.from_pretrained(
        model_id,
        revision=model_revision,
        add_pooling_layer=False,
        dtype=torch.float32,
    )
    _validate_float32_model_state(model)
    return tokenizer, model


def embed_sequences(
    sequences: Sequence[str],
    *,
    model_id: str,
    model_revision: str,
    batch_size: int,
    device: str | torch.device,
) -> np.ndarray:
    """Embed sequences with one explicit revision-pinned representation."""
    tokenizer, model = _load_hf_components(model_id, model_revision)
    return embed_sequences_with_model(
        sequences,
        tokenizer=tokenizer,
        model=model,
        batch_size=batch_size,
        device=device,
    )


def ordered_row_hash_digest(row_hashes: Sequence[str]) -> str:
    """Digest fixed-width SHA-256 row hashes in their exact manifest order."""
    if isinstance(row_hashes, (str, bytes)) or not isinstance(row_hashes, Sequence):
        raise TypeError("row_hashes must be a non-string sequence")
    materialized = tuple(row_hashes)
    if not materialized:
        raise ValueError("row_hashes must not be empty")
    if len(set(materialized)) != len(materialized):
        raise ValueError("row_hashes must be unique")
    digest = hashlib.sha256()
    for value in materialized:
        if (
            type(value) is not str
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError("each row_hash must be a lowercase SHA-256 digest")
        digest.update(bytes.fromhex(value))
    return digest.hexdigest()


def _validate_cache_identity(
    dms_id: str,
    row_hashes: Sequence[str],
    model_id: str,
    model_revision: str,
    sources: EmbeddingCacheSources,
) -> tuple[tuple[str, ...], str]:
    if type(dms_id) is not str or not dms_id:
        raise ValueError("dms_id must be a non-empty string")
    if type(model_id) is not str or not model_id:
        raise ValueError("model_id must be a non-empty string")
    if (
        type(model_revision) is not str
        or len(model_revision) != 40
        or any(character not in "0123456789abcdef" for character in model_revision)
    ):
        raise ValueError("model_revision must be a lowercase 40-character commit")
    if not isinstance(sources, EmbeddingCacheSources):
        raise TypeError("sources must be EmbeddingCacheSources")
    materialized_hashes = tuple(row_hashes)
    return materialized_hashes, ordered_row_hash_digest(materialized_hashes)


def _validate_embedding_array(
    embeddings: np.ndarray,
    *,
    row_count: int,
    embedding_dim: int | None = None,
) -> np.ndarray:
    if not isinstance(embeddings, np.ndarray):
        raise TypeError("embeddings must be a numpy.ndarray")
    if embeddings.dtype != np.dtype(np.float32):
        raise TypeError("embeddings must have dtype float32")
    if embeddings.ndim != 2:
        raise ValueError("embeddings must be a 2-D matrix")
    if embeddings.shape[0] != row_count:
        raise ValueError("embedding row count does not match row hashes")
    if embeddings.shape[0] <= 0 or embeddings.shape[1] <= 0:
        raise ValueError("embedding shape dimensions must be positive")
    if embedding_dim is not None and embeddings.shape[1] != embedding_dim:
        raise ValueError("embedding width does not match expected embedding dimension")
    if not np.isfinite(embeddings).all():
        raise ValueError("embeddings contain non-finite values")
    return np.ascontiguousarray(embeddings)


def _fsync_directory(directory: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(directory, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _remove_cache_pair(npy_path: Path, metadata_path: Path) -> None:
    """Remove cache payload files and durably record their absence."""
    directories: set[Path] = set()
    for path in (npy_path, metadata_path):
        path.unlink(missing_ok=True)
        directories.add(path.parent)
    for directory in directories:
        if directory.exists():
            _fsync_directory(directory)


def _best_effort_remove_cache_pair(npy_path: Path, metadata_path: Path) -> None:
    """Try every cleanup step without masking the originating write error."""
    directories: set[Path] = set()
    for path in (npy_path, metadata_path):
        try:
            path.unlink(missing_ok=True)
        except OSError:
            continue
        directories.add(path.parent)
    for directory in directories:
        try:
            if directory.exists():
                _fsync_directory(directory)
        except OSError:
            continue


@contextmanager
def _exclusive_cache_lock(metadata_path: Path) -> Iterator[None]:
    """Serialize cache inspection and writes across cluster processes."""
    lock_path = metadata_path.with_name(f".{metadata_path.name}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as lock_file:
        os.fsync(lock_file.fileno())
        _fsync_directory(lock_path.parent)
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _atomic_write_npy(path: Path, embeddings: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            np.save(handle, embeddings, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        digest = sha256_file(temporary_path)
        os.replace(temporary_path, path)
        temporary_path = None
        _fsync_directory(path.parent)
        return digest
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def write_embedding_cache(
    npy_path: Path | str,
    metadata_path: Path | str,
    embeddings: np.ndarray,
    *,
    dms_id: str,
    row_hashes: Sequence[str],
    model_id: str,
    model_revision: str,
    sources: EmbeddingCacheSources,
) -> EmbeddingCacheMetadata:
    """Lock-free low-level write for prevalidated data; prefer get-or-create."""
    array_destination = Path(npy_path)
    metadata_destination = Path(metadata_path)
    if array_destination == metadata_destination:
        raise ValueError("array and metadata paths must be distinct")
    hashes, row_digest = _validate_cache_identity(
        dms_id,
        row_hashes,
        model_id,
        model_revision,
        sources,
    )
    checked_embeddings = _validate_embedding_array(
        embeddings,
        row_count=len(hashes),
    )
    npy_digest = _atomic_write_npy(array_destination, checked_embeddings)
    try:
        metadata = EmbeddingCacheMetadata(
            schema_version=1,
            dms_id=dms_id,
            row_hash_digest=row_digest,
            row_count=len(hashes),
            model_id=model_id,
            model_revision=model_revision,
            pooling=POOLING_DEFINITION,
            shape=cast(tuple[int, int], checked_embeddings.shape),
            dtype="float32",
            sources=sources,
            npy_sha256=npy_digest,
        )
        atomic_write_json(metadata_destination, metadata.model_dump(mode="json"))
    except Exception:
        _best_effort_remove_cache_pair(array_destination, metadata_destination)
        raise
    return metadata


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _load_cache_metadata(path: Path) -> EmbeddingCacheMetadata:
    with path.open(encoding="utf-8") as source:
        payload = json.load(source, object_pairs_hook=_unique_json_object)
    return EmbeddingCacheMetadata.model_validate(payload)


def _validate_expected_embedding_dim(expected_embedding_dim: int) -> int:
    if type(expected_embedding_dim) is not int:
        raise TypeError("expected_embedding_dim must be an int")
    if expected_embedding_dim <= 0:
        raise ValueError("expected_embedding_dim must be positive")
    return expected_embedding_dim


def load_embedding_cache(
    npy_path: Path | str,
    metadata_path: Path | str,
    *,
    dms_id: str,
    row_hashes: Sequence[str],
    model_id: str,
    model_revision: str,
    sources: EmbeddingCacheSources,
    expected_embedding_dim: int,
) -> np.ndarray:
    """Load a cache only after validating provenance, bytes, shape, and values."""
    array_source = Path(npy_path)
    metadata_source = Path(metadata_path)
    if not array_source.exists() or not metadata_source.exists():
        raise ValueError("embedding cache pair is missing or incomplete")
    hashes, row_digest = _validate_cache_identity(
        dms_id,
        row_hashes,
        model_id,
        model_revision,
        sources,
    )
    checked_dim = _validate_expected_embedding_dim(expected_embedding_dim)
    metadata = _load_cache_metadata(metadata_source)
    expected_fields: dict[str, object] = {
        "dms_id": dms_id,
        "row_hash_digest": row_digest,
        "row_count": len(hashes),
        "model_id": model_id,
        "model_revision": model_revision,
        "pooling": POOLING_DEFINITION,
        "shape": (len(hashes), checked_dim),
        "dtype": "float32",
        "sources": sources,
    }
    mismatches = [
        field
        for field, expected in expected_fields.items()
        if getattr(metadata, field) != expected
    ]
    if mismatches:
        raise ValueError(
            "embedding cache metadata mismatch: " + ", ".join(mismatches)
        )
    try:
        handle = array_source.open("rb")
    except OSError as error:
        raise ValueError("embedding cache NPY is corrupt") from error
    with handle:
        digest = hashlib.sha256()
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
        if digest.hexdigest() != metadata.npy_sha256:
            raise ValueError("embedding cache NPY checksum mismatch")
        handle.seek(0)
        try:
            loaded = np.load(handle, allow_pickle=False)
            if handle.read(1):
                raise ValueError("embedding cache NPY contains trailing bytes")
        except (OSError, ValueError) as error:
            raise ValueError("embedding cache NPY is corrupt") from error
    return _validate_embedding_array(
        loaded,
        row_count=len(hashes),
        embedding_dim=checked_dim,
    )


def get_or_create_embedding_cache(
    npy_path: Path | str,
    metadata_path: Path | str,
    *,
    dms_id: str,
    row_hashes: Sequence[str],
    model_id: str,
    model_revision: str,
    sources: EmbeddingCacheSources,
    expected_embedding_dim: int,
    sequences: Sequence[str],
    batch_size: int,
    device: str | torch.device,
) -> np.ndarray:
    """Create one identity-coupled cache under a cross-process writer lock."""
    array_path = Path(npy_path)
    json_path = Path(metadata_path)
    ordered_sequences = _validate_sequences(sequences)
    checked_batch_size = _validate_batch_size(batch_size)
    checked_dim = _validate_expected_embedding_dim(expected_embedding_dim)
    hashes, _ = _validate_cache_identity(
        dms_id,
        row_hashes,
        model_id,
        model_revision,
        sources,
    )
    if len(ordered_sequences) != len(hashes):
        raise ValueError("sequence count must match ordered row-hash count")

    with _exclusive_cache_lock(json_path):
        array_exists = array_path.exists()
        metadata_exists = json_path.exists()
        if array_exists and metadata_exists:
            return load_embedding_cache(
                array_path,
                json_path,
                dms_id=dms_id,
                row_hashes=hashes,
                model_id=model_id,
                model_revision=model_revision,
                sources=sources,
                expected_embedding_dim=checked_dim,
            )
        if array_exists != metadata_exists:
            _remove_cache_pair(array_path, json_path)

        embeddings = embed_sequences(
            ordered_sequences,
            model_id=model_id,
            model_revision=model_revision,
            batch_size=checked_batch_size,
            device=device,
        )
        checked_embeddings = _validate_embedding_array(
            embeddings,
            row_count=len(hashes),
            embedding_dim=checked_dim,
        )
        write_embedding_cache(
            array_path,
            json_path,
            checked_embeddings,
            dms_id=dms_id,
            row_hashes=hashes,
            model_id=model_id,
            model_revision=model_revision,
            sources=sources,
        )
        return load_embedding_cache(
            array_path,
            json_path,
            dms_id=dms_id,
            row_hashes=row_hashes,
            model_id=model_id,
            model_revision=model_revision,
            sources=sources,
            expected_embedding_dim=checked_dim,
        )
