"""Deterministic hashes, random-stream seeds, and atomic JSON artifacts."""

import hashlib
import json
import os
import tempfile
from pathlib import Path

_HASH_CHUNK_SIZE = 1024 * 1024


def sha256_bytes(payload: bytes) -> str:
    """Return the lowercase SHA-256 hex digest of *payload*."""
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path | str) -> str:
    """Return a streaming SHA-256 hex digest for a file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(_HASH_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def derive_seed(dms_id: str, seed: int, purpose: str) -> int:
    """Derive a deterministic, purpose-separated unsigned 64-bit seed."""
    payload = f"{dms_id}\0{seed}\0{purpose}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def atomic_write_json(path: Path | str, payload: object) -> None:
    """Serialize finite JSON and atomically replace *path* with the result."""
    serialized = json.dumps(
        payload,
        allow_nan=False,
        indent=2,
        sort_keys=True,
    ) + "\n"
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, destination)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
