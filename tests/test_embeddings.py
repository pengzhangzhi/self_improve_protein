import hashlib
import json
import subprocess
import sys
import types
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from pydantic import ValidationError

from self_improve_protein import embeddings as embedding_module
from self_improve_protein.embeddings import (
    DEFAULT_MODEL_ID,
    DEFAULT_MODEL_REVISION,
    POOLING_DEFINITION,
    EmbeddingCacheMetadata,
    EmbeddingCacheSources,
    embed_sequences,
    embed_sequences_with_model,
    get_or_create_embedding_cache,
    load_embedding_cache,
    mean_pool_residues,
    ordered_row_hash_digest,
    write_embedding_cache,
)
from self_improve_protein.provenance import atomic_write_json, sha256_file


def _row_hash(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _sources() -> EmbeddingCacheSources:
    return EmbeddingCacheSources(
        proteingym_upstream_commit="1" * 40,
        substitutions_sha256="2" * 64,
        zero_shot_scores_sha256="3" * 64,
        metadata_sha256="4" * 64,
    )


def _cache_kwargs(row_hashes: Sequence[str]) -> dict[str, object]:
    return {
        "dms_id": "TINY",
        "row_hashes": tuple(row_hashes),
        "model_id": DEFAULT_MODEL_ID,
        "model_revision": DEFAULT_MODEL_REVISION,
        "sources": _sources(),
    }


class FakeTokenizer:
    def __init__(self) -> None:
        self.calls: list[tuple[tuple[str, ...], dict[str, object]]] = []

    def __call__(self, sequences: Sequence[str], **kwargs: object) -> dict[str, Any]:
        values = tuple(sequences)
        self.calls.append((values, dict(kwargs)))
        rows = [
            [101, *(ord(residue) - 64 for residue in value), 102]
            for value in values
        ]
        width = max(map(len, rows))
        input_ids = torch.zeros((len(rows), width), dtype=torch.int64)
        attention = torch.zeros_like(input_ids)
        special = torch.ones_like(input_ids)
        for index, row in enumerate(rows):
            input_ids[index, : len(row)] = torch.tensor(row)
            attention[index, : len(row)] = 1
            special[index, 1 : len(row) - 1] = 0
        return {
            "input_ids": input_ids,
            "attention_mask": attention,
            "special_tokens_mask": special,
        }


class FakeModel:
    def __init__(self) -> None:
        self.eval_calls = 0
        self.devices: list[torch.device] = []
        self.calls: list[dict[str, Any]] = []

    def eval(self) -> "FakeModel":
        self.eval_calls += 1
        return self

    def to(self, device: torch.device) -> "FakeModel":
        self.devices.append(torch.device(device))
        return self

    def __call__(self, **kwargs: Any) -> SimpleNamespace:
        assert torch.is_inference_mode_enabled()
        self.calls.append(kwargs)
        hidden = kwargs["input_ids"].to(torch.float32).unsqueeze(-1)
        return SimpleNamespace(last_hidden_state=hidden)


class AutocastCheckingFakeModel(FakeModel):
    def __call__(self, **kwargs: Any) -> SimpleNamespace:
        device_type = kwargs["input_ids"].device.type
        assert not torch.is_autocast_enabled(device_type)
        return super().__call__(**kwargs)


def test_mean_pool_excludes_bos_eos_and_padding() -> None:
    hidden = torch.tensor([[[100.0], [1.0], [3.0], [200.0], [300.0]]])
    attention = torch.tensor([[1, 1, 1, 1, 0]])
    special = torch.tensor([[1, 0, 0, 1, 1]])

    pooled = mean_pool_residues(hidden, attention, special)

    torch.testing.assert_close(pooled, torch.tensor([[2.0]]))
    assert pooled.dtype == torch.float32


def test_mean_pool_casts_floating_input_to_float32() -> None:
    hidden = torch.tensor([[[1.0, 3.0], [3.0, 5.0]]], dtype=torch.float64)
    attention = torch.ones((1, 2), dtype=torch.bool)
    special = torch.zeros((1, 2), dtype=torch.bool)

    pooled = mean_pool_residues(hidden, attention, special)

    assert pooled.dtype == torch.float32
    torch.testing.assert_close(pooled, torch.tensor([[2.0, 4.0]]))


@pytest.mark.parametrize(
    ("hidden", "attention", "special", "message"),
    [
        (torch.ones((2, 3)), torch.ones((2, 3)), torch.zeros((2, 3)), "3-D"),
        (
            torch.ones((2, 3, 4)),
            torch.ones((2, 3, 1)),
            torch.zeros((2, 3)),
            "2-D",
        ),
        (
            torch.ones((2, 3, 4)),
            torch.ones((2, 2)),
            torch.zeros((2, 3)),
            "shape",
        ),
        (
            torch.ones((2, 3, 4), dtype=torch.int64),
            torch.ones((2, 3)),
            torch.zeros((2, 3)),
            "floating",
        ),
    ],
)
def test_mean_pool_rejects_malformed_shapes_and_hidden_dtype(
    hidden: torch.Tensor,
    attention: torch.Tensor,
    special: torch.Tensor,
    message: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        mean_pool_residues(hidden, attention, special)


@pytest.mark.parametrize(
    ("attention", "special"),
    [
        (torch.tensor([[1.0, 1.0]]), torch.tensor([[0, 0]])),
        (torch.tensor([[1, 2]]), torch.tensor([[0, 0]])),
        (torch.tensor([[1, 1]]), torch.tensor([[0, -1]])),
    ],
)
def test_mean_pool_rejects_non_boolean_masks(
    attention: torch.Tensor,
    special: torch.Tensor,
) -> None:
    with pytest.raises((TypeError, ValueError), match="mask"):
        mean_pool_residues(torch.ones((1, 2, 1)), attention, special)


def test_mean_pool_rejects_device_mismatch_before_computation() -> None:
    hidden = torch.ones((1, 2, 1))
    attention = torch.ones((1, 2), dtype=torch.int64)
    special = torch.zeros((1, 2), dtype=torch.int64, device="meta")

    with pytest.raises(ValueError, match="device"):
        mean_pool_residues(hidden, attention, special)


def test_mean_pool_rejects_zero_residue_rows_and_nonfinite_hidden() -> None:
    hidden = torch.ones((2, 2, 1))
    attention = torch.tensor([[1, 1], [1, 0]])
    special = torch.tensor([[1, 1], [0, 1]])

    with pytest.raises(ValueError, match="zero residue"):
        mean_pool_residues(hidden, attention, special)

    hidden[0, 0, 0] = torch.nan
    with pytest.raises(ValueError, match="non-finite"):
        mean_pool_residues(
            hidden,
            torch.ones((2, 2), dtype=torch.int64),
            torch.zeros((2, 2), dtype=torch.int64),
        )


def test_mean_pool_rejects_nonfinite_output_from_finite_overflow() -> None:
    largest = torch.finfo(torch.float32).max
    hidden = torch.full((1, 2, 1), largest)

    with pytest.raises(ValueError, match=r"mean-pooled.*non-finite"):
        mean_pool_residues(
            hidden,
            torch.ones((1, 2), dtype=torch.int64),
            torch.zeros((1, 2), dtype=torch.int64),
        )


def test_batch_embedding_preserves_order_and_uses_residue_pooling() -> None:
    tokenizer = FakeTokenizer()
    model = FakeModel()
    sequences = ("A", "CD", "E", "FG", "H")

    embeddings = embed_sequences_with_model(
        sequences,
        tokenizer=tokenizer,
        model=model,
        batch_size=2,
        device="cpu",
    )

    np.testing.assert_array_equal(
        embeddings[:, 0],
        np.array([1.0, 3.5, 5.0, 6.5, 8.0], dtype=np.float32),
    )
    assert embeddings.dtype == np.float32
    assert [call[0] for call in tokenizer.calls] == [
        ("A", "CD"),
        ("E", "FG"),
        ("H",),
    ]
    assert model.eval_calls == 1
    assert model.devices == [torch.device("cpu")]


def test_batch_embedding_locks_tokenizer_and_model_call_contract() -> None:
    tokenizer = FakeTokenizer()
    model = FakeModel()

    embed_sequences_with_model(
        ("AC",),
        tokenizer=tokenizer,
        model=model,
        batch_size=1,
        device="cpu",
    )

    _, tokenizer_kwargs = tokenizer.calls[0]
    assert tokenizer_kwargs == {
        "add_special_tokens": True,
        "padding": True,
        "return_attention_mask": True,
        "return_special_tokens_mask": True,
        "return_tensors": "pt",
        "truncation": False,
    }
    assert len(model.calls) == 1
    assert set(model.calls[0]) == {"input_ids", "attention_mask", "return_dict"}
    assert model.calls[0]["return_dict"] is True
    assert "special_tokens_mask" not in model.calls[0]


def test_batch_embedding_disables_ambient_device_autocast() -> None:
    model = AutocastCheckingFakeModel()

    with torch.autocast("cpu", enabled=True):
        assert torch.is_autocast_enabled("cpu")
        actual = embed_sequences_with_model(
            ("AC",),
            tokenizer=FakeTokenizer(),
            model=model,
            batch_size=1,
            device="cpu",
        )

    assert actual.dtype == np.float32
    assert model.eval_calls == 1


@pytest.mark.parametrize("batch_size", [0, -1, True, 1.5])
def test_batch_embedding_rejects_invalid_batch_size(batch_size: object) -> None:
    with pytest.raises((TypeError, ValueError), match="batch_size"):
        embed_sequences_with_model(
            ("AC",),
            tokenizer=FakeTokenizer(),
            model=FakeModel(),
            batch_size=batch_size,  # type: ignore[arg-type]
            device="cpu",
        )


@pytest.mark.parametrize("sequences", [(), ("",), ("AC", 1), "AC"])
def test_batch_embedding_rejects_empty_or_malformed_sequences(
    sequences: object,
) -> None:
    with pytest.raises((TypeError, ValueError), match="sequence"):
        embed_sequences_with_model(
            sequences,  # type: ignore[arg-type]
            tokenizer=FakeTokenizer(),
            model=FakeModel(),
            batch_size=2,
            device="cpu",
        )


def test_real_embedding_path_uses_exact_hf_revision_and_no_pooler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokenizer = FakeTokenizer()
    model = FakeModel()
    calls: dict[str, tuple[tuple[object, ...], dict[str, object]]] = {}

    class AutoTokenizer:
        @classmethod
        def from_pretrained(cls, *args: object, **kwargs: object) -> FakeTokenizer:
            calls["tokenizer"] = (args, kwargs)
            return tokenizer

    class AutoModel:
        @classmethod
        def from_pretrained(cls, *args: object, **kwargs: object) -> FakeModel:
            calls["model"] = (args, kwargs)
            return model

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoTokenizer = AutoTokenizer  # type: ignore[attr-defined]
    fake_transformers.AutoModel = AutoModel  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    actual = embed_sequences(("AC",), batch_size=1, device="cpu")

    assert actual.dtype == np.float32
    assert calls == {
        "tokenizer": (
            (DEFAULT_MODEL_ID,),
            {"revision": DEFAULT_MODEL_REVISION},
        ),
        "model": (
            (DEFAULT_MODEL_ID,),
            {
                "revision": DEFAULT_MODEL_REVISION,
                "add_pooling_layer": False,
            },
        ),
    }


def test_embedding_module_import_does_not_import_transformers() -> None:
    code = """
import builtins
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name == 'transformers' or name.startswith('transformers.'):
        raise AssertionError('transformers imported eagerly')
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
import self_improve_protein.embeddings
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_ordered_row_hash_digest_is_order_sensitive_and_strict() -> None:
    hashes = (_row_hash("a"), _row_hash("b"))
    expected = hashlib.sha256(b"".join(bytes.fromhex(value) for value in hashes))

    assert ordered_row_hash_digest(hashes) == expected.hexdigest()
    assert ordered_row_hash_digest(hashes[::-1]) != expected.hexdigest()
    with pytest.raises(ValueError, match="row_hash"):
        ordered_row_hash_digest(())
    with pytest.raises(ValueError, match="row_hash"):
        ordered_row_hash_digest(("A" * 64,))


def test_embedding_cache_roundtrip_and_bytes_are_deterministic(tmp_path: Path) -> None:
    embeddings = np.arange(12, dtype=np.float32).reshape(3, 4)
    row_hashes = tuple(_row_hash(str(index)) for index in range(3))
    first_npy = tmp_path / "first.npy"
    first_json = tmp_path / "first.json"
    second_npy = tmp_path / "second.npy"
    second_json = tmp_path / "second.json"

    first = write_embedding_cache(
        first_npy,
        first_json,
        embeddings,
        **_cache_kwargs(row_hashes),
    )
    second = write_embedding_cache(
        second_npy,
        second_json,
        embeddings.copy(),
        **_cache_kwargs(row_hashes),
    )
    loaded = load_embedding_cache(
        first_npy,
        first_json,
        expected_embedding_dim=4,
        **_cache_kwargs(row_hashes),
    )

    np.testing.assert_array_equal(loaded, embeddings)
    assert loaded.dtype == np.float32
    assert first == second
    assert first_npy.read_bytes() == second_npy.read_bytes()
    assert first_json.read_bytes() == second_json.read_bytes()
    assert first.npy_sha256 == sha256_file(first_npy)
    assert first.shape == (3, 4)
    assert first.dtype == "float32"
    assert first.pooling == POOLING_DEFINITION
    assert first.row_hash_digest == ordered_row_hash_digest(row_hashes)


def test_embedding_cache_hit_does_not_recompute(tmp_path: Path) -> None:
    row_hashes = (_row_hash("a"), _row_hash("b"))
    npy_path = tmp_path / "cache.npy"
    json_path = tmp_path / "cache.json"
    expected = np.array([[1.0], [2.0]], dtype=np.float32)
    write_embedding_cache(
        npy_path,
        json_path,
        expected,
        **_cache_kwargs(row_hashes),
    )

    def forbidden() -> np.ndarray:
        raise AssertionError("cache hit recomputed embeddings")

    actual = get_or_create_embedding_cache(
        npy_path,
        json_path,
        expected_embedding_dim=1,
        compute=forbidden,
        **_cache_kwargs(row_hashes),
    )

    np.testing.assert_array_equal(actual, expected)


def test_embedding_cache_miss_computes_once_and_validates(tmp_path: Path) -> None:
    row_hashes = (_row_hash("a"), _row_hash("b"))
    calls = 0
    expected = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

    def compute() -> np.ndarray:
        nonlocal calls
        calls += 1
        return expected.copy()

    actual = get_or_create_embedding_cache(
        tmp_path / "cache.npy",
        tmp_path / "cache.json",
        expected_embedding_dim=2,
        compute=compute,
        **_cache_kwargs(row_hashes),
    )

    assert calls == 1
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    "mismatch",
    ["dms_id", "row_order", "model_id", "model_revision", "sources"],
)
def test_embedding_cache_rejects_identity_or_source_mismatch(
    tmp_path: Path,
    mismatch: str,
) -> None:
    row_hashes = (_row_hash("a"), _row_hash("b"))
    npy_path = tmp_path / "cache.npy"
    json_path = tmp_path / "cache.json"
    write_embedding_cache(
        npy_path,
        json_path,
        np.ones((2, 3), dtype=np.float32),
        **_cache_kwargs(row_hashes),
    )
    kwargs = _cache_kwargs(row_hashes)
    if mismatch == "dms_id":
        kwargs["dms_id"] = "OTHER"
    elif mismatch == "row_order":
        kwargs["row_hashes"] = row_hashes[::-1]
    elif mismatch == "model_id":
        kwargs["model_id"] = "other/model"
    elif mismatch == "model_revision":
        kwargs["model_revision"] = "a" * 40
    elif mismatch == "sources":
        kwargs["sources"] = _sources().model_copy(
            update={"metadata_sha256": "f" * 64}
        )

    with pytest.raises(ValueError, match="metadata mismatch"):
        load_embedding_cache(
            npy_path,
            json_path,
            expected_embedding_dim=3,
            **kwargs,
        )


def test_embedding_cache_rejects_missing_or_incomplete_pair(tmp_path: Path) -> None:
    row_hashes = (_row_hash("a"),)
    npy_path = tmp_path / "cache.npy"
    json_path = tmp_path / "cache.json"
    np.save(npy_path, np.ones((1, 1), dtype=np.float32), allow_pickle=False)

    with pytest.raises(ValueError, match="incomplete"):
        load_embedding_cache(
            npy_path,
            json_path,
            expected_embedding_dim=1,
            **_cache_kwargs(row_hashes),
        )


def test_embedding_cache_rejects_corrupt_npy_and_json(tmp_path: Path) -> None:
    row_hashes = (_row_hash("a"),)
    npy_path = tmp_path / "cache.npy"
    json_path = tmp_path / "cache.json"
    write_embedding_cache(
        npy_path,
        json_path,
        np.ones((1, 2), dtype=np.float32),
        **_cache_kwargs(row_hashes),
    )
    npy_path.write_bytes(npy_path.read_bytes()[:20])
    with pytest.raises(ValueError, match="checksum"):
        load_embedding_cache(
            npy_path,
            json_path,
            expected_embedding_dim=2,
            **_cache_kwargs(row_hashes),
        )

    json_path.write_text("{truncated", encoding="utf-8")
    with pytest.raises((json.JSONDecodeError, ValueError)):
        load_embedding_cache(
            npy_path,
            json_path,
            expected_embedding_dim=2,
            **_cache_kwargs(row_hashes),
        )


def test_embedding_cache_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    row_hashes = (_row_hash("a"),)
    npy_path = tmp_path / "cache.npy"
    json_path = tmp_path / "cache.json"
    write_embedding_cache(
        npy_path,
        json_path,
        np.ones((1, 2), dtype=np.float32),
        **_cache_kwargs(row_hashes),
    )
    serialized = json_path.read_text(encoding="utf-8")
    marker = '  "schema_version": 1,\n'
    json_path.write_text(serialized.replace(marker, marker + marker), encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate JSON key"):
        load_embedding_cache(
            npy_path,
            json_path,
            expected_embedding_dim=2,
            **_cache_kwargs(row_hashes),
        )


@pytest.mark.parametrize(
    "array",
    [
        np.ones((2, 3), dtype=np.float64),
        np.array([[1.0], [np.nan]], dtype=np.float32),
        np.ones((2,), dtype=np.float32),
        np.ones((0, 3), dtype=np.float32),
    ],
)
def test_embedding_cache_write_rejects_wrong_dtype_shape_or_nonfinite(
    tmp_path: Path,
    array: np.ndarray,
) -> None:
    row_hashes = (_row_hash("a"), _row_hash("b"))

    with pytest.raises((TypeError, ValueError)):
        write_embedding_cache(
            tmp_path / "cache.npy",
            tmp_path / "cache.json",
            array,
            **_cache_kwargs(row_hashes),
        )


@pytest.mark.parametrize("mutation", ["nonfinite", "wrong_shape"])
def test_embedding_cache_load_rejects_consistently_rehashed_bad_array(
    tmp_path: Path,
    mutation: str,
) -> None:
    row_hashes = (_row_hash("a"), _row_hash("b"))
    npy_path = tmp_path / "cache.npy"
    json_path = tmp_path / "cache.json"
    metadata = write_embedding_cache(
        npy_path,
        json_path,
        np.ones((2, 3), dtype=np.float32),
        **_cache_kwargs(row_hashes),
    )
    bad = (
        np.array([[np.inf, 1.0, 1.0], [1.0, 1.0, 1.0]], dtype=np.float32)
        if mutation == "nonfinite"
        else np.ones((2, 4), dtype=np.float32)
    )
    with npy_path.open("wb") as handle:
        np.save(handle, bad, allow_pickle=False)
    metadata = metadata.model_copy(
        update={
            "shape": bad.shape,
            "npy_sha256": sha256_file(npy_path),
        }
    )
    atomic_write_json(json_path, metadata.model_dump(mode="json"))

    message = "non-finite" if mutation == "nonfinite" else "metadata mismatch"
    with pytest.raises(ValueError, match=message):
        load_embedding_cache(
            npy_path,
            json_path,
            expected_embedding_dim=3,
            **_cache_kwargs(row_hashes),
        )


def test_embedding_cache_atomic_array_failure_cleans_temp_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_replace(source: object, destination: object) -> None:
        raise OSError("simulated atomic replacement failure")

    monkeypatch.setattr(embedding_module.os, "replace", fail_replace)

    with pytest.raises(OSError, match="simulated"):
        write_embedding_cache(
            tmp_path / "cache.npy",
            tmp_path / "cache.json",
            np.ones((1, 2), dtype=np.float32),
            **_cache_kwargs((_row_hash("a"),)),
        )

    assert list(tmp_path.iterdir()) == []


def test_embedding_cache_metadata_failure_leaves_only_rejected_incomplete_pair(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    npy_path = tmp_path / "cache.npy"
    json_path = tmp_path / "cache.json"
    row_hashes = (_row_hash("a"),)

    def fail_metadata(path: object, payload: object) -> None:
        raise OSError("simulated metadata replacement failure")

    monkeypatch.setattr(embedding_module, "atomic_write_json", fail_metadata)
    with pytest.raises(OSError, match="metadata"):
        write_embedding_cache(
            npy_path,
            json_path,
            np.ones((1, 2), dtype=np.float32),
            **_cache_kwargs(row_hashes),
        )

    assert npy_path.is_file()
    assert not json_path.exists()
    assert not list(tmp_path.glob("*.tmp"))
    with pytest.raises(ValueError, match="incomplete"):
        load_embedding_cache(
            npy_path,
            json_path,
            expected_embedding_dim=2,
            **_cache_kwargs(row_hashes),
        )


def test_embedding_cache_metadata_is_strict_deeply_immutable_and_extra_forbid() -> None:
    row_hashes = (_row_hash("a"),)
    payload = {
        "schema_version": 1,
        "dms_id": "TINY",
        "row_hash_digest": ordered_row_hash_digest(row_hashes),
        "row_count": 1,
        "model_id": DEFAULT_MODEL_ID,
        "model_revision": DEFAULT_MODEL_REVISION,
        "pooling": POOLING_DEFINITION,
        "shape": [1, 2],
        "dtype": "float32",
        "sources": _sources().model_dump(mode="json"),
        "npy_sha256": "5" * 64,
    }
    metadata = EmbeddingCacheMetadata.model_validate(payload)

    with pytest.raises(ValidationError, match="frozen"):
        metadata.dms_id = "OTHER"  # type: ignore[misc]
    with pytest.raises(ValidationError, match="frozen"):
        metadata.sources.metadata_sha256 = "6" * 64  # type: ignore[misc]
    with pytest.raises(ValidationError, match="extra"):
        EmbeddingCacheMetadata.model_validate({**payload, "unexpected": True})
    assert isinstance(metadata.shape, tuple)

    for field, invalid in [
        ("schema_version", True),
        ("row_count", "1"),
        ("shape", [1, True]),
        ("model_revision", "A" * 40),
        ("npy_sha256", "A" * 64),
    ]:
        malformed = {**payload, field: invalid}
        with pytest.raises(ValidationError):
            EmbeddingCacheMetadata.model_validate(malformed)


def test_embedding_cache_metadata_validates_shape_against_row_count() -> None:
    row_hashes = (_row_hash("a"),)
    payload = {
        "schema_version": 1,
        "dms_id": "TINY",
        "row_hash_digest": ordered_row_hash_digest(row_hashes),
        "row_count": 1,
        "model_id": DEFAULT_MODEL_ID,
        "model_revision": DEFAULT_MODEL_REVISION,
        "pooling": POOLING_DEFINITION,
        "shape": [2, 3],
        "dtype": "float32",
        "sources": _sources().model_dump(mode="json"),
        "npy_sha256": "5" * 64,
    }

    with pytest.raises(ValidationError, match="row_count"):
        EmbeddingCacheMetadata.model_validate(payload)
