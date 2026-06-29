import hashlib
import json
from pathlib import Path

import pytest

from self_improve_protein import provenance
from self_improve_protein.provenance import (
    atomic_write_json,
    derive_seed,
    sha256_bytes,
    sha256_file,
)


def test_seed_derivation_matches_locked_sha256_formula() -> None:
    dms_id = "ADRB2_HUMAN_Jones_2020"
    seed = 3
    purpose = "split"
    expected = int.from_bytes(
        hashlib.sha256(f"{dms_id}\0{seed}\0{purpose}".encode()).digest()[:8],
        "little",
    )

    assert derive_seed(dms_id, seed, purpose) == expected


def test_seed_derivation_is_deterministic_and_purpose_separated() -> None:
    split = derive_seed("ADRB2_HUMAN_Jones_2020", 0, "split")
    random_selection = derive_seed(
        "ADRB2_HUMAN_Jones_2020", 0, "random_selection"
    )

    assert split == derive_seed("ADRB2_HUMAN_Jones_2020", 0, "split")
    assert split != random_selection


def test_sha256_helpers_are_stable_and_file_matches_bytes(tmp_path: Path) -> None:
    payload = b"protein\x00fitness\n"
    artifact = tmp_path / "payload.bin"
    artifact.write_bytes(payload)
    expected = hashlib.sha256(payload).hexdigest()

    assert sha256_bytes(payload) == expected
    assert sha256_bytes(payload) == sha256_bytes(payload)
    assert sha256_file(artifact) == expected
    assert sha256_file(artifact) == sha256_file(artifact)


def test_atomic_json_write_creates_parents_and_round_trips(tmp_path: Path) -> None:
    destination = tmp_path / "nested" / "report.json"
    payload = {"z": [3, 2, 1], "a": {"finite": 1.25}}

    atomic_write_json(destination, payload)

    assert json.loads(destination.read_text(encoding="utf-8")) == payload
    assert destination.read_text(encoding="utf-8") == (
        '{\n  "a": {\n    "finite": 1.25\n  },\n  "z": [\n'
        "    3,\n    2,\n    1\n  ]\n}\n"
    )


def test_atomic_json_write_replaces_existing_file_without_temp_file(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "report.json"
    destination.write_text('{"old": true}\n', encoding="utf-8")

    atomic_write_json(destination, {"new": True})

    assert json.loads(destination.read_text(encoding="utf-8")) == {"new": True}
    assert list(tmp_path.iterdir()) == [destination]


def test_atomic_json_write_fsyncs_destination_local_temp_before_replace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "nested" / "report.json"
    events: list[str] = []
    replace_sources: list[Path] = []
    real_fsync = provenance.os.fsync
    real_replace = provenance.os.replace

    def recording_fsync(file_descriptor: int) -> None:
        events.append("fsync")
        real_fsync(file_descriptor)

    def recording_replace(source: Path, target: Path) -> None:
        events.append("replace")
        replace_sources.append(Path(source))
        real_replace(source, target)

    monkeypatch.setattr(provenance.os, "fsync", recording_fsync)
    monkeypatch.setattr(provenance.os, "replace", recording_replace)

    atomic_write_json(destination, {"ready": True})

    assert events == ["fsync", "replace"]
    assert len(replace_sources) == 1
    assert replace_sources[0].parent == destination.parent


def test_atomic_json_write_cleans_destination_local_temp_after_replace_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "report.json"
    original = '{"preserved": true}\n'
    destination.write_text(original, encoding="utf-8")
    fsync_called = False
    captured_source: Path | None = None
    real_fsync = provenance.os.fsync

    def recording_fsync(file_descriptor: int) -> None:
        nonlocal fsync_called
        fsync_called = True
        real_fsync(file_descriptor)

    def failing_replace(source: Path, target: Path) -> None:
        nonlocal captured_source
        captured_source = Path(source)
        assert Path(target) == destination
        assert json.loads(captured_source.read_text(encoding="utf-8")) == {
            "replacement": True
        }
        raise OSError("simulated replace failure")

    monkeypatch.setattr(provenance.os, "fsync", recording_fsync)
    monkeypatch.setattr(provenance.os, "replace", failing_replace)

    with pytest.raises(OSError, match="simulated replace failure"):
        atomic_write_json(destination, {"replacement": True})

    assert fsync_called
    assert destination.read_text(encoding="utf-8") == original
    assert captured_source is not None
    assert captured_source.parent == destination.parent
    assert not captured_source.exists()
    assert list(tmp_path.iterdir()) == [destination]


@pytest.mark.parametrize("non_finite", [float("nan"), float("inf"), -float("inf")])
def test_atomic_json_write_rejects_non_finite_values_and_cleans_temp(
    tmp_path: Path, non_finite: float
) -> None:
    destination = tmp_path / "report.json"
    destination.write_text('{"preserved": true}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="JSON"):
        atomic_write_json(destination, {"bad": non_finite})

    assert json.loads(destination.read_text(encoding="utf-8")) == {
        "preserved": True
    }
    assert list(tmp_path.iterdir()) == [destination]
