from __future__ import annotations

import copy
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd  # type: ignore[import-untyped]
import pytest
from pydantic import ValidationError

from self_improve_protein.config import Protocol
from self_improve_protein.crossfit_data import (
    CROSSFIT_CARD_ID,
    CROSSFIT_CARD_SHA256,
    CROSSFIT_POOL_SCHEMA_ID,
    CrossfitPoolManifest,
    build_crossfit_pool,
    load_crossfit_pool_manifest,
    validate_crossfit_pool_provenance,
    write_crossfit_pool_manifest,
)
from self_improve_protein.data import (
    DataManifest,
    ManifestSource,
    ManifestSources,
    SelectedAssayManifest,
    build_working_set,
    filter_usable_variants,
    load_assay_from_archives,
    write_data_manifest,
)
from self_improve_protein.provenance import sha256_file


@dataclass(frozen=True, slots=True)
class PoolFixture:
    protocol: Protocol
    base_manifest: DataManifest
    base_manifest_path: Path
    dms_zip: Path
    scores_zip: Path
    metadata_csv: Path
    eligible_ids: tuple[str, ...]


def _write_sources(tmp_path: Path) -> tuple[Path, Path, Path, tuple[str, ...]]:
    eligible_ids = tuple(f"ASSAY_{index:02d}" for index in range(35))
    dms_zip = tmp_path / "dms.zip"
    scores_zip = tmp_path / "scores.zip"
    metadata_csv = tmp_path / "metadata.csv"
    mutants = ("A1C", "A2D", "A3E", "A4F")
    sequences = ("ACDE", "ACDF", "ACDG", "ACDH")

    with (
        zipfile.ZipFile(dms_zip, "w") as dms_archive,
        zipfile.ZipFile(scores_zip, "w") as scores_archive,
    ):
        for assay_index, dms_id in reversed(tuple(enumerate(eligible_ids))):
            dms = pd.DataFrame(
                {
                    "mutant": mutants,
                    "mutated_sequence": sequences,
                    "DMS_score": [
                        float(assay_index + offset) for offset in range(len(mutants))
                    ],
                }
            )
            scores = pd.DataFrame(
                {
                    "mutant": mutants,
                    "ESM1v_ensemble": [0.1, 0.2, 0.3, 0.4],
                }
            )
            dms_archive.writestr(
                f"DMS_ProteinGym_substitutions/{dms_id}.csv",
                dms.to_csv(index=False),
            )
            scores_archive.writestr(dms_id + ".csv", scores.to_csv(index=False))

    pd.DataFrame(
        {"DMS_id": eligible_ids[::-1], "seq_len": [4] * len(eligible_ids)}
    ).to_csv(metadata_csv, index=False)
    return dms_zip, scores_zip, metadata_csv, eligible_ids


def _protocol(dms_zip: Path, scores_zip: Path, metadata_csv: Path) -> Protocol:
    return Protocol(
        data_release="v1.3",
        substitutions_url="https://example.test/dms.zip",
        zero_shot_scores_url="https://example.test/scores.zip",
        metadata_url="https://example.test/metadata.csv",
        proteingym_upstream_commit="1" * 40,
        substitutions_sha256=sha256_file(dms_zip),
        zero_shot_scores_sha256=sha256_file(scores_zip),
        metadata_sha256=sha256_file(metadata_csv),
        teacher_column="ESM1v_ensemble",
        model="facebook/esm2_t12_35M_UR50D",
        model_revision="6fbf070e65b0b7291e7bbcd451118c216cff79d8",
        working_size=3,
        n_labeled=1,
        n_unlabeled=1,
        n_test=1,
        q=1,
        pseudo_weight=0.1,
        ridge_lambda=0.01,
        damping=0.0001,
        seeds=(0,),
        assay_count=8,
        max_length=512,
        preprocessing={
            "feature_scaling": "scalar_rms",
            "student_fit": "no_intercept",
            "label_ddof": 0,
        },
        analysis_seed=20260629,
        random_diagnostic_replicates=100,
    )


def _rebuilt_record(
    dms_zip: Path,
    scores_zip: Path,
    protocol: Protocol,
    dms_id: str,
) -> SelectedAssayManifest:
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
    return SelectedAssayManifest(
        dms_id=dms_id,
        usable_count=len(usable),
        sequence_length=4,
        row_hashes=tuple(str(value) for value in working["sequence_hash"]),
    )


def _pool_fixture(tmp_path: Path) -> PoolFixture:
    dms_zip, scores_zip, metadata_csv, eligible_ids = _write_sources(tmp_path)
    protocol = _protocol(dms_zip, scores_zip, metadata_csv)
    confirmatory_ids = eligible_ids[:8]
    development_id = eligible_ids[8]
    base_manifest = DataManifest(
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
        confirmatory_ids=confirmatory_ids,
        development_id=development_id,
        max_length=protocol.max_length,
        working_size=protocol.working_size,
        selected_assays=tuple(
            _rebuilt_record(dms_zip, scores_zip, protocol, dms_id)
            for dms_id in (*confirmatory_ids, development_id)
        ),
    )
    base_manifest_path = tmp_path / "base-manifest.json"
    write_data_manifest(base_manifest_path, base_manifest)
    return PoolFixture(
        protocol=protocol,
        base_manifest=base_manifest,
        base_manifest_path=base_manifest_path,
        dms_zip=dms_zip,
        scores_zip=scores_zip,
        metadata_csv=metadata_csv,
        eligible_ids=eligible_ids,
    )


def _build(
    fixture: PoolFixture,
) -> tuple[CrossfitPoolManifest, dict[str, pd.DataFrame]]:
    return build_crossfit_pool(
        protocol=fixture.protocol,
        base_manifest_path=fixture.base_manifest_path,
        dms_zip=fixture.dms_zip,
        scores_zip=fixture.scores_zip,
        metadata_csv=fixture.metadata_csv,
    )


def test_card_binding_matches_committed_pre_outcome_card() -> None:
    card_path = Path("docs/research/experiment-card-crossfit.md")

    assert CROSSFIT_POOL_SCHEMA_ID == "self-improve-protein.crossfit-pool.v1"
    assert CROSSFIT_CARD_ID == "docs/research/experiment-card-crossfit.md"
    assert CROSSFIT_CARD_SHA256 == (
        "383afd7a5bae9c2ebd6768a112a82980236540fc0f66e3a294ef298961b8596f"
    )
    assert sha256_file(card_path) == CROSSFIT_CARD_SHA256


def test_builder_partitions_all_assays_and_emits_deterministic_records(
    tmp_path: Path,
) -> None:
    fixture = _pool_fixture(tmp_path)
    emitted: list[tuple[str, tuple[str, ...]]] = []

    first, first_frames = build_crossfit_pool(
        protocol=fixture.protocol,
        base_manifest_path=fixture.base_manifest_path,
        dms_zip=fixture.dms_zip,
        scores_zip=fixture.scores_zip,
        metadata_csv=fixture.metadata_csv,
        on_untouched_frame=lambda dms_id, frame: emitted.append(
            (dms_id, tuple(str(value) for value in frame["sequence_hash"]))
        ),
    )
    second, second_frames = _build(fixture)

    expected_screen = (fixture.eligible_ids[8], *fixture.eligible_ids[:8])
    expected_untouched = fixture.eligible_ids[9:]
    assert first.schema_id == CROSSFIT_POOL_SCHEMA_ID
    assert first.schema_version == 1
    assert first.card_id == CROSSFIT_CARD_ID
    assert first.card_sha256 == CROSSFIT_CARD_SHA256
    assert first.base_manifest_sha256 == sha256_file(fixture.base_manifest_path)
    assert len(first.protocol_sha256) == 64
    assert first.eligible_assay_ids == fixture.eligible_ids
    assert first.screen_ids == expected_screen
    assert first.untouched_ids == expected_untouched
    assert len(first.untouched_ids) == 26
    assert tuple(record.dms_id for record in first.selected_assays) == (
        fixture.eligible_ids
    )
    assert tuple(first_frames) == expected_untouched
    assert tuple(second_frames) == expected_untouched
    assert tuple(dms_id for dms_id, _ in emitted) == expected_untouched
    assert first == second
    assert [record.model_dump_json() for record in first.selected_assays] == [
        record.model_dump_json() for record in second.selected_assays
    ]
    for record in first.selected_assays:
        assert len(record.row_hashes) == fixture.protocol.working_size
    for dms_id, frame in first_frames.items():
        record = next(item for item in first.selected_assays if item.dms_id == dms_id)
        assert tuple(frame["sequence_hash"]) == record.row_hashes


def test_manifest_roundtrip_is_canonical_label_free_and_deeply_immutable(
    tmp_path: Path,
) -> None:
    fixture = _pool_fixture(tmp_path)
    manifest, _ = _build(fixture)
    first_path = tmp_path / "crossfit.json"
    second_path = tmp_path / "nested" / "crossfit.json"

    write_crossfit_pool_manifest(first_path, manifest)
    write_crossfit_pool_manifest(second_path, manifest)

    assert first_path.read_bytes() == second_path.read_bytes()
    assert load_crossfit_pool_manifest(first_path) == manifest
    assert "DMS_score" not in first_path.read_text(encoding="utf-8")
    with pytest.raises(ValidationError, match="frozen"):
        manifest.card_id = "changed"  # type: ignore[misc]
    with pytest.raises(ValidationError, match="frozen"):
        manifest.selected_assays[0].usable_count = 1  # type: ignore[misc]


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("card_sha256", "0" * 64, "card_sha256"),
        ("screen_ids", None, "screen_ids"),
        ("untouched_ids", None, "untouched_ids"),
        ("selected_assays", None, "selected_assays"),
    ],
)
def test_manifest_rejects_card_partition_and_record_order_tamper(
    tmp_path: Path,
    field: str,
    replacement: object,
    message: str,
) -> None:
    fixture = _pool_fixture(tmp_path)
    manifest, _ = _build(fixture)
    payload = manifest.model_dump(mode="json")
    if replacement is not None:
        payload[field] = replacement
    else:
        payload[field] = list(reversed(payload[field]))

    with pytest.raises(ValidationError, match=message):
        CrossfitPoolManifest.model_validate(payload)


def test_builder_rejects_source_content_tamper_before_archive_use(
    tmp_path: Path,
) -> None:
    fixture = _pool_fixture(tmp_path)
    with fixture.dms_zip.open("ab") as destination:
        destination.write(b"tamper")

    with pytest.raises(ValueError, match=r"source checksum mismatch.*substitutions"):
        _build(fixture)


@pytest.mark.parametrize("tamper_kind", ["row_hashes", "record_bytes"])
def test_builder_rejects_rebuilt_base_record_mismatch(
    tmp_path: Path,
    tamper_kind: str,
) -> None:
    fixture = _pool_fixture(tmp_path)
    payload = fixture.base_manifest.model_dump(mode="json")
    first = copy.deepcopy(payload["selected_assays"][0])
    if tamper_kind == "row_hashes":
        first["row_hashes"] = sorted(["0" * 64, *first["row_hashes"][1:]])
    else:
        first["usable_count"] += 1
    payload["selected_assays"][0] = first
    write_data_manifest(
        fixture.base_manifest_path,
        DataManifest.model_validate(payload),
    )

    expected = "row hashes" if tamper_kind == "row_hashes" else "record bytes"
    with pytest.raises(ValueError, match=expected):
        _build(fixture)


@pytest.mark.parametrize("tamper_kind", ["base_hash", "source_identity"])
def test_exact_provenance_validation_rejects_tamper(
    tmp_path: Path,
    tamper_kind: str,
) -> None:
    fixture = _pool_fixture(tmp_path)
    manifest, _ = _build(fixture)
    payload = manifest.model_dump(mode="json")
    if tamper_kind == "base_hash":
        payload["base_manifest_sha256"] = "f" * 64
    else:
        payload["sources"]["metadata"]["url"] = "https://tampered.test/metadata"
    tampered = CrossfitPoolManifest.model_validate(payload)

    expected = "base manifest SHA" if tamper_kind == "base_hash" else "source identity"
    with pytest.raises(ValueError, match=expected):
        validate_crossfit_pool_provenance(
            tampered,
            protocol=fixture.protocol,
            base_manifest_path=fixture.base_manifest_path,
        )


def test_loader_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.json"
    path.write_text(
        json.dumps({"schema_id": CROSSFIT_POOL_SCHEMA_ID})[:-1]
        + ', "schema_id": "duplicate"}',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"duplicate JSON key.*schema_id"):
        load_crossfit_pool_manifest(path)
