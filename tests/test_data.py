import hashlib
import inspect
import json
import zipfile
from dataclasses import FrozenInstanceError
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from self_improve_protein import data as data_module
from self_improve_protein.data import (
    AssayEligibility,
    DataManifest,
    ManifestSource,
    ManifestSources,
    SelectedAssayManifest,
    SplitIndices,
    build_working_set,
    filter_usable_variants,
    load_assay_from_archives,
    load_data_manifest,
    make_split,
    merge_assay_frames,
    row_hash,
    select_eligible_assays,
    write_data_manifest,
)
from self_improve_protein.provenance import derive_seed

TEACHER = "ESM1v_ensemble"
FIXTURES = Path(__file__).parent / "fixtures"


def _tiny_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    return (
        pd.read_csv(FIXTURES / "dms_tiny.csv"),
        pd.read_csv(FIXTURES / "scores_tiny.csv"),
    )


def _usable_frame(size: int = 24, dms_id: str = "TINY") -> pd.DataFrame:
    alphabet = "ACDEFGHIKLMNPQRSTVWY"
    mutants = [f"A{i + 1}{alphabet[i % len(alphabet)]}" for i in range(size)]
    sequences = [
        "ACDE"
        + alphabet[(i // len(alphabet)) % len(alphabet)]
        + alphabet[i % len(alphabet)]
        for i in range(size)
    ]
    frame = pd.DataFrame(
        {
            "mutant": mutants,
            "mutated_sequence": sequences,
            "DMS_score": np.linspace(-1.0, 1.0, size),
            TEACHER: np.linspace(2.0, 3.0, size),
            "dms_id": dms_id,
            "source_row": np.arange(size, dtype=np.int64),
        }
    )
    return filter_usable_variants(frame, TEACHER, max_length=512)


def _split_bytes(split: SplitIndices) -> bytes:
    index_parts = (
        split.labeled,
        split.unlabeled,
        split.test,
        split.buffer,
    )
    hash_parts = (
        split.labeled_sequence_hashes,
        split.unlabeled_sequence_hashes,
        split.test_sequence_hashes,
        split.buffer_sequence_hashes,
    )
    indices = b"".join(
        np.asarray(part, dtype=np.int64).tobytes() for part in index_parts
    )
    hashes = "\0".join(value for part in hash_parts for value in part).encode()
    return indices + hashes


def _manifest_row_hashes(dms_id: str, size: int) -> tuple[str, ...]:
    hashes = [
        row_hash(dms_id, f"A{index + 1}C", f"ACDE{index}A") for index in range(size)
    ]
    return tuple(sorted(hashes))


def _tiny_manifest(working_size: int = 3) -> DataManifest:
    selected_ids = ("ASSAY_A", "ASSAY_B", "ASSAY_C")
    return DataManifest(
        schema_version=1,
        data_release="ProteinGym-v1.3",
        teacher_column=TEACHER,
        sources=ManifestSources(
            substitutions=ManifestSource(
                url="https://example.test/substitutions.zip",
                sha256="a" * 64,
            ),
            scores=ManifestSource(
                url="https://example.test/scores.zip",
                sha256="b" * 64,
            ),
            metadata=ManifestSource(
                url="https://example.test/metadata.csv",
                sha256="c" * 64,
            ),
        ),
        upstream_revision="proteingym@abc123",
        eligible_assay_ids=("ASSAY_A", "ASSAY_B", "ASSAY_C", "ASSAY_D"),
        confirmatory_ids=("ASSAY_A", "ASSAY_B"),
        development_id="ASSAY_C",
        max_length=512,
        working_size=working_size,
        selected_assays=tuple(
            SelectedAssayManifest(
                dms_id=dms_id,
                usable_count=working_size + 2,
                sequence_length=6,
                row_hashes=_manifest_row_hashes(dms_id, working_size),
            )
            for dms_id in selected_ids
        ),
    )


def test_row_hash_matches_exact_utf8_nul_separated_formula() -> None:
    dms_id = "ADRB2_HUMAN_Jones_2020"
    mutant = "A12C"
    sequence = "ACDEFGHIKLMNPQRSTVWY"
    expected = hashlib.sha256(f"{dms_id}\0{mutant}\0{sequence}".encode()).hexdigest()

    actual = row_hash(dms_id, mutant, sequence)

    assert actual == expected
    assert len(actual) == 64
    assert actual == actual.lower()
    assert set(actual) <= set("0123456789abcdef")
    assert tuple(inspect.signature(row_hash).parameters) == (
        "dms_id",
        "mutant",
        "mutated_sequence",
    )


def test_merge_is_one_to_one_inner_join_and_retains_dms_source_identity() -> None:
    dms, scores = _tiny_frames()
    scores = scores.iloc[[2, 0, 3]].reset_index(drop=True)

    merged = merge_assay_frames(dms, scores, "TINY", TEACHER)

    assert merged.columns.tolist() == [
        "mutant",
        "mutated_sequence",
        "DMS_score",
        "DMS_score_bin",
        "source_row",
        "dms_id",
        TEACHER,
    ]
    assert merged["mutant"].tolist() == ["A1C", "A3E", "A4F"]
    assert merged["source_row"].tolist() == [0, 2, 3]
    assert merged["dms_id"].tolist() == ["TINY"] * 3
    assert merged[TEACHER].tolist() == pytest.approx([0.1, 0.3, 0.4])
    assert "unused_teacher" not in merged


@pytest.mark.parametrize("side", ["dms", "scores"])
def test_merge_rejects_duplicate_mutant_keys(side: str) -> None:
    dms, scores = _tiny_frames()
    if side == "dms":
        dms = pd.concat([dms, dms.iloc[[0]]], ignore_index=True)
    else:
        scores = pd.concat([scores, scores.iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match=r"duplicate.*mutant"):
        merge_assay_frames(dms, scores, "TINY", TEACHER)


@pytest.mark.parametrize(
    ("side", "column"),
    [
        ("dms", "mutant"),
        ("dms", "mutated_sequence"),
        ("dms", "DMS_score"),
        ("scores", "mutant"),
        ("scores", TEACHER),
    ],
)
def test_merge_rejects_missing_required_columns(side: str, column: str) -> None:
    dms, scores = _tiny_frames()
    if side == "dms":
        dms = dms.drop(columns=column)
    else:
        scores = scores.drop(columns=column)

    with pytest.raises(ValueError, match=column):
        merge_assay_frames(dms, scores, "TINY", TEACHER)


def test_merge_missing_teacher_coverage_is_explicitly_dropped_by_inner_join() -> None:
    dms, scores = _tiny_frames()
    scores = scores.loc[scores["mutant"] != "A2D"]

    merged = merge_assay_frames(dms, scores, "TINY", TEACHER)

    assert merged["mutant"].tolist() == ["A1C", "A3E", "A4F"]
    assert len(merged) == 3


def test_filter_applies_numeric_finite_canonical_nonempty_and_length_rules() -> None:
    frame = pd.DataFrame(
        {
            "mutant": ["A1C", "A2D", "A3E", "A4F", "A5G", "A6H", "A7I"],
            "mutated_sequence": [
                "ACDE",
                "ACDF",
                "ACDG",
                "",
                "ACDX",
                "acde",
                "ACDEG",
            ],
            "DMS_score": ["1.25", np.nan, 3.0, 4.0, 5.0, 6.0, 7.0],
            TEACHER: ["-0.5", 0.2, np.inf, 0.4, 0.5, 0.6, 0.7],
            "dms_id": ["TINY"] * 7,
            "source_row": range(7),
        }
    )

    usable = filter_usable_variants(frame, TEACHER, max_length=4)

    assert usable["mutant"].tolist() == ["A1C"]
    assert usable["DMS_score"].tolist() == pytest.approx([1.25])
    assert usable[TEACHER].tolist() == pytest.approx([-0.5])
    assert usable["sequence_hash"].tolist() == [row_hash("TINY", "A1C", "ACDE")]


def test_filter_rejects_max_length_above_locked_proteingym_limit() -> None:
    frame = pd.DataFrame(
        {
            "mutant": ["A1C"],
            "mutated_sequence": ["A" * 513],
            "DMS_score": [1.0],
            TEACHER: [2.0],
            "dms_id": ["TINY"],
        }
    )

    with pytest.raises(ValueError, match=r"max_length.*512"):
        filter_usable_variants(frame, TEACHER, max_length=513)


@pytest.mark.parametrize("column", ["DMS_score", TEACHER])
def test_filter_rejects_non_numeric_garbage_instead_of_silently_dropping(
    column: str,
) -> None:
    frame = _usable_frame(4).drop(columns="sequence_hash")
    frame[column] = frame[column].astype(object)
    frame.loc[0, column] = "not-a-number"

    with pytest.raises(ValueError, match=f"non-numeric.*{column}"):
        filter_usable_variants(frame, TEACHER, max_length=512)


def test_filter_deduplicates_sequence_by_smallest_row_hash() -> None:
    duplicate_sequence = "ACDE"
    mutants = ["A1C", "A2D"]
    expected_mutant = min(
        mutants,
        key=lambda mutant: row_hash("TINY", mutant, duplicate_sequence),
    )
    frame = pd.DataFrame(
        {
            "mutant": [*mutants, "A3E"],
            "mutated_sequence": [duplicate_sequence, duplicate_sequence, "ACDF"],
            "DMS_score": [100.0, -100.0, 0.0],
            TEACHER: [-500.0, 500.0, 1.0],
            "dms_id": ["TINY"] * 3,
            "source_row": [0, 1, 2],
        }
    )

    first = filter_usable_variants(frame, TEACHER, max_length=512)
    second = filter_usable_variants(
        frame.iloc[::-1].reset_index(drop=True), TEACHER, max_length=512
    )

    assert first.loc[
        first["mutated_sequence"] == duplicate_sequence, "mutant"
    ].item() == (expected_mutant)
    assert first["sequence_hash"].tolist() == second["sequence_hash"].tolist()
    assert first["sequence_hash"].tolist() == sorted(first["sequence_hash"])


def test_filter_hash_selection_ignores_source_order_labels_and_teacher_values() -> None:
    base = _usable_frame(12).drop(columns="sequence_hash")
    changed = base.iloc[[8, 1, 10, 3, 6, 0, 11, 2, 9, 5, 7, 4]].reset_index(drop=True)
    changed["DMS_score"] = base["DMS_score"].to_numpy()[::-1]
    changed[TEACHER] = np.roll(base[TEACHER].to_numpy(), 3)

    first = filter_usable_variants(base, TEACHER, max_length=512)
    second = filter_usable_variants(changed, TEACHER, max_length=512)

    assert first["sequence_hash"].tolist() == second["sequence_hash"].tolist()
    assert first["mutated_sequence"].tolist() == second["mutated_sequence"].tolist()


def test_build_working_set_has_exact_size_and_hash_order() -> None:
    usable = _usable_frame(14).sample(frac=1.0, random_state=17)

    working = build_working_set(usable, size=9)

    expected_hashes = sorted(usable["sequence_hash"].tolist())[:9]
    assert len(working) == 9
    assert working.index.tolist() == list(range(9))
    assert working["sequence_hash"].tolist() == expected_hashes


def test_build_working_set_rejects_insufficient_rows() -> None:
    with pytest.raises(ValueError, match="insufficient"):
        build_working_set(_usable_frame(5), size=6)


def test_build_working_set_rejects_tampered_hash_before_selection() -> None:
    usable = _usable_frame(8)
    untampered = build_working_set(usable, size=3)
    outside_member = usable.loc[
        ~usable["sequence_hash"].isin(untampered["sequence_hash"])
    ].index[0]
    usable.loc[outside_member, "sequence_hash"] = "0" * 64

    with pytest.raises(ValueError, match=r"sequence_hash.*exact row hash"):
        build_working_set(usable, size=3)


@pytest.mark.parametrize("duplicate", ["sequence_hash", "mutated_sequence"])
def test_build_working_set_rejects_duplicate_hashes_or_sequences(
    duplicate: str,
) -> None:
    usable = _usable_frame(6)
    usable.loc[1, duplicate] = usable.loc[0, duplicate]

    with pytest.raises(ValueError, match=f"duplicate.*{duplicate}"):
        build_working_set(usable, size=5)


def _write_fixture_archives(
    tmp_path: Path, dms_member: str, score_member: str
) -> tuple[Path, Path]:
    dms_zip = tmp_path / "dms.zip"
    scores_zip = tmp_path / "scores.zip"
    with zipfile.ZipFile(dms_zip, "w") as archive:
        archive.writestr(dms_member, (FIXTURES / "dms_tiny.csv").read_bytes())
        archive.writestr("metadata/README.txt", b"fixture")
    with zipfile.ZipFile(scores_zip, "w") as archive:
        archive.writestr(score_member, (FIXTURES / "scores_tiny.csv").read_bytes())
        archive.writestr("README.txt", b"fixture")
    return dms_zip, scores_zip


def test_load_assay_uses_exact_v13_member_layout_without_extracting(
    tmp_path: Path,
) -> None:
    dms_zip, scores_zip = _write_fixture_archives(
        tmp_path,
        "DMS_ProteinGym_substitutions/TINY.csv",
        "TINY.csv",
    )

    merged = load_assay_from_archives(dms_zip, scores_zip, "TINY", TEACHER)

    assert merged["mutant"].tolist() == ["A1C", "A2D", "A3E", "A4F"]
    assert merged[TEACHER].tolist() == pytest.approx([0.1, 0.2, 0.3, 0.4])
    assert sorted(path.name for path in tmp_path.iterdir()) == ["dms.zip", "scores.zip"]


def test_load_assay_projects_required_columns_during_csv_parsing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dms_zip, scores_zip = _write_fixture_archives(
        tmp_path,
        "DMS_ProteinGym_substitutions/TINY.csv",
        "TINY.csv",
    )
    read_csv = Mock(wraps=data_module.pd.read_csv)
    monkeypatch.setattr(data_module.pd, "read_csv", read_csv)

    merged = load_assay_from_archives(dms_zip, scores_zip, "TINY", TEACHER)

    assert [entry.kwargs.get("usecols") for entry in read_csv.call_args_list] == [
        ("mutant", "mutated_sequence", "DMS_score"),
        ("mutant", TEACHER),
    ]
    assert merged.columns.tolist() == [
        "mutant",
        "mutated_sequence",
        "DMS_score",
        "source_row",
        "dms_id",
        TEACHER,
    ]


@pytest.mark.parametrize(
    ("archive_kind", "csv_payload", "missing_column"),
    [
        (
            "dms",
            b"mutant,DMS_score\nA1C,1.0\n",
            "mutated_sequence",
        ),
        (
            "scores",
            b"mutant,unused_teacher\nA1C,0.1\n",
            TEACHER,
        ),
    ],
)
def test_load_assay_projection_rejects_missing_required_columns(
    tmp_path: Path,
    archive_kind: str,
    csv_payload: bytes,
    missing_column: str,
) -> None:
    dms_zip, scores_zip = _write_fixture_archives(
        tmp_path,
        "DMS_ProteinGym_substitutions/TINY.csv",
        "TINY.csv",
    )
    archive_path = dms_zip if archive_kind == "dms" else scores_zip
    member = (
        "DMS_ProteinGym_substitutions/TINY.csv" if archive_kind == "dms" else "TINY.csv"
    )
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr(member, csv_payload)

    with pytest.raises(ValueError, match=missing_column):
        load_assay_from_archives(dms_zip, scores_zip, "TINY", TEACHER)


@pytest.mark.parametrize(
    ("dms_member", "score_member", "missing_pattern"),
    [
        ("TINY.csv", "TINY.csv", "DMS_ProteinGym_substitutions/TINY.csv"),
        (
            "DMS_ProteinGym_substitutions/TINY.csv",
            "scores/TINY.csv",
            "TINY.csv",
        ),
    ],
)
def test_load_assay_rejects_missing_exact_archive_member(
    tmp_path: Path,
    dms_member: str,
    score_member: str,
    missing_pattern: str,
) -> None:
    dms_zip, scores_zip = _write_fixture_archives(tmp_path, dms_member, score_member)

    with pytest.raises(ValueError, match=missing_pattern):
        load_assay_from_archives(dms_zip, scores_zip, "TINY", TEACHER)


def test_data_manifest_atomic_roundtrip_is_byte_deterministic_and_label_free(
    tmp_path: Path,
) -> None:
    manifest = _tiny_manifest()
    first_path = tmp_path / "nested" / "manifest.json"
    second_path = tmp_path / "manifest-copy.json"

    write_data_manifest(first_path, manifest)
    write_data_manifest(second_path, manifest)

    assert first_path.read_bytes() == second_path.read_bytes()
    assert load_data_manifest(first_path) == manifest
    payload = json.loads(first_path.read_text(encoding="utf-8"))
    assert "DMS_score" not in first_path.read_text(encoding="utf-8")
    assert set(payload["selected_assays"][0]) == {
        "dms_id",
        "usable_count",
        "sequence_length",
        "row_hashes",
    }
    assert list((tmp_path / "nested").iterdir()) == [first_path]


def test_data_manifest_is_deeply_immutable_and_forbids_extra_fields() -> None:
    manifest = _tiny_manifest()

    with pytest.raises(ValidationError, match="frozen"):
        manifest.teacher_column = "other"  # type: ignore[misc]
    with pytest.raises(ValidationError, match="frozen"):
        manifest.sources.substitutions.url = "https://changed.test"  # type: ignore[misc]
    with pytest.raises(ValidationError, match="frozen"):
        manifest.selected_assays[0].usable_count = 1  # type: ignore[misc]
    with pytest.raises(ValidationError, match="extra"):
        DataManifest.model_validate(
            {**manifest.model_dump(mode="json"), "DMS_score": [1.0]}
        )
    assert isinstance(manifest.eligible_assay_ids, tuple)
    assert isinstance(manifest.selected_assays, tuple)
    assert isinstance(manifest.selected_assays[0].row_hashes, tuple)


def test_data_manifest_requires_lowercase_sha256_source_digests() -> None:
    manifest = _tiny_manifest()
    payload = manifest.model_dump(mode="json")
    payload["sources"]["scores"]["sha256"] = "B" * 64

    with pytest.raises(ValidationError, match="sha256"):
        DataManifest.model_validate(payload)


@pytest.mark.parametrize(
    "eligible_ids",
    [
        ["ASSAY_B", "ASSAY_A", "ASSAY_C", "ASSAY_D"],
        ["ASSAY_A", "ASSAY_A", "ASSAY_C", "ASSAY_D"],
    ],
)
def test_data_manifest_requires_unique_lexically_sorted_eligible_ids(
    eligible_ids: list[str],
) -> None:
    payload = _tiny_manifest().model_dump(mode="json")
    payload["eligible_assay_ids"] = eligible_ids

    with pytest.raises(ValidationError, match=r"eligible_assay_ids.*unique.*sorted"):
        DataManifest.model_validate(payload)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("confirmatory_ids", ["ASSAY_A", "ASSAY_C"]),
        ("development_id", "ASSAY_D"),
    ],
)
def test_data_manifest_requires_confirmatory_prefix_and_next_development_id(
    field: str,
    value: object,
) -> None:
    payload = _tiny_manifest().model_dump(mode="json")
    payload[field] = value

    with pytest.raises(ValidationError, match="eligible assay prefix"):
        DataManifest.model_validate(payload)


def test_data_manifest_selected_records_cover_confirmatory_and_dev_in_order() -> None:
    payload = _tiny_manifest().model_dump(mode="json")
    payload["selected_assays"] = [
        payload["selected_assays"][1],
        payload["selected_assays"][0],
        payload["selected_assays"][2],
    ]

    with pytest.raises(ValidationError, match=r"selected_assays.*coverage"):
        DataManifest.model_validate(payload)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("short_hashes", "working_size"),
        ("duplicate_hashes", "unique"),
        ("unsorted_hashes", "sorted"),
        ("insufficient_usable", "usable_count"),
        ("too_long", "sequence_length"),
        ("cross_assay_duplicate_hash", "across selected assays"),
    ],
)
def test_data_manifest_validates_selected_working_set_cardinality_and_hashes(
    mutation: str,
    message: str,
) -> None:
    payload = _tiny_manifest().model_dump(mode="json")
    first = payload["selected_assays"][0]
    if mutation == "short_hashes":
        first["row_hashes"] = first["row_hashes"][:-1]
    elif mutation == "duplicate_hashes":
        first["row_hashes"][1] = first["row_hashes"][0]
    elif mutation == "unsorted_hashes":
        first["row_hashes"] = list(reversed(first["row_hashes"]))
    elif mutation == "insufficient_usable":
        first["usable_count"] = payload["working_size"] - 1
    elif mutation == "too_long":
        first["sequence_length"] = payload["max_length"] + 1
    elif mutation == "cross_assay_duplicate_hash":
        payload["selected_assays"][1]["row_hashes"] = first["row_hashes"]
    else:
        raise AssertionError(f"unknown test mutation {mutation}")

    with pytest.raises(ValidationError, match=message):
        DataManifest.model_validate(payload)


def test_data_manifest_caps_max_length_at_512() -> None:
    payload = _tiny_manifest().model_dump(mode="json")
    payload["max_length"] = 513

    with pytest.raises(ValidationError, match="max_length"):
        DataManifest.model_validate(payload)


def test_load_data_manifest_rejects_tampered_working_set_cardinality(
    tmp_path: Path,
) -> None:
    path = tmp_path / "manifest.json"
    write_data_manifest(path, _tiny_manifest())
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["selected_assays"][0]["row_hashes"].pop()
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValidationError, match="working_size"):
        load_data_manifest(path)


def test_load_data_manifest_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    write_data_manifest(path, _tiny_manifest())
    serialized = path.read_text(encoding="utf-8")
    marker = '  "schema_version": 1,\n'
    assert serialized.count(marker) == 1
    path.write_text(serialized.replace(marker, marker + marker), encoding="utf-8")

    with pytest.raises(ValueError, match=r"duplicate JSON key.*schema_version"):
        load_data_manifest(path)


@pytest.mark.parametrize("malformed", ["true", "1.0"])
def test_load_data_manifest_requires_strict_integer_schema_version(
    tmp_path: Path,
    malformed: str,
) -> None:
    path = tmp_path / "manifest.json"
    write_data_manifest(path, _tiny_manifest())
    serialized = path.read_text(encoding="utf-8")
    marker = '"schema_version": 1'
    assert serialized.count(marker) == 1
    path.write_text(
        serialized.replace(marker, f'"schema_version": {malformed}'),
        encoding="utf-8",
    )

    with pytest.raises(ValidationError, match="schema_version"):
        load_data_manifest(path)


def test_select_eligible_assays_uses_lexical_first_eight_and_ninth_dev() -> None:
    eligible_ids = [f"ASSAY_{index:02d}" for index in range(10)]
    records = [
        AssayEligibility(dms_id, usable_count=6000 + index, sequence_length=100)
        for index, dms_id in enumerate(reversed(eligible_ids))
    ]
    records.append(AssayEligibility("INELIGIBLE", 5999, 100))
    records.append(AssayEligibility("A_TOO_LONG", 7000, 513))

    confirmatory, development = select_eligible_assays(
        records, minimum=6000, assay_count=8
    )

    assert confirmatory == tuple(eligible_ids[:8])
    assert development == eligible_ids[8]


def test_assay_eligibility_is_immutable() -> None:
    record = AssayEligibility("TINY", 6000, 4)

    with pytest.raises(FrozenInstanceError):
        record.usable_count = 1  # type: ignore[misc]


def test_select_eligible_assays_rejects_too_few_for_confirmatory_plus_dev() -> None:
    records = [AssayEligibility(f"ASSAY_{index}", 6000, 100) for index in range(8)]

    with pytest.raises(ValueError, match="at least 9"):
        select_eligible_assays(records, minimum=6000, assay_count=8)


def test_select_eligible_assays_rejects_duplicate_records() -> None:
    records = [AssayEligibility(f"ASSAY_{index}", 6000, 100) for index in range(9)]
    records.append(AssayEligibility("ASSAY_0", 6000, 100))

    with pytest.raises(ValueError, match=r"duplicate.*ASSAY_0"):
        select_eligible_assays(records, minimum=6000, assay_count=8)


def test_select_eligible_assays_rejects_inconsistent_lengths_for_same_assay() -> None:
    records = [AssayEligibility(f"ASSAY_{index}", 6000, 100) for index in range(9)]
    records.append(AssayEligibility("ASSAY_0", 6000, 101))

    with pytest.raises(ValueError, match=r"inconsistent.*sequence_length.*ASSAY_0"):
        select_eligible_assays(records, minimum=6000, assay_count=8)


def test_make_split_is_byte_deterministic_complete_disjoint_and_exact() -> None:
    working = build_working_set(_usable_frame(24), size=20)

    first = make_split(working, "TINY", 3, 3, 6, 4)
    second = make_split(working.copy(), "TINY", 3, 3, 6, 4)
    different = make_split(working, "TINY", 4, 3, 6, 4)

    assert _split_bytes(first) == _split_bytes(second)
    assert _split_bytes(first) != _split_bytes(different)
    assert tuple(
        map(len, (first.labeled, first.unlabeled, first.test, first.buffer))
    ) == (
        3,
        6,
        4,
        7,
    )
    index_groups = [
        set(first.labeled),
        set(first.unlabeled),
        set(first.test),
        set(first.buffer),
    ]
    assert set.union(*index_groups) == set(range(20))
    assert sum(len(group) for group in index_groups) == len(set.union(*index_groups))
    hash_groups = [
        set(first.labeled_sequence_hashes),
        set(first.unlabeled_sequence_hashes),
        set(first.test_sequence_hashes),
        set(first.buffer_sequence_hashes),
    ]
    assert set.union(*hash_groups) == set(working["sequence_hash"])
    assert sum(len(group) for group in hash_groups) == len(set.union(*hash_groups))
    first.validate_against(working)


def test_make_split_matches_direct_split_purpose_pcg64_permutation() -> None:
    working = build_working_set(_usable_frame(18), size=18)
    split = make_split(working, "TINY", 7, 4, 5, 3)
    split_seed = derive_seed("TINY", 7, "split")
    expected = np.random.Generator(np.random.PCG64(split_seed)).permutation(18)

    assert split.labeled == tuple(int(value) for value in expected[:4])
    assert split.unlabeled == tuple(int(value) for value in expected[4:9])
    assert split.test == tuple(int(value) for value in expected[9:12])
    assert split.buffer == tuple(int(value) for value in expected[12:])


def test_split_sequence_hash_check_rejects_tampered_working_set() -> None:
    working = build_working_set(_usable_frame(12), size=12)
    split = make_split(working, "TINY", 0, 2, 4, 3)
    tampered = working.copy()
    tampered.loc[split.labeled[0], "sequence_hash"] = "0" * 64

    with pytest.raises(ValueError, match="sequence_hash"):
        split.validate_against(tampered)


def test_make_split_recomputes_sequence_hashes_before_partitioning() -> None:
    working = build_working_set(_usable_frame(12), size=12)
    working.loc[0, "sequence_hash"] = "not-a-sha256"

    with pytest.raises(ValueError, match=r"sequence_hash.*exact row hash"):
        make_split(working, "TINY", 0, 2, 4, 3)


@pytest.mark.parametrize(
    ("seed", "n_labeled", "n_unlabeled", "n_test"),
    [
        (0.0, 2, 4, 3),
        (True, 2, 4, 3),
        (0, 2.0, 4, 3),
        (0, 2, np.int64(4), 3),
        (0, 2, 4, False),
        (-1, 2, 4, 3),
        (0, 0, 4, 3),
        (0, 2, -1, 3),
        (0, 2, 4, 20),
    ],
)
def test_make_split_rejects_invalid_types_and_sizes(
    seed: object,
    n_labeled: object,
    n_unlabeled: object,
    n_test: object,
) -> None:
    working = build_working_set(_usable_frame(12), size=12)

    with pytest.raises((TypeError, ValueError)):
        make_split(
            working,
            "TINY",
            seed,  # type: ignore[arg-type]
            n_labeled,  # type: ignore[arg-type]
            n_unlabeled,  # type: ignore[arg-type]
            n_test,  # type: ignore[arg-type]
        )


def test_make_split_does_not_use_global_numpy_rng(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    working = build_working_set(_usable_frame(12), size=12)

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("global RNG API used")

    monkeypatch.setattr(np.random, "seed", forbidden)
    monkeypatch.setattr(np.random, "shuffle", forbidden)
    monkeypatch.setattr(np.random, "permutation", forbidden)
    monkeypatch.setattr(np.random, "choice", forbidden)
    monkeypatch.setattr(np.random, "default_rng", forbidden)

    split = make_split(working, "TINY", 0, 2, 4, 3)

    assert len(split.buffer) == 3


def test_split_api_has_no_unlabeled_or_test_label_inputs() -> None:
    assert tuple(inspect.signature(make_split).parameters) == (
        "working_set",
        "dms_id",
        "seed",
        "n_labeled",
        "n_unlabeled",
        "n_test",
    )
