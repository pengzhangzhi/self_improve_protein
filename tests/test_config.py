import tomllib
from pathlib import Path

import pytest
from pydantic import ValidationError

from self_improve_protein.config import Protocol, load_protocol

CONFIG_PATH = Path("configs/v0.yaml")


def _protocol_data() -> dict[str, object]:
    return load_protocol(CONFIG_PATH).model_dump(mode="python")


def test_package_does_not_publish_console_script_before_cli_exists() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

    assert "scripts" not in pyproject["project"]


def test_random_selection_environment_exactly_pins_numpy() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    lockfile = tomllib.loads(Path("uv.lock").read_text(encoding="utf-8"))

    assert "numpy==2.3.5" in pyproject["project"]["dependencies"]
    numpy_versions = {
        package["version"]
        for package in lockfile["package"]
        if package["name"] == "numpy"
    }
    assert numpy_versions == {"2.3.5"}


def test_embedding_environment_exactly_pins_torch_and_transformers() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    lockfile = tomllib.loads(Path("uv.lock").read_text(encoding="utf-8"))

    assert set(pyproject["project"]["optional-dependencies"]["embed"]) == {
        "torch==2.10.0",
        "transformers==4.57.6",
    }
    for name, expected_version in {
        "torch": "2.10.0",
        "transformers": "4.57.6",
    }.items():
        versions = {
            package["version"]
            for package in lockfile["package"]
            if package["name"] == name
        }
        assert versions == {expected_version}


def test_v0_protocol_has_all_locked_values() -> None:
    protocol = load_protocol(CONFIG_PATH)

    assert protocol.working_size == 6000
    assert protocol.n_labeled == 96
    assert protocol.n_unlabeled == 2000
    assert protocol.n_test == 1000
    assert protocol.q == 192
    assert protocol.pseudo_weight == 0.1
    assert protocol.ridge_lambda == 0.01
    assert protocol.damping == 0.0001
    assert protocol.teacher_column == "ESM1v_ensemble"
    assert protocol.model == "facebook/esm2_t12_35M_UR50D"
    assert protocol.data_release == "v1.3"
    assert protocol.seeds == (0, 1, 2, 3, 4)
    assert protocol.assay_count == 8
    assert protocol.max_length == 512
    assert protocol.random_diagnostic_replicates == 100


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("model", "other/model"),
        ("model_revision", "a" * 40),
    ],
)
def test_v0_protocol_rejects_any_other_embedding_model_identity(
    field: str,
    value: str,
) -> None:
    data = _protocol_data()
    data[field] = value

    with pytest.raises(ValidationError, match=field):
        Protocol.model_validate(data)


def test_v0_protocol_has_official_sources_and_preprocessing() -> None:
    protocol = load_protocol(CONFIG_PATH)

    assert protocol.substitutions_url == (
        "https://zenodo.org/api/records/15293562/files/"
        "DMS_ProteinGym_substitutions.zip/content"
    )
    assert protocol.zero_shot_scores_url == (
        "https://zenodo.org/api/records/15293562/files/"
        "zero_shot_substitutions_scores.zip/content"
    )
    assert protocol.metadata_url == (
        "https://zenodo.org/api/records/15293562/files/DMS_substitutions.csv/content"
    )
    assert (
        protocol.proteingym_upstream_commit
        == "1f8de974dead8ff7501eff087b725d14a965e9f9"
    )
    assert protocol.model_revision == "6fbf070e65b0b7291e7bbcd451118c216cff79d8"
    assert (
        protocol.substitutions_sha256
        == "3a83766254ac9ac9984ec25cb73c6e010ea4418f5e35f143933e6b6e6473b921"
    )
    assert (
        protocol.zero_shot_scores_sha256
        == "3fd7cdb5e78f1d43cabfabfeb6578c252b63af23ba2ab44db0094dc3a42de36d"
    )
    assert (
        protocol.metadata_sha256
        == "a8f498011532a74aa9fe556a50555a75e928c5837d19c06a87592ae04049b308"
    )
    assert protocol.preprocessing.feature_scaling == "scalar_rms"
    assert protocol.preprocessing.student_fit == "no_intercept"
    assert protocol.preprocessing.label_ddof == 0
    assert protocol.analysis_seed == 20260629


@pytest.mark.parametrize(
    "field",
    ["working_size", "n_labeled", "n_unlabeled", "n_test", "q"],
)
def test_protocol_rejects_nonpositive_sizes(field: str) -> None:
    data = _protocol_data()
    data[field] = 0

    with pytest.raises(ValidationError):
        Protocol.model_validate(data)


@pytest.mark.parametrize(
    "field",
    [
        "working_size",
        "n_labeled",
        "n_unlabeled",
        "n_test",
        "q",
        "assay_count",
        "max_length",
        "analysis_seed",
        "random_diagnostic_replicates",
    ],
)
@pytest.mark.parametrize("invalid_value", [True, "1"])
def test_protocol_rejects_coercible_integer_values(
    field: str, invalid_value: object
) -> None:
    data = _protocol_data()
    data[field] = invalid_value

    with pytest.raises(ValidationError):
        Protocol.model_validate(data)


@pytest.mark.parametrize("field", ["pseudo_weight", "ridge_lambda", "damping"])
@pytest.mark.parametrize("invalid_value", [True, "0.1"])
def test_protocol_rejects_coercible_float_values(
    field: str, invalid_value: object
) -> None:
    data = _protocol_data()
    data[field] = invalid_value

    with pytest.raises(ValidationError):
        Protocol.model_validate(data)


@pytest.mark.parametrize("field", ["pseudo_weight", "ridge_lambda", "damping"])
@pytest.mark.parametrize("non_finite", [float("nan"), float("inf"), -float("inf")])
def test_protocol_rejects_non_finite_float_values(
    field: str, non_finite: float
) -> None:
    data = _protocol_data()
    data[field] = non_finite

    with pytest.raises(ValidationError):
        Protocol.model_validate(data)


@pytest.mark.parametrize("invalid_value", [False, "0"])
def test_preprocessing_rejects_coercible_label_ddof(invalid_value: object) -> None:
    data = _protocol_data()
    preprocessing = dict(data["preprocessing"])
    preprocessing["label_ddof"] = invalid_value
    data["preprocessing"] = preprocessing

    with pytest.raises(ValidationError):
        Protocol.model_validate(data)


@pytest.mark.parametrize("seeds", [[True, 2, 3, 4, 5], [0, "5", 2, 3, 4]])
def test_protocol_rejects_coercible_seed_elements(seeds: list[object]) -> None:
    data = _protocol_data()
    data["seeds"] = seeds

    with pytest.raises(ValidationError):
        Protocol.model_validate(data)


def test_protocol_rejects_active_sizes_exceeding_working_size() -> None:
    data = _protocol_data()
    data["n_test"] = 4000

    with pytest.raises(ValidationError, match="working_size"):
        Protocol.model_validate(data)


def test_protocol_rejects_q_exceeding_unlabeled_pool() -> None:
    data = _protocol_data()
    data["q"] = 2001

    with pytest.raises(ValidationError, match="n_unlabeled"):
        Protocol.model_validate(data)


def test_protocol_rejects_duplicate_seeds() -> None:
    data = _protocol_data()
    data["seeds"] = [0, 1, 1]

    with pytest.raises(ValidationError, match="distinct"):
        Protocol.model_validate(data)


@pytest.mark.parametrize("invalid_value", [0, -1, True, "100"])
def test_protocol_rejects_invalid_random_diagnostic_replicates(
    invalid_value: object,
) -> None:
    data = _protocol_data()
    data["random_diagnostic_replicates"] = invalid_value

    with pytest.raises(ValidationError, match="random_diagnostic_replicates"):
        Protocol.model_validate(data)


def test_random_diagnostic_is_carded_as_purpose_separated_exploratory() -> None:
    card = Path("docs/research/experiment-card-v0.md").read_text(encoding="utf-8")
    assert "random_diagnostic_replicates=100" in card
    assert "random_diagnostic:{replicate}" in card
    assert "exploratory" in card.lower()


def test_development_pilot_expects_five_methods_by_two_seeds() -> None:
    plan = Path(
        "docs/superpowers/plans/2026-06-29-proteingym-v0-implementation.md"
    ).read_text(encoding="utf-8")
    assert "Expected: 10 method rows total" in plan
    assert "Expected: eight method rows total" not in plan


def test_protocol_rejects_unknown_fields() -> None:
    data = _protocol_data()
    data["unlocked_override"] = True

    with pytest.raises(ValidationError):
        Protocol.model_validate(data)


@pytest.mark.parametrize(
    ("field", "length"),
    [
        ("proteingym_upstream_commit", 40),
        ("model_revision", 40),
        ("substitutions_sha256", 64),
        ("zero_shot_scores_sha256", 64),
        ("metadata_sha256", 64),
    ],
)
@pytest.mark.parametrize("invalid", [True, 1, "ABCDEF"])
def test_protocol_rejects_non_strict_or_malformed_revision_and_digest_fields(
    field: str,
    length: int,
    invalid: object,
) -> None:
    data = _protocol_data()
    if invalid == "ABCDEF":
        invalid = "A" * length
    data[field] = invalid

    with pytest.raises(ValidationError):
        Protocol.model_validate(data)


def test_protocol_is_immutable() -> None:
    protocol = load_protocol(CONFIG_PATH)

    with pytest.raises(ValidationError, match="frozen"):
        protocol.q = 1

    with pytest.raises(ValidationError, match="frozen"):
        protocol.preprocessing.label_ddof = 1


def test_load_protocol_rejects_duplicate_top_level_keys(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate-top-level.yaml"
    duplicate.write_text(
        CONFIG_PATH.read_text(encoding="utf-8") + "q: 191\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"duplicate key 'q'"):
        load_protocol(duplicate)


def test_load_protocol_rejects_duplicate_nested_keys(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate-nested.yaml"
    config_text = CONFIG_PATH.read_text(encoding="utf-8")
    duplicate.write_text(
        config_text.replace(
            "  label_ddof: 0\n",
            "  label_ddof: 0\n  label_ddof: 0\n",
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"duplicate key 'label_ddof'"):
        load_protocol(duplicate)
