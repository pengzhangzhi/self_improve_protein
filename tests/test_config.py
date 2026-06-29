from pathlib import Path

import pytest
from pydantic import ValidationError

from self_improve_protein.config import Protocol, load_protocol

CONFIG_PATH = Path("configs/v0.yaml")


def _protocol_data() -> dict[str, object]:
    return load_protocol(CONFIG_PATH).model_dump(mode="python")


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
        "https://zenodo.org/api/records/15293562/files/"
        "DMS_substitutions.csv/content"
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


def test_protocol_rejects_unknown_fields() -> None:
    data = _protocol_data()
    data["unlocked_override"] = True

    with pytest.raises(ValidationError):
        Protocol.model_validate(data)


def test_protocol_is_immutable() -> None:
    protocol = load_protocol(CONFIG_PATH)

    with pytest.raises(ValidationError, match="frozen"):
        protocol.q = 1

    with pytest.raises(ValidationError, match="frozen"):
        protocol.preprocessing.label_ddof = 1
