"""Pure, hidden-label-safe fitting and post-fit evaluation for one task."""

from __future__ import annotations

import dataclasses
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray

from self_improve_protein.config import Protocol
from self_improve_protein.metrics import (
    ndcg_at_10_percent,
    spearman_correlation,
    standardized_mse,
)
from self_improve_protein.provenance import derive_seed, sha256_bytes
from self_improve_protein.ridge import (
    FeatureTransform,
    LabelTransform,
    TeacherCalibration,
    fit_feature_transform,
    fit_label_transform,
    fit_teacher_calibration,
    fit_weighted_ridge,
    labeled_gradient_hessian,
    squared_loss,
)
from self_improve_protein.selection import (
    influence_scores,
    no_hessian_scores,
    random_indices,
    stable_top_k,
    top_teacher_indices,
)

FloatArray: TypeAlias = NDArray[np.float64]
MethodName: TypeAlias = Literal[
    "supervised",
    "random",
    "top_teacher",
    "ours",
    "no_hessian",
]

METHOD_NAMES: tuple[MethodName, ...] = (
    "supervised",
    "random",
    "top_teacher",
    "ours",
    "no_hessian",
)
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_SOURCE_IDENTITY_FIELDS = (
    "data_release",
    "substitutions_url",
    "zero_shot_scores_url",
    "metadata_url",
    "proteingym_upstream_commit",
    "substitutions_sha256",
    "zero_shot_scores_sha256",
    "metadata_sha256",
    "teacher_column",
    "model",
    "model_revision",
    "max_length",
)


def _canonical_payload_digest(payload: object) -> str:
    try:
        encoded = json.dumps(
            payload,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
    except (TypeError, ValueError) as error:
        raise ValueError(
            "canonical payload must be finite and JSON serializable"
        ) from error
    return sha256_bytes(encoded)


def canonical_protocol_digest(protocol: Protocol) -> str:
    """Hash every field of one validated immutable protocol canonically."""
    if not isinstance(protocol, Protocol):
        raise ValueError("protocol must be Protocol")
    return _canonical_payload_digest(protocol.model_dump(mode="json"))


def canonical_source_digest(protocol: Protocol) -> str:
    """Hash only pinned ProteinGym, teacher, and embedding source identity."""
    if not isinstance(protocol, Protocol):
        raise ValueError("protocol must be Protocol")
    protocol_payload = protocol.model_dump(mode="json")
    payload = {field: protocol_payload[field] for field in _SOURCE_IDENTITY_FIELDS}
    return _canonical_payload_digest(payload)


def _identity(assay_id: str, seed: int, source_digest: str) -> tuple[str, int, str]:
    if not isinstance(assay_id, str) or not assay_id.strip():
        raise ValueError("assay_id must be a non-empty string")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if not isinstance(source_digest, str) or not _SHA256_PATTERN.fullmatch(
        source_digest
    ):
        raise ValueError("source_digest must be a lowercase SHA-256 digest")
    return assay_id, seed, source_digest


def _float_array(
    value: object,
    *,
    name: str,
    ndim: int,
    allow_empty: bool = False,
) -> FloatArray:
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a real numeric {ndim}D array") from error
    if raw.ndim != ndim:
        raise ValueError(f"{name} must be a {ndim}D array")
    if raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a real numeric {ndim}D array")
    if not allow_empty and (raw.size == 0 or any(size == 0 for size in raw.shape)):
        raise ValueError(f"{name} must be non-empty")
    array = np.array(raw, dtype=np.float64, copy=True, order="C")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    array.setflags(write=False)
    return array


def _hashes(value: Sequence[str], *, name: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be a sequence of SHA-256 strings")
    hashes = tuple(value)
    if not hashes:
        raise ValueError(f"{name} must be non-empty")
    if any(
        not isinstance(item, str) or not _SHA256_PATTERN.fullmatch(item)
        for item in hashes
    ):
        raise ValueError(f"{name} must contain lowercase SHA-256 strings")
    if len(set(hashes)) != len(hashes):
        raise ValueError(f"{name} must be unique")
    return hashes


def _constant(vector: FloatArray) -> bool:
    span = float(np.max(vector) - np.min(vector))
    tolerance = (
        16.0
        * np.finfo(np.float64).eps
        * max(
            1.0,
            float(np.max(np.abs(vector))),
        )
    )
    return bool(span <= tolerance)


@dataclass(frozen=True)
class FitInputs:
    """All and only information permitted before hidden-label unblinding."""

    assay_id: str
    seed: int
    source_digest: str
    labeled_hashes: tuple[str, ...]
    unlabeled_hashes: tuple[str, ...]
    test_hashes: tuple[str, ...]
    x_l: FloatArray
    y_l: FloatArray
    z_l: FloatArray
    x_u: FloatArray
    z_u: FloatArray
    x_test: FloatArray
    z_test: FloatArray

    def __post_init__(self) -> None:
        assay_id, seed, source_digest = _identity(
            self.assay_id,
            self.seed,
            self.source_digest,
        )
        labeled_hashes = _hashes(self.labeled_hashes, name="labeled_hashes")
        unlabeled_hashes = _hashes(self.unlabeled_hashes, name="unlabeled_hashes")
        test_hashes = _hashes(self.test_hashes, name="test_hashes")
        if len(set(labeled_hashes + unlabeled_hashes + test_hashes)) != (
            len(labeled_hashes) + len(unlabeled_hashes) + len(test_hashes)
        ):
            raise ValueError("labeled, unlabeled, and test hashes must be disjoint")

        x_l = _float_array(self.x_l, name="x_l", ndim=2)
        y_l = _float_array(self.y_l, name="y_l", ndim=1)
        z_l = _float_array(self.z_l, name="z_l", ndim=1)
        x_u = _float_array(self.x_u, name="x_u", ndim=2)
        z_u = _float_array(self.z_u, name="z_u", ndim=1)
        x_test = _float_array(self.x_test, name="x_test", ndim=2)
        z_test = _float_array(self.z_test, name="z_test", ndim=1)
        widths = {x_l.shape[1], x_u.shape[1], x_test.shape[1]}
        if len(widths) != 1:
            raise ValueError("x_l, x_u, and x_test must have the same feature width")
        expected = (
            ("labeled_hashes", len(labeled_hashes), x_l.shape[0]),
            ("y_l", y_l.size, x_l.shape[0]),
            ("z_l", z_l.size, x_l.shape[0]),
            ("unlabeled_hashes", len(unlabeled_hashes), x_u.shape[0]),
            ("z_u", z_u.size, x_u.shape[0]),
            ("test_hashes", len(test_hashes), x_test.shape[0]),
            ("z_test", z_test.size, x_test.shape[0]),
        )
        for name, actual, required in expected:
            if actual != required:
                raise ValueError(f"{name} length must match its feature rows")

        for name, value in (
            ("assay_id", assay_id),
            ("seed", seed),
            ("source_digest", source_digest),
            ("labeled_hashes", labeled_hashes),
            ("unlabeled_hashes", unlabeled_hashes),
            ("test_hashes", test_hashes),
            ("x_l", x_l),
            ("y_l", y_l),
            ("z_l", z_l),
            ("x_u", x_u),
            ("z_u", z_u),
            ("x_test", x_test),
            ("z_test", z_test),
        ):
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class EvaluationLabels:
    """Hidden outcomes and immutable provenance supplied only after fitting."""

    assay_id: str
    seed: int
    source_digest: str
    labeled_hashes: tuple[str, ...]
    unlabeled_hashes: tuple[str, ...]
    test_hashes: tuple[str, ...]
    y_u: FloatArray
    y_test: FloatArray

    def __post_init__(self) -> None:
        assay_id, seed, source_digest = _identity(
            self.assay_id,
            self.seed,
            self.source_digest,
        )
        labeled_hashes = _hashes(self.labeled_hashes, name="labeled_hashes")
        unlabeled_hashes = _hashes(self.unlabeled_hashes, name="unlabeled_hashes")
        test_hashes = _hashes(self.test_hashes, name="test_hashes")
        if len(set(labeled_hashes + unlabeled_hashes + test_hashes)) != (
            len(labeled_hashes) + len(unlabeled_hashes) + len(test_hashes)
        ):
            raise ValueError("labeled, unlabeled, and test hashes must be disjoint")
        y_u = _float_array(self.y_u, name="y_u", ndim=1)
        y_test = _float_array(self.y_test, name="y_test", ndim=1)
        if y_u.size != len(unlabeled_hashes):
            raise ValueError("y_u length must match unlabeled_hashes")
        if y_test.size != len(test_hashes):
            raise ValueError("y_test length must match test_hashes")
        for name, value in (
            ("assay_id", assay_id),
            ("seed", seed),
            ("source_digest", source_digest),
            ("labeled_hashes", labeled_hashes),
            ("unlabeled_hashes", unlabeled_hashes),
            ("test_hashes", test_hashes),
            ("y_u", y_u),
            ("y_test", y_test),
        ):
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class CorrelationDiagnostic:
    defined: bool
    value: float | None
    reason: str | None


@dataclass(frozen=True)
class CosineDiagnostic:
    defined: bool
    value: float | None
    reason: str | None


@dataclass(frozen=True)
class TeacherScoreDiagnostics:
    count: int
    finite_count: int
    finite_fraction: float
    variance: float


@dataclass(frozen=True)
class DistributionDiagnostics:
    count: int
    minimum: float
    maximum: float
    mean: float
    standard_deviation: float
    quantiles: tuple[float, float, float, float, float]


@dataclass(frozen=True)
class ScoreDiagnostics:
    minimum: float
    maximum: float
    mean: float
    standard_deviation: float
    quantiles: tuple[float, float, float, float, float]
    positive_fraction: float
    unique_count: int
    mean_selected_method: float
    maximum_selected_method: float
    mean_selected_random: float


@dataclass(frozen=True)
class NormalMatrixDiagnostics:
    minimum_eigenvalue: float
    maximum_eigenvalue: float
    condition_number: float
    numerical_rank: int
    effective_df: float
    spectrum: tuple[float, ...]


@dataclass(frozen=True)
class MethodFitDiagnostics:
    name: MethodName
    stationarity_residual: float
    normal_matrix: NormalMatrixDiagnostics
    first_order_labeled_loss_change: float | None
    realized_labeled_loss_change: float | None
    displacement_cosine: float | None
    displacement_cosine_defined: bool
    displacement_relative_error: float | None
    locality_index: float | None


@dataclass(frozen=True)
class FitDiagnostics:
    feature_scale: float
    feature_mean_norm: float
    label_mean: float
    label_scale: float
    calibration_slope: float
    calibration_intercept: float
    calibration_labeled_rmse: float
    calibration_labeled_spearman: CorrelationDiagnostic
    teacher_scores_labeled: TeacherScoreDiagnostics
    teacher_scores_unlabeled: TeacherScoreDiagnostics
    teacher_scores_test: TeacherScoreDiagnostics
    teacher_student_unlabeled_residual: DistributionDiagnostics
    supervised_stationarity_identity_residual: float
    full_score: ScoreDiagnostics
    no_hessian_score: ScoreDiagnostics
    ours_top_teacher_overlap: float
    methods: tuple[MethodFitDiagnostics, ...]


@dataclass(frozen=True)
class MethodArtifact:
    name: MethodName
    selected_indices: tuple[int, ...]
    selected_hashes: tuple[str, ...]
    selected_pseudo_labels: FloatArray
    pseudo_weight: float
    ridge_lambda: float
    training_weight_sum: float
    coefficients: FloatArray
    test_predictions: FloatArray

    def __post_init__(self) -> None:
        if self.name not in METHOD_NAMES:
            raise ValueError("unknown method name")
        indices = tuple(self.selected_indices)
        if any(type(index) is not int or index < 0 for index in indices):
            raise ValueError("selected_indices must contain non-negative integers")
        if len(set(indices)) != len(indices):
            raise ValueError("selected_indices must be unique")
        selected_hashes = tuple(self.selected_hashes)
        selected_pseudo_labels = _float_array(
            self.selected_pseudo_labels,
            name="selected_pseudo_labels",
            ndim=1,
            allow_empty=True,
        )
        coefficients = _float_array(
            self.coefficients,
            name="coefficients",
            ndim=1,
        )
        predictions = _float_array(
            self.test_predictions,
            name="test_predictions",
            ndim=1,
        )
        if len(selected_hashes) != len(indices) or selected_pseudo_labels.size != len(
            indices
        ):
            raise ValueError("selected artifact lengths must match")
        for value, name in (
            (self.pseudo_weight, "pseudo_weight"),
            (self.ridge_lambda, "ridge_lambda"),
            (self.training_weight_sum, "training_weight_sum"),
        ):
            if isinstance(value, (bool, str, bytes)) or not np.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        object.__setattr__(self, "selected_indices", indices)
        object.__setattr__(self, "selected_hashes", selected_hashes)
        object.__setattr__(self, "selected_pseudo_labels", selected_pseudo_labels)
        object.__setattr__(self, "coefficients", coefficients)
        object.__setattr__(self, "test_predictions", predictions)


def _array_identity(array: FloatArray) -> dict[str, object]:
    contiguous = np.ascontiguousarray(array)
    return {
        "shape": list(contiguous.shape),
        "dtype": str(contiguous.dtype),
        "sha256": sha256_bytes(contiguous.tobytes(order="C")),
    }


@dataclass(frozen=True)
class FitResult:
    assay_id: str
    seed: int
    source_digest: str
    protocol_digest: str
    labeled_hashes: tuple[str, ...]
    unlabeled_hashes: tuple[str, ...]
    test_hashes: tuple[str, ...]
    q: int
    pseudo_weight: float
    ridge_lambda: float
    damping: float
    feature_transform: FeatureTransform
    label_transform: LabelTransform
    teacher_calibration: TeacherCalibration
    y_l_raw: FloatArray
    z_l: FloatArray
    z_u: FloatArray
    z_test: FloatArray
    x_l: FloatArray
    y_l_standardized: FloatArray
    x_u: FloatArray
    x_test: FloatArray
    pseudo_labels_u: FloatArray
    teacher_predictions_test: FloatArray
    full_scores: FloatArray
    no_hessian_scores: FloatArray
    random_diagnostic_replicates: int
    random_diagnostic_indices: tuple[tuple[int, ...], ...]
    methods: tuple[MethodArtifact, ...]
    diagnostics: FitDiagnostics

    def to_payload(self) -> dict[str, object]:
        """Return a compact canonical payload containing no hidden outcomes."""
        return {
            "assay_id": self.assay_id,
            "seed": self.seed,
            "source_digest": self.source_digest,
            "protocol_digest": self.protocol_digest,
            "labeled_hashes": list(self.labeled_hashes),
            "unlabeled_hashes": list(self.unlabeled_hashes),
            "test_hashes": list(self.test_hashes),
            "q": self.q,
            "pseudo_weight": self.pseudo_weight,
            "ridge_lambda": self.ridge_lambda,
            "damping": self.damping,
            "random_diagnostic_replicates": self.random_diagnostic_replicates,
            "feature_transform": {
                "mean": _array_identity(self.feature_transform.mean),
                "scale": self.feature_transform.scale,
            },
            "label_transform": dataclasses.asdict(self.label_transform),
            "teacher_calibration": dataclasses.asdict(self.teacher_calibration),
            "fit_state": {
                "x_l": _array_identity(self.x_l),
                "labeled_raw": _array_identity(self.y_l_raw),
                "labeled_standardized": _array_identity(self.y_l_standardized),
                "z_l": _array_identity(self.z_l),
                "z_u": _array_identity(self.z_u),
                "z_test": _array_identity(self.z_test),
                "x_u": _array_identity(self.x_u),
                "x_test": _array_identity(self.x_test),
                "pseudo_labels_u": _array_identity(self.pseudo_labels_u),
                "teacher_predictions_test": _array_identity(
                    self.teacher_predictions_test
                ),
                "full_scores": _array_identity(self.full_scores),
                "no_hessian_scores": _array_identity(self.no_hessian_scores),
            },
            "random_diagnostic_indices": [
                list(indices) for indices in self.random_diagnostic_indices
            ],
            "methods": [
                {
                    "name": method.name,
                    "selected_indices": list(method.selected_indices),
                    "selected_hashes": list(method.selected_hashes),
                    "selected_pseudo_labels": _array_identity(
                        method.selected_pseudo_labels
                    ),
                    "pseudo_weight": method.pseudo_weight,
                    "ridge_lambda": method.ridge_lambda,
                    "training_weight_sum": method.training_weight_sum,
                    "coefficients": _array_identity(method.coefficients),
                    "test_predictions": _array_identity(method.test_predictions),
                }
                for method in self.methods
            ],
            "diagnostics": dataclasses.asdict(self.diagnostics),
        }


def _canonical_fit_digest_unchecked(result: FitResult) -> str:
    return _canonical_payload_digest(result.to_payload())


def canonical_fit_digest(result: FitResult) -> str:
    """Validate and hash the deterministic, hidden-label-free fit payload."""
    if not isinstance(result, FitResult):
        raise ValueError("result must be FitResult")
    _validate_fit_integrity(result)
    return _canonical_fit_digest_unchecked(result)


def _score_diagnostics(
    scores: FloatArray,
    selected: tuple[int, ...],
    random_selected: tuple[int, ...],
) -> ScoreDiagnostics:
    quantiles = tuple(
        float(value) for value in np.quantile(scores, [0.0, 0.25, 0.5, 0.75, 1.0])
    )
    selected_array = scores[np.asarray(selected, dtype=np.int64)]
    random_array = scores[np.asarray(random_selected, dtype=np.int64)]
    return ScoreDiagnostics(
        minimum=float(np.min(scores)),
        maximum=float(np.max(scores)),
        mean=float(np.mean(scores)),
        standard_deviation=float(np.std(scores)),
        quantiles=quantiles,  # type: ignore[arg-type]
        positive_fraction=float(np.mean(scores > 0.0)),
        unique_count=int(np.unique(scores).size),
        mean_selected_method=float(np.mean(selected_array)),
        maximum_selected_method=float(np.max(selected_array)),
        mean_selected_random=float(np.mean(random_array)),
    )


def _matrix_diagnostics(
    x: FloatArray,
    weights: FloatArray,
    ridge_lambda: float,
) -> NormalMatrixDiagnostics:
    denominator = float(np.sum(weights))
    data_gram = x.T @ (weights[:, None] * x) / denominator
    normal = data_gram + ridge_lambda * np.eye(x.shape[1])
    eigenvalues = np.linalg.eigvalsh(normal)
    data_eigenvalues = np.linalg.eigvalsh(data_gram)
    tolerance = (
        max(normal.shape) * np.finfo(np.float64).eps * float(np.max(eigenvalues))
    )
    return NormalMatrixDiagnostics(
        minimum_eigenvalue=float(eigenvalues[0]),
        maximum_eigenvalue=float(eigenvalues[-1]),
        condition_number=float(eigenvalues[-1] / eigenvalues[0]),
        numerical_rank=int(np.count_nonzero(eigenvalues > tolerance)),
        effective_df=float(
            np.sum(data_eigenvalues / (data_eigenvalues + ridge_lambda))
        ),
        spectrum=tuple(float(value) for value in eigenvalues),
    )


def _cosine_diagnostic(
    first: FloatArray,
    second: FloatArray,
) -> CosineDiagnostic:
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    if denominator <= np.finfo(np.float64).tiny:
        return CosineDiagnostic(False, None, "zero_norm")
    return CosineDiagnostic(
        True,
        float(np.clip(first @ second / denominator, -1.0, 1.0)),
        None,
    )


def _locality_index(
    x_l: FloatArray,
    x_selected: FloatArray,
    hessian: FloatArray,
    t: float,
) -> float:
    eigenvalues, eigenvectors = np.linalg.eigh(hessian)
    inverse_sqrt = (eigenvectors / np.sqrt(eigenvalues)) @ eigenvectors.T
    difference = (
        x_selected.T @ x_selected / x_selected.shape[0] - x_l.T @ x_l / x_l.shape[0]
    )
    return float(t * np.linalg.norm(inverse_sqrt @ difference @ inverse_sqrt, ord=2))


def _method_diagnostics(
    method: MethodArtifact,
    *,
    x_l: FloatArray,
    y_l: FloatArray,
    x_u: FloatArray,
    pseudo_labels_u: FloatArray,
    theta_zero: FloatArray,
    gradient: FloatArray,
    hessian: FloatArray,
    q: int,
    pseudo_weight: float,
    ridge_lambda: float,
) -> MethodFitDiagnostics:
    if method.name == "supervised":
        x = x_l
        y = y_l
        weights = np.ones(x_l.shape[0], dtype=np.float64)
    else:
        indices = np.asarray(method.selected_indices, dtype=np.int64)
        x_selected = x_u[indices]
        y_selected = pseudo_labels_u[indices]
        x = np.concatenate([x_l, x_selected])
        y = np.concatenate([y_l, y_selected])
        weights = np.concatenate(
            [
                np.ones(x_l.shape[0], dtype=np.float64),
                np.full(q, pseudo_weight, dtype=np.float64),
            ]
        )
    denominator = float(np.sum(weights))
    stationarity = x.T @ (weights * (x @ method.coefficients - y)) / denominator
    stationarity += ridge_lambda * method.coefficients
    stationarity_residual = float(np.linalg.norm(stationarity))
    normal_diagnostics = _matrix_diagnostics(x, weights, ridge_lambda)
    if method.name == "supervised":
        return MethodFitDiagnostics(
            name=method.name,
            stationarity_residual=stationarity_residual,
            normal_matrix=normal_diagnostics,
            first_order_labeled_loss_change=None,
            realized_labeled_loss_change=None,
            displacement_cosine=None,
            displacement_cosine_defined=False,
            displacement_relative_error=None,
            locality_index=None,
        )

    indices = np.asarray(method.selected_indices, dtype=np.int64)
    x_selected = x_u[indices]
    y_selected = pseudo_labels_u[indices]
    selected_gradient = x_selected.T @ (x_selected @ theta_zero - y_selected) / q
    t = pseudo_weight * q / (x_l.shape[0] + pseudo_weight * q)
    predicted_displacement = -t * np.linalg.solve(
        hessian,
        selected_gradient - gradient,
    )
    realized_displacement = method.coefficients - theta_zero
    realized_norm = float(np.linalg.norm(realized_displacement))
    tiny = float(np.finfo(np.float64).tiny)
    relative_error = float(
        np.linalg.norm(realized_displacement - predicted_displacement)
        / max(realized_norm, tiny)
    )
    displacement_cosine = _cosine_diagnostic(
        predicted_displacement,
        realized_displacement,
    )
    return MethodFitDiagnostics(
        name=method.name,
        stationarity_residual=stationarity_residual,
        normal_matrix=normal_diagnostics,
        first_order_labeled_loss_change=float(gradient @ predicted_displacement),
        realized_labeled_loss_change=float(
            squared_loss(x_l, y_l, method.coefficients)
            - squared_loss(x_l, y_l, theta_zero)
        ),
        displacement_cosine=displacement_cosine.value,
        displacement_cosine_defined=displacement_cosine.defined,
        displacement_relative_error=relative_error,
        locality_index=_locality_index(x_l, x_selected, hessian, t),
    )


def _teacher_score_diagnostics(scores: FloatArray) -> TeacherScoreDiagnostics:
    finite_count = int(np.count_nonzero(np.isfinite(scores)))
    return TeacherScoreDiagnostics(
        count=int(scores.size),
        finite_count=finite_count,
        finite_fraction=float(finite_count / scores.size),
        variance=float(np.var(scores)),
    )


def _distribution_diagnostics(values: FloatArray) -> DistributionDiagnostics:
    quantiles = tuple(
        float(value) for value in np.quantile(values, [0.0, 0.25, 0.5, 0.75, 1.0])
    )
    return DistributionDiagnostics(
        count=int(values.size),
        minimum=float(np.min(values)),
        maximum=float(np.max(values)),
        mean=float(np.mean(values)),
        standard_deviation=float(np.std(values)),
        quantiles=quantiles,  # type: ignore[arg-type]
    )


def _build_fit_diagnostics(
    *,
    feature_transform: FeatureTransform,
    label_transform: LabelTransform,
    calibration: TeacherCalibration,
    z_l: FloatArray,
    z_u: FloatArray,
    z_test: FloatArray,
    x_l: FloatArray,
    y_l: FloatArray,
    x_u: FloatArray,
    pseudo_labels_u: FloatArray,
    theta_zero: FloatArray,
    full_scores: FloatArray,
    no_h_scores: FloatArray,
    selections: dict[MethodName, tuple[int, ...]],
    methods: Sequence[MethodArtifact],
    q: int,
    pseudo_weight: float,
    ridge_lambda: float,
) -> FitDiagnostics:
    gradient, hessian = labeled_gradient_hessian(
        x_l,
        y_l,
        theta_zero,
        ridge_lambda,
    )
    method_diagnostics = tuple(
        _method_diagnostics(
            method,
            x_l=x_l,
            y_l=y_l,
            x_u=x_u,
            pseudo_labels_u=pseudo_labels_u,
            theta_zero=theta_zero,
            gradient=gradient,
            hessian=hessian,
            q=q,
            pseudo_weight=pseudo_weight,
            ridge_lambda=ridge_lambda,
        )
        for method in methods
    )
    calibrated_labeled = calibration.predict(z_l)
    residual_u = x_u @ theta_zero - pseudo_labels_u
    ours = selections["ours"]
    top_teacher = selections["top_teacher"]
    random_selection = selections["random"]
    return FitDiagnostics(
        feature_scale=feature_transform.scale,
        feature_mean_norm=float(np.linalg.norm(feature_transform.mean)),
        label_mean=label_transform.mean,
        label_scale=label_transform.scale,
        calibration_slope=calibration.slope,
        calibration_intercept=calibration.intercept,
        calibration_labeled_rmse=float(
            np.sqrt(np.mean((calibrated_labeled - y_l) ** 2))
        ),
        calibration_labeled_spearman=_correlation(y_l, calibrated_labeled),
        teacher_scores_labeled=_teacher_score_diagnostics(z_l),
        teacher_scores_unlabeled=_teacher_score_diagnostics(z_u),
        teacher_scores_test=_teacher_score_diagnostics(z_test),
        teacher_student_unlabeled_residual=_distribution_diagnostics(residual_u),
        supervised_stationarity_identity_residual=float(
            np.linalg.norm(gradient + ridge_lambda * theta_zero)
        ),
        full_score=_score_diagnostics(full_scores, ours, random_selection),
        no_hessian_score=_score_diagnostics(
            no_h_scores,
            selections["no_hessian"],
            random_selection,
        ),
        ours_top_teacher_overlap=len(set(ours) & set(top_teacher)) / q,
        methods=method_diagnostics,
    )


def _validate_against_protocol(inputs: FitInputs, protocol: Protocol) -> None:
    counts = {
        "n_labeled": inputs.x_l.shape[0],
        "n_unlabeled": inputs.x_u.shape[0],
        "n_test": inputs.x_test.shape[0],
    }
    for name, actual in counts.items():
        if actual != getattr(protocol, name):
            raise ValueError(f"FitInputs {name} does not match Protocol")
    if protocol.preprocessing.feature_scaling != "scalar_rms":
        raise ValueError("unsupported feature transform")
    if protocol.preprocessing.student_fit != "no_intercept":
        raise ValueError("unsupported student fit")
    expected_source_digest = canonical_source_digest(protocol)
    if inputs.source_digest != expected_source_digest:
        raise ValueError(
            "FitInputs source_digest does not match the pinned protocol sources"
        )


def fit_task(inputs: FitInputs, protocol: Protocol) -> FitResult:
    """Fit all locked methods without accepting or accessing hidden labels."""
    if not isinstance(inputs, FitInputs):
        raise ValueError("inputs must be FitInputs")
    if not isinstance(protocol, Protocol):
        raise ValueError("protocol must be Protocol")
    _validate_against_protocol(inputs, protocol)

    feature_transform = fit_feature_transform(inputs.x_l)
    x_l = feature_transform.transform(inputs.x_l)
    x_u = feature_transform.transform(inputs.x_u)
    x_test = feature_transform.transform(inputs.x_test)
    label_transform = fit_label_transform(
        inputs.y_l,
        ddof=protocol.preprocessing.label_ddof,
    )
    y_l = label_transform.transform(inputs.y_l)
    calibration = fit_teacher_calibration(inputs.z_l, y_l)
    pseudo_labels_u = calibration.predict(inputs.z_u)
    teacher_predictions_test = calibration.predict(inputs.z_test)
    theta_zero = fit_weighted_ridge(x_l, y_l, protocol.ridge_lambda)
    full_scores = influence_scores(
        x_l,
        y_l,
        x_u,
        pseudo_labels_u,
        theta_zero,
        protocol.ridge_lambda,
        protocol.damping,
    )
    no_h_scores = no_hessian_scores(
        x_l,
        y_l,
        x_u,
        pseudo_labels_u,
        theta_zero,
        protocol.ridge_lambda,
    )
    selections: dict[MethodName, tuple[int, ...]] = {
        "supervised": (),
        "random": tuple(
            int(value)
            for value in random_indices(
                protocol.n_unlabeled,
                protocol.q,
                derive_seed(inputs.assay_id, inputs.seed, "random_selection"),
            )
        ),
        "top_teacher": tuple(
            int(value)
            for value in top_teacher_indices(
                pseudo_labels_u,
                inputs.unlabeled_hashes,
                protocol.q,
            )
        ),
        "ours": tuple(
            int(value)
            for value in stable_top_k(
                full_scores,
                inputs.unlabeled_hashes,
                protocol.q,
            )
        ),
        "no_hessian": tuple(
            int(value)
            for value in stable_top_k(
                no_h_scores,
                inputs.unlabeled_hashes,
                protocol.q,
            )
        ),
    }

    methods: list[MethodArtifact] = []
    for name in METHOD_NAMES:
        selected = selections[name]
        if name == "supervised":
            coefficients = theta_zero
            selected_labels = np.empty(0, dtype=np.float64)
            training_weight_sum = float(protocol.n_labeled)
        else:
            indices = np.asarray(selected, dtype=np.int64)
            selected_labels = pseudo_labels_u[indices]
            combined_x = np.concatenate([x_l, x_u[indices]], axis=0)
            combined_y = np.concatenate([y_l, selected_labels])
            weights = np.concatenate(
                [
                    np.ones(protocol.n_labeled, dtype=np.float64),
                    np.full(protocol.q, protocol.pseudo_weight, dtype=np.float64),
                ]
            )
            coefficients = fit_weighted_ridge(
                combined_x,
                combined_y,
                protocol.ridge_lambda,
                sample_weight=weights,
            )
            training_weight_sum = float(np.sum(weights))
        predictions = np.asarray(x_test @ coefficients, dtype=np.float64)
        if not np.all(np.isfinite(predictions)) or _constant(predictions):
            raise ValueError(f"{name} produced a constant primary prediction")
        methods.append(
            MethodArtifact(
                name=name,
                selected_indices=selected,
                selected_hashes=tuple(
                    inputs.unlabeled_hashes[index] for index in selected
                ),
                selected_pseudo_labels=selected_labels,
                pseudo_weight=0.0 if name == "supervised" else protocol.pseudo_weight,
                ridge_lambda=protocol.ridge_lambda,
                training_weight_sum=training_weight_sum,
                coefficients=coefficients,
                test_predictions=predictions,
            )
        )

    random_diagnostic_indices = tuple(
        tuple(
            int(value)
            for value in random_indices(
                protocol.n_unlabeled,
                protocol.q,
                derive_seed(
                    inputs.assay_id,
                    inputs.seed,
                    f"random_diagnostic:{replicate}",
                ),
            )
        )
        for replicate in range(protocol.random_diagnostic_replicates)
    )
    diagnostics = _build_fit_diagnostics(
        feature_transform=feature_transform,
        label_transform=label_transform,
        calibration=calibration,
        z_l=inputs.z_l,
        z_u=inputs.z_u,
        z_test=inputs.z_test,
        x_l=x_l,
        y_l=y_l,
        x_u=x_u,
        pseudo_labels_u=pseudo_labels_u,
        theta_zero=theta_zero,
        full_scores=full_scores,
        no_h_scores=no_h_scores,
        selections=selections,
        methods=methods,
        q=protocol.q,
        pseudo_weight=protocol.pseudo_weight,
        ridge_lambda=protocol.ridge_lambda,
    )
    return FitResult(
        assay_id=inputs.assay_id,
        seed=inputs.seed,
        source_digest=inputs.source_digest,
        protocol_digest=canonical_protocol_digest(protocol),
        labeled_hashes=inputs.labeled_hashes,
        unlabeled_hashes=inputs.unlabeled_hashes,
        test_hashes=inputs.test_hashes,
        q=protocol.q,
        pseudo_weight=protocol.pseudo_weight,
        ridge_lambda=protocol.ridge_lambda,
        damping=protocol.damping,
        feature_transform=feature_transform,
        label_transform=label_transform,
        teacher_calibration=calibration,
        y_l_raw=_float_array(inputs.y_l, name="y_l_raw", ndim=1),
        z_l=_float_array(inputs.z_l, name="z_l", ndim=1),
        z_u=_float_array(inputs.z_u, name="z_u", ndim=1),
        z_test=_float_array(inputs.z_test, name="z_test", ndim=1),
        x_l=_float_array(x_l, name="x_l", ndim=2),
        y_l_standardized=_float_array(y_l, name="y_l_standardized", ndim=1),
        x_u=_float_array(x_u, name="x_u", ndim=2),
        x_test=_float_array(x_test, name="x_test", ndim=2),
        pseudo_labels_u=_float_array(
            pseudo_labels_u,
            name="pseudo_labels_u",
            ndim=1,
        ),
        teacher_predictions_test=_float_array(
            teacher_predictions_test,
            name="teacher_predictions_test",
            ndim=1,
        ),
        full_scores=_float_array(full_scores, name="full_scores", ndim=1),
        no_hessian_scores=_float_array(
            no_h_scores,
            name="no_hessian_scores",
            ndim=1,
        ),
        random_diagnostic_replicates=protocol.random_diagnostic_replicates,
        random_diagnostic_indices=random_diagnostic_indices,
        methods=tuple(methods),
        diagnostics=diagnostics,
    )


@dataclass(frozen=True)
class MethodEvaluation:
    name: MethodName
    spearman: float
    mse: float
    ndcg_10pct: float
    selected_pseudo_label_mae: float | None


@dataclass(frozen=True)
class OracleDiagnostics:
    """Test-risk oracle using hidden test outer gradient and frozen pseudo-gradients."""

    score_alignment: CorrelationDiagnostic
    score_vs_absolute_error: CorrelationDiagnostic
    gradient_cosine: float | None
    gradient_cosine_defined: bool


@dataclass(frozen=True)
class EvaluationResult:
    assay_id: str
    seed: int
    methods: tuple[MethodEvaluation, ...]
    teacher_test_spearman: CorrelationDiagnostic
    pool_pseudo_label_mae: float
    random_error_reference: tuple[float, ...]
    full_test_risk_oracle: OracleDiagnostics
    no_hessian_test_risk_oracle: OracleDiagnostics


def _correlation(first: FloatArray, second: FloatArray) -> CorrelationDiagnostic:
    if _constant(first):
        return CorrelationDiagnostic(False, None, "constant_truth")
    if _constant(second):
        return CorrelationDiagnostic(False, None, "constant_prediction")
    return CorrelationDiagnostic(
        True,
        spearman_correlation(first, second),
        None,
    )


def _metric_cosine(
    first: FloatArray,
    second: FloatArray,
    metric: FloatArray | None,
) -> tuple[float | None, bool]:
    if metric is None:
        first_mapped = first
        second_mapped = second
        numerator = float(first @ second)
    else:
        first_mapped = np.asarray(
            np.linalg.solve(metric, first),
            dtype=np.float64,
        )
        second_mapped = np.asarray(
            np.linalg.solve(metric, second),
            dtype=np.float64,
        )
        numerator = float(first @ second_mapped)
    denominator = float(
        np.sqrt(max(float(first @ first_mapped), 0.0))
        * np.sqrt(max(float(second @ second_mapped), 0.0))
    )
    if denominator <= np.finfo(np.float64).tiny:
        return None, False
    return float(np.clip(numerator / denominator, -1.0, 1.0)), True


def _validate_evaluation_provenance(
    fit: FitResult,
    labels: EvaluationLabels,
) -> None:
    for name in ("assay_id", "seed", "source_digest"):
        if getattr(fit, name) != getattr(labels, name):
            raise ValueError(f"EvaluationLabels {name} does not match fit result")
    for name in ("labeled_hashes", "unlabeled_hashes", "test_hashes"):
        if getattr(fit, name) != getattr(labels, name):
            raise ValueError(f"EvaluationLabels {name} does not match fit result")
    if labels.y_u.size != fit.x_u.shape[0] or labels.y_test.size != fit.x_test.shape[0]:
        raise ValueError("EvaluationLabels cardinalities do not match fit result")


def _validate_fit_integrity(fit: FitResult) -> None:
    """Fail closed when a frozen fit artifact no longer matches its own card."""
    _identity(fit.assay_id, fit.seed, fit.source_digest)
    if not isinstance(fit.protocol_digest, str) or not _SHA256_PATTERN.fullmatch(
        fit.protocol_digest
    ):
        raise ValueError("fit protocol_digest must be a lowercase SHA-256 digest")
    labeled_hashes = _hashes(fit.labeled_hashes, name="labeled_hashes")
    unlabeled_hashes = _hashes(fit.unlabeled_hashes, name="unlabeled_hashes")
    test_hashes = _hashes(fit.test_hashes, name="test_hashes")
    if len(set(labeled_hashes + unlabeled_hashes + test_hashes)) != (
        len(labeled_hashes) + len(unlabeled_hashes) + len(test_hashes)
    ):
        raise ValueError("fit split hashes must be disjoint")
    if type(fit.q) is not int or not 0 < fit.q <= len(unlabeled_hashes):
        raise ValueError("fit q must be a valid positive selection count")
    for value, name in (
        (fit.pseudo_weight, "pseudo_weight"),
        (fit.ridge_lambda, "ridge_lambda"),
        (fit.damping, "damping"),
    ):
        if (
            isinstance(value, (bool, str, bytes))
            or not np.isfinite(float(value))
            or float(value) <= 0.0
        ):
            raise ValueError(f"fit {name} must be finite and positive")

    if fit.x_l.shape[0] != len(labeled_hashes):
        raise ValueError("fit labeled_hashes cardinality mismatch")
    if fit.x_u.shape[0] != len(unlabeled_hashes):
        raise ValueError("fit unlabeled_hashes cardinality mismatch")
    if fit.x_test.shape[0] != len(test_hashes):
        raise ValueError("fit test_hashes cardinality mismatch")
    if len({fit.x_l.shape[1], fit.x_u.shape[1], fit.x_test.shape[1]}) != 1:
        raise ValueError("fit feature width mismatch")
    vector_lengths = (
        ("y_l_raw", fit.y_l_raw.size, fit.x_l.shape[0]),
        ("y_l_standardized", fit.y_l_standardized.size, fit.x_l.shape[0]),
        ("z_l", fit.z_l.size, fit.x_l.shape[0]),
        ("z_u", fit.z_u.size, fit.x_u.shape[0]),
        ("z_test", fit.z_test.size, fit.x_test.shape[0]),
        ("pseudo_labels_u", fit.pseudo_labels_u.size, fit.x_u.shape[0]),
        (
            "teacher_predictions_test",
            fit.teacher_predictions_test.size,
            fit.x_test.shape[0],
        ),
        ("full_scores", fit.full_scores.size, fit.x_u.shape[0]),
        ("no_hessian_scores", fit.no_hessian_scores.size, fit.x_u.shape[0]),
    )
    for name, actual, expected in vector_lengths:
        if actual != expected:
            raise ValueError(f"fit {name} cardinality mismatch")

    expected_label_transform = fit_label_transform(
        fit.y_l_raw,
        ddof=fit.label_transform.ddof,
    )
    if not np.isclose(
        fit.label_transform.mean,
        expected_label_transform.mean,
        atol=0.0,
        rtol=0.0,
    ) or not np.isclose(
        fit.label_transform.scale,
        expected_label_transform.scale,
        atol=0.0,
        rtol=0.0,
    ):
        raise ValueError("fit label transform is inconsistent with labeled outcomes")
    if not np.array_equal(
        fit.y_l_standardized,
        fit.label_transform.transform(fit.y_l_raw),
    ):
        raise ValueError("fit standardized labeled outcomes are inconsistent")
    expected_calibration = fit_teacher_calibration(fit.z_l, fit.y_l_standardized)
    if not np.isclose(
        fit.teacher_calibration.slope,
        expected_calibration.slope,
        atol=0.0,
        rtol=0.0,
    ) or not np.isclose(
        fit.teacher_calibration.intercept,
        expected_calibration.intercept,
        atol=0.0,
        rtol=0.0,
    ):
        raise ValueError("fit teacher calibration is inconsistent with labeled data")
    if not np.array_equal(
        fit.pseudo_labels_u,
        fit.teacher_calibration.predict(fit.z_u),
    ) or not np.array_equal(
        fit.teacher_predictions_test,
        fit.teacher_calibration.predict(fit.z_test),
    ):
        raise ValueError("fit teacher calibration predictions are inconsistent")

    names = tuple(method.name for method in fit.methods)
    if names != METHOD_NAMES:
        raise ValueError("fit method order does not match the locked card")
    theta_zero = fit.methods[0].coefficients
    expected_full_scores = influence_scores(
        fit.x_l,
        fit.y_l_standardized,
        fit.x_u,
        fit.pseudo_labels_u,
        theta_zero,
        fit.ridge_lambda,
        fit.damping,
    )
    expected_no_h_scores = no_hessian_scores(
        fit.x_l,
        fit.y_l_standardized,
        fit.x_u,
        fit.pseudo_labels_u,
        theta_zero,
        fit.ridge_lambda,
    )
    if not np.array_equal(fit.full_scores, expected_full_scores) or not np.array_equal(
        fit.no_hessian_scores,
        expected_no_h_scores,
    ):
        raise ValueError("fit score arrays are inconsistent with the supervised model")
    expected_selections: dict[MethodName, tuple[int, ...]] = {
        "supervised": (),
        "random": tuple(
            int(value)
            for value in random_indices(
                fit.x_u.shape[0],
                fit.q,
                derive_seed(fit.assay_id, fit.seed, "random_selection"),
            )
        ),
        "top_teacher": tuple(
            int(value)
            for value in top_teacher_indices(
                fit.pseudo_labels_u,
                fit.unlabeled_hashes,
                fit.q,
            )
        ),
        "ours": tuple(
            int(value)
            for value in stable_top_k(
                fit.full_scores,
                fit.unlabeled_hashes,
                fit.q,
            )
        ),
        "no_hessian": tuple(
            int(value)
            for value in stable_top_k(
                fit.no_hessian_scores,
                fit.unlabeled_hashes,
                fit.q,
            )
        ),
    }
    for method in fit.methods:
        selected = method.selected_indices
        expected_count = 0 if method.name == "supervised" else fit.q
        if len(selected) != expected_count:
            raise ValueError(f"{method.name} selected count does not match q")
        if selected != expected_selections[method.name]:
            raise ValueError(f"{method.name} selected indices do not match its policy")
        expected_hashes = tuple(fit.unlabeled_hashes[index] for index in selected)
        if method.selected_hashes != expected_hashes:
            raise ValueError(
                f"{method.name} selected hashes do not correspond to selected indices"
            )
        expected_labels = fit.pseudo_labels_u[np.asarray(selected, dtype=np.int64)]
        if not np.array_equal(method.selected_pseudo_labels, expected_labels):
            raise ValueError(f"{method.name} selected pseudo-labels are inconsistent")
        expected_weight = 0.0 if method.name == "supervised" else fit.pseudo_weight
        if method.pseudo_weight != expected_weight:
            raise ValueError(f"{method.name} pseudo_weight does not match the fit card")
        if method.ridge_lambda != fit.ridge_lambda:
            raise ValueError(f"{method.name} ridge_lambda does not match the fit card")
        expected_weight_sum = float(fit.x_l.shape[0])
        if method.name != "supervised":
            expected_weight_sum += fit.pseudo_weight * fit.q
        if not np.isclose(
            method.training_weight_sum,
            expected_weight_sum,
            atol=8.0 * np.finfo(np.float64).eps * max(1.0, expected_weight_sum),
            rtol=0.0,
        ):
            raise ValueError(f"{method.name} training_weight_sum does not match n + wq")
        if method.coefficients.size != fit.x_l.shape[1]:
            raise ValueError(f"{method.name} coefficient width mismatch")
        expected_predictions = fit.x_test @ method.coefficients
        if not np.array_equal(method.test_predictions, expected_predictions):
            raise ValueError(f"{method.name} test predictions are inconsistent")
        if _constant(method.test_predictions):
            raise ValueError(f"{method.name} produced a constant primary prediction")
        if method.name == "supervised":
            training_x = fit.x_l
            training_y = fit.y_l_standardized
            training_weights = np.ones(fit.x_l.shape[0], dtype=np.float64)
        else:
            indices = np.asarray(selected, dtype=np.int64)
            training_x = np.concatenate([fit.x_l, fit.x_u[indices]], axis=0)
            training_y = np.concatenate(
                [fit.y_l_standardized, fit.pseudo_labels_u[indices]]
            )
            training_weights = np.concatenate(
                [
                    np.ones(fit.x_l.shape[0], dtype=np.float64),
                    np.full(fit.q, fit.pseudo_weight, dtype=np.float64),
                ]
            )
        denominator = float(np.sum(training_weights))
        normalized_stationarity = (
            training_x.T
            @ (training_weights * (training_x @ method.coefficients - training_y))
            / denominator
            + fit.ridge_lambda * method.coefficients
        )
        right_hand_scale = float(
            np.linalg.norm(training_x.T @ (training_weights * training_y) / denominator)
        )
        stationarity_tolerance = (
            256.0
            * np.finfo(np.float64).eps
            * max(1.0, right_hand_scale, float(np.linalg.norm(method.coefficients)))
            * max(1, training_x.shape[1])
        )
        if float(np.linalg.norm(normalized_stationarity)) > stationarity_tolerance:
            raise ValueError(
                f"{method.name} coefficients violate the normalized normal equations"
            )

    if (
        type(fit.random_diagnostic_replicates) is not int
        or fit.random_diagnostic_replicates <= 0
        or len(fit.random_diagnostic_indices) != fit.random_diagnostic_replicates
    ):
        raise ValueError(
            "random diagnostic replicate count does not match the fit card"
        )
    for replicate, diagnostic_indices in enumerate(fit.random_diagnostic_indices):
        expected = tuple(
            int(value)
            for value in random_indices(
                fit.x_u.shape[0],
                fit.q,
                derive_seed(
                    fit.assay_id,
                    fit.seed,
                    f"random_diagnostic:{replicate}",
                ),
            )
        )
        if diagnostic_indices != expected:
            raise ValueError("random diagnostic indices do not match their seed stream")

    expected_diagnostics = _build_fit_diagnostics(
        feature_transform=fit.feature_transform,
        label_transform=fit.label_transform,
        calibration=fit.teacher_calibration,
        z_l=fit.z_l,
        z_u=fit.z_u,
        z_test=fit.z_test,
        x_l=fit.x_l,
        y_l=fit.y_l_standardized,
        x_u=fit.x_u,
        pseudo_labels_u=fit.pseudo_labels_u,
        theta_zero=theta_zero,
        full_scores=fit.full_scores,
        no_h_scores=fit.no_hessian_scores,
        selections=expected_selections,
        methods=fit.methods,
        q=fit.q,
        pseudo_weight=fit.pseudo_weight,
        ridge_lambda=fit.ridge_lambda,
    )
    try:
        actual_diagnostics = json.dumps(
            dataclasses.asdict(fit.diagnostics),
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        expected_diagnostics_payload = json.dumps(
            dataclasses.asdict(expected_diagnostics),
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as error:
        raise ValueError("fit diagnostics must contain only finite values") from error
    if actual_diagnostics != expected_diagnostics_payload:
        raise ValueError("fit diagnostics do not match recomputed diagnostics")


def evaluate_task(
    fit: FitResult,
    labels: EvaluationLabels,
    *,
    expected_fit_digest: str,
) -> EvaluationResult:
    """Evaluate a frozen fit using hidden labels without refitting or mutation."""
    if not isinstance(fit, FitResult):
        raise ValueError("fit must be a FitResult")
    if not isinstance(labels, EvaluationLabels):
        raise ValueError("labels must be EvaluationLabels")
    if not isinstance(expected_fit_digest, str) or not _SHA256_PATTERN.fullmatch(
        expected_fit_digest
    ):
        raise ValueError("expected_fit_digest must be a lowercase SHA-256 digest")
    if _canonical_fit_digest_unchecked(fit) != expected_fit_digest:
        raise ValueError("current fit digest does not match expected_fit_digest")
    _validate_fit_integrity(fit)
    _validate_evaluation_provenance(fit, labels)
    y_u = fit.label_transform.transform(labels.y_u)
    y_test = fit.label_transform.transform(labels.y_test)
    absolute_errors = np.abs(fit.pseudo_labels_u - y_u)
    method_results: list[MethodEvaluation] = []
    for method in fit.methods:
        selected_mae = None
        if method.selected_indices:
            selected_mae = float(
                np.mean(absolute_errors[np.asarray(method.selected_indices)])
            )
        method_results.append(
            MethodEvaluation(
                name=method.name,
                spearman=spearman_correlation(y_test, method.test_predictions),
                mse=standardized_mse(y_test, method.test_predictions),
                ndcg_10pct=ndcg_at_10_percent(y_test, method.test_predictions),
                selected_pseudo_label_mae=selected_mae,
            )
        )

    theta_zero = fit.methods[0].coefficients
    gradient_l, hessian = labeled_gradient_hessian(
        fit.x_l,
        fit.y_l_standardized,
        theta_zero,
        fit.ridge_lambda,
    )
    gradient_test = fit.x_test.T @ (fit.x_test @ theta_zero - y_test) / y_test.size
    pseudo_residual = fit.x_u @ theta_zero - fit.pseudo_labels_u
    candidate_difference = pseudo_residual[:, None] * fit.x_u - gradient_l
    damped_hessian = hessian + fit.damping * np.eye(hessian.shape[0])
    full_oracle_scores = candidate_difference @ np.linalg.solve(
        damped_hessian,
        gradient_test,
    )
    no_h_oracle_scores = candidate_difference @ gradient_test
    full_cosine, full_cosine_defined = _metric_cosine(
        gradient_l,
        gradient_test,
        damped_hessian,
    )
    no_h_cosine, no_h_cosine_defined = _metric_cosine(
        gradient_l,
        gradient_test,
        None,
    )
    return EvaluationResult(
        assay_id=fit.assay_id,
        seed=fit.seed,
        methods=tuple(method_results),
        teacher_test_spearman=_correlation(
            y_test,
            fit.teacher_predictions_test,
        ),
        pool_pseudo_label_mae=float(np.mean(absolute_errors)),
        random_error_reference=tuple(
            float(np.mean(absolute_errors[np.asarray(indices)]))
            for indices in fit.random_diagnostic_indices
        ),
        full_test_risk_oracle=OracleDiagnostics(
            score_alignment=_correlation(full_oracle_scores, fit.full_scores),
            score_vs_absolute_error=_correlation(
                absolute_errors,
                fit.full_scores,
            ),
            gradient_cosine=full_cosine,
            gradient_cosine_defined=full_cosine_defined,
        ),
        no_hessian_test_risk_oracle=OracleDiagnostics(
            score_alignment=_correlation(no_h_oracle_scores, fit.no_hessian_scores),
            score_vs_absolute_error=_correlation(
                absolute_errors,
                fit.no_hessian_scores,
            ),
            gradient_cosine=no_h_cosine,
            gradient_cosine_defined=no_h_cosine_defined,
        ),
    )
