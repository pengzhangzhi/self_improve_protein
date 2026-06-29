"""Float64 preprocessing, calibration, and normalized ridge utilities."""

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

FloatArray: TypeAlias = NDArray[np.float64]


def _as_float64_matrix(value: FloatArray, *, name: str) -> FloatArray:
    """Return *value* as a finite, non-empty float64 matrix."""
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a numeric 2D array") from error
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2D array")
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"{name} must be non-empty in both dimensions")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _as_float64_vector(value: FloatArray, *, name: str) -> FloatArray:
    """Return *value* as a finite, non-empty float64 vector."""
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a numeric 1D array") from error
    if array.ndim != 1:
        raise ValueError(f"{name} must be a 1D array")
    if array.size == 0:
        raise ValueError(f"{name} must be non-empty")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _finite_scalar(value: float, *, name: str) -> float:
    """Return a finite Python float or raise a parameter-specific error."""
    try:
        scalar = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite scalar") from error
    if not np.isfinite(scalar):
        raise ValueError(f"{name} must be finite")
    return scalar


def _validated_xy(x: FloatArray, y: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Validate a finite design matrix and matching finite response vector."""
    matrix = _as_float64_matrix(x, name="X")
    vector = _as_float64_vector(y, name="y")
    if matrix.shape[0] != vector.shape[0]:
        raise ValueError("X and y must have the same number of rows")
    return matrix, vector


def _validated_ridge_lambda(ridge_lambda: float) -> float:
    """Validate a strict numeric, finite, non-negative ridge coefficient."""
    if isinstance(ridge_lambda, (bool, str, bytes)):
        raise ValueError("ridge_lambda must be a finite non-negative number")
    value = _finite_scalar(ridge_lambda, name="ridge_lambda")
    if value < 0.0:
        raise ValueError("ridge_lambda must be non-negative")
    return value


@dataclass(frozen=True)
class FeatureTransform:
    """Labeled-only column centering with one global RMS scale."""

    mean: FloatArray
    scale: float

    def __post_init__(self) -> None:
        mean = np.array(
            _as_float64_vector(self.mean, name="mean"),
            dtype=np.float64,
            copy=True,
        )
        mean.setflags(write=False)
        scale = _finite_scalar(self.scale, name="scale")
        if scale <= 0.0:
            raise ValueError("scale must be strictly positive")
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "scale", scale)

    def transform(self, x: FloatArray) -> FloatArray:
        """Center and scale a finite matrix with the fitted feature width."""
        matrix = _as_float64_matrix(x, name="X")
        if matrix.shape[1] != self.mean.size:
            raise ValueError(
                f"X width {matrix.shape[1]} does not match fitted width "
                f"{self.mean.size}"
            )
        return np.asarray((matrix - self.mean) / self.scale, dtype=np.float64)


def fit_feature_transform(x_l: FloatArray) -> FeatureTransform:
    """Fit labeled-only centering and scalar-RMS feature scaling."""
    matrix = _as_float64_matrix(x_l, name="X_l")
    mean = np.asarray(matrix.mean(axis=0), dtype=np.float64)
    centered = matrix - mean
    scale = float(np.sqrt(np.mean(centered * centered)))
    if not np.all(np.isfinite(mean)) or not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("feature scale must be finite and strictly positive")
    return FeatureTransform(mean=mean, scale=scale)


@dataclass(frozen=True)
class LabelTransform:
    """Population standardization fitted on labeled responses only."""

    mean: float
    scale: float
    ddof: int = 0

    def __post_init__(self) -> None:
        mean = _finite_scalar(self.mean, name="mean")
        scale = _finite_scalar(self.scale, name="scale")
        if scale <= 0.0:
            raise ValueError("scale must be strictly positive")
        if type(self.ddof) is not int or self.ddof != 0:
            raise ValueError("ddof must be the integer 0")
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "scale", scale)

    def transform(self, y: FloatArray) -> FloatArray:
        """Standardize a finite response vector."""
        vector = _as_float64_vector(y, name="y")
        return np.asarray((vector - self.mean) / self.scale, dtype=np.float64)

    def inverse(self, y_standardized: FloatArray) -> FloatArray:
        """Map a finite standardized response vector to its original scale."""
        vector = _as_float64_vector(y_standardized, name="y_standardized")
        return np.asarray(vector * self.scale + self.mean, dtype=np.float64)


def fit_label_transform(y_l: FloatArray, *, ddof: int = 0) -> LabelTransform:
    """Fit labeled-only population response standardization."""
    if type(ddof) is not int or ddof != 0:
        raise ValueError("ddof must be the integer 0")
    vector = _as_float64_vector(y_l, name="y_l")
    mean = float(np.mean(vector))
    scale = float(np.std(vector, ddof=ddof))
    if not np.isfinite(mean) or not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("label scale must be finite and strictly positive")
    return LabelTransform(mean=mean, scale=scale, ddof=ddof)


@dataclass(frozen=True)
class TeacherCalibration:
    """Affine calibration from teacher scores to standardized responses."""

    slope: float
    intercept: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "slope", _finite_scalar(self.slope, name="slope"))
        object.__setattr__(
            self,
            "intercept",
            _finite_scalar(self.intercept, name="intercept"),
        )

    def predict(self, z: FloatArray) -> FloatArray:
        """Apply the fitted affine calibration to finite teacher scores."""
        vector = _as_float64_vector(z, name="z")
        return np.asarray(self.slope * vector + self.intercept, dtype=np.float64)


def fit_teacher_calibration(
    z_l: FloatArray,
    y_l_standardized: FloatArray,
) -> TeacherCalibration:
    """Fit OLS teacher calibration with an intercept in standardized space."""
    teacher = _as_float64_vector(z_l, name="z_l")
    labels = _as_float64_vector(y_l_standardized, name="y_l_standardized")
    if teacher.shape[0] != labels.shape[0]:
        raise ValueError("z_l and y_l_standardized must have the same number of rows")
    if np.all(teacher == teacher[0]):
        return TeacherCalibration(slope=0.0, intercept=float(np.mean(labels)))
    design = np.column_stack((teacher, np.ones_like(teacher)))
    coefficients, _, _, _ = np.linalg.lstsq(design, labels, rcond=None)
    return TeacherCalibration(
        slope=float(coefficients[0]),
        intercept=float(coefficients[1]),
    )


def fit_weighted_ridge(
    x: FloatArray,
    y: FloatArray,
    ridge_lambda: float,
    sample_weight: FloatArray | None = None,
) -> FloatArray:
    """Solve the exact no-intercept, weight-normalized ridge equations."""
    matrix, response = _validated_xy(x, y)
    regularization = _validated_ridge_lambda(ridge_lambda)
    if sample_weight is None:
        weights = np.ones(matrix.shape[0], dtype=np.float64)
    else:
        weights = _as_float64_vector(sample_weight, name="sample_weight")
        if weights.shape[0] != matrix.shape[0]:
            raise ValueError("sample_weight length must match the number of X rows")
        if np.any(weights <= 0.0):
            raise ValueError("sample_weight values must be strictly positive")
    denominator = float(np.sum(weights))
    if not np.isfinite(denominator):
        raise ValueError("sample_weight sum must be finite")
    if regularization == 0.0:
        weighted_matrix = np.sqrt(weights)[:, None] * matrix
        if not np.all(np.isfinite(weighted_matrix)):
            raise ValueError("weighted ridge normal equations must be finite")
        if np.linalg.matrix_rank(weighted_matrix) < matrix.shape[1]:
            raise np.linalg.LinAlgError("weighted ridge normal matrix is singular")
    identity = np.eye(matrix.shape[1], dtype=np.float64)
    gram = (
        matrix.T @ (weights[:, None] * matrix)
        + denominator * regularization * identity
    )
    right_hand_side = matrix.T @ (weights * response)
    if not np.all(np.isfinite(gram)) or not np.all(np.isfinite(right_hand_side)):
        raise ValueError("weighted ridge normal equations must be finite")
    try:
        solution = np.linalg.solve(gram, right_hand_side)
    except np.linalg.LinAlgError as error:
        raise np.linalg.LinAlgError(
            "weighted ridge normal matrix is singular"
        ) from error
    return np.asarray(solution, dtype=np.float64)


def labeled_gradient_hessian(
    x: FloatArray,
    y: FloatArray,
    theta: FloatArray,
    ridge_lambda: float,
) -> tuple[FloatArray, FloatArray]:
    """Return the unregularized loss gradient and regularized Hessian."""
    matrix, response = _validated_xy(x, y)
    parameters = _as_float64_vector(theta, name="theta")
    if parameters.shape[0] != matrix.shape[1]:
        raise ValueError("theta length must match the number of X columns")
    regularization = _validated_ridge_lambda(ridge_lambda)
    sample_count = matrix.shape[0]
    gradient = matrix.T @ (matrix @ parameters - response) / sample_count
    hessian = (
        matrix.T @ matrix / sample_count
        + regularization * np.eye(matrix.shape[1], dtype=np.float64)
    )
    return (
        np.asarray(gradient, dtype=np.float64),
        np.asarray(hessian, dtype=np.float64),
    )


def squared_loss(x: FloatArray, y: FloatArray, theta: FloatArray) -> float:
    """Return one half of the mean squared residual without regularization."""
    matrix, response = _validated_xy(x, y)
    parameters = _as_float64_vector(theta, name="theta")
    if parameters.shape[0] != matrix.shape[1]:
        raise ValueError("theta length must match the number of X columns")
    residual = matrix @ parameters - response
    return float(np.mean(residual * residual) / 2.0)
