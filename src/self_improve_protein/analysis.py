"""Assay-clustered summaries and predeclared v0 decision rules."""

from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral, Real
from typing import TypeAlias

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from numpy.typing import ArrayLike, NDArray

from self_improve_protein.config import Protocol

FloatArray: TypeAlias = NDArray[np.float64]

CONFIRMATORY_METHODS = ("supervised", "random", "top_teacher", "ours")
NO_HESSIAN_METHOD = "no_hessian"
METRIC_COLUMNS = ("spearman", "mse", "ndcg_10pct")
_KEY_COLUMNS = ("assay_id", "seed", "method")
_REQUIRED_COLUMNS = _KEY_COLUMNS + METRIC_COLUMNS


def _strict_int(value: int, *, name: str, minimum: int) -> int:
    if type(value) is not int or value < minimum:
        relation = "non-negative" if minimum == 0 else "positive"
        raise ValueError(f"{name} must be a {relation} integer")
    return value


def _finite_float(value: float, *, name: str) -> float:
    if isinstance(value, (bool, str, bytes)):
        raise ValueError(f"{name} must be finite")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be finite") from error
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


@dataclass(frozen=True)
class BootstrapInterval:
    """Immutable percentile interval from a hierarchical bootstrap."""

    lower: float
    upper: float
    confidence_level: float
    n_resamples: int
    analysis_seed: int

    def __post_init__(self) -> None:
        lower = _finite_float(self.lower, name="bootstrap lower bound")
        upper = _finite_float(self.upper, name="bootstrap upper bound")
        confidence = _finite_float(
            self.confidence_level,
            name="confidence_level",
        )
        if lower > upper:
            raise ValueError("bootstrap lower bound must not exceed upper bound")
        if not 0.0 < confidence < 1.0:
            raise ValueError("confidence_level must be between zero and one")
        resamples = _strict_int(
            self.n_resamples,
            name="n_resamples",
            minimum=1,
        )
        seed = _strict_int(self.analysis_seed, name="analysis_seed", minimum=0)
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)
        object.__setattr__(self, "confidence_level", confidence)
        object.__setattr__(self, "n_resamples", resamples)
        object.__setattr__(self, "analysis_seed", seed)


@dataclass(frozen=True)
class PairwiseSummary:
    """Immutable task- and assay-level summary for one paired contrast."""

    first: str
    second: str
    metric: str
    mean_gain: float
    standard_error: float
    task_wins: int
    task_total: int
    task_win_rate: float
    assay_wins: int
    assay_total: int
    assay_win_rate: float
    exact_sign_flip_pvalue: float
    assay_deltas: tuple[float, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.first, str) or not self.first:
            raise ValueError("first method must be a non-empty string")
        if not isinstance(self.second, str) or not self.second:
            raise ValueError("second method must be a non-empty string")
        if self.first == self.second:
            raise ValueError("paired methods must be distinct")
        if self.metric not in METRIC_COLUMNS:
            raise ValueError(f"metric must be one of {METRIC_COLUMNS}")
        mean = _finite_float(self.mean_gain, name="mean_gain")
        standard_error = _finite_float(
            self.standard_error,
            name="standard_error",
        )
        if standard_error < 0.0:
            raise ValueError("standard_error must be non-negative")
        task_total = _strict_int(self.task_total, name="task_total", minimum=1)
        assay_total = _strict_int(self.assay_total, name="assay_total", minimum=1)
        task_wins = _strict_int(self.task_wins, name="task_wins", minimum=0)
        assay_wins = _strict_int(self.assay_wins, name="assay_wins", minimum=0)
        if task_wins > task_total or assay_wins > assay_total:
            raise ValueError("wins must not exceed their totals")
        task_rate = _finite_float(self.task_win_rate, name="task_win_rate")
        assay_rate = _finite_float(self.assay_win_rate, name="assay_win_rate")
        if not np.isclose(task_rate, task_wins / task_total, atol=1e-15, rtol=0.0):
            raise ValueError("task_win_rate must equal task_wins / task_total")
        if not np.isclose(
            assay_rate,
            assay_wins / assay_total,
            atol=1e-15,
            rtol=0.0,
        ):
            raise ValueError("assay_win_rate must equal assay_wins / assay_total")
        pvalue = _finite_float(
            self.exact_sign_flip_pvalue,
            name="exact_sign_flip_pvalue",
        )
        if not 0.0 <= pvalue <= 1.0:
            raise ValueError("exact_sign_flip_pvalue must be in [0, 1]")
        deltas = tuple(
            _finite_float(delta, name="assay delta") for delta in self.assay_deltas
        )
        if len(deltas) != assay_total:
            raise ValueError("assay_deltas length must equal assay_total")

        object.__setattr__(self, "mean_gain", mean)
        object.__setattr__(self, "standard_error", standard_error)
        object.__setattr__(self, "task_wins", task_wins)
        object.__setattr__(self, "task_total", task_total)
        object.__setattr__(self, "task_win_rate", task_rate)
        object.__setattr__(self, "assay_wins", assay_wins)
        object.__setattr__(self, "assay_total", assay_total)
        object.__setattr__(self, "assay_win_rate", assay_rate)
        object.__setattr__(self, "exact_sign_flip_pvalue", pvalue)
        object.__setattr__(self, "assay_deltas", deltas)


@dataclass(frozen=True)
class AnalysisVerdict:
    """Primary-only selection and practical-self-improvement decisions."""

    selection_success: bool
    practical_self_improvement: bool
    ours_minus_random: PairwiseSummary
    ours_minus_supervised: PairwiseSummary

    def __post_init__(self) -> None:
        if type(self.selection_success) is not bool:
            raise ValueError("selection_success must be a boolean")
        if type(self.practical_self_improvement) is not bool:
            raise ValueError("practical_self_improvement must be a boolean")


def _string_tuple(values: Sequence[str], *, name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be a sequence of strings")
    result = tuple(values)
    if not result or any(not isinstance(value, str) or not value for value in result):
        raise ValueError(f"{name} must contain non-empty strings")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must be unique")
    return result


def _seed_tuple(values: Sequence[int]) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError("seeds must be a sequence of integers")
    result = tuple(values)
    if not result or any(type(value) is not int for value in result):
        raise ValueError("seeds must contain integers")
    if len(set(result)) != len(result):
        raise ValueError("seeds must be unique")
    return result


def validate_result_table(
    results: pd.DataFrame,
    *,
    assay_ids: Sequence[str],
    seeds: Sequence[int],
    required_methods: Sequence[str] = CONFIRMATORY_METHODS,
) -> pd.DataFrame:
    """Validate and deterministically sort a complete long-form result grid."""
    if not isinstance(results, pd.DataFrame):
        raise ValueError("results must be a pandas DataFrame")
    missing_columns = [
        column for column in _REQUIRED_COLUMNS if column not in results.columns
    ]
    if missing_columns:
        raise ValueError(f"results missing required columns: {missing_columns}")
    expected_assays = _string_tuple(assay_ids, name="assay_ids")
    expected_seeds = _seed_tuple(seeds)
    required = _string_tuple(required_methods, name="required_methods")
    table = results.copy(deep=True)

    if table.empty:
        raise ValueError("results must be non-empty")
    if any(not isinstance(value, str) or not value for value in table["assay_id"]):
        raise ValueError("assay_id values must be non-empty strings")
    if any(not isinstance(value, str) or not value for value in table["method"]):
        raise ValueError("method values must be non-empty strings")
    if any(
        not isinstance(value, Integral) or isinstance(value, (bool, np.bool_))
        for value in table["seed"]
    ):
        raise ValueError("seed values must be integers")

    for metric in METRIC_COLUMNS:
        if any(
            not isinstance(value, Real) or isinstance(value, (bool, np.bool_))
            for value in table[metric]
        ):
            raise ValueError(f"{metric} values must be numeric")
        numeric = table[metric].to_numpy(dtype=np.float64, copy=True)
        if not np.all(np.isfinite(numeric)):
            raise ValueError(f"{metric} values must be finite")
        table[metric] = numeric

    duplicate_mask = table.duplicated(list(_KEY_COLUMNS), keep=False)
    if bool(duplicate_mask.any()):
        raise ValueError("duplicate assay_id, seed, method task rows")

    present_assays = set(table["assay_id"])
    if not present_assays.issubset(expected_assays):
        extras = sorted(present_assays - set(expected_assays))
        raise ValueError(f"unexpected assay IDs: {extras}")
    present_methods = set(table["method"])
    absent_methods = sorted(set(required) - present_methods)
    if absent_methods:
        raise ValueError(f"required method rows are missing: {absent_methods}")

    expected_task_keys = {
        (assay_id, seed) for assay_id in expected_assays for seed in expected_seeds
    }
    for method in sorted(present_methods):
        method_rows = table.loc[table["method"] == method, ["assay_id", "seed"]]
        actual_keys = {
            (str(assay_id), int(seed))
            for assay_id, seed in method_rows.itertuples(index=False, name=None)
        }
        missing = expected_task_keys - actual_keys
        extra = actual_keys - expected_task_keys
        if missing or extra:
            raise ValueError(
                f"method {method!r} has missing or extra assay-seed tasks "
                f"(missing={len(missing)}, extra={len(extra)})"
            )

    return table.sort_values(list(_KEY_COLUMNS), kind="stable").reset_index(drop=True)


def validate_v0_result_table(
    results: pd.DataFrame,
    *,
    assay_ids: Sequence[str],
    protocol: Protocol,
    require_no_hessian: bool = False,
) -> pd.DataFrame:
    """Validate the exact assay and seed cardinalities declared by a protocol."""
    assays = _string_tuple(assay_ids, name="assay_ids")
    if len(assays) != protocol.assay_count:
        raise ValueError(
            f"assay_ids length must equal Protocol assay_count ({protocol.assay_count})"
        )
    if type(require_no_hessian) is not bool:
        raise ValueError("require_no_hessian must be a boolean")
    methods: tuple[str, ...] = CONFIRMATORY_METHODS
    if require_no_hessian:
        methods += (NO_HESSIAN_METHOD,)
    return validate_result_table(
        results,
        assay_ids=assays,
        seeds=protocol.seeds,
        required_methods=methods,
    )


def _paired_deltas(
    results: pd.DataFrame,
    *,
    first: str,
    second: str,
    metric: str,
) -> pd.DataFrame:
    if (
        not isinstance(first, str)
        or not first
        or not isinstance(second, str)
        or not second
    ):
        raise ValueError("method names must be non-empty strings")
    if first == second:
        raise ValueError("paired methods must be distinct")
    if metric not in METRIC_COLUMNS:
        raise ValueError(f"metric must be one of {METRIC_COLUMNS}")
    for column in (*_KEY_COLUMNS, metric):
        if column not in results.columns:
            raise ValueError(f"results missing required column {column!r}")
    subset = results.loc[
        results["method"].isin((first, second)),
        ["assay_id", "seed", "method", metric],
    ]
    missing_methods = {first, second} - set(subset["method"])
    if missing_methods:
        raise ValueError(f"method rows are missing: {sorted(missing_methods)}")
    if bool(subset.duplicated(list(_KEY_COLUMNS)).any()):
        raise ValueError("duplicate paired task rows")
    wide = subset.pivot(index=["assay_id", "seed"], columns="method", values=metric)
    missing_pairs = bool(wide[[first, second]].isna().any().any())
    if first not in wide or second not in wide or missing_pairs:
        raise ValueError("paired methods must have matching assay-seed tasks")
    values = wide[[first, second]].to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("paired metric values must be finite")
    output = wide.reset_index()[["assay_id", "seed"]].copy()
    output["delta"] = values[:, 0] - values[:, 1]
    return output.sort_values(["assay_id", "seed"], kind="stable").reset_index(
        drop=True
    )


def exact_sign_flip_pvalue(assay_deltas: ArrayLike) -> float:
    """Enumerate the exact two-sided sign-flip null over assay means."""
    try:
        deltas = np.asarray(assay_deltas, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("assay_deltas must be a finite 1D array") from error
    if deltas.ndim != 1 or deltas.size == 0:
        raise ValueError("assay_deltas must be a non-empty 1D array")
    if not np.all(np.isfinite(deltas)):
        raise ValueError("assay_deltas must be finite")
    assay_count = deltas.size
    total_assignments = 1 << assay_count
    observed = abs(float(np.mean(deltas)))
    extreme_count = 0
    bit_positions = np.arange(assay_count, dtype=np.int64)
    for mask in range(total_assignments):
        signs = np.where((mask >> bit_positions) & 1, 1.0, -1.0)
        permuted = abs(float(np.mean(signs * deltas)))
        if permuted >= observed:
            extreme_count += 1
    return extreme_count / total_assignments


def pairwise_summary(
    results: pd.DataFrame,
    *,
    first: str,
    second: str,
    metric: str = "spearman",
) -> PairwiseSummary:
    """Average paired tasks within assay before cross-assay inference.

    ``results`` must be the complete output of :func:`validate_result_table` or
    :func:`validate_v0_result_table`.
    """
    paired = _paired_deltas(results, first=first, second=second, metric=metric)
    seed_counts = paired.groupby("assay_id", sort=True)["seed"].nunique()
    if seed_counts.nunique() != 1:
        raise ValueError("each assay must have the same paired seed grid")
    assay_means = paired.groupby("assay_id", sort=True)["delta"].mean()
    assay_values = assay_means.to_numpy(dtype=np.float64)
    task_values = paired["delta"].to_numpy(dtype=np.float64)
    assay_total = assay_values.size
    standard_error = (
        float(np.std(assay_values, ddof=1) / np.sqrt(assay_total))
        if assay_total > 1
        else 0.0
    )
    task_wins = int(np.count_nonzero(task_values > 0.0))
    assay_wins = int(np.count_nonzero(assay_values > 0.0))
    return PairwiseSummary(
        first=first,
        second=second,
        metric=metric,
        mean_gain=float(np.mean(assay_values)),
        standard_error=standard_error,
        task_wins=task_wins,
        task_total=task_values.size,
        task_win_rate=task_wins / task_values.size,
        assay_wins=assay_wins,
        assay_total=assay_total,
        assay_win_rate=assay_wins / assay_total,
        exact_sign_flip_pvalue=exact_sign_flip_pvalue(assay_values),
        assay_deltas=tuple(float(value) for value in assay_values),
    )


def hierarchical_bootstrap_interval(
    results: pd.DataFrame,
    *,
    first: str,
    second: str,
    metric: str = "spearman",
    analysis_seed: int,
    n_resamples: int = 10_000,
    confidence_level: float = 0.95,
) -> BootstrapInterval:
    """Resample assays, then paired seed rows within each sampled assay.

    ``results`` must be the complete output of :func:`validate_result_table` or
    :func:`validate_v0_result_table`.
    """
    seed = _strict_int(analysis_seed, name="analysis_seed", minimum=0)
    resample_count = _strict_int(n_resamples, name="n_resamples", minimum=1)
    confidence = _finite_float(confidence_level, name="confidence_level")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence_level must be between zero and one")
    paired = _paired_deltas(results, first=first, second=second, metric=metric)
    grouped = [
        group["delta"].to_numpy(dtype=np.float64)
        for _, group in paired.groupby("assay_id", sort=True)
    ]
    if not grouped:
        raise ValueError("at least one assay is required")
    if len({values.size for values in grouped}) != 1:
        raise ValueError("each assay must have the same paired seed grid")

    generator = np.random.Generator(np.random.PCG64(seed))
    assay_count = len(grouped)
    samples = np.empty(resample_count, dtype=np.float64)
    for bootstrap_index in range(resample_count):
        sampled_assays = generator.integers(0, assay_count, size=assay_count)
        cluster_means = np.empty(assay_count, dtype=np.float64)
        for cluster_index, assay_index in enumerate(sampled_assays):
            seed_values = grouped[int(assay_index)]
            sampled_seeds = generator.integers(
                0,
                seed_values.size,
                size=seed_values.size,
            )
            cluster_means[cluster_index] = float(np.mean(seed_values[sampled_seeds]))
        samples[bootstrap_index] = float(np.mean(cluster_means))

    alpha = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(samples, [alpha, 1.0 - alpha])
    return BootstrapInterval(
        lower=float(lower),
        upper=float(upper),
        confidence_level=confidence,
        n_resamples=resample_count,
        analysis_seed=seed,
    )


def _ordered_methods(results: pd.DataFrame) -> tuple[str, ...]:
    present = set(str(method) for method in results["method"])
    confirmatory = tuple(method for method in CONFIRMATORY_METHODS if method in present)
    extras = tuple(sorted(present - set(confirmatory)))
    return confirmatory + extras


def method_summary_table(results: pd.DataFrame) -> pd.DataFrame:
    """Return a compact assay-macro table for every included method.

    ``results`` must be the complete output of :func:`validate_result_table` or
    :func:`validate_v0_result_table`.
    """
    for column in _REQUIRED_COLUMNS:
        if column not in results.columns:
            raise ValueError(f"results missing required column {column!r}")
    assay_means = (
        results.groupby(["method", "assay_id"], sort=True)[list(METRIC_COLUMNS)]
        .mean()
        .reset_index()
    )
    rows: list[dict[str, float | str]] = []
    for method in _ordered_methods(results):
        method_assays = assay_means.loc[assay_means["method"] == method]
        spearman_values = method_assays["spearman"].to_numpy(dtype=np.float64)
        if not np.all(
            np.isfinite(method_assays[list(METRIC_COLUMNS)].to_numpy(dtype=np.float64))
        ):
            raise ValueError("method metric values must be finite")
        se = (
            float(np.std(spearman_values, ddof=1) / np.sqrt(spearman_values.size))
            if spearman_values.size > 1
            else 0.0
        )
        rows.append(
            {
                "method": method,
                "mean_spearman": float(method_assays["spearman"].mean()),
                "se_spearman": se,
                "mean_mse": float(method_assays["mse"].mean()),
                "mean_ndcg_10pct": float(method_assays["ndcg_10pct"].mean()),
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "method",
            "mean_spearman",
            "se_spearman",
            "mean_mse",
            "mean_ndcg_10pct",
        ],
    )


def comparison_summary_table(
    results: pd.DataFrame,
    *,
    comparisons: Sequence[tuple[str, str]],
    analysis_seed: int,
    n_resamples: int = 10_000,
    confidence_level: float = 0.95,
) -> pd.DataFrame:
    """Return confirmatory-style clustered summaries for requested contrasts.

    ``results`` must be the complete output of :func:`validate_result_table` or
    :func:`validate_v0_result_table`.
    """
    rows: list[dict[str, float | int | str]] = []
    for first, second in comparisons:
        summary = pairwise_summary(
            results,
            first=first,
            second=second,
            metric="spearman",
        )
        interval = hierarchical_bootstrap_interval(
            results,
            first=first,
            second=second,
            metric="spearman",
            analysis_seed=analysis_seed,
            n_resamples=n_resamples,
            confidence_level=confidence_level,
        )
        rows.append(
            {
                "comparison": f"{first} - {second}",
                "mean_spearman_gain": summary.mean_gain,
                "se": summary.standard_error,
                "task_wins": summary.task_wins,
                "task_total": summary.task_total,
                "task_win_rate": summary.task_win_rate,
                "assay_wins": summary.assay_wins,
                "assay_total": summary.assay_total,
                "assay_win_rate": summary.assay_win_rate,
                "exact_sign_flip_pvalue": summary.exact_sign_flip_pvalue,
                "bootstrap_lower": interval.lower,
                "bootstrap_upper": interval.upper,
            }
        )
    return pd.DataFrame(rows)


def analysis_verdict(
    results: pd.DataFrame,
    *,
    ours_method: str = "ours",
    random_method: str = "random",
    supervised_method: str = "supervised",
    minimum_task_wins: int = 25,
    minimum_assay_wins: int = 5,
) -> AnalysisVerdict:
    """Apply the primary-only selection rule and separate practical criterion.

    ``results`` must be the complete output of :func:`validate_result_table` or
    :func:`validate_v0_result_table`.
    """
    task_threshold = _strict_int(
        minimum_task_wins,
        name="minimum_task_wins",
        minimum=0,
    )
    assay_threshold = _strict_int(
        minimum_assay_wins,
        name="minimum_assay_wins",
        minimum=0,
    )
    versus_random = pairwise_summary(
        results,
        first=ours_method,
        second=random_method,
        metric="spearman",
    )
    versus_supervised = pairwise_summary(
        results,
        first=ours_method,
        second=supervised_method,
        metric="spearman",
    )
    if task_threshold > versus_random.task_total:
        raise ValueError("minimum_task_wins must not exceed task_total")
    if assay_threshold > versus_random.assay_total:
        raise ValueError("minimum_assay_wins must not exceed assay_total")
    selection_success = (
        versus_random.mean_gain > 0.0
        and versus_random.task_wins >= task_threshold
        and versus_random.assay_wins >= assay_threshold
    )
    practical = versus_supervised.mean_gain > 0.0
    return AnalysisVerdict(
        selection_success=selection_success,
        practical_self_improvement=practical,
        ours_minus_random=versus_random,
        ours_minus_supervised=versus_supervised,
    )
