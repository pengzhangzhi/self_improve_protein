from dataclasses import FrozenInstanceError
from inspect import signature

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from self_improve_protein.analysis import (
    CONFIRMATORY_METHODS,
    NO_HESSIAN_METHOD,
    BootstrapInterval,
    PairwiseSummary,
    comparison_summary_table,
    exact_sign_flip_pvalue,
    hierarchical_bootstrap_interval,
    method_summary_table,
    pairwise_summary,
    v0_analysis_verdict,
    validate_result_table,
    validate_v0_result_table,
)
from self_improve_protein.config import load_protocol

ASSAYS = tuple(f"assay_{index}" for index in range(8))
SEEDS = (0, 1, 2, 3, 4)
ASSAY_DELTAS = np.array(
    [-0.20, -0.10, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50],
    dtype=np.float64,
)


def _complete_results(*, practical_gain: float = 0.02) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for assay_index, assay_id in enumerate(ASSAYS):
        for seed in SEEDS:
            random_spearman = -0.2 + 0.03 * assay_index + 0.002 * seed
            ours_spearman = random_spearman + ASSAY_DELTAS[assay_index]
            values = {
                "supervised": ours_spearman - practical_gain,
                "random": random_spearman,
                "top_teacher": ours_spearman - 0.01,
                "ours": ours_spearman,
            }
            for method, spearman in values.items():
                rows.append(
                    {
                        "assay_id": assay_id,
                        "seed": seed,
                        "method": method,
                        "spearman": spearman,
                        "mse": 1.0 - 0.1 * spearman,
                        "ndcg_10pct": 0.5 + 0.1 * spearman,
                    }
                )
    return pd.DataFrame(rows)


def test_validate_result_table_accepts_exact_grid_and_returns_sorted_copy() -> None:
    source = _complete_results().sample(frac=1.0, random_state=7)
    source_before = source.copy(deep=True)

    validated = validate_result_table(
        source,
        assay_ids=ASSAYS,
        seeds=SEEDS,
        required_methods=CONFIRMATORY_METHODS,
    )

    pdt.assert_frame_equal(source, source_before)
    assert len(validated) == 8 * 5 * 4
    assert validated[["assay_id", "seed", "method"]].to_records(
        index=False
    ).tolist() == sorted(
        validated[["assay_id", "seed", "method"]].to_records(index=False).tolist()
    )


@pytest.mark.parametrize("defect", ["missing", "duplicate", "nonfinite", "extra"])
def test_validate_result_table_rejects_incomplete_or_invalid_task_grids(
    defect: str,
) -> None:
    table = _complete_results()
    if defect == "missing":
        table = table.iloc[1:].copy()
        message = "missing"
    elif defect == "duplicate":
        table = pd.concat([table, table.iloc[[0]]], ignore_index=True)
        message = "duplicate"
    elif defect == "nonfinite":
        table.loc[0, "spearman"] = np.nan
        message = "finite"
    else:
        extra = table.iloc[[0]].copy()
        extra["assay_id"] = "not_predeclared"
        table = pd.concat([table, extra], ignore_index=True)
        message = "assay"

    with pytest.raises(ValueError, match=message):
        validate_result_table(
            table,
            assay_ids=ASSAYS,
            seeds=SEEDS,
            required_methods=CONFIRMATORY_METHODS,
        )


def test_no_hessian_is_optional_but_must_be_complete_when_required() -> None:
    confirmatory = _complete_results()

    validate_result_table(confirmatory, assay_ids=ASSAYS, seeds=SEEDS)
    with pytest.raises(ValueError, match="required method"):
        validate_result_table(
            confirmatory,
            assay_ids=ASSAYS,
            seeds=SEEDS,
            required_methods=(*CONFIRMATORY_METHODS, NO_HESSIAN_METHOD),
        )

    no_hessian = confirmatory.loc[confirmatory["method"] == "ours"].copy()
    no_hessian["method"] = NO_HESSIAN_METHOD
    extended = pd.concat([confirmatory, no_hessian], ignore_index=True)
    validate_result_table(
        extended,
        assay_ids=ASSAYS,
        seeds=SEEDS,
        required_methods=(*CONFIRMATORY_METHODS, NO_HESSIAN_METHOD),
    )


def test_validate_v0_uses_protocol_assay_and_seed_cardinalities() -> None:
    protocol = load_protocol("configs/v0.yaml")

    validated = validate_v0_result_table(
        _complete_results(),
        assay_ids=ASSAYS,
        protocol=protocol,
    )
    assert len(validated) == protocol.assay_count * len(protocol.seeds) * 4

    with pytest.raises(ValueError, match="assay_count"):
        validate_v0_result_table(
            _complete_results(),
            assay_ids=ASSAYS[:-1],
            protocol=protocol,
        )


def test_pairwise_summary_clusters_within_assay_and_counts_strict_wins() -> None:
    validated = validate_result_table(
        _complete_results(), assay_ids=ASSAYS, seeds=SEEDS
    )

    summary = pairwise_summary(validated, first="ours", second="random")

    assert summary.assay_deltas == pytest.approx(tuple(ASSAY_DELTAS))
    assert summary.mean_gain == pytest.approx(0.15625)
    assert summary.standard_error == pytest.approx(0.08527136305432709)
    assert summary.task_wins == 30
    assert summary.task_total == 40
    assert summary.task_win_rate == pytest.approx(0.75)
    assert summary.assay_wins == 6
    assert summary.assay_total == 8
    assert summary.assay_win_rate == pytest.approx(0.75)
    assert summary.exact_sign_flip_pvalue == pytest.approx(34 / 256)
    with pytest.raises(FrozenInstanceError):
        summary.mean_gain = 1.0


def test_strict_ties_are_not_counted_as_task_or_assay_wins() -> None:
    table = _complete_results()
    ours = table["method"] == "ours"
    random = table["method"] == "random"
    table.loc[ours, "spearman"] = table.loc[random, "spearman"].to_numpy()
    validated = validate_result_table(table, assay_ids=ASSAYS, seeds=SEEDS)

    summary = pairwise_summary(validated, first="ours", second="random")

    assert summary.task_wins == 0
    assert summary.assay_wins == 0
    assert summary.mean_gain == pytest.approx(0.0)


def test_exact_sign_flip_enumerates_all_2_to_the_a_assignments() -> None:
    assert exact_sign_flip_pvalue(ASSAY_DELTAS) == pytest.approx(34 / 256)
    assert exact_sign_flip_pvalue(np.zeros(8)) == 1.0
    assert exact_sign_flip_pvalue(np.array([1e-16, 1e-16])) == 0.5


def test_hierarchical_bootstrap_is_deterministic_and_resamples_both_levels() -> None:
    table = _complete_results()
    # Add seed variation to distinguish seed-within-assay resampling from a
    # bootstrap over already averaged assay effects.
    mask = table["method"] == "ours"
    table.loc[mask, "spearman"] += np.tile(
        np.array([-0.04, -0.02, 0.0, 0.02, 0.04]),
        len(ASSAYS),
    )
    validated = validate_result_table(table, assay_ids=ASSAYS, seeds=SEEDS)

    first = hierarchical_bootstrap_interval(
        validated,
        first="ours",
        second="random",
        analysis_seed=991,
        n_resamples=500,
    )
    second = hierarchical_bootstrap_interval(
        validated,
        first="ours",
        second="random",
        analysis_seed=991,
        n_resamples=500,
    )
    changed_seed = hierarchical_bootstrap_interval(
        validated,
        first="ours",
        second="random",
        analysis_seed=992,
        n_resamples=500,
    )

    assert first == second
    assert first != changed_seed
    assert first.n_resamples == 500
    assert first.analysis_seed == 991
    assert first.lower < 0.15625 < first.upper
    with pytest.raises(FrozenInstanceError):
        first.lower = 0.0


def test_compact_method_and_comparison_tables_are_assay_macro() -> None:
    validated = validate_result_table(
        _complete_results(), assay_ids=ASSAYS, seeds=SEEDS
    )

    methods = method_summary_table(validated)
    comparisons = comparison_summary_table(
        validated,
        comparisons=(
            ("ours", "supervised"),
            ("ours", "random"),
            ("ours", "top_teacher"),
        ),
        analysis_seed=123,
        n_resamples=100,
    )

    assert list(methods.columns) == [
        "method",
        "mean_spearman",
        "se_spearman",
        "mean_mse",
        "mean_ndcg_10pct",
    ]
    ours_row = methods.set_index("method").loc["ours"]
    expected_ours_assay_means = np.array(
        [
            _complete_results()
            .query("method == 'ours' and assay_id == @assay_id")["spearman"]
            .mean()
            for assay_id in ASSAYS
        ]
    )
    assert ours_row["mean_spearman"] == pytest.approx(expected_ours_assay_means.mean())
    assert ours_row["se_spearman"] == pytest.approx(
        expected_ours_assay_means.std(ddof=1) / np.sqrt(8)
    )
    assert list(comparisons["comparison"]) == [
        "ours - supervised",
        "ours - random",
        "ours - top_teacher",
    ]
    random_row = comparisons.set_index("comparison").loc["ours - random"]
    assert random_row["mean_spearman_gain"] == pytest.approx(0.15625)
    assert random_row["task_wins"] == 30
    assert random_row["assay_wins"] == 6


def test_v0_verdict_signature_does_not_expose_methods_or_thresholds() -> None:
    assert tuple(signature(v0_analysis_verdict).parameters) == ("results",)


def test_v0_verdict_separates_selection_success_from_practical_improvement() -> None:
    positive_selection = validate_result_table(
        _complete_results(practical_gain=-0.03),
        assay_ids=ASSAYS,
        seeds=SEEDS,
    )

    verdict = v0_analysis_verdict(positive_selection)

    assert verdict.selection_success is True
    assert verdict.practical_self_improvement is False
    assert verdict.ours_minus_random.mean_gain > 0.0
    assert verdict.ours_minus_supervised.mean_gain < 0.0


def test_secondary_metrics_or_supervised_gain_cannot_rescue_primary_failure() -> None:
    table = _complete_results(practical_gain=0.20)
    # Reverse ours-random only, while making secondary metrics look favorable.
    ours = table["method"] == "ours"
    random = table["method"] == "random"
    table.loc[ours, "spearman"] = table.loc[random, "spearman"].to_numpy() - 0.01
    table.loc[ours, "mse"] = 0.0
    table.loc[ours, "ndcg_10pct"] = 1.0
    supervised = table["method"] == "supervised"
    table.loc[supervised, "spearman"] = table.loc[ours, "spearman"].to_numpy() - 0.5
    validated = validate_result_table(table, assay_ids=ASSAYS, seeds=SEEDS)

    verdict = v0_analysis_verdict(validated)

    assert verdict.selection_success is False
    assert verdict.practical_self_improvement is True


@pytest.mark.parametrize("assay_count", [7, 9])
def test_v0_verdict_requires_exactly_40_tasks_and_8_assays(assay_count: int) -> None:
    table = _complete_results()
    if assay_count == 7:
        table = table.loc[table["assay_id"].isin(ASSAYS[:7])].copy()
    else:
        extra = table.loc[table["assay_id"] == ASSAYS[0]].copy()
        extra["assay_id"] = "assay_8"
        table = pd.concat([table, extra], ignore_index=True)

    with pytest.raises(ValueError, match=r"exactly 40.*8 assays"):
        v0_analysis_verdict(table)


def test_v0_verdict_hard_codes_25_task_win_threshold() -> None:
    table = _complete_results()
    deltas = {
        ASSAYS[0]: (0.2, 0.2, 0.2, 0.2, 0.2),
        ASSAYS[1]: (0.2, 0.2, 0.2, 0.2, 0.2),
        ASSAYS[2]: (0.2, 0.2, 0.2, 0.2, 0.2),
        ASSAYS[3]: (0.2, 0.2, 0.2, 0.2, 0.2),
        ASSAYS[4]: (0.2, 0.2, 0.2, 0.2, -0.05),
        ASSAYS[5]: (-0.05, -0.05, -0.05, -0.05, -0.05),
        ASSAYS[6]: (-0.05, -0.05, -0.05, -0.05, -0.05),
        ASSAYS[7]: (-0.05, -0.05, -0.05, -0.05, -0.05),
    }
    for assay_id, assay_deltas in deltas.items():
        for seed, delta in zip(SEEDS, assay_deltas, strict=True):
            random_mask = (
                (table["assay_id"] == assay_id)
                & (table["seed"] == seed)
                & (table["method"] == "random")
            )
            random_value = float(table.loc[random_mask, "spearman"].iloc[0])
            ours_mask = (
                (table["assay_id"] == assay_id)
                & (table["seed"] == seed)
                & (table["method"] == "ours")
            )
            supervised_mask = (
                (table["assay_id"] == assay_id)
                & (table["seed"] == seed)
                & (table["method"] == "supervised")
            )
            table.loc[ours_mask, "spearman"] = random_value + delta
            table.loc[supervised_mask, "spearman"] = random_value + delta - 0.01

    verdict = v0_analysis_verdict(table)

    assert verdict.ours_minus_random.mean_gain > 0.0
    assert verdict.ours_minus_random.task_wins == 24
    assert verdict.ours_minus_random.assay_wins == 5
    assert verdict.selection_success is False
    assert verdict.practical_self_improvement is True


def test_v0_verdict_hard_codes_five_assay_win_threshold() -> None:
    table = _complete_results()
    deltas = {
        ASSAYS[0]: (0.2, 0.2, 0.2, 0.2, 0.2),
        ASSAYS[1]: (0.2, 0.2, 0.2, 0.2, 0.2),
        ASSAYS[2]: (0.2, 0.2, 0.2, 0.2, 0.2),
        ASSAYS[3]: (0.2, 0.2, 0.2, 0.2, 0.2),
        ASSAYS[4]: (0.01, 0.01, -0.1, -0.1, -0.1),
        ASSAYS[5]: (0.01, -0.1, -0.1, -0.1, -0.1),
        ASSAYS[6]: (0.01, -0.1, -0.1, -0.1, -0.1),
        ASSAYS[7]: (0.01, -0.1, -0.1, -0.1, -0.1),
    }
    for assay_id, assay_deltas in deltas.items():
        for seed, delta in zip(SEEDS, assay_deltas, strict=True):
            base_mask = (
                (table["assay_id"] == assay_id)
                & (table["seed"] == seed)
                & (table["method"] == "random")
            )
            random_value = float(table.loc[base_mask, "spearman"].iloc[0])
            ours_mask = (
                (table["assay_id"] == assay_id)
                & (table["seed"] == seed)
                & (table["method"] == "ours")
            )
            supervised_mask = (
                (table["assay_id"] == assay_id)
                & (table["seed"] == seed)
                & (table["method"] == "supervised")
            )
            table.loc[ours_mask, "spearman"] = random_value + delta
            table.loc[supervised_mask, "spearman"] = random_value + delta - 0.01

    verdict = v0_analysis_verdict(table)

    assert verdict.ours_minus_random.mean_gain > 0.0
    assert verdict.ours_minus_random.task_wins == 25
    assert verdict.ours_minus_random.assay_wins == 4
    assert verdict.selection_success is False


def test_no_hessian_rows_cannot_alter_v0_verdict() -> None:
    table = _complete_results()
    baseline = v0_analysis_verdict(table)
    no_hessian = table.loc[table["method"] == "ours"].copy()
    no_hessian["method"] = NO_HESSIAN_METHOD
    no_hessian["spearman"] = 1e6
    no_hessian["mse"] = -1e6
    no_hessian["ndcg_10pct"] = 1e6
    extended = pd.concat([table, no_hessian], ignore_index=True)

    assert v0_analysis_verdict(extended) == baseline


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (
            lambda frame: pairwise_summary(frame, first="missing", second="random"),
            "method",
        ),
        (
            lambda frame: hierarchical_bootstrap_interval(
                frame,
                first="ours",
                second="random",
                analysis_seed=-1,
                n_resamples=10,
            ),
            "analysis_seed",
        ),
        (
            lambda frame: hierarchical_bootstrap_interval(
                frame,
                first="ours",
                second="random",
                analysis_seed=1,
                n_resamples=0,
            ),
            "n_resamples",
        ),
    ],
)
def test_analysis_functions_reject_invalid_parameters(
    call: object,
    message: str,
) -> None:
    validated = validate_result_table(
        _complete_results(), assay_ids=ASSAYS, seeds=SEEDS
    )
    with pytest.raises(ValueError, match=message):
        call(validated)  # type: ignore[operator]


def test_result_dataclasses_reject_nonfinite_or_inconsistent_values() -> None:
    with pytest.raises(ValueError, match="finite"):
        BootstrapInterval(
            lower=np.nan,
            upper=1.0,
            confidence_level=0.95,
            n_resamples=100,
            analysis_seed=1,
        )
    with pytest.raises(ValueError, match="wins"):
        PairwiseSummary(
            first="ours",
            second="random",
            metric="spearman",
            mean_gain=0.1,
            standard_error=0.01,
            task_wins=11,
            task_total=10,
            task_win_rate=1.1,
            assay_wins=1,
            assay_total=2,
            assay_win_rate=0.5,
            exact_sign_flip_pvalue=0.5,
            assay_deltas=(0.0, 0.2),
        )
