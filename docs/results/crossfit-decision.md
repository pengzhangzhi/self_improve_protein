# Cross-Fit Screen Decision

## CONFIRMED

No new confirmatory scientific claim is made. The provenance-locked exploratory
screen completed all 45 GFP-plus-v0 assay-seed tasks under commit
`3fdb3dad404c725cbceefa79103ba5d96d6a0470`; the eight-assay primary slice
contains 40 tasks. The official aggregate is
`artifacts/exploratory/crossfit_v1/screen/aggregate.json`, SHA-256
`5cb8ca71bd2540e32ca467d4de233ec0727572f48ffbcaa62c8a3a3e3e1b88fc`.

## EXPLORATORY

Replacing the training-root outer gradient with the frozen four-fold
cross-fitted gradient did not repair selection. Cross-fit minus random was
`-0.05537` Spearman (assay-cluster SE `0.00697`, 2/40 task wins, 0/8 assay
wins, exact assay sign-flip `p=0.0078125`). It was essentially tied with the
original full score (`-0.00011` Spearman) and worse than top-teacher and
no-Hessian selection.

Its candidate ordering was negatively aligned with the hidden test-risk oracle
on all 40 primary tasks (mean correlation `-0.3719`). The full score was also
negative on all 40 (`-0.3404`).

## FAILED / INCOMPLETE

The preregistered promotion gate failed, so no untouched-assay replication was
authorized. The 26 protected assay outcomes were not read or summarized. This
screen also intentionally retained the global 96-label standardization and
teacher calibration, so it is a single-change diagnostic rather than a fully
held-out risk estimator.

## HIGHEST VERIFIED RUNG

R5: development/exposed-assay exploratory screen. R6 untouched replication was
intentionally not run.

## EVIDENCE GAPS

With only 96 labels and 480 features, a four-fold outer-gradient estimate may
be too noisy. More importantly, both learned score variants point opposite the
hidden test-risk oracle, so variance alone is not an adequate positive
explanation.

## RECOMMENDED NEXT

Test whether the first-order approximation becomes faithful at genuinely
smaller pseudo perturbations before attempting any replication.
