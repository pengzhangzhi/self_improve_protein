# Exact-CV Screen Decision

## CONFIRMED

No new confirmatory scientific claim is made. The preregistered exploratory
screen completed 45 GFP-plus-v0 tasks and 45 independent exact-rebuild
verification tasks under commit
`f94b94e5bb17f885b7d6afb2be64b2cb849ee2c0`. The receipt-verified aggregate
is `artifacts/exploratory/exact_cv_v1/screen/aggregate.json`, SHA-256
`5c415a12513f8f9d47aaeb75d79529f2aa720bbeb55e94750fec8d03a6d1759d`.
The required hidden-outcome-free, 480-dimensional fit probe is SHA-256
`93a129d6a6527ef3822b6cd62d39ca775fbecad26842c340e6b99a427ae38127`.

## EXPLORATORY

Exact greedy selection fit its four reused validation folds extremely well.
Across the 40 primary tasks, mean fold-validation MSE fell from `1.7297` to
`0.2174`; all 192 selected steps reduced the current fold objective on every
task. This did not generalize.

At the frozen `q=192` endpoint, exact-CV test MSE was `1.6177`, versus `1.1857`
for the locked random selector. The predeclared MSE gain
`random - exact-CV` was `-0.43205` (assay-cluster SE `0.10268`), with 2/40 task
wins, 0/8 assay wins, and exact assay sign-flip `p=0.0078125`.

Mean Spearman was `0.30336` for exact-CV and `0.34971` for random, a gain of
`-0.04635` (SE `0.01388`), with 6/40 task wins, 1/8 assay win, and exact
sign-flip `p=0.015625`. NDCG@10% was also lower by `0.00724` on average.

The validation/test contradiction is systematic: the task-level size of the
fold-CV improvement correlates negatively with test-MSE gain (Spearman
`-0.442`). Exact-CV selected slightly lower-error pseudo-labels than random on
average (`0.6675` versus `0.6896` MAE), so pseudo-label error alone does not
explain the failure.

## FAILED / INCOMPLETE

Both frozen promotion gates failed. A label-aware selector using exact finite
held-out squared-loss updates did not beat random pseudo-sample selection. The
result rules out the first-order/Hessian approximation as the sole cause of v0
failure in this setup.

The 26 untouched assay outcomes were not accessed. No replication was
authorized.

## HIGHEST VERIFIED RUNG

R5: fully verified exposed-assay mechanism/feasibility screen. The screen is
not an untouched confirmatory study.

## EVIDENCE GAPS

The common greedy ordering repeatedly adapts to four 24-example validation
folds, so validation overfitting is expected. Fold-specific transforms and the
final full-96 transform also define related but non-identical coordinate
systems. These limitations explain why exact-CV is an upper-bound feasibility
probe rather than a proposed production algorithm; they do not rescue its
failed hidden-test gate.

## RECOMMENDED NEXT

Stop further influence, damping, Hessian, and locality repairs for this frozen
teacher/student/task. Do not unseal the untouched assays. A future project
should change the research question—for example, introduce a genuinely
independent validation source, a teacher uncertainty model, or a different
label budget—and preregister that as a new study rather than tuning this result.
