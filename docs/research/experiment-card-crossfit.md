# Experiment Card: Cross-Fitted Outer-Gradient Repair

**Status:** Predeclared after the locked v0 negative decision and before computing any cross-fitted selector outcome.

**Question:** Does replacing the in-sample labeled outer gradient in the paper score with an out-of-fold labeled-risk gradient repair pseudo-sample selection on the frozen ProteinGym task?

**Motivation:** In locked v0, the full score lost to random selection on all eight assay means. Its rank alignment with the hidden test-gradient oracle was negative in all 40 tasks, while its favorable infinitesimal labeled-loss prediction failed after finite retraining in all 40. This card targets only the outer-gradient proxy; it does not rewrite v0.

**Baseline:** The exact locked random pseudo-label method: same 192 candidates, calibrated `ESM1v_ensemble` labels, `w=0.1`, ridge `lambda=0.01`, frozen ESM-2 embeddings, splits, and deterministic random stream.

**Variant:** Keep the full supervised estimator, candidate pseudo-gradients, regularized Hessian, damping, pseudo-labels, selected count, and exact retraining objective fixed. Replace only the score's left gradient.

For each assay-seed task, deterministically permute the 96 labeled indices with the purpose-separated seed `crossfit_outer_folds_v1` and form four folds of exactly 24 labels. For each fold, train the same ridge model on the other 72 standardized labels and compute prediction residual gradients on the 24 held-out labels. Average all 96 out-of-fold contributions:

\[
g_{\mathrm{CF}}
=
\frac{1}{96}\sum_{k=1}^{4}\sum_{i\in F_k}
\bigl(x_i^\top\hat\theta_{-k}-y_i\bigr)x_i.
\]

The cross-fitted score is

\[
s_j^{\mathrm{CF}}
=
g_{\mathrm{CF}}^\top(H+\rho I)^{-1}(g_j-g_L),
\]

where `g_j`, `g_L`, `H`, and `rho=1e-4` are exactly the locked v0 quantities at the full 96-label supervised estimator. Select the largest 192 scores with the same stable hash tie-break, then perform the exact locked weighted-ridge retraining.

**Single-change rule:** No teacher change, no pseudo-label filtering, no positive-score requirement, no hyperparameter tuning, no PCA, no change to `q`, `w`, `lambda`, damping, preprocessing, metrics, splits, or retraining. Full-data feature and label transforms remain fixed so the cross-fitted gradient is expressed in the same coordinates as the candidate gradients and Hessian.

**Screening data:** The development GFP assay plus the eight now-unblinded v0 assays. Every result on these assays is exploratory, even though the repair itself was predeclared before it was run.

**Screening metric:** Assay-macro paired test Spearman gain, `crossfit - random`, on the 8-by-5 v0 grid. Report task and assay wins, MSE, NDCG@10%, score/test-oracle alignment, gradient cosine, score distribution, overlap, selected pseudo-label error, and locality diagnostics.

**Screening promotion rule:** Positive mean Spearman gain, at least 25/40 task wins, and at least 5/8 assay-mean wins. This mirrors v0 and is a gate to replication, not a confirmatory claim.

**Untouched replication:** If the screen promotes, freeze the implementation and evaluate on the 26 eligible assays not used by v0 or GFP, with all five seeds. The primary replication comparison remains `crossfit - random`, aggregated within assay before inference. A positive claim requires positive assay-macro gain, at least 60% task wins, at least 60% assay wins, and an exact assay sign-flip test reported with uncertainty.

**Leakage guard:** Cross-fitted selection may consume only the 96 labeled outcomes and pre-unblinding inputs. Hidden unlabeled/test labels remain inaccessible until selections, coefficients, predictions, and a fit digest are frozen. Hidden labels may enter only evaluation and oracle diagnostics.

**Negative controls:** The locked v0 full score and no-Hessian method are carried as references but cannot be relabeled as confirmatory. A fixed-cardinality positive-only full-score branch is not run because all 40 v0 tasks had at least 192 positive full scores, so it would select exactly the v0 set.

**Stop/branch:** If cross-fitting fails the screen, next isolate perturbation size with a predeclared `w`/`q` locality curve before changing Hessian geometry. If it promotes but fails untouched replication, conclude that the apparent repair did not generalize.
