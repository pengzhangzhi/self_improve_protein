# ProteinGym Low-Label Pseudo-Sample Selection Design

**Status:** Approved by the user's autonomous execution mandate on 2026-06-29.

**Pre-outcome amendment (2026-06-29):** Inspection of the v1.3 merged files corrected the literal teacher column from the upstream input-score name `Ensemble_ESM1v` to the distributed merged-column name `ESM1v_ensemble`. The source was pinned to Zenodo record `15293562` because ProteinGymPy 0.9.3 still points its zero-shot loader at v1.2. No method outcome had been computed.

**Pre-outcome provenance amendment (2026-06-29):** The ProteinGym tag `PG_v1.3` was peeled to upstream commit `1f8de974dead8ff7501eff087b725d14a965e9f9`; the Zenodo metadata was confirmed byte-identical at SHA-256 `a8f498011532a74aa9fe556a50555a75e928c5837d19c06a87592ae04049b308`. The representation is pinned to ESM-2 revision `6fbf070e65b0b7291e7bbcd451118c216cff79d8`. These values and all three source-archive digests were written into the strict protocol before embedding. No method outcome had been evaluated.

## Research goal contract

The research outcome is a defensible yes/no answer to whether the proposed first-order score chooses externally pseudo-labeled protein variants that improve low-label DMS fitness ranking relative to random selection. The evidence bar is a predeclared 8-assay by 5-seed study with a locked primary metric, followed by an untouched-assay replication if the result is positive or ambiguous. A green software test suite is necessary but is not evidence for the method.

The starting state is a new repository, the supplied paper and experiment proposal, ProteinGym v1.3, and a CPU login node with Slurm access to A100 workers. Compute is used to minimize wall-clock time after cheap gates pass. The first stopping point is an R7 decision memo that separates supported conclusions from exploratory observations. Work pauses only for an external blocker such as missing data access or GitHub repository-creation credentials, not for routine implementation choices.

## Claim boundary from the mathematical audit

For squared-loss ridge, the paper's literal self-teacher is degenerate. At the supervised optimum,

\[
g_L=\frac{1}{n}X_L^\top(X_L\hat\theta-y_L)=-\lambda\hat\theta.
\]

If the pseudo-label is the student's own prediction, then every pseudo-gradient is zero. Consequently,

\[
s_j=-g_L^\top(H+\rho I)^{-1}g_L\leq 0
\]

for every candidate. There is no ranking signal, and the paper's positive-only algorithm selects nothing. This identity is a required regression test and a negative control.

The confirmatory experiment therefore tests a narrower generalized method:

> external-teacher, influence-ranked pseudo-label selection for low-label regression.

A positive result supports this empirical selection rule. It does not validate the paper's literal self-teacher algorithm or its current asymptotic proof. The score predicts an infinitesimal decrease in unregularized labeled training loss; held-out improvement is an empirical hypothesis. The proposed configuration has \(q=192>n=96\) and \(t=1/6\), so it is described as a finite moderate perturbation rather than as an asymptotic local-regime test.

## Approaches considered

1. **Literal self-teacher.** This is retained only as an algebraic negative control because its scores tie at a non-positive constant.
2. **Fixed external teacher.** This is the recommended and locked v0. A non-identical ProteinGym ESM-1v ensemble supplies pseudo-labels; all methods differ only in candidate selection.
3. **Cross-fitted risk-gradient score.** This is the first exploratory repair if v0 fails. It estimates the outer gradient on held-out labeled folds instead of reusing the in-sample training gradient. It is not allowed to replace v0 after results are observed.

## Locked confirmatory protocol

### Data release, teacher, and assay selection

- Pin the official ProteinGym substitution benchmark to release `v1.3` at Zenodo record `15293562`.
- Use the literal zero-shot score column `ESM1v_ensemble` for every assay, with no per-assay fallback and no outcome-based teacher choice.
- Join DMS rows and teacher scores one-to-one on `mutant`.
- A usable row has a finite `DMS_score`, a finite `ESM1v_ensemble` score, a non-empty `mutated_sequence`, a sequence length no greater than 512, and a unique mutated sequence within its assay.
- An eligible assay has at least 6,000 usable rows and target length no greater than 512.
- Sort eligible assays lexicographically by `DMS_id`; the first eight form the confirmatory set. The ninth eligible assay is development-only and is excluded from confirmatory and replication summaries.
- Record release URLs, SHA-256 checksums, upstream metadata commit, selected assay IDs, row counts, and sequence hashes in an immutable manifest before embeddings are computed.

The pre-outcome eligibility result fixes the confirmatory IDs as `ADRB2_HUMAN_Jones_2020`, `AMIE_PSEAE_Wrenbeck_2017`, `CCR5_HUMAN_Gill_2023`, `CP2C9_HUMAN_Amorosi_2021_abundance`, `CP2C9_HUMAN_Amorosi_2021_activity`, `D7PM05_CLYGR_Somermeyer_2022`, `F7YBW8_MESOW_Aakre_2015`, and `F7YBW8_MESOW_Ding_2023`. `GFP_AEQVI_Sarkisyan_2016` is development-only.

Within each assay, sort usable rows by SHA-256 of the UTF-8 tuple `(DMS_id, mutant, mutated_sequence)` and take the first 6,000 as the fixed working set. This avoids dependence on source-file row order without using labels to choose examples.

### Splits and random streams

Use seeds `[0, 1, 2, 3, 4]`. Derive independent PCG64 streams from SHA-256 of `(DMS_id, seed, purpose)` for split construction and random pseudo-selection. For each split permutation, assign the first 96 rows to labeled training, the next 2,000 to the unlabeled pool, and the next 1,000 to test. Assert uniqueness and pairwise disjointness by sequence hash.

The selector and fitting APIs never receive hidden unlabeled or test labels. Hidden unlabeled labels are joined only by the diagnostic stage after all selection hashes and model predictions are written.

### Frozen representation and preprocessing

Use `facebook/esm2_t12_35M_UR50D` in evaluation mode. Mean-pool the last hidden state over residue tokens only, excluding BOS, EOS, and padding. Cache one float32 embedding matrix for each 6,000-row assay working set; use float64 for ridge solves, scores, and metrics.

For each assay-seed task, compute from the 96 labeled embeddings only:

\[
\mu_x=\frac{1}{n}\sum_i e_i,\qquad
c_x=\sqrt{\frac{1}{np}\sum_i\lVert e_i-\mu_x\rVert_2^2}.
\]

Transform every embedding as \(x=(e-\mu_x)/c_x\). This single scalar scale avoids noisy per-coordinate variance estimates while fixing the numerical meaning of \(\lambda\). Fit no intercept. Standardize labels with labeled-only mean and population standard deviation (`ddof=0`). Fail the task if either scale is non-finite or effectively zero.

### Teacher calibration

Fit ordinary least squares with an intercept on the 96 labeled examples in standardized-label space:

\[
(a,b)=\arg\min_{a,b}\sum_{i\in L}(az_i+b-y_i^{\mathrm{std}})^2,
\qquad \hat y_j=az_j+b.
\]

Do not constrain the slope sign. Save slope, intercept, rank correlation, variance, and finite-coverage diagnostics.

### Student, score, and retraining

The supervised objective is

\[
\frac{1}{2n}\lVert X_L\theta-y_L\rVert^2+\frac{\lambda}{2}\lVert\theta\rVert^2,
\qquad \lambda=10^{-2}.
\]

The exact solution is

\[
\hat\theta_0=(X_L^\top X_L+n\lambda I)^{-1}X_L^\top y_L.
\]

Define

\[
g_L=X_L^\top(X_L\hat\theta_0-y_L)/n,
\quad H=X_L^\top X_L/n+\lambda I,
\quad v=(H+\rho I)^{-1}g_L,
\]

with \(\rho=10^{-4}\). For external pseudo-label residuals \(r_U=X_U\hat\theta_0-\hat y_U\), compute scores efficiently as

\[
s=r_U\odot(X_Uv)-g_L^\top v.
\]

Select exactly \(q=192\) candidates by descending score with stable hash tie-breaking. Do not require positive scores in v0. The no-Hessian ablation replaces \(v\) by \(g_L\).

For a selected set \(S\), use pseudo weight \(w=0.1\), \(D=n+wq\), and the exact normalized-objective solution

\[
\hat\theta_S=
\left[X_L^\top X_L+wX_S^\top X_S+D\lambda I\right]^{-1}
\left[X_L^\top y_L+wX_S^\top\hat y_S\right].
\]

The factor `D * lambda` is part of the method and must not be replaced by a constant library default.

### Methods

The four confirmatory methods are supervised-only, random pseudo, top-teacher pseudo, and full score-selected pseudo. All pseudo methods use the same calibrated labels, \(q=192\), \(w=0.1\), features, and solver. Random selection uses its independent deterministic stream. Top-teacher and score ties use the same stable sequence-hash rule.

The self-teacher, no-Hessian, cross-fitted, positive-only, regularization, locality, and teacher-student-disagreement variants are separate negative-control or exploratory cards. They cannot alter the v0 result. The no-Hessian card is executed in the same task wave because it is nearly free and was predeclared before ProteinGym method outcomes, but it is analyzed as a separate mechanism experiment.

### Metrics and inference

The locked primary endpoint is the assay-macro mean paired Spearman gain of score-selected over random pseudo-labeling. Compute task-level Spearman on the 1,000 held-out examples, then average the five paired deltas within each assay, then average the eight assay means.

Secondary metrics are standardized-label MSE and ProteinGym-style NDCG@10%: min-max normalize true test fitness to non-negative gains, compute DCG over the top 10% of the predicted ranking, and divide by ideal DCG over the true top 10%. Save raw predictions so all metrics can be recomputed.

Report both task-level descriptive summaries and assay-clustered inference. The confirmatory standard error is the standard deviation of eight assay-mean deltas divided by \(\sqrt{8}\). Compute an exact two-sided sign-flip test over all \(2^8\) assay-level sign assignments and a deterministic hierarchical bootstrap that resamples assays, then seeds within assays. A paired iid test over 40 rows may be shown only as a non-confirmatory legacy comparison.

The v0 success rule is fixed before results:

- mean assay-macro `ours - random` Spearman is positive;
- at least 25 of 40 task-level pairs favor ours; and
- at least 5 of 8 assay-mean pairs favor ours.

This rule decides the **selection hypothesis**. A separate **practical self-improvement** label requires the full score to have positive assay-macro Spearman gain over supervised-only. Beating top-teacher is a stronger secondary result. Neither secondary comparison can substitute for failure against random, and beating random while losing to supervised is reported as selection success without practical self-improvement.

### Required diagnostics

Save teacher test Spearman; score min/max/standard deviation/quantiles/positive fraction/unique count; mean selected score for ours and random; overlap with top-teacher; hidden unlabeled pseudo-label absolute error for analysis only; teacher-student residual distributions; Hessian eigenvalue spectrum, numerical effective rank, condition number, and effective ridge degrees of freedom; \(\lVert g_L+\lambda\theta_0\rVert\); selection hashes; and random-selection Monte Carlo sensitivity.

After predictions and selection hashes are frozen, compute an analysis-only oracle influence score by replacing the labeled outer gradient with the test gradient. Report its rank correlation with the proposed score and the \((H+\rho I)^{-1}\)-metric cosine between labeled and test gradients. These diagnostics explain proxy alignment but never enter selection.

Lower hidden pseudo-label error is not required by the influence argument. It is interpretive evidence, not a success criterion.

## Software architecture

The package is divided into focused units:

- `config`: validated immutable protocol values and paths;
- `data`: official downloads, one-to-one joins, usable-row filtering, hashing, manifests, and splits;
- `embeddings`: residue-only ESM mean pooling and atomic caches;
- `ridge`: exact supervised and weighted normalized ridge solvers;
- `selection`: random, teacher-top, full-score, no-Hessian, and negative-control selectors;
- `metrics`: Spearman, MSE, and exact NDCG@10%;
- `experiment`: one assay-seed task with strict hidden-label boundaries;
- `analysis`: aggregation, clustered inference, diagnostics, and decision tables;
- `provenance`: checksums, environment, git revision, and artifact schemas.

Command-line entry points prepare data, embed one assay, run one task, aggregate completed shards, and render a decision memo. Every stage is restart-safe: write to a temporary path, validate schema and checksum, then atomically rename. Completion is determined by validated artifacts rather than by the mere existence of a file.

The public repository contains code, small manifests, tests, and final compact result tables. It excludes benchmark data, model weights, embeddings, task shards, logs, credentials, and cluster-specific local profiles.

## Correctness and empirical verification

Before a cluster study, tests must establish the ridge normal equations, vectorized versus explicit score equivalence, finite-difference score sign, self-teacher constant non-positive scores, zero OLS scores at \(\lambda=0\), weighted regularization normalization, no-Hessian equivalence, deterministic disjoint splits, hidden-label invariance, residue-token pooling, NDCG parity with the pinned ProteinGym implementation, and atomic artifact behavior.

The staged R0-R7 gates are defined in `docs/research/feedback-ladder.md`. The confirmatory run is not launched until a development-only assay completes through R5. A development result can reveal broken plumbing or a clearly unusable teacher, but it cannot count toward the confirmatory endpoint.

## Planned branching after v0

If v0 is positive or ambiguous, run the same locked protocol on all remaining eligible assays except the development assay as an untouched replication. If v0 is negative, preserve that result and use the already-predeclared no-Hessian result before opening new exploratory cards in this order: cross-fitted risk gradient, positive-only matched-cardinality selection, labeled-only regularization choice, locality curves over \(t\), PCA/effective-dimension stress tests, and simple teacher-student disagreement baselines. Any promising exploratory method requires a new untouched assay set or a new benchmark for confirmation.

## Publication

The local repository uses the personal identity `fredpeng <pengzhangzhics@gmail.com>` and will target `git@github.com:pengzhangzhi/self_improve_protein.git`. Before publication, scan tracked files and history for secrets, private cluster paths, proprietary code, large artifacts, and benchmark redistribution. The GitHub repository must be explicitly public.
