# Experiment Card: Exact Cross-Validated Greedy Selection

**Status:** Predeclared before computing any exact-CV selector outcome. This is
an exploratory mechanism screen on the already exposed GFP development assay
and eight v0 assays. It cannot revise the locked v0 result.

**Question:** If candidates are ranked by their exact marginal effect on honest
held-out squared loss, rather than by a first-order influence proxy, can a
common pseudo-labeled set improve the final low-label ridge model over random
pseudo-labeling?

**Motivation:** The influence selectors did not beat random selection on the
exposed assays. This branch isolates whether that failure comes from the local
score approximation or from the pseudo-labeling problem itself. It replaces
the influence score with exact fold-held-out squared-loss evaluation while
keeping the final v0 student, teacher, pseudo-label budget, and random comparator
fixed. This is a new greedy cross-validation method, not a test of the paper's
first-order theorem.

## Frozen inputs and screen

Use the GFP development assay plus the eight v0 assays, all five frozen seeds:
45 assay-seed tasks in total. The promotion gate uses only the 40 tasks from the
eight v0 assays; GFP remains development-only. Each task reuses the exact v0
working set, 96 labeled examples, 2,000-candidate unlabeled pool, 1,000-example
test set, frozen ESM-2 embeddings, raw `ESM1v_ensemble` teacher score,
`lambda=0.01`, `w=0.1`, preprocessing definitions, metrics, and stable row-hash
tie-breaking.

The final selected count is fixed at

\[
q=192.
\]

There is no positive-gain filter, early stopping, hyperparameter sweep, or
outcome-dependent choice of `q`.

## Honest four-fold construction

Deterministically permute the 96 labeled indices with a purpose-separated
`exact_cv_folds_v1` seed and partition them into four folds
\(F_1,\ldots,F_4\) of exactly 24 examples. For fold \(k\), its 72-example
training complement \(T_k=L\setminus F_k\) alone determines:

- the feature mean and scalar RMS scale;
- the target mean and population standard deviation;
- the affine teacher calibration, including intercept; and
- every ridge state used to score candidates for that fold.

Apply those training-only transforms to \(F_k\) and to the common 2,000-row
candidate pool. No statistic fitted on \(F_k\), another fold's validation
labels, or a hidden candidate/test outcome may enter fold \(k\)'s transform,
teacher calibration, pseudo-labels, or normal equations. The only declared
coupling across folds is the common candidate decision made from their four
validation losses.

All four folds advance through one **common** greedy ordering. They do not
produce four sets that are subsequently unioned. At step \(r\), every remaining
candidate is evaluated in all four fold-specific states, the four validation
MSEs are averaged, and the single candidate with the lowest mean post-addition
MSE is appended. Exact ties use the frozen stable row-hash order. The selected
candidate is then added to all four states with that fold's transformed feature
and fold-specific calibrated pseudo-label.

This is honest with respect to each candidate's fold-level fit, but the common
ordering is adaptively chosen from repeated use of the same four validation
folds. The card therefore treats the entire screen as exploratory and does not
equate cross-validation greed with independent generalization evidence.

## Fixed-final-denominator ridge states

The greedy path is defined relative to the final requested budget, not a
sequence of differently normalized prefix objectives. For every fold and every
prefix length \(r=0,\ldots,q\), fix

\[
D=72+wq=72+0.1\times192=91.2.
\]

For a current common prefix \(S_r\), fold \(k\)'s exact state is

\[
\theta_{k,r}
=
A_{k,r}^{-1}b_{k,r},
\]

with

\[
A_{k,r}
=
X_{T_k}^{\top}X_{T_k}
+w\sum_{j\in S_r}x_{j,k}x_{j,k}^{\top}
+D\lambda I,
\]

\[
b_{k,r}
=
X_{T_k}^{\top}y_{T_k}
+w\sum_{j\in S_r}x_{j,k}\hat y_{j,k}.
\]

Equivalently, this minimizes the prefix data terms divided by the fixed
final denominator \(D\), plus \(\lambda\lVert\theta\rVert_2^2/2\). Crucially,
the construction does **not** use \(72+wr\) at step \(r\). A changing
denominator would change the effective regularization at every step and would
make the path answer a different question. Because \(D\) is fixed, each
candidate is an exact rank-one addition to a common final-budget objective.

For candidate \(j\notin S_r\), compute its hypothetical state without
refitting from scratch:

\[
\theta_{k,r+1}^{(j)}
=
\theta_{k,r}
+
\frac{w A_{k,r}^{-1}x_{j,k}
(\hat y_{j,k}-x_{j,k}^{\top}\theta_{k,r})}
{1+w x_{j,k}^{\top}A_{k,r}^{-1}x_{j,k}}.
\]

Its selection criterion is the mean **unhalved** validation MSE of
\(\theta_{k,r+1}^{(j)}\) over the four 24-example folds. After selecting the
best candidate, update the inverse exactly by Sherman--Morrison:

\[
A_{k,r+1}^{-1}
=
A_{k,r}^{-1}
-
\frac{w A_{k,r}^{-1}x_{j,k}x_{j,k}^{\top}A_{k,r}^{-1}}
{1+w x_{j,k}^{\top}A_{k,r}^{-1}x_{j,k}}.
\]

Direct-solve parity checks on synthetic and real-shape hidden-label-free inputs
must pass before an official task is allowed to run. The implementation must
also verify finite Sherman--Morrison denominators greater than or equal to one
up to declared numerical tolerance, and reconstruct the final four states from
the frozen ordering.

## Final model and baseline

After the complete 192-candidate ordering is frozen, discard the four fold
models. Refit the feature transform, target transform, and teacher calibration
on all 96 labels exactly as in v0. Recompute the selected candidates'
pseudo-labels under this full-96 calibration and fit the exact normalized v0
weighted ridge objective with denominator

\[
96+wq=115.2.
\]

Thus the deployed exact-CV and random methods differ only in the selected
candidate indices. The comparator is the locked v0 random-pseudo set of 192
candidates, with the same full-96 transforms, calibrated pseudo-labels,
`w=0.1`, `lambda=0.01`, solver, predictions, and metrics.

## MSE-first dual promotion gate

The squared-loss endpoint is evaluated first because it matches the greedy
selection objective. Define task-level MSE gain as

\[
\Delta_{\mathrm{MSE}}
=
\mathrm{MSE}_{\mathrm{random}}
-
\mathrm{MSE}_{\mathrm{exactCV}},
\]

and Spearman gain as

\[
\Delta_{\rho}
=
\rho_{\mathrm{exactCV}}
-
\rho_{\mathrm{random}}.
\]

Aggregate the five seeds within assay before the assay-macro mean. Promotion
requires **both** gates on the eight-assay, 40-task screen:

1. MSE gain is positive in the assay-macro mean, exact-CV wins at least 25/40
   tasks, and it wins at least 5/8 assay means.
2. Spearman gain is positive in the assay-macro mean, exact-CV wins at least
   25/40 tasks, and it wins at least 5/8 assay means.

If the MSE gate fails, stop: exact held-out squared-loss greed did not repair
the method on exposed assays, regardless of an isolated rank observation. If
MSE passes but Spearman fails, conclude that squared-risk selection did not
transfer to the protein-ranking endpoint. Only passing both gates may motivate
a separately preregistered replication; passing is not itself a confirmatory
claim.

## Hidden-label boundary and diagnostics

Selection may consume the 96 labeled outcomes through the declared honest
folds. Candidate true DMS scores and all 1,000 test outcomes are forbidden
until the full 192-step ordering, the selected set, all full-96 coefficients
and predictions, and a canonical fit digest have been frozen and verified.
Candidate/test labels may then enter evaluation only. Hidden-label permutation
must leave the ordering, selected set, coefficients, predictions, and fit digest
unchanged.

Before unblinding, also freeze full-96 fits and predictions for the nested
prefixes

```text
q_prefix in {24, 48, 72, 96, 192}.
```

Each descriptive prefix is refit with the actual normalized deployed objective
for that prefix, using denominator \(96+wq_{\mathrm{prefix}}\). The fixed
endpoint denominator \(96+w192\) is used only for the 192-point deployed fit;
the fold-level greedy surrogate retains its separately frozen
\(D=72+w192\) at every selection step.

After unblinding, report their fold-CV MSE trajectory, test MSE and Spearman,
marginal validation gains, selected pseudo-label error, and overlap with the
random and prior influence selections. These prefix results are **descriptive
only**: they cannot select a different `q`, enter the promotion gate except at
the frozen `q=192` endpoint, or be described as a tuned result.

## Untouched-outcome prohibition

No outcome from any of the 26 untouched eligible assays may be read, joined,
evaluated, summarized, or used to modify this method under this card. The
exact-CV screen remains exploratory because all nine assays it uses are already
exposed. Even if both promotion gates pass, untouched outcomes remain sealed
until a new replication card, frozen implementation/trust root, and explicit
replication launch are in place.
