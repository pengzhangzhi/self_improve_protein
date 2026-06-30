# Conditional Experiment Card: Pseudo-Perturbation Locality Curve

**Status:** Predeclared before evaluating the cross-fitted screen. Run only if
the cross-fitted outer-gradient repair fails its screening gate. This is an
exploratory mechanism study on already unblinded assays, not a confirmatory
rescue of v0.

**Question:** Did v0 fail because its finite pseudo-labeled update was too far
from the paper's local perturbation regime, and does score usefulness emerge at
smaller pseudo-sample counts or weights?

**Motivation:** Locked v0 used `q=192=2n`, `w=0.1`, and `t=1/6`. The theorem's
stated regime requires both a small objective perturbation and
`sqrt(n) << q << n`; the v0 count violates `q << n` even though down-weighting
makes `t` moderate. V0 also showed very large locality indices and poor
first-order displacement accuracy. This card varies only perturbation size.

**Frozen inputs:** Use the GFP development assay and the eight v0 assays, all
five seeds, the same 96 labels, 2,000 candidates, 1,000 test examples, frozen
ESM-2 embeddings, calibrated `ESM1v_ensemble` teacher, ridge `lambda=0.01`,
Hessian damping `rho=1e-4`, preprocessing, splits, metrics, tie-breaking, and
hidden-label boundary. No untouched replication assay may be used.

**Factorial curve:** Evaluate

```text
q in {24, 48, 72, 96, 192}
w in {0.01, 0.03, 0.10}
```

for random, the locked full score, the cross-fitted score, and no-Hessian score.
For each selector, compute one deterministic complete candidate ordering per
task and use prefixes so selected sets are nested across `q`. Retrain the exact
weighted ridge objective separately for every `(q, w)` pair. Selection scores
do not depend on hidden labels or `w`.

**Interpretation axes:** Report effective pseudo fraction
`t=wq/(96+wq)`, mean Spearman gain over the matched random prefix, task and
assay wins, MSE, NDCG@10%, selected pseudo-label error, predicted versus
realized labeled-loss change, test-oracle score alignment, displacement cosine
and relative error, and locality index.

**Mechanism readout:** The local-perturbation explanation is supported only if
first-order displacement/sign diagnostics improve monotonically as `t`
shrinks and at least one influence selector has positive assay-macro Spearman
gain over matched random in the sub-`n` region `q in {24, 48, 72}`. Performance
at `q>=96` cannot by itself support the theorem's stated count regime.

**No tuning claim:** The best cell is descriptive and may motivate a new,
separately frozen replication. It cannot be reported as confirmatory and may
not be run on the 26 untouched assays under this card.

**Stop/branch:** If no selector shows improving first-order fidelity as `t`
shrinks, reject finite-step locality as the primary explanation and next test
the score's outer-risk proxy or pseudo-label-error sensitivity under a new
card. If fidelity improves but ranking does not, conclude that the local
parameter approximation is not sufficient for useful variant selection.
