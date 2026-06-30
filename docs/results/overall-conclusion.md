# Overall Scientific Conclusion

## Bottom line

The proposed influence score does not identify pseudo-labeled ProteinGym
variants that outperform random pseudo-sample selection in the frozen
ESM1v-teacher, ESM-2-embedding, ridge-student setting. This is a stable negative
result, not a compute-limited or incomplete run.

The strongest supported positive finding is narrower: calibrated external
teacher pseudo-labels are useful. Randomly adding them improves mean Spearman
from `0.29309` to `0.34971`. The failure is in deciding which pseudo-labels to
add, not in the existence of teacher signal.

## Evidence chain

1. The locked 8-assay by 5-seed v0 study found full-Hessian influence selection
   `-0.05526` Spearman below random, with 0/8 assay wins. This is the R7
   confirmatory result for the implemented external-teacher selection rule.
2. Cross-fitting the labeled outer gradient did not help: crossfit was
   `-0.05537` Spearman below random, with 0/8 assay wins. Both the original and
   cross-fitted scores were negatively aligned with an analysis-only test-risk
   oracle on every task.
3. Reducing the pseudo perturbation made the first-order displacement
   approximation substantially more faithful, but did not recover selection
   utility. At the smallest tested cell, `q=24, w=0.01`, full-Hessian selection
   remained `-0.01067` Spearman below random with 0/8 assay wins. Neither full
   nor cross-fitted influence selection beat random in any of 15 locality cells.
4. Exact four-fold validation-risk greedy selection removed the first-order and
   Hessian approximations. It drove reused fold-validation MSE from `1.7297` to
   `0.2174`, yet its test MSE was `1.6177` versus random's `1.1857`, and its
   Spearman was `0.30336` versus `0.34971`. The larger the apparent CV gain, the
   worse the test-MSE gain tended to be (task Spearman `-0.442`).

Together these studies rule out outer-gradient leakage, perturbation size, and
the Hessian/Taylor approximation as sole explanations of the failure. The
validation/test contradiction is consistent with adaptive validation
overfitting and/or selection-surrogate mismatch; these screens do not identify
one of those mechanisms as the unique cause.

## Claim boundary

This conclusion applies to one teacher, one representation, one 96-label
budget, and the selected ProteinGym substitution assays. It is not a general
impossibility result for pseudo-label selection or protein fitness prediction.
The literal squared-loss self-teacher algorithm in the manuscript is not an
alternative rescue: at the supervised ridge optimum its first-round
pseudo-gradients vanish, so every candidate receives the same score.

The 26 designated untouched assay outcomes remained sealed because every
predeclared exploratory promotion gate failed. Exact-CV therefore reached R5
(verified exposed-assay mechanism screen), while the original locked v0 reached
R7 for its narrow confirmatory claim.

## Research decision

Do not tune damping, Hessian variants, score sign, pseudo weight, or selection
budget further on these outcomes, and do not unseal the untouched assays. A new
study should change the information available to selection—for example, a
genuinely independent validation source, teacher uncertainty or ensembles, or
a larger labeled selection set—and preregister that question independently.

The current ProteinGym result is publishable as a rigorous negative or
diagnostic result, but it does not support the manuscript's intended positive
self-improvement claim.

## Evidence roots

- v0 aggregate SHA-256: `d9426a88554f6e84d9a2b6995a4aa319f9c756570f8df4b22240136e885fabe3`
- crossfit aggregate SHA-256: `5cb8ca71bd2540e32ca467d4de233ec0727572f48ffbcaa62c8a3a3e3e1b88fc`
- locality aggregate SHA-256: `917ac35d9ef2e9f885c1b774cb590e231056e35e97d43a579e0f362372ec97cf`
- exact-CV aggregate SHA-256: `5c415a12513f8f9d47aaeb75d79529f2aa720bbeb55e94750fec8d03a6d1759d`
