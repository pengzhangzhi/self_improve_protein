# Experiment Card: Full Hessian versus Identity Geometry

**Question:** Does the full inverse-Hessian geometry improve pseudo-sample selection over replacing \((H+\rho I)^{-1}\) by the identity?

**Hypothesis:** The full score increases assay-macro paired test Spearman over no-Hessian because curvature-corrected directions better approximate the effect of retraining.

**Baseline:** Select the top 192 values of `g_L.T @ (g_j - g_L)` and retrain with the same calibrated `ESM1v_ensemble` labels, `w=0.1`, `lambda=0.01`, features, and normalized solver.

**Variant:** Replace the identity in the baseline score with `inv(H + 1e-4 I)`; no other value changes.

**Primary metric:** Assay-macro mean paired test Spearman gain, `full - no_hessian`.

**Success:** Positive mean primary gain and at least 5/8 assay-mean wins. This result cannot alter the v0 `full - random` verdict.

**Guardrails:** Identical teacher labels/count/weight and deterministic ties; no hidden labels enter either selector; report overlap and effective-rank diagnostics.

**Data / split:** Exactly the frozen v0 assay IDs, working sets, seeds, and splits. The method runs in the same task wave but is keyed as a separate exploratory method.

**Seeds:** `[0, 1, 2, 3, 4]`, aggregated within assay before inference.

**Budget:** Negligible incremental GPU cost and less than one additional CPU-hour because embeddings are shared.

**Start rung:** Reuse v0 R1-R4 only after the no-Hessian explicit/vectorized parity test passes.

**Exploratory-only:** Relationships with Hessian spectrum, numerical effective rank, teacher quality, oracle-gradient alignment, and PCA dimension. These may explain the result but cannot change its success rule.
