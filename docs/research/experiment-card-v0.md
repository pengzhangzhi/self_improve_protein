# Experiment Card: External-Teacher Score-Ranking v0

**Question:** With identical calibrated `ESM1v_ensemble` pseudo-labels, does the paper-inspired full-Hessian score choose 192 variants that improve low-label ProteinGym test Spearman more than random choice?

**Hypothesis:** Full-score selection increases assay-macro paired test Spearman over random selection because its candidates have the largest predicted first-order decrease in labeled squared loss.

**Baseline:** Randomly select 192 candidates using the locked independent deterministic RNG stream, retrain the exact normalized weighted ridge with `w=0.1` and `lambda=0.01`, and evaluate on the same test set.

**Variant:** Replace only the random candidate indices with the top 192 values of `g_L.T @ inv(H + 1e-4 I) @ (g_j - g_L)`.

**Primary metric:** Assay-macro mean paired test Spearman gain, `ours - random`.

**Success:** Positive mean primary gain, at least 25/40 task wins, and at least 5/8 assay-mean wins.

**Guardrails:** No non-finite tasks; selections and fitted parameters are invariant to hidden-label perturbation; all pseudo methods use identical labels/count/weight; report MSE and NDCG@10% without using them to overturn the primary decision.

**Data / split:** ProteinGym v1.3 substitutions from Zenodo record `15293562`, literal teacher column `ESM1v_ensemble`, first eight lexicographically sorted assays with at least 6,000 usable variants and length at most 512. Per seed: 96 labeled, 2,000 unlabeled, 1,000 test from a fixed hash-sorted 6,000-row working set. The ninth eligible assay is development-only.

**Leakage check:** Selector and fit functions cannot accept hidden unlabeled/test labels. Freeze data checksums, assay IDs, working-set hashes, and split hashes before model evaluation; verify selection invariance after permuting hidden labels.

**Seeds:** `[0, 1, 2, 3, 4]`, fixed. Aggregate within assay before confirmatory inference.

**Budget:** At most one 8-GPU embedding wave, 32 A100-hours, and four wall-clock hours for v0 before stopping for diagnosis. Ridge tasks and aggregation use CPU arrays. Later ablations require separate cards.

**Start rung:** R1 static/import/config after this R0 card is committed.

**Exploratory-only:** Teacher quality, hidden pseudo-label error, score/error correlation, supervised and top-teacher comparisons, no-Hessian behavior, regularization/locality sweeps, cross-fitted scores, and results on any assay examined during development.
