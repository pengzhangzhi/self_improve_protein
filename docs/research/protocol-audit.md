# Protocol-to-Evidence Audit

**Scope:** External-teacher ProteinGym v0, locked 2026-06-29. This audit maps
the approved design to executable code, tests, and retained artifacts. It is a
software/science traceability document, not a result. No ProteinGym outcome was
read or summarized to produce it.

## Evidence boundary

There are four different claims in this project, and evidence for one must not
be presented as evidence for another:

1. **Paper theory.** The supplied theorem studies sequential pseudo-labels from
   the current student in a local asymptotic regime.
2. **External-teacher extension.** Confirmatory v0 instead freezes calibrated
   ProteinGym `ESM1v_ensemble` labels and ranks them with the paper-inspired
   score. This is a scientifically motivated extension, not a literal test of
   the theorem.
3. **Plumbing and algebra.** R1-R5 establish that the implementation is
   deterministic, leakage-safe, numerically faithful, and runnable. Synthetic
   finite differences can verify the local score calculation.
4. **Method effectiveness.** Only the frozen 8-assay by 5-seed R6 outcomes and
   the R7 locked decision rule can establish whether the method improves
   protein fitness ranking. Passing tests or a development-assay result cannot.

The literal squared-loss self-teacher is retained as a negative control. At the
supervised ridge optimum, `g_L = -lambda * theta`; its pseudo-gradient is zero,
so every full-H score equals the same non-positive constant. The confirmatory
teacher avoids this algebraic degeneracy, but doing so changes the method being
tested.

## Design-to-code traceability

| Locked requirement | Implementation / enforcement | Tests | Retained evidence |
| --- | --- | --- | --- |
| ProteinGym v1.3, Zenodo record `15293562`, three exact source digests, peeled upstream commit | Strict immutable fields in `configs/v0.yaml` and `config.Protocol`; source verification and manifest construction in `data` | `test_config.py`, `test_data.py`, `test_provenance.py` | Frozen data manifest at R4-R6; R1 config dump and digest |
| Fixed teacher `ESM1v_ensemble`, no per-assay fallback | Required config field; exact source projection and one-to-one `mutant` join in `data` | `test_config.py`, `test_data.py` | Data-manifest coverage and source hashes |
| Usable row: finite label/teacher, non-empty unique sequence, length at most 512 | `data.usable_rows` validation/filtering and uniqueness checks | `test_data.py` | Per-assay usable counts and exclusion counts in manifest |
| Eligible assays have at least 6,000 rows; lexicographic first eight confirmatory, ninth development-only | Deterministic eligibility and assay selection in `data`; config fixes `working_size=6000`, `assay_count=8` | `test_data.py`, `test_config.py` | Immutable selected-assay list in manifest |
| Working set independent of source order and outcomes | SHA-256 identity of `(DMS_id, mutant, mutated_sequence)`, hash sort, first 6,000 | `test_data.py`, `test_provenance.py` | Ordered row hashes in data manifest |
| Five fixed seeds and purpose-separated PCG64 streams | `provenance.derive_seed`; split and selector call sites use distinct purpose strings | `test_provenance.py`, `test_data.py`, `test_selection.py`, `test_experiment.py` | Split/selection hashes in task artifacts |
| Per task: 96 labeled, 2,000 unlabeled, 1,000 test; unique and disjoint | Strict config cardinalities and deterministic split construction | `test_config.py`, `test_data.py` | Ordered split hashes and task identity |
| Selector and fit never receive hidden unlabeled/test outcomes | Separate immutable `experiment.FitInputs` and `EvaluationLabels`; external trusted fit/evaluation digests; verified identity/hash join before evaluation | `test_experiment.py` hidden-label invariance, digest tampering, reordering, and unblinding tests | Fit artifact frozen before evaluation; separate evaluation artifact |
| ESM-2 35M exact revision, evaluation mode, last-state residue-only mean pooling | Model/revision literals in config; strict special-token mask and identity-coupled cache in `embeddings` | `test_config.py`, `test_embeddings.py` | Embedding metadata, row digest, shape/dtype/checksum |
| Cache 6,000 x 480 float32 embeddings; use float64 downstream | `embeddings` cache schema; float64 coercion/validation in `ridge`, `selection`, `experiment`, and metrics | `test_embeddings.py`, `test_ridge.py`, `test_selection.py`, `test_experiment.py` | Cache metadata plus fit diagnostic dtypes |
| Labeled-only mean and one scalar RMS feature scale; no intercept | `ridge.fit_feature_transform`; immutable transform stored in fit artifact | `test_ridge.py`, `test_experiment.py` | Feature transform and finite/variation diagnostics |
| Labeled-only target standardization with population SD (`ddof=0`) | `ridge.fit_label_transform`; config literal | `test_ridge.py`, `test_config.py`, `test_experiment.py` | Label transform in fit artifact |
| Affine OLS teacher calibration with intercept and unconstrained sign | `ridge.fit_teacher_calibration` | `test_ridge.py`, `test_experiment.py` | Slope/intercept and teacher diagnostics |
| Supervised no-intercept ridge, `lambda=0.01`, exact normalized normal equations | `ridge.fit_weighted_ridge`; task algebra in `experiment` | `test_ridge.py`, `test_experiment.py`; R3 learnability probe | R2 algebra probe and task normal-equation diagnostics |
| Full score uses `g_L`, `H`, and `rho=1e-4`; efficient vectorization; descending stable-hash ties | `selection.influence_scores` and `stable_top_k`; protocol-checked task assembly in `experiment` | `test_selection.py`, `test_experiment.py`; R3 exact zero-damping finite difference | R2 score statistics; per-task scores and selection hashes |
| Select exactly `q=192`; do not filter to positive scores | Strict protocol count and task-artifact verification | `test_config.py`, `test_selection.py`, `test_experiment.py` | Per-method selected count and positive-score fraction |
| Retrain pseudo methods with `w=0.1` and `D=n+wq`, including `D * lambda` | Weight-normalized solver in `ridge.fit_weighted_ridge`; task validation recomputes normal equations | `test_ridge.py`, `test_experiment.py` | Coefficients, weights, stationarity residuals |
| Three pseudo-label methods differ only in selection; the supervised-only method uses no pseudo-samples; random has an independent stream | `experiment.METHOD_NAMES`, method-specific selection with common transformed inputs and pseudo-labels | `test_experiment.py`, `test_selection.py` | Five method rows per task wave, with no-H separately marked exploratory |
| No-Hessian ablation replaces inverse-H geometry by identity only | `selection.no_hessian_scores`; separately carded method in `experiment` | `test_selection.py`, `test_experiment.py`, R3 explicit-formula equality | Full/no-H selection hashes and overlap |
| Primary metric is assay-macro paired Spearman `ours-random`; MSE/NDCG are secondary | Strict metric implementations in `metrics`; method and contrast constants in `analysis` | `test_metrics.py`, `test_analysis.py` | Task metrics, assay table, aggregate table |
| NDCG@10% uses true-fitness min-max gains and floor 10% cutoff | `metrics.ndcg_at_10_percent` | `test_metrics.py` including pinned ProteinGym parity | Recomputable predictions and metric table |
| Confirmatory inference clusters by assay | Within-assay seed means, 8-assay SE, exact `2^8` sign flip, and deterministic hierarchical bootstrap in `analysis` | `test_analysis.py` | Assay-level effects, exact p-value, bootstrap interval |
| Success is positive macro gain, at least 25/40 task wins, and at least 5/8 assay wins | Hard-coded v0 verdict validation in `analysis`; no caller-overridable methods or thresholds | `test_analysis.py` | Machine-readable R7 verdict |
| Practical self-improvement additionally requires positive `ours-supervised` macro gain | Separate hard-coded verdict field in `analysis` | `test_analysis.py` | R7 decision memo and verdict JSON |
| Required fit diagnostics: score distribution, overlap, Hessian spectrum/condition, data rank, effective degrees of freedom, stationarity | Fit-time diagnostics and artifact revalidation in `experiment` | `test_experiment.py` | Fit task shards |
| Required hidden diagnostics: teacher test Spearman, pseudo-label error, oracle test-gradient alignment | Post-fit-only `evaluate_task`; oracle retains the frozen teacher pseudo-gradient | `test_experiment.py` | Evaluation task shards, explicitly diagnostic |
| Random sensitivity uses 100 purpose-separated exploratory draws and cannot replace baseline | `random_diagnostic_replicates=100` config/card and task diagnostics | `test_config.py`, `test_experiment.py` | Exploratory random-diagnostic block |
| BLAS calls serialize at one thread and runtime identity is recorded | Locked `threadpoolctl:blas_threads=1` scope and `NumericalRuntimeFingerprint` in `experiment`; launchers pin active `OPENBLAS_CORETYPE=Haswell` | `test_experiment.py`, `test_slurm_contract.py` | R1 fingerprint and every official fit artifact |
| Stages are restart-safe and fail closed | Atomic JSON/cache writes, content/digest/schema validation, command-level temporary siblings, `afterok` Slurm stages | `test_provenance.py`, `test_embeddings.py`, `test_cli.py`, `test_slurm_contract.py` | R1-R6 completion manifests and launcher logs |
| Public repository excludes data, weights, embeddings, task shards, logs, local profiles, and credentials | `.gitignore`, publication scan, tracked-file allowlist review | Publication gate at R7 | Public compact tables/docs only |

## R1-R3 artifact contract

`scripts/verify_r1_r3.sh` is an offline, CPU-only, fail-fast gate. It discovers
the repository rather than assuming a checkout path, refuses to start from a
dirty worktree, and requires the same clean HEAD again after all commands. It
stages every output in a temporary sibling of the configured
`SELF_IMPROVE_VERIFICATION_ROOT`. After schema and hash validation, it replaces
the exact R1-R3 directories and atomically publishes `completion.json` last.
Without that final marker, existing files are not a passed verification run.

- `r1/fresh-environment-resolution.json` records an offline `uv sync --dry-run`
  against a nonexistent environment with both `dev` and `embed` extras. It
  proves the lock declares pinned `torch==2.10.0` and
  `transformers==4.57.6`, independent of packages already in `.venv`.
- `r1/report.json` records UTC timestamps, exact command strings and exit codes,
  bounded output tails and complete output hashes, the stable clean git HEAD,
  exact torch/transformers versions, the validated config dump, numerical
  fingerprint, and SHA-256 values for `pyproject.toml`, `uv.lock`, config,
  verification script, Python, uv, pytest, Ruff, and mypy.
- `r2/pytest.txt` captures both targeted algebra/data/
  leakage/pooling/metrics/task tests and the complete test suite.
- `r2/algebra_probe.json` records dimensions, dtypes,
  finite checks, normal-equation/gradient residuals, and score statistics.
- `r3/synthetic_probe.json` records a seeded noiseless
  learnability problem, exact one-candidate perturbation results at
  `epsilon=1e-6`, full-H/no-H scores, the literal self-teacher negative control,
  selected hashes, and two matching deterministic run digests.
- `completion.json` binds the configured artifact root, git/trust root, and
  SHA-256 of every R1-R3 file. It is the sole authoritative success marker.

The R3 causal check intentionally uses zero score damping so that the score is
the exact derivative of the explicitly perturbed objective. Confirmatory v0
uses the locked damping `rho=1e-4`; unit tests verify that implementation, but
R3 does not claim that a damped score is the exact undamped derivative.

### TDD record

- **RED (2026-06-29):** `uv run --extra dev pytest
  tests/test_synthetic_probe.py -q` stopped during collection with
  `ModuleNotFoundError: No module named 'self_improve_protein.probes'` after the
  probe contract tests were added and before the helper existed.
- **GREEN (2026-06-29):** after the minimal probe/helper and artifact validator
  were implemented, the focused probe suite passed. The final authoritative
  counts, static checks, commands, and hashes are the locally generated R1-R3
  artifacts, not this narrative entry.
- **Hardening RED/GREEN (2026-06-29):** review tests first failed collection on
  the missing bundle publication API, then passed after clean-HEAD binding,
  fresh-environment proof, custom-root paths, exact output hashes, and
  interruption/stale-artifact fail-closed publication were implemented.

## Known scientific and numerical caveats

- The effective pseudo fraction is
  `t = (0.1 * 192) / (96 + 0.1 * 192) = 1/6`. This is a moderate finite
  perturbation, not the theorem's `t -> 0` regime. First-order ranking can fail
  after selecting and jointly retraining on 192 points even when the
  infinitesimal single-point calculation is correct.
- Labeled-only feature centering with `n=96` gives
  `rank(X_L) <= n - 1 = 95`, while ESM-2 has 480 features. The unregularized
  data Gram matrix is necessarily singular. Ridge regularization supplies
  invertibility; data rank and regularized conditioning must be reported as
  different diagnostics.
- The labeled training gradient is a proxy for test risk. A positive score
  predicts lower local labeled loss, not better held-out Spearman. The
  analysis-only test-gradient oracle measures this alignment after selection
  is frozen and cannot enter the method.
- Teacher calibration uses only 96 labels. Weak or anti-correlated teacher
  behavior can prevent pseudo-labeling from helping even if ranking by the
  score is internally correct. Teacher quality is diagnostic, not a criterion
  for dropping an assay.
- The 40 assay-seed rows are not 40 independent biological replicates. Seeds
  share an assay, sequence context, and source study. Confirmatory uncertainty
  therefore aggregates the five seeds within assay and performs inference over
  eight assay clusters. An iid 40-row paired test is non-confirmatory only.
- Exact floating hashes are comparable only under the same recorded numerical
  fingerprint. Across fingerprints, the protocol requires identical discrete
  selections plus tight numerical equivalence while retaining distinct exact
  hashes.
- The ninth eligible assay is development-only. Its results can expose a bug or
  unusable configuration but cannot enter the confirmatory table or alter the
  locked decision rule.

## Promotion interpretation

R1-R3 prove only static integrity and controlled algebra. R4 proves the real
launcher/cache path. R5 proves end-to-end plumbing on the development assay.
R6 supplies the first confirmatory outcomes. R7 must report pass, fail, or
ambiguous without rewriting v0, and must separately label confirmatory,
secondary, diagnostic, and exploratory observations.
