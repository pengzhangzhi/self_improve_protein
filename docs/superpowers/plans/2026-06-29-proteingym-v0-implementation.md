# ProteinGym v0 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers-ml:subagent-driven-development (recommended) or superpowers-ml:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build, verify, run, analyze, and publicly release the locked external-teacher ProteinGym pseudo-sample selection study.

**Architecture:** A small typed Python package separates immutable protocol, data manifests, frozen embeddings, closed-form ridge/selection math, hidden-label-safe task execution, and clustered analysis. Each batch stage writes validated atomic artifacts, so Slurm arrays are idempotent and independently retryable. The confirmatory path is frozen; negative controls and later repairs live behind separate commands/cards.

**Tech Stack:** Python 3.12, uv, NumPy/SciPy/pandas/PyArrow, PyTorch/Transformers for ESM-2, Pydantic/PyYAML/Typer, pytest/Hypothesis, Ruff, mypy, Slurm, Git/GitHub.

---

## File map

- `pyproject.toml`: package metadata, dependencies, console entry point, lint/test settings.
- `configs/v0.yaml`: literal locked protocol values.
- `src/self_improve_protein/config.py`: immutable validated configuration.
- `src/self_improve_protein/provenance.py`: hashes, seed derivation, atomic JSON/Parquet writes.
- `src/self_improve_protein/data.py`: joins, usable-row rules, assay selection, working sets, and splits.
- `src/self_improve_protein/embeddings.py`: residue-only ESM-2 pooling and assay caches.
- `src/self_improve_protein/ridge.py`: preprocessing, calibration, exact normalized ridge solves.
- `src/self_improve_protein/selection.py`: full score, no-Hessian, random, teacher-top, self-teacher control.
- `src/self_improve_protein/metrics.py`: Spearman, MSE, ProteinGym NDCG@10%.
- `src/self_improve_protein/experiment.py`: hidden-label-safe fit/evaluate task boundary.
- `src/self_improve_protein/analysis.py`: task aggregation, exact sign flips, hierarchical bootstrap, tables.
- `src/self_improve_protein/cli.py`: prepare, embed, run-task, aggregate, and verify commands.
- `slurm/*.sbatch`: generic restart-safe launch stages.
- `tests/`: one focused test module per package responsibility.

### Task 1: Package, immutable protocol, and provenance primitives

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `LICENSE`
- Create: `README.md`
- Create: `configs/v0.yaml`
- Create: `src/self_improve_protein/__init__.py`
- Create: `src/self_improve_protein/config.py`
- Create: `src/self_improve_protein/provenance.py`
- Test: `tests/test_config.py`
- Test: `tests/test_provenance.py`

- [ ] **Step 1: Write failing protocol and deterministic-seed tests**

```python
def test_v0_protocol_is_locked():
    p = load_protocol(Path("configs/v0.yaml"))
    assert (p.n_labeled, p.n_unlabeled, p.n_test, p.q) == (96, 2000, 1000, 192)
    assert (p.pseudo_weight, p.ridge_lambda, p.damping) == (0.1, 0.01, 0.0001)
    assert p.teacher_column == "ESM1v_ensemble"
    assert p.seeds == (0, 1, 2, 3, 4)

def test_seed_derivation_is_purpose_separated():
    split = derive_seed("ADRB2_HUMAN_Jones_2020", 0, "split")
    random = derive_seed("ADRB2_HUMAN_Jones_2020", 0, "random_selection")
    assert split == derive_seed("ADRB2_HUMAN_Jones_2020", 0, "split")
    assert split != random
```

- [ ] **Step 2: Run the tests and verify RED**

Run: `uv run --extra dev pytest tests/test_config.py tests/test_provenance.py -q`

Expected: collection fails because `self_improve_protein.config` and `self_improve_protein.provenance` do not exist.

- [ ] **Step 3: Add the minimal typed package and literal YAML**

Implement a frozen `Protocol` model with validators for positive sizes, `n_labeled + n_unlabeled + n_test <= working_size`, `q <= n_unlabeled`, distinct seeds, and the literal v0 values. Implement `derive_seed` as the first eight bytes of SHA-256 over NUL-separated fields, interpreted unsigned little-endian. Implement atomic writes with `tempfile.NamedTemporaryFile` in the destination directory, `fsync`, schema validation, and `os.replace`.

```python
def derive_seed(dms_id: str, seed: int, purpose: str) -> int:
    payload = f"{dms_id}\0{seed}\0{purpose}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")
```

- [ ] **Step 4: Verify GREEN and static checks**

Run: `uv run --extra dev pytest tests/test_config.py tests/test_provenance.py -q && uv run --extra dev ruff check . && uv run --extra dev mypy src`

Expected: all tests pass; Ruff and mypy exit zero.

- [ ] **Step 5: Commit**

Run: `git add pyproject.toml .gitignore LICENSE README.md configs src/self_improve_protein tests && git commit -m "feat: add locked protocol and provenance core"`

### Task 2: Exact ridge, calibration, and influence selectors

**Files:**
- Create: `src/self_improve_protein/ridge.py`
- Create: `src/self_improve_protein/selection.py`
- Test: `tests/test_ridge.py`
- Test: `tests/test_selection.py`

- [ ] **Step 1: Write failing algebra tests**

Cover: supervised normal equations; `g_L + lambda * theta == 0`; weighted solve uses `(sum weights) * lambda`; vectorized and explicit scores match; finite-difference labeled-loss derivative equals `-score`; self-teacher scores tie at a non-positive value; full-rank OLS scores are zero; no-Hessian vectorization matches explicit gradients; and simultaneous feature scaling with `lambda` and `rho` scaling by the squared factor preserves predictions and score ordering.

```python
def test_self_teacher_is_constant_nonpositive(ridge_case):
    theta = fit_weighted_ridge(ridge_case.x_l, ridge_case.y_l, 0.01)
    yhat_u = ridge_case.x_u @ theta
    scores = influence_scores(ridge_case.x_l, ridge_case.y_l, ridge_case.x_u, yhat_u, theta, 0.01, 1e-4)
    np.testing.assert_allclose(scores, scores[0], atol=1e-11)
    assert scores[0] <= 0.0
```

- [ ] **Step 2: Run and verify RED**

Run: `uv run --extra dev pytest tests/test_ridge.py tests/test_selection.py -q`

Expected: import failure for the missing math modules.

- [ ] **Step 3: Implement minimal float64 closed forms**

Use one scalar labeled-only embedding scale, label `ddof=0`, OLS teacher calibration with `np.linalg.lstsq`, Cholesky solve with a deterministic `np.linalg.solve` fallback, and stable hash tie-breaking. The weighted solver must implement:

```python
denom = float(weights.sum())
gram = x.T @ (weights[:, None] * x) + denom * ridge_lambda * np.eye(x.shape[1])
rhs = x.T @ (weights * y)
theta = np.linalg.solve(gram, rhs)
```

The full score must use one solve and no per-candidate matrix:

```python
g_l = x_l.T @ (x_l @ theta - y_l) / x_l.shape[0]
h = x_l.T @ x_l / x_l.shape[0] + ridge_lambda * np.eye(x_l.shape[1])
v = np.linalg.solve(h + damping * np.eye(x_l.shape[1]), g_l)
residual_u = x_u @ theta - yhat_u
scores = residual_u * (x_u @ v) - g_l @ v
```

- [ ] **Step 4: Verify GREEN, including the sign regression**

Run: `uv run --extra dev pytest tests/test_ridge.py tests/test_selection.py -q`

Expected: every algebra test passes, including the self-teacher negative control and finite-difference sign.

- [ ] **Step 5: Commit**

Run: `git add src/self_improve_protein/ridge.py src/self_improve_protein/selection.py tests/test_ridge.py tests/test_selection.py && git commit -m "feat: implement exact ridge influence selection"`

### Task 3: ProteinGym joins, eligibility, manifests, and splits

**Files:**
- Create: `src/self_improve_protein/data.py`
- Create: `tests/fixtures/dms_tiny.csv`
- Create: `tests/fixtures/scores_tiny.csv`
- Test: `tests/test_data.py`

- [ ] **Step 1: Write failing data-contract tests**

Test one-to-one `mutant` joins, finite teacher/label filtering, duplicate-sequence removal, SHA-256 row ordering, exactly 6,000 rows when eligible, lexicographic assay selection, independent deterministic splits, and rejection of overlap or malformed sequences.

```python
def test_hidden_values_do_not_affect_working_set(tiny_joined):
    first = build_working_set(tiny_joined, size=6)
    changed = tiny_joined.assign(DMS_score=tiny_joined.DMS_score[::-1].to_numpy())
    second = build_working_set(changed, size=6)
    assert first.sequence_hash.tolist() == second.sequence_hash.tolist()
```

- [ ] **Step 2: Run and verify RED**

Run: `uv run --extra dev pytest tests/test_data.py -q`

Expected: import failure for `self_improve_protein.data`.

- [ ] **Step 3: Implement strict joins and immutable manifest records**

Use `pandas.merge(validate="one_to_one")`; reject duplicate mutant keys before merging; compute hashes from `(DMS_id, mutant, mutated_sequence)`; apply teacher/label/sequence filtering before counting eligibility; write selected IDs and row hashes to JSON with source SHA-256 values. `make_split` returns only integer indices and sequence hashes.

- [ ] **Step 4: Verify GREEN and fixture determinism**

Run: `uv run --extra dev pytest tests/test_data.py -q`

Expected: all data tests pass twice with identical hashes.

- [ ] **Step 5: Commit**

Run: `git add src/self_improve_protein/data.py tests && git commit -m "feat: add deterministic ProteinGym manifests"`

### Task 4: Residue-only ESM-2 embedding cache

**Files:**
- Create: `src/self_improve_protein/embeddings.py`
- Test: `tests/test_embeddings.py`

- [ ] **Step 1: Write failing mask, pooling, batching, and cache tests**

```python
def test_mean_pool_excludes_bos_eos_and_padding():
    hidden = torch.tensor([[[100.0], [1.0], [3.0], [200.0], [300.0]]])
    attention = torch.tensor([[1, 1, 1, 1, 0]])
    special = torch.tensor([[1, 0, 0, 1, 1]])
    pooled = mean_pool_residues(hidden, attention, special)
    torch.testing.assert_close(pooled, torch.tensor([[2.0]]))
```

Also require float32 output, preserved manifest row order, atomic cache metadata, sequence/model hash validation, and cache rejection after metadata corruption.

- [ ] **Step 2: Run and verify RED**

Run: `uv run --extra dev --extra embed pytest tests/test_embeddings.py -q`

Expected: import failure for the missing embedding module.

- [ ] **Step 3: Implement lazy model loading and atomic cache writes**

Load tokenizer/model inside the command function, call the tokenizer with `return_special_tokens_mask=True`, pool only `attention_mask & ~special_tokens_mask`, use `torch.inference_mode()`, write float32 `.npy` plus JSON metadata, and validate both before cache reuse. Autotune batch size only before the frozen study; the chosen value is runtime-only and cannot affect embeddings.

- [ ] **Step 4: Verify GREEN on CPU tensors**

Run: `uv run --extra dev --extra embed pytest tests/test_embeddings.py -q`

Expected: pooling and cache tests pass without downloading model weights.

- [ ] **Step 5: Commit**

Run: `git add src/self_improve_protein/embeddings.py tests/test_embeddings.py && git commit -m "feat: add validated ESM embedding caches"`

### Task 5: Hidden-label-safe task execution

**Files:**
- Create: `src/self_improve_protein/experiment.py`
- Test: `tests/test_experiment.py`

- [ ] **Step 1: Write failing end-to-end synthetic task tests**

Define `FitInputs` with labeled features/labels, unlabeled features, labeled/unlabeled teacher scores, test features, and stable hashes—but no unlabeled/test labels. Define `EvaluationLabels` separately. Test all four confirmatory methods plus the separately carded no-Hessian method, equal pseudo count and labels, deterministic selection, selection invariance after hidden-label permutation, and exact weighted normal equations.

- [ ] **Step 2: Run and verify RED**

Run: `uv run --extra dev pytest tests/test_experiment.py -q`

Expected: import failure for `self_improve_protein.experiment`.

- [ ] **Step 3: Implement fit then evaluate as separate pure functions**

`fit_task(FitInputs, Protocol)` returns selected hashes, coefficients, test predictions, protocol/source digests, and non-label diagnostics. Freeze its canonical digest immediately. `evaluate_task(FitArtifact, EvaluationLabels, expected_fit_digest=...)` must verify that external digest before accessing hidden labels, then add metrics and hidden pseudo-error diagnostics without changing the fit artifact. Reject any method artifact whose selected count, pseudo weight, calibration, source hashes, exact weighted normal equations, or diagnostics differ from protocol. The analysis-only **test-risk oracle influence** uses the hidden test outer gradient with the frozen teacher pseudo-gradient; it does not replace the pseudo-label with the hidden unlabeled label.

- [ ] **Step 4: Verify GREEN and leakage invariance**

Run: `uv run --extra dev pytest tests/test_experiment.py -q`

Expected: all methods complete; hidden-label permutations leave fit artifacts byte-identical.

- [ ] **Step 5: Commit**

Run: `git add src/self_improve_protein/experiment.py tests/test_experiment.py && git commit -m "feat: add leakage-safe assay task runner"`

### Task 6: Exact metrics and assay-clustered inference

**Files:**
- Create: `src/self_improve_protein/metrics.py`
- Create: `src/self_improve_protein/analysis.py`
- Test: `tests/test_metrics.py`
- Test: `tests/test_analysis.py`

- [ ] **Step 1: Write failing metric and inference tests**

Copy small expected values from the pinned ProteinGym v1.3 `calc_ndcg` behavior; test Spearman constant handling and MSE argument order. For analysis, use an eight-assay toy table to test within-assay averaging, `sd / sqrt(8)`, exact enumeration of 256 sign flips, deterministic hierarchical bootstrap, win counts, distinct selection/practical-self-improvement verdicts, and failure on missing/duplicate tasks.

- [ ] **Step 2: Run and verify RED**

Run: `uv run --extra dev pytest tests/test_metrics.py tests/test_analysis.py -q`

Expected: import failures for both modules.

- [ ] **Step 3: Implement exact metrics and paired analysis**

Use stable descending ranks for NDCG, min-max true gains, `k=floor(0.10*n)`, and zero only when ideal DCG is zero. Exact sign flips enumerate integer masks `0..255`; the two-sided p-value is the fraction of absolute permuted means at least the observed absolute mean. Bootstrap resamples assay IDs first and seed rows second with a fixed analysis seed.

- [ ] **Step 4: Verify GREEN**

Run: `uv run --extra dev pytest tests/test_metrics.py tests/test_analysis.py -q`

Expected: all reference values and aggregation invariants pass.

- [ ] **Step 5: Commit**

Run: `git add src/self_improve_protein/metrics.py src/self_improve_protein/analysis.py tests && git commit -m "feat: add ProteinGym metrics and clustered inference"`

### Task 7: CLI, validated artifacts, and generic Slurm stages

**Files:**
- Create: `src/self_improve_protein/cli.py`
- Create: `slurm/prepare.sbatch`
- Create: `slurm/embed_array.sbatch`
- Create: `slurm/task_array.sbatch`
- Create: `slurm/aggregate.sbatch`
- Create: `slurm/submit_pipeline.sh`
- Test: `tests/test_cli.py`
- Test: `tests/test_slurm_contract.py`

- [ ] **Step 1: Write failing CLI and shell-contract tests**

Test `--help`, config rendering, dry-run stage plans, refusal to overwrite mismatched artifacts, exact array ranges from manifests, `set -euo pipefail`, `--requeue`, reaper comment, and placement of `SLURM_CONF` after `#SBATCH` directives. Cluster account, partitions, repository, and data root must come from environment variables rather than tracked literals.

- [ ] **Step 2: Run and verify RED**

Run: `uv run --extra dev pytest tests/test_cli.py tests/test_slurm_contract.py -q`

Expected: missing CLI and Slurm files.

- [ ] **Step 3: Implement five idempotent commands and four stages**

Commands: `prepare-data`, `embed-assay`, `run-task`, `aggregate`, and `verify`. Every command prints a normalized JSON start record, validates inputs, writes to a temporary sibling, validates output, atomically renames, and prints a terminal JSON record. The submit script submits prepare, embedding array, task array, and aggregate with `afterok` dependencies and records job IDs in an ignored local run directory.

- [ ] **Step 4: Verify GREEN and shell syntax**

Run: `uv run --extra dev pytest tests/test_cli.py tests/test_slurm_contract.py -q && for f in slurm/*.sbatch slurm/*.sh; do bash -n "$f"; done`

Expected: tests and shell syntax checks exit zero.

- [ ] **Step 5: Commit**

Run: `git add src/self_improve_protein/cli.py slurm tests && git commit -m "feat: add restart-safe experiment launchers"`

### Task 8: R1-R3 verification artifacts and independent review

**Files:**
- Create: `tests/test_synthetic_probe.py`
- Create: `scripts/verify_r1_r3.sh`
- Create: `docs/research/protocol-audit.md`

- [ ] **Step 1: Add the failing synthetic learnability/causal-score test**

Construct a seeded noiseless linear problem with enough dimensions to interpolate, fit at `lambda=1e-10`, and assert tiny train MSE. Construct an external teacher where finite-difference labeled losses can be ordered and assert score/order agreement at epsilon `1e-6`.

- [ ] **Step 2: Verify RED, then implement only the reusable probe helper needed by the test**

Run: `uv run --extra dev pytest tests/test_synthetic_probe.py -q`

Expected RED: missing probe helper. After the helper is added, expected GREEN: both numerical assertions pass.

- [ ] **Step 3: Run fresh R1-R3 commands and write machine-readable artifacts**

Run: `bash scripts/verify_r1_r3.sh`

Expected: `artifacts/verification/r1/report.json`, `r2/pytest.txt`, and `r3/synthetic_probe.json` exist, validate, and record zero exit status. These local artifacts remain ignored by git.

- [ ] **Step 4: Request an independent code/science review and resolve only verified findings**

Review the design-to-code mapping, normalizations, sign, leakage boundary, metrics, and missing tests. Reproduce every actionable finding with a failing test before changing production code.

- [ ] **Step 5: Run full verification and commit**

Run: `uv run --extra dev --extra embed pytest -q && uv run --extra dev ruff check . && uv run --extra dev mypy src && git diff --check`

Expected: zero failed tests and zero static errors. Then run `git add tests/test_synthetic_probe.py scripts/verify_r1_r3.sh docs/research/protocol-audit.md src && git commit -m "test: verify local research pipeline"`.

### Task 9: Official data, R4 Slurm smoke, and R5 development pilot

**Files:**
- Create locally, ignored: `local/cluster.env`
- Create via commands, ignored: `data/`, `artifacts/verification/r4/`, `artifacts/verification/r5/`
- Modify only if a test proves a bug: package or launcher files from Tasks 1-7

- [ ] **Step 1: Download official v1.3 archives and freeze checksums**

Run `prepare-data` against the pinned official v1.3 substitution, zero-shot, and metadata URLs from Zenodo record `15293562`. Expected: one immutable manifest with one-to-one `ESM1v_ensemble` coverage, eligible assay list, the eight confirmatory IDs, ninth development ID, 6,000 row hashes per chosen assay, and source SHA-256 values.

- [ ] **Step 2: Inspect the manifest without evaluating outcomes**

Run the manifest verifier and confirm only protocol/coverage/hash fields. Do not compute method metrics or choose assays/teacher from DMS performance.

- [ ] **Step 3: Submit and monitor R4**

Submit one A100 smoke job for a small development-assay shard through `slurm/embed_array.sbatch`, then a reduced task through the real task launcher. Expected R4 artifacts: terminal-success scheduler state, finite float32 embeddings, valid checksums, one reduced result, and successful cache reuse on rerun.

- [ ] **Step 4: Submit and monitor R5**

Embed the full ninth eligible assay and run two full development seeds. Expected: 10 method rows total (five methods by two development seeds), finite teacher/score/prediction metrics, deterministic rerun hashes, and a written pilot note that does not enter v0 tables.

- [ ] **Step 5: Classify failures at their lowest rung**

For any failure, preserve logs, write a reproducing test, classify it as code/config/data/numerical/resource, fix only the root cause, and rerun from the required rung. A method-direction change is a new experiment card, not a bugfix.

### Task 10: R6 confirmatory run, R7 decision, and public release

**Files:**
- Create via commands, ignored: `artifacts/studies/v0/`
- Create: `results/v0/summary.csv`
- Create: `results/v0/effects.csv`
- Create: `results/v0/assay_diagnostics.csv`
- Create: `docs/results/v0-decision.md`
- Modify: `README.md`

- [ ] **Step 1: Freeze the R6 run manifest and git revision**

Require a clean tracked worktree and record the exact commit, config digest, data checksums, model revision, eight assay IDs, five seeds, 40 task IDs, and expected artifact schemas. Reject mixed revisions.

- [ ] **Step 2: Launch maximum-parallel idempotent arrays**

Run one embedding task per assay and one CPU task per assay-seed. Monitor terminal states and artifact validation, not merely queue disappearance. Retry only missing/invalid shards with the same manifest.

- [ ] **Step 3: Aggregate only after all 40 tasks validate**

Generate method means, paired effects, task and assay win rates, assay-clustered SE, exact sign-flip p-value, hierarchical bootstrap interval, diagnostics, and the locked success-rule boolean. Expected: no silent row drops or duplicate task-method keys.

- [ ] **Step 4: Apply the ML result-review skill and write R7**

State the highest verified rung, primary verdict, exact effect/uncertainty/wins, secondary results, diagnostics, limitations, and the next experiment card. Do not call exploratory outcomes confirmatory.

- [ ] **Step 5: Sanitize, verify, commit, and publish**

Run the full suite, static checks, build, tracked-file size scan, secret scan, private-path scan, and `git diff --check`. Commit compact results and documentation. Create `pengzhangzhi/self_improve_protein` explicitly as a public repository, add the SSH remote, push `main`, and verify the public URL and remote commit match local HEAD.
