# Verification Ladder: External-Teacher Score-Ranking v0

## R0: Protocol lock

**Check:** The design and experiment card fix one primary endpoint, one baseline contrast, data release, teacher, split algorithm, seed policy, and decision rule.

**Artifact:** This file, `experiment-card-v0.md`, and the approved design in one git commit.

**Promote when:** No placeholders or contradictory protocol values remain.

## R1: Static package and configuration

**Check:** The package imports, CLI help renders, the v0 config parses into validated immutable values, deterministic seed derivation is stable, and lint/type checks are clean.

**Artifact:** `artifacts/verification/r1/report.json` with command lines, exit codes, config dump, Python/package versions, and git revision.

**Promote when:** All commands exit zero and the dump exactly matches the experiment card.

## R2: Algebra, shapes, leakage, and one-pass model checks

**Check:** Unit/property tests cover exact ridge normal equations, weighted `D * lambda`, score vectorization, finite-difference sign, self-teacher degeneracy, no-Hessian formula, deterministic disjoint splits, hidden-label invariance, residue-only pooling, finite embeddings, and metric parity.

**Artifact:** `artifacts/verification/r2/pytest.txt` plus a JSON algebra probe containing dimensions, dtypes, finite checks, gradient identity residual, and score statistics.

**Promote when:** The full R2 test set passes with no non-finite values and all stated tolerances hold.

## R3: Tiny synthetic learnability and causal score probe

**Check:** Near-unregularized ridge recovers a tiny noiseless linear problem; a controlled external-teacher construction yields the finite-difference loss ordering predicted by the score; repeated execution is deterministic.

**Artifact:** `artifacts/verification/r3/synthetic_probe.json` with train MSE, parameter error, predicted first-order changes, realized epsilon changes, and selection hashes.

**Promote when:** Tiny train MSE is below the declared numerical tolerance, predicted and realized first-order changes agree in sign and tolerance, and hashes repeat exactly.

## R4: Real Slurm launcher smoke

**Check:** The actual Slurm path loads the frozen ESM model on one A100, embeds a small real sequence shard, atomically writes and reloads its cache, runs one reduced assay-seed task, emits metrics/diagnostics, and exits zero.

**Artifact:** `artifacts/verification/r4/` containing scheduler metadata, stdout/stderr, embedding metadata/checksum, one task result, and launcher exit report.

**Promote when:** The scheduler job is terminal-success, all schemas/checksums validate, and rerunning reuses rather than corrupts valid artifacts.

## R5: Development-only pilot

**Check:** On the ninth eligible assay and two fixed development seeds, all four confirmatory methods plus the separately carded no-Hessian method complete at full task sizes; teacher and predictions vary; metrics and score diagnostics are finite; independent reruns match.

**Artifact:** `artifacts/verification/r5/` with the frozen development manifest, two task shards, aggregate table, and a pilot note.

**Promote when:** There is no data, leakage, numerical, or launcher failure and the teacher has nonzero coverage/variance. Pilot direction is recorded but cannot count as confirmatory evidence.

## R6: Full confirmatory study

**Check:** All 40 predeclared assay-seed tasks complete for the four confirmatory methods and the separately carded no-Hessian method under one immutable manifest and git revision.

**Artifact:** `artifacts/studies/v0/` with 40 validated task shards, predictions, diagnostics, aggregate tables, exact sign-flip result, hierarchical bootstrap, and completion manifest.

**Promote when:** Every planned task is present exactly once, selection hashes and protocol values match the frozen manifest, and no failed task was silently dropped.

## R7: Result review and decision

**Check:** Compare the locked primary endpoint and success rule, then separate confirmatory, secondary, diagnostic, and exploratory observations.

**Artifact:** `docs/results/v0-decision.md` plus machine-readable summary tables.

**Stop or branch:** Conclude pass, fail, or ambiguous for external-teacher v0. A positive or ambiguous result may advance to untouched-assay replication; a negative result may start separate exploratory cards without rewriting v0.
