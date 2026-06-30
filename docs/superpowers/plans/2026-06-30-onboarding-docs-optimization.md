# Onboarding Documentation Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `subagent-driven-development` (recommended) or `executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the public documentation into a concise scientific front door, a theory-friendly conceptual guide, and an exact maintainer operations runbook.

**Architecture:** Keep `README.md` focused on the answer and navigation, keep `docs/GETTING_STARTED.md` focused on biology/theory/code orientation, and move reproducibility operations into a new `docs/OPERATIONS.md`. Validate prose against committed results and commands against the actual CLI and Slurm scripts, then independently review the integrated reading path before publishing.

**Tech Stack:** GitHub-flavored Markdown, Typer CLI, Bash/Slurm, `uv`, pytest, Ruff, mypy, Git, and GitHub.

---

## File map

- Create `docs/OPERATIONS.md`: the only command-complete maintainer runbook.
- Modify `docs/GETTING_STARTED.md`: conceptual onboarding for the theory collaborator.
- Modify `README.md`: public conclusion, concise project map, and reader routing.
- Read but do not modify `docs/results/overall-conclusion.md`, `configs/v0.yaml`,
  `slurm/submit_pipeline.sh`, `slurm/*.sbatch`, and
  `docs/research/feedback-ladder.md`: authoritative evidence for the rewrite.

### Task 1: Build the maintainer operations runbook

**Files:**

- Create: `docs/OPERATIONS.md`
- Reference: `scripts/verify_r1_r3.sh`
- Reference: `slurm/submit_pipeline.sh`
- Reference: `slurm/prepare.sbatch`
- Reference: `slurm/embed_array.sbatch`
- Reference: `slurm/task_array.sbatch`
- Reference: `slurm/aggregate.sbatch`

- [ ] **Step 1: Add the runbook purpose and safety boundary**

Start the document with these facts in plain language:

```markdown
# Operations

This is the maintainer runbook for reproducing and extending the experiment.
If you want the scientific explanation first, read [Getting started](GETTING_STARTED.md).

The local commands below inspect or verify the repository. The cluster submission
section creates scheduled jobs and should be used only after site values and storage
paths have been reviewed. No public default contains a real account, partition, or
private filesystem path.
```

- [ ] **Step 2: Document local setup and CLI discovery**

Include the CPU setup, the large embedding/developer sync, and all five stage help
commands exactly as executable commands:

```bash
uv sync --frozen
uv run self-improve-protein --show-config
uv run self-improve-protein --help

uv sync --frozen --extra dev --extra embed
uv run self-improve-protein prepare-data --help
uv run self-improve-protein embed-assay --help
uv run self-improve-protein run-task --help
uv run self-improve-protein aggregate --help
uv run self-improve-protein verify --help
```

State that the first embedding sync installs Torch and Transformers, may be large,
and does not submit jobs. State that successful `--help` commands print usage and
exit zero.

- [ ] **Step 3: Document the local R1--R3 verification contract**

Include:

```bash
bash scripts/verify_r1_r3.sh
```

Explain that the script requires a clean worktree and an already cached offline
environment, runs import/config/lint/type/test/algebra/synthetic checks, and publishes
receipts under `artifacts/verification/r1/`, `r2/`, and `r3/`. Explicitly say this
establishes code-path correctness and synthetic learnability, not protein-task method
quality.

- [ ] **Step 4: Add the data and stage map**

Use a compact table covering `prepare-data`, `embed-assay`, `run-task`, `aggregate`,
and `verify`. Name the required upstream inputs: the pinned ProteinGym substitution
tables, `ESM1v_ensemble` zero-shot scores, the frozen ESM-2 checkpoint, the locked
configuration, writable processed/embedding/results roots, and the immutable data
manifest produced by preparation.

- [ ] **Step 5: Add exact launcher configuration**

List the seven variables required by `slurm/submit_pipeline.sh`:

```bash
export SI_ACCOUNT="your-cluster-account"
export SI_CPU_PARTITION="your-cpu-partition"
export SI_GPU_PARTITION="your-gpu-partition"
export SI_REPO_ROOT="$(pwd)"
export SI_DATA_ROOT="/path/to/project-data"
export SI_ARTIFACT_ROOT="/path/to/project-artifacts"
export SI_SLURM_CONF="/path/to/slurm.conf"
```

Explain the default-derived variables (`SI_CONFIG`, `SI_MANIFEST`,
`SI_PROCESSED_ROOT`, `SI_EMBEDDING_ROOT`, `SI_RESULTS_ROOT`, and `SI_MODE`) and say
that values are examples the operator must replace.

- [ ] **Step 6: Document development and confirmatory submission accurately**

Label both command blocks `SUBMITS CLUSTER JOBS`. For development mode, document:

```bash
export SI_MODE=development
export SI_RUN_ID="dev-$(date -u +%Y%m%dT%H%M%SZ)"
bash slurm/submit_pipeline.sh
```

State the exact footprint: one CPU preparation job, a nine-element one-GPU embedding
array, a two-element CPU task array, and one CPU aggregation job. Clarify that this
is the development pilot and is not a dedicated reduced R4 launcher.

For confirmation, document that `SI_MODE=confirmatory` requires the pre-authorized
`SI_R5_GATE` receipt and creates a 40-element task array. Do not provide a fake gate
path or imply that the 26 sealed assays may now be opened.

- [ ] **Step 7: Add monitoring, logs, cancellation, and success criteria**

Explain that submission prints and records
`local/slurm/<run-id>/job_ids.json`. Show how to read its four IDs, then use:

```bash
squeue --jobs PREPARE_ID,EMBED_ID,TASK_ID,AGGREGATE_ID
sacct -j PREPARE_ID,EMBED_ID,TASK_ID,AGGREGATE_ID \
  --format=JobID,JobName%40,State,ExitCode,Elapsed
```

Document log patterns in the submission directory:

```text
slurm-sip-prepare-<job-id>.out/.err
slurm-sip-embed-<array-job-id>_<index>.out/.err
slurm-sip-task-<array-job-id>_<index>.out/.err
slurm-sip-aggregate-<job-id>.out/.err
```

Label the following `CANCELS CLUSTER JOBS`:

```bash
scancel PREPARE_ID EMBED_ID TASK_ID AGGREGATE_ID
```

Define success as every job and array element reaching `COMPLETED` with exit code
`0:0`, plus a valid `${SI_RESULTS_ROOT}/development/aggregate.json` or
`${SI_RESULTS_ROOT}/confirmatory/aggregate.json` verified by the CLI.

- [ ] **Step 8: Add the safe extension workflow**

Give a concrete sequence: write a new experiment card; identify the one changed
factor; add focused tests; run R1--R3; arrange an explicit real-data R4 smoke with a
maintainer because there is no public one-command reduced launcher; run the
development pilot; lock artifacts and gate; run confirmation only if authorized;
write a separate decision memo. Link `research/feedback-ladder.md` and the existing
experiment cards.

- [ ] **Step 9: Validate and commit Task 1**

Run:

```bash
rg -n "PREPARE_ID|EMBED_ID|TASK_ID|AGGREGATE_ID|COMPLETED|0:0|nine-element|40-element" docs/OPERATIONS.md
git diff --check
```

Expected: every operational contract appears, and `git diff --check` prints nothing.
Then commit:

```bash
git add docs/OPERATIONS.md
git commit -m "docs: add experiment operations runbook"
```

### Task 2: Refocus the getting-started guide on concepts

**Files:**

- Modify: `docs/GETTING_STARTED.md`
- Reference: `docs/OPERATIONS.md`
- Reference: `docs/results/overall-conclusion.md`
- Reference: `docs/research/theory-audit.md`

- [ ] **Step 1: Update reader routes**

Keep the 10-minute science and 30-minute code routes. Replace the command-heavy
hands-on route with links to local orientation in this guide and the complete
maintainer runbook in `OPERATIONS.md`.

- [ ] **Step 2: Preserve the biology and experiment explanation**

Retain the definitions of sequence, substitution variant, assay, DMS, fitness,
teacher, pseudo-label, student, and embedding. Keep the 96/2,000/1,000/192/0.1 split
table and the four-stage pipeline. Remove repetition that does not help a first-time
reader.

- [ ] **Step 3: Make effective regularization concrete**

Immediately after the weighted training explanation, add a short paragraph with this
meaning: the ridge penalty stays fixed while weighted pseudo-examples change the
scale and composition of the data-fitting term. Therefore random pseudo-labeling
versus supervised-only changes both information and the effective strength of ridge;
full influence versus random is the clean selection comparison.

- [ ] **Step 4: Tighten the theory-to-code bridge**

Keep the notation table for `X`, `Y`, `D^L`, `D^U`, `f_theta`, `H`, `g_L`, `g_j`,
and `S_j`. Retain the self-teacher degeneracy derivation and the boundary that v0 is
an external-teacher heuristic test, not a direct test of the manuscript's literal
self-teaching setup.

- [ ] **Step 5: Preserve the complete negative-result interpretation**

Keep the v0 timeline and the exact values: full minus random Spearman `-0.05526`,
`0/8` assay wins, exact-CV Spearman `0.30336` versus `0.34971`, exact-CV MSE
`1.6177` versus `1.1857`, exploratory no-Hessian contrast `+0.00654` with
`p=0.6172` and worse MSE, and 26 sealed outcomes. State that no selector established
superiority.

- [ ] **Step 6: Keep code orientation, remove cluster operations**

Retain the symbol-level code tour, repository map, local CPU setup, and troubleshooting
for missing `uv`, optional dependencies, missing data, and CLI discovery. Replace the
full Slurm submission/monitoring/cancellation section with a short maintainer boundary
and a link to `OPERATIONS.md`.

- [ ] **Step 7: Tighten the closing reference path**

End with separate next actions for a theory reader and a code maintainer, followed by
the glossary and ordered sources of truth. Ensure `docs/results/overall-conclusion.md`
remains authoritative.

- [ ] **Step 8: Validate and commit Task 2**

Run:

```bash
test "$(wc -w < docs/GETTING_STARTED.md)" -ge 2200
test "$(wc -w < docs/GETTING_STARTED.md)" -le 2600
rg -n "OPERATIONS.md|-0.05526|0/8|1.6177|0.30336|0.6172|26" docs/GETTING_STARTED.md
git diff --check
```

Expected: both word-count tests exit zero, all scientific anchors appear, and the
diff check is silent. Then commit:

```bash
git add docs/GETTING_STARTED.md
git commit -m "docs: focus onboarding on science and code"
```

### Task 3: Tighten the README and navigation

**Files:**

- Modify: `README.md`
- Reference: `docs/GETTING_STARTED.md`
- Reference: `docs/OPERATIONS.md`

- [ ] **Step 1: Add a two-route navigation block**

Near the opening, route “understand the study” to `docs/GETTING_STARTED.md` and
“reproduce or extend it” to `docs/OPERATIONS.md`. Do not duplicate the runbook's
environment variables or Slurm commands.

- [ ] **Step 2: Preserve the scientific answer and caveats**

Keep the five-method table, full-versus-random effect, external-teacher claim
boundary, same-student degeneracy, effective-regularization caveat, exact-CV result,
and link to the overall conclusion. Keep the wording setup-specific and avoid an
impossibility claim.

- [ ] **Step 3: Make the first-hour path executable**

Keep only safe local orientation commands in the README. Point full verification and
cluster operation to `docs/OPERATIONS.md`. State that the first embedding environment
sync is large and that tests verify code paths rather than scientific performance.

- [ ] **Step 4: Validate and commit Task 3**

Run:

```bash
test "$(wc -w < README.md)" -ge 650
test "$(wc -w < README.md)" -le 850
rg -n "GETTING_STARTED.md|OPERATIONS.md|-0.05526|effective regularization" README.md
git diff --check
```

Expected: both word-count tests exit zero, both reader routes and scientific anchors
appear, and the diff check is silent. Then commit:

```bash
git add README.md
git commit -m "docs: sharpen project entry point"
```

### Task 4: Integrated review, verification, and publication

**Files:**

- Verify: `README.md`
- Verify: `docs/GETTING_STARTED.md`
- Verify: `docs/OPERATIONS.md`
- Verify: all repository-relative links in those files

- [ ] **Step 1: Run an automated Markdown link and private-path check**

Run this read-only checker, which extracts non-URL Markdown targets, strips anchors,
resolves each target relative to its document, and fails on missing paths:

```bash
uv run python - <<'PY'
import re
from pathlib import Path

documents = (Path("README.md"), Path("docs/GETTING_STARTED.md"), Path("docs/OPERATIONS.md"))
missing = []
for document in documents:
    text = document.read_text(encoding="utf-8")
    for raw in re.findall(r"\[[^]]+\]\(([^)]+)\)", text):
        target = raw.strip().strip("<>").split("#", 1)[0]
        if not target or "://" in target or target.startswith("mailto:"):
            continue
        resolved = (document.parent / target).resolve()
        if not resolved.exists():
            missing.append(f"{document}: {raw}")
if missing:
    raise SystemExit("missing local links:\n" + "\n".join(missing))
print("all local Markdown targets exist")
PY
```

Also run:

```bash
if rg -n "/lustre/|/home/fredp" README.md docs/GETTING_STARTED.md docs/OPERATIONS.md; then
  exit 1
fi
```

Expected: the link checker reports all local targets present; the private-path scan
prints nothing. Manually confirm that cluster account and partition values remain
clearly marked examples rather than real site values.

- [ ] **Step 2: Verify every documented CLI help command**

Run:

```bash
uv run self-improve-protein prepare-data --help
uv run self-improve-protein embed-assay --help
uv run self-improve-protein run-task --help
uv run self-improve-protein aggregate --help
uv run self-improve-protein verify --help
```

Expected: all five commands print usage and exit zero.

- [ ] **Step 3: Cross-check claims against sources**

Compare the five README means with `results/v0-method-means.csv`; compare branch
effects and exact-CV values with `results/branch-effects.csv` and
`docs/results/overall-conclusion.md`; compare all job counts, variable names, log
patterns, and result paths with `slurm/submit_pipeline.sh` and `slurm/*.sbatch`.

- [ ] **Step 4: Run repository verification**

Run:

```bash
uv run pytest -q
uv run ruff check .
uv run mypy src
git diff --check
```

Expected: the full test suite passes, Ruff and mypy report success, and the diff
check is silent. These checks establish repository integrity, not method quality.

- [ ] **Step 5: Obtain independent specification and fresh-reader reviews**

Ask one reviewer to check every requirement in
`docs/superpowers/specs/2026-06-30-onboarding-docs-optimization-design.md` against the
three documents. Ask a separate reviewer, without implementation context, to follow
the README as a theory collaborator and the operations guide as a maintainer. Fix all
high- and medium-severity findings, then rerun Steps 1--4.

- [ ] **Step 6: Integrate and publish**

Merge the reviewed documentation branch into `main`, rerun the full checks on
`main`, and push:

```bash
git push origin main
```

Expected: `git status --short --branch` reports `main...origin/main` with no changes,
and `git rev-parse HEAD` equals `git ls-remote origin refs/heads/main`.

- [ ] **Step 7: Verify the public rendering inputs**

Fetch the raw GitHub URLs for `README.md`, `docs/GETTING_STARTED.md`, and
`docs/OPERATIONS.md` at the pushed commit and compare their SHA-256 hashes with local
files. Expected: all three hashes match exactly.
