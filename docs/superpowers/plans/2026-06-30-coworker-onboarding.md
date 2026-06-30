# Coworker Onboarding Documentation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers-ml:subagent-driven-development (recommended) or
> superpowers-ml:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give a theory-first collaborator with no protein-ML or coding
background a friendly, accurate path from the research question to reading,
running, and extending this repository.

**Architecture:** Use two layers. `README.md` is the approachable landing page
and first-hour checklist; `docs/GETTING_STARTED.md` is the complete conceptual
and operational handoff. Existing experiment cards, audits, and result memos
remain authoritative and are linked rather than rewritten as competing sources.

**Tech Stack:** GitHub-flavored Markdown, Python/uv CLI commands, current
Typer command-line interface, Git, and Slurm documentation.

---

### Task 1: Write the complete coworker guide

**Files:**

- Create: `docs/GETTING_STARTED.md`
- Reference: `configs/v0.yaml`
- Reference: `docs/research/theory-audit.md`
- Reference: `docs/results/overall-conclusion.md`
- Reference: `results/v0-method-means.csv`
- Reference: `results/branch-effects.csv`

- [ ] **Step 1: Write the opening and reading paths**

  Begin with a direct note explaining that the collaborator supplied the
  theory and this repository is an empirical test bed. Provide three routes:
  a 10-minute scientific overview, a 30-minute code tour, and a hands-on setup.
  State immediately that the code worked but the proposed selector did not
  beat random pseudo-labeling in this frozen experiment.

- [ ] **Step 2: Explain the biology from first principles**

  Define protein sequence, amino acid substitution, variant, deep mutational
  scanning (DMS), fitness score, assay, and ProteinGym. Include one compact
  mutation example such as `A42V`, explaining that the measured target is a
  scalar assay-specific effect and not a universal biological fitness value.

- [ ] **Step 3: Map the theory to the implemented task**

  Present the low-label setup in plain language and with
  `n=96`, `N_U=2000`, `N_test=1000`, `q=192`, and `w=0.1`. Add a table mapping
  manuscript symbols (`X`, `Y`, `D^L`, `D^U`, `f_theta`, `H`, `g_L`, `g_j`,
  `S_j`) to protein examples and code objects. Explain ESM-2 embeddings,
  ESM-1v teacher calibration, ridge regression, the four v0 methods, Spearman,
  MSE, and NDCG@10%.

- [ ] **Step 4: Explain the claim boundary and theory audit**

  Show the key identity for squared-loss self-teaching:

  ```text
  self pseudo-label = current student prediction
  => pseudo residual = 0
  => every first-round pseudo-gradient = 0
  => the score cannot rank candidates
  ```

  Explain that v0 therefore tests a non-identical external teacher. Distinguish
  the manuscript theorem, the implemented heuristic, and the protein
  application claim. Link to `docs/research/theory-audit.md` for proof details.

- [ ] **Step 5: Tell the experimental story and result**

  Describe v0, crossfit, locality, and exact-CV as a sequence of questions.
  Report the reviewed values exactly: supervised Spearman `0.29309`, random
  `0.34971`, full influence `0.29445`; full-minus-random `-0.05526` with 0/8
  assay wins; exact-CV MSE `1.6177` versus random `1.1857`, and Spearman
  `0.30336` versus `0.34971`. Explain that teacher signal exists, selection
  fails, and the result is consistent with validation overfitting and/or
  surrogate mismatch. State that 26 untouched outcomes remain sealed.

- [ ] **Step 6: Explain the repository by scientific responsibility**

  Add a table for `config.py`, `data.py`, `embeddings.py`, `ridge.py`,
  `selection.py`, `experiment.py`, `metrics.py`, `analysis.py`, exploratory
  modules, tests, Slurm scripts, experiment cards, and decision memos. For each
  row answer: what it does, when the collaborator would read it, and what it
  depends on.

- [ ] **Step 7: Add setup and safe first commands**

  Include exact commands and label their cost:

  ```bash
  git clone https://github.com/pengzhangzhi/self_improve_protein.git
  cd self_improve_protein
  uv sync --frozen --extra dev --extra embed
  uv run self-improve-protein --show-config
  uv run self-improve-protein --help
  uv run pytest -q
  ```

  Explain expected output, that tests do not rerun the scientific study, and
  that full data/embedding reproduction requires ProteinGym downloads, GPUs,
  Slurm, and site-specific `SI_*` variables.

- [ ] **Step 8: Add extension workflow, troubleshooting, and glossary**

  Require a new experiment card before a new selector, a development-only
  smoke/pilot before full runs, exact baseline parity, and untouched outcomes
  for confirmation. Add common fixes for missing `uv`, missing CUDA, absent raw
  artifacts, and provenance failures. End with a glossary and ordered links to
  the protocol, theory audit, result conclusion, experiment cards, and compact
  CSV tables.

- [ ] **Step 9: Scan the guide for unsupported or unfriendly language**

  Run:

  ```bash
  rg -n 'obviously|simply|just |failed project|proves the theorem' \
    docs/GETTING_STARTED.md
  ```

  Expected: no placeholders, dismissive phrasing, or claim inflation. Any match
  must be reviewed and removed or explicitly justified.

### Task 2: Rebuild README as the friendly front door

**Files:**

- Modify: `README.md`
- Reference: `docs/GETTING_STARTED.md`
- Reference: `docs/results/overall-conclusion.md`

- [ ] **Step 1: Replace the expert-first opening**

  Start with a two-paragraph plain-language explanation of the collaborator's
  question and the protein task. Add a prominent link to
  `docs/GETTING_STARTED.md` for readers new to biology or code.

- [ ] **Step 2: Present the result as a scientific answer**

  Use a small table for supervised, random pseudo-labeling, top-teacher,
  influence, and no-Hessian v0 means. Follow it with three bullets: teacher
  signal exists; influence ranking does not extract it; the conclusion is
  specific to the frozen setup.

- [ ] **Step 3: Add a first-hour path and repository map**

  Provide a numbered sequence: read the guide, inspect config, run help, run
  tests, then read the decision memo. Add a short tree showing where protocol,
  source, tests, Slurm launchers, results, and research records live.

- [ ] **Step 4: Preserve reproducibility and development details**

  Keep pinned ProteinGym/ESM sources, install commands, optional embedding
  dependency behavior, test/lint/type commands, licensing, data exclusions,
  and links to all result memos. Avoid repeating the full guide.

### Task 3: Verify the documentation against live code and evidence

**Files:**

- Verify: `README.md`
- Verify: `docs/GETTING_STARTED.md`
- Verify: `results/v0-method-means.csv`
- Verify: `results/branch-effects.csv`

- [ ] **Step 1: Check all local Markdown links**

  Run an inline Python scanner over README and the guide. For every relative
  Markdown target, strip anchors and require the referenced path to exist.
  Expected output: `all local markdown links resolve`.

- [ ] **Step 2: Verify every lightweight command**

  Run:

  ```bash
  uv run --frozen self-improve-protein --show-config
  uv run --frozen self-improve-protein --help
  uv run --frozen pytest -q
  uv run --frozen ruff check .
  uv run --frozen mypy src
  ```

  Expected: protocol JSON and help text print successfully; all tests pass;
  Ruff and strict mypy report no errors.

- [ ] **Step 3: Verify reported numbers from tracked tables**

  Parse both CSVs with Python and assert the values quoted in README and the
  guide equal the tracked table values to the shown precision. Expected output:
  `headline result values match tracked tables`.

- [ ] **Step 4: Scan for private paths and secrets**

  Run:

  ```bash
  git grep -nE '(/lustre/|BEGIN (RSA|OPENSSH|EC) PRIVATE KEY|ghp_|github_pat_)' \
    -- README.md docs/GETTING_STARTED.md
  ```

  Expected: no output.

### Task 4: Review, commit, and publish

**Files:**

- Modify: `README.md`
- Create: `docs/GETTING_STARTED.md`

- [ ] **Step 1: Run a non-expert review**

  Check that the guide defines every domain term before use, separates biology,
  method, code, and results, and contains no unexplained command placeholder.
  Correct any issues inline.

- [ ] **Step 2: Inspect the exact diff**

  Run:

  ```bash
  git diff --check
  git diff -- README.md docs/GETTING_STARTED.md
  ```

  Expected: no whitespace errors; only intended documentation changes.

- [ ] **Step 3: Commit the documentation**

  ```bash
  git add README.md docs/GETTING_STARTED.md \
    docs/superpowers/plans/2026-06-30-coworker-onboarding.md
  git commit -m "docs: add coworker onboarding guide"
  ```

- [ ] **Step 4: Push and verify publication**

  ```bash
  git push origin main
  test "$(git rev-parse HEAD)" = \
    "$(git ls-remote origin refs/heads/main | awk '{print $1}')"
  ```

  Expected: `main` advances on GitHub and local HEAD equals the remote SHA.
