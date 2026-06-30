# self-improve-protein

This project asks a practical question: when laboratory measurements are
scarce, can a score choose model-generated labels that help predict protein
fitness? A protein variant is a sequence with one or more amino-acid changes,
and its fitness here is the value measured by one particular laboratory assay.
A model-generated training target is called a **pseudo-label**.

We tested several ways to select the same number of pseudo-labels for a small
prediction model. The software and predeclared study ran correctly, but the
scientific answer was negative: the proposed influence-based selector did not
beat choosing candidates at random.

> **Start here:** If protein biology or research code is new to you, read
> [Getting started](docs/GETTING_STARTED.md). It explains the experiment from
> first principles and gives a safe, guided code tour.

## What we learned

The locked v0 study covered eight ProteinGym assays and five random seeds. The
table reports tracked means, rounded to five decimals. Spearman correlation and
NDCG@10% are ranking metrics (higher is better); mean squared error (MSE)
measures numerical prediction error (lower is better).

| Method | Spearman | MSE | NDCG@10% |
| --- | ---: | ---: | ---: |
| Supervised only | 0.29309 | 1.73293 | 0.67166 |
| Random pseudo-labels | 0.34971 | 1.18568 | 0.68573 |
| Top teacher score | 0.34230 | 1.40425 | 0.63870 |
| Full influence (ours) | 0.29445 | 1.56129 | 0.66225 |
| No-Hessian influence | 0.33722 | 1.72346 | 0.68960 |

- Random pseudo-labeling improved over supervised-only training. That is
  consistent with useful signal in the calibrated teacher, but this comparison
  also changes the loss-to-penalty balance (effective regularization), so it is
  not a clean test of selection alone.

- Full influence versus random is the clean selection comparison: both use the
  same teacher, pseudo-label count, weight, student, and test set. Full
  influence was `-0.05526` mean Spearman below random, with 0/8 assay wins.

- This is a setup-specific result for one teacher, representation, student,
  label budget, and assay slice. It is not an impossibility theorem for
  pseudo-label selection or protein fitness prediction.

An exact cross-validation (exact-CV) diagnosis removed the influence
approximation and selected candidates by their exact effect on reused
validation folds. Validation loss fell sharply while hidden-test performance
worsened, which is consistent with adaptive validation overfitting and/or a
mismatch between the selection target and the final test goal. The 26
designated untouched assay outcomes remained sealed. See the
[overall scientific conclusion](docs/results/overall-conclusion.md) for the
complete evidence chain and decision.

## What claim does the study test?

The literal same-student idea uses a student's own prediction as its
pseudo-label. Under squared loss, prediction minus pseudo-label is then zero,
so each candidate has zero residual and zero pseudo-gradient. Every candidate
ties; the score has no ranking signal. This is an algebraic boundary, not a
software defect.

The study therefore uses ProteinGym's external `ESM1v_ensemble` teacher. Its
scores are calibrated on the 96 measured training variants and used as
pseudo-labels for a separate ridge-regression student. V0 tests whether the
proposed influence score can select helpful labels from that external teacher;
it does not directly test literal self-teaching or prove the manuscript's
theorem. The [theory-to-experiment audit](docs/research/theory-audit.md) gives
the derivation and the remaining assumptions.

## Your first hour

1. Read [Getting started](docs/GETTING_STARTED.md), using its 10-minute science
   route if you want the shortest conceptual introduction.
2. Clone the repository and create the CPU environment in
   [Development](#development).
3. Inspect the locked protocol in [`configs/v0.yaml`](configs/v0.yaml).
4. Run `uv run self-improve-protein --show-config` and
   `uv run self-improve-protein --help` to see the validated configuration and
   available pipeline stages.
5. Run `uv run pytest`. These are code-path and traceability checks; they do
   **not** rerun the scientific study.
6. Read the [overall conclusion](docs/results/overall-conclusion.md) to see what
   the evidence supports and what remains open.

## Repository map

| Path | What it contains |
| --- | --- |
| [`configs/v0.yaml`](configs/v0.yaml) | Locked protocol, data hashes, model revision, sample sizes, and seeds |
| [`src/self_improve_protein/`](src/self_improve_protein/) | Data preparation, embeddings, student fitting, selection, evaluation, and diagnostic implementations |
| [`tests/`](tests/) | Small-fixture, algebra, provenance, and command-line code-path checks |
| [`slurm/`](slurm/) | Site-configured launchers for full CPU/GPU reproduction |
| [`docs/research/`](docs/research/) | Experiment cards, protocol audit, theory audit, and verification ladder |
| [`docs/results/`](docs/results/) | Reviewed decision memos and the overall scientific conclusion |
| [`results/`](results/) | Compact, public CSV tables rather than large raw artifacts |

The reviewed decisions are
[v0](docs/results/v0-decision.md),
[crossfit](docs/results/crossfit-decision.md),
[locality](docs/results/locality-decision.md), and
[exact-CV](docs/results/exact-cv-decision.md). The compact tables are
[v0 method means](results/v0-method-means.csv) and
[diagnostic branch effects](results/branch-effects.csv).

## Pinned inputs

- Substitution assays and zero-shot scores come from the
  [ProteinGym v1.3 Zenodo record](https://zenodo.org/records/15293562).
- Sequence representations use
  [`facebook/esm2_t12_35M_UR50D`](https://huggingface.co/facebook/esm2_t12_35M_UR50D);
  its exact revision is pinned in [`configs/v0.yaml`](configs/v0.yaml).
- Pseudo-labels use the `ESM1v_ensemble` teacher distributed with the pinned
  ProteinGym release.

## Development

Python 3.11 or newer and [uv](https://docs.astral.sh/uv/) are required. The
clone and first sync need network access. This fast orientation path is CPU-only
and does not install the large Torch embedding stack:

```bash
git clone https://github.com/pengzhangzhi/self_improve_protein.git
cd self_improve_protein
uv sync --frozen
uv run self-improve-protein --show-config
uv run self-improve-protein --help
```

For the complete developer environment, including the optional embedding
dependencies, run:

```bash
uv sync --frozen --extra dev --extra embed
uv run pytest
uv run ruff check .
uv run mypy src
```

The `embed` extra installs a large Torch/Transformers stack. The tests exercise
the implementation with small fixtures and synthetic probes; passing them
checks software behavior, not the study's scientific conclusion. Full data
preparation, embedding, and Slurm reproduction require site-specific resources
and maintainer coordination. Follow the guide's
[maintainer-assisted Slurm section](docs/GETTING_STARTED.md#full-reproduction-with-slurm)
rather than treating it as a local first-run command.

The MIT license covers this repository's code. Public ProteinGym data and ESM
model artifacts retain their upstream licenses and terms; large datasets,
embeddings, model weights, logs, and task artifacts are not redistributed in
this repository.
