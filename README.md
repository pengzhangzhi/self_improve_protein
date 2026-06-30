# self-improve-protein

This project asks whether a score can choose model-generated labels that help
low-label protein-fitness prediction. A protein variant is a sequence with
amino-acid changes; its fitness here is the value measured in one laboratory
assay. A model-generated training target is a **pseudo-label**.

We compared ways to select the same number of pseudo-labels. The study completed
and passed software and result-traceability checks, but the proposed
influence-based selector did not beat random selection.

> **Start here:** New to protein biology or research code? [Getting
> started](docs/GETTING_STARTED.md) explains the experiment from first
> principles and gives a safe code tour.

## What we learned

An external **teacher** is a pretrained model that supplies pseudo-labels; the
**student** is a ridge regressor on protein representations. The teacher is
calibrated on 96 laboratory measurements; every pseudo-label method receives
the same candidate scores and selects 192 candidates for student training. The
primary predeclared comparison was full influence versus random. Top-teacher
was secondary; no-Hessian was a separately carded exploratory ablation.

These reviewed means across eight assays and five seeds are rounded to five
decimals. Spearman and NDCG@10% are higher-is-better; mean squared error (MSE)
is lower-is-better.

| Method | Spearman | MSE | NDCG@10% |
| --- | ---: | ---: | ---: |
| Supervised only | 0.29309 | 1.73293 | 0.67166 |
| Random pseudo-labels | 0.34971 | 1.18568 | 0.68573 |
| Top teacher score | 0.34230 | 1.40425 | 0.63870 |
| Full influence (ours) | 0.29445 | 1.56129 | 0.66225 |
| No-Hessian (exploratory) | 0.33722 | 1.72346 | 0.68960 |

- Random pseudo-labeling improved over supervised-only training, consistent
  with useful teacher signal. It also changes effective regularization, so this
  is not a clean selection test.

- Full versus random fixes the teacher, pseudo-label count, weight, student, and
  test set. Full influence was `-0.05526` mean Spearman below random, with 0/8
  assay wins.

- This is a setup-specific result for one teacher, representation, student,
  label budget, and assay slice. It is not an impossibility theorem for
  pseudo-label selection or protein fitness prediction.

Exact-CV added one diagnostic: removing the influence approximation did not
rescue selection; its validation objective improved while hidden-test
performance worsened. The [overall scientific
conclusion](docs/results/overall-conclusion.md) records the mechanics,
interpretation, and 26 untouched assays that remained sealed.

## What claim does the study test?

The literal same-student rule uses its prediction as the pseudo-label. With
squared loss, the residual and candidate pseudo-gradient are zero; all
candidates tie and cannot be ranked. This is algebra, not a software defect.

V0 instead uses ProteinGym's external `ESM1v_ensemble` teacher, calibrated on
the 96 measured variants, to train a separate ridge student. It tests
external-teacher influence selection, not literal self-teaching or the
manuscript theorem. The [theory-to-experiment
audit](docs/research/theory-audit.md) gives the derivation and assumptions.

## Your first hour

1. Read [Getting started](docs/GETTING_STARTED.md), beginning with the
   10-minute science route.
2. Clone and create the CPU environment in [Development](#development).
3. Inspect the locked protocol in [`configs/v0.yaml`](configs/v0.yaml).
4. Run `uv run self-improve-protein --show-config` and
   `uv run self-improve-protein --help` to inspect the CLI.
5. Read the [overall conclusion](docs/results/overall-conclusion.md).

## Repository map

| Path | What it contains |
| --- | --- |
| [`configs/v0.yaml`](configs/v0.yaml) | Locked protocol, data/model pins, sizes, and seeds |
| [`src/self_improve_protein/`](src/self_improve_protein/) | Experiment implementation and diagnostics |
| [`tests/`](tests/) | Small-fixture, algebra, provenance, and CLI checks |
| [`slurm/`](slurm/) | Site-configured CPU/GPU launchers |
| [`docs/research/`](docs/research/) | Experiment cards, audits, and verification ladder |
| [`docs/results/`](docs/results/) | Reviewed scientific decisions |
| [`results/`](results/) | Compact public CSV tables |

Reviewed results: [overall conclusion](docs/results/overall-conclusion.md);
[v0](docs/results/v0-decision.md),
[crossfit](docs/results/crossfit-decision.md),
[locality](docs/results/locality-decision.md), and
[exact-CV](docs/results/exact-cv-decision.md) decisions; and the compact
[v0 means](results/v0-method-means.csv) and
[branch-effects](results/branch-effects.csv) tables.

## Pinned inputs

- Substitution assays and zero-shot scores come from the
  [ProteinGym v1.3 Zenodo record](https://zenodo.org/records/15293562).
- Sequence representations use
  [`facebook/esm2_t12_35M_UR50D`](https://huggingface.co/facebook/esm2_t12_35M_UR50D);
  its exact revision is pinned in [`configs/v0.yaml`](configs/v0.yaml).
- Pseudo-labels use the `ESM1v_ensemble` teacher distributed with the pinned
  ProteinGym release.

## Development

Python 3.11 or newer and [uv](https://docs.astral.sh/uv/) are required. Cloning
and the first sync need network access. This fast CPU path omits Torch:

```bash
git clone https://github.com/pengzhangzhi/self_improve_protein.git
cd self_improve_protein
uv sync --frozen
uv run self-improve-protein --show-config
uv run self-improve-protein --help
```

Optional full code-path checks require the developer and embedding extras:

```bash
uv sync --frozen --extra dev --extra embed
uv run pytest -q
uv run ruff check .
uv run mypy src
```

The first `embed` sync is a large download with substantial disk cost because
it installs Torch/Transformers. Tests use small fixtures and synthetic probes;
they check code paths, not the scientific conclusion. Full reproduction needs
site-specific resources and maintainer coordination. Follow the guide's
[maintainer-assisted Slurm section](docs/GETTING_STARTED.md#full-reproduction-with-slurm)
rather than treating it as a local first-run command.

The MIT license covers this repository's code. Public ProteinGym data and ESM
model artifacts retain their upstream licenses and terms; large datasets,
embeddings, model weights, logs, and task artifacts are not redistributed in
this repository.
