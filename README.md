# self-improve-protein

Can a score choose model-generated labels that improve protein-fitness prediction
when laboratory labels are scarce? In the completed v0 study, the proposed
influence-based selector did **not** beat random selection. The software and result
lineage passed their checks; the scientific result is negative for this setup.

Choose a route:

- **Understand the study:** [Getting started](docs/GETTING_STARTED.md) explains the
  biology, theory-to-code mapping, and evidence from first principles.
- **Reproduce or extend it:** [Operations](docs/OPERATIONS.md) is the complete
  verification and execution runbook.

## What we learned

ProteinGym deep-mutational-scanning (DMS) assays provide measured fitness values for
protein sequence variants. Frozen ESM-2 embeddings turn each sequence into numeric
features. ProteinGym's external `ESM1v_ensemble` zero-shot model supplies the
pseudo-labels, and a separate ridge-regression student learns from the embeddings.
The teacher is calibrated on 96 measured variants; every selection method receives
the same candidate scores and chooses 192 candidates for student training.

The primary predeclared comparison was full influence versus random. Top-teacher was
secondary; no-Hessian was a separately carded exploratory ablation. These reviewed
means cover eight assays and five seeds and are rounded to five decimals. Spearman
and NDCG@10% are higher-is-better; mean squared error (MSE) is lower-is-better.

| Method | Spearman | MSE | NDCG@10% |
| --- | ---: | ---: | ---: |
| Supervised only | 0.29309 | 1.73293 | 0.67166 |
| Random pseudo-labels | 0.34971 | 1.18568 | 0.68573 |
| Top teacher score | 0.34230 | 1.40425 | 0.63870 |
| Full influence (ours) | 0.29445 | 1.56129 | 0.66225 |
| No-Hessian (exploratory) | 0.33722 | 1.72346 | 0.68960 |

- Random pseudo-labeling improved over supervised-only training, consistent with
  useful teacher signal. Adding weighted examples changes the **effective regularization**
  of the ridge model, so this is not a clean selection test.
- Full influence versus random fixes the teacher, pseudo-label count, weight,
  student, and test set. Full influence was `-0.05526` mean Spearman below random,
  with `0/8` assay wins.
- This is setup-specific evidence for one teacher, representation, student, label
  budget, and assay slice—not an impossibility result for pseudo-label selection or
  protein fitness prediction.

Exact cross-validation added a diagnostic: removing the influence approximation did
not rescue selection. Its reused validation objective improved while hidden-test
performance worsened. The 26 designated untouched assay outcomes remained sealed.
The [overall scientific conclusion](docs/results/overall-conclusion.md) records the
full evidence and decision.

## What claim does the study test?

The literal same-student rule uses the student's own prediction as a pseudo-label.
With squared loss, the residual and candidate pseudo-gradient are zero, so all
candidates tie and cannot be ranked. This is algebra, not a software defect.

V0 therefore uses the external `ESM1v_ensemble` teacher to train a separate ridge
student. It tests external-teacher influence selection, not literal self-teaching or
the manuscript theorem. The [theory-to-experiment
audit](docs/research/theory-audit.md) gives the derivation and assumptions.

## Your first hour

1. Follow the 10-minute science route in [Getting started](docs/GETTING_STARTED.md).
2. Use the safe local setup in [Development](#development).
3. Inspect the locked protocol in [`configs/v0.yaml`](configs/v0.yaml).
4. Read the [overall conclusion](docs/results/overall-conclusion.md).
5. Use [Operations](docs/OPERATIONS.md) before full verification or any Slurm work.

## Repository map

| Path | What it contains |
| --- | --- |
| [`configs/v0.yaml`](configs/v0.yaml) | Locked protocol, data/model pins, sizes, and seeds |
| [`src/self_improve_protein/`](src/self_improve_protein/) | Experiment implementation and diagnostics |
| [`tests/`](tests/) | Small-fixture, algebra, provenance, and CLI checks |
| [`docs/research/`](docs/research/) | Experiment cards, audits, and verification ladder |
| [`docs/results/`](docs/results/) | Reviewed scientific decisions |
| [`results/`](results/) | Compact public result tables |

Reviewed artifacts include the [v0](docs/results/v0-decision.md),
[crossfit](docs/results/crossfit-decision.md),
[locality](docs/results/locality-decision.md), and
[exact-CV](docs/results/exact-cv-decision.md) decisions, plus the compact
[method means](results/v0-method-means.csv) and
[branch effects](results/branch-effects.csv).

## Pinned inputs

- Assays and zero-shot scores come from the
  [ProteinGym v1.3 Zenodo record](https://zenodo.org/records/15293562).
- Embeddings use
  [`facebook/esm2_t12_35M_UR50D`](https://huggingface.co/facebook/esm2_t12_35M_UR50D)
  at the revision pinned in [`configs/v0.yaml`](configs/v0.yaml).
- Pseudo-labels use `ESM1v_ensemble` from the pinned ProteinGym release.

## Development

Python 3.11 or newer and [uv](https://docs.astral.sh/uv/) are required. These are
safe local orientation commands; cloning and the first sync require network access.

```bash
git clone https://github.com/pengzhangzhi/self_improve_protein.git
cd self_improve_protein
uv sync --frozen
uv run self-improve-protein --show-config
uv run self-improve-protein --help
```

For developer and embedding extras, full verification, data staging, and all Slurm
operations, follow [Operations](docs/OPERATIONS.md). The first embedding-environment
sync is a large download because it installs Torch and Transformers. Tests verify
paths, invariants, and controlled code behavior; they do not establish method
quality on protein tasks.

The MIT license covers this repository's code. ProteinGym data and ESM artifacts
retain their upstream licenses and terms; large datasets, embeddings, weights, logs,
and task artifacts are not redistributed here.
