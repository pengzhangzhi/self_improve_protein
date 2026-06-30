# Getting started

Welcome. You supplied the mathematical theory; this repository is the empirical
test bed built to ask how its selection idea behaves in a controlled protein
fitness problem. The implementation completed the frozen study and passed its
software and provenance checks. The scientific answer was negative for the
selector: it did **not** beat random selection of model-generated training
targets in this setup.

Choose the route that matches what you need today:

| Route | Read |
| --- | --- |
| **10-minute science** | “The biological problem,” “The low-label experiment,” “What v0 tests,” and “The experimental story” |
| **30-minute code tour** | The science sections, then “Repository map”; open `config.py`, `ridge.py`, `selection.py`, and `experiment.py` in that order |
| **Hands-on setup** | “Local setup,” then “Extending the study safely” and “Troubleshooting” |

## The biological problem, from first principles

A **protein sequence** is an ordered chain of amino acids. Code represents it as
a string over the 20 standard one-letter amino-acid symbols. Changing one amino
acid creates a **substitution variant**. For example, `A42V` means that alanine
(`A`) at sequence position 42 is replaced by valine (`V`); the numbering belongs
to that assay's reference sequence.

A biological **assay** is a laboratory procedure that measures a particular
property, such as activity, binding, or growth. **Deep mutational scanning
(DMS)** measures that property for many sequence variants in parallel. Each row
then has a variant, its mutated sequence, and a scalar **fitness score** for the
measured phenotype. Higher and lower values describe performance in that assay;
they are not a universal measure of an organism's or protein's fitness. A score
from one assay should not be compared directly with a score from another.

[ProteinGym](https://proteingym.org/) standardizes collections of these assays
for evaluating protein models. This study uses substitution assays from the
pinned ProteinGym v1.3 release. Its prediction task is regression: use a small
number of measured variants to predict the assay-specific DMS score of unseen
variants.

## The low-label experiment

Because laboratory labels are scarce, a pretrained **teacher** model supplies
predicted scores called **pseudo-labels**: model-generated values used as
weighted training targets for a smaller **student** model.

For each assay and random seed, the code creates disjoint sets from a
deterministic 6,000-variant working set:

| Quantity | Value | Role |
| --- | ---: | --- |
| `n` | 96 | Labeled variants available to fit and calibrate models |
| `N_U` | 2,000 | Candidate variants whose true DMS scores are hidden during selection |
| `N_test` | 1,000 | Held-out variants used only after fitting |
| `q` | 192 | Candidates selected for pseudo-labeling |
| `w` | 0.1 | Training weight of each pseudo-labeled candidate |

The remaining working-set rows form an unused buffer. There are eight assays
and five fixed seeds, giving 40 confirmatory tasks. The 192 pseudo-labels carry
total weight `wq = 19.2` beside 96 real labels, so their final objective fraction
is `19.2 / 115.2 = 1/6`.

The model pipeline has four stages:

1. **Representation.** ESM-2, a pretrained protein language model, converts
   each sequence into a 480-number **embedding** by averaging its residue
   representations. “Frozen” means the ESM-2 parameters are not trained here;
   embeddings are fixed inputs.
2. **External teacher.** ProteinGym's `ESM1v_ensemble` predicts mutation effects
   from sequence without fitting to this assay, often called a *zero-shot*
   score. A slope and intercept fitted on the 96 labels align those scores to
   the assay's standardized scale. The calibrated predictions become candidate
   pseudo-labels.
3. **Student.** A no-intercept linear ridge regressor predicts standardized DMS
   score from the ESM-2 embedding. Ridge adds a quadratic penalty to stabilize
   a 480-feature fit from only 96 labels.
4. **Selection and refit.** A method chooses 192 of the 2,000 candidates. The
   student is refit on the 96 real labels plus those weighted pseudo-labels,
   then evaluated on the untouched test split.

The four v0 comparison methods differ only in whether and how candidates are
chosen:

| Method | Candidate rule |
| --- | --- |
| Supervised | Use no pseudo-labels; this measures the 96-label student |
| Random | Sample 192 candidates without replacement; this is the primary comparator |
| Top teacher | Select the 192 largest calibrated teacher predictions |
| Full influence | Select the 192 largest paper-inspired influence scores |

A separately carded exploratory **no-Hessian** ablation replaces inverse-Hessian
geometry with the identity. All pseudo-label methods otherwise share the same
teacher labels, count, weight, preprocessing, student, and test set.

### Manuscript notation to code

| Symbol | Meaning | Protein instantiation / code object |
| --- | --- | --- |
| `X` | Input features | ESM-2 embedding matrices such as `FitInputs.x_l`, `x_u`, and `x_test` |
| `Y` | Measured response | Assay-specific DMS scores; labeled values are standardized and hidden values are isolated in `EvaluationLabels` |
| `D^L` | Labeled dataset | The 96 embedding/score pairs passed to the student |
| `D^U` | Unlabeled dataset | The 2,000 candidate embeddings and external-teacher predictions, without true candidate scores |
| `f_theta` | Student prediction function | Linear prediction `x @ theta` from the ridge coefficients |
| `H` | Local curvature of labeled ridge loss | `X_L.T @ X_L / n + lambda I` in `selection.py`; scoring also adds damping `rho I` |
| `g_L` | Labeled loss gradient | Mean unregularized residual gradient at the supervised ridge fit |
| `g_j` | Candidate pseudo-gradient | Candidate residual times its embedding, using the calibrated external pseudo-label |
| `S_j` | Candidate selection score | `g_L.T @ inv(H + rho I) @ (g_j - g_L)`; larger values rank first |

The primary metric is **Spearman correlation**, which compares the rank order of
predicted and measured fitness; higher is better. Standardized **mean squared
error (MSE)** measures numerical prediction error and is the closer match to the
theory's squared-risk objective; lower is better. **Normalized discounted
cumulative gain at 10% (NDCG@10%)** measures how well the model ranks the
highest-fitness tenth of the test variants; higher is better.

## What v0 tests—and what it does not

The manuscript's literal squared-loss self-teacher cannot provide a first-round
ranking at the supervised optimum:

```text
self pseudo-label = current student prediction
=> pseudo residual = 0
=> pseudo-gradient = 0
=> every candidate receives the same score, so there is no ranking signal
```

With ridge regularization, the shared score is a non-positive constant; without
regularization it is zero. This is an algebraic property, not a software issue.
V0 therefore substitutes a pretrained ESM-1v model external to the ridge
student; its assay calibration still uses the same 96 labels. Its pseudo-label
generally differs from the current ridge prediction. V0 tests an
**external-teacher influence heuristic**, not literal self-teaching and not a
direct empirical proof of the manuscript theorem.

Keep three claims separate:

1. The manuscript theorem concerns a local population-risk expansion.
2. The implemented score adapts that idea to fixed-cardinality selection from a
   finite candidate pool using external pseudo-labels.
3. The experiment asks whether that adapted ranking improves held-out protein
   fitness prediction relative to random selection.

The [theory-to-experiment audit](research/theory-audit.md) gives the derivation
and records additional boundaries, including adaptive top-score selection and
the finite rather than asymptotic perturbation. The completed result is specific
to this teacher, representation, ridge student, label budget, and assay slice;
it is not an impossibility theorem for pseudo-label selection.

## The experimental story

Each later study was a predeclared diagnosis on already exposed assays. None
replaces the locked v0 comparison. “Exact-CV” below means exact
cross-validation: repeatedly fitting on part of the labeled data and scoring
candidates on the held-out part.

| Study | Question | Answer |
| --- | --- | --- |
| **v0** | Does full influence ranking beat random when labels and retraining are fixed? | No. The full selector lost on the primary paired comparison. |
| **Crossfit** | Is the in-sample outer gradient the problem? | Replacing it with a four-fold out-of-fold gradient did not improve selection. |
| **Locality** | Is the 192-point, weight-0.1 update too large for a first-order score? | Smaller updates made the parameter approximation much more faithful, but influence selection still did not beat random in any tested cell. |
| **Exact-CV** | Does exact greedy held-out-loss lookahead work after removing the Hessian and Taylor approximations? | It fit the reused validation folds strongly but generalized worse than random. |

The reviewed v0 mean Spearman values were:

| Supervised | Random pseudo-labeling | Full influence |
| ---: | ---: | ---: |
| `0.29309` | `0.34971` | `0.29445` |

Full influence minus random was `-0.05526` Spearman and won **0/8 assay
means**. Random's improvement over supervised shows that the calibrated
external teacher contains useful signal. Full influence's result shows that
this selector did not extract it better than a random subset.

Exact-CV reinforced the distinction between finding an attractive selection
surrogate and improving hidden outcomes. Its test MSE was `1.6177`, versus
random's `1.1857`; its Spearman was `0.30336`, versus random's `0.34971`.
Meanwhile, its reused fold-validation MSE fell sharply. The combined evidence
is consistent with adaptive validation overfitting and/or mismatch between the
selection surrogate and test performance. It does not identify either mechanism
as the unique cause.

All exploratory promotion gates failed, so the 26 designated untouched assay
outcomes remain sealed. The authoritative interpretation, including evidence
hashes, is in the [overall conclusion](results/overall-conclusion.md).

## Repository map

Read by scientific responsibility rather than alphabetical filename:

| Responsibility | Where | Read it when… | Main dependencies |
| --- | --- | --- | --- |
| Frozen protocol | [`configs/v0.yaml`](../configs/v0.yaml), [`config.py`](../src/self_improve_protein/config.py) | You need the exact data, model, split, and hyperparameter values | YAML, Pydantic |
| Data cohort and splits | [`data.py`](../src/self_improve_protein/data.py) | You need to understand joins, eligibility, row hashes, manifests, or disjoint splits | Protocol, ProteinGym archives |
| Sequence representation | [`embeddings.py`](../src/self_improve_protein/embeddings.py) | You need ESM-2 residue pooling, GPU inference, or validated caches | Torch, Transformers, source hashes |
| Teacher and student math | [`ridge.py`](../src/self_improve_protein/ridge.py) | You need feature/label transforms, ESM-1v calibration, ridge fitting, gradients, or Hessians | NumPy |
| Candidate rules | [`selection.py`](../src/self_improve_protein/selection.py) | You need random, top-teacher, full, no-Hessian, or crossfit scoring and tie-breaking | Ridge utilities, deterministic seeds |
| End-to-end task | [`experiment.py`](../src/self_improve_protein/experiment.py) | You need to trace one fit, the hidden-label boundary, refitting, diagnostics, and evaluation | Config, ridge, selection, metrics, provenance |
| Metric definitions | [`metrics.py`](../src/self_improve_protein/metrics.py) | You need exact Spearman, MSE, or NDCG@10% behavior | NumPy, SciPy |
| Confirmatory summaries | [`analysis.py`](../src/self_improve_protein/analysis.py) | You need assay-level aggregation, paired contrasts, uncertainty, or v0 decision rules | Pandas, SciPy, metrics tables |
| Exploratory diagnoses | [`crossfit.py`](../src/self_improve_protein/crossfit.py), [`locality.py`](../src/self_improve_protein/locality.py), [`exact_cv.py`](../src/self_improve_protein/exact_cv.py) | You need the later mechanism screens after understanding v0 | Frozen v0 objects plus branch-specific cards |
| Executable checks | [`tests/`](../tests/), [`verify_r1_r3.sh`](../scripts/verify_r1_r3.sh) | You change code or want evidence for algebra, leakage safety, determinism, and CLI contracts | Small fixtures; no full study rerun |
| Cluster orchestration | [`slurm/`](../slurm/), especially [`submit_pipeline.sh`](../slurm/submit_pipeline.sh) | You have approved data/GPU storage and are reproducing a card | Slurm, site variables, synced environment, raw artifacts |
| Experiment cards | [`experiment-card-v0.md`](research/experiment-card-v0.md), [`crossfit`](research/experiment-card-crossfit.md), [`locality`](research/experiment-card-locality.md), [`exact-CV`](research/experiment-card-exact-cv.md) | Before reading or changing an experiment implementation | Scientific question and frozen protocol |
| Decision memos | [`overall-conclusion.md`](results/overall-conclusion.md) and [`docs/results/`](results/) | You need reviewed outcomes rather than implementation detail | Aggregated, provenance-checked artifacts |

## Local setup

Python 3.11 or newer and [uv](https://docs.astral.sh/uv/) are required. From a
machine with network access:

```bash
git clone https://github.com/pengzhangzhi/self_improve_protein.git
cd self_improve_protein
uv sync --frozen --extra dev --extra embed
uv run self-improve-protein --show-config
uv run self-improve-protein --help
uv run pytest -q
```

What to expect:

| Command | Cost | Successful outcome |
| --- | --- | --- |
| `git clone` | Small source download | A new `self_improve_protein/` checkout |
| `uv sync ...` | One-time network/disk cost; the Torch embedding extra is large | A lockfile-faithful `.venv`; no ProteinGym data or model weights yet |
| `--show-config` | Seconds, CPU-only | One JSON record containing the validated protocol and its digest |
| `--help` | Seconds, CPU-only | The preparation, embedding, task, aggregation, and verification commands |
| `pytest -q` | CPU-only unit/property tests, normally minutes or less | A passing test summary |

The tests exercise code, fixtures, algebra, provenance checks, and synthetic
probes. They do not download ProteinGym, run ESM-2 over the study sequences,
submit Slurm jobs, or recompute the scientific result.

A full reproduction additionally needs the pinned ProteinGym archives, ESM
model weights, GPU embedding inference, artifact storage, and a Slurm cluster.
The launcher requires site-specific variables such as `SI_ACCOUNT`,
`SI_CPU_PARTITION`, `SI_GPU_PARTITION`, `SI_REPO_ROOT`, `SI_DATA_ROOT`,
`SI_ARTIFACT_ROOT`, and `SI_SLURM_CONF`. Set them in a private local profile;
never publish cluster paths, credentials, or raw-data locations in the
repository.

## Extending the study safely

Use this sequence for a new selector or research question:

1. Write a new experiment card before inspecting new outcomes. State the
   question, changed coordinate, comparator, metric, success gate, data slice,
   and stop rule.
2. Preserve baseline parity. Reuse the same split, teacher predictions, `q`,
   `w`, student objective, random baseline, and metrics unless the card names a
   change. A method comparison should differ only where declared.
3. Add focused tests, then run a development-only smoke task and pilot. Passing
   them establishes execution and data-flow integrity, not method quality.
4. Freeze the implementation, config, selections, predictions, and provenance
   receipts before hidden outcomes enter evaluation.
5. Apply the predeclared promotion gate. Use untouched outcomes only for an
   authorized confirmation; do not tune on them or unseal them after a failed
   development gate.
6. Record a separate decision memo. Exploratory repairs remain diagnoses and
   do not replace the completed v0 result.

## Troubleshooting

| Symptom | Response |
| --- | --- |
| `uv: command not found` | Install uv using its [official instructions](https://docs.astral.sh/uv/getting-started/installation/), open a new shell, and rerun the frozen sync. |
| CUDA is unavailable | Config inspection and tests can still run on CPU. Check `uv run python -c "import torch; print(torch.cuda.is_available())"`; run `embed-assay` only in an approved GPU environment. |
| ProteinGym archives or embeddings are absent | This is expected in a fresh clone: large/raw artifacts are excluded from Git. Use the pinned URLs and SHA-256 values in `configs/v0.yaml`, then follow the staged launcher. |
| A provenance, digest, or manifest check fails | Stop. Confirm the config, source files, Git revision, ordered row hashes, runtime, and artifact root. Do not bypass a confirmatory check or edit a receipt to make it pass. |

## Glossary

| Term | Meaning here |
| --- | --- |
| Assay | One laboratory measurement context and its variant-score table |
| DMS | Deep mutational scanning: measuring many sequence variants in parallel |
| Variant | A protein sequence that differs from the assay reference |
| Embedding | A fixed numeric representation of a sequence |
| Teacher | ESM-1v, whose calibrated prediction supplies pseudo-labels |
| Pseudo-label | A model prediction treated as a weighted training target |
| Student | The ridge regressor trained on ESM-2 embeddings |
| Hessian | Matrix describing local curvature of the fitted objective |
| Influence score | First-order estimate used to rank a candidate's effect |
| Held-out | Withheld from fitting or selection until the declared evaluation stage |
| Slurm | Cluster scheduler used for dependency-ordered CPU/GPU jobs |

## Sources of truth, in reading order

1. [Frozen v0 configuration](../configs/v0.yaml)
2. [Theory-to-experiment audit](research/theory-audit.md)
3. [Overall scientific conclusion](results/overall-conclusion.md)
4. Experiment cards: [v0](research/experiment-card-v0.md), [crossfit](research/experiment-card-crossfit.md), [locality](research/experiment-card-locality.md), and [exact-CV](research/experiment-card-exact-cv.md)
5. Compact reviewed tables: [v0 method means](../results/v0-method-means.csv) and [branch effects](../results/branch-effects.csv)
