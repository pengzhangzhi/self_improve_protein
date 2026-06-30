# Getting started

Welcome. You supplied the mathematical theory; this repository is its empirical
test bed for a protein prediction problem. The completed, predeclared study
passed its software and result-traceability checks. Its scientific answer was
negative: the tested rule for choosing model-generated training examples did
**not** beat choosing the same number at random in this setup.

Choose the route that matches what you need today:

| Route | Read |
| --- | --- |
| **10-minute science** | [Biology](#the-biological-problem-from-first-principles) → [low-label experiment](#the-low-label-experiment) → [claim boundary](#what-v0-tests) → [results timeline](#results-timeline). Skip [notation-to-code](#manuscript-notation-to-code) on the first pass. |
| **30-minute code tour** | Follow the science route, include [notation-to-code](#manuscript-notation-to-code), then use the [first code tour](#first-code-tour). |
| **Hands-on setup** | Start with [local setup](#local-setup) and [troubleshooting](#troubleshooting). Use [Slurm reproduction](#full-reproduction-with-slurm) only with a maintainer. |

## The biological problem, from first principles

A **protein sequence** is an ordered chain of amino acids. Code represents it as
a string over the 20 standard one-letter amino-acid symbols. Changing one amino
acid creates a **substitution variant**. For example, `A42V` means that alanine
(`A`) at sequence position 42 is replaced by valine (`V`); the numbering follows
the experiment's reference sequence.

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
| `N_test` | 1,000 | Sealed test variants unavailable until final evaluation |
| `q` | 192 | Candidates selected for pseudo-labeling |
| `w` | 0.1 | Training weight of each pseudo-labeled candidate |

The remaining working-set rows form an unused buffer. There are eight assays
and five fixed seeds, giving 40 confirmatory tasks. The 192 pseudo-labels carry
aggregate weight `wq = 192 * 0.1 = 19.2` beside 96 real labels. The total
training weight is therefore `96 + 19.2 = 115.2`, and pseudo-labels contribute
`19.2 / 115.2 = 1/6` of it.

The model pipeline has four stages:

1. **Representation.** ESM-2, a pretrained protein language model, converts
   each sequence into a 480-number **embedding**. A **residue** is the amino
   acid at one sequence position; ESM-2 averages its residue representations.
   Its model parameters are not trained in this study.
2. **External teacher.** ProteinGym's `ESM1v_ensemble` predicts mutation effects
   from sequence without fitting to this assay, often called a *zero-shot*
   score. A slope and intercept fitted on the 96 labels align those scores to
   the assay's standardized scale. The calibrated predictions become candidate
   pseudo-labels.
3. **Student.** A linear ridge regressor predicts standardized DMS score from
   the ESM-2 embedding. Ridge adds a quadratic penalty to stabilize a
   480-feature fit from only 96 labels. Before fitting, the code centers
   embeddings with labeled-only means and divides by one scalar
   root-mean-square (RMS) scale.
4. **Selection and refit.** A method chooses 192 of the 2,000 candidates. The
   student is refit on the 96 real labels plus those weighted pseudo-labels,
   then evaluated on the test split.

Two similar words have different jobs here. The **locked protocol** is the
study plan written down before results and then held fixed. The **frozen ESM-2
model** is a pretrained network whose parameters are not updated here.

The four v0 comparison methods differ only in whether and how candidates are
chosen:

| Method | Candidate rule |
| --- | --- |
| Supervised | Use no pseudo-labels; this measures the 96-label student |
| Random | Sample 192 candidates without replacement; this is the primary comparator |
| Top teacher | Select the 192 largest calibrated teacher predictions |
| Full influence | Select the 192 largest paper-inspired influence scores |

An **ablation** is a controlled removal of one method component. The separately
carded exploratory [no-Hessian
ablation](research/experiment-card-no-hessian.md) replaces inverse-Hessian
geometry with the identity. All pseudo-label methods otherwise share the same
teacher labels, count, weight, preprocessing, student, and test set.

### Manuscript notation to code

| Symbol | Meaning | Protein instantiation / code object |
| --- | --- | --- |
| `X` | Input features | Labeled-only centered, scalar-RMS-scaled ESM-2 matrices. `FitInputs.x_*` hold raw embeddings; `experiment.py` creates the transformed local `x_l`, `x_u`, and `x_test` used by ridge. |
| `Y` | Measured response | Assay-specific DMS scores; labeled values are standardized and hidden values are isolated in `EvaluationLabels` |
| `D^L` | Labeled dataset | The 96 embedding/score pairs passed to the student |
| `D^U` | Unlabeled dataset | The 2,000 candidate embeddings and external-teacher predictions, without true candidate scores |
| `f_theta` | Student prediction function | Linear prediction `x @ theta` using a transformed embedding `x` and the ridge coefficients |
| `H` | Local curvature of labeled ridge loss | `x_l.T @ x_l / n + lambda I` from the transformed labeled matrix; scoring also adds damping `rho I` |
| `g_L` | Labeled loss gradient | Mean unregularized residual gradient at the supervised ridge fit |
| `g_j` | Candidate pseudo-gradient | Candidate residual times its embedding, using the calibrated external pseudo-label |
| `S_j` | Candidate selection score | `g_L.T @ inv(H + rho I) @ (g_j - g_L)`; larger values rank first |

The primary metric is **Spearman correlation**, which compares the rank order of
predicted and measured fitness; higher is better. Standardized **mean squared
error (MSE)** measures numerical prediction error and is the closer match to the
theory's squared-risk objective; lower is better. **Normalized discounted
cumulative gain at 10% (NDCG@10%)** measures how well the model ranks the
highest-fitness tenth of the test variants; higher is better.

## What v0 tests

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
V0 therefore uses ProteinGym's `ESM1v_ensemble` score as an external teacher;
its assay calibration still uses the same 96 labels. Its pseudo-label generally
differs from the current ridge prediction. V0 tests an
**external-teacher influence heuristic**, not literal self-teaching and not a
direct empirical proof of the manuscript theorem.

Keep three claims separate:

1. The manuscript theorem concerns a local population-risk expansion.
2. The implemented score adapts that idea to fixed-cardinality selection from a
   finite candidate pool using external pseudo-labels.
3. The experiment asks whether that adapted ranking improves test-set protein
   fitness prediction relative to random selection.

The [theory-to-experiment audit](research/theory-audit.md) gives the derivation
and records additional boundaries, including adaptive top-score selection and
the finite rather than asymptotic perturbation. The completed result is specific
to this teacher, representation, ridge student, label budget, and assay slice;
it is not an impossibility theorem for pseudo-label selection.

## Results timeline

V0 ran first under the locked plan. After its result was reviewed, the crossfit,
locality, and exact-CV studies were written down and run on assay outcomes that
had already been viewed in v0 or development. These later studies diagnose why
the method behaved as it did; they do not replace the v0 result.

Exact-CV reuses the same 96 labels in four folds. At each selection step, a fold
fits on 72 labels and uses the other 24 as validation data. Those 24 are
excluded from that fold's fit but deliberately reused while choosing 192
candidates. The separate 1,000-variant sealed test labels are unavailable to
fitting, selection, and tuning until final evaluation, after the candidate
order, student fit, and predictions are fixed.

| Study | Question | Answer |
| --- | --- | --- |
| **v0** | Does full influence ranking beat random when labels and retraining are fixed? | No. The full selector lost on the primary paired comparison. |
| **Crossfit** | Is the in-sample outer gradient the problem? | Replacing it with a four-fold out-of-fold gradient did not improve selection. |
| **Locality** | Is the 192-point, weight-0.1 update too large for a first-order score? | Smaller updates made the parameter approximation more faithful, but full-Hessian and cross-fitted influence each failed to beat random in all 15 cells. |
| **Exact-CV** | Does exact greedy fold-validation-loss lookahead work after removing the Hessian and Taylor approximations? | It fit the reused validation folds strongly but generalized worse than random. |

At one locality-grid setting (`q=72`, `w=0.1`), no-Hessian was `+0.00654`
Spearman above random. This descriptive result was not significant (assay
sign-flip `p=0.6172`), and its MSE was worse than random in all 15 settings. No
selection method established superiority to random.

The reviewed v0 mean Spearman values were:

| Supervised | Random pseudo-labeling | Top teacher | Full influence |
| ---: | ---: | ---: | ---: |
| `0.29309` | `0.34971` | `0.34230` | `0.29445` |

Full influence minus random was `-0.05526` Spearman; full influence had lower
mean Spearman in all eight assays. Random's improvement over supervised is
consistent with useful signal in the calibrated external teacher, but that
contrast also changes the loss-to-penalty balance (effective regularization).
Full influence versus random holds pseudo-labels, count, weight, and retraining
objective fixed, so it is the clean selection test—and full influence lost it.

A **selection surrogate** is the quantity used to choose candidates, which may
differ from the final goal. Exact-CV's surrogate was reused fold-validation
MSE. It fell sharply, yet test MSE was `1.6177` versus random's `1.1857`, and
Spearman was `0.30336` versus `0.34971`. This is consistent with adaptive
validation overfitting and/or a mismatch between the selection surrogate and
test performance; the study does not identify one unique cause.

A **promotion gate** is a criterion written before a study for deciding whether
to proceed to untouched assays. Every exploratory gate failed, so the 26
designated untouched assay outcomes remain **sealed**: they were not read,
summarized, or used for selection or tuning. The authoritative interpretation,
including evidence hashes, is in the [overall
conclusion](results/overall-conclusion.md).

## First code tour

Follow four symbol-level steps rather than reading whole modules:

| Step | Exact symbols | What to follow |
| --- | --- | --- |
| 1. Load the plan | [`Protocol`, `load_protocol`](../src/self_improve_protein/config.py) | Validate the locked YAML values from [`configs/v0.yaml`](../configs/v0.yaml). |
| 2. Build the student | [`fit_feature_transform`, `fit_label_transform`, `fit_teacher_calibration`, `fit_weighted_ridge`](../src/self_improve_protein/ridge.py) | Center embeddings with labeled-only means and one scalar RMS scale, calibrate the teacher, and fit ridge. |
| 3. Rank candidates | [`influence_scores`, `stable_top_k`](../src/self_improve_protein/selection.py) | Compute the full score, order candidates, and break exact ties reproducibly. |
| 4. Run one task | [`FitInputs`, `fit_task`, `evaluate_task`](../src/self_improve_protein/experiment.py) | Keep hidden outcomes outside fitting, refit every method, then admit evaluation labels and compute results. |

### Broader reference map

| Responsibility | Where to look |
| --- | --- |
| ProteinGym joins, eligibility, manifests, and splits | [`data.py`](../src/self_improve_protein/data.py) |
| ESM-2 pooling and validated embedding caches | [`embeddings.py`](../src/self_improve_protein/embeddings.py) |
| Metric definitions and assay-level summaries | [`metrics.py`](../src/self_improve_protein/metrics.py), [`analysis.py`](../src/self_improve_protein/analysis.py) |
| Later diagnostic implementations | [`crossfit.py`](../src/self_improve_protein/crossfit.py), [`locality.py`](../src/self_improve_protein/locality.py), [`exact_cv.py`](../src/self_improve_protein/exact_cv.py) |
| Executable checks | [`tests/`](../tests/), [`verify_r1_r3.sh`](../scripts/verify_r1_r3.sh) |
| Experiment plans and reviewed decisions | [`docs/research/`](research/), [`docs/results/`](results/) |

## Local setup

Python 3.11 or newer and [uv](https://docs.astral.sh/uv/) are required.

### Fast orientation

The clone and first sync require network access. This path does not install
Torch or require a GPU:

```bash
git clone https://github.com/pengzhangzhi/self_improve_protein.git
cd self_improve_protein
uv sync --frozen
uv run self-improve-protein --show-config
uv run self-improve-protein --help
```

`uv sync --frozen` creates or updates the project-local `.venv` using the
applicable base-package versions recorded in `uv.lock`; it does not install the
optional developer or embedding groups. `uv run` executes the following command
inside `.venv`, so you do not need to activate it manually. The config command
prints one validated JSON record; help lists the pipeline stages. Both are
CPU-only and take seconds after installation.

### Developer checks and embedding support

Install the developer tools plus the optional Torch/Transformers embedding
stack, then run the tests:

```bash
uv sync --frozen --extra dev --extra embed
uv run pytest -q
```

The embedding extra has a larger network and disk cost. The tests use small
fixtures, algebra checks, traceability checks, and synthetic probes. They do not
download ProteinGym, embed the study sequences, submit cluster jobs, or
recompute the scientific result.

## Troubleshooting

| Symptom | Response |
| --- | --- |
| `uv: command not found` | Install uv using its [official instructions](https://docs.astral.sh/uv/getting-started/installation/), open a new shell, and rerun the base sync. |
| CUDA is unavailable | Orientation and tests can run on CPU. After installing the embedding extra, check `uv run python -c "import torch; print(torch.cuda.is_available())"`; run `embed-assay` only in an approved GPU environment. |
| ProteinGym archives or embeddings are absent | This is expected in a fresh clone: large artifacts are excluded from Git. Use the pinned URLs and SHA-256 values in [`configs/v0.yaml`](../configs/v0.yaml), then ask the maintainer for the approved data layout. |
| A digest, manifest, or traceability check fails | Stop. Confirm the config, source files, Git revision, ordered row hashes, runtime, and artifact root. Do not bypass a confirmatory check or edit a receipt to make it pass. |

## Full reproduction with Slurm

**Maintainer-assisted.** Slurm is the scheduler that queues CPU and GPU work on
a shared cluster. Submitting through it changes cluster state and consumes
shared resources. Before any submission, obtain an approved private site
profile, data/model access, storage locations, and account/partition approval
from the maintainer.

The [pipeline launcher](../slurm/submit_pipeline.sh) expects site-specific
`SI_ACCOUNT`, `SI_CPU_PARTITION`, `SI_GPU_PARTITION`, `SI_REPO_ROOT`,
`SI_DATA_ROOT`, `SI_ARTIFACT_ROOT`, and `SI_SLURM_CONF` values. Keep their
values private. It defaults to `SI_MODE=development`, which schedules two
assay-seed tasks. One development submission creates this dependency-ordered
footprint: one CPU preparation job, a nine-member GPU embedding array using one
GPU per member, a two-member CPU task array, and one CPU aggregate job. Each
stage waits for the preceding stage to succeed. The full eight-assay by
five-seed study uses a 40-member task array and requires
`SI_MODE=confirmatory`.

Confirmatory mode also requires `SI_R5_GATE` set to the filesystem path of a
completed, validated R5 verification-gate JSON. R5 records that the
development-only pilot and its evidence passed; see the [verification
ladder](research/feedback-ladder.md).

After the approved profile exports those values:

```bash
# SUBMITS JOBS and changes cluster state
bash slurm/submit_pipeline.sh
```

Expected output is the path to `local/slurm/<run-id>/job_ids.json`, which
records the submitted prepare, embedding, task-array, and aggregate job IDs.
This is not a local orientation command. Copy every non-null ID from that file.
Monitor the pending/running chain with `squeue --job=ID1,ID2,ID3,ID4`. To stop
the whole chain, pass all non-null IDs as separate arguments to
`scancel ID1 ID2 ID3 ID4`; include downstream pending jobs as well as any
running job. `squeue` is read-only, while `scancel` changes cluster state.

## Extending the study safely

Use this sequence for a new selector or research question:

1. Start from the [v0 experiment card](research/experiment-card-v0.md) and write
   a new card before inspecting new outcomes. State the question, the one factor
   this study changes, comparator, metric, success gate, data slice, and stop
   rule.
2. Preserve baseline parity. Reuse the same split, teacher predictions, `q`,
   `w`, student objective, random baseline, and metrics unless the card names a
   change. A method comparison should differ only where declared.
3. Add focused tests, then run a [development-only smoke task and
   pilot](research/feedback-ladder.md). Passing them establishes execution and
   data-flow integrity, not method quality.
4. Record and lock the implementation, config, selections, predictions, and
   traceability receipts before hidden outcomes enter evaluation.
5. Apply the predeclared promotion gate. Use untouched outcomes only for an
   authorized confirmation; do not tune on them or unseal them after a failed
   development gate.
6. Record a separate decision memo. Exploratory repairs remain diagnoses and
   do not replace the completed v0 result.

## Glossary

| Term | Meaning here |
| --- | --- |
| Assay | One laboratory measurement context and its variant-score table |
| DMS | Deep mutational scanning: measuring many sequence variants in parallel |
| Variant | A protein sequence that differs from the assay reference |
| Embedding | A fixed numeric representation of a sequence |
| Teacher | ProteinGym's `ESM1v_ensemble` score, affine-calibrated to supply pseudo-labels |
| Pseudo-label | A model prediction treated as a weighted training target |
| Student | The ridge regressor trained on ESM-2 embeddings |
| Hessian | Matrix describing local curvature of the fitted objective |
| Influence score | First-order estimate used to rank a candidate's effect |
| Fold-validation data | Labels excluded from one exact-CV fold fit but repeatedly reused to guide selection |
| Sealed data | Outcomes unavailable to fitting, selection, and tuning until the declared evaluation; the 26 untouched assay outcomes were never opened |
| Slurm | Cluster scheduler used for dependency-ordered CPU/GPU jobs |

## Sources of truth, in reading order

1. [Locked v0 configuration](../configs/v0.yaml)
2. [Theory-to-experiment audit](research/theory-audit.md)
3. [Overall scientific conclusion](results/overall-conclusion.md)
4. Experiment cards: [v0](research/experiment-card-v0.md), [no-Hessian](research/experiment-card-no-hessian.md), [crossfit](research/experiment-card-crossfit.md), [locality](research/experiment-card-locality.md), and [exact-CV](research/experiment-card-exact-cv.md)
5. Compact reviewed tables: [v0 method means](../results/v0-method-means.csv) and [branch effects](../results/branch-effects.csv)
