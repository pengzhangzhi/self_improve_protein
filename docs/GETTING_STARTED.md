# Getting started

Welcome. You supplied the mathematical theory; this repository is its empirical
test bed for a protein prediction problem. The completed, predeclared study
passed its software and result-traceability checks. Its scientific answer was
negative: the tested rule for choosing model-generated training examples did
**not** beat choosing the same number at random in this setup.

Choose the route that matches what you need today:

| Route | Read |
| --- | --- |
| **10-minute science** | [Biology](#the-biological-problem) -> [experiment](#the-low-label-experiment) -> [claim boundary](#what-v0-tests) -> [results](#results-and-interpretation). Skip notation and code on the first pass. |
| **30-minute code tour** | Follow the science route, include [notation-to-code](#manuscript-notation-to-code), then use the [symbol-level tour](#symbol-level-code-tour). |
| **Hands-on** | Run the safe [local orientation](#safe-local-setup), then use the maintainer [operations runbook](OPERATIONS.md) for data, verification, and cluster work. |

## The biological problem

A **protein sequence** is an ordered chain of amino acids. Code represents it as
a string over the 20 standard one-letter amino-acid symbols. Changing one amino
acid creates a **substitution variant**. For example, `A42V` means that alanine
(`A`) at position 42 is replaced by valine (`V`), using the experiment's
reference-sequence numbering.

A biological **assay** is a laboratory procedure that measures a property such
as activity, binding, or growth. **Deep mutational scanning (DMS)** measures that
property for many sequence variants in parallel. Each data row has a variant,
its sequence, and a scalar **fitness** score for the measured phenotype. Here,
fitness means performance in one assay—not universal organism or protein
fitness—and scores from different assays are not directly comparable.

[ProteinGym](https://proteingym.org/) standardizes collections of these assays
for evaluating protein models. This study uses substitution assays from its
pinned v1.3 release. The task is regression: use a small number of measured
variants to predict assay-specific DMS scores for unseen variants.

## The low-label experiment

Laboratory labels are scarce, so a pretrained **teacher** supplies predicted
scores called **pseudo-labels**: model-generated values used as weighted
training targets for a smaller **student** model. An **embedding** is a fixed
numeric representation of a sequence that the student can use as input.

For each assay and random seed, the code draws disjoint sets from a deterministic
6,000-variant working set:

| Quantity | Value | Role |
| --- | ---: | --- |
| `n` | 96 | Labeled variants used to fit and calibrate models |
| `N_U` | 2,000 | Candidates whose true DMS scores are hidden during selection |
| `N_test` | 1,000 | Sealed test variants used only for final evaluation |
| `q` | 192 | Candidates selected for pseudo-labeling |
| `w` | 0.1 | Training weight of each pseudo-example |

The remaining rows are an unused buffer. Eight assays and five fixed seeds give
40 confirmatory tasks. The pseudo-examples carry aggregate weight
`wq = 192 * 0.1 = 19.2`, compared with 96 real labels: one sixth of the total
training weight `115.2`.

The pipeline has four stages:

1. **Representation.** Frozen ESM-2 converts each sequence into a 480-number
   embedding by averaging representations across its amino-acid positions. Its
   pretrained parameters are never updated here.
2. **External teacher.** ProteinGym's `ESM1v_ensemble` predicts mutation effects
   without fitting this assay. A slope and intercept fitted on the 96 labels
   align those zero-shot scores to the assay's standardized scale; the
   calibrated predictions are the candidate pseudo-labels.
3. **Student.** A linear ridge regressor predicts standardized DMS score from
   ESM-2 embeddings. Ridge adds a quadratic penalty to stabilize a 480-feature
   fit from 96 labels. Embeddings are centered with labeled-only means and
   divided by one scalar root-mean-square (RMS) scale.
4. **Selection and refit.** A method chooses 192 of 2,000 candidates. The same
   student is refit on 96 real labels plus the selected weighted pseudo-labels,
   then evaluated on the sealed test split.

The **locked protocol** is the study plan held fixed after preregistration; the
**frozen model** is a pretrained network whose parameters are not updated.

The four v0 methods are:

| Method | Candidate rule |
| --- | --- |
| Supervised | Use no pseudo-labels: the 96-label student |
| Random | Sample 192 candidates without replacement: the primary comparator |
| Top teacher | Choose the 192 largest calibrated teacher predictions |
| Full influence | Choose the 192 largest paper-inspired influence scores |

An **ablation** removes one method component to diagnose its role. The separately
carded exploratory [no-Hessian ablation](research/experiment-card-no-hessian.md)
replaces inverse-Hessian geometry with the identity. It was not part of the
confirmatory v0 claim. All pseudo-label methods otherwise share the teacher,
count, weight, preprocessing, student, and test set.

### Effective regularization matters

The ridge coefficient is numerically fixed, but adding weighted pseudo-examples
changes the scale and composition of the data-loss term while leaving the ridge
penalty fixed. Random pseudo-labeling versus supervised training therefore
changes both the information available to the student and the ridge penalty's
strength *relative to the data loss*—its effective regularization. That contrast
cannot isolate candidate information from the changed loss-to-penalty balance.

Full influence versus random is the clean selection test: both use the same
number and weight of pseudo-labels, the same external teacher, and the same
refitting objective. Their declared difference is which candidates are chosen.

## Manuscript notation to code

| Symbol | Meaning | Protein instantiation / code object |
| --- | --- | --- |
| `X` | Input features | ESM-2 matrices. `FitInputs.x_*` hold raw embeddings; `experiment.py` creates labeled-centered, scalar-RMS-scaled `x_l`, `x_u`, and `x_test`. |
| `Y` | Measured response | Assay DMS scores; labeled values are standardized, while hidden values live in `EvaluationLabels`. |
| `D^L` | Labeled dataset | The 96 embedding/score pairs passed to the student. |
| `D^U` | Unlabeled dataset | The 2,000 candidate embeddings and teacher predictions, without true candidate scores. |
| `f_theta` | Student prediction | Linear prediction `x @ theta` from a transformed embedding and ridge coefficients. |
| `H` | Labeled ridge curvature | `x_l.T @ x_l / n + lambda I`; scoring adds damping `rho I`. |
| `g_L` | Labeled loss gradient | Mean unregularized residual gradient at the supervised ridge fit. |
| `g_j` | Candidate pseudo-gradient | Candidate residual times its embedding, using the external pseudo-label. |
| `S_j` | Selection score | `g_L.T @ inv(H + rho I) @ (g_j - g_L)`; larger values rank first. |

The primary metric is **Spearman correlation**, comparing predicted and measured
rank order; higher is better. Standardized **mean squared error (MSE)** measures
numeric prediction error and most closely matches the theory's squared-risk
objective; lower is better. **NDCG@10%** measures how well the model ranks the
highest-fitness tenth of test variants; higher is better.

## What v0 tests

The manuscript's literal same-student squared-loss pseudo-label cannot produce a
first-round ranking at the supervised optimum:

```text
same-student pseudo-label = current student prediction
=> pseudo residual = 0
=> pseudo-gradient = 0
=> every candidate receives the same score: no ranking signal
```

With ridge regularization, the shared score is a non-positive constant; without
regularization it is zero. This degeneracy is algebraic, not a software bug. V0
therefore uses an external `ESM1v_ensemble` teacher, calibrated with the same 96
labels. Its pseudo-label usually differs from the ridge student's prediction.
V0 is an **external-teacher influence heuristic**, not literal self-teaching or
a literal test of the theorem, and not direct empirical proof or disproof of it.

Keep the claim levels separate:

1. The theorem concerns a local population-risk expansion.
2. The implementation adapts it to fixed-cardinality selection from a finite
   candidate pool using external pseudo-labels.
3. The experiment asks whether that ranking improves test prediction relative
   to random selection.

The [theory-to-experiment audit](research/theory-audit.md) records the derivation
and further boundaries, including adaptive top-score selection and the finite,
rather than asymptotic, update. The result is specific to this teacher,
representation, ridge student, label budget, and assay slice; it is not an
impossibility theorem for pseudo-label selection.

## Results and interpretation

V0 ran under the locked plan. Crossfit, locality, and exact-CV were subsequently
carded and run on outcomes already viewed in v0 or development. They diagnose
failure modes; they do not replace the confirmatory result.

| Study | Question | Result |
| --- | --- | --- |
| **v0** | Does full influence beat random under fixed labels and refitting? | No: full influence lost the primary paired comparison. |
| **Crossfit** | Is the in-sample outer gradient the problem? | A four-fold out-of-fold gradient did not improve selection. |
| **Locality** | Is the weighted 192-point update too large for the approximation? | Smaller updates improved parameter approximation, but full-Hessian and cross-fitted influence lost to random in all 15 cells. |
| **Exact-CV** | Does greedy fold-validation-loss lookahead work without Hessian or Taylor approximations? | It fit reused validation folds strongly but generalized worse than random. |

The reviewed v0 mean Spearman values were:

| Supervised | Random pseudo-labeling | Top teacher | Full influence |
| ---: | ---: | ---: | ---: |
| `0.29309` | `0.34971` | `0.34230` | `0.29445` |

Full influence minus random was **`-0.05526` Spearman**, with **0/8 assay
wins**. Random's gain over supervised is compatible with useful external-teacher
signal, but, as explained above, that contrast also changes effective
regularization. Full versus random isolates selection, and full influence lost.

Exact-CV repeatedly reused four 24-label validation folds while greedily
choosing candidates; its separate 1,000 test labels stayed sealed until choices
and predictions were fixed. Exact-CV reached Spearman **`0.30336` versus
`0.34971`** for random and MSE **`1.6177` versus `1.1857`**. Its selection
surrogate improved while test performance worsened, consistent with adaptive
validation overfitting and/or surrogate mismatch; the study does not identify a
unique cause.

At one locality setting (`q=72`, `w=0.1`), exploratory no-Hessian was
**`+0.00654` Spearman** over random, with assay sign-flip **`p=0.6172`** and
worse MSE than random in all 15 locality settings. It did not pass its gate.
Across all work, no selector established superiority to random.

Every exploratory promotion gate failed, so the **26 designated untouched assay
outcomes remain sealed**: they were never read, summarized, selected on, or
tuned against. The [overall conclusion](results/overall-conclusion.md) is the
authoritative scientific interpretation and evidence index.

## Symbol-level code tour

Follow symbols rather than reading whole modules:

| Step | Exact symbols | What to follow |
| --- | --- | --- |
| 1. Load the plan | [`Protocol`, `load_protocol`](../src/self_improve_protein/config.py) | Validate [`configs/v0.yaml`](../configs/v0.yaml). |
| 2. Build the student | [`fit_feature_transform`, `fit_label_transform`, `fit_teacher_calibration`, `fit_weighted_ridge`](../src/self_improve_protein/ridge.py) | Transform embeddings, calibrate the teacher, and fit ridge. |
| 3. Rank candidates | [`influence_scores`, `stable_top_k`](../src/self_improve_protein/selection.py) | Compute scores and break exact ties reproducibly. |
| 4. Run one task | [`FitInputs`, `fit_task`, `evaluate_task`](../src/self_improve_protein/experiment.py) | Keep outcomes hidden during fitting, refit each method, then evaluate. |

### Repository map

| Responsibility | Location |
| --- | --- |
| Data joins, eligibility, manifests, splits | [`data.py`](../src/self_improve_protein/data.py) |
| ESM-2 pooling and embedding-cache validation | [`embeddings.py`](../src/self_improve_protein/embeddings.py) |
| Metrics and assay summaries | [`metrics.py`](../src/self_improve_protein/metrics.py), [`analysis.py`](../src/self_improve_protein/analysis.py) |
| Later diagnostics | [`crossfit.py`](../src/self_improve_protein/crossfit.py), [`locality.py`](../src/self_improve_protein/locality.py), [`exact_cv.py`](../src/self_improve_protein/exact_cv.py) |
| Executable checks | [`tests/`](../tests/), [`verify_r1_r3.sh`](../scripts/verify_r1_r3.sh) |
| Plans and reviewed decisions | [`docs/research/`](research/), [`docs/results/`](results/) |

## Safe local setup

Python 3.11 or newer and [uv](https://docs.astral.sh/uv/) are required. This
orientation path creates `.venv`, but does not install Torch, require a GPU,
download study data, or submit cluster jobs:

```bash
git clone https://github.com/pengzhangzhi/self_improve_protein.git
cd self_improve_protein
uv sync --frozen
uv run self-improve-protein --show-config
uv run self-improve-protein --help
```

`uv sync --frozen` uses versions recorded in `uv.lock`; `uv run` executes inside
the project environment, so activation is unnecessary. To run the fixture-based
checks and install optional embedding support:

```bash
uv sync --frozen --extra dev --extra embed
uv run pytest -q
```

The embedding extra is substantially larger. Tests exercise algebra, data-flow,
traceability, and synthetic probes, but do not establish method quality.

## Troubleshooting and CLI discovery

| Symptom or question | Response |
| --- | --- |
| `uv: command not found` | Install uv with its [official instructions](https://docs.astral.sh/uv/getting-started/installation/), open a new shell, and rerun the sync. |
| What does data preparation accept? | Run `uv run self-improve-protein prepare-data --help`; help is read-only. |
| What does embedding require? | Run `uv run self-improve-protein embed-assay --help`; the help command is CPU-only and does not embed anything. |
| CUDA is unavailable | Orientation and tests can run on CPU. Run actual embedding only in an approved GPU environment. |
| Data archives or embeddings are absent | Expected in a fresh clone: large artifacts are excluded from Git. See pinned sources in [`configs/v0.yaml`](../configs/v0.yaml) and ask a maintainer for the approved layout. |
| A digest, manifest, or traceability check fails | Stop and confirm config, source files, revision, row hashes, runtime, and artifact root. Never bypass a confirmatory check or edit a receipt to pass it. |

Cluster reproduction is intentionally outside this conceptual guide. Slurm
submission and cancellation change shared state and consume resources; use the
[operations runbook](OPERATIONS.md) with an approved site profile and maintainer.

## Next steps

**For a theory reader:** start with the
[theory audit](research/theory-audit.md), especially the same-student degeneracy,
finite-update boundary, and external-teacher substitution. Then compare the
declared score with the exact-CV diagnostic and formulate a new experiment card
before proposing a new selector.

**For a maintainer:** use [Operations](OPERATIONS.md) for the stage contracts,
local verification, artifact layout, approved cluster workflow, and safe study
extension. Preserve comparator parity, sealed outcomes, traceability receipts,
and predeclared promotion gates.

## Glossary

| Term | Meaning here |
| --- | --- |
| Sequence | Ordered chain of amino acids, represented as letters |
| Substitution variant | Sequence with one amino acid replaced relative to a reference |
| Assay | One laboratory measurement context and its variant-score table |
| DMS | Deep mutational scanning: measuring many sequence variants in parallel |
| Fitness | Assay-specific measured performance, not a universal property |
| Embedding | Fixed numeric representation of a sequence |
| Teacher | External model supplying calibrated predictions |
| Pseudo-label | Model prediction used as a weighted training target |
| Student | Ridge regressor fitted to ESM-2 embeddings |
| Hessian | Matrix describing local objective curvature |
| Influence score | First-order estimate used to rank a candidate's effect |
| Sealed outcome | Value unavailable to fitting, selection, and tuning under the declared protocol |

## Sources of truth

1. [Locked v0 configuration](../configs/v0.yaml)
2. [Theory-to-experiment audit](research/theory-audit.md)
3. [Overall scientific conclusion](results/overall-conclusion.md)
4. Experiment cards: [v0](research/experiment-card-v0.md), [no-Hessian](research/experiment-card-no-hessian.md), [crossfit](research/experiment-card-crossfit.md), [locality](research/experiment-card-locality.md), and [exact-CV](research/experiment-card-exact-cv.md)
5. Reviewed tables: [v0 method means](../results/v0-method-means.csv) and [branch effects](../results/branch-effects.csv)
