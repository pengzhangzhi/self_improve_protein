# self-improve-protein

This repository implements a predeclared study of influence-ranked,
external-teacher pseudo-label selection for low-label protein fitness
regression. The study uses the substitution assays from
[ProteinGym v1.3](https://zenodo.org/records/15293562) and frozen
[`facebook/esm2_t12_35M_UR50D`](https://huggingface.co/facebook/esm2_t12_35M_UR50D)
representations. The literal protocol is tracked in `configs/v0.yaml`.

## Claim boundary

The confirmatory question is deliberately narrow: when every pseudo-labeling
method receives the same calibrated external `ESM1v_ensemble` teacher scores,
does influence ranking select a fixed-size pseudo-labeled subset that improves
held-out ProteinGym ranking relative to random selection? A positive result
would support that empirical external-teacher selection rule only. It would not
validate a literal self-teacher algorithm, establish the paper's asymptotic
proof, or show that pseudo-labeling generally improves protein models.

For squared-loss ridge, a literal self-teacher is degenerate. At the supervised
optimum, `g_L = -lambda * theta`, while assigning each candidate the student's
own prediction makes every candidate pseudo-gradient zero. The resulting score
is the same non-positive constant for every candidate, so it supplies no
ranking signal. The project retains this identity as a negative control rather
than presenting self-teaching as the confirmatory method.

## Current evidence

The locked v0 result is negative for the proposed selector. Across eight assays
and five seeds, full-Hessian influence selection was `-0.05526` Spearman below
random pseudo-label selection, with 0/8 assay wins. Cross-fitted outer gradients
and much smaller pseudo perturbations did not repair the ranking.

The strongest diagnostic replaced the influence approximation entirely with
exact four-fold validation-risk greedy selection. It still underperformed
random by `0.43205` MSE and `0.04635` Spearman on the eight-assay exposed
screen. Its validation loss improved sharply while hidden-test performance
worsened, identifying adaptive validation overfitting / surrogate mismatch as
a deeper failure than Hessian geometry alone. The preregistered gate failed,
so 26 untouched assay outcomes remain sealed.

Detailed, hash-bound decisions are tracked in:

- [`docs/results/v0-decision.md`](docs/results/v0-decision.md)
- [`docs/results/crossfit-decision.md`](docs/results/crossfit-decision.md)
- [`docs/results/locality-decision.md`](docs/results/locality-decision.md)
- [`docs/results/exact-cv-decision.md`](docs/results/exact-cv-decision.md)

These findings apply to the frozen ProteinGym/ESM1v/ESM-2-ridge setup. They do
not establish that pseudo-label selection is impossible in other protein
fitness regimes. The theory-to-experiment limitations are recorded in
[`docs/research/theory-audit.md`](docs/research/theory-audit.md).

## Sources

- [ProteinGym](https://proteingym.org/)
- [Pinned ProteinGym v1.3 data record](https://zenodo.org/records/15293562)
- [ESM-2 35M model card](https://huggingface.co/facebook/esm2_t12_35M_UR50D)

## Development

Python 3.11 or newer is required. Reproduce the executable development
environment with `uv sync --frozen --extra dev`, then run `uv run pytest`,
`uv run ruff check .`, and `uv run mypy src`. The tracked lockfile pins NumPy
2.3.5 to freeze the `PCG64`/`Generator.choice` behavior used by the random
selection baseline. As a local fallback, install with `pip install -e '.[dev]'`.

The MIT license covers this repository's code. ProteinGym data and ESM model
artifacts remain governed by their upstream terms and are not redistributed
here.
