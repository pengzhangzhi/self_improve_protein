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

This repository currently contains the locked protocol and software
foundations. It makes no experimental-results claim.

## Sources

- [ProteinGym](https://proteingym.org/)
- [Pinned ProteinGym v1.3 data record](https://zenodo.org/records/15293562)
- [ESM-2 35M model card](https://huggingface.co/facebook/esm2_t12_35M_UR50D)

## Development

Python 3.11 or newer is required. Install development dependencies with
`pip install -e '.[dev]'`, then run `pytest`, `ruff check .`, and `mypy src`.

The MIT license covers this repository's code. ProteinGym data and ESM model
artifacts remain governed by their upstream terms and are not redistributed
here.
