# Locality Screen Decision

## CONFIRMED

No confirmatory scientific claim is made. The predeclared exploratory screen
completed all 45 GFP-plus-v0 tasks under commit
`f1f9c4cd4b5caa8cc29110f8085defb6925b3131`. Its official aggregate contains
the exact 45-task grid, 40-task primary slice, 2,400 primary result rows, and 60
selector-by-perturbation cells. The aggregate is
`artifacts/exploratory/locality_v1/screen/aggregate.json`, SHA-256
`917ac35d9ef2e9f885c1b774cb590e231056e35e97d43a579e0f362372ec97cf`.

## EXPLORATORY

First-order displacement fidelity improved monotonically as pseudo weight
decreased at every fixed `q` for random, full, cross-fit, and no-Hessian
selection. At the smallest cell (`q=24`, `w=0.01`, `t=0.00249`), mean
displacement cosine was `0.8972` for full and `0.8940` for cross-fit, versus
`0.5843` and `0.5854` at the v0-like cell (`q=192`, `w=0.1`). Their relative
displacement errors fell from `21.35`/`19.34` to `3.25`/`2.72`.

Useful ranking did not emerge. At the smallest cell, full minus random was
`-0.01067` Spearman (SE `0.00172`, 0/8 assay wins, exact sign-flip
`p=0.0078125`) and cross-fit minus random was `-0.00787` (SE `0.00294`, 1/8
assay wins, `p=0.0390625`). Both had worse MSE and NDCG@10% than random. Their
Spearman deficit grew with perturbation size, and neither beat random in any of
the 15 cells.

No-Hessian had one descriptive positive Spearman cell (`q=72`, `w=0.1`:
`+0.00654`, SE `0.01281`, 5/8 assay wins, `p=0.6172`), but its MSE was worse
than random in every assay-macro cell. All three learned score orderings were
negatively aligned with the hidden test-risk oracle: full `-0.3404`, cross-fit
`-0.3719`, and no-Hessian `-0.3270`.

## FAILED / INCOMPLETE

The local-perturbation explanation is insufficient. Making the finite update
smaller improves approximation fidelity but does not make the paper-derived
ranking useful. Even the smallest full/cross-fit cell is not quantitatively
very local: mean locality indices are `7.95` and `7.65`, and relative errors
remain greater than two.

## HIGHEST VERIFIED RUNG

R5: exposed-assay exploratory mechanism screen. No untouched-assay outcome was
accessed and no confirmatory replication was run.

## EVIDENCE GAPS

The grid cannot rule out still smaller perturbations, but the hidden-oracle
anti-alignment identifies the outer-risk proxy as a more fundamental failure.
The no-Hessian ranking blip does not satisfy the squared-risk mechanism.

## RECOMMENDED NEXT

Replace both approximations with the preregistered exact four-fold
validation-risk greedy selector. If exact held-out MSE lookahead cannot beat
random with the same teacher and final objective, stop pursuing further
influence/locality geometry repairs in this setup.
