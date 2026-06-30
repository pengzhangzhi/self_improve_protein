# V0 Decision: External-Teacher Influence Selection

## CONFIRMED

The locked 8-assay by 5-seed study completed all 40 tasks under commit
`1b632a3b6bc9cee9cea3d8422ecc4e4d6c55a81f`. Random pseudo-labeling improved
mean Spearman over supervised-only ridge (`0.3497` versus `0.2931`), so the
external teacher contained usable signal. The proposed full-Hessian selector
did not extract that signal: its mean Spearman was `0.2944`.

The primary paired contrast was decisively negative. Full-Hessian selection
minus random selection was `-0.05526` Spearman (assay-cluster SE `0.00865`,
hierarchical 95% interval `[-0.07616, -0.03513]`, exact assay sign-flip
`p=0.0078125`). It won 5/40 tasks and 0/8 assay means. The predeclared v0
selection-success criterion therefore failed.

The official aggregate is
`artifacts/studies/v0/confirmatory/aggregate.json`, SHA-256
`d9426a88554f6e84d9a2b6995a4aa319f9c756570f8df4b22240136e885fabe3`.

## EXPLORATORY

No-Hessian selection was closer to random in Spearman (`0.3372`) and had the
best mean NDCG@10% (`0.6896`), but its mean MSE was `1.7235`, much worse than
random's `1.1857`. Top-teacher selection reached `0.3423` Spearman. These
secondary results motivate mechanism diagnosis; they do not reverse the
primary failure.

## FAILED / INCOMPLETE

The data reject the claim that the supplied full-Hessian score improves
low-label ProteinGym ranking over random pseudo-label selection in this frozen
setting. They do not test the manuscript's literal squared-loss self-teacher,
which is analytically degenerate at the first step, and they do not establish a
general impossibility result for pseudo-label selection.

## HIGHEST VERIFIED RUNG

R7: the complete predeclared v0 study and result review are verified. The
scientific decision is a negative result for the external-teacher influence
ranking rule.

## EVIDENCE GAPS

The experiment uses eight assays, one teacher, one embedding, one label budget,
and a high-dimensional ridge student. It does not validate the manuscript's
adaptive-selection asymptotics, and its primary Spearman endpoint is not the
squared population risk in the theorem.

## RECOMMENDED NEXT

Do not tune v0 post hoc or launch the same selector on more assays. Diagnose the
outer-risk proxy and finite-step approximation under separately frozen,
development-only experiment cards.
