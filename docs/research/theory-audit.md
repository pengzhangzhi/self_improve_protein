# Theory-to-Experiment Audit

This note records the claim boundary found by an independent audit of the
supplied manuscript. It separates the score heuristic tested in this repository
from what the current theorem actually establishes.

## Critical gaps

### Squared-loss self-teaching is degenerate at the first step

At the exact penalized supervised optimum,

\[
g_L=\nabla L_n(\hat\theta^{(0)})
=-\lambda\nabla\Omega(\hat\theta^{(0)}).
\]

For the manuscript's literal self-label
\(\hat y_j=f_{\hat\theta^{(0)}}(x_j)\), squared loss gives \(g_j=0\). Hence

\[
S_j=g_L^\top H^{-1}(g_j-g_L)
=-\lVert H^{-1/2}g_L\rVert^2\le 0
\]

for every candidate. With no regularization all scores are zero; otherwise all
are the same non-positive constant. The positive-score algorithm therefore
stops before selecting a first batch.

The ProteinGym experiment deliberately uses a calibrated external teacher to
avoid this identity. It tests an external-teacher score heuristic, not the
manuscript's literal self-teacher algorithm. A theorem for this version needs a
teacher risk \(P_T(\theta)=E[\ell(\theta;X,T(X))]\) and must account for the
label-fitted calibration parameters.

### The proof does not cover score-selected samples

The empirical proof centers selected pseudo terms by their unconditional
population expectation and then applies independent-sample concentration. A
top-score index is an order statistic: its distribution is not the original
covariate distribution, selected points are dependent, and in general
\(E[h(X_J)]\ne E[h(X)]\). The proposed population objective is therefore not
the population analogue of the selected empirical objective.

A selection theorem also needs the unlabeled pool size, a random selection
threshold, score-margin or ranking-stability assumptions, and treatment of the
data-dependent stopping time. None currently appears. As written, the rate
argument is closest to non-adaptive or random pseudo-batches.

## Important interpretation limits

### The empirical left gradient is coupled to the training root

The score substitutes the in-sample gradient at the same penalized fit for
\(\nabla R\). For ridge it is exactly \(-\lambda\hat\theta\), and it vanishes
without regularization. Cross-fitting in this repository targets this specific
failure while retaining the objective perturbation term:

\[
S_j^{\mathrm{CF}}
=g_{\mathrm{eval}}^\top H_{\mathrm{fit}}^{-1}
  (g_j-g_{\mathrm{fit}}).
\]

The two gradient roles must remain distinct; replacing both occurrences by a
validation gradient would not be the derivative of the fitted objective.

The preregistered cross-fit repair intentionally keeps the 96-label
standardization and teacher calibration fixed to preserve all other v0
coordinates. Consequently, each held-out fold still affects its own global
label transform and calibrated teacher. It is a single-change diagnostic, not
a fully outcome-held-out risk estimator.

### The advertised asymptotic window is insufficient

Let \(q=mk\) and, for small perturbations, \(t\asymp wq/n\). For the displayed
\(O(t)\) term to dominate both \(n^{-1/2}\) and \(q^{-1/2}\), one needs

\[
wq\gg\sqrt n,\qquad wq^{3/2}\gg n,\qquad wq\ll n.
\]

For fixed \(w\), the second lower bound is \(q\gg n^{2/3}\), stronger than
\(q\gg\sqrt n\). The v0 choice \(q=192=2n\), \(w=0.1\), \(t=1/6\) violates
\(q\ll n\) and does not define a \(t\to0\) sequence. It is a finite heuristic
test, not an asymptotic verification. The conditional locality card probes this
gap explicitly.

### Normalization changes effective regularization

The path \(M_t=(1-t)L+tP+\lambda\Omega\) downweights labeled loss while leaving
the penalty fixed. Under a literal self-teacher, \(\nabla P=0\) at the raw fit,
so first-order movement is driven by this changed loss-to-penalty ratio and the
population expansion predicts non-improvement. Fixed-\(q,w\) comparisons among
pseudo-selection methods remain fair, but pseudo versus supervised comparisons
partly mix selection with effective regularization.

### Positive-score and sequential claims are narrower than implemented

Fixed-cardinality evaluation tests score ranking. It matches the positive-score
gate only on tasks with at least \(q\) positive scores. The manuscript also
defines later pseudo-labels using the current student while its displayed score
is anchored to the raw fit and Hessian; a genuine iterative derivative should
use the current accumulated objective and state its treatment of frozen labels.

The cross-fit diagnostic named predicted outer-loss change uses the cross-fit
gradient, whereas its recorded realized labeled-loss change is evaluated on the
in-sample labeled objective. These are different outer functionals and should
not be interpreted as a direct sign-agreement test.

Finally, the theorem concerns squared population risk while the experiment's
primary endpoint is Spearman rank correlation. Test MSE is the closer empirical
analogue; Spearman tests the protein-ranking application rather than the
displayed risk expansion.

## Proof details requiring repair

- Bounds established on a local compact set are used outside that set.
- Membership in a larger neighborhood is used to assert membership in a
  smaller one, while an assumption already places the targets there.
- Pointwise smoothness does not justify differentiation through expectations
  without domination/integrability conditions.
- Fixed-dimensional concentration does not transfer directly to the actual
  `d=480`, `n=96` experiment.

## Sign convention

For a fixed pseudo perturbation,

\[
\theta'(0)=-H^{-1}(g_j-g_L),\qquad
R'(0)=-\nabla R^\top H^{-1}(g_j-g_L).
\]

Therefore, if the left gradient consistently estimated population risk,
positive scores would predict reduced risk and descending by the largest score
has the correct sign. The central problems are self-teacher degeneracy,
training-root coupling, adaptive selection, and finite-step locality—not a
missing minus sign.
