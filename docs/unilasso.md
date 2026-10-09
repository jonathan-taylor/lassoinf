---
jupytext:
  formats: md:myst,ipynb
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.1
kernelspec:
  name: python3
  display_name: Python 3 (ipykernel)
  language: python
---

# Inference after the uniLasso

The uniLasso {cite}`UniLasso` fits a univariate regression of $y$ on each feature, then a
lasso whose penalty factors and signs come from those fits:

$$
\min_{\beta_0, \beta} \ \frac{1}{2n}\|y - \beta_0 - X\beta\|_2^2 + \lambda \sum_j \frac{|\beta_j|}{|\hat{\beta}^{\text{uni}}_j|}
\quad \text{subject to} \quad \text{sign}(\beta_j) \in \{0, \text{sign}(\hat{\beta}^{\text{uni}}_j)\},
$$

where $\hat{\beta}^{\text{uni}}_j$ is the slope of the univariate regression of $y$ on $x_j$.
This is what the R package `uniLasso` fits with `loo = FALSE`. With $\lambda = 0$ it is
**unireg**: least squares with the sign constraints.

Both the penalty factors and the sign constraints depend on the data, so the selection event
is not a polyhedron and the polyhedral lemma does not apply directly.
`unilasso_inference(X, y, beta_hat, lam)` takes a uniLasso fit and gives selective intervals
and $p$-values for the **full-model** coefficient $\beta_k$ of each selected variable, in a
Gaussian linear model with $n > p$. The estimate is the $k$-th coefficient of the least
squares fit on all $p$ features.

For these targets, only the target's own univariate coefficient
$\hat{\beta}^{\text{uni}}_k$ varies along the line the lemma conditions on. Each constraint
then has the form $(AY)_i \leq B_i / C_i$ with $C_i = \hat{\beta}^{\text{uni}}_k$, and the
truncation set is found exactly, as a finite union of intervals; see
[](data_dependent_penalty.md). We condition on the selected variables and on the signs of
all the univariate coefficients. There is no randomization.

The fit must solve the problem above. The `uniLasso` package's default `loo = TRUE`
regresses $y$ on leave-one-out univariate fits instead. That is a different selection
event, not covered here; `unilasso_inference` warns when `beta_hat` does not satisfy the
KKT conditions above.

```{code-cell} ipython3
import numpy as np
from lassoinf import unilasso_inference, unilasso_fit

rng = np.random.default_rng(0)
n, p = 200, 8
X = rng.standard_normal((n, p)) @ np.linalg.cholesky(0.5 * np.eye(p) + 0.5).T
beta = np.r_[0.3, -0.2, 0.15, np.zeros(p - 3)]
y = X @ beta + rng.standard_normal(n)
```

## uniLasso

Here `unilasso_fit` stands in for the R package. It solves the problem in score
coordinates, $\frac{1}{2}\beta' Q \beta - \beta' Z + n\lambda \sum_j |\beta_j| / |C_j|$, with
centered $X$ and $y$, $Q = X'X$, $Z = X'y$ and $C = Z / \text{diag}(Q)$.

```{code-cell} ipython3
lam = 0.02
Xc, yc = X - X.mean(0), y - y.mean()
beta_hat = unilasso_fit(Xc.T @ Xc, Xc.T @ yc, n * lam)
UI = unilasso_inference(X, y, beta_hat, lam, level=0.9)
UI.summary_
```

`beta_hat` in the summary is the full-model least squares coefficient. The noise variance
is estimated from the full least squares fit. The truncation set for each target is
available as an array of intervals:

```{code-cell} ipython3
{k: UI._results[k].intervals for k in UI.E}
```

## Unireg

```{code-cell} ipython3
UR = unilasso_inference(X, y, unilasso_fit(Xc.T @ Xc, Xc.T @ yc, 0.), 0., level=0.9)
UR.summary_
```

## Randomization

`UniLassoInference.from_selection` takes a uniLasso solution for randomized selection data
$X'y + \omega$, with the randomization's covariance `scalar_noise * Sigma`:

```{code-cell} ipython3
from lassoinf import UniLassoInference

Q, Z = Xc.T @ Xc, Xc.T @ yc
sigma2 = np.sum((yc - Xc @ np.linalg.solve(Q, Z))**2) / (n - p - 1)
Z_noisy = Z + np.linalg.cholesky(sigma2 * Q) @ rng.standard_normal(p)
UniLassoInference.from_selection(unilasso_fit(Q, Z_noisy, n * lam), Z_noisy, Q, n * lam, Z,
                                 sigma2 * Q, scalar_noise=1., level=0.9).summary_
```

## Calibration

The pivots at the true coefficients are uniform. The tests in `tests/test_unilasso.py`
check this by simulation for the uniLasso and unireg, with and without randomization.
