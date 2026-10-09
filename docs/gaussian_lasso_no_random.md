---
jupytext:
  main_language: python
  cell_metadata_filter: -all
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

# The gaussian LASSO without randomization

The polyhedral lemma of {cite}`LeeLasso` gives exact selective inference after the LASSO
fitted on all the data, with no randomization: conditionally on the selected variables and
their signs, each estimate is a truncated Gaussian. Here we fit a gaussian LASSO, compute
the selective intervals with `LassoInference`, and check their calibration by simulation.
The R package has the same example, `vignette("gaussian_lasso_polyhedral")`.

The targets are the coefficients of the selected model $E$: the population least squares
coefficients of $y$ on $X_E$. The estimate is the least squares coefficient on $X_E$, and
$\sigma^2$ is estimated from the full least squares fit.

Without randomization the score used for selection is $Z = X'y$ itself, which is
`scalar_noise=0`. Internally this is replaced by a randomization of variance
$0.001\, \text{Var}(Z)$ for numerical stability; the observed randomization is zero.

```{code-cell} ipython3
import numpy as np
import pandas as pd
from scipy.stats import norm

from lassoinf import LassoInference
from lassoinf.bivariate_normal import TruncBivariateNormal

n, p = 200, 20
beta = np.r_[0.5, 0.25, -0.25, np.zeros(p - 3)]

def make_data(rng):
    X = rng.standard_normal((n, p))
    y = X @ beta + rng.standard_normal(n)
    return X - X.mean(0), y - y.mean()   # centered: no intercept
```

We solve the LASSO $\frac{1}{2}\|y - X\beta\|_2^2 + \lambda \|\beta\|_1$ by coordinate
descent, to a tight tolerance so that the KKT conditions hold closely. In glmnet's scaling,
with the loss divided by $n$, this is $\lambda / n = 0.1$.

```{code-cell} ipython3
def lasso_cd(Q, Z, lam, tol=1e-12, max_iter=10000):
    # minimize 1/2 b'Qb - b'Z + lam ||b||_1
    b = np.zeros(len(Z))
    d = np.diag(Q)
    for _ in range(max_iter):
        delta = 0.
        for j in range(len(Z)):
            r = Z[j] - Q[j] @ b + d[j] * b[j]
            new = np.sign(r) * max(abs(r) - lam, 0) / d[j]
            delta = max(delta, abs(new - b[j]) * np.sqrt(d[j]))
            b[j] = new
        if delta < tol:
            break
    return b

lam = 0.1 * n

def lasso_inference(X, y, level=0.9):
    Q, Z = X.T @ X, X.T @ y
    beta_hat = lasso_cd(Q, Z, lam)
    sigma2 = np.sum((y - X @ np.linalg.solve(Q, Z))**2) / (n - p - 1)
    return LassoInference(beta_hat=beta_hat, G_hat=Q @ beta_hat - Z, Q_hat=Q,
                          D=np.full(p, lam), L=np.full(p, -np.inf), U=np.full(p, np.inf),
                          Z_full=Z, Sigma=sigma2 * Q, scalar_noise=0, level=level)
```

## Inference

```{code-cell} ipython3
rng = np.random.default_rng(0)
X, y = make_data(rng)
LI = lasso_inference(X, y)
LI.summary_
```

`beta_hat` is the estimate of each target, the least squares coefficient on the selected
variables. Its target is $\eta' X'X\beta$, with $\eta$ the contrast:

```{code-cell} ipython3
def target(LI, j):
    return LI._contrasts[j].direction @ (LI.Q_hat @ beta)

summary = LI.summary_.copy()
summary['truth'] = [target(LI, j) for j in summary.index]
summary['cover'] = (summary['lower_conf'] <= summary['truth']) & (summary['truth'] <= summary['upper_conf'])
summary
```

## Calibration

The pivot of each target, the conditional CDF of its estimate at the true value, should be
uniform. We compute it from the contrast, its truncation interval and the truncated
bivariate normal, as `LassoInference` does. The naive intervals and pivot ignore the
selection.

```{code-cell} ipython3
def pivot(LI, j, truth):
    c = LI._contrasts[j]
    L0, U0 = c.get_interval(0., LI.A, LI.b)
    variance = float(c.naive_variance)
    bar_s = float(c.bar_s)
    tbn = TruncBivariateNormal(a_coeff=bar_s**2 / variance, b_coeff=1., L=L0, U=U0,
                               sig_omega=bar_s, sig_x=np.sqrt(variance))
    return tbn.cdf(truth / variance, float(c.theta_hat))

def one_replication(rng):
    X, y = make_data(rng)
    LI = lasso_inference(X, y)
    q = norm.ppf(0.95)
    rows = []
    for j, s in LI.summary_.iterrows():
        truth = target(LI, j)
        sd = np.sqrt(float(LI._contrasts[j].naive_variance))
        rows.append(dict(signal=beta[j] != 0,
                         pivot=pivot(LI, j, truth),
                         cover=s.lower_conf <= truth <= s.upper_conf,
                         length=s.upper_conf - s.lower_conf,
                         naive_pivot=norm.cdf((s.beta_hat - truth) / sd),
                         naive_cover=abs(s.beta_hat - truth) <= q * sd,
                         naive_length=2 * q * sd))
    return rows

rng = np.random.default_rng(1)
results = pd.DataFrame([row for _ in range(300) for row in one_replication(rng)])
results.groupby('signal')[['cover', 'naive_cover', 'length', 'naive_length']].agg(
    {'cover': 'mean', 'naive_cover': 'mean', 'length': 'median', 'naive_length': 'median'}).assign(
    variables=results.groupby('signal').size())
```

```{code-cell} ipython3
:tags: [hide-input]

import matplotlib.pyplot as plt

ink, ink2, grid = '#0b0b0b', '#52514e', '#e5e4e0'
pal = ['#2a78d6', '#eb6834']

fig, ax = plt.subplots(figsize=(6.5, 4.2))
xs = np.linspace(0, 1, 401)
for (col, label), color in zip([('pivot', 'polyhedral (LassoInference)'),
                                ('naive_pivot', 'naive (ignores selection)')], pal):
    v = np.sort(results[col].values)
    ax.plot(xs, np.searchsorted(v, xs, side='right') / len(v), color=color, lw=2, label=label)
ax.plot([0, 1], [0, 1], ls='--', lw=1, color=ink2)
ax.set(xlim=(0, 1), ylim=(0, 1), xlabel='Uniform quantile', ylabel='Proportion of pivots below')
ax.set_title('Pivots at the selected-model targets', color=ink, fontsize=11)
ax.grid(color=grid, lw=1)
ax.set_axisbelow(True)
for side in ['top', 'right']:
    ax.spines[side].set_visible(False)
ax.tick_params(colors=ink2)
ax.legend(frameon=False, loc='upper left', labelcolor=ink)
plt.show()
```

The polyhedral pivots follow the diagonal, and the intervals cover close to the nominal 90%
for signal and null variables alike. The naive intervals cover the selected null variables
far less often than 90%: those were selected because their estimates are large. The
polyhedral intervals are about as long as the naive ones for strong signals, and much longer
for variables selected near the selection boundary. Randomization gives shorter intervals
there; see [](gaussian_lasso.md).
