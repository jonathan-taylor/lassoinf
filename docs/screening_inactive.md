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

# Screening inactive variables

After the LASSO selects an active set $E$, it is natural to ask about the variables it left
out. A common heuristic looks at the KKT "near misses": the inactive variables whose
gradient at the LASSO solution is largest in absolute value. Here we keep the **top 10**
inactive variables by $|\nabla \ell(\hat{\beta})_j|$ and, for each, test its coefficient in the
model $E \cup \{j\}$.

Both the LASSO and the screening step look at the data, so naive tests of the screened
variables are invalid. `ScreenedSelection` adds the screening step to the selection event,
and the polyhedral lemma then gives valid intervals and p-values.

## The selection event in two coordinates

With $Q = X'X$ and $W = Q_{E,E}^{-1}$, the LASSO selection event depends on the data
only through

- $\bar{\beta}_E = W Z_E = (X_E'X_E)^{-1}X_E'y$, via the active sign constraints;
- the inactive scores $U_{-E} = Z_{-E} - Q_{-E,E} W Z_E = X_{-E}'(I - P_E)y$, via the
  subgradient constraints $|\nabla \ell(\hat{\beta})_j| \leq \lambda$.

On the selection event the inactive gradient at the LASSO solution is
$\nabla \ell(\hat{\beta})_{-E} = -U_{-E}(Z + \omega) + \text{const}$, an affine function of the same
coordinates, where $Z + \omega$ is the randomized score used for selection. Keeping the top $K$
variables by $|\nabla \ell(\hat{\beta})_j|$, with the signs of their gradients, is then also a
polyhedral event:

- **`conditioning='first_dropped'`** (default): also condition on the largest dropped variable
  $l^*$ and its sign $s^*$. This gives the rows $s_k G_k \geq s^* G_{l^*}$ for kept $k$ and
  $|G_l| \leq s^* G_{l^*}$ for the other dropped $l$, which is $O(p)$ rows.
- **`conditioning='exact'`**: the pairwise rows $s_k G_k \geq |G_l|$ for every kept $k$ and dropped $l$,
  which is $2K(p - |E| - K)$ rows. It conditions on less, at the price of more constraints.

The coefficient of inactive variable $j$ in the model $E \cup \{j\}$ is
$U_j / S_{jj}$ with $S_{jj} = Q_{jj} - Q_{j,E} W Q_{E,j}$: a contrast $\eta'Z$ in the same
coordinates. All of this is computed with matrix-vector products by $Q$, so $X'X$ is never formed.

## Data: a wide design

We take $n = 100$ observations and $p = 300$ features, with 15 nonzero coefficients of moderate
size, and treat $\sigma^2 = 1$ as known. Selection uses a randomized response
$y_{noisy} = y + \epsilon$ with $\epsilon \sim N(0, \frac{1-\pi}{\pi}\sigma^2 I)$ and $\pi = 0.8$, as in the
parametric bootstrap example.

```{code-cell} ipython3
import numpy as np
import pandas as pd
import cvxpy as cp
import scipy.sparse as sp
from scipy.sparse.linalg import aslinearoperator
from scipy.stats import norm

from lassoinf import LassoInference
from lassoinf.operators import XTVXOperator
from lassoinf.custom_estimand import (ScreenedSelection, SelectionCoordinates,
                                      contrast_inference, inactive_summary)
```

```{code-cell} ipython3
n, p, k = 100, 300, 15
pi_selection = 0.8
scalar_noise = (1 - pi_selection) / pi_selection   # Var(omega) = scalar_noise * Sigma

def make_data(rng):
    X = rng.standard_normal((n, p))
    beta = np.zeros(p)
    beta[:k] = 3 / np.sqrt(n) * rng.choice([-1, 1], k)
    mu = X @ beta
    y = mu + rng.standard_normal(n)
    y_noisy = y + rng.normal(0, np.sqrt(scalar_noise), n)
    return X, beta, mu, y, y_noisy

rng = np.random.default_rng(0)
X, beta, mu, y, y_noisy = make_data(rng)
```

## Fit the LASSO on the randomized data

```{code-cell} ipython3
lam = 0.8 * np.sqrt(2 * n * np.log(p))

def fit_lasso(X, y_noisy):
    b = cp.Variable(p)
    objective = 0.5 * cp.sum_squares(y_noisy - X @ b) + lam * cp.norm1(b)
    cp.Problem(cp.Minimize(objective)).solve(solver=cp.CLARABEL)
    beta_hat = np.where(np.abs(b.value) > 1e-6, b.value, 0.)
    G_hat = X.T @ (X @ beta_hat - y_noisy)
    return beta_hat, G_hat

def lasso_inference(X, y, y_noisy):
    beta_hat, G_hat = fit_lasso(X, y_noisy)
    # Q = X'X and Sigma = sigma^2 X'X as matrix-free operators
    Q = XTVXOperator(X, aslinearoperator(sp.identity(X.shape[0])))
    return LassoInference(beta_hat=beta_hat, G_hat=G_hat, Q_hat=Q, D=np.full(p, lam),
                          L=None, U=None, Z_full=X.T @ y, Sigma=Q,
                          scalar_noise=scalar_noise, level=0.9)

LI = lasso_inference(X, y, y_noisy)
print(f'{len(LI.E)} active variables, {np.sum(beta[LI.E] != 0)} of them true signals; '
      f'{np.sum(beta[LI.E_c] != 0)} true signals were left out')
```

## Screen the top 10 inactive variables

```{code-cell} ipython3
screen = ScreenedSelection(LI, top_k=10)
print('screened:', screen.screened)
print('gradient signs:', screen.screened_signs)
print('first dropped:', screen.first_dropped, 'with sign', screen.first_dropped_sign)
print('constraint rows: LASSO', LI.A.shape[0], '+ screening', screen.A_screen.shape[0])
```

`inactive_summary` gives the coefficient of each screened variable in the model $E \cup \{j\}$,
conditional on both selection steps. Its target is the population coefficient
$\eta_j' X'\mu$, the coefficient of $x_j$ in the regression of $\mu = E[y]$ on $X_{E \cup \{j\}}$.

```{code-cell} ipython3
coords = SelectionCoordinates(LI)
truth = {j: coords.inactive_coef(j).value(X.T @ mu) for j in screen.screened}

screened_df = inactive_summary(screen)
screened_df['truth'] = pd.Series(truth)
screened_df['signal'] = beta[screened_df.index] != 0
screened_df
```

### Ignoring the screening step

Using the LASSO constraints alone treats the 10 variables as if they had been chosen in advance,
and naive intervals ignore selection altogether. Both are shown for the same variables:

```{code-cell} ipython3
def naive_summary(LI, variables, level=0.9):
    q = norm.ppf(1 - (1 - level) / 2)
    rows = []
    for j in variables:
        est = coords.inactive_coef(j)
        contrast = LI.si.compute_contrast(est.eta)
        sd = np.sqrt(contrast.naive_variance)
        theta = est.value(LI.Z_full)
        rows.append({'index': j, 'estimate': theta, 'lower_conf': theta - q * sd,
                     'upper_conf': theta + q * sd, 'p_value': 2 * norm.sf(abs(theta) / sd)})
    return pd.DataFrame(rows).set_index('index')

compare = pd.concat({'screened': inactive_summary(screen),
                     'lasso only': inactive_summary(LI, variables=screen.screened),
                     'naive': naive_summary(LI, screen.screened)}, axis=1)
compare.xs('p_value', axis=1, level=1).assign(signal=beta[screen.screened] != 0)
```

Ignoring the screening step generally gives smaller p-values, since variables are kept
*because* their gradients are large. The repeated experiments below show that those
smaller p-values are not valid.

### The LASSO-solution basis

The same estimands can be centered at the LASSO solution instead of the refit: with
`basis='lasso'` the coefficient is the one-step estimate $-\nabla \ell(\hat{\beta})_j / S_{jj}$. It uses the
same contrast, so the intervals are shifted copies of the refit intervals, and the
estimand is shifted by the same constant. The p-values change because $\theta = 0$ is now
a null hypothesis about the shifted estimand.

```{code-cell} ipython3
inactive_summary(screen, basis='lasso').head()
```

### First-dropped versus exact conditioning

```{code-cell} ipython3
exact = ScreenedSelection(LI, top_k=10, conditioning='exact')
print('constraint rows: first_dropped', screen.A_screen.shape[0], ' exact', exact.A_screen.shape[0])
pd.concat({'first_dropped': inactive_summary(screen)[['lower_conf', 'upper_conf', 'p_value']],
           'exact': inactive_summary(exact)[['lower_conf', 'upper_conf', 'p_value']]}, axis=1)
```

Conditioning on the first-dropped variable restricts the selection event further, so its
truncation intervals are subsets of the exact ones, and often the same.

## Coverage over repeated experiments

We repeat the experiment and record, for each screened variable, whether the 90% interval covers
its target. With 50 replications that is 500 intervals per method.

```{code-cell} ipython3
def one_replication(rng):
    X, beta, mu, y, y_noisy = make_data(rng)
    LI = lasso_inference(X, y, y_noisy)
    screen = ScreenedSelection(LI, top_k=10)
    coords = SelectionCoordinates(LI)
    q = norm.ppf(0.95)
    rows = []
    for j in screen.screened:
        est = coords.inactive_coef(j)
        target = est.value(X.T @ mu)
        sd = np.sqrt(LI.si.compute_contrast(est.eta).naive_variance)
        theta = est.value(LI.Z_full)
        for method, sel in [('screened', screen), ('lasso only', LI)]:
            res = contrast_inference(sel, est)
            rows.append({'method': method, 'cover': res.lower_conf <= target <= res.upper_conf,
                         'length': res.upper_conf - res.lower_conf, 'reject': res.p_value < 0.1,
                         'signal': beta[j] != 0})
        rows.append({'method': 'naive', 'cover': abs(theta - target) <= q * sd, 'length': 2 * q * sd,
                     'reject': abs(theta) > q * sd, 'signal': beta[j] != 0})
    return rows

rng = np.random.default_rng(1)
results = pd.DataFrame([row for _ in range(50) for row in one_replication(rng)])
results.groupby('method').agg(coverage=('cover', 'mean'), median_length=('length', 'median'))
```

Only the intervals that condition on the screening step come close to the nominal 90% coverage.
Rejection rates at level 0.1, split by whether the screened variable is a true signal:

```{code-cell} ipython3
results.pivot_table(index='method', columns='signal', values='reject', aggfunc='mean')
```

Among screened null variables, the screened tests reject at about the nominal 10%, while the naive
and LASSO-only tests reject far more often. The screened tests still find a share of the true
signals the LASSO missed.
