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

# Surrogate screening: swapping selected and screened variables

In [](screening_inactive.md) we kept the **top 10** inactive variables by
$|\nabla \ell(\hat{\beta})_j|$ and tested each one in the model $E \cup \{j\}$. Here we screen
the same way but ask a different question. A screened variable $j$ may be a *surrogate* for a
selected variable $s$: the two are correlated and the LASSO picked $s$ where it could have
picked $j$. To check this, for each selected $s \in E$ and screened $j$ we drop $s$ from the model, put
$j$ in its place, and test

$$
H_0: \beta_{j \cdot F_{s,j}} = 0, \qquad F_{s,j} = (E \setminus \{s\}) \cup \{j\},
$$

the coefficient of $x_j$ in the model $F_{s,j}$, or build a confidence interval for it.

Only the **target** changes. The selection event (LASSO, then screening) is the same as
before, so we reuse `ScreenedSelection` as it is and pass it a new contrast.

## The swapped coefficient as a contrast

As before, $Q = X'X$ and $W = Q_{E,E}^{-1}$. The coefficient of $j$ in $F = F_{s,j}$ is

$$
\hat{\theta}_{s,j} = e_j'\, Q_{F,F}^{-1} Z_F = \eta_{s,j}' Z, \qquad
(\eta_{s,j})_F = Q_{F,F}^{-1} e_j, \quad (\eta_{s,j})_{-F} = 0,
$$

which for least squares is the coefficient of $x_j$ in the regression of $y$ on $X_F$. The
estimand is $\eta_{s,j}' X'\mu$, the coefficient of $x_j$ in the regression of
$\mu = E[y]$ on $X_F$.

The matrix $Q_{F,F}$ is built from the information matrix of the model $E$. Its block on
$E \setminus \{s\}$ is a submatrix of $Q_{E,E} = W^{-1}$, which we already have. The rest is
$Q_{j,E \setminus \{s\}}$ and $Q_{jj}$, which come from one matvec $Q e_j$ per screened
variable, shared by all $s$. For a GLM, $Q$ is the `Q_hat` used for selection. We do
not refit the model $F$ or recompute its information matrix.

Equivalently, with $G = E \cup \{j\}$ and $M = Q_{G,G}^{-1}$, dropping $s$ from $G$ gives

$$
\hat{\theta}_{s,j} = \hat{\beta}_{j \cdot G} - \frac{M_{js}}{M_{ss}}\, \hat{\beta}_{s \cdot G}.
$$

This is the screened coefficient from [](screening_inactive.md) minus a multiple of the
coefficient of $s$ in the same model. Because $x_j$ and $x_s$ are correlated, part of the
effect of $s$ moves onto $j$. The contrast is linear in $Z$, so the polyhedral lemma applies
with the screening constraints we already have.

## Setup

We use two designs, each with $n = 100$ observations, $p = 300$ features and 15 nonzero
coefficients on every 20th feature (0, 20, ..., 280), and $\sigma^2 = 1$ treated as known:

1. **Independent features** with signals of size $3/\sqrt{n}$, the setting of
   [](screening_inactive.md). Here, ignoring the screening step badly breaks inference.
2. **AR(1) features**, with $\text{corr}(x_i, x_l) = \rho^{|i - l|}$, $\rho = 0.8$, and signals
   of size $5/\sqrt{n}$. Each signal's neighbors are correlated null features that can stand in
   for it, so this design produces surrogates.

Selection uses a randomized response $y_{noisy} = y + \epsilon$ with
$\epsilon \sim N(0, \frac{1-\pi}{\pi}\sigma^2 I)$ and $\pi = 0.8$.

```{code-cell} ipython3
import numpy as np
import pandas as pd
import cvxpy as cp
import scipy.sparse as sp
from scipy.sparse.linalg import aslinearoperator
from scipy.stats import norm

from lassoinf import LassoInference
from lassoinf.operators import XTVXOperator
from lassoinf.custom_estimand import (ScreenedSelection, ContrastEstimand,
                                      contrast_inference, estimand_summary)
```

```{code-cell} ipython3
n, p, k = 100, 300, 15
pi_selection = 0.8
scalar_noise = (1 - pi_selection) / pi_selection   # Var(omega) = scalar_noise * Sigma
signals = np.arange(0, p, p // k)

def make_data(rng, rho, amp):
    # AR(1) columns: corr(x_i, x_l) = rho^|i - l|; rho = 0 gives independent columns
    X = rng.standard_normal((n, p))
    for i in range(1, p):
        X[:, i] = rho * X[:, i - 1] + np.sqrt(1 - rho**2) * X[:, i]
    beta = np.zeros(p)
    beta[signals] = amp / np.sqrt(n) * rng.choice([-1, 1], k)
    mu = X @ beta
    y = mu + rng.standard_normal(n)
    y_noisy = y + rng.normal(0, np.sqrt(scalar_noise), n)
    return X, beta, mu, y, y_noisy
```

We fit the LASSO on the randomized data as before:

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
```

### Swap estimands

`SwapEstimands` builds the contrasts $\eta_{s,j}$. It stores $Q_{E,E} = W^{-1}$ and caches
the column $Q e_j$ of each screened variable. Each contrast then costs one
$|E| \times |E|$ solve.

```{code-cell} ipython3
class SwapEstimands:
    """
    Coefficient of inactive variable j in the model (E minus {s}) plus {j}, for s in E,
    using the information matrix Q_hat of the model E.
    """

    def __init__(self, lasso_inference):
        self.Q = lasso_inference.Q_hat
        self.E = np.asarray(lasso_inference.E, dtype=int)
        Q_EE = np.linalg.inv(lasso_inference.W)
        self.Q_EE = (Q_EE + Q_EE.T) / 2
        self.p = self.Q.shape[0]
        self._Q_col = {}

    def _column(self, j):
        if j not in self._Q_col:
            e_j = np.zeros(self.p)
            e_j[j] = 1.
            self._Q_col[j] = np.ravel(self.Q @ e_j)
        return self._Q_col[j]

    def model(self, s, j):
        """F = (E minus {s}) plus {j}, with j last."""
        return np.append(self.E[self.E != s], j)

    def swap_coef(self, s, j) -> ContrastEstimand:
        if s not in self.E or j in self.E:
            raise ValueError('need s in E and j not in E')
        Q_j = self._column(j)
        keep = self.E != s
        F = self.model(s, j)
        Q_FF = np.empty((len(F), len(F)))
        Q_FF[:-1, :-1] = self.Q_EE[np.ix_(keep, keep)]
        Q_FF[:-1, -1] = Q_FF[-1, :-1] = Q_j[F[:-1]]
        Q_FF[-1, -1] = Q_j[j]
        eta = np.zeros(self.p)
        eta[F] = np.linalg.solve(Q_FF, np.eye(len(F))[-1])
        return ContrastEstimand(eta)
```

`analyze` runs the whole pipeline on one data set. It fits the LASSO, screens the top 10
inactive variables with the unchanged `ScreenedSelection`, and then loops over every pair
$(s, j)$, giving $|E| \times 10$ estimands. `estimand_summary` with the `ScreenedSelection`
conditions on both selection steps. The target $\eta_{s,j}' X'\mu$ is shown in the `truth`
column.

```{code-cell} ipython3
def swap_table(selection, swaps, selected, screened, **kwargs):
    estimands = {(s, j): swaps.swap_coef(s, j) for s in selected for j in screened}
    df = estimand_summary(selection, estimands, **kwargs)
    df.index = pd.MultiIndex.from_tuples(df.index, names=['dropped', 'added'])
    return df, estimands

def analyze(X, beta, mu, y, y_noisy):
    LI = lasso_inference(X, y, y_noisy)
    screen = ScreenedSelection(LI, top_k=10)
    swaps = SwapEstimands(LI)
    print(f'{len(LI.E)} active variables, {np.sum(beta[LI.E] != 0)} of them true signals; '
          f'{np.sum(beta[LI.E_c] != 0)} true signals were left out')
    print('screened:', screen.screened)
    swap_df, swap_estimands = swap_table(screen, swaps, LI.E, screen.screened)
    dropped = swap_df.index.get_level_values('dropped')
    added = swap_df.index.get_level_values('added')
    swap_df['truth'] = [est.value(X.T @ mu) for est in swap_estimands.values()]
    swap_df['dropped_signal'] = beta[dropped] != 0
    swap_df['added_signal'] = beta[added] != 0
    swap_df['adjacent'] = np.abs(dropped - added) <= 2
    return LI, screen, swaps, swap_df, swap_estimands
```

To compare, we also compute the LASSO-only tests, which use the LASSO constraints but ignore
the screening step, and naive tests that ignore selection altogether:

```{code-cell} ipython3
def naive_summary(LI, estimands, level=0.9):
    q = norm.ppf(1 - (1 - level) / 2)
    rows = []
    for name, est in estimands.items():
        sd = np.sqrt(LI.si.compute_contrast(est.eta).naive_variance)
        theta = est.value(LI.Z_full)
        rows.append({'estimate': theta, 'lower_conf': theta - q * sd,
                     'upper_conf': theta + q * sd, 'p_value': 2 * norm.sf(abs(theta) / sd)})
    return pd.DataFrame(rows, index=pd.MultiIndex.from_tuples(estimands, names=['dropped', 'added']))

def compare_p_values(LI, screen, swaps, swap_df, swap_estimands):
    compare = pd.concat({'screened': swap_df[['estimate', 'lower_conf', 'upper_conf', 'p_value']],
                         'lasso only': swap_table(LI, swaps, LI.E, screen.screened)[0],
                         'naive': naive_summary(LI, swap_estimands)}, axis=1)
    return compare.xs('p_value', axis=1, level=1)
```

For repeated experiments, we record for each pair $(s, j)$ whether the 90% interval covers
$\eta_{s,j}' X'\mu$, and whether the test rejects at level 0.1. Intervals from the same data
set are dependent, so the averages are over both replications and pairs.

```{code-cell} ipython3
def one_replication(rng, rho, amp):
    X, beta, mu, y, y_noisy = make_data(rng, rho, amp)
    LI = lasso_inference(X, y, y_noisy)
    screen = ScreenedSelection(LI, top_k=10)
    swaps = SwapEstimands(LI)
    q = norm.ppf(0.95)
    rows = []
    for s in LI.E:
        for j in screen.screened:
            est = swaps.swap_coef(s, j)
            target = est.value(X.T @ mu)
            sd = np.sqrt(LI.si.compute_contrast(est.eta).naive_variance)
            theta = est.value(LI.Z_full)
            info = {'dropped_signal': beta[s] != 0, 'added_signal': beta[j] != 0,
                    'adjacent': abs(s - j) <= 2}
            for method, sel in [('screened', screen), ('lasso only', LI)]:
                res = contrast_inference(sel, est)
                rows.append({'method': method, 'cover': res.lower_conf <= target <= res.upper_conf,
                             'length': res.upper_conf - res.lower_conf,
                             'reject': res.p_value < 0.1, **info})
            rows.append({'method': 'naive', 'cover': abs(theta - target) <= q * sd,
                         'length': 2 * q * sd, 'reject': abs(theta) > q * sd, **info})
    return rows

def simulate(rho, amp, nsim, seed=1):
    rng = np.random.default_rng(seed)
    return pd.DataFrame([row for _ in range(nsim) for row in one_replication(rng, rho, amp)])
```

## Example 1: independent features

```{code-cell} ipython3
rng = np.random.default_rng(0)
X, beta, mu, y, y_noisy = make_data(rng, rho=0., amp=3)
LI, screen, swaps, swap_df, swap_estimands = analyze(X, beta, mu, y, y_noisy)
```

As a check, the estimate is the least squares coefficient of $x_j$ on $X_F$:

```{code-cell} ipython3
s, j = LI.E[0], screen.screened[0]
F = swaps.model(s, j)
ols = np.linalg.lstsq(X[:, F], y, rcond=None)[0][-1]
print(ols, swaps.swap_coef(s, j).value(X.T @ y))
```

```{code-cell} ipython3
swap_df.head(10)
```

The p-values are easier to read as a grid, with selected variables as rows and screened
variables as columns:

```{code-cell} ipython3
swap_df['p_value'].unstack('added').round(3)
```

A small p-value in row $s$, column $j$ means that $x_j$ still carries signal once
$x_s$ is removed from the model. When $x_j$ is a surrogate for $x_s$ these p-values are small
in row $s$ and large in the other rows. A screened variable whose p-value is small in every
row is a candidate for a missed signal in its own right, as in [](screening_inactive.md).
With independent features, the columns of the grid are nearly constant: dropping one selected
variable barely changes the coefficient of $x_j$.

### Does screening matter?

The three sets of p-values for the same pairs:

```{code-cell} ipython3
compare_p_values(LI, screen, swaps, swap_df, swap_estimands).head(10)
```

The naive p-values are much smaller. Over 50 replications:

```{code-cell} ipython3
results_indep = simulate(rho=0., amp=3, nsim=50)
results_indep.groupby('method').agg(coverage=('cover', 'mean'), median_length=('length', 'median'))
```

```{code-cell} ipython3
results_indep.pivot_table(index='method', columns='added_signal', values='reject', aggfunc='mean')
```

Only the intervals that condition on the screening step come close to the nominal 90% coverage.
When the added variable is null, the screened tests reject at about the nominal 10%, and the
LASSO-only and naive tests reject far more often. With independent features,
$M_{js} \approx 0$ in the decomposition above, so $\hat{\theta}_{s,j}$ is nearly the screened
coefficient $\hat{\beta}_{j \cdot G}$. That coefficient is large partly because $x_j$ was
chosen for its large gradient, which is the selection effect that screening accounts for.

## Example 2: correlated features and surrogates

```{code-cell} ipython3
rng = np.random.default_rng(0)
X, beta, mu, y, y_noisy = make_data(rng, rho=0.8, amp=5)
LI, screen, swaps, swap_df, swap_estimands = analyze(X, beta, mu, y, y_noisy)
```

The interesting pairs are a selected null variable next to a screened true signal: the
LASSO chose $x_s$ in place of the nearby $x_j$. Here are the pairs at distance at most 2:

```{code-cell} ipython3
swap_df[swap_df['adjacent']]
```

```{code-cell} ipython3
swap_df['p_value'].unstack('added').round(3)
```

Both patterns appear here. The LASSO selected the null feature 161 instead of the signal 160.
Column 160 is large in most rows, but dropping 161 gives a p-value of about 0.01. Column 261
is small in every row: the signal 260 was missed, and its neighbor 261 carries its effect
whichever selected variable is dropped.

```{code-cell} ipython3
compare_p_values(LI, screen, swaps, swap_df, swap_estimands).head(10)
```

Over 30 replications:

```{code-cell} ipython3
results_ar = simulate(rho=0.8, amp=5, nsim=30)
results_ar.groupby('method').agg(coverage=('cover', 'mean'), median_length=('length', 'median'))
```

```{code-cell} ipython3
results_ar.pivot_table(index='method', columns=['dropped_signal', 'added_signal'],
                       values='reject', aggfunc='mean')
```

The screened intervals again have close to the nominal coverage. In this design the
LASSO-only and naive intervals undercover only slightly, so here ignoring screening does
much less harm than in Example 1.

When the added variable is null, the target $\beta_{j \cdot F}$ is not exactly zero. If $x_j$
is correlated with the dropped $x_s$ and $x_s$ carries signal, the coefficient of $x_j$
absorbs part of that signal. The table below restricts to the surrogate pairs, where a
selected null sits next to a screened signal. In those pairs, the screened tests detect the
swap in a large share of cases:

```{code-cell} ipython3
surrogate = results_ar[results_ar['adjacent'] & ~results_ar['dropped_signal']
                       & results_ar['added_signal']]
surrogate.groupby('method').agg(pairs=('reject', 'size'), coverage=('cover', 'mean'),
                                reject=('reject', 'mean'))
```

Conditioning on screening can be changed in the same way as in [](screening_inactive.md),
for example `ScreenedSelection(LI, top_k=10, conditioning='exact')` or a `threshold`. The
swap contrasts do not depend on that choice.
