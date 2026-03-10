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

# No Randomization

This notebook demonstrates how to perform selective inference after fitting an Ordinary Least Squares (OLS) model with a lasso penalty. 
Instead of data splitting, we use a **parametric bootstrap** (randomization) approach. We add Gaussian noise to the data such that the "size" of the randomization is equivalent to using a specific proportion of the data for selection.

In this example we'll  run the lasso without randomization and construct
confidence intervals. In the background, the code assumes a "minimum" 
of `scalar_noise` of 0.001 corresponding to retaining about 1/1000-th of the
data for inference.

We generate $n=100$ observations and $p=20$ features. We include 1 strong effect, 2 weak effects, and set the rest to truly 0. The true noise variance is $\sigma^2 = 2$.

```{code-cell} ipython3
import numpy as np
import cvxpy as cp
import pandas as pd
from lassoinf import LassoInference
```

```{code-cell} ipython3
n, p = 100, 20
signal_strength = 4
seed = 0
```

```{code-cell} ipython3
# 1. Generate data
rng = np.random.default_rng(seed)
X = rng.standard_normal((n, p)) 

true_beta = np.zeros(p)
true_beta[0] = 2 * signal_strength / np.sqrt(n)   # 1 strong effect
true_beta[1] = signal_strength / np.sqrt(n) # 1 weak effect
true_beta[2] = -signal_strength / np.sqrt(n) # 1 weak effect

sigma2_true = 2.0
y = X @ true_beta + rng.normal(0, np.sqrt(sigma2_true), size=n)
```

Now we configure the parametric bootstrap to mimic $\pi = 0.9$:

```{code-cell} ipython3
# Target statistic Z_full
Z_full = X.T @ y
Q_hat = X.T @ X
beta_OLS = np.linalg.solve(Q_hat, Z_full)
sigma2_est = np.linalg.norm(y - X @ beta_OLS)**2 / (n - p)

# True covariance of the score
Sigma = sigma2_est * (X.T @ X)
```

## Fitting the Penalized Model on Noisy Data

We fit the lasso penalty on the randomized sample $y_{noisy}$ to select our active set and signs.

```{code-cell} ipython3
# 3. Fit the Lasso model on the noisy data
beta_lasso = cp.Variable(p)
loss =  0.5 * cp.sum_squares(y - X @ beta_lasso)
lam = 2 * np.sqrt(n) # L1 penalty parameter
D_weight = lam * np.ones(p)
penalty = cp.sum(cp.multiply(D_weight, cp.abs(beta_lasso)))

prob_lasso = cp.Problem(cp.Minimize(loss + penalty))
# Using SCS or ECOS solver
prob_lasso.solve(solver=cp.SCS)

beta_hat = beta_lasso.value

# Post-selection quantities
# Gradient of the unpenalized loss at beta_hat
G_hat = X.T @ (X @ beta_hat - y)
Q_hat = X.T @ X
```

## Post-Selection Inference

With all ingredients gathered, we pass the selection parameters and the theoretical covariances into `LassoInference`.

```{code-cell} ipython3
D = D_weight
# Standard lasso with no additional bounds
L_bound = np.full(p, -np.inf)
U_bound = np.full(p, np.inf)

# 4. Inference
inference = LassoInference(
    beta_hat=beta_hat,
    G_hat=G_hat,
    Q_hat=Q_hat,
    D=D,
    L=L_bound,
    U=U_bound,
    Z_full=Z_full,
    Sigma=Sigma,
    Sigma_noise=None,
    scalar_noise=0
)

# 5. View the summary of free (selected) variables
carve_df = inference.summary_
carve_df['length'] = carve_df['upper_conf'] - carve_df['lower_conf']
carve_df
```

### Finding the true parameter

The contrasts are retained, which allow us to compute
*true* projected parameters:

```{code-cell} ipython3
true_Z = X.T @ (X @ true_beta)
carve_df['truth'] = [(inference._contrasts[j].direction * true_Z).sum() for j in df.index]
```

### Checking coverage

```{code-cell} ipython3
carve_df['cover'] = (carve_df['lower_conf'] < carve_df['truth']) * (carve_df['upper_conf'] > carve_df['truth'])
carve_df
```

```{code-cell} ipython3

```
