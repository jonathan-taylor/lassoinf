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

# Logistic Lasso Selective Inference Example

This notebook demonstrates how to perform selective inference after fitting a logistic lasso model. 
We will generate some synthetic data, use the bootstrap to estimate the covariance of the unpenalized score, and finally compute post-selection confidence intervals and p-values for the parameters.

## Setup Data and Bootstrap

We start by generating a dataset of $n=300$ observations and $p=10$ features, where only the first few features are truly active. Then, we approximate the variance of the unpenalized score $Z_{full} = Q \bar{\beta}$ via the bootstrap.

```{code-cell} ipython3
import numpy as np
import cvxpy as cp
import pandas as pd
from scipy.special import expit
from lassoinf import LassoInference

# 1. Generate data
rng = np.random.default_rng()#20)
n, p = 800, 10
X = rng.standard_normal((n, p)) 

true_beta = np.zeros(p)
true_beta[:3] = [2.0, -2.0, 1.0] / np.sqrt(n)

logits = X @ true_beta
probs = expit(logits)
y = rng.binomial(1, probs)
```

```{code-cell} ipython3
# 2. Estimate Sigma via bootstrap
B = 300
Z_boot = []

print("Running bootstrap...")
for b in range(B):
    indices = rng.choice(n, n, replace=True)
    X_b, y_b = X[indices], y[indices]
    
    # Unpenalized logistic fit
    beta_b = cp.Variable(p)
    loss_b = cp.sum(
        cp.logistic(X_b @ beta_b) - cp.multiply(y_b, X_b @ beta_b)
    )
    prob_b = cp.Problem(cp.Minimize(loss_b))
    
    # Using SCS solver for reliability here
    prob_b.solve(solver=cp.SCS)
    
    if beta_b.value is None:
        continue
        
    b_val = beta_b.value
    p_b = expit(X_b @ b_val)
    W_b = np.diag(p_b * (1 - p_b))
    Q_b = X_b.T @ W_b @ X_b
    Z_b = Q_b @ b_val
    
    Z_boot.append(Z_b)

Z_boot = np.array(Z_boot)
Sigma = np.cov(Z_boot, rowvar=False)
print("Bootstrap finished.")
```

## Fitting the Unpenalized and Penalized Models

We now fit the unpenalized model on the original data to obtain $Z_{full}$, which acts as the target for our inference. Then, we simulate a "noisy" experiment by selecting a random bootstrap sample to act as our dataset for selection. We fit the lasso penalty on this sample to choose our model.

```{code-cell} ipython3
# 3. Unpenalized fit on full data
beta_orig = cp.Variable(p)
loss_orig = cp.sum(
    cp.logistic(X @ beta_orig) - cp.multiply(y, X @ beta_orig)
)
prob_orig = cp.Problem(cp.Minimize(loss_orig))
prob_orig.solve(solver=cp.SCS)

bar_beta = beta_orig.value
p_orig = expit(X @ bar_beta)
W_orig = np.diag(p_orig * (1 - p_orig))
Q_full = X.T @ W_orig @ X

# Target statistic Z_full
Z_full = Q_full @ bar_beta

# 4. "Noisy" example: selection on a bootstrap sample
indices_noisy = rng.choice(n, n, replace=True)
X_noisy, y_noisy = X[indices_noisy], y[indices_noisy]

beta_lasso = cp.Variable(p)
lam = 1. # L1 penalty
D_weight = lam * (np.ones(p) + np.linspace(-0.1, 0.1, p))
D_weight[2] = 0.0 # 3rd feature unpenalized

loss_noisy = cp.sum(
    cp.logistic(X_noisy @ beta_lasso) - cp.multiply(y_noisy, X_noisy @ beta_lasso)
)
penalty = cp.sum(cp.multiply(D_weight, cp.abs(beta_lasso)))
L1, U1 = -2, 2
constraints = [
    beta_lasso[0] >= L1,
    beta_lasso[0] <= U1,
    beta_lasso[1] >= 0.0
]
prob_lasso = cp.Problem(cp.Minimize(loss_noisy + penalty), constraints)
prob_lasso.solve(solver=cp.SCS)

beta_hat = beta_lasso.value

# Post-selection quantities
p_noisy = expit(X_noisy @ beta_hat)
G_hat = -X_noisy.T @ (y_noisy - p_noisy)
W_noisy = np.diag(p_noisy * (1 - p_noisy))
Q_hat = X_noisy.T @ W_noisy @ X_noisy
```

## Post-Selection Inference

With all ingredients gathered, we can pass the selection parameters, original constraints, and statistics into `LassoInference`.

```{code-cell} ipython3
D = D_weight
L_bound = np.full(p, -np.inf)
L_bound[0] = L1
L_bound[1] = 0.0

U_bound = np.full(p, np.inf)
U_bound[0] = U1

# 5. Inference
inference = LassoInference(
    beta_hat=beta_hat,
    G_hat=G_hat,
    Q_hat=Q_hat,
    D=D,
    L=L_bound,
    U=U_bound,
    Z_full=Z_full,
    Sigma=Sigma,
    Sigma_noise=Sigma  # Re-use bootstrap covariance
)

# 6. View the summary of free (selected) variables
carve_df = inference.summary_
carve_df
```

## The same selection with glmstar

[glmstar](https://github.com/jonathan-taylor/glmstar) fits glmnet's problem in Python. Its path
estimator `LogNet` runs R's glmnet code, and `glmstar_problem` reads off the LASSO problem a
fit solved, in the original coordinates: the solution, the gradient and Hessian of the smooth
part, the penalty weights and the bounds. These are the selection inputs of `LassoInference`.
Here we refit the selection problem above with glmstar instead of cvxpy.

glmnet minimizes

$$
\frac{1}{n} \sum_i \ell(y_i, x_i^T\beta) + \lambda \sum_j \text{pf}_j |\beta_j|,
$$

with $\ell$ the negative log-likelihood and the penalty factors $\text{pf}$ rescaled to sum
to $p$. Dividing the cvxpy objective
above by $n$, we need $\lambda\, \text{pf}_j = D_j / n$: we take $\text{pf} = D$ (including
its 0 for the unpenalized feature) and $\lambda = \sum_j D_j / (pn)$. There is no intercept
and no standardization, and the bounds are passed as limits. glmnet fits a path of $\lambda$
values, so we fit one ending at ours. `glmstar_problem` warns if the fit does not satisfy
the KKT conditions closely enough, so a small `thresh` matters. We use glmstar's path
solvers (`LogNet`, and `GaussNet` or `FishNet` for other families) rather than its IRLS
estimator `GLMNet`: they solve glmnet's problem exactly, including penalty factors of 0.

```{code-cell} ipython3
:tags: [remove-stderr]

from glmnet import LogNet
from glmnet.paths.fastnet import FastNetControl
from lassoinf import glmstar_problem
from lassoinf.glm_problem import kkt_violation

lam_glmnet = D_weight.sum() / (p * n)
glmstar_fit = LogNet(fit_intercept=False,
                     standardize=False,
                     penalty_factor=D_weight.copy(),  # glmstar rescales it in place
                     lower_limits=L_bound,
                     upper_limits=U_bound,
                     lambda_values=lam_glmnet * np.exp(np.linspace(np.log(20), 0, 30)),
                     control=FastNetControl(thresh=1e-14))
glmstar_fit.fit(X_noisy, y_noisy)
glm_prob = glmstar_problem(glmstar_fit, X_noisy, y_noisy, lambda_val=lam_glmnet)
```

glmstar's problem is on the per-observation scale. Multiplying the gradient, the Hessian and
the penalty weights by $n$ puts it on the scale of the cvxpy problem, where it should agree
with the cvxpy solution up to the accuracy of SCS:

```{code-cell} ipython3
glm_args = glm_prob.lasso_args()
for k in ['G_hat', 'Q_hat', 'D']:
    glm_args[k] = n * glm_args[k]

pd.DataFrame({'cvxpy': beta_hat, 'glmstar': glm_args['beta_hat']}).round(4)
```

```{code-cell} ipython3
print('max |D difference|:', np.abs(glm_args['D'] - D).max())
print('max |beta difference|:', np.abs(glm_args['beta_hat'] - beta_hat).max())
print('max |Q difference| / max |Q|:', np.abs(glm_args['Q_hat'] - Q_hat).max() / np.abs(Q_hat).max())
```

The inference uses the same $Z_{full}$ and $\Sigma$:

```{code-cell} ipython3
glmstar_inference = LassoInference(**glm_args,
                                   Z_full=Z_full,
                                   Sigma=Sigma,
                                   Sigma_noise=Sigma)
glmstar_inference.summary_
```

Side by side with the cvxpy results:

```{code-cell} ipython3
cols = ['lower_conf', 'upper_conf', 'p_value']
pd.concat({'cvxpy': carve_df[cols], 'glmstar': glmstar_inference.summary_[cols]}, axis=1).round(4)
```

The two fits agree up to the accuracy of the cvxpy solution. Both satisfy the KKT conditions
of the same problem; `kkt_violation` checks a fit from any solver against them. SCS returns
small nonzero values instead of exact zeros, so for cvxpy we treat coefficients below `1e-4`
in absolute value as 0:

```{code-cell} ipython3
print('KKT violation, glmstar:', glm_prob.kkt_violation() * n)
print('KKT violation, cvxpy:',
      np.max(kkt_violation(beta_hat, G_hat, D, L_bound, U_bound, tol=1e-4)))
```
