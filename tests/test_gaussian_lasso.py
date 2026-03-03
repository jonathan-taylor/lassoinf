# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     default_lexer: ipython3
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Gaussian Lasso Selective Inference Example
#
# ## Non-parametric (Pairs) Bootstrap
#
# This notebook demonstrates how to perform selective inference after fitting an Ordinary Least Squares (OLS) model with a lasso penalty. 
# Instead of data splitting, we use a **parametric bootstrap** (randomization) approach. We add Gaussian noise to the data such that the "size" of the randomization is equivalent to using a specific proportion of the data for selection.
#
# ## Proportion for Selection and Gamma ($\pi$)
#
# In classical data splitting, a proportion $\pi \in (0, 1)$ of the data is used for model selection, and the remaining $1-\pi$ is used for inference. For a score statistic $Z_{full} = X^T y$ with variance $\Sigma = \sigma^2 X^T X$, the selection score computed on $\pi n$ observations has variance $\Sigma / \pi$.
#
# In randomized selection, we instead construct a noisy score $Z_{noisy} = Z_{full} + \omega$, where $\omega \sim N(0, \Omega)$ so that $Z_{noisy}$ is such that the estimate
# $$
# \hat{\beta}^* = (X'X)^{-1}Z_{noisy} \sim N\left(\hat{\beta}, \frac{1}{\pi} \sigma^2 (X'X)^{-1}\right).
# $$
# To achieve an equivalent amount of information for selection as data splitting with proportion $\pi$, we set the variance of the noisy score to match:
#
# $$ \text{Var}(Z_{noisy}) = \Sigma + \Omega = \frac{1}{\pi} \Sigma. $$
#
# This implies that the injected noise must have variance:
#
# $$ \Omega = \frac{1 - \pi}{\pi} \Sigma $$
#
# In our example, we use $\pi = 0.9$ (90% of the data for selection). In the non-parametric bootstrap context, this proportion of variance added to the score is controlled by $m$ in the $m$-out-of-$n$ bootstrap with the classical choice of $n$ corresponding to
# $\pi=1/2$. Generally, we want to use $m=\pi/(1-\pi)$.
#
# When $\pi \to 1$, we add very little noise (almost all information is used for selection), which results in less power for post-selection inference. When $\pi \to 0$, we add a lot of noise, leaving more information for the inference stage.
#
# ## Setup Data and Non-Parametric (Pairs) Bootstrap
#
# We generate $n=100$ observations and $p=20$ features. We include 1 strong effect, 2 weak effects, and set the rest to truly 0. The true noise variance is $\sigma^2 = 2$.

# %%
import numpy as np
import cvxpy as cp
import pandas as pd
from lassoinf import LassoInference

# %%
n, p = 100, 20
signal_strength = 4
seed = 0

# %%
# 1. Generate data
rng = np.random.default_rng(seed)

def test_gaussian_lasso():
    X = rng.standard_normal((n, p)) 

    true_beta = np.zeros(p)
    true_beta[0] = signal_strength / np.sqrt(n)   # 1 strong effect
    true_beta[1] = signal_strength / np.sqrt(n) # 1 weak effect
    true_beta[2] = -signal_strength / np.sqrt(n) # 1 weak effect

    sigma2_true = 2.0
    y = X @ true_beta + rng.normal(0, np.sqrt(sigma2_true), size=n)

    # %% [markdown]
    # Now we configure the non-parametric bootstrap to mimic $\pi = 0.9$.
    # This corresponds to running a $m=n \pi /(1 - \pi)=9n$-out-of-$n$ bootstrap.
    #
    # If we want to keep our penalty the same and just think of perturbing the
    # original loss, we should scale the bootstrapped quadratic part by $f=n/m=1/9$
    # leaving the penalty as it is.

    # %%
    # 2. Non-parametric bootstrap setup
    pi_selection = 0.9
    f = frac_m_of_n = (1 - pi_selection) / pi_selection
    # Target statistic Z_full
    Z_full = X.T @ y
    Q_hat = X.T @ X

    idx = rng.choice(n, int(n / f), replace=True)
    y_star = y[idx]
    X_star = X[idx]

    # %% [markdown]
    # ## Fitting the Penalized Model on Noisy Data
    #
    # We fit the lasso penalty on the randomized sample $y_{noisy}$ to select our active set and signs.

    # %%
    # 3. Fit the Lasso model on the noisy data
    beta_lasso = cp.Variable(p)

    # %% [markdown]
    # It will be convenient to normalize the smooth loss so it is roughly
    # our original loss plus a small perturbation. The quadratic
    # term in the log-likelihood for $m$-out-of-$n$ grows like $m$, hence
    # to keep the quadratic term growing like $n$ we scale the smooth loss by $f$.

    # %%
    loss_noisy = 0.5 * cp.sum_squares(y_star - X_star @ beta_lasso) * f

    # %% [markdown]
    # We may want to change penalty as well. That is somewhat up to the user.

    # %%
    lam = 2 * np.sqrt(n) # L1 penalty parameter
    D_weight = lam * np.ones(p)
    penalty = cp.sum(cp.multiply(D_weight, cp.abs(beta_lasso)))

    # %%
    prob_lasso = cp.Problem(cp.Minimize(loss_noisy + penalty))
    # Using SCS or ECOS solver
    prob_lasso.solve(solver=cp.SCS)

    beta_hat = beta_lasso.value

    # Post-selection quantities
    # Gradient of the unpenalized loss at beta_hat
    G_hat = X_star.T @ (X_star @ beta_hat - y_star) * f

    # Hessian of the unpenalized loss
    Q_hat = X_star.T @ X_star * f

    # %% [markdown]
    # ### Estimating $\Sigma$

    # %%
    B = 1000
    full_score = X.T @ (y - X @ beta_hat)
    M_1 = np.zeros(p)
    M_2 = np.zeros((p, p))
    for _ in range(B):
        idx = rng.choice(n, n, replace=True)
        centered_score_star = X[idx].T @ (y[idx] - X[idx] @ beta_hat) - full_score
        M_1 += centered_score_star
        M_2 += np.multiply.outer(centered_score_star, centered_score_star)
    M_1 /= B
    M_2 /= B
    Sigma = M_2 - np.multiply.outer(M_1, M_1)
    Sigma_noisy = frac_m_of_n * Sigma

    # %% [markdown]
    # ## Post-Selection Inference
    #
    # With all ingredients gathered, we pass the selection parameters and the theoretical covariances into `LassoInference`.

    # %%
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
        Sigma_noise=Sigma_noisy
    )

    # %% [markdown]
    # ### Using `scalar_noise`
    #
    # The "well specified" assumption essentially assumes that
    # `Sigma_noisy` is proportional to `Sigma`. It simplifies
    # some calculations,  by avoiding a $p \times p$ solve. It is used via the
    # `scalar_noise` parameter which is used when `Sigma_noisy` is `None`:

    # %%
    inference_scalar = LassoInference(
        beta_hat=beta_hat,
        G_hat=G_hat,
        Q_hat=Q_hat,
        D=D,
        L=L_bound,
        U=U_bound,
        Z_full=Z_full,
        Sigma=Sigma,
        Sigma_noise=None,
        scalar_noise=frac_m_of_n
    )

    scalar_df = inference_scalar.summary()
    scalar_df

    carve_df = inference.summary()
    carve_df

    print(scalar_df, 'scalar')
    print(carve_df, 'carve')
    
    assert np.allclose(scalar_df.values, carve_df.values, rtol=1e-5, atol=1e-5)

    carve_df['length'] = carve_df['upper_conf'] - carve_df['lower_conf']

    split_df = inference._splitting
    split_df['length'] = split_df['upper_conf'] - split_df['lower_conf']

    assert np.all(carve_df['length'] <= split_df['length'] * (1 + 1e-5))
    naive_df = inference._naive
