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
# # Logistic Lasso Selective Inference Example
#
# This notebook demonstrates how to perform selective inference after fitting a logistic lasso model. 
# We will generate some synthetic data, use the bootstrap to estimate the covariance of the unpenalized score, and finally compute post-selection confidence intervals and p-values for the parameters.
#
# ## Setup Data and Bootstrap
#
# We start by generating a dataset of $n=300$ observations and $p=10$ features, where only the first few features are truly active. Then, we approximate the variance of the unpenalized score $Z_{full} = Q \bar{\beta}$ via the bootstrap.

# %%
import numpy as np
import cvxpy as cp
import pandas as pd
from scipy.special import expit

# 1. Generate data
rng = np.random.default_rng()#20)
n, p = 800, 20
X = rng.standard_normal((n, p)) 

true_beta = np.zeros(p)
true_beta[:3] = [2.0, -2.0, 1.0] / np.sqrt(n)

logits = X @ true_beta
probs = expit(logits)
y = rng.binomial(1, probs)

# %%
# 2. Estimate Sigma via bootstrap
B = 30
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

# %% [markdown]
# ## Fitting the Unpenalized and Penalized Models
#
# We now fit the unpenalized model on the original data to obtain $Z_{full}$, which acts as the target for our inference. Then, we simulate a "noisy" experiment by selecting a random bootstrap sample to act as our dataset for selection. We fit the lasso penalty on this sample to choose our model.

# %%
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
lam = 1.0 * np.sqrt(n) # L1 penalty
D_weight = lam * (np.ones(p) + np.linspace(-0.1, 0.1, p))
D_weight[2] = 0.0 # 3rd feature unpenalized

loss_noisy = cp.sum(
    cp.logistic(X_noisy @ beta_lasso) - cp.multiply(y_noisy, X_noisy @ beta_lasso)
)
penalty = cp.sum(cp.multiply(D_weight, cp.abs(beta_lasso)))
# constraints = [
#     beta_lasso[0] >= -2.0,
#     beta_lasso[0] <= 2.0,
#     beta_lasso[1] >= 0.0
# ]
prob_lasso = cp.Problem(cp.Minimize(loss_noisy + penalty)) # , constraints)
prob_lasso.solve(solver=cp.SCS)

beta_hat = beta_lasso.value

# Post-selection quantities
p_noisy = expit(X_noisy @ beta_hat)
G_hat = -X_noisy.T @ (y_noisy - p_noisy)
W_noisy = np.diag(p_noisy * (1 - p_noisy))
Q_hat = X_noisy.T @ W_noisy @ X_noisy

# %% [markdown]
# ## Post-Selection Inference
#
# With all ingredients gathered, we can pass the selection parameters, original constraints, and statistics into `LassoInference`.

# %%
from lassoinf.selective_inference import LassoInference

D = D_weight
L_bound = np.full(p, -np.inf)
#L_bound[0] = -2.0
#L_bound[1] = 0.0

U_bound = np.full(p, np.inf)
#U_bound[0] = 2.0

# 5. Inference
print(f"Z_full shape: {Z_full.shape}")
print(f"Sigma shape: {Sigma.shape}")
print(f"Sigma trace: {np.trace(Sigma)}")
print(f"beta_hat: {beta_hat}")
print(f"G_hat: {G_hat}")

inference = LassoInference(
    beta_hat=beta_hat,
    G_hat=G_hat,
    Q_hat=Q_hat,
    D=D,
    L=L_bound,
    U=U_bound,
    Z_full=Z_full,
    Sigma=Sigma,
    Sigma_noisy=Sigma  # Re-use bootstrap covariance
)

# 6. View the summary of free (selected) variables
summary_df = inference.summary()
print(summary_df)

# %%
