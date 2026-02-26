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

## Classic LASSO

The classic LASSO problem in Lagrange form ($n > p$ and full rank where needed)
is 
$$
\text{minimize}_{\beta} \frac{1}{2} \|Y-X\beta\|^2_2 + \|D\beta\|_1 = 
\text{minimize}_{\beta} -\beta'Z + \frac{1}{2} \beta'Q\beta + \|D\beta\|_1
$$
where $Z=X'y$, $Q=X'X$ and $D$ is some diagonal matrix. Setting $\bar{\ell}(\beta)=
\frac{1}{2} \|Y-X\beta\|^2_2$ we can write this as
$$
\text{minimize}_{\beta} - \beta'\nabla \bar{\ell}(\beta_0) + \frac{1}{2}(\beta-\beta_0)'
\nabla^2 \bar{\ell}(\beta_0)(\beta - \beta_0) + \|D\beta\|_1
$$
with the first expression using $\beta_0=0$. For population parameter
$$
\beta^*=\beta^*_F = \left(E_F[X'X]\right)^{-1} E_F[X'Y]
$$
this becomes
$$
\text{minimize}_{\beta} -\beta'\nabla \bar{\ell}(\beta^*_F) + 
\frac{1}{2}(\beta - \beta^*_F)'\nabla^2 \bar{\ell}(\beta^*_F)(\beta-\beta^*_F) + \|D\beta\|_1
$$
with $\nabla \bar{\ell}(\beta^*_F)$ approximately normally distributed (depending on 
the model we are using).

It is straightforward to check
that in the squared-error case the following identity holds **for all $\beta$:**
$$
Z=X'y = \nabla \bar{\ell}(\beta) + Q\beta.
$$
This is discussed in {cite}`https://purl.stanford.edu/sm283fq0116` (Jelena's thesis) among other places.

## M-estimator form

This last form generalizes. The GLM versions take on a more M-estimator form
$$
\text{minimize}_{\beta} \sum_{i=1}^n \ell(Y_i, X_i'\beta) + \|D\beta\|_1
= \text{minimize}_{\beta} \bar{\ell}(\beta) + \|D\beta\|_1
$$
for some sufficiently smooth $\ell$ with the classic lasso setting $\ell(r)=r^2$.

Given a solution $\hat{\beta}$ we can find an equivalent quadratic LASSO problem that
$\hat{\beta}$ similarly solves. Specifically, set
$$
\begin{aligned}
Q &= Q(\hat{\beta}) \\
&= \nabla^2 \left(\sum_{i=1}^n \ell(Y_i, X_i'\hat{\beta}) \right) \\
\cup{Z} &= Z(\hat{\beta}) \\
&= - \nabla \left(\sum_{i=1}^n \ell(Y_i, X_i'\hat{\beta}) \right) + Q \hat{\beta} \\
&= - \nabla \left(\sum_{i=1}^n \ell(Y_i, X_i'\hat{\beta}) \right) + Q[:,E] \hat{\beta}_E
\end{aligned}
$$
with $E=E(\hat{\beta})$ the support of $\hat{\beta}$. 



As in {cite}`CJS` set our one-step estimator of $\beta^*_E$ to be
$$
\bar{\beta}_E = \hat{\beta}_E + Q[E,E]^{-1}D[E]
$$

+++

We seek the residual of the stacked vector $X_{stack} = [Z, \bar{Z}]^\top$ after projecting onto the basis $V = [\hat{\theta}, W]^\top$.

## Covariance Analysis and Sherman-Morrison-Woodbury
The covariance of $W$ and the cross-covariance with $\hat{\theta}$ involve the Wiener filter matrix $A = \Sigma(\Sigma + \bar{\Sigma})^{-1}$. 
The variance-covariance matrix of the basis $V$ is given by:
$$ Var(V) = \begin{pmatrix} \sigma_{\hat{\theta}}^2 & \gamma \\ \gamma & \gamma \end{pmatrix} $$
where:
- $\sigma_{\hat{\theta}}^2 = \eta^\top \Sigma \eta$
- $\gamma = \eta^\top \Sigma (\Sigma + \bar{\Sigma})^{-1} \Sigma \eta$

Applying the \textbf{Sherman-Morrison-Woodbury Identity} to the expression for $\gamma$:
$$ (\Sigma + \bar{\Sigma})^{-1} = \Sigma^{-1} - \Sigma^{-1}(\Sigma^{-1} + \bar{\Sigma}^{-1})^{-1}\Sigma^{-1} $$
Substituting this into $\gamma$ yields the residual variance $\delta$:
$$ \delta = \sigma_{\hat{\theta}}^2 - \gamma = \eta^\top (\Sigma^{-1} + \bar{\Sigma}^{-1})^{-1} \eta $$

## The $\bar{Z}$-Residual
The projection of $\bar{Z}$ onto the span of $V$ simplifies significantly because $Cov(\bar{Z}, \hat{\theta}) = Cov(\bar{Z}, W) = \Sigma \eta$. The resulting $\bar{Z}$-residual is:
$$ \bar{Z}_{res} = \left[ I - \frac{\Sigma \eta \eta^\top \Sigma (\Sigma + \bar{\Sigma})^{-1}}{\gamma} \right] \bar{Z} $$

### Special Case: Homoscedastic Noise ($\bar{\Sigma} = \alpha \Sigma$)
When the noise is proportional to the signal covariance, the matrix terms cancel:
$$ \bar{Z}_{res} = \bar{Z} - \frac{\eta^\top \bar{Z}}{\eta^\top \Sigma \eta} \Sigma \eta $$
In this case, the residual is independent of the noise magnitude $\alpha$.

```{code-cell} ipython3
import numpy as np

def compute_projection_system(Sigma, Sigma_bar, eta):
    """
    Computes the projection components for a Gaussian stacked system.
    
    Parameters:
    Sigma (n x n array): Covariance of Z
    Sigma_bar (n x n array): Covariance of omega
    eta (n, array): Vector defining theta_hat = eta'Z
    
    Returns:
    dict: Contains the 2x2 basis covariance and the rank-1 projection matrix.
    """
    # Ensure eta is a column vector for matrix ops
    eta = eta.reshape(-1, 1)
    
    # 1. Precompute the shared signal vector: v = Sigma @ eta
    v = Sigma @ eta
    
    # 2. Compute the filtered vector x = (Sigma + Sigma_bar)^-1 @ v
    # Using solve() is more stable than inv()
    S_sum = Sigma + Sigma_bar
    x = np.linalg.solve(S_sum, v)
    
    # 3. Scalar components
    # sigma_theta_sq = eta' @ Sigma @ eta
    sigma_theta_sq = (eta.T @ v).item()
    
    # gamma = eta' @ Sigma @ (Sigma + Sigma_bar)^-1 @ Sigma @ eta
    # This is equivalent to v' @ (S_sum^-1) @ v, or more simply:
    gamma = (v.T @ x).item()
    
    # 4. Construct the 2x2 Basis Covariance Matrix Var(V)
    # V = [theta_hat, W]
    cov_basis = np.array([
        [sigma_theta_sq, gamma],
        [gamma,          gamma]
    ])
    
    # 5. Projection Matrix M such that Y_proj = M @ Y
    # Based on our derivation: M = (v @ x.T) / gamma
    M = (v @ x.T) / gamma
    
    return {
        "cov_basis": cov_basis,
        "projection_matrix_M": M,
        "gamma": gamma,
        "residual_variance": sigma_theta_sq - gamma
    }

# --- Example Setup ---
n = 5
eta_vec = np.random.randn(n)
S = np.diag(np.linspace(1, 2, n))  # Example Sigma
S_b = np.eye(n) * 0.1             # Example Sigma_bar

res = compute_projection_system(S, S_b, eta_vec)

print("--- 2x2 Basis Covariance Matrix ---")
print(res["cov_basis"])
print("\n--- Projection Matrix M (Rank-1) ---")
print(res["projection_matrix_M"])
print(f"\nCheck: Rank of M is {np.linalg.matrix_rank(res['projection_matrix_M'])}")
```

```{code-cell} ipython3
# -*- coding: utf-8 -*-
"""Optimization Problems

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1OnSk5QfnJUJrKu9Y35Vvr2fHQwgVLYbw
"""

# Optimization Solvers using CVXPY
# This script addresses two specific problems:
# 1. Lasso in quadratic form with diagonal regularization weight.
# 2. Finding the step-size 't' for an affine constraint.

# Install cvxpy if running in a fresh Colab environment
try:
    import cvxpy as cp
except ImportError:
    print("Installing cvxpy...")
    !pip install cvxpy
    import cvxpy as cp

import numpy as np

def solve_lasso_quadratic(Q, Z, D_diag):
    """
    Solves min -b'Z + 1/2 b'Qb + ||D beta||_1
    where D is diagonal.

    Parameters:
    - Q: (n, n) Positive Semidefinite Matrix
    - Z: (n,) vector
    - D_diag: (n,) vector representing diagonal of D
    """
    n = Q.shape[0]
    beta = cp.Variable(n)

    # 0.5 * beta.T @ Q @ beta
    quad_term = 0.5 * cp.quad_form(beta, Q)

    # -beta.T @ Z
    linear_term = -beta @ Z

    # ||D @ beta||_1
    # Since D is diagonal, D @ beta is element-wise multiplication
    reg_term = cp.norm1(cp.multiply(D_diag, beta))

    objective = cp.Minimize(quad_term + linear_term + reg_term)
    problem = cp.Problem(objective)

    problem.solve()

    return beta.value, problem.value

def solve_step_size(A, V, b, eta, mode='max'):
    """
    Solves min/max t subject to A(V + t * eta) <= b

    Parameters:
    - A: (m, n) matrix
    - V: (n,) initial vector
    - b: (m,) constraint vector
    - eta: (n,) direction vector
    - mode: 'min' or 'max'
    """
    t = cp.Variable()

    # Constraint: A @ (V + t * eta) <= b
    # CVXPY handles the affine expression distribution automatically
    constraints = [A @ (V + t * cp.reshape(eta, (len(eta), 1))) <= cp.reshape(b, (len(b), 1))]

    if mode == 'max':
        objective = cp.Maximize(t)
    else:
        objective = cp.Minimize(t)

    problem = cp.Problem(objective, constraints)
    problem.solve()

    return t.value, problem.status

def lasso_post_selection_constraints(Q, Z, D_diag, tol=1e-5):
    """
    Derives the linear constraints A * Z <= b characterizing the polytope
    where the active set and signs of the Lasso remain constant,
    based on Lee, Sun, Sun, and Taylor (2016).
    Allows for unpenalized variables by passing 0 in D_diag.
    """
    # 1. Solve the lasso to get the empirical active set and signs
    beta_hat, _ = solve_lasso_quadratic(Q, Z, D_diag)
    n = Q.shape[0]

    # 2. Extract active set (M) and inactive set (M_c)
    # Unpenalized variables are forced into the active set
    M_empirical = np.where(np.abs(beta_hat) > tol)[0]
    unpenalized = np.where(D_diag <= tol)[0]
    M = np.union1d(M_empirical, unpenalized).astype(int)
    M_c = np.setdiff1d(np.arange(n), M).astype(int)

    # Edge case: No variables are active
    if len(M) == 0:
        A = np.vstack([np.eye(n), -np.eye(n)])
        b = np.concatenate([D_diag, D_diag])
        return A, b, M, np.array([])

    s_M = np.sign(beta_hat[M])
    # Handle exact zeros to avoid multiplying by 0 incorrectly in subgradients
    s_M[s_M == 0] = 1

    # Matrix blocks
    Q_MM = Q[np.ix_(M, M)]
    Q_McM = Q[np.ix_(M_c, M)]

    # Inverse of Q_MM
    invQ_MM = np.linalg.inv(Q_MM)

    D_M = D_diag[M]
    D_Mc = D_diag[M_c]

    # Precompute shared terms to avoid redundant multiplications
    invQ_D_s = invQ_MM @ (D_M * s_M)

    # --- Constraint 1: Sign Constraints for Active Variables ---
    # Condition: diag(s_M) @ beta_M > 0
    # Only enforce for PENALIZED variables (where D_M > 0)
    penalized_in_M = np.where(D_M > tol)[0]

    if len(penalized_in_M) > 0:
        A1_full = -np.diag(s_M) @ invQ_MM
        A1 = np.zeros((len(penalized_in_M), n))
        A1[:, M] = A1_full[penalized_in_M]

        b1_full = -np.diag(s_M) @ invQ_D_s
        b1 = b1_full[penalized_in_M]
    else:
        A1 = np.zeros((0, n))
        b1 = np.zeros(0)

    # --- Constraint 2 & 3: Subgradient Bounds for Inactive Variables ---
    if len(M_c) > 0:
        Q_Mc_invQ = Q_McM @ invQ_MM

        # Condition: Z_Mc - Q_McM @ beta_M <= D_Mc
        # A2 * Z <= b2
        A2 = np.zeros((len(M_c), n))
        A2[:, M] = -Q_Mc_invQ
        A2[:, M_c] = np.eye(len(M_c))
        b2 = D_Mc - Q_Mc_invQ @ (D_M * s_M)

        # Condition: -Z_Mc + Q_McM @ beta_M <= D_Mc
        # A3 * Z <= b3
        A3 = np.zeros((len(M_c), n))
        A3[:, M] = Q_Mc_invQ
        A3[:, M_c] = -np.eye(len(M_c))
        b3 = D_Mc + Q_Mc_invQ @ (D_M * s_M)
    else:
        A2 = np.zeros((0, n))
        b2 = np.zeros(0)
        A3 = np.zeros((0, n))
        b3 = np.zeros(0)

    # Combine all constraints into a single Polytope A*Z <= b
    A = np.vstack([A1, A2, A3])
    b = np.concatenate([b1, b2, b3])

    return A, b, M, s_M

def compute_affine_w_constraints(A, b, z0, C, w0):
    """
    Computes the equivalent affine constraints for w, given the constraint
    set {Z : AZ <= b} and the substitution Z = z0 - C*w0 + C*w.

    This finds A_bar and b_bar such that the set can be written as
    {w : A_bar * w <= b_bar}.

    Parameters:
    - A: (m, n) constraint matrix for Z
    - b: (m,) constraint vector for Z
    - z0: (n,) feasible vector in the Z space
    - C: (n, k) transformation matrix
    - w0: (k,) reference vector in the w space

    Returns:
    - A_bar: (m, k) equivalent constraint matrix for w
    - b_bar: (m,) equivalent constraint vector for w
    """
    # From A @ (z0 - C @ w0 + C @ w) <= b
    # We expand to: A @ z0 - A @ C @ w0 + A @ C @ w <= b
    # Isolate w: (A @ C) @ w <= b - A @ z0 + A @ C @ w0

    A_bar = A @ C
    b_bar = b - A @ z0 + A_bar @ w0

    return A_bar, b_bar

# --- Example Usage ---

if __name__ == "__main__":
    print("--- Problem 1: Lasso Quadratic Form ---")
    n_lasso = 5
    # Generate random PSD matrix Q
    _tmp = np.random.randn(n_lasso, n_lasso)
    Q_val = _tmp.T @ _tmp + np.eye(n_lasso) * 0.1
    Z_val = np.random.randn(n_lasso)
    D_val = np.random.uniform(0.1, 1.0, n_lasso)

    # Test unpenalized variables by setting the first weight to 0
    D_val[0] = 0.0

    opt_beta, opt_val = solve_lasso_quadratic(Q_val, Z_val, D_val)
    print(f"Optimal Beta:\n{opt_beta}")
    print(f"Optimal Value: {opt_val:.4f}\n")

    print("--- Problem 2: Step Size Optimization ---")
    m, n_step = 10, 3
    A_val = np.random.randn(m, n_step)
    V_val = np.zeros(n_step) # Start at origin
    b_val = np.ones(m)      # Constraints are A @ x <= 1
    eta_val = np.random.randn(n_step)

    max_t, status_max = solve_step_size(A_val, V_val, b_val, eta_val, mode='max')
    min_t, status_min = solve_step_size(A_val, V_val, b_val, eta_val, mode='min')

    print(f"Direction eta: {eta_val}")
    print(f"Max t: {max_t} (Status: {status_max})")
    print(f"Min t: {min_t} (Status: {status_min})\n")

    print("--- Problem 3: Post-Selection Inference Polytope (Lee et al.) ---")
    # Reuse Q_val, Z_val, D_val from Problem 1
    A_poly, b_poly, active_set, signs = lasso_post_selection_constraints(Q_val, Z_val, D_val)

    print(f"Active Set M: {active_set}")
    print(f"Signs s_M: {signs}")
    print(f"Polytope constraints A shape: {A_poly.shape}, b shape: {b_poly.shape}")

    # Verify the current Z lies inside the polytope (A @ Z <= b)
    # We add a small tolerance for floating point inaccuracies from cvxpy
    is_inside = np.all(A_poly @ Z_val <= b_poly + 1e-6)
    print(f"Does the original Z satisfy A @ Z <= b? {is_inside}\n")

    print("--- Problem 4: Equivalent Affine Constraints for w ---")
    k = 2  # dimension of w
    C_val = np.random.randn(n_lasso, k)
    w0_val = np.random.randn(k)

    A_bar, b_bar = compute_affine_w_constraints(A_poly, b_poly, Z_val, C_val, w0_val)

    print(f"A_bar shape: {A_bar.shape}, b_bar shape: {b_bar.shape}")

    # Verify that w0 is inside the new polytope (A_bar @ w0 <= b_bar)
    # This must be true since Z_val is feasible and substituting w0 yields Z_val
    w0_is_feasible = np.all(A_bar @ w0_val <= b_bar + 1e-6)
    print(f"Is w0 feasible in the new constraint set A_bar @ w <= b_bar? {w0_is_feasible}")
```
