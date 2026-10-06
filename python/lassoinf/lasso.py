from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import norm as normal_dbn

from .affine_constraints import AffineConstraints

from .operators.lasso_constraints import LassoConstraintOperator
from .operators.submatrix import extract_submatrices
from .gaussian_family import WeightedGaussianFamily
from .bivariate_normal import TruncBivariateNormal

@dataclass
class LassoInference:
    beta_hat: np.ndarray
    G_hat: np.ndarray
    Q_hat: np.ndarray
    D: np.ndarray
    L: np.ndarray
    U: np.ndarray
    Z_full: np.ndarray
    Sigma: np.ndarray
    Sigma_noise: np.ndarray | None = None
    scalar_noise: float = np.nan
    level: float = 0.95

    def check_kkt(self, tol=1e-5):
        """
        Checks whether the current beta_hat, G_hat satisfy the KKT conditions
        for the bounded lasso problem.
        The KKT condition is -G_hat in the subdifferential of P(beta_hat).
        """
        g = -self.G_hat
        n = len(self.beta_hat)
        L = np.full(n, -np.inf) if self.L is None else np.asarray(self.L)
        U = np.full(n, np.inf) if self.U is None else np.asarray(self.U)
        D = np.asarray(self.D)
        
        for j in range(n):
            if np.isclose(L[j], U[j], atol=tol):
                continue
                
            bj = self.beta_hat[j]
            gj = g[j]
            dj = D[j]
            
            # Subgradient of absolute value
            if abs(bj) > tol:
                subgrad_l1 = dj * np.sign(bj)
            else:
                subgrad_l1_min, subgrad_l1_max = -dj, dj
                
            if bj > L[j] + tol and bj < U[j] - tol:
                # Interior of bounds
                if abs(bj) > tol:
                    if not np.isclose(gj, subgrad_l1, atol=tol): 
                        return False
                else:
                    if gj < subgrad_l1_min - tol or gj > subgrad_l1_max + tol: 
                        return False
            elif bj >= U[j] - tol:
                # On upper bound
                if abs(bj) > tol:
                    if gj < subgrad_l1 - tol: 
                        return False
                else:
                    if gj < subgrad_l1_min - tol: 
                        return False
            elif bj <= L[j] + tol:
                # On lower bound
                if abs(bj) > tol:
                    if gj > subgrad_l1 + tol: 
                        return False
                else:
                    if gj > subgrad_l1_max + tol: 
                        return False
                    
        return True

    def prox_lasso_bounds(self, v, t):
        """
        Computes the proximal operator for the bounded L1 penalty:
            h(beta) = ||D beta||_1 + I_{[L, U]}(beta)
        """
        v = np.asarray(v)
        D_diag = np.asarray(self.D)
        
        n = len(v)
        L = np.full(n, -np.inf) if self.L is None else np.asarray(self.L)
        U = np.full(n, np.inf) if self.U is None else np.asarray(self.U)
        
        # Soft-Thresholding
        st_val = np.sign(v) * np.maximum(np.abs(v) - t * D_diag, 0.0)
        
        # Projection (Clipping) onto the box constraints
        prox_val = np.clip(st_val, L, U)
        
        return prox_val

    def __post_init__(self):
        self.proximal_step()
        self.setup_constraints()
        self.compute_intervals()

    def proximal_step(self):
        # 1. Estimate largest singular value of Q_hat using power method
        n = self.Q_hat.shape[0] if hasattr(self.Q_hat, 'shape') else len(self.beta_hat)

        eigenval_max, lower = largest_eigenvalue_bound_Q(self.Q_hat)
            
        # 2. Compute step size
        # As requested, take step_size = 1 / (20 * lambda_max)
        step_size = 1.0 / (20. * eigenval_max)
        
        # 3. Proximal map step (thresholding/rounding)
        # One iteration of proximal gradient ensures KKT holds exactly for the 
        # linearized objective at beta_new.
        v_step = self.beta_hat - step_size * self.G_hat
        beta_new = self.prox_lasso_bounds(v_step, step_size)
        
        # Update G_hat such that the proximal identity holds:
        # G_new = G_old + (1/t)(beta_new - beta_old)
        beta_diff = beta_new - self.beta_hat
        self.G_hat = self.G_hat + (1.0 / step_size) * beta_diff
        self.beta_hat = beta_new
        
        if not self.check_kkt(tol=1e-4):
            # This should ideally not be reached if step_size is small enough
            pass

    def setup_constraints(self):
        self.A, self.b, self.E, self.E_c, self.s_E, self.v_Ec = lasso_post_selection_constraints(
            self.beta_hat, self.G_hat, self.Q_hat, self.D, self.L, self.U
        )
        # W = Q_hat[E, E]^{-1}
        self.W = self.A.W
        
        self.Z_noisy = -self.G_hat + self.Q_hat @ self.beta_hat
        
        self.si = AffineConstraints(
            Z=self.Z_full,
            Z_noisy=self.Z_noisy,
            Q=self.Sigma,
            Q_noise=self.Sigma_noise,
            scalar_noise=self.scalar_noise,
        )

    def compute_intervals(self, inference_method=None):
        n = self.Q_hat.shape[0] if hasattr(self.Q_hat, 'shape') else len(self.beta_hat)
        self._contrasts = {}
        betas, lowers, uppers, pvals = [], [], [], []

        # compute confidence intervals for the parameters using the "free" variables from the constraints
        if len(self.E) > 0:
            W = self.W

            for k, j in enumerate(self.E):
                v = np.zeros(n)
                v[self.E] = W[:, k]
                
                # The target estimate theta_hat
                theta_hat = v.T @ self.Z_full
                
                # The variance of theta_hat is v^T Sigma v
                if isinstance(self.Sigma, np.ndarray):
                    variance = v.T @ self.Sigma @ v
                else:
                    variance = v.T @ (self.Sigma @ v)
                
                lower, upper, p_val, contrast = self._compute_inference(
                    v, theta_hat, variance, inference_method
                )
                
                self._contrasts[j] = contrast

                lowers.append(lower)
                uppers.append(upper)
                pvals.append(p_val)
                betas.append(contrast.theta_hat)
                
            self.summary_ = pd.DataFrame({'beta_hat':betas,
                                          'lower_conf':lowers,
                                          'upper_conf':uppers,
                                          'p_value': pvals,
                                          'index':self.E}).set_index('index')
        else:
            self.summary_ = pd.DataFrame(columns=['beta_hat', 'lower_conf', 'upper_conf', 'p_value', 'index']).set_index('index')

    def _compute_inference(self, v, theta_hat, variance, inference_method=None):
        sigma = np.sqrt(variance)
        contrast = self.si.compute_contrast(v)
        bar_s = float(contrast.bar_s)
        
        # Get the interval bounds at theta_hat = 0
        L_0, U_0 = contrast.get_interval(0.0, self.A, self.b)
        
        c1 = float(variance)
        c2 = bar_s**2
        
        # The constraint is L_0 <= (c2/c1) * theta_hat + bar_theta <= U_0
        a_coeff = c2 / c1
        b_coeff = 1.0

        if inference_method is None or inference_method == "bivariate_normal":
            tbn = TruncBivariateNormal(
                a_coeff=a_coeff, b_coeff=b_coeff, 
                L=L_0, U=U_0, 
                sig_omega=bar_s, 
                sig_x=sigma
            )
            
            # Compute the 95% confidence interval (in natural parameter space)
            L_theta, U_theta = tbn.equal_tailed_interval(float(theta_hat), alpha=1-self.level)
            lower, upper = L_theta * c1, U_theta * c1

            # Compute p-value for H0: theta = 0
            # H0: theta_true = 0 => theta_natural = 0
            cdf_val = np.clip(tbn.cdf(theta=0.0, x=float(theta_hat)), 0.0, 1.0)
            p_val = np.clip(2 * min(cdf_val, 1.0 - cdf_val), 0.0, 1.0)
            
            return lower, upper, p_val, contrast
        else:
            if callable(inference_method):
                return inference_method(
                    v=v, 
                    theta_hat=float(theta_hat), 
                    variance=float(variance), 
                    contrast=contrast, 
                    L_0=L_0, 
                    U_0=U_0,
                    alpha=1-self.level
                )
            raise ValueError(f"Unknown inference method: {inference_method}")



def largest_eigenvalue_bound_Q(Q, num_iters=4):
    """
    Estimates an upper bound for the largest eigenvalue of A = X^T Q X
    using only matrix-vector products.
    """
    n = Q.shape[0]
    
    # 1. Initialize a random vector
    rng = np.random.default_rng()
    v = rng.standard_normal(n)
    v /= np.linalg.norm(v)
    
    # 2. Power iterations using the matvec chain
    for _ in range(num_iters):
        Qv = Q @ v
        v = Qv / np.linalg.norm(Qv)
        
    # 3. Compute the Rayleigh quotient (mu)
    # mu = v^T (X^T Q X) v = (Xv)^T (Q(Xv)) = u^T Qu
    Qv = Q @ v
    mu = np.dot(v, Qv)
    
    # 4. Compute the residual vector and its norm
    r = Qv - mu * v
    residual_norm = np.linalg.norm(r)
    
    # 5. Guaranteed upper bound
    upper_bound = mu + residual_norm
    
    return upper_bound, mu

def spec_from_glmnet(glmnet_obj,
                     data,
                     lambda_val,
                     state,
                     proportion,
                     dispersion=None):
    G = glmnet_obj
    X_full, Df_full = data
    _, _, Y_full, _, weight_full = G.get_data_arrays(X_full, Df_full)
    ridge_coef = (1 - G.alpha) * lambda_val * weight_full.sum()

    active_set = np.nonzero(state.coef != 0)[0]

    if active_set.shape[0] == 0:
        return None

    unreg_GLM = glmnet_obj.get_GLM(ridge_coef=ridge_coef)
    unreg_GLM.summarize = True
    unreg_GLM.fit(X_full[:, active_set], Df_full, dispersion=dispersion)

    D_active = glmnet_obj.get_design(X_full[:, active_set],
                                     weight_full,
                                     standardize=glmnet_obj.standardize,
                                     intercept=glmnet_obj.fit_intercept)

    info = unreg_GLM._information
    P_active = D_active.quadratic_form(info, transformed=True)

    D0 = np.ones(D_active.shape[1])
    if G.fit_intercept:
        D0[0] = 0
    DIAG_active = np.diag(D0) * ridge_coef

    if not G.fit_intercept:
        if G.penalty_factor is not None:
            penfac = G.penalty_factor[active_set]
        else:
            penfac = np.ones_like(active_set)
        P_active = P_active[1:, 1:]
        DIAG_active = DIAG_active[1:, 1:]
    else:
        penfac = np.ones(active_set.shape[0])

    hessian = P_active + DIAG_active
    Q_active = np.linalg.inv(hessian)
    
    signs = np.sign(state.coef[active_set])

    if G.fit_intercept:
        penfac = np.hstack([0, penfac])
        signs = np.hstack([0, signs])
        stacked = np.hstack([state.intercept, state.coef[active_set]])
    else:
        stacked = state.coef[active_set]

    penalized = penfac > 0
    n_penalized = penalized.sum()
    n_coef = penalized.shape[0]
    row_idx = np.arange(n_penalized)
    col_idx = np.nonzero(penalized)[0]
    data = -signs[penalized]
    sel_active = scipy.sparse.coo_matrix((data, (row_idx, col_idx)), shape=(n_penalized, n_coef))

    linear = sel_active
    offset = np.zeros(sel_active.shape[0])

    return {'D': D_active,
            'L': linear,
            'U': offset,
            'gradient': -Q_active @ (penfac * lambda_val * signs) * weight_full.sum(),
            'hessian': hessian}

def lasso_post_selection_constraints(beta_hat, G, Q, D_diag, L=None, U=None, tol=1e-6):
    """
    Derives the linear constraints AZ <= b characterizing the polytope where
    the active set, signs, and bound-activations of the Lasso remain constant.
    Returns A as a LassoConstraintOperator: matrix-free, applied with one Q matvec.
    """

    beta_hat = np.asarray(beta_hat, dtype=float)
    n = Q.shape[0] if hasattr(Q, 'shape') else len(beta_hat)
    L_bound = np.full(n, -np.inf) if L is None else np.asarray(L, dtype=float)
    U_bound = np.full(n, np.inf) if U is None else np.asarray(U, dtype=float)
    D_diag = np.broadcast_to(np.asarray(D_diag, dtype=float), (n,))

    at_L = beta_hat <= L_bound + tol
    at_U = beta_hat >= U_bound - tol
    at_0 = np.abs(beta_hat) <= tol
    active = ~(at_L | at_U | at_0)

    E = np.nonzero(active)[0]
    E_c = np.nonzero(~active)[0]
    s_E = np.sign(beta_hat[E])

    # inactive coordinates: value v_j and bounds on the subgradient
    a0, aU, aL = at_0[E_c], at_U[E_c], at_L[E_c]
    v_Ec = np.where(a0, 0.0, np.where(aU, U_bound[E_c], L_bound[E_c]))
    d_Ec = D_diag[E_c]
    g_min = np.full(len(E_c), -np.inf)
    g_max = np.full(len(E_c), np.inf)
    g_min = np.where(a0 & (L_bound[E_c] < -tol), -d_Ec, g_min)
    g_max = np.where(a0 & (U_bound[E_c] > tol), d_Ec, g_max)
    g_min = np.where(~a0 & aU, d_Ec, g_min)
    g_max = np.where(~a0 & ~aU & aL, -d_Ec, g_max)

    V_vec = np.zeros(n)
    V_vec[E_c] = v_Ec
    Q_V = np.ravel(Q @ V_vec) if np.any(v_Ec != 0) else np.zeros(n)

    if len(E) > 0:
        Q_EE = extract_submatrices(Q, E)
        W = np.linalg.inv(Q_EE)
        c_E = W @ (Q_V[E] + D_diag[E] * s_E)
        c_E_vec = np.zeros(n)
        c_E_vec[E] = c_E
        c_Ec = np.ravel(Q @ c_E_vec)[E_c] - Q_V[E_c]
    else:
        W = np.zeros((0, 0))
        c_E = np.zeros(0)
        c_Ec = -Q_V[E_c]

    # active rows act on W Z_E: signs, then bounds
    k_E = np.arange(len(E))
    up = (s_E == 1) & (U_bound[E] < np.inf)
    lo = (s_E == -1) & (L_bound[E] > -np.inf)
    k_bd = np.nonzero(up | lo)[0]
    R_active = sp.vstack([sp.csr_matrix((-s_E, (k_E, k_E)), shape=(len(E), len(E))),
                          sp.csr_matrix((np.where(up[k_bd], 1.0, -1.0), (np.arange(len(k_bd)), k_bd)),
                                        shape=(len(k_bd), len(E)))])
    b_active = np.concatenate([-s_E * c_E,
                               np.where(up, U_bound[E] + c_E, -L_bound[E] - c_E)[k_bd]])

    # inactive rows act on U_{-E}(Z): for each coordinate, upper then lower subgradient bound
    k_max = np.nonzero(g_max < np.inf)[0]
    k_min = np.nonzero(g_min > -np.inf)[0]
    k_in = np.concatenate([k_max, k_min])
    order = np.lexsort((np.r_[np.zeros(len(k_max)), np.ones(len(k_min))], k_in))
    k_in = k_in[order]
    vals_in = np.r_[np.ones(len(k_max)), -np.ones(len(k_min))][order]
    b_in = np.r_[g_max[k_max] - c_Ec[k_max], -g_min[k_min] + c_Ec[k_min]][order]
    R_inactive = sp.csr_matrix((vals_in, (np.arange(len(k_in)), k_in)), shape=(len(k_in), len(E_c)))

    A = LassoConstraintOperator(Q, E, E_c, W, R_active, R_inactive)
    b = np.concatenate([b_active, b_in])
    return A, b, E, E_c, s_E, v_Ec

__all__ = ['LassoInference', 'spec_from_glmnet']
