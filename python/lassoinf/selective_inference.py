import numpy as np
import scipy.sparse
from dataclasses import dataclass
from scipy.stats import norm
from functools import partial
from scipy.sparse.linalg import cg, LinearOperator
from .gaussian_family import WeightedGaussianFamily
from .bivariate_normal import TruncBivariateNormal

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

@dataclass
class SelectiveInference:
    Z: np.ndarray
    Z_noisy: np.ndarray
    Q: np.ndarray        # Sigma (can be np.ndarray or LinearOperator)
    Q_noise: np.ndarray  # Bar Sigma (can be np.ndarray or LinearOperator)

    def solve_contrast(self, v: np.ndarray) -> np.ndarray:
        """
        Computes c = BarSigma^-1 * Sigma * eta
        This isolates the linear system solve so it can be optimized.
        """
        # Right hand side: Q_v = Sigma * eta
        Q_v = self.Q @ v
        
        if isinstance(self.Q_noise, LinearOperator):
            # Iterative solve for Matrix-Free / Sparse operators
            c, info = cg(self.Q_noise, Q_v, rtol=1e-8)
            if info != 0:
                raise RuntimeError(f"Conjugate gradient did not converge (info={info})")
            return c
        else:
            # Direct solve for dense matrices
            return np.linalg.solve(self.Q_noise, Q_v)

    def compute_params(self, v: np.ndarray):
        """
        v is the contrast vector (eta in the doc).
        """
        Q_v = self.Q @ v
        
        # eta' * Sigma * eta
        v_sigma_v = v.T @ Q_v
        
        # Gamma = Sigma * eta * (eta' * Sigma * eta)^-1
        gamma = Q_v / v_sigma_v
        
        # c = BarSigma^-1 * Sigma * eta
        c = self.solve_contrast(v)
        
        # bar_s^2 = c' * BarSigma * c = eta' * Sigma * BarSigma^-1 * Sigma * eta
        Q_noise_c = self.Q_noise @ c
        bar_s2 = c.T @ Q_noise_c
        bar_s = np.sqrt(bar_s2)
        
        # bar_Gamma = (c' * BarSigma * c)^-1 * Cov(omega, c'omega)
        # Cov(omega, c'omega) = BarSigma * c = Sigma * eta
        # So bar_Gamma = (Sigma * eta) / bar_s2
        bar_gamma = Q_v / bar_s2
        
        # N_o = Z - Gamma * (v' * Z)
        theta_hat = v.T @ self.Z
        n_o = self.Z - gamma * theta_hat
        
        # bar_N_o = omega - bar_Gamma * (c' * omega)
        # omega = Z_noisy - Z
        omega = self.Z_noisy - self.Z
        bar_theta = c.T @ omega
        bar_n_o = omega - bar_gamma * bar_theta
        
        return {
            'gamma': gamma,
            'c': c,
            'bar_gamma': bar_gamma,
            'bar_s': bar_s,
            'n_o': n_o,
            'bar_n_o': bar_n_o,
            'theta_hat': theta_hat,
            'bar_theta': bar_theta
        }

    def data_splitting_estimator(self, v: np.ndarray):
        """
        Computes the data splitting estimator and its variance.
        """
        params = self.compute_params(v)
        variance = (v.T @ self.Q @ v) + params['bar_s']**2
        estimator = params['theta_hat'] - params['bar_theta']
        return estimator, variance

    def get_interval(self, v: np.ndarray, t: float, A: np.ndarray, b: np.ndarray):
        params = self.compute_params(v)
        
        # Selection event: A(Z + omega) <= b
        # Z + omega = (N_o + Gamma*theta_hat) + (bar_N_o + bar_Gamma*bar_theta)
        # We condition on N_o, bar_N_o, and theta_hat = t.
        # The condition becomes:
        # A(N_o + Gamma*t + bar_N_o + bar_Gamma*bar_theta) <= b
        # A * bar_Gamma * bar_theta <= b - A(N_o + bar_N_o + Gamma*t)
        
        alpha = A @ params['bar_gamma']
        beta = b - A @ (params['n_o'] + params['bar_n_o'] + params['gamma'] * t)
        
        # alpha * bar_theta <= beta
        # We want the interval [L, U] for bar_theta.
        # For each row i: alpha[i] * bar_theta <= beta[i]
        
        lower = -np.inf
        upper = np.inf
        
        for a_i, b_i in zip(alpha, beta):
            if a_i > 1e-10:
                upper = min(upper, b_i / a_i)
            elif a_i < -1e-10:
                lower = max(lower, b_i / a_i)
            elif b_i < -1e-10:
                # Infeasible
                return (np.nan, np.nan)
        
        if lower > upper:
            return (np.nan, np.nan)
            
        return lower, upper

    def get_weight(self, v: np.ndarray, A, b: np.ndarray):
        """
        Returns a function of t that computes the selection probability.
        """
        # Ensure A is handled as a linear operator if possible
        if not isinstance(A, LinearOperator) and hasattr(A, 'matvec'):
             # it might be our CompositeOperator but isinstance failed due to imports
             pass 

        b = np.atleast_1d(b).ravel()
        
        # Precompute parts that don't depend on t
        params_fixed = self.compute_params(v)
        bar_s = params_fixed['bar_s']
        
        bar_gamma = np.atleast_1d(params_fixed['bar_gamma'])
        n_o = np.atleast_1d(params_fixed['n_o'])
        bar_n_o = np.atleast_1d(params_fixed['bar_n_o'])
        gamma = np.atleast_1d(params_fixed['gamma'])

        if hasattr(A, 'matvec'):
            alpha = A.matvec(bar_gamma)
            beta_0 = b - A.matvec(n_o + bar_n_o)
            beta_1 = -A.matvec(gamma)
        else:
            A_arr = np.atleast_2d(np.asarray(A))
            alpha = A_arr.dot(bar_gamma)
            beta_0 = b - A_arr.dot(n_o + bar_n_o)
            beta_1 = -A_arr.dot(gamma)
        
        pos_mask = alpha > 1e-10
        neg_mask = alpha < -1e-10
        zero_mask = np.abs(alpha) <= 1e-10
        
        def weight_func(t):
            t_arr = np.asarray(t)
            is_scalar = t_arr.ndim == 0
            t_1d = np.atleast_1d(t_arr)
            
            # Compute beta for all t at once
            # beta shape: (len(t_1d), len(beta_0))
            beta = beta_0[None, :] + t_1d[:, None] * beta_1[None, :]
            
            L = np.full(t_1d.shape, -np.inf)
            U = np.full(t_1d.shape, np.inf)
            valid = np.ones(t_1d.shape, dtype=bool)
            
            if np.any(pos_mask):
                U = np.min(beta[:, pos_mask] / alpha[pos_mask], axis=1)
            if np.any(neg_mask):
                L = np.max(beta[:, neg_mask] / alpha[neg_mask], axis=1)
            if np.any(zero_mask):
                valid &= ~np.any(beta[:, zero_mask] < -1e-10, axis=1)
                
            valid &= (L <= U)
            
            prob = np.zeros_like(t_1d, dtype=float)
            if np.any(valid):
                prob[valid] = norm.cdf(U[valid] / bar_s) - norm.cdf(L[valid] / bar_s)
                prob[valid] = np.maximum(prob[valid], 0.0)
                
            return prob[0] if is_scalar else prob

        return weight_func

def lasso_post_selection_constraints(beta_hat, G, Q, D_diag, L=None, U=None, tol=1e-6):
    """
    Derives the linear constraints AZ <= b characterizing the polytope where
    the active set, signs, and bound-activations of the Lasso remain constant.
    Returns A as a CompositeOperator to support matrix-free operations.
    """
    from .operators.composite import CompositeOperator
    import scipy.sparse as sp

    n = Q.shape[0] if hasattr(Q, 'shape') else len(beta_hat)
    L_bound = np.full(n, -np.inf) if L is None else np.asarray(L)
    U_bound = np.full(n, np.inf) if U is None else np.asarray(U)

    E = []
    E_c = []
    s_E = []
    v_Ec = []
    g_min = []
    g_max = []

    for j in range(n):
        beta_val = beta_hat[j]
        at_L = (beta_val <= L_bound[j] + tol)
        at_U = (beta_val >= U_bound[j] - tol)
        at_0 = (abs(beta_val) <= tol)

        if not at_L and not at_U and not at_0:
            E.append(j)
            s_E.append(np.sign(beta_val))
        else:
            E_c.append(j)
            if at_0: v_j = 0.0
            elif at_U: v_j = U_bound[j]
            else: v_j = L_bound[j]
            v_Ec.append(v_j)

            dj = D_diag[j]
            gmin, gmax = -np.inf, np.inf
            if at_0:
                if L_bound[j] < -tol: gmin = -dj
                if U_bound[j] > tol:  gmax = dj
            elif at_U: gmin = dj
            elif at_L: gmax = -dj

            g_min.append(gmin)
            g_max.append(gmax)

    E = np.array(E, dtype=int)
    E_c = np.array(E_c, dtype=int)
    s_E = np.array(s_E)
    v_Ec = np.array(v_Ec)
    g_min = np.array(g_min)
    g_max = np.array(g_max)

    S_list, U_list, b_list = [], [], []

    if len(E) > 0:
        E_M = np.zeros((n, len(E)))
        for i, j in enumerate(E): E_M[j, i] = 1.0
        
        Q_E = Q @ E_M if not isinstance(Q, np.ndarray) else Q[:, E]
        Q_EE = Q_E[E, :]
        Q_EcE = Q_E[E_c, :]
        
        W = np.linalg.inv(Q_EE)
        
        c_E = W @ (Q_EcE.T @ v_Ec + D_diag[E] * s_E)
        
        U_list.append(-np.diag(s_E) @ W)
        S_list.append(sp.csr_matrix((len(E), n)))
        b_list.append(-np.diag(s_E) @ c_E)

        for k, j in enumerate(E):
            if s_E[k] == 1 and U_bound[j] < np.inf:
                U_list.append(W[k:k+1, :])
                S_list.append(sp.csr_matrix((1, n)))
                b_list.append(np.array([U_bound[j] + c_E[k]]))
            elif s_E[k] == -1 and L_bound[j] > -np.inf:
                U_list.append(-W[k:k+1, :])
                S_list.append(sp.csr_matrix((1, n)))
                b_list.append(np.array([-L_bound[j] - c_E[k]]))
    else:
        Q_EcE = np.zeros((len(E_c), 0))
        W = np.zeros((0, 0))
        c_E = np.zeros(0)

    if len(E_c) > 0:
        V_vec = np.zeros(n)
        V_vec[E_c] = v_Ec
        Q_V = Q @ V_vec if not isinstance(Q, np.ndarray) else Q @ V_vec
        Q_EcEc_v_Ec = Q_V[E_c]
        
        U_part = - Q_EcE @ W if len(E) > 0 else np.zeros((len(E_c), 0))
        c_Ec = Q_EcE @ c_E - Q_EcEc_v_Ec if len(E) > 0 else - Q_EcEc_v_Ec

        for k, j in enumerate(E_c):
            if g_max[k] < np.inf:
                U_list.append(U_part[k:k+1, :])
                row = sp.csr_matrix(([1.0], ([0], [j])), shape=(1, n))
                S_list.append(row)
                b_list.append(np.array([g_max[k] - c_Ec[k]]))
            if g_min[k] > -np.inf:
                U_list.append(-U_part[k:k+1, :])
                row = sp.csr_matrix(([-1.0], ([0], [j])), shape=(1, n))
                S_list.append(row)
                b_list.append(np.array([-g_min[k] + c_Ec[k]]))

    if not U_list:
        m = 0
        S_final = sp.csr_matrix((0, n))
        U_final = np.zeros((0, len(E)))
        b_final = np.zeros(0)
    else:
        S_final = sp.vstack(S_list)
        U_final = np.vstack(U_list)
        b_final = np.concatenate(b_list)
        m = S_final.shape[0]

    V_final = np.zeros((n, len(E)))
    for i, j in enumerate(E):
        V_final[j, i] = 1.0

    A = CompositeOperator((m, n), S=S_final, U=U_final, V=V_final)
    return A, b_final, E, E_c, s_E, v_Ec

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
    Sigma_noisy: np.ndarray

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
            
        self.A, self.b, self.E, self.E_c, self.s_E, self.v_Ec = lasso_post_selection_constraints(
            self.beta_hat, self.G_hat, self.Q_hat, self.D, self.L, self.U
        )
        
        self.Z_noisy = -self.G_hat + self.Q_hat @ self.beta_hat
        
        self.si = SelectiveInference(
            Z=self.Z_full,
            Z_noisy=self.Z_noisy,
            Q=self.Sigma,
            Q_noise=self.Sigma_noisy
        )
        
        self.intervals = {}
        # compute confidence intervals for the parameters using the "free" variables from the constraints
        if len(self.E) > 0:
            n = self.Q_hat.shape[0] if hasattr(self.Q_hat, 'shape') else len(self.beta_hat)
            
            # Reconstruct W (inverse of Q_EE)
            E_M = np.zeros((n, len(self.E)))
            for i, j in enumerate(self.E): 
                E_M[j, i] = 1.0
            
            if isinstance(self.Q_hat, np.ndarray):
                Q_EE = self.Q_hat[self.E][:, self.E]
            else:
                Q_E = self.Q_hat @ E_M
                Q_EE = Q_E[self.E, :]
                
            W = np.linalg.inv(Q_EE)
            
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
                sigma = np.sqrt(variance)
                
                # Use TruncBivariateNormal for exact inference
                params_fixed = self.si.compute_params(v)
                bar_s = float(params_fixed['bar_s'])
                
                # Get the interval bounds at theta_hat = 0
                L_0, U_0 = self.si.get_interval(v, 0.0, self.A, self.b)
                
                c1 = float(variance)
                c2 = bar_s**2
                
                # The constraint is L_0 <= (c2/c1) * theta_hat + bar_theta <= U_0
                a_coeff = c2 / c1
                b_coeff = 1.0
                
                tbn = TruncBivariateNormal(
                    a_coeff=a_coeff, b_coeff=b_coeff, 
                    L=L_0, U=U_0, 
                    sig_omega=bar_s, 
                    sig_x=sigma
                )
                
                # Compute the 95% confidence interval (in natural parameter space)
                L_theta, U_theta = tbn.equal_tailed_interval(float(theta_hat), alpha=0.05)
                lower, upper = L_theta * c1, U_theta * c1
                
                # Compute p-value for H0: theta = 0
                # H0: theta_true = 0 => theta_natural = 0
                cdf_val = np.clip(tbn.cdf(theta=0.0, x=float(theta_hat)), 0.0, 1.0)
                p_val = np.clip(2 * min(cdf_val, 1.0 - cdf_val), 0.0, 1.0)
                
                self.intervals[j] = (lower, upper, p_val)

    def summary(self):
        """
        Returns a summary of the inference results.
        """
        import pandas as pd
        
        indices = sorted(self.intervals.keys())
        beta_vals = [self.beta_hat[j] for j in indices]
        lowers = [self.intervals[j][0] for j in indices]
        uppers = [self.intervals[j][1] for j in indices]
        p_vals = [self.intervals[j][2] for j in indices]
        
        data = {
            'index': indices,
            'beta_hat': beta_vals,
            'lower_conf': lowers,
            'upper_conf': uppers,
            'p_value': p_vals
        }
        
        try:
            return pd.DataFrame(data).set_index('index')
        except ImportError:
            # Fallback to numpy array if pandas is not installed
            res = np.column_stack([indices, beta_vals, lowers, uppers, p_vals])
            return res

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

