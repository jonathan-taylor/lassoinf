import numpy as np
from dataclasses import dataclass
from scipy.stats import norm
from functools import partial
from scipy.sparse.linalg import cg, LinearOperator

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

    def get_weight(self, v: np.ndarray, A: np.ndarray, b: np.ndarray):
        """
        Returns a function of t that computes the selection probability.
        """
        A = np.atleast_2d(A)
        b = np.atleast_1d(b).ravel()
        
        # Precompute parts that don't depend on t
        params_fixed = self.compute_params(v)
        bar_s = params_fixed['bar_s']
        
        alpha = A @ params_fixed['bar_gamma']
        beta_0 = b - A @ (params_fixed['n_o'] + params_fixed['bar_n_o'])
        beta_1 = -A @ params_fixed['gamma']
        
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
    
