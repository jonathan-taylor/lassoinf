from dataclasses import dataclass
from functools import partial
import warnings

import numpy as np
import scipy.sparse
from scipy.stats import norm as normal_dbn
from scipy.sparse.linalg import cg, LinearOperator

@dataclass
class AffineConstraintsContrast:

    theta_hat: float
    bar_theta: float
    splitting_variance: float
    splitting_estimator: float
    naive_variance: float
    gamma: np.ndarray
    c: np.ndarray
    bar_gamma: np.ndarray
    bar_s: float
    n_o: np.ndarray
    bar_n_o: np.ndarray

    def get_interval(self, t: float, A: np.ndarray, b: np.ndarray):
        
        # Selection event: A(Z + omega) <= b
        # Z + omega = (N_o + Gamma*theta_hat) + (bar_N_o + bar_Gamma*bar_theta)
        # We condition on N_o, bar_N_o, and theta_hat = t.
        # The condition becomes:
        # A(N_o + Gamma*t + bar_N_o + bar_Gamma*bar_theta) <= b
        # A * bar_Gamma * bar_theta <= b - A(N_o + bar_N_o + Gamma*t)
        
        alpha = A @ self.bar_gamma
        beta = b - A @ (self.n_o + self.bar_n_o + self.gamma * t)
        
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

    def get_weight(self, A, b: np.ndarray):
        """
        Returns a function of t that computes the selection probability.
        """
        # # Ensure A is handled as a linear operator if possible
        # if not isinstance(A, LinearOperator) and hasattr(A, 'matvec'):
        #      # it might be our CompositeOperator but isinstance failed due to imports
        #      pass 

        b = np.atleast_1d(b).ravel()
        
        # Precompute parts that don't depend on t
        bar_s = self.bar_s
        
        bar_gamma = np.atleast_1d(self.bar_gamma)
        n_o = np.atleast_1d(self.n_o)
        bar_n_o = np.atleast_1d(self.bar_n_o)
        gamma = np.atleast_1d(self.gamma)

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
                prob[valid] = normal_dbn.cdf(U[valid] / bar_s) - normal_dbn.cdf(L[valid] / bar_s)
                prob[valid] = np.maximum(prob[valid], 0.0)
                
            return prob[0] if is_scalar else prob

        return weight_func


@dataclass
class AffineConstraints:
    Z: np.ndarray
    Z_noisy: np.ndarray
    Q: np.ndarray        # Sigma (can be np.ndarray or LinearOperator)
    Q_noise: np.ndarray  # Bar Sigma (can be np.ndarray or LinearOperator)
    scalar_noise: float = np.nan # if > 0 and Q_noise is None, it is assumed that Q_noise = scalar_noise * Q
    
    def __post_init__(self):
        if self.Q_noise is None:
            if not self.scalar_noise >= 0:
                raise ValueError('if Q_noise is None, scalar_noise must be >= 0')
            if self.scalar_noise is None or self.scalar_noise < 0.001:
                warnings.warn('For numerical stability using scalar_noise=0.001')
                self.scalar_noise = 0.001
                
    def solve_contrast(self, v: np.ndarray) -> np.ndarray:
        """
        Computes c = BarSigma^-1 * Sigma * eta
        This isolates the linear system solve so it can be optimized.
        """
        if self.Q_noise is not None:
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
        else:
            return v / self.scalar_noise
                

    def compute_contrast(self, v: np.ndarray):
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
        if self.Q_noise is not None:
            Q_noise_c = Q_v # self.Q_noise @ c
        else:
            Q_noise_c = Q_v # self.scalar_noise * self.Q @ c

        bar_s2 = c.T  @ Q_noise_c
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
        
        naive_var = (v.T @ self.Q @ v)

        result = AffineConstraintsContrast(theta_hat=theta_hat,
                                           gamma=gamma,
                                           c=c,
                                           bar_gamma=bar_gamma,
                                           bar_s=bar_s,
                                           n_o=n_o,
                                           bar_n_o=bar_n_o,
                                           bar_theta=bar_theta,
                                           splitting_variance=naive_var + bar_s**2,
                                           splitting_estimator=theta_hat - bar_theta,
                                           naive_variance=naive_var)

        return result





