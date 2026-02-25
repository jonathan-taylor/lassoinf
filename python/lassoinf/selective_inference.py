import numpy as np
from dataclasses import dataclass
from scipy.stats import norm
from functools import partial

@dataclass
class SelectiveInference:
    Z: np.ndarray
    Z_noisy: np.ndarray
    Q: np.ndarray        # Sigma
    Q_noise: np.ndarray  # Bar Sigma

    def compute_params(self, v: np.ndarray):
        """
        v is the contrast vector (eta in the doc).
        """
        # eta' * Sigma * eta
        v_sigma_v = v.T @ self.Q @ v
        
        # Gamma = Sigma * eta * (eta' * Sigma * eta)^-1
        gamma = (self.Q @ v) / v_sigma_v
        
        # c = BarSigma^-1 * Sigma * eta
        # Solving BarSigma * c = Sigma * eta
        c = np.linalg.solve(self.Q_noise, self.Q @ v)
        
        # bar_s^2 = c' * BarSigma * c = eta' * Sigma * BarSigma^-1 * Sigma * eta
        bar_s2 = c.T @ self.Q_noise @ c
        bar_s = np.sqrt(bar_s2)
        
        # bar_Gamma = (c' * BarSigma * c)^-1 * Cov(omega, c'omega)
        # Cov(omega, c'omega) = BarSigma * c = Sigma * eta
        # So bar_Gamma = (Sigma * eta) / bar_s2
        bar_gamma = (self.Q @ v) / bar_s2
        
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
        
        return lower, upper

    def get_weight(self, v: np.ndarray, A: np.ndarray, b: np.ndarray):
        """
        Returns a function of t that computes the selection probability.
        """
        # Precompute parts that don't depend on t
        params_fixed = self.compute_params(v)
        bar_s = params_fixed['bar_s']
        
        def weight_func(t, si_obj, v_vec, A_mat, b_vec, s_val):
            L, U = si_obj.get_interval(v_vec, t, A_mat, b_vec)
            if np.isnan(L) or np.isnan(U):
                return 0.0
            return norm.cdf(U / s_val) - norm.cdf(L / s_val)

        return partial(weight_func, si_obj=self, v_vec=v, A_mat=A, b_vec=b, s_val=bar_s)
