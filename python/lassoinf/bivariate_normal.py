import numpy as np
from scipy.stats import norm, multivariate_normal
from .discrete_family import discrete_family

def bivariate_normal_cdf(h, k, rho):
    """
    Computes P(Z1 <= h, Z2 <= k) with correlation rho.
    """
    if abs(rho) < 1e-12:
        return norm.cdf(h) * norm.cdf(k)
    if rho > 0.999999:
        return norm.cdf(min(h, k))
    if rho < -0.999999:
        return max(0, norm.cdf(h) + norm.cdf(k) - 1)

    return multivariate_normal.cdf([h, k], mean=[0, 0], cov=[[1, rho], [rho, 1]])


def compute_gaussian_conditional_stats(mu_x, sig_x, sig_omega, cx, comega, a, b, t=None):
    """
    Computes Probability, Mean, and Variance of X
    given the constraint: a <= cx*X + comega*Omega <= b.

    X ~ N(mu_x, sig_x^2)
    Omega ~ N(0, sig_omega^2)

    Returns a dictionary of results.
    """
    # 1. Properties of the constraint variable S = cx*X + comega*Omega
    var_x = sig_x**2
    var_omega = sig_omega**2

    mu_s = cx * mu_x
    var_s = (cx**2 * var_x) + (comega**2 * var_omega)
    sig_s = np.sqrt(var_s)

    # Covariance and correlation between X and S
    cov_xs = cx * var_x
    rho = cov_xs / (sig_x * sig_s)

    # 2. Denominator: P(a <= S <= b)
    # Standardize the bounds for S
    alpha = (a - mu_s) / sig_s if a != -np.inf else -np.inf
    beta = (b - mu_s) / sig_s if b != np.inf else np.inf

    p_constraint = (norm.cdf(beta) if beta != np.inf else 1.0) - \
                   (norm.cdf(alpha) if alpha != -np.inf else 0.0)

    if p_constraint < 1e-15:
        return {"error": "The constraint interval has negligible probability."}

    # 3. Conditional Probability: P(X > t | a <= S <= b)
    prob_gt_t = None
    if t is not None:
        # We need P(X > t AND a <= S <= b) / P(a <= S <= b)
        # Let Z_x = (X - mu_x)/sig_x. X > t => Z_x > (t - mu_x)/sig_x
        # For CDF function, we use P(-Z_x <= -(t - mu_x)/sig_x)
        h_val = -(t - mu_x) / sig_x

        # New correlation with -Z_x is -rho
        p_num_high = bivariate_normal_cdf(h_val, beta, -rho) if beta != np.inf else norm.cdf(h_val)
        p_num_low = bivariate_normal_cdf(h_val, alpha, -rho) if alpha != -np.inf else 0.0

        prob_gt_t = (p_num_high - p_num_low) / p_constraint

    # 4. Conditional Mean: E[X | S in [a, b]]
    # Using the truncated normal mean formula for S and linear regression
    phi_alpha = norm.pdf(alpha) if alpha != -np.inf else 0
    phi_beta = norm.pdf(beta) if beta != np.inf else 0

    # Ratio term for first moment of truncated S
    # E[S|S in [a,b]] = mu_s + sig_s * (phi(alpha) - phi(beta)) / P(a<S<b)
    ratio_mean = (phi_alpha - phi_beta) / p_constraint
    mean_x_cond = mu_x + (cov_xs / sig_s) * ratio_mean

    # 5. Conditional Variance: Var(X | S in [a, b])
    # Law of Total Variance: Var(X|S) + Var(E[X|S] | S in [a,b])
    # Term 1: Residual Variance Var(X|S) = sig_x^2 * (1 - rho^2)
    residual_var = var_x * (1 - rho**2)

    # Term 2: Variance of the regression component
    # Var(S|S in [a,b]) = sig_s^2 * [1 + (alpha*phi(alpha) - beta*phi(beta))/P - ratio_mean^2]
    term_alpha = alpha * phi_alpha if alpha != -np.inf else 0
    term_beta = beta * phi_beta if beta != np.inf else 0
    ratio_var = (term_alpha - term_beta) / p_constraint

    var_s_cond = var_s * (1 + ratio_var - ratio_mean**2)
    explained_var_cond = (cov_xs / var_s)**2 * var_s_cond

    var_x_cond = residual_var + explained_var_cond

    return {
        "params": {
            "mu_x": mu_x, "sig_x": sig_x, "sig_omega": sig_omega,
            "cx": cx, "comega": comega, "a": a, "b": b, "t": t
        },
        "stats": {
            "p_constraint": p_constraint,
            "p_x_gt_t_cond": prob_gt_t,
            "e_x_cond": mean_x_cond,
            "var_x_cond": var_x_cond,
            "std_x_cond": np.sqrt(var_x_cond),
            "rho_xs": rho
        }
    }

class TruncBivariateNormal(discrete_family):
    def __init__(self, a_coeff, b_coeff, L, U, sig_omega, sig_x=1.0, theta=0.):
        """
        TruncBivariateNormal models a continuous random variable Z ~ N(theta, sig_x^2)
        conditioned on the event L <= a_coeff * Z + b_coeff * omega <= U,
        where omega ~ N(0, sig_omega^2).
        
        This overrides the discrete_family methods to use exact continuous formulas
        from gaussian_conditional_moments.py.
        """
        self.a_coeff = a_coeff
        self.b_coeff = b_coeff
        self.L = L
        self.U = U
        self.sig_omega = sig_omega
        self.sig_x = sig_x

        # We call super().__init__ with dummy grid to bypass discrete_family initialization
        super().__init__(np.array([0.0]), np.array([1.0]), theta)

    def _get_stats(self, theta, x=None):
        return compute_gaussian_conditional_stats(
            mu_x=theta, sig_x=self.sig_x, sig_omega=self.sig_omega,
            cx=self.a_coeff, comega=self.b_coeff, a=self.L, b=self.U, t=x
        )

    def ccdf(self, theta, x=None, gamma=0, return_unnorm=False):
        if x is None:
            raise NotImplementedError("ccdf requires an observation x for TruncBivariateNormal")
        stats = self._get_stats(theta, x)
        if "error" in stats:
            return np.nan
        return stats['stats']['p_x_gt_t_cond']

    def cdf(self, theta, x=None, gamma=1):
        if x is None:
            raise NotImplementedError("cdf requires an observation x for TruncBivariateNormal")
        ccdf_val = self.ccdf(theta, x)
        return 1.0 - ccdf_val if not np.isnan(ccdf_val) else np.nan

    def E(self, theta, func):
        dummy = np.array([1.0, 2.0])
        try:
            val = func(dummy)
            is_identity = np.allclose(val, dummy)
        except Exception:
            is_identity = False

        if is_identity:
            stats = self._get_stats(theta)
            if "error" in stats:
                return np.nan
            return stats['stats']['e_x_cond']
        else:
            raise NotImplementedError("TruncBivariateNormal.E only supports the identity function")

    def Var(self, theta, func):
        dummy = np.array([1.0, 2.0])
        try:
            val = func(dummy)
            is_identity = np.allclose(val, dummy)
        except Exception:
            is_identity = False

        if is_identity:
            stats = self._get_stats(theta)
            if "error" in stats:
                return np.nan
            return stats['stats']['var_x_cond']
        else:
            raise NotImplementedError("TruncBivariateNormal.Var only supports the identity function")
