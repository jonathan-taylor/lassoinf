"""
Selective inference after the uniLasso, or unireg (its lam = 0 case), for the
coefficients of the full model (n > p).

The uniLasso fits univariate regressions and then a lasso with penalty factors
1 / |beta_uni_j| and the sign constraints sign(beta_j) in {0, sign(beta_uni_j)}. In score
coordinates (Z = X'y, Q = X'X) with selection data Y = Z + omega, it solves

    minimize 1/2 beta' Q beta - beta' Y + sum_j D_j |beta_j|,  D_j = lam / |C_j|,
    subject to sign(beta_j) in {0, s_j},

with C = Y / diag(Q) the univariate coefficients and s = sign(C). The penalties depend on
the data, so the selection event is A Y <= b(D(Y)) rather than a polyhedron.

For the full-model coefficient theta_k = e_k' Q^{-1} E[Z] in the well-specified model
(Sigma = sigma^2 Q), Cov(Z, theta_hat_k) = sigma^2 e_k, so along the line the polyhedral
lemma conditions on, only C_k moves. Every constraint row is then a single ratio in D_k,
handled by RatioConstraints. See docs/data_dependent_penalty.md.

Unireg is lam = 0: least squares with the uniLasso's sign constraints. There is no
penalty, but the sign constraints still depend on the data, through the sign of C_k.

unilasso_inference takes a uniLasso fit, e.g. from the R package uniLasso with
loo = FALSE, which solves exactly this problem. Its default loo = TRUE regresses y on
leave-one-out univariate fits, a different selection event that this does not cover.
"""
from dataclasses import dataclass
import warnings

import numpy as np
import pandas as pd

from .lasso import LassoInference, lasso_post_selection_constraints
from .ratio_constraints import RatioConstraints, ratio_contrast_inference


def unilasso_penalty(Z_noisy, Q, lam):
    """
    The uniLasso's univariate coefficients C, signs s, penalty weights D = lam / |C| and
    bounds (L, U) for the selection data Z_noisy.
    """
    C = np.asarray(Z_noisy, dtype=float) / np.diag(Q)
    s = np.sign(C)
    if np.any(s == 0):
        raise ValueError('a univariate coefficient is exactly zero')
    D = lam / np.abs(C)
    L = np.where(s > 0, 0., -np.inf)
    U = np.where(s > 0, np.inf, 0.)
    return C, s, D, L, U


def unilasso_fit(Q, Z_noisy, lam, tol=1e-12, max_iter=100000):
    """
    The uniLasso solution for the selection data Z_noisy, by coordinate descent.
    """
    Q = np.asarray(Q, dtype=float)
    Y = np.asarray(Z_noisy, dtype=float)
    _, _, D, L, U = unilasso_penalty(Y, Q, lam)
    dq = np.diag(Q)
    beta = np.zeros(len(Y))
    for _ in range(max_iter):
        delta = 0.
        for j in range(len(Y)):
            r = Y[j] - Q[j] @ beta + dq[j] * beta[j]
            new = np.clip(np.sign(r) * max(abs(r) - D[j], 0.) / dq[j], L[j], U[j])
            delta = max(delta, abs(new - beta[j]) * np.sqrt(dq[j]))
            beta[j] = new
        if delta < tol:
            break
    return beta


def unilasso_kkt_violation(beta_hat, Q, Z_noisy, lam):
    """
    The largest violation of the uniLasso's KKT conditions by beta_hat, for the
    selection data Z_noisy and lam in the units of Z_noisy.
    """
    C, s, D, _, _ = unilasso_penalty(Z_noisy, Q, lam)
    beta_hat = np.asarray(beta_hat, dtype=float)
    g = np.asarray(Z_noisy, dtype=float) - np.asarray(Q) @ beta_hat   # minus the gradient
    active = beta_hat != 0
    v = np.r_[np.abs(g[active] - D[active] * s[active]),
              np.maximum(s[~active] * g[~active] - D[~active], 0.)]
    return float(v.max()) if v.size else 0.


@dataclass
class UniLassoInference(LassoInference):
    """
    Selective inference for the full-model coefficients of the variables the uniLasso
    selects. Construct with :meth:`from_selection`.

    summary_ has, for each selected k, the full-model least squares coefficient
    (Q^{-1} Z_full)_k with its selective interval and p-value.
    """
    lam: float = None
    C: np.ndarray = None   # univariate coefficients from the selection data

    @classmethod
    def from_selection(cls,
                       beta_hat,
                       Z_noisy,
                       Q_hat,
                       lam,
                       Z_full,
                       Sigma,
                       Sigma_noise=None,
                       scalar_noise=np.nan,
                       level=0.95):
        """
        beta_hat : the uniLasso solution for the selection data Z_noisy (e.g. unilasso_fit)
        Z_noisy : selection data, X'y + omega
        Q_hat : X'X, invertible (n > p)
        lam : the uniLasso's lambda, in the units of Z
        Z_full : X'y
        Sigma : Var(Z_full), sigma^2 X'X in the well-specified model
        Sigma_noise, scalar_noise : the randomization, as for LassoInference
        """
        Z_noisy = np.asarray(Z_noisy, dtype=float)
        Q_hat = np.asarray(Q_hat, dtype=float)
        C, _, D, L, U = unilasso_penalty(Z_noisy, Q_hat, lam)
        beta_hat = np.asarray(beta_hat, dtype=float)
        return cls(beta_hat=beta_hat,
                   G_hat=Q_hat @ beta_hat - Z_noisy,
                   Q_hat=Q_hat,
                   D=D,
                   L=L,
                   U=U,
                   Z_full=np.asarray(Z_full, dtype=float),
                   Sigma=Sigma,
                   Sigma_noise=Sigma_noise,
                   scalar_noise=scalar_noise,
                   level=level,
                   lam=lam,
                   C=C)

    def setup_constraints(self):
        # without randomization the selection score is the data (the polyhedral case, as in
        # glmnet_inference): after the proximal step, take Z_full to be it
        if self.Sigma_noise is None and self.scalar_noise == 0:
            self.Z_full = -self.G_hat + self.Q_hat @ self.beta_hat
        super().setup_constraints()

    def ratio_constraints(self, k):
        """
        The selection event as RatioConstraints along the line for the target of
        coordinate k: every row is (A Y)_i <= b0_i + M_ik D_k, D_k = lam s_k / C_k.
        """
        D = np.asarray(self.D, dtype=float)
        b_of = lambda D: lasso_post_selection_constraints(self.beta_hat, self.G_hat, self.Q_hat,
                                                          D, self.L, self.U)[1]
        # b is affine in D: its coefficient on D_k (any step works; D_k = 0 for unireg)
        step = D[k] if D[k] > 0 else 1.
        M_k = (b_of(D + step * np.eye(len(D))[k]) - self.b) / step
        b0 = self.b - M_k * D[k]
        s_k = np.sign(self.C[k])
        # C_k = Y_k / Q_kk, so Cov(C_k, Z) = Sigma[k] / Q_kk
        Sigma_k = np.ravel(self.Sigma @ np.eye(len(D))[k]) / self.Q_hat[k, k]
        m = len(self.b)
        return RatioConstraints(A=self.A,
                                B=b0 * self.C[k] + M_k * self.lam * s_k,
                                C=np.full(m, self.C[k]),
                                s_B=np.zeros(m),
                                s_C=np.full(m, s_k),
                                cov_BZ=np.outer(b0, Sigma_k),
                                cov_CZ=np.tile(Sigma_k, (m, 1)))

    def compute_intervals(self, inference_method=None):
        if self.lam is None or self.C is None:
            raise ValueError('construct UniLassoInference with from_selection')
        p = len(self.beta_hat)
        Q = self.Q_hat
        betas, lowers, uppers, pvals = [], [], [], []
        self._contrasts = {}
        self._results = {}
        for k in self.E:
            # full-model target theta_k = e_k' Q^{-1} E[Z]
            eta = np.linalg.solve(Q, np.eye(p)[k])
            result = ratio_contrast_inference(self.si, self.ratio_constraints(k), eta, level=self.level)
            self._results[k] = result
            betas.append(result.theta_hat)
            lowers.append(result.lower_conf)
            uppers.append(result.upper_conf)
            pvals.append(result.p_value)
        self.summary_ = pd.DataFrame({'beta_hat': betas,
                                      'lower_conf': lowers,
                                      'upper_conf': uppers,
                                      'p_value': pvals,
                                      'index': list(self.E)}).set_index('index')


def unilasso_inference(X,
                       y,
                       beta_hat,
                       lam,
                       intercept=True,
                       sigma2=None,
                       level=0.95,
                       kkt_tol=1e-3):
    """
    Selective inference after a uniLasso (or unireg, lam = 0) fit on (X, y), for the
    full-model coefficients of the selected variables, in a Gaussian linear model with n > p.

    The fit is assumed to solve

        minimize (1/2n) ||y - b0 - X beta||^2 + lam sum_j |beta_j| / |b_uni_j|
        subject to sign(beta_j) in {0, sign(b_uni_j)},

    with b_uni_j the univariate regression slopes of y on X_j (with intercepts if intercept).
    This is the uniLasso of the R package uniLasso with loo = FALSE, and lam is its lambda.
    The default loo = TRUE selects differently and is not covered.

    X, y : the data the fit used
    beta_hat : the fit's coefficients (without the intercept)
    lam : the fit's lambda, in glmnet's scaling (loss divided by n)
    intercept : whether the fit (and the univariate regressions) have intercepts
    sigma2 : noise variance; default the residual variance of the full least squares fit
    kkt_tol : warn if beta_hat violates the KKT conditions by more than kkt_tol * n * lam

    There is no randomization: this is the polyhedral approach. Returns a
    UniLassoInference; its summary_ has the full-model least squares coefficients of the
    selected variables, with selective intervals and p-values.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    beta_hat = np.asarray(beta_hat, dtype=float)
    n, p = X.shape
    df = n - p - int(intercept)
    if df <= 0:
        raise ValueError('unilasso_inference requires n > p')
    if intercept:
        X = X - X.mean(0)
        y = y - y.mean()
    Q = X.T @ X
    Z = X.T @ y
    lam_Z = n * lam
    C, s, _, _, _ = unilasso_penalty(Z, Q, lam_Z)
    if np.any(beta_hat * s < 0):
        raise ValueError('beta_hat has a sign opposite to its univariate coefficient: '
                         'it is not a uniLasso fit for these data')
    scale = lam_Z if lam > 0 else np.abs(Z).max()
    violation = unilasso_kkt_violation(beta_hat, Q, Z, lam_Z)
    if violation > kkt_tol * scale:
        unit = 'n lam' if lam > 0 else "max |X'y|"
        warnings.warn(f'beta_hat violates the uniLasso KKT conditions by {violation / scale:.1e} * '
                      f'{unit}; it may not solve the problem with '
                      'penalty factors 1 / |b_uni| (e.g. a loo = TRUE fit), or may not have '
                      'converged (tighten the threshold)')
    if sigma2 is None:
        resid = y - X @ np.linalg.solve(Q, Z)
        sigma2 = resid @ resid / df
    return UniLassoInference.from_selection(beta_hat=beta_hat,
                                            Z_noisy=Z,
                                            Q_hat=Q,
                                            lam=lam_Z,
                                            Z_full=Z,
                                            Sigma=sigma2 * Q,
                                            scalar_noise=0.,
                                            level=level)


__all__ = ['UniLassoInference', 'unilasso_inference', 'unilasso_fit', 'unilasso_penalty',
           'unilasso_kkt_violation']
