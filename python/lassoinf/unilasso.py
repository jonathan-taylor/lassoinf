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
solved in closed form (a quadratic per row). See docs/data_dependent_penalty.md.

Unireg is lam = 0: least squares with the uniLasso's sign constraints. There is no
penalty, but the sign constraints still depend on the data, through the sign of C_k.

unilasso_inference takes a uniLasso fit, e.g. from the R package uniLasso or the Python
package unilasso. With loo = False these solve exactly this problem. With loo = True (their
default) they regress y on leave-one-out univariate fits instead; that solution solves the
problem above with penalties (n lam + kappa_j) / |b_uni_j|, kappa_j computed by
unilasso_loo_kappa, and the inference uses those penalties. kappa_j depends on the data,
which the inference ignores, so it is approximate. See R's vignette("unilasso_loo").

The computations (the fit, the KKT check, kappa_j and the truncation sets) are in the C++
library (cpp/src/unilasso.cpp), shared with the R package.
"""
from dataclasses import dataclass
import warnings

import numpy as np
import pandas as pd

import lassoinf_cpp as _cpp

from .lasso import LassoInference
from .bivariate_normal import TruncBivariateNormal


def _lam(lam):
    return np.atleast_1d(np.asarray(lam, dtype=float))


def unilasso_penalty(Z_noisy, Q, lam):
    """
    The uniLasso's univariate coefficients C, signs s, penalty weights D = lam / |C| and
    bounds (L, U) for the selection data Z_noisy.
    """
    C = np.asarray(Z_noisy, dtype=float) / np.diag(Q)
    try:
        pen = _cpp.unilasso_penalty(C, _lam(lam))
    except ValueError as e:
        raise ValueError(str(e)) from None
    return pen.C, pen.s, pen.D, pen.L, pen.U


def unilasso_fit(Q, Z_noisy, lam, tol=1e-12, max_iter=100000):
    """
    The uniLasso solution for the selection data Z_noisy, by coordinate descent.
    """
    return _cpp.unilasso_fit(np.asarray(Q, dtype=float), np.asarray(Z_noisy, dtype=float),
                             _lam(lam), tol, max_iter)


def unilasso_kkt_violation(beta_hat, Q, Z_noisy, lam):
    """
    The largest violation of the uniLasso's KKT conditions by beta_hat, for the
    selection data Z_noisy and lam in the units of Z_noisy.
    """
    return _cpp.unilasso_kkt_violation(np.asarray(beta_hat, dtype=float),
                                       np.asarray(Q, dtype=float),
                                       np.asarray(Z_noisy, dtype=float), _lam(lam))


@dataclass
class UniLassoContrastResult:
    theta_hat: float
    lower_conf: float
    upper_conf: float
    p_value: float
    intervals: np.ndarray
    family: TruncBivariateNormal
    variance: float

    def pivot(self, theta):
        """P(theta_hat' <= theta_hat | selection) when the target equals theta."""
        return self.family.cdf(theta / self.variance, self.theta_hat)


def _contrast_inference(contrast, intervals, level):
    # theta_hat and bar_omega are independent Gaussians, truncated to
    # (bar_s^2 / sigma^2) theta_hat + bar_omega in the union of intervals
    if len(intervals) == 0:
        raise ValueError('the observed data do not satisfy the selection event')
    variance = float(contrast.naive_variance)
    bar_s = float(contrast.bar_s)
    theta_hat = float(contrast.theta_hat)
    family = TruncBivariateNormal(a_coeff=bar_s**2 / variance, b_coeff=1.,
                                  L=intervals[:, 0], U=intervals[:, 1],
                                  sig_omega=bar_s, sig_x=np.sqrt(variance))
    L_theta, U_theta = family.equal_tailed_interval(theta_hat, alpha=1 - level)
    cdf0 = np.clip(family.cdf(0., theta_hat), 0., 1.)
    return UniLassoContrastResult(theta_hat=theta_hat,
                                  lower_conf=L_theta * variance,
                                  upper_conf=U_theta * variance,
                                  p_value=float(np.clip(2 * min(cdf0, 1 - cdf0), 0., 1.)),
                                  intervals=intervals,
                                  family=family,
                                  variance=variance)


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
        lam : the uniLasso's lambda, in the units of Z; an array gives each feature its own
            penalty lam[j] / |C[j]|
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

    def truncation_set(self, k, contrast=None):
        """
        The truncation set for the target of coordinate k, as an array (j, 2) of disjoint
        intervals for w (equivalently bar_omega at theta_hat = 0).

        b is affine in D, and along the line only D_k = lam_k s_k / C_k moves, so every row is
        (A Y)_i <= b0_i + M_ik lam_k s_k / C_k.
        """
        p = len(self.beta_hat)
        e_k = np.eye(p)[k]
        if contrast is None:
            contrast = self.si.compute_contrast(np.linalg.solve(self.Q_hat, e_k))
        lam_k = np.broadcast_to(np.asarray(self.lam, dtype=float), (p,))[k]
        # C_k = Y_k / Q_kk is affine in w, with slope Cov(C_k, theta_hat) / bar_s^2
        bar_s2 = float(contrast.bar_s)**2
        gamma = float(np.ravel(self.Sigma @ e_k) @ contrast.direction) / self.Q_hat[k, k] / bar_s2
        w_obs = bar_s2 / float(contrast.naive_variance) * float(contrast.theta_hat) + float(contrast.bar_theta)
        # at theta_hat = 0 the selection data are Y(w) = N_o + bar_N_o + bar_Gamma w
        L = np.full(p, -np.inf) if self.L is None else np.asarray(self.L, dtype=float)
        U = np.full(p, np.inf) if self.U is None else np.asarray(self.U, dtype=float)
        return _cpp.unilasso_truncation_set(
            beta_hat=self.beta_hat, G_hat=self.G_hat, Q=np.asarray(self.Q_hat, dtype=float),
            D=np.broadcast_to(np.asarray(self.D, dtype=float), (p,)), L=L, U=U, k=int(k),
            lam_k=float(lam_k), C_k=float(self.C[k]),
            y0=np.ravel(np.atleast_1d(contrast.n_o) + np.atleast_1d(contrast.bar_n_o)),
            direction=np.ravel(np.atleast_1d(contrast.bar_gamma)),
            w_obs=w_obs, gamma=gamma)

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
            contrast = self.si.compute_contrast(np.linalg.solve(Q, np.eye(p)[k]))
            result = _contrast_inference(contrast, self.truncation_set(k, contrast), self.level)
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


def unilasso_loo_kappa(X, y, beta_hat):
    """
    kappa_j for a leave-one-out uniLasso fit beta_hat on centered X and y (univariate fits
    with intercepts): beta_hat solves the plain uniLasso with penalty
    (n lam + kappa_j) / |b_uni_j| on feature j. kappa_j = 2 sigma^2 + O_p(n^{-1/2}).
    """
    return _cpp.unilasso_loo_kappa(np.asarray(X, dtype=float), np.asarray(y, dtype=float),
                                   np.asarray(beta_hat, dtype=float))


def unilasso_inference(X,
                       y,
                       beta_hat,
                       lam,
                       *,
                       loo,
                       intercept=True,
                       sigma2=None,
                       level=0.95,
                       kkt_tol=1e-3):
    """
    Selective inference after a uniLasso (or unireg, lam = 0) fit on (X, y), for the
    full-model coefficients of the selected variables, in a Gaussian linear model with n > p.

    With loo=False the fit is assumed to solve

        minimize (1/2n) ||y - b0 - X beta||^2 + lam sum_j |beta_j| / |b_uni_j|
        subject to sign(beta_j) in {0, sign(b_uni_j)},

    with b_uni_j the univariate regression slopes of y on X_j (with intercepts if intercept),
    and the inference is exact. This is the uniLasso of the R package uniLasso, or of the
    Python package unilasso, with loo = False; lam is its lambda. The fit must use the
    lasso penalty (glmnet's alpha = 1): an elastic net would add a ridge term with
    data-dependent factors, which is not supported (the fit cannot be checked for this).

    With loo=True (those packages' default), the fit regresses y on leave-one-out univariate
    fits. Its solution solves the problem above with the penalty n lam on feature j replaced
    by n lam + kappa_j (unilasso_loo_kappa), and the inference uses those penalties. Because
    kappa_j depends on the data, which the inference ignores, it is approximate, and a
    warning says so.

    X, y : the data the fit used
    beta_hat : the fit's coefficients (without the intercept)
    lam : the fit's lambda, in glmnet's scaling (loss divided by n)
    loo : whether the fit used leave-one-out univariate fits; required
    intercept : whether the fit (and the univariate regressions) have intercepts
    sigma2 : noise variance; default the residual variance of the full least squares fit
    kkt_tol : warn if beta_hat violates the KKT conditions by more than kkt_tol times the penalty

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
    if loo and not intercept:
        raise NotImplementedError('loo=True is supported only with intercepts')
    if intercept:
        X = X - X.mean(0)
        y = y - y.mean()
    Q = X.T @ X
    Z = X.T @ y
    lam_Z = np.asarray(n * lam, dtype=float)
    if loo:
        # the leave-one-out uniLasso is the plain one with penalty n lam + kappa_j on feature j
        # (it can be negative for a strong feature at small n; the identity still holds)
        lam_Z = lam_Z + unilasso_loo_kappa(X, y, beta_hat)
        warnings.warn('loo=True: the fit is treated as the uniLasso with penalties '
                      '(n lam + kappa_j) / |b_uni_j|, which it solves exactly; kappa_j depends '
                      'on the data, which the inference ignores, so it is approximate')
    C, s, _, _, _ = unilasso_penalty(Z, Q, lam_Z)
    if np.any(beta_hat * s < 0):
        raise ValueError('beta_hat has a sign opposite to its univariate coefficient: '
                         'it is not a uniLasso fit for these data')
    scale = np.mean(np.abs(lam_Z)) if np.max(np.abs(lam_Z)) > 0 else np.abs(Z).max()
    violation = unilasso_kkt_violation(beta_hat, Q, Z, lam_Z)
    if violation > kkt_tol * scale:
        unit = 'the penalty' if np.max(np.abs(lam_Z)) > 0 else "max |X'y|"
        warnings.warn(f'beta_hat violates the uniLasso KKT conditions by {violation / scale:.1e} * '
                      f'{unit}; it may not solve the uniLasso at this lambda, or may not have '
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
           'unilasso_kkt_violation', 'unilasso_loo_kappa']
