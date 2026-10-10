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
"""
from dataclasses import dataclass
import warnings

import numpy as np
import pandas as pd

from .lasso import LassoInference, lasso_post_selection_constraints
from .bivariate_normal import TruncBivariateNormal


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


# ---- the truncation set along the conditioning line ----
#
# For the target of coordinate k, every constraint row reads, as a function of the scalar w
# the selection depends on (see docs/data_dependent_penalty.md),
#
#     u_i + a_i w <= b0_i + m_i / C(w),   C(w) = c + gamma w,   s C(w) > 0,
#
# with C = C_k, the univariate coefficient of the target, and s its sign. On the half-line
# s C(w) > 0, multiplying by C(w) gives a quadratic inequality, so each row holds on at most
# two intervals and the truncation set is a finite union of intervals.

# rows with |linear coefficient| below this are treated as constant, as in
# AffineConstraintsContrast.get_interval
_ZERO_TOL = 1e-10


def _quadratic_le_zero(q2, q1, q0):
    """
    The set {w : q2 w^2 + q1 w + q0 <= 0} for each row, as two intervals
    (lo1, hi1), (lo2, hi2); a missing interval has lo > hi.
    """
    m = q2.shape[0]
    lo1 = np.full(m, np.inf)
    hi1 = np.full(m, -np.inf)
    lo2 = np.full(m, np.inf)
    hi2 = np.full(m, -np.inf)

    # linear rows: q1 w + q0 <= 0
    lin = q2 == 0
    pos = lin & (q1 > _ZERO_TOL)
    neg = lin & (q1 < -_ZERO_TOL)
    const = lin & ~pos & ~neg
    with np.errstate(divide='ignore', invalid='ignore'):
        root = -q0 / q1
    lo1 = np.where(pos, -np.inf, lo1)
    hi1 = np.where(pos, root, hi1)
    lo1 = np.where(neg, root, lo1)
    hi1 = np.where(neg, np.inf, hi1)
    feasible = const & (q0 <= _ZERO_TOL)
    lo1 = np.where(feasible, -np.inf, lo1)
    hi1 = np.where(feasible, np.inf, hi1)

    # quadratic rows, roots in the numerically stable form
    quad = ~lin
    disc = q1**2 - 4 * q2 * q0
    real = quad & (disc >= 0)
    with np.errstate(divide='ignore', invalid='ignore'):
        sq = np.sqrt(np.where(real, disc, 0.))
        w = -0.5 * (q1 + np.where(q1 >= 0, sq, -sq))
        ra = w / q2
        rb = np.where(w != 0, q0 / w, ra)
    r1 = np.minimum(ra, rb)
    r2 = np.maximum(ra, rb)

    up = real & (q2 > 0)        # between the roots
    lo1 = np.where(up, r1, lo1)
    hi1 = np.where(up, r2, hi1)

    down = real & (q2 < 0)      # outside the roots
    lo1 = np.where(down, -np.inf, lo1)
    hi1 = np.where(down, r1, hi1)
    lo2 = np.where(down, r2, lo2)
    hi2 = np.where(down, np.inf, hi2)

    everywhere = quad & ~real & (q2 < 0)
    lo1 = np.where(everywhere, -np.inf, lo1)
    hi1 = np.where(everywhere, np.inf, hi1)
    return (lo1, hi1), (lo2, hi2)


def _row_sets(u, a, b0, m, c, gamma, s):
    """
    For each row i, the set of w with s (c + gamma w) > 0 and
    u_i + a_i w <= b0_i + m_i / (c + gamma w), as an array (rows, 2, 2) of two intervals
    (lo, hi) per row; a missing interval has lo > hi.
    """
    u, a, b0, m = [np.atleast_1d(np.asarray(v, dtype=float)) for v in (u, a, b0, m)]
    # the half-line where C(w) has sign s
    k, h = s * gamma, -s * c
    if k > 0:
        lo_I, hi_I = h / k, np.inf
    elif k < 0:
        lo_I, hi_I = -np.inf, h / k
    elif s * c > 0:
        lo_I, hi_I = -np.inf, np.inf
    else:
        lo_I, hi_I = np.inf, -np.inf
    # on it, the row is s [C(w) (u_i - b0_i + a_i w) - m_i] <= 0
    v = u - b0
    pieces = _quadratic_le_zero(s * gamma * a, s * (c * a + gamma * v), s * (c * v - m))
    out = np.empty((len(u), 2, 2))
    for j, (lo, hi) in enumerate(pieces):
        out[:, j, 0] = np.maximum(lo, lo_I)
        out[:, j, 1] = np.minimum(hi, hi_I)
    return out


def _intersect(row_sets):
    """
    The intersection over rows of the unions in row_sets, as an array (k, 2) of disjoint
    intervals in increasing order.
    """
    rows = row_sets.shape[0]
    if rows == 0:
        return np.array([[-np.inf, np.inf]])
    lo = row_sets[..., 0].ravel()
    hi = row_sets[..., 1].ravel()
    keep = lo < hi
    lo, hi = lo[keep], hi[keep]
    if lo.size == 0:
        return np.zeros((0, 2))
    # sweep over the endpoints counting covering rows (each row's pieces are disjoint);
    # at ties, closing before opening drops zero-length pieces
    pos = np.r_[lo, hi]
    delta = np.r_[np.ones(lo.size), -np.ones(hi.size)]
    order = np.lexsort((delta, pos))
    pos, delta = pos[order], delta[order]
    count = np.cumsum(delta)
    seg = np.nonzero((count[:-1] == rows) & (pos[:-1] < pos[1:]))[0]
    intervals = np.column_stack([pos[seg], pos[seg + 1]])
    if len(intervals) > 1:
        merged = [intervals[0]]
        for l, h in intervals[1:]:
            if l <= merged[-1][1]:
                merged[-1] = np.array([merged[-1][0], max(merged[-1][1], h)])
            else:
                merged.append(np.array([l, h]))
        intervals = np.array(merged)
    return intervals


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


def _apply(M, x):
    return np.ravel(M.matvec(x) if hasattr(M, 'matvec') else np.asarray(M) @ x)


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
        D = np.asarray(self.D, dtype=float)
        b_of = lambda D: lasso_post_selection_constraints(self.beta_hat, self.G_hat, self.Q_hat,
                                                          D, self.L, self.U)[1]
        # b is affine in D: its coefficient on D_k (any step works; D_k = 0 for unireg)
        step = D[k] if D[k] > 0 else 1.
        M_k = (b_of(D + step * e_k) - self.b) / step
        b0 = self.b - M_k * D[k]
        s_k = float(np.sign(self.C[k]))
        lam_k = np.broadcast_to(np.asarray(self.lam, dtype=float), (p,))[k]
        # at theta_hat = 0, A Y = A (N_o + bar_N_o) + A bar_Gamma w
        u = _apply(self.A, np.atleast_1d(contrast.n_o + contrast.bar_n_o))
        a = _apply(self.A, np.atleast_1d(contrast.bar_gamma))
        # C_k = Y_k / Q_kk is affine in w, with slope Cov(C_k, theta_hat) / bar_s^2
        bar_s2 = float(contrast.bar_s)**2
        gamma = float(np.ravel(self.Sigma @ e_k) @ contrast.direction) / self.Q_hat[k, k] / bar_s2
        w_obs = bar_s2 / float(contrast.naive_variance) * float(contrast.theta_hat) + float(contrast.bar_theta)
        c = self.C[k] - gamma * w_obs
        return _intersect(_row_sets(u, a, b0, M_k * lam_k * s_k, c, gamma, s_k))

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
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    beta_hat = np.asarray(beta_hat, dtype=float)
    n = X.shape[0]
    S = np.sum(X**2, axis=0)
    b = X.T @ y / S
    h = 1 / n + X**2 / S                                # leverages of the univariate fits
    delta = -h / (1 - h) * (y[:, None] - X * b)         # leave-one-out fits minus fits
    delta = delta - delta.mean(0)
    theta = beta_hat / b
    r = y - X @ beta_hat
    return -delta.T @ r + b * (X.T @ (delta @ theta)) + delta.T @ (delta @ theta)


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
    Python package unilasso, with loo = False; lam is its lambda.

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
