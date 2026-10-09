"""
Selection events A X <= B / C with a data-dependent right-hand side.

B and C are (asymptotically) jointly Gaussian with X and their signs are part of the
selection event. Along the one-dimensional affine family the polyhedral lemma conditions
on, each row is a quadratic inequality on an interval, so the truncation set is a finite
union of intervals. See docs/data_dependent_penalty.md.
"""
from dataclasses import dataclass

import numpy as np

from .bivariate_normal import TruncBivariateNormal

# rows with |linear coefficient| below this are treated as constant, as in
# AffineConstraintsContrast.get_interval
_ZERO_TOL = 1e-10


def _sign_interval(sign, v0, v1):
    """
    The interval {t : sign * (v0 + v1 t) > 0} for each row, as (lo, hi, nonempty);
    rows with sign 0 are unconstrained.
    """
    k = sign * v1
    h = -sign * v0
    lo = np.full(k.shape, -np.inf)
    hi = np.full(k.shape, np.inf)
    with np.errstate(divide='ignore', invalid='ignore'):
        lo = np.where(k > 0, h / k, lo)
        hi = np.where(k < 0, h / k, hi)
    nonempty = ~((sign != 0) & (k == 0) & (h >= 0))
    return lo, hi, nonempty


def _quadratic_le_zero(q2, q1, q0):
    """
    The set {t : q2 t^2 + q1 t + q0 <= 0} for each row, as two intervals
    (lo1, hi1), (lo2, hi2); a missing interval has lo > hi.
    """
    m = q2.shape[0]
    lo1 = np.full(m, np.inf)
    hi1 = np.full(m, -np.inf)
    lo2 = np.full(m, np.inf)
    hi2 = np.full(m, -np.inf)

    # linear rows: q1 t + q0 <= 0
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


def ratio_row_sets(u, a, b, beta, c, gamma, s_B, s_C):
    """
    For each row i, the set of t with

        sign(b_i + beta_i t) = s_B[i]   (not required if s_B[i] == 0),
        sign(c_i + gamma_i t) = s_C[i],
        u_i + a_i t <= (b_i + beta_i t) / (c_i + gamma_i t),

    as two intervals per row: arrays of shape (m, 2, 2) of (lo, hi), a missing interval
    having lo > hi.
    """
    u, a, b, beta, c, gamma = [np.atleast_1d(np.asarray(x, dtype=float))
                               for x in (u, a, b, beta, c, gamma)]
    m = u.shape[0]
    s_B = np.broadcast_to(np.asarray(s_B, dtype=float), (m,))
    s_C = np.broadcast_to(np.asarray(s_C, dtype=float), (m,))
    if np.any(np.abs(s_C) != 1):
        raise ValueError('the sign of each C must be conditioned on: s_C must be +1 or -1')

    lo_B, hi_B, ok_B = _sign_interval(s_B, b, beta)
    lo_C, hi_C, ok_C = _sign_interval(s_C, c, gamma)
    lo_I = np.maximum(lo_B, lo_C)
    hi_I = np.minimum(hi_B, hi_C)
    ok = ok_B & ok_C

    # on the sign interval, the ratio constraint is s_C * [C (u + a t) - B] <= 0
    q2 = s_C * gamma * a
    q1 = s_C * (c * a + gamma * u - beta)
    q0 = s_C * (c * u - b)
    pieces = _quadratic_le_zero(q2, q1, q0)

    out = np.empty((m, 2, 2))
    for k, (lo, hi) in enumerate(pieces):
        out[:, k, 0] = np.where(ok, np.maximum(lo, lo_I), np.inf)
        out[:, k, 1] = np.where(ok, np.minimum(hi, hi_I), -np.inf)
    return out


def intersect_row_sets(row_sets):
    """
    The intersection over rows of the unions in row_sets (as from ratio_row_sets), as an
    array of shape (k, 2) of disjoint intervals in increasing order.
    """
    m = row_sets.shape[0]
    if m == 0:
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
    seg = np.nonzero((count[:-1] == m) & (pos[:-1] < pos[1:]))[0]
    intervals = np.column_stack([pos[seg], pos[seg + 1]])

    # merge touching segments
    if len(intervals) > 1:
        merged = [intervals[0]]
        for l, h in intervals[1:]:
            if l <= merged[-1][1]:
                merged[-1] = np.array([merged[-1][0], max(merged[-1][1], h)])
            else:
                merged.append(np.array([l, h]))
        intervals = np.array(merged)
    return intervals


def _apply(M, x):
    return np.ravel(M.matvec(x) if hasattr(M, 'matvec') else np.asarray(M) @ x)


@dataclass
class RatioConstraints:
    """
    The selection event A (Z + omega) <= B / C (row by row), with sign(B) = s_B and
    sign(C) = s_C.

    B and C must be functions of the selection data Z + omega (plus independent noise),
    so that the event depends on (theta_hat, bar_omega) only through
    w = (bar_s^2 / sigma^2) theta_hat + bar_omega; see docs/data_dependent_penalty.md.

    A : matrix or operator (m x p)
    B : observed numerators (m,)
    C : observed denominators (m,); default 1
    s_B : conditioned signs of B; default 0 (not conditioned, for constant B)
    s_C : conditioned signs of C (+1 or -1); default sign(C)
    cov_BZ, cov_CZ : Cov(B, Z) and Cov(C, Z) (m x p, matrix or operator); default 0

    With the defaults this is the usual event A (Z + omega) <= B.
    """
    A: object
    B: np.ndarray
    C: np.ndarray = None
    s_B: np.ndarray = None
    s_C: np.ndarray = None
    cov_BZ: object = None
    cov_CZ: object = None

    def __post_init__(self):
        self.B = np.atleast_1d(np.asarray(self.B, dtype=float))
        m = self.B.shape[0]
        self.C = np.ones(m) if self.C is None else np.broadcast_to(
            np.asarray(self.C, dtype=float), (m,)).copy()
        self.s_B = np.zeros(m) if self.s_B is None else np.broadcast_to(
            np.asarray(self.s_B, dtype=float), (m,)).copy()
        self.s_C = np.sign(self.C) if self.s_C is None else np.broadcast_to(
            np.asarray(self.s_C, dtype=float), (m,)).copy()

    def _affine(self, V, cov_VZ, contrast):
        # V = N_V + (k_V / bar_s^2) w with k_V = Cov(V, theta_hat) = Cov(V, Z) eta;
        # returns the intercept and slope in w, evaluated so that w is bar_omega at theta_hat = 0
        if cov_VZ is None:
            return V, np.zeros_like(V)
        k = _apply(cov_VZ, contrast.direction)
        bar_s2 = float(contrast.bar_s)**2
        w_obs = bar_s2 / float(contrast.naive_variance) * float(contrast.theta_hat) + float(contrast.bar_theta)
        slope = k / bar_s2
        return V - slope * w_obs, slope

    def row_sets(self, contrast):
        """Each row's set of w, as from ratio_row_sets."""
        # at theta_hat = 0, A (Z + omega) = A (N_o + bar_N_o) + A bar_Gamma w
        u = _apply(self.A, np.atleast_1d(contrast.n_o + contrast.bar_n_o))
        a = _apply(self.A, np.atleast_1d(contrast.bar_gamma))
        b, beta = self._affine(self.B, self.cov_BZ, contrast)
        c, gamma = self._affine(self.C, self.cov_CZ, contrast)
        return ratio_row_sets(u, a, b, beta, c, gamma, self.s_B, self.s_C)

    def truncation_set(self, contrast):
        """
        The set of w (equivalently of bar_omega at theta_hat = 0) on which the selection
        event holds, as an array (k, 2) of disjoint intervals.
        """
        return intersect_row_sets(self.row_sets(contrast))


@dataclass
class RatioContrastResult:
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


def ratio_contrast_inference(affine_constraints, ratio_constraints, direction, level=0.95):
    """
    Selective inference for theta = direction' E[Z] after the selection event of
    ratio_constraints, with the randomization of affine_constraints (an AffineConstraints).
    """
    contrast = affine_constraints.compute_contrast(direction)
    intervals = ratio_constraints.truncation_set(contrast)
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
    return RatioContrastResult(theta_hat=theta_hat,
                               lower_conf=L_theta * variance,
                               upper_conf=U_theta * variance,
                               p_value=float(np.clip(2 * min(cdf0, 1 - cdf0), 0., 1.)),
                               intervals=intervals,
                               family=family,
                               variance=variance)


__all__ = ['RatioConstraints', 'ratio_contrast_inference', 'ratio_row_sets',
           'intersect_row_sets', 'RatioContrastResult']
