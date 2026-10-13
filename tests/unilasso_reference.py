"""
Pure-Python reference implementations of the uniLasso computations that lassoinf runs in
C++ (cpp/src/unilasso.cpp). The tests check the C++ code against these.
"""
import numpy as np

from lassoinf.lasso import lasso_post_selection_constraints


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


def truncation_set(beta_hat, G_hat, Q, D, L, U, k, lam_k, C_k, y0, direction, w_obs, gamma,
                   tol=1e-6):
    """
    The truncation set for the target of coordinate k, as lassoinf_cpp.unilasso_truncation_set.
    """
    D = np.asarray(D, dtype=float)
    A, b = lasso_post_selection_constraints(beta_hat, G_hat, Q, D, L, U, tol=tol)[:2]
    e_k = np.eye(len(D))[k]
    step = D[k] if D[k] > 0 else 1.
    b_step = lasso_post_selection_constraints(beta_hat, G_hat, Q, D + step * e_k, L, U, tol=tol)[1]
    M_k = (b_step - b) / step
    b0 = b - M_k * D[k]
    s_k = float(np.sign(C_k))
    u = np.ravel(A @ np.asarray(y0, dtype=float))
    a = np.ravel(A @ np.asarray(direction, dtype=float))
    return _intersect(_row_sets(u, a, b0, M_k * lam_k * s_k, C_k - gamma * w_obs, gamma, s_k))
