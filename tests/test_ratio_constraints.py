import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import kstest, norm

from lassoinf import AffineConstraints, LassoInference
from lassoinf.bivariate_normal import TruncBivariateNormal
from lassoinf.ratio_constraints import (RatioConstraints,
                                        ratio_contrast_inference,
                                        ratio_row_sets,
                                        intersect_row_sets)


def _in_sets(t, sets):
    # membership of the points t in each row's union of intervals: (len(t), m)
    lo, hi = sets[..., 0], sets[..., 1]
    inside = (t[:, None, None] >= lo[None]) & (t[:, None, None] <= hi[None])
    return inside.any(axis=2)


# ---- one row ----

def test_two_pieces():
    # 3 - t <= 1/t on t > 0: B = 1, C = t, (AX) = 3 - t
    sets = ratio_row_sets(u=[3.], a=[-1.], b=[1.], beta=[0.], c=[0.], gamma=[1.], s_B=0., s_C=1.)
    S = intersect_row_sets(sets)
    r = np.sort(np.roots([1, -3, 1]))
    np.testing.assert_allclose(S, [[0., r[0]], [r[1], np.inf]])


def test_row_sets_against_grid():
    rng = np.random.default_rng(0)
    m = 400
    u, a, b, beta, c, gamma = rng.standard_normal((6, m))
    # some rows with constant B or C, and the lasso case
    beta[:50] = 0
    gamma[50:100] = 0
    beta[100:120] = gamma[100:120] = 0
    c[100:120] = 1
    s_B = rng.choice([-1., 0., 1.], m)
    s_C = rng.choice([-1., 1.], m)
    sets = ratio_row_sets(u, a, b, beta, c, gamma, s_B, s_C)

    t = np.linspace(-20, 20, 40001)
    B = b + beta * t[:, None]
    C = c + gamma * t[:, None]
    with np.errstate(divide='ignore', invalid='ignore'):
        direct = ((s_B == 0) | (np.sign(B) == s_B)) & (np.sign(C) == s_C) & (u + a * t[:, None] <= B / C)
    found = _in_sets(t, sets)

    # ignore grid points next to a boundary of either set
    ends = np.r_[sets[..., 0].ravel(), sets[..., 1].ravel()]
    ends = ends[np.isfinite(ends)]
    with np.errstate(divide='ignore', invalid='ignore'):
        ends = np.r_[ends, -b / beta, -c / gamma]
    ends = np.sort(ends[np.isfinite(ends)])
    i = np.clip(np.searchsorted(ends, t), 1, len(ends) - 1)
    near = np.minimum(np.abs(t - ends[i]), np.abs(t - ends[i - 1])) < 1e-3
    np.testing.assert_array_equal(found[~near], direct[~near])
    # the two-interval case occurs
    assert np.sum((sets[:, 1, 0] < sets[:, 1, 1]) & (sets[:, 0, 0] < sets[:, 0, 1])) > 0


def test_intersection():
    sets = np.array([[[-np.inf, 1.], [3., np.inf]],
                     [[0., 4.], [np.inf, -np.inf]],
                     [[-np.inf, 0.5], [0.8, np.inf]]])
    np.testing.assert_allclose(intersect_row_sets(sets), [[0., 0.5], [0.8, 1.], [3., 4.]])
    empty = np.array([[[0., 1.], [np.inf, -np.inf]], [[2., 3.], [np.inf, -np.inf]]])
    assert intersect_row_sets(empty).shape == (0, 2)


# ---- the defaults are the polyhedral lemma ----

def test_defaults_match_get_interval():
    rng = np.random.default_rng(1)
    p, m = 6, 15
    X = rng.standard_normal((40, p))
    Q = X.T @ X / 40
    Z = rng.standard_normal(p)
    Z_noisy = Z + rng.standard_normal(p)
    A = rng.standard_normal((m, p))
    b = A @ Z_noisy + rng.uniform(0.1, 2, m)
    si = AffineConstraints(Z=Z, Z_noisy=Z_noisy, Q=Q, Q_noise=None, scalar_noise=0.5)
    for j in range(p):
        contrast = si.compute_contrast(np.eye(p)[j])
        S = RatioConstraints(A=A, B=b).truncation_set(contrast)
        np.testing.assert_allclose(S, [contrast.get_interval(0., A, b)], rtol=1e-12)


def _lasso_cd(Q, Z, lam, n_iter=500):
    beta = np.zeros(Q.shape[0])
    for _ in range(n_iter):
        for j in range(Q.shape[0]):
            r = Z[j] - Q[j] @ beta + Q[j, j] * beta[j]
            beta[j] = np.sign(r) * max(abs(r) - lam, 0) / Q[j, j]
    return beta


def test_defaults_match_lasso_inference():
    rng = np.random.default_rng(2)
    n, p, kappa = 100, 8, 0.5
    X = rng.standard_normal((n, p))
    y = X[:, :2] @ np.r_[0.4, -0.3] + rng.standard_normal(n)
    Q, Z = X.T @ X, X.T @ y
    Z_noisy = Z + rng.multivariate_normal(np.zeros(p), kappa * Q)
    lam = 1.5 * np.sqrt(n)
    beta_hat = _lasso_cd(Q, Z_noisy, lam)
    LI = LassoInference(beta_hat=beta_hat, G_hat=Q @ beta_hat - Z_noisy, Q_hat=Q, D=np.full(p, lam),
                        L=np.full(p, -np.inf), U=np.full(p, np.inf), Z_full=Z, Sigma=Q, scalar_noise=kappa)
    assert len(LI.E) > 0
    ratio = RatioConstraints(A=LI.A, B=LI.b)
    for k, j in enumerate(LI.E):
        v = np.zeros(p)
        v[LI.E] = LI.W[:, k]
        r = ratio_contrast_inference(LI.si, ratio, v, level=LI.level)
        np.testing.assert_allclose([r.lower_conf, r.upper_conf, r.p_value],
                                   LI.summary_.loc[j, ['lower_conf', 'upper_conf', 'p_value']].values.astype(float),
                                   rtol=1e-6, atol=1e-8)


# ---- the truncated bivariate normal on a union of intervals ----

def _ccdf_by_quadrature(x, mu, sig_x, sig_omega, a_coeff, intervals):
    # P(X > x | a_coeff X + omega in the union) by integrating over X
    def p_in(z):
        lo, hi = intervals[:, 0] - a_coeff * z, intervals[:, 1] - a_coeff * z
        return np.sum(norm.cdf(hi / sig_omega) - norm.cdf(lo / sig_omega))
    f = lambda z: norm.pdf(z, mu, sig_x) * p_in(z)
    num = quad(f, x, np.inf, limit=200)[0]
    den = quad(f, -np.inf, x, limit=200)[0] + num
    return num / den


@pytest.mark.parametrize('intervals', [np.array([[-0.5, 1.2]]),
                                       np.array([[-np.inf, -1.], [0.2, 0.7], [2., np.inf]]),
                                       np.array([[-2., -1.5], [1., 3.]])])
def test_union_matches_quadrature(intervals):
    sig_x, sig_omega, a_coeff, theta = 1.3, 0.7, 0.4, 0.3
    tbn = TruncBivariateNormal(a_coeff=a_coeff, b_coeff=1., L=intervals[:, 0], U=intervals[:, 1],
                               sig_omega=sig_omega, sig_x=sig_x)
    mu = theta * sig_x**2
    for x in [-1., 0.4, 1.5]:
        np.testing.assert_allclose(tbn.ccdf(theta, x),
                                   _ccdf_by_quadrature(x, mu, sig_x, sig_omega, a_coeff, intervals),
                                   rtol=1e-5, atol=1e-7)
    if len(intervals) == 1:
        scalar = TruncBivariateNormal(a_coeff=a_coeff, b_coeff=1., L=intervals[0, 0], U=intervals[0, 1],
                                      sig_omega=sig_omega, sig_x=sig_x)
        assert scalar.ccdf(theta, 0.4) == tbn.ccdf(theta, 0.4)


# ---- calibration ----

def _pivots(rng, n_accept, Sigma, kappa, mu, eta, A, b0, K, c0, J, s_B):
    # rejection sampling of A Y <= B / C with Y = Z + omega, B = b0 + K Y, C = c0 + J Y > 0
    Lc = np.linalg.cholesky(Sigma)
    p = len(mu)
    ratio, naive = [], []
    while len(ratio) < n_accept:
        Z = mu + Lc @ rng.standard_normal(p)
        Y = Z + np.sqrt(kappa) * Lc @ rng.standard_normal(p)
        B, C = b0 + K @ Y, c0 + J @ Y
        if not (np.all(C > 0) and np.all((s_B == 0) | (np.sign(B) == s_B)) and np.all(A @ Y <= B / C)):
            continue
        si = AffineConstraints(Z=Z, Z_noisy=Y, Q=Sigma, Q_noise=None, scalar_noise=kappa)
        rc = RatioConstraints(A=A, B=B, C=C, s_B=s_B, cov_BZ=K @ Sigma, cov_CZ=J @ Sigma)
        ratio.append(ratio_contrast_inference(si, rc, eta).pivot(eta @ mu))
        # ignoring that B / C depends on the data
        naive.append(ratio_contrast_inference(si, RatioConstraints(A=A, B=B / C), eta).pivot(eta @ mu))
    return np.array(ratio), np.array(naive)


def test_calibration_two_pieces():
    # B = 1 - 3Y, C = Y > 0, -Y <= B / C: the event 3 - Y <= 1 / Y, two intervals
    rng = np.random.default_rng(3)
    ratio, naive = _pivots(rng, 400, np.eye(1), 1., np.r_[1.], np.ones(1),
                           A=-np.eye(1), b0=np.ones(1), K=-3 * np.eye(1),
                           c0=np.zeros(1), J=np.eye(1), s_B=np.zeros(1))
    assert kstest(ratio, 'uniform').pvalue > 1e-3
    assert kstest(naive, 'uniform').pvalue < 1e-3


def test_calibration_correlated():
    rng = np.random.default_rng(4)
    Sigma = np.array([[1., .3, 0], [.3, 1., .2], [0, .2, 1.]])
    K = np.zeros((3, 3))
    K[0, 0] = 0.3
    ratio, _ = _pivots(rng, 300, Sigma, 1., np.r_[0.5, 0., -0.3], np.r_[1., 0., 0.],
                       A=np.array([[1., 0.2, 0.], [-1., 0.5, 0.], [0.3, -0.5, 1.]]),
                       b0=np.ones(3), K=K, c0=np.full(3, 0.5), J=np.eye(3), s_B=np.r_[1., 0., 0.])
    assert kstest(ratio, 'uniform').pvalue > 1e-3
