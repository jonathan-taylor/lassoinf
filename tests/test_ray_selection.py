import numpy as np
import pytest
from scipy.stats import kstest

from lassoinf import AffineConstraints, LassoInference
from lassoinf.ray_selection import (find_truncation_set,
                                    lasso_event,
                                    ray_contrast_inference)
from lassoinf.unilasso import UniLassoInference, unilasso_fit, unilasso_penalty


def _design(rng, n, p, rho=0.5):
    return rng.standard_normal((n, p)) @ np.linalg.cholesky((1 - rho) * np.eye(p) + rho).T


def _weighted_lasso(Q, Y, D, tol=1e-13, max_iter=100000):
    # minimize 1/2 b'Qb - b'Y + sum_j D_j |b_j| by coordinate descent
    b = np.zeros(len(Y))
    d = np.diag(Q)
    for _ in range(max_iter):
        delta = 0.
        for j in range(len(Y)):
            r = Y[j] - Q[j] @ b + d[j] * b[j]
            new = np.sign(r) * max(abs(r) - D[j], 0.) / d[j]
            delta = max(delta, abs(new - b[j]) * np.sqrt(d[j]))
            b[j] = new
        if delta < tol:
            break
    return b


def test_fixed_penalty_is_the_polyhedral_lemma():
    # with fixed penalties the event is a polyhedron: one interval, the usual one
    rng = np.random.default_rng(0)
    n, p = 100, 6
    X = _design(rng, n, p)
    Q, Z = X.T @ X, X.T @ (X[:, :2] @ np.r_[0.3, -0.2] + rng.standard_normal(n))
    Y = Z + np.linalg.cholesky(0.5 * Q) @ rng.standard_normal(p)
    D = np.full(p, 15.)
    b = _weighted_lasso(Q, Y, D)
    LI = LassoInference(beta_hat=b, G_hat=Q @ b - Y, Q_hat=Q, D=D, L=np.full(p, -np.inf),
                        U=np.full(p, np.inf), Z_full=Z, Sigma=Q, scalar_noise=0.5)
    event = lasso_event(LI.beta_hat, LI.G_hat, Q, lambda Y: (D, LI.L, LI.U))
    assert len(LI.E) > 0
    for j, contrast in LI._contrasts.items():
        S = find_truncation_set(event, contrast)
        np.testing.assert_allclose(S, [contrast.get_interval(0., LI.A, LI.b)], rtol=1e-7, atol=1e-7)


def test_matches_the_unilasso_closed_form():
    # the uniLasso's truncation sets, found along the ray, match its closed form
    rng = np.random.default_rng(1)
    checked = 0
    while checked < 10:
        n, p = 100, 6
        X = _design(rng, n, p)
        Q, Z = X.T @ X, X.T @ (X @ np.r_[0.3, -0.25, 0.2, 0, 0, 0] + rng.standard_normal(n))
        Y = Z + np.linalg.cholesky(0.5 * Q) @ rng.standard_normal(p)
        lam = 4.
        UI = UniLassoInference.from_selection(unilasso_fit(Q, Y, lam), Y, Q, lam, Z, Q, scalar_noise=0.5)
        s_obs = np.sign(UI.C)

        def penalty(Y):
            C, s, D, L, U = unilasso_penalty(Y, Q, lam)
            return (D, L, U) if np.all(s == s_obs) else None

        event = lasso_event(UI.beta_hat, UI.G_hat, Q, penalty)
        for k in UI.E:
            contrast = UI.si.compute_contrast(np.linalg.solve(Q, np.eye(p)[k]))
            closed = UI.truncation_set(k, contrast)
            ray = find_truncation_set(event, contrast, n_sd=8)
            # compare within the searched window
            w0 = (float(contrast.bar_s)**2 / float(contrast.naive_variance) * float(contrast.theta_hat)
                  + float(contrast.bar_theta))
            sd = np.sqrt(float(contrast.bar_s)**4 / float(contrast.naive_variance) + float(contrast.bar_s)**2)
            clip = lambda S: np.clip(S, w0 - 8 * sd, w0 + 8 * sd)
            closed = clip(closed[(closed[:, 1] > w0 - 8 * sd) & (closed[:, 0] < w0 + 8 * sd)])
            np.testing.assert_allclose(clip(ray), closed, atol=1e-6 * sd)
            checked += 1


def _adaptive_lasso_case(rng, n, p, beta, lam, kappa):
    # the adaptive lasso: penalties lam / |b_ols_j(Y)|, b_ols = Q^{-1} Y from the selection data
    X = _design(rng, n, p)
    Q, Z = X.T @ X, X.T @ (X @ beta + rng.standard_normal(n))
    Y = Z + np.linalg.cholesky(kappa * Q) @ rng.standard_normal(p) if kappa > 0 else Z
    Qinv = np.linalg.inv(Q)
    penalty = lambda Y: (lam / np.abs(Qinv @ Y), np.full(p, -np.inf), np.full(p, np.inf))
    D = penalty(Y)[0]
    b = _weighted_lasso(Q, Y, D)
    return Q, Z, Y, b, penalty


@pytest.mark.parametrize('kappa', [0., 1.])
def test_adaptive_lasso_calibration(kappa):
    # every penalty moves along the ray, so the rows are sums of ratios: no closed form
    rng = np.random.default_rng(2)
    n, p, lam = 80, 4, 3.
    beta = np.r_[0.35, -0.25, 0., 0.]
    pivots = []
    while len(pivots) < 150:
        Q, Z, Y, b, penalty = _adaptive_lasso_case(rng, n, p, beta, lam, kappa)
        E = np.nonzero(b)[0]
        if len(E) == 0:
            continue
        si = AffineConstraints(Z=Z, Z_noisy=Y, Q=Q, Q_noise=None, scalar_noise=kappa)
        event = lasso_event(b, Q @ b - Y, Q, penalty)
        for k in E:
            eta = np.linalg.solve(Q, np.eye(p)[k])          # full-model coefficient
            res = ray_contrast_inference(si, eta, event, level=0.9, n_sd=8, n_grid=801)
            pivots.append(res.pivot(beta[k]))
    assert kstest(pivots, 'uniform').pvalue > 1e-3
