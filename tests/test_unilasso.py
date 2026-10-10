import warnings

import numpy as np
import pytest
from scipy.stats import kstest

from lassoinf.lasso import lasso_post_selection_constraints
from lassoinf.unilasso import (UniLassoInference,
                               _intersect,
                               _row_sets,
                               unilasso_inference,
                               unilasso_fit,
                               unilasso_loo_kappa,
                               unilasso_penalty)


def _data(rng, n=60, p=5, rho=0.5, beta=(0.35, -0.25, 0.15, 0, 0)):
    X = rng.standard_normal((n, p)) @ np.linalg.cholesky((1 - rho) * np.eye(p) + rho).T
    beta = np.asarray(beta, dtype=float)
    return X, X @ beta + rng.standard_normal(n), beta


def _fit(X, y, lam):
    # the uniLasso with intercept at lam (glmnet scaling), as uniLasso(loo = FALSE) fits it
    Xc, yc = X - X.mean(0), y - y.mean()
    return unilasso_fit(Xc.T @ Xc, Xc.T @ yc, X.shape[0] * lam)


def _randomized(rng, X, y, lam, kappa=0.5):
    # randomized uniLasso on X'y + omega, omega ~ N(0, kappa X'X), sigma^2 = 1; lam in Z units
    Q, Z = X.T @ X, X.T @ y
    Z_noisy = Z + np.linalg.cholesky(kappa * Q) @ rng.standard_normal(len(Z))
    return UniLassoInference.from_selection(unilasso_fit(Q, Z_noisy, lam), Z_noisy, Q, lam, Z, Q,
                                            scalar_noise=kappa)


@pytest.mark.parametrize('lam', [0., 3.])
def test_fit_kkt(lam):
    rng = np.random.default_rng(0)
    X, y, _ = _data(rng)
    Q, Z = X.T @ X, X.T @ y
    beta = unilasso_fit(Q, Z, lam)
    C, s, D, L, U = unilasso_penalty(Z, Q, lam)
    assert np.all(beta * s >= 0)
    g = Z - Q @ beta    # minus the gradient of the smooth part
    active = beta != 0
    np.testing.assert_allclose(g[active], D[active] * s[active], atol=1e-8)
    # inactive: the subgradient bound on the side the sign constraint allows
    assert np.all(s[~active] * g[~active] <= D[~active] + 1e-8)


@pytest.mark.parametrize('lam', [0., 4.])
def test_truncation_set_matches_event(lam):
    # along the line for each target, the truncation set is where the uniLasso event holds,
    # recomputing every univariate coefficient, penalty and constraint
    rng = np.random.default_rng(1)
    n_checked = 0
    while n_checked < 6:
        X, y, _ = _data(rng, n=80, p=6, beta=(0.3, -0.25, 0.2, 0, 0, 0))
        UI = _randomized(rng, X, y, lam)
        Q = UI.Q_hat
        for k in UI.E:
            eta = np.linalg.solve(Q, np.eye(len(UI.C))[k])
            con = UI.si.compute_contrast(eta)
            S = UI.truncation_set(k, con)
            bar_s2 = float(con.bar_s)**2
            w_obs = bar_s2 / float(con.naive_variance) * float(con.theta_hat) + float(con.bar_theta)
            sd = np.sqrt(bar_s2**2 / float(con.naive_variance) + bar_s2)
            ws = w_obs + np.linspace(-6, 6, 1201) * sd
            event = np.zeros(len(ws), bool)
            for i, w in enumerate(ws):
                Yw = con.n_o + con.bar_n_o + con.bar_gamma * w
                Cw = Yw / np.diag(Q)
                if np.any(np.sign(Cw) != np.sign(UI.C)):
                    continue
                bw = lasso_post_selection_constraints(UI.beta_hat, UI.G_hat, Q, lam / np.abs(Cw),
                                                      UI.L, UI.U)[1]
                event[i] = np.all(UI.A @ Yw <= bw)
            in_S = np.zeros(len(ws), bool)
            for lo, hi in S:
                in_S |= (ws >= lo) & (ws <= hi)
            ends = S[np.isfinite(S)]
            near = (np.min(np.abs(ws[:, None] - ends[None]), axis=1) < 1e-2 * sd if ends.size
                    else np.zeros(len(ws), bool))
            np.testing.assert_array_equal(in_S[~near], event[~near])
            n_checked += 1


@pytest.mark.parametrize('lam,n_rep', [(0., 120), (0.06, 250)])
def test_calibration(lam, n_rep):
    # a uniLasso fit on all the data (no randomization): pivots at the true full-model
    # coefficients are uniform; lam = 0 is unireg
    rng = np.random.default_rng(2)
    pivots = []
    for _ in range(n_rep):
        X, y, beta = _data(rng)
        UI = unilasso_inference(X, y, _fit(X, y, lam), lam, loo=False, sigma2=1.)
        pivots.extend(UI._results[k].pivot(beta[k]) for k in UI.E)
    assert len(pivots) > 200
    assert kstest(pivots, 'uniform').pvalue > 1e-3


@pytest.mark.parametrize('lam_scale', [0., 0.5])
def test_calibration_randomized(lam_scale):
    rng = np.random.default_rng(5)
    pivots = []
    while len(pivots) < 250:
        X, y, beta = _data(rng)
        UI = _randomized(rng, X, y, lam_scale * np.sqrt(X.shape[0]), kappa=1.)
        pivots.extend(UI._results[k].pivot(beta[k]) for k in UI.E)
    assert kstest(pivots, 'uniform').pvalue > 1e-3


def test_summary_and_intercept():
    rng = np.random.default_rng(3)
    X, y, _ = _data(rng)
    X, y = X + 2., y + 5.          # the fit has an intercept
    lam = 0.03
    UI = unilasso_inference(X, y, _fit(X, y, lam), lam, loo=False)
    X1 = np.column_stack([np.ones(len(y)), X])
    ols = np.linalg.lstsq(X1, y, rcond=None)[0][1:]
    np.testing.assert_allclose(UI.summary_['beta_hat'].values, ols[UI.E])
    assert np.all(np.isfinite(UI.summary_[['lower_conf', 'upper_conf']].values))
    assert np.all(UI.summary_['lower_conf'] < UI.summary_['upper_conf'])


def test_checks_the_fit():
    rng = np.random.default_rng(4)
    X, y, _ = _data(rng)
    lam = 0.03
    beta = _fit(X, y, lam)
    with pytest.warns(UserWarning, match='KKT'):
        unilasso_inference(X, y, beta * 1.2, lam, loo=False)
    wrong_sign = beta.copy()
    wrong_sign[np.argmax(np.abs(beta))] *= -1
    with pytest.raises(ValueError, match='sign'):
        unilasso_inference(X, y, wrong_sign, lam, loo=False)
    with pytest.raises(ValueError, match='n > p'):
        unilasso_inference(rng.standard_normal((5, 8)), rng.standard_normal(5), np.zeros(8), 1., loo=False)


def _loo_fit(X, y, lam, tol=1e-13, max_iter=100000):
    # the uniLasso with leave-one-out univariate fits (uniLasso(loo = TRUE) in R): the
    # nonnegative lasso (1/2n)||y - theta_0 - F theta||^2 + lam sum theta_j, theta >= 0, on the
    # leave-one-out fits F; returns beta = b * theta, b the univariate slopes
    n = X.shape[0]
    Xc, yc = X - X.mean(0), y - y.mean()
    S = np.sum(Xc**2, axis=0)
    b = Xc.T @ yc / S
    h = 1 / n + Xc**2 / S
    F = y.mean() + Xc * b - h / (1 - h) * (yc[:, None] - Xc * b)
    F = F - F.mean(0)
    theta = np.zeros(X.shape[1])
    r = yc.copy()
    d = np.sum(F**2, axis=0)
    for _ in range(max_iter):
        delta = 0.
        for j in range(len(theta)):
            new = max(theta[j] + (F[:, j] @ r - n * lam) / d[j], 0.)
            if new != theta[j]:
                r -= F[:, j] * (new - theta[j])
                delta = max(delta, abs(new - theta[j]) * np.sqrt(d[j]))
                theta[j] = new
        if delta < tol:
            break
    return b * theta


def test_loo_kappa_identity():
    # a leave-one-out fit satisfies the KKT conditions with penalties (n lam + kappa_j) / |b_j|
    rng = np.random.default_rng(6)
    X, y, _ = _data(rng, n=200, p=6, beta=(0.35, -0.25, 0.15, 0, 0, 0))
    lam = 0.004
    beta = _loo_fit(X, y, lam)
    Xc, yc = X - X.mean(0), y - y.mean()
    kappa = unilasso_loo_kappa(Xc, yc, beta)
    b = Xc.T @ yc / np.sum(Xc**2, axis=0)
    score = b * (Xc.T @ (yc - Xc @ beta))
    active = beta != 0
    assert active.any() and not active.all()
    np.testing.assert_allclose(score[active], X.shape[0] * lam + kappa[active], rtol=1e-6)
    assert np.all(score[~active] <= X.shape[0] * lam + kappa[~active] + 1e-8)
    sigma2 = np.sum((yc - Xc @ np.linalg.solve(Xc.T @ Xc, Xc.T @ yc))**2) / (len(y) - X.shape[1] - 1)
    assert np.all(np.abs(kappa / (2 * sigma2) - 1) < 0.5)


def test_loo_inference():
    rng = np.random.default_rng(7)
    X, y, _ = _data(rng, n=200, p=6, beta=(0.35, -0.25, 0.15, 0, 0, 0))
    lam = 0.004
    with pytest.warns(UserWarning, match='approximate'):
        UI = unilasso_inference(X, y, _loo_fit(X, y, lam), lam, loo=True)
    X1 = np.column_stack([np.ones(len(y)), X])
    ols = np.linalg.lstsq(X1, y, rcond=None)[0][1:]
    # the fit solves the shifted problem exactly, so the estimates are least squares
    np.testing.assert_allclose(UI.summary_['beta_hat'].values, ols[UI.E], atol=1e-6)
    with pytest.raises(TypeError):
        unilasso_inference(X, y, _loo_fit(X, y, lam), lam)          # loo is required


def test_loo_calibration():
    rng = np.random.default_rng(8)
    pivots = []
    while len(pivots) < 250:
        X, y, beta = _data(rng, n=200, p=5)
        lam = 0.004
        b = _loo_fit(X, y, lam)
        if not np.any(b):
            continue
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            UI = unilasso_inference(X, y, b, lam, loo=True, sigma2=1.)
        pivots.extend(UI._results[k].pivot(beta[k]) for k in UI.E)
    assert kstest(pivots, 'uniform').pvalue > 1e-3


# ---- the row solver ----

def test_row_two_pieces():
    # 3 - w <= 1 / w on w > 0: u = 3, a = -1, b0 = 0, m = 1, C(w) = w
    S = _intersect(_row_sets(u=[3.], a=[-1.], b0=[0.], m=[1.], c=0., gamma=1., s=1.))
    r = np.sort(np.roots([1, -3, 1]))
    np.testing.assert_allclose(S, [[0., r[0]], [r[1], np.inf]])


@pytest.mark.parametrize('s', [1., -1.])
def test_row_sets_against_grid(s):
    rng = np.random.default_rng(0)
    rows = 300
    u, a, b0, m = rng.standard_normal((4, rows))
    c, gamma = 0.3 * s, 0.8
    sets = _row_sets(u, a, b0, m, c, gamma, s)
    w = np.linspace(-15, 15, 30001)
    C = c + gamma * w
    with np.errstate(divide='ignore', invalid='ignore'):
        direct = (np.sign(C)[:, None] == s) & (u + a * w[:, None] <= b0 + m / C[:, None])
    found = ((w[:, None, None] >= sets[None, ..., 0]) & (w[:, None, None] <= sets[None, ..., 1])).any(axis=2)
    ends = np.sort(np.r_[sets[np.isfinite(sets)], -c / gamma])
    i = np.clip(np.searchsorted(ends, w), 1, len(ends) - 1)
    near = np.minimum(np.abs(w - ends[i]), np.abs(w - ends[i - 1])) < 1e-3
    np.testing.assert_array_equal(found[~near], direct[~near])
    # rows with two intervals occur
    assert np.sum((sets[:, 0, 0] < sets[:, 0, 1]) & (sets[:, 1, 0] < sets[:, 1, 1])) > 0


def test_intersect():
    sets = np.array([[[-np.inf, 1.], [3., np.inf]],
                     [[0., 4.], [np.inf, -np.inf]],
                     [[-np.inf, 0.5], [0.8, np.inf]]])
    np.testing.assert_allclose(_intersect(sets), [[0., 0.5], [0.8, 1.], [3., 4.]])
    empty = np.array([[[0., 1.], [np.inf, -np.inf]], [[2., 3.], [np.inf, -np.inf]]])
    assert _intersect(empty).shape == (0, 2)


def test_loo_unireg():
    # uniReg with loo=True (lam = 0) is the uniLasso with penalties kappa_j / |b_j|
    rng = np.random.default_rng(9)
    X, y, _ = _data(rng, n=200, p=6, beta=(0.35, -0.25, 0.15, 0, 0, 0))
    beta = _loo_fit(X, y, 0.)
    assert np.any(beta == 0) and np.any(beta != 0)
    with pytest.warns(UserWarning, match='approximate'):
        UI = unilasso_inference(X, y, beta, 0., loo=True)
    X1 = np.column_stack([np.ones(len(y)), X])
    np.testing.assert_allclose(UI.summary_['beta_hat'].values,
                               np.linalg.lstsq(X1, y, rcond=None)[0][1:][UI.E], atol=1e-6)
