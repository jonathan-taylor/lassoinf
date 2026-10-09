import numpy as np
import pytest
from scipy.stats import kstest

from lassoinf.lasso import lasso_post_selection_constraints
from lassoinf.unilasso import (UniLassoInference,
                               unilasso_inference,
                               unilasso_fit,
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
            S = UI.ratio_constraints(k).truncation_set(con)
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
        UI = unilasso_inference(X, y, _fit(X, y, lam), lam, sigma2=1.)
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
    UI = unilasso_inference(X, y, _fit(X, y, lam), lam)
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
        unilasso_inference(X, y, beta * 1.2, lam)
    wrong_sign = beta.copy()
    wrong_sign[np.argmax(np.abs(beta))] *= -1
    with pytest.raises(ValueError, match='sign'):
        unilasso_inference(X, y, wrong_sign, lam)
    with pytest.raises(ValueError, match='n > p'):
        unilasso_inference(rng.standard_normal((5, 8)), rng.standard_normal(5), np.zeros(8), 1.)
