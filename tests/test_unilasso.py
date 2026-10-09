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
        UI = unilasso_inference(X, y, lam, scalar_noise=0.5, sigma2=1., rng=rng)
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


@pytest.mark.parametrize('lam_scale,n_rep', [(0., 120), (0.5, 250)])
def test_calibration(lam_scale, n_rep):
    # pivots at the true full-model coefficients are uniform; lam = 0 is unireg
    rng = np.random.default_rng(2)
    pivots = []
    for _ in range(n_rep):
        X, y, beta = _data(rng)
        UI = unilasso_inference(X, y, lam_scale * np.sqrt(X.shape[0]), scalar_noise=1., sigma2=1., rng=rng)
        pivots.extend(UI._results[k].pivot(beta[k]) for k in UI.E)
    assert len(pivots) > 200
    assert kstest(pivots, 'uniform').pvalue > 1e-3


def test_summary_targets_full_model():
    rng = np.random.default_rng(3)
    X, y, _ = _data(rng)
    UI = unilasso_inference(X, y, 2., rng=rng)
    ols = np.linalg.solve(X.T @ X, X.T @ y)
    np.testing.assert_allclose(UI.summary_['beta_hat'].values, ols[UI.E])
    assert np.all(np.isfinite(UI.summary_[['lower_conf', 'upper_conf']].values))
    assert np.all(UI.summary_['lower_conf'] < UI.summary_['upper_conf'])


def test_requires_n_greater_than_p():
    rng = np.random.default_rng(4)
    with pytest.raises(ValueError, match='n > p'):
        unilasso_inference(rng.standard_normal((5, 8)), rng.standard_normal(5), 1.)
