import warnings

import numpy as np
import pandas as pd
import pytest

from lassoinf import LassoInference, covariance_lasso_inference, box_lasso_fit, selective_mle
from lassoinf.covariance_lasso import box_lasso_kkt_violation, expected_max_abs


def _cov(rng, p, rho=0.3):
    A = rng.standard_normal((p, p))
    return rho * A @ A.T / p + np.diag(rng.uniform(0.5, 2, p))


def _prox_grad(Q, Y, D, L, U, n_iter=20000):
    # reference: proximal gradient for the box-constrained lasso
    t = 1 / np.linalg.eigvalsh(Q).max()
    b = np.zeros(len(Y))
    for _ in range(n_iter):
        v = b - t * (Q @ b - Y)
        b = np.clip(np.sign(v) * np.maximum(np.abs(v) - t * D, 0), L, U)
    return b


def test_box_lasso_fit():
    rng = np.random.default_rng(0)
    p = 12
    Q = np.linalg.inv(_cov(rng, p))
    Y = 3 * rng.standard_normal(p)
    D = rng.uniform(0, 1, p)
    D[0] = 0
    L, U = np.full(p, -np.inf), np.full(p, np.inf)
    L[1] = U[1] = 0          # excluded
    U[2] = 0.05              # an upper bound
    L[3] = 0                 # a sign constraint
    b = box_lasso_fit(Q, Y, D, L, U)
    assert b[1] == 0 and b[2] <= 0.05 and b[3] >= 0
    assert box_lasso_kkt_violation(b, Q, Y, D, L, U) < 1e-8
    np.testing.assert_allclose(b, _prox_grad(Q, Y, D, L, U), atol=1e-6)
    # a warm start gives the same solution
    np.testing.assert_allclose(box_lasso_fit(Q, Y, D, L, U, beta0=b + 1), b, atol=1e-8)


def test_box_lasso_errors():
    Q = np.eye(2)
    with pytest.raises(ValueError):
        box_lasso_fit(Q, np.ones(2), -np.ones(2), -np.ones(2), np.ones(2))
    with pytest.raises(ValueError):
        box_lasso_fit(Q, np.ones(3), np.ones(3), -np.ones(3), np.ones(3))


def test_diagonal_selection():
    # with uncorrelated estimates, j is selected when |beta_star_j| / SE(beta_star_j) > pf_j t
    rng = np.random.default_rng(1)
    p, pi = 40, 0.7
    se = rng.uniform(0.5, 3, p)
    beta_hat = se * rng.standard_normal(p) * 2
    pf = rng.uniform(0.5, 1.5, p)
    res = covariance_lasso_inference(beta_hat, np.diag(se**2), prop_select=pi,
                                     penalty_factor=pf, threshold=1.5, rng=2)
    z_star = res.beta_star.values / (se / np.sqrt(pi))
    np.testing.assert_array_equal(np.asarray(res.selected), np.nonzero(np.abs(z_star) > 1.5 * pf)[0])
    # and the lasso soft-thresholds beta_star
    lasso = np.sign(z_star) * np.maximum(np.abs(z_star) - 1.5 * pf, 0) * se / np.sqrt(pi)
    np.testing.assert_allclose(res.lasso_coef.values, lasso, atol=1e-6 * se.max())


def test_names_exclude_penalty_factor():
    rng = np.random.default_rng(3)
    p = 15
    Sigma = _cov(rng, p)
    names = [f'x{j}' for j in range(p)]
    beta = np.r_[5, 5, 0.1, np.zeros(p - 3)] * np.sqrt(np.diag(Sigma))
    beta_hat = pd.Series(beta + np.linalg.cholesky(Sigma) @ rng.standard_normal(p), index=names)
    pf = np.ones(p)
    pf[2] = 0
    res = covariance_lasso_inference(beta_hat, pd.DataFrame(Sigma, index=names, columns=names),
                                     exclude=['x1'], penalty_factor=pf, rng=4)
    assert 'x1' not in res.selected
    assert res.lasso_coef['x1'] == 0
    assert 'x2' in res.selected          # unpenalized
    assert list(res.summary_.index) == list(res.selected)
    # positions and a mask exclude the same coefficient
    for exclude in ([1], np.arange(p) == 1):
        other = covariance_lasso_inference(beta_hat, Sigma, exclude=exclude, penalty_factor=pf,
                                           rng=4)
        np.testing.assert_allclose(other.summary_.values, res.summary_.values)


def test_perturbation_and_threshold():
    rng = np.random.default_rng(5)
    p, pi = 10, 0.8
    Sigma = _cov(rng, p)
    beta_hat = rng.standard_normal(p)
    omega = rng.standard_normal(p)
    res = covariance_lasso_inference(beta_hat, Sigma, prop_select=pi, perturbation=omega,
                                     lambda_mult=1.3, rng=0)
    np.testing.assert_allclose(res.beta_star.values, beta_hat + np.sqrt((1 - pi) / pi) * omega)
    assert np.isclose(res.threshold, 1.3 * expected_max_abs(np.linalg.inv(Sigma), rng=0))
    assert res.inference.scalar_noise == pytest.approx((1 - pi) / pi)


def test_global_null_threshold():
    # with beta_P = 0, nothing is selected exactly when the standardized score of the
    # penalized coefficients, unpenalized ones profiled out, has max |xi_j| <= t
    rng = np.random.default_rng(11)
    p, pi, t = 12, 0.7, 2.2
    Sigma = _cov(rng, p, rho=2)
    pf = np.ones(p)
    pf[0] = 0
    exclude = [1]
    C = np.linalg.cholesky(Sigma)
    empty, maxes = [], []
    Q = np.linalg.inv(Sigma)
    P, N = np.arange(2, p), np.array([0])
    for _ in range(300):
        beta_hat = C @ rng.standard_normal(p)
        res = covariance_lasso_inference(beta_hat, Sigma, prop_select=pi, exclude=exclude,
                                         penalty_factor=pf, threshold=t, rng=rng)
        # the KKT check at b_P = 0, with b_N profiled and b_1 = 0
        K = np.r_[N, P]
        S = (Q @ res.beta_star.values)[K]
        QK = Q[np.ix_(K, K)]
        g = S[1:] - QK[1:, :1] @ np.linalg.solve(QK[:1, :1], S[:1])
        M = QK[1:, 1:] - QK[1:, :1] @ np.linalg.solve(QK[:1, :1], QK[:1, 1:])
        maxes.append(np.max(np.abs(g) / np.sqrt(np.diag(M) / pi)))
        empty.append(set(res.selected) <= {0})
    np.testing.assert_array_equal(np.array(maxes) <= t, empty)
    # and the default threshold is E max |xi| for xi ~ N(0, corr(M))
    res = covariance_lasso_inference(beta_hat, Sigma, exclude=exclude, penalty_factor=pf,
                                     rng=0)
    assert np.isclose(res.threshold, expected_max_abs(M, rng=0))


def test_expected_max_abs():
    # independent coordinates: compare with the integral of P(max |xi_j| > t)
    from scipy.stats import norm
    from scipy.integrate import quad
    p = 20
    exact = quad(lambda t: 1 - (2 * norm.cdf(t) - 1)**p, 0, np.inf)[0]
    assert np.isclose(expected_max_abs(np.diag(np.arange(1., p + 1)), n_mc=200000, rng=0),
                      exact, rtol=5e-3)


def test_selected_target_matches_lasso_inference():
    rng = np.random.default_rng(6)
    p = 20
    Sigma = _cov(rng, p)
    beta = np.r_[4, -3, 2, np.zeros(p - 3)] * np.sqrt(np.diag(Sigma))
    beta_hat = beta + np.linalg.cholesky(Sigma) @ rng.standard_normal(p)
    res = covariance_lasso_inference(beta_hat, Sigma, rng=7)
    assert len(res.selected) > 0
    Q = np.linalg.inv(Sigma)
    E = np.asarray(res.selected)
    np.testing.assert_allclose(res.summary_['estimate'],
                               np.linalg.solve(Q[np.ix_(E, E)], (Q @ beta_hat)[E]), rtol=1e-8)
    np.testing.assert_allclose(res.summary_[['lower_conf', 'upper_conf', 'p_value']].values,
                               res.inference.summary_[['lower_conf', 'upper_conf',
                                                       'p_value']].values)
    full = covariance_lasso_inference(beta_hat, Sigma, rng=7, target='full')
    np.testing.assert_array_equal(full.selected, res.selected)
    np.testing.assert_allclose(full.summary_['estimate'], beta_hat[E])
    np.testing.assert_allclose(full.summary_['std_err'], np.sqrt(np.diag(Sigma))[E])
    # the split estimate has 1 / (1 - pi) times the variance
    np.testing.assert_allclose(full.splitting_['std_err'],
                               np.sqrt(np.diag(Sigma))[E] / np.sqrt(1 - 0.8))


def test_selective_mle():
    rng = np.random.default_rng(8)
    p = 20
    Sigma = _cov(rng, p)
    beta_hat = np.r_[3, 2.5, np.zeros(p - 2)] * np.sqrt(np.diag(Sigma)) + \
        np.linalg.cholesky(Sigma) @ rng.standard_normal(p)
    res = covariance_lasso_inference(beta_hat, Sigma, rng=9)
    inf = res.inference
    for k, j in enumerate(inf.E):
        v = np.zeros(p)
        v[inf.E] = inf.W[:, k]
        variance = v @ inf.Q_hat @ v
        family, _ = inf.truncated_family(v, variance)
        mle, se = selective_mle(family, res.summary_['estimate'].iloc[k], variance)
        # the score equation: E_mle[theta_hat | selection] = observed
        assert np.isclose(family.E(mle / variance, lambda x: x), res.summary_['estimate'].iloc[k])
        assert np.isclose(mle, res.summary_['mle'].iloc[k])
        assert se > 0


def test_calibration():
    # p-values of null 'full' targets are roughly uniform
    rng = np.random.default_rng(10)
    p = 15
    Sigma = _cov(rng, p)
    beta = np.r_[3, -2, np.zeros(p - 2)] * np.sqrt(np.diag(Sigma))
    C = np.linalg.cholesky(Sigma)
    pvals, cover = [], []
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for _ in range(150):
            res = covariance_lasso_inference(beta + C @ rng.standard_normal(p), Sigma,
                                             target='full', lambda_mult=0.7, level=0.9, rng=rng)
            E = np.asarray(res.selected)
            s = res.summary_
            cover.extend((s.lower_conf <= beta[E]) & (beta[E] <= s.upper_conf))
            pvals.extend(s.p_value[beta[E] == 0])
    assert abs(np.mean(cover) - 0.9) < 0.05
    assert len(pvals) > 50
    assert abs(np.mean(np.asarray(pvals) < 0.2) - 0.2) < 0.08
