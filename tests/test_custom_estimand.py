import numpy as np
import pytest
from scipy.stats import norm as normal_dbn
from scipy.sparse.linalg import aslinearoperator

from lassoinf import LassoInference
from lassoinf.custom_estimand import (SelectionCoordinates,
                                      CovarianceEstimand,
                                      contrast_inference,
                                      custom_estimand_inference,
                                      estimand_summary,
                                      inactive_summary,
                                      truncated_normal_inference)


def _lasso_cd(Q, Z, lam, n_iter=500):
    # minimize 1/2 b'Qb - b'Z + lam ||b||_1 by coordinate descent
    beta = np.zeros(Q.shape[0])
    for _ in range(n_iter):
        for j in range(Q.shape[0]):
            r = Z[j] - Q[j] @ beta + Q[j, j] * beta[j]
            beta[j] = np.sign(r) * max(abs(r) - lam, 0) / Q[j, j]
    return beta


def _gaussian_instance(rng, n=100, p=10, sigma2=1.0, kappa=0.5, use_Sigma_noise=False):
    X = rng.standard_normal((n, p))
    beta = np.zeros(p)
    beta[:2] = 3 / np.sqrt(n)
    mu = X @ beta
    y = mu + rng.normal(0, np.sqrt(sigma2), n)

    Q = X.T @ X
    Z = X.T @ y
    Sigma = sigma2 * Q
    omega = rng.multivariate_normal(np.zeros(p), kappa * Sigma)
    Z_noisy = Z + omega

    lam = 1.5 * np.sqrt(n * sigma2)
    beta_hat = _lasso_cd(Q, Z_noisy, lam)
    G_hat = Q @ beta_hat - Z_noisy

    kwargs = dict(Sigma_noise=kappa * Sigma) if use_Sigma_noise else dict(scalar_noise=kappa)
    LI = LassoInference(beta_hat=beta_hat,
                        G_hat=G_hat,
                        Q_hat=Q,
                        D=np.full(p, lam),
                        L=np.full(p, -np.inf),
                        U=np.full(p, np.inf),
                        Z_full=Z,
                        Sigma=Sigma,
                        **kwargs)
    return X, y, mu, LI


def _coef_contrasts(LI):
    W = np.linalg.inv(LI.Q_hat[np.ix_(LI.E, LI.E)])
    etas = {}
    for k, j in enumerate(LI.E):
        eta = np.zeros(LI.Q_hat.shape[0])
        eta[LI.E] = W[:, k]
        etas[j] = eta
    return etas


def test_selection_coordinates():
    rng = np.random.default_rng(1)
    X, y, _, LI = _gaussian_instance(rng)
    E, E_c = LI.E, LI.E_c
    assert len(E) > 0 and len(E_c) > 0

    # constraint coordinates in OLS form
    bar_beta = np.linalg.lstsq(X[:, E], y, rcond=None)[0]
    resid = y - X[:, E] @ bar_beta
    U = X[:, E_c].T @ resid

    coords = SelectionCoordinates(LI)
    a_E, a_Ec = rng.standard_normal(len(E)), rng.standard_normal(len(E_c))
    eta = coords.contrast(a_E, a_Ec)
    np.testing.assert_allclose(eta @ LI.Z_full, a_E @ bar_beta + a_Ec @ U)

    for k, j in enumerate(E_c):
        np.testing.assert_allclose(coords.inactive_score(j) @ LI.Z_full, U[k])
        coef = np.linalg.lstsq(X[:, list(E) + [j]], y, rcond=None)[0][-1]
        np.testing.assert_allclose(coords.inactive_coef(j) @ LI.Z_full, coef)

    with pytest.raises(ValueError):
        coords.inactive_score(E[0])

    # Q_hat as a matrix-free operator gives the same contrasts
    LI_op = LassoInference(beta_hat=LI.beta_hat, G_hat=LI.G_hat, Q_hat=aslinearoperator(LI.Q_hat),
                           D=LI.D, L=LI.L, U=LI.U, Z_full=LI.Z_full, Sigma=LI.Sigma,
                           scalar_noise=LI.si.scalar_noise)
    coords_op = SelectionCoordinates(LI_op)
    np.testing.assert_allclose(coords_op.contrast(a_E, a_Ec), eta)
    np.testing.assert_allclose(coords_op.inactive_coef(E_c[0]), coords.inactive_coef(E_c[0]))


@pytest.mark.parametrize('use_Sigma_noise', [False, True])
def test_reduces_to_coefficient_inference(use_Sigma_noise):
    rng = np.random.default_rng(1)
    _, _, _, LI = _gaussian_instance(rng, use_Sigma_noise=use_Sigma_noise)
    assert len(LI.E) > 0

    for j, eta in _coef_contrasts(LI).items():
        row = LI.summary_.loc[j]
        results = [contrast_inference(LI, eta)]
        if use_Sigma_noise:
            cov = LI.Sigma @ eta
            results.append(custom_estimand_inference(LI,
                                                     theta_hat=eta @ LI.Z_full,
                                                     variance=eta @ cov,
                                                     score_cov=cov))
        for res in results:
            np.testing.assert_allclose(res.estimate, row['beta_hat'], rtol=1e-8)
            np.testing.assert_allclose(res.lower_conf, row['lower_conf'], rtol=1e-5)
            np.testing.assert_allclose(res.upper_conf, row['upper_conf'], rtol=1e-5)
            np.testing.assert_allclose(res.p_value, row['p_value'], rtol=1e-5, atol=1e-10)


def test_inactive_constraints_matter():
    # for the score U_j of an inactive variable, only the inactive
    # (subgradient) rows of A involve theta_hat
    rng = np.random.default_rng(1)
    _, _, _, LI = _gaussian_instance(rng)
    E, E_c = LI.E, LI.E_c
    # no bounds: rows are |E| sign rows then 2 rows per inactive variable
    assert LI.A.shape[0] == len(E) + 2 * len(E_c)

    j = E_c[0]
    eta = SelectionCoordinates(LI).inactive_score(j)
    contrast = LI.si.compute_contrast(eta)
    variance = contrast.naive_variance

    full = contrast_inference(LI, eta)
    A_dense = LI.A.to_dense()
    np.testing.assert_allclose(truncated_normal_inference(contrast, variance, A_dense, LI.b, level=LI.level),
                               (full.lower_conf, full.upper_conf, full.p_value), rtol=1e-6)

    active_only = truncated_normal_inference(contrast, variance, A_dense[:len(E)], LI.b[:len(E)], level=LI.level)
    width_full = full.upper_conf - full.lower_conf
    width_active = active_only[1] - active_only[0]
    assert abs(width_full - width_active) > 0.05 * width_active


def test_summaries():
    rng = np.random.default_rng(1)
    _, _, _, LI = _gaussian_instance(rng, use_Sigma_noise=True)
    coords = SelectionCoordinates(LI)
    j = LI.E_c[0]
    eta = coords.inactive_coef(j)
    cov = LI.Sigma @ eta

    df = estimand_summary(LI, {'coef_contrast': eta,
                               'coef_cov': CovarianceEstimand(eta @ LI.Z_full, eta @ cov, cov)})
    assert list(df.columns) == ['estimate', 'lower_conf', 'upper_conf', 'p_value']
    np.testing.assert_allclose(df.loc['coef_contrast'], df.loc['coef_cov'], rtol=1e-5)

    for estimand in ['coef', 'score']:
        df = inactive_summary(LI, estimand=estimand)
        assert list(df.index) == list(LI.E_c)
        assert np.all(df['lower_conf'] <= df['estimate']) and np.all(df['estimate'] <= df['upper_conf'])
    np.testing.assert_allclose(inactive_summary(LI).loc[j], estimand_summary(LI, {j: eta}).loc[j])

    with pytest.raises(ValueError):
        inactive_summary(LI, estimand='other')


def test_covariance_estimand_requires_Sigma_noise():
    rng = np.random.default_rng(3)
    _, _, _, LI = _gaussian_instance(rng)
    eta = _coef_contrasts(LI)[LI.E[0]]
    cov = LI.Sigma @ eta
    with pytest.raises(ValueError, match='Sigma_noise'):
        custom_estimand_inference(LI, eta @ LI.Z_full, eta @ cov, cov)


def test_zero_score_cov_raises():
    rng = np.random.default_rng(3)
    _, _, _, LI = _gaussian_instance(rng, use_Sigma_noise=True)
    with pytest.raises(ValueError, match='zero'):
        custom_estimand_inference(LI, 0.0, 1.0, np.zeros(LI.Z_full.shape[0]))


@pytest.mark.parametrize('estimand', ['coef', 'score'])
def test_coverage_inactive(estimand):
    rng = np.random.default_rng(4)
    n_rep, level = 200, 0.9
    covered = []
    for _ in range(n_rep):
        X, y, mu, LI = _gaussian_instance(rng)
        if len(LI.E) == 0 or len(LI.E_c) == 0:
            continue
        j = LI.E_c[0]
        coords = SelectionCoordinates(LI)
        eta = coords.inactive_coef(j) if estimand == 'coef' else coords.inactive_score(j)
        res = contrast_inference(LI, eta, level=level)
        # true value: population coefficient of x_j in model E u {j},
        # or x_j'(I-P_E)mu for the score
        covered.append(res.lower_conf <= eta @ (X.T @ mu) <= res.upper_conf)
    se = np.sqrt(level * (1 - level) / len(covered))
    assert abs(np.mean(covered) - level) < 4 * se, np.mean(covered)


def test_coverage_covariance_estimand():
    # theta_hat = a'y with a not in the column span of X, so theta_hat
    # is not a function of Z_full; selective intervals should still cover
    rng = np.random.default_rng(2)
    sigma2, n_rep, level = 1.0, 300, 0.9
    q = normal_dbn.ppf(1 - (1 - level) / 2)
    covered, naive_covered = [], []
    for _ in range(n_rep):
        X, y, mu, LI = _gaussian_instance(rng, sigma2=sigma2, kappa=0.1, use_Sigma_noise=True)
        # first two features are the true signals
        nulls = [j for j in LI.E if j >= 2]
        if not nulls:
            continue
        # estimator depending on the selected model (a selected null column
        # plus a fixed perturbation), true value a'mu
        n = len(y)
        a = (X[:, nulls[0]] + 0.5 * np.linspace(-1, 1, n)) / n
        res = custom_estimand_inference(LI,
                                        theta_hat=a @ y,
                                        variance=sigma2 * a @ a,
                                        score_cov=sigma2 * X.T @ a,
                                        level=level)
        covered.append(res.lower_conf <= a @ mu <= res.upper_conf)
        naive_covered.append(abs(a @ y - a @ mu) <= q * np.sqrt(sigma2 * a @ a))

    se = np.sqrt(level * (1 - level) / len(covered))
    assert abs(np.mean(covered) - level) < 4 * se, np.mean(covered)
    # selection matters here: naive intervals undercover
    assert np.mean(naive_covered) < level - 4 * se, np.mean(naive_covered)
