import numpy as np
import pytest
from scipy.stats import norm as normal_dbn
from scipy.sparse.linalg import aslinearoperator

from lassoinf import LassoInference
from lassoinf.custom_estimand import (SelectionCoordinates,
                                      ScreenedSelection,
                                      ContrastEstimand,
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
        np.testing.assert_allclose(coords.inactive_score(j).value(LI.Z_full), U[k])
        coef = np.linalg.lstsq(X[:, list(E) + [j]], y, rcond=None)[0][-1]
        np.testing.assert_allclose(coords.inactive_coef(j).value(LI.Z_full), coef)

    with pytest.raises(ValueError):
        coords.inactive_score(E[0])

    # Q_hat as a matrix-free operator gives the same contrasts
    LI_op = LassoInference(beta_hat=LI.beta_hat, G_hat=LI.G_hat, Q_hat=aslinearoperator(LI.Q_hat),
                           D=LI.D, L=LI.L, U=LI.U, Z_full=LI.Z_full, Sigma=LI.Sigma,
                           scalar_noise=LI.si.scalar_noise)
    coords_op = SelectionCoordinates(LI_op)
    np.testing.assert_allclose(coords_op.contrast(a_E, a_Ec), eta)
    np.testing.assert_allclose(coords_op.inactive_coef(E_c[0]).eta, coords.inactive_coef(E_c[0]).eta)


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
    eta = SelectionCoordinates(LI).inactive_score(j).eta
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
    eta = coords.inactive_coef(j).eta
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
        est = coords.inactive_coef(j) if estimand == 'coef' else coords.inactive_score(j)
        res = contrast_inference(LI, est, level=level)
        # true value: population coefficient of x_j in model E u {j},
        # or x_j'(I-P_E)mu for the score
        covered.append(res.lower_conf <= est.value(X.T @ mu) <= res.upper_conf)
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


def test_lasso_basis():
    # lasso basis: beta_hat_E and -grad(beta_hat)_{-E}, same contrast, shifted value
    rng = np.random.default_rng(1)
    _, _, _, LI = _gaussian_instance(rng)
    coords = SelectionCoordinates(LI)
    E, E_c, Z_noisy = LI.E, LI.E_c, LI.si.Z_noisy
    a_E, a_Ec = rng.standard_normal(len(E)), rng.standard_normal(len(E_c))

    refit = coords.estimand(a_E, a_Ec, basis='refit')
    lasso = coords.estimand(a_E, a_Ec, basis='lasso')
    np.testing.assert_allclose(lasso.eta, refit.eta)
    # at the data used for selection the maps give the LASSO solution and gradient
    np.testing.assert_allclose(lasso.value(Z_noisy), a_E @ LI.beta_hat[E] - a_Ec @ LI.G_hat[E_c])
    j = E_c[0]
    np.testing.assert_allclose(coords.inactive_score(j, basis='lasso').value(Z_noisy), -LI.G_hat[j])

    res_refit = contrast_inference(LI, refit)
    res_lasso = contrast_inference(LI, lasso, null_value=lasso.offset)
    for attr in ['estimate', 'lower_conf', 'upper_conf']:
        np.testing.assert_allclose(getattr(res_lasso, attr), getattr(res_refit, attr) + lasso.offset)
    np.testing.assert_allclose(res_lasso.p_value, res_refit.p_value)

    with pytest.raises(ValueError):
        coords.estimand(a_E, a_Ec, basis='other')


SCREENING_RULES = [('threshold', None), ('top_k', 'first_dropped'), ('top_k', 'exact')]


def _screening_kwargs(LI, rule, conditioning, top_k=2):
    if rule == 'threshold':
        return dict(threshold=0.5 * LI.D[0])
    return dict(top_k=top_k, conditioning=conditioning)


def _screening_event(G, threshold=None, top_k=None, conditioning=None):
    # what ScreenedSelection conditions on: kept set, their signs and,
    # for first_dropped, the largest dropped variable and its sign
    order = np.argsort(-np.abs(G), kind='stable')
    if threshold is not None:
        keep = np.sort(np.nonzero(np.abs(G) > threshold)[0])
        extra = ()
    else:
        keep = np.sort(order[:top_k])
        extra = ()
        if conditioning == 'first_dropped' and len(order) > top_k:
            l0 = order[top_k]
            extra = (l0, np.sign(G[l0]))
    return (tuple(keep), tuple(np.sign(G[keep]))) + extra


@pytest.mark.parametrize('rule,conditioning', SCREENING_RULES)
def test_screening_polytope_exact(rule, conditioning):
    # on the LASSO event, A_screen (Z + omega) <= b_screen exactly when
    # the screening event is the same as observed
    rng = np.random.default_rng(1)
    _, _, _, LI = _gaussian_instance(rng)
    coords = SelectionCoordinates(LI)
    m = len(LI.E_c)
    kw = _screening_kwargs(LI, rule, conditioning)
    S = ScreenedSelection(LI, **kw)

    G_obs = LI.G_hat[LI.E_c]
    event_obs = _screening_event(G_obs, **kw)
    assert set(np.searchsorted(LI.E_c, S.screened)) == set(event_obs[0])
    K = len(S.screened)
    if rule == 'threshold':
        n_rows = K + 2 * (m - K)
    elif conditioning == 'first_dropped':
        n_rows = K + 2 * (m - K - 1)
        assert S.first_dropped == LI.E_c[event_obs[2]]
        assert S.first_dropped_sign == event_obs[3]
    else:
        n_rows = 2 * K * (m - K)
    assert S.A_screen.shape[0] == n_rows
    assert S.A.shape[0] == LI.A.shape[0] + n_rows

    L_chol = np.linalg.cholesky(LI.Sigma)
    agree, outcomes = 0, set()
    n_in_event = 0
    for _ in range(2000):
        Zp = LI.si.Z_noisy + 0.3 * L_chol @ rng.standard_normal(m + len(LI.E))
        if np.any(LI.A @ Zp > LI.b):
            continue
        n_in_event += 1
        G = -(coords.refit_score(Zp) + coords.lasso_score_offset)
        same = _screening_event(G, **kw) == event_obs
        in_poly = np.all(S.A_screen @ Zp <= S.b_screen)
        agree += (same == in_poly)
        outcomes.add(same)
    assert n_in_event > 100
    assert outcomes == {True, False}
    assert agree == n_in_event


def test_screened_summary():
    rng = np.random.default_rng(1)
    _, _, _, LI = _gaussian_instance(rng)
    S = ScreenedSelection(LI, top_k=2)
    assert S.conditioning == 'first_dropped'
    G = LI.G_hat[LI.E_c]
    order = np.argsort(-np.abs(G))
    top = LI.E_c[order[:2]]
    np.testing.assert_array_equal(S.screened, top)
    np.testing.assert_array_equal(S.screened_signs, np.sign(LI.G_hat[top]))
    assert S.first_dropped == LI.E_c[order[2]]

    for basis in ['refit', 'lasso']:
        df = inactive_summary(S, basis=basis)
        assert list(df.index) == list(S.screened)
    # screening constraints change the answer
    assert not np.allclose(inactive_summary(S).values, inactive_summary(LI, variables=S.screened).values)
    # first_dropped conditions on more than exact: its truncation intervals
    # are nested inside the exact ones, usually strictly
    S_exact = ScreenedSelection(LI, top_k=2, conditioning='exact')
    assert S_exact.first_dropped is None
    coords = SelectionCoordinates(LI)
    n_strict = 0
    for _ in range(20):
        eta = coords.contrast(rng.standard_normal(len(LI.E)), rng.standard_normal(len(LI.E_c)))
        contrast = LI.si.compute_contrast(eta)
        L_fd, U_fd = contrast.get_interval(0., S.A, S.b)
        L_ex, U_ex = contrast.get_interval(0., S_exact.A, S_exact.b)
        assert L_ex - 1e-9 <= L_fd <= U_fd <= U_ex + 1e-9
        n_strict += (L_fd > L_ex + 1e-9) or (U_fd < U_ex - 1e-9)
    assert n_strict > 0

    # nothing to condition on when all inactive variables are kept
    for conditioning in ['first_dropped', 'exact']:
        S_all = ScreenedSelection(LI, top_k=len(LI.E_c), conditioning=conditioning)
        assert S_all.A_screen.shape[0] == 0
        np.testing.assert_allclose(inactive_summary(S_all).values,
                                   inactive_summary(LI, variables=S_all.screened).values)

    # only one dropped variable: its sign is conditioned on directly
    S_one = ScreenedSelection(LI, top_k=len(LI.E_c) - 1)
    assert S_one.A_screen.shape[0] == len(LI.E_c)

    with pytest.raises(ValueError):
        ScreenedSelection(LI)
    with pytest.raises(ValueError):
        ScreenedSelection(LI, threshold=1., top_k=1)
    with pytest.raises(ValueError):
        ScreenedSelection(LI, top_k=1, conditioning='other')


def test_matrix_free_constraints():
    # A built from Q_hat matvecs equals the explicit dense constraints
    rng = np.random.default_rng(1)
    _, _, _, LI = _gaussian_instance(rng)
    Q, E, E_c, W, s = LI.Q_hat, LI.E, LI.E_c, LI.W, LI.s_E
    p = Q.shape[0]
    np.testing.assert_allclose(W, np.linalg.inv(Q[np.ix_(E, E)]))

    R = np.zeros((len(E_c), p))          # U_{-E} = R Z
    R[np.arange(len(E_c)), E_c] = 1.
    R[:, E] = -Q[np.ix_(E_c, E)] @ W
    B = np.zeros((len(E), p))            # bar_beta_E = B Z
    B[:, E] = W
    rows = [-s[:, None] * B]
    for k in range(len(E_c)):
        rows.extend([R[k:k+1], -R[k:k+1]])
    A_explicit = np.vstack(rows)
    np.testing.assert_allclose(LI.A.to_dense(), A_explicit, atol=1e-10)
    y = rng.standard_normal(A_explicit.shape[0])
    np.testing.assert_allclose(LI.A.rmatvec(y), A_explicit.T @ y, atol=1e-10)

    coords = SelectionCoordinates(LI)
    U = coords.score_operator()
    z = rng.standard_normal(len(E_c))
    np.testing.assert_allclose(U.rmatvec(z), R.T @ z, atol=1e-10)

    # S_jj: batch (|E| matvecs, needs diag(Q)) and per-variable agree,
    # including for a matrix-free Q_hat with Q_diag supplied
    LI_op = LassoInference(beta_hat=LI.beta_hat, G_hat=LI.G_hat, Q_hat=aslinearoperator(Q),
                           D=LI.D, L=LI.L, U=LI.U, Z_full=LI.Z_full, Sigma=aslinearoperator(LI.Sigma),
                           scalar_noise=LI.si.scalar_noise)
    S_direct = np.array([Q[j, j] - Q[j, E] @ W @ Q[E, j] for j in E_c])
    np.testing.assert_allclose(SelectionCoordinates(LI).inactive_S(E_c), S_direct)
    np.testing.assert_allclose(SelectionCoordinates(LI_op).inactive_S(E_c), S_direct)
    np.testing.assert_allclose(SelectionCoordinates(LI_op, Q_diag=np.diag(Q)).inactive_S(E_c), S_direct)
    np.testing.assert_allclose(inactive_summary(LI_op, Q_diag=np.diag(Q)).values,
                               inactive_summary(LI).values, rtol=1e-6)


@pytest.mark.parametrize('rule,conditioning', SCREENING_RULES)
def test_coverage_screened(rule, conditioning):
    # coefficient of the inactive variable with the largest gradient
    rng = np.random.default_rng(7)
    n_rep, level = 200, 0.9
    covered, lasso_only_covered = [], []
    for _ in range(n_rep):
        X, y, mu, LI = _gaussian_instance(rng, kappa=0.2)
        if len(LI.E) == 0 or len(LI.E_c) < 2:
            continue
        S = ScreenedSelection(LI, **_screening_kwargs(LI, rule, conditioning, top_k=1))
        if len(S.screened) == 0:
            continue
        est = SelectionCoordinates(LI).inactive_coef(S.screened[0], basis='lasso')
        truth = est.value(X.T @ mu)
        res = contrast_inference(S, est, level=level)
        covered.append(res.lower_conf <= truth <= res.upper_conf)
        res = contrast_inference(LI, est, level=level)
        lasso_only_covered.append(res.lower_conf <= truth <= res.upper_conf)

    se = np.sqrt(level * (1 - level) / len(covered))
    assert abs(np.mean(covered) - level) < 4 * se, np.mean(covered)
    # ignoring the screening step undercovers
    assert np.mean(lasso_only_covered) < level - 4 * se, np.mean(lasso_only_covered)
