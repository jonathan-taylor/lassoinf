import numpy as np
import pytest
from scipy.sparse.linalg import aslinearoperator

import lassoinf_cpp
from lassoinf.lasso import lasso_post_selection_constraints
from lassoinf.custom_estimand import (SelectionCoordinates,
                                      ScreenedSelection,
                                      covariance_contrast)

from test_custom_estimand import _gaussian_instance, SCREENING_RULES, _screening_kwargs


def _dense(op):
    n = op.cols()
    return np.column_stack([op.multiply(np.eye(n)[:, i]) for i in range(n)])


def _random_problem(rng):
    # mix of zeros, active coefficients, coordinates at upper / lower bounds,
    # nonnegativity bounds and unpenalized coordinates
    p = rng.integers(3, 15)
    X = rng.standard_normal((2 * p, p))
    Q = X.T @ X
    L = np.where(rng.random(p) < 0.3, rng.choice([0., -1.], p), -np.inf)
    U = np.where(rng.random(p) < 0.3, rng.choice([0., 1.5], p), np.inf)
    L, U = np.minimum(L, U), np.maximum(L, U)
    kind = rng.integers(0, 4, p)
    act = kind == 1
    beta = np.zeros(p)
    beta[act] = rng.choice([-1, 1], act.sum()) * rng.uniform(0.2, 0.9, act.sum())
    beta = np.clip(beta, L + 0.05, U - 0.05) * act
    at_U = (kind == 2) & np.isfinite(U)
    at_L = (kind == 3) & np.isfinite(L)
    beta[at_U] = U[at_U]
    beta[at_L] = L[at_L]
    D = rng.uniform(0.5, 2, p)
    D[rng.random(p) < 0.2] = 0.
    return beta, rng.standard_normal(p), Q, D, L, U


def test_constraints_match_python():
    rng = np.random.default_rng(0)
    for trial in range(100):
        beta, G, Q, D, L, U = _random_problem(rng)
        if trial % 7 == 0:
            beta[:] = 0
        A_py, b_py, E, E_c, s_E, v_Ec = lasso_post_selection_constraints(beta, G, Q, D, L, U)
        cons = lassoinf_cpp.lasso_post_selection_constraints(beta, G, lassoinf_cpp.DenseOperator(Q), D, L, U)
        np.testing.assert_array_equal(cons.E, E)
        np.testing.assert_array_equal(cons.E_c, E_c)
        np.testing.assert_array_equal(cons.s_E, s_E)
        np.testing.assert_array_equal(cons.v_Ec, v_Ec)
        np.testing.assert_allclose(cons.W, A_py.W, atol=1e-10)
        np.testing.assert_allclose(cons.b, b_py, atol=1e-10)
        assert (cons.A.rows(), cons.A.cols()) == A_py.shape
        if A_py.shape[0]:
            np.testing.assert_allclose(_dense(cons.A), A_py.to_dense(), atol=1e-10)
            y = rng.standard_normal(A_py.shape[0])
            np.testing.assert_allclose(cons.A.multiply_transpose(y), A_py.rmatvec(y), atol=1e-10)


def _cpp_setup(LI, Q_op=None, Q_diag=None):
    Q_op = lassoinf_cpp.DenseOperator(LI.Q_hat) if Q_op is None else Q_op
    cons = lassoinf_cpp.lasso_post_selection_constraints(LI.beta_hat, LI.G_hat, Q_op, LI.D, LI.L, LI.U)
    kw = {} if Q_diag is None else dict(Q_diag=Q_diag)
    coords = lassoinf_cpp.SelectionCoordinates(cons.A.score, LI.si.Z_noisy, LI.beta_hat, LI.G_hat, **kw)
    return cons, coords


def test_coordinates_match_python():
    rng = np.random.default_rng(1)
    _, _, _, LI = _gaussian_instance(rng)
    coords_py = SelectionCoordinates(LI)
    E, E_c = LI.E, LI.E_c
    # dense Q (diagonal known) and matrix-free X'X (diagonal supplied or not)
    X_op = lassoinf_cpp.XTVXOperator(np.linalg.cholesky(LI.Q_hat).T, lassoinf_cpp.DenseOperator(np.eye(len(E) + len(E_c))))
    for Q_op, Q_diag in [(None, None), (X_op, None), (X_op, np.diag(LI.Q_hat))]:
        cons, coords = _cpp_setup(LI, Q_op, Q_diag)
        np.testing.assert_allclose(coords.lasso_coef_offset, coords_py.lasso_coef_offset, atol=1e-8)
        np.testing.assert_allclose(coords.lasso_score_offset, coords_py.lasso_score_offset, atol=1e-8)
        Z = LI.Z_full
        np.testing.assert_allclose(coords.refit_score(Z), coords_py.refit_score(Z), atol=1e-8)

        a_E, a_Ec = rng.standard_normal(len(E)), rng.standard_normal(len(E_c))
        np.testing.assert_allclose(coords.contrast(a_E, a_Ec), coords_py.contrast(a_E, a_Ec), atol=1e-8)
        for basis in ['refit', 'lasso']:
            est, est_py = coords.estimand(a_E, a_Ec, basis), coords_py.estimand(a_E, a_Ec, basis)
            np.testing.assert_allclose(est.eta, est_py.eta, atol=1e-8)
            np.testing.assert_allclose(est.offset, est_py.offset, atol=1e-8)
            for j in E_c:
                for c, c_py in [(coords.inactive_score(j, basis), coords_py.inactive_score(j, basis)),
                                (coords.inactive_coef(j, basis), coords_py.inactive_coef(j, basis))]:
                    np.testing.assert_allclose(c.eta, c_py.eta, atol=1e-8)
                    np.testing.assert_allclose(c.offset, c_py.offset, atol=1e-8)
        np.testing.assert_allclose(coords.inactive_S(list(E_c)), coords_py.inactive_S(E_c), rtol=1e-10)

        with pytest.raises(ValueError):
            coords.inactive_score(int(E[0]))
        with pytest.raises(ValueError):
            coords.estimand(a_E, a_Ec, 'other')


@pytest.mark.parametrize('rule,conditioning', SCREENING_RULES)
def test_screening_match_python(rule, conditioning):
    rng = np.random.default_rng(1)
    _, _, _, LI = _gaussian_instance(rng)
    kw = _screening_kwargs(LI, rule, conditioning)
    S_py = ScreenedSelection(LI, **kw)
    cons, coords = _cpp_setup(LI)
    S = lassoinf_cpp.ScreenedSelection(cons.A, cons.b, coords, LI.G_hat, LI.si.Z_noisy, **kw)

    np.testing.assert_array_equal(S.screened, S_py.screened)
    np.testing.assert_array_equal(S.screened_signs, S_py.screened_signs)
    if S_py.first_dropped is None:
        assert S.first_dropped == -1
    else:
        assert S.first_dropped == S_py.first_dropped
        assert S.first_dropped_sign == S_py.first_dropped_sign
    assert S.observed_feasible
    np.testing.assert_allclose(S.b_screen, S_py.b_screen, atol=1e-8)
    np.testing.assert_allclose(_dense(S.A_screen), S_py.A_screen.matmat(np.eye(LI.Q_hat.shape[0])), atol=1e-8)
    np.testing.assert_allclose(S.b, S_py.b, atol=1e-8)
    np.testing.assert_allclose(_dense(S.A), S_py.A.to_dense(), atol=1e-8)

    # truncation intervals agree on the stacked constraints
    si = lassoinf_cpp.AffineConstraints(LI.Z_full, LI.si.Z_noisy, LI.Sigma, LI.si.scalar_noise * LI.Sigma)
    for j in S_py.screened:
        eta = SelectionCoordinates(LI).inactive_coef(j).eta
        np.testing.assert_allclose(si.compute_contrast(eta).get_interval(0., S.A, S.b),
                                   LI.si.compute_contrast(eta).get_interval(0., S_py.A, S_py.b), rtol=1e-6)

    with pytest.raises(ValueError):
        lassoinf_cpp.ScreenedSelection(cons.A, cons.b, coords, LI.G_hat, LI.si.Z_noisy)
    with pytest.raises(ValueError):
        lassoinf_cpp.ScreenedSelection(cons.A, cons.b, coords, LI.G_hat, LI.si.Z_noisy, top_k=1, conditioning='other')


def test_covariance_contrast_match_python():
    rng = np.random.default_rng(1)
    _, _, _, LI = _gaussian_instance(rng, use_Sigma_noise=True)
    eta = SelectionCoordinates(LI).inactive_coef(LI.E_c[0]).eta
    cov = LI.Sigma @ eta + 0.01 * rng.standard_normal(len(eta))
    theta_hat, variance = eta @ LI.Z_full, eta @ LI.Sigma @ eta + 1.

    si = lassoinf_cpp.AffineConstraints(LI.Z_full, LI.si.Z_noisy, LI.Sigma, LI.Sigma_noise)
    c_cpp = si.compute_covariance_contrast(theta_hat, variance, cov)
    c_py = covariance_contrast(LI.si, theta_hat, variance, cov)
    for attr in ['gamma', 'c', 'bar_gamma', 'bar_s', 'n_o', 'bar_n_o', 'theta_hat', 'bar_theta',
                 'splitting_variance', 'splitting_estimator', 'naive_variance']:
        np.testing.assert_allclose(getattr(c_cpp, attr), getattr(c_py, attr), rtol=1e-8, atol=1e-10)

    with pytest.raises(ValueError):
        si.compute_covariance_contrast(theta_hat, variance, np.zeros_like(cov))

    # compute_contrast matches Python, including the noise variance shortcut
    c_cpp, c_py = si.compute_contrast(eta), LI.si.compute_contrast(eta)
    for attr in ['bar_s', 'bar_theta', 'splitting_variance']:
        np.testing.assert_allclose(getattr(c_cpp, attr), getattr(c_py, attr), rtol=1e-8)
