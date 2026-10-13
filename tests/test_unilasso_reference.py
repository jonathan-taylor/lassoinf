"""
The C++ uniLasso computations (lassoinf_cpp) against the pure-Python reference
implementations in unilasso_reference.py.
"""
import numpy as np
import pytest

import lassoinf_cpp
import unilasso_reference as ref
from lassoinf.unilasso import (UniLassoInference,
                               unilasso_fit,
                               unilasso_kkt_violation,
                               unilasso_loo_kappa,
                               unilasso_penalty)


def _problem(rng, n=80, p=6, kappa=0.5):
    X = rng.standard_normal((n, p)) @ np.linalg.cholesky(0.5 * np.eye(p) + 0.5).T
    y = X @ np.r_[0.3, -0.25, 0.2, np.zeros(p - 3)] + rng.standard_normal(n)
    Q, Z = X.T @ X, X.T @ y
    Z_noisy = Z + np.linalg.cholesky(kappa * Q) @ rng.standard_normal(p)
    return X, y, Q, Z, Z_noisy


def _drop_slivers(intervals):
    # as intersect_row_sets, drop intervals of rounding width
    intervals = np.asarray(intervals).reshape(-1, 2)
    bounded = np.isfinite(intervals).all(1)
    with np.errstate(invalid='ignore'):
        scale = np.maximum(1., np.abs(intervals).max(1))
        wide = intervals[:, 1] - intervals[:, 0] > 64 * np.finfo(float).eps * scale
    return intervals[~bounded | wide]


def _as_pairs(sets):
    # C++ row sets (rows, 4) as the reference's (rows, 2, 2)
    return sets.reshape(-1, 2, 2)


def test_quadratic_le_zero():
    rng = np.random.default_rng(0)
    q2, q1, q0 = rng.standard_normal((3, 400))
    q2[:50] = 0.                 # linear rows
    q1[:10] = 0.                 # constant rows
    q0[:5] = -1.
    (lo1, hi1), (lo2, hi2) = ref._quadratic_le_zero(q2, q1, q0)
    np.testing.assert_allclose(lassoinf_cpp.quadratic_le_zero(q2, q1, q0),
                               np.column_stack([lo1, hi1, lo2, hi2]), rtol=1e-14)


@pytest.mark.parametrize('s', [1., -1.])
@pytest.mark.parametrize('gamma', [0.8, -0.5, 0.])
def test_row_sets_and_intersection(s, gamma):
    rng = np.random.default_rng(1)
    for _ in range(20):
        rows = rng.integers(1, 30)
        u, a, b0, m = rng.standard_normal((4, rows))
        u -= 2.                  # so the intersection is often nonempty
        c = rng.standard_normal()
        expected = ref._row_sets(u, a, b0, m, c, gamma, s)
        sets = lassoinf_cpp.unilasso_row_sets(u, a, b0, m, c, gamma, s)
        np.testing.assert_allclose(_as_pairs(sets), expected, rtol=1e-14)
        np.testing.assert_allclose(lassoinf_cpp.intersect_row_sets(sets), _drop_slivers(ref._intersect(expected)),
                                   rtol=1e-14)


def test_intersection_edge_cases():
    np.testing.assert_array_equal(lassoinf_cpp.intersect_row_sets(np.zeros((0, 4))),
                                  [[-np.inf, np.inf]])
    empty = np.array([[0., 1., np.inf, -np.inf], [2., 3., np.inf, -np.inf]])
    assert lassoinf_cpp.intersect_row_sets(empty).shape == (0, 2)
    # touching pieces merge, as in the reference
    touching = np.array([[0., 1., 1., 2.]])
    np.testing.assert_array_equal(lassoinf_cpp.intersect_row_sets(touching),
                                  ref._intersect(touching.reshape(-1, 2, 2)))


@pytest.mark.parametrize('lam', [0., 5., 'vector'])
def test_fit_penalty_kkt(lam):
    rng = np.random.default_rng(2)
    _, _, Q, _, Z_noisy = _problem(rng)
    if lam == 'vector':
        lam = rng.uniform(1., 8., len(Z_noisy))
    for got, expected in zip(unilasso_penalty(Z_noisy, Q, lam), ref.unilasso_penalty(Z_noisy, Q, lam)):
        np.testing.assert_allclose(got, expected, rtol=1e-14)
    beta = unilasso_fit(Q, Z_noisy, lam)
    np.testing.assert_allclose(beta, ref.unilasso_fit(Q, Z_noisy, lam), rtol=1e-10, atol=1e-12)
    for b in (beta, 1.1 * beta, np.zeros_like(beta)):
        np.testing.assert_allclose(unilasso_kkt_violation(b, Q, Z_noisy, lam),
                                   ref.unilasso_kkt_violation(b, Q, Z_noisy, lam), rtol=1e-12, atol=1e-12)


def test_penalty_rejects_a_zero_coefficient():
    with pytest.raises(ValueError, match='exactly zero'):
        unilasso_penalty(np.array([1., 0.]), np.eye(2), 1.)


def test_loo_kappa():
    rng = np.random.default_rng(3)
    X, y, _, _, _ = _problem(rng, n=50)
    X, y = X - X.mean(0), y - y.mean()
    beta = rng.standard_normal(X.shape[1]) * 0.1
    np.testing.assert_allclose(unilasso_loo_kappa(X, y, beta), ref.unilasso_loo_kappa(X, y, beta),
                               rtol=1e-12)


@pytest.mark.parametrize('lam', [0., 4.])
def test_truncation_sets(lam):
    # the whole truncation set, from the solution and the line for each target
    rng = np.random.default_rng(4)
    n_checked = 0
    while n_checked < 10:
        _, _, Q, Z, Z_noisy = _problem(rng)
        UI = UniLassoInference.from_selection(unilasso_fit(Q, Z_noisy, lam), Z_noisy, Q, lam, Z, Q,
                                              scalar_noise=0.5)
        UI.setup_constraints()
        p = len(Z)
        for k in UI.E:
            contrast = UI.si.compute_contrast(np.linalg.solve(Q, np.eye(p)[k]))
            bar_s2 = float(contrast.bar_s)**2
            gamma = float(Q[k] @ contrast.direction) / Q[k, k] / bar_s2
            w_obs = bar_s2 / float(contrast.naive_variance) * float(contrast.theta_hat) + float(contrast.bar_theta)
            expected = ref.truncation_set(UI.beta_hat, UI.G_hat, Q, UI.D, UI.L, UI.U, k, lam, UI.C[k],
                                          contrast.n_o + contrast.bar_n_o, contrast.bar_gamma,
                                          w_obs, gamma)
            expected = _drop_slivers(expected)
            got = UI.truncation_set(k, contrast)
            assert got.shape == expected.shape
            np.testing.assert_allclose(got, expected, rtol=1e-9, atol=1e-9)
            n_checked += 1
