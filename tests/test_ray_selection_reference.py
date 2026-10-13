"""
The C++ ray-selection computations (lassoinf_cpp) against the pure-Python reference
implementations in ray_selection_reference.py.
"""
import numpy as np
import pytest

import lassoinf_cpp
import ray_selection_reference as ref
from lassoinf import AffineConstraints
from lassoinf.ray_selection import find_truncation_set, lasso_event, ray_scale

from test_ray_selection import _adaptive_lasso_case


def _contrasts(rng, kappa, n_cases=4):
    # adaptive lasso fits and the contrasts for their full-model targets
    p = 4
    for _ in range(n_cases):
        Q, Z, Y, b, penalty = _adaptive_lasso_case(rng, 80, p, np.r_[0.35, -0.25, 0., 0.], 3., kappa)
        E = np.nonzero(b)[0]
        si = AffineConstraints(Z=Z, Z_noisy=Y, Q=Q, Q_noise=None, scalar_noise=kappa)
        for k in E:
            yield Q, Y, b, penalty, si.compute_contrast(np.linalg.solve(Q, np.eye(p)[k]))


def test_ray_scale():
    rng = np.random.default_rng(0)
    for _, _, _, _, contrast in _contrasts(rng, 1.):
        np.testing.assert_allclose(ray_scale(contrast), ref.ray_scale(contrast), rtol=1e-15)


@pytest.mark.parametrize('kappa', [0., 1.])
def test_lasso_event(kappa):
    # the C++ event agrees with the reference along the ray, including off the event
    rng = np.random.default_rng(1)
    for Q, Y, b, penalty, contrast in _contrasts(rng, kappa):
        event = lasso_event(b, Q @ b - Y, Q, penalty)
        expected = ref.lasso_event(b, Q @ b - Y, Q, penalty)
        w_obs, sd_w = ref.ray_scale(contrast)
        values = [(event(ref.ray_point(contrast, w)), expected(ref.ray_point(contrast, w)))
                  for w in w_obs + np.linspace(-6, 6, 241) * sd_w]
        got, want = np.array(values).T
        np.testing.assert_array_equal(got, want)
        assert not got.all()     # the ray leaves the event


@pytest.mark.parametrize('kappa', [0., 1.])
def test_truncation_sets(kappa):
    rng = np.random.default_rng(2)
    for Q, Y, b, penalty, contrast in _contrasts(rng, kappa):
        expected = ref.find_truncation_set(ref.lasso_event(b, Q @ b - Y, Q, penalty), contrast,
                                           n_sd=8, n_grid=401)
        sd_w = ref.ray_scale(contrast)[1]
        # with the C++ event, and with a Python callable
        for event in (lasso_event(b, Q @ b - Y, Q, penalty),
                      ref.lasso_event(b, Q @ b - Y, Q, penalty)):
            got = find_truncation_set(event, contrast, n_sd=8, n_grid=401)
            assert got.shape == expected.shape
            np.testing.assert_allclose(got, expected, rtol=0, atol=1e-9 * sd_w)


def test_python_event_errors_propagate():
    rng = np.random.default_rng(3)
    _, _, _, _, contrast = next(_contrasts(rng, 1.))
    with pytest.raises(ValueError, match='not in the selection event'):
        find_truncation_set(lambda Y: False, contrast)

    def broken(Y):
        raise RuntimeError('broken event')
    with pytest.raises(RuntimeError, match='broken event'):
        find_truncation_set(broken, contrast)


def test_interval_bookkeeping():
    # an event that is a known union of intervals in w, on the identity ray Y(w) = w
    pieces = [(-np.inf, -2.), (-0.5, 0.25), (1., 3.)]
    event = lambda Y: any(lo < Y[0] < hi for lo, hi in pieces)
    S = lassoinf_cpp.find_truncation_set(event, np.zeros(1), np.ones(1), 0., 1., n_sd=5., n_grid=201)
    np.testing.assert_allclose(S, pieces, atol=1e-8)


def _box_lasso(Q, Y, D, L, U, n_iter=20000):
    # minimize 1/2 b'Qb - b'Y + sum_j D_j |b_j| subject to L <= b <= U by coordinate descent
    b = np.zeros(len(Y))
    d = np.diag(Q)
    for _ in range(n_iter):
        old = b.copy()
        for j in range(len(Y)):
            r = Y[j] - Q[j] @ b + d[j] * b[j]
            b[j] = np.clip(np.sign(r) * max(abs(r) - D[j], 0.) / d[j], L[j], U[j])
        if np.max(np.abs(b - old)) < 1e-14:
            break
    return b


@pytest.mark.parametrize('moving', [False, True])
def test_bounds(moving):
    # coordinates at nonzero bounds (the offsets Q V of the bound rows), with bounds fixed at
    # their observed values or moving with Y. A coordinate at a moving bound stays at it only
    # near w_obs, so the bound activations change along the ray: each pattern gets its own
    # constraints in the precomputed path.
    rng = np.random.default_rng(4)
    n, p, lam = 100, 6, 2.
    at_bound = 0
    for _ in range(6):
        X = rng.standard_normal((n, p))
        Q, Z = X.T @ X, X.T @ (X @ np.r_[0.6, -0.5, 0.3, 0, 0, 0] + rng.standard_normal(n))
        Y = Z + np.linalg.cholesky(Q) @ rng.standard_normal(p)
        Qinv = np.linalg.inv(Q)

        L_obs, U_obs = -0.6 * np.abs(Qinv @ Y), 0.6 * np.abs(Qinv @ Y)

        def penalty(Y):
            b_ols = Qinv @ Y
            if moving:
                return lam / np.abs(b_ols), -0.6 * np.abs(b_ols), 0.6 * np.abs(b_ols)
            return lam / np.abs(b_ols), L_obs, U_obs

        D, L, U = penalty(Y)
        b = _box_lasso(Q, Y, D, L, U)
        at_bound += np.sum(np.isclose(b, L, atol=1e-8) | np.isclose(b, U, atol=1e-8))
        si = AffineConstraints(Z=Z, Z_noisy=Y, Q=Q, Q_noise=None, scalar_noise=1.)
        event = lasso_event(b, Q @ b - Y, Q, penalty)
        expected_event = ref.lasso_event(b, Q @ b - Y, Q, penalty)
        for k in np.nonzero(b)[0]:
            contrast = si.compute_contrast(Qinv[:, k])
            sd_w = ref.ray_scale(contrast)[1]
            expected = ref.find_truncation_set(expected_event, contrast, n_sd=8, n_grid=401)
            # the precomputed path, and the C++ event evaluated at each point
            for got in (find_truncation_set(event, contrast, n_sd=8, n_grid=401),
                        find_truncation_set(lambda Y: event(Y), contrast, n_sd=8, n_grid=401)):
                assert got.shape == expected.shape
                np.testing.assert_allclose(got, expected, rtol=0, atol=1e-9 * sd_w)
    assert at_bound > 0
