import numpy as np
import pytest
from scipy.sparse.linalg import aslinearoperator

from lassoinf.lasso import lasso_post_selection_constraints, coordinate_tolerance


def _problem(seed=0, n=6):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((50, n))
    Q = A.T @ A / 50
    beta = np.r_[0.8, -0.5, 0.2, 0., 0., 0.]
    D = np.full(n, 0.1)
    # KKT: G = -D sign(beta) on the active set, |G| < D off it
    G = np.where(beta != 0, -D * np.sign(beta), 0.05 * rng.uniform(-1, 1, n))
    return beta, G, Q, D


def _rescale(beta, G, Q, D, c):
    # coordinate scales c: beta_j / c_j, G_j c_j, Q_jk c_j c_k, D_j c_j
    return beta / c, G * c, Q * np.outer(c, c), D * c


@pytest.mark.parametrize('c', [1e-8, 1e-4, 1e4, 1e8])
def test_feature_rescaling(c):
    beta, G, Q, D = _problem()
    scales = np.ones(len(beta))
    scales[2] = c                         # an active coordinate
    cons = lasso_post_selection_constraints(beta, G, Q, D)
    cons_c = lasso_post_selection_constraints(*_rescale(beta, G, Q, D, scales))
    np.testing.assert_array_equal(cons[2], cons_c[2])      # E
    np.testing.assert_array_equal(cons[4], cons_c[4])      # s_E


@pytest.mark.parametrize('k', [1e-8, 1e8])
def test_response_rescaling(k):
    # rescaling the response (gaussian: beta, G and D scale with it) keeps the active set
    beta, G, Q, D = _problem()
    cons = lasso_post_selection_constraints(beta, G, Q, D)
    cons_k = lasso_post_selection_constraints(k * beta, k * G, Q, k * D)
    np.testing.assert_array_equal(cons[2], cons_k[2])


def test_small_coefficient_of_a_large_feature():
    # 5e-7 is a small coefficient only for a feature on a small scale
    beta, G, Q, D = _problem()
    beta[2] = 5e-7
    scales = np.ones(len(beta))
    scales[2] = 1e-6          # the feature is 1e6 times larger: its coefficient 5e-7 is 0.5 per unit of it
    rescaled = _rescale(beta * np.where(np.arange(len(beta)) == 2, 1e6, 1.), G, Q, D, scales)
    assert 2 in lasso_post_selection_constraints(*rescaled)[2]
    # on the scale of the other features it is numerically 0
    assert 2 not in lasso_post_selection_constraints(beta, G, Q, D)[2]


def test_matrix_free_Q_uses_common_scale():
    beta, _, Q, _ = _problem()
    tol_dense = coordinate_tolerance(beta, Q)
    tol_op = coordinate_tolerance(beta, aslinearoperator(Q))
    np.testing.assert_allclose(tol_dense, 1e-6 * np.max(np.abs(beta) * np.sqrt(np.diag(Q))) / np.sqrt(np.diag(Q)))
    np.testing.assert_allclose(tol_op, np.full(len(beta), 1e-6 * np.abs(beta).max()))


def test_all_zero():
    beta, G, Q, D = _problem()
    cons = lasso_post_selection_constraints(np.zeros_like(beta), np.zeros_like(G), Q, D)
    assert len(cons[2]) == 0
