import numpy as np
import pytest
import scipy.sparse as sp
from scipy.sparse.linalg import aslinearoperator

from lassoinf import LassoInference
from lassoinf.lasso import full_model_contrasts


def _lasso_cd(Q, Z, lam, n_iter=500):
    beta = np.zeros(Q.shape[0])
    for _ in range(n_iter):
        for j in range(Q.shape[0]):
            r = Z[j] - Q[j] @ beta + Q[j, j] * beta[j]
            beta[j] = np.sign(r) * max(abs(r) - lam, 0) / Q[j, j]
    return beta


def _instance(rng, n=100, p=10, kappa=0.5):
    X = rng.standard_normal((n, p))
    beta = np.zeros(p)
    beta[:2] = 3 / np.sqrt(n)
    y = X @ beta + rng.standard_normal(n)
    Q, Z = X.T @ X, X.T @ y
    Z_noisy = Z + rng.multivariate_normal(np.zeros(p), kappa * Q)
    lam = 1.5 * np.sqrt(n)
    beta_hat = _lasso_cd(Q, Z_noisy, lam)
    args = dict(beta_hat=beta_hat, G_hat=Q @ beta_hat - Z_noisy, D=np.full(p, lam),
                L=np.full(p, -np.inf), U=np.full(p, np.inf), Z_full=Z, Sigma=Q,
                scalar_noise=kappa)
    return X, y, Q, args


def test_full_target():
    rng = np.random.default_rng(0)
    X, y, Q, args = _instance(rng)
    LI = LassoInference(Q_hat=Q, target='full', **args)
    E = LI.E
    assert len(E) > 0

    # estimates are the full-model OLS coefficients
    ols = np.linalg.lstsq(X, y, rcond=None)[0]
    np.testing.assert_allclose(LI.summary_['beta_hat'], ols[E])
    Q_inv = np.linalg.inv(Q)
    for k, j in enumerate(E):
        np.testing.assert_allclose(LI._contrasts[j].direction, Q_inv[:, j], atol=1e-10)

    # sparse and matrix-free Q_hat agree with the dense Cholesky solve
    for Q_alt in [sp.csr_matrix(Q), aslinearoperator(Q)]:
        LI_alt = LassoInference(Q_hat=Q_alt, target='full', **args)
        np.testing.assert_allclose(LI_alt.summary_.to_numpy(), LI.summary_.to_numpy(), rtol=1e-6)

    # switching targets
    selected = LassoInference(Q_hat=Q, **args).summary_
    LI.compute_intervals(target='selected')
    np.testing.assert_allclose(LI.summary_.to_numpy(), selected.to_numpy())
    with pytest.raises(ValueError):
        LI.compute_intervals(target='other')


def test_full_target_singular():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((10, 20))
    Q = X.T @ X
    with pytest.raises(ValueError):
        full_model_contrasts(Q, [0, 1])
    with pytest.raises(ValueError):
        full_model_contrasts(aslinearoperator(Q), [0, 1])


def test_full_target_Q_solve():
    from scipy.linalg import cho_factor, cho_solve
    rng = np.random.default_rng(2)
    X, y, Q, args = _instance(rng)
    LI = LassoInference(Q_hat=Q, target='full', **args)

    # a cached factorization, used with a matrix-free Q_hat
    factor = cho_factor(Q)
    LI_solve = LassoInference(Q_hat=aslinearoperator(Q), target='full',
                              Q_solve=lambda B: cho_solve(factor, B), **args)
    np.testing.assert_allclose(LI_solve.summary_.to_numpy(), LI.summary_.to_numpy())

    with pytest.raises(ValueError):
        full_model_contrasts(Q, LI.E, Q_solve=lambda B: B[:, 0])
