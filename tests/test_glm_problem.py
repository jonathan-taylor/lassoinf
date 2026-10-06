import itertools
import warnings

import numpy as np
import pandas as pd
import pytest
import cvxpy as cp
from scipy.special import expit
import statsmodels.genmod.families as smf

from glmnet import GLMNet
from glmnet.glmnet import GLMNetControl

from lassoinf import LassoInference
from lassoinf.lasso import lasso_post_selection_constraints
from lassoinf.glm_problem import glmnet_problem, glmstar_problem, kkt_violation
from lassoinf.custom_estimand import ScreenedSelection, inactive_summary

N, P = 120, 8
SM_FAMILIES = {'gaussian': smf.Gaussian(), 'binomial': smf.Binomial(), 'poisson': smf.Poisson()}


@pytest.fixture(scope='module')
def data():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((N, P)) * rng.uniform(0.5, 3, P) + rng.uniform(-1, 1, P)
    eta = X[:, :3] @ np.array([0.4, -0.3, 0.2])
    df = pd.DataFrame({'gaussian': eta + rng.standard_normal(N),
                       'binomial': rng.binomial(1, expit(eta)).astype(float),
                       'poisson': rng.poisson(np.exp(0.3 * eta)).astype(float),
                       'w': rng.uniform(0.5, 2, N),
                       'o': rng.normal(0, 0.2, N)})
    return X, df


# ---- an independent statement of glmnet's objective (smooth part) ----

def _scales(X, w, standardize):
    w = w / w.sum()
    return np.sqrt(w @ X**2 - (w @ X)**2) if standardize else np.ones(X.shape[1])


def _penalty_factor(pf, exclude):
    pf = np.ones(P) if pf is None else np.array(pf, dtype=float)
    excluded = np.isinf(pf)
    excluded[list(exclude)] = True
    pf[excluded] = 1.
    return pf * P / pf.sum(), excluded


def _smooth_objective(theta, X, y, family, w, offset, fit_intercept, ridge):
    a0, beta = (theta[0], theta[1:]) if fit_intercept else (0., theta)
    eta = offset + a0 + X @ beta
    if family == 'gaussian':
        nll = 0.5 * (y - eta)**2
    elif family == 'binomial':
        nll = np.logaddexp(0, eta) - y * eta
    else:
        nll = np.exp(eta) - y * eta
    return w @ nll / w.sum() + 0.5 * np.sum(ridge * beta**2)


def _numerical_gradient(f, theta, h=1e-6):
    E = np.eye(len(theta)) * h
    return np.array([(f(theta + e) - f(theta - e)) / (2 * h) for e in E])


def _check_problem(prob, X, y, family, w, offset, fit_intercept, lam, alpha=1., pf=None,
                   exclude=(), standardize=True, y_scale=1., kkt_tol=2e-5):
    # 1. the fit satisfies the KKT conditions of the extracted problem
    assert prob.kkt_violation() < kkt_tol * lam, prob.kkt_violation() / lam

    # 2. the gradient is that of glmnet's objective
    pf_star, excluded = _penalty_factor(pf, exclude)
    ridge = lam * (1 - alpha) * pf_star * _scales(X, w, standardize)**2 / y_scale
    ridge[excluded] = 0
    f = lambda theta: _smooth_objective(theta, X, y, family, w, offset, fit_intercept, ridge)
    np.testing.assert_allclose(prob.G_hat, _numerical_gradient(f, prob.beta_hat), atol=1e-6 * (1 + lam))

    # 3. Q is the Hessian: derivative of the extracted gradient
    def gradient(theta):
        a0, coef = (theta[0], theta[1:]) if fit_intercept else (0., theta)
        return glmnet_problem(X, y, coef, a0, lam, family=family, weights=w, offset=offset, alpha=alpha,
                              penalty_factor=pf, exclude=exclude, standardize=standardize,
                              fit_intercept=fit_intercept, y_scale=y_scale).G_hat
    h = 1e-6
    H = np.column_stack([(gradient(prob.beta_hat + h * e) - gradient(prob.beta_hat - h * e)) / (2 * h)
                         for e in np.eye(len(prob.beta_hat))])
    np.testing.assert_allclose(prob.Q_hat, H, atol=1e-5 * np.abs(H).max())

    # 4. the selection constraints hold at the data used for selection
    Z_noisy = -prob.G_hat + prob.Q_hat @ prob.beta_hat
    A, b, E, E_c, s_E, v_Ec = lasso_post_selection_constraints(prob.beta_hat, prob.G_hat, prob.Q_hat,
                                                               prob.D, prob.L, prob.U)
    free = (np.abs(prob.beta_hat) > 1e-6) & (prob.beta_hat > prob.L + 1e-6) & (prob.beta_hat < prob.U - 1e-6)
    np.testing.assert_array_equal(E, np.nonzero(free)[0])
    np.testing.assert_allclose(v_Ec, np.where(np.abs(prob.beta_hat[E_c]) <= 1e-6, 0., prob.beta_hat[E_c]), atol=1e-8)
    assert np.all(A @ Z_noisy <= b + kkt_tol * lam * (1 + np.abs(b)))


# ---- glmstar ----

OPTIONS = {
    'plain': {},
    'limits': dict(upper_limits=np.r_[0.1, np.full(P - 1, np.inf)],
                   lower_limits=np.r_[-np.inf, -0.1, np.full(P - 2, -np.inf)]),
    'penalty_factor': dict(penalty_factor=np.r_[0.5, 2., np.ones(P - 2)]),
    'alpha': dict(alpha=0.5),
    'weights': dict(weight_id='w'),
    'offset': dict(offset_id='o'),
    'alpha_weights': dict(alpha=0.3, weight_id='w'),
}

# glmstar 0.1.1 bugs: exclude is applied to the next variable (1-based indices
# used on a 0-based array in glmnet/elnet.py), and fits with a penalty factor
# of 0 or inf fail in IRLS
GLMSTAR_BROKEN = {
    'exclude': dict(exclude=[3, 5]),
    'unpenalized': dict(penalty_factor=np.r_[0., 2., np.ones(P - 2)]),
    'penalty_factor_inf': dict(penalty_factor=np.r_[1., 1., 1., np.inf, np.ones(P - 4)]),
}


def _fit_glmstar(X, df, family, standardize, fit_intercept, opts):
    G = GLMNet(family=SM_FAMILIES[family], standardize=standardize, fit_intercept=fit_intercept,
               response_id=family, nlambda=20, control=GLMNetControl(thresh=1e-14, fdev=0),
               # copies: glmstar rescales penalty_factor in place
               **{k: (list(v) if k == 'exclude' else np.copy(v) if isinstance(v, np.ndarray) else v)
                  for k, v in opts.items()})
    G.fit(X, df)
    return G


def _check_glmstar(data, family, standardize, fit_intercept, opts):
    X, df = data
    G = _fit_glmstar(X, df, family, standardize, fit_intercept, opts)
    lam = G.lambda_values_[8]
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        prob = glmstar_problem(G, X, df, lambda_val=lam)
    w = df['w'].values if opts.get('weight_id') else np.ones(N)
    offset = df['o'].values if opts.get('offset_id') else np.zeros(N)
    _check_problem(prob, X, df[family].values, family, w, offset, fit_intercept, lam,
                   alpha=opts.get('alpha', 1.), pf=opts.get('penalty_factor'),
                   exclude=opts.get('exclude', ()), standardize=standardize)
    return prob


@pytest.mark.parametrize('family,standardize,fit_intercept,option',
                         list(itertools.product(SM_FAMILIES, [True, False], [True, False], OPTIONS)))
def test_glmstar_problem(data, family, standardize, fit_intercept, option):
    prob = _check_glmstar(data, family, standardize, fit_intercept, OPTIONS[option])
    if option == 'limits':
        # the limits bind
        assert np.sum((prob.beta_hat >= prob.U - 1e-8) | (prob.beta_hat <= prob.L + 1e-8)) >= 1


@pytest.mark.xfail(strict=True, reason='glmstar 0.1.1: exclude off by one; penalty factor 0 / inf fails')
@pytest.mark.parametrize('option', GLMSTAR_BROKEN)
def test_glmstar_broken_options(data, option):
    _check_glmstar(data, 'gaussian', True, True, GLMSTAR_BROKEN[option])


def test_glmstar_unconverged_warns(data):
    # a deliberately loose fit is flagged
    X, df = data
    G = GLMNet(family=SM_FAMILIES['poisson'], response_id='poisson', lambda_values=np.array([0.08, 0.04]),
               control=GLMNetControl(thresh=1e-2, fdev=0))
    G.fit(X, df)
    with pytest.warns(UserWarning, match='KKT'):
        glmstar_problem(G, X, df, lambda_val=0.04)
    with pytest.raises(ValueError, match='lambda_values_'):
        glmstar_problem(G, X, df, lambda_val=0.05)


# ---- the core extraction against an independent solve of glmnet's objective ----

def _solve_cvxpy(X, y, family, w, offset, fit_intercept, lam, alpha, pf, exclude, lower, upper, standardize, y_scale):
    pf_star, excluded = _penalty_factor(pf, exclude)
    s = _scales(X, w, standardize)
    beta = cp.Variable(P)
    a0 = cp.Variable() if fit_intercept else 0.
    eta = offset + a0 + X @ beta
    wn = w / w.sum()
    if family == 'gaussian':
        nll = 0.5 * cp.sum(cp.multiply(wn, cp.square(y - eta)))
    elif family == 'binomial':
        nll = cp.sum(cp.multiply(wn, cp.logistic(eta) - cp.multiply(y, eta)))
    else:
        nll = cp.sum(cp.multiply(wn, cp.exp(eta) - cp.multiply(y, eta)))
    ridge = lam * (1 - alpha) * pf_star * s**2 / y_scale
    penalty = (0.5 * cp.sum(cp.multiply(ridge, cp.square(beta)))
               + cp.sum(cp.multiply(lam * alpha * pf_star * s, cp.abs(beta))))
    cons = [beta >= lower, beta <= upper]
    if excluded.any():
        cons.append(beta[np.nonzero(excluded)[0]] == 0)
    cp.Problem(cp.Minimize(nll + penalty), cons).solve(solver=cp.CLARABEL, tol_gap_abs=1e-10, tol_gap_rel=1e-10,
                                                       tol_feas=1e-10, max_iter=500)
    coef = np.where(np.abs(beta.value) < 1e-6, 0., beta.value)
    coef = np.clip(coef, lower, upper)
    coef[excluded] = 0.
    return coef, (float(a0.value) if fit_intercept else 0.)


CORE_OPTIONS = {
    # factors not summing to the number of variables, so the rescaling matters
    'penalty_factor': dict(pf=np.r_[0.5, 3., np.ones(P - 2)]),
    'exclude': dict(exclude=[3, 5]),
    'unpenalized': dict(pf=np.r_[0., 2., np.ones(P - 2)]),
    'penalty_factor_inf': dict(pf=np.r_[1., 1., 1., np.inf, np.ones(P - 4)]),
    'limits_alpha': dict(upper=np.r_[0.1, np.full(P - 1, np.inf)], lower=np.r_[-np.inf, -0.1, np.full(P - 2, -np.inf)],
                         alpha=0.6),
}


@pytest.mark.parametrize('family,standardize,fit_intercept,option',
                         list(itertools.product(SM_FAMILIES, [True, False], [True, False], CORE_OPTIONS)))
def test_glmnet_problem_matches_objective(data, family, standardize, fit_intercept, option):
    X, df = data
    y, w, offset = df[family].values, df['w'].values, df['o'].values
    o = dict(alpha=1., pf=None, exclude=(), lower=np.full(P, -np.inf), upper=np.full(P, np.inf))
    o.update(CORE_OPTIONS[option])
    lam = 0.03
    y_scale = 1.
    if family == 'gaussian':
        wn = w / w.sum()
        m2 = wn @ y**2
        y_scale = np.sqrt(m2 - (wn @ y)**2 if fit_intercept else m2)
    coef, a0 = _solve_cvxpy(X, y, family, w, offset, fit_intercept, lam, o['alpha'], o['pf'], o['exclude'],
                            o['lower'], o['upper'], standardize, y_scale)
    # default y_scale is R glmnet's convention
    prob = glmnet_problem(X, y, coef, a0, lam, family=family, weights=w, offset=offset, alpha=o['alpha'],
                          penalty_factor=o['pf'], exclude=o['exclude'], lower_limits=o['lower'],
                          upper_limits=o['upper'], standardize=standardize, fit_intercept=fit_intercept)
    _check_problem(prob, X, y, family, w, offset, fit_intercept, lam, alpha=o['alpha'], pf=o['pf'],
                   exclude=o['exclude'], standardize=standardize, y_scale=y_scale, kkt_tol=5e-4)
    excluded = list(o['exclude']) + ([] if o['pf'] is None else list(np.nonzero(np.isinf(o['pf']))[0]))
    offset_i = 1 if fit_intercept else 0
    for j in excluded:
        assert prob.L[j + offset_i] == prob.U[j + offset_i] == 0


def test_glmnet_problem_arguments(data):
    X, df = data
    with pytest.raises(ValueError, match='family'):
        glmnet_problem(X, df['gaussian'], np.zeros(P), 0., 0.1, family='gamma')
    with pytest.raises(ValueError, match='hessian'):
        glmnet_problem(X, df['gaussian'], np.zeros(P), 0., 0.1, hessian='sparse')
    beta = np.array([0., 1., -1., 0.5])
    G = np.array([0.5, -1., 1., -2.])
    D = np.ones(4)
    np.testing.assert_allclose(kkt_violation(beta, G, D, -np.inf, np.inf), [0, 0, 0, 1.])
    np.testing.assert_allclose(kkt_violation(beta, G, D, -np.inf, np.r_[np.inf, np.inf, np.inf, 0.5]), 0.)


# ---- end to end: inference after a GLM fit ----

@pytest.mark.parametrize('family', ['binomial', 'poisson'])
def test_glm_inference_end_to_end(data, family):
    X, df = data
    opts = OPTIONS['limits']
    G = _fit_glmstar(X, df, family, True, True, opts)
    lam = G.lambda_values_[6]
    dense = glmstar_problem(G, X, df, lambda_val=lam)
    op = glmstar_problem(G, X, df, lambda_val=lam, hessian='operator')
    np.testing.assert_allclose(op.Q_hat @ np.eye(P + 1), dense.Q_hat, atol=1e-12)

    Z = -dense.G_hat + dense.Q_hat @ dense.beta_hat
    results = []
    for prob in [dense, op]:
        LI = LassoInference(**prob.lasso_args(), Z_full=Z, Sigma=dense.Q_hat, scalar_noise=0.5)
        np.testing.assert_allclose(LI.beta_hat, prob.beta_hat, atol=1e-6)
        assert np.all(LI.A @ LI.si.Z_noisy <= LI.b + 1e-8)
        assert len(LI.summary_) == len(LI.E)
        screen = ScreenedSelection(LI, top_k=2)
        results.append((LI.summary_.values, inactive_summary(screen).values))
    np.testing.assert_allclose(results[0][0], results[1][0], rtol=1e-5)
    np.testing.assert_allclose(results[0][1], results[1][1], rtol=1e-5)
