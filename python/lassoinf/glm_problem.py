"""
The LASSO problem solved by glmnet, in the original coordinates.

glmnet (R) and glmstar (Python) solve, for family log-likelihood ``l``,

.. math::

    \\min_{\\beta_0, \\beta} -\\frac{1}{\\sum_i w_i} \\sum_i w_i \\, l(y_i, \\eta_i)
    + \\lambda \\sum_j pf_j \\left[\\frac{1-\\alpha}{2} (s_j \\beta_j)^2 + \\alpha |s_j \\beta_j|\\right]
    \\quad \\text{s.t.} \\quad L_j \\leq \\beta_j \\leq U_j

with :math:`\\eta = \\text{offset} + \\beta_0 + X\\beta`. Here :math:`s_j` is the standardization
scale of column :math:`j` (its weighted standard deviation, 1 if ``standardize=False``),
:math:`pf` the penalty factors rescaled to sum to the number of variables, and excluded
variables are fixed at 0.

For the gaussian family R's glmnet fits the standardized response :math:`y / s_y`, which
divides the ridge term by :math:`s_y` (the weighted standard deviation of :math:`y`, or its
weighted root mean square without an intercept); glmstar's ``GLMNet`` does not, its C++
path estimators (``GaussNet``) do. See ``y_scale``.

In the original coordinates :math:`(\\beta_0, \\beta)` this is a bounded LASSO with weights
:math:`D_j = \\lambda \\alpha pf_j s_j` (0 for the intercept) and smooth part
:math:`f` = negative log-likelihood plus the ridge term. :func:`glmnet_problem` returns the
solution, the gradient and Hessian of :math:`f` at it, and ``D``, ``L``, ``U``: the inputs
of :class:`lassoinf.LassoInference`. Standardization only enters through ``D`` and the
ridge term, so coefficients and limits stay on the original scale.

Supported families: gaussian, binomial and poisson (canonical links).
"""

from dataclasses import dataclass, asdict
import warnings

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator, aslinearoperator
from scipy.special import expit

from .operators.xtvx import XTVXOperator

FAMILIES = ('gaussian', 'binomial', 'poisson')


@dataclass
class GLMProblem:
    """
    Bounded LASSO problem in the coordinates (intercept, coefficients) or
    (coefficients) when there is no intercept.
    """
    beta_hat: np.ndarray
    G_hat: np.ndarray
    Q_hat: object
    D: np.ndarray
    L: np.ndarray
    U: np.ndarray
    fit_intercept: bool

    def lasso_args(self):
        """Keyword arguments for LassoInference (add Z_full, Sigma, ...)."""
        return {k: v for k, v in asdict(self).items() if k != 'fit_intercept'}

    def kkt_violation(self):
        """Largest violation of the KKT conditions at beta_hat (0 at the exact solution)."""
        return float(np.max(kkt_violation(self.beta_hat, self.G_hat, self.D, self.L, self.U), initial=0.))


def kkt_violation(beta, G, D, L, U, tol=1e-8):
    """
    Coordinatewise violation of 0 in G + D d|beta| + N_[L, U](beta), the KKT
    conditions of the bounded LASSO; coordinates with L = U are fixed and skipped.
    """
    beta, G, D, L, U = (np.asarray(v, dtype=float) for v in (beta, G, D, L, U))
    zero = np.abs(beta) <= tol
    # range of G + D * subgradient of |beta|
    lo = np.where(zero, G - D, G + D * np.sign(beta))
    hi = np.where(zero, G + D, lo)
    at_U = beta >= U - tol
    at_L = beta <= L + tol
    viol = np.where(at_U, np.maximum(lo, 0),
                    np.where(at_L, np.maximum(-hi, 0), np.maximum(np.maximum(lo, -hi), 0)))
    viol[(U - L <= tol) | (at_U & at_L)] = 0.
    return viol


def _check_converged(problem, lambda_val, tol=1e-4):
    violation = problem.kkt_violation()
    if violation > tol * lambda_val:
        warnings.warn(f'fit violates the KKT conditions by {violation / lambda_val:.1e} * lambda; '
                      'it may not have converged (tighten the convergence threshold)')


def glmnet_scaling(X, weights, standardize=True):
    """
    glmnet's column scales: weighted standard deviations (with or without
    an intercept), 1 if not standardizing.
    """
    X = np.asarray(X, dtype=float)
    p = X.shape[1]
    if not standardize:
        return np.ones(p)
    w = np.asarray(weights, dtype=float) / np.sum(weights)
    m1 = w @ X
    return np.sqrt(w @ (X * X) - m1**2)


def glmnet_response_scale(y, weights, fit_intercept=True):
    """
    Scale of the response used by R's glmnet for the gaussian family:
    weighted standard deviation with an intercept, weighted root mean
    square without one.
    """
    y = np.asarray(y, dtype=float)
    w = np.asarray(weights, dtype=float) / np.sum(weights)
    m2 = w @ y**2
    return float(np.sqrt(m2 - (w @ y)**2 if fit_intercept else m2))


def glmnet_penalty_factor(penalty_factor, p, exclude=()):
    """
    Penalty factors as used by glmnet: infinite factors mark exclusions,
    excluded variables get factor 1, then factors are rescaled to sum to p.

    Returns (pf, excluded) with excluded a boolean mask.
    """
    pf = np.ones(p) if penalty_factor is None else np.array(np.broadcast_to(penalty_factor, (p,)), dtype=float)
    excluded = np.zeros(p, bool)
    excluded[np.asarray(list(exclude), dtype=int)] = True
    excluded |= np.isinf(pf)
    pf[excluded] = 1.
    pf = np.maximum(pf, 0)
    return pf * p / pf.sum(), excluded


def _mean_and_variance(family, eta):
    if family == 'gaussian':
        return eta, np.ones_like(eta)
    if family == 'binomial':
        mu = expit(eta)
        return mu, mu * (1 - mu)
    if family == 'poisson':
        mu = np.exp(eta)
        return mu, mu
    raise ValueError(f'family must be one of {FAMILIES}, got {family!r}')


def glmnet_problem(X,
                   y,
                   coef,
                   intercept,
                   lambda_val,
                   family='gaussian',
                   weights=None,
                   offset=None,
                   alpha=1.,
                   penalty_factor=None,
                   exclude=(),
                   lower_limits=-np.inf,
                   upper_limits=np.inf,
                   standardize=True,
                   fit_intercept=True,
                   scaling=None,
                   y_scale=None,
                   hessian='dense'):
    """
    The bounded LASSO problem glmnet solved, at its solution (coef, intercept).

    Parameters
    ----------
    X, y : design (n x p) and response
    coef, intercept : fitted coefficients on the original scale
    lambda_val : the lambda of the fit
    family : 'gaussian', 'binomial' or 'poisson'
    weights, offset : observation weights and offset, as passed to glmnet
    alpha, penalty_factor, exclude, lower_limits, upper_limits,
    standardize, fit_intercept : as passed to glmnet (exclude 0-based)
    scaling : column scales s_j; default is glmnet's convention (see glmnet_scaling)
    y_scale : the ridge term is divided by y_scale; default is R glmnet's convention
        (glmnet_response_scale for gaussian, 1 otherwise). glmstar uses 1.
    hessian : 'dense' (array) or 'operator' (matrix-free X'VX, for wide designs)

    Returns
    -------
    GLMProblem, in coordinates (intercept, coef) if fit_intercept else coef.
    """
    if family not in FAMILIES:
        raise ValueError(f'family must be one of {FAMILIES}, got {family!r}')
    if hessian not in ('dense', 'operator'):
        raise ValueError("hessian must be 'dense' or 'operator'")

    X = np.asarray(X, dtype=float)
    n, p = X.shape
    y = np.asarray(y, dtype=float)
    coef = np.asarray(coef, dtype=float)
    weights = np.ones(n) if weights is None else np.asarray(weights, dtype=float)
    offset = np.zeros(n) if offset is None else np.asarray(offset, dtype=float)
    w = weights / weights.sum()

    if scaling is None:
        scaling = glmnet_scaling(X, weights, standardize=standardize)
    if y_scale is None:
        y_scale = glmnet_response_scale(y, weights, fit_intercept) if family == 'gaussian' else 1.
    pf, excluded = glmnet_penalty_factor(penalty_factor, p, exclude)

    lower = np.array(np.broadcast_to(lower_limits, (p,)), dtype=float)
    upper = np.array(np.broadcast_to(upper_limits, (p,)), dtype=float)
    # excluded variables are fixed at 0
    lower[excluded] = 0.
    upper[excluded] = 0.

    eta = offset + X @ coef + (intercept if fit_intercept else 0.)
    mu, var = _mean_and_variance(family, eta)

    ridge = lambda_val * (1 - alpha) * pf * scaling**2 / y_scale
    ridge[excluded] = 0.
    grad = X.T @ (w * (mu - y)) + ridge * coef
    D = lambda_val * alpha * pf * scaling

    if hessian == 'dense':
        Q = X.T @ (X * (w * var)[:, None]) + np.diag(ridge)
    else:
        Q = XTVXOperator(X, aslinearoperator(sp.diags(w * var))) + aslinearoperator(sp.diags(ridge))

    if fit_intercept:
        beta_hat = np.r_[intercept, coef]
        G_hat = np.r_[w @ (mu - y), grad]
        D = np.r_[0., D]
        L = np.r_[-np.inf, lower]
        U = np.r_[np.inf, upper]
        if hessian == 'dense':
            x_bar = X.T @ (w * var)
            Q = np.block([[np.sum(w * var), x_bar[None, :]],
                          [x_bar[:, None], Q]])
        else:
            X1 = np.column_stack([np.ones(n), X])
            Q = (XTVXOperator(X1, aslinearoperator(sp.diags(w * var)))
                 + aslinearoperator(sp.diags(np.r_[0., ridge])))
    else:
        beta_hat, G_hat, L, U = coef, grad, lower, upper

    return GLMProblem(beta_hat=beta_hat, G_hat=G_hat, Q_hat=Q, D=D, L=L, U=U,
                      fit_intercept=fit_intercept)


def _glmstar_family(glmnet_obj):
    import statsmodels.genmod.families as sm_family
    base = glmnet_obj._family.base
    for name, cls in [('gaussian', sm_family.Gaussian),
                      ('binomial', sm_family.Binomial),
                      ('poisson', sm_family.Poisson)]:
        if isinstance(base, cls):
            if not isinstance(base.link, type(cls().link)):
                raise ValueError(f'only the canonical link is supported for {name}')
            return name
    raise ValueError(f'unsupported family {type(base).__name__}; supported: {FAMILIES}')


def _is_cpp_path(glmnet_obj):
    from glmnet.paths.fastnet import FastNetMixin
    return isinstance(glmnet_obj, FastNetMixin)


def glmstar_problem(glmnet_obj, X, y, lambda_val=None, hessian='dense'):
    """
    GLMProblem for a fitted glmstar ``GLMNet`` at one of its lambda values.

    Also accepts glmstar's C++ path estimators ``GaussNet``, ``LogNet`` and
    ``FishNet``. These run R's glmnet code (glmnetpp), so they follow R's
    conventions: they standardize internally (``design_.scaling_`` is 1) and,
    for ``GaussNet``, divide the ridge term by the scale of y.

    X, y : the data passed to ``fit`` (y may be a DataFrame with response,
        weight and offset columns, as for glmstar).
    lambda_val : a value in ``glmnet_obj.lambda_values_``; default the last one.
    """
    G = glmnet_obj
    lambdas = np.asarray(G.lambda_values_)
    if lambda_val is None:
        k = len(lambdas) - 1
    else:
        matches = np.nonzero(np.isclose(lambdas, lambda_val, rtol=1e-10, atol=0))[0]
        if len(matches) == 0:
            raise ValueError('lambda_val must be one of the fitted lambda_values_; refit at it')
        k = matches[0]

    X_arr, _, response, offset, weight = G.get_data_arrays(X, y)
    X_arr = np.asarray(X_arr.toarray() if sp.issparse(X_arr) else X_arr, dtype=float)

    lower, upper = G.lower_limits, G.upper_limits
    if _is_cpp_path(G):
        scaling, y_scale = None, None
        # infinite limits are stored as glmnet's sentinel +-control.big
        big = G.control.big
        lower = np.where(np.asarray(lower) <= -big, -np.inf, lower)
        upper = np.where(np.asarray(upper) >= big, np.inf, upper)
    else:
        scaling, y_scale = np.asarray(G.design_.scaling_), 1.
    problem = glmnet_problem(X_arr,
                          response,
                          coef=G.coefs_[k],
                          intercept=G.intercepts_[k],
                          lambda_val=lambdas[k],
                          family=_glmstar_family(G),
                          weights=weight,
                          offset=offset,
                          alpha=G.alpha,
                          penalty_factor=G.penalty_factor,
                          exclude=G.excluded_,
                          lower_limits=lower,
                          upper_limits=upper,
                          standardize=G.standardize,
                          fit_intercept=G.fit_intercept,
                          scaling=scaling,
                          y_scale=y_scale,
                          hessian=hessian)
    _check_converged(problem, lambdas[k])
    return problem


__all__ = ['GLMProblem', 'glmnet_problem', 'glmstar_problem', 'glmnet_scaling',
           'glmnet_response_scale', 'glmnet_penalty_factor', 'kkt_violation']
