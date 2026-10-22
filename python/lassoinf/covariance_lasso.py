"""
Selective inference after the lasso in covariance form, from an estimate and its covariance.

Given an asymptotically Gaussian estimate beta_hat ~ N(beta, Sigma), such as the MLE of a GLM
with its estimated covariance, covariance_lasso_inference selects coefficients with the lasso

    minimize 1/2 (beta_star - b)' Sigma^{-1} (beta_star - b) + sum_j D_j |b_j|,

where beta_star = beta_hat + omega and omega ~ N(0, (1 - pi) / pi Sigma) mimics selecting on a
random fraction pi (prop_select) of the data: beta_star has the variance Sigma / pi of an
estimate computed on that fraction. It then returns p-values, confidence intervals and
selective MLEs for the selected coefficients, conditional on the selection.

In the score coordinates of LassoInference this is Q_hat = Sigma^{-1}, Z_full = Sigma^{-1}
beta_hat, Cov(Z_full) = Sigma^{-1} and scalar_noise = (1 - pi) / pi.

The penalty standardizes the score. At b = 0 the lasso compares the score
S = Sigma^{-1} beta_star with D; with unpenalized coefficients N profiled out (and excluded
ones removed) the score of the penalized coefficients P has covariance M / pi, with M the
Schur complement of N in Sigma^{-1}. With

    D_j = penalty_factor_j * t * sqrt(M_jj / pi),

nothing is selected exactly when max_j |xi_j| / penalty_factor_j <= t for the standardized
score xi_j = S_j / sqrt(M_jj / pi). This is the covariance-form analog of glmnet's
standardized X'y: in least squares xi_j is the marginal z-statistic of feature j. By
default t is lambda_mult times E ||xi||_inf under the global null, xi ~ N(0, corr(M)), a
Bonferroni-like threshold. For uncorrelated estimates M_jj = 1 / Var(beta_hat_j), and j is
selected when |beta_star_j| / SE(beta_star_j) > penalty_factor_j * t.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm

import lassoinf_cpp as _cpp

from .lasso import LassoInference


def box_lasso_fit(Q, Y, D, L, U, beta0=None, tol=1e-12, max_iter=100000):
    """
    The solution of minimize 1/2 b' Q b - b' Y + sum_j D_j |b_j| subject to L <= b <= U,
    by coordinate descent. L_j = U_j = 0 excludes coordinate j; D_j = 0 leaves it unpenalized.
    """
    beta0 = np.zeros(0) if beta0 is None else np.asarray(beta0, dtype=float)
    args = [np.asarray(a, dtype=float) for a in (Q, Y, D, L, U)]
    try:
        return _cpp.box_lasso_fit(*args, beta0, tol, max_iter)
    except ValueError as e:
        raise ValueError(str(e)) from None


def box_lasso_kkt_violation(beta_hat, Q, Y, D, L, U):
    """The largest violation of the KKT conditions of box_lasso_fit by beta_hat."""
    return _cpp.box_lasso_kkt_violation(*[np.asarray(a, dtype=float)
                                          for a in (beta_hat, Q, Y, D, L, U)])


def expected_max_abs(cov, n_mc=4000, rng=None):
    """
    Monte Carlo estimate of E ||xi||_inf for xi ~ N(0, R), R the correlation matrix of cov.
    """
    cov = np.atleast_2d(np.asarray(cov, dtype=float))
    if cov.shape[0] == 0:
        return 0.
    s = np.sqrt(np.diag(cov))
    R = cov / np.outer(s, s)
    rng = np.random.default_rng(rng)
    chol = np.linalg.cholesky(R)
    xi = rng.standard_normal((n_mc, R.shape[0])) @ chol.T
    return np.abs(xi).max(1).mean()


def selective_mle(family, observed, variance, width=40.):
    """
    The selective MLE of the target for the TruncBivariateNormal ``family`` of
    theta_hat (see LassoInference.truncated_family), and its standard error from the
    conditional Fisher information. Both are on the scale of the target.

    The MLE solves E_theta[theta_hat | selection] = observed; the conditional mean is
    increasing in the natural parameter theta.
    """
    sd = np.sqrt(variance)

    def gap(theta):
        mean = family.E(theta, lambda x: x)
        return np.nan if mean is None else float(mean) - observed

    # a grid around the unadjusted estimate, then a root in the first bracket found
    grid = observed / variance + np.linspace(-width, width, 161) / sd
    values = np.array([gap(t) for t in grid])
    ok = np.isfinite(values)
    grid, values = grid[ok], values[ok]
    change = np.nonzero(np.sign(values[:-1]) != np.sign(values[1:]))[0]
    if len(change) == 0:
        return np.nan, np.nan
    k = change[0]
    theta = brentq(gap, grid[k], grid[k + 1], xtol=1e-10 / sd)
    info = family.Var(theta, lambda x: x)
    return theta * variance, variance / np.sqrt(info)


@dataclass
class CovarianceLassoResult:
    """
    The output of covariance_lasso_inference.

    summary_ has a row for each selected coefficient:

    - estimate, std_err: the unadjusted estimate of the target and its standard error;
    - mle, mle_std_err: the selective MLE and its standard error from the conditional
      Fisher information;
    - lower_conf, upper_conf: the selective confidence interval at ``level``;
    - p_value: the selective p-value for the target being 0.

    splitting_ has the estimate that is independent of the selection, with variance
    1 / (1 - pi) times that of the full estimate, and its interval: the analog of
    computing the estimate on the held-out fraction 1 - pi of a split.
    """
    summary_: pd.DataFrame
    splitting_: pd.DataFrame
    selected: pd.Index
    lasso_coef: pd.Series
    beta_star: pd.Series
    threshold: float
    D: np.ndarray
    inference: LassoInference
    target: str


def _penalized_score_cov(Q, P, N):
    """
    The covariance (times pi) of the score of the penalized coefficients P at b_P = 0, with
    the unpenalized N profiled out: the Schur complement of N in Q, restricted to P.
    """
    M = Q[np.ix_(P, P)]
    if len(N):
        M = M - Q[np.ix_(P, N)] @ np.linalg.solve(Q[np.ix_(N, N)], Q[np.ix_(N, P)])
    return (M + M.T) / 2


def _positions(spec, names, p):
    """Integer positions for ``spec``: names, integer positions or a boolean mask."""
    if spec is None:
        return np.zeros(0, dtype=int)
    spec = np.asarray(spec)
    if spec.dtype == bool:
        if spec.shape != (p,):
            raise ValueError('a boolean mask must have one entry per coefficient')
        return np.nonzero(spec)[0]
    if spec.dtype.kind in 'iu':
        return spec.astype(int)
    return names.get_indexer(spec)


def covariance_lasso_inference(beta_hat,
                               cov_beta,
                               prop_select=0.8,
                               exclude=None,
                               penalty_factor=None,
                               lambda_mult=1.,
                               threshold=None,
                               perturbation=None,
                               target='selected',
                               level=0.95,
                               n_mc=4000,
                               rng=None):
    """
    Select coefficients with a randomized lasso in covariance form and return selective
    inference for them.

    Parameters
    ----------
    beta_hat : array or pd.Series, shape (p,)
        An asymptotically Gaussian estimate, e.g. a GLM's MLE. With a Series, its index names
        the coefficients. Leave the intercept out: dropping it from beta_hat and cov_beta
        profiles it out.
    cov_beta : array or pd.DataFrame, shape (p, p)
        The covariance of beta_hat (model-based, sandwich or bootstrap).
    prop_select : float in (0, 1)
        The fraction pi of the data whose selection is mimicked: the randomization omega has
        covariance (1 - pi) / pi cov_beta.
    exclude : names, positions or boolean mask, optional
        Coefficients kept out of the model (fixed at 0), as glmnet's ``exclude``.
    penalty_factor : array, shape (p,), optional
        Multiplies the penalty of each coefficient, as glmnet's ``penalty.factor``; 0 leaves a
        coefficient unpenalized. Default 1.
    lambda_mult : float
        The threshold t is lambda_mult * E ||xi||_inf, with xi the standardized score of the
        penalized coefficients under the global null (see the module docstring; estimated
        with n_mc draws). Nothing is selected when max_j |xi_j| / penalty_factor_j <= t.
    threshold : float, optional
        The threshold t itself, overriding lambda_mult.
    perturbation : array, shape (p,), optional
        A draw with covariance cov_beta, independent of beta_hat given the data, for example a
        bootstrap replicate of beta_hat minus the bootstrap mean. It is multiplied by
        sqrt((1 - pi) / pi) to form omega. By default omega is drawn from a Gaussian.
    target : 'selected' or 'full'
        'selected': for each selected j, the coefficient of j in the model of the selected
        coefficients: the projection of beta onto them in the Sigma^{-1} metric, estimated
        by refitting the selected model in covariance form. 'full': beta_j itself, estimated
        by beta_hat_j.
    level : float
        The confidence level.
    n_mc : int
        Monte Carlo draws for E ||xi||_inf.
    rng : seed or np.random.Generator, optional

    Returns
    -------
    CovarianceLassoResult
    """
    if target not in ('selected', 'full'):
        raise ValueError("target must be 'selected' or 'full'")
    pi = float(prop_select)
    if not 0 < pi < 1:
        raise ValueError('prop_select must be in (0, 1)')
    rng = np.random.default_rng(rng)

    names = beta_hat.index if isinstance(beta_hat, pd.Series) else None
    beta_hat = np.asarray(beta_hat, dtype=float)
    p = beta_hat.shape[0]
    names = pd.Index(range(p)) if names is None else pd.Index(names)
    Sigma = np.asarray(cov_beta, dtype=float)
    if Sigma.shape != (p, p):
        raise ValueError('cov_beta must be p x p')
    Sigma = (Sigma + Sigma.T) / 2

    excluded = np.zeros(p, dtype=bool)
    pos = _positions(exclude, names, p)
    if np.any(pos < 0):
        raise ValueError('exclude has names not in beta_hat')
    excluded[pos] = True
    pf = np.ones(p) if penalty_factor is None else np.asarray(penalty_factor, dtype=float)
    if pf.shape != (p,) or np.any(pf < 0):
        raise ValueError('penalty_factor must have p entries >= 0')

    Q = np.linalg.inv(Sigma)
    Q = (Q + Q.T) / 2
    penalized = (pf > 0) & ~excluded
    P = np.nonzero(penalized)[0]
    M = _penalized_score_cov(Q, P, np.nonzero((pf == 0) & ~excluded)[0])
    if threshold is None:
        threshold = lambda_mult * expected_max_abs(M, n_mc=n_mc, rng=rng)

    # the randomization, on the scale of beta_hat
    if perturbation is None:
        perturbation = np.linalg.cholesky(Sigma) @ rng.standard_normal(p)
    perturbation = np.asarray(perturbation, dtype=float)
    if perturbation.shape != (p,):
        raise ValueError('perturbation must have p entries')
    beta_star = beta_hat + np.sqrt((1 - pi) / pi) * perturbation

    # covariance form in score coordinates
    Z_full = Q @ beta_hat
    Y = Q @ beta_star
    # the standardized score of the penalized coefficients is S_j / sqrt(M_jj / pi)
    D = np.zeros(p)
    D[P] = pf[P] * threshold * np.sqrt(np.diag(M) / pi)
    L = np.where(excluded, 0., -np.inf)
    U = np.where(excluded, 0., np.inf)

    b = box_lasso_fit(Q, Y, D, L, U)
    G = Q @ b - Y

    inference = LassoInference(beta_hat=b,
                               G_hat=G,
                               Q_hat=Q,
                               D=D,
                               L=L,
                               U=U,
                               Z_full=Z_full,
                               Sigma=Q,
                               scalar_noise=(1 - pi) / pi,
                               level=level)

    E = np.asarray(inference.E, dtype=int)
    rows, split_rows = [], []
    q = norm.ppf(1 - (1 - level) / 2)
    for k, j in enumerate(E):
        if target == 'selected':
            v = np.zeros(p)
            v[E] = inference.W[:, k]
            contrast = inference._contrasts[j]
            lower, upper, p_value = inference.summary_.loc[j, ['lower_conf', 'upper_conf',
                                                               'p_value']]
        else:
            v = Sigma[:, j]
            contrast = None
        theta_hat = float(v @ Z_full)
        variance = float(v @ Q @ v)
        family, contrast = inference.truncated_family(v, variance, contrast)
        if target == 'full':
            L_nat, U_nat = family.equal_tailed_interval(theta_hat, alpha=1 - level)
            lower, upper = L_nat * variance, U_nat * variance
            cdf = np.clip(family.cdf(theta=0., x=theta_hat), 0., 1.)
            p_value = np.clip(2 * min(cdf, 1 - cdf), 0., 1.)
        mle, mle_se = selective_mle(family, theta_hat, variance)
        rows.append({'estimate': theta_hat,
                     'std_err': np.sqrt(variance),
                     'mle': mle,
                     'mle_std_err': mle_se,
                     'lower_conf': lower,
                     'upper_conf': upper,
                     'p_value': p_value})
        split_est = float(contrast.splitting_estimator)
        split_se = np.sqrt(float(contrast.splitting_variance))
        split_rows.append({'estimate': split_est,
                           'std_err': split_se,
                           'lower_conf': split_est - q * split_se,
                           'upper_conf': split_est + q * split_se,
                           'p_value': 2 * norm.sf(abs(split_est) / split_se)})

    selected = names[E]
    columns = ['estimate', 'std_err', 'mle', 'mle_std_err', 'lower_conf', 'upper_conf',
               'p_value']
    summary = pd.DataFrame(rows, index=selected, columns=columns)
    splitting = pd.DataFrame(split_rows, index=selected,
                             columns=['estimate', 'std_err', 'lower_conf', 'upper_conf',
                                      'p_value'])
    return CovarianceLassoResult(summary_=summary,
                                 splitting_=splitting,
                                 selected=selected,
                                 lasso_coef=pd.Series(inference.beta_hat, index=names),
                                 beta_star=pd.Series(beta_star, index=names),
                                 threshold=float(threshold),
                                 D=D,
                                 inference=inference,
                                 target=target)
