"""
Selective inference for user-specified one-dimensional estimands.

The selection event of :class:`lassoinf.LassoInference` is the polytope
:math:`A(Z + \\omega) \\leq b` where :math:`Z` is the score statistic
(``Z_full``, in the same coordinates as ``beta_hat``) and :math:`\\omega`
is randomization noise with covariance :math:`\\bar{\\Sigma}`.

With :math:`W = Q_{E,E}^{-1}` (``Q = Q_hat``), the constraints depend on
:math:`Z` only through

- :math:`\\bar{\\beta}_E = W Z_E` (in OLS, :math:`(X_E'X_E)^{-1}X_E'y`), via the active sign / bound rows;
- :math:`U_{-E} = Z_{-E} - Q_{-E,E} W Z_E` (in OLS, :math:`X_{-E}'(I-P_E)y`), via the
  subgradient rows of the inactive coordinates.

Estimands linear in these, :math:`\\hat{\\theta} = a_E'\\bar{\\beta}_E + a_{-E}'U_{-E}`,
are contrasts :math:`\\hat{\\theta} = \\eta'Z` with

.. math::

    \\eta_{-E} = a_{-E}, \\qquad \\eta_E = W(a_E - Q_{E,-E} a_{-E})

and are handled exactly as the coefficients in ``LassoInference.summary_``,
using only ``Q_hat`` submatrices and ``Sigma`` / ``A`` matvecs.
Examples: the score :math:`U_j` of an inactive variable, or its coefficient
:math:`U_j / S_{jj}` in the model :math:`E \\cup \\{j\\}`, with
:math:`S_{jj} = Q_{jj} - Q_{j,E} W Q_{E,j}`.

Estimators that are not functions of :math:`Z` can be specified by
:math:`\\text{Var}(\\hat{\\theta})` and :math:`\\text{Cov}(Z, \\hat{\\theta})`
(:class:`CovarianceEstimand`). This needs a solve with :math:`\\bar{\\Sigma}`
so it is only supported when ``Sigma_noise`` is given.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.sparse.linalg import cg, LinearOperator

from .affine_constraints import AffineConstraintsContrast
from .bivariate_normal import TruncBivariateNormal
from .operators.submatrix import extract_submatrices


@dataclass
class CustomEstimandResult:
    estimate: float
    lower_conf: float
    upper_conf: float
    p_value: float
    variance: float
    contrast: AffineConstraintsContrast


@dataclass
class CovarianceEstimand:
    """
    Estimator specified by its value and second moments with the score.

    theta_hat : observed value
    variance : Var(theta_hat)
    score_cov : Cov(Z_full, theta_hat)
    """
    theta_hat: float
    variance: float
    score_cov: np.ndarray


class SelectionCoordinates:
    """
    The coordinates (bar_beta_E, U_{-E}) in which the selection constraints
    of a LassoInference object are expressed. Builds contrasts eta with
    theta_hat = eta' Z_full from coefficients in these coordinates.
    """

    def __init__(self, lasso_inference):
        self.lasso_inference = lasso_inference
        Q = lasso_inference.Q_hat
        self.E = np.asarray(lasso_inference.E, dtype=int)
        self.E_c = np.asarray(lasso_inference.E_c, dtype=int)
        self.n = Q.shape[0]

        if len(self.E) > 0:
            Q_EE, self.Q_EcE = extract_submatrices(Q, self.E, self.E_c)
            self.W = np.linalg.inv(Q_EE)
        else:
            self.Q_EcE = np.zeros((len(self.E_c), 0))
            self.W = np.zeros((0, 0))

        self._pos_Ec = {j: k for k, j in enumerate(self.E_c)}

    def contrast(self, a_E=None, a_Ec=None):
        """
        eta such that eta' Z = a_E' bar_beta_E + a_Ec' U_{-E}.
        """
        a_E = np.zeros(len(self.E)) if a_E is None else np.asarray(a_E, dtype=float)
        a_Ec = np.zeros(len(self.E_c)) if a_Ec is None else np.asarray(a_Ec, dtype=float)
        if a_E.shape != (len(self.E),) or a_Ec.shape != (len(self.E_c),):
            raise ValueError(f'expected a_E of length {len(self.E)} and a_Ec of length {len(self.E_c)}')

        eta = np.zeros(self.n)
        eta[self.E_c] = a_Ec
        eta[self.E] = self.W @ (a_E - self.Q_EcE.T @ a_Ec)
        return eta

    def _inactive_position(self, j):
        if j not in self._pos_Ec:
            raise ValueError(f'variable {j} is not inactive')
        return self._pos_Ec[j]

    def inactive_score(self, j):
        """
        eta for U_j = Z_j - Q_{j,E} W Z_E (in OLS, x_j'(I-P_E)y).
        """
        a_Ec = np.zeros(len(self.E_c))
        a_Ec[self._inactive_position(j)] = 1.
        return self.contrast(a_Ec=a_Ec)

    def inactive_coef(self, j):
        """
        eta for U_j / S_jj, the coefficient of variable j in the model E u {j}
        (one-step / OLS coefficient with Hessian Q_hat).
        """
        k = self._inactive_position(j)
        Q_jj = extract_submatrices(self.lasso_inference.Q_hat, np.array([j]))[0, 0]
        Q_jE = self.Q_EcE[k]
        S_jj = Q_jj - Q_jE @ self.W @ Q_jE
        return self.inactive_score(j) / S_jj


def _solve(M, rhs):
    # same solve as AffineConstraints.solve_contrast
    if isinstance(M, LinearOperator):
        x, info = cg(M, rhs, rtol=1e-8)
        if info != 0:
            raise RuntimeError(f"Conjugate gradient did not converge (info={info})")
        return x
    return np.linalg.solve(M, rhs)


def covariance_contrast(si, theta_hat, variance, score_cov) -> AffineConstraintsContrast:
    """
    Analog of AffineConstraints.compute_contrast for an estimator specified
    by Var(theta_hat) and Cov(Z, theta_hat) rather than a direction eta.
    Requires an explicit noise covariance (si.Q_noise).
    """
    theta_hat = float(theta_hat)
    variance = float(variance)
    score_cov = np.asarray(score_cov, dtype=float)

    if si.Q_noise is None:
        raise ValueError('estimands specified by covariance require Sigma_noise; with scalar_noise, '
                         'express the estimand as a contrast (see SelectionCoordinates)')
    if variance <= 0:
        raise ValueError('variance must be positive')
    if score_cov.shape != np.asarray(si.Z).shape:
        raise ValueError(f'score_cov has shape {score_cov.shape}, expected {np.asarray(si.Z).shape}')
    if np.allclose(score_cov, 0):
        raise ValueError('score_cov is zero: estimator is independent of the score, '
                         'so selection does not affect its law -- use unconditional inference')

    # c = BarSigma^{-1} Cov(Z, theta_hat), bar_s^2 = c' BarSigma c
    c = _solve(si.Q_noise, score_cov)
    bar_s2 = c @ score_cov
    bar_s = np.sqrt(bar_s2)

    gamma = score_cov / variance
    bar_gamma = score_cov / bar_s2

    n_o = si.Z - gamma * theta_hat

    omega = si.Z_noisy - si.Z
    bar_theta = c @ omega
    bar_n_o = omega - bar_gamma * bar_theta

    return AffineConstraintsContrast(direction=None,
                                     theta_hat=theta_hat,
                                     gamma=gamma,
                                     c=c,
                                     bar_gamma=bar_gamma,
                                     bar_s=bar_s,
                                     n_o=n_o,
                                     bar_n_o=bar_n_o,
                                     bar_theta=bar_theta,
                                     splitting_variance=variance + bar_s2,
                                     splitting_estimator=theta_hat - bar_theta,
                                     naive_variance=variance)


def truncated_normal_inference(contrast: AffineConstraintsContrast,
                               variance: float,
                               A,
                               b: np.ndarray,
                               level: float = 0.95,
                               null_value: float = 0.):
    """
    Equal-tailed confidence interval and two-sided p-value for H0: theta = null_value
    from the truncated bivariate normal implied by the polyhedral lemma.

    Returns (lower, upper, p_value).
    """
    variance = float(variance)
    bar_s = float(contrast.bar_s)

    # interval for bar_theta at theta_hat = 0; the constraint is
    # L_0 <= (bar_s^2 / variance) * theta_hat + bar_theta <= U_0
    L_0, U_0 = contrast.get_interval(0.0, A, b)
    if np.isnan(L_0):
        raise ValueError('observed data do not satisfy the selection constraints')

    tbn = TruncBivariateNormal(a_coeff=bar_s**2 / variance,
                               b_coeff=1.0,
                               L=L_0,
                               U=U_0,
                               sig_omega=bar_s,
                               sig_x=np.sqrt(variance))

    theta_hat = float(contrast.theta_hat)

    # natural parameter: mean = natural * variance
    L_nat, U_nat = tbn.equal_tailed_interval(theta_hat, alpha=1 - level)
    lower, upper = L_nat * variance, U_nat * variance

    cdf_val = np.clip(tbn.cdf(theta=null_value / variance, x=theta_hat), 0.0, 1.0)
    p_value = np.clip(2 * min(cdf_val, 1.0 - cdf_val), 0.0, 1.0)

    return lower, upper, p_value


def _inference(lasso_inference, contrast, variance, level, null_value):
    if level is None:
        level = lasso_inference.level
    lower, upper, p_value = truncated_normal_inference(contrast,
                                                       variance,
                                                       lasso_inference.A,
                                                       lasso_inference.b,
                                                       level=level,
                                                       null_value=null_value)
    return CustomEstimandResult(estimate=float(contrast.theta_hat),
                                lower_conf=lower,
                                upper_conf=upper,
                                p_value=p_value,
                                variance=float(variance),
                                contrast=contrast)


def contrast_inference(lasso_inference,
                       eta: np.ndarray,
                       level: float | None = None,
                       null_value: float = 0.) -> CustomEstimandResult:
    """
    Selective inference for theta = eta' E[Z_full], as in LassoInference.summary_.
    """
    contrast = lasso_inference.si.compute_contrast(np.asarray(eta, dtype=float))
    # naive_variance = eta' Sigma eta
    return _inference(lasso_inference, contrast, contrast.naive_variance, level, null_value)


def custom_estimand_inference(lasso_inference,
                              theta_hat: float,
                              variance: float,
                              score_cov: np.ndarray,
                              level: float | None = None,
                              null_value: float = 0.) -> CustomEstimandResult:
    """
    Selective inference for an estimator specified by Var(theta_hat) and
    Cov(Z_full, theta_hat), with (theta_hat, Z_full) jointly Gaussian.
    Requires Sigma_noise.
    """
    contrast = covariance_contrast(lasso_inference.si, theta_hat, variance, score_cov)
    return _inference(lasso_inference, contrast, variance, level, null_value)


def estimand_summary(lasso_inference,
                     estimands: dict,
                     level: float | None = None,
                     null_value: float = 0.) -> pd.DataFrame:
    """
    Table of selective inference results, one row per estimand.

    estimands : dict
        name -> eta (np.ndarray, theta_hat = eta' Z_full) or CovarianceEstimand.
    """
    rows = []
    for name, spec in estimands.items():
        if isinstance(spec, CovarianceEstimand):
            res = custom_estimand_inference(lasso_inference,
                                            spec.theta_hat,
                                            spec.variance,
                                            spec.score_cov,
                                            level=level,
                                            null_value=null_value)
        else:
            res = contrast_inference(lasso_inference, spec, level=level, null_value=null_value)
        rows.append({'index': name,
                     'estimate': res.estimate,
                     'lower_conf': res.lower_conf,
                     'upper_conf': res.upper_conf,
                     'p_value': res.p_value})
    if not rows:
        return pd.DataFrame(columns=['estimate', 'lower_conf', 'upper_conf', 'p_value', 'index']).set_index('index')
    return pd.DataFrame(rows).set_index('index')


def inactive_summary(lasso_inference,
                     estimand: str = 'coef',
                     level: float | None = None,
                     null_value: float = 0.) -> pd.DataFrame:
    """
    Table of selective inference for each inactive variable j.

    estimand : 'coef' (coefficient of j in model E u {j}) or 'score' (U_j).
    """
    coords = SelectionCoordinates(lasso_inference)
    if estimand == 'coef':
        make_eta = coords.inactive_coef
    elif estimand == 'score':
        make_eta = coords.inactive_score
    else:
        raise ValueError(f"estimand must be 'coef' or 'score', got {estimand!r}")
    estimands = {j: make_eta(j) for j in coords.E_c}
    return estimand_summary(lasso_inference, estimands, level=level, null_value=null_value)


__all__ = ['SelectionCoordinates',
           'CovarianceEstimand',
           'CustomEstimandResult',
           'contrast_inference',
           'custom_estimand_inference',
           'estimand_summary',
           'inactive_summary',
           'covariance_contrast',
           'truncated_normal_inference']
