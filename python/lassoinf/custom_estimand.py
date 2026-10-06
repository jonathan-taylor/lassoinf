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

On the selection event the LASSO solution :math:`\\hat{\\beta}_E` and the
negative gradient :math:`-\\nabla \\ell(\\hat{\\beta})_{-E}` are the same linear
maps of the data plus constants (the effect of the penalty and bounds),
so estimands may instead be expressed in these coordinates (``basis='lasso'``):
same contrast :math:`\\eta`, same variances and covariances, shifted observed values.

:class:`ScreenedSelection` adds a second selection step choosing inactive
variables by the size of the LASSO gradient (threshold or top K); conditioning
on the chosen variables and the signs of their gradients (for top K, by default
also on the largest dropped variable and its sign) keeps the event a polytope
with O(p) rows.

Estimators that are not functions of :math:`Z` can be specified by
:math:`\\text{Var}(\\hat{\\theta})` and :math:`\\text{Cov}(Z, \\hat{\\theta})`
(:class:`CovarianceEstimand`). This needs a solve with :math:`\\bar{\\Sigma}`
so it is only supported when ``Sigma_noise`` is given.
"""

from dataclasses import dataclass, replace
import warnings

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import cg, LinearOperator, aslinearoperator

from .affine_constraints import AffineConstraintsContrast
from .bivariate_normal import TruncBivariateNormal
from .operators.lasso_constraints import InactiveScoreOperator


@dataclass
class CustomEstimandResult:
    estimate: float
    lower_conf: float
    upper_conf: float
    p_value: float
    variance: float
    contrast: AffineConstraintsContrast


@dataclass
class ContrastEstimand:
    """
    Estimator theta_hat = eta' Z_full + offset, estimand theta = eta' E[Z_full] + offset.
    """
    eta: np.ndarray
    offset: float = 0.

    def value(self, Z):
        return self.eta @ Z + self.offset


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
    The coordinates in which the selection constraints of a LassoInference
    object are expressed:

    - basis='refit': (bar_beta_E, U_{-E}) = (W Z_E, Z_{-E} - Q_{-E,E} W Z_E)
    - basis='lasso': (beta_hat_E, -grad_{-E}), the LASSO solution and the negative
      gradient of the loss at it, as maps of the data on the selection event.

    The two differ by constants, so they give the same contrast eta.
    Everything is applied with Q_hat matvecs; nothing of size p x |E| is stored.

    Q_diag : optional diagonal of Q_hat, used to compute S_jj for many
        inactive variables with |E| matvecs instead of one per variable.
        Taken from Q_hat when it is an array or has a ``diagonal`` method.
    """

    def __init__(self, lasso_inference, Q_diag=None):
        self.lasso_inference = lasso_inference
        self.Q = lasso_inference.Q_hat
        self.W = lasso_inference.W
        self.E = np.asarray(lasso_inference.E, dtype=int)
        self.E_c = np.asarray(lasso_inference.E_c, dtype=int)
        self.n = self.Q.shape[0]
        self._score = InactiveScoreOperator(self.Q, self.E, self.E_c, self.W)
        self._Q_diag = Q_diag
        self._S = {}

        self._pos_Ec = {j: k for k, j in enumerate(self.E_c)}

        # the LASSO solution and gradient are affine in the data on the
        # selection event: constants are observed values minus the linear
        # maps evaluated at the data used for selection
        Z_noisy = lasso_inference.si.Z_noisy
        beta_hat = np.asarray(lasso_inference.beta_hat)
        G_hat = np.asarray(lasso_inference.G_hat)
        self.lasso_coef_offset = beta_hat[self.E] - self.refit_coef(Z_noisy)
        self.lasso_score_offset = -G_hat[self.E_c] - self.refit_score(Z_noisy)

    def refit_coef(self, Z):
        """bar_beta_E = W Z_E."""
        return self._score.coef(Z)

    def refit_score(self, Z):
        """U_{-E} = Z_{-E} - Q_{-E,E} W Z_E."""
        return self._score.matvec(Z)

    def score_operator(self):
        """U_{-E} as a matrix-free LinearOperator acting on Z."""
        return self._score

    def contrast(self, a_E=None, a_Ec=None):
        """
        eta such that eta' Z = a_E' bar_beta_E + a_Ec' U_{-E}.
        """
        a_E = np.zeros(len(self.E)) if a_E is None else np.asarray(a_E, dtype=float)
        a_Ec = np.zeros(len(self.E_c)) if a_Ec is None else np.asarray(a_Ec, dtype=float)
        if a_E.shape != (len(self.E),) or a_Ec.shape != (len(self.E_c),):
            raise ValueError(f'expected a_E of length {len(self.E)} and a_Ec of length {len(self.E_c)}')

        # eta = U_{-E}' a_Ec + [W a_E on E]
        eta = self._score.rmatvec(a_Ec) if np.any(a_Ec != 0) else np.zeros(self.n)
        eta[self.E] += self.W.T @ a_E
        return eta

    def estimand(self, a_E=None, a_Ec=None, basis='refit') -> ContrastEstimand:
        """
        theta_hat = a_E' bar_beta_E + a_Ec' U_{-E}               (basis='refit')
        theta_hat = a_E' beta_hat_E + a_Ec' (-grad(beta_hat)_{-E})  (basis='lasso')
        """
        eta = self.contrast(a_E, a_Ec)
        if basis == 'refit':
            return ContrastEstimand(eta)
        if basis == 'lasso':
            offset = 0.
            if a_E is not None:
                offset += np.asarray(a_E, dtype=float) @ self.lasso_coef_offset
            if a_Ec is not None:
                offset += np.asarray(a_Ec, dtype=float) @ self.lasso_score_offset
            return ContrastEstimand(eta, float(offset))
        raise ValueError(f"basis must be 'refit' or 'lasso', got {basis!r}")

    def _inactive_position(self, j):
        if j not in self._pos_Ec:
            raise ValueError(f'variable {j} is not inactive')
        return self._pos_Ec[j]

    def _Q_diagonal(self):
        if self._Q_diag is not None:
            return np.asarray(self._Q_diag, dtype=float)
        if isinstance(self.Q, np.ndarray):
            return np.diag(self.Q)
        if hasattr(self.Q, 'diagonal'):
            return np.asarray(self.Q.diagonal(), dtype=float)
        return None

    def inactive_S(self, variables):
        """
        S_jj = Q_jj - Q_{j,E} W Q_{E,j} for inactive variables j, cached.

        One Q matvec per variable, or |E| matvecs in total when diag(Q_hat)
        is available and there are more than |E| variables.
        """
        variables = np.atleast_1d(np.asarray(variables, dtype=int))
        missing = [j for j in dict.fromkeys(variables.tolist()) if j not in self._S]
        for j in missing:
            self._inactive_position(j)
        diag = self._Q_diagonal() if len(missing) > len(self.E) else None
        if missing and diag is not None:
            # Q_{j,E} W Q_{E,j} = ||C' Q_{E,j}||^2 with W = C C'
            S = diag[missing].copy()
            if len(self.E) > 0:
                C = np.linalg.cholesky(self.W)
                for i in range(len(self.E)):
                    col = np.zeros(self.n)
                    col[self.E] = C[:, i]
                    S -= np.ravel(self.Q @ col)[missing]**2
            self._S.update(zip(missing, S))
        else:
            for j in missing:
                e_j = np.zeros(self.n)
                e_j[j] = 1.
                Q_j = np.ravel(self.Q @ e_j)
                Q_Ej = Q_j[self.E]
                self._S[j] = Q_j[j] - Q_Ej @ self.W @ Q_Ej
        return np.array([self._S[j] for j in variables.tolist()])

    def inactive_score(self, j, basis='refit') -> ContrastEstimand:
        """
        U_j = Z_j - Q_{j,E} W Z_E (in OLS, x_j'(I-P_E)y) for basis='refit';
        -grad(beta_hat)_j for basis='lasso'.
        """
        a_Ec = np.zeros(len(self.E_c))
        a_Ec[self._inactive_position(j)] = 1.
        return self.estimand(a_Ec=a_Ec, basis=basis)

    def inactive_coef(self, j, basis='refit') -> ContrastEstimand:
        """
        Coefficient of variable j in the model E u {j}: the inactive score
        divided by S_jj = Q_jj - Q_{j,E} W Q_{E,j} (OLS / one-step coefficient).
        """
        S_jj = self.inactive_S([j])[0]
        score = self.inactive_score(j, basis=basis)
        return ContrastEstimand(score.eta / S_jj, score.offset / S_jj)


class _VStackOperator(LinearOperator):

    def __init__(self, ops):
        self.ops = [aslinearoperator(op) for op in ops]
        self._splits = np.cumsum([op.shape[0] for op in self.ops])[:-1]
        shape = (sum(op.shape[0] for op in self.ops), self.ops[0].shape[1])
        super().__init__(np.float64, shape)

    def _matvec(self, x):
        return np.concatenate([np.ravel(op.matvec(x)) for op in self.ops])

    def _rmatvec(self, y):
        y = np.ravel(y)
        return sum(np.ravel(op.rmatvec(y_i)) for op, y_i in zip(self.ops, np.split(y, self._splits)))

    def to_dense(self):
        # one matvec per column: for testing / small problems only
        return self.matmat(np.eye(self.shape[1]))


def _sparse_rows(entries, n_rows, n_cols):
    # entries: list of (row, col, val) arrays
    if not entries:
        return sp.csr_matrix((n_rows, n_cols))
    rows, cols, vals = (np.concatenate(x) for x in zip(*entries))
    return sp.csr_matrix((vals, (rows, cols)), shape=(n_rows, n_cols))


class ScreenedSelection:
    """
    LASSO selection followed by screening of inactive variables on the
    gradient of the loss at the LASSO solution, G = grad(beta_hat)_{-E}:

    - threshold: keep j with |G_j| > threshold;
    - top_k: keep the top_k inactive variables by |G_j|.

    We condition on the kept variables and the signs of their gradients, which
    adds affine constraints in Z + omega (G_{-E} = -U_{-E}(Z + omega) + const
    on the LASSO selection event):

    - threshold: s_j G_j > threshold for kept j, |G_l| <= threshold otherwise
      (|E_c| + #dropped rows);
    - top_k, conditioning='first_dropped' (default): also condition on the
      identity l* and sign s* of the largest dropped gradient:
      s_k G_k >= s* G_l* for kept k, |G_l| <= s* G_l* for the other dropped l
      (|E_c| + #dropped - 1 rows);
    - top_k, conditioning='exact': s_k G_k >= |G_l| for every kept k and
      dropped l (2 top_k #dropped rows).

    Constraints are matrix-free: one Q_hat matvec plus a sparse product.

    Can be passed anywhere a LassoInference is accepted in this module;
    other attributes are taken from the LassoInference object.

    Attributes
    ----------
    screened : np.ndarray
        Kept inactive variables, by decreasing |G_j|.
    screened_signs : np.ndarray
        Signs of their gradients.
    first_dropped, first_dropped_sign :
        l* and s* for top_k with conditioning='first_dropped', else None.
    A, b : constraints of the combined selection event.
    """

    def __init__(self, lasso_inference, threshold=None, top_k=None, conditioning='first_dropped'):
        if (threshold is None) == (top_k is None):
            raise ValueError('specify exactly one of threshold or top_k')
        if conditioning not in ('first_dropped', 'exact'):
            raise ValueError(f"conditioning must be 'first_dropped' or 'exact', got {conditioning!r}")
        if top_k is not None and top_k < 1:
            raise ValueError('top_k must be at least 1')

        self.lasso_inference = lasso_inference
        self.threshold = threshold
        self.top_k = top_k
        self.conditioning = conditioning if top_k is not None else None
        self.first_dropped = None
        self.first_dropped_sign = None

        coords = SelectionCoordinates(lasso_inference)
        m = len(coords.E_c)
        G = np.asarray(lasso_inference.G_hat)[coords.E_c]
        abs_G = np.abs(G)
        signs = np.sign(G)
        order = np.argsort(-abs_G, kind='stable')

        # constraints P G <= q, P built from (row, col, val) arrays
        entries, q = [], []

        def add_rows(cols_vals, rhs):
            # cols_vals: list of (col array, val array), one entry per row each
            r0 = sum(len(x) for x in q)
            rows = r0 + np.arange(len(rhs))
            for c, v in cols_vals:
                entries.append((rows, np.asarray(c), np.broadcast_to(np.asarray(v, dtype=float), rows.shape)))
            q.append(np.asarray(rhs, dtype=float))

        if threshold is not None:
            kept_mask = abs_G > threshold
            keep = order[kept_mask[order]]
            kept, dropped = np.nonzero(kept_mask)[0], np.nonzero(~kept_mask)[0]
            add_rows([(kept, -signs[kept])], np.full(len(kept), -threshold))
            add_rows([(dropped, 1.)], np.full(len(dropped), threshold))
            add_rows([(dropped, -1.)], np.full(len(dropped), threshold))
        else:
            keep, drop = order[:top_k], order[top_k:]
            if len(drop) > 0 and conditioning == 'exact':
                kk = np.repeat(keep, len(drop))
                ll = np.tile(drop, len(keep))
                zeros = np.zeros(len(kk))
                add_rows([(ll, 1.), (kk, -signs[kk])], zeros)
                add_rows([(ll, -1.), (kk, -signs[kk])], zeros)
            elif len(drop) > 0:
                l0, rest = drop[0], drop[1:]
                s0 = signs[l0]
                self.first_dropped, self.first_dropped_sign = coords.E_c[l0], s0
                # s0 G_l0 <= s_k G_k for kept k
                add_rows([(np.full(len(keep), l0), s0), (keep, -signs[keep])], np.zeros(len(keep)))
                # |G_l| <= s0 G_l0 for other dropped l
                add_rows([(rest, 1.), (np.full(len(rest), l0), -s0)], np.zeros(len(rest)))
                add_rows([(rest, -1.), (np.full(len(rest), l0), -s0)], np.zeros(len(rest)))
                if len(rest) == 0:
                    add_rows([(np.array([l0]), -s0)], np.zeros(1))

        self.screened = coords.E_c[keep]
        self.screened_signs = signs[keep]

        q = np.concatenate(q) if q else np.zeros(0)
        P = _sparse_rows(entries, len(q), m)

        # G = -(U(Z) + lasso_score_offset), so P G <= q  <=>  -P U(Z) <= q + P lasso_score_offset
        A_screen = aslinearoperator(-P) @ coords.score_operator()
        b_screen = q + P @ coords.lasso_score_offset

        if np.any(A_screen @ lasso_inference.si.Z_noisy > b_screen + 1e-8 * (1 + np.abs(b_screen))):
            warnings.warn('observed data do not satisfy the screening constraints')

        self.A_screen, self.b_screen = A_screen, b_screen
        self.A = _VStackOperator([lasso_inference.A, A_screen])
        self.b = np.concatenate([lasso_inference.b, b_screen])

    def __getattr__(self, name):
        # only called for attributes not set in __init__
        if name == 'lasso_inference':
            raise AttributeError(name)
        return getattr(self.lasso_inference, name)


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
                       estimand,
                       level: float | None = None,
                       null_value: float = 0.) -> CustomEstimandResult:
    """
    Selective inference for theta = eta' E[Z_full] + offset, as in LassoInference.summary_.

    estimand : ContrastEstimand or np.ndarray (eta, with offset 0)
    """
    if not isinstance(estimand, ContrastEstimand):
        estimand = ContrastEstimand(np.asarray(estimand, dtype=float))
    contrast = lasso_inference.si.compute_contrast(estimand.eta)
    # naive_variance = eta' Sigma eta
    res = _inference(lasso_inference, contrast, contrast.naive_variance, level,
                     null_value - estimand.offset)
    return replace(res,
                   estimate=res.estimate + estimand.offset,
                   lower_conf=res.lower_conf + estimand.offset,
                   upper_conf=res.upper_conf + estimand.offset)


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
        name -> ContrastEstimand, eta (np.ndarray, theta_hat = eta' Z_full)
        or CovarianceEstimand.
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
                     basis: str = 'refit',
                     variables=None,
                     level: float | None = None,
                     null_value: float = 0.,
                     Q_diag=None) -> pd.DataFrame:
    """
    Table of selective inference for inactive variables.

    estimand : 'coef' (coefficient of j in model E u {j}) or 'score'.
    basis : 'refit' (at bar_beta_E) or 'lasso' (at the LASSO solution).
    variables : inactive variables to report; defaults to ``screened`` for a
        ScreenedSelection and all inactive variables otherwise.
    Q_diag : optional diagonal of Q_hat (see SelectionCoordinates).
    """
    coords = SelectionCoordinates(lasso_inference, Q_diag=Q_diag)
    if estimand == 'coef':
        make = coords.inactive_coef
    elif estimand == 'score':
        make = coords.inactive_score
    else:
        raise ValueError(f"estimand must be 'coef' or 'score', got {estimand!r}")
    if variables is None:
        variables = getattr(lasso_inference, 'screened', coords.E_c)
    if estimand == 'coef' and len(variables) > 0:
        coords.inactive_S(variables)
    estimands = {j: make(j, basis=basis) for j in variables}
    return estimand_summary(lasso_inference, estimands, level=level, null_value=null_value)


__all__ = ['SelectionCoordinates',
           'ScreenedSelection',
           'ContrastEstimand',
           'CovarianceEstimand',
           'CustomEstimandResult',
           'contrast_inference',
           'custom_estimand_inference',
           'estimand_summary',
           'inactive_summary',
           'covariance_contrast',
           'truncated_normal_inference']
