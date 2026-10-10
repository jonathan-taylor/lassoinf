"""
Selective inference for a selection event evaluated along the conditioning ray.

The polyhedral lemma conditions on everything but one direction: for a contrast eta, the
selection data move along the ray

    Y(w) = N_o + bar_N_o + bar_Gamma w,

and the selection event, as a function of (theta_hat, bar_omega), depends only on the
scalar w = (bar_s^2 / sigma^2) theta_hat + bar_omega (see docs/main.md). If the selection
event has an explicit characterization event(Y) -> bool, for example the KKT conditions of
a lasso whose penalties are functions of Y, then the truncation set

    S = {w : event(Y(w))}

can be found numerically: S is a finite union of intervals, located by evaluating the
event on a grid of w and bisecting each change. No refitting is needed, unlike estimating
the selection probability from refits to resampled data (Liu, Markovic-Voronov and Taylor,
arXiv:2203.14504).

Given S, the law of theta_hat is that of a Gaussian, convolved with the randomization
bar_omega and truncated to w in S. This is TruncBivariateNormal with a union of intervals,
and with no randomization (scalar_noise = 0) it is the polyhedral case.

Anything the event depends on, such as penalty factors computed from the data, must be a
function of the selection data Y (plus independent noise). Otherwise the event depends on
(theta_hat, bar_omega) through more than w. The uniLasso has a closed form for its
truncation set (lassoinf.unilasso), which this module does not replace.
"""
from dataclasses import dataclass

import numpy as np

from .bivariate_normal import TruncBivariateNormal
from .lasso import lasso_post_selection_constraints


def ray_scale(contrast):
    """
    The observed w and the standard deviation of w along the ray of an
    AffineConstraintsContrast.
    """
    bar_s2 = float(contrast.bar_s)**2
    variance = float(contrast.naive_variance)
    w_obs = bar_s2 / variance * float(contrast.theta_hat) + float(contrast.bar_theta)
    sd_w = np.sqrt(bar_s2**2 / variance + bar_s2)
    return w_obs, sd_w


def ray_point(contrast, w):
    """The selection data Y(w) = N_o + bar_N_o + bar_Gamma w along the ray."""
    return (np.atleast_1d(contrast.n_o) + np.atleast_1d(contrast.bar_n_o)
            + np.atleast_1d(contrast.bar_gamma) * w)


def find_truncation_set(event, contrast, n_sd=10., n_grid=2001, tol=1e-10):
    """
    The truncation set {w : event(Y(w))} along the ray of a contrast, as an array (k, 2) of
    disjoint intervals in increasing order.

    event : function of the selection data Y, True on the selection event
    n_sd, n_grid : the event is evaluated on n_grid equally spaced points within n_sd standard
        deviations of the observed w, and assumed constant beyond them (where the Gaussian
        has negligible mass)
    tol : each change of the event between grid points is located by bisection, to tol
        times the standard deviation of w

    The event can change more than once between two grid points; such short pieces are
    missed. A missed piece of length e changes the truncated probabilities by at most
    e / (bar_s sqrt(2 pi)).
    """
    w_obs, sd_w = ray_scale(contrast)
    member = lambda w: bool(event(ray_point(contrast, w)))
    if not member(w_obs):
        raise ValueError('the observed data are not in the selection event')
    grid = w_obs + np.linspace(-n_sd, n_sd, n_grid) * sd_w
    inside = np.array([member(w) for w in grid])

    def boundary(lo, hi, lo_inside):
        # the change between lo and hi, where member(lo) == lo_inside != member(hi)
        while hi - lo > tol * sd_w:
            mid = 0.5 * (lo + hi)
            if member(mid) == lo_inside:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    intervals = []
    start = -np.inf if inside[0] else None
    for i in range(1, n_grid):
        if inside[i] != inside[i - 1]:
            b = boundary(grid[i - 1], grid[i], inside[i - 1])
            if inside[i]:
                start = b
            else:
                intervals.append((start, b))
                start = None
    if start is not None:
        intervals.append((start, np.inf))
    return np.array(intervals, dtype=float).reshape(-1, 2)


@dataclass
class RayContrastResult:
    theta_hat: float
    lower_conf: float
    upper_conf: float
    p_value: float
    intervals: np.ndarray
    family: TruncBivariateNormal
    variance: float

    def pivot(self, theta):
        """P(theta_hat' <= theta_hat | selection) when the target equals theta."""
        return self.family.cdf(theta / self.variance, self.theta_hat)


def ray_contrast_inference(affine_constraints, direction, event, level=0.95, **find_args):
    """
    Selective inference for direction' E[Z] after the selection event event(Y), with the
    randomization of affine_constraints (an AffineConstraints).

    find_args are passed to find_truncation_set.
    """
    contrast = affine_constraints.compute_contrast(np.asarray(direction, dtype=float))
    intervals = find_truncation_set(event, contrast, **find_args)
    variance = float(contrast.naive_variance)
    bar_s = float(contrast.bar_s)
    theta_hat = float(contrast.theta_hat)
    family = TruncBivariateNormal(a_coeff=bar_s**2 / variance, b_coeff=1.,
                                  L=intervals[:, 0], U=intervals[:, 1],
                                  sig_omega=bar_s, sig_x=np.sqrt(variance))
    L_theta, U_theta = family.equal_tailed_interval(theta_hat, alpha=1 - level)
    cdf0 = np.clip(family.cdf(0., theta_hat), 0., 1.)
    return RayContrastResult(theta_hat=theta_hat,
                             lower_conf=L_theta * variance,
                             upper_conf=U_theta * variance,
                             p_value=float(np.clip(2 * min(cdf0, 1 - cdf0), 0., 1.)),
                             intervals=intervals,
                             family=family,
                             variance=variance)


def lasso_event(beta_hat, G_hat, Q, penalty, tol=1e-6, slack=1e-9):
    """
    The selection event of a lasso with data-dependent penalties and bounds: the same
    active set, signs and bound activations as beta_hat.

    The lasso solves minimize 1/2 beta' Q beta - beta' Y + sum_j D_j |beta_j| subject to
    L <= beta <= U, with (D, L, U) = penalty(Y). penalty returns None where Y is outside the
    conditioning event for the penalties (for example, where a conditioned sign is wrong).

    beta_hat, G_hat : the solution and the gradient Q beta_hat - Y at the observed Y; they
        fix the active set, signs and bound activations
    slack : the constraints A Y <= b are checked to slack * (1 + |b|)

    Returns event(Y) -> bool.
    """
    beta_hat = np.asarray(beta_hat, dtype=float)
    G_hat = np.asarray(G_hat, dtype=float)

    def event(Y):
        pen = penalty(Y)
        if pen is None:
            return False
        D, L, U = pen
        A, b = lasso_post_selection_constraints(beta_hat, G_hat, Q, D, L, U, tol=tol)[:2]
        AY = np.ravel(A @ Y)
        return bool(np.all(AY <= b + slack * (1 + np.abs(b))))

    return event


__all__ = ['find_truncation_set', 'ray_contrast_inference', 'lasso_event', 'ray_point',
           'ray_scale', 'RayContrastResult']
