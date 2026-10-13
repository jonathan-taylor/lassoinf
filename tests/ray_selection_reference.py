"""
Pure-Python reference implementations of the ray-selection computations that lassoinf runs
in C++ (cpp/src/ray_selection.cpp). The tests check the C++ code against these.
"""
import numpy as np

from lassoinf.lasso import lasso_post_selection_constraints


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
            if not lo < mid < hi:
                break
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


def lasso_event(beta_hat, G_hat, Q, penalty, tol=1e-6, slack=1e-9):
    """
    The selection event of a lasso with data-dependent penalties and bounds: the same
    active set, signs and bound activations as beta_hat.

    The lasso solves minimize 1/2 beta' Q beta - beta' Y + sum_j D_j |beta_j| subject to
    L <= beta <= U, with (D, L, U) = penalty(Y). penalty returns None where Y is outside the
    conditioning event for the penalties (for example, where a conditioned sign is wrong).

    beta_hat, G_hat : the solution and the gradient Q beta_hat - Y at the observed Y; they
        fix the active set, signs and bound activations
    slack : the constraints A Y <= b are checked to slack * (|b_i| + |(A Y)_i|)

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
        return bool(np.all(AY <= b + slack * (np.abs(b) + np.abs(AY))))

    return event
