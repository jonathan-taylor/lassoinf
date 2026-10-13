# --- Selection events evaluated along the conditioning ray ---
#
# For a contrast, the selection data move along the ray Y(w) = N_o + bar_N_o + bar_Gamma w,
# and the selection event depends on (theta_hat, bar_omega) only through the scalar w. Given
# an explicit characterization event(Y), the truncation set {w : event(Y(w))} is found
# numerically. The computations are in the C++ library (cpp/src/ray_selection.cpp), shared
# with Python (lassoinf.ray_selection). The uniLasso has a closed form for its truncation
# set (UniLassoInference), which this does not replace.

#' Truncation set of a selection event along the conditioning ray
#'
#' @description Experimental. For a contrast from \code{AffineConstraints$compute_contrast()},
#' the selection data move along the ray
#' \eqn{Y(w) = N_o + \bar N_o + \bar\Gamma w}, and the selection event depends on the data
#' only through the scalar \eqn{w}. Given an explicit characterization of the event, for
#' example the KKT conditions of a lasso whose penalties are functions of \eqn{Y}
#' (\code{lasso_event()}), the truncation set \eqn{\{w : \mathrm{event}(Y(w))\}} is located
#' by evaluating the event on a grid of \eqn{w} and bisecting each change. No refitting is
#' needed.
#'
#' Anything the event depends on, such as penalty factors computed from the data, must be a
#' function of the selection data \eqn{Y}; otherwise the event depends on more than \eqn{w}.
#' @param event A function of the selection data \code{Y} returning \code{TRUE} on the
#'   selection event, or an event from \code{lasso_event()} (evaluated in C++).
#' @param contrast An \code{AffineConstraintsContrast}.
#' @param n_sd,n_grid The event is evaluated on \code{n_grid} equally spaced points within
#'   \code{n_sd} standard deviations of the observed \eqn{w}, and assumed constant beyond
#'   them. The event can change more than once between two grid points; such short pieces
#'   are missed. A missed piece of length \eqn{e} changes the truncated probabilities by at
#'   most \eqn{e / (\bar s \sqrt{2\pi})}.
#' @param tol Each change between grid points is located by bisection, to \code{tol} times
#'   the standard deviation of \eqn{w}.
#' @return A matrix with two columns: disjoint intervals of \eqn{w} in increasing order.
#' @seealso \code{\link{ray_contrast_inference}}, \code{\link{lasso_event}}
#' @export
find_truncation_set <- function(event, contrast, n_sd = 10, n_grid = 2001, tol = 1e-10) {
  scale <- ray_scale_cpp(contrast$bar_s, contrast$naive_variance, contrast$theta_hat,
                         contrast$bar_theta)
  find_truncation_set_cpp(event, as.numeric(contrast$n_o + contrast$bar_n_o),
                          as.numeric(contrast$bar_gamma), scale[1], scale[2], n_sd,
                          as.integer(n_grid), tol)
}

#' Selection event of a lasso with data-dependent penalties
#'
#' @description Experimental. The selection event of the lasso
#' \deqn{\min_\beta \frac{1}{2} \beta' Q \beta - \beta' Y + \sum_j D_j |\beta_j|,
#' \quad L \le \beta \le U,}
#' with \eqn{(D, L, U)} functions of the selection data \eqn{Y}: the same active set, signs
#' and bound activations as \code{beta_hat}. For \code{find_truncation_set()}.
#' @param beta_hat The solution at the observed selection data.
#' @param G_hat The gradient \code{Q beta_hat - Y} at the observed selection data.
#' @param Q The quadratic, a matrix or \code{LinearOp}.
#' @param penalty A function of \code{Y} returning \code{list(D, L, U)} (\code{L} or
#'   \code{U} \code{NULL} for no bounds), or \code{NULL} where \code{Y} is outside the
#'   conditioning event for the penalties (for example, where a conditioned sign is wrong).
#' @param tol Tolerance for the active set.
#' @param slack The constraints \eqn{AY \le b} are checked to
#'   \code{slack * (|b_i| + |(AY)_i|)}, relative to the scale of the problem.
#' @return An event, evaluated in C++ (\code{penalty} is called in R); \code{event$call(Y)}
#'   evaluates it at \code{Y}.
#' @export
lasso_event <- function(beta_hat, G_hat, Q, penalty, tol = 1e-6, slack = 1e-9) {
  p <- length(beta_hat)
  vec <- function(v) if (is.null(v)) NULL else rep_len(as.numeric(v), p)
  pen <- function(Y) {
    out <- penalty(Y)
    if (is.null(out)) return(NULL)
    list(vec(out[[1]]), vec(out[[2]]), vec(out[[3]]))
  }
  if (is.matrix(Q)) Q <- unname(Q)
  new(LassoRayEventCpp, as.numeric(beta_hat), as.numeric(G_hat), Q, pen, tol, slack)
}

#' Selective inference along the conditioning ray
#'
#' @description Experimental. Selective inference for \eqn{\eta^T E[Z]}, \eqn{\eta} = \code{direction}, after the
#' selection event \code{event}, with the randomization of \code{affine_constraints}. The
#' truncation set is found by \code{find_truncation_set()}; given it, the estimate is a
#' Gaussian convolved with the randomization and truncated to the set, as in the polyhedral
#' lemma (no randomization: \code{scalar_noise = 0}).
#' @param affine_constraints An \code{AffineConstraints}.
#' @param direction The contrast vector.
#' @param event As for \code{find_truncation_set()}.
#' @param level Confidence level.
#' @param ... Passed to \code{find_truncation_set()}.
#' @return A list with the estimate \code{theta_hat}, the interval
#'   (\code{lower_conf}, \code{upper_conf}), \code{p_value}, the truncation set
#'   \code{intervals}, the \code{family} (a \code{TruncBivariateNormal}), the
#'   \code{variance}, and \code{pivot(theta)}, uniform when \code{theta} is the true value.
#' @examples
#' # the adaptive lasso: penalties lam / |b_ols(Y)| move along the ray
#' set.seed(1)
#' n <- 100; p <- 4
#' x <- matrix(rnorm(n * p), n, p)
#' y <- drop(x %*% c(0.5, -0.4, 0, 0)) + rnorm(n)
#' Q <- crossprod(x); Z <- drop(crossprod(x, y))
#' Y <- Z + drop(t(chol(Q)) %*% rnorm(p))        # randomization, scalar_noise = 1
#' penalty <- function(Y) list(3 / abs(solve(Q, Y)), NULL, NULL)
#' D <- penalty(Y)[[1]]
#' lasso_cd <- function(Q, Y, D) {   # coordinate descent
#'   b <- rep(0, length(Y))
#'   for (it in 1:2000) for (j in seq_along(Y)) {
#'     r <- Y[j] - sum(Q[j, ] * b) + Q[j, j] * b[j]
#'     b[j] <- sign(r) * max(abs(r) - D[j], 0) / Q[j, j]
#'   }
#'   b
#' }
#' b <- lasso_cd(Q, Y, D)
#' si <- new(AffineConstraints, Z, Y, Q, NULL, 1)
#' event <- lasso_event(b, drop(Q %*% b) - Y, Q, penalty)
#' k <- which(b != 0)[1]
#' res <- ray_contrast_inference(si, solve(Q, replace(rep(0, p), k, 1)), event, level = 0.9)
#' c(res$lower_conf, res$upper_conf)
#' @export
ray_contrast_inference <- function(affine_constraints, direction, event, level = 0.95, ...) {
  contrast <- affine_constraints$compute_contrast(as.numeric(direction))
  S <- find_truncation_set(event, contrast, ...)
  variance <- contrast$naive_variance
  bar_s <- contrast$bar_s
  theta_hat <- contrast$theta_hat
  family <- TruncBivariateNormal$new(a_coeff = bar_s^2 / variance, b_coeff = 1,
                                     L = S[, 1], U = S[, 2],
                                     sig_omega = bar_s, sig_x = sqrt(variance))
  L_U <- family$equal_tailed_interval(theta_hat, alpha = 1 - level)
  cdf0 <- min(max(family$cdf(theta = 0, x = theta_hat), 0), 1)
  list(theta_hat = theta_hat,
       lower_conf = L_U[1] * variance,
       upper_conf = L_U[2] * variance,
       p_value = min(max(2 * min(cdf0, 1 - cdf0), 0), 1),
       intervals = S,
       family = family,
       variance = variance,
       pivot = function(theta) family$cdf(theta = theta / variance, x = theta_hat))
}
