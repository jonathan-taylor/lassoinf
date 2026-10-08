# Selective inference from a fitted glmnet object in one call; mirrors
# python/lassoinf/glm_inference.py.
#
# With the LASSO problem (beta_hat, G, Q_hat) on the rows used for selection
# (glmnet_problem_from_fit; Q_hat at the one-step relaxed fit by default), the
# selection score is Z_noisy = -G + Q_hat beta_hat, and
#   Var(Z_full) = dispersion * Q_hat / sum(weights).
# Without held-out data Z_full = Z_noisy (scalar_noise = 0). With held-out data, selecting
# on a proportion pi = sum(weights[rows]) / sum(weights) is randomization with
# scalar_noise = (1 - pi) / pi, and Z_full linearizes the full-data gradient at one
# Newton step for the full-data loss on the selected coordinates from beta_hat,
#   beta_tilde = beta_hat - H_EE^{-1} grad_E,   Z_full = -G_full(beta_tilde) + Q_hat beta_tilde.
# Linearizing at the shrunk beta_hat instead biases Z_full and undercovers.

.eta <- function(x, beta, offset, intercept) {
  if (intercept) offset + beta[1] + drop(x %*% beta[-1]) else offset + drop(x %*% beta)
}

# gradient of the unpenalized loss (1 / sum(w)) sum_i w_i l_i at beta (intercept first
# when intercept = TRUE)
.loss_gradient <- function(x, y, beta, family, weights, offset, intercept) {
  r <- weights / sum(weights) * (.mean_variance(family, .eta(x, beta, offset, intercept))$mu - y)
  g <- drop(crossprod(x, r))
  if (intercept) c(sum(r), g) else g
}

# one Newton step for the unpenalized loss on the nonzero coordinates of beta (and the
# intercept), with the Hessian at beta
.newton_step <- function(x, y, beta, family, weights, offset, intercept) {
  A <- which(beta != 0 | (intercept & seq_along(beta) == 1))
  if (length(A) == 0) return(beta)
  cols <- if (intercept) A[A > 1] - 1 else A
  XA <- if (intercept && 1 %in% A) cbind(1, x[, cols, drop = FALSE]) else x[, cols, drop = FALSE]
  w <- weights / sum(weights)
  mv <- .mean_variance(family, .eta(x, beta, offset, intercept))
  step <- tryCatch(solve(crossprod(XA, XA * (w * mv$var)), crossprod(XA, w * (mv$mu - y))),
                   error = function(e) NULL)
  if (is.null(step)) {
    warning("the full-data Hessian on the selected coordinates is singular; ",
            "linearizing at the LASSO solution")
    return(beta)
  }
  beta[A] <- beta[A] - drop(step)
  beta
}

# residual variance of the weighted least squares fit on all the variables
.gaussian_dispersion <- function(x, y, weights, offset, intercept) {
  X1 <- if (intercept) cbind(1, x) else x
  if (nrow(X1) <= ncol(X1)) {
    stop("cannot estimate the dispersion with nrow(x) <= ncol(x) (+ intercept); supply dispersion")
  }
  ols <- stats::lm.wfit(X1, y - offset, weights)
  sum(weights * ols$residuals^2) / (nrow(X1) - ols$rank)
}

.scale_operator <- function(Q, c) if (inherits(Q, "Rcpp_LinearOp")) Q$scaled(c) else c * Q

# argument passed to glmnet, or NULL
.fit_arg <- function(fit, name, envir) {
  call_args <- as.list(fit$call)[-1]
  if (name %in% names(call_args)) eval(call_args[[name]], envir) else NULL
}

#' Selective inference after a glmnet fit
#'
#' @description Intervals and p-values for the coefficients selected by a \code{glmnet}
#' fit at \code{lambda = s}, accounting for the selection. Without
#' \code{selection_rows} the fit is on all the data and there is no randomization (the
#' polyhedral approach). With \code{selection_rows}, the fit is on those rows only and
#' inference uses all the data (carving): selecting on a proportion \eqn{\pi} of the data
#' is randomization with \code{scalar_noise = (1 - pi) / pi}.
#'
#' The selection score is \code{Z_noisy = -G + Q_hat beta_hat} from
#' \code{glmnet_problem_from_fit} on the selection rows, and
#' \code{Sigma = dispersion * Q_hat / sum(weights)}. Without held-out data
#' \code{Z_full = Z_noisy}. With held-out data, \code{Z_full = -G_full(beta_tilde) +
#' Q_hat beta_tilde}, with \code{G_full} the full-data gradient and \code{beta_tilde} one
#' Newton step for the full-data loss on the selected coordinates from \code{beta_hat}. Weights are
#' treated as prior (frequency or precision) weights, as in \code{glm}.
#'
#' @param fit A fitted \code{glmnet} object (gaussian, binomial or poisson).
#' @param x,y The full data. With \code{selection_rows}, \code{fit} must have been fitted
#'   on \code{x[selection_rows, ]} and \code{y[selection_rows]}.
#' @param s A value in \code{fit$lambda} (default: the last one).
#' @param selection_rows Rows (indices or logical) the fit used for selection; \code{NULL}
#'   (default) for all of them, i.e. no randomization.
#' @param weights,offset Full-data observation weights and offset; default those passed to
#'   \code{glmnet} (which must then be for the full data).
#' @param dispersion Dispersion; default 1 for binomial and poisson and, for gaussian, the
#'   residual variance of the least squares fit on all the variables (requires
#'   \code{nrow(x) > ncol(x)}; supply it otherwise).
#' @param level Confidence level.
#' @param hessian,information As in \code{\link{glmnet_problem}}.
#' @param envir Environment in which to evaluate the arguments of \code{fit$call}.
#' @return A \code{\link{LassoInference}} object; \code{$summary()} gives the selective
#'   intervals and p-values. Indices are 0-based, with 0 the intercept when the fit has one.
#' @references
#' Lee, J. D., Sun, D. L., Sun, Y. and Taylor, J. E. (2016). Exact post-selection
#' inference, with application to the lasso. \emph{The Annals of Statistics}, 44(3),
#' 907--927. \doi{10.1214/15-AOS1371}
#'
#' Panigrahi, S., Fry, K. and Taylor, J. (2024). Exact selective inference with
#' randomization. \emph{Biometrika}, 111(4), 1109--1127. \doi{10.1093/biomet/asae019}
#' @examplesIf requireNamespace("glmnet", quietly = TRUE)
#' set.seed(1)
#' n <- 200; p <- 10
#' x <- matrix(rnorm(n * p), n, p)
#' y <- rbinom(n, 1, plogis(x[, 1] - x[, 2]))
#' fit <- glmnet::glmnet(x, y, family = "binomial", lambda = c(0.2, 0.1, 0.05),
#'                       control = list(thresh = 1e-14))
#' glmnet_inference(fit, x, y, level = 0.9)$summary()
#'
#' # carving: select on half of the data, inference with all of it
#' rows <- sample(n, n / 2)
#' fit <- glmnet::glmnet(x[rows, ], y[rows], family = "binomial",
#'                       lambda = c(0.2, 0.1, 0.05), control = list(thresh = 1e-14))
#' glmnet_inference(fit, x, y, selection_rows = rows, level = 0.9)$summary()
#' @export
glmnet_inference <- function(fit, x, y, s = NULL, selection_rows = NULL, weights = NULL,
                             offset = NULL, dispersion = NULL, level = 0.95,
                             hessian = c("dense", "operator"),
                             information = c("relaxed", "lasso"), envir = parent.frame()) {
  hessian <- match.arg(hessian)
  information <- match.arg(information)
  family <- .glmnet_fit_family(fit)
  x <- as.matrix(x)
  n <- nrow(x)
  if (family == "binomial" && is.factor(y)) y <- as.numeric(y == levels(y)[2])
  y <- as.numeric(y)
  if (length(y) != n) stop("y must have length nrow(x)")
  if (is.null(weights)) weights <- .fit_arg(fit, "weights", envir)
  if (is.null(offset)) offset <- .fit_arg(fit, "offset", envir)
  if (is.null(weights)) weights <- rep(1, n)
  if (is.null(offset)) offset <- rep(0, n)
  if (length(weights) != n || length(offset) != n) {
    stop("weights and offset must be for the full data (length nrow(x)); ",
         "with selection_rows, pass them explicitly")
  }
  rows <- if (is.null(selection_rows)) seq_len(n) else seq_len(n)[selection_rows]
  if (anyNA(rows) || anyDuplicated(rows) || length(rows) == 0) stop("invalid selection_rows")

  prob <- glmnet_problem_from_fit(fit, x[rows, , drop = FALSE], y[rows], s = s,
                                  weights = weights[rows], offset = offset[rows],
                                  hessian = hessian, information = information, envir = envir)
  Q <- prob$Q_hat
  b <- prob$beta_hat
  if (length(rows) == n) {
    Z_full <- -prob$G_hat + matvec(Q, b)
    scalar_noise <- 0
  } else {
    b_tilde <- .newton_step(x, y, b, family, weights, offset, prob$intercept)
    # G = loss gradient + ridge * beta; recover ridge * beta_hat from the selection problem
    ridge_b <- prob$G_hat - .loss_gradient(x[rows, , drop = FALSE], y[rows], b, family,
                                           weights[rows], offset[rows], prob$intercept)
    ridge_b_tilde <- ifelse(b != 0, ridge_b * b_tilde / b, 0)
    G_full <- .loss_gradient(x, y, b_tilde, family, weights, offset, prob$intercept) + ridge_b_tilde
    Z_full <- -G_full + matvec(Q, b_tilde)
    pi_selection <- sum(weights[rows]) / sum(weights)
    scalar_noise <- (1 - pi_selection) / pi_selection
  }
  if (is.null(dispersion)) {
    dispersion <- if (family == "gaussian") {
      .gaussian_dispersion(x, y, weights, offset, prob$intercept)
    } else 1
  }
  LassoInference$new(b, prob$G_hat, Q, prob$D, prob$L, prob$U,
                     Z_full = Z_full,
                     Sigma = .scale_operator(Q, dispersion / sum(weights)),
                     Sigma_noise = NULL, scalar_noise = scalar_noise, level = level)
}
