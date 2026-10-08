# The LASSO problem solved by glmnet, in the original coordinates
# (intercept, coefficients); mirrors python/lassoinf/glm_problem.py.
#
# glmnet solves, for family log-likelihood l,
#   min -(1 / sum(w)) sum_i w_i l(y_i, eta_i)
#       + lambda sum_j pf_j [(1 - alpha) / 2 (s_j beta_j)^2 + alpha |s_j beta_j|]
#   subject to lower_j <= beta_j <= upper_j,
# with eta = offset + beta_0 + X beta, s_j the standardization scale (weighted
# standard deviation, 1 if standardize = FALSE), pf rescaled to sum to the number
# of variables and
# excluded variables fixed at 0. In the original coordinates this is a bounded
# LASSO with weights D_j = lambda alpha pf_j s_j and smooth part the negative
# log-likelihood plus the ridge term. For the gaussian family glmnet fits the
# standardized response y / s_y, which divides the ridge term by s_y (weighted
# standard deviation of y, or its weighted root mean square without an intercept).

.glmnet_families <- c("gaussian", "binomial", "poisson")

#' glmnet's column scales
#'
#' @description Weighted standard deviations of the columns (with or without an
#' intercept), 1 if not standardizing.
#' @param x Design matrix.
#' @param weights Observation weights.
#' @param standardize As passed to \code{glmnet}.
#' @return The column scales, a vector of length \code{ncol(x)}.
#' @examples
#' x <- matrix(rnorm(40), 10, 4)
#' glmnet_scaling(x)
#' @export
glmnet_scaling <- function(x, weights = rep(1, nrow(x)), standardize = TRUE) {
  if (!standardize) return(rep(1, ncol(x)))
  w <- weights / sum(weights)
  sqrt(drop(crossprod(w, x^2)) - drop(crossprod(w, x))^2)
}

# scale of the gaussian response used by glmnet
.glmnet_response_scale <- function(y, weights, intercept) {
  w <- weights / sum(weights)
  m2 <- sum(w * y^2)
  sqrt(if (intercept) m2 - sum(w * y)^2 else m2)
}

.glmnet_penalty_factor <- function(penalty.factor, p, exclude) {
  pf <- rep_len(as.numeric(penalty.factor), p)
  excluded <- rep(FALSE, p)
  excluded[exclude] <- TRUE
  excluded <- excluded | is.infinite(pf)
  pf[excluded] <- 1
  pf <- pmax(pf, 0)
  list(pf = pf * p / sum(pf), excluded = excluded)
}

.mean_variance <- function(family, eta) {
  switch(family,
         gaussian = list(mu = eta, var = rep(1, length(eta))),
         binomial = { mu <- plogis(eta); list(mu = mu, var = mu * (1 - mu)) },
         poisson = { mu <- exp(eta); list(mu = mu, var = mu) },
         stop("family must be one of ", paste(.glmnet_families, collapse = ", ")))
}

#' The bounded LASSO problem glmnet solved
#'
#' @description Gradient and Hessian of the smooth part at the solution, penalty
#' weights and limits, in the original coordinates: the inputs of
#' \code{LassoInference}. \code{glmnet_problem_from_fit} reads the settings from a
#' fitted \code{glmnet} object.
#' @param x Design matrix.
#' @param y Response (0/1 or a two-level factor for binomial).
#' @param coef Fitted coefficients (original scale).
#' @param a0 Fitted intercept.
#' @param lambda The lambda of the fit.
#' @param family \code{"gaussian"}, \code{"binomial"} or \code{"poisson"}.
#' @param weights,offset Observation weights and offset.
#' @param alpha,penalty.factor,exclude,lower.limits,upper.limits,standardize,intercept As passed to \code{glmnet} (\code{exclude} 1-based).
#' @param scaling Column scales; default glmnet's convention (\code{glmnet_scaling}).
#' @param y_scale The ridge term is divided by \code{y_scale}; default glmnet's
#'   convention (scale of \code{y} for gaussian, 1 otherwise).
#' @param hessian \code{"dense"} (matrix) or \code{"operator"} (matrix-free \code{LinearOp}).
#' @param information Where \code{Q_hat} is evaluated: \code{"relaxed"} (default), a
#'   one-step relaxed fit, one Newton step for the unpenalized loss on the selected
#'   coordinates (the intercept and the nonzero coefficients) from the LASSO solution,
#'   \code{beta_E - H_EE^{-1} grad_E} with the Hessian \code{H} at the LASSO solution
#'   (limits and the ridge term are ignored); or \code{"lasso"}, the LASSO solution. The LASSO
#'   solution is shrunk, so for binomial and poisson its information overstates the
#'   information at the truth and \code{Q_hat / n} understates \code{Var(Z)}; the relaxed fit
#'   avoids this. The two agree for gaussian. \code{G_hat} is always at the LASSO solution.
#' @return A list with \code{beta_hat}, \code{G_hat}, \code{Q_hat}, \code{D}, \code{L},
#'   \code{U} and \code{intercept}; coordinates are \code{(intercept, coef)} when
#'   \code{intercept = TRUE}.
#' @examplesIf requireNamespace("glmnet", quietly = TRUE)
#' set.seed(1)
#' x <- matrix(rnorm(100 * 5), 100, 5)
#' y <- rbinom(100, 1, plogis(x[, 1]))
#' fit <- glmnet::glmnet(x, y, family = "binomial", lambda = c(0.1, 0.05),
#'                       control = list(thresh = 1e-14))
#' prob <- glmnet_problem_from_fit(fit, x, y)
#' prob$beta_hat
#' max(kkt_violation(prob$beta_hat, prob$G_hat, prob$D, prob$L, prob$U))
#' @export
glmnet_problem <- function(x, y, coef, a0 = 0, lambda, family = "gaussian",
                           weights = NULL, offset = NULL, alpha = 1,
                           penalty.factor = rep(1, ncol(x)), exclude = NULL,
                           lower.limits = -Inf, upper.limits = Inf,
                           standardize = TRUE, intercept = TRUE,
                           scaling = NULL, y_scale = NULL, hessian = c("dense", "operator"),
                           information = c("relaxed", "lasso")) {
  hessian <- match.arg(hessian)
  information <- match.arg(information)
  if (!family %in% .glmnet_families) stop("family must be one of ", paste(.glmnet_families, collapse = ", "))
  x <- as.matrix(x)
  n <- nrow(x); p <- ncol(x)
  if (family == "binomial" && is.factor(y)) y <- as.numeric(y == levels(y)[2])
  y <- as.numeric(y)
  if (is.null(weights)) weights <- rep(1, n)
  if (is.null(offset)) offset <- rep(0, n)
  w <- weights / sum(weights)
  if (is.null(scaling)) scaling <- glmnet_scaling(x, weights, standardize)
  if (is.null(y_scale)) y_scale <- if (family == "gaussian") .glmnet_response_scale(y, weights, intercept) else 1
  pen <- .glmnet_penalty_factor(penalty.factor, p, exclude)
  pf <- pen$pf

  lower <- rep_len(as.numeric(lower.limits), p)
  upper <- rep_len(as.numeric(upper.limits), p)
  # excluded variables are fixed at 0
  lower[pen$excluded] <- 0
  upper[pen$excluded] <- 0

  coef <- as.numeric(coef)
  eta <- offset + drop(x %*% coef) + if (intercept) a0 else 0
  mv <- .mean_variance(family, eta)

  ridge <- lambda * (1 - alpha) * pf * scaling^2 / y_scale
  ridge[pen$excluded] <- 0
  grad <- drop(crossprod(x, w * (mv$mu - y))) + ridge * coef
  D <- lambda * alpha * pf * scaling

  if (intercept) {
    X1 <- cbind(1, x)
    out <- list(beta_hat = c(a0, coef), G_hat = c(sum(w * (mv$mu - y)), grad),
                D = c(0, D), L = c(-Inf, lower), U = c(Inf, upper))
    ridge1 <- c(0, ridge)
  } else {
    X1 <- x
    out <- list(beta_hat = coef, G_hat = grad, D = D, L = lower, U = upper)
    ridge1 <- ridge
  }
  if (information == "relaxed" && family != "gaussian") {
    mv <- .relaxed_mean_variance(X1, y, w, eta,
                                 which(out$beta_hat != 0 | (intercept & seq_along(out$beta_hat) == 1)),
                                 family, mv)
  }
  out$Q_hat <- if (hessian == "dense") {
    crossprod(X1, X1 * (w * mv$var)) + diag(ridge1, length(ridge1))
  } else {
    new(LinearOp, X1, w * mv$var, ridge1)
  }
  out$intercept <- intercept
  out
}

# mean and variance at the one-step relaxed fit on the columns `active` of X1: one Newton
# step for the unpenalized loss from the LASSO solution, with the Hessian there,
#   beta_E - H_EE^{-1} grad_E
# (for the LASSO grad_E = -D_E sign(beta_E), so this undoes the shrinkage). Falls back to
# `lasso_mv` if H_EE is singular.
.relaxed_mean_variance <- function(X1, y, w, eta, active, family, lasso_mv) {
  if (length(active) == 0) return(lasso_mv)
  XE <- X1[, active, drop = FALSE]
  grad <- crossprod(XE, w * (lasso_mv$mu - y))
  hess <- crossprod(XE, XE * (w * lasso_mv$var))
  step <- tryCatch(solve(hess, grad), error = function(e) NULL)
  if (is.null(step)) {
    warning("the Hessian on the selected coordinates is singular; ",
            "using the information at the LASSO solution")
    return(lasso_mv)
  }
  .mean_variance(family, eta - drop(XE %*% step))
}

#' KKT violation of the bounded LASSO
#'
#' @description Coordinatewise violation of \code{0 in G + D d|beta| + N_[L, U](beta)};
#' 0 at the exact solution. Coordinates with \code{L = U} are fixed and skipped.
#' @param beta,G,D,L,U Solution, gradient of the smooth part, penalty weights and limits
#'   (e.g. from \code{glmnet_problem}).
#' @param tol Tolerance for being at 0 or at a limit.
#' @return The violation for each coordinate, nonnegative.
#' @examples
#' # beta = (1, 0) solves min (beta - c(2, 0.5))^2 / 2 + |beta|_1
#' beta <- c(1, 0)
#' kkt_violation(beta, G = beta - c(2, 0.5), D = c(1, 1), L = c(-Inf, -Inf), U = c(Inf, Inf))
#' @export
kkt_violation <- function(beta, G, D, L, U, tol = 1e-8) {
  zero <- abs(beta) <= tol
  lo <- ifelse(zero, G - D, G + D * sign(beta))
  hi <- ifelse(zero, G + D, lo)
  at_U <- beta >= U - tol
  at_L <- beta <= L + tol
  viol <- ifelse(at_U, pmax(lo, 0), ifelse(at_L, pmax(-hi, 0), pmax(lo, -hi, 0)))
  viol[(U - L <= tol) | (at_U & at_L)] <- 0
  viol
}

.glmnet_fit_family <- function(fit) {
  if (inherits(fit, "elnet")) return("gaussian")
  if (inherits(fit, "lognet")) return("binomial")
  if (inherits(fit, "fishnet")) return("poisson")
  if (inherits(fit, "glmnetfit") && !is.null(fit$family)) {
    fam <- fit$family
    canonical <- c(gaussian = "identity", binomial = "logit", poisson = "log")
    if (fam$family %in% names(canonical) && fam$link == canonical[[fam$family]]) return(fam$family)
  }
  stop("unsupported glmnet family; supported: ", paste(.glmnet_families, collapse = ", "),
       " with canonical links")
}

#' @rdname glmnet_problem
#' @param fit A fitted \code{glmnet} object.
#' @param s A value in \code{fit$lambda} (default: the last one); refit at other values.
#' @param envir Environment in which to evaluate the arguments of \code{fit$call}.
#' @export
glmnet_problem_from_fit <- function(fit, x, y, s = NULL, weights = NULL, offset = NULL,
                                    hessian = c("dense", "operator"),
                                    information = c("relaxed", "lasso"), envir = parent.frame()) {
  hessian <- match.arg(hessian)
  information <- match.arg(information)
  family <- .glmnet_fit_family(fit)
  if (is.null(s)) {
    k <- length(fit$lambda)
  } else {
    k <- which(abs(fit$lambda - s) <= 1e-10 * s)
    if (length(k) == 0) stop("s must be one of fit$lambda; refit glmnet at it")
    k <- k[1]
  }
  # settings used for the fit, with glmnet's defaults
  call_args <- as.list(fit$call)[-1]
  arg <- function(name, default) {
    if (name %in% names(call_args)) eval(call_args[[name]], envir) else default
  }
  p <- ncol(x)
  if (is.null(weights)) weights <- arg("weights", NULL)
  if (is.null(offset)) offset <- arg("offset", NULL)
  exclude <- arg("exclude", NULL)
  if (is.function(exclude)) {
    exclude <- exclude(x = x, y = y, weights = if (is.null(weights)) rep(1, nrow(x)) else weights)
  }
  problem <- glmnet_problem(x, y,
                 coef = as.numeric(fit$beta[, k]),
                 a0 = as.numeric(fit$a0[k]),
                 lambda = fit$lambda[k],
                 family = family,
                 weights = weights,
                 offset = offset,
                 alpha = arg("alpha", 1),
                 penalty.factor = arg("penalty.factor", rep(1, p)),
                 exclude = exclude,
                 lower.limits = arg("lower.limits", -Inf),
                 upper.limits = arg("upper.limits", Inf),
                 standardize = arg("standardize", TRUE),
                 intercept = arg("intercept", TRUE),
                 hessian = hessian,
                 information = information)
  violation <- max(kkt_violation(problem$beta_hat, problem$G_hat, problem$D, problem$L, problem$U))
  if (violation > 1e-4 * fit$lambda[k]) {
    warning(sprintf("fit violates the KKT conditions by %.1e * lambda; it may not have converged (decrease thresh, e.g. control = list(thresh = 1e-14))",
                    violation / fit$lambda[k]))
  }
  problem
}
