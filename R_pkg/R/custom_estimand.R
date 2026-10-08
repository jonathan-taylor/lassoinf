# Selective inference for user-specified one-dimensional estimands.
# Mirrors python/lassoinf/custom_estimand.py; the geometry is done in C++
# (cpp/src/custom_estimand.cpp), truncated-normal inference in R.
#
# Variable indices (E, E_c, screened, index columns) are 0-based, as in
# LassoInference$summary() and the Python package.

#' Linear operators
#'
#' @description Matrix-free operators for \code{Q_hat}, \code{Sigma} and
#' \code{Sigma_noise} in \code{LassoInference}. \code{dense_operator(M)} wraps a
#' matrix; \code{xtvx_operator(X, weights)} represents \code{t(X) \%*\% diag(weights) \%*\% X}
#' without forming it, which is what wide designs (n << p) need.
#' @param M A square numeric matrix.
#' @param X A numeric design matrix.
#' @param weights Observation weights, length \code{nrow(X)}.
#' @return A \code{LinearOp} with methods \code{multiply}, \code{multiply_transpose},
#'   \code{rows}, \code{cols}, \code{diagonal}, \code{to_dense} and \code{scaled(c)}
#'   (\code{c} times the operator, sharing its data).
#' @examples
#' set.seed(1)
#' x <- matrix(rnorm(50 * 5), 50, 5)
#' v <- rnorm(5)
#' Q <- xtvx_operator(x)
#' all.equal(Q$multiply(v), drop(crossprod(x) %*% v))
#' all.equal(Q$scaled(2)$to_dense(), 2 * crossprod(x))
#' dense_operator(crossprod(x))$rows()
#' @export
dense_operator <- function(M) new(LinearOp, as.matrix(M))

#' @rdname dense_operator
#' @export
xtvx_operator <- function(X, weights = rep(1, nrow(X))) {
  new(LinearOp, as.matrix(X), as.numeric(weights))
}

#' Estimand specifications
#'
#' @description \code{contrast_estimand(eta, offset)}: estimator
#' \code{sum(eta * Z_full) + offset} for the estimand \code{sum(eta * E[Z_full]) + offset}.
#' \code{covariance_estimand(theta_hat, variance, score_cov)}: estimator given by its
#' value, \code{Var(theta_hat)} and \code{Cov(Z_full, theta_hat)}; requires \code{Sigma_noise}.
#' @param eta Contrast vector.
#' @param offset Constant added to the estimator and the estimand.
#' @param theta_hat Observed value of the estimator.
#' @param variance Variance of the estimator.
#' @param score_cov Covariance of \code{Z_full} with the estimator.
#' @param estimand A contrast estimand.
#' @param Z A score vector.
#' @return \code{contrast_estimand} and \code{covariance_estimand}: estimand
#'   specifications (lists of class \code{ContrastEstimand} and \code{CovarianceEstimand}),
#'   for \code{contrast_inference}, \code{custom_estimand_inference} and
#'   \code{estimand_summary}. \code{estimand_value}: the value of the estimator at \code{Z}.
#' @examples
#' e <- contrast_estimand(c(1, -1, 0), offset = 0.5)
#' estimand_value(e, c(2, 1, 5))
#' @export
contrast_estimand <- function(eta, offset = 0) {
  structure(list(eta = as.numeric(eta), offset = offset), class = "ContrastEstimand")
}

#' @rdname contrast_estimand
#' @export
covariance_estimand <- function(theta_hat, variance, score_cov) {
  structure(list(theta_hat = theta_hat, variance = variance, score_cov = as.numeric(score_cov)),
            class = "CovarianceEstimand")
}

#' @rdname contrast_estimand
#' @export
estimand_value <- function(estimand, Z) sum(estimand$eta * Z) + estimand$offset

.lasso_inference <- function(selection) {
  if (inherits(selection, "ScreenedSelection")) selection$lasso_inference else selection
}

.or_empty <- function(x) if (is.null(x)) numeric(0) else as.numeric(x)

#' SelectionCoordinates Class
#'
#' @description The coordinates in which the selection constraints are expressed.
#' With \code{basis = "refit"}: \code{bar_beta_E = W Z_E} and the inactive scores
#' \code{U_{-E} = Z_{-E} - Q_{-E,E} W Z_E} (in OLS, \code{t(X_{-E}) (I - P_E) y}).
#' With \code{basis = "lasso"}: the LASSO solution \code{beta_hat_E} and the
#' negative gradient at it, \code{-grad_{-E}}; these differ from the refit
#' coordinates by constants, so they give the same contrast with a shifted value.
#' Everything is computed with \code{Q_hat} matrix-vector products.
#'
#' @examplesIf requireNamespace("glmnet", quietly = TRUE)
#' set.seed(1)
#' n <- 100; p <- 10
#' x <- matrix(rnorm(n * p), n, p)
#' y <- x[, 1] - x[, 2] + rnorm(n)
#' fit <- glmnet::glmnet(x, y, lambda = c(0.3, 0.2), control = list(thresh = 1e-14))
#' li <- glmnet_inference(fit, x, y, level = 0.9)   # index 0 is the intercept
#' coords <- SelectionCoordinates$new(li)
#' coords$E_c
#' # coefficient of the first inactive variable in the model E u {j}
#' res <- contrast_inference(li, coords$inactive_coef(coords$E_c[1]))
#' res[c("lower_conf", "upper_conf", "p_value")]
#' @export
SelectionCoordinates <- R6::R6Class("SelectionCoordinates",
  public = list(
    #' @field cpp The C++ object.
    cpp = NULL,
    #' @field E Active set (0-based).
    E = NULL,
    #' @field E_c Inactive set (0-based).
    E_c = NULL,

    #' @description Create the coordinates for a selection.
    #' @param selection A \code{LassoInference} or \code{ScreenedSelection}.
    #' @param Q_diag Optional diagonal of \code{Q_hat}; lets \code{inactive_S} use
    #'   \code{|E|} matvecs for many variables. Taken from \code{Q_hat} when known.
    initialize = function(selection, Q_diag = NULL) {
      li <- .lasso_inference(selection)
      if (is.null(Q_diag)) {
        Q_diag <- if (inherits(li$Q_hat, "Rcpp_LinearOp")) li$Q_hat$diagonal() else diag(as.matrix(li$Q_hat))
      }
      self$cpp <- selection_coordinates_cpp(li$score, li$Z_noisy, li$beta_hat, li$G_hat, as.numeric(Q_diag))
      self$E <- self$cpp$E()
      self$E_c <- self$cpp$E_c()
    },

    #' @description Contrast \code{eta} with \code{sum(eta * Z) = sum(a_E * bar_beta_E) + sum(a_Ec * U_{-E})}.
    #' @param a_E Coefficients on \code{bar_beta_E} (or \code{NULL}).
    #' @param a_Ec Coefficients on \code{U_{-E}} (or \code{NULL}).
    contrast = function(a_E = NULL, a_Ec = NULL) {
      self$cpp$contrast(.or_empty(a_E), .or_empty(a_Ec))
    },

    #' @description Estimand linear in the selection coordinates.
    #' @param a_E Coefficients on the active coordinates.
    #' @param a_Ec Coefficients on the inactive scores.
    #' @param basis \code{"refit"} or \code{"lasso"}.
    estimand = function(a_E = NULL, a_Ec = NULL, basis = "refit") {
      self$cpp$estimand(.or_empty(a_E), .or_empty(a_Ec), basis)
    },

    #' @description Score of inactive variable \code{j}.
    #' @param j Inactive variable (0-based).
    #' @param basis \code{"refit"} or \code{"lasso"}.
    inactive_score = function(j, basis = "refit") self$cpp$inactive_score(as.integer(j), basis),

    #' @description Coefficient of inactive variable \code{j} in the model \code{E u {j}}.
    #' @param j Inactive variable (0-based).
    #' @param basis \code{"refit"} or \code{"lasso"}.
    inactive_coef = function(j, basis = "refit") self$cpp$inactive_coef(as.integer(j), basis),

    #' @description \code{S_jj = Q_jj - Q_{j,E} W Q_{E,j}} for inactive variables.
    #' @param variables Inactive variables (0-based).
    inactive_S = function(variables) self$cpp$inactive_S(as.integer(variables)),

    #' @description \code{bar_beta_E} for a score vector.
    #' @param Z A score vector.
    refit_coef = function(Z) self$cpp$refit_coef(Z),

    #' @description \code{U_{-E}} for a score vector.
    #' @param Z A score vector.
    refit_score = function(Z) self$cpp$refit_score(Z),

    #' @description Offsets of the lasso basis: active coefficients and inactive scores.
    lasso_offsets = function() {
      list(coef = self$cpp$lasso_coef_offset(), score = self$cpp$lasso_score_offset())
    },

    #' @description The inactive score operator.
    score_operator = function() self$cpp$score_operator()
  )
)

#' ScreenedSelection Class
#'
#' @description LASSO selection followed by screening of inactive variables on the
#' gradient at the LASSO solution, \code{G = grad(beta_hat)_{-E}}: keep variables with
#' \code{|G_j| > threshold}, or the \code{top_k} by \code{|G_j|}. Inference conditions on
#' the kept variables and the signs of their gradients; for \code{top_k} with
#' \code{conditioning = "first_dropped"} (default) also on the largest dropped variable
#' and its sign (O(p) constraints), with \code{"exact"} on all (kept, dropped) pairs.
#' Can be passed wherever a \code{LassoInference} is accepted by the estimand functions.
#'
#' @examplesIf requireNamespace("glmnet", quietly = TRUE)
#' set.seed(1)
#' n <- 100; p <- 10
#' x <- matrix(rnorm(n * p), n, p)
#' y <- x[, 1] - x[, 2] + rnorm(n)
#' fit <- glmnet::glmnet(x, y, lambda = c(0.3, 0.2), control = list(thresh = 1e-14))
#' li <- glmnet_inference(fit, x, y, level = 0.9)   # index 0 is the intercept
#' screen <- ScreenedSelection$new(li, top_k = 3)
#' screen$screened
#' inactive_summary(screen)
#' @export
ScreenedSelection <- R6::R6Class("ScreenedSelection",
  public = list(
    #' @field lasso_inference The underlying \code{LassoInference}.
    lasso_inference = NULL,
    #' @field threshold Screening threshold (or \code{NULL}).
    threshold = NULL,
    #' @field top_k Number of variables kept (or \code{NULL}).
    top_k = NULL,
    #' @field conditioning Conditioning for \code{top_k}.
    conditioning = NULL,
    #' @field screened Kept inactive variables (0-based), by decreasing \code{|G_j|}.
    screened = NULL,
    #' @field screened_signs Signs of their gradients.
    screened_signs = NULL,
    #' @field first_dropped Largest dropped variable (0-based) for \code{"first_dropped"}, else \code{NULL}.
    first_dropped = NULL,
    #' @field first_dropped_sign Sign of its gradient.
    first_dropped_sign = NULL,
    #' @field A Constraints of the combined selection event (\code{LinearOp}).
    A = NULL,
    #' @field b Constraint offsets.
    b = NULL,
    #' @field A_screen Screening constraints alone (\code{LinearOp}).
    A_screen = NULL,
    #' @field b_screen Screening constraint offsets.
    b_screen = NULL,
    #' @field si The \code{AffineConstraints} of the LASSO inference.
    si = NULL,
    #' @field level Confidence level.
    level = NULL,

    #' @description Screen inactive variables.
    #' @param lasso_inference A \code{LassoInference}.
    #' @param threshold Keep \code{|G_j| > threshold}.
    #' @param top_k Keep the \code{top_k} largest \code{|G_j|}.
    #' @param conditioning \code{"first_dropped"} or \code{"exact"} (for \code{top_k}).
    initialize = function(lasso_inference, threshold = NULL, top_k = NULL, conditioning = "first_dropped") {
      if (is.null(threshold) == is.null(top_k)) stop("specify exactly one of threshold or top_k")
      li <- lasso_inference
      coords <- SelectionCoordinates$new(li)
      cpp <- screened_selection_cpp(li$A, li$b, coords$cpp, li$G_hat, li$Z_noisy,
                                    if (is.null(threshold)) NaN else threshold,
                                    if (is.null(top_k)) -1L else as.integer(top_k),
                                    conditioning)
      if (!cpp$observed_feasible()) warning("observed data do not satisfy the screening constraints")
      self$lasso_inference <- li
      self$threshold <- threshold
      self$top_k <- top_k
      self$conditioning <- if (is.null(top_k)) NULL else conditioning
      self$screened <- cpp$screened()
      self$screened_signs <- cpp$screened_signs()
      if (cpp$first_dropped() >= 0) {
        self$first_dropped <- cpp$first_dropped()
        self$first_dropped_sign <- cpp$first_dropped_sign()
      }
      self$A <- cpp$A()
      self$b <- cpp$b()
      self$A_screen <- cpp$A_screen()
      self$b_screen <- cpp$b_screen()
      self$si <- li$si
      self$level <- li$level
    }
  )
)

#' Truncated normal inference from the polyhedral lemma
#'
#' @description Equal-tailed interval and two-sided p-value for \code{H0: theta = null_value}.
#' @param contrast An \code{AffineConstraintsContrast}.
#' @param variance Variance of the estimator.
#' @param A Constraint matrix or \code{LinearOp}.
#' @param b Constraint offsets.
#' @param level Confidence level.
#' @param null_value Null value of the estimand.
#' @return Named vector \code{c(lower, upper, p_value)}.
#' @examplesIf requireNamespace("glmnet", quietly = TRUE)
#' set.seed(1)
#' n <- 100; p <- 10
#' x <- matrix(rnorm(n * p), n, p)
#' y <- x[, 1] - x[, 2] + rnorm(n)
#' fit <- glmnet::glmnet(x, y, lambda = c(0.3, 0.2), control = list(thresh = 1e-14))
#' li <- glmnet_inference(fit, x, y, level = 0.9)   # index 0 is the intercept
#' v <- li$contrasts[["1"]]   # the coefficient of x1 in the selected model
#' contrast <- li$si$compute_contrast(v)
#' truncated_normal_inference(contrast, contrast$naive_variance, li$A, li$b, level = 0.9)
#' @export
truncated_normal_inference <- function(contrast, variance, A, b, level = 0.95, null_value = 0) {
  bar_s <- contrast$bar_s
  # interval for bar_theta at theta_hat = 0; the constraint is
  # L_0 <= (bar_s^2 / variance) * theta_hat + bar_theta <= U_0
  interval <- contrast$get_interval(0.0, A, b)
  if (is.nan(interval[1])) stop("observed data do not satisfy the selection constraints")
  tbn <- TruncBivariateNormal$new(a_coeff = bar_s^2 / variance, b_coeff = 1.0,
                                  L = interval[1], U = interval[2],
                                  sig_omega = bar_s, sig_x = sqrt(variance))
  theta_hat <- contrast$theta_hat
  # natural parameter: mean = natural * variance
  nat <- tbn$equal_tailed_interval(theta_hat, alpha = 1 - level)
  cdf_val <- min(max(tbn$cdf(theta = null_value / variance, x = theta_hat), 0.0), 1.0)
  p_value <- min(max(2 * min(cdf_val, 1.0 - cdf_val), 0.0), 1.0)
  c(lower = nat[1] * variance, upper = nat[2] * variance, p_value = p_value)
}

.inference <- function(selection, contrast, variance, level, null_value) {
  if (is.null(level)) level <- selection$level
  res <- truncated_normal_inference(contrast, variance, selection$A, selection$b,
                                    level = level, null_value = null_value)
  list(estimate = contrast$theta_hat, lower_conf = unname(res["lower"]),
       upper_conf = unname(res["upper"]), p_value = unname(res["p_value"]),
       variance = variance, contrast = contrast)
}

#' Selective inference for a contrast estimand
#'
#' @description Inference for \code{theta = sum(eta * E[Z_full]) + offset}, as in
#' \code{LassoInference$summary()}.
#' @param selection A \code{LassoInference} or \code{ScreenedSelection}.
#' @param estimand A \code{contrast_estimand} or a contrast vector \code{eta}.
#' @param level Confidence level (default: the selection's level).
#' @param null_value Null value for the p-value.
#' @return A list with \code{estimate}, \code{lower_conf}, \code{upper_conf},
#'   \code{p_value}, \code{variance} and \code{contrast}.
#' @examplesIf requireNamespace("glmnet", quietly = TRUE)
#' set.seed(1)
#' n <- 100; p <- 10
#' x <- matrix(rnorm(n * p), n, p)
#' y <- x[, 1] - x[, 2] + rnorm(n)
#' fit <- glmnet::glmnet(x, y, lambda = c(0.3, 0.2), control = list(thresh = 1e-14))
#' li <- glmnet_inference(fit, x, y, level = 0.9)   # index 0 is the intercept
#' # the sum of the coefficients of x1 and x2 in the selected model
#' eta <- li$contrasts[["1"]] + li$contrasts[["2"]]
#' contrast_inference(li, eta)[c("estimate", "lower_conf", "upper_conf", "p_value")]
#' @export
contrast_inference <- function(selection, estimand, level = NULL, null_value = 0) {
  if (!inherits(estimand, "ContrastEstimand")) estimand <- contrast_estimand(estimand)
  li <- .lasso_inference(selection)
  contrast <- li$si$compute_contrast(estimand$eta)
  # naive_variance = eta' Sigma eta
  res <- .inference(selection, contrast, contrast$naive_variance, level, null_value - estimand$offset)
  res$estimate <- res$estimate + estimand$offset
  res$lower_conf <- res$lower_conf + estimand$offset
  res$upper_conf <- res$upper_conf + estimand$offset
  res
}

#' Selective inference for an estimand given by its covariance with the score
#'
#' @description For \code{(theta_hat, Z_full)} jointly Gaussian with known
#' \code{Var(theta_hat)} and \code{Cov(Z_full, theta_hat)}. Requires \code{Sigma_noise};
#' with \code{scalar_noise}, express the estimand as a contrast.
#' @param selection A \code{LassoInference} or \code{ScreenedSelection}.
#' @param theta_hat Observed value of the estimator.
#' @param variance Variance of the estimator.
#' @param score_cov Covariance of \code{Z_full} with the estimator.
#' @param level Confidence level (default: the selection's level).
#' @param null_value Null value for the p-value.
#' @return A list as from \code{contrast_inference}: \code{estimate}, \code{lower_conf},
#'   \code{upper_conf}, \code{p_value}, \code{variance} and \code{contrast}.
#' @examplesIf requireNamespace("glmnet", quietly = TRUE)
#' set.seed(1)
#' n <- 100; p <- 10
#' x <- matrix(rnorm(n * p), n, p)
#' y <- x[, 1] - x[, 2] + rnorm(n)
#' # randomized response y + omega with Var(omega) = kappa, sigma^2 = 1 known
#' kappa <- 0.5
#' y_noisy <- y + rnorm(n, sd = sqrt(kappa))
#' Q <- crossprod(x); Z <- drop(crossprod(x, y))
#' lam <- 2 * sqrt(n)
#' fit <- glmnet::glmnet(x, y_noisy, lambda = lam / n, intercept = FALSE,
#'                       standardize = FALSE, control = list(thresh = 1e-14))
#' b <- as.numeric(coef(fit))[-1]
#' G <- drop(crossprod(x, x %*% b - y_noisy))
#' li <- LassoInference$new(b, G, Q, rep(lam, p), NULL, NULL, Z, Q, kappa * Q, level = 0.9)
#' # the estimator sum(eta * Z), specified by its variance and covariance with Z
#' eta <- li$contrasts[[1]]
#' custom_estimand_inference(li, sum(eta * Z), sum(eta * (Q %*% eta)),
#'                           drop(Q %*% eta))[c("lower_conf", "upper_conf", "p_value")]
#' @export
custom_estimand_inference <- function(selection, theta_hat, variance, score_cov, level = NULL, null_value = 0) {
  li <- .lasso_inference(selection)
  if (is.null(li$Sigma_noise)) {
    stop("estimands specified by covariance require Sigma_noise; with scalar_noise, ",
         "express the estimand as a contrast (see SelectionCoordinates)")
  }
  contrast <- li$si$compute_covariance_contrast(theta_hat, variance, as.numeric(score_cov))
  .inference(selection, contrast, variance, level, null_value)
}

#' Table of selective inference results
#'
#' @param selection A \code{LassoInference} or \code{ScreenedSelection}.
#' @param estimands Named list of \code{contrast_estimand}s, contrast vectors or
#'   \code{covariance_estimand}s.
#' @param level Confidence level (default: the selection's level).
#' @param null_value Null value for the p-values.
#' @return A data frame with columns \code{index}, \code{estimate}, \code{lower_conf},
#'   \code{upper_conf}, \code{p_value}.
#' @examplesIf requireNamespace("glmnet", quietly = TRUE)
#' set.seed(1)
#' n <- 100; p <- 10
#' x <- matrix(rnorm(n * p), n, p)
#' y <- x[, 1] - x[, 2] + rnorm(n)
#' fit <- glmnet::glmnet(x, y, lambda = c(0.3, 0.2), control = list(thresh = 1e-14))
#' li <- glmnet_inference(fit, x, y, level = 0.9)   # index 0 is the intercept
#' estimand_summary(li, list(x1 = li$contrasts[["1"]],
#'                            sum = li$contrasts[["1"]] + li$contrasts[["2"]]))
#' @export
estimand_summary <- function(selection, estimands, level = NULL, null_value = 0) {
  rows <- lapply(estimands, function(spec) {
    res <- if (inherits(spec, "CovarianceEstimand")) {
      custom_estimand_inference(selection, spec$theta_hat, spec$variance, spec$score_cov,
                                level = level, null_value = null_value)
    } else {
      contrast_inference(selection, spec, level = level, null_value = null_value)
    }
    c(estimate = res$estimate, lower_conf = res$lower_conf, upper_conf = res$upper_conf, p_value = res$p_value)
  })
  if (length(rows) == 0) {
    return(data.frame(index = character(), estimate = numeric(), lower_conf = numeric(),
                      upper_conf = numeric(), p_value = numeric()))
  }
  df <- cbind(index = names(estimands), as.data.frame(do.call(rbind, rows)), stringsAsFactors = FALSE)
  rownames(df) <- NULL
  df
}

#' Selective inference for inactive variables
#'
#' @param selection A \code{LassoInference} or \code{ScreenedSelection}.
#' @param estimand \code{"coef"} (coefficient of \code{j} in model \code{E u {j}}) or \code{"score"}.
#' @param basis \code{"refit"} or \code{"lasso"}.
#' @param variables Inactive variables (0-based); defaults to \code{screened} for a
#'   \code{ScreenedSelection} and all inactive variables otherwise.
#' @param level Confidence level (default: the selection's level).
#' @param null_value Null value for the p-values.
#' @param Q_diag Optional diagonal of \code{Q_hat}.
#' @return A data frame as from \code{estimand_summary}, with integer \code{index}.
#' @examplesIf requireNamespace("glmnet", quietly = TRUE)
#' set.seed(1)
#' n <- 100; p <- 10
#' x <- matrix(rnorm(n * p), n, p)
#' y <- x[, 1] - x[, 2] + rnorm(n)
#' fit <- glmnet::glmnet(x, y, lambda = c(0.3, 0.2), control = list(thresh = 1e-14))
#' li <- glmnet_inference(fit, x, y, level = 0.9)   # index 0 is the intercept
#' head(inactive_summary(li))
#' @export
inactive_summary <- function(selection, estimand = "coef", basis = "refit", variables = NULL,
                             level = NULL, null_value = 0, Q_diag = NULL) {
  coords <- SelectionCoordinates$new(selection, Q_diag = Q_diag)
  make <- switch(estimand,
                 coef = coords$inactive_coef,
                 score = coords$inactive_score,
                 stop("estimand must be 'coef' or 'score'"))
  if (is.null(variables)) {
    variables <- if (inherits(selection, "ScreenedSelection")) selection$screened else coords$E_c
  }
  if (estimand == "coef" && length(variables) > 0) coords$inactive_S(variables)
  estimands <- lapply(variables, function(j) make(j, basis = basis))
  names(estimands) <- variables
  df <- estimand_summary(selection, estimands, level = level, null_value = null_value)
  df$index <- as.integer(df$index)
  df
}
