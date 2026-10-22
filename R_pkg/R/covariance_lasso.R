# Selective inference after the lasso in covariance form, from an estimate and its
# covariance. See python/lassoinf/covariance_lasso.py.

#' Fit the lasso with box constraints
#'
#' @description Solves
#' \deqn{\min_b \frac{1}{2} b' Q b - b' Y + \sum_j D_j |b_j|, \quad L \le b \le U,}
#' by coordinate descent. \code{L[j] = U[j] = 0} excludes coordinate \code{j} and
#' \code{D[j] = 0} leaves it unpenalized. With \code{Q = solve(Sigma)} and
#' \code{Y = Q \%*\% beta_star} this is the lasso in covariance form.
#' @param Q A positive definite matrix.
#' @param Y The linear term.
#' @param D Penalty weights, \code{>= 0}.
#' @param L,U Bounds; default \code{-Inf} and \code{Inf}.
#' @param beta0 A starting value; default 0.
#' @param tol Convergence tolerance, in the units of \code{sqrt(diag(Q))}.
#' @param max_iter Maximum number of passes over the coordinates.
#' @return The solution, a numeric vector.
#' @examples
#' Q <- diag(3)
#' box_lasso_fit(Q, c(3, -0.5, 2), D = rep(1, 3), U = c(Inf, Inf, 0.5))
#' @export
box_lasso_fit <- function(Q, Y, D, L = NULL, U = NULL, beta0 = NULL, tol = 1e-12,
                          max_iter = 100000) {
  p <- length(Y)
  if (is.null(L)) L <- rep(-Inf, p)
  if (is.null(U)) U <- rep(Inf, p)
  if (is.null(beta0)) beta0 <- numeric(0)
  box_lasso_fit_cpp(as.matrix(Q), as.numeric(Y), as.numeric(D), as.numeric(L),
                    as.numeric(U), as.numeric(beta0), tol, as.integer(max_iter))
}

#' Expected maximum of standardized Gaussians
#'
#' @description Monte Carlo estimate of \eqn{E \|\xi\|_\infty}, \eqn{\xi \sim N(0, R)},
#' with \eqn{R} the correlation matrix of \code{cov}.
#' @param cov A covariance matrix.
#' @param n_mc Number of Monte Carlo draws.
#' @return A number.
#' @keywords internal
expected_max_abs <- function(cov, n_mc = 4000) {
  cov <- as.matrix(cov)
  if (nrow(cov) == 0) return(0)
  R <- stats::cov2cor(cov)
  xi <- matrix(rnorm(n_mc * nrow(R)), n_mc) %*% chol(R)
  mean(apply(abs(xi), 1, max))
}

#' Selective MLE
#'
#' @description The selective MLE of a target from the \code{TruncBivariateNormal} law
#' of its estimate given the selection, and its standard error from the conditional
#' Fisher information, both on the scale of the target. It solves
#' \eqn{E_\theta[\hat\theta \mid \mathrm{selection}] = } \code{observed}.
#' @param family A \code{TruncBivariateNormal} with natural parameter the target divided
#'   by \code{variance}.
#' @param observed The estimate of the target.
#' @param variance Its variance.
#' @param width Half-width of the search, in standard errors.
#' @return A named vector: \code{mle} and \code{std_err}.
#' @keywords internal
selective_mle <- function(family, observed, variance, width = 40) {
  sd <- sqrt(variance)
  gap <- function(theta) {
    m <- family$E(theta, function(x) x)
    if (is.null(m) || !is.finite(m)) NaN else m - observed
  }
  grid <- observed / variance + seq(-width, width, length.out = 161) / sd
  values <- vapply(grid, gap, numeric(1))
  ok <- is.finite(values)
  grid <- grid[ok]
  values <- values[ok]
  change <- which(sign(values[-length(values)]) != sign(values[-1]))
  if (length(change) == 0) return(c(mle = NaN, std_err = NaN))
  k <- change[1]
  theta <- uniroot(gap, c(grid[k], grid[k + 1]), tol = 1e-10 / sd)$root
  info <- family$Var(theta, function(x) x)
  c(mle = theta * variance, std_err = variance / sqrt(info))
}

.truncated_family <- function(inference, v, variance) {
  contrast <- inference$si$compute_contrast(v)
  interval <- contrast$get_interval(0.0, inference$A, inference$b)
  bar_s <- contrast$bar_s
  family <- TruncBivariateNormal$new(a_coeff = bar_s^2 / variance, b_coeff = 1,
                                     L = interval[[1]], U = interval[[2]],
                                     sig_omega = bar_s, sig_x = sqrt(variance))
  list(family = family, contrast = contrast)
}

# The covariance (times pi) of the score of the penalized coefficients P at b_P = 0, with the
# unpenalized N profiled out: the Schur complement of N in Q, restricted to P.
.penalized_score_cov <- function(Q, P, N) {
  M <- Q[P, P, drop = FALSE]
  if (length(N) > 0) {
    M <- M - Q[P, N, drop = FALSE] %*% solve(Q[N, N, drop = FALSE], Q[N, P, drop = FALSE])
  }
  (M + t(M)) / 2
}

.positions <- function(spec, names, p) {
  if (is.null(spec)) return(integer(0))
  if (is.logical(spec)) {
    if (length(spec) != p) stop("a logical exclude must have one entry per coefficient")
    return(which(spec))
  }
  if (is.character(spec)) {
    pos <- match(spec, names)
    if (anyNA(pos)) stop("exclude has names not in beta_hat")
    return(pos)
  }
  as.integer(spec)
}

#' Selective inference after the lasso in covariance form
#'
#' @description Given an asymptotically Gaussian estimate
#' \eqn{\hat\beta \sim N(\beta, \Sigma)}, such as a GLM's MLE and its estimated
#' covariance, selects coefficients with the randomized lasso in covariance form
#' \deqn{\min_b \frac{1}{2}(\hat\beta^* - b)' \Sigma^{-1} (\hat\beta^* - b) + \sum_j D_j |b_j|,}
#' with \eqn{\hat\beta^* = \hat\beta + \omega} and
#' \eqn{\omega \sim N(0, \frac{1-\pi}{\pi}\Sigma)}, which mimics selecting on a random
#' fraction \eqn{\pi} (\code{prop_select}) of the data. It then returns p-values,
#' confidence intervals and selective MLEs for the selected coefficients, conditional on
#' the selection.
#'
#' The penalty standardizes the score \eqn{S = \Sigma^{-1}\hat\beta^*}. With the
#' unpenalized coefficients profiled out and the excluded ones removed, the score of the
#' penalized coefficients has covariance \eqn{M/\pi}, \eqn{M} the Schur complement of
#' the unpenalized coefficients in \eqn{\Sigma^{-1}}, and
#' \eqn{D_j = f_j t \sqrt{M_{jj}/\pi}}, with \eqn{f_j} the penalty factor. Nothing is
#' selected exactly when \eqn{\max_j |\xi_j| / f_j \le t} for the standardized score
#' \eqn{\xi_j = S_j / \sqrt{M_{jj}/\pi}}, the analog of glmnet's standardized
#' \eqn{X'y} (in least squares, the marginal z-statistics). By default \eqn{t} is
#' \code{lambda_mult} times \eqn{E\|\xi\|_\infty} under the global null,
#' \eqn{\xi \sim N(0, \mathrm{corr}(M))}: a Bonferroni-like threshold. For uncorrelated
#' estimates, \eqn{j} is selected when \eqn{|\hat\beta^*_j| / SE(\hat\beta^*_j) > f_j t}.
#' @param beta_hat The estimate, a (named) numeric vector. Leave the intercept out.
#' @param cov_beta Its covariance matrix.
#' @param prop_select The fraction \eqn{\pi} of the data whose selection is mimicked.
#' @param exclude Coefficients fixed at 0 (names, positions or a logical vector), as
#'   in \code{glmnet}.
#' @param penalty_factor Multiplies each coefficient's penalty; 0 leaves it unpenalized.
#' @param lambda_mult Multiplies \eqn{E\|\xi\|_\infty}, for \eqn{\xi} the standardized
#'   score under the global null, to give the threshold.
#' @param threshold The threshold \eqn{t} itself, overriding \code{lambda_mult}.
#' @param perturbation A draw with covariance \code{cov_beta}, independent of
#'   \code{beta_hat} given the data (e.g. a bootstrap replicate minus the bootstrap
#'   mean). It is multiplied by \eqn{\sqrt{(1-\pi)/\pi}} to form \eqn{\omega}. By default
#'   \eqn{\omega} is drawn from a Gaussian.
#' @param target \code{"selected"}: the coefficients of the model of the selected
#'   coefficients, \eqn{(\Sigma^{-1})_{EE}^{-1}(\Sigma^{-1}\beta)_E}.
#'   \code{"full"}: \eqn{\beta_j} itself.
#' @param level Confidence level.
#' @param n_mc Monte Carlo draws for \eqn{E\|\xi\|_\infty}.
#' @return A list with \code{summary} (for each selected coefficient: the unadjusted
#'   \code{estimate} and \code{std_err}, the selective \code{mle} and
#'   \code{mle_std_err}, \code{lower_conf}, \code{upper_conf} and \code{p_value}),
#'   \code{splitting} (the estimate independent of the selection, with
#'   \eqn{1/(1-\pi)} times the variance, and its interval), \code{selected},
#'   \code{lasso_coef}, \code{beta_star}, \code{threshold}, \code{D} and the
#'   \code{LassoInference} object \code{inference}.
#' @examples
#' set.seed(1)
#' p <- 10
#' Sigma <- 0.5^abs(outer(1:p, 1:p, "-"))
#' beta_hat <- c(4, -3, rep(0, p - 2)) + drop(rnorm(p) %*% chol(Sigma))
#' names(beta_hat) <- paste0("x", 1:p)
#' res <- covariance_lasso_inference(beta_hat, Sigma, level = 0.9)
#' res$summary
#' @export
covariance_lasso_inference <- function(beta_hat, cov_beta, prop_select = 0.8, exclude = NULL,
                                       penalty_factor = NULL, lambda_mult = 1, threshold = NULL,
                                       perturbation = NULL, target = c("selected", "full"),
                                       level = 0.95, n_mc = 4000) {
  target <- match.arg(target)
  pi_sel <- prop_select
  if (!(pi_sel > 0 && pi_sel < 1)) stop("prop_select must be in (0, 1)")
  p <- length(beta_hat)
  names <- if (is.null(names(beta_hat))) as.character(seq_len(p)) else names(beta_hat)
  beta_hat <- as.numeric(beta_hat)
  Sigma <- as.matrix(cov_beta)
  if (!all(dim(Sigma) == p)) stop("cov_beta must be p x p")
  Sigma <- (Sigma + t(Sigma)) / 2
  dimnames(Sigma) <- NULL

  excluded <- rep(FALSE, p)
  excluded[.positions(exclude, names, p)] <- TRUE
  pf <- if (is.null(penalty_factor)) rep(1, p) else as.numeric(penalty_factor)
  if (length(pf) != p || any(pf < 0)) stop("penalty_factor must have p entries >= 0")

  Q <- solve(Sigma)
  Q <- (Q + t(Q)) / 2
  P <- which(pf > 0 & !excluded)
  M <- .penalized_score_cov(Q, P, which(pf == 0 & !excluded))
  if (is.null(threshold)) threshold <- lambda_mult * expected_max_abs(M, n_mc)

  if (is.null(perturbation)) perturbation <- drop(rnorm(p) %*% chol(Sigma))
  perturbation <- as.numeric(perturbation)
  if (length(perturbation) != p) stop("perturbation must have p entries")
  beta_star <- beta_hat + sqrt((1 - pi_sel) / pi_sel) * perturbation

  Z_full <- drop(Q %*% beta_hat)
  Y <- drop(Q %*% beta_star)
  # the standardized score of the penalized coefficients is S_j / sqrt(M_jj / pi)
  D <- rep(0, p)
  D[P] <- pf[P] * threshold * sqrt(diag(M) / pi_sel)
  L <- ifelse(excluded, 0, -Inf)
  U <- ifelse(excluded, 0, Inf)

  b <- box_lasso_fit(Q, Y, D, L, U)
  G <- drop(Q %*% b) - Y
  inference <- LassoInference$new(beta_hat = b, G_hat = G, Q_hat = Q, D = D, L = L, U = U,
                                  Z_full = Z_full, Sigma = Q,
                                  scalar_noise = (1 - pi_sel) / pi_sel, level = level)

  E <- inference$E + 1
  q <- qnorm(1 - (1 - level) / 2)
  rows <- lapply(seq_along(E), function(k) {
    j <- E[k]
    if (target == "selected") {
      v <- rep(0, p)
      v[E] <- inference$W[, k]
    } else {
      v <- Sigma[, j]
    }
    theta_hat <- sum(v * Z_full)
    variance <- drop(crossprod(v, Q %*% v))
    tf <- .truncated_family(inference, v, variance)
    if (target == "selected") {
      ci <- inference$intervals[[as.character(j - 1)]]
      lower <- ci[[1]]
      upper <- ci[[2]]
      p_value <- ci[[3]]
    } else {
      nat <- tf$family$equal_tailed_interval(theta_hat, alpha = 1 - level)
      lower <- nat[1] * variance
      upper <- nat[2] * variance
      cdf <- min(max(tf$family$cdf(theta = 0, x = theta_hat), 0), 1)
      p_value <- min(max(2 * min(cdf, 1 - cdf), 0), 1)
    }
    mle <- selective_mle(tf$family, theta_hat, variance)
    split_est <- tf$contrast$splitting_estimator
    split_se <- sqrt(tf$contrast$splitting_variance)
    list(summary = c(estimate = theta_hat, std_err = sqrt(variance), mle = mle[["mle"]],
                     mle_std_err = mle[["std_err"]], lower_conf = lower, upper_conf = upper,
                     p_value = p_value),
         splitting = c(estimate = split_est, std_err = split_se,
                       lower_conf = split_est - q * split_se,
                       upper_conf = split_est + q * split_se,
                       p_value = 2 * pnorm(abs(split_est) / split_se, lower.tail = FALSE)))
  })
  frame <- function(part, cols) {
    m <- matrix(NA_real_, length(E), length(cols), dimnames = list(names[E], cols))
    for (k in seq_along(E)) m[k, ] <- rows[[k]][[part]][cols]
    as.data.frame(m)
  }
  summary <- frame("summary", c("estimate", "std_err", "mle", "mle_std_err", "lower_conf",
                                "upper_conf", "p_value"))
  splitting <- frame("splitting", c("estimate", "std_err", "lower_conf", "upper_conf",
                                    "p_value"))
  list(summary = summary,
       splitting = splitting,
       selected = names[E],
       lasso_coef = stats::setNames(inference$beta_hat, names),
       beta_star = stats::setNames(beta_star, names),
       threshold = threshold,
       D = D,
       inference = inference,
       target = target)
}
