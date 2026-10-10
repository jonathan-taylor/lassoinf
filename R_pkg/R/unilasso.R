# --- The truncation set along the conditioning line ---
#
# For the target of coordinate k, every constraint row reads, as a function of the scalar w
# the selection depends on (see docs/data_dependent_penalty.md),
#
#     u_i + a_i w <= b0_i + m_i / C(w),   C(w) = c + gamma w,   s C(w) > 0,
#
# with C = C_k, the univariate coefficient of the target, and s its sign. On the half-line
# s C(w) > 0, multiplying by C(w) gives a quadratic inequality, so each row holds on at most
# two intervals and the truncation set is a finite union of intervals. Mirrors
# python/lassoinf/unilasso.py.

# rows with |linear coefficient| below this are treated as constant, as in get_interval
.unilasso_zero_tol <- 1e-10

# {t : q2 t^2 + q1 t + q0 <= 0} for each row, as two intervals (columns lo1, hi1, lo2, hi2);
# a missing interval has lo > hi
.quadratic_le_zero <- function(q2, q1, q0) {
  m <- length(q2)
  out <- cbind(lo1 = rep(Inf, m), hi1 = rep(-Inf, m), lo2 = rep(Inf, m), hi2 = rep(-Inf, m))

  # linear rows: q1 t + q0 <= 0
  lin <- q2 == 0
  pos <- lin & q1 > .unilasso_zero_tol
  neg <- lin & q1 < -.unilasso_zero_tol
  feasible <- lin & !pos & !neg & q0 <= .unilasso_zero_tol
  root <- -q0 / q1
  out[pos, "lo1"] <- -Inf
  out[pos, "hi1"] <- root[pos]
  out[neg, "lo1"] <- root[neg]
  out[neg, "hi1"] <- Inf
  out[feasible, "lo1"] <- -Inf
  out[feasible, "hi1"] <- Inf

  # quadratic rows, roots in the numerically stable form
  quad <- !lin
  disc <- q1^2 - 4 * q2 * q0
  real <- quad & disc >= 0
  sq <- sqrt(pmax(disc, 0))
  w <- -0.5 * (q1 + ifelse(q1 >= 0, sq, -sq))
  ra <- w / q2
  rb <- ifelse(w != 0, q0 / w, ra)
  r1 <- pmin(ra, rb)
  r2 <- pmax(ra, rb)

  up <- real & q2 > 0        # between the roots
  out[up, "lo1"] <- r1[up]
  out[up, "hi1"] <- r2[up]

  down <- real & q2 < 0      # outside the roots
  out[down, "lo1"] <- -Inf
  out[down, "hi1"] <- r1[down]
  out[down, "lo2"] <- r2[down]
  out[down, "hi2"] <- Inf

  everywhere <- quad & !real & q2 < 0
  out[everywhere, "lo1"] <- -Inf
  out[everywhere, "hi1"] <- Inf
  out
}

# For each row i, the set of w with s (c + gamma w) > 0 and
# u_i + a_i w <= b0_i + m_i / (c + gamma w), as an m x 4 matrix of two intervals per row
# (lo1, hi1, lo2, hi2); a missing interval has lo > hi.
.unilasso_row_sets <- function(u, a, b0, m, c, gamma, s) {
  # the half-line where C(w) has sign s
  k <- s * gamma
  h <- -s * c
  if (k > 0) {
    lo_I <- h / k; hi_I <- Inf
  } else if (k < 0) {
    lo_I <- -Inf; hi_I <- h / k
  } else if (s * c > 0) {
    lo_I <- -Inf; hi_I <- Inf
  } else {
    lo_I <- Inf; hi_I <- -Inf
  }
  # on it, the row is s [C(w) (u_i - b0_i + a_i w) - m_i] <= 0
  v <- u - b0
  pieces <- .quadratic_le_zero(s * gamma * a, s * (c * a + gamma * v), s * (c * v - m))
  cbind(lo1 = pmax(pieces[, "lo1"], lo_I), hi1 = pmin(pieces[, "hi1"], hi_I),
        lo2 = pmax(pieces[, "lo2"], lo_I), hi2 = pmin(pieces[, "hi2"], hi_I))
}

# The intersection over rows of the unions from .unilasso_row_sets, as a k x 2 matrix of
# disjoint intervals in increasing order.
.intersect_row_sets <- function(sets) {
  m <- nrow(sets)
  if (m == 0) return(matrix(c(-Inf, Inf), 1, 2))
  lo <- c(sets[, "lo1"], sets[, "lo2"])
  hi <- c(sets[, "hi1"], sets[, "hi2"])
  keep <- lo < hi
  lo <- lo[keep]
  hi <- hi[keep]
  if (length(lo) == 0) return(matrix(numeric(0), 0, 2))

  # sweep over the endpoints counting covering rows (each row's pieces are disjoint);
  # at ties, closing before opening drops zero-length pieces
  pos <- c(lo, hi)
  delta <- c(rep(1, length(lo)), rep(-1, length(hi)))
  o <- order(pos, delta)
  pos <- pos[o]
  delta <- delta[o]
  count <- cumsum(delta)
  n <- length(pos)
  seg <- which(count[-n] == m & pos[-n] < pos[-1])
  iv <- cbind(pos[seg], pos[seg + 1])

  # merge touching segments
  if (nrow(iv) > 1) {
    merged <- iv[1, , drop = FALSE]
    for (r in 2:nrow(iv)) {
      last <- nrow(merged)
      if (iv[r, 1] <= merged[last, 2]) {
        merged[last, 2] <- max(merged[last, 2], iv[r, 2])
      } else {
        merged <- rbind(merged, iv[r, ])
      }
    }
    iv <- merged
  }
  unname(iv)
}

# --- uniLasso and unireg ---

#' uniLasso penalty
#'
#' @description The univariate coefficients, signs, penalty weights and bounds of the
#' uniLasso for selection data \code{Z_noisy}: \code{C = Z_noisy / diag(Q)},
#' \code{s = sign(C)}, \code{D = lam / |C|}, and bounds that restrict \code{beta_j} to
#' the sign \code{s_j}.
#' @param Z_noisy Selection data, \code{crossprod(x, y)} plus any randomization.
#' @param Q \code{crossprod(x)}.
#' @param lam The uniLasso's lambda, in the units of \code{Z_noisy}. \code{lam = 0} is unireg.
#' @return A list with components \code{C}, \code{s}, \code{D}, \code{L} and \code{U}.
#' @keywords internal
unilasso_penalty <- function(Z_noisy, Q, lam) {
  C <- as.vector(Z_noisy) / diag(Q)
  s <- sign(C)
  if (any(s == 0)) stop("a univariate coefficient is exactly zero")
  list(C = C, s = s, D = lam / abs(C),
       L = ifelse(s > 0, 0, -Inf), U = ifelse(s > 0, Inf, 0))
}

#' Fit the uniLasso
#'
#' @description Solves the uniLasso
#' \deqn{\min_\beta \frac{1}{2} \beta' Q \beta - \beta' Z + \lambda \sum_j |\beta_j| / |C_j|,
#' \quad \mathrm{sign}(\beta_j) \in \{0, \mathrm{sign}(C_j)\},}
#' with \eqn{C = Z / \mathrm{diag}(Q)} the univariate regression coefficients, by
#' coordinate descent. \code{lam = 0} is unireg: least squares with the sign constraints.
#' @param Q \code{crossprod(x)}.
#' @param Z_noisy Selection data, \code{crossprod(x, y)} plus any randomization.
#' @param lam The uniLasso's lambda, in the units of \code{Z_noisy}.
#' @param tol Convergence tolerance.
#' @param max_iter Maximum number of passes over the coordinates.
#' @return The solution, a numeric vector.
#' @examples
#' set.seed(1)
#' x <- matrix(rnorm(100 * 5), 100, 5)
#' y <- x[, 1] - 0.5 * x[, 2] + rnorm(100)
#' unilasso_fit(crossprod(x), drop(crossprod(x, y)), lam = 5)
#' @export
unilasso_fit <- function(Q, Z_noisy, lam, tol = 1e-12, max_iter = 100000) {
  Q <- as.matrix(Q)
  Y <- as.vector(Z_noisy)
  pen <- unilasso_penalty(Y, Q, lam)
  dq <- diag(Q)
  beta <- rep(0, length(Y))
  for (it in seq_len(max_iter)) {
    delta <- 0
    for (j in seq_along(Y)) {
      r <- Y[j] - sum(Q[j, ] * beta) + dq[j] * beta[j]
      new <- sign(r) * max(abs(r) - pen$D[j], 0) / dq[j]
      new <- min(max(new, pen$L[j]), pen$U[j])
      delta <- max(delta, abs(new - beta[j]) * sqrt(dq[j]))
      beta[j] <- new
    }
    if (delta < tol) break
  }
  beta
}

#' UniLassoInference Class
#'
#' @description Selective inference after the uniLasso, or unireg (\code{lam = 0}), for
#' the full-model coefficients of the selected variables, in a Gaussian linear model with
#' \eqn{n > p}. Mirrors the Python \code{lassoinf.UniLassoInference}.
#'
#' The uniLasso's penalty factors \eqn{1 / |\hat\beta^{uni}_j|} and sign constraints depend
#' on the data, so the selection event is not a polyhedron. For the full-model coefficient
#' \eqn{\theta_k = e_k^T Q^{-1} E[Z]} in the well-specified model
#' (\code{Sigma = sigma^2 Q}), only the target's own univariate coefficient moves along
#' the line the polyhedral lemma conditions on. Each constraint row is then a single ratio,
#' and the truncation set is a finite union of intervals.
#'
#' @examples
#' set.seed(1)
#' n <- 100; p <- 5
#' x <- matrix(rnorm(n * p), n, p)
#' y <- x[, 1] - 0.5 * x[, 2] + rnorm(n)
#' Q <- crossprod(x)
#' Z <- drop(crossprod(x, y))
#' kappa <- 1
#' Z_noisy <- Z + drop(t(chol(kappa * Q)) %*% rnorm(p))
#' beta_hat <- unilasso_fit(Q, Z_noisy, lam = 5)
#' ui <- UniLassoInference$new(beta_hat, Z_noisy, Q, lam = 5, Z_full = Z, Sigma = Q,
#'                             scalar_noise = kappa)
#' ui$summary()
#' @export
UniLassoInference <- R6::R6Class("UniLassoInference",
  public = list(
    #' @field beta_hat The uniLasso solution for the selection data.
    beta_hat = NULL,
    #' @field G_hat The gradient of the smooth part at beta_hat.
    G_hat = NULL,
    #' @field Q_hat \code{crossprod(x)}.
    Q_hat = NULL,
    #' @field lam The uniLasso's lambda.
    lam = NULL,
    #' @field C The univariate coefficients from the selection data.
    C = NULL,
    #' @field D The penalty weights \code{lam / |C|}.
    D = NULL,
    #' @field L Lower bounds from the sign constraints.
    L = NULL,
    #' @field U Upper bounds from the sign constraints.
    U = NULL,
    #' @field Z_full \code{crossprod(x, y)}.
    Z_full = NULL,
    #' @field Sigma Covariance of Z_full.
    Sigma = NULL,
    #' @field Sigma_noise Covariance of the randomization, or \code{NULL}.
    Sigma_noise = NULL,
    #' @field scalar_noise Scale of the randomization if \code{Sigma_noise} is \code{NULL}.
    scalar_noise = NaN,
    #' @field Z_noisy The selection data.
    Z_noisy = NULL,
    #' @field A The constraint operator of the selection event (a \code{LinearOp}).
    A = NULL,
    #' @field b The constraint vector at the observed penalties.
    b = NULL,
    #' @field E The selected variables (0-based, as in \code{LassoInference}).
    E = NULL,
    #' @field si AffineConstraints C++ wrapper instance.
    si = NULL,
    #' @field level Confidence level.
    level = 0.95,
    #' @field results For each selected variable, the estimate, interval, p-value and
    #'   truncation set (a matrix of intervals).
    results = NULL,

    #' @description
    #' Initialize the UniLassoInference object.
    #' @param beta_hat The uniLasso solution for \code{Z_noisy} (e.g. from \code{unilasso_fit}).
    #' @param Z_noisy Selection data, \code{crossprod(x, y)} plus the randomization.
    #' @param Q_hat \code{crossprod(x)}, invertible.
    #' @param lam The uniLasso's lambda, in the units of \code{Z_noisy}; 0 for unireg. A
    #'   vector gives each feature its own penalty \code{lam[j] / |C[j]|}.
    #' @param Z_full \code{crossprod(x, y)}.
    #' @param Sigma Covariance of \code{Z_full}: \code{sigma2 * crossprod(x)} in the
    #'   well-specified model.
    #' @param Sigma_noise Covariance of the randomization. Default is \code{NULL}.
    #' @param scalar_noise If \code{Sigma_noise} is \code{NULL}, the randomization covariance
    #'   is \code{scalar_noise * Sigma}, as for \code{LassoInference}. Default is \code{NaN}.
    #' @param tol Tolerance for the active set. Default is \code{1e-6}.
    #' @param level Confidence level. Default is 0.95.
    initialize = function(beta_hat, Z_noisy, Q_hat, lam, Z_full, Sigma, Sigma_noise = NULL,
                          scalar_noise = NaN, tol = 1e-6, level = 0.95) {
      Q_hat <- as.matrix(Q_hat)
      Z_noisy <- as.vector(Z_noisy)
      pen <- unilasso_penalty(Z_noisy, Q_hat, lam)
      self$Q_hat <- Q_hat
      self$lam <- lam
      self$C <- pen$C
      self$D <- pen$D
      self$L <- pen$L
      self$U <- pen$U
      self$Z_full <- as.vector(Z_full)
      self$Sigma <- Sigma
      self$Sigma_noise <- Sigma_noise
      no_randomization <- is.null(Sigma_noise) && isTRUE(scalar_noise == 0)
      if (is.null(Sigma_noise)) {
        if (!isTRUE(scalar_noise >= 0)) stop("if Sigma_noise is NULL, scalar_noise must be >= 0")
        if (scalar_noise > 0 && scalar_noise < 0.001) warning("for numerical stability using scalar_noise = 0.001")
        scalar_noise <- max(scalar_noise, 0.001)
      }
      self$scalar_noise <- scalar_noise
      self$level <- level
      p <- length(Z_noisy)

      # one proximal gradient step so that the KKT conditions hold exactly, as in
      # LassoInference
      G_hat <- drop(Q_hat %*% beta_hat) - Z_noisy
      lambda_max <- max(eigen(Q_hat, symmetric = TRUE, only.values = TRUE)$values)
      step_size <- 1.0 / (20.0 * lambda_max)
      beta_new <- prox_lasso_bounds(beta_hat - step_size * G_hat, step_size, self$D, self$L, self$U)
      self$G_hat <- G_hat + (beta_new - beta_hat) / step_size
      self$beta_hat <- beta_new

      cons <- lasso_post_selection_constraints(self$beta_hat, self$G_hat, Q_hat, self$D,
                                               self$L, self$U, tol)
      self$A <- cons$A
      self$b <- cons$b
      self$E <- cons$E
      self$Z_noisy <- -self$G_hat + drop(Q_hat %*% self$beta_hat)
      # without randomization the selection score is the data (the polyhedral case, as in
      # glmnet_inference)
      if (no_randomization) self$Z_full <- self$Z_noisy
      self$si <- new(AffineConstraints, self$Z_full, self$Z_noisy, self$Sigma, self$Sigma_noise,
                     self$scalar_noise)

      self$results <- list()
      for (k0 in self$E) {
        k <- k0 + 1
        e_k <- replace(rep(0, p), k, 1)

        # b is affine in D: its coefficient on D_k (any step works; D_k = 0 for unireg)
        step <- if (self$D[k] > 0) self$D[k] else 1
        b_step <- lasso_post_selection_constraints(self$beta_hat, self$G_hat, Q_hat,
                                                   self$D + step * e_k, self$L, self$U, tol)$b
        M_k <- (b_step - self$b) / step
        b0 <- self$b - M_k * self$D[k]
        s_k <- sign(self$C[k])
        lam_k <- if (length(lam) == 1) lam else lam[k]
        # C_k = Z_k / Q_kk, so Cov(C_k, Z) = Sigma[k, ] / Q_kk
        Sigma_k <- matvec(self$Sigma, e_k) / Q_hat[k, k]

        # full-model target theta_k = e_k' Q^{-1} E[Z]
        eta <- solve(Q_hat, e_k)
        contrast <- self$si$compute_contrast(eta)
        # every row: (A Y)_i <= b0_i + M_ik lam_k s_k / C_k, along the line in w; at
        # theta_hat = 0, A Y = A (N_o + bar_N_o) + A bar_Gamma w, and C_k is affine in w with
        # slope Cov(C_k, theta_hat) / bar_s^2
        bar_s2 <- contrast$bar_s^2
        w_obs <- bar_s2 / contrast$naive_variance * contrast$theta_hat + contrast$bar_theta
        gamma <- sum(Sigma_k * eta) / bar_s2
        S <- .intersect_row_sets(.unilasso_row_sets(
          u = matvec(self$A, contrast$n_o + contrast$bar_n_o),
          a = matvec(self$A, contrast$bar_gamma),
          b0 = b0, m = M_k * lam_k * s_k,
          c = self$C[k] - gamma * w_obs, gamma = gamma, s = s_k))
        if (nrow(S) == 0) stop("the observed data do not satisfy the selection event")

        variance <- contrast$naive_variance
        bar_s <- contrast$bar_s
        theta_hat <- contrast$theta_hat
        tbn <- TruncBivariateNormal$new(a_coeff = bar_s^2 / variance, b_coeff = 1,
                                        L = S[, 1], U = S[, 2],
                                        sig_omega = bar_s, sig_x = sqrt(variance))
        L_U <- tbn$equal_tailed_interval(theta_hat, alpha = 1 - level)
        cdf0 <- min(max(tbn$cdf(theta = 0, x = theta_hat), 0), 1)
        self$results[[as.character(k0)]] <- list(
          beta_hat = theta_hat,
          lower_conf = L_U[1] * variance,
          upper_conf = L_U[2] * variance,
          p_value = min(max(2 * min(cdf0, 1 - cdf0), 0), 1),
          intervals = S,
          family = tbn,
          variance = variance)
      }
    },

    #' @description
    #' The pivot \eqn{P(\hat\theta_k' \le \hat\theta_k \mid \mathrm{selection})} at a value of
    #' the target, uniform when \code{theta} is the true coefficient.
    #' @param k A selected variable (0-based).
    #' @param theta The value of the full-model coefficient.
    #' @return The pivot.
    pivot = function(k, theta) {
      r <- self$results[[as.character(k)]]
      r$family$cdf(theta = theta / r$variance, x = r$beta_hat)
    },

    #' @description
    #' Summary of post-selection inference results.
    #' @return A data frame with the selected variables (0-based \code{index}), the
    #'   full-model least squares coefficients (\code{beta_hat}), their selective
    #'   confidence intervals and p-values.
    summary = function() {
      if (length(self$E) == 0) {
        return(data.frame(index = integer(), beta_hat = numeric(), lower_conf = numeric(),
                          upper_conf = numeric(), p_value = numeric()))
      }
      field <- function(name) sapply(self$results, function(r) r[[name]])
      data.frame(index = as.integer(names(self$results)),
                 beta_hat = field("beta_hat"),
                 lower_conf = field("lower_conf"),
                 upper_conf = field("upper_conf"),
                 p_value = field("p_value"),
                 row.names = NULL)
    }
  )
)

#' Selective inference after a uniLasso fit
#'
#' @description Selective inference after a \code{uniLasso} (or \code{uniReg}) fit from the
#' \pkg{uniLasso} package, for the full-model coefficients of the selected variables, in a
#' Gaussian linear model with \eqn{n > p}.
#'
#' With \code{loo = FALSE} the uniLasso solves
#' \deqn{\min_{\beta_0, \beta} \frac{1}{2n} \|y - \beta_0 - X\beta\|_2^2 +
#' \lambda \sum_j \frac{|\beta_j|}{|\hat\beta^{uni}_j|}, \quad
#' \mathrm{sign}(\beta_j) \in \{0, \mathrm{sign}(\hat\beta^{uni}_j)\},}
#' with \eqn{\hat\beta^{uni}_j} the univariate regression slopes (with intercepts). Its
#' penalty factors and sign constraints depend on the data. For the full-model coefficients,
#' the truncation set is still found exactly, as a finite union of intervals, and the
#' inference is exact in the Gaussian linear model.
#'
#' With \code{loo = TRUE} (the uniLasso package's default), the uniLasso regresses \code{y} on
#' leave-one-out univariate fits instead. Its solution solves, exactly, the problem above with
#' the penalty \eqn{n\lambda} on feature \eqn{j} replaced by \eqn{n\lambda + \kappa_j}, for an
#' explicit \eqn{\kappa_j \approx 2\sigma^2} computed from the data and the fit (see
#' \code{vignette("unilasso_loo")}). The inference then uses these penalties. Because
#' \eqn{\kappa_j} itself depends on the data, which the inference ignores, it is
#' approximate, and a warning says so.
#'
#' There is no randomization: this is the polyhedral approach of Lee et al. (2016).
#' @param fit A \code{uniLasso} or \code{uniReg} object, fitted with the default
#'   \code{lower.limits}, \code{standardize} and family, and no weights.
#' @param x,y The data the fit used.
#' @param s The value of \code{lambda} for the inference; it must be one of
#'   \code{fit$lambda}. Default is the only one, for a \code{uniReg} fit.
#' @param sigma2 Noise variance. Default is the residual variance of the full least
#'   squares fit.
#' @param level Confidence level. Default is 0.95.
#' @param kkt_tol Warn if the fit violates the KKT conditions by more than
#'   \code{kkt_tol} times the penalty. Default is \code{1e-3}.
#' @return A \code{UniLassoInference} object; its \code{summary()} has the full-model least
#'   squares coefficients of the selected variables, with selective intervals and p-values.
#' @references
#' Chatterjee, S., Hastie, T. and Tibshirani, R. (2025). Univariate-guided sparse
#' regression. \emph{arXiv preprint}. \doi{10.48550/arXiv.2501.18360}
#'
#' Lee, J. D., Sun, D. L., Sun, Y. and Taylor, J. E. (2016). Exact post-selection
#' inference, with application to the lasso. \emph{The Annals of Statistics}, 44(3),
#' 907--927. \doi{10.1214/15-AOS1371}
#' @examplesIf requireNamespace("uniLasso", quietly = TRUE)
#' set.seed(1)
#' n <- 200; p <- 8
#' x <- matrix(rnorm(n * p), n, p)
#' y <- drop(x %*% c(0.4, -0.3, 0.2, rep(0, p - 3))) + rnorm(n)
#' fit <- uniLasso::uniLasso(x, y, loo = FALSE, control = list(thresh = 1e-14))
#' unilasso_inference(fit, x, y, s = fit$lambda[20], level = 0.9)$summary()
#' # the default loo = TRUE: approximate, with a warning
#' fit_loo <- uniLasso::uniLasso(x, y, control = list(thresh = 1e-14))
#' unilasso_inference(fit_loo, x, y, s = fit_loo$lambda[20], level = 0.9)$summary()
#' @export
unilasso_inference <- function(fit, x, y, s = NULL, sigma2 = NULL, level = 0.95, kkt_tol = 1e-3) {
  if (!inherits(fit, "uniLasso")) stop("fit must be a uniLasso or uniReg object")
  arg <- function(name, default) {
    v <- fit$call[[name]]
    if (is.null(v)) default else eval(v, parent.frame(2))
  }
  loo <- arg("loo", TRUE)
  if (!isTRUE(loo) && !isFALSE(loo)) stop("could not determine loo from the fit")
  if (!identical(arg("family", "gaussian"), "gaussian")) stop("only the gaussian family is supported")
  if (!isTRUE(all(arg("lower.limits", 0) == 0))) stop("the fit must use the default lower.limits = 0")
  if (!isFALSE(arg("standardize", FALSE))) stop("the fit must use standardize = FALSE")
  if (!is.null(fit$call$weights)) stop("weights are not supported")
  if (!isTRUE(arg("intercept", TRUE))) stop("the fit must have an intercept")

  lambdas <- fit$lambda
  if (is.null(s)) {
    if (length(lambdas) != 1) stop("s must be given: the fit has a path of lambda values")
    k <- 1
  } else {
    k <- which(abs(lambdas - s) <= 1e-10 * max(abs(s), 1e-300))
    if (length(k) == 0) stop("s must be one of fit$lambda")
    k <- k[1]
  }
  unilasso_inference_beta(x, y, as.numeric(fit$beta[, k]), lambdas[k], loo = loo,
                          sigma2 = sigma2, level = level, kkt_tol = kkt_tol)
}

# kappa_j of vignette("unilasso_loo"): the leave-one-out uniLasso solution beta_hat solves
# the plain uniLasso with penalty (n lambda + kappa_j) / |b_j| on feature j. x and y are
# centered; the univariate fits have intercepts.
unilasso_loo_kappa <- function(x, y, beta_hat) {
  n <- nrow(x)
  S <- colSums(x^2)
  b <- drop(crossprod(x, y)) / S
  h <- 1 / n + sweep(x^2, 2, S, "/")                       # leverages of the univariate fits
  delta <- -h / (1 - h) * (y - sweep(x, 2, b, "*"))         # leave-one-out fits minus fits
  delta <- sweep(delta, 2, colMeans(delta))
  theta <- beta_hat / b
  r <- y - drop(x %*% beta_hat)
  -drop(crossprod(delta, r)) + b * drop(crossprod(x, delta %*% theta)) +
    drop(crossprod(delta, delta %*% theta))
}

# The same from the coefficients of a fit at lam (glmnet's scaling) on (x, y), with intercept;
# mirrors the Python lassoinf.unilasso_inference.
unilasso_inference_beta <- function(x, y, beta_hat, lam, loo = FALSE, sigma2 = NULL,
                                    level = 0.95, kkt_tol = 1e-3) {
  x <- as.matrix(x)
  y <- as.vector(y)
  n <- nrow(x)
  p <- ncol(x)
  df <- n - p - 1
  if (df <= 0) stop("unilasso_inference requires n > p")
  x <- sweep(x, 2, colMeans(x))
  y <- y - mean(y)
  Q <- crossprod(x)
  Z <- drop(crossprod(x, y))
  lam_Z <- n * lam
  if (loo) {
    # the leave-one-out uniLasso is the plain one with penalty n lam + kappa_j on feature j
    # (it can be negative for a strong feature at small n; the identity still holds)
    lam_Z <- lam_Z + unilasso_loo_kappa(x, y, beta_hat)
    warning("loo = TRUE: the fit is treated as the uniLasso with penalties ",
            "(n * lambda + kappa_j) / |b_uni_j|, which it solves exactly; kappa_j depends on ",
            "the data, which the inference ignores, so it is approximate. ",
            "See vignette(\"unilasso_loo\").", call. = FALSE)
  }
  pen <- unilasso_penalty(Z, Q, lam_Z)
  if (any(beta_hat * pen$s < 0)) {
    stop("beta_hat has a sign opposite to its univariate coefficient: it is not a uniLasso ",
         "fit for these data")
  }
  g <- Z - drop(Q %*% beta_hat)    # minus the gradient
  act <- beta_hat != 0
  violation <- max(c(abs(g[act] - pen$D[act] * pen$s[act]),
                     pmax(pen$s[!act] * g[!act] - pen$D[!act], 0)), 0)
  scale <- if (max(abs(lam_Z)) > 0) mean(abs(lam_Z)) else max(abs(Z))
  if (violation > kkt_tol * scale) {
    warning(sprintf(paste0("the fit violates the uniLasso KKT conditions by %.1e * %s; it may ",
                           "not solve the uniLasso at this lambda, or may not have converged ",
                           "(tighten thresh)"),
                    violation / scale, if (max(abs(lam_Z)) > 0) "the penalty" else "max |X'y|"))
  }
  if (is.null(sigma2)) {
    resid <- y - drop(x %*% solve(Q, Z))
    sigma2 <- sum(resid^2) / df
  }
  UniLassoInference$new(beta_hat, Z, Q, lam = lam_Z, Z_full = Z, Sigma = sigma2 * Q,
                        scalar_noise = 0, level = level)
}
