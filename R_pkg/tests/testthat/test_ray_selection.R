weighted_lasso <- function(Q, Y, D, tol = 1e-13, max_iter = 100000) {
  # minimize 1/2 b'Qb - b'Y + sum_j D_j |b_j| by coordinate descent
  b <- rep(0, length(Y))
  for (it in seq_len(max_iter)) {
    delta <- 0
    for (j in seq_along(Y)) {
      r <- Y[j] - sum(Q[j, ] * b) + Q[j, j] * b[j]
      new <- sign(r) * max(abs(r) - D[j], 0) / Q[j, j]
      delta <- max(delta, abs(new - b[j]) * sqrt(Q[j, j]))
      b[j] <- new
    }
    if (delta < tol) break
  }
  b
}

design <- function(n, p, rho = 0.5) {
  matrix(rnorm(n * p), n, p) %*% chol((1 - rho) * diag(p) + rho)
}

ray_w <- function(contrast) {
  bar_s2 <- contrast$bar_s^2
  c(w = bar_s2 / contrast$naive_variance * contrast$theta_hat + contrast$bar_theta,
    sd = sqrt(bar_s2^2 / contrast$naive_variance + bar_s2))
}

test_that("with fixed penalties the truncation set is the polyhedral lemma's interval", {
  set.seed(0)
  n <- 100; p <- 6
  x <- design(n, p)
  Q <- crossprod(x)
  Z <- drop(crossprod(x, x[, 1:2] %*% c(0.3, -0.2) + rnorm(n)))
  Y <- Z + drop(t(chol(0.5 * Q)) %*% rnorm(p))
  D <- rep(15, p)
  b <- weighted_lasso(Q, Y, D)
  G <- drop(Q %*% b) - Y
  cons <- lassoinf:::lasso_post_selection_constraints(b, G, Q, D, rep(-Inf, p), rep(Inf, p), 1e-6)
  expect_gt(length(cons$E), 0)
  si <- new(AffineConstraints, Z, Y, Q, NULL, 0.5)
  event <- lasso_event(b, G, Q, function(Y) list(D, NULL, NULL))
  for (k in cons$E + 1) {
    contrast <- si$compute_contrast(solve(Q, replace(rep(0, p), k, 1)))
    S <- find_truncation_set(event, contrast)
    expect_equal(dim(S), c(1L, 2L))
    expect_equal(drop(S), contrast$get_interval(0, cons$A, cons$b), tolerance = 1e-7)
  }
})

test_that("rescaling the response rescales the truncation sets", {
  # the ray's patterns use the same scale-relative tolerance as the constraints; finite
  # bounds, not binding, add rows for the active coordinates that the pattern must find
  set.seed(0)
  n <- 100; p <- 6
  x <- design(n, p)
  Q <- crossprod(x)
  Z <- drop(crossprod(x, x[, 1:2] %*% c(0.3, -0.2) + rnorm(n)))
  Y <- Z + drop(t(chol(0.5 * Q)) %*% rnorm(p))
  D <- rep(15, p)
  b <- weighted_lasso(Q, Y, D)
  G <- drop(Q %*% b) - Y
  for (k in c(1e-8, 1e8)) {
    for (bound in c(Inf, 100)) {
      L <- rep(-k * bound, p); U <- rep(k * bound, p)
      cons <- lassoinf:::lasso_post_selection_constraints(k * b, k * G, Q, k * D, L, U, 1e-6)
      expect_gt(length(cons$E), 0)
      si <- new(AffineConstraints, k * Z, k * Y, k^2 * Q, NULL, 0.5)
      event <- lasso_event(k * b, k * G, Q, function(Y) list(k * D, L, U))
      for (j in cons$E + 1) {
        contrast <- si$compute_contrast(solve(Q, replace(rep(0, p), j, 1)))
        S <- find_truncation_set(event, contrast)
        expect_equal(drop(S), contrast$get_interval(0, cons$A, cons$b), tolerance = 1e-6)
      }
    }
  }
})

test_that("an R function event and the C++ event agree", {
  set.seed(1)
  n <- 100; p <- 5
  x <- design(n, p)
  Q <- crossprod(x)
  Z <- drop(crossprod(x, x[, 1:2] %*% c(0.4, -0.3) + rnorm(n)))
  Y <- Z + drop(t(chol(Q)) %*% rnorm(p))
  penalty <- function(Y) list(0.2 / abs(solve(Q, Y)), NULL, NULL)
  b <- weighted_lasso(Q, Y, penalty(Y)[[1]])
  G <- drop(Q %*% b) - Y
  expect_gt(sum(b != 0), 0)
  event <- lasso_event(b, G, Q, penalty)
  r_event <- function(Y) event$call(Y)
  si <- new(AffineConstraints, Z, Y, Q, NULL, 1)
  for (k in which(b != 0)) {
    contrast <- si$compute_contrast(solve(Q, replace(rep(0, p), k, 1)))
    expect_equal(find_truncation_set(r_event, contrast, n_grid = 401),
                 find_truncation_set(event, contrast, n_grid = 401))
  }
  expect_error(find_truncation_set(function(Y) FALSE, contrast), "not in the selection event")
  expect_error(find_truncation_set(function(Y) stop("broken event"), contrast), "broken event")
})

test_that("the ray finder matches the uniLasso's closed form", {
  set.seed(2)
  checked <- 0
  while (checked < 8) {
    n <- 100; p <- 6
    x <- design(n, p)
    Q <- crossprod(x)
    Z <- drop(crossprod(x, x %*% c(0.3, -0.25, 0.2, 0, 0, 0) + rnorm(n)))
    Y <- Z + drop(t(chol(0.5 * Q)) %*% rnorm(p))
    lam <- 4
    ui <- UniLassoInference$new(unilasso_fit(Q, Y, lam), Y, Q, lam = lam, Z_full = Z, Sigma = Q,
                                scalar_noise = 0.5)
    s_obs <- sign(ui$C)
    penalty <- function(Y) {
      pen <- lassoinf:::unilasso_penalty(Y, Q, lam)
      if (all(pen$s == s_obs)) list(pen$D, pen$L, pen$U) else NULL
    }
    event <- lasso_event(ui$beta_hat, ui$G_hat, Q, penalty)
    for (k0 in ui$E) {
      contrast <- ui$si$compute_contrast(solve(Q, replace(rep(0, p), k0 + 1, 1)))
      ws <- ray_w(contrast)
      lo <- ws[["w"]] - 8 * ws[["sd"]]
      hi <- ws[["w"]] + 8 * ws[["sd"]]
      closed <- ui$results[[as.character(k0)]]$intervals
      closed <- closed[closed[, 2] > lo & closed[, 1] < hi, , drop = FALSE]
      ray <- find_truncation_set(event, contrast, n_sd = 8)
      clip <- function(S) pmin(pmax(S, lo), hi)
      expect_equal(clip(ray), clip(closed), tolerance = 1e-6 * ws[["sd"]])
      checked <- checked + 1
    }
  }
})

test_that("adaptive lasso pivots are uniform (no closed form)", {
  skip_on_cran()
  set.seed(3)
  n <- 80; p <- 4; lam <- 3; kappa <- 1
  beta <- c(0.35, -0.25, 0, 0)
  pivots <- c()
  while (length(pivots) < 120) {
    x <- design(n, p)
    Q <- crossprod(x)
    Z <- drop(crossprod(x, x %*% beta + rnorm(n)))
    Y <- Z + drop(t(chol(kappa * Q)) %*% rnorm(p))
    Qinv <- solve(Q)
    penalty <- function(Y) list(lam / abs(drop(Qinv %*% Y)), NULL, NULL)
    b <- weighted_lasso(Q, Y, penalty(Y)[[1]])
    if (all(b == 0)) next
    si <- new(AffineConstraints, Z, Y, Q, NULL, kappa)
    event <- lasso_event(b, drop(Q %*% b) - Y, Q, penalty)
    for (k in which(b != 0)) {
      res <- ray_contrast_inference(si, Qinv[, k], event, level = 0.9, n_sd = 8, n_grid = 801)
      pivots <- c(pivots, res$pivot(beta[k]))
    }
  }
  expect_gt(suppressWarnings(ks.test(pivots, "punif"))$p.value, 1e-3)
})
