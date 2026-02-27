library(testthat)
library(lassoinf)

test_that("selective_inference consistency", {
  set.seed(42)
  Z <- rnorm(5)
  Z_noisy <- Z + rnorm(5) * 0.5
  Q_base <- matrix(rnorm(25), 5, 5)
  Q <- diag(5) + 0.1 * Q_base
  Q <- t(Q) %*% Q
  Q_noise_base <- matrix(rnorm(25), 5, 5)
  Q_noise <- diag(5) * 0.25 + 0.05 * Q_noise_base
  Q_noise <- t(Q_noise) %*% Q_noise

  si_cpp <- new(SelectiveInference, Z, Z_noisy, Q, Q_noise)

  v <- c(1.0, -0.5, 0.2, 0.0, 0.0)

  # 1. Test compute_params
  params <- si_cpp$compute_params(v)
  expect_type(params, "list") # Rcpp wrapped objects behave as lists now
  expect_true(!is.null(params$gamma))
  expect_true(!is.null(params$c))
  expect_true(!is.null(params$bar_gamma))
  expect_true(is.numeric(params$bar_s))

  # 2. Test data_splitting_estimator
  dse <- si_cpp$data_splitting_estimator(v)
  expect_length(dse, 2)
  expect_true(is.numeric(dse))

  # 3. Test get_interval
  A <- matrix(rnorm(20), 4, 5)
  b <- rnorm(4)
  
  for (t_val in c(-5.0, 0.0, 1.23, 10.0)) {
    interval <- si_cpp$get_interval(v, t_val, A, b)
    expect_length(interval, 2)
    # interval might be NaN, which is fine, we just check shape
  }
})

test_that("selective_inference infeasible interval", {
  Z <- rep(0, 2)
  Z_noisy <- rep(0, 2)
  Q <- diag(2)
  Q_noise <- diag(2)
  v <- c(1.0, 0.0)
  
  A <- matrix(c(1.0, 0.0, -1.0, 0.0), nrow = 2, byrow = TRUE)
  b <- c(-1.0, -1.0)
  
  si_cpp <- new(SelectiveInference, Z, Z_noisy, Q, Q_noise)
  interval <- si_cpp$get_interval(v, 0.0, A, b)
  
  expect_true(is.nan(interval[1]))
  expect_true(is.nan(interval[2]))
})

test_that("lasso_inference logic", {
  set.seed(42)
  p <- 5
  beta_hat <- c(1.2, -0.8, 0, 0, 0)
  G_hat <- c(0, 0, 0.5, -0.2, 0.1)
  Q_hat <- diag(p) + 0.1 * matrix(rnorm(p*p), p, p)
  Q_hat <- t(Q_hat) %*% Q_hat
  
  D <- rep(1.0, p)
  L <- rep(-Inf, p)
  U <- rep(Inf, p)
  
  Z_full <- rnorm(p)
  Sigma <- diag(p)
  Sigma_noisy <- diag(p)
  
  res <- lasso_inference(beta_hat, G_hat, Q_hat, D, L, U, Z_full, Sigma, Sigma_noisy)
  
  expect_true(is.data.frame(res))
  if (nrow(res) > 0) {
    expect_true("p_value" %in% colnames(res))
    expect_true("lower_conf" %in% colnames(res))
    expect_true("upper_conf" %in% colnames(res))
    expect_true(all(res$lower_conf <= res$upper_conf, na.rm=TRUE))
  }
})