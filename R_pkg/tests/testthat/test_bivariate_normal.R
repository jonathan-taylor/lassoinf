library(testthat)
library(lassoinf)

test_that("TruncBivariateNormal CDF works", {
  a <- 1.0; b <- 1.2
  L <- 1.0; U <- 5.0
  sig_omega <- 1.0
  sig_x <- 1.5
  theta <- 2.0
  
  tbn <- TruncBivariateNormal$new(a, b, L, U, sig_omega, sig_x = sig_x, theta = theta)
  
  x <- 2.5
  cdf_val <- tbn$cdf(theta, x)
  expect_true(cdf_val >= 0 && cdf_val <= 1)
  
  ccdf_val <- tbn$ccdf(theta, x)
  expect_equal(cdf_val + ccdf_val, 1.0, tolerance = 1e-6)
})

test_that("TruncBivariateNormal moments work", {
  a <- 1.0; b <- 0.0
  L <- -1.0; U <- 1.0
  sig_omega <- 1.0
  sig_x <- 1.0
  theta <- 0.0
  
  tbn <- TruncBivariateNormal$new(a, b, L, U, sig_omega, sig_x = sig_x, theta = theta)
  
  mean_val <- tbn$E(theta, function(x) matrix(x, ncol=1))
  expect_equal(mean_val, 0.0, tolerance = 1e-7)
  
  expected_var <- 1 - 2 * dnorm(1) / (pnorm(1) - pnorm(-1))
  var_val <- tbn$Var(theta, function(x) matrix(x, ncol=1))
  expect_equal(var_val, expected_var, tolerance = 1e-7)
})

test_that("TruncBivariateNormal coverage mean", {
  set.seed(42)
  a_coeff <- 1.0; b_coeff <- 0.5
  L <- 0.0; U <- 3.0
  sig_omega <- 1.0
  sig_x <- 2.0
  
  mu_true <- 1.5
  theta_true <- mu_true / sig_x^2
  
  tbn <- TruncBivariateNormal$new(a_coeff, b_coeff, L, U, sig_omega, sig_x = sig_x, theta = theta_true)
  
  n_sim <- 100
  alpha <- 0.1
  coverage_tbn <- 0
  
  samples <- numeric(0)
  while (length(samples) < n_sim) {
    z <- rnorm(1000, mean = mu_true, sd = sig_x)
    omega <- rnorm(1000, mean = 0.0, sd = sig_omega)
    s <- a_coeff * z + b_coeff * omega
    
    valid_z <- z[s >= L & s <= U]
    samples <- c(samples, valid_z)
  }
  
  samples <- samples[1:n_sim]
  
  for (x_obs in samples) {
    L_U_theta <- tbn$equal_tailed_interval(x_obs, alpha = alpha)
    lower_mean_tbn <- L_U_theta[1] * sig_x^2
    upper_mean_tbn <- L_U_theta[2] * sig_x^2
    
    if (lower_mean_tbn <= mu_true && mu_true <= upper_mean_tbn) {
      coverage_tbn <- coverage_tbn + 1
    }
  }
  
  cov_rate_tbn <- coverage_tbn / n_sim
  expect_true(cov_rate_tbn >= 0.85 && cov_rate_tbn <= 0.95)
})