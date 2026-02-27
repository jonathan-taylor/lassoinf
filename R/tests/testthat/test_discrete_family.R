library(testthat)
library(lassoinf)

test_that("discrete_family cdf, ccdf, equal_tailed_interval", {
  set.seed(42)
  grid <- seq(-5, 5, length.out = 200)
  weights <- exp(-0.5 * grid^2)
  weights <- weights / sum(weights)

  df_cpp <- new(DiscreteFamily, grid, weights, 0.0)

  x <- 1.0
  cdf_cpp <- df_cpp$cdf(0.0, x, 1.0)
  expect_true(is.numeric(cdf_cpp))

  ccdf_cpp <- df_cpp$ccdf(0.0, x, 0.0)
  expect_true(is.numeric(ccdf_cpp))

  theta <- 1.5
  cdf_cpp_t <- df_cpp$cdf(theta, x, 1.0)
  expect_true(is.numeric(cdf_cpp_t))

  interval <- df_cpp$equal_tailed_interval(x, 0.1, 1e-6)
  expect_length(interval, 2)
  expect_true(interval[1] <= interval[2])
})