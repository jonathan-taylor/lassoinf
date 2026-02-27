library(testthat)
library(lassoinf)

test_that("TruncatedGaussian weights", {
  estimate <- 1.73
  sigma <- 1.0
  lower_bound <- 2.0
  upper_bound <- 10.0
  smoothing_sigma <- 0.5
  noisy_estimate <- estimate

  tg <- new(TruncatedGaussian, estimate, sigma, smoothing_sigma, lower_bound, upper_bound, noisy_estimate, 1.0)
  
  t_val <- c(0.5, 1.0, 2.0)
  w <- tg$weight(t_val)
  expect_length(w, 3)
  expect_true(is.numeric(w))
})

test_that("WeightedGaussianFamily pvalue and interval", {
  estimate <- 1.73
  sigma <- 1.0
  
  # A simple mock weight function in R
  mock_weight <- function(x) {
    # It must take a numeric vector and return a numeric vector
    return(exp(-0.5 * x^2))
  }
  
  # Create list of weight functions
  w_fns <- list(mock_weight)

  wgf <- new(WeightedGaussianFamily, estimate, sigma, w_fns, 10.0, 4000)

  pval <- wgf$pvalue(0.0, "twosided", NaN)
  expect_true(is.numeric(pval))
  expect_true(pval >= 0 && pval <= 1)

  int <- wgf$interval(NaN, 0.9)
  expect_length(int, 2)
  expect_true(int[1] <= int[2])
})