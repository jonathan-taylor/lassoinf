library(testthat)
library(lassoinf)

skip_if_not_installed("glmnet")

set.seed(3)
n <- 300; p <- 12
x <- matrix(rnorm(n * p), n, p)
eta <- drop(x[, 1:3] %*% c(0.8, -0.6, 0.5))
ys <- list(gaussian = eta + rnorm(n), binomial = rbinom(n, 1, plogis(eta)), poisson = rpois(n, exp(0.3 * eta)))
path <- function(lam) lam * exp(seq(log(10), 0, length.out = 20))
lams <- c(gaussian = 0.08, binomial = 0.03, poisson = 0.05)
fit_glmnet <- function(family, rows = seq_len(n), ...) {
  glmnet::glmnet(x[rows, ], ys[[family]][rows], family = family, lambda = path(lams[[family]]),
                 control = list(thresh = 1e-14), ...)
}

test_that("no randomization: Z_full is the selection score and Sigma = dispersion * Q_hat / n", {
  for (family in names(ys)) {
    fit <- fit_glmnet(family)
    li <- glmnet_inference(fit, x, ys[[family]], dispersion = 2)
    prob <- glmnet_problem_from_fit(fit, x, ys[[family]])
    expect_equal(li$scalar_noise, 0.001)
    expect_equal(li$Z_full, -prob$G_hat + drop(prob$Q_hat %*% prob$beta_hat))
    expect_equal(li$Sigma, 2 * prob$Q_hat / n)
    expect_equal(li$Q_hat, prob$Q_hat)
    summ <- li$summary()
    expect_true(all(summ$lower_conf <= summ$upper_conf))
  }
})

test_that("carving: scalar_noise from the proportion, Z_full at a full-data Newton step", {
  rows <- sample(n, 200)
  family <- "binomial"
  y <- ys[[family]]
  fit <- fit_glmnet(family, rows)
  li <- glmnet_inference(fit, x, y, selection_rows = rows)
  expect_equal(li$scalar_noise, (1 - 2 / 3) / (2 / 3))
  prob <- glmnet_problem_from_fit(fit, x[rows, ], y[rows])
  X1 <- cbind(1, x)
  b <- prob$beta_hat; A <- which(b != 0)
  mu <- plogis(drop(X1 %*% b))
  b[A] <- b[A] - solve(crossprod(X1[, A], X1[, A] * (mu * (1 - mu))) / n,
                       crossprod(X1[, A], mu - y) / n)
  G_full <- drop(crossprod(X1, plogis(drop(X1 %*% b)) - y)) / n
  expect_equal(li$Z_full, -G_full + drop(prob$Q_hat %*% b), tolerance = 1e-10)
  expect_equal(li$Sigma, prob$Q_hat / n)
  # logical selection_rows
  li2 <- glmnet_inference(fit, x, y, selection_rows = seq_len(n) %in% rows)
  expect_equal(li2$Z_full, li$Z_full)
})

test_that("weights, offset and the ridge term enter the full-data score", {
  rows <- 1:200
  w <- runif(n, 0.5, 2); o <- rnorm(n, sd = 0.2)
  y <- ys$poisson
  fit <- glmnet::glmnet(x[rows, ], y[rows], family = "poisson", weights = w[rows], offset = o[rows],
                        alpha = 0.5, lambda = path(0.05), control = list(thresh = 1e-14))
  expect_error(glmnet_inference(fit, x, y, selection_rows = rows), "full data")
  li <- glmnet_inference(fit, x, y, selection_rows = rows, weights = w, offset = o)
  expect_equal(li$scalar_noise, (sum(w) - sum(w[rows])) / sum(w[rows]))
  # Z_full is -grad + Q b at the Newton step, with the ridge term of glmnet's objective
  prob <- glmnet_problem_from_fit(fit, x[rows, ], y[rows], weights = w[rows], offset = o[rows])
  X1 <- cbind(1, x); wn <- w / sum(w)
  b <- prob$beta_hat; A <- which(b != 0)
  mu <- exp(o + drop(X1 %*% b))
  b[A] <- b[A] - solve(crossprod(X1[, A], X1[, A] * (wn * mu)), crossprod(X1[, A], wn * (mu - y)))
  ridge <- c(0, 0.05 * 0.5 * glmnet_scaling(x[rows, ], w[rows])^2)   # pf = 1, y_scale = 1
  G_full <- drop(crossprod(X1, wn * (exp(o + drop(X1 %*% b)) - y))) + ridge * b
  expect_equal(li$Z_full, -G_full + drop(prob$Q_hat %*% b), tolerance = 1e-8)
  expect_equal(li$Sigma, prob$Q_hat / sum(w))
})

test_that("gaussian dispersion is the least squares residual variance", {
  fit <- fit_glmnet("gaussian")
  y <- ys$gaussian
  li <- glmnet_inference(fit, x, y)
  prob <- glmnet_problem_from_fit(fit, x, y)
  sigma2 <- sum(lm.fit(cbind(1, x), y)$residuals^2) / (n - p - 1)
  expect_equal(li$Sigma, sigma2 * prob$Q_hat / n)
  wide <- matrix(rnorm(20 * 30), 20, 30)
  y_w <- rnorm(20)
  fit_w <- glmnet::glmnet(wide, y_w, lambda = c(0.5, 0.3), control = list(thresh = 1e-14))
  expect_error(glmnet_inference(fit_w, wide, y_w), "dispersion")
  expect_silent(glmnet_inference(fit_w, wide, y_w, dispersion = 1))
})

test_that("hessian = 'operator' matches dense", {
  rows <- 1:200
  for (family in c("gaussian", "binomial")) {
    fit <- fit_glmnet(family, rows)
    dense <- glmnet_inference(fit, x, ys[[family]], selection_rows = rows)
    op <- glmnet_inference(fit, x, ys[[family]], selection_rows = rows, hessian = "operator")
    expect_s4_class(op$Q_hat, "Rcpp_LinearOp")
    expect_equal(op$Z_full, dense$Z_full, tolerance = 1e-10)
    expect_equal(op$Sigma$to_dense(), dense$Sigma, tolerance = 1e-10)
    s_d <- dense$summary(); s_o <- op$summary()
    expect_equal(s_o$lower_conf, s_d$lower_conf, tolerance = 1e-6)
    expect_equal(s_o$upper_conf, s_d$upper_conf, tolerance = 1e-6)
  }
})

test_that("LinearOp$scaled shares the operator", {
  Q <- xtvx_operator(x)
  v <- rnorm(p)
  expect_equal(Q$scaled(2.5)$multiply(v), 2.5 * Q$multiply(v))
  expect_equal(Q$scaled(2.5)$diagonal(), 2.5 * Q$diagonal())
})
