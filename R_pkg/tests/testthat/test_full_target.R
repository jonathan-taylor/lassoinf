library(testthat)
library(lassoinf)

full_instance <- function(n = 100, p = 10, kappa = 0.5) {
  repeat {
    X <- matrix(rnorm(n * p), n, p)
    y <- drop(X %*% c(3, 3, rep(0, p - 2))) / sqrt(n) + rnorm(n)
    Q <- crossprod(X)
    Z <- drop(crossprod(X, y))
    Z_noisy <- Z + drop(t(chol(kappa * Q)) %*% rnorm(p))
    lam <- 1.5 * sqrt(n)
    b <- rep(0, p)
    for (it in 1:500) for (j in 1:p) {
      r <- Z_noisy[j] - sum(Q[j, ] * b) + Q[j, j] * b[j]
      b[j] <- sign(r) * max(abs(r) - lam, 0) / Q[j, j]
    }
    if (any(b != 0)) break
  }
  G <- drop(Q %*% b) - Z_noisy
  fit <- function(Q_arg, target, Q_solve = NULL)
    LassoInference$new(b, G, Q_arg, rep(lam, p), NULL, NULL, Z, Q, NULL, kappa,
                       target = target, Q_solve = Q_solve)
  list(X = X, y = y, Q = Q, fit = fit)
}

test_that("full-model target uses the full-model coefficients", {
  set.seed(3)
  inst <- full_instance()
  li <- inst$fit(inst$Q, "full")
  E <- li$E + 1
  ols <- drop(solve(inst$Q, crossprod(inst$X, inst$y)))
  Q_inv <- solve(inst$Q)
  for (k in seq_along(E)) {
    j <- as.character(li$E[k])
    expect_equal(li$contrasts[[j]], Q_inv[, E[k]], tolerance = 1e-8)
    expect_equal(li$naive[[j]][[1]], ols[E[k]], tolerance = 1e-8)
  }

  # conjugate gradient with a LinearOp agrees with the Cholesky solve
  li_op <- inst$fit(dense_operator(inst$Q), "full")
  expect_equal(li_op$summary(), li$summary(), tolerance = 1e-6)

  expect_false(isTRUE(all.equal(inst$fit(inst$Q, "selected")$intervals, li$intervals)))
  expect_error(inst$fit(inst$Q, "other"))
})

test_that("full-model target requires invertible Q_hat", {
  set.seed(4)
  X <- matrix(rnorm(200), 10, 20)
  Q <- crossprod(X)
  expect_error(lassoinf:::full_model_contrasts(Q, 1:2), "positive definite")
  expect_error(lassoinf:::full_model_contrasts(dense_operator(Q), 1:2), "positive definite")
})

test_that("Q_solve replaces the solves with Q_hat", {
  set.seed(5)
  inst <- full_instance()
  li <- inst$fit(inst$Q, "full")
  R <- chol(inst$Q)
  li_solve <- inst$fit(dense_operator(inst$Q), "full",
                       Q_solve = function(B) backsolve(R, forwardsolve(t(R), B)))
  expect_equal(li_solve$summary(), li$summary(), tolerance = 1e-10)
  expect_error(inst$fit(inst$Q, "full", Q_solve = function(B) B[, 1]), "Q_solve returned")
})
