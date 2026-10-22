.cov <- function(p, rho = 0.3) {
  A <- matrix(rnorm(p * p), p)
  rho * A %*% t(A) / p + diag(runif(p, 0.5, 2))
}

test_that("box_lasso_fit solves the box-constrained lasso", {
  set.seed(0)
  p <- 12
  Q <- solve(.cov(p))
  Y <- 3 * rnorm(p)
  D <- runif(p)
  D[1] <- 0
  L <- rep(-Inf, p); U <- rep(Inf, p)
  L[2] <- U[2] <- 0
  U[3] <- 0.05
  L[4] <- 0
  b <- box_lasso_fit(Q, Y, D, L, U)
  expect_equal(b[2], 0)
  expect_lte(b[3], 0.05)
  expect_gte(b[4], 0)
  expect_lt(box_lasso_kkt_violation_cpp(b, Q, Y, D, L, U), 1e-8)
  # proximal gradient reference
  t_step <- 1 / max(eigen(Q, symmetric = TRUE, only.values = TRUE)$values)
  r <- rep(0, p)
  for (i in 1:20000) {
    v <- r - t_step * drop(Q %*% r - Y)
    r <- pmin(pmax(sign(v) * pmax(abs(v) - t_step * D, 0), L), U)
  }
  expect_equal(b, r, tolerance = 1e-6)
  expect_error(box_lasso_fit(diag(2), c(1, 1), c(-1, 1)))
})

test_that("uncorrelated estimates are selected by their standardized values", {
  set.seed(1)
  p <- 40
  pi_sel <- 0.7
  se <- runif(p, 0.5, 3)
  beta_hat <- 2 * se * rnorm(p)
  pf <- runif(p, 0.5, 1.5)
  res <- covariance_lasso_inference(beta_hat, diag(se^2), prop_select = pi_sel,
                                    penalty_factor = pf, threshold = 1.5)
  z_star <- res$beta_star / (se / sqrt(pi_sel))
  expect_equal(res$selected, as.character(which(abs(z_star) > 1.5 * pf)))
  lasso <- sign(z_star) * pmax(abs(z_star) - 1.5 * pf, 0) * se / sqrt(pi_sel)
  expect_equal(unname(res$lasso_coef), unname(lasso), tolerance = 1e-6)
})

test_that("names, exclude, penalty_factor and perturbation", {
  set.seed(3)
  p <- 15
  Sigma <- .cov(p)
  beta <- c(5, 5, 0.1, rep(0, p - 3)) * sqrt(diag(Sigma))
  beta_hat <- beta + drop(rnorm(p) %*% chol(Sigma))
  names(beta_hat) <- paste0("x", 1:p)
  pf <- rep(1, p)
  pf[3] <- 0
  omega <- rnorm(p)
  res <- covariance_lasso_inference(beta_hat, Sigma, exclude = "x2", penalty_factor = pf,
                                    perturbation = omega)
  expect_false("x2" %in% res$selected)
  expect_equal(res$lasso_coef[["x2"]], 0)
  expect_true("x3" %in% res$selected)
  expect_equal(rownames(res$summary), res$selected)
  expect_equal(unname(res$beta_star), unname(beta_hat + sqrt(0.25) * omega))
  for (ex in list(2, seq_len(p) == 2)) {
    other <- covariance_lasso_inference(beta_hat, Sigma, exclude = ex, penalty_factor = pf,
                                        perturbation = omega, threshold = res$threshold)
    expect_equal(other$summary, res$summary)
  }
})

test_that("targets, the splitting estimate and the selective MLE", {
  set.seed(6)
  p <- 20
  Sigma <- .cov(p)
  beta <- c(4, -3, 2, rep(0, p - 3)) * sqrt(diag(Sigma))
  beta_hat <- beta + drop(rnorm(p) %*% chol(Sigma))
  omega <- drop(rnorm(p) %*% chol(Sigma))
  res <- covariance_lasso_inference(beta_hat, Sigma, perturbation = omega)
  E <- as.integer(res$selected)
  expect_gt(length(E), 0)
  Q <- solve(Sigma)
  expect_equal(res$summary$estimate,
               drop(solve(Q[E, E, drop = FALSE], (Q %*% beta_hat)[E])), tolerance = 1e-8)
  full <- covariance_lasso_inference(beta_hat, Sigma, perturbation = omega, target = "full",
                                     threshold = res$threshold)
  expect_equal(full$selected, res$selected)
  expect_equal(full$summary$estimate, beta_hat[E])
  expect_equal(full$splitting$std_err, sqrt(diag(Sigma))[E] / sqrt(0.2))
  inf <- res$inference
  for (k in seq_along(E)) {
    v <- rep(0, p)
    v[E] <- inf$W[, k]
    variance <- drop(crossprod(v, Q %*% v))
    family <- lassoinf:::.truncated_family(inf, v, variance)$family
    expect_equal(family$E(res$summary$mle[k] / variance, function(x) x),
                 res$summary$estimate[k], tolerance = 1e-6)
  }
})

test_that("agrees with Python", {
  skip_if_not_installed("reticulate")
  skip_if_not(reticulate::py_module_available("lassoinf"))
  li <- reticulate::import("lassoinf")
  set.seed(8)
  p <- 12
  Sigma <- .cov(p)
  beta_hat <- c(3, -2.5, 1, rep(0, p - 3)) * sqrt(diag(Sigma)) + drop(rnorm(p) %*% chol(Sigma))
  omega <- rnorm(p)
  for (target in c("selected", "full")) {
    r <- covariance_lasso_inference(beta_hat, Sigma, perturbation = omega, threshold = 1.2,
                                    target = target, level = 0.9)
    py <- li$covariance_lasso_inference(beta_hat, Sigma, perturbation = omega, threshold = 1.2,
                                        target = target, level = 0.9)
    s <- reticulate::py_to_r(py$summary_)
    expect_equal(unname(as.matrix(r$summary)), unname(as.matrix(s)), tolerance = 1e-4)
  }
})
