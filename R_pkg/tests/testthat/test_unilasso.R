unilasso_data <- function(n = 60, p = 5, rho = 0.5, beta = c(0.35, -0.25, 0.15, 0, 0)) {
  x <- matrix(rnorm(n * p), n, p) %*% chol((1 - rho) * diag(p) + rho)
  list(x = x, y = drop(x %*% beta) + rnorm(n), beta = beta)
}

test_that("a row can be a union of two intervals", {
  # 3 - t <= 1 / t on t > 0: B = 1, C = t, (A X) = 3 - t
  S <- lassoinf:::intersect_row_sets(
    lassoinf:::ratio_row_sets(u = 3, a = -1, b = 1, beta = 0, c = 0, gamma = 1, s_B = 0, s_C = 1))
  r <- sort(Re(polyroot(c(1, -3, 1))))
  expect_equal(S, rbind(c(0, r[1]), c(r[2], Inf)))
})

test_that("row sets match a grid evaluation", {
  set.seed(0)
  m <- 300
  u <- rnorm(m); a <- rnorm(m); b <- rnorm(m); beta <- rnorm(m); cc <- rnorm(m); gamma <- rnorm(m)
  beta[1:40] <- 0
  gamma[41:80] <- 0
  s_B <- sample(c(-1, 0, 1), m, replace = TRUE)
  s_C <- sample(c(-1, 1), m, replace = TRUE)
  sets <- lassoinf:::ratio_row_sets(u, a, b, beta, cc, gamma, s_B, s_C)
  ts <- seq(-10, 10, length.out = 4001)
  ends <- c(sets[is.finite(sets)], (-b / beta)[is.finite(-b / beta)], (-cc / gamma)[is.finite(-cc / gamma)])
  for (t in ts) {
    if (min(abs(t - ends)) < 1e-3) next
    B <- b + beta * t; C <- cc + gamma * t
    direct <- (s_B == 0 | sign(B) == s_B) & sign(C) == s_C & (u + a * t <= B / C)
    found <- (t >= sets[, "lo1"] & t <= sets[, "hi1"]) | (t >= sets[, "lo2"] & t <= sets[, "hi2"])
    expect_identical(found, direct)
  }
})

test_that("TruncBivariateNormal on a union of intervals matches quadrature", {
  sig_x <- 1.3; sig_omega <- 0.7; a_coeff <- 0.4; theta <- 0.3
  L <- c(-Inf, 0.2, 2); U <- c(-1, 0.7, Inf)
  tbn <- TruncBivariateNormal$new(a_coeff = a_coeff, b_coeff = 1, L = L, U = U,
                                  sig_omega = sig_omega, sig_x = sig_x)
  mu <- theta * sig_x^2
  p_in <- function(z) sapply(z, function(zz) sum(pnorm((U - a_coeff * zz) / sig_omega) -
                                                 pnorm((L - a_coeff * zz) / sig_omega)))
  f <- function(z) dnorm(z, mu, sig_x) * p_in(z)
  for (x in c(-1, 0.4, 1.5)) {
    num <- integrate(f, x, Inf, rel.tol = 1e-10)$value
    den <- num + integrate(f, -Inf, x, rel.tol = 1e-10)$value
    expect_equal(tbn$ccdf(theta, x), num / den, tolerance = 1e-5)
  }
  # a single interval is unchanged
  one <- TruncBivariateNormal$new(a_coeff = a_coeff, b_coeff = 1, L = 0.2, U = 0.7,
                                  sig_omega = sig_omega, sig_x = sig_x)
  expect_equal(one$ccdf(theta, 0.4),
               TruncBivariateNormal$new(a_coeff = a_coeff, b_coeff = 1, L = c(0.2), U = c(0.7),
                                        sig_omega = sig_omega, sig_x = sig_x)$ccdf(theta, 0.4))
})

test_that("unilasso_fit satisfies the KKT conditions", {
  set.seed(1)
  d <- unilasso_data()
  Q <- crossprod(d$x); Z <- drop(crossprod(d$x, d$y))
  for (lam in c(0, 3)) {
    beta <- unilasso_fit(Q, Z, lam)
    pen <- lassoinf:::unilasso_penalty(Z, Q, lam)
    expect_true(all(beta * pen$s >= 0))
    g <- Z - drop(Q %*% beta)
    act <- beta != 0
    expect_equal(g[act], pen$D[act] * pen$s[act], tolerance = 1e-8)
    expect_true(all(pen$s[!act] * g[!act] <= pen$D[!act] + 1e-8))
  }
})

# the uniLasso with intercept at lam (glmnet's scaling), as uniLasso(loo = FALSE) fits it
unilasso_beta <- function(x, y, lam) {
  xc <- sweep(x, 2, colMeans(x))
  unilasso_fit(crossprod(xc), drop(crossprod(xc, y - mean(y))), nrow(x) * lam)
}

test_that("summary reports the full-model least squares coefficients", {
  set.seed(2)
  d <- unilasso_data()
  x <- d$x + 2; y <- d$y + 5          # the fit has an intercept
  ui <- lassoinf:::unilasso_inference_beta(x, y, unilasso_beta(x, y, 0.03), 0.03)
  ols <- coef(lm(y ~ x))[-1]
  s <- ui$summary()
  expect_equal(unname(s$beta_hat), unname(ols[s$index + 1]))
  expect_true(all(is.finite(s$lower_conf) & is.finite(s$upper_conf)))
  expect_true(all(s$lower_conf < s$upper_conf))
})

test_that("the fit is checked", {
  set.seed(4)
  d <- unilasso_data()
  b <- unilasso_beta(d$x, d$y, 0.03)
  expect_warning(lassoinf:::unilasso_inference_beta(d$x, d$y, 1.2 * b, 0.03), "KKT")
  wrong <- b
  wrong[which.max(abs(b))] <- -wrong[which.max(abs(b))]
  expect_error(lassoinf:::unilasso_inference_beta(d$x, d$y, wrong, 0.03), "sign")
  expect_error(lassoinf:::unilasso_inference_beta(matrix(rnorm(40), 5, 8), rnorm(5), rep(0, 8), 1),
               "n > p")
})

test_that("pivots at the true coefficients are uniform, without randomization", {
  skip_on_cran()
  set.seed(3)
  for (lam in c(0, 0.06)) {
    pivots <- c()
    while (length(pivots) < 150) {
      d <- unilasso_data()
      ui <- lassoinf:::unilasso_inference_beta(d$x, d$y, unilasso_beta(d$x, d$y, lam), lam, sigma2 = 1)
      for (k in ui$E) pivots <- c(pivots, ui$pivot(k, d$beta[k + 1]))
    }
    expect_gt(suppressWarnings(ks.test(pivots, "punif")$p.value), 1e-3)
  }
})

test_that("pivots are uniform with randomization", {
  skip_on_cran()
  set.seed(5)
  pivots <- c()
  while (length(pivots) < 150) {
    d <- unilasso_data()
    Q <- crossprod(d$x); Z <- drop(crossprod(d$x, d$y))
    Z_noisy <- Z + drop(t(chol(Q)) %*% rnorm(ncol(Q)))
    lam <- 0.5 * sqrt(nrow(d$x))
    ui <- UniLassoInference$new(unilasso_fit(Q, Z_noisy, lam), Z_noisy, Q, lam = lam,
                                Z_full = Z, Sigma = Q, scalar_noise = 1)
    for (k in ui$E) pivots <- c(pivots, ui$pivot(k, d$beta[k + 1]))
  }
  expect_gt(suppressWarnings(ks.test(pivots, "punif")$p.value), 1e-3)
})

test_that("unilasso_inference takes uniLasso and uniReg fits", {
  skip_if_not_installed("uniLasso")
  set.seed(6)
  n <- 150; p <- 6
  x <- matrix(rnorm(n * p), n, p)
  y <- drop(x %*% c(0.4, -0.3, 0.2, 0, 0, 0)) + rnorm(n)
  fit <- uniLasso::uniLasso(x, y, loo = FALSE, control = list(thresh = 1e-14))
  k <- 15
  s <- unilasso_inference(fit, x, y, s = fit$lambda[k])$summary()
  ref <- lassoinf:::unilasso_inference_beta(x, y, unilasso_beta(x, y, fit$lambda[k]), fit$lambda[k])$summary()
  expect_equal(s, ref, tolerance = 1e-6)
  expect_error(unilasso_inference(fit, x, y), "s must be given")
  expect_error(unilasso_inference(fit, x, y, s = 1.2345), "one of fit")
  expect_error(unilasso_inference(uniLasso::uniLasso(x, y), x, y, s = 0.01), "loo = FALSE")
  reg <- uniLasso::uniReg(x, y, loo = FALSE, control = list(thresh = 1e-14))
  expect_equal(unilasso_inference(reg, x, y)$summary(),
               lassoinf:::unilasso_inference_beta(x, y, unilasso_beta(x, y, reg$lambda), reg$lambda)$summary(),
               tolerance = 1e-6)
})
