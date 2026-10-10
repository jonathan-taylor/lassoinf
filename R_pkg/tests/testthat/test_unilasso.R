unilasso_data <- function(n = 60, p = 5, rho = 0.5, beta = c(0.35, -0.25, 0.15, 0, 0)) {
  x <- matrix(rnorm(n * p), n, p) %*% chol((1 - rho) * diag(p) + rho)
  list(x = x, y = drop(x %*% beta) + rnorm(n), beta = beta)
}

test_that("a row can be a union of two intervals", {
  # 3 - w <= 1 / w on w > 0: u = 3, a = -1, b0 = 0, m = 1, C(w) = w
  S <- lassoinf:::.intersect_row_sets(
    lassoinf:::.unilasso_row_sets(u = 3, a = -1, b0 = 0, m = 1, c = 0, gamma = 1, s = 1))
  r <- sort(Re(polyroot(c(1, -3, 1))))
  expect_equal(S, rbind(c(0, r[1]), c(r[2], Inf)))
})

test_that("row sets match a grid evaluation", {
  set.seed(0)
  rows <- 300
  u <- rnorm(rows); a <- rnorm(rows); b0 <- rnorm(rows); m <- rnorm(rows)
  for (s in c(1, -1)) {
    cc <- 0.3 * s; gamma <- 0.8
    sets <- lassoinf:::.unilasso_row_sets(u, a, b0, m, cc, gamma, s)
    ends <- c(sets[is.finite(sets)], -cc / gamma)
    for (w in seq(-15, 15, length.out = 3001)) {
      if (min(abs(w - ends)) < 1e-3) next
      C <- cc + gamma * w
      direct <- sign(C) == s & (u + a * w <= b0 + m / C)
      found <- (w >= sets[, "lo1"] & w <= sets[, "hi1"]) | (w >= sets[, "lo2"] & w <= sets[, "hi2"])
      expect_identical(found, direct)
    }
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
  # loo = TRUE: approximate, with a warning; the fit solves the shifted problem exactly,
  # so the estimates are the least squares coefficients
  fit_loo <- uniLasso::uniLasso(x, y, control = list(thresh = 1e-14))
  expect_warning(ui_loo <- unilasso_inference(fit_loo, x, y, s = fit_loo$lambda[k]), "approximate")
  s_loo <- ui_loo$summary()
  expect_equal(s_loo$beta_hat, unname(coef(lm(y ~ x))[-1][s_loo$index + 1]), tolerance = 1e-6)
  reg <- uniLasso::uniReg(x, y, loo = FALSE, control = list(thresh = 1e-14))
  expect_equal(unilasso_inference(reg, x, y)$summary(),
               lassoinf:::unilasso_inference_beta(x, y, unilasso_beta(x, y, reg$lambda), reg$lambda)$summary(),
               tolerance = 1e-6)
})

test_that("a loo = TRUE fit solves the uniLasso with penalties n lambda + kappa_j", {
  skip_if_not_installed("uniLasso")
  set.seed(7)
  n <- 200; p <- 6
  x <- matrix(rnorm(n * p), n, p) %*% chol(0.5 * diag(p) + 0.5)
  y <- drop(x %*% c(0.35, -0.25, 0.15, 0, 0, 0)) + rnorm(n)
  fit <- uniLasso::uniLasso(x, y, control = list(thresh = 1e-14))
  k <- 20
  b <- as.numeric(fit$beta[, k])
  xc <- sweep(x, 2, colMeans(x)); yc <- y - mean(y)
  kappa <- lassoinf:::unilasso_loo_kappa(xc, yc, b)
  b_uni <- drop(crossprod(xc, yc)) / colSums(xc^2)
  score <- b_uni * drop(crossprod(xc, yc - xc %*% b))
  act <- b != 0
  expect_true(any(act))
  expect_equal(score[act], n * fit$lambda[k] + kappa[act], tolerance = 1e-6)
  expect_true(all(score[!act] <= n * fit$lambda[k] + kappa[!act] + 1e-8))
  sigma2 <- sum(resid(lm(y ~ x))^2) / (n - p - 1)
  expect_true(all(abs(kappa / (2 * sigma2) - 1) < 0.5))
})
