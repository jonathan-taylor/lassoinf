library(testthat)
library(lassoinf)

skip_if_not_installed("glmnet")
library(glmnet)

set.seed(0)
n <- 120; p <- 8
x <- sweep(matrix(rnorm(n * p), n, p), 2, runif(p, 0.5, 3), "*") + rep(runif(p, -1, 1), each = n)
eta <- drop(x[, 1:3] %*% c(0.4, -0.3, 0.2))
ys <- list(gaussian = eta + rnorm(n), binomial = rbinom(n, 1, plogis(eta)), poisson = rpois(n, exp(0.3 * eta)))
obs_w <- runif(n, 0.5, 2)
obs_o <- rnorm(n, sd = 0.2)

options_grid <- list(
  plain = list(),
  limits = list(upper.limits = c(0.1, rep(Inf, p - 1)), lower.limits = c(-Inf, -0.1, rep(-Inf, p - 2))),
  penalty_factor = list(penalty.factor = c(0.5, 3, rep(1, p - 2))),
  unpenalized = list(penalty.factor = c(0, 2, rep(1, p - 2))),
  penalty_factor_inf = list(penalty.factor = c(1, 1, 1, Inf, rep(1, p - 4))),
  exclude = list(exclude = c(4, 6)),
  alpha = list(alpha = 0.5),
  alpha_weights = list(alpha = 0.3, weights = obs_w),
  weights = list(weights = obs_w),
  offset = list(offset = obs_o)
)

# an independent statement of glmnet's objective (smooth part)
smooth_objective <- function(theta, family, w, offset, intercept, ridge) {
  a0 <- if (intercept) theta[1] else 0
  beta <- if (intercept) theta[-1] else theta
  e <- offset + a0 + drop(x %*% beta)
  y <- ys[[family]]
  nll <- switch(family,
                gaussian = 0.5 * (y - e)^2,
                binomial = log1p(exp(e)) - y * e,
                poisson = exp(e) - y * e)
  sum(w * nll) / sum(w) + 0.5 * sum(ridge * beta^2)
}

numerical_gradient <- function(f, theta, h = 1e-6) {
  sapply(seq_along(theta), function(j) {
    e <- rep(0, length(theta)); e[j] <- h
    (f(theta + e) - f(theta - e)) / (2 * h)
  })
}

check_glmnet_fit <- function(family, standardize, intercept, opts, use_family_object = FALSE) {
  fam_arg <- if (use_family_object) get(family)() else family
  args <- c(list(x = x, y = ys[[family]], family = fam_arg, standardize = standardize,
                 intercept = intercept, control = list(thresh = 1e-14), nlambda = 20), opts)
  fit <- do.call(glmnet, args)
  k <- min(8, length(fit$lambda))
  lam <- fit$lambda[k]
  prob <- expect_silent(glmnet_problem_from_fit(fit, x, ys[[family]], s = lam,
                                                weights = opts$weights, offset = opts$offset))

  # 1. the fit satisfies the KKT conditions of the extracted problem
  viol <- kkt_violation(prob$beta_hat, prob$G_hat, prob$D, prob$L, prob$U)
  expect_lt(max(viol), 2e-5 * lam)

  # 2. the gradient is that of glmnet's objective
  w <- if (is.null(opts$weights)) rep(1, n) else opts$weights
  offset <- if (is.null(opts$offset)) rep(0, n) else opts$offset
  wn <- w / sum(w)
  s <- if (standardize) sqrt(colSums(wn * x^2) - colSums(wn * x)^2) else rep(1, p)
  pf <- if (is.null(opts$penalty.factor)) rep(1, p) else opts$penalty.factor
  excluded <- is.infinite(pf) | seq_len(p) %in% opts$exclude
  pf[excluded] <- 1
  pf <- pf * p / sum(pf)
  y <- ys[[family]]
  y_scale <- if (family == "gaussian") sqrt(if (intercept) sum(wn * y^2) - sum(wn * y)^2 else sum(wn * y^2)) else 1
  alpha <- if (is.null(opts$alpha)) 1 else opts$alpha
  ridge <- lam * (1 - alpha) * pf * s^2 / y_scale
  ridge[excluded] <- 0
  f <- function(theta) smooth_objective(theta, family, w, offset, intercept, ridge)
  expect_equal(prob$G_hat, numerical_gradient(f, prob$beta_hat), tolerance = 1e-5)

  # excluded variables are fixed at 0, unpenalized ones have D = 0
  shift <- if (intercept) 1 else 0
  expect_true(all(prob$L[which(excluded) + shift] == 0 & prob$U[which(excluded) + shift] == 0))
  expect_true(all(prob$D[which(pf == 0) + shift] == 0))

  # 3. matrix-free Hessian equals the dense one
  op <- glmnet_problem_from_fit(fit, x, ys[[family]], s = lam, weights = opts$weights,
                                offset = opts$offset, hessian = "operator")
  expect_equal(op$Q_hat$to_dense(), prob$Q_hat, tolerance = 1e-10)

  # 4. the selection constraints hold at the data used for selection
  cons <- lassoinf:::lasso_post_selection_constraints(prob$beta_hat, prob$G_hat, prob$Q_hat,
                                                       prob$D, prob$L, prob$U, 1e-6)
  free <- abs(prob$beta_hat) > 1e-6 & prob$beta_hat > prob$L + 1e-6 & prob$beta_hat < prob$U - 1e-6
  expect_equal(cons$E, which(free) - 1)
  Z_noisy <- -prob$G_hat + drop(prob$Q_hat %*% prob$beta_hat)
  expect_true(all(cons$A$multiply(Z_noisy) <= cons$b + 2e-5 * lam * (1 + abs(cons$b))))
  invisible(prob)
}

for (family in names(ys)) for (standardize in c(TRUE, FALSE)) for (intercept in c(TRUE, FALSE)) {
  for (opt in names(options_grid)) {
    test_that(sprintf("glmnet problem: %s standardize=%s intercept=%s %s", family, standardize, intercept, opt), {
      prob <- check_glmnet_fit(family, standardize, intercept, options_grid[[opt]])
      if (opt == "limits") {
        expect_gte(sum(prob$beta_hat >= prob$U - 1e-8 | prob$beta_hat <= prob$L + 1e-8), 1)
      }
    })
  }
}

test_that("family objects and exclude as a function", {
  for (family in c("binomial", "poisson")) {
    check_glmnet_fit(family, TRUE, TRUE, list(), use_family_object = TRUE)
  }
  fit <- glmnet(x, ys$gaussian, exclude = function(x, y, weights) c(2, 5), control = list(thresh = 1e-14), nlambda = 20)
  prob <- glmnet_problem_from_fit(fit, x, ys$gaussian, s = fit$lambda[8])
  expect_equal(prob$L[c(2, 5) + 1], c(0, 0))
  expect_lt(max(kkt_violation(prob$beta_hat, prob$G_hat, prob$D, prob$L, prob$U)), 2e-5 * fit$lambda[8])
})

test_that("unconverged fits warn; lambda must be on the path", {
  fit <- glmnet(x, ys$poisson, family = "poisson", lambda = c(0.08, 0.04), control = list(thresh = 1e-2))
  expect_warning(glmnet_problem_from_fit(fit, x, ys$poisson, s = 0.04), "KKT")
  expect_error(glmnet_problem_from_fit(fit, x, ys$poisson, s = 0.05), "fit\\$lambda")
  expect_error(glmnet_problem(x, ys$gaussian, rep(0, p), 0, 0.1, family = "gamma"), "family")
})

test_that("inference after a GLM fit", {
  set.seed(5)
  fit <- glmnet(x, ys$binomial, family = "binomial", upper.limits = c(0.1, rep(Inf, p - 1)),
                control = list(thresh = 1e-14), nlambda = 20)
  lam <- fit$lambda[6]
  prob <- glmnet_problem_from_fit(fit, x, ys$binomial, s = lam)
  op <- glmnet_problem_from_fit(fit, x, ys$binomial, s = lam, hessian = "operator")
  Z <- -prob$G_hat + drop(prob$Q_hat %*% prob$beta_hat)
  summaries <- lapply(list(prob, op), function(pr) {
    li <- LassoInference$new(pr$beta_hat, pr$G_hat, pr$Q_hat, pr$D, pr$L, pr$U, Z, prob$Q_hat, NULL, 0.5)
    expect_equal(li$beta_hat, pr$beta_hat, tolerance = 1e-6)
    expect_true(all(li$A$multiply(li$Z_noisy) <= li$b + 1e-8))
    screen <- ScreenedSelection$new(li, top_k = 2)
    list(li$summary(), inactive_summary(screen))
  })
  expect_equal(summaries[[1]], summaries[[2]], tolerance = 1e-5)
  expect_equal(nrow(summaries[[1]][[1]]), length(LassoInference$new(prob$beta_hat, prob$G_hat, prob$Q_hat, prob$D,
                                                                     prob$L, prob$U, Z, prob$Q_hat, NULL, 0.5)$E))
})

test_that("information = 'relaxed' evaluates Q_hat at the one-step relaxed fit", {
  set.seed(11)
  y <- ys$binomial
  fit <- glmnet(x, y, family = "binomial", lambda = c(0.2, 0.1, 0.05), control = list(thresh = 1e-14))
  relaxed <- glmnet_problem_from_fit(fit, x, y, s = 0.05)
  lasso <- glmnet_problem_from_fit(fit, x, y, s = 0.05, information = "lasso")
  expect_equal(relaxed$G_hat, lasso$G_hat)
  X1 <- cbind(1, x)
  active <- which(relaxed$beta_hat != 0)
  mu_lasso <- plogis(drop(X1 %*% lasso$beta_hat))
  # one Newton step on the active coordinates from the LASSO solution
  b <- lasso$beta_hat
  b[active] <- b[active] - solve(lasso$Q_hat[active, active], crossprod(X1[, active], mu_lasso - y) / n)
  mu <- plogis(drop(X1 %*% b))
  expect_equal(relaxed$Q_hat, crossprod(X1, X1 * (mu * (1 - mu))) / n, tolerance = 1e-10)
  # on the active set the gradient is -D sign(beta), so the step undoes the shrinkage
  expect_equal(b[active], lasso$beta_hat[active] +
                 drop(solve(lasso$Q_hat[active, active], (lasso$D * sign(lasso$beta_hat))[active])),
               tolerance = 1e-6)
  expect_equal(lasso$Q_hat, crossprod(X1, X1 * (mu_lasso * (1 - mu_lasso))) / n, tolerance = 1e-10)
  # gaussian: the information does not depend on the fit
  fit_g <- glmnet(x, ys$gaussian, lambda = c(0.2, 0.1), control = list(thresh = 1e-14))
  expect_equal(glmnet_problem_from_fit(fit_g, x, ys$gaussian)$Q_hat,
               glmnet_problem_from_fit(fit_g, x, ys$gaussian, information = "lasso")$Q_hat)
})

test_that("scalar_noise = 0 (no randomization) runs", {
  y <- ys$binomial
  fit <- glmnet(x, y, family = "binomial", lambda = c(0.2, 0.1, 0.05), control = list(thresh = 1e-14))
  prob <- glmnet_problem_from_fit(fit, x, y, s = 0.05)
  Z <- -prob$G_hat + drop(prob$Q_hat %*% prob$beta_hat)
  li <- LassoInference$new(prob$beta_hat, prob$G_hat, prob$Q_hat, prob$D, prob$L, prob$U,
                           Z, prob$Q_hat / n, NULL, 0)
  expect_equal(li$scalar_noise, 0.001)
  summ <- li$summary()
  expect_true(all(summ$lower_conf <= summ$upper_conf))
  expect_true(all(summ$p_value >= 0 & summ$p_value <= 1))
  expect_error(LassoInference$new(prob$beta_hat, prob$G_hat, prob$Q_hat, prob$D, prob$L, prob$U,
                                  Z, prob$Q_hat / n, NULL, -1), "scalar_noise")
})
