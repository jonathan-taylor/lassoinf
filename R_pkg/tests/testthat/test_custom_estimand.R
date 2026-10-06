library(testthat)
library(lassoinf)

lasso_cd <- function(Q, Z, lam, n_iter = 500) {
  # minimize 1/2 b'Qb - b'Z + lam ||b||_1 by coordinate descent
  b <- rep(0, nrow(Q))
  for (it in seq_len(n_iter)) for (j in seq_along(b)) {
    r <- Z[j] - sum(Q[j, ] * b) + Q[j, j] * b[j]
    b[j] <- sign(r) * max(abs(r) - lam, 0) / Q[j, j]
  }
  b
}

gaussian_instance <- function(n = 100, p = 10, kappa = 0.5, use_Sigma_noise = FALSE, operator = FALSE) {
  X <- matrix(rnorm(n * p), n, p)
  beta <- c(3, 3, rep(0, p - 2)) / sqrt(n)
  mu <- drop(X %*% beta)
  y <- mu + rnorm(n)
  Q <- crossprod(X)
  Z <- drop(crossprod(X, y))
  Z_noisy <- Z + drop(t(chol(kappa * Q)) %*% rnorm(p))
  lam <- 1.5 * sqrt(n)
  b <- lasso_cd(Q, Z_noisy, lam)
  G <- drop(Q %*% b) - Z_noisy
  Q_arg <- if (operator) xtvx_operator(X) else Q
  li <- LassoInference$new(b, G, Q_arg, rep(lam, p), NULL, NULL, Z, Q_arg,
                           if (use_Sigma_noise) kappa * Q else NULL,
                           if (use_Sigma_noise) NaN else kappa)
  list(X = X, y = y, mu = mu, li = li)
}

# an instance with at least one active and two inactive variables
selected_instance <- function(...) {
  repeat {
    inst <- gaussian_instance(...)
    if (length(inst$li$E) > 0 && length(inst$li$E_c) > 1) return(inst)
  }
}

test_that("selection coordinates match OLS quantities", {
  set.seed(1)
  inst <- selected_instance()
  X <- inst$X; y <- inst$y; li <- inst$li
  E <- li$E + 1; E_c <- li$E_c + 1
  expect_true(length(E) > 0 && length(E_c) > 0)

  coords <- SelectionCoordinates$new(li)
  resid <- y - X[, E, drop = FALSE] %*% qr.solve(X[, E, drop = FALSE], y)
  for (k in seq_along(E_c)) {
    j <- E_c[k]
    score <- coords$inactive_score(j - 1)
    expect_equal(estimand_value(score, li$Z_full), sum(X[, j] * resid), tolerance = 1e-8)
    coef <- coords$inactive_coef(j - 1)
    expect_equal(estimand_value(coef, li$Z_full), unname(tail(qr.solve(X[, c(E, j)], y), 1)), tolerance = 1e-8)
  }
  expect_error(coords$inactive_score(li$E[1]), "not inactive")
  expect_error(coords$estimand(NULL, rep(1, length(E_c)), "other"), "basis")

  # matrix-free constraints equal the explicit dense ones
  Q <- li$Q_hat; W <- li$W; s <- li$s_E
  R <- matrix(0, length(E_c), ncol(Q)); R[cbind(seq_along(E_c), E_c)] <- 1
  R[, E] <- -Q[E_c, E, drop = FALSE] %*% W
  B <- matrix(0, length(E), ncol(Q)); B[, E] <- W
  A_explicit <- rbind(-s * B, R[rep(seq_along(E_c), each = 2), ] * rep(c(1, -1), length(E_c)))
  expect_equal(li$A$to_dense(), A_explicit, tolerance = 1e-10)
  expect_equal(W, solve(Q[E, E]), tolerance = 1e-10)
})

test_that("matrix-free X'X operator gives the same inference as the dense matrix", {
  set.seed(2)
  inst <- selected_instance()
  li <- inst$li
  li_op <- LassoInference$new(li$beta_hat, li$G_hat, xtvx_operator(inst$X), li$D, NULL, NULL,
                              li$Z_full, xtvx_operator(inst$X), NULL, li$scalar_noise)
  expect_equal(li_op$A$to_dense(), li$A$to_dense(), tolerance = 1e-8)
  expect_equal(li_op$summary(), li$summary(), tolerance = 1e-6)
  expect_equal(inactive_summary(li_op), inactive_summary(li), tolerance = 1e-6)
  expect_equal(xtvx_operator(inst$X)$diagonal(), diag(crossprod(inst$X)), tolerance = 1e-10)
})

test_that("contrast inference reproduces coefficient inference", {
  set.seed(1)
  for (use_Sigma_noise in c(FALSE, TRUE)) {
    li <- selected_instance(use_Sigma_noise = use_Sigma_noise)$li
    summ <- li$summary()
    for (k in seq_along(li$E)) {
      eta <- rep(0, length(li$beta_hat)); eta[li$E + 1] <- li$W[, k]
      res <- contrast_inference(li, eta)
      expect_equal(c(res$lower_conf, res$upper_conf, res$p_value),
                   unlist(summ[k, c("lower_conf", "upper_conf", "p_value")], use.names = FALSE),
                   tolerance = 1e-6)
      if (use_Sigma_noise) {
        cov <- drop(li$Sigma %*% eta)
        res_cov <- custom_estimand_inference(li, sum(eta * li$Z_full), sum(eta * cov), cov)
        expect_equal(res_cov[c("lower_conf", "upper_conf", "p_value")],
                     res[c("lower_conf", "upper_conf", "p_value")], tolerance = 1e-6)
      }
    }
  }
})

test_that("lasso basis shifts the refit basis", {
  set.seed(1)
  li <- selected_instance()$li
  coords <- SelectionCoordinates$new(li)
  E <- li$E + 1; E_c <- li$E_c + 1
  a_E <- rnorm(length(E)); a_Ec <- rnorm(length(E_c))
  refit <- coords$estimand(a_E, a_Ec, "refit")
  lasso <- coords$estimand(a_E, a_Ec, "lasso")
  expect_equal(lasso$eta, refit$eta)
  expect_equal(estimand_value(lasso, li$Z_noisy), sum(a_E * li$beta_hat[E]) - sum(a_Ec * li$G_hat[E_c]), tolerance = 1e-8)
  expect_equal(estimand_value(coords$inactive_score(li$E_c[1], "lasso"), li$Z_noisy), -li$G_hat[E_c[1]], tolerance = 1e-8)

  r_refit <- contrast_inference(li, refit)
  r_lasso <- contrast_inference(li, lasso, null_value = lasso$offset)
  expect_equal(r_lasso$lower_conf, r_refit$lower_conf + lasso$offset, tolerance = 1e-8)
  expect_equal(r_lasso$upper_conf, r_refit$upper_conf + lasso$offset, tolerance = 1e-8)
  expect_equal(r_lasso$p_value, r_refit$p_value, tolerance = 1e-8)
})

screening_event <- function(G, threshold = NULL, top_k = NULL, conditioning = "first_dropped") {
  ord <- order(-abs(G))
  if (!is.null(threshold)) {
    keep <- sort(which(abs(G) > threshold)); extra <- NULL
  } else {
    keep <- sort(ord[seq_len(top_k)]); extra <- NULL
    if (conditioning == "first_dropped" && length(ord) > top_k) {
      l0 <- ord[top_k + 1]; extra <- c(l0, sign(G[l0]))
    }
  }
  c(keep, sign(G[keep]), extra)
}

test_that("screening constraints describe the screening event exactly", {
  set.seed(1)
  li <- selected_instance()$li
  m <- length(li$E_c)
  coords <- SelectionCoordinates$new(li)
  offs <- coords$lasso_offsets()$score
  L_chol <- t(chol(li$Sigma))
  rules <- list(list(threshold = 0.5 * li$D[1]),
                list(top_k = 2, conditioning = "first_dropped"),
                list(top_k = 2, conditioning = "exact"))
  for (rule in rules) {
    S <- do.call(ScreenedSelection$new, c(list(li), rule))
    G_obs <- li$G_hat[li$E_c + 1]
    event_obs <- do.call(screening_event, c(list(G_obs), rule))
    K <- length(S$screened)
    n_rows <- if (!is.null(rule$threshold)) K + 2 * (m - K) else
      if (rule$conditioning == "first_dropped") K + 2 * (m - K - 1) else 2 * K * (m - K)
    expect_equal(S$A_screen$rows(), n_rows)
    expect_equal(S$A$rows(), li$A$rows() + n_rows)

    agree <- 0; n_in <- 0; outcomes <- c()
    for (i in 1:500) {
      Zp <- li$Z_noisy + 0.3 * drop(L_chol %*% rnorm(nrow(L_chol)))
      if (any(li$A$multiply(Zp) > li$b)) next
      n_in <- n_in + 1
      G <- -(coords$refit_score(Zp) + offs)
      same <- identical(do.call(screening_event, c(list(G), rule)), event_obs)
      in_poly <- all(S$A_screen$multiply(Zp) <= S$b_screen)
      agree <- agree + (same == in_poly)
      outcomes <- union(outcomes, same)
    }
    expect_gt(n_in, 50)
    expect_setequal(outcomes, c(TRUE, FALSE))
    expect_equal(agree, n_in)
  }
})

test_that("screened summaries", {
  set.seed(1)
  li <- selected_instance()$li
  S <- ScreenedSelection$new(li, top_k = 2)
  G <- li$G_hat[li$E_c + 1]
  ord <- order(-abs(G))
  expect_equal(S$screened, li$E_c[ord[1:2]])
  expect_equal(S$screened_signs, sign(G[ord[1:2]]))
  expect_equal(S$first_dropped, li$E_c[ord[3]])

  for (basis in c("refit", "lasso")) expect_equal(inactive_summary(S, basis = basis)$index, S$screened)
  expect_false(isTRUE(all.equal(inactive_summary(S), inactive_summary(li, variables = S$screened))))

  # first_dropped intervals are nested in the exact ones
  S_exact <- ScreenedSelection$new(li, top_k = 2, conditioning = "exact")
  expect_null(S_exact$first_dropped)
  coords <- SelectionCoordinates$new(li)
  n_strict <- 0
  for (i in 1:20) {
    eta <- coords$contrast(rnorm(length(li$E)), rnorm(length(li$E_c)))
    contrast <- li$si$compute_contrast(eta)
    fd <- contrast$get_interval(0, S$A, S$b)
    ex <- contrast$get_interval(0, S_exact$A, S_exact$b)
    expect_true(ex[1] - 1e-9 <= fd[1] && fd[2] <= ex[2] + 1e-9)
    n_strict <- n_strict + (fd[1] > ex[1] + 1e-9 || fd[2] < ex[2] - 1e-9)
  }
  expect_gt(n_strict, 0)

  S_all <- ScreenedSelection$new(li, top_k = length(li$E_c))
  expect_equal(S_all$A_screen$rows(), 0)
  expect_equal(inactive_summary(S_all), inactive_summary(li, variables = S_all$screened))

  expect_error(ScreenedSelection$new(li), "exactly one")
  expect_error(ScreenedSelection$new(li, top_k = 1, conditioning = "other"), "conditioning")
})

test_that("covariance estimands require Sigma_noise", {
  set.seed(3)
  li <- selected_instance()$li
  eta <- SelectionCoordinates$new(li)$inactive_coef(li$E_c[1])$eta
  cov <- drop(li$Sigma %*% eta)
  expect_error(custom_estimand_inference(li, sum(eta * li$Z_full), sum(eta * cov), cov), "Sigma_noise")
  li_n <- selected_instance(use_Sigma_noise = TRUE)$li
  expect_error(custom_estimand_inference(li_n, 0, 1, rep(0, length(li_n$Z_full))), "zero")
})

test_that("screened inference covers; ignoring screening does not", {
  set.seed(7)
  level <- 0.9
  covered <- c(); lasso_only <- c()
  for (rep in 1:150) {
    inst <- gaussian_instance(kappa = 0.2)
    li <- inst$li
    if (length(li$E) == 0 || length(li$E_c) < 2) next
    S <- ScreenedSelection$new(li, top_k = 1)
    est <- SelectionCoordinates$new(li)$inactive_coef(S$screened[1], "lasso")
    truth <- estimand_value(est, drop(crossprod(inst$X, inst$mu)))
    r <- contrast_inference(S, est, level = level)
    covered <- c(covered, r$lower_conf <= truth && truth <= r$upper_conf)
    r <- contrast_inference(li, est, level = level)
    lasso_only <- c(lasso_only, r$lower_conf <= truth && truth <= r$upper_conf)
  }
  se <- sqrt(level * (1 - level) / length(covered))
  expect_lt(abs(mean(covered) - level), 4 * se)
  expect_lt(mean(lasso_only), level - 4 * se)
})
