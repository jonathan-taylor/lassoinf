args <- commandArgs(TRUE); S <- args[1]; n <- as.integer(args[2]); loo <- as.logical(args[3]); variant <- args[4]
.libPaths(c(file.path(S, "rlib"), .libPaths())); suppressMessages({library(uniLasso); library(lassoinf)})
binomial_unilasso_inference <- function(fit, x, y, k, loo, level = 0.9) {
  n <- nrow(x); p <- ncol(x)
  lam <- fit$lambda[k]; beta_hat <- as.numeric(fit$beta[, k]); a0 <- fit$a0[k]
  # univariate logistic fits: slopes b_j and the information for each slope
  uni <- sapply(seq_len(p), function(j) {
    g <- glm.fit(cbind(1, x[, j]), y, family = binomial())
    w <- g$weights; xw <- x[, j] - sum(w * x[, j]) / sum(w)
    c(b = unname(g$coefficients[2]), info = sum(w * xw^2))
  })
  # Hessian at the full-model MLE, intercept profiled out
  full <- glm.fit(cbind(1, x), y, family = binomial()); w <- full$weights
  xw <- sweep(x, 2, colSums(w * x) / sum(w)); Q_mle <- crossprod(xw, w * xw)
  m <- plogis(a0 + drop(x %*% beta_hat)); wb <- m * (1 - m)
  xb <- sweep(x, 2, colSums(wb * x) / sum(wb)); Q_bh <- crossprod(xb, wb * xb)
  Q <- if (variant %in% c("Q_at_beta_hat", "hybrid")) Q_bh else Q_mle
  Sigma <- if (variant == "hybrid") Q_mle else Q
  mu_hat <- plogis(a0 + drop(x %*% beta_hat))
  Z <- drop(Q %*% beta_hat) + drop(crossprod(x, y - mu_hat))     # linearized score at beta_hat
  lam_Z <- n * lam + if (loo) 2 else 0
  UniLassoInference$new(beta_hat, Z, Q, lam = lam_Z, Z_full = Z, Sigma = Sigma, scalar_noise = 0,
                        level = level, C = uni["b", ], C_scale = if (variant == "C_scale_full") diag(Q) else uni["info", ])
}
set.seed(n + loo)
p <- 6; beta <- c(0.6, -0.5, 0.4, 0, 0, 0) * sqrt(300 / n) * 1.5
piv <- c(); cover <- c(); naive_cover <- c(); kkt <- c(); t0 <- Sys.time()
while (length(piv) < 300) {
  x <- matrix(rnorm(n * p), n, p) %*% chol(0.5 * diag(p) + 0.5); y <- rbinom(n, 1, plogis(drop(x %*% beta)))
  fit <- uniLasso(x, y, family = "binomial", loo = loo, control = list(thresh = 1e-12))
  k <- round(length(fit$lambda) * 0.4); if (all(fit$beta[, k] == 0)) next
  ui <- tryCatch(binomial_unilasso_inference(fit, x, y, k, loo), error = function(e) NULL)
  if (is.null(ui)) next
  kkt <- c(kkt, max(abs(ui$beta_hat - as.numeric(fit$beta[, k]))))
  se <- sqrt(diag(solve(ui$Q_hat)))
  for (j in ui$E) {
    r <- ui$results[[as.character(j)]]
    piv <- c(piv, ui$pivot(j, beta[j + 1])); cover <- c(cover, r$lower_conf <= beta[j + 1] && beta[j + 1] <= r$upper_conf)
    naive_cover <- c(naive_cover, abs(r$beta_hat - beta[j + 1]) <= qnorm(0.95) * se[j + 1])
  }
}
cat(variant, sprintf("binomial n = %4d loo = %-5s: %d targets, KS p %.3g, coverage %.3f (naive %.3f); prox step moved beta by at most %.1e; %.0fs\n",
    n, loo, length(piv), suppressWarnings(ks.test(piv, "punif")$p.value), mean(cover), mean(naive_cover), max(kkt), as.numeric(Sys.time() - t0, units = "secs")))
