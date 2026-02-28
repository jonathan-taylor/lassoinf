#' @useDynLib lassoinf, .registration=TRUE
#' @importFrom Rcpp loadModule
NULL

loadModule("lassoinf_cpp", TRUE)

# --- Bivariate Normal Implementation ---
bivariate_normal_cdf <- function(h, k, rho) {
  if (abs(rho) < 1e-12) {
    return(pnorm(h) * pnorm(k))
  }
  if (rho > 0.999999) {
    return(pnorm(min(h, k)))
  }
  if (rho < -0.999999) {
    return(max(0, pnorm(h) + pnorm(k) - 1))
  }
  
  requireNamespace("mvtnorm", quietly = TRUE)
  sigma <- matrix(c(1, rho, rho, 1), 2, 2)
  as.numeric(mvtnorm::pmvnorm(upper = c(h, k), mean = c(0, 0), sigma = sigma))
}

compute_gaussian_conditional_stats <- function(mu_x, sig_x, sig_omega, cx, comega, a, b, t=NULL) {
  var_x <- sig_x^2
  var_omega <- sig_omega^2
  
  mu_s <- cx * mu_x
  var_s <- (cx^2 * var_x) + (comega^2 * var_omega)
  sig_s <- sqrt(var_s)
  
  cov_xs <- cx * var_x
  rho <- cov_xs / (sig_x * sig_s)
  
  alpha <- if (a != -Inf) (a - mu_s) / sig_s else -Inf
  beta <- if (b != Inf) (b - mu_s) / sig_s else Inf
  
  p_constraint <- (if (beta != Inf) pnorm(beta) else 1.0) - 
                  (if (alpha != -Inf) pnorm(alpha) else 0.0)
                  
  if (p_constraint < 1e-15) {
    return(list(error = "The constraint interval has negligible probability."))
  }
  
  prob_gt_t <- NULL
  if (!is.null(t)) {
    h_val <- -(t - mu_x) / sig_x
    p_num_high <- if (beta != Inf) bivariate_normal_cdf(h_val, beta, -rho) else pnorm(h_val)
    p_num_low <- if (alpha != -Inf) bivariate_normal_cdf(h_val, alpha, -rho) else 0.0
    prob_gt_t <- (p_num_high - p_num_low) / p_constraint
  }
  
  phi_alpha <- if (alpha != -Inf) dnorm(alpha) else 0
  phi_beta <- if (beta != Inf) dnorm(beta) else 0
  
  ratio_mean <- (phi_alpha - phi_beta) / p_constraint
  mean_x_cond <- mu_x + (cov_xs / sig_s) * ratio_mean
  
  residual_var <- var_x * (1 - rho^2)
  
  term_alpha <- if (alpha != -Inf) alpha * phi_alpha else 0
  term_beta <- if (beta != Inf) beta * phi_beta else 0
  ratio_var <- (term_alpha - term_beta) / p_constraint
  
  var_s_cond <- var_s * (1 + ratio_var - ratio_mean^2)
  explained_var_cond <- (cov_xs / var_s)^2 * var_s_cond
  var_x_cond <- max(0.0, residual_var + explained_var_cond)
  
  list(
    params = list(mu_x=mu_x, sig_x=sig_x, sig_omega=sig_omega, cx=cx, comega=comega, a=a, b=b, t=t),
    stats = list(
      p_constraint = p_constraint,
      p_x_gt_t_cond = prob_gt_t,
      e_x_cond = mean_x_cond,
      var_x_cond = var_x_cond,
      std_x_cond = sqrt(var_x_cond),
      rho_xs = rho
    )
  )
}

#' TruncBivariateNormal
#' 
#' @export
TruncBivariateNormal <- R6::R6Class("TruncBivariateNormal",
  public = list(
    a_coeff = NULL,
    b_coeff = NULL,
    L = NULL,
    U = NULL,
    sig_omega = NULL,
    sig_x = NULL,
    
    initialize = function(a_coeff, b_coeff, L, U, sig_omega, sig_x=1.0, theta=0.0) {
      self$a_coeff <- a_coeff
      self$b_coeff <- b_coeff
      self$L <- L
      self$U <- U
      self$sig_omega <- sig_omega
      self$sig_x <- sig_x
    },
    
    .get_stats = function(theta, x=NULL) {
      compute_gaussian_conditional_stats(
        mu_x = theta * self$sig_x^2, sig_x = self$sig_x, sig_omega = self$sig_omega,
        cx = self$a_coeff, comega = self$b_coeff, a = self$L, b = self$U, t = x
      )
    },
    
    ccdf = function(theta, x=NULL, gamma=0) {
      if (is.null(x)) stop("ccdf requires an observation x for TruncBivariateNormal")
      stats <- self$.get_stats(theta, x)
      if (!is.null(stats$error)) {
        mu_x <- theta * self$sig_x^2
        if (mu_x > x) return(1.0) else return(0.0)
      }
      stats$stats$p_x_gt_t_cond
    },
    
    cdf = function(theta, x=NULL, gamma=1) {
      if (is.null(x)) stop("cdf requires an observation x for TruncBivariateNormal")
      ccdf_val <- self$ccdf(theta, x)
      if (is.nan(ccdf_val)) return(NaN)
      1.0 - ccdf_val
    },
    
    E = function(theta, func) {
      stats <- self$.get_stats(theta)
      if (!is.null(stats$error)) return(NaN)
      
      test_val <- c(0.0, 1.0)
      f_val <- as.matrix(func(test_val))
      
      mean <- stats$stats$e_x_cond
      variance <- stats$stats$var_x_cond
      
      if (ncol(f_val) == 1 && all(abs(f_val - test_val) < 1e-12)) {
        return(mean)
      } else if (ncol(f_val) == 2 && nrow(f_val) == 2 && all(abs(f_val[,1] - test_val) < 1e-12) && all(abs(f_val[,2] - test_val^2) < 1e-12)) {
        return(c(mean, variance + mean^2))
      }
      stop("TruncBivariateNormal.E only supports identity or [x, x^2] functions.")
    },
    
    Var = function(theta, func) {
      stats <- self$.get_stats(theta)
      if (!is.null(stats$error)) return(NaN)
      
      test_val <- c(0.0, 1.0)
      f_val <- as.matrix(func(test_val))
      
      if (ncol(f_val) == 1 && all(abs(f_val - test_val) < 1e-12)) {
        return(stats$stats$var_x_cond)
      }
      stop("TruncBivariateNormal.Var only supports the identity function.")
    },
    
    equal_tailed_interval = function(observed, alpha=0.05, tol=1e-6) {
      theta_est <- observed / self$sig_x^2
      margin <- 40.0 / self$sig_x
      lb <- theta_est - margin
      ub <- theta_est + margin
      
      F_fun <- function(th) self$cdf(th, observed)
      
      L_root <- tryCatch({
        uniroot(function(th) F_fun(th) - (1.0 - 0.5 * alpha), lower = lb, upper = ub, tol = tol)$root
      }, error = function(e) NaN)
      
      U_root <- tryCatch({
        uniroot(function(th) F_fun(th) - (0.5 * alpha), lower = lb, upper = ub, tol = tol)$root
      }, error = function(e) NaN)
      
      c(L_root, U_root)
    }
  )
)


check_kkt <- function(beta_hat, G_hat, L, U, D, tol=1e-5) {
  g <- -G_hat
  n <- length(beta_hat)
  
  if (is.null(L)) L <- rep(-Inf, n)
  if (is.null(U)) U <- rep(Inf, n)
  
  for (j in seq_len(n)) {
    if (abs(L[j] - U[j]) <= tol) next
    
    bj <- beta_hat[j]
    gj <- g[j]
    dj <- D[j]
    
    if (abs(bj) > tol) {
      subgrad_l1 <- dj * sign(bj)
    } else {
      subgrad_l1_min <- -dj
      subgrad_l1_max <- dj
    }
    
    if (bj > L[j] + tol && bj < U[j] - tol) {
      if (abs(bj) > tol) {
        if (abs(gj - subgrad_l1) > tol) return(FALSE)
      } else {
        if (gj < subgrad_l1_min - tol || gj > subgrad_l1_max + tol) return(FALSE)
      }
    } else if (bj >= U[j] - tol) {
      if (abs(bj) > tol) {
        if (gj < subgrad_l1 - tol) return(FALSE)
      } else {
        if (gj < subgrad_l1_min - tol) return(FALSE)
      }
    } else if (bj <= L[j] + tol) {
      if (abs(bj) > tol) {
        if (gj > subgrad_l1 + tol) return(FALSE)
      } else {
        if (gj > subgrad_l1_max + tol) return(FALSE)
      }
    }
  }
  return(TRUE)
}

prox_lasso_bounds <- function(v, t, D, L, U) {
  n <- length(v)
  if (is.null(L)) L <- rep(-Inf, n)
  if (is.null(U)) U <- rep(Inf, n)
  
  st_val <- sign(v) * pmax(abs(v) - t * D, 0.0)
  pmin(pmax(st_val, L), U)
}


#' Lasso Inference
#'
#' @description Computes post-selection inference for the lasso.
#'
#' @param beta_hat The lasso solution.
#' @param G_hat The gradient at beta_hat.
#' @param Q_hat The Hessian / design matrix crossprod.
#' @param D The penalty weights.
#' @param L Lower bounds.
#' @param U Upper bounds.
#' @param Z_full The unpenalized score.
#' @param Sigma Covariance of Z_full.
#' @param Sigma_noisy Covariance of Z_noisy.
#' @param tol Tolerance for active set.
#'
#' @return A data frame with inference results.
#' @export
lasso_inference <- function(beta_hat, G_hat, Q_hat, D, L=NULL, U=NULL, Z_full, Sigma, Sigma_noisy, tol = 1e-6) {
  
  n <- length(beta_hat)
  if (is.null(L)) L <- rep(-Inf, n)
  if (is.null(U)) U <- rep(Inf, n)

  # 1. Estimate largest singular value of Q_hat using power method
  v_pow <- rnorm(n)
  v_pow <- v_pow / sqrt(sum(v_pow^2))
  for (i in 1:10) {
    Q_v <- as.vector(Q_hat %*% v_pow)
    lambda_max <- sqrt(sum(Q_v^2))
    v_pow <- Q_v / lambda_max
  }
  
  step_size <- 1.0 / (20.0 * lambda_max)
  
  v_step <- beta_hat - step_size * G_hat
  beta_new <- prox_lasso_bounds(v_step, step_size, D, L, U)
  
  beta_diff <- beta_new - beta_hat
  G_hat <- G_hat + (1.0 / step_size) * beta_diff
  beta_hat <- beta_new
  
  # check_kkt(beta_hat, G_hat, L, U, D, tol=1e-4) # optional debug check
  
  # Get constraints from C++ wrapper
  if (!exists("lasso_post_selection_constraints", envir = parent.frame()) && 
      !exists("lasso_post_selection_constraints", envir = globalenv())) {
    # If using Rcpp modules, it's typically loaded into the package namespace.
  }
  
  constraints <- lasso_post_selection_constraints(
    beta_hat, G_hat, Q_hat, D, L, U, tol
  )
  
  E <- constraints$E
  if (length(E) == 0) {
    return(data.frame(index=integer(), beta_hat=numeric(), lower_conf=numeric(), upper_conf=numeric(), p_value=numeric()))
  }
  
  E_r <- E + 1
  Z_noisy <- -G_hat + as.vector(Q_hat %*% beta_hat)
  
  si <- new(SelectiveInference, Z_full, Z_noisy, Sigma, Sigma_noisy)
  
  Q_EE <- Q_hat[E_r, E_r, drop=FALSE]
  W <- solve(Q_EE)
  
  A <- constraints$A_dense
  b <- constraints$b
  
  results <- list()
  
  for (k in seq_along(E_r)) {
    j <- E_r[k]
    
    v <- rep(0, nrow(Q_hat))
    v[E_r] <- W[, k]
    
    theta_hat <- sum(v * Z_full)
    variance <- sum(v * (Sigma %*% v))
    sigma <- sqrt(variance)
    
    params <- si$compute_params(v)
    bar_s <- params$bar_s
    
    interval <- si$get_interval(v, 0.0, A, b)
    L_0 <- interval[[1]]
    U_0 <- interval[[2]]
    
    c1 <- variance
    c2 <- bar_s^2
    a_coeff <- c2 / c1
    b_coeff <- 1.0
    
    tbn <- TruncBivariateNormal$new(
      a_coeff = a_coeff, b_coeff = b_coeff,
      L = L_0, U = U_0,
      sig_omega = bar_s, sig_x = sigma
    )
    
    L_U_theta <- tbn$equal_tailed_interval(theta_hat, alpha = 0.05)
    lower <- L_U_theta[1] * c1
    upper <- L_U_theta[2] * c1
    
    cdf_val <- tbn$cdf(theta = 0.0, x = theta_hat)
    cdf_val <- min(max(cdf_val, 0.0), 1.0)
    pval <- min(max(2 * min(cdf_val, 1.0 - cdf_val), 0.0), 1.0)
    
    results[[k]] <- data.frame(
      index = E[k], 
      beta_hat = beta_hat[j],
      lower_conf = lower,
      upper_conf = upper,
      p_value = pval
    )
  }
  
  return(do.call(rbind, results))
}