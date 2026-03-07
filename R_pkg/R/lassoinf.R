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

#' TruncBivariateNormal Class
#'
#' @description Evaluates exact bounds and inference parameters for the selectively truncated bivariate normal distribution.
#'
#' @export
TruncBivariateNormal <- R6::R6Class("TruncBivariateNormal",
  public = list(
    #' @field a_coeff Internal coefficient.
    a_coeff = NULL,
    #' @field b_coeff Internal coefficient.
    b_coeff = NULL,
    #' @field L Lower bound of the truncation interval.
    L = NULL,
    #' @field U Upper bound of the truncation interval.
    U = NULL,
    #' @field sig_omega Standard deviation of the noise component.
    sig_omega = NULL,
    #' @field sig_x Standard deviation of the target statistic.
    sig_x = NULL,
    
    #' @description
    #' Initialize the TruncBivariateNormal object.
    #' @param a_coeff The 'a' coefficient for the conditional statistic.
    #' @param b_coeff The 'b' coefficient for the noise statistic.
    #' @param L Lower bound of the condition.
    #' @param U Upper bound of the condition.
    #' @param sig_omega Noise standard deviation.
    #' @param sig_x Target statistic standard deviation. Default is 1.0.
    #' @param theta True underlying parameter. Default is 0.0.
    initialize = function(a_coeff, b_coeff, L, U, sig_omega, sig_x=1.0, theta=0.0) {
      self$a_coeff <- a_coeff
      self$b_coeff <- b_coeff
      self$L <- L
      self$U <- U
      self$sig_omega <- sig_omega
      self$sig_x <- sig_x
    },
    
    #' @description
    #' Get the computed statistics given the parameter.
    #' @param theta True parameter value.
    #' @param x Observed value. Default is NULL.
    #' @return A list containing the computed stats.
    .get_stats = function(theta, x=NULL) {
      compute_gaussian_conditional_stats(
        mu_x = theta * self$sig_x^2, sig_x = self$sig_x, sig_omega = self$sig_omega,
        cx = self$a_coeff, comega = self$b_coeff, a = self$L, b = self$U, t = x
      )
    },
    
    #' @description
    #' Complementary CDF.
    #' @param theta Parameter.
    #' @param x Observed value.
    #' @param gamma Ignored.
    #' @return The complementary CDF value.
    ccdf = function(theta, x=NULL, gamma=0) {
      if (is.null(x)) stop("ccdf requires an observation x for TruncBivariateNormal")
      stats <- self$.get_stats(theta, x)
      if (!is.null(stats$error)) {
        mu_x <- theta * self$sig_x^2
        if (mu_x > x) return(1.0) else return(0.0)
      }
      stats$stats$p_x_gt_t_cond
    },
    
    #' @description
    #' Cumulative Distribution Function.
    #' @param theta Parameter.
    #' @param x Observed value.
    #' @param gamma Ignored.
    #' @return The CDF value.
    cdf = function(theta, x=NULL, gamma=1) {
      if (is.null(x)) stop("cdf requires an observation x for TruncBivariateNormal")
      ccdf_val <- self$ccdf(theta, x)
      if (is.nan(ccdf_val)) return(NaN)
      1.0 - ccdf_val
    },
    
    #' @description
    #' Expected value.
    #' @param theta Parameter.
    #' @param func A function returning identity or x^2.
    #' @return Expected value.
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
    
    #' @description
    #' Variance.
    #' @param theta Parameter.
    #' @param func Identity function.
    #' @return The conditional variance.
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
    
    #' @description
    #' Find the equal tailed confidence interval by inverting the CDF.
    #' @param observed The observed value.
    #' @param alpha Type-I error rate. Default is 0.05.
    #' @param tol Tolerance for uniroot.
    #' @return A length 2 vector containing the lower and upper bounds.
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


#' LassoInference Class
#'
#' @description Computes post-selection inference for the lasso.
#' This class mirrors the Python `lassoinf.LassoInference` dataclass, 
#' providing identical parameters and functional parity.
#'
#' @export
LassoInference <- R6::R6Class("LassoInference",
  public = list(
    #' @field beta_hat The lasso solution from the randomized objective.
    beta_hat = NULL,
    #' @field G_hat The gradient at beta_hat of the unpenalized loss.
    G_hat = NULL,
    #' @field Q_hat The Hessian / design matrix crossprod.
    Q_hat = NULL,
    #' @field D The penalty weights.
    D = NULL,
    #' @field L Lower bounds.
    L = NULL,
    #' @field U Upper bounds.
    U = NULL,
    #' @field Z_full The unpenalized score on the full data.
    Z_full = NULL,
    #' @field Sigma Covariance matrix of Z_full.
    Sigma = NULL,
    #' @field Sigma_noise Covariance matrix of the randomized score Z_noisy.
    Sigma_noise = NULL,
    #' @field scalar_noise Scalar controlling variance of noise if Sigma_noise is not provided.
    scalar_noise = NaN,
    #' @field A Affine constraint matrix (from C++ backend).
    A = NULL,
    #' @field b Affine constraint vector (from C++ backend).
    b = NULL,
    #' @field E The active set (selected indices).
    E = NULL,
    #' @field E_c The inactive set.
    E_c = NULL,
    #' @field s_E Signs of the active set coefficients.
    s_E = NULL,
    #' @field v_Ec Values at the bounds for the inactive set.
    v_Ec = NULL,
    #' @field si AffineConstraints C++ wrapper instance.
    si = NULL,
    #' @field intervals Selective inference intervals.
    intervals = NULL,
    #' @field splitting Data splitting intervals.
    splitting = NULL,
    #' @field naive Naive intervals.
    naive = NULL,
    #' @field contrasts The contrast vectors for inference.
    contrasts = NULL,
    #' @field _naive Data frame storing naive results.
    `_naive` = NULL,
    #' @field _splitting Data frame storing data splitting results.
    `_splitting` = NULL,
    
    #' @description
    #' Initialize the LassoInference object.
    #' @param beta_hat The lasso solution.
    #' @param G_hat The gradient at beta_hat.
    #' @param Q_hat The Hessian / design matrix crossprod.
    #' @param D The penalty weights.
    #' @param L Lower bounds for beta. Default is \code{NULL} (equivalent to \code{-Inf}).
    #' @param U Upper bounds for beta. Default is \code{NULL} (equivalent to \code{Inf}).
    #' @param Z_full The unpenalized score.
    #' @param Sigma Covariance of Z_full.
    #' @param Sigma_noise Covariance of Z_noisy. Default is \code{NULL}.
    #' @param scalar_noise Variance scaling if Sigma_noise is NULL. Default is \code{NaN}.
    #' @param tol Tolerance for active set and KKT conditions. Default is \code{1e-6}.
    initialize = function(beta_hat, G_hat, Q_hat, D, L=NULL, U=NULL, Z_full, Sigma, Sigma_noise=NULL, scalar_noise=NaN, tol = 1e-6) {
      self$beta_hat <- beta_hat
      self$G_hat <- G_hat
      self$Q_hat <- Q_hat
      self$D <- D
      self$L <- L
      self$U <- U
      self$Z_full <- Z_full
      self$Sigma <- Sigma
      self$Sigma_noise <- Sigma_noise
      self$scalar_noise <- scalar_noise
      
      n <- length(self$beta_hat)
      if (is.null(self$L)) self$L <- rep(-Inf, n)
      if (is.null(self$U)) self$U <- rep(Inf, n)

      v_pow <- rnorm(n)
      v_pow <- v_pow / sqrt(sum(v_pow^2))
      for (i in 1:10) {
        Q_v <- as.vector(self$Q_hat %*% v_pow)
        lambda_max <- sqrt(sum(Q_v^2))
        v_pow <- Q_v / lambda_max
      }
      
      step_size <- 1.0 / (20.0 * lambda_max)
      
      v_step <- self$beta_hat - step_size * self$G_hat
      beta_new <- prox_lasso_bounds(v_step, step_size, self$D, self$L, self$U)
      
      beta_diff <- beta_new - self$beta_hat
      self$G_hat <- self$G_hat + (1.0 / step_size) * beta_diff
      self$beta_hat <- beta_new
      
      constraints <- lasso_post_selection_constraints(
        self$beta_hat, self$G_hat, self$Q_hat, self$D, self$L, self$U, tol
      )
      
      self$A <- constraints$A_dense
      self$b <- constraints$b
      self$E <- constraints$E
      self$E_c <- constraints$E_c
      self$s_E <- constraints$s_E
      self$v_Ec <- constraints$v_Ec
      
      Z_noisy <- -self$G_hat + as.vector(self$Q_hat %*% self$beta_hat)
      
      self$si <- new(AffineConstraints, self$Z_full, Z_noisy, self$Sigma, self$Sigma_noise, self$scalar_noise)
      
      self$intervals <- list()
      self$splitting <- list()
      self$naive <- list()
      self$contrasts <- list()
      
      if (length(self$E) > 0) {
        E_r <- self$E + 1
        Q_EE <- self$Q_hat[E_r, E_r, drop=FALSE]
        W <- solve(Q_EE)
        
        for (k in seq_along(E_r)) {
          j <- E_r[k]
          v <- rep(0, nrow(self$Q_hat))
          v[E_r] <- W[, k]
          
          theta_hat <- sum(v * self$Z_full)
          variance <- sum(v * (self$Sigma %*% v))
          sigma <- sqrt(variance)
          
          contrast <- self$si$compute_contrast(v)
          bar_s <- contrast$bar_s
          
          interval <- contrast$get_interval(0.0, self$A, self$b)
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
          
          j_idx <- as.character(self$E[k])
          self$intervals[[j_idx]] <- list(lower, upper, pval)
          self$splitting[[j_idx]] <- list(contrast$splitting_estimator, contrast$splitting_variance)
          self$naive[[j_idx]] <- list(contrast$theta_hat, contrast$naive_variance)
          self$contrasts[[j_idx]] <- v
        }
      }
    },
    
    #' @description
    #' Summary of post-selection inference results.
    #' @return A data frame containing the parameter indices, estimates, selective confidence intervals, and selective p-values.
    summary = function() {
      alpha <- 0.05
      q <- qnorm(1 - alpha / 2)
      
      indices <- sort(self$E)
      indices_char <- as.character(indices)
      
      if (length(indices) == 0) {
        self$`_naive` <- data.frame(index=integer(), beta_hat=numeric(), lower_conf=numeric(), upper_conf=numeric(), p_value=numeric())
        self$`_splitting` <- data.frame(index=integer(), beta_hat=numeric(), lower_conf=numeric(), upper_conf=numeric(), p_value=numeric())
        return(data.frame(index=integer(), beta_hat=numeric(), lower_conf=numeric(), upper_conf=numeric(), p_value=numeric()))
      }
      
      est_n <- sapply(indices_char, function(idx) self$naive[[idx]][[1]])
      sd_n <- sqrt(sapply(indices_char, function(idx) self$naive[[idx]][[2]]))
      self$`_naive` <- data.frame(
        index = indices,
        beta_hat = est_n,
        lower_conf = est_n - q * sd_n,
        upper_conf = est_n + q * sd_n,
        p_value = 2 * pnorm(abs(est_n / sd_n), lower.tail = FALSE)
      )
      
      est_s <- sapply(indices_char, function(idx) self$splitting[[idx]][[1]])
      sd_s <- sqrt(sapply(indices_char, function(idx) self$splitting[[idx]][[2]]))
      self$`_splitting` <- data.frame(
        index = indices,
        beta_hat = est_s,
        lower_conf = est_s - q * sd_s,
        upper_conf = est_s + q * sd_s,
        p_value = 2 * pnorm(abs(est_s / sd_s), lower.tail = FALSE)
      )
      
      beta_vals <- sapply(indices, function(i) self$beta_hat[i + 1])
      lowers <- sapply(indices_char, function(idx) self$intervals[[idx]][[1]])
      uppers <- sapply(indices_char, function(idx) self$intervals[[idx]][[2]])
      p_vals <- sapply(indices_char, function(idx) self$intervals[[idx]][[3]])
      
      df <- data.frame(
        index = indices,
        beta_hat = beta_vals,
        lower_conf = lowers,
        upper_conf = uppers,
        p_value = p_vals
      )
      return(df)
    }
  )
)