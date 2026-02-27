#' @useDynLib lassoinf, .registration=TRUE
#' @importFrom Rcpp loadModule
NULL

loadModule("lassoinf_cpp", TRUE)

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
lasso_inference <- function(beta_hat, G_hat, Q_hat, D, L, U, Z_full, Sigma, Sigma_noisy, tol = 1e-6) {
  
  # Ensure lassoinf_cpp module is available
  if (!exists("lasso_post_selection_constraints", envir = parent.frame()) && 
      !exists("lasso_post_selection_constraints", envir = globalenv())) {
    # If using Rcpp modules, it's typically loaded into the package namespace.
    # We will assume lassoinf_cpp is loaded.
  }
  
  # Get constraints from C++ wrapper
  constraints <- lasso_post_selection_constraints(
    beta_hat, G_hat, Q_hat, D, L, U, tol
  )
  
  E <- constraints$E
  if (length(E) == 0) {
    return(data.frame(index=integer(), beta_hat=numeric(), lower_conf=numeric(), upper_conf=numeric(), p_value=numeric()))
  }
  
  # C++ indices are 0-based, R is 1-based.
  E_r <- E + 1
  
  # Construct Z_noisy = -G_hat + Q_hat %*% beta_hat
  Z_noisy <- -G_hat + as.vector(Q_hat %*% beta_hat)
  
  # Initialize SelectiveInference C++ object from module
  si <- new(SelectiveInference, Z_full, Z_noisy, Sigma, Sigma_noisy)
  
  # Compute W = inverse(Q_EE)
  Q_EE <- Q_hat[E_r, E_r, drop=FALSE]
  W <- solve(Q_EE)
  
  # Extract dense A
  A <- constraints$A_dense
  b <- constraints$b
  
  results <- list()
  
  for (k in seq_along(E_r)) {
    j <- E_r[k]
    
    # Contrast vector
    v <- rep(0, nrow(Q_hat))
    v[E_r] <- W[, k]
    
    # Target estimate
    theta_hat <- sum(v * Z_full)
    
    # Variance
    variance <- sum(v * (Sigma %*% v))
    sigma <- sqrt(variance)
    
    # Weight function
    params <- si$compute_params(v)
    bar_s <- params$bar_s
    
    get_weight <- function(t_vec) {
      weights <- numeric(length(t_vec))
      for (i in seq_along(t_vec)) {
        t_val <- t_vec[i]
        interval <- si$get_interval(v, t_val, A, b)
        lower_bound <- interval[[1]]
        upper_bound <- interval[[2]]
        
        if (is.nan(lower_bound) || is.nan(upper_bound) || lower_bound > upper_bound) {
          weights[i] <- 0
        } else {
          weights[i] <- pnorm(upper_bound / bar_s) - pnorm(lower_bound / bar_s)
        }
      }
      return(weights)
    }
    
    # Grid for numerical integration
    num_sd <- 10
    num_grid <- 4000
    grid <- seq(theta_hat - num_sd * sigma, theta_hat + num_sd * sigma, length.out = num_grid)
    w <- get_weight(grid)
    
    compute_cdf <- function(theta) {
      log_w <- log(pmax(w, 1e-16))
      log_w <- log_w - max(log_w) - 10
      dens <- exp(log_w) * dnorm(grid, mean = theta, sd = sigma)
      dens <- dens / sum(dens)
      cdf <- cumsum(dens)
      obs_idx <- which.min(abs(grid - theta_hat))
      return(cdf[obs_idx])
    }
    
    # P-value for H0: theta = 0
    cdf_null <- compute_cdf(0)
    pval <- 2 * min(cdf_null, 1 - cdf_null)
    
    # Confidence intervals (Root finding)
    alpha <- 0.05
    
    find_root <- function(target) {
      # We search in [theta_hat - 10*sigma, theta_hat + 10*sigma]
      res <- tryCatch({
        uniroot(function(theta) compute_cdf(theta) - target, 
                lower = theta_hat - 10 * sigma, 
                upper = theta_hat + 10 * sigma)$root
      }, error = function(e) NA)
      return(res)
    }
    
    lower_conf <- find_root(1 - alpha / 2)
    upper_conf <- find_root(alpha / 2)
    
    results[[k]] <- data.frame(
      index = E[k], # keep 0-based for consistency with C++
      beta_hat = beta_hat[j],
      lower_conf = lower_conf,
      upper_conf = upper_conf,
      p_value = pval
    )
  }
  
  return(do.call(rbind, results))
}
