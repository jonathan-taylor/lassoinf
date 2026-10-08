#' lassoinf: Selective Inference for the Randomized Lasso
#'
#' @description Post-selection confidence intervals and p-values for the
#' randomized lasso. Start with \code{\link{LassoInference}}; see
#' \code{vignette("gaussian_lasso", package = "lassoinf")}.
#'
#' @references
#' Lee, J. D., Sun, D. L., Sun, Y. and Taylor, J. E. (2016). Exact
#' post-selection inference, with application to the lasso.
#' \emph{The Annals of Statistics}, 44(3), 907--927.
#' \doi{10.1214/15-AOS1371}
#'
#' Tian, X. and Taylor, J. (2018). Selective inference with a randomized
#' response. \emph{The Annals of Statistics}, 46(2), 679--710.
#' \doi{10.1214/17-AOS1564}
#'
#' Panigrahi, S., Fry, K. and Taylor, J. (2024). Exact selective inference
#' with randomization. \emph{Biometrika}, 111(4), 1109--1127.
#' \doi{10.1093/biomet/asae019}
#'
#' Liu, S. (2023). An exact sampler for inference after polyhedral model
#' selection. \emph{arXiv preprint}. \doi{10.48550/arXiv.2308.10346}
#'
#' @keywords internal
#' @useDynLib lassoinf, .registration = TRUE
#' @importFrom Rcpp loadModule
#' @importFrom methods new
#' @importFrom R6 R6Class
#' @importFrom stats dnorm plogis pnorm qnorm rnorm uniroot
"_PACKAGE"

#' C++ classes
#'
#' @description Classes exposed from the C++ backend through an Rcpp module;
#' construct them with \code{new()}.
#' \describe{
#'   \item{\code{new(AffineConstraints, Z, Z_noisy, Sigma, Sigma_noise, scalar_noise = NaN)}}{
#'     The randomized selection setup: full-data score \code{Z}, the noisy
#'     score used for selection, and their covariances (matrices or
#'     \code{LinearOp}s; \code{Sigma_noise} may be \code{NULL}, in which case
#'     \code{scalar_noise} scales \code{Sigma}). Methods
#'     \code{compute_contrast(v)}, \code{compute_covariance_contrast(...)} and
#'     \code{solve_contrast(...)}.}
#'   \item{\code{new(DiscreteFamily, sufficient_stat, weights, theta)}}{
#'     Exponential family with reference measure
#'     \code{sum_j weights[j] delta(sufficient_stat[j])}. Methods \code{pdf},
#'     \code{cdf}, \code{ccdf}, \code{equal_tailed_interval},
#'     \code{get_theta}, \code{set_theta} and \code{get_partition}.}
#'   \item{\code{new(TruncatedGaussian, estimate, sigma, smoothing_sigma, lower_bound, upper_bound, noisy_estimate, factor)}}{
#'     Selection weight of a Gaussian estimate with randomized truncation;
#'     method \code{weight(x)}.}
#'   \item{\code{new(WeightedGaussianFamily, estimate, sigma, weight_fns, num_sd, num_grid)}}{
#'     A Gaussian reweighted by the product of the functions in the list
#'     \code{weight_fns}, discretized on \code{num_grid} points within
#'     \code{num_sd} standard deviations. Methods
#'     \code{pvalue(null_value, alternative, basept)} and
#'     \code{interval(basept, level)}.}
#' }
#' @name AffineConstraints
#' @aliases DiscreteFamily TruncatedGaussian WeightedGaussianFamily
#' @examples
#' grid <- seq(-5, 5, length.out = 201)
#' fam <- new(DiscreteFamily, grid, dnorm(grid), 0)
#' fam$cdf(0, 1, 0)
#' @export AffineConstraints DiscreteFamily TruncatedGaussian WeightedGaussianFamily
NULL
