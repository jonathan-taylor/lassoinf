#pragma once

// Selective inference after the uniLasso, or unireg (its lam = 0 case), for the
// coefficients of the full model.
//
// In score coordinates (Z = X'y, Q = X'X) with selection data Y = Z + omega, the uniLasso
// solves
//
//     minimize 1/2 beta' Q beta - beta' Y + sum_j D_j |beta_j|,  D_j = lam_j / |C_j|,
//     subject to sign(beta_j) in {0, s_j},
//
// with C the univariate coefficients (Y / diag(Q) for the gaussian family) and s = sign(C).
// For the target of coordinate k, along the line the polyhedral lemma conditions on, only
// C_k moves, and every constraint row reads
//
//     u_i + a_i w <= b0_i + m_i / C(w),   C(w) = c + gamma w,   s C(w) > 0.
//
// On the half-line s C(w) > 0 that is a quadratic inequality, so each row holds on at most
// two intervals, and the truncation set is a finite union of intervals.
// See docs/data_dependent_penalty.md.

#include <Eigen/Dense>
#include <memory>
#include "affine_constraints.hpp"

namespace lassoinf {

struct UniLassoPenalty {
    Eigen::VectorXd C;  // univariate coefficients
    Eigen::VectorXd s;  // their signs
    Eigen::VectorXd D;  // penalty weights lam / |C|
    Eigen::VectorXd L;  // bounds from the sign constraints
    Eigen::VectorXd U;
};

// lam has length 1 (shared) or length(C) (per feature)
UniLassoPenalty unilasso_penalty(const Eigen::VectorXd& C, const Eigen::VectorXd& lam);

// The uniLasso solution for the selection data Z_noisy, by coordinate descent, with the
// gaussian univariate coefficients C = Z_noisy / diag(Q).
Eigen::VectorXd unilasso_fit(const Eigen::MatrixXd& Q, const Eigen::VectorXd& Z_noisy,
                             const Eigen::VectorXd& lam, double tol = 1e-12,
                             int max_iter = 100000);

// The largest violation of the uniLasso's KKT conditions by beta_hat, for the selection
// data Z_noisy and lam in the units of Z_noisy (C = Z_noisy / diag(Q)).
double unilasso_kkt_violation(const Eigen::VectorXd& beta_hat, const Eigen::MatrixXd& Q,
                              const Eigen::VectorXd& Z_noisy, const Eigen::VectorXd& lam);

// kappa_j for a leave-one-out uniLasso fit beta_hat on centered X and y (univariate fits
// with intercepts): beta_hat solves the plain uniLasso with penalty
// (n lam + kappa_j) / |b_uni_j| on feature j. kappa_j = 2 sigma^2 + O_p(n^{-1/2}).
Eigen::VectorXd unilasso_loo_kappa(const Eigen::MatrixXd& X, const Eigen::VectorXd& y,
                                   const Eigen::VectorXd& beta_hat);

// {w : q2 w^2 + q1 w + q0 <= 0} for each row, as two intervals per row, the columns
// (lo1, hi1, lo2, hi2); a missing interval has lo > hi.
Eigen::MatrixXd quadratic_le_zero(const Eigen::VectorXd& q2, const Eigen::VectorXd& q1,
                                  const Eigen::VectorXd& q0);

// For each row i, the set of w with s (c + gamma w) > 0 and
// u_i + a_i w <= b0_i + m_i / (c + gamma w), as two intervals per row (lo1, hi1, lo2, hi2).
Eigen::MatrixXd unilasso_row_sets(const Eigen::VectorXd& u, const Eigen::VectorXd& a,
                                  const Eigen::VectorXd& b0, const Eigen::VectorXd& m,
                                  double c, double gamma, double s);

// The intersection over rows of the unions from unilasso_row_sets, as a (k, 2) matrix of
// disjoint intervals in increasing order.
Eigen::MatrixXd intersect_row_sets(const Eigen::MatrixXd& sets);

// The truncation set, in w, for the target of coordinate k of a uniLasso solution.
//
// beta_hat, G_hat, Q, D, L, U, tol : the solution, the gradient Q beta_hat - Y and the
//     penalty and bounds at the observed data, as for lasso_post_selection_constraints
// lam_k : the penalty of coordinate k, D_k = lam_k / |C_k|
// C_k : the observed univariate coefficient of coordinate k
// y0, direction : the selection data along the line, Y(w) = y0 + direction w
// w_obs : the observed w
// gamma : the slope of C_k in w
//
// b is affine in D, and only D_k moves, so every row is (A Y)_i <= b0_i + M_ik lam_k s_k / C_k.
Eigen::MatrixXd unilasso_truncation_set(const Eigen::VectorXd& beta_hat,
                                        const Eigen::VectorXd& G_hat,
                                        std::shared_ptr<LinearOperator> Q,
                                        const Eigen::VectorXd& D,
                                        const Eigen::VectorXd& L,
                                        const Eigen::VectorXd& U,
                                        int k,
                                        double lam_k,
                                        double C_k,
                                        const Eigen::VectorXd& y0,
                                        const Eigen::VectorXd& direction,
                                        double w_obs,
                                        double gamma,
                                        double tol = 1e-6);

} // namespace lassoinf
