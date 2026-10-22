#pragma once

// The lasso with box constraints, for a quadratic loss:
//
//     minimize 1/2 beta' Q beta - beta' Y + sum_j D_j |beta_j|,  subject to L <= beta <= U,
//
// with Q positive definite. This is the selection problem of LassoInference with
// Z_noisy = Y. With Q = Sigma^{-1} and Y = Sigma^{-1} beta_star it is the lasso in covariance
// form, 1/2 (beta_star - beta)' Sigma^{-1} (beta_star - beta) + sum_j D_j |beta_j|.
// L_j = U_j = 0 excludes coordinate j, and D_j = 0 leaves it unpenalized.

#include <Eigen/Dense>

namespace lassoinf {

// The solution by coordinate descent, started from beta0 (zero when empty). It stops when no
// coordinate moves by more than tol in the units of sqrt(Q_jj).
Eigen::VectorXd box_lasso_fit(const Eigen::MatrixXd& Q, const Eigen::VectorXd& Y,
                              const Eigen::VectorXd& D, const Eigen::VectorXd& L,
                              const Eigen::VectorXd& U,
                              const Eigen::VectorXd& beta0 = Eigen::VectorXd(),
                              double tol = 1e-12, int max_iter = 100000);

// The largest violation of the KKT conditions by beta_hat, in the units of Y. Coordinates
// with L_j = U_j are not checked.
double box_lasso_kkt_violation(const Eigen::VectorXd& beta_hat, const Eigen::MatrixXd& Q,
                               const Eigen::VectorXd& Y, const Eigen::VectorXd& D,
                               const Eigen::VectorXd& L, const Eigen::VectorXd& U);

}  // namespace lassoinf
