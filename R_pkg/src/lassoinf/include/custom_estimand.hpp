#pragma once

// Geometry for selective inference on user-specified one-dimensional
// estimands after the LASSO; mirrors python/lassoinf/custom_estimand.py.
// Truncated-normal inference itself is done by the Python / R front ends.

#include "affine_constraints.hpp"

#include <string>
#include <unordered_map>

namespace lassoinf {

// theta_hat = eta' Z + offset
struct ContrastEstimand {
    Eigen::VectorXd eta;
    double offset = 0.0;

    double value(const Eigen::VectorXd& Z) const { return eta.dot(Z) + offset; }
};

// Coordinates of the selection constraints:
//   basis "refit": (bar_beta_E, U_{-E}) = (W Z_E, Z_{-E} - Q_{-E,E} W Z_E)
//   basis "lasso": (beta_hat_E, -grad(beta_hat)_{-E}) as affine maps on the selection event
// Everything uses Q matvecs (Q symmetric).
class SelectionCoordinates {
public:
    // Z_noisy: data used for selection; beta_hat / G_hat: LASSO solution and gradient.
    // Q_diag: optional diagonal of Q (empty = unknown; taken from a DenseOperator).
    SelectionCoordinates(std::shared_ptr<InactiveScoreOperator> score,
                         const Eigen::VectorXd& Z_noisy,
                         const Eigen::VectorXd& beta_hat,
                         const Eigen::VectorXd& G_hat,
                         Eigen::VectorXd Q_diag = Eigen::VectorXd());

    const std::vector<int>& E() const { return score_->E(); }
    const std::vector<int>& E_c() const { return score_->E_c(); }
    const std::shared_ptr<InactiveScoreOperator>& score_operator() const { return score_; }
    const Eigen::VectorXd& lasso_coef_offset() const { return lasso_coef_offset_; }
    const Eigen::VectorXd& lasso_score_offset() const { return lasso_score_offset_; }

    Eigen::VectorXd refit_coef(const Eigen::VectorXd& Z) const { return score_->coef(Z); }
    Eigen::VectorXd refit_score(const Eigen::VectorXd& Z) const { return score_->multiply(Z); }

    // eta with eta' Z = a_E' bar_beta_E + a_Ec' U_{-E}; empty vectors mean zero
    Eigen::VectorXd contrast(const Eigen::VectorXd& a_E, const Eigen::VectorXd& a_Ec) const;
    ContrastEstimand estimand(const Eigen::VectorXd& a_E, const Eigen::VectorXd& a_Ec,
                              const std::string& basis = "refit") const;

    ContrastEstimand inactive_score(int j, const std::string& basis = "refit") const;
    ContrastEstimand inactive_coef(int j, const std::string& basis = "refit") const;

    // S_jj = Q_jj - Q_{j,E} W Q_{E,j}, cached; one matvec per variable, or |E|
    // in total when diag(Q) is known and there are more than |E| variables
    Eigen::VectorXd inactive_S(const std::vector<int>& variables) const;

private:
    int inactive_position(int j) const;

    std::shared_ptr<InactiveScoreOperator> score_;
    Eigen::VectorXd Q_diag_;
    Eigen::VectorXd lasso_coef_offset_;
    Eigen::VectorXd lasso_score_offset_;
    std::unordered_map<int, int> pos_Ec_;
    mutable std::unordered_map<int, double> S_cache_;
};

// Screening of inactive variables on the gradient at the LASSO solution,
// G = grad(beta_hat)_{-E}, conditioning on kept variables and their signs:
//   threshold >= 0: keep |G_j| > threshold
//   top_k >= 1: keep the top_k by |G_j|; conditioning "first_dropped" (default)
//     also conditions on the largest dropped variable and its sign, "exact"
//     uses all (kept, dropped) pairs.
class ScreenedSelection {
public:
    ScreenedSelection(const std::shared_ptr<LinearOperator>& A_lasso,
                      const Eigen::VectorXd& b_lasso,
                      const SelectionCoordinates& coords,
                      const Eigen::VectorXd& G_hat,
                      const Eigen::VectorXd& Z_noisy,
                      double threshold = std::numeric_limits<double>::quiet_NaN(),
                      int top_k = -1,
                      const std::string& conditioning = "first_dropped");

    std::vector<int> screened;          // by decreasing |G_j|
    Eigen::VectorXd screened_signs;
    int first_dropped = -1;             // -1 unless top_k with first_dropped
    double first_dropped_sign = 0.0;
    bool observed_feasible = true;

    std::shared_ptr<LinearOperator> A_screen;
    Eigen::VectorXd b_screen;
    std::shared_ptr<LinearOperator> A;  // [A_lasso; A_screen]
    Eigen::VectorXd b;
};

} // namespace lassoinf
