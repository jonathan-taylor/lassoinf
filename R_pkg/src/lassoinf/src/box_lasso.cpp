#include "box_lasso.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace lassoinf {

namespace {

void check_sizes(const Eigen::MatrixXd& Q, const Eigen::VectorXd& Y, const Eigen::VectorXd& D,
                 const Eigen::VectorXd& L, const Eigen::VectorXd& U) {
    const Eigen::Index p = Y.size();
    if (Q.rows() != p || Q.cols() != p || D.size() != p || L.size() != p || U.size() != p)
        throw std::invalid_argument("box_lasso: Q must be p x p and Y, D, L, U of length p");
    if ((D.array() < 0).any()) throw std::invalid_argument("box_lasso: D must be >= 0");
    if ((L.array() > U.array()).any()) throw std::invalid_argument("box_lasso: L must be <= U");
}

double soft(double r, double d) {
    return r > d ? r - d : (r < -d ? r + d : 0.0);
}

}  // namespace

Eigen::VectorXd box_lasso_fit(const Eigen::MatrixXd& Q, const Eigen::VectorXd& Y,
                              const Eigen::VectorXd& D, const Eigen::VectorXd& L,
                              const Eigen::VectorXd& U, const Eigen::VectorXd& beta0,
                              double tol, int max_iter) {
    check_sizes(Q, Y, D, L, U);
    const Eigen::Index p = Y.size();
    Eigen::VectorXd dq = Q.diagonal();
    if ((dq.array() <= 0).any()) throw std::invalid_argument("box_lasso: diag(Q) must be > 0");
    Eigen::VectorXd beta = beta0.size() == p ? beta0 : Eigen::VectorXd::Zero(p);
    beta = beta.cwiseMax(L).cwiseMin(U);
    Eigen::VectorXd Qbeta = Q * beta;
    for (int it = 0; it < max_iter; ++it) {
        double delta = 0.0;
        for (Eigen::Index j = 0; j < p; ++j) {
            if (L(j) == U(j)) continue;
            double r = Y(j) - Qbeta(j) + dq(j) * beta(j);
            double next = std::min(std::max(soft(r, D(j)) / dq(j), L(j)), U(j));
            double step = next - beta(j);
            if (step != 0.0) {
                Qbeta += step * Q.col(j);
                beta(j) = next;
                delta = std::max(delta, std::abs(step) * std::sqrt(dq(j)));
            }
        }
        if (delta < tol) break;
    }
    return beta;
}

double box_lasso_kkt_violation(const Eigen::VectorXd& beta_hat, const Eigen::MatrixXd& Q,
                               const Eigen::VectorXd& Y, const Eigen::VectorXd& D,
                               const Eigen::VectorXd& L, const Eigen::VectorXd& U) {
    check_sizes(Q, Y, D, L, U);
    Eigen::VectorXd g = Y - Q * beta_hat;   // minus the gradient of the loss
    double v = 0.0;
    for (Eigen::Index j = 0; j < beta_hat.size(); ++j) {
        if (L(j) == U(j)) continue;
        double b = beta_hat(j);
        // the subgradient of D_j |b| at b, as an interval [lo, hi]
        double lo = b > 0 ? D(j) : -D(j);
        double hi = b < 0 ? -D(j) : D(j);
        // a bound active in the direction g pushes adds a normal cone
        if (b >= U(j)) hi = INFINITY;
        if (b <= L(j)) lo = -INFINITY;
        v = std::max(v, std::max(lo - g(j), g(j) - hi));
    }
    return v;
}

}  // namespace lassoinf
