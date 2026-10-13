#include "unilasso.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace lassoinf {

using detail::kInf;

namespace {

// rows with |linear coefficient| below this are treated as constant, as in
// AffineConstraintsContrast::get_interval
const double kZeroTol = 1e-10;

// intervals narrower than this, relative to their endpoints, are rounding error
const double kSliver = 64 * std::numeric_limits<double>::epsilon();

// lam of length 1 or p, as a vector of length p
Eigen::VectorXd broadcast(const Eigen::VectorXd& lam, Eigen::Index p) {
    if (lam.size() == p) return lam;
    if (lam.size() == 1) return Eigen::VectorXd::Constant(p, lam(0));
    throw std::invalid_argument("lam must have length 1 or the number of features");
}

} // namespace

UniLassoPenalty unilasso_penalty(const Eigen::VectorXd& C, const Eigen::VectorXd& lam) {
    const Eigen::Index p = C.size();
    Eigen::VectorXd lam_p = broadcast(lam, p);
    UniLassoPenalty out;
    out.C = C;
    out.s.resize(p);
    out.D.resize(p);
    out.L.resize(p);
    out.U.resize(p);
    for (Eigen::Index j = 0; j < p; ++j) {
        double s = detail::sign(C(j));
        if (s == 0) throw std::invalid_argument("a univariate coefficient is exactly zero");
        out.s(j) = s;
        out.D(j) = lam_p(j) / std::abs(C(j));
        out.L(j) = s > 0 ? 0.0 : -kInf;
        out.U(j) = s > 0 ? kInf : 0.0;
    }
    return out;
}

Eigen::VectorXd unilasso_fit(const Eigen::MatrixXd& Q, const Eigen::VectorXd& Z_noisy,
                             const Eigen::VectorXd& lam, double tol, int max_iter) {
    const Eigen::Index p = Z_noisy.size();
    Eigen::VectorXd dq = Q.diagonal();
    UniLassoPenalty pen = unilasso_penalty(Z_noisy.cwiseQuotient(dq), lam);
    Eigen::VectorXd beta = Eigen::VectorXd::Zero(p);
    for (int it = 0; it < max_iter; ++it) {
        double delta = 0.0;
        for (Eigen::Index j = 0; j < p; ++j) {
            double r = Z_noisy(j) - Q.row(j).dot(beta) + dq(j) * beta(j);
            double soft = detail::sign(r) * std::max(std::abs(r) - pen.D(j), 0.0) / dq(j);
            double next = std::min(std::max(soft, pen.L(j)), pen.U(j));
            delta = std::max(delta, std::abs(next - beta(j)) * std::sqrt(dq(j)));
            beta(j) = next;
        }
        if (delta < tol) break;
    }
    return beta;
}

double unilasso_kkt_violation(const Eigen::VectorXd& beta_hat, const Eigen::MatrixXd& Q,
                              const Eigen::VectorXd& Z_noisy, const Eigen::VectorXd& lam) {
    UniLassoPenalty pen = unilasso_penalty(Z_noisy.cwiseQuotient(Q.diagonal()), lam);
    Eigen::VectorXd g = Z_noisy - Q * beta_hat;   // minus the gradient
    double v = 0.0;
    for (Eigen::Index j = 0; j < beta_hat.size(); ++j) {
        if (beta_hat(j) != 0) {
            v = std::max(v, std::abs(g(j) - pen.D(j) * pen.s(j)));
        } else {
            v = std::max(v, pen.s(j) * g(j) - pen.D(j));
        }
    }
    return v;
}

Eigen::VectorXd unilasso_loo_kappa(const Eigen::MatrixXd& X, const Eigen::VectorXd& y,
                                   const Eigen::VectorXd& beta_hat) {
    const double n = static_cast<double>(X.rows());
    Eigen::RowVectorXd S = X.array().square().colwise().sum();
    Eigen::VectorXd b = (X.transpose() * y).cwiseQuotient(S.transpose());
    // leverages of the univariate fits, then leave-one-out fits minus fits
    Eigen::ArrayXXd h = (X.array().square().rowwise() / S.array()) + 1.0 / n;
    Eigen::ArrayXXd fit_resid = (-(X.array().rowwise() * b.transpose().array())).colwise() + y.array();
    Eigen::MatrixXd delta = (-h / (1.0 - h) * fit_resid).matrix();
    delta.rowwise() -= delta.colwise().mean();
    Eigen::VectorXd theta = beta_hat.cwiseQuotient(b);
    Eigen::VectorXd r = y - X * beta_hat;
    Eigen::VectorXd dt = delta * theta;
    return -delta.transpose() * r + b.cwiseProduct(X.transpose() * dt) + delta.transpose() * dt;
}

Eigen::MatrixXd quadratic_le_zero(const Eigen::VectorXd& q2, const Eigen::VectorXd& q1,
                                  const Eigen::VectorXd& q0) {
    const Eigen::Index m = q2.size();
    Eigen::MatrixXd out(m, 4);
    for (Eigen::Index i = 0; i < m; ++i) {
        double lo1 = kInf, hi1 = -kInf, lo2 = kInf, hi2 = -kInf;
        if (q2(i) == 0) {
            // linear: q1 w + q0 <= 0
            if (q1(i) > kZeroTol) {
                lo1 = -kInf;
                hi1 = -q0(i) / q1(i);
            } else if (q1(i) < -kZeroTol) {
                lo1 = -q0(i) / q1(i);
                hi1 = kInf;
            } else if (q0(i) <= kZeroTol) {
                lo1 = -kInf;
                hi1 = kInf;
            }
        } else {
            double disc = q1(i) * q1(i) - 4 * q2(i) * q0(i);
            if (disc >= 0) {
                // roots in the numerically stable form
                double sq = std::sqrt(disc);
                double w = -0.5 * (q1(i) + (q1(i) >= 0 ? sq : -sq));
                double ra = w / q2(i);
                double rb = w != 0 ? q0(i) / w : ra;
                double r1 = std::min(ra, rb), r2 = std::max(ra, rb);
                if (q2(i) > 0) {          // between the roots
                    lo1 = r1;
                    hi1 = r2;
                } else {                  // outside the roots
                    lo1 = -kInf;
                    hi1 = r1;
                    lo2 = r2;
                    hi2 = kInf;
                }
            } else if (q2(i) < 0) {
                lo1 = -kInf;
                hi1 = kInf;
            }
        }
        out(i, 0) = lo1;
        out(i, 1) = hi1;
        out(i, 2) = lo2;
        out(i, 3) = hi2;
    }
    return out;
}

Eigen::MatrixXd unilasso_row_sets(const Eigen::VectorXd& u, const Eigen::VectorXd& a,
                                  const Eigen::VectorXd& b0, const Eigen::VectorXd& m,
                                  double c, double gamma, double s) {
    // the half-line where C(w) has sign s
    double k = s * gamma, h = -s * c;
    double lo_I, hi_I;
    if (k > 0) {
        lo_I = h / k;
        hi_I = kInf;
    } else if (k < 0) {
        lo_I = -kInf;
        hi_I = h / k;
    } else if (s * c > 0) {
        lo_I = -kInf;
        hi_I = kInf;
    } else {
        lo_I = kInf;
        hi_I = -kInf;
    }
    // on it, the row is s [C(w) (u_i - b0_i + a_i w) - m_i] <= 0
    Eigen::VectorXd v = u - b0;
    Eigen::MatrixXd out = quadratic_le_zero(s * gamma * a, s * (c * a + gamma * v),
                                            s * (c * v - m));
    for (Eigen::Index i = 0; i < out.rows(); ++i) {
        for (int j = 0; j < 2; ++j) {
            out(i, 2 * j) = std::max(out(i, 2 * j), lo_I);
            out(i, 2 * j + 1) = std::min(out(i, 2 * j + 1), hi_I);
        }
    }
    return out;
}

Eigen::MatrixXd intersect_row_sets(const Eigen::MatrixXd& sets) {
    const Eigen::Index rows = sets.rows();
    if (rows == 0) {
        Eigen::MatrixXd all(1, 2);
        all << -kInf, kInf;
        return all;
    }
    if (sets.cols() != 4) throw std::invalid_argument("sets must have 4 columns");
    // sweep over the endpoints counting covering rows (each row's pieces are disjoint);
    // at ties, closing before opening drops zero-length pieces
    std::vector<std::pair<double, int>> events;
    for (Eigen::Index i = 0; i < rows; ++i) {
        for (int j = 0; j < 2; ++j) {
            double lo = sets(i, 2 * j), hi = sets(i, 2 * j + 1);
            if (lo < hi) {
                events.emplace_back(lo, 1);
                events.emplace_back(hi, -1);
            }
        }
    }
    std::sort(events.begin(), events.end());
    std::vector<std::pair<double, double>> intervals;
    Eigen::Index count = 0;
    for (size_t i = 0; i + 1 < events.size(); ++i) {
        count += events[i].second;
        if (count == rows && events[i].first < events[i + 1].first) {
            double lo = events[i].first, hi = events[i + 1].first;
            // merge touching segments
            if (!intervals.empty() && lo <= intervals.back().second) {
                intervals.back().second = std::max(intervals.back().second, hi);
            } else {
                intervals.emplace_back(lo, hi);
            }
        }
    }
    // drop pieces of rounding width, e.g. where a root meets the sign boundary C(w) = 0
    std::vector<std::pair<double, double>> kept;
    for (const auto& iv : intervals) {
        bool bounded = std::isfinite(iv.first) && std::isfinite(iv.second);
        double scale = std::max(1.0, std::max(std::abs(iv.first), std::abs(iv.second)));
        if (!bounded || iv.second - iv.first > kSliver * scale) kept.push_back(iv);
    }
    Eigen::MatrixXd out(kept.size(), 2);
    for (size_t i = 0; i < kept.size(); ++i) {
        out(i, 0) = kept[i].first;
        out(i, 1) = kept[i].second;
    }
    return out;
}

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
                                        double tol) {
    if (k < 0 || k >= D.size()) throw std::out_of_range("k is not a coordinate");
    LassoConstraints cons = lasso_post_selection_constraints(beta_hat, G_hat, Q, D, L, U, tol);
    // b is affine in D: its coefficient on D_k (any step works; D_k = 0 for unireg)
    double step = D(k) > 0 ? D(k) : 1.0;
    Eigen::VectorXd D_step = D;
    D_step(k) += step;
    Eigen::VectorXd b_step = lasso_post_selection_constraints(beta_hat, G_hat, Q, D_step, L, U, tol).b;
    Eigen::VectorXd M_k = (b_step - cons.b) / step;
    Eigen::VectorXd b0 = cons.b - M_k * D(k);
    double s_k = detail::sign(C_k);
    // C_k(w) = c + gamma w; along the line, A Y(w) = A y0 + (A direction) w
    double c = C_k - gamma * w_obs;
    return intersect_row_sets(unilasso_row_sets(cons.A->multiply(y0), cons.A->multiply(direction),
                                                b0, M_k * (lam_k * s_k), c, gamma, s_k));
}

} // namespace lassoinf
