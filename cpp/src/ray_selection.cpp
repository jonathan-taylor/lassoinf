#include "ray_selection.hpp"

#include <cmath>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace lassoinf {

using detail::kInf;
using detail::take;

std::function<bool(double)> RayEvent::along(const Eigen::VectorXd& y0, const Eigen::VectorXd& direction) const {
    return [this, y0, direction](double w) { return (*this)(y0 + direction * w); };
}

namespace {

double bound(const Eigen::VectorXd& v, int j, double none) { return v.size() > 0 ? v(j) : none; }

// The active set, signs and bound activations of beta_hat under (L, U), as in
// lasso_post_selection_constraints, with which bounds give constraints: this fixes A and the
// layout of b. One character per coordinate.
std::string lasso_pattern(const Eigen::VectorXd& beta, const Eigen::VectorXd& L, const Eigen::VectorXd& U,
                          double tol) {
    std::string code(beta.size(), ' ');
    for (Eigen::Index j = 0; j < beta.size(); ++j) {
        double Lj = bound(L, j, -kInf), Uj = bound(U, j, kInf);
        bool at_L = beta(j) <= Lj + tol;
        bool at_U = beta(j) >= Uj - tol;
        bool at_0 = std::abs(beta(j)) <= tol;
        if (!at_L && !at_U && !at_0) {
            code[j] = beta(j) > 0 ? (Uj < kInf ? 'P' : 'p') : (Lj > -kInf ? 'N' : 'n');
        } else if (at_0) {
            code[j] = static_cast<char>('0' + (Lj < -tol) + 2 * (Uj > tol));
        } else {
            code[j] = at_U ? 'U' : 'L';
        }
    }
    return code;
}

// The constraints of one pattern along the ray: A Y(w) = u + a w, and what b needs.
struct RayConstraints {
    std::vector<int> E, E_c;
    Eigen::VectorXd s_E;
    Eigen::MatrixXd W;      // Q_{E,E}^{-1}
    Eigen::MatrixXd Q_EcE;  // Q_{E^c,E}
    std::vector<int> B;     // coordinates at a bound other than 0
    Eigen::MatrixXd Q_B;    // Q_{:,B}
    Eigen::VectorXd u, a;
};

RayConstraints ray_constraints(const LassoConstraints& cons, const LinearOperator& Q, const std::string& code,
                               const Eigen::VectorXd& y0, const Eigen::VectorXd& direction) {
    RayConstraints rc{cons.E, cons.E_c, cons.s_E, cons.W, {}, {}, {}, cons.A->multiply(y0),
                      cons.A->multiply(direction)};
    Eigen::Index n = Q.rows();
    rc.Q_EcE.resize(rc.E_c.size(), rc.E.size());
    for (size_t i = 0; i < rc.E.size(); ++i) {
        Eigen::VectorXd e = Eigen::VectorXd::Zero(n);
        e(rc.E[i]) = 1.0;
        rc.Q_EcE.col(i) = take(Q.multiply(e), rc.E_c);
    }
    for (int j : rc.E_c) {
        if (code[j] == 'U' || code[j] == 'L') rc.B.push_back(j);
    }
    rc.Q_B.resize(n, rc.B.size());
    for (size_t i = 0; i < rc.B.size(); ++i) {
        Eigen::VectorXd e = Eigen::VectorXd::Zero(n);
        e(rc.B[i]) = 1.0;
        rc.Q_B.col(i) = Q.multiply(e);
    }
    return rc;
}

// b of lasso_post_selection_constraints at (D, L, U), in the same row order
Eigen::VectorXd ray_offsets(const RayConstraints& rc, const std::string& code, const RayPenalty& pen) {
    Eigen::Index n = static_cast<Eigen::Index>(code.size());
    Eigen::VectorXd Q_V = Eigen::VectorXd::Zero(n);
    if (!rc.B.empty()) {
        Eigen::VectorXd v_B(rc.B.size());
        for (size_t i = 0; i < rc.B.size(); ++i) {
            int j = rc.B[i];
            v_B(i) = code[j] == 'U' ? pen.U(j) : pen.L(j);
        }
        Q_V = rc.Q_B * v_B;
    }
    Eigen::VectorXd c_E = rc.W * (take(Q_V, rc.E) + take(pen.D, rc.E).cwiseProduct(rc.s_E));
    Eigen::VectorXd c_Ec = rc.Q_EcE * c_E - take(Q_V, rc.E_c);

    std::vector<double> b;
    b.reserve(2 * n);
    for (size_t k = 0; k < rc.E.size(); ++k) b.push_back(-rc.s_E(k) * c_E(k));
    for (size_t k = 0; k < rc.E.size(); ++k) {
        int j = rc.E[k];
        if (code[j] == 'P') b.push_back(pen.U(j) + c_E(k));
        else if (code[j] == 'N') b.push_back(-pen.L(j) - c_E(k));
    }
    for (size_t k = 0; k < rc.E_c.size(); ++k) {
        int j = rc.E_c[k];
        double d = pen.D(j);
        double gmin = -kInf, gmax = kInf;
        char c = code[j];
        if (c == 'U') {
            gmin = d;
        } else if (c == 'L') {
            gmax = -d;
        } else {
            int flags = c - '0';
            if (flags & 1) gmin = -d;
            if (flags & 2) gmax = d;
        }
        if (gmax < kInf) b.push_back(gmax - c_Ec(k));
        if (gmin > -kInf) b.push_back(-gmin + c_Ec(k));
    }
    return Eigen::Map<Eigen::VectorXd>(b.data(), static_cast<Eigen::Index>(b.size()));
}

} // namespace

bool LassoRayEvent::operator()(const Eigen::VectorXd& Y) const {
    RayPenalty pen = penalty_(Y);
    if (!pen.valid) return false;
    LassoConstraints cons = lasso_post_selection_constraints(beta_hat_, G_hat_, Q_, pen.D, pen.L,
                                                             pen.U, tol_);
    Eigen::VectorXd AY = cons.A->multiply(Y);
    for (Eigen::Index i = 0; i < AY.size(); ++i) {
        if (!(AY(i) <= cons.b(i) + slack_ * (1 + std::abs(cons.b(i))))) return false;
    }
    return true;
}

std::function<bool(double)> LassoRayEvent::along(const Eigen::VectorXd& y0,
                                                 const Eigen::VectorXd& direction) const {
    auto cache = std::make_shared<std::map<std::string, RayConstraints>>();
    return [this, y0, direction, cache](double w) {
        Eigen::VectorXd Y = y0 + direction * w;
        RayPenalty pen = penalty_(Y);
        if (!pen.valid) return false;
        std::string code = lasso_pattern(beta_hat_, pen.L, pen.U, tol_);
        auto it = cache->find(code);
        if (it == cache->end()) {
            LassoConstraints cons = lasso_post_selection_constraints(beta_hat_, G_hat_, Q_, pen.D, pen.L,
                                                                     pen.U, tol_);
            it = cache->emplace(code, ray_constraints(cons, *Q_, code, y0, direction)).first;
        }
        const RayConstraints& rc = it->second;
        Eigen::VectorXd b = ray_offsets(rc, code, pen);
        for (Eigen::Index i = 0; i < b.size(); ++i) {
            if (!(rc.u(i) + rc.a(i) * w <= b(i) + slack_ * (1 + std::abs(b(i))))) return false;
        }
        return true;
    };
}

std::pair<double, double> ray_scale(double bar_s, double variance, double theta_hat, double bar_theta) {
    double bar_s2 = bar_s * bar_s;
    double w_obs = bar_s2 / variance * theta_hat + bar_theta;
    double sd_w = std::sqrt(bar_s2 * bar_s2 / variance + bar_s2);
    return {w_obs, sd_w};
}

Eigen::MatrixXd find_truncation_set(const RayEvent& event,
                                    const Eigen::VectorXd& y0,
                                    const Eigen::VectorXd& direction,
                                    double w_obs,
                                    double sd_w,
                                    double n_sd,
                                    int n_grid,
                                    double tol) {
    if (y0.size() != direction.size()) throw std::invalid_argument("y0 and direction differ in length");
    if (n_grid < 2) throw std::invalid_argument("n_grid must be at least 2");
    std::function<bool(double)> member = event.along(y0, direction);
    if (!member(w_obs)) throw std::invalid_argument("the observed data are not in the selection event");

    // the grid w_obs + linspace(-n_sd, n_sd, n_grid) * sd_w
    std::vector<double> grid(n_grid);
    const double step = 2 * n_sd / (n_grid - 1);
    for (int i = 0; i < n_grid; ++i) {
        double t = (i == n_grid - 1) ? n_sd : -n_sd + i * step;
        grid[i] = w_obs + t * sd_w;
    }
    std::vector<char> inside(n_grid);
    for (int i = 0; i < n_grid; ++i) inside[i] = member(grid[i]);

    // the change between lo and hi, where member(lo) == lo_inside != member(hi)
    auto boundary = [&](double lo, double hi, bool lo_inside) {
        while (hi - lo > tol * sd_w) {
            double mid = 0.5 * (lo + hi);
            if (member(mid) == lo_inside) {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        return 0.5 * (lo + hi);
    };

    const double inf = std::numeric_limits<double>::infinity();
    std::vector<std::pair<double, double>> intervals;
    bool open = inside[0];
    double start = -inf;
    for (int i = 1; i < n_grid; ++i) {
        if (inside[i] != inside[i - 1]) {
            double b = boundary(grid[i - 1], grid[i], inside[i - 1]);
            if (inside[i]) {
                start = b;
                open = true;
            } else {
                intervals.emplace_back(start, b);
                open = false;
            }
        }
    }
    if (open) intervals.emplace_back(start, inf);

    Eigen::MatrixXd out(intervals.size(), 2);
    for (size_t i = 0; i < intervals.size(); ++i) {
        out(i, 0) = intervals[i].first;
        out(i, 1) = intervals[i].second;
    }
    return out;
}

} // namespace lassoinf
