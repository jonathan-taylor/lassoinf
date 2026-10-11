#include "ray_selection.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace lassoinf {

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
    auto member = [&](double w) -> bool { return event(y0 + direction * w); };
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
