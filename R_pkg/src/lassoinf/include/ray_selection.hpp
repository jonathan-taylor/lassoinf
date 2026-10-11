#pragma once

// Selection events evaluated along the conditioning ray.
//
// For a contrast, the selection data move along the ray Y(w) = y0 + direction w, with
// y0 = N_o + bar_N_o and direction = bar_Gamma, and the selection event depends on
// (theta_hat, bar_omega) only through the scalar w (see docs/main.md). Given an explicit
// characterization event(Y) -> bool, the truncation set {w : event(Y(w))} is located by
// evaluating the event on a grid of w and bisecting each change. No refitting is needed.
//
// The uniLasso has a closed form for its truncation set (unilasso.hpp), which this does not
// replace.

#include <Eigen/Dense>
#include <functional>
#include <memory>
#include <utility>
#include "affine_constraints.hpp"

namespace lassoinf {

// A selection event: true where the selection data Y give the observed selection
class RayEvent {
public:
    virtual ~RayEvent() = default;
    virtual bool operator()(const Eigen::VectorXd& Y) const = 0;
};

// An event given by a function
class FunctionRayEvent : public RayEvent {
public:
    explicit FunctionRayEvent(std::function<bool(const Eigen::VectorXd&)> f) : f_(std::move(f)) {}
    bool operator()(const Eigen::VectorXd& Y) const override { return f_(Y); }
private:
    std::function<bool(const Eigen::VectorXd&)> f_;
};

// The penalty weights and bounds (D, L, U) of a lasso at the selection data Y; valid is
// false where Y is outside the conditioning event for the penalties (for example where a
// conditioned sign is wrong).
struct RayPenalty {
    bool valid;
    Eigen::VectorXd D;
    Eigen::VectorXd L;  // empty means no lower bounds
    Eigen::VectorXd U;  // empty means no upper bounds
};

// The selection event of a lasso with data-dependent penalties and bounds: the same active
// set, signs and bound activations as beta_hat.
//
// The lasso solves minimize 1/2 beta' Q beta - beta' Y + sum_j D_j |beta_j| subject to
// L <= beta <= U, with (D, L, U) = penalty(Y). beta_hat and G_hat (the gradient
// Q beta_hat - Y at the observed Y) fix the active set, signs and bound activations. The
// constraints A Y <= b are checked to slack * (1 + |b|).
class LassoRayEvent : public RayEvent {
public:
    LassoRayEvent(Eigen::VectorXd beta_hat, Eigen::VectorXd G_hat, std::shared_ptr<LinearOperator> Q,
                  std::function<RayPenalty(const Eigen::VectorXd&)> penalty, double tol = 1e-6,
                  double slack = 1e-9)
        : beta_hat_(std::move(beta_hat)), G_hat_(std::move(G_hat)), Q_(std::move(Q)),
          penalty_(std::move(penalty)), tol_(tol), slack_(slack) {}

    bool operator()(const Eigen::VectorXd& Y) const override;

private:
    Eigen::VectorXd beta_hat_;
    Eigen::VectorXd G_hat_;
    std::shared_ptr<LinearOperator> Q_;
    std::function<RayPenalty(const Eigen::VectorXd&)> penalty_;
    double tol_;
    double slack_;
};

// The observed w and the standard deviation of w along the ray of a contrast with the given
// bar_s, naive variance, theta_hat and bar_theta.
std::pair<double, double> ray_scale(double bar_s, double variance, double theta_hat, double bar_theta);

// The truncation set {w : event(y0 + direction w)}, as a (k, 2) matrix of disjoint
// intervals in increasing order.
//
// The event is evaluated on n_grid equally spaced points within n_sd standard deviations
// sd_w of w_obs, and assumed constant beyond them (where the Gaussian has negligible mass).
// Each change between grid points is located by bisection, to tol * sd_w. The event can
// change more than once between two grid points; such short pieces are missed. A missed
// piece of length e changes the truncated probabilities by at most e / (bar_s sqrt(2 pi)).
//
// Throws std::invalid_argument if the event does not hold at w_obs.
Eigen::MatrixXd find_truncation_set(const RayEvent& event,
                                    const Eigen::VectorXd& y0,
                                    const Eigen::VectorXd& direction,
                                    double w_obs,
                                    double sd_w,
                                    double n_sd = 10.0,
                                    int n_grid = 2001,
                                    double tol = 1e-10);

} // namespace lassoinf
