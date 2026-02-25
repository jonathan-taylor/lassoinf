#include "lassoinf.hpp"

namespace lassoinf {

SelectiveInference::SelectiveInference(Eigen::VectorXd Z, 
                                       Eigen::VectorXd Z_noisy, 
                                       Eigen::MatrixXd Q, 
                                       Eigen::MatrixXd Q_noise)
    : Z_(std::move(Z)), Z_noisy_(std::move(Z_noisy)), Q_(std::move(Q)), Q_noise_(std::move(Q_noise)) {}

Params SelectiveInference::compute_params(const Eigen::VectorXd& v) const {
    Params p;
    double v_sigma_v = v.dot(Q_ * v);
    p.gamma = (Q_ * v) / v_sigma_v;
    p.c = Q_noise_.colPivHouseholderQr().solve(Q_ * v);
    double bar_s2 = p.c.dot(Q_noise_ * p.c);
    p.bar_s = std::sqrt(bar_s2);
    p.bar_gamma = (Q_ * v) / bar_s2;
    p.theta_hat = v.dot(Z_);
    p.n_o = Z_ - p.gamma * p.theta_hat;
    
    Eigen::VectorXd omega = Z_noisy_ - Z_;
    p.bar_theta = p.c.dot(omega);
    p.bar_n_o = omega - p.bar_gamma * p.bar_theta;
    
    return p;
}

std::pair<double, double> SelectiveInference::get_interval(const Eigen::VectorXd& v, double t, 
                                                           const Eigen::MatrixXd& A, const Eigen::VectorXd& b) const {
    Params p = compute_params(v);
    
    Eigen::VectorXd alpha = A * p.bar_gamma;
    Eigen::VectorXd beta = b - A * (p.n_o + p.bar_n_o + p.gamma * t);
    
    double lower = -std::numeric_limits<double>::infinity();
    double upper = std::numeric_limits<double>::infinity();
    
    for (int i = 0; i < alpha.size(); ++i) {
        double a_i = alpha(i);
        double b_i = beta(i);
        if (a_i > 1e-10) {
            upper = std::min(upper, b_i / a_i);
        } else if (a_i < -1e-10) {
            lower = std::max(lower, b_i / a_i);
        } else if (b_i < -1e-10) {
            return {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN()};
        }
    }
    
    return {lower, upper};
}

std::function<double(double)> SelectiveInference::get_weight(const Eigen::VectorXd& v, const Eigen::MatrixXd& A, const Eigen::VectorXd& b) const {
    Params p = compute_params(v);
    double bar_s = p.bar_s;
    
    auto obj = *this;
    
    return [obj, v, A, b, bar_s](double t) -> double {
        auto interval = obj.get_interval(v, t, A, b);
        double L = interval.first;
        double U = interval.second;
        if (std::isnan(L) || std::isnan(U)) {
            return 0.0;
        }
        
        auto norm_cdf = [](double x) {
            return 0.5 * std::erfc(-x / std::sqrt(2.0));
        };
        
        return norm_cdf(U / bar_s) - norm_cdf(L / bar_s);
    };
}

} // namespace lassoinf
