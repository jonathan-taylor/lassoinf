#include "gaussian_family.hpp"
#include <cmath>
#include <algorithm>
#include <numeric>

namespace lassoinf {

static double norm_cdf(double x) {
    return 0.5 * std::erfc(-x / std::sqrt(2.0));
}

static double norm_pdf(double x) {
    return std::exp(-0.5 * x * x) / std::sqrt(2.0 * M_PI);
}

TruncatedGaussian::TruncatedGaussian(double estimate, double sigma, double smoothing_sigma, double lower_bound, double upper_bound, double noisy_estimate, double factor)
    : estimate_(estimate), sigma_(sigma), smoothing_sigma_(smoothing_sigma), lower_bound_(lower_bound), upper_bound_(upper_bound), noisy_estimate_(noisy_estimate), factor_(factor) {}

Eigen::VectorXd TruncatedGaussian::weight(const Eigen::VectorXd& x) const {
    Eigen::VectorXd w(x.size());
    for (int i = 0; i < x.size(); ++i) {
        double t = x(i) * factor_;
        w(i) = norm_cdf((upper_bound_ - t) / smoothing_sigma_) - norm_cdf((lower_bound_ - t) / smoothing_sigma_);
    }
    return w;
}

WeightedGaussianFamily::WeightedGaussianFamily(double estimate, double sigma, const std::vector<std::function<Eigen::VectorXd(const Eigen::VectorXd&)>>& weight_fns, double num_sd, int num_grid)
    : estimate_(estimate), sigma_(sigma), weight_fns_(weight_fns), num_sd_(num_sd), num_grid_(num_grid) {}

std::shared_ptr<DiscreteFamily> WeightedGaussianFamily::get_family(double basept) {
    if (std::isnan(basept)) {
        basept = estimate_;
    }

    Eigen::VectorXd grid = Eigen::VectorXd::LinSpaced(num_grid_, basept - num_sd_ * sigma_, basept + num_sd_ * sigma_);
    
    Eigen::VectorXd log_weight = Eigen::VectorXd::Zero(num_grid_);
    for (const auto& w_fn : weight_fns_) {
        Eigen::VectorXd w = w_fn(grid);
        for (int i = 0; i < w.size(); ++i) {
            double w_val = std::max(w(i), 1e-16);
            log_weight(i) += std::log(w_val);
        }
    }

    double max_log_weight = log_weight.maxCoeff();
    for (int i = 0; i < num_grid_; ++i) {
        log_weight(i) -= (max_log_weight + 10.0);
    }

    std::vector<double> sufficient_stat(num_grid_);
    std::vector<double> weight(num_grid_);

    for (int i = 0; i < num_grid_; ++i) {
        sufficient_stat[i] = grid(i);
        weight[i] = std::exp(log_weight(i)) * norm_pdf((grid(i) - basept) / sigma_);
    }

    return std::make_shared<DiscreteFamily>(sufficient_stat, weight);
}

double WeightedGaussianFamily::pvalue(double null_value, const std::string& alternative, double basept) {
    if (std::isnan(basept)) {
        basept = null_value;
    }

    auto family = get_family(basept);
    double tilt = (null_value - basept) / (sigma_ * sigma_);
    
    if (alternative == "less" || alternative == "twosided") {
        double cdf = family->cdf(tilt, estimate_);
        if (alternative == "less") {
            return cdf;
        } else {
            return 2.0 * std::min(cdf, 1.0 - cdf);
        }
    } else if (alternative == "greater") {
        return family->ccdf(tilt, estimate_);
    } else {
        throw std::invalid_argument("alternative should be one of ['twosided', 'greater', 'less']");
    }
}

std::pair<double, double> WeightedGaussianFamily::interval(double basept, double level) {
    if (std::isnan(basept)) {
        basept = estimate_;
    }

    auto family = get_family(basept);
    auto interval = family->equal_tailed_interval(estimate_, 1.0 - level);
    
    double L = interval.first * sigma_ * sigma_ + basept;
    double U = interval.second * sigma_ * sigma_ + basept;
    
    return {L, U};
}

} // namespace lassoinf
