#pragma once

#include <vector>
#include <functional>
#include <memory>
#include <string>
#include <limits>
#include <Eigen/Dense>
#include "discrete_family.h"

namespace lassoinf {

class TruncatedGaussian {
public:
    TruncatedGaussian(double estimate, double sigma, double smoothing_sigma, double lower_bound, double upper_bound, double noisy_estimate, double factor = 1.0);

    Eigen::VectorXd weight(const Eigen::VectorXd& x) const;

private:
    double estimate_;
    double sigma_;
    double smoothing_sigma_;
    double lower_bound_;
    double upper_bound_;
    double noisy_estimate_;
    double factor_;
};

class WeightedGaussianFamily {
public:
    WeightedGaussianFamily(double estimate, double sigma, const std::vector<std::function<Eigen::VectorXd(const Eigen::VectorXd&)>>& weight_fns, double num_sd = 10.0, int num_grid = 4000);

    double pvalue(double null_value = 0.0, const std::string& alternative = "twosided", double basept = std::numeric_limits<double>::quiet_NaN());
    std::pair<double, double> interval(double basept = std::numeric_limits<double>::quiet_NaN(), double level = 0.9);
    // MLE is defined in Python, but we don't have MLE in DiscreteFamily C++ yet? Let's check.

private:
    std::shared_ptr<DiscreteFamily> get_family(double basept);

    double estimate_;
    double sigma_;
    std::vector<std::function<Eigen::VectorXd(const Eigen::VectorXd&)>> weight_fns_;
    double num_sd_;
    int num_grid_;
};

} // namespace lassoinf
