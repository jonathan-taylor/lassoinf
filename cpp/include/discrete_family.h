#pragma once

#include <vector>
#include <functional>
#include <utility>

namespace lassoinf {

class DiscreteFamily {
public:
    DiscreteFamily(const std::vector<double>& sufficient_stat, const std::vector<double>& weights, double theta = 0.0);

    double get_theta() const;
    void set_theta(double theta);

    double get_partition() const;
    const std::vector<double>& get_sufficient_stat() const;
    const std::vector<double>& get_weights() const;

    std::vector<double> pdf(double theta);
    double cdf(double theta, double x, double gamma = 1.0);
    double ccdf(double theta, double x, double gamma = 0.0);

    double E(double theta, const std::function<double(double)>& func);
    double Var(double theta, const std::function<double(double)>& func);
    double Cov(double theta, const std::function<double(double)>& func1, const std::function<double(double)>& func2);

    std::pair<double, double> equal_tailed_interval(double observed, double alpha = 0.05, double tol = 1e-6);

private:
    std::vector<double> x_;
    std::vector<double> w_;
    std::vector<double> lw_;
    double theta_;
    double partition_;
    std::vector<double> pdf_;

    void update_partition(double theta);
};

// Root finder utility
double bisection_find_root(const std::function<double(double)>& f, double value, double lower, double upper, double tol = 1e-6);

} // namespace lassoinf
