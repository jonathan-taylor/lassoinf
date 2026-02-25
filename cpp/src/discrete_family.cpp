#include "discrete_family.h"
#include <cmath>
#include <numeric>
#include <algorithm>
#include <stdexcept>

namespace lassoinf {

double bisection_find_root(const std::function<double(double)>& f, double value, double lower, double upper, double tol) {
    if (lower > upper) {
        std::swap(lower, upper);
    }
    double f_lower = f(lower) - value;
    double f_upper = f(upper) - value;

    if (f_lower * f_upper > 0) {
        if (std::abs(f_lower) < std::abs(f_upper)) return lower;
        return upper;
    }

    while ((upper - lower) > tol) {
        double mid = (lower + upper) / 2.0;
        double f_mid = f(mid) - value;
        if (f_mid == 0.0) return mid;
        if (f_lower * f_mid < 0) {
            upper = mid;
            f_upper = f_mid;
        } else {
            lower = mid;
            f_lower = f_mid;
        }
    }
    return (lower + upper) / 2.0;
}

DiscreteFamily::DiscreteFamily(const std::vector<double>& sufficient_stat, const std::vector<double>& weights, double theta) {
    if (sufficient_stat.size() != weights.size()) {
        throw std::invalid_argument("sufficient_stat and weights must have the same size");
    }
    
    std::vector<std::pair<double, double>> xw;
    for (size_t i = 0; i < sufficient_stat.size(); ++i) {
        xw.push_back({sufficient_stat[i], weights[i]});
    }
    std::sort(xw.begin(), xw.end());

    double sum_w = 0.0;
    for (const auto& p : xw) sum_w += p.second;

    for (const auto& p : xw) {
        x_.push_back(p.first);
        double w_norm = p.second / sum_w;
        w_.push_back(w_norm);
        lw_.push_back(std::log(w_norm));
    }
    
    theta_ = std::numeric_limits<double>::quiet_NaN();
    set_theta(theta);
}

double DiscreteFamily::get_theta() const { return theta_; }

void DiscreteFamily::set_theta(double theta) {
    if (theta != theta_ || std::isnan(theta_)) {
        update_partition(theta);
        theta_ = theta;
    }
}

void DiscreteFamily::update_partition(double theta) {
    std::vector<double> thetaX(x_.size());
    double max_val = -std::numeric_limits<double>::infinity();
    for (size_t i = 0; i < x_.size(); ++i) {
        thetaX[i] = theta * x_[i] + lw_[i];
        if (thetaX[i] > max_val) max_val = thetaX[i];
    }
    
    double largest = max_val - 5.0;
    partition_ = 0.0;
    pdf_.resize(x_.size());
    for (size_t i = 0; i < x_.size(); ++i) {
        pdf_[i] = std::exp(thetaX[i] - largest);
        partition_ += pdf_[i];
    }
    
    for (size_t i = 0; i < x_.size(); ++i) {
        pdf_[i] /= partition_;
    }
    partition_ *= std::exp(largest);
}

double DiscreteFamily::get_partition() const { return partition_; }
const std::vector<double>& DiscreteFamily::get_sufficient_stat() const { return x_; }
const std::vector<double>& DiscreteFamily::get_weights() const { return w_; }

std::vector<double> DiscreteFamily::pdf(double theta) {
    set_theta(theta);
    return pdf_;
}

double DiscreteFamily::cdf(double theta, double x, double gamma) {
    std::vector<double> p = pdf(theta);
    double tr = 0.0;
    for (size_t i = 0; i < x_.size(); ++i) {
        if (x_[i] < x) tr += p[i];
        else if (x_[i] == x) tr += gamma * p[i];
    }
    return tr;
}

double DiscreteFamily::ccdf(double theta, double x, double gamma) {
    std::vector<double> p = pdf(theta);
    double tr = 0.0;
    for (size_t i = 0; i < x_.size(); ++i) {
        if (x_[i] > x) tr += p[i];
        else if (x_[i] == x) tr += gamma * p[i];
    }
    return tr;
}

double DiscreteFamily::E(double theta, const std::function<double(double)>& func) {
    std::vector<double> p = pdf(theta);
    double expected = 0.0;
    for (size_t i = 0; i < x_.size(); ++i) {
        expected += func(x_[i]) * p[i];
    }
    return expected;
}

double DiscreteFamily::Var(double theta, const std::function<double(double)>& func) {
    double mu = E(theta, func);
    return E(theta, [func, mu](double x) {
        double val = x - mu; // Assuming func is identity for simplicity, but using func correctly
        double f_val = func(x) - mu;
        return f_val * f_val;
    });
}

double DiscreteFamily::Cov(double theta, const std::function<double(double)>& func1, const std::function<double(double)>& func2) {
    double mu1 = E(theta, func1);
    double mu2 = E(theta, func2);
    return E(theta, [func1, func2, mu1, mu2](double x) {
        return (func1(x) - mu1) * (func2(x) - mu2);
    });
}

std::pair<double, double> DiscreteFamily::equal_tailed_interval(double observed, double alpha, double tol) {
    // Make sure we have the correct theta_ partition initialized before computing moments
    set_theta(theta_);
    
    double mu = E(theta_, [](double x){ return x; });
    double sigma = std::sqrt(Var(theta_, [](double x){ return x; }));
    double lb = mu - 20 * sigma;
    double ub = mu + 20 * sigma;
    
    auto F = [this, observed](double th) {
        // cast to non-const member calls
        DiscreteFamily* self = const_cast<DiscreteFamily*>(this);
        return self->cdf(th, observed, 1.0);
    };
    
    double L = bisection_find_root(F, 1.0 - 0.5 * alpha, lb, ub, tol);
    double U = bisection_find_root(F, 0.5 * alpha, lb, ub, tol);
    return {L, U};
}

} // namespace lassoinf
