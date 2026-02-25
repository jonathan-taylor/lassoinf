#pragma once

#include <Eigen/Dense>
#include <cmath>
#include <limits>
#include <functional>
#include <utility>

namespace lassoinf {

struct Params {
    Eigen::VectorXd gamma;
    Eigen::VectorXd c;
    Eigen::VectorXd bar_gamma;
    double bar_s;
    Eigen::VectorXd n_o;
    Eigen::VectorXd bar_n_o;
    double theta_hat;
    double bar_theta;
};

class SelectiveInference {
public:
    SelectiveInference(Eigen::VectorXd Z, 
                       Eigen::VectorXd Z_noisy, 
                       Eigen::MatrixXd Q, 
                       Eigen::MatrixXd Q_noise);

    Params compute_params(const Eigen::VectorXd& v) const;

    std::pair<double, double> get_interval(const Eigen::VectorXd& v, double t, 
                                           const Eigen::MatrixXd& A, const Eigen::VectorXd& b) const;

    std::function<double(double)> get_weight(const Eigen::VectorXd& v, const Eigen::MatrixXd& A, const Eigen::VectorXd& b) const;

private:
    Eigen::VectorXd Z_;
    Eigen::VectorXd Z_noisy_;
    Eigen::MatrixXd Q_;
    Eigen::MatrixXd Q_noise_;
};

} // namespace lassoinf
