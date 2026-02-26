#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <cmath>
#include <limits>
#include <functional>
#include <utility>
#include <vector>
#include <memory>

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

// Abstract base class for Matrix-Free Operations
class LinearOperator {
public:
    virtual ~LinearOperator() = default;
    virtual Eigen::Index rows() const = 0;
    virtual Eigen::Index cols() const = 0;
    virtual Eigen::VectorXd multiply(const Eigen::VectorXd& x) const = 0;
};

// Wrapper for standard Dense Matrices
class DenseOperator : public LinearOperator {
public:
    explicit DenseOperator(Eigen::MatrixXd mat) : mat_(std::move(mat)) {}
    Eigen::Index rows() const override { return mat_.rows(); }
    Eigen::Index cols() const override { return mat_.cols(); }
    Eigen::VectorXd multiply(const Eigen::VectorXd& x) const override {
        return mat_ * x;
    }
    const Eigen::MatrixXd& mat() const { return mat_; }
private:
    Eigen::MatrixXd mat_;
};

// Sparse + Low Rank + Diagonal component
struct CompositeComponent {
    Eigen::SparseMatrix<double> S;
    Eigen::MatrixXd U;
    Eigen::MatrixXd V;
    Eigen::VectorXd b;
};

// Matrix-Free operator wrapping a list of components
class CompositeOperator : public LinearOperator {
public:
    CompositeOperator(Eigen::Index size, std::vector<CompositeComponent> components) 
        : size_(size), components_(std::move(components)) {}

    Eigen::Index rows() const override { return size_; }
    Eigen::Index cols() const override { return size_; }

    Eigen::VectorXd multiply(const Eigen::VectorXd& x) const override {
        Eigen::VectorXd res = Eigen::VectorXd::Zero(size_);
        for (const auto& comp : components_) {
            if (comp.S.nonZeros() > 0) {
                res += comp.S * x;
            }
            if (comp.U.cols() > 0) {
                res += comp.U * (comp.V.transpose() * x);
            }
            if (comp.b.size() > 0) {
                res += comp.b.cwiseProduct(x);
            }
        }
        return res;
    }

private:
    Eigen::Index size_;
    std::vector<CompositeComponent> components_;
};

class SelectiveInference {
public:
    SelectiveInference(Eigen::VectorXd Z, 
                       Eigen::VectorXd Z_noisy, 
                       std::shared_ptr<LinearOperator> Q, 
                       std::shared_ptr<LinearOperator> Q_noise);

    // Provide a convenience constructor for backwards compatibility
    SelectiveInference(Eigen::VectorXd Z, 
                       Eigen::VectorXd Z_noisy, 
                       Eigen::MatrixXd Q, 
                       Eigen::MatrixXd Q_noise);

    // Expose the solve step explicitly
    Eigen::VectorXd solve_contrast(const Eigen::VectorXd& v) const;

    Params compute_params(const Eigen::VectorXd& v) const;
    std::pair<double, double> data_splitting_estimator(const Eigen::VectorXd& v) const;

    std::pair<double, double> get_interval(const Eigen::VectorXd& v, double t, 
                                           const Eigen::MatrixXd& A, const Eigen::VectorXd& b) const;

    std::function<Eigen::VectorXd(const Eigen::VectorXd&)> get_weight(const Eigen::VectorXd& v, const Eigen::MatrixXd& A, const Eigen::VectorXd& b) const;

private:
    Eigen::VectorXd Z_;
    Eigen::VectorXd Z_noisy_;
    std::shared_ptr<LinearOperator> Q_;
    std::shared_ptr<LinearOperator> Q_noise_;
};

} // namespace lassoinf
