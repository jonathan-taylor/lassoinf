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
    virtual Eigen::VectorXd multiply_transpose(const Eigen::VectorXd& x) const = 0;
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
    Eigen::VectorXd multiply_transpose(const Eigen::VectorXd& x) const override {
        return mat_.transpose() * x;
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
    CompositeOperator(Eigen::Index rows, Eigen::Index cols, std::vector<CompositeComponent> components) 
        : rows_(rows), cols_(cols), components_(std::move(components)) {}

    Eigen::Index rows() const override { return rows_; }
    Eigen::Index cols() const override { return cols_; }

    Eigen::VectorXd multiply(const Eigen::VectorXd& x) const override {
        Eigen::VectorXd res = Eigen::VectorXd::Zero(rows_);
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

    Eigen::VectorXd multiply_transpose(const Eigen::VectorXd& x) const override {
        Eigen::VectorXd res = Eigen::VectorXd::Zero(cols_);
        for (const auto& comp : components_) {
            if (comp.S.nonZeros() > 0) {
                res += comp.S.transpose() * x;
            }
            if (comp.V.cols() > 0) {
                res += comp.V * (comp.U.transpose() * x);
            }
            if (comp.b.size() > 0) {
                res += comp.b.cwiseProduct(x);
            }
        }
        return res;
    }

private:
    Eigen::Index rows_;
    Eigen::Index cols_;
    std::vector<CompositeComponent> components_;
};

class XTVXOperator : public LinearOperator {
public:
    XTVXOperator(Eigen::MatrixXd X, std::shared_ptr<LinearOperator> V)
        : X_(std::move(X)), V_(std::move(V)) {}

    Eigen::Index rows() const override { return X_.cols(); }
    Eigen::Index cols() const override { return X_.cols(); }

    Eigen::VectorXd multiply(const Eigen::VectorXd& x) const override {
        return X_.transpose() * V_->multiply(X_ * x);
    }

    Eigen::VectorXd multiply_transpose(const Eigen::VectorXd& x) const override {
        // (X^T V X)^T = X^T V^T X
        return X_.transpose() * V_->multiply_transpose(X_ * x);
    }

private:
    Eigen::MatrixXd X_;
    std::shared_ptr<LinearOperator> V_;
};

struct LassoConstraints {
    std::shared_ptr<CompositeOperator> A;
    Eigen::VectorXd b;
    std::vector<int> E;
    std::vector<int> E_c;
    Eigen::VectorXd s_E;
    Eigen::VectorXd v_Ec;
};

LassoConstraints lasso_post_selection_constraints(
    const Eigen::VectorXd& beta_hat,
    const Eigen::VectorXd& G,
    std::shared_ptr<LinearOperator> Q,
    const Eigen::VectorXd& D_diag,
    const Eigen::VectorXd& L, // Empty vector means None
    const Eigen::VectorXd& U, // Empty vector means None
    double tol = 1e-6
);

struct InferenceResult {
    int index;
    double beta_hat;
    double lower_conf;
    double upper_conf;
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
                                           const LinearOperator& A, const Eigen::VectorXd& b) const;

    std::pair<double, double> get_interval(const Eigen::VectorXd& v, double t, 
                                           const Eigen::MatrixXd& A, const Eigen::VectorXd& b) const;

    std::function<Eigen::VectorXd(const Eigen::VectorXd&)> get_weight(const Eigen::VectorXd& v, const LinearOperator& A, const Eigen::VectorXd& b) const;

    std::function<Eigen::VectorXd(const Eigen::VectorXd&)> get_weight(const Eigen::VectorXd& v, const Eigen::MatrixXd& A, const Eigen::VectorXd& b) const;

private:
    Eigen::VectorXd Z_;
    Eigen::VectorXd Z_noisy_;
    std::shared_ptr<LinearOperator> Q_;
    std::shared_ptr<LinearOperator> Q_noise_;
};

class LassoInference {
public:
    LassoInference(Eigen::VectorXd beta_hat,
                   Eigen::VectorXd G_hat,
                   std::shared_ptr<LinearOperator> Q_hat,
                   Eigen::VectorXd D,
                   Eigen::VectorXd L,
                   Eigen::VectorXd U,
                   Eigen::VectorXd Z_full,
                   std::shared_ptr<LinearOperator> Sigma,
                   std::shared_ptr<LinearOperator> Sigma_noisy);

    std::vector<InferenceResult> summary() const;

private:
    Eigen::VectorXd beta_hat_;
    Eigen::VectorXd G_hat_;
    std::shared_ptr<LinearOperator> Q_hat_;
    Eigen::VectorXd D_;
    Eigen::VectorXd L_;
    Eigen::VectorXd U_;
    Eigen::VectorXd Z_full_;
    std::shared_ptr<LinearOperator> Sigma_;
    std::shared_ptr<LinearOperator> Sigma_noisy_;

    LassoConstraints constraints_;
    std::vector<InferenceResult> results_;
};

} // namespace lassoinf
