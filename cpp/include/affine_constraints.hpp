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

// Abstract base class for Matrix-Free Operations
class LinearOperator {
public:
    virtual ~LinearOperator() = default;
    virtual Eigen::Index rows() const = 0;
    virtual Eigen::Index cols() const = 0;
    virtual Eigen::VectorXd multiply(const Eigen::VectorXd& x) const = 0;
    virtual Eigen::VectorXd multiply_transpose(const Eigen::VectorXd& x) const = 0;
};

struct AffineConstraintsContrast {
    Eigen::VectorXd gamma;
    Eigen::VectorXd c;
    Eigen::VectorXd bar_gamma;
    double bar_s;
    Eigen::VectorXd n_o;
    Eigen::VectorXd bar_n_o;
    double theta_hat;
    double bar_theta;
    double splitting_variance;
    double splitting_estimator;
    double naive_variance;

    std::pair<double, double> get_interval(double t, const LinearOperator& A, const Eigen::VectorXd& b) const;
    std::pair<double, double> get_interval(double t, const Eigen::MatrixXd& A, const Eigen::VectorXd& b) const;

    std::function<Eigen::VectorXd(const Eigen::VectorXd&)> get_weight(const LinearOperator& A, const Eigen::VectorXd& b) const;
    std::function<Eigen::VectorXd(const Eigen::VectorXd&)> get_weight(const Eigen::MatrixXd& A, const Eigen::VectorXd& b) const;
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

namespace detail {

inline Eigen::VectorXd embed(Eigen::Index n, const std::vector<int>& idx, const Eigen::VectorXd& vals) {
    Eigen::VectorXd z = Eigen::VectorXd::Zero(n);
    for (size_t i = 0; i < idx.size(); ++i) z(idx[i]) = vals(i);
    return z;
}

inline Eigen::VectorXd take(const Eigen::VectorXd& x, const std::vector<int>& idx) {
    Eigen::VectorXd out(idx.size());
    for (size_t i = 0; i < idx.size(); ++i) out(i) = x(idx[i]);
    return out;
}

inline double sign(double x) { return x > 0 ? 1.0 : (x < 0 ? -1.0 : 0.0); }

} // namespace detail

// Inactive scores U_{-E}(x) = x_{-E} - Q_{-E,E} W x_E with W = Q_{E,E}^{-1},
// applied with one Q matvec (Q symmetric); nothing of size p x |E| is stored.
class InactiveScoreOperator : public LinearOperator {
public:
    InactiveScoreOperator(std::shared_ptr<LinearOperator> Q, std::vector<int> E,
                          std::vector<int> E_c, Eigen::MatrixXd W)
        : Q_(std::move(Q)), E_(std::move(E)), E_c_(std::move(E_c)), W_(std::move(W)) {}

    Eigen::Index rows() const override { return static_cast<Eigen::Index>(E_c_.size()); }
    Eigen::Index cols() const override { return Q_->rows(); }

    // W x_E
    Eigen::VectorXd coef(const Eigen::VectorXd& x) const;
    Eigen::VectorXd multiply(const Eigen::VectorXd& x) const override;
    Eigen::VectorXd multiply_transpose(const Eigen::VectorXd& y) const override;

    const std::shared_ptr<LinearOperator>& Q() const { return Q_; }
    const std::vector<int>& E() const { return E_; }
    const std::vector<int>& E_c() const { return E_c_; }
    const Eigen::MatrixXd& W() const { return W_; }

private:
    std::shared_ptr<LinearOperator> Q_;
    std::vector<int> E_;
    std::vector<int> E_c_;
    Eigen::MatrixXd W_;
};

// A x = [R_active (W x_E); R_inactive U_{-E}(x)]
class LassoConstraintOperator : public LinearOperator {
public:
    LassoConstraintOperator(std::shared_ptr<InactiveScoreOperator> score,
                            Eigen::SparseMatrix<double> R_active,
                            Eigen::SparseMatrix<double> R_inactive)
        : score_(std::move(score)), R_active_(std::move(R_active)), R_inactive_(std::move(R_inactive)) {}

    Eigen::Index rows() const override { return R_active_.rows() + R_inactive_.rows(); }
    Eigen::Index cols() const override { return score_->cols(); }

    Eigen::VectorXd multiply(const Eigen::VectorXd& x) const override;
    Eigen::VectorXd multiply_transpose(const Eigen::VectorXd& y) const override;

    const std::shared_ptr<InactiveScoreOperator>& score() const { return score_; }
    const Eigen::SparseMatrix<double>& R_active() const { return R_active_; }
    const Eigen::SparseMatrix<double>& R_inactive() const { return R_inactive_; }

private:
    std::shared_ptr<InactiveScoreOperator> score_;
    Eigen::SparseMatrix<double> R_active_;
    Eigen::SparseMatrix<double> R_inactive_;
};

// x -> P (R x) with P sparse
class SparseProductOperator : public LinearOperator {
public:
    SparseProductOperator(Eigen::SparseMatrix<double> P, std::shared_ptr<LinearOperator> R)
        : P_(std::move(P)), R_(std::move(R)) {}

    Eigen::Index rows() const override { return P_.rows(); }
    Eigen::Index cols() const override { return R_->cols(); }

    Eigen::VectorXd multiply(const Eigen::VectorXd& x) const override {
        return P_ * R_->multiply(x);
    }
    Eigen::VectorXd multiply_transpose(const Eigen::VectorXd& y) const override {
        return R_->multiply_transpose(P_.transpose() * y);
    }

private:
    Eigen::SparseMatrix<double> P_;
    std::shared_ptr<LinearOperator> R_;
};

// [A_1; A_2; ...]
class VStackOperator : public LinearOperator {
public:
    explicit VStackOperator(std::vector<std::shared_ptr<LinearOperator>> ops);

    Eigen::Index rows() const override { return rows_; }
    Eigen::Index cols() const override { return ops_.front()->cols(); }

    Eigen::VectorXd multiply(const Eigen::VectorXd& x) const override;
    Eigen::VectorXd multiply_transpose(const Eigen::VectorXd& y) const override;

private:
    std::vector<std::shared_ptr<LinearOperator>> ops_;
    Eigen::Index rows_;
};

struct LassoConstraints {
    std::shared_ptr<LassoConstraintOperator> A;
    Eigen::VectorXd b;
    std::vector<int> E;
    std::vector<int> E_c;
    Eigen::VectorXd s_E;
    Eigen::VectorXd v_Ec;
    Eigen::MatrixXd W;  // Q_{E,E}^{-1}
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

class AffineConstraints {
public:
    AffineConstraints(Eigen::VectorXd Z, 
                       Eigen::VectorXd Z_noisy, 
                       std::shared_ptr<LinearOperator> Q, 
                       std::shared_ptr<LinearOperator> Q_noise,
                       double scalar_noise = std::numeric_limits<double>::quiet_NaN());

    // Provide a convenience constructor for backwards compatibility
    AffineConstraints(Eigen::VectorXd Z, 
                       Eigen::VectorXd Z_noisy, 
                       Eigen::MatrixXd Q, 
                       Eigen::MatrixXd Q_noise,
                       double scalar_noise = std::numeric_limits<double>::quiet_NaN());

    // Expose the solve step explicitly
    Eigen::VectorXd solve_contrast(const Eigen::VectorXd& v) const;

    AffineConstraintsContrast compute_contrast(const Eigen::VectorXd& v) const;

    // Contrast for an estimator specified by Var(theta_hat) and Cov(Z, theta_hat)
    // instead of a direction; requires Q_noise.
    AffineConstraintsContrast compute_covariance_contrast(double theta_hat,
                                                          double variance,
                                                          const Eigen::VectorXd& score_cov) const;

    const Eigen::VectorXd& Z() const { return Z_; }
    const Eigen::VectorXd& Z_noisy() const { return Z_noisy_; }

private:
    // Q_noise^{-1} rhs
    Eigen::VectorXd solve_noise(const Eigen::VectorXd& rhs) const;

    Eigen::VectorXd Z_;
    Eigen::VectorXd Z_noisy_;
    std::shared_ptr<LinearOperator> Q_;
    std::shared_ptr<LinearOperator> Q_noise_;
    double scalar_noise_;
};

} // namespace lassoinf
