#include "selective_inference.hpp"
#include "gaussian_family.hpp"
#include <Eigen/IterativeLinearSolvers>

namespace lassoinf {

SelectiveInference::SelectiveInference(Eigen::VectorXd Z, 
                                       Eigen::VectorXd Z_noisy, 
                                       std::shared_ptr<LinearOperator> Q, 
                                       std::shared_ptr<LinearOperator> Q_noise)
    : Z_(std::move(Z)), Z_noisy_(std::move(Z_noisy)), Q_(std::move(Q)), Q_noise_(std::move(Q_noise)) {}

SelectiveInference::SelectiveInference(Eigen::VectorXd Z, 
                                       Eigen::VectorXd Z_noisy, 
                                       Eigen::MatrixXd Q, 
                                       Eigen::MatrixXd Q_noise)
    : Z_(std::move(Z)), Z_noisy_(std::move(Z_noisy)), 
      Q_(std::make_shared<DenseOperator>(std::move(Q))), 
      Q_noise_(std::make_shared<DenseOperator>(std::move(Q_noise))) {}

} // namespace lassoinf

namespace lassoinf {
class EigenLinearOperatorProxy;
}

namespace Eigen {
namespace internal {
  template<>
  struct traits<lassoinf::EigenLinearOperatorProxy> : public Eigen::internal::traits<Eigen::SparseMatrix<double>> {};
}
}

namespace lassoinf {

// A proxy class to allow Eigen's ConjugateGradient to work with our polymorphic LinearOperator
class EigenLinearOperatorProxy : public Eigen::EigenBase<EigenLinearOperatorProxy> {
public:
    typedef double Scalar;
    typedef double RealScalar;
    typedef Eigen::Index StorageIndex;
    enum {
        ColsAtCompileTime = Eigen::Dynamic,
        MaxColsAtCompileTime = Eigen::Dynamic,
        IsRowMajor = false
    };

    using Index = Eigen::Index;
    explicit EigenLinearOperatorProxy(const LinearOperator* op) : op_(op) {}

    Index rows() const { return op_->rows(); }
    Index cols() const { return op_->cols(); }

    template<typename Rhs>
    Eigen::Product<EigenLinearOperatorProxy, Rhs, Eigen::AliasFreeProduct> operator*(const Eigen::MatrixBase<Rhs>& x) const {
        return Eigen::Product<EigenLinearOperatorProxy, Rhs, Eigen::AliasFreeProduct>(*this, x.derived());
    }

    Eigen::VectorXd multiply(const Eigen::VectorXd& x) const {
        return op_->multiply(x);
    }
private:
    const LinearOperator* op_;
};

} // namespace lassoinf

namespace Eigen {
namespace internal {
  template<typename Rhs>
  struct generic_product_impl<lassoinf::EigenLinearOperatorProxy, Rhs, SparseShape, DenseShape, GemvProduct>
  : generic_product_impl_base<lassoinf::EigenLinearOperatorProxy, Rhs, generic_product_impl<lassoinf::EigenLinearOperatorProxy, Rhs>>
  {
    typedef typename Product<lassoinf::EigenLinearOperatorProxy, Rhs>::Scalar Scalar;

    template<typename Dest>
    static void scaleAndAddTo(Dest& dst, const lassoinf::EigenLinearOperatorProxy& lhs, const Rhs& rhs, const Scalar& alpha)
    {
      dst += alpha * lhs.multiply(rhs);
    }
  };
}
}

namespace lassoinf {

Eigen::VectorXd SelectiveInference::solve_contrast(const Eigen::VectorXd& v) const {
    Eigen::VectorXd Q_v = Q_->multiply(v);

    // If Q_noise is just a dense matrix, we can use a direct solver for efficiency
    if (auto dense_op = dynamic_cast<const DenseOperator*>(Q_noise_.get())) {
        return dense_op->mat().colPivHouseholderQr().solve(Q_v);
    }

    // Otherwise, use Conjugate Gradient for matrix-free / composite operators
    EigenLinearOperatorProxy proxy(Q_noise_.get());
    Eigen::ConjugateGradient<EigenLinearOperatorProxy, Eigen::Lower|Eigen::Upper, Eigen::IdentityPreconditioner> cg;
    cg.compute(proxy);
    Eigen::VectorXd c = cg.solve(Q_v);
    
    if (cg.info() != Eigen::Success) {
        throw std::runtime_error("Conjugate Gradient did not converge in solve_contrast");
    }
    
    return c;
}

std::pair<double, double> SelectiveInference::data_splitting_estimator(const Eigen::VectorXd& v) const {
    Params p = compute_params(v);
    double variance = v.dot(Q_->multiply(v)) + std::pow(p.bar_s, 2);
    double estimator = p.theta_hat - p.bar_theta;
    return {estimator, variance};
}

Params SelectiveInference::compute_params(const Eigen::VectorXd& v) const {
    Params p;
    Eigen::VectorXd Q_v = Q_->multiply(v);
    double v_sigma_v = v.dot(Q_v);
    
    p.gamma = Q_v / v_sigma_v;
    p.c = solve_contrast(v);
    
    Eigen::VectorXd Q_noise_c = Q_noise_->multiply(p.c);
    double bar_s2 = p.c.dot(Q_noise_c);
    p.bar_s = std::sqrt(bar_s2);
    p.bar_gamma = Q_v / bar_s2;
    p.theta_hat = v.dot(Z_);
    p.n_o = Z_ - p.gamma * p.theta_hat;
    
    Eigen::VectorXd omega = Z_noisy_ - Z_;
    p.bar_theta = p.c.dot(omega);
    p.bar_n_o = omega - p.bar_gamma * p.bar_theta;
    
    return p;
}

std::pair<double, double> SelectiveInference::get_interval(const Eigen::VectorXd& v, double t, 
                                                           const LinearOperator& A, const Eigen::VectorXd& b) const {
    Params p = compute_params(v);
    
    Eigen::VectorXd alpha = A.multiply(p.bar_gamma);
    Eigen::VectorXd beta = b - A.multiply(p.n_o + p.bar_n_o + p.gamma * t);
    
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
    
    if (lower > upper) {
        return {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN()};
    }
    
    return {lower, upper};
}

std::pair<double, double> SelectiveInference::get_interval(const Eigen::VectorXd& v, double t, 
                                                           const Eigen::MatrixXd& A, const Eigen::VectorXd& b) const {
    return get_interval(v, t, DenseOperator(A), b);
}

std::function<Eigen::VectorXd(const Eigen::VectorXd&)> SelectiveInference::get_weight(const Eigen::VectorXd& v, const LinearOperator& A, const Eigen::VectorXd& b) const {
    Params p = compute_params(v);
    double bar_s = p.bar_s;
    
    // We need a way to capture A. Since it's a reference, we might need a copy if it's used asynchronously,
    // but here we assume it lives as long as the weight function is used for grid evaluation.
    // Actually, A is often a shared_ptr elsewhere. For now, let's assume it's stable.
    
    return [this, v, &A, b, bar_s](const Eigen::VectorXd& t_vec) -> Eigen::VectorXd {
        Eigen::VectorXd result(t_vec.size());
        auto norm_cdf = [](double x) {
            return 0.5 * std::erfc(-x / std::sqrt(2.0));
        };
        for (int i = 0; i < t_vec.size(); ++i) {
            double t = t_vec(i);
            auto interval = this->get_interval(v, t, A, b);
            double L = interval.first;
            double U = interval.second;
            if (std::isnan(L) || std::isnan(U)) {
                result(i) = 0.0;
            } else {
                result(i) = norm_cdf(U / bar_s) - norm_cdf(L / bar_s);
            }
        }
        return result;
    };
}

std::function<Eigen::VectorXd(const Eigen::VectorXd&)> SelectiveInference::get_weight(const Eigen::VectorXd& v, const Eigen::MatrixXd& A, const Eigen::VectorXd& b) const {
    // Note: This creates a temporary DenseOperator which might be dangerous if captured by reference.
    // However, get_weight above captures by reference. 
    // Let's make a version that works safely.
    
    auto dense_A = std::make_shared<DenseOperator>(A);
    Params p = compute_params(v);
    double bar_s = p.bar_s;
    
    return [this, v, dense_A, b, bar_s](const Eigen::VectorXd& t_vec) -> Eigen::VectorXd {
        Eigen::VectorXd result(t_vec.size());
        auto norm_cdf = [](double x) {
            return 0.5 * std::erfc(-x / std::sqrt(2.0));
        };
        for (int i = 0; i < t_vec.size(); ++i) {
            double t = t_vec(i);
            auto interval = this->get_interval(v, t, *dense_A, b);
            double L = interval.first;
            double U = interval.second;
            if (std::isnan(L) || std::isnan(U)) {
                result(i) = 0.0;
            } else {
                result(i) = norm_cdf(U / bar_s) - norm_cdf(L / bar_s);
            }
        }
        return result;
    };
}

LassoInference::LassoInference(Eigen::VectorXd beta_hat,
                               Eigen::VectorXd G_hat,
                               std::shared_ptr<LinearOperator> Q_hat,
                               Eigen::VectorXd D,
                               Eigen::VectorXd L,
                               Eigen::VectorXd U,
                               Eigen::VectorXd Z_full,
                               std::shared_ptr<LinearOperator> Sigma,
                               std::shared_ptr<LinearOperator> Sigma_noisy)
    : beta_hat_(std::move(beta_hat)), G_hat_(std::move(G_hat)), Q_hat_(std::move(Q_hat)),
      D_(std::move(D)), L_(std::move(L)), U_(std::move(U)), Z_full_(std::move(Z_full)),
      Sigma_(std::move(Sigma)), Sigma_noisy_(std::move(Sigma_noisy))
{
    constraints_ = lasso_post_selection_constraints(beta_hat_, G_hat_, Q_hat_, D_, L_, U_);
    
    Eigen::VectorXd Z_noisy = -G_hat_ + Q_hat_->multiply(beta_hat_);
    SelectiveInference si(Z_full_, Z_noisy, Sigma_, Sigma_noisy_);
    
    if (!constraints_.E.empty()) {
        int n = Q_hat_->rows();
        Eigen::MatrixXd Q_EE(constraints_.E.size(), constraints_.E.size());
        for (size_t i = 0; i < constraints_.E.size(); ++i) {
            Eigen::VectorXd e_i = Eigen::VectorXd::Zero(n);
            e_i(constraints_.E[i]) = 1.0;
            Eigen::VectorXd Q_ei = Q_hat_->multiply(e_i);
            for (size_t j = 0; j < constraints_.E.size(); ++j) {
                Q_EE(j, i) = Q_ei(constraints_.E[j]);
            }
        }
        
        Eigen::MatrixXd W = Q_EE.inverse();
        
        for (size_t k = 0; k < constraints_.E.size(); ++k) {
            int idx = constraints_.E[k];
            Eigen::VectorXd v = Eigen::VectorXd::Zero(n);
            for (size_t i = 0; i < constraints_.E.size(); ++i) {
                v(constraints_.E[i]) = W(i, k);
            }
            
            double theta_hat = v.dot(Z_full_);
            double variance = v.dot(Sigma_->multiply(v));
            double sigma = std::sqrt(variance);
            
            auto weight_f = si.get_weight(v, *constraints_.A, constraints_.b);
            WeightedGaussianFamily wgf(theta_hat, sigma, {weight_f});
            auto interval = wgf.interval(theta_hat, 0.95);
            double p_val = wgf.pvalue(0.0, "twosided", theta_hat);
            
            results_.push_back({idx, beta_hat_(idx), interval.first, interval.second, p_val});
        }
    }
}

std::vector<InferenceResult> LassoInference::summary() const {
    return results_;
}

} // namespace lassoinf

