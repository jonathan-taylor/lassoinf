#include <RcppEigen.h>
#include "../cpp/include/selective_inference.hpp"

// [[Rcpp::depends(RcppEigen)]]

using namespace Rcpp;

// Wrapper for lasso_post_selection_constraints to handle shared_ptr and defaults
lassoinf::LassoConstraints lasso_post_selection_constraints_dense(
    const Eigen::VectorXd& beta_hat,
    const Eigen::VectorXd& G,
    const Eigen::MatrixXd& Q,
    const Eigen::VectorXd& D_diag,
    const Eigen::VectorXd& L,
    const Eigen::VectorXd& U,
    double tol) {
    
    std::shared_ptr<lassoinf::LinearOperator> Q_ptr = std::make_shared<lassoinf::DenseOperator>(Q);
    return lassoinf::lasso_post_selection_constraints(beta_hat, G, Q_ptr, D_diag, L, U, tol);
}

// Helper to convert CompositeOperator to Dense matrix
Eigen::MatrixXd operator_to_dense(const std::shared_ptr<lassoinf::LinearOperator>& op) {
    if (!op) return Eigen::MatrixXd(0, 0);
    int n = op->cols();
    int m = op->rows();
    Eigen::MatrixXd dense = Eigen::MatrixXd::Zero(m, n);
    for (int i = 0; i < n; ++i) {
        Eigen::VectorXd e = Eigen::VectorXd::Zero(n);
        e(i) = 1.0;
        dense.col(i) = op->multiply(e);
    }
    return dense;
}

Eigen::MatrixXd get_A_dense(const lassoinf::LassoConstraints& constraints) {
    return operator_to_dense(constraints.A);
}

// Helper to convert std::pair to NumericVector
NumericVector get_interval_wrapper(const lassoinf::SelectiveInference& si, const Eigen::VectorXd& v, double t, const Eigen::MatrixXd& A, const Eigen::VectorXd& b) {
    auto res = si.get_interval(v, t, A, b);
    return NumericVector::create(res.first, res.second);
}

RCPP_MODULE(lassoinf_cpp) {
    class_<lassoinf::Params>("Params")
        .constructor()
        .field("gamma", &lassoinf::Params::gamma)
        .field("c", &lassoinf::Params::c)
        .field("bar_gamma", &lassoinf::Params::bar_gamma)
        .field("bar_s", &lassoinf::Params::bar_s)
        .field("n_o", &lassoinf::Params::n_o)
        .field("bar_n_o", &lassoinf::Params::bar_n_o)
        .field("theta_hat", &lassoinf::Params::theta_hat)
        .field("bar_theta", &lassoinf::Params::bar_theta);

    class_<lassoinf::LassoConstraints>("LassoConstraints")
        .constructor()
        .method("get_A_dense", &get_A_dense)
        .field("b", &lassoinf::LassoConstraints::b)
        .field("E", &lassoinf::LassoConstraints::E)
        .field("E_c", &lassoinf::LassoConstraints::E_c)
        .field("s_E", &lassoinf::LassoConstraints::s_E)
        .field("v_Ec", &lassoinf::LassoConstraints::v_Ec);

    function("lasso_post_selection_constraints", &lasso_post_selection_constraints_dense, 
             List::create(_["beta_hat"], _["G"], _["Q"], _["D_diag"], _["L"], _["U"], _["tol"] = 1e-6));

    class_<lassoinf::SelectiveInference>("SelectiveInference")
        .constructor<Eigen::VectorXd, Eigen::VectorXd, Eigen::MatrixXd, Eigen::MatrixXd>()
        .method("solve_contrast", &lassoinf::SelectiveInference::solve_contrast)
        .method("compute_params", &lassoinf::SelectiveInference::compute_params)
        .method("data_splitting_estimator", &lassoinf::SelectiveInference::data_splitting_estimator)
        .method("get_interval", &get_interval_wrapper)
        ;
}
