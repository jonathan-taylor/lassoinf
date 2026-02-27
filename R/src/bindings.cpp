#include <RcppEigen.h>

// [[Rcpp::depends(RcppEigen)]]

// 1. Include the headers
#include "../cpp/include/selective_inference.hpp"
#include "../cpp/include/discrete_family.h"
#include "../cpp/include/gaussian_family.hpp"

// 2. Unity build: include the C++ sources directly to avoid duplicate symbols
//    and bypass the need for a complex Makefile to compile them individually.
#include "../cpp/src/selective_inference.cpp"
#include "../cpp/src/lasso_post_selection_constraints.cpp"
#include "../cpp/src/discrete_family.cpp"
#include "../cpp/src/gaussian_family.cpp"

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
NumericVector get_interval_wrapper(lassoinf::SelectiveInference* si, const Eigen::VectorXd& v, double t, const Eigen::MatrixXd& A, const Eigen::VectorXd& b) {
    auto res = si->get_interval(v, t, A, b);
    return NumericVector::create(res.first, res.second);
}

NumericVector data_splitting_estimator_wrapper(lassoinf::SelectiveInference* si, const Eigen::VectorXd& v) {
    auto res = si->data_splitting_estimator(v);
    return NumericVector::create(res.first, res.second);
}

Rcpp::List compute_params_wrapper(lassoinf::SelectiveInference* si, const Eigen::VectorXd& v) {
    auto p = si->compute_params(v);
    Rcpp::Environment env = Rcpp::Environment::empty_env();
    Rcpp::List list = Rcpp::List::create(
        Rcpp::Named("gamma") = p.gamma,
        Rcpp::Named("c") = p.c,
        Rcpp::Named("bar_gamma") = p.bar_gamma,
        Rcpp::Named("bar_s") = p.bar_s,
        Rcpp::Named("n_o") = p.n_o,
        Rcpp::Named("bar_n_o") = p.bar_n_o,
        Rcpp::Named("theta_hat") = p.theta_hat,
        Rcpp::Named("bar_theta") = p.bar_theta
    );
    return list; // returning list is easier
}

Rcpp::List lasso_post_selection_constraints_wrapper(
    const Eigen::VectorXd& beta_hat,
    const Eigen::VectorXd& G,
    const Eigen::MatrixXd& Q,
    const Eigen::VectorXd& D_diag,
    const Eigen::VectorXd& L,
    const Eigen::VectorXd& U,
    double tol) {
    
    std::shared_ptr<lassoinf::LinearOperator> Q_ptr = std::make_shared<lassoinf::DenseOperator>(Q);
    auto constraints = lassoinf::lasso_post_selection_constraints(beta_hat, G, Q_ptr, D_diag, L, U, tol);
    
    return Rcpp::List::create(
        Rcpp::Named("A_dense") = get_A_dense(constraints),
        Rcpp::Named("b") = constraints.b,
        Rcpp::Named("E") = constraints.E,
        Rcpp::Named("E_c") = constraints.E_c,
        Rcpp::Named("s_E") = constraints.s_E,
        Rcpp::Named("v_Ec") = constraints.v_Ec
    );
}

// DiscreteFamily wrappers
double df_cdf_wrapper(lassoinf::DiscreteFamily* df, double theta, double x, double gamma) {
    return df->cdf(theta, x, gamma);
}
double df_ccdf_wrapper(lassoinf::DiscreteFamily* df, double theta, double x, double gamma) {
    return df->ccdf(theta, x, gamma);
}
NumericVector df_equal_tailed_interval_wrapper(lassoinf::DiscreteFamily* df, double observed, double alpha, double tol) {
    auto res = df->equal_tailed_interval(observed, alpha, tol);
    return NumericVector::create(res.first, res.second);
}

// WeightedGaussianFamily wrapper for constructor
lassoinf::WeightedGaussianFamily* create_weighted_gaussian_family(
    double estimate, double sigma, Rcpp::List r_weight_fns, double num_sd, int num_grid) {
    
    std::vector<std::function<Eigen::VectorXd(const Eigen::VectorXd&)>> weight_fns;
    for (int i = 0; i < r_weight_fns.size(); ++i) {
        Rcpp::Function r_fn = r_weight_fns[i];
        weight_fns.push_back([r_fn](const Eigen::VectorXd& x) -> Eigen::VectorXd {
            Rcpp::NumericVector rx = Rcpp::wrap(x);
            Rcpp::NumericVector r_res = r_fn(rx);
            return Rcpp::as<Eigen::VectorXd>(r_res);
        });
    }
    return new lassoinf::WeightedGaussianFamily(estimate, sigma, weight_fns, num_sd, num_grid);
}

// WeightedGaussianFamily wrappers
NumericVector wgf_interval_wrapper(lassoinf::WeightedGaussianFamily* wgf, double basept, double level) {
    auto res = wgf->interval(basept, level);
    return NumericVector::create(res.first, res.second);
}

RCPP_MODULE(lassoinf_cpp) {
    function("lasso_post_selection_constraints", &lasso_post_selection_constraints_wrapper, 
             List::create(_["beta_hat"], _["G"], _["Q"], _["D_diag"], _["L"], _["U"], _["tol"] = 1e-6));

    class_<lassoinf::SelectiveInference>("SelectiveInference")
        .constructor<Eigen::VectorXd, Eigen::VectorXd, Eigen::MatrixXd, Eigen::MatrixXd>()
        .method("solve_contrast", &lassoinf::SelectiveInference::solve_contrast)
        .method("compute_params", &compute_params_wrapper)
        .method("data_splitting_estimator", &data_splitting_estimator_wrapper)
        .method("get_interval", &get_interval_wrapper)
        ;

    class_<lassoinf::DiscreteFamily>("DiscreteFamily")
        .constructor<std::vector<double>, std::vector<double>, double>()
        .method("get_theta", &lassoinf::DiscreteFamily::get_theta)
        .method("set_theta", &lassoinf::DiscreteFamily::set_theta)
        .method("get_partition", &lassoinf::DiscreteFamily::get_partition)
        .method("pdf", &lassoinf::DiscreteFamily::pdf)
        .method("cdf", &df_cdf_wrapper)
        .method("ccdf", &df_ccdf_wrapper)
        .method("equal_tailed_interval", &df_equal_tailed_interval_wrapper)
        ;

    class_<lassoinf::TruncatedGaussian>("TruncatedGaussian")
        .constructor<double, double, double, double, double, double, double>()
        .method("weight", &lassoinf::TruncatedGaussian::weight)
        ;

    class_<lassoinf::WeightedGaussianFamily>("WeightedGaussianFamily")
        .factory<double, double, Rcpp::List, double, int>(&create_weighted_gaussian_family)
        .method("pvalue", &lassoinf::WeightedGaussianFamily::pvalue)
        .method("interval", &wgf_interval_wrapper)
        ;
}