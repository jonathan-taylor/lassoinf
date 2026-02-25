#include <RcppEigen.h>
#include "../cpp/include/lassoinf.hpp"

// [[Rcpp::depends(RcppEigen)]]

using namespace Rcpp;

RCPP_MODULE(lassoinf_cpp) {
    class_<lassoinf::Params>("Params")
        .field("gamma", &lassoinf::Params::gamma)
        .field("c", &lassoinf::Params::c)
        .field("bar_gamma", &lassoinf::Params::bar_gamma)
        .field("bar_s", &lassoinf::Params::bar_s)
        .field("n_o", &lassoinf::Params::n_o)
        .field("bar_n_o", &lassoinf::Params::bar_n_o)
        .field("theta_hat", &lassoinf::Params::theta_hat)
        .field("bar_theta", &lassoinf::Params::bar_theta);

    class_<lassoinf::SelectiveInference>("SelectiveInference")
        .constructor<Eigen::VectorXd, Eigen::VectorXd, Eigen::MatrixXd, Eigen::MatrixXd>()
        .method("compute_params", &lassoinf::SelectiveInference::compute_params)
        .method("get_interval", &lassoinf::SelectiveInference::get_interval)
        // NOTE: get_weight returns a std::function which Rcpp modules do not natively export without custom glue
        // .method("get_weight", &lassoinf::SelectiveInference::get_weight)
        ;
}
