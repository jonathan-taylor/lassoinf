#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <memory>
#include "unilasso.hpp"

namespace py = pybind11;

void init_unilasso(py::module_ &m) {
    py::class_<lassoinf::UniLassoPenalty>(m, "UniLassoPenalty")
        .def_readonly("C", &lassoinf::UniLassoPenalty::C)
        .def_readonly("s", &lassoinf::UniLassoPenalty::s)
        .def_readonly("D", &lassoinf::UniLassoPenalty::D)
        .def_readonly("L", &lassoinf::UniLassoPenalty::L)
        .def_readonly("U", &lassoinf::UniLassoPenalty::U);

    m.def("unilasso_penalty", &lassoinf::unilasso_penalty, py::arg("C"), py::arg("lam"));
    m.def("unilasso_fit", &lassoinf::unilasso_fit, py::arg("Q"), py::arg("Z_noisy"), py::arg("lam"),
          py::arg("tol") = 1e-12, py::arg("max_iter") = 100000);
    m.def("unilasso_kkt_violation", &lassoinf::unilasso_kkt_violation, py::arg("beta_hat"),
          py::arg("Q"), py::arg("Z_noisy"), py::arg("lam"));
    m.def("unilasso_loo_kappa", &lassoinf::unilasso_loo_kappa, py::arg("X"), py::arg("y"),
          py::arg("beta_hat"));
    m.def("quadratic_le_zero", &lassoinf::quadratic_le_zero, py::arg("q2"), py::arg("q1"), py::arg("q0"));
    m.def("unilasso_row_sets", &lassoinf::unilasso_row_sets, py::arg("u"), py::arg("a"), py::arg("b0"),
          py::arg("m"), py::arg("c"), py::arg("gamma"), py::arg("s"));
    m.def("intersect_row_sets", &lassoinf::intersect_row_sets, py::arg("sets"));
    m.def("unilasso_truncation_set", &lassoinf::unilasso_truncation_set,
          py::arg("beta_hat"), py::arg("G_hat"), py::arg("Q"), py::arg("D"), py::arg("L"), py::arg("U"),
          py::arg("k"), py::arg("lam_k"), py::arg("C_k"), py::arg("y0"), py::arg("direction"),
          py::arg("w_obs"), py::arg("gamma"), py::arg("tol") = 1e-6);
    m.def("unilasso_truncation_set",
          [](const Eigen::VectorXd& beta_hat, const Eigen::VectorXd& G_hat, const Eigen::MatrixXd& Q,
             const Eigen::VectorXd& D, const Eigen::VectorXd& L, const Eigen::VectorXd& U, int k,
             double lam_k, double C_k, const Eigen::VectorXd& y0, const Eigen::VectorXd& direction,
             double w_obs, double gamma, double tol) {
              return lassoinf::unilasso_truncation_set(beta_hat, G_hat,
                                                       std::make_shared<lassoinf::DenseOperator>(Q),
                                                       D, L, U, k, lam_k, C_k, y0, direction,
                                                       w_obs, gamma, tol);
          },
          py::arg("beta_hat"), py::arg("G_hat"), py::arg("Q"), py::arg("D"), py::arg("L"), py::arg("U"),
          py::arg("k"), py::arg("lam_k"), py::arg("C_k"), py::arg("y0"), py::arg("direction"),
          py::arg("w_obs"), py::arg("gamma"), py::arg("tol") = 1e-6);
}
