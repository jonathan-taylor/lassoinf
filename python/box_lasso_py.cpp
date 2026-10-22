#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include "box_lasso.hpp"

namespace py = pybind11;

void init_box_lasso(py::module_ &m) {
    m.def("box_lasso_fit", &lassoinf::box_lasso_fit, py::arg("Q"), py::arg("Y"), py::arg("D"),
          py::arg("L"), py::arg("U"), py::arg("beta0") = Eigen::VectorXd(),
          py::arg("tol") = 1e-12, py::arg("max_iter") = 100000);
    m.def("box_lasso_kkt_violation", &lassoinf::box_lasso_kkt_violation, py::arg("beta_hat"),
          py::arg("Q"), py::arg("Y"), py::arg("D"), py::arg("L"), py::arg("U"));
}
