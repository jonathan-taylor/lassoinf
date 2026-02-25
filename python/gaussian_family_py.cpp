#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <pybind11/eigen.h>
#include "gaussian_family.hpp"

namespace py = pybind11;

void init_gaussian_family(py::module_ &m) {
    py::class_<lassoinf::TruncatedGaussian>(m, "TruncatedGaussian")
        .def(py::init<double, double, double, double, double, double, double>(),
             py::arg("estimate"), py::arg("sigma"), py::arg("smoothing_sigma"), py::arg("lower_bound"), py::arg("upper_bound"), py::arg("noisy_estimate"), py::arg("factor") = 1.0)
        .def("weight", [](const lassoinf::TruncatedGaussian& tg, py::object t) -> py::object {
            if (py::isinstance<py::float_>(t) || py::isinstance<py::int_>(t)) {
                Eigen::VectorXd t_vec(1);
                t_vec(0) = t.cast<double>();
                return py::cast(tg.weight(t_vec)(0));
            } else {
                return py::cast(tg.weight(t.cast<Eigen::VectorXd>()));
            }
        }, py::arg("x"));

    py::class_<lassoinf::WeightedGaussianFamily>(m, "WeightedGaussianFamily")
        .def(py::init<double, double, const std::vector<std::function<Eigen::VectorXd(const Eigen::VectorXd&)>>&, double, int>(),
             py::arg("estimate"), py::arg("sigma"), py::arg("weight_fns"), py::arg("num_sd") = 10.0, py::arg("num_grid") = 4000)
        .def("pvalue", &lassoinf::WeightedGaussianFamily::pvalue,
             py::arg("null_value") = 0.0, py::arg("alternative") = "twosided", py::arg("basept") = std::numeric_limits<double>::quiet_NaN())
        .def("interval", &lassoinf::WeightedGaussianFamily::interval,
             py::arg("basept") = std::numeric_limits<double>::quiet_NaN(), py::arg("level") = 0.9);
}