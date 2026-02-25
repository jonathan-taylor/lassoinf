#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include "discrete_family.h"

namespace py = pybind11;

void init_discrete_family(py::module_ &m) {
    py::class_<lassoinf::DiscreteFamily>(m, "DiscreteFamilyCPP")
        .def(py::init<const std::vector<double>&, const std::vector<double>&, double>(),
             py::arg("sufficient_stat"), py::arg("weights"), py::arg("theta") = 0.0)
        .def_property("theta", &lassoinf::DiscreteFamily::get_theta, &lassoinf::DiscreteFamily::set_theta)
        .def_property_readonly("partition", &lassoinf::DiscreteFamily::get_partition)
        .def("pdf", &lassoinf::DiscreteFamily::pdf, py::arg("theta"))
        .def("cdf", &lassoinf::DiscreteFamily::cdf, py::arg("theta"), py::arg("x"), py::arg("gamma") = 1.0)
        .def("ccdf", &lassoinf::DiscreteFamily::ccdf, py::arg("theta"), py::arg("x"), py::arg("gamma") = 0.0)
        .def("equal_tailed_interval", &lassoinf::DiscreteFamily::equal_tailed_interval, 
             py::arg("observed"), py::arg("alpha") = 0.05, py::arg("tol") = 1e-6);
}
