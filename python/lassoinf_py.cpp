#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>
#include <memory>
#include "selective_inference.hpp"

namespace py = pybind11;

void init_discrete_family(py::module_ &m);
void init_gaussian_family(py::module_ &m);

// Pybind11 trampoline class for LinearOperator
class PyLinearOperator : public lassoinf::LinearOperator {
public:
    using lassoinf::LinearOperator::LinearOperator;

    Eigen::Index rows() const override {
        PYBIND11_OVERRIDE_PURE(Eigen::Index, lassoinf::LinearOperator, rows);
    }

    Eigen::Index cols() const override {
        PYBIND11_OVERRIDE_PURE(Eigen::Index, lassoinf::LinearOperator, cols);
    }

    Eigen::VectorXd multiply(const Eigen::VectorXd& x) const override {
        PYBIND11_OVERRIDE_PURE(Eigen::VectorXd, lassoinf::LinearOperator, multiply, x);
    }
};

PYBIND11_MODULE(lassoinf_cpp, m) {
    m.doc() = "C++ implementation of SelectiveInference using Eigen and pybind11";

    init_discrete_family(m);
    init_gaussian_family(m);

    py::class_<lassoinf::Params>(m, "Params")
        .def_readonly("gamma", &lassoinf::Params::gamma)
        .def_readonly("c", &lassoinf::Params::c)
        .def_readonly("bar_gamma", &lassoinf::Params::bar_gamma)
        .def_readonly("bar_s", &lassoinf::Params::bar_s)
        .def_readonly("n_o", &lassoinf::Params::n_o)
        .def_readonly("bar_n_o", &lassoinf::Params::bar_n_o)
        .def_readonly("theta_hat", &lassoinf::Params::theta_hat)
        .def_readonly("bar_theta", &lassoinf::Params::bar_theta);

    py::class_<lassoinf::LinearOperator, PyLinearOperator, std::shared_ptr<lassoinf::LinearOperator>>(m, "LinearOperator")
        .def(py::init<>())
        .def("rows", &lassoinf::LinearOperator::rows)
        .def("cols", &lassoinf::LinearOperator::cols)
        .def("multiply", &lassoinf::LinearOperator::multiply, py::arg("x"));

    py::class_<lassoinf::DenseOperator, lassoinf::LinearOperator, std::shared_ptr<lassoinf::DenseOperator>>(m, "DenseOperator")
        .def(py::init<Eigen::MatrixXd>(), py::arg("mat"));

    py::class_<lassoinf::CompositeComponent>(m, "CompositeComponent")
        .def(py::init<Eigen::SparseMatrix<double>, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::VectorXd>(),
             py::arg("S"), py::arg("U"), py::arg("V"), py::arg("b"));

    py::class_<lassoinf::CompositeOperator, lassoinf::LinearOperator, std::shared_ptr<lassoinf::CompositeOperator>>(m, "CompositeOperator")
        .def(py::init<Eigen::Index, std::vector<lassoinf::CompositeComponent>>(), py::arg("size"), py::arg("components"));

    py::class_<lassoinf::SelectiveInference>(m, "SelectiveInference")
        .def(py::init<Eigen::VectorXd, Eigen::VectorXd, std::shared_ptr<lassoinf::LinearOperator>, std::shared_ptr<lassoinf::LinearOperator>>(),
             py::arg("Z"), py::arg("Z_noisy"), py::arg("Q"), py::arg("Q_noise"))
        .def(py::init<Eigen::VectorXd, Eigen::VectorXd, Eigen::MatrixXd, Eigen::MatrixXd>(),
             py::arg("Z"), py::arg("Z_noisy"), py::arg("Q"), py::arg("Q_noise"))
        .def("solve_contrast", &lassoinf::SelectiveInference::solve_contrast, py::arg("v"))
        .def("compute_params", &lassoinf::SelectiveInference::compute_params, py::arg("v"))
        .def("data_splitting_estimator", &lassoinf::SelectiveInference::data_splitting_estimator, py::arg("v"))
        .def("get_interval", &lassoinf::SelectiveInference::get_interval, py::arg("v"), py::arg("t"), py::arg("A"), py::arg("b"))
        .def("get_weight", [](const lassoinf::SelectiveInference& si, const Eigen::VectorXd& v, const Eigen::MatrixXd& A, const Eigen::VectorXd& b) {
            auto func = si.get_weight(v, A, b);
            return py::cpp_function([func](py::object t) -> py::object {
                if (py::isinstance<py::float_>(t) || py::isinstance<py::int_>(t)) {
                    Eigen::VectorXd t_vec(1);
                    t_vec(0) = t.cast<double>();
                    return py::cast(func(t_vec)(0));
                } else {
                    return py::cast(func(t.cast<Eigen::VectorXd>()));
                }
            });
        }, py::arg("v"), py::arg("A"), py::arg("b"));
}
