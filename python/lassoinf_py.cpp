#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>
#include <memory>
#include "affine_constraints.hpp"
#include "custom_estimand.hpp"

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

    Eigen::VectorXd multiply_transpose(const Eigen::VectorXd& x) const override {
        PYBIND11_OVERRIDE_PURE(Eigen::VectorXd, lassoinf::LinearOperator, multiply_transpose, x);
    }
};

PYBIND11_MODULE(lassoinf_cpp, m) {
    m.doc() = "C++ implementation of SelectiveInference using Eigen and pybind11";

    init_discrete_family(m);
    init_gaussian_family(m);

    py::class_<lassoinf::AffineConstraintsContrast>(m, "AffineConstraintsContrast")
        .def_readonly("gamma", &lassoinf::AffineConstraintsContrast::gamma)
        .def_readonly("c", &lassoinf::AffineConstraintsContrast::c)
        .def_readonly("bar_gamma", &lassoinf::AffineConstraintsContrast::bar_gamma)
        .def_readonly("bar_s", &lassoinf::AffineConstraintsContrast::bar_s)
        .def_readonly("n_o", &lassoinf::AffineConstraintsContrast::n_o)
        .def_readonly("bar_n_o", &lassoinf::AffineConstraintsContrast::bar_n_o)
        .def_readonly("theta_hat", &lassoinf::AffineConstraintsContrast::theta_hat)
        .def_readonly("bar_theta", &lassoinf::AffineConstraintsContrast::bar_theta)
        .def_readonly("splitting_variance", &lassoinf::AffineConstraintsContrast::splitting_variance)
        .def_readonly("splitting_estimator", &lassoinf::AffineConstraintsContrast::splitting_estimator)
        .def_readonly("naive_variance", &lassoinf::AffineConstraintsContrast::naive_variance)
        .def("get_interval", py::overload_cast<double, const Eigen::MatrixXd&, const Eigen::VectorXd&>(&lassoinf::AffineConstraintsContrast::get_interval, py::const_), py::arg("t"), py::arg("A"), py::arg("b"))
        .def("get_interval", py::overload_cast<double, const lassoinf::LinearOperator&, const Eigen::VectorXd&>(&lassoinf::AffineConstraintsContrast::get_interval, py::const_), py::arg("t"), py::arg("A"), py::arg("b"))
        .def("get_weight", [](const lassoinf::AffineConstraintsContrast& contrast, const Eigen::MatrixXd& A, const Eigen::VectorXd& b) {
            auto func = contrast.get_weight(A, b);
            return py::cpp_function([func](py::object t) -> py::object {
                if (py::isinstance<py::float_>(t) || py::isinstance<py::int_>(t)) {
                    Eigen::VectorXd t_vec(1);
                    t_vec(0) = t.cast<double>();
                    return py::cast(func(t_vec)(0));
                } else {
                    return py::cast(func(t.cast<Eigen::VectorXd>()));
                }
            });
        }, py::arg("A"), py::arg("b"));

    py::class_<lassoinf::LinearOperator, PyLinearOperator, std::shared_ptr<lassoinf::LinearOperator>>(m, "LinearOperator")
        .def(py::init<>())
        .def("rows", &lassoinf::LinearOperator::rows)
        .def("cols", &lassoinf::LinearOperator::cols)
        .def("multiply", &lassoinf::LinearOperator::multiply, py::arg("x"))
        .def("multiply_transpose", &lassoinf::LinearOperator::multiply_transpose, py::arg("x"));

    py::class_<lassoinf::DenseOperator, lassoinf::LinearOperator, std::shared_ptr<lassoinf::DenseOperator>>(m, "DenseOperator")
        .def(py::init<Eigen::MatrixXd>(), py::arg("mat"));

    py::class_<lassoinf::CompositeComponent>(m, "CompositeComponent")
        .def(py::init<Eigen::SparseMatrix<double>, Eigen::MatrixXd, Eigen::MatrixXd, Eigen::VectorXd>(),
             py::arg("S"), py::arg("U"), py::arg("V"), py::arg("b"));

    py::class_<lassoinf::CompositeOperator, lassoinf::LinearOperator, std::shared_ptr<lassoinf::CompositeOperator>>(m, "CompositeOperator")
        .def(py::init<Eigen::Index, Eigen::Index, std::vector<lassoinf::CompositeComponent>>(), py::arg("rows"), py::arg("cols"), py::arg("components"));

    py::class_<lassoinf::XTVXOperator, lassoinf::LinearOperator, std::shared_ptr<lassoinf::XTVXOperator>>(m, "XTVXOperator")
        .def(py::init<Eigen::MatrixXd, std::shared_ptr<lassoinf::LinearOperator>>(), py::arg("X"), py::arg("V"));

    py::class_<lassoinf::InactiveScoreOperator, lassoinf::LinearOperator, std::shared_ptr<lassoinf::InactiveScoreOperator>>(m, "InactiveScoreOperator")
        .def(py::init<std::shared_ptr<lassoinf::LinearOperator>, std::vector<int>, std::vector<int>, Eigen::MatrixXd>(),
             py::arg("Q"), py::arg("E"), py::arg("E_c"), py::arg("W"))
        .def("coef", &lassoinf::InactiveScoreOperator::coef, py::arg("x"));

    py::class_<lassoinf::LassoConstraintOperator, lassoinf::LinearOperator, std::shared_ptr<lassoinf::LassoConstraintOperator>>(m, "LassoConstraintOperator")
        .def_property_readonly("score", &lassoinf::LassoConstraintOperator::score)
        .def_property_readonly("R_active", &lassoinf::LassoConstraintOperator::R_active)
        .def_property_readonly("R_inactive", &lassoinf::LassoConstraintOperator::R_inactive);

    py::class_<lassoinf::SparseProductOperator, lassoinf::LinearOperator, std::shared_ptr<lassoinf::SparseProductOperator>>(m, "SparseProductOperator")
        .def(py::init<Eigen::SparseMatrix<double>, std::shared_ptr<lassoinf::LinearOperator>>(), py::arg("P"), py::arg("R"));

    py::class_<lassoinf::VStackOperator, lassoinf::LinearOperator, std::shared_ptr<lassoinf::VStackOperator>>(m, "VStackOperator")
        .def(py::init<std::vector<std::shared_ptr<lassoinf::LinearOperator>>>(), py::arg("ops"));

    py::class_<lassoinf::LassoConstraints>(m, "LassoConstraints")
        .def_readonly("A", &lassoinf::LassoConstraints::A)
        .def_readonly("b", &lassoinf::LassoConstraints::b)
        .def_readonly("E", &lassoinf::LassoConstraints::E)
        .def_readonly("E_c", &lassoinf::LassoConstraints::E_c)
        .def_readonly("s_E", &lassoinf::LassoConstraints::s_E)
        .def_readonly("v_Ec", &lassoinf::LassoConstraints::v_Ec)
        .def_readonly("W", &lassoinf::LassoConstraints::W);

    py::class_<lassoinf::ContrastEstimand>(m, "ContrastEstimand")
        .def(py::init<>())
        .def_readwrite("eta", &lassoinf::ContrastEstimand::eta)
        .def_readwrite("offset", &lassoinf::ContrastEstimand::offset)
        .def("value", &lassoinf::ContrastEstimand::value, py::arg("Z"));

    py::class_<lassoinf::SelectionCoordinates>(m, "SelectionCoordinates")
        .def(py::init<std::shared_ptr<lassoinf::InactiveScoreOperator>, const Eigen::VectorXd&, const Eigen::VectorXd&,
                      const Eigen::VectorXd&, Eigen::VectorXd>(),
             py::arg("score"), py::arg("Z_noisy"), py::arg("beta_hat"), py::arg("G_hat"), py::arg("Q_diag") = Eigen::VectorXd())
        .def_property_readonly("E", &lassoinf::SelectionCoordinates::E)
        .def_property_readonly("E_c", &lassoinf::SelectionCoordinates::E_c)
        .def_property_readonly("score_operator", &lassoinf::SelectionCoordinates::score_operator)
        .def_property_readonly("lasso_coef_offset", &lassoinf::SelectionCoordinates::lasso_coef_offset)
        .def_property_readonly("lasso_score_offset", &lassoinf::SelectionCoordinates::lasso_score_offset)
        .def("refit_coef", &lassoinf::SelectionCoordinates::refit_coef, py::arg("Z"))
        .def("refit_score", &lassoinf::SelectionCoordinates::refit_score, py::arg("Z"))
        .def("contrast", &lassoinf::SelectionCoordinates::contrast, py::arg("a_E"), py::arg("a_Ec"))
        .def("estimand", &lassoinf::SelectionCoordinates::estimand, py::arg("a_E"), py::arg("a_Ec"), py::arg("basis") = "refit")
        .def("inactive_score", &lassoinf::SelectionCoordinates::inactive_score, py::arg("j"), py::arg("basis") = "refit")
        .def("inactive_coef", &lassoinf::SelectionCoordinates::inactive_coef, py::arg("j"), py::arg("basis") = "refit")
        .def("inactive_S", &lassoinf::SelectionCoordinates::inactive_S, py::arg("variables"));

    py::class_<lassoinf::ScreenedSelection>(m, "ScreenedSelection")
        .def(py::init<const std::shared_ptr<lassoinf::LinearOperator>&, const Eigen::VectorXd&,
                      const lassoinf::SelectionCoordinates&, const Eigen::VectorXd&, const Eigen::VectorXd&,
                      double, int, const std::string&>(),
             py::arg("A_lasso"), py::arg("b_lasso"), py::arg("coords"), py::arg("G_hat"), py::arg("Z_noisy"),
             py::arg("threshold") = std::numeric_limits<double>::quiet_NaN(), py::arg("top_k") = -1,
             py::arg("conditioning") = "first_dropped")
        .def_readonly("screened", &lassoinf::ScreenedSelection::screened)
        .def_readonly("screened_signs", &lassoinf::ScreenedSelection::screened_signs)
        .def_readonly("first_dropped", &lassoinf::ScreenedSelection::first_dropped)
        .def_readonly("first_dropped_sign", &lassoinf::ScreenedSelection::first_dropped_sign)
        .def_readonly("observed_feasible", &lassoinf::ScreenedSelection::observed_feasible)
        .def_readonly("A_screen", &lassoinf::ScreenedSelection::A_screen)
        .def_readonly("b_screen", &lassoinf::ScreenedSelection::b_screen)
        .def_readonly("A", &lassoinf::ScreenedSelection::A)
        .def_readonly("b", &lassoinf::ScreenedSelection::b);

    m.def("lasso_post_selection_constraints", &lassoinf::lasso_post_selection_constraints,
          py::arg("beta_hat"), py::arg("G"), py::arg("Q"), py::arg("D_diag"),
          py::arg("L") = Eigen::VectorXd(), py::arg("U") = Eigen::VectorXd(), py::arg("tol") = 1e-6);

    py::class_<lassoinf::AffineConstraints>(m, "AffineConstraints")
        .def(py::init<Eigen::VectorXd, Eigen::VectorXd, std::shared_ptr<lassoinf::LinearOperator>, std::shared_ptr<lassoinf::LinearOperator>, double>(),
             py::arg("Z"), py::arg("Z_noisy"), py::arg("Q"), py::arg("Q_noise"), py::arg("scalar_noise") = std::numeric_limits<double>::quiet_NaN())
        .def(py::init<Eigen::VectorXd, Eigen::VectorXd, Eigen::MatrixXd, Eigen::MatrixXd, double>(),
             py::arg("Z"), py::arg("Z_noisy"), py::arg("Q"), py::arg("Q_noise"), py::arg("scalar_noise") = std::numeric_limits<double>::quiet_NaN())
        .def("solve_contrast", &lassoinf::AffineConstraints::solve_contrast, py::arg("v"))
        .def("compute_contrast", &lassoinf::AffineConstraints::compute_contrast, py::arg("v"))
        .def("compute_covariance_contrast", &lassoinf::AffineConstraints::compute_covariance_contrast,
             py::arg("theta_hat"), py::arg("variance"), py::arg("score_cov"));
}
