#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <memory>
#include "ray_selection.hpp"

namespace py = pybind11;

namespace {

// penalty(Y) -> None or (D, L, U), with L or U None for no bounds
std::function<lassoinf::RayPenalty(const Eigen::VectorXd&)> wrap_penalty(py::function penalty) {
    return [penalty](const Eigen::VectorXd& Y) {
        py::object out = penalty(Y);
        lassoinf::RayPenalty pen{false, Eigen::VectorXd(), Eigen::VectorXd(), Eigen::VectorXd()};
        if (out.is_none()) return pen;
        py::tuple t = out.cast<py::tuple>();
        if (t.size() != 3) throw std::invalid_argument("penalty must return None or (D, L, U)");
        pen.valid = true;
        pen.D = t[0].cast<Eigen::VectorXd>();
        if (!t[1].is_none()) pen.L = t[1].cast<Eigen::VectorXd>();
        if (!t[2].is_none()) pen.U = t[2].cast<Eigen::VectorXd>();
        return pen;
    };
}

} // namespace

void init_ray_selection(py::module_ &m) {
    py::class_<lassoinf::RayEvent, std::shared_ptr<lassoinf::RayEvent>>(m, "RayEvent")
        .def("__call__", [](const lassoinf::RayEvent& e, const Eigen::VectorXd& Y) { return e(Y); },
             py::arg("Y"));

    py::class_<lassoinf::LassoRayEvent, lassoinf::RayEvent, std::shared_ptr<lassoinf::LassoRayEvent>>(m, "LassoRayEvent")
        .def(py::init([](Eigen::VectorXd beta_hat, Eigen::VectorXd G_hat, std::shared_ptr<lassoinf::LinearOperator> Q,
                         py::function penalty, double tol, double slack) {
                 return std::make_shared<lassoinf::LassoRayEvent>(std::move(beta_hat), std::move(G_hat), std::move(Q),
                                                                  wrap_penalty(penalty), tol, slack);
             }),
             py::arg("beta_hat"), py::arg("G_hat"), py::arg("Q"), py::arg("penalty"),
             py::arg("tol") = 1e-6, py::arg("slack") = 1e-9)
        .def(py::init([](Eigen::VectorXd beta_hat, Eigen::VectorXd G_hat, Eigen::MatrixXd Q,
                         py::function penalty, double tol, double slack) {
                 return std::make_shared<lassoinf::LassoRayEvent>(std::move(beta_hat), std::move(G_hat),
                                                                  std::make_shared<lassoinf::DenseOperator>(std::move(Q)),
                                                                  wrap_penalty(penalty), tol, slack);
             }),
             py::arg("beta_hat"), py::arg("G_hat"), py::arg("Q"), py::arg("penalty"),
             py::arg("tol") = 1e-6, py::arg("slack") = 1e-9);

    m.def("ray_scale", &lassoinf::ray_scale, py::arg("bar_s"), py::arg("variance"), py::arg("theta_hat"),
          py::arg("bar_theta"));

    m.def("find_truncation_set",
          [](const lassoinf::RayEvent& event, const Eigen::VectorXd& y0, const Eigen::VectorXd& direction,
             double w_obs, double sd_w, double n_sd, int n_grid, double tol) {
              return lassoinf::find_truncation_set(event, y0, direction, w_obs, sd_w, n_sd, n_grid, tol);
          },
          py::arg("event"), py::arg("y0"), py::arg("direction"), py::arg("w_obs"), py::arg("sd_w"),
          py::arg("n_sd") = 10.0, py::arg("n_grid") = 2001, py::arg("tol") = 1e-10);
    // any Python callable event(Y) -> bool
    m.def("find_truncation_set",
          [](py::function event, const Eigen::VectorXd& y0, const Eigen::VectorXd& direction,
             double w_obs, double sd_w, double n_sd, int n_grid, double tol) {
              lassoinf::FunctionRayEvent e([event](const Eigen::VectorXd& Y) {
                  return static_cast<bool>(py::bool_(event(Y)));
              });
              return lassoinf::find_truncation_set(e, y0, direction, w_obs, sd_w, n_sd, n_grid, tol);
          },
          py::arg("event"), py::arg("y0"), py::arg("direction"), py::arg("w_obs"), py::arg("sd_w"),
          py::arg("n_sd") = 10.0, py::arg("n_grid") = 2001, py::arg("tol") = 1e-10);
}
