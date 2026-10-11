#include <RcppEigen.h>

// [[Rcpp::depends(RcppEigen)]]

// 1. Include the headers
#include "lassoinf/include/affine_constraints.hpp"
#include "lassoinf/include/discrete_family.h"
#include "lassoinf/include/gaussian_family.hpp"
#include "lassoinf/include/custom_estimand.hpp"
#include "lassoinf/include/unilasso.hpp"
#include "lassoinf/include/ray_selection.hpp"

// 2. Unity build: include the C++ sources directly to avoid duplicate symbols
//    and bypass the need for a complex Makefile to compile them individually.
#include "lassoinf/src/affine_constraints.cpp"
#include "lassoinf/src/lasso_post_selection_constraints.cpp"
#include "lassoinf/src/discrete_family.cpp"
#include "lassoinf/src/gaussian_family.cpp"
#include "lassoinf/src/custom_estimand.cpp"
#include "lassoinf/src/unilasso.cpp"
#include "lassoinf/src/ray_selection.cpp"

using namespace Rcpp;

// ---- linear operators ----

// R-facing holder for any C++ LinearOperator; diag is the operator's
// diagonal when known (empty otherwise)
struct LinearOp {
    std::shared_ptr<lassoinf::LinearOperator> op;
    Eigen::VectorXd diag;

    explicit LinearOp(std::shared_ptr<lassoinf::LinearOperator> op_, Eigen::VectorXd diag_ = Eigen::VectorXd())
        : op(std::move(op_)), diag(std::move(diag_)) {}

    Eigen::VectorXd multiply(const Eigen::VectorXd& x) { return op->multiply(x); }
    Eigen::VectorXd multiply_transpose(const Eigen::VectorXd& x) { return op->multiply_transpose(x); }
    int rows() { return static_cast<int>(op->rows()); }
    int cols() { return static_cast<int>(op->cols()); }
    Eigen::VectorXd diagonal() { return diag; }

    // one matvec per column: for testing / small problems only
    Eigen::MatrixXd to_dense() {
        Eigen::MatrixXd dense(op->rows(), op->cols());
        for (Eigen::Index i = 0; i < op->cols(); ++i) {
            Eigen::VectorXd e = Eigen::VectorXd::Zero(op->cols());
            e(i) = 1.0;
            dense.col(i) = op->multiply(e);
        }
        return dense;
    }
};

// dense matrix
LinearOp* dense_linear_op(Eigen::MatrixXd M) {
    Eigen::VectorXd d = M.rows() == M.cols() ? Eigen::VectorXd(M.diagonal()) : Eigen::VectorXd();
    return new LinearOp(std::make_shared<lassoinf::DenseOperator>(std::move(M)), d);
}

// X' diag(w) X, matrix-free
LinearOp* xtvx_linear_op(Eigen::MatrixXd X, Eigen::VectorXd w) {
    if (w.size() != X.rows()) Rcpp::stop("weights must have length nrow(X)");
    lassoinf::CompositeComponent comp;
    comp.S = Eigen::SparseMatrix<double>(X.rows(), X.rows());
    comp.U = Eigen::MatrixXd(X.rows(), 0);
    comp.V = Eigen::MatrixXd(X.rows(), 0);
    comp.b = w;
    auto V = std::make_shared<lassoinf::CompositeOperator>(X.rows(), X.rows(), std::vector<lassoinf::CompositeComponent>{comp});
    Eigen::VectorXd d = (X.array().square().colwise() * w.array()).colwise().sum().transpose();
    return new LinearOp(std::make_shared<lassoinf::XTVXOperator>(std::move(X), V), d);
}

// X' diag(w) X + diag(d), matrix-free
class XTVXDiagOperator : public lassoinf::LinearOperator {
public:
    XTVXDiagOperator(Eigen::MatrixXd X, Eigen::VectorXd w, Eigen::VectorXd d)
        : X_(std::move(X)), w_(std::move(w)), d_(std::move(d)) {}
    Eigen::Index rows() const override { return X_.cols(); }
    Eigen::Index cols() const override { return X_.cols(); }
    Eigen::VectorXd multiply(const Eigen::VectorXd& x) const override {
        return X_.transpose() * (w_.cwiseProduct(X_ * x)) + d_.cwiseProduct(x);
    }
    Eigen::VectorXd multiply_transpose(const Eigen::VectorXd& x) const override { return multiply(x); }
private:
    Eigen::MatrixXd X_;
    Eigen::VectorXd w_;
    Eigen::VectorXd d_;
};

LinearOp* xtvx_diag_linear_op(Eigen::MatrixXd X, Eigen::VectorXd w, Eigen::VectorXd d) {
    if (w.size() != X.rows()) Rcpp::stop("weights must have length nrow(X)");
    if (d.size() != X.cols()) Rcpp::stop("diagonal must have length ncol(X)");
    Eigen::VectorXd diag = (X.array().square().colwise() * w.array()).colwise().sum().transpose() + d.array();
    return new LinearOp(std::make_shared<XTVXDiagOperator>(std::move(X), std::move(w), std::move(d)), diag);
}

template <class T>
T* unwrap_cpp_object(SEXP obj, const char* cls) {
    if (!Rf_inherits(obj, cls)) Rcpp::stop(std::string("expected an object of class ") + cls);
    Rcpp::Environment env(obj);
    SEXP xp = env.get(".pointer");
    T* ptr = static_cast<T*>(R_ExternalPtrAddr(xp));
    if (!ptr) Rcpp::stop("invalid (null) C++ object");
    return ptr;
}

// numeric matrix or LinearOp
std::shared_ptr<lassoinf::LinearOperator> as_operator(SEXP x) {
    if (Rf_isMatrix(x)) {
        return std::make_shared<lassoinf::DenseOperator>(Rcpp::as<Eigen::MatrixXd>(x));
    }
    return unwrap_cpp_object<LinearOp>(x, "Rcpp_LinearOp")->op;
}

SEXP wrap_operator(std::shared_ptr<lassoinf::LinearOperator> op) {
    return Rcpp::internal::make_new_object(new LinearOp(std::move(op)));
}

// c * op, sharing op (no copy of its data)
class ScaledOperator : public lassoinf::LinearOperator {
public:
    ScaledOperator(std::shared_ptr<lassoinf::LinearOperator> op, double c) : op_(std::move(op)), c_(c) {}
    Eigen::Index rows() const override { return op_->rows(); }
    Eigen::Index cols() const override { return op_->cols(); }
    Eigen::VectorXd multiply(const Eigen::VectorXd& x) const override { return c_ * op_->multiply(x); }
    Eigen::VectorXd multiply_transpose(const Eigen::VectorXd& x) const override { return c_ * op_->multiply_transpose(x); }
private:
    std::shared_ptr<lassoinf::LinearOperator> op_;
    double c_;
};

SEXP linear_op_scaled(LinearOp* L, double c) {
    return Rcpp::internal::make_new_object(
        new LinearOp(std::make_shared<ScaledOperator>(L->op, c), c * L->diag));
}

// ---- constraints ----

Rcpp::List lasso_post_selection_constraints_wrapper(
    const Eigen::VectorXd& beta_hat,
    const Eigen::VectorXd& G,
    SEXP Q,
    const Eigen::VectorXd& D_diag,
    const Eigen::VectorXd& L,
    const Eigen::VectorXd& U,
    double tol) {

    auto constraints = lassoinf::lasso_post_selection_constraints(beta_hat, G, as_operator(Q), D_diag, L, U, tol);

    return Rcpp::List::create(
        Rcpp::Named("A") = wrap_operator(constraints.A),
        Rcpp::Named("score") = wrap_operator(constraints.A->score()),
        Rcpp::Named("b") = constraints.b,
        Rcpp::Named("E") = constraints.E,
        Rcpp::Named("E_c") = constraints.E_c,
        Rcpp::Named("s_E") = constraints.s_E,
        Rcpp::Named("v_Ec") = constraints.v_Ec,
        Rcpp::Named("W") = constraints.W
    );
}

// ---- affine constraints ----

NumericVector get_interval_wrapper(lassoinf::AffineConstraintsContrast* contrast, double t, SEXP A, const Eigen::VectorXd& b) {
    auto res = contrast->get_interval(t, *as_operator(A), b);
    return NumericVector::create(res.first, res.second);
}

lassoinf::AffineConstraints* create_affine_constraints(
    Eigen::VectorXd Z, Eigen::VectorXd Z_noisy, SEXP Q, SEXP Q_noise_sexp, double scalar_noise) {
    std::shared_ptr<lassoinf::LinearOperator> Q_noise_op = nullptr;
    if (!Rf_isNull(Q_noise_sexp)) Q_noise_op = as_operator(Q_noise_sexp);
    return new lassoinf::AffineConstraints(Z, Z_noisy, as_operator(Q), Q_noise_op, scalar_noise);
}

lassoinf::AffineConstraintsContrast* compute_contrast_wrapper(lassoinf::AffineConstraints* si, const Eigen::VectorXd& v) {
    return new lassoinf::AffineConstraintsContrast(si->compute_contrast(v));
}

lassoinf::AffineConstraintsContrast* compute_covariance_contrast_wrapper(lassoinf::AffineConstraints* si, double theta_hat,
                                                                        double variance, const Eigen::VectorXd& score_cov) {
    try {
        return new lassoinf::AffineConstraintsContrast(si->compute_covariance_contrast(theta_hat, variance, score_cov));
    } catch (const std::invalid_argument& e) {
        Rcpp::stop(e.what());
    }
}

// ---- custom estimands ----

Rcpp::List estimand_list(const lassoinf::ContrastEstimand& est) {
    Rcpp::List out = Rcpp::List::create(Rcpp::Named("eta") = est.eta, Rcpp::Named("offset") = est.offset);
    out.attr("class") = "ContrastEstimand";
    return out;
}

SEXP selection_coordinates_cpp(SEXP score, const Eigen::VectorXd& Z_noisy, const Eigen::VectorXd& beta_hat,
                               const Eigen::VectorXd& G_hat, const Eigen::VectorXd& Q_diag) {
    auto score_op = std::dynamic_pointer_cast<lassoinf::InactiveScoreOperator>(as_operator(score));
    if (!score_op) Rcpp::stop("score must be the inactive score operator from lasso_post_selection_constraints");
    return Rcpp::internal::make_new_object(new lassoinf::SelectionCoordinates(score_op, Z_noisy, beta_hat, G_hat, Q_diag));
}

template <class F>
auto rethrow(F f) -> decltype(f()) {
    try {
        return f();
    } catch (const std::invalid_argument& e) {
        Rcpp::stop(e.what());
    }
}

Eigen::VectorXd coords_contrast(lassoinf::SelectionCoordinates* c, const Eigen::VectorXd& a_E, const Eigen::VectorXd& a_Ec) {
    return rethrow([&] { return c->contrast(a_E, a_Ec); });
}
Rcpp::List coords_estimand(lassoinf::SelectionCoordinates* c, const Eigen::VectorXd& a_E, const Eigen::VectorXd& a_Ec,
                           std::string basis) {
    return estimand_list(rethrow([&] { return c->estimand(a_E, a_Ec, basis); }));
}
Rcpp::List coords_inactive_score(lassoinf::SelectionCoordinates* c, int j, std::string basis) {
    return estimand_list(rethrow([&] { return c->inactive_score(j, basis); }));
}
Rcpp::List coords_inactive_coef(lassoinf::SelectionCoordinates* c, int j, std::string basis) {
    return estimand_list(rethrow([&] { return c->inactive_coef(j, basis); }));
}
Eigen::VectorXd coords_inactive_S(lassoinf::SelectionCoordinates* c, std::vector<int> variables) {
    return rethrow([&] { return c->inactive_S(variables); });
}
Eigen::VectorXd coords_refit_coef(lassoinf::SelectionCoordinates* c, const Eigen::VectorXd& Z) { return c->refit_coef(Z); }
Eigen::VectorXd coords_refit_score(lassoinf::SelectionCoordinates* c, const Eigen::VectorXd& Z) { return c->refit_score(Z); }
Eigen::VectorXd coords_lasso_coef_offset(lassoinf::SelectionCoordinates* c) { return c->lasso_coef_offset(); }
Eigen::VectorXd coords_lasso_score_offset(lassoinf::SelectionCoordinates* c) { return c->lasso_score_offset(); }
std::vector<int> coords_E(lassoinf::SelectionCoordinates* c) { return c->E(); }
std::vector<int> coords_E_c(lassoinf::SelectionCoordinates* c) { return c->E_c(); }
SEXP coords_score_operator(lassoinf::SelectionCoordinates* c) { return wrap_operator(c->score_operator()); }

SEXP screened_selection_cpp(SEXP A_lasso, const Eigen::VectorXd& b_lasso, SEXP coords,
                            const Eigen::VectorXd& G_hat, const Eigen::VectorXd& Z_noisy,
                            double threshold, int top_k, std::string conditioning) {
    auto* c = unwrap_cpp_object<lassoinf::SelectionCoordinates>(coords, "Rcpp_SelectionCoordinatesCpp");
    return rethrow([&] {
        return Rcpp::internal::make_new_object(
            new lassoinf::ScreenedSelection(as_operator(A_lasso), b_lasso, *c, G_hat, Z_noisy, threshold, top_k, conditioning));
    });
}

std::vector<int> screened_screened(lassoinf::ScreenedSelection* s) { return s->screened; }
Eigen::VectorXd screened_signs(lassoinf::ScreenedSelection* s) { return s->screened_signs; }
int screened_first_dropped(lassoinf::ScreenedSelection* s) { return s->first_dropped; }
double screened_first_dropped_sign(lassoinf::ScreenedSelection* s) { return s->first_dropped_sign; }
bool screened_observed_feasible(lassoinf::ScreenedSelection* s) { return s->observed_feasible; }
Eigen::VectorXd screened_b(lassoinf::ScreenedSelection* s) { return s->b; }
Eigen::VectorXd screened_b_screen(lassoinf::ScreenedSelection* s) { return s->b_screen; }
SEXP screened_A(lassoinf::ScreenedSelection* s) { return wrap_operator(s->A); }
SEXP screened_A_screen(lassoinf::ScreenedSelection* s) { return wrap_operator(s->A_screen); }

// ---- uniLasso ----

Rcpp::List unilasso_penalty_cpp(const Eigen::VectorXd& C, const Eigen::VectorXd& lam) {
    auto pen = rethrow([&] { return lassoinf::unilasso_penalty(C, lam); });
    return Rcpp::List::create(Rcpp::Named("C") = pen.C, Rcpp::Named("s") = pen.s,
                              Rcpp::Named("D") = pen.D, Rcpp::Named("L") = pen.L,
                              Rcpp::Named("U") = pen.U);
}

Eigen::VectorXd unilasso_fit_cpp(const Eigen::MatrixXd& Q, const Eigen::VectorXd& Z_noisy,
                                 const Eigen::VectorXd& lam, double tol, int max_iter) {
    return rethrow([&] { return lassoinf::unilasso_fit(Q, Z_noisy, lam, tol, max_iter); });
}

double unilasso_kkt_violation_cpp(const Eigen::VectorXd& beta_hat, const Eigen::MatrixXd& Q,
                                  const Eigen::VectorXd& Z_noisy, const Eigen::VectorXd& lam) {
    return rethrow([&] { return lassoinf::unilasso_kkt_violation(beta_hat, Q, Z_noisy, lam); });
}

Eigen::VectorXd unilasso_loo_kappa_cpp(const Eigen::MatrixXd& X, const Eigen::VectorXd& y,
                                       const Eigen::VectorXd& beta_hat) {
    return lassoinf::unilasso_loo_kappa(X, y, beta_hat);
}

Eigen::MatrixXd unilasso_row_sets_cpp(const Eigen::VectorXd& u, const Eigen::VectorXd& a,
                                      const Eigen::VectorXd& b0, const Eigen::VectorXd& m,
                                      double c, double gamma, double s) {
    return lassoinf::unilasso_row_sets(u, a, b0, m, c, gamma, s);
}

Eigen::MatrixXd intersect_row_sets_cpp(const Eigen::MatrixXd& sets) {
    return rethrow([&] { return lassoinf::intersect_row_sets(sets); });
}

Eigen::MatrixXd unilasso_truncation_set_cpp(const Eigen::VectorXd& beta_hat, const Eigen::VectorXd& G_hat,
                                            SEXP Q, const Eigen::VectorXd& D, const Eigen::VectorXd& L,
                                            const Eigen::VectorXd& U, int k, double lam_k, double C_k,
                                            const Eigen::VectorXd& y0, const Eigen::VectorXd& direction,
                                            double w_obs, double gamma, double tol) {
    return rethrow([&] {
        return lassoinf::unilasso_truncation_set(beta_hat, G_hat, as_operator(Q), D, L, U, k, lam_k,
                                                 C_k, y0, direction, w_obs, gamma, tol);
    });
}

// ---- ray selection ----

// penalty(Y) returns NULL or list(D, L, U), with L or U NULL for no bounds
lassoinf::LassoRayEvent* create_lasso_ray_event(Eigen::VectorXd beta_hat, Eigen::VectorXd G_hat, SEXP Q,
                                                Rcpp::Function penalty, double tol, double slack) {
    auto pen = [penalty](const Eigen::VectorXd& Y) {
        SEXP out = penalty(Rcpp::wrap(Y));
        lassoinf::RayPenalty r{false, Eigen::VectorXd(), Eigen::VectorXd(), Eigen::VectorXd()};
        if (Rf_isNull(out)) return r;
        Rcpp::List l(out);
        if (l.size() != 3) Rcpp::stop("penalty must return NULL or list(D, L, U)");
        r.valid = true;
        r.D = Rcpp::as<Eigen::VectorXd>(l[0]);
        if (!Rf_isNull(l[1])) r.L = Rcpp::as<Eigen::VectorXd>(l[1]);
        if (!Rf_isNull(l[2])) r.U = Rcpp::as<Eigen::VectorXd>(l[2]);
        return r;
    };
    return new lassoinf::LassoRayEvent(std::move(beta_hat), std::move(G_hat), as_operator(Q), pen, tol, slack);
}

bool lasso_ray_event_call(lassoinf::LassoRayEvent* e, const Eigen::VectorXd& Y) { return (*e)(Y); }

Rcpp::NumericVector ray_scale_cpp(double bar_s, double variance, double theta_hat, double bar_theta) {
    auto r = lassoinf::ray_scale(bar_s, variance, theta_hat, bar_theta);
    return Rcpp::NumericVector::create(r.first, r.second);
}

// event: an R function of Y returning TRUE / FALSE, or a LassoRayEventCpp
Eigen::MatrixXd find_truncation_set_cpp(SEXP event, const Eigen::VectorXd& y0, const Eigen::VectorXd& direction,
                                        double w_obs, double sd_w, double n_sd, int n_grid, double tol) {
    return rethrow([&] {
        if (Rf_isFunction(event)) {
            Rcpp::Function f(event);
            lassoinf::FunctionRayEvent e([f](const Eigen::VectorXd& Y) {
                return Rcpp::as<bool>(f(Rcpp::wrap(Y)));
            });
            return lassoinf::find_truncation_set(e, y0, direction, w_obs, sd_w, n_sd, n_grid, tol);
        }
        auto* e = unwrap_cpp_object<lassoinf::LassoRayEvent>(event, "Rcpp_LassoRayEventCpp");
        return lassoinf::find_truncation_set(*e, y0, direction, w_obs, sd_w, n_sd, n_grid, tol);
    });
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

lassoinf::AffineConstraints* create_affine_constraints_default(
    Eigen::VectorXd Z, Eigen::VectorXd Z_noisy, SEXP Q, SEXP Q_noise_sexp) {
    return create_affine_constraints(Z, Z_noisy, Q, Q_noise_sexp, std::numeric_limits<double>::quiet_NaN());
}

RCPP_MODULE(lassoinf_cpp) {
    function("lasso_post_selection_constraints", &lasso_post_selection_constraints_wrapper, 
             List::create(_["beta_hat"], _["G"], _["Q"], _["D_diag"], _["L"], _["U"], _["tol"] = 1e-6));

    class_<lassoinf::AffineConstraintsContrast>("AffineConstraintsContrast")
        .field("gamma", &lassoinf::AffineConstraintsContrast::gamma)
        .field("c", &lassoinf::AffineConstraintsContrast::c)
        .field("bar_gamma", &lassoinf::AffineConstraintsContrast::bar_gamma)
        .field("bar_s", &lassoinf::AffineConstraintsContrast::bar_s)
        .field("n_o", &lassoinf::AffineConstraintsContrast::n_o)
        .field("bar_n_o", &lassoinf::AffineConstraintsContrast::bar_n_o)
        .field("theta_hat", &lassoinf::AffineConstraintsContrast::theta_hat)
        .field("bar_theta", &lassoinf::AffineConstraintsContrast::bar_theta)
        .field("splitting_variance", &lassoinf::AffineConstraintsContrast::splitting_variance)
        .field("splitting_estimator", &lassoinf::AffineConstraintsContrast::splitting_estimator)
        .field("naive_variance", &lassoinf::AffineConstraintsContrast::naive_variance)
        .method("get_interval", &get_interval_wrapper)
        ;

    class_<LinearOp>("LinearOp")
        .factory<Eigen::MatrixXd>(&dense_linear_op)
        .factory<Eigen::MatrixXd, Eigen::VectorXd>(&xtvx_linear_op)
        .factory<Eigen::MatrixXd, Eigen::VectorXd, Eigen::VectorXd>(&xtvx_diag_linear_op)
        .method("multiply", &LinearOp::multiply)
        .method("multiply_transpose", &LinearOp::multiply_transpose)
        .method("rows", &LinearOp::rows)
        .method("cols", &LinearOp::cols)
        .method("diagonal", &LinearOp::diagonal)
        .method("to_dense", &LinearOp::to_dense)
        .method("scaled", &linear_op_scaled)
        ;

    class_<lassoinf::SelectionCoordinates>("SelectionCoordinatesCpp")
        .method("contrast", &coords_contrast)
        .method("estimand", &coords_estimand)
        .method("inactive_score", &coords_inactive_score)
        .method("inactive_coef", &coords_inactive_coef)
        .method("inactive_S", &coords_inactive_S)
        .method("refit_coef", &coords_refit_coef)
        .method("refit_score", &coords_refit_score)
        .method("lasso_coef_offset", &coords_lasso_coef_offset)
        .method("lasso_score_offset", &coords_lasso_score_offset)
        .method("E", &coords_E)
        .method("E_c", &coords_E_c)
        .method("score_operator", &coords_score_operator)
        ;

    class_<lassoinf::ScreenedSelection>("ScreenedSelectionCpp")
        .method("screened", &screened_screened)
        .method("screened_signs", &screened_signs)
        .method("first_dropped", &screened_first_dropped)
        .method("first_dropped_sign", &screened_first_dropped_sign)
        .method("observed_feasible", &screened_observed_feasible)
        .method("b", &screened_b)
        .method("b_screen", &screened_b_screen)
        .method("A", &screened_A)
        .method("A_screen", &screened_A_screen)
        ;

    function("unilasso_penalty_cpp", &unilasso_penalty_cpp, List::create(_["C"], _["lam"]));
    function("unilasso_fit_cpp", &unilasso_fit_cpp,
             List::create(_["Q"], _["Z_noisy"], _["lam"], _["tol"], _["max_iter"]));
    function("unilasso_kkt_violation_cpp", &unilasso_kkt_violation_cpp,
             List::create(_["beta_hat"], _["Q"], _["Z_noisy"], _["lam"]));
    function("unilasso_loo_kappa_cpp", &unilasso_loo_kappa_cpp, List::create(_["X"], _["y"], _["beta_hat"]));
    function("unilasso_row_sets_cpp", &unilasso_row_sets_cpp,
             List::create(_["u"], _["a"], _["b0"], _["m"], _["c"], _["gamma"], _["s"]));
    function("intersect_row_sets_cpp", &intersect_row_sets_cpp, List::create(_["sets"]));
    function("unilasso_truncation_set_cpp", &unilasso_truncation_set_cpp,
             List::create(_["beta_hat"], _["G_hat"], _["Q"], _["D"], _["L"], _["U"], _["k"], _["lam_k"],
                          _["C_k"], _["y0"], _["direction"], _["w_obs"], _["gamma"], _["tol"]));

    class_<lassoinf::LassoRayEvent>("LassoRayEventCpp")
        .factory<Eigen::VectorXd, Eigen::VectorXd, SEXP, Rcpp::Function, double, double>(&create_lasso_ray_event)
        .method("call", &lasso_ray_event_call)
        ;
    function("ray_scale_cpp", &ray_scale_cpp, List::create(_["bar_s"], _["variance"], _["theta_hat"], _["bar_theta"]));
    function("find_truncation_set_cpp", &find_truncation_set_cpp,
             List::create(_["event"], _["y0"], _["direction"], _["w_obs"], _["sd_w"], _["n_sd"], _["n_grid"], _["tol"]));

    function("selection_coordinates_cpp", &selection_coordinates_cpp,
             List::create(_["score"], _["Z_noisy"], _["beta_hat"], _["G_hat"], _["Q_diag"]));
    function("screened_selection_cpp", &screened_selection_cpp,
             List::create(_["A_lasso"], _["b_lasso"], _["coords"], _["G_hat"], _["Z_noisy"],
                          _["threshold"], _["top_k"], _["conditioning"]));

    class_<lassoinf::AffineConstraints>("AffineConstraints")
        .factory<Eigen::VectorXd, Eigen::VectorXd, SEXP, SEXP, double>(&create_affine_constraints)
        .factory<Eigen::VectorXd, Eigen::VectorXd, SEXP, SEXP>(&create_affine_constraints_default)
        .method("solve_contrast", &lassoinf::AffineConstraints::solve_contrast)
        .method("compute_contrast", &compute_contrast_wrapper)
        .method("compute_covariance_contrast", &compute_covariance_contrast_wrapper)
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