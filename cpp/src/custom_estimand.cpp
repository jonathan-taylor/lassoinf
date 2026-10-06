#include "custom_estimand.hpp"

#include <algorithm>
#include <numeric>

namespace lassoinf {

namespace {

using detail::take;
using detail::sign;

void check_basis(const std::string& basis) {
    if (basis != "refit" && basis != "lasso") {
        throw std::invalid_argument("basis must be 'refit' or 'lasso'");
    }
}

} // namespace

// ---- SelectionCoordinates ----

SelectionCoordinates::SelectionCoordinates(std::shared_ptr<InactiveScoreOperator> score,
                                           const Eigen::VectorXd& Z_noisy,
                                           const Eigen::VectorXd& beta_hat,
                                           const Eigen::VectorXd& G_hat,
                                           Eigen::VectorXd Q_diag)
    : score_(std::move(score)), Q_diag_(std::move(Q_diag)) {
    const auto& E_c = score_->E_c();
    for (size_t k = 0; k < E_c.size(); ++k) pos_Ec_[E_c[k]] = static_cast<int>(k);

    if (Q_diag_.size() == 0) {
        if (auto dense = dynamic_cast<const DenseOperator*>(score_->Q().get())) {
            Q_diag_ = dense->mat().diagonal();
        }
    }

    // the LASSO solution and gradient are affine in the data on the selection
    // event: constants are observed values minus the linear maps at Z_noisy
    lasso_coef_offset_ = take(beta_hat, score_->E()) - refit_coef(Z_noisy);
    lasso_score_offset_ = -take(G_hat, E_c) - refit_score(Z_noisy);
}

int SelectionCoordinates::inactive_position(int j) const {
    auto it = pos_Ec_.find(j);
    if (it == pos_Ec_.end()) throw std::invalid_argument("variable " + std::to_string(j) + " is not inactive");
    return it->second;
}

Eigen::VectorXd SelectionCoordinates::contrast(const Eigen::VectorXd& a_E, const Eigen::VectorXd& a_Ec) const {
    const auto& E = score_->E();
    Eigen::Index n = score_->cols();
    if ((a_E.size() != 0 && a_E.size() != static_cast<Eigen::Index>(E.size())) ||
        (a_Ec.size() != 0 && a_Ec.size() != score_->rows())) {
        throw std::invalid_argument("a_E / a_Ec have the wrong length");
    }
    // eta = U_{-E}' a_Ec + [W' a_E on E]
    Eigen::VectorXd eta = (a_Ec.size() > 0 && a_Ec.cwiseAbs().maxCoeff() > 0)
        ? score_->multiply_transpose(a_Ec) : Eigen::VectorXd::Zero(n);
    if (a_E.size() > 0) {
        Eigen::VectorXd eta_E = score_->W().transpose() * a_E;
        for (size_t i = 0; i < E.size(); ++i) eta(E[i]) += eta_E(i);
    }
    return eta;
}

ContrastEstimand SelectionCoordinates::estimand(const Eigen::VectorXd& a_E, const Eigen::VectorXd& a_Ec,
                                                const std::string& basis) const {
    check_basis(basis);
    ContrastEstimand est{contrast(a_E, a_Ec), 0.0};
    if (basis == "lasso") {
        if (a_E.size() > 0) est.offset += a_E.dot(lasso_coef_offset_);
        if (a_Ec.size() > 0) est.offset += a_Ec.dot(lasso_score_offset_);
    }
    return est;
}

ContrastEstimand SelectionCoordinates::inactive_score(int j, const std::string& basis) const {
    Eigen::VectorXd a_Ec = Eigen::VectorXd::Zero(score_->rows());
    a_Ec(inactive_position(j)) = 1.0;
    return estimand(Eigen::VectorXd(), a_Ec, basis);
}

ContrastEstimand SelectionCoordinates::inactive_coef(int j, const std::string& basis) const {
    double S_jj = inactive_S({j})(0);
    ContrastEstimand score = inactive_score(j, basis);
    return ContrastEstimand{score.eta / S_jj, score.offset / S_jj};
}

Eigen::VectorXd SelectionCoordinates::inactive_S(const std::vector<int>& variables) const {
    const auto& E = score_->E();
    const auto& W = score_->W();
    const auto& Q = score_->Q();
    Eigen::Index n = score_->cols();

    std::vector<int> missing;
    for (int j : variables) {
        inactive_position(j);
        if (!S_cache_.count(j) && std::find(missing.begin(), missing.end(), j) == missing.end()) {
            missing.push_back(j);
        }
    }

    if (!missing.empty() && Q_diag_.size() > 0 && missing.size() > E.size()) {
        // Q_{j,E} W Q_{E,j} = ||C' Q_{E,j}||^2 with W = C C'
        Eigen::VectorXd S = take(Q_diag_, missing);
        if (!E.empty()) {
            Eigen::MatrixXd C = W.llt().matrixL();
            for (size_t i = 0; i < E.size(); ++i) {
                Eigen::VectorXd QC = Q->multiply(detail::embed(n, E, C.col(i)));
                S -= take(QC, missing).cwiseAbs2();
            }
        }
        for (size_t i = 0; i < missing.size(); ++i) S_cache_[missing[i]] = S(i);
    } else {
        for (int j : missing) {
            Eigen::VectorXd e_j = Eigen::VectorXd::Zero(n);
            e_j(j) = 1.0;
            Eigen::VectorXd Q_j = Q->multiply(e_j);
            Eigen::VectorXd Q_Ej = take(Q_j, E);
            S_cache_[j] = Q_j(j) - Q_Ej.dot(W * Q_Ej);
        }
    }

    Eigen::VectorXd out(variables.size());
    for (size_t i = 0; i < variables.size(); ++i) out(i) = S_cache_.at(variables[i]);
    return out;
}

// ---- ScreenedSelection ----

namespace {

// constraints P G <= q, built row by row
struct RowBuilder {
    std::vector<Eigen::Triplet<double>> triplets;
    std::vector<double> q;

    void add(std::initializer_list<std::pair<int, double>> entries, double rhs) {
        int r = static_cast<int>(q.size());
        for (const auto& e : entries) triplets.emplace_back(r, e.first, e.second);
        q.push_back(rhs);
    }
};

} // namespace

ScreenedSelection::ScreenedSelection(const std::shared_ptr<LinearOperator>& A_lasso,
                                     const Eigen::VectorXd& b_lasso,
                                     const SelectionCoordinates& coords,
                                     const Eigen::VectorXd& G_hat,
                                     const Eigen::VectorXd& Z_noisy,
                                     double threshold,
                                     int top_k,
                                     const std::string& conditioning) {
    bool use_threshold = !std::isnan(threshold);
    bool use_top_k = top_k >= 0;
    if (use_threshold == use_top_k) throw std::invalid_argument("specify exactly one of threshold or top_k");
    if (conditioning != "first_dropped" && conditioning != "exact") {
        throw std::invalid_argument("conditioning must be 'first_dropped' or 'exact'");
    }
    if (use_top_k && top_k < 1) throw std::invalid_argument("top_k must be at least 1");

    const auto& E_c = coords.E_c();
    int m = static_cast<int>(E_c.size());
    Eigen::VectorXd G = take(G_hat, E_c);
    Eigen::VectorXd abs_G = G.cwiseAbs();

    std::vector<int> order(m);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return abs_G(a) > abs_G(b); });

    RowBuilder rows;
    std::vector<int> keep;

    if (use_threshold) {
        for (int k : order) if (abs_G(k) > threshold) keep.push_back(k);
        for (int k = 0; k < m; ++k) if (abs_G(k) > threshold) rows.add({{k, -sign(G(k))}}, -threshold);
        for (int k = 0; k < m; ++k) if (!(abs_G(k) > threshold)) rows.add({{k, 1.0}}, threshold);
        for (int k = 0; k < m; ++k) if (!(abs_G(k) > threshold)) rows.add({{k, -1.0}}, threshold);
    } else {
        int K = std::min(top_k, m);
        keep.assign(order.begin(), order.begin() + K);
        std::vector<int> drop(order.begin() + K, order.end());
        if (!drop.empty() && conditioning == "exact") {
            for (int k : keep) for (int l : drop) rows.add({{l, 1.0}, {k, -sign(G(k))}}, 0.0);
            for (int k : keep) for (int l : drop) rows.add({{l, -1.0}, {k, -sign(G(k))}}, 0.0);
        } else if (!drop.empty()) {
            int l0 = drop[0];
            double s0 = sign(G(l0));
            first_dropped = E_c[l0];
            first_dropped_sign = s0;
            // s0 G_l0 <= s_k G_k for kept k
            for (int k : keep) rows.add({{l0, s0}, {k, -sign(G(k))}}, 0.0);
            // |G_l| <= s0 G_l0 for the other dropped l
            for (size_t i = 1; i < drop.size(); ++i) rows.add({{drop[i], 1.0}, {l0, -s0}}, 0.0);
            for (size_t i = 1; i < drop.size(); ++i) rows.add({{drop[i], -1.0}, {l0, -s0}}, 0.0);
            if (drop.size() == 1) rows.add({{l0, -s0}}, 0.0);
        }
    }

    for (int k : keep) screened.push_back(E_c[k]);
    screened_signs.resize(keep.size());
    for (size_t i = 0; i < keep.size(); ++i) screened_signs(i) = sign(G(keep[i]));

    int n_rows = static_cast<int>(rows.q.size());
    Eigen::SparseMatrix<double> P(n_rows, m);
    P.setFromTriplets(rows.triplets.begin(), rows.triplets.end());
    Eigen::VectorXd q(n_rows);
    for (int i = 0; i < n_rows; ++i) q(i) = rows.q[i];

    // G = -(U(Z) + lasso_score_offset): P G <= q  <=>  -P U(Z) <= q + P lasso_score_offset
    Eigen::SparseMatrix<double> minus_P = -P;
    A_screen = std::make_shared<SparseProductOperator>(minus_P, coords.score_operator());
    b_screen = q + P * coords.lasso_score_offset();

    if (n_rows > 0) {
        Eigen::VectorXd slack = b_screen - A_screen->multiply(Z_noisy);
        Eigen::VectorXd tol = 1e-8 * (Eigen::VectorXd::Ones(n_rows) + b_screen.cwiseAbs());
        observed_feasible = ((slack + tol).array() >= 0).all();
    }

    A = std::make_shared<VStackOperator>(std::vector<std::shared_ptr<LinearOperator>>{A_lasso, A_screen});
    b.resize(b_lasso.size() + b_screen.size());
    b << b_lasso, b_screen;
}

} // namespace lassoinf
