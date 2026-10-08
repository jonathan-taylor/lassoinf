#include "affine_constraints.hpp"

namespace lassoinf {

namespace {

Eigen::VectorXd to_vector(const std::vector<double>& v) {
    Eigen::VectorXd out(v.size());
    for (size_t i = 0; i < v.size(); ++i) out(i) = v[i];
    return out;
}

} // namespace

using detail::embed;
using detail::take;

// ---- InactiveScoreOperator ----

Eigen::VectorXd InactiveScoreOperator::coef(const Eigen::VectorXd& x) const {
    return W_ * take(x, E_);
}

Eigen::VectorXd InactiveScoreOperator::multiply(const Eigen::VectorXd& x) const {
    Eigen::VectorXd out = take(x, E_c_);
    if (!E_.empty()) {
        Eigen::VectorXd QWx = Q_->multiply(embed(cols(), E_, coef(x)));
        out -= take(QWx, E_c_);
    }
    return out;
}

Eigen::VectorXd InactiveScoreOperator::multiply_transpose(const Eigen::VectorXd& y) const {
    Eigen::VectorXd out = embed(cols(), E_c_, y);
    if (!E_.empty()) {
        Eigen::VectorXd Qy = Q_->multiply(out);
        Eigen::VectorXd out_E = -W_.transpose() * take(Qy, E_);
        for (size_t i = 0; i < E_.size(); ++i) out(E_[i]) = out_E(i);
    }
    return out;
}

// ---- LassoConstraintOperator ----

Eigen::VectorXd LassoConstraintOperator::multiply(const Eigen::VectorXd& x) const {
    Eigen::VectorXd out(rows());
    out.head(R_active_.rows()) = R_active_ * score_->coef(x);
    if (R_inactive_.rows() > 0) {
        out.tail(R_inactive_.rows()) = R_inactive_ * score_->multiply(x);
    }
    return out;
}

Eigen::VectorXd LassoConstraintOperator::multiply_transpose(const Eigen::VectorXd& y) const {
    Eigen::VectorXd out;
    if (R_inactive_.rows() > 0) {
        out = score_->multiply_transpose(R_inactive_.transpose() * y.tail(R_inactive_.rows()));
    } else {
        out = Eigen::VectorXd::Zero(cols());
    }
    const auto& E = score_->E();
    if (!E.empty()) {
        Eigen::VectorXd y_active = y.head(R_active_.rows());
        Eigen::VectorXd out_E = score_->W().transpose() * (R_active_.transpose() * y_active);
        for (size_t i = 0; i < E.size(); ++i) out(E[i]) += out_E(i);
    }
    return out;
}

// ---- VStackOperator ----

VStackOperator::VStackOperator(std::vector<std::shared_ptr<LinearOperator>> ops)
    : ops_(std::move(ops)), rows_(0) {
    if (ops_.empty()) throw std::invalid_argument("VStackOperator needs at least one operator");
    for (const auto& op : ops_) {
        if (op->cols() != ops_.front()->cols()) throw std::invalid_argument("VStackOperator: column mismatch");
        rows_ += op->rows();
    }
}

Eigen::VectorXd VStackOperator::multiply(const Eigen::VectorXd& x) const {
    Eigen::VectorXd out(rows_);
    Eigen::Index r = 0;
    for (const auto& op : ops_) {
        if (op->rows() > 0) out.segment(r, op->rows()) = op->multiply(x);
        r += op->rows();
    }
    return out;
}

Eigen::VectorXd VStackOperator::multiply_transpose(const Eigen::VectorXd& y) const {
    Eigen::VectorXd out = Eigen::VectorXd::Zero(cols());
    Eigen::Index r = 0;
    for (const auto& op : ops_) {
        if (op->rows() > 0) out += op->multiply_transpose(y.segment(r, op->rows()));
        r += op->rows();
    }
    return out;
}

// ---- constraints ----

LassoConstraints lasso_post_selection_constraints(
    const Eigen::VectorXd& beta_hat,
    const Eigen::VectorXd& G,
    std::shared_ptr<LinearOperator> Q,
    const Eigen::VectorXd& D_diag,
    const Eigen::VectorXd& L,
    const Eigen::VectorXd& U,
    double tol
) {
    const double inf = std::numeric_limits<double>::infinity();
    Eigen::Index n = Q->rows();
    Eigen::VectorXd L_bound = L.size() > 0 ? L : Eigen::VectorXd::Constant(n, -inf);
    Eigen::VectorXd U_bound = U.size() > 0 ? U : Eigen::VectorXd::Constant(n, inf);

    std::vector<int> E;
    std::vector<int> E_c;
    std::vector<double> s_E_vec;
    std::vector<double> v_Ec_vec;
    std::vector<double> g_min_vec;
    std::vector<double> g_max_vec;

    for (int j = 0; j < n; ++j) {
        double beta_val = beta_hat(j);
        bool at_L = (beta_val <= L_bound(j) + tol);
        bool at_U = (beta_val >= U_bound(j) - tol);
        bool at_0 = (std::abs(beta_val) <= tol);

        if (!at_L && !at_U && !at_0) {
            E.push_back(j);
            s_E_vec.push_back(beta_val > 0 ? 1.0 : (beta_val < 0 ? -1.0 : 0.0));
        } else {
            E_c.push_back(j);
            double v_j;
            if (at_0) v_j = 0.0;
            else if (at_U) v_j = U_bound(j);
            else v_j = L_bound(j);
            v_Ec_vec.push_back(v_j);

            double dj = D_diag(j);
            double gmin = -inf;
            double gmax = inf;
            if (at_0) {
                if (L_bound(j) < -tol) gmin = -dj;
                if (U_bound(j) > tol) gmax = dj;
            } else if (at_U) {
                gmin = dj;
            } else if (at_L) {
                gmax = -dj;
            }
            g_min_vec.push_back(gmin);
            g_max_vec.push_back(gmax);
        }
    }

    Eigen::VectorXd s_E = to_vector(s_E_vec);
    Eigen::VectorXd v_Ec = to_vector(v_Ec_vec);

    // Q V with V = v on E_c (zero when all inactive coordinates are at 0)
    Eigen::VectorXd V_vec = embed(n, E_c, v_Ec);
    Eigen::VectorXd Q_V = v_Ec.size() > 0 && v_Ec.cwiseAbs().maxCoeff() > 0
        ? Q->multiply(V_vec) : Eigen::VectorXd::Zero(n);

    Eigen::MatrixXd W(E.size(), E.size());
    Eigen::VectorXd c_E(E.size());
    Eigen::VectorXd c_Ec;

    if (!E.empty()) {
        // Q_{E,E} from |E| matvecs, keeping only the E rows of each column
        Eigen::MatrixXd Q_EE(E.size(), E.size());
        for (size_t i = 0; i < E.size(); ++i) {
            Eigen::VectorXd e_i = Eigen::VectorXd::Zero(n);
            e_i(E[i]) = 1.0;
            Q_EE.col(i) = take(Q->multiply(e_i), E);
        }
        W = Q_EE.inverse();

        Eigen::VectorXd D_E = take(D_diag, E);
        c_E = W * (take(Q_V, E) + D_E.cwiseProduct(s_E));
        c_Ec = take(Q->multiply(embed(n, E, c_E)), E_c) - take(Q_V, E_c);
    } else {
        c_Ec = -take(Q_V, E_c);
    }

    // active rows act on W Z_E: signs, then bounds
    std::vector<Eigen::Triplet<double>> active_triplets;
    std::vector<double> b_list;
    int row = 0;
    for (size_t k = 0; k < E.size(); ++k) {
        active_triplets.emplace_back(row++, k, -s_E(k));
        b_list.push_back(-s_E(k) * c_E(k));
    }
    for (size_t k = 0; k < E.size(); ++k) {
        int j = E[k];
        if (s_E(k) == 1.0 && U_bound(j) < inf) {
            active_triplets.emplace_back(row++, k, 1.0);
            b_list.push_back(U_bound(j) + c_E(k));
        } else if (s_E(k) == -1.0 && L_bound(j) > -inf) {
            active_triplets.emplace_back(row++, k, -1.0);
            b_list.push_back(-L_bound(j) - c_E(k));
        }
    }
    Eigen::SparseMatrix<double> R_active(row, E.size());
    R_active.setFromTriplets(active_triplets.begin(), active_triplets.end());

    // inactive rows act on U_{-E}(Z): upper then lower subgradient bound
    std::vector<Eigen::Triplet<double>> inactive_triplets;
    row = 0;
    for (size_t k = 0; k < E_c.size(); ++k) {
        if (g_max_vec[k] < inf) {
            inactive_triplets.emplace_back(row++, k, 1.0);
            b_list.push_back(g_max_vec[k] - c_Ec(k));
        }
        if (g_min_vec[k] > -inf) {
            inactive_triplets.emplace_back(row++, k, -1.0);
            b_list.push_back(-g_min_vec[k] + c_Ec(k));
        }
    }
    Eigen::SparseMatrix<double> R_inactive(row, E_c.size());
    R_inactive.setFromTriplets(inactive_triplets.begin(), inactive_triplets.end());

    Eigen::VectorXd b_final = to_vector(b_list);

    auto score = std::make_shared<InactiveScoreOperator>(Q, E, E_c, W);
    auto A = std::make_shared<LassoConstraintOperator>(score, R_active, R_inactive);

    return LassoConstraints{A, b_final, E, E_c, s_E, v_Ec, W};
}

} // namespace lassoinf
