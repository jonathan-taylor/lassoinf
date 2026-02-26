#include "selective_inference.hpp"

namespace lassoinf {

LassoConstraints lasso_post_selection_constraints(
    const Eigen::VectorXd& beta_hat,
    const Eigen::VectorXd& G,
    std::shared_ptr<LinearOperator> Q,
    const Eigen::VectorXd& D_diag,
    const Eigen::VectorXd& L,
    const Eigen::VectorXd& U,
    double tol
) {
    Eigen::Index n = Q->rows();
    Eigen::VectorXd L_bound = L.size() > 0 ? L : Eigen::VectorXd::Constant(n, -std::numeric_limits<double>::infinity());
    Eigen::VectorXd U_bound = U.size() > 0 ? U : Eigen::VectorXd::Constant(n, std::numeric_limits<double>::infinity());

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
            double gmin = -std::numeric_limits<double>::infinity();
            double gmax = std::numeric_limits<double>::infinity();
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

    Eigen::VectorXd s_E(s_E_vec.size());
    if(!s_E_vec.empty()) s_E = Eigen::Map<Eigen::VectorXd>(s_E_vec.data(), s_E_vec.size());
    
    Eigen::VectorXd v_Ec(v_Ec_vec.size());
    if(!v_Ec_vec.empty()) v_Ec = Eigen::Map<Eigen::VectorXd>(v_Ec_vec.data(), v_Ec_vec.size());
    
    Eigen::VectorXd g_min(g_min_vec.size());
    if(!g_min_vec.empty()) g_min = Eigen::Map<Eigen::VectorXd>(g_min_vec.data(), g_min_vec.size());
    
    Eigen::VectorXd g_max(g_max_vec.size());
    if(!g_max_vec.empty()) g_max = Eigen::Map<Eigen::VectorXd>(g_max_vec.data(), g_max_vec.size());

    std::vector<double> b_list;
    std::vector<Eigen::Triplet<double>> S_triplets;
    int current_row = 0;
    std::vector<Eigen::VectorXd> U_rows;

    Eigen::MatrixXd Q_E;
    Eigen::MatrixXd W;
    Eigen::VectorXd c_E;

    if (!E.empty()) {
        Q_E.resize(n, E.size());
        for (size_t i = 0; i < E.size(); ++i) {
            Eigen::VectorXd e_i = Eigen::VectorXd::Zero(n);
            e_i(E[i]) = 1.0;
            Q_E.col(i) = Q->multiply(e_i);
        }

        Eigen::MatrixXd Q_EE(E.size(), E.size());
        for (size_t i = 0; i < E.size(); ++i) {
            Q_EE.row(i) = Q_E.row(E[i]);
        }
        Eigen::MatrixXd Q_EcE(E_c.size(), E.size());
        for (size_t i = 0; i < E_c.size(); ++i) {
            Q_EcE.row(i) = Q_E.row(E_c[i]);
        }

        W = Q_EE.inverse();
        
        Eigen::VectorXd D_E(E.size());
        for(size_t i=0; i<E.size(); ++i) D_E(i) = D_diag(E[i]);

        c_E = W * (Q_EcE.transpose() * v_Ec + D_E.cwiseProduct(s_E));

        Eigen::MatrixXd diag_s_E = s_E.asDiagonal();
        Eigen::MatrixXd U_part = -diag_s_E * W;
        for (int i = 0; i < U_part.rows(); ++i) {
            U_rows.push_back(U_part.row(i));
            b_list.push_back((-diag_s_E * c_E)(i));
            current_row++;
        }

        for (size_t k = 0; k < E.size(); ++k) {
            int j = E[k];
            if (s_E(k) == 1.0 && U_bound(j) < std::numeric_limits<double>::infinity()) {
                U_rows.push_back(W.row(k));
                b_list.push_back(U_bound(j) + c_E(k));
                current_row++;
            } else if (s_E(k) == -1.0 && L_bound(j) > -std::numeric_limits<double>::infinity()) {
                U_rows.push_back(-W.row(k));
                b_list.push_back(-L_bound(j) - c_E(k));
                current_row++;
            }
        }
    } else {
        W.resize(0, 0);
        c_E.resize(0);
    }

    if (!E_c.empty()) {
        Eigen::VectorXd V_vec = Eigen::VectorXd::Zero(n);
        for (size_t i = 0; i < E_c.size(); ++i) V_vec(E_c[i]) = v_Ec(i);

        Eigen::VectorXd Q_V = Q->multiply(V_vec);
        Eigen::VectorXd Q_EcEc_v_Ec(E_c.size());
        for (size_t i = 0; i < E_c.size(); ++i) {
            Q_EcEc_v_Ec(i) = Q_V(E_c[i]);
        }

        Eigen::MatrixXd Q_EcE(E_c.size(), E.size());
        if (!E.empty()) {
            for (size_t i = 0; i < E_c.size(); ++i) {
                Q_EcE.row(i) = Q_E.row(E_c[i]);
            }
        }

        Eigen::MatrixXd U_part = E.empty() ? Eigen::MatrixXd::Zero(E_c.size(), 0) : Eigen::MatrixXd(-Q_EcE * W);
        Eigen::VectorXd c_Ec = E.empty() ? Eigen::VectorXd(-Q_EcEc_v_Ec) : Eigen::VectorXd(Q_EcE * c_E - Q_EcEc_v_Ec);

        for (size_t k = 0; k < E_c.size(); ++k) {
            int j = E_c[k];
            if (g_max(k) < std::numeric_limits<double>::infinity()) {
                if (!E.empty()) U_rows.push_back(U_part.row(k));
                else U_rows.push_back(Eigen::VectorXd::Zero(0));
                
                S_triplets.push_back(Eigen::Triplet<double>(current_row, j, 1.0));
                b_list.push_back(g_max(k) - c_Ec(k));
                current_row++;
            }
            if (g_min(k) > -std::numeric_limits<double>::infinity()) {
                if (!E.empty()) U_rows.push_back(-U_part.row(k));
                else U_rows.push_back(Eigen::VectorXd::Zero(0));

                S_triplets.push_back(Eigen::Triplet<double>(current_row, j, -1.0));
                b_list.push_back(-g_min(k) + c_Ec(k));
                current_row++;
            }
        }
    }

    int m = current_row;
    Eigen::SparseMatrix<double> S_final(m, n);
    S_final.setFromTriplets(S_triplets.begin(), S_triplets.end());

    Eigen::MatrixXd U_final(m, E.size());
    for (int i = 0; i < m; ++i) {
        if (!E.empty()) U_final.row(i) = U_rows[i];
    }

    Eigen::MatrixXd V_final = Eigen::MatrixXd::Zero(n, E.size());
    for (size_t i = 0; i < E.size(); ++i) {
        V_final(E[i], i) = 1.0;
    }

    Eigen::VectorXd b_final(b_list.size());
    if(!b_list.empty()) {
        b_final = Eigen::Map<Eigen::VectorXd>(b_list.data(), b_list.size());
    }

    CompositeComponent comp;
    comp.S = S_final;
    comp.U = U_final;
    comp.V = V_final;
    comp.b = Eigen::VectorXd();

    std::vector<CompositeComponent> components;
    components.push_back(comp);
    auto A = std::make_shared<CompositeOperator>(m, n, components);

    return LassoConstraints{A, b_final, E, E_c, s_E, v_Ec};
}

} // namespace lassoinf