import numpy as np

def spec_from_glmnet(glmnet_obj,
                     data,
                     lambda_val,
                     state,
                     proportion,
                     dispersion=None):
    G = glmnet_obj
    X_full, Df_full = data
    _, _, Y_full, _, weight_full = G.get_data_arrays(X_full, Df_full)
    ridge_coef = (1 - G.alpha) * lambda_val * weight_full.sum()

    active_set = np.nonzero(state.coef != 0)[0]

    if active_set.shape[0] == 0:
        return None

    unreg_GLM = glmnet_obj.get_GLM(ridge_coef=ridge_coef)
    unreg_GLM.summarize = True
    unreg_GLM.fit(X_full[:, active_set], Df_full, dispersion=dispersion)

    D_active = glmnet_obj.get_design(X_full[:, active_set],
                                     weight_full,
                                     standardize=glmnet_obj.standardize,
                                     intercept=glmnet_obj.fit_intercept)

    info = unreg_GLM._information
    P_active = D_active.quadratic_form(info, transformed=True)

    D0 = np.ones(D_active.shape[1])
    if G.fit_intercept:
        D0[0] = 0
    DIAG_active = np.diag(D0) * ridge_coef

    if not G.fit_intercept:
        if G.penalty_factor is not None:
            penfac = G.penalty_factor[active_set]
        else:
            penfac = np.ones_like(active_set)
        P_active = P_active[1:, 1:]
        DIAG_active = DIAG_active[1:, 1:]
    else:
        penfac = np.ones(active_set.shape[0])

    hessian = P_active + DIAG_active
    Q_active = np.linalg.inv(hessian)
    
    signs = np.sign(state.coef[active_set])

    if G.fit_intercept:
        penfac = np.hstack([0, penfac])
        signs = np.hstack([0, signs])
        stacked = np.hstack([state.intercept, state.coef[active_set]])
    else:
        stacked = state.coef[active_set]

    penalized = penfac > 0
    n_penalized = penalized.sum()
    n_coef = penalized.shape[0]
    row_idx = np.arange(n_penalized)
    col_idx = np.nonzero(penalized)[0]
    data = -signs[penalized]
    sel_active = scipy.sparse.coo_matrix((data, (row_idx, col_idx)), shape=(n_penalized, n_coef))

    linear = sel_active
    offset = np.zeros(sel_active.shape[0])

    return {'D': D_active,
            'L': linear,
            'U': offset,
            'gradient': -Q_active @ (penfac * lambda_val * signs) * weight_full.sum(),
            'hessian': hessian}
