import numpy as np
import pandas as pd
from ISLP.models import summarize
import statsmodels.api as sm

import adelie as ad
from selected_lasso import LassoInference


def test_intervals(n_features=30, dispersion=2, level=0.9):

    X = rng.standard_normal((5 * n_features, n_features))
    Y = rng.standard_normal(X.shape[0]) * np.sqrt(dispersion)
    weights = np.ones(X.shape[0])
    glm = ad.glm.gaussian(y=Y, weights=weights)
    lambda_max = np.fabs(X.T @ (Y * weights)).max() / weights.sum()

    penalty = np.ones(n_features)
    grp = ad.grpnet(X,
                    glm=glm,
                    lmda_path=lambda_max*np.linspace(1, 0.8, 20),
                    intercept=False)

    r = sm.OLS(Y, X).fit()
    est_dispersion = np.sum(r.resid**2) / r.df_resid
    
    initial_soln = grp.betas[-1].toarray().reshape(-1) # solve_lasso_adelie(sufficient_stat, 0, Q_mat, lambda_val, progress_bar=False)[0]
    active_set = np.where(np.fabs(initial_soln) > 0)[0]
    
    lasso_inf = LassoInference(
        initial_soln=initial_soln,
        sufficient_stat=X.T @ (Y * weights),
        Q_mat=X.T @ X,
        lambda_val=(grp.lmda_path[-1] * np.sum(weights)) * penalty,
        active_set=active_set)

    r_sel = sm.OLS(Y, X[:,active_set]).fit()
    nominal = summarize(r_sel, conf_int=True, level=level)

    return lasso_inf.summary(dispersion=est_dispersion, level=level), nominal








if __name__ == "__main__":

    rng = np.random.default_rng()# 0)
    level = 0.95
    df_sel = []
    df_nom = []
    for _ in range(100):
        sel, nominal = test_intervals(n_features=500, level=level)
        df_sel.append(sel)
        df_nom.append(nominal)
    df_sel = pd.concat(df_sel)
    df_nom = pd.concat(df_nom)
    L, U = df_nom.columns[-2:]
    df_nom = df_nom.rename(columns={L:f'L ({100*level:0.1f}%)',
                                     U:f'U ({100*level:0.1f}%)'})
    print('coverage (selective):', ((df_sel['L (95.0%)'] < 0) * (df_sel['U (95.0%)'] > 0)).mean())
    print('coverage (nominal):', ((df_nom['L (95.0%)'] < 0) * (df_nom['U (95.0%)'] > 0)).mean())
    print('type I error at 5% (selective):', (df_sel['p-value (twosided)'] < 0.05).mean())
    print('type I error at 5% (nominal):', (df_nom['P>|t|'] < 0.05).mean())
    print('mean length (selective):', (df_sel['U (95.0%)'] - df_sel['L (95.0%)']).mean())
    print('mean length (nominal):', (df_nom['U (95.0%)'] - df_nom['L (95.0%)']).mean())
