import numpy as np
import pandas as pd
from homotopy import (homotopy_path,
                      solve_lasso_adelie)

'''
Suppose out target is $\hat{\theta} = \eta'\bar{\beta}_E=\eta'Q_E^{-1}T_E$ with $T_E=X_E'y$ in OLS regression.

The decomposition of $T_E$ is therefore

$$
T_E = N_E + \Gamma_E \cdot \hat{\theta}
$$

with

$$
\Gamma_E = \frac{\text{Cov}(S_E, \hat{\theta}) }{\text{Var}(\hat{\theta})} = \frac{1}{\eta'Q_E^{-1}\eta} \cdot \eta
$$
'''

def truncation_path(restr_Q,
                    restr_soln,
                    restr_stat,
                    restr_lambda,
                    restr_dir,
                    restr_Qi=None,
                    check_adelie=False,
                    adelie_tol=1e-3):
    
    if restr_Qi is None:
        restr_Qi = np.linalg.inv(restr_Q)

    observed_target = (restr_dir * (restr_Qi @ restr_stat)).sum() # not quite right -- restr_soln is \REG not \OLS
    initial_t = observed_target
    unscaled_var = (restr_dir * (restr_Qi @ restr_dir)).sum()
    
    limit_signs = np.sign(restr_Qi @ restr_dir)

    # find a t such that signs are the limit signs

    unscaled_sd = np.sqrt(unscaled_var)
    t_neg = -10 * unscaled_var
    reached_limit = False
    _Qi_dir = restr_Qi @ restr_dir / unscaled_var
    _unreg_soln = restr_Qi @ restr_stat
    
    _delta = restr_Qi @ (restr_lambda * limit_signs)
    
    while not reached_limit:
        _beta = _unreg_soln + t_neg * _Qi_dir + _delta
        if np.all(np.sign(_beta) == -limit_signs):
            reached_limit = True
        else:
            t_neg -= 10 * unscaled_var

    forward_path, _ = homotopy_path(_beta, # restr_soln,
                                    restr_stat + t_neg * restr_dir / unscaled_var,
                                    restr_dir / unscaled_var,
                                    restr_Q,
                                    restr_lambda)

    t_arr = []
    signs_arr = []
    events_arr = []
    beta_arr = []
    
    for t, beta, _, signs, event in forward_path:
        t_arr.append(t + t_neg + initial_t)
        signs_arr.append(signs)
        events_arr.append(event)
        beta_arr.append(beta.copy())

    # make a nominal infinite endpoint

    t_arr.append(np.inf)
    signs_arr.append([True] * len(signs_arr[-1]))
    events_arr.append(('+infty', None))
    beta_arr.append(np.nan * restr_soln)
    
    df = pd.DataFrame({'t':t_arr, 'signs':signs_arr, 'events':events_arr, 'beta':beta_arr})
    df = df[lambda df: df['events'] != ('init', None)] # remove the initial events at t=0
    df = df.set_index('t').sort_index()
    
    if check_adelie:
        for i in range(df.shape[0]):
            t = df.index[i]
            if np.isfinite(t):
                A = solve_lasso_adelie(restr_stat,
                                       restr_dir / unscaled_var,
                                       restr_Q,
                                       restr_lambda,
                                       t=t-initial_t)
                assert np.linalg.norm(A - df.loc[t,'beta']) / max(np.linalg.norm(A), 1) < adelie_tol

    # now compute the intervals where all are active

    active_ints, active_signs, betas, events = [], [], [], []

    for t_c, t_n in zip(df.index, df.index[1:]):
        if np.all(np.fabs(df.loc[t_c,'signs']) > 0):
            active_ints.append((t_c, t_n))
            active_signs.append(df.loc[t_c, 'signs'].copy())
            if np.isfinite(t_c):
                betas.append((df.loc[t_c, 'beta'].copy(), df.loc[t_n, 'beta'].copy()))
                events.append(df.loc[t_c, 'events'])
            else:
                betas.append((df.loc[t_c, 'beta'].copy(), df.loc[t_n, 'beta'].copy()))
                events.append(df.loc[t_n, 'events'])

    active_df = pd.DataFrame({'intervals':active_ints, 'signs':active_signs, 'beta':betas, 'event':events})
    return active_df


def test_intervals(n_features=30):

    n_features = 30

    sufficient_stat = rng.standard_normal(n_features)
    W = []
    W = [rng.standard_normal(2 * n_features)]
    for i in range(n_features - 1):
        W.append(0.7 * W[-1] + rng.standard_normal(2 * n_features))
    W = np.array(W)
    Q_mat = W @ W.T / n_features
    lambda_val = 1.2 * np.ones(n_features)

    initial_soln = solve_lasso_adelie(sufficient_stat, 0, Q_mat, lambda_val)
    active_set = np.where(np.fabs(initial_soln) > 0)[0]
    
    # restrict the problem now

    restr_soln = initial_soln[active_set]
    restr_stat = sufficient_stat[active_set]
    restr_Q = Q_mat[np.ix_(active_set, active_set)]
    restr_lambda = lambda_val[active_set]
    restr_Qi = np.linalg.inv(restr_Q)
    unreg_soln = restr_Qi @ restr_stat
    
    paths = []
    for elem_basis in np.eye(restr_Q.shape[0]):
        path = truncation_path(restr_Q,
                               restr_soln,
                               restr_stat,
                               restr_lambda,
                               elem_basis,
                               restr_Qi=restr_Qi,
                               check_adelie=True)
        paths.append(path)
        path['weights'] = _approx_probability_signs(Q_mat, path)

    #path = forward_path
    
    adelie_solns = [
        (
            t,
            solve_lasso_adelie(restr_stat, direction, restr_Q, restr_lambda, t=t),
        )
        for t, _, _, _ in path
    ]

    H = np.array([beta for _, beta, _, _ in path])
    A = np.array([beta for _, beta in adelie_solns])


    print(np.linalg.norm(A - H) / max(np.linalg.norm(A), 1), np.linalg.norm(A))
    if hasattr(hpath, "_flag"):
        print('flagged a problem')

def _approx_probability_signs(Q, path):
    return np.ones(path.shape[0])

if __name__ == "__main__":

    rng = np.random.default_rng()# 0)
    test_intervals()


