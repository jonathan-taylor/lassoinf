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
                    restr_dir,
                    restr_soln,
                    restr_stat,
                    restr_lambda,
                    restr_Qi=None):
    
    if restr_Qi is None:
        restr_Qi = np.linalg.inv(restr_Q)

    observed_target = (restr_dir * (restr_Qi @ restr_stat)).sum() # not quite right -- restr_soln is \REG not \OLS
    initial_t = observed_target
    unscaled_var = (restr_dir * (restr_Qi @ restr_dir)).sum()
    
    forward_path, _ = homotopy_path(restr_soln,
                                    restr_stat,
                                    restr_dir / unscaled_var,
                                    restr_Q, restr_lambda)

    t_arr = []
    signs_arr = []
    events_arr = []
    for t, _, _, signs, event in forward_path:
        t_arr.append(t)
        signs_arr.append(signs)
        events_arr.append(event)
    
    # make a nominal infinite endpoint

    t_arr.append(np.inf)
    signs_arr.append([True] * len(signs_arr[-1]))
    events_arr.append(('+infty', None))

    backward_path, _ = homotopy_path(restr_soln,
                                     restr_stat,
                                     -restr_dir / unscaled_var,
                                     restr_Q,
                                     restr_lambda)

    for t, _, _, signs, event in backward_path:
        t_arr.append(-t)   # negate the time for backward path
        signs_arr.append(signs)
        events_arr.append(event)

    # make a nominal negative infinite endpoint

    t_arr.append(-np.inf)
    signs_arr.append(signs_arr[-1].copy()) # last sign 
    events_arr.append(('-infty', None))


    df = pd.DataFrame({'t':t_arr, 'signs':signs_arr, 'events':events_arr})
    df = df.set_index('t').sort_index()
    df = df[lambda df: df['events'] != ('init', None)] # remove the initial events at t=0

    # convert the times to intervals of constancy for restr_dir.T @ restr_Qi @ restr_stat

    # now compute the intervals where all are active

    active_ints, active_signs = [], []

    for t_c, t_n in zip(df.index, df.index[1:]):
        if np.all(np.fabs(df.loc[t_c,'signs']) > 0):
            active_ints.append((t_c, t_n))
            active_signs.append(df.loc[t_c, 'signs'])

    active_df = pd.DataFrame({'intervals':active_ints, 'signs':active_signs})
    return active_df # path


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
    target_direction = rng.standard_normal(restr_Q.shape[0]) * 0.2 # target parameter is target_direction' beta_E
    
    restr_lambda = lambda_val[active_set]

    path = truncation_path(restr_Q,
                           target_direction,
                           restr_soln,
                           restr_stat,
                           restr_lambda)
    print(path)
    stop
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

if __name__ == "__main__":

    rng = np.random.default_rng(0)
    test_intervals()


