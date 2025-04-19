from dataclasses import dataclass

import numpy as np
import pandas as pd

from .homotopy import (homotopy_path,
                       solve_lasso_adelie)
from .truncated_gaussian import truncated_gaussian

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
    """
    Computes the truncation path for inference.

    Parameters
    ----------
    restr_Q : ndarray
        Restricted Q matrix.
    restr_soln : ndarray
        Restricted solution.
    restr_stat : ndarray
        Restricted sufficient statistic.
    restr_lambda : ndarray
        Restricted lambda values.
    restr_dir : ndarray
        Restriction direction.
    restr_Qi : ndarray, optional
        Inverse of restricted Q matrix, by default None.
    check_adelie : bool, optional
        Whether to check against Adelie solution, by default False.
    adelie_tol : float, optional
        Tolerance for Adelie check, by default 1e-3.

    Returns
    -------
    DataFrame
        DataFrame containing the truncation path information.
    """
    
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
            t_neg *= 2

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
    df.loc[0,'t'] = -np.inf
    df = df.set_index('t').sort_index()
    
    if check_adelie:
        for i in range(df.shape[0]):
            t = df.index[i]
            if np.isfinite(t):
                A, _ = solve_lasso_adelie(restr_stat,
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

@dataclass
class LassoInference(object):
    """
    Class for performing inference on Lasso estimates.

    Attributes
    ----------
    initial_soln : ndarray
        Initial solution vector.
    sufficient_stat : ndarray
        Sufficient statistic vector.
    Q_mat : ndarray
        Q matrix.
    lambda_val : ndarray
        Lambda values.
    active_set : ndarray
        Active set indices.
    check_adelie : bool, optional
        Whether to check against Adelie solution, by default False.
    B : int, optional
        Number of bootstrap iterations, by default 100000.
    seed : int, optional
        Random seed, by default 0.
    """

    initial_soln: np.ndarray
    sufficient_stat: np.ndarray
    Q_mat: np.ndarray
    lambda_val: np.ndarray
    active_set: np.ndarray
    check_adelie: bool = False  # Default value
    B: int = 100000
    seed: int = 0
    
    def __post_init__(self):
        self.Q_mat = np.asfortranarray(self.Q_mat)
        self._restr_soln = self.initial_soln[self.active_set]
        self._restr_stat = self.sufficient_stat[self.active_set]
        self._restr_Q = self.Q_mat[np.ix_(self.active_set, self.active_set)]
        self._restr_lambda = self.lambda_val[self.active_set]
        self._restr_Qi = np.linalg.inv(self._restr_Q)
        self._unreg_soln = self._restr_Qi @ self._restr_stat
        self._cache = {}
        self._sign_cache = {}
        self._inactive_set = np.ones(self.Q_mat.shape[0], bool)
        self._inactive_set[self.active_set] = 0
        self._irrep = self.Q_mat[np.ix_(self._inactive_set, self.active_set)] @ self._restr_Qi

    def confint(self,
                contrast,
                method='chernoff',
                level=0.95,
                dispersion=1):
        """
        Computes confidence intervals for a linear contrast of the Lasso estimate.

        Parameters
        ----------
        contrast : ndarray
            Contrast vector.
        method : str, optional
            Method for computing confidence intervals, by default 'chernoff'.
        level : float, optional
            Confidence level, by default 0.95.
        dispersion : float, optional
            Dispersion parameter, by default 1.

        Returns
        -------
        tuple
            Lower and upper bounds of the confidence interval.
        """
        
        obs = contrast @ self._unreg_soln
        law, scale = self._retrieve_law(contrast, method, dispersion)

        return law.equal_tailed_interval(obs, level=level)

    def pvalue(self, 
               contrast,
               null_value=0,
               method='chernoff',
               dispersion=1,
               alternative='twosided'):
        """
        Computes the p-value for a hypothesis test on a linear contrast of the Lasso estimate.

        Parameters
        ----------
        contrast : ndarray
            Contrast vector.
        null_value : float, optional
            Null hypothesis value, by default 0.
        method : str, optional
            Method for computing the p-value, by default 'chernoff'.
        dispersion : float, optional
            Dispersion parameter, by default 1.
        alternative : str, optional
            Alternative hypothesis, by default 'twosided'.

        Returns
        -------
        float
            The p-value.
        """
        law, scale = self._retrieve_law(contrast, method, dispersion)
        law.set_mu(null_value)

        obs = contrast @ self._unreg_soln
        pval = law.cdf(obs)

        if alternative == 'greater':
            return 1 - pval
        elif alternative == 'less':
            return pval
        elif alternative == 'twosided':
            return 2 * min(pval, 1 - pval)
        else:
            raise ValueError('alternative should be one of ["greater", "less", "twosided"]')
        
    def summary(self,
                method='chernoff',
                dispersion=1,
                alternative='twosided',
                level=0.95):
        """
        Computes a summary of confidence intervals and p-values for each active variable.

        Parameters
        ----------
        method : str, optional
            Method for inference, by default 'chernoff'.
        dispersion : float, optional
            Dispersion parameter, by default 1.
        alternative : str, optional
            Alternative hypothesis for p-values, by default 'twosided'.
        level : float, optional
            Confidence level, by default 0.95.

        Returns
        -------
        DataFrame
            DataFrame containing confidence intervals and p-values.
        """
        
        L, U, pvals = [], [], []
        for elem_basis in np.eye(len(self.active_set)):
            l, u = self.confint(elem_basis,
                                dispersion=dispersion,
                                method=method,
                                level=level)
            L.append(l); U.append(u)
            pvals.append(self.pvalue(elem_basis,
                                     method=method,
                                     dispersion=dispersion))
        df = pd.DataFrame({f'L ({100*level:0.1f}%)': L,
                           f'U ({100*level:0.1f}%)': U,
                           f'p-value ({alternative})':pvals,
                           'ID':self.active_set})
        return df.set_index('ID')

    # Private API

    def _retrieve_law(self,
                      contrast,
                      method,
                      dispersion=1):
        """
        Retrieves the law (truncated Gaussian) for inference, computing it if necessary.

        Parameters
        ----------
        contrast : ndarray
            Contrast vector.
        method : str
            Method for approximating the distribution.
        dispersion : float, optional
            Dispersion parameter, by default 1.

        Returns
        -------
        tuple
            Tuple containing the truncated Gaussian distribution and scale.
        """

        if (tuple(contrast), method, dispersion) not in self._cache:
            path = truncation_path(self._restr_Q,
                                   self._restr_soln,
                                   self._restr_stat,
                                   self._restr_lambda,
                                   contrast,
                                   restr_Qi=self._restr_Qi,
                                   check_adelie=False)

            chernoff, MC = self._approx_probability_signs(path,
                                                          dispersion=dispersion,
                                                          do_MC=method == 'MC')
            chernoff /= chernoff.sum()
            MC /= MC.sum()
            
            method = 'mix'
            if method == 'chernoff':
                weights = chernoff
            elif method == 'MC':
                weights = MC
            elif method == 'mix':
                if np.all(np.isfinite(MC)):
                    weights = 0.1 * chernoff + 0.9 * MC
                else:
                    weights = chernoff
            else:
                raise ValueError('method for weights should be in ["chernoff", "MC"]')

            L = [i[0] for i in path['intervals']]
            R = [i[1] for i in path['intervals']]
            scale = np.sqrt((contrast @ self._restr_Qi @ contrast) * dispersion)

            law = truncated_gaussian(left=L,
                                     right=R,
                                     weights=weights,
                                     scale=scale)
            self._cache[(tuple(contrast), method, dispersion)] = (law, scale)
        return self._cache[(tuple(contrast), method, dispersion)]

    def _approx_probability_signs(self,
                                  path,
                                  dispersion=1,
                                  do_MC=False):
        """
        Approximates the probability of sign patterns along the truncation path.

        Parameters
        ----------
        path : DataFrame
            Truncation path DataFrame.
        dispersion : float, optional
            Dispersion parameter, by default 1.
        do_MC : bool, optional
            Whether to perform Monte Carlo approximation, by default False.

        Returns
        -------
        tuple
            Tuple containing Chernoff and Monte Carlo weights.
        """

        (Q_mat,
         active_set,
         inactive_set,
         lambda_val,
         B) = (self.Q_mat,
               self.active_set,
               self._inactive_set,
               self.lambda_val,
               self.B)

        restr_lambda = self._restr_lambda
        new_lambda = lambda_val.copy()
        new_lambda[active_set] = 0

        weights = []
        weights_MC = []

        Q_i = self._restr_Qi

        irrep = self._irrep
        linear_term = np.asfortranarray(np.zeros(Q_mat.shape[0]))
        inact_lambda = lambda_val[inactive_set]

        rng = np.random.default_rng(self.seed)
        if do_MC: 
            if not hasattr(self, "_chol"):
                cond_cov = Q_mat - Q_mat[:,active_set] @ (Q_i @ Q_mat[active_set])
                chol = self._chol = np.linalg.cholesky(cond_cov[np.ix_(inactive_set, inactive_set)]) * np.sqrt(dispersion)
            else:
                chol = self._chol
            N = rng.standard_normal((B, chol.shape[0])) @ chol.T

        for i in range(path.shape[0]):
            sign_tuple = tuple(path.loc[i, 'signs'])
            if sign_tuple not in self._sign_cache:
                bias = irrep @ (restr_lambda * path.loc[i, 'signs'])
                linear_term[active_set] = -path.loc[i,'signs'] * restr_lambda
                q_active = linear_term[active_set] @ Q_i @ linear_term[active_set] / 2 # \|v\|^2_2 / 2 
                soln, S = solve_lasso_adelie(linear_term, 0, Q_mat, new_lambda, progress_bar=False)

                value = -linear_term @ soln + S.devs[0] / 2 + np.fabs(inact_lambda * soln[inactive_set]).sum()  
                value += q_active 
                weight = np.exp(-value)
                if do_MC:
                    weight_MC = (np.fabs((N + bias) / inact_lambda[None,:]).max(1) < 1).mean()
                else:
                    weight_MC = np.nan
                self._sign_cache[sign_tuple] = (weight, weight_MC)
            weight, weight_MC = self._sign_cache[sign_tuple]
            weights.append(weight)
            weights_MC.append(weight_MC)
            
        return np.array(weights), np.array(weights_MC)

