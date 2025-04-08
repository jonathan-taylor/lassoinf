"""
[Gemini](https://gemini.google.com/app/e24858f8011e1558)

Solves a homotopy continuation problem for a modified LASSO-like optimization problem.

This code implements a path-following algorithm to trace the solution of the
following optimization problem as a function of the homotopy parameter 't':

$$
\textrm{minimize}_{b}  -beta'(N + t * V) + 1/2 * b' * Q * b + \lambda * \|b\|_1
$$

where:

- $b$: vector of variables being optimized.
- $N$: vector corresponding to the initial solution
- $V$: the direction of the homotopy `dir_vec` below
- $Q$: A known positive semi-definite matrix.
- $\lambda$: A fixed regularization parameter (lambda >= 0).
- $t$: The homotopy parameter.

The algorithm tracks the solution b(t) as 't' varies, efficiently updating the
Cholesky decomposition of relevant submatrices to improve performance.
It handles events where variables enter or leave the active set.

The implementation uses the 'cholupdates' library for Cholesky updates
and downdates, improving numerical stability and efficiency compared to
recomputing the Cholesky decomposition at each step.

# Homotopy problem

We want to construct a path of solutions
$$
\textrm{minimize}_{b}  -b'(N + t \cdot v) + \frac{1}{2} b'  Q  b + \|\Lambda \beta\|_1
$$
with $\Lambda = \text{diag}(\lambda)$ a set of penalty factors (possibly 0).

## Leaving time calculation                                                                                                               
                                                                                                                                           
Suppose at $t_c$ we know the solution is $b_c = \hat{\beta}(t_c)$ with signs and active set                                                
$A_c=A(t_c), s_c=s_A(t_c)$. Then, for $t > t_c$ the proposed solution is                                                                   
                                                                                                                                           
$$                                                                                                                                         
b_c + (t - t_c) * Q_{A_c}^{-1} v_{A_c}                                                                                                     
$$                                                                                                                                         
                                                                                                                                           
The active conditions say that to keep the active set fixed we must have                                                                  $$                                                                                                                                        s_{A_c} * (b_c + (t - t_c) * Q_{A_c}^{-1} v_{A_c}) \geq 0                                                                                 $$ 

The zero time for variable $j$ satisfies
$$
\begin{aligned}
s_{A_c,j} * (b_{c,j} + (t_j - t_c) (Q_{A_c}^{-1} v_{A_c})_j) = 0
  & \iff b_{c,j} - (t_j - t_c) (Q_{A_c}^{-1} v_{A_c})_j = 0 \\
t_{j,d} = t_c - \frac{b_{c,j}}{(Q_{A_c}^{-1} v_{A_c})_j}.
\end{aligned}
$$
We're only interested in $t_{j,d} > t_c$ because we are going forward in time.

Therefore, the next deletion time is
$$
t_d = \min(t_{j,d}: t_{j,d} > t_c)
$$

## Adding time

At time $t_c$, with inactive set $I(t_c)=I_c$ we know that the inactive subgradient's value is
the negative of the gradient of the quadratic part at $b_c$:
$$
U_c = N_{I_c} - Q_{I_c,A_c}b_c
$$
$$
\|\Lambda_{I_c}^{-1}U_c\|_{\infty} \leq 1
$$

Moving along the solution path as $b_A the subgradient is therefore
$$
U_c + (t - t_c) \cdot(v_{I_c,j} - Q_{I_c,A_c} Q_{A_c}^{-1} v_{A_c})
$$

Setting the coordinates to $\pm \lambda_j$ we get
$$
t_{j,a}^+ = t_c + \frac{\lambda_j - U_{c,j}}{ v_{I_c,j} - (Q_{I_c,A_c} Q_{A_c}^{-1} v_{A_c})_j }
$$
and
$$
t_{j,a}^- = t_c + \frac{-\lambda_j - U_{c,j}}{v_{I_c,j} - (Q_{I_c,A_c} Q_{A_c}^{-1} v_{A_c})_j}
$$
so
$$
t_{j,a} = \min(t_{j,a}^{\pm}: t_{j,a}^{\pm} > t_c)
$$
(I think only one will be above $t_c$).
and the next added variable will be
$$
t_a = \min(t_j: t_j > t_c)
$$


Therefore,
$$
t_{j,a} = \min(t_{j,a}^{\pm}: t_{j,a}^{\pm} > t_c)
$$
(I think only one will be above $t_c$).
and the next added variable will be
$$
t_a = \min(t_j: t_j > t_c)
$$

In either case, the $\beta$ solution at the next time
$$
t_n = \min(t_a, t_d)
$$
we'll have our terminal $\beta$ value for the $A_c$ coordinates
$$
\beta_c + (t_n - t_c) Q_{A_c}^{-1} v_{A_c}
$$

The $I_c$ coordinates at $t_n$ for $\beta$ will also be 0. We'll similarly have our terminal
subgradient 
$$
U_c + (t_n - t_c) Q_{I_c,A_c} Q_{A_c}^{-1} v_{A_c}
$$

"""

import numpy as np
from scipy.linalg import solve
from cholupdates.rank_1 import update as update_chol 
from cholupdates.rank_1 import downdate as downdate_chol
from dataclasses import dataclass
from adelie.solver import gaussian_cov


@dataclass
class ActiveCholesky(object):

    Q_mat: np.ndarray
    active_indices: list
    inactive_indices: list
    chol: np.ndarray # lower cholesky

    def __post_init__(self):
        if self.chol is None:
            Q_A = Q_mat[np.ix_(self.active_indices, self.active_indices)]
            self.chol = np.linalg.cholesky(Q_A) # lower triangular

    def update(self,
               event_index):
        if event_index in self.active_indices:
            raise ValueError(f'feature {event_index} already in active set')
        self.active_indices.append(event_index)
        self.inactive_indices.remove(event_index)
        self.chol = np.linalg.cholesky(Q_mat[np.ix_(self.active_indices, self.active_indices)]) # lower triangular

    def downdate(self,
                 event_index):
        if event_index not in self.active_indices:
            raise ValueError(f'feature {event_index} not in active set')
        self.active_indices.remove(event_index)
        self.inactive_indices.append(event_index)
        if len(self.active_indices) > 0:
            self.chol = np.linalg.cholesky(Q_mat[np.ix_(self.active_indices, self.active_indices)]) # lower triangular
        else:
            self.chol = None

    def solve_active(self,
                     V):
        active_V = V[self.active_indices]
        return solve(self.chol.T, solve(self.chol, active_V, lower=True), lower=False)

    def next_event(self,
                   current_beta,
                   current_subgrad,
                   current_t,
                   dir_vec,
                   lambda_values):

        """Tracks the next event (variable entering or leaving the active
        set) as t increases.

        Parameters
        ----------
        beta_vec : np.ndarray, shape (n_features,)
            Vector beta.
        initial_vec : np.ndarray, shape (n_features,)
            Vector of sufficient statistics corresponding to nuisance parameters.
        dir_vec : np.ndarray, shape (n_features,)
            Vector v.
        Q_mat : np.ndarray, shape (n_features, n_features)
            Matrix Q.
        lambda_values : np.ndarray, shape (n_features,)
            Regularization parameter lambda.
        active_mask : np.ndarray, shape (n_features,), dtype=bool
            Boolean mask indicating the currently active features.
        active_signs : np.ndarray, shape (n_active,)
            Array of signs (+1 or -1) for active variables.
        current_beta : np.ndarray, shape (n_features,)
            Current solution vector.
        current_t : float
            Current value of t.
        active_chol : np.ndarray, shape (n_active, n_active) or None
            The current lower triangular Cholesky factor.

        Returns
        -------
        tuple: (next_event_t, event_type, event_index)
            - next_event_t (float): The value of t at the next event.
            - event_type (str or None): 'enter', 'leave', or None
                if no event found.
            - event_index (int): The index of the variable
                involved in the event (-1 if no event).
        """

        active_indices = self.active_indices
        inactive_indices = self.inactive_indices

        next_event_t = np.inf
        event_type = None
        event_index = -1

        # current_beta is the solution at current_t
        # current_subgrad is the negative gradient of quadratic at current_t, i.e. the subgradient at t

        # 1. Check for active variables hitting zero
        if len(active_indices) > 0:
            active_path = self.solve_active(dir_vec)
            for i, active_idx in enumerate(active_indices):
                if abs(active_path[i]) > 1e-9:
                    t_zero = -current_beta[active_idx] / active_path[i]
                    if t_zero > current_t + 1e-9 and t_zero < next_event_t:
                        next_event_t = t_zero
                        event_type = 'leave'
                        event_index = active_idx

        # 2. Check for inactive variables becoming active
        if len(inactive_indices) > 0:
            subgrad_inact = current_subgrad[inactive_indices]
            inact_path = dir_vec[inactive_indices]

            if len(active_indices) > 0:
                Q_IA = self.Q_mat[np.ix_(inactive_indices, active_indices)]
                inact_path -= Q_IA @ active_path

            for i, inactive_idx in enumerate(inactive_indices):
                t_zero_pos = (lambda_values[inactive_idx] - current_subgrad[inactive_idx]) / inact_path[i]
                t_zero_neg = (-lambda_values[inactive_idx] - current_subgrad[inactive_idx]) / inact_path[i]

                if t_zero_pos > current_t + 1e-9 and t_zero_pos < next_event_t:
                    next_event_t = t_zero_pos
                    event_type = 'enter'
                    event_index = inactive_idx

                if t_zero_neg > current_t + 1e-9 and t_zero_neg < next_event_t:
                    next_event_t = t_zero_neg
                    event_type = 'enter'
                    event_index = inactive_idx

        # update the beta and subgradient now that the time has been computed

        delta_t = next_event_t - current_t

        # 3. update the beta 
        if len(active_indices) > 0:
            for i, active_idx in enumerate(active_indices):
                current_beta[active_idx] += delta_t * active_path[i]

        # 4. update the subgrad
        if len(inactive_indices) > 0:
            for i, inactive_idx in enumerate(inactive_indices):
                current_subgrad[inactive_idx] += delta_t * inact_path[i]

        if event_type == 'enter':
            # for now this just straight up recomputes
            self.update(event_index)

        elif event_type == 'leave':
            # for now this just straight up recomputes
            self.downdate(event_index)

        return (next_event_t, event_type, event_index)

def homotopy_path(initial_soln,
                  S_vec,
                  dir_vec,
                  Q_mat,
                  lambda_val,
                  t_end=1.0):
    """Computes the homotopy path of the LASSO-like problem as t varies,
    using cholupdates for Cholesky factor updates and downdates.

    Parameters
    ----------
    initial_soln : np.ndarray, shape (n_features,)
        Vector b that should be solution to the problem at t_start.
    initial_vec : np.ndarray, shape (n_features,)
        Vector of sufficient statistics corresponding to nuisance parameters.
    dir_vec : np.ndarray, shape (n_features,)
        Vector v.
    Q_mat : np.ndarray, shape (n_features, n_features)
        Matrix Q.
    lambda_val : float
        Regularization parameter lambda.
    t_end : float, optional
        Ending value of t, by default 1.0.
    t_step : float, optional
        Step size for the homotopy parameter t (used for the outer loop).

    Returns
    -------
    list of tuple: (t, beta, active_mask)
        - t (float): The value of the homotopy parameter.
        - beta (np.ndarray, shape (n_features,)): The solution vector at t.
        - active_mask (np.ndarray, shape (n_features,), dtype=bool): Boolean mask of active features.
    """

    n_features = len(initial_soln)
    current_beta = initial_soln # presumed solution at 0
    current_subgrad = S_vec - Q_mat @ current_beta
    active_mask = np.fabs(current_beta) > 1e-9
    active_indices = list(np.where(active_mask)[0])
    inactive_indices = list(np.where(~active_mask)[0])

    active_chol = ActiveCholesky(active_indices=active_indices,
                                 inactive_indices=inactive_indices,
                                 Q_mat=Q_mat,
                                 chol=None)

    current_t = 0

    path = [(current_t, current_beta.copy(), active_mask.copy(), 'init')]

    while current_t < np.inf:

        (next_t,
         event_type,
         event_index) = active_chol.next_event(current_beta, # modified in place
                                               current_subgrad, # modified in place
                                               current_t,
                                               dir_vec,
                                               lambda_val)

        if next_t < np.inf:
            path.append((next_t, current_beta.copy(), active_mask.copy(), event_type))
        current_t = next_t

    return path

def solve_lasso_adelie(S_vec, dir_vec, Q_mat, lambda_val, t=0):
    """
    Solves the optimization problem using scikit-learn's Lasso class.

    Args:
        beta_vec (np.ndarray): Vector beta.
        nuisance_vec (np.ndarray): Vector gamma.
        dir_vec (np.ndarray): Vector v.
        Q_mat (np.ndarray): Matrix Q.
        lambda_val (float): Regularization parameter lambda.
        t (float): Homotopy parameter t.

    Returns:
        np.ndarray: The solution vector b.
    """

    S = gaussian_cov(A=Q_mat,
                     v=S_vec + t * dir_vec,
                     lmda_path=[lambda_val])

    return np.asarray(S.betas.todense()[-1]).reshape(-1)


if __name__ == '__main__':
    # Example usage (in low dimensions)
    n_features = 5
    rng = np.random.default_rng()
    S_vec = rng.standard_normal(n_features)
    dir_vec = rng.standard_normal(n_features) * 0.2
    W = rng.standard_normal((2 * n_features, n_features))
    Q_mat = W.T @ W
    lambda_val = 0.2 * np.ones(n_features)

    initial_soln = solve_lasso_adelie(S_vec,
                                      dir_vec,
                                      Q_mat,
                                      lambda_val)

    forward_path = homotopy_path(initial_soln,
                                 S_vec,
                                 dir_vec,
                                 Q_mat,
                                 lambda_val,
                                 t_end=1.0)
    backward_path = homotopy_path(initial_soln,
                                  S_vec,
                                  -dir_vec,
                                  Q_mat,
                                  lambda_val,
                                  t_end=1.0)[::-1]
    backward_path = [(-t, beta, active, event_type) for  t, beta, active, event_type in backward_path]
    path = backward_path + forward_path
    
    adelie_solns = [(t, solve_lasso_adelie(S_vec,
                                           dir_vec,
                                           Q_mat,
                                           lambda_val,
                                           t=t)) for t, _, _, _ in path]

#    print("Homotopy Path (t, beta, active_mask):")

    print(np.array([beta for _, beta, _, _ in path]))
    print(np.array([beta for _, beta in adelie_solns]))

    # for t, beta, active, event_type in path:
    #     print(f"t = {t:.4f}, beta = {beta}, active = {active}, event_type = {event_type}")
    # print(adelie_solns)


