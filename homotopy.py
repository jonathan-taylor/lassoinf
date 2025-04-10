"""
Solves a homotopy continuation problem for a modified LASSO-like optimization problem.

This code implements a path-following algorithm to trace the solution of the
following optimization problem as a function of the homotopy parameter 't':

$$
\textrm{minimize}_{b}  b' @ (initial_soln + t * dir_vec) + 0.5 * b' @ Q_mat @ b + ||lambda_val * b||_1
$$

where:

-   $b$: vector of variables being optimized.
-   $initial_soln$: vector corresponding to the initial solution at t=0.
-   $dir_vec$: the direction of the homotopy.
-   $Q_mat$: A known positive semi-definite matrix.
-   $lambda_val$: A fixed regularization parameter (lambda >= 0). Can be a scalar or vector.
-   $t$: The homotopy parameter.

The algorithm tracks the solution b(t) as 't' varies, efficiently updating the
Cholesky decomposition of relevant submatrices to improve performance.
It handles events where variables enter or leave the active set.

The implementation uses the 'cholupdates' library for Cholesky updates
and downdates, improving numerical stability and efficiency compared to
recomputing the Cholesky decomposition at each step.

Homotopy problem

We want to construct a path of solutions

$$
\textrm{minimize}_{b}  b' @ (initial_soln + t * dir_vec) + 0.5 * b' @ Q_mat @ b + ||lambda_val * b||_1
$$

with $lambda_val$ a set of penalty factors (possibly 0).

Leaving time calculation

Suppose at $t_c$ we know the solution is $b_c = \hat{\beta}(t_c)$ with signs and active set
$A_c=A(t_c), s_c=s_A(t_c)$. Then, for $t > t_c$ the proposed solution is

$$
b_c + (t - t_c) * Q_{A_c}^{-1} @ dir_vec_{A_c}
$$

The active conditions say that to keep the active set fixed we must have
$$
s_{A_c} * (b_c + (t - t_c) * Q_{A_c}^{-1} @ dir_vec_{A_c}) >= 0
$$

The zero time for variable $j$ satisfies

$$
\begin{aligned}
s_{A_c,j} * (b_{c,j} + (t_j - t_c) * (Q_{A_c}^{-1} @ dir_vec_{A_c})_j) = 0
  & \iff b_{c,j} - (t_j - t_c) * (Q_{A_c}^{-1} @ dir_vec_{A_c})_j = 0 \\
t_{j,d} = t_c - b_{c,j} / (Q_{A_c}^{-1} @ dir_vec_{A_c})_j.
\end{aligned}
$$
We're only interested in $t_{j,d} > t_c$ because we are going forward in time.

Therefore, the next deletion time is

$$
t_d = min(t_{j,d}: t_{j,d} > t_c)
$$

Adding time

At time $t_c$, with inactive set $I(t_c)=I_c$ we know that the inactive subgradient's value is
the negative of the gradient of the quadratic part at $b_c$:

$$
U_c = initial_soln_{I_c} - Q_{I_c,A_c} @ b_c
$$

$$
||lambda_val_{I_c}^{-1} * U_c||_{inf} <= 1
$$

Moving along the solution path as $beta_A$ the subgradient is therefore

$$
U_c + (t - t_c) *(dir_vec_{I_c,j} - Q_{I_c,A_c} @ Q_{A_c}^{-1} @ dir_vec_{A_c})
$$

Setting the coordinates to $\pm lambda_j$ we get

$$
t_{j,a}^+ = t_c + (lambda_val_j - U_{c,j}) / ( dir_vec_{I_c,j} - (Q_{I_c,A_c} @ Q_{A_c}^{-1} @ dir_vec_{A_c})_j )
$$

and

$$
t_{j,a}^- = t_c + (-lambda_val_j - U_{c,j}) / (dir_vec_{I_c,j} - (Q_{I_c,A_c} @ Q_{A_c}^{-1} @ dir_vec_{A_c})_j)
$$

so

$$
t_{j,a} = min(t_{j,a}^+-: t_{j,a}^+- > t_c, t_{j,a}^--: t_{j,a}^-- > t_c)
$$

and the next added variable will be

$$
t_a = min(t_j: t_j > t_c)
$$

In either case, the $beta$ solution at the next time

$$
t_n = min(t_a, t_d)
$$

we'll have our terminal $beta$ value for the $A_c$ coordinates

$$
beta_c + (t_n - t_c) @ Q_{A_c}^{-1} @ dir_vec_{A_c}
$$

The $I_c$ coordinates at $t_n$ for $beta$ will also be 0. We'll similarly have our terminal
subgradient

$$
U_c + (t_n - t_c) @ Q_{I_c,A_c} @ Q_{A_c}^{-1} @ dir_vec_{A_c}
$$

"""

import numpy as np
from scipy.linalg import solve
from dataclasses import dataclass
from adelie.solver import gaussian_cov

@dataclass
class HomotopyState(object):

    beta: np.ndarray
    subgrad: np.ndarray
    t: float

    def __copy__(self):
        return self.__class__(beta=self.beta.copy(),
                              subgrad=self.subgrad.copy(),
                              t=self.t)

    def __eq__(self, other):
        val = ((self.t == other.t) and
               np.allclose(self.beta, other.beta) and
               np.allclose(self.subgrad, other.subgrad))
        if not val:
            stop
        return val

@dataclass
class HomotopyPath(object):
    """
    Manages the Cholesky decomposition of the active set's submatrix.

    Efficiently updates and downdates the Cholesky factorization as the
    active set changes.
    """

    Q_mat: np.ndarray
    direction: np.ndarray
    active_indices: list
    inactive_indices: list
    active_signs: np.ndarray
    chol: np.ndarray  # lower cholesky
    lambda_values: np.ndarray
    sufficient_stat: np.ndarray
    initial_soln: np.ndarray | None
    initial_t: float = 0    

    def __post_init__(self):
        """Initializes the Cholesky decomposition."""
        if self.chol is None:
            Q_A = self.Q_mat[np.ix_(self.active_indices, self.active_indices)]
            self.chol = np.linalg.cholesky(Q_A)  # lower triangular
            self._last_event = (None, None, None)
        
        S = self.sufficient_stat

        self._adelie_solver = gaussian_cov(A=self.Q_mat,
                                           v=S, lmda_path=[1],
                                           penalty=self.lambda_values)

        if self.initial_soln is None:
            beta = self._adelie_solver.solve().betas[-1]
        else:
            beta = self.initial_soln.copy()

        _state = HomotopyState(beta=beta,
                               subgrad=S - self.Q_mat @ beta,
                               t=self.initial_t)

        if not _check_kkt(self,
                          _state):
            warn('initial KKT check not passing, refitting with adelie')
            _state.beta = self._adelie_solver.solve().betas[-1]
            _state.subgrad = S - self.Q_mat @ _state.beta
        
        self._state = _state

    def update(self, event_index: int, sign: int):
        """
        Updates the Cholesky decomposition when a variable enters the active set.

        Parameters
        ----------
        event_index : int
            The index of the variable entering the active set.

        Raises
        ------
        ValueError
            If the variable is already in the active set.
        """
        if event_index in self.active_indices:
            raise ValueError(f"feature {event_index} already in active set")
        self.active_indices.append(event_index)
        self.active_signs[event_index] = sign
        self.inactive_indices.remove(event_index)
        self.chol = np.linalg.cholesky(
            self.Q_mat[np.ix_(self.active_indices, self.active_indices)]
        )  # lower triangular

    def downdate(self, event_index: int):
        """
        Downdates the Cholesky decomposition when a variable leaves the active set.

        Parameters
        ----------
        event_index : int
            The index of the variable leaving the active set.

        Raises
        ------
        ValueError
            If the variable is not in the active set.
        """
        if event_index not in self.active_indices:
            raise ValueError(f"feature {event_index} not in active set")
        self.active_indices.remove(event_index)
        self.active_signs[event_index] = 0
        self.inactive_indices.append(event_index)
        if len(self.active_indices) > 0:
            self.chol = np.linalg.cholesky(
                self.Q_mat[np.ix_(self.active_indices, self.active_indices)]
            )  # lower triangular
        else:
            self.chol = None

    def solve_active(self) -> np.ndarray:
        """
        Solves the linear system involving the active set's submatrix.

        Parameters
        ----------
        dir_vec : np.ndarray
            The direction vector.

        Returns
        -------
        np.ndarray
            The solution to the linear system.
        """
        dir_vec = self.direction
        active_dir_vec = dir_vec[self.active_indices].copy()
        unconstrained =  solve(
            self.chol.T, solve(self.chol, active_dir_vec, lower=True), lower=False
        )
        if (self._last_event[0] == 'enter' and unconstrained[-1] * self.active_signs[self.active_indices[-1]] < 0):
            self._flag = True

        return unconstrained
    
    def next_event(self) -> tuple[float, str, int]:
        """
        Tracks the next event (variable entering or leaving the active set) as t increases.

        Parameters
        ----------
        current_beta : np.ndarray, shape (n_features,)
            Current solution vector.
        current_subgrad : np.ndarray, shape (n_features,)
            Current subgradient vector.
        current_t : float
            Current value of t.
        dir_vec : np.ndarray, shape (n_features,)
            Direction vector.
        lambda_values : np.ndarray, shape (n_features,)
            Regularization parameter lambda.

        Returns
        -------
        tuple: (next_event_t, event_type, event_index)
            - next_event_t (float): The value of t at the next event.
            - event_type (str or None): 'enter', 'leave', or None if no event found.
            - event_index (int): The index of the variable involved in the event (-1 if no event).
        """

        state = self._state
        dir_vec = self.direction

        (current_beta,
         current_subgrad,
         current_t) = (state.beta,
                       state.subgrad,
                       state.t)

        active_indices = self.active_indices
        inactive_indices = self.inactive_indices
        lambda_values = self.lambda_values

        next_event_t = np.inf
        event_type = None
        event_index = -1

        # current_beta is the solution at current_t
        # current_subgrad is the negative gradient of quadratic at current_t, i.e. the subgradient at t

        # 1. Check for active variables hitting zero (leaving event)
        if len(active_indices) > 0:
            active_path = self.solve_active()

            for i, active_idx in enumerate(active_indices):
                if abs(active_path[i]) > 1e-9:
                    t_zero = current_t - current_beta[active_idx] / active_path[i]
                    if t_zero > current_t + 1e-9 and t_zero < next_event_t:
                        next_event_t = t_zero
                        event_type = "leave"
                        event_index = active_idx
                        event_sign = np.sign(current_beta[active_idx])
                        
        # 2. Check for inactive variables becoming active (entering event)
        if len(inactive_indices) > 0:
            subgrad_inact = current_subgrad[inactive_indices]
            inact_path = dir_vec[inactive_indices]

            if len(active_indices) > 0:
                Q_IA = self.Q_mat[np.ix_(inactive_indices, active_indices)]
                inact_path -= Q_IA @ self.solve_active()

            for i, inactive_idx in enumerate(inactive_indices):
                t_zero_pos = current_t + (lambda_values[inactive_idx] - current_subgrad[inactive_idx]) / inact_path[i]
                t_zero_neg = current_t + (-lambda_values[inactive_idx] - current_subgrad[inactive_idx]) / inact_path[i]

                if t_zero_pos > current_t + 1e-9 and t_zero_pos < next_event_t:
                    if self._last_event != ('leave', inactive_idx, +1):
                        next_event_t = t_zero_pos
                        event_type = "enter"
                        event_sign = 1
                        event_index = inactive_idx

                if t_zero_neg > current_t + 1e-9 and t_zero_neg < next_event_t:
                    if self._last_event != ('leave', inactive_idx, -1):
                        next_event_t = t_zero_neg
                        event_type = "enter"
                        event_sign = -1
                        event_index = inactive_idx
                        
        # 3. update the beta and subgradient now that the time has been computed
        delta_t = next_event_t - current_t

        if len(active_indices) > 0:
            for i, active_idx in enumerate(active_indices):
                current_beta[active_idx] += delta_t * active_path[i]

        if event_type == 'leave':
            self._last_event = ('leave', event_index, event_sign)

        if len(inactive_indices) > 0:
            for i, inactive_idx in enumerate(inactive_indices):
                current_subgrad[inactive_idx] += delta_t * inact_path[i]

        if event_type == 'enter':
            self._last_event = ('enter', event_index, event_sign)

        # 4. update the active chol
        if event_type == "enter":
            self.update(event_index, event_sign)

        elif event_type == "leave":
            self.downdate(event_index)

        next_state = HomotopyState(beta=current_beta,
                                   subgrad=current_subgrad,
                                   t=next_event_t)
        
        self._state.t = next_event_t
        print(_check_kkt(self,
                         self._state))
        self._state = next_state 

        return (next_state, event_type, event_index)


def _check_kkt(hpath: HomotopyPath,
               state: HomotopyState):
    dir_vec = hpath.direction
    S = hpath.sufficient_stat + state.t * dir_vec
    G = S - hpath.Q_mat @ state.beta
    active_indices = hpath.active_indices
    inactive_indices = hpath.inactive_indices

    if hpath.lambda_values[active_indices].sum() > 0:
        # check the signs
        active_signs = hpath.active_signs[active_indices]
        active_subgrad = state.subgrad[active_indices]
        active_check = np.all(state.beta[active_indices] * active_signs >= -1e-3)
        active_lambda = hpath.lambda_values[active_indices]
        # check the active gradients have close to correct values
        active_check = active_check & (np.linalg.norm(active_subgrad * active_signs
                                                      + active_lambda) /
                                       np.linalg.norm(active_lambda) < 1e-3)
    else:
        active_check = True
        
    if len(inactive_indices) > 0:
        inactive_subgrad = state.subgrad[inactive_indices]
        inactive_lambda = hpath.lambda_values[inactive_indices]
        inactive_check = np.fabs(inactive_subgrad / inactive_lambda).max() <= 1 + 1e-3
        inactive_check = inactive_check & (np.linalg.norm(G[inactive_indices] - inactive_subgrad) / np.linalg.norm(G[inactive_indices]) < 1e-3)
    else:
        inactive_check = True
        
    return {'active': active_check, 'inactive':inactive_check, 'G':G, 'subgrad':state.subgrad}

def homotopy_path(
        initial_soln: np.ndarray,
        sufficient_stat: np.ndarray,
        direction: np.ndarray,
        Q_mat: np.ndarray,
        lambda_values: np.ndarray,
        initial_t: float=0,
        ) -> list[tuple[float, np.ndarray, np.ndarray, str]]:
    """
    Computes the homotopy path of the LASSO-like problem as t varies.

    Parameters
    ----------
    initial_soln : np.ndarray, shape (n_features,)
        Vector b that should be solution to the problem at initial_t
    sufficient_stat : np.ndarray, shape (n_features,)
        Vector of sufficient statistics
    initial_t:
        Initial value of target for initial_soln
    direction : np.ndarray, shape (n_features,)
        Vector v, the direction of the homotopy.
    Q_mat : np.ndarray, shape (n_features, n_features)
        Matrix Q.
    lambda_val : np.ndarray
        Regularization parameter lambda.

    Returns
    -------
    list of tuple: (t, beta, active_mask, event_type)
        - t (float): The value of the homotopy parameter.
        - beta (np.ndarray, shape (n_features,)): The solution vector at t.
        - active_mask (np.ndarray, shape (n_features,), dtype=bool): Boolean mask of active features.
        - event_type (str): Type of event ('init', 'enter', 'leave').
    """

    n_features = len(initial_soln)
    current_beta = initial_soln.copy()  # presumed solution at 0
    current_subgrad = sufficient_stat - Q_mat @ current_beta
    active_mask = np.fabs(current_beta) > 1e-9
    active_indices = list(np.where(active_mask)[0])
    inactive_indices = list(np.where(~active_mask)[0])

    active_signs = np.sign(current_beta)
    active_signs[inactive_indices] = 0

    hpath = HomotopyPath(
        sufficient_stat=sufficient_stat,
        active_indices=active_indices,
        inactive_indices=inactive_indices,
        active_signs=active_signs,
        Q_mat=Q_mat,
        direction=direction,
        chol=None,
        lambda_values=lambda_values,
        initial_soln=initial_soln,
    )

    current_t = 0

    path = [(current_t, current_beta.copy(), active_mask.copy(), ("init", None))]

    while current_t < np.inf:
        (state,
         event_type,
         event_index) = hpath.next_event()

        active_mask[:] = 0
        active_mask[hpath.active_indices] = 1
        if state.t < np.inf:
            path.append((state.t, state.beta.copy(), active_mask.copy(), (event_type, event_index)))
        current_t = state.t

    print(len(hpath.active_indices), hpath.Q_mat.shape)
    return path, hpath


def solve_lasso_adelie(
    sufficient_stat: np.ndarray,
    direction: np.ndarray,
    Q_mat: np.ndarray,
    lambda_val: np.ndarray,
    t: float = 0,
) -> np.ndarray:
    """
    Solves the optimization problem using adelie.

    Args:
        sufficient_stat (np.ndarray): Vector of sufficient statistics.
        direction (np.ndarray): Vector defining the homotopy direction.
        Q_mat (np.ndarray): Matrix Q.
        lambda_val (np.ndarray): Regularization parameter lambda.
        t (float, optional): Homotopy parameter t. Defaults to 0.

    Returns:
        np.ndarray: The solution vector b.
    """

    S = gaussian_cov(A=Q_mat, v=sufficient_stat + t * direction, lmda_path=[1],
                     penalty=lambda_val)

    return np.asarray(S.betas.todense()[-1]).reshape(-1)



if __name__ == "__main__":
    # Example usage (in low dimensions)
    n_features = 30
    rng = np.random.default_rng()
    sufficient_stat = rng.standard_normal(n_features)
    direction = rng.standard_normal(n_features) * 0.2
    W = []
    W = [rng.standard_normal(2 * n_features)]
    for i in range(n_features - 1):
        W.append(0.7 * W[-1] + rng.standard_normal(2 * n_features))
    W = np.array(W)
    Q_mat = W @ W.T / n_features
    lambda_val = 1.2 * np.ones(n_features)

    initial_soln = solve_lasso_adelie(sufficient_stat, direction, Q_mat, lambda_val)
    active_set = np.where(np.fabs(initial_soln) > 0)[0]
    
    initial_soln = initial_soln[active_set]
    sufficient_stat = sufficient_stat[active_set]
    direction = direction[active_set]
    Q_mat = Q_mat[np.ix_(active_set, active_set)]
    lambda_val = lambda_val[active_set]
    forward_path, hpath = homotopy_path(
        initial_soln, sufficient_stat, direction,
        Q_mat, lambda_val)

    backward_path, hpath = homotopy_path(
        initial_soln, sufficient_stat, -direction, Q_mat, lambda_val
    )
    backward_path = backward_path[::-1]
    backward_path = [
        (-t, beta, active, event_type) for t, beta, active, event_type in backward_path
    ]
    path = backward_path + forward_path
    #path = forward_path
    
    adelie_solns = [
        (
            t,
            solve_lasso_adelie(sufficient_stat, direction, Q_mat, lambda_val, t=t),
        )
        for t, _, _, _ in path
    ]

    H = np.array([beta for _, beta, _, _ in path])
    A = np.array([beta for _, beta in adelie_solns])


    print(np.linalg.norm(A - H) / max(np.linalg.norm(A), 1), np.linalg.norm(A))
    if hasattr(hpath, "_flag"):
        print('flagged a problem')
    

