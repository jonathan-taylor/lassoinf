"""
Matrix-free operators for the LASSO selection constraints.

With W = Q_{E,E}^{-1}, the constraints depend on x only through
W x_E and the inactive scores U_{-E}(x) = x_{-E} - Q_{-E,E} W x_E.
Both are applied with at most one Q matvec, so nothing of size
p x |E| is stored.
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator


def _embed(n, idx, vals):
    z = np.zeros(n)
    z[idx] = vals
    return z


class InactiveScoreOperator(LinearOperator):
    """
    U_{-E}(x) = x_{-E} - Q_{-E,E} W x_E, with W = Q_{E,E}^{-1}.
    """

    def __init__(self, Q, E, E_c, W):
        self.Q = Q
        self.E = np.asarray(E, dtype=int)
        self.E_c = np.asarray(E_c, dtype=int)
        self.W = W
        super().__init__(np.float64, (len(self.E_c), Q.shape[0]))

    def coef(self, x):
        """W x_E."""
        return self.W @ np.ravel(x)[self.E]

    def _matvec(self, x):
        x = np.ravel(x)
        if len(self.E) == 0:
            return x[self.E_c].copy()
        QWx = np.ravel(self.Q @ _embed(self.shape[1], self.E, self.coef(x)))
        return x[self.E_c] - QWx[self.E_c]

    def _rmatvec(self, y):
        y = np.ravel(y)
        out = np.zeros(self.shape[1])
        out[self.E_c] = y
        if len(self.E) > 0:
            Qy = np.ravel(self.Q @ _embed(self.shape[1], self.E_c, y))
            out[self.E] = -self.W.T @ Qy[self.E]
        return out


class LassoConstraintOperator(LinearOperator):
    """
    A x = [R_active @ (W x_E); R_inactive @ U_{-E}(x)]

    R_active (sparse, rows x |E|) holds the sign and bound rows of the active
    coordinates; R_inactive (sparse, rows x |E_c|) the subgradient rows of the
    inactive coordinates.
    """

    def __init__(self, Q, E, E_c, W, R_active, R_inactive):
        self.score = InactiveScoreOperator(Q, E, E_c, W)
        self.R_active = sp.csr_matrix(R_active)
        self.R_inactive = sp.csr_matrix(R_inactive)
        self.n_active = self.R_active.shape[0]
        super().__init__(np.float64,
                         (self.n_active + self.R_inactive.shape[0], Q.shape[0]))

    @property
    def W(self):
        return self.score.W

    @property
    def E(self):
        return self.score.E

    @property
    def E_c(self):
        return self.score.E_c

    def _matvec(self, x):
        x = np.ravel(x)
        parts = [self.R_active @ self.score.coef(x)]
        if self.R_inactive.shape[0] > 0:
            parts.append(self.R_inactive @ self.score.matvec(x))
        return np.concatenate(parts)

    def _rmatvec(self, y):
        y = np.ravel(y)
        y_active, y_inactive = y[:self.n_active], y[self.n_active:]
        if self.R_inactive.shape[0] > 0:
            out = self.score.rmatvec(self.R_inactive.T @ y_inactive)
        else:
            out = np.zeros(self.shape[1])
        out[self.E] += self.W.T @ (self.R_active.T @ y_active)
        return out

    def to_dense(self):
        # one matvec per column: for testing / small problems only
        return self.matmat(np.eye(self.shape[1]))
