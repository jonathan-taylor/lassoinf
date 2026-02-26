import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator

class CompositeOperator(LinearOperator):
    """
    A linear operator representing a matrix defined as a sum of components:
    Q = sum_i (S_i + U_i @ V_i^T + diag(b_i))
    """
    def __init__(self, shape, S=None, U=None, V=None, b=None, components=None):
        self.shape = shape
        self.dtype = np.dtype(np.float64)
        self.components = list(components) if components is not None else []
        
        m, n = shape
        if S is not None or U is not None or V is not None or b is not None:
            self.components.append((S, U, V, b))

        for S_i, U_i, V_i, b_i in self.components:
            if S_i is not None and S_i.shape != shape:
                raise ValueError(f"S shape {S_i.shape} must match {shape}")
            if U_i is not None and U_i.shape[0] != m:
                raise ValueError(f"U must have {m} rows")
            if V_i is not None and V_i.shape[0] != n:
                raise ValueError(f"V must have {n} rows")
            if U_i is not None and V_i is not None and U_i.shape[1] != V_i.shape[1]:
                raise ValueError("U and V must have the same number of columns (rank)")
            if b_i is not None:
                if m != n:
                    raise ValueError("b (diagonal) only supported for square operators")
                if b_i.shape != (n,):
                    raise ValueError(f"b shape {b_i.shape} must be ({n},)")

    def _matvec(self, x):
        res = np.zeros(self.shape[0])
        for S_i, U_i, V_i, b_i in self.components:
            if S_i is not None:
                res += S_i @ x
            if U_i is not None and V_i is not None and U_i.shape[1] > 0:
                res += U_i @ (V_i.T @ x)
            if b_i is not None:
                res += b_i * x
        return res
        
    def _rmatvec(self, x):
        res = np.zeros(self.shape[1])
        for S_i, U_i, V_i, b_i in self.components:
            if S_i is not None:
                res += S_i.T @ x
            if U_i is not None and V_i is not None and U_i.shape[1] > 0:
                res += V_i @ (U_i.T @ x)
            if b_i is not None:
                res += b_i * x
        return res
        
    def to_dense(self):
        m, n = self.shape
        dense = np.zeros((m, n))
        for S_i, U_i, V_i, b_i in self.components:
            if S_i is not None:
                if sp.issparse(S_i):
                    dense += S_i.toarray()
                else:
                    dense += S_i
            if U_i is not None and V_i is not None and U_i.shape[1] > 0:
                dense += U_i @ V_i.T
            if b_i is not None:
                dense += np.diag(b_i)
        return dense