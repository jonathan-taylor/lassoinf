import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator

class CompositeOperator(LinearOperator):
    """
    A linear operator representing a symmetric matrix Q defined as:
    Q = S + U @ V^T + diag(b)
    where:
    - S is a sparse symmetric matrix
    - U, V are dense matrices representing a low-rank component (U @ V^T)
    - b is a vector representing a diagonal component
    """
    def __init__(self, shape, S=None, U=None, V=None, b=None):
        self.shape = shape
        self.dtype = np.dtype(np.float64)
        
        n = shape[0]
        self.S = S if S is not None else sp.csr_matrix((n, n), dtype=np.float64)
        self.U = U if U is not None else np.zeros((n, 0))
        self.V = V if V is not None else np.zeros((n, 0))
        self.b = b if b is not None else np.zeros(n)
        
        # Validation
        if self.S.shape != shape:
            raise ValueError(f"S shape {self.S.shape} must match {shape}")
        if self.U.shape[0] != n or self.V.shape[0] != n:
            raise ValueError("U and V must have n rows")
        if self.U.shape[1] != self.V.shape[1]:
            raise ValueError("U and V must have the same number of columns (rank)")
        if self.b.shape != (n,):
            raise ValueError(f"b shape {self.b.shape} must be ({n},)")

    def _matvec(self, x):
        """
        Compute Q @ x = S @ x + U @ (V^T @ x) + b * x
        """
        res = self.S @ x
        
        if self.U.shape[1] > 0:
            res += self.U @ (self.V.T @ x)
            
        res += self.b * x
        return res
        
    def to_dense(self):
        """
        Materialize the full dense matrix (for testing/debugging).
        """
        n = self.shape[0]
        dense = self.S.toarray()
        if self.U.shape[1] > 0:
            dense += self.U @ self.V.T
        dense += np.diag(self.b)
        return dense
