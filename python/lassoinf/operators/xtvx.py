import numpy as np
from scipy.sparse.linalg import LinearOperator

class XTVXOperator(LinearOperator):
    """
    A linear operator representing Q = X^T V X, 
    where X is a dense/sparse matrix (tall) and V is a LinearOperator.
    """
    def __init__(self, X, V):
        self.X = X
        self.V = V
        n = X.shape[1]
        self.shape = (n, n)
        self.dtype = np.dtype(np.float64)

    def _matvec(self, x):
        return self.X.T @ (self.V @ (self.X @ x))
        
    def _rmatvec(self, x):
        # (X.T V X)^T = X^T V^T X
        return self.X.T @ (self.V.T @ (self.X @ x))
        
    def _matmat(self, X_mat):
        return self.X.T @ (self.V @ (self.X @ X_mat))
        
    def _rmatmat(self, X_mat):
        return self.X.T @ (self.V.T @ (self.X @ X_mat))
