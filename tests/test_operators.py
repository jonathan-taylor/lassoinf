import numpy as np
import scipy.sparse as sp
import pytest

from lassoinf.operators.composite import CompositeOperator
from lassoinf.operators.xtvx import XTVXOperator
import lassoinf_cpp

def test_composite_operator_adjoint():
    np.random.seed(42)
    m, n = 10, 8
    
    # Python CompositeOperator
    S_mat = sp.random(m, n, density=0.2, random_state=42).tocsr()
    U_mat = np.random.randn(m, 3)
    V_mat = np.random.randn(n, 3)
    
    op_py = CompositeOperator((m, n), S=S_mat, U=U_mat, V=V_mat)
    op_dense = op_py.to_dense()
    
    # Test _matvec
    x = np.random.randn(n)
    np.testing.assert_allclose(op_py @ x, op_dense @ x, atol=1e-10)
    
    # Test _rmatvec
    y = np.random.randn(m)
    np.testing.assert_allclose(op_py.T @ y, op_dense.T @ y, atol=1e-10)
    
    # C++ CompositeOperator
    comp_cpp = lassoinf_cpp.CompositeComponent(S_mat, U_mat, V_mat, np.array([]))
    op_cpp = lassoinf_cpp.CompositeOperator(m, n, [comp_cpp])
    
    np.testing.assert_allclose(op_cpp.multiply(x), op_dense @ x, atol=1e-10)
    np.testing.assert_allclose(op_cpp.multiply_transpose(y), op_dense.T @ y, atol=1e-10)


def test_xtvx_operator_adjoint():
    np.random.seed(43)
    n_samples, n_features = 20, 5
    
    X = np.random.randn(n_samples, n_features)
    
    # V will be a diagonal matrix represented as a CompositeOperator
    b_diag = np.random.uniform(1.0, 2.0, n_samples)
    V_py = CompositeOperator((n_samples, n_samples), b=b_diag)
    
    # Python XTVXOperator
    xtvx_py = XTVXOperator(X, V_py)
    
    dense_V = np.diag(b_diag)
    dense_xtvx = X.T @ dense_V @ X
    
    x = np.random.randn(n_features)
    np.testing.assert_allclose(xtvx_py @ x, dense_xtvx @ x, atol=1e-10)
    
    # _rmatvec should equal _matvec since X^T V X is symmetric when V is symmetric
    y = np.random.randn(n_features)
    np.testing.assert_allclose(xtvx_py.T @ y, dense_xtvx.T @ y, atol=1e-10)
    
    # C++ XTVXOperator
    S_empty = sp.csr_matrix((n_samples, n_samples))
    U_empty = np.zeros((n_samples, 0))
    V_empty = np.zeros((n_samples, 0))
    
    comp_cpp = lassoinf_cpp.CompositeComponent(S_empty, U_empty, V_empty, b_diag)
    V_cpp = lassoinf_cpp.CompositeOperator(n_samples, n_samples, [comp_cpp])
    
    xtvx_cpp = lassoinf_cpp.XTVXOperator(X, V_cpp)
    
    np.testing.assert_allclose(xtvx_cpp.multiply(x), dense_xtvx @ x, atol=1e-10)
    np.testing.assert_allclose(xtvx_cpp.multiply_transpose(y), dense_xtvx.T @ y, atol=1e-10)
