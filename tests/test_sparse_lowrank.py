import numpy as np
import scipy.sparse as sp
import pytest
from lassoinf.operators.composite import CompositeOperator
from lassoinf.selective_inference import SelectiveInference

def test_composite_operator_dense_equivalence():
    n = 10
    np.random.seed(42)
    
    # Generate random components
    S_dense = np.random.randn(n, n)
    S_dense = (S_dense + S_dense.T) / 2  # Make symmetric
    S_sparse = sp.csr_matrix(S_dense)
    
    rank = 3
    U = np.random.randn(n, rank)
    V = np.random.randn(n, rank)
    
    b = np.random.randn(n)
    
    # Build full dense matrix conceptually: S + U @ V.T + diag(b)
    Q_dense = S_dense + U @ V.T + np.diag(b)
    
    # Build operator
    Q_op = CompositeOperator((n, n), S=S_sparse, U=U, V=V, b=b)
    
    # 1. Test to_dense() matches
    np.testing.assert_allclose(Q_op.to_dense(), Q_dense)
    
    # 2. Test matvec
    x = np.random.randn(n)
    y_dense = Q_dense @ x
    y_op = Q_op @ x
    np.testing.assert_allclose(y_op, y_dense)

def test_selective_inference_solve_contrast_with_operator():
    n = 15
    np.random.seed(42)
    
    # Generate some well-conditioned symmetric positive definite matrices
    A = np.random.randn(n, n)
    Q_dense = A @ A.T + np.eye(n) * 0.1
    
    B = np.random.randn(n, n)
    Q_noise_dense = B @ B.T + np.eye(n) * 0.1
    
    # We'll treat Q_noise as an operator, built entirely from the dense representation (S=dense, others 0)
    Q_noise_op = CompositeOperator((n, n), S=sp.csr_matrix(Q_noise_dense))
    
    # Data vectors
    Z = np.random.randn(n)
    Z_noisy = Z + np.random.randn(n)
    v = np.random.randn(n)
    
    # Dense Inference
    si_dense = SelectiveInference(Z, Z_noisy, Q_dense, Q_noise_dense)
    c_dense = si_dense.solve_contrast(v)
    
    # Operator Inference
    si_op = SelectiveInference(Z, Z_noisy, Q_dense, Q_noise_op)
    c_op = si_op.solve_contrast(v)
    
    # c = Q_noise^-1 * Q * v
    # Both should yield the same vector 'c'
    np.testing.assert_allclose(c_op, c_dense, rtol=1e-5, atol=1e-5)

def test_selective_inference_full_params_with_operator():
    n = 15
    np.random.seed(42)
    
    # Diagonal + Low Rank structure
    b_diag = np.random.uniform(1.0, 2.0, n)
    U = np.random.randn(n, 2)
    
    Q_op = CompositeOperator((n, n), b=b_diag, U=U, V=U) # S is 0 implicitly
    Q_noise_op = CompositeOperator((n, n), b=b_diag * 0.5, U=U * 0.7, V=U * 0.7)
    
    Q_dense = Q_op.to_dense()
    Q_noise_dense = Q_noise_op.to_dense()
    
    Z = np.random.randn(n)
    Z_noisy = Z + np.random.randn(n)
    v = np.ones(n)
    
    si_dense = SelectiveInference(Z, Z_noisy, Q_dense, Q_noise_dense)
    si_op = SelectiveInference(Z, Z_noisy, Q_op, Q_noise_op)
    
    params_dense = si_dense.compute_params(v)
    params_op = si_op.compute_params(v)
    
    np.testing.assert_allclose(params_op['c'], params_dense['c'], rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(params_op['bar_s'], params_dense['bar_s'], rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(params_op['bar_theta'], params_dense['bar_theta'], rtol=1e-5, atol=1e-5)