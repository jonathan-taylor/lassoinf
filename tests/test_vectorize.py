import numpy as np
import pytest

from lassoinf.selective_inference import SelectiveInference
import lassoinf_cpp

def test_get_weight_vectorized():
    """
    Test that the weight function correctly handles scalar and vector inputs
    and that Python and C++ implementations agree.
    """
    np.random.seed(0)
    Z = np.zeros(5)
    Z_noisy = np.zeros(5)
    Q = np.eye(5)
    Q_noise = np.eye(5)
    v = np.array([1, 0, 0, 0, 0])
    
    # A * x <= b constraint
    A = np.array([[1, 0, 0, 0, 0], [-1, 0, 0, 0, 0]])
    b = np.array([2.0, 2.0]) 

    # 1. Python implementation
    si_py = SelectiveInference(Z, Z_noisy, Q, Q_noise)
    wf_py = si_py.get_weight(v, A, b)
    
    t_scalar = 0.5
    t_vector = np.linspace(-1, 1, 5)
    
    py_scalar = wf_py(t_scalar)
    py_vector = wf_py(t_vector)
    
    # 2. C++ implementation
    si_cpp = lassoinf_cpp.SelectiveInference(Z, Z_noisy, Q, Q_noise)
    wf_cpp = si_cpp.get_weight(v, A, b)
    
    cpp_scalar = wf_cpp(t_scalar)
    cpp_vector = wf_cpp(t_vector)
    
    # Assertions
    # Scalar should be a float
    assert isinstance(py_scalar, float)
    assert isinstance(cpp_scalar, float)
    
    # Vector should be a numpy array
    assert isinstance(py_vector, np.ndarray)
    assert isinstance(cpp_vector, np.ndarray)
    
    # Dimensions match
    assert py_vector.shape == t_vector.shape
    assert cpp_vector.shape == t_vector.shape
    
    # Values match
    np.testing.assert_allclose(py_scalar, cpp_scalar, atol=1e-10)
    np.testing.assert_allclose(py_vector, cpp_vector, atol=1e-10)
    
    # They shouldn't be zero for this feasible region
    assert np.all(py_vector > 0.0)

def test_get_weight_random_data():
    """
    Test with random (potentially infeasible) regions.
    """
    np.random.seed(42)
    Z = np.random.randn(5)
    Z_noisy = Z + np.random.randn(5)*0.5
    Q = np.eye(5)
    Q_noise = np.eye(5)*0.25
    v = np.array([1, 0, 0, 0, 0])
    A = np.random.randn(3, 5)
    b = np.random.randn(3)

    si_py = SelectiveInference(Z, Z_noisy, Q, Q_noise)
    wf_py = si_py.get_weight(v, A, b)
    
    si_cpp = lassoinf_cpp.SelectiveInference(Z, Z_noisy, Q, Q_noise)
    wf_cpp = si_cpp.get_weight(v, A, b)
    
    t_vector = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    
    py_vector = wf_py(t_vector)
    cpp_vector = wf_cpp(t_vector)
    
    np.testing.assert_allclose(py_vector, cpp_vector, atol=1e-10)
