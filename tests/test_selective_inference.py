import numpy as np
import pytest

from lassoinf.selective_inference import SelectiveInference
import lassoinf_cpp

def test_selective_inference_consistency():
    """
    Test that the Python and C++ implementations of SelectiveInference
    produce perfectly consistent outputs across all core methods.
    """
    np.random.seed(42)
    Z = np.random.randn(5)
    Z_noisy = Z + np.random.randn(5) * 0.5
    Q = np.eye(5) + 0.1 * np.random.randn(5, 5)
    Q = Q.T @ Q  # ensure positive definite
    Q_noise = np.eye(5) * 0.25 + 0.05 * np.random.randn(5, 5)
    Q_noise = Q_noise.T @ Q_noise

    si_py = SelectiveInference(Z, Z_noisy, Q, Q_noise)
    si_cpp = lassoinf_cpp.SelectiveInference(Z, Z_noisy, Q, Q_noise)

    v = np.array([1.0, -0.5, 0.2, 0.0, 0.0])

    # 1. Test compute_params
    params_py = si_py.compute_params(v)
    params_cpp = si_cpp.compute_params(v)

    np.testing.assert_allclose(params_py['gamma'], params_cpp.gamma, atol=1e-10)
    np.testing.assert_allclose(params_py['c'], params_cpp.c, atol=1e-10)
    np.testing.assert_allclose(params_py['bar_gamma'], params_cpp.bar_gamma, atol=1e-10)
    np.testing.assert_allclose(params_py['bar_s'], params_cpp.bar_s, atol=1e-10)
    np.testing.assert_allclose(params_py['n_o'], params_cpp.n_o, atol=1e-10)
    np.testing.assert_allclose(params_py['bar_n_o'], params_cpp.bar_n_o, atol=1e-10)
    np.testing.assert_allclose(params_py['theta_hat'], params_cpp.theta_hat, atol=1e-10)
    np.testing.assert_allclose(params_py['bar_theta'], params_cpp.bar_theta, atol=1e-10)

    # 2. Test data_splitting_estimator
    dse_py = si_py.data_splitting_estimator(v)
    dse_cpp = si_cpp.data_splitting_estimator(v)
    np.testing.assert_allclose(dse_py, dse_cpp, atol=1e-10)

    # 3. Test get_interval
    A = np.random.randn(4, 5)
    b = np.random.randn(4)
    t = 1.23

    # get_interval can return (np.nan, np.nan) for infeasible regions, let's test a few t values
    for t_val in [-5.0, 0.0, 1.23, 10.0]:
        interval_py = si_py.get_interval(v, t_val, A, b)
        interval_cpp = si_cpp.get_interval(v, t_val, A, b)
        
        # Check if both return NaN
        if np.isnan(interval_py[0]):
            assert np.isnan(interval_cpp[0])
            assert np.isnan(interval_cpp[1])
        else:
            np.testing.assert_allclose(interval_py, interval_cpp, atol=1e-10)

def test_selective_inference_infeasible_interval():
    """
    Test explicitly that infeasible intervals are handled similarly.
    """
    Z = np.zeros(2)
    Z_noisy = np.zeros(2)
    Q = np.eye(2)
    Q_noise = np.eye(2)
    v = np.array([1.0, 0.0])
    
    # An impossible set of constraints (A * x <= b)
    A = np.array([[1.0, 0.0], [-1.0, 0.0]])
    b = np.array([-1.0, -1.0])  # x1 <= -1 AND -x1 <= -1  => x1 <= -1 AND x1 >= 1 => Infeasible
    
    si_py = SelectiveInference(Z, Z_noisy, Q, Q_noise)
    si_cpp = lassoinf_cpp.SelectiveInference(Z, Z_noisy, Q, Q_noise)
    
    interval_py = si_py.get_interval(v, 0.0, A, b)
    interval_cpp = si_cpp.get_interval(v, 0.0, A, b)
    
    assert np.isnan(interval_py[0]) and np.isnan(interval_py[1])
    assert np.isnan(interval_cpp[0]) and np.isnan(interval_cpp[1])
