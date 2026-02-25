import numpy as np
import pytest

from lassoinf.gaussian_family import WeightedGaussianFamily, TruncatedGaussian
from lassoinf.selective_inference import SelectiveInference
import lassoinf_cpp

def test_truncated_gaussian():
    estimate = 1.73
    sigma = 1.0
    lower_bound = 2.0
    upper_bound = 10.0
    smoothing_sigma = 0.5
    noisy_estimate = estimate

    # Python implementation
    tg_py = TruncatedGaussian(estimate, sigma, smoothing_sigma, lower_bound, upper_bound, noisy_estimate)
    
    # C++ implementation
    tg_cpp = lassoinf_cpp.TruncatedGaussian(estimate, sigma, smoothing_sigma, lower_bound, upper_bound, noisy_estimate)

    t_scalar = 0.5
    t_vector = np.linspace(-1, 5, 10)

    # Check scalar
    assert np.isclose(tg_py.weight(t_scalar), tg_cpp.weight(t_scalar))

    # Check vector
    py_w = tg_py.weight(t_vector)
    cpp_w = tg_cpp.weight(t_vector)
    
    np.testing.assert_allclose(py_w, cpp_w)


def test_weighted_gaussian_family():
    estimate = 1.73
    sigma = 1.0
    lower_bound = 2.0
    upper_bound = np.inf
    smoothing_sigma = 0.5

    tg_py = TruncatedGaussian(estimate, sigma, smoothing_sigma, lower_bound, upper_bound, estimate)
    tg_cpp = lassoinf_cpp.TruncatedGaussian(estimate, sigma, smoothing_sigma, lower_bound, upper_bound, estimate)

    wgf_py = WeightedGaussianFamily(estimate, sigma, [tg_py.weight], seed=0)
    wgf_cpp = lassoinf_cpp.WeightedGaussianFamily(estimate, sigma, [tg_cpp.weight], 10.0, 4000)

    # test pvalue
    pval_py = wgf_py.pvalue()
    pval_cpp = wgf_cpp.pvalue()
    
    # Python uses discrete_family for its interval, let's see if they match C++
    np.testing.assert_allclose(pval_py, pval_cpp, atol=1e-5)

    # test interval
    int_py = wgf_py.interval()
    int_cpp = wgf_cpp.interval()
    
    np.testing.assert_allclose(int_py, int_cpp, atol=1e-5)

def test_file_drawer_with_gaussian_family():
    mu_null = 0
    gamma = 0.5
    threshold = 2.0
    z_obs = 1.73

    Z = np.array([z_obs])
    Q = np.eye(1)
    Q_noise = np.array([[gamma**2]])
    Z_noisy = Z.copy() 

    si = SelectiveInference(Z, Z_noisy, Q, Q_noise)
    v = np.array([1.0])
    A = np.array([[-1.0]])
    b = np.array([[-threshold]])

    weight_f_py = si.get_weight(v, A, b)
    
    si_cpp = lassoinf_cpp.SelectiveInference(Z, Z_noisy, Q, Q_noise)
    weight_f_cpp = si_cpp.get_weight(v, A, b)

    wgf_py = WeightedGaussianFamily(z_obs, 1.0, [weight_f_py], seed=0)
    wgf_cpp = lassoinf_cpp.WeightedGaussianFamily(z_obs, 1.0, [weight_f_cpp], 10.0, 4000)

    pval_py = wgf_py.pvalue()
    pval_cpp = wgf_cpp.pvalue()
    np.testing.assert_allclose(pval_py, pval_cpp, atol=1e-5)

    int_py = wgf_py.interval()
    int_cpp = wgf_cpp.interval()
    np.testing.assert_allclose(int_py, int_cpp, atol=1e-5)
