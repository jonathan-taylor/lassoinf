import numpy as np
import pytest

from lassoinf.discrete_family import discrete_family
import lassoinf_cpp

def test_discrete_family():
    np.random.seed(42)
    grid = np.linspace(-5, 5, 200)
    weights = np.exp(-0.5 * grid**2)
    
    # Normalize weights so they represent a valid probability distribution base measure
    weights /= weights.sum()

    df_py = discrete_family(grid, weights)
    df_cpp = lassoinf_cpp.DiscreteFamilyCPP(grid.tolist(), weights.tolist(), 0.0)

    # test cdf and ccdf at theta = 0
    x = 1.0
    cdf_py = df_py.cdf(0.0, x, gamma=1.0)
    cdf_cpp = df_cpp.cdf(0.0, x, gamma=1.0)
    np.testing.assert_allclose(cdf_py, cdf_cpp, atol=1e-5)

    ccdf_py = df_py.ccdf(0.0, x, gamma=0.0)
    ccdf_cpp = df_cpp.ccdf(0.0, x, gamma=0.0)
    np.testing.assert_allclose(ccdf_py, ccdf_cpp, atol=1e-5)

    # test cdf and ccdf at theta = 1.5
    theta = 1.5
    cdf_py_t = df_py.cdf(theta, x, gamma=1.0)
    cdf_cpp_t = df_cpp.cdf(theta, x, gamma=1.0)
    np.testing.assert_allclose(cdf_py_t, cdf_cpp_t, atol=1e-5)

    ccdf_py_t = df_py.ccdf(theta, x, gamma=0.0)
    ccdf_cpp_t = df_cpp.ccdf(theta, x, gamma=0.0)
    np.testing.assert_allclose(ccdf_py_t, ccdf_cpp_t, atol=1e-5)

    # test equal_tailed_interval
    interval_py = df_py.equal_tailed_interval(x, alpha=0.1)
    interval_cpp = df_cpp.equal_tailed_interval(x, alpha=0.1)
    np.testing.assert_allclose(interval_py, interval_cpp, atol=1e-5)
