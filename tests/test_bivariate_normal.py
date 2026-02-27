import numpy as np
import pytest
from scipy.stats import norm
from lassoinf.bivariate_normal import TruncBivariateNormal
from lassoinf.discrete_family import discrete_family

def test_trunc_bivariate_normal_cdf():
    # Parameters for the constraint: L <= a*Z + b*omega <= U
    # Z ~ N(theta, sig_x^2), omega ~ N(0, sig_omega^2)
    a, b = 1.0, 1.2
    L, U = 1.0, 5.0
    sig_omega = 1.0
    sig_x = 1.5
    theta = 2.0
    
    tbn = TruncBivariateNormal(a, b, L, U, sig_omega, sig_x=sig_x, theta=theta)
    
    # Check CDF at some point x
    x = 2.5
    cdf_val = tbn.cdf(theta, x)
    assert 0 <= cdf_val <= 1
    
    # Check CCDF
    ccdf_val = tbn.ccdf(theta, x)
    np.testing.assert_allclose(cdf_val + ccdf_val, 1.0)

def test_trunc_bivariate_normal_moments():
    a, b = 1.0, 0.0 # No noise case, pure truncated normal
    L, U = -1.0, 1.0
    sig_omega = 1.0
    sig_x = 1.0
    theta = 0.0
    
    tbn = TruncBivariateNormal(a, b, L, U, sig_omega, sig_x=sig_x, theta=theta)
    
    # For a=1, b=0, theta=0, sig_x=1, we have Z ~ N(0, 1) | -1 <= Z <= 1
    # The mean should be 0 by symmetry
    mean = tbn.E(theta, lambda x: x)
    np.testing.assert_allclose(mean, 0.0, atol=1e-7)
    
    # The variance of truncated normal N(0, 1) on [-1, 1] is:
    # 1 - [phi(1)*1 - phi(-1)*(-1)]/[Phi(1) - Phi(-1)]
    # = 1 - 2*phi(1) / (Phi(1) - Phi(-1))
    expected_var = 1 - 2 * norm.pdf(1) / (norm.cdf(1) - norm.cdf(-1))
    var = tbn.Var(theta, lambda x: x)
    np.testing.assert_allclose(var, expected_var, atol=1e-7)

def test_trunc_bivariate_normal_vs_discrete():
    # Large sig_omega to make it smooth
    a, b = 1.0, 1.0
    L, U = 0.0, 2.0
    sig_omega = 2.0
    sig_x = 1.0
    theta = 0.5
    
    tbn = TruncBivariateNormal(a, b, L, U, sig_omega, sig_x=sig_x, theta=theta)
    
    # Create a dense discrete approximation
    z_grid = np.linspace(-5, 5, 1000)
    # Weight for discrete family: P(L <= a*z + b*omega <= U) * p(z)
    # a*z + b*omega ~ N(a*z, (b*sig_omega)^2)
    s_std = np.abs(b * sig_omega)
    weights = (norm.cdf((U - a * z_grid) / s_std) - norm.cdf((L - a * z_grid) / s_std)) * norm.pdf(z_grid, loc=0, scale=sig_x)
    df = discrete_family(z_grid, weights)
    
    x_test = 0.7
    # Use theta=0 for discrete family because we already included p(z) in weights (which is N(0, sig_x^2))
    # Actually discrete_family.cdf(theta) multiplies weights by exp(theta * x)
    # Our tbn.cdf(theta) means Z ~ N(theta, sig_x^2)
    # So if we use df with theta=theta, it should match if we correctly set the base weights.
    # Base weights should be the weight function P(L <= a*z + b*omega <= U) * exp(-0.5 * z^2 / sig_x^2)
    
    cdf_tbn = tbn.cdf(theta, x_test)
    cdf_df = df.cdf(theta / (sig_x**2), x_test) # discrete_family uses exp(theta * x)
    
    np.testing.assert_allclose(cdf_tbn, cdf_df, atol=1e-3)
