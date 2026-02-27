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
    sig_x = 1.2
    theta = 0.5
    
    tbn = TruncBivariateNormal(a, b, L, U, sig_omega, sig_x=sig_x, theta=theta)
    
    # Create a dense discrete approximation
    z_grid = np.linspace(-5, 5, 2000)
    # Base weight function: P(L <= a*z + b*omega <= U) * exp(-0.5 * z^2 / sig_x^2)
    s_std = np.abs(b * sig_omega)
    weights = (norm.cdf((U - a * z_grid) / s_std) - norm.cdf((L - a * z_grid) / s_std)) * norm.pdf(z_grid, loc=0, scale=sig_x)
    df = discrete_family(z_grid, weights)
    
    x_test = 0.7
    # Both now use natural parameterization
    cdf_tbn = tbn.cdf(theta, x_test)
    cdf_df = df.cdf(theta, x_test)
    
    np.testing.assert_allclose(cdf_tbn, cdf_df, atol=1e-3)

def test_trunc_bivariate_normal_mle():
    a, b = 1.0, 0.5
    L, U = 0.0, 3.0
    sig_omega = 1.0
    sig_x = 1.0
    theta_true = 0.5
    
    tbn = TruncBivariateNormal(a, b, L, U, sig_omega, sig_x=sig_x, theta=theta_true)
    
    # True mean under theta_true
    mu_true = tbn.E(theta_true, lambda x: x)
    
    # If we observe exactly the mean, MLE should be theta_true
    mle_est, _, _ = tbn.MLE(mu_true, initial=theta_true)
    np.testing.assert_allclose(mle_est, theta_true, atol=1e-5)
    
    # Test with a different observation
    obs = 1.2
    mle_est, std_err, _ = tbn.MLE(obs)
    # Check if E[X | mle_est] is close to obs
    np.testing.assert_allclose(tbn.E(mle_est, lambda x: x), obs, atol=1e-5)

def test_trunc_bivariate_normal_coverage_mean():
    rng = np.random.default_rng(42)
    
    a_coeff, b_coeff = 1.0, 0.5
    L, U = 0.0, 3.0
    sig_omega = 1.0
    sig_x = 2.0 # sig_x != 1 to distinguish mean from natural parameter
    
    mu_true = 1.5
    theta_true = mu_true / sig_x**2
    
    tbn = TruncBivariateNormal(a_coeff, b_coeff, L, U, sig_omega, sig_x=sig_x, theta=theta_true)
    
    # Weight function for WeightedGaussianFamily
    def tbn_weight(x):
        s_std = np.abs(b_coeff * sig_omega)
        return norm.cdf((U - a_coeff * x) / s_std) - norm.cdf((L - a_coeff * x) / s_std)
    
    from lassoinf.gaussian_family import WeightedGaussianFamily
    
    n_sim = 100
    alpha = 0.1
    coverage_tbn = 0
    coverage_wgf = 0
    
    # Rejection sampling to simulate from the exact distribution
    samples = []
    while len(samples) < n_sim:
        z = rng.normal(loc=mu_true, scale=sig_x, size=1000)
        omega = rng.normal(loc=0.0, scale=sig_omega, size=1000)
        s = a_coeff * z + b_coeff * omega
        
        valid_z = z[(s >= L) & (s <= U)]
        samples.extend(valid_z)
    
    samples = samples[:n_sim]
    
    for x_obs in samples:
        # TBN interval (returns natural parameter bounds)
        lower_theta, upper_theta = tbn.equal_tailed_interval(x_obs, alpha=alpha)
        
        # Convert to mean space
        lower_mean_tbn = lower_theta * sig_x**2
        upper_mean_tbn = upper_theta * sig_x**2
        
        if lower_mean_tbn <= mu_true <= upper_mean_tbn:
            coverage_tbn += 1
            
        # WGF interval (returns mean bounds directly)
        wgf = WeightedGaussianFamily(estimate=x_obs, sigma=sig_x, weight_fns=[tbn_weight])
        lower_mean_wgf, upper_mean_wgf = wgf.interval(level=1 - alpha)
        
        if lower_mean_wgf <= mu_true <= upper_mean_wgf:
            coverage_wgf += 1
            
        # The intervals should be quite similar
        np.testing.assert_allclose([lower_mean_tbn, upper_mean_tbn], 
                                   [lower_mean_wgf, upper_mean_wgf], rtol=0.05, atol=0.5)
            
    cov_rate_tbn = coverage_tbn / n_sim
    cov_rate_wgf = coverage_wgf / n_sim
    
    assert 0.85 <= cov_rate_tbn <= 0.95, f"TBN Coverage {cov_rate_tbn} is outside bounds"
    assert 0.85 <= cov_rate_wgf <= 0.95, f"WGF Coverage {cov_rate_wgf} is outside bounds"
