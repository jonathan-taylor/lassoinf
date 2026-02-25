import numpy as np
import pytest

from lassoinf.selective_inference import SelectiveInference
from lassoinf.discrete_family import discrete_family
from scipy.stats import norm

def test_file_drawer():
    """
    Test based on docs/file_drawer.md to verify root-finding in equal_tails_interval
    and behavior of the selective inference weight function in 1D.
    """
    mu_null = 0
    gamma = 0.5
    threshold = 2.0
    z_obs = 1.73

    # SelectiveInference expects arrays
    Z = np.array([z_obs])
    Q = np.eye(1)
    Q_noise = np.array([[gamma**2]])

    # For the weight function calculation, the specific value of omega doesn't change 
    # the probability P(Z + omega > threshold | Z=t) analytically but we need Z_noisy to instantiate.
    Z_noisy = Z.copy() 

    si = SelectiveInference(Z, Z_noisy, Q, Q_noise)

    # Define target and constraints
    v = np.array([1.0])  # Target is Z itself
    A = np.array([[-1.0]])  # -(Z + omega) <= -threshold  => Z + omega >= threshold
    b = np.array([[-threshold]])

    # Get the weight function
    weight_f = si.get_weight(v, A, b)
    
    # Verify weight function evaluations
    t_grid = np.linspace(0, 4, 10)
    weights = weight_f(t_grid)
    assert len(weights) == len(t_grid)
    assert np.all(weights >= 0) and np.all(weights <= 1.0)
    
    # Compute the discrete family distribution
    grid = np.linspace(-10, 10, 500)
    
    # Reference measure is N(mu_null, 1) * weight_f(t)
    reference_pdf = norm.pdf(grid, loc=mu_null, scale=1.0)
    sel_weights = weight_f(grid)
    reference_weights = reference_pdf * sel_weights
    
    # Instantiate discrete family
    fam = discrete_family(grid, reference_weights)
    
    # Check that equal_tails_interval doesn't raise an exception
    # (i.e. root finding converges).
    try:
        lower, upper = fam.equal_tailed_interval(z_obs, alpha=0.05)
        # Should be a valid interval
        assert lower <= upper
    except Exception as e:
        pytest.fail(f"Root finding in equal_tails_interval failed: {e}")
