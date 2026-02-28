import numpy as np
import pytest
from lassoinf.selective_inference import LassoInference

# Define some interesting values for D, L, U
Ds = [0.0, 1.0, 5.0]
Ls = [-np.inf, -2.0, 0.0, 1.0]
Us = [np.inf, 2.0, 0.0, -1.0]

# All valid (D, L, U) combinations
all_combos = [(d, l, u) for d in Ds for l in Ls for u in Us if l <= u]

# Generate scenarios of size p=6. 
# We'll include some hand-picked and some random ones.
scenarios = []

# Hand-picked scenario 1: mixed
scenarios.append([
    (1.0, -np.inf, np.inf), # standard lasso
    (0.0, -np.inf, np.inf), # unpenalized
    (1.0, 0.0, np.inf),     # non-negative lasso
    (1.0, -np.inf, 0.0),    # non-positive lasso
    (1.0, -1.0, 1.0),      # bounded lasso
    (2.0, 0.5, 0.5)        # fixed at 0.5
])

# Hand-picked scenario 2: all unpenalized but bounded
scenarios.append([
    (0.0, 0.0, 1.0),
    (0.0, -1.0, 0.0),
    (0.0, -1.0, 1.0),
    (0.0, 0.0, 0.0),
    (0.0, -np.inf, 3.0),
    (0.0, -3.0, np.inf)
])

# Random scenarios
np.random.seed(42)
for _ in range(18):
    indices = np.random.choice(len(all_combos), 6)
    scenarios.append([all_combos[i] for i in indices])

@pytest.mark.parametrize("scenario", scenarios)
@pytest.mark.parametrize("q_type", ["identity", "random_pd"])
def test_prox_map_extensive(scenario, q_type):
    p = 6
    D = np.array([s[0] for s in scenario])
    L = np.array([s[1] for s in scenario])
    U = np.array([s[2] for s in scenario])
    
    # Random seed based on scenario to ensure reproducibility
    # but still varied enough
    np.random.seed(hash(tuple(scenario)) % 2**32)
    
    if q_type == "identity":
        Q_hat = np.eye(p)
    else:
        # Use a poorly conditioned matrix too
        A = np.random.randn(p, p)
        Q_hat = A.T @ A + 1e-3 * np.eye(p)
        
    Sigma = np.eye(p)
    Sigma_noisy = np.eye(p)
    Z_full = np.random.randn(p)
    
    # Starting point for beta
    beta_start = np.random.randn(p) * 5.0
    
    # Initial G_hat - unpenalized gradient
    G_hat = np.random.randn(p) * 2.0
    
    inf = LassoInference(
        beta_hat=beta_start,
        G_hat=G_hat,
        Q_hat=Q_hat,
        D=D,
        L=L,
        U=U,
        Z_full=Z_full,
        Sigma=Sigma,
        Sigma_noisy=Sigma_noisy
    )
    
    # 1. Check KKT
    # We use a slightly looser tolerance for poorly conditioned matrices
    tol = 1e-4 if q_type == "identity" else 1e-3
    assert inf.check_kkt(tol=tol), f"KKT Failed for scenario {scenario} and q_type {q_type}"
    
    # 2. Check that beta_hat is feasible
    # We use a loose tolerance because check_kkt already verified the logic
    assert np.all(inf.beta_hat >= L - 1e-6)
    assert np.all(inf.beta_hat <= U + 1e-6)
    
    # 3. Check that the SelectiveInference object can actually compute an interval
    # (even if it's NaN for some reason, it shouldn't crash)
    if len(inf.E) > 0:
        idx = inf.E[0]
        assert idx in inf.intervals
        lower, upper, p_val = inf.intervals[idx]
        if not np.isnan(lower):
            assert lower <= upper
            assert 0 <= p_val <= 1
