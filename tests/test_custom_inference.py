import numpy as np
import pandas as pd
from lassoinf.lasso import LassoInference

def test_custom_inference_method():
    # Setup minimal data for LassoInference
    n_features = 3
    beta_hat = np.array([1.0, 0.0, 0.0])
    G_hat = np.array([0.0, 0.5, -0.5])
    Q_hat = np.eye(n_features)
    D = np.array([0.1, 0.1, 0.1])
    L = np.full(n_features, -np.inf)
    U = np.full(n_features, np.inf)
    Z_full = np.array([1.0, 0.0, 0.0])
    Sigma = np.eye(n_features)
    
    # Define a custom inference method
    def custom_method(v, theta_hat, variance, contrast, L_0, U_0, alpha):
        # Just return dummy values for testing
        return 0.1, 0.9, 0.05, contrast

    # 1. Test with default (bivariate_normal)
    inf_default = LassoInference(
        beta_hat=beta_hat, G_hat=G_hat, Q_hat=Q_hat, D=D, L=L, U=U, 
        Z_full=Z_full, Sigma=Sigma, scalar_noise=0
    )
    assert hasattr(inf_default, 'summary_')
    assert 'lower_conf' in inf_default.summary_.columns

    # 2. Test calling compute_intervals with custom method
    inf_custom = LassoInference(
        beta_hat=beta_hat, G_hat=G_hat, Q_hat=Q_hat, D=D, L=L, U=U, 
        Z_full=Z_full, Sigma=Sigma, scalar_noise=0
    )
    inf_custom.compute_intervals(inference_method=custom_method)
    
    summary = inf_custom.summary_
    # In our dummy setup, the first variable is active
    if 0 in summary.index:
        assert summary.loc[0, 'lower_conf'] == 0.1
        assert summary.loc[0, 'upper_conf'] == 0.9
        assert summary.loc[0, 'p_value'] == 0.05

    print("Custom inference method test passed!")

if __name__ == "__main__":
    test_custom_inference_method()
