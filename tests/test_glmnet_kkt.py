import numpy as np
import pandas as pd
from glmnet.glmnet import GLMNet
from lassoinf.selective_inference import spec_from_glmnet, LassoInference

def test_kkt_conditions():
    # Generate synthetic data
    n_samples, n_features = 100, 20
    X = np.random.randn(n_samples, n_features)
    true_coef = np.zeros(n_features)
    true_coef[:5] = np.random.randn(5)
    y = X @ true_coef + np.random.randn(n_samples)

    # Fit GLMNet model
    glmnet = GLMNet(lambda_values=[0.1])
    df = pd.DataFrame(X)
    df['y'] = y
    glmnet.fit(df.drop('y', axis=1), df['y'])
    
    state = glmnet.state_
    lambda_val = glmnet.lambda_val

    # Get spec from glmnet
    spec = spec_from_glmnet(glmnet,
                            (df.drop('y', axis=1), df),
                            lambda_val,
                            state,
                            proportion=1.0)

    assert spec is not None, "spec_from_glmnet should not return None"

    # Create a LassoInference instance to check KKT
    # We need to construct the arguments for LassoInference
    # For the purpose of testing check_kkt, some can be placeholders
    
    Q_hat = spec['hessian']
    beta_hat = state.coef
    
    # G_hat = -(gradient + Q_hat @ beta_hat)
    G_hat = - (spec['gradient'] + Q_hat @ beta_hat)

    lasso_inf = LassoInference(beta_hat=beta_hat,
                               G_hat=G_hat,
                               Q_hat=Q_hat,
                               D=np.ones(n_features) * lambda_val,
                               L=np.full(n_features, -np.inf),
                               U=np.full(n_features, np.inf),
                               Z_full=np.zeros(n_features), # Placeholder
                               Sigma=np.eye(n_features),      # Placeholder
                               Sigma_noisy=np.eye(n_features) # Placeholder
                               )
    
    assert lasso_inf.check_kkt(), "KKT conditions should be satisfied"
