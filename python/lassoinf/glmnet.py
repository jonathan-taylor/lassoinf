import numpy as np
import pandas as pd
from glmnet.glmnet import GLMNet, GLMState
from glmnet.glm import compute_grad
from lassoinf.operators.xtvx import XTVXOperator

def extract_glmnet_problem(glmnet_obj: GLMNet, X, Df, lambda_val, state=None, return_operator=False):
    """
    Extracts the actual solution, gradient, and Hessian of the LASSO problem
    solved by GLMNet, including penalty_factor, exclude, and upper/lower limits.
    The reported solutions are on the scaled coordinates if standardize=True.
    
    Returns a dictionary:
    {
       'beta_hat': array,    # Actual solution (scaled)
       'gradient': array,    # Gradient of the smooth loss (scaled)
       'hessian': array | LinearOperator, # Hessian of the smooth loss (scaled)
       'D': array,           # Penalty factors * lambda_val (scaled)
       'L': array,           # Lower bounds (scaled)
       'U': array            # Upper bounds (scaled)
    }
    """
    G = glmnet_obj
    if state is None:
        state = G.state_
        
    response_id = getattr(G, 'response_id', 'y')
    weight_id = getattr(G, 'weight_id', None)
    
    if isinstance(Df, pd.DataFrame):
        Y = Df[response_id] if response_id in Df.columns else Df.iloc[:, 0]
    else:
        Y = Df
        
    if weight_id and isinstance(Df, pd.DataFrame) and weight_id in Df.columns:
        W = Df[weight_id]
    else:
        W = np.ones(X.shape[0])
        
    scaled_state = G.design_.raw_to_scaled(state)
    
    scaled_grad, resid = compute_grad(G,
                                      state.intercept,
                                      state.coef,
                                      G.design_,
                                      Y,
                                      scaled_output=True,
                                      norm_weights=True)
                                      
    # Hessian on the scaled space
    # The smooth part loss is evaluated at the current state
    info = G._family.information(state, W / W.sum())
    
    if return_operator:
        Hessian = XTVXOperator(G.design_, info)
    else:
        Hessian = G.design_.quadratic_form(info, transformed=True)

    
    # Penalty factor
    p = G.design_.shape[1] - (1 if G.fit_intercept else 0)
    if G.penalty_factor is not None:
        penfac = np.array(G.penalty_factor, dtype=float)
    else:
        penfac = np.ones(p, dtype=float)
        
    # Exclude
    if hasattr(G, 'exclude') and G.exclude is not None:
        for idx in G.exclude:
            penfac[idx] = np.inf
            
    if G.fit_intercept:
        penfac = np.hstack([0.0, penfac])
        
    D = penfac * lambda_val
    
    # Limits
    def _expand_limits(lim, val, size):
        if lim is None:
            return np.full(size, val, dtype=float)
        if np.isscalar(lim):
            return np.full(size, lim, dtype=float)
        return np.array(lim, dtype=float)
        
    L_raw = _expand_limits(getattr(G, 'lower_limits', None), -np.inf, p)
    U_raw = _expand_limits(getattr(G, 'upper_limits', None), np.inf, p)
    
    # Transform limits to scaled space
    scaling = getattr(G.design_, 'scaling_', np.ones(p))
    
    L_scaled = np.where(L_raw == -np.inf, -np.inf, L_raw * scaling)
    U_scaled = np.where(U_raw == np.inf, np.inf, U_raw * scaling)
    
    if G.fit_intercept:
        L_scaled = np.hstack([-np.inf, L_scaled])
        U_scaled = np.hstack([np.inf, U_scaled])
        
    beta_hat = scaled_state.coef
    if G.fit_intercept:
        beta_hat = np.hstack([scaled_state.intercept, beta_hat])
        
    return {
        'beta_hat': beta_hat,
        'gradient': scaled_grad,
        'hessian': Hessian,
        'D': D,
        'L': L_scaled,
        'U': U_scaled
    }
