import numpy as np
import pylops

def extract_submatrices(Q, E, E_c=None):
    """
    Extracts Q_{E, E} and optionally Q_{E_c, E} from an operator Q (or numpy array)
    using only matrix-vector products, which is efficient when Q is an implicit operator
    and |E| is small.
    
    Args:
        Q: A dense matrix or a LinearOperator.
        E: Array of indices defining the columns and rows for Q_EE.
        E_c: (Optional) Array of indices defining the rows for Q_EcE.
        
    Returns:
        Q_EE if E_c is None, else (Q_EE, Q_EcE).
    """
    if not hasattr(Q, 'shape'):
        raise ValueError("Operator Q must have a 'shape' attribute")
        
    n = Q.shape[0]
    
    if isinstance(Q, np.ndarray):
        Q_E = Q[:, E]
        Q_EE = Q_E[E, :]
        if E_c is not None:
            Q_EcE = Q_E[E_c, :]
            return Q_EE, Q_EcE
        return Q_EE
        
    R_E = pylops.Restriction(n, E)
    Q_EE = np.zeros((len(E), len(E)))
    
    if E_c is not None:
        R_Ec = pylops.Restriction(n, E_c)
        Q_EcE = np.zeros((len(E_c), len(E)))
        
    for i in range(len(E)):
        v = np.zeros(n)
        v[E[i]] = 1.0
        Q_v = Q.matvec(v) if hasattr(Q, 'matvec') else Q @ v
        Q_EE[:, i] = R_E.matvec(Q_v)
        if E_c is not None:
            Q_EcE[:, i] = R_Ec.matvec(Q_v)
            
    if E_c is not None:
        return Q_EE, Q_EcE
    return Q_EE
