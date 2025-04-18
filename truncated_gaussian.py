import bisect
import numpy as np
import mpmath as mp
from scipy.stats import norm as normal_dbn
from dataclasses import dataclass

@dataclass
class truncated_gaussian(object):

    left: np.ndarray
    right: np.ndarray
    weights: np.ndarray | None
    mu: float=0
    scale: float=1


    """
    A class representing a Gaussian distribution, truncated to specified intervals.

    Attributes
    ----------
    left : np.ndarray
        Array of left endpoints of the truncation intervals.
    right : np.ndarray
        Array of right endpoints of the truncation intervals.
    weights : np.ndarray | None
        Array of weights for each interval.
        If None, defaults to an array of ones.
    mu : float, default=0
        Mean of the underlying Gaussian distribution.
    scale : float, default=1
        Standard deviation (scale) of the underlying Gaussian distribution.
    """

    def __post_init__(self):
        """
        Initializes the truncated Gaussian object.

        Performs sorting of intervals and checks for validity (non-overlapping, non-empty).
        """

        if self.weights is None:
            self.weights = np.ones(len(self.right), float)
            
        # sort by the left end point
        order_left = np.argsort(self.left)

        # this will copy, probably not a bad idea
        self.left = np.asarray(self.left)[order_left]
        self.right = np.asarray(self.right)[order_left]
        self.weights = np.asarray(self.weights)[order_left]

        # check that self.right[:,:-1] < self.left[:,1:] to
        # verify non-overlapping
        if np.any(self.right == self.left):
            raise ValueError('empty interval: left - right should be > 0')

        overlap_test = np.greater.outer(self.left[1:], self.right[:-1])
        overlap_checks = [np.all(overlap_test[k,:k]) for k in range(overlap_test.shape[0])]
        if not np.all(overlap_checks):
            raise ValueError('intervals should be non-overlapping')
        
        self._compute_P()
    
    def set_mu(self, mu):
        '''
        Set value of mu and recalculate some private quantities.

        Arguments
        ---------

        mu: float

        Notes
        -----

        When setting mu or scale, some additional calculations must be done.
        This will not happen my simply setting the corresponding attribute's value.
        '''

        self.mu = mu
        self._compute_P()
        
    def set_scale(self, scale):
        '''
        Set value of scale and recalculate some private quantities.

        Arguments
        ---------

        mu: float

        Notes
        -----

        When setting mu or scale, some additional calculations must be done.
        This will not happen my simply setting the corresponding attribute's value.
        '''

        self.scale = scale
        self._compute_P()

    def __repr__(self):
        """
        Returns a string representation of the truncated_gaussian object.

        Returns
        -------
        str
            A string representation of the object.
        """
        return f'''{self.__class__.__name__}(left={self.left},
                                         right={self.right},
                                         weights={self.weights},
                                         mu={self.mu:0.3e},
                                         scale={self.scale:0.3e})'''
    
    def cdf(self, observed):
        """
        Computes the cumulative distribution function (CDF) at a given point.

        Parameters
        ----------
        observed : float
            The point at which to evaluate the CDF.

        Returns
        -------
        float
            The CDF value at the given point.

        Raises
        ------
        ValueError
            If the observed value is not within the support of the density.
        """
        # self._P will have weights applied to them by the previous
        # call to self._mu_or_scale_changed()
        
        P, mu, scale = self._P, self.mu, self.scale
        z = observed
        idx = find_interval_sorted(z, self.left, self.right)
        if idx is not None:
            if observed > self.left[idx]:
                val = (P[:idx].sum() + 
                       (self.weights[idx] * norm_interval((self.left[idx] - mu) / scale,
                                                        (observed - mu) / scale))
                        ) / P.sum()
            else:
                val = P[:idx].sum() / P.sum()
            return val
        else:
            raise ValueError('observed is not within support of this density')

    def equal_tailed_interval(self,
                              observed,
                              level=0.9):
        """
        Computes the equal-tailed interval for a given observed value and level.

        Parameters
        ----------
        observed : float
            The observed value.
        level : float
            The desired level for the interval.

        Returns
        -------
        np.ndarray
            An array containing the lower and upper bounds of the equal-tailed interval.
        """

        old_mu = self.mu
        lb = self.mu - 20. * self.scale
        ub = self.mu + 20. * self.scale
        def F(param):
            self.set_mu(param)
            return self.cdf(observed)

        alpha = 1 - level
        L = find_root(F, 1.0 - 0.5 * alpha, lb, ub)
        U = find_root(F, 0.5 * alpha, lb, ub)
        self.set_mu(old_mu)
        return np.array([L, U])

            
    # private method to update P after a change of parameters

    def _compute_P(self):
        """
        Updates the precomputed interval probabilities (_P) after a change in mu or scale.
        """
        mu, scale = self.mu, self.scale
        self._P = np.array([norm_interval((a-mu)/scale,
                                         (b-mu)/scale) 
                           for a, b in zip(self.left, self.right)])
        self._P *= self.weights

def find_root(f,
              y,
              lb,
              ub,
              tol=1e-6):
    """
    Searches for a solution to f(x) = y in (lb, ub), where 
    f is a monotone decreasing function

    Parameters
    ----------
    f : callable
        The monotone function.
    y : float
        The target value.
    lb : float
        The lower bound of the search interval.
    ub : float
        The upper bound of the search interval.
    tol : float
        The desired tolerance for the solution.

    Returns
    -------
    float
        An approximation of the root of the equation f(x) = y.
    """       
    
    # make sure solution is in range
    a, b   = lb, ub
    fa, fb = f(a), f(b)
    
    # assume a < b
    if fa > y and fb > y:
        while fb > y : 
            b, fb = b + (b-a), f(b + (b-a))
    elif fa < y and fb < y:
        while fa < y : 
            a, fa = a - (b-a), f(a - (b-a))
    
    # determine the necessary number of iterations
    max_iter = int( np.ceil( ( np.log(tol) - np.log(b-a) ) / np.log(0.5) ) )

    # bisect (slow but sure) until solution is obtained
    for _ in range(max_iter):
        c, fc  = (a+b)/2, f((a+b)/2)
        if fc > y: a = c
        elif fc < y: b = c
    
    return c

def norm_interval(lower, upper):
    """
    Evaluates the difference of the standard normal CDF between two limits.

    Computes Phi(upper) - Phi(lower), where Phi is the standard normal CDF.

    Parameters
    ----------
    lower : float
        The lower limit.
    upper : float
        The upper limit.

    Returns
    -------
    float
        The difference of the standard normal CDF.
    """
    if lower > 0 and upper > 0:
        return mp.ncdf(-lower) - mp.ncdf(-upper)
    else:
        return mp.ncdf(upper) - mp.ncdf(lower)

def find_interval_sorted(point, left_endpoints, right_endpoints):
    """
    Finds the interval containing a given point from disjoint intervals
    that are sorted by their left endpoints, using binary search.

    Parameters
    ----------
    point : float
        The point to search for.
    left_endpoints : list of float
        A sorted list of the left endpoints of the disjoint intervals.
    right_endpoints : list of float
        The corresponding list of the right endpoints.

    Returns
    -------
    int or None
        The index of the interval containing the point,
        or None if the point is not within any of the intervals.
    """
    # Find the index of the first left endpoint that is greater than the point
    right_index = bisect.bisect_right(left_endpoints, point)

    # If right_index is 0, the point is smaller than all left endpoints
    if right_index == 0:
        return None

    # Check the interval just before the found index
    potential_index = right_index - 1
    if left_endpoints[potential_index] <= point <= right_endpoints[potential_index]:
        return potential_index
    else:
        return None
