---
jupytext:
  main_language: python
  cell_metadata_filter: -all
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.1
  formats: ipynb,md:myst
kernelspec:
  name: python3
  display_name: Python 3 (ipykernel)
  language: python
---

# Discrete Family

The `discrete_family` module provides tools for working with 1-dimensional discrete exponential families. These are often used for exact (post-selection) hypothesis tests and confidence intervals.

## Example: Selective Inference with Discrete Family

Here is an example evaluating a normal density on a grid and using a selection weight from `SelectiveInference`.

```{code-cell} ipython3
import numpy as np
from scipy.stats import norm
from lassoinf.selective_inference import SelectiveInference
from lassoinf.discrete_family import discrete_family

# 1. Setup a SelectiveInference problem
np.random.seed(42)
n = 10
Z = np.random.randn(n)
Q = np.eye(n)
gamma_val = 0.5
Q_noise = (gamma_val**2) * Q
omega = np.random.multivariate_normal(np.zeros(n), Q_noise)
Z_noisy = np.fabs(Z + omega)

si = SelectiveInference(Z, Z_noisy, Q, Q_noise)

# Define contrast eta (v)
v = np.zeros(n)
v[0] = 1.0  # target is Z[0]

# Define constraints
A = -np.eye(n)
b = np.zeros(n)

# 2. Get the selection weight function for the target
weight_f = si.get_weight(v, A, b)

# 3. Create a grid of sufficient statistics (e.g. possible values of theta_hat)
grid = np.linspace(-15, 15, 1000)

# 4. The reference measure is the unselected normal distribution for theta_hat
# In this case, theta_hat ~ N(0, v'Qv) -> N(0, 1) under the null
# We multiply the density by the selection weight to get the selective density
reference_weights = norm.pdf(grid, loc=0, scale=1) * np.array([weight_f(t) for t in grid])
reference_weights /= reference_weights.sum()

# 5. Initialize the discrete exponential family
family = discrete_family(grid, reference_weights)

# 6. Perform Exact Inference
observed = 1.5

# Test the hypothesis H0: theta = 0 given the observed value
p_val_rejects = family.two_sided_test(theta0=0.0, observed=observed, alpha=0.05)
print(f"Rejects H0: {p_val_rejects}")

# Compute a 95% Equal-Tailed Confidence Interval
lower, upper = family.equal_tailed_interval(observed=observed, alpha=0.05)
print(f"95% Confidence Interval: [{lower:.4f}, {upper:.4f}]")
```
