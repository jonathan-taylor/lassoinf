---
jupytext:
  main_language: python
  cell_metadata_filter: -all
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.1
kernelspec:
  name: python3
  display_name: Python 3 (ipykernel)
  language: python
---

# File Drawer Example

This example demonstrates the "File Drawer" problem, a classic form of selection bias where a test statistic $Z$ is only observed or reported if it exceeds a certain threshold—often representing "statistical significance." In this scenario, results that fail to meet the threshold are effectively left in the "file drawer," leading to overestimates of effect sizes and inflated type I errors if standard inference is used.

## Problem Setup

We model a simple version of this phenomenon:
- **Full Data**: $Z \sim \text{Normal}(\mu, 1)$, where $\mu$ is the true effect size we wish to estimate.
- **Reporting Noise**: $\omega \sim \text{Normal}(0, \gamma^2)$, representing additional variability in the selection process (e.g., small variations in experimental conditions or data cleaning). We set $\gamma = 0.5$.
- **Selection Event**: The result is only "published" if the noisy version of the statistic, $Z + \omega$, exceeds a threshold of 2.0.
- **Observation**: We observe $Z = 1.73$. 

Note that $Z=1.73$ is actually below the threshold of 2.0, but it was "selected" because the unobserved noise $\omega$ was large enough to push $Z + \omega$ over 2.0. If we ignore this selection process, we may produce confidence intervals
with poor coverage and $p$-values with poor Type I error. Selective inference allows us to adjust for the fact that we are only looking at this data point because it passed the filter.

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm as normal_dbn
from lassoinf.selective_inference import SelectiveInference
from lassoinf.discrete_family import discrete_family

# 1. Parameters
mu_null = 0
gamma = 0.5
threshold = 2.0
z_obs = 1.73

# SelectiveInference expects arrays
Z = np.array([z_obs])
Q = np.eye(1)
Q_noise = np.array([[gamma**2]])

# Z_noisy is Z + omega. 
# For the weight function calculation, the specific value of omega doesn't change 
# the probability P(Z + omega > threshold | Z=t), but we need to provide a Z_noisy.
Z_noisy = Z.copy() 

si = SelectiveInference(Z, Z_noisy, Q, Q_noise)

# 2. Define target and constraints
v = np.array([1.0])  # Target is Z itself
A = np.array([[-1.0]])  # -(Z + omega) <= -threshold  => Z + omega >= threshold
b = np.array([-threshold])

# 3. Get the weight function
weight_f = si.get_weight(v, A, b)

# 4. Plot the weight function
t_grid = np.linspace(0, 4, 100)
weights = weight_f(t_grid)
```

### Exact selection adjustment

In this problem we can compute the exact adjustment as
$$
t \mapsto P(Z+\omega > 2 | Z=t) = 1 - \Phi((2-t)/\gamma)
$$

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(t_grid, weights, label='Selection Weight $W(t)$', c='k', linewidth=5)
ax.plot(t_grid, normal_dbn.sf((2 - t_grid) / gamma), c='r', label=r'$1 - \Phi((2-t)/\gamma)$')
assert np.allclose(weights, normal_dbn.sf((2 - t_grid) / gamma))
ax.axvline(z_obs, color='red', linestyle='--', label=f'Observed $Z={z_obs}$')
ax.set_xlabel('Value of $Z$')
ax.set_ylabel('Probability of Selection')
ax.set_title('File Drawer Selection Weight')
ax.legend()
ax.grid(True)
```

```{code-cell} ipython3

# 5. Form the Discrete Family
# We use a grid for the sufficient statistic (observed Z)
grid = np.linspace(-2, 5, 500)

# Reference measure is N(mu_null, 1) * weight_f(t)
# We assume the null mu=0 for the reference distribution
reference_pdf = normal_dbn.pdf(grid, loc=mu_null, scale=1.0)
sel_weights = weight(grid)
```

```{code-cell} ipython3
reference_weights = reference_pdf * sel_weights

# Initialize family
family = discrete_family(grid, reference_weights)

# 6. Inference
# 95% Confidence Interval
lower, upper = family.interval(z_obs, alpha=0.05, randomize=False)
print(f"95% Confidence Interval for mu: ({lower:.3f}, {upper:.3f})")

# P-value for H0: mu = 0
# We can use the CDF at the observed value under theta=0 (which is our reference)
p_val_cdf = family.cdf(0, z_obs, gamma=0.5)
p_val_two_sided = 2 * min(p_val_cdf, 1 - p_val_cdf)
print(f"Two-sided p-value for H0 (mu=0): {p_val_two_sided:.4f}")
```

```{code-cell} ipython3
family.cdf(1, z_obs)
```

## Selective Inference Analysis

### The Weight Function
The weight function $W(t) = P(Z + \omega > 2 \mid Z = t)$ is critical. It calculates the probability that the selection criterion is met for any possible realization $t$ of $Z$. 
- When $t$ is very large, the probability of selection is near 1.
- When $t$ is very small, selection is unlikely but still possible if $\omega$ is large.
The plot generated above shows how this probability "filters" our view of the data.

### Adjusting the Distribution
In standard inference, we would use $Z \sim \text{Normal}(\mu, 1)$. However, given selection, the conditional distribution of $Z$ is:
$$f_{\mu}(z \mid \text{selected}) \propto \phi(z-\mu) \cdot W(z)$$
The `discrete_family` class takes a grid of values and their corresponding weights (the product of the base density and the selection weight) to represent this adjusted exponential family.

### Valid Post-Selection Inference
By inverting the tests in this adjusted family, we obtain confidence intervals and p-values that are valid even though the data point was chosen specifically because it was "large." This approach directly mitigates the bias inherent in the file-drawer effect, providing a more honest assessment of the evidence.

### Using WeightedGaussianFamily
We can also perform the exact same analysis using the built-in `WeightedGaussianFamily` class, which abstracts away the grid creation and base density multiplication:

```{code-cell} ipython3
from lassoinf.gaussian_family import WeightedGaussianFamily

wgf = WeightedGaussianFamily(estimate=z_obs, sigma=1.0, weight_fns=[weight_f], seed=0)

# 95% Confidence Interval
L, U = wgf.interval(level=0.95)
print(f"95% Confidence Interval for mu (WeightedGaussianFamily): ({L:.3f}, {U:.3f})")

# P-value for H0: mu = 0
p_val_wgf = wgf.pvalue(null_value=0, alternative='twosided')
print(f"Two-sided p-value for H0 (mu=0) (WeightedGaussianFamily): {p_val_wgf:.4f}")
```
