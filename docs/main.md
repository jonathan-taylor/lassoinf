---
jupytext:
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

# Inference for $\hat{\theta}=\eta'Z$ conditioned on selection of $\bar{Z}=Z+\omega$

Suppose $Z \sim N(\zeta^*, \Sigma)$ and $\omega|Z \sim N(0, \bar{\Sigma})$. We're interested in conditional inference 
on a target $\hat{\theta}=\eta'Z$ based on the joint law of $(Z,\omega)$ truncated to $\{(Z,\omega):A(Z+\omega) \leq b\}$. We could consider
some soft truncation of the form $\pi(Z+\omega)$.
but affine constraints are perhaps the most tractable.

As in {cite}`LeeLasso` we will condition on additional information to make the distribution tractable and to eliminate
dependence on nuisance parameters. This means we are essentially forced to condition on
$$
N = Z - \Sigma \eta (\eta'\Sigma \eta)^{-1}\hat{\theta} = Z - \Gamma \hat{\theta} = RZ.
$$

We can condition on more if convenient of course. Conditioning on too much leads to intervals that 
may be too short even if selection is trivial. For instance, we can condition on $Z+\omega$ (this is
"data splitting / thinning" but then we are left only with the "second half" of the data to form unbiased estimates.

This second half is easily seen to be
$$
Z - \Sigma \bar{\Sigma}^{-1}\omega
$$
with variance
$$
\Sigma + \Sigma \bar{\Sigma}^{-1}\Sigma > \Sigma.
$$
Hence the variance of the "thinned" estimator is
$$
\eta'\Sigma \eta + \eta'\Sigma \bar{\Sigma}^{-1}\Sigma \eta.
$$

In general, we can condition on anything linear function of $(Z+\omega)$ (assuming invertibility of $\Sigma, \bar{\Sigma}$ where necessary). 
But which contrasts will keep the nominal variance under trivial selection the same as $\text{Var}(\hat{\theta})$?

It is not hard to see that conditioning on any linear contrast $v'(Z+\omega)$ such that $\text{Cov}(v'(Z+\omega), \hat{\theta} | N)=0$ preserves
this "best case variance". Well, as $\text{Cov}(\hat{\theta},N)=0$
$$
\begin{aligned}
\text{Cov}(v'(Z+\omega), \hat{\theta}|N) &= \text{Cov}(v'(Z+\omega), \hat{\theta})\\
&= \text{Cov}(v'Z, \hat{\theta}).
\end{aligned}
$$

We see then, that the *maximal* contrast of $Z+\omega$ we can condition on without affecting this "best" case variance is $R(Z+\omega)$ where 
$N=RZ$. Of course, conditioning on $\sigma(RZ, R(Z+\omega))$ is equivalent to conditioning on $\sigma(RZ,R\omega)$.

Conditioning on this pair of vectors restricts variation in our original joint law of $(Z,\omega)$ to a 2-dimensional affine plane
with "linear" part given by $\eta'Z$ and $c'\omega$ where $c$ is chosen such that $\text{Cov}(c'\omega, R\omega) = c'\bar{\Sigma}R=0$. Of course, $c$ is defined
only up to scaling which we can choose.  It is not difficult to verify that  we can (and will) take
$$
c = \bar{\Sigma}^{-1}\Sigma \eta.
$$

The computational complexity here is the cost of solving for $c$ in a linear system $\bar{\Sigma}c=\Sigma \eta$.

+++

#### Decomposition of $\omega$ given $c$

Having computed $c$, we can decompose $\omega$ as
$$
\omega - \bar{\Gamma} \cdot c'\omega + \bar{\Gamma} \cdot c'\omega
$$
with
$$
\bar{\Gamma} = (c'\bar{\Sigma}c)^{-1} \text{Cov}(\omega, c'\omega) = (\eta'\Sigma \bar{\Sigma}^{-1} \Sigma \eta)^{-1} \Sigma \eta
$$

+++

#### Well-specified assumption

In certain scenarios, such as classical data splitting or other resampling and inference procedures such as {cite}`AssumptionLean` it may be the case that for some $\gamma > 0$
$$
\bar{\Sigma} = \gamma^2 \cdot \Sigma.
$$
**In this case we can take $c=\eta$ which implies $\bar{\Gamma}=\Gamma$.**

### Reduced affine constraints

Given $c$, we'll write $\bar{\omega}=c'\omega$, a centered univariate Gaussian with variance $c'\bar{\Sigma}c=\eta'\Sigma \bar{\Sigma}^{-1}\Sigma \eta = \bar{s}^2$. We can always write
$$
\omega = \left(\omega - (c'\bar{\Sigma}c)^{-1}\bar{\Sigma}c \cdot \bar{\omega} \right) + (c'\bar{\Sigma}c)^{-1}\bar{\Sigma}c \cdot \bar{\omega} = \bar{N} + \bar{\Gamma} \bar{\omega}
$$
with $\bar{N}$ a linear functional of $R\omega$.

```{code-cell} python
import inspect
import numpy as np
from lassoinf.selective_inference import SelectiveInference

# Compute the required parameters for inference
print(inspect.getsource(SelectiveInference.compute_params))
```

We have decomposed the joint law $(Z,\omega)$ into 4 independent pieces $(\hat{\theta}, N, \bar{\omega}, \bar{N})$ such that $\text{Var}(\hat{\theta} | N, \bar{N}) = \text{Var}(\hat{\theta})$. Hence,
when selection is trivial the corresponding confidence intervals and $p$-values for testing $H_0:\eta'\zeta^*=\theta_0$ will be essentially as if we had done no selection.

Our selection event can be rewritten as
$$
\left\{(\hat{\theta}, N, \bar{\omega}, \bar{N}): A\left(N + \Gamma \hat{\theta} + \bar{N} + \bar{\Gamma} \bar{\omega}\right) \leq b \right\}.
$$
Equivalently, fixing $(N,\bar{N})$ at observed values $(N_o, \bar{N}_o)$ this is
$$
\left\{(\hat{\theta}, \bar{\omega}): A (\Gamma \hat{\theta} + \bar{\Gamma} \bar{\omega}) \leq b - A(N_o + \bar{N}_o) \right\}
$$

+++

### Post-selection density

Fixing $\hat{\theta}$ at some nominal density argument $t$ we can now compute the selective adjustment to the marginal law $N(\eta'\zeta^*, \eta'\Sigma \eta)$:
$$
\begin{aligned}
t &\mapsto \mathbb{P} \left( \bar{A} \bar{\omega}\leq \bar{b}(N,\bar{N}, \hat{\theta}) \biggl| \hat{\theta}=t, N=N_o, \bar{N}=N_o\right) \\
&= \Phi(U(N_o, \bar{N}_o, t)/\bar{s}) - \Phi(L(N_o, \bar{N}_o, t)/\bar{s}).
\end{aligned}
$$
with
$$
\bar{b}(N,\bar{N},\hat{\theta}) = b - A( N +  \bar{N} + \Gamma \hat{\theta}).
$$

```{code-cell} python
# Compute the truncation interval [L, U]
print(inspect.getsource(SelectiveInference.get_interval))
```

The conclusion follows from the fact that, as discussed in {cite}`LeeLasso` $\left\{\bar{\omega}: \bar{A} \bar{\omega} \leq \bar{b}(N_o,\bar{N}_o,t)\right\}$ is an interval $[L(N_o,\bar{N}_o, t), U(N_o, \bar{N}_o, t)]$. 

```{code-cell} python
# Calculate the selection probability (weight)
print(inspect.getsource(SelectiveInference.get_weight))
```

## Implementation

Below is a Python implementation of the framework described above.

```{code-cell} python
# The core dataclass holding the problem parameters
source = inspect.getsource(SelectiveInference)
print(source[:source.find("    def compute_params")].strip())
```

### Step-by-Step Computation

We can demonstrate the computation by instantiating the class and looking at the intermediate results.

```{code-cell} python
# Simulation setup
np.random.seed(42)
n = 10
Z = np.random.randn(n)
Q = np.eye(n)
gamma_val = 0.5
Q_noise = (gamma_val**2) * Q
omega = np.random.multivariate_normal(np.zeros(n), Q_noise)
Z_noisy = Z + omega

# Instantiate class
si = SelectiveInference(Z, Z_noisy, Q, Q_noise)

# Define contrast eta (v)
v = np.zeros(n)
v[0] = 1.0  # target is Z[0]

# Compute parameters
params = si.compute_params(v)

print("Target theta_hat:", params['theta_hat'])
print("Contrast c (should be eta under well-specified):", params['c'][:3], "...")
print("Variance bar_s:", params['bar_s'])
```

And computing the interval $[L, U]$ given some constraints $AZ_{noisy} \leq b$:

```{code-cell} python
# Example constraints: Z_noisy > 0 (or -Z_noisy <= 0)
A = -np.eye(n)
b = np.zeros(n)

# Observed value of theta_hat
theta_obs = params['theta_hat']

# Compute interval for bar_theta
L, U = si.get_interval(v, theta_obs, A, b)
print(f"Interval [L, U] for bar_theta: [{L:.4f}, {U:.4f}]")
print(f"Observed bar_theta: {params['bar_theta']:.4f}")

# Compute selection weight function
weight_f = si.get_weight(v, A, b)
print(f"Selection weight at theta_obs: {weight_f(theta_obs):.4f}")

# Evaluate over a range of t values
t_grid = np.linspace(theta_obs - 5, theta_obs + 5, 100)
weights = [weight_f(t) for t in t_grid]

import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(t_grid, weights, label="Selection Weight", color="blue")
ax.axvline(theta_obs, color="red", linestyle="--", label="Observed $\hat{\\theta}$")
ax.set_xlabel("Target value $t$")
ax.set_ylabel("Selection Probability")
ax.set_title("Selection Weight Function")
ax.legend()
fig.tight_layout()
plt.show()
```

## C++ Implementation via pybind11

For performance-critical applications, a C++ implementation using Eigen is also available. It is mirrored from the Python logic and exposed via `pybind11`.

```{code-cell} python
from lassoinf.lassoinf_cpp import SelectiveInference as SelectiveInferenceCPP

# Instantiate C++ class
si_cpp = SelectiveInferenceCPP(Z, Z_noisy, Q, Q_noise)

# Compute parameters using C++
params_cpp = si_cpp.compute_params(v)
print("C++ Target theta_hat:", params_cpp.theta_hat)

# Compare selection weight function
weight_f_cpp = si_cpp.get_weight(v, A, b)
print(f"C++ Selection weight at theta_obs: {weight_f_cpp(theta_obs):.4f}")

# Cross-check weights
weights_cpp = [weight_f_cpp(t) for t in t_grid]
diff = np.abs(np.array(weights) - np.array(weights_cpp)).max()
print(f"Maximum difference between Python and C++ weights: {diff:.2e}")
```

### Post-selection density under the well-specified assumption

Under the well-specified assumption with $c=\eta$ and $\bar{\Gamma}=\Gamma$ we see that $N + \bar{N} = L(Z+\omega)$ and $\Gamma \hat{\theta}+\bar{\Gamma}\bar{\omega} = \Gamma \eta'(Z+\omega)$ with $\eta'(Z+\omega)$ which we might call $\bar{\theta}$, a noisy
estimate of our target. In this case, it is more natural to consider the equivalent law of $\hat{\theta}, \bar{\theta}$: 
$$
\left\{(\hat{\theta}, \bar{\theta}): A\bar{\theta} \leq b - AL(Z+\omega) \right\} = \left\{(\hat{\theta}, \bar{\theta}): \bar{\theta} \in [L(Z+\omega), U(Z+\omega)]\right\}
$$
In this case the adjustment to the selection density takes the form
$$
t \mapsto \Phi((\bar{U}(Z+\omega)-t)/s) - \Phi((\bar{L}(Z+\omega)-t)/s)
$$
with $s = \text{Var}(\bar{\theta} | \hat{\theta})$. These upper and lower limits are exactly the truncation intervals of {cite}`LeeLasso` when applied to the noisy data $Z+\omega$!

