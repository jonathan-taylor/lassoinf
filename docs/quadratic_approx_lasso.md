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

## Classic LASSO

The classic LASSO problem in Lagrange form ($n > p$ and full rank where needed)
is 
$$
\text{minimize}_{\beta} \frac{1}{2} \|Y-X\beta\|^2_2 + \|D\beta\|_1 = 
\text{minimize}_{\beta} -\beta'Z + \frac{1}{2} \beta'Q\beta + \|D\beta\|_1
$$
where $Z=X'y$, $Q=X'X$ and $D$ is some diagonal matrix. Setting $\bar{\ell}(\beta)=
\frac{1}{2} \|Y-X\beta\|^2_2$ we can write this as
$$
\text{minimize}_{\beta} - \beta'\nabla \bar{\ell}(\beta_0) + \frac{1}{2}(\beta-\beta_0)'
\nabla^2 \bar{\ell}(\beta_0)(\beta - \beta_0) + \|D\beta\|_1
$$
with the first expression using $\beta_0=0$. For population parameter
$$
\beta^*=\beta^*_F = \left(E_F[X'X]\right)^{-1} E_F[X'Y]
$$
this becomes
$$
\text{minimize}_{\beta} -\beta'\nabla \bar{\ell}(\beta^*_F) + 
\frac{1}{2}(\beta - \beta^*_F)'\nabla^2 \bar{\ell}(\beta^*_F)(\beta-\beta^*_F) + \|D\beta\|_1
$$
with $\nabla \bar{\ell}(\beta^*_F)$ approximately normally distributed (depending on 
the statistical model we are using).

It is straightforward to check
that in the squared-error case the following identity holds **for all $\beta$:**
$$
Z=X'y = -\nabla \bar{\ell}(\beta) + \nabla^2 \bar{\ell}(\beta)\beta.
$$

The LASSO problem then takes the form
$$
\text{minimize}_{\beta} -\beta'Z + \frac{1}{2} \beta'Q\beta + \|D\beta\|_1.
$$

## M-estimator form

This last form generalizes. The GLM versions take on a more M-estimator form
$$
\text{minimize}_{\beta} \sum_{i=1}^n \ell(Y_i, X_i'\beta) + \|D\beta\|_1
= \text{minimize}_{\beta} \bar{\ell}(\beta) + \|D\beta\|_1
$$
for some sufficiently smooth $\ell$ with the classic lasso setting $\ell(r)=r^2$.

Given a solution $\hat{\beta}$, set, as above
$$
\begin{aligned}
Z &= Z(\hat{\beta}) \\
& -\nabla \bar{\ell}(\hat{\beta}) + \nabla^2 \bar{\ell}(\hat{\beta})\hat{\beta} \\
&= \hat{u} + \nabla^2 \bar{\ell}(\hat{\beta})\hat{\beta}.
\end{aligned}
$$
Fixing $Q=Q(\hat{\beta}) = \nabla^2 \bar{\ell}(\hat{\beta})$, $Z$ is an
affine function of $(\hat{\beta}, \hat{u}) \equiv (\hat{\beta}_E, \hat{u}_{-E})$ with inverse
$$
\begin{aligned}
\hat{\beta}_E &= Q_{E,E}^{-1}(Z_E - u_E) \\
\hat{u}_{-E} &= Z_{-E} - Q_{-E,E}\hat{\beta}_E.
\end{aligned}
$$
with $u_E=D[E] \cdot \text{sign}(\hat{\beta}_E)$.
It is straightforward to verify that $\hat{\beta}$ solves the problem
$$
\text{minimize}_{\beta} - \beta'Z + \frac{1}{2}\beta'Q\beta+ \|D\beta\|_1.
$$

### Affine constraints

Given $Q$, the affine constraints of {cite}`LeeLasso` can therefore be expressed in terms of $Z$
as
$$
\begin{aligned}
\text{sign}\left(Q_{E,E}^{-1}(Z_E - u_E)\right) &= \text{sign}(\hat{\beta}_E) \\
\|D_{-E}^{-1}\left(Z_{-E} - Q_{-E,E}Q_{E,E}^{-1}Z_E + Q_{-E,E}Q_{E,E}^{-1}u_E\right) \|_{\infty} \leq 1
\end{aligned}
$$

## Upper and lower limits

Packages such as `glmnet` offer the additional possibility to impose hard constraints on $\beta_j$ of the form $\beta_j \in [L_j,U_j]$
with $L_j \in [-\infty, 0]$ and $U_j \in [0, \infty]$. This leads to the (quadratic) problem
$$
\text{minimize}_{\beta: \beta \in [L, U]} -\beta'Z + \frac{1}{2} \beta'Q\beta + \|D\beta\|_{-1}.
$$

## Quadratic approximation when bootstrapping

- When simply given a bootstrap sample, the natural quadratic form is to use the boostrap *precision*, i.e.
the inverse of the bootstrap covariance. This will generally be invertible under conditions
where the bootstrap is appropriate.

## Implementation

This modifies the affine constraints of the KKT somewhat but not substantially. In this context,
the usual notion of *active set* $E$ is best replaced with the notion of *free set* (i.e. those
coordinates whose values are not in $\{L_j,O, U_j\}$. The function below
takes a triple $(\hat{\beta}, \nabla \bar{\ell}(\hat{\beta}), \nabla^2 \bar{\ell}(\hat{\beta})$ as well
as the problem specifications $D, U, L$ and constructs the corresponding set of affine constraints analogous
to those in {cite}`LeeLasso`.

```{code-cell} ipython3
import inspect
from lassoinf.lasso import lasso_post_selection_constraints

# Compute the required parameters for inference
print(inspect.getsource(lasso_post_selection_constraints))
```
