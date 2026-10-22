# Changelog

Changes to the Python package `lassoinf`. The R package's changes are in
[`R_pkg/NEWS.md`](R_pkg/NEWS.md).

## Unreleased

- `unilasso_inference`: the docstring states that the fit must use the lasso penalty
  (glmnet's `alpha = 1`); an elastic net would add a ridge term with data-dependent
  factors, which is not supported (#26).

## 0.3.0

- `covariance_lasso_inference(beta_hat, cov_beta, prop_select=0.8, exclude=None,
  penalty_factor=None, ...)`: selection and inference from an estimate and its covariance
  alone, e.g. a GLM's coefficients and `cov_params()`. It selects with the lasso in
  covariance form, randomized to mimic selecting on a fraction `prop_select` of the data,
  with a penalty that standardizes the score `Sigma^-1 beta`, the analog of glmnet's
  standardized `X'y`, at a multiple of its expected largest entry under the
  global null. For the selected coefficients it returns selective p-values, confidence
  intervals and MLEs (`target='selected'` or `'full'`), and the estimates independent of
  the selection. The randomization can be a given draw (`perturbation`), such as a
  bootstrap replicate minus the bootstrap mean. See `docs/covariance_lasso.md`.
- `box_lasso_fit`: the lasso with box constraints for a quadratic loss, by coordinate
  descent in the C++ library (`cpp/src/box_lasso.cpp`), shared with the R package.
- `LassoInference.truncated_family(v, variance)`: the `TruncBivariateNormal` law of
  `v' Z_full` given the selection.

## 0.2.1

- The active set is now found with a tolerance relative to the scale of the fit:
  coordinate j counts as 0 (or at a bound) within `tol * max_k |beta_k| s_k / s_j`, with
  `s_j = sqrt(Q_jj)` when `Q` is dense, sparse or has a `diagonal()` (1 for a matrix-free `Q`, to avoid matvecs). The
  default `tol = 1e-6` was an absolute tolerance on the coefficients, so a variable on a
  small scale (a large coefficient) or a large one (a small, still active, coefficient)
  could be misclassified, and rescaling a feature or the response could change the
  selected set.
- The uniLasso computations (the fit, the KKT check, `kappa_j` and the truncation sets) are
  now in the C++ library (`cpp/src/unilasso.cpp`), shared with the R package;
  `lassoinf.unilasso` calls them through `lassoinf_cpp`. Results are unchanged, except that
  truncation-set pieces of rounding width (about 1e-16) are dropped. The pure-Python
  versions are kept as test references in `tests/unilasso_reference.py`.

## 0.2.0

### uniLasso and uniReg

- `unilasso_inference(X, y, beta_hat, lam, loo=...)`: selective inference after a uniLasso
  (or uniReg, `lam = 0`) fit, for the full-model coefficients of the selected variables, in
  a Gaussian linear model with n > p. The uniLasso's penalty factors `1 / |b_uni_j|` and
  sign constraints depend on the data; the truncation set is still found exactly, as a
  finite union of intervals.
  - With `loo=False` the inference is exact.
  - With `loo=True` (the default of the uniLasso packages) it is approximate, with a
    warning. The fit solves the `loo=False` problem exactly at the penalties
    `(n lam + kappa_j) / |b_uni_j|`, and `unilasso_loo_kappa` computes `kappa_j`. This
    includes uniReg with `loo=True` (`lam = 0`), whose penalties are then
    `kappa_j / |b_uni_j|`.
  - `loo` must be given.
- `UniLassoInference` does the same for a given solution, with or without randomization,
  and accepts a per-feature `lam`. `unilasso_fit` fits the uniLasso by coordinate descent.
- `TruncBivariateNormal` accepts a union of intervals.
- Docs: `docs/unilasso.md`, and the math in `docs/data_dependent_penalty.md`.

### Removed: the glmstar adapters

- `glmnet_problem`, `glmstar_problem`, `GLMProblem`, `glm_inference`,
  `glmstar_inference`, `extract_glmnet_problem` and `spec_from_glmnet` are removed. The
  code that extracts the problem a glmnet / glmstar fit solved now lives in glmstar, as
  `glmnet.inference`.
- `glmstar` is no longer a dependency.

### Fixes

- `XTVXOperator` and `CompositeOperator` call `LinearOperator.__init__`, which scipy 1.18
  needs.

### Other changes

- Alden Green and Sifan Liu are listed as authors, with Kevin Fry and Snigdha Panigrahi.

### Docs

- `docs/gaussian_lasso_no_random.md` is rewritten: the gaussian LASSO without
  randomization, with a calibration simulation.

## 0.1.4 and earlier

See the git history and the
[GitHub releases](https://github.com/jonathan-taylor/lassoinf/tags).
