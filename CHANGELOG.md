# Changelog

Changes to the Python package `lassoinf`. The R package's changes are in
[`R_pkg/NEWS.md`](R_pkg/NEWS.md).

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
