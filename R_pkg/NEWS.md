# lassoinf 0.2.0

## uniLasso and uniReg

* `unilasso_inference()`: selective inference after a `uniLasso::uniLasso()` or
  `uniLasso::uniReg()` fit, for the full-model coefficients of the selected variables
  (n > p). The uniLasso's penalty factors and sign constraints depend on the data; the
  truncation set is still found exactly, as a finite union of intervals.
  * With `loo = FALSE` the inference is exact.
  * With the default `loo = TRUE` it is approximate, with a warning: the fit solves the
    `loo = FALSE` problem exactly at the penalties `(n * lambda + kappa_j) / |b_uni_j|`,
    with `kappa_j` computed from the data. This includes `uniReg(loo = TRUE)`
    (`lambda = 0`), whose penalties are then `kappa_j / |b_uni_j|`, with
    `kappa_j` close to `2 sigma^2`.
* `UniLassoInference` does the same for a given solution, with or without randomization,
  and accepts a per-feature `lam`. `unilasso_fit()` fits the uniLasso.
  See `vignette("unilasso")`.

## Vignettes

* New vignettes with calibration simulations (results cached in `vignettes/*_sims.rds`;
  set `LASSOINF_RERUN_SIMS=true` to recompute): `vignette("unilasso")`,
  `vignette("unireg")` (comparing with `uniLasso::ci.uniReg()`) and
  `vignette("gaussian_lasso_polyhedral")`.
* `vignette("unilasso_loo")`: a proof that the leave-one-out uniLasso (`loo = TRUE`) solves
  the plain uniLasso at the penalty `n * lambda + kappa_j`, with
  `kappa_j = 2 sigma^2 + O_p(n^(-1/2))`, with numerical checks. For GLM families it is, to
  first order, the shift `2 * phi` (`phi` the dispersion, 1 for the binomial), checked
  with `uniLasso(family = "binomial")`.
* `vignette("unireg_loo")`: why `uniLasso::ci.uniReg()` covers so differently with and
  without leave-one-out: with `loo = TRUE`, uniReg shrinks each coefficient by the factor
  `(1 - 2 / t_j^2)_+` (orthogonal case), zeroing weak coefficients whatever their sign.

## Other changes

* `TruncBivariateNormal` accepts a union of intervals.
* Alden Green and Sifan Liu are listed as authors.

# lassoinf 0.1.0

* First CRAN release.
* `LassoInference`: selective intervals and p-values for the (randomized) bounded LASSO,
  with dense matrices or matrix-free operators (`dense_operator()`, `xtvx_operator()`).
* `glmnet_inference()`: inference from a fitted `glmnet` object at a given `lambda`,
  without randomization or with carving (`selection_rows`).
* `glmnet_problem()` and `glmnet_problem_from_fit()`: the LASSO problem glmnet solved, with
  the Hessian at a one-step relaxed fit by default (`information = "relaxed"`).
* Inference for user-specified estimands (`contrast_inference()`,
  `custom_estimand_inference()`, `estimand_summary()`), inactive variables
  (`inactive_summary()`, `SelectionCoordinates`) and screening (`ScreenedSelection`).
