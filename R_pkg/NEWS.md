# lassoinf (development version)

* `unilasso_inference()`: selective inference after a `uniLasso::uniLasso()` or
  `uniLasso::uniReg()` fit with `loo = FALSE`, for the full-model coefficients of the
  selected variables (n > p). The penalty factors and sign constraints depend on the data;
  the truncation set is a finite union of intervals. `UniLassoInference` does the same for
  a given solution, with or without randomization, and `unilasso_fit()` fits the uniLasso.
  See `vignette("unilasso")`.
* `TruncBivariateNormal` accepts a union of intervals.
* New vignettes with calibration simulations (results cached in `vignettes/*_sims.rds`;
  set `LASSOINF_RERUN_SIMS=true` to recompute): `vignette("unilasso")`,
  `vignette("unireg")` (comparing with `uniLasso::ci.uniReg()`) and
  `vignette("gaussian_lasso_polyhedral")`.
* `vignette("unilasso_loo")`: a proof that the leave-one-out uniLasso (`loo = TRUE`) solves
  the plain uniLasso at the penalty `n * lambda + kappa_j`, with
  `kappa_j = 2 sigma^2 + O_p(n^(-1/2))`, with numerical checks.
* `vignette("unireg_loo")`: why `uniLasso::ci.uniReg()` covers so differently with and
  without leave-one-out: with `loo = TRUE`, uniReg shrinks each coefficient by the factor
  `(1 - 2 / t_j^2)_+` (orthogonal case), zeroing weak coefficients whatever their sign.

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
