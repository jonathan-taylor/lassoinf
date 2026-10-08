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
