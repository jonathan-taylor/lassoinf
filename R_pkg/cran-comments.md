## Submission

This is an update of lassoinf (0.1.0 on CRAN) to 0.3.1. It adds selective inference after
the uniLasso and uniReg (`unilasso_inference()`), and after a randomized lasso in
covariance form from an estimate and its covariance (`covariance_lasso_inference()`), with
vignettes for each, and fixes the score covariance for elastic-net fits in
`glmnet_inference()`. Versions 0.2.0, 0.2.1 and 0.3.0 were not submitted to CRAN; their
changes are in NEWS.md.

## Test environments

* local macOS (aarch64), R 4.5.3
* TODO before submitting: GitHub Actions (macOS, Windows, Ubuntu devel/release/oldrel-1)
  and win-builder (R-devel) for 0.3.1

## R CMD check results

0 errors | 0 warnings | 0 notes

(Local `R CMD check --as-cran` on 2026-10-21 also gave "Days since last update: 1", which
no longer applies at submission, and a note that the local HTML Tidy is too old.)

## Notes for the reviewers

* The C++ sources in `src/lassoinf/` are shared with the package's Python implementation
  (https://github.com/jonathan-taylor/lassoinf).
* The truncated bivariate normal computations are adapted from MIT-licensed code by
  Sifan Liu (listed as author and copyright holder); see `inst/COPYRIGHTS`.
* The vignettes' simulation results are precomputed and saved in `vignettes/*.rds`, so
  building the vignettes does not rerun the simulations.
