# Binomial uniLasso inference: prototype (to do after 0.2.0)

`dev/binomial_unilasso_prototype.R` runs selective inference for the full-model coefficients
after `uniLasso(family = "binomial")`, through the quadratic approximation:

- the univariate logistic fits give slopes `b_j` and slope informations `info_j` (weighted,
  centered sums), so `b_j ~ Z_j / info_j` along the conditioning line;
- `Q`: the Hessian with the intercept profiled out, **evaluated at the uniLasso solution
  `beta_hat`**;
- `Z = Q beta_hat + X'(y - mu_hat)`, `Sigma = Q`, no randomization;
- penalties `n lambda / |b_j|`, plus `2` (the GLM shift `2 phi`) for `loo = TRUE`;
- `UniLassoInference$new(..., C = b, C_scale = info)`. `C` and `C_scale` are new arguments
  on this branch; the defaults, `Z / diag(Q)` and `diag(Q)`, are the gaussian case.

Run it with `Rscript dev/binomial_unilasso_prototype.R <lib dir> <n> <loo> <variant>`, where
`<lib dir>/rlib` has lassoinf (this branch) and uniLasso. The variants are `baseline` (`Q` at
the full-model MLE), `C_scale_full`, `Q_at_beta_hat` and `hybrid` (`Q` at `beta_hat`, `Sigma`
at the MLE).

Coverage of 90% intervals, 300 targets each, design: equicorrelation 0.5, p = 6,
beta = (0.6, -0.5, 0.4, 0, 0, 0) * sqrt(300 / n) * 1.5:

| n | loo | Q at beta_hat: coverage (KS p) | Q at the MLE: coverage (KS p) |
|---|---|---|---|
| 300 | FALSE | 0.88 (0.92) | 0.84 (5e-5) |
| 1000 | FALSE | 0.89 (0.95) | 0.86 (0.053) |
| 3000 | FALSE | 0.90 (0.78) | 0.90 (0.89) |
| 300 | TRUE | 0.93 (0.42) | 0.81 (2e-12) |
| 1000 | TRUE | 0.89 (0.12) | 0.88 (0.016) |

The hybrid is about the same as `Q` at `beta_hat`. Changing `C_scale` to `diag(Q)` makes no
difference.

## To do

- `unilasso_inference(fit, x, y, s)` for `family = "binomial"` fits, in R and Python.
- A linearized `kappa_j` for `loo = TRUE`, in place of the plain shift 2.
- Tests and calibration checks, including imbalanced classes and other designs.
- A section in `vignette("unilasso")` or a new vignette.
