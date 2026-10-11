# Selection events along the conditioning ray (`lassoinf.ray_selection`)

For a contrast, the selection data move along the ray `Y(w) = N_o + bar_N_o + bar_Gamma w`.
If the selection event has an explicit characterization `event(Y)`, the truncation set
`{w : event(Y(w))}` can be found numerically, by a grid over `w_obs +- n_sd sd(w)` and
bisection of each change. Inference is then the truncated bivariate normal on a union of
intervals: the Gaussian convolution through `scalar_noise`.

This handles data-dependent penalties with no closed form: rows that are sums of ratios.
It needs no refitting, unlike Liu, Markovic-Voronov and Taylor (arXiv:2203.14504). The
uniLasso and uniReg keep their closed form in `lassoinf.unilasso` and do not use this module.

- `find_truncation_set(event, contrast, n_sd, n_grid, tol)`
- `ray_contrast_inference(affine_constraints, direction, event, level, ...)`
- `lasso_event(beta_hat, G_hat, Q, penalty)`: the lasso KKT event at the observed active set
  and signs, with `(D, L, U) = penalty(Y)`, or `None` outside the conditioning event.

## Checks (`tests/test_ray_selection.py`)

- With fixed penalties it reproduces the polyhedral interval (`get_interval`).
- For the uniLasso it reproduces the closed-form truncation sets.
- The adaptive lasso is calibrated: penalties `lam / |b_ols_j(Y)|` with `b_ols = Q^{-1} Y`,
  targeting the full-model coefficients.

`dev/ray_adaptive_lasso_calibration.py`, 500 targets each, 90% intervals:

| n | kappa | ray: KS p (coverage) | penalties treated as fixed: KS p |
|---|---|---|---|
| 80 | 0 | 0.68 (0.916) | 0.68 |
| 80 | 1 | 0.66 (0.890) | 0.97 |
| 400 | 0 | 0.39 (0.908) | 6e-6 |
| 400 | 1 | 0.99 (0.908) | 0.09 |

All truncation sets were single intervals. Each inference takes about 0.15 s with an
801-point grid.

## Next

- Use the per-row structure: return the slack of each row instead of a boolean, use
  `brentq` on the sign changes of each row, and use the degree bound to check that no
  pieces are missed.
- Other targets: selected-model coefficients (`W Z_E`), and inactive variables.
- The R port, and a vignette, e.g. the adaptive lasso.
- Designs where sets have several pieces (strong correlation, small n), to test the grid.
