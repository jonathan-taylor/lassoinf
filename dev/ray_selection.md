# Selection events along the conditioning ray (`lassoinf.ray_selection`)

The root finding (`find_truncation_set`) and the lasso event (`LassoRayEvent`) are in the
C++ library, in `cpp/src/ray_selection.cpp`. Python (`lassoinf.ray_selection`) and R
(`find_truncation_set()`, `lasso_event()` and `ray_contrast_inference()`) call it. An event
can be a C++ `lasso_event`, which is evaluated in C++ with only `penalty(Y)` called back,
or any Python or R function. The pure-Python versions are test references, in
`tests/ray_selection_reference.py`.

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

## Checks (`tests/test_ray_selection*.py`, `R_pkg/tests/testthat/test_ray_selection.R`)

- The C++ results match the pure-Python reference: the event is identical along the ray,
  and the truncation sets agree to 1e-9 sd.

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

## Speed: the structure along the ray

`A` depends only on the active set, signs and bound activations, and `b` is affine in
`(D, L, U)`. Along a ray (`RayEvent::along`), `LassoRayEvent` computes `A Y(w) = u + a w`
once, and caches `Q_{E,E}^{-1}`, `Q_{E^c,E}` and the columns of `Q` at nonzero bounds. Each
point then costs `penalty(Y(w))` and `O(|E| p)`, instead of a rebuild of the constraints
(`|E|` matvecs and an `|E|^3` inverse). The structure is cached for each pattern of bound
activations met along the ray, so the event is unchanged. A coordinate at a bound that
moves with `Y` leaves it just off `w_obs`, which changes the pattern.

One truncation set of an adaptive lasso, default 2001-point grid (`penalty` in Python):

| p | active | Python reference | C++, rebuilding | C++, along the ray |
|---|---|---|---|---|
| 10 | 5 | 0.30 s | 0.02 s | 0.01 s |
| 40 | 11 | 0.34 s | 0.03 s | 0.02 s |
| 160 | 35 | 0.40 s | 0.26 s | 0.02 s |
| 640 | 164 | 0.89 s | 12.5 s | 0.08 s |

At p = 640, about a third of the remaining time is the Python `penalty` calls.

## Next

- Use the per-row structure: return the slack of each row instead of a boolean, use
  `brentq` on the sign changes of each row, and use the degree bound to check that no
  pieces are missed.
- Other targets: selected-model coefficients (`W Z_E`), and inactive variables.
- A vignette, e.g. the adaptive lasso.
- Designs where sets have several pieces (strong correlation, small n), to test the grid.
