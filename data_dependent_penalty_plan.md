# Plan: selective inference after the uniLasso

Branch: `data_dependent_penalty`. The math is in `docs/data_dependent_penalty.md`, the
uniLasso page in `docs/unilasso.md`.

Goal: selective inference after the uniLasso and unireg (`λ = 0`). Its penalty factors
`1/|β̂_j^uni|` and sign constraints depend on the data. The approach generalizes the
polyhedral lemma from `AX <= b` with `b` fixed to `AX <= B/C`, with `B`, `C` jointly Gaussian
with `X` and the signs of `C` conditioned on. The focus is the uniLasso; the general
data-dependent case (finding 3) is deferred.

## Status

- Done, in Python and R: the uniLasso and unireg for **full-model** coefficients, n > p
  (finding 6). `unilasso_inference` takes a fit: in R a `uniLasso` / `uniReg` object
  (`unilasso_inference(fit, x, y, s)`), in Python `(X, y, beta_hat, lam)`. lam uses glmnet's
  scaling, and there is no randomization (polyhedral). `UniLassoInference` takes a solution
  for given (possibly randomized) selection data, and `unilasso_fit` fits the uniLasso. R
  is pure R on top of the existing C++ contrasts. Python and R agree to ~1e-8.
- The fit must use `loo = FALSE` (finding 7); R refuses `loo = TRUE` fits.
- Next: selected-model coefficients `e_k' Q_EE⁻¹ Z_E` (cubic rows, finding 5).
- Deferred: misspecified `Σ` and other targets (many ratios per row), and the general
  `LassoInference` API for data-dependent penalties (Phase 2).

## Findings that shape the plan

1. **A row is a union of at most two intervals, not always one.** On the interval where the
   sign conditions hold, the row is a quadratic inequality. When the quadratic opens the
   "wrong" way, its solution set is the complement of an interval. Example: `3 - t <= 1/t` on
   `t > 0` holds on `(0, 0.382] ∪ [2.618, ∞)`. The truncation set is therefore a finite union
   of intervals, and the truncated (bivariate) normal must handle unions. This is easy, since
   all probabilities and moments are sums over the intervals.
2. **Randomized case: `B` and `C` must be functions of the selection data `Z + ω`.** Then
   `Cov(B, θ̂) = Cov(B, ω̄)`, and the event still depends on `(θ̂, ω̄)` only through the scalar
   `w = (s̄²/σ²) θ̂ + ω̄`, as it does now. Otherwise the event is a 2-d region bounded by
   conics, which is out of scope.
3. **The lasso's constraint rows are not single ratios once the active penalties are
   data-dependent.** `b = b_0 + M D`, and every row depends on every active `D_E`, through
   `c_E = W(… + D_E s_E)`. An inactive `D_j` enters only its own two rows. So a lasso with
   `D_k = B_k / C_k` gives rows `(AX)_i <= b_0i + Σ_k M_ik B_k/C_k`. These are polynomial
   inequalities of degree `<= |E| + 2` after clearing denominators, still finite unions of
   intervals, but with no closed form. Phase 3 needs a decision on this (see Open questions).

4. **For the uniLasso, a single interval is typical but not guaranteed.** A brute-force count
   along `w` gave one interval in 449 of 450 random instances. The exception had a second piece
   3.7 to 4.4 sd from the observed `w`, coming from an inactive row (a cubic, see 5). So the
   code must handle unions, but in practice the extra pieces are rare and far in the tails.
5. **For the well-specified uniLasso, rows have at most two moving ratios.** With `Σ = σ²Q`
   and the target `e_k' Q_EE⁻¹ Z_E`, `Cov(C_E, θ̂) = σ² e_k / diag(Q)_E` is one-hot. So the
   only active penalty that moves along `w` is `D_k`. Active rows are single ratios
   (quadratics). Inactive rows also involve their own `D_j`, whose `C_j` moves, so they are
   cubics. Both checked numerically. This resolves finding 3 for the uniLasso, with closed
   forms and no bracketing. Misspecified `Σ` brings back the general case.
6. **Full-model targets (n > p) have one moving ratio per row.** For `θ̂_k = e_k' Q⁻¹ Z`,
   `Ση = σ² e_k`, so `Cov(C, θ̂)` is one-hot over all coordinates and only `C_k` moves. `b` is
   affine in `D`, so every row is `(AY)_i <= b0_i + M_ik λ s_k / C_k = B_i / C_i` with
   `C_i = C_k` and `B_i = b0_i C_k + M_ik λ s_k`. The quadratic solver applies. For unireg
   (`λ = 0`), the rows reduce to the polyhedron plus the sign condition on `C_k`.
7. **The `uniLasso` package's default `loo = TRUE` is a different selection event.** It
   regresses `y` on leave-one-out univariate fits `F`, which are linear in `y`, so its KKT
   conditions are quadratic in the data. Its fits violate the idealized KKT conditions by
   about 17 `nλ`, and the idealized uniLasso at the same `λ` selected the same variables and
   signs in only 159 of 300 fits. With `loo = FALSE` the fit is exactly the idealized
   problem (violation 4e-15 at `thresh = 1e-14`).

## Phase 1: single-ratio constraints (Python prototype) — done

Status: implemented on this branch, with tests passing (see below). The defaults reproduce
`get_interval` and `LassoInference` exactly. In the 1-d two-piece simulation the ratio pivot is
uniform (KS p ≈ 0.8), while treating `B/C` as fixed is not (KS p ≈ 1e-14).

New module `python/lassoinf/ratio_constraints.py`:

- `RatioConstraints` (dataclass): `A` (matrix or operator), observed `B`, `C`, signs `s_B`
  (default 0, i.e. not conditioned) and `s_C` (default `sign(C)`), and `cov_BZ`, `cov_CZ`
  (`Cov(B, Z)`, `Cov(C, Z)`; default 0). Its `truncation_set(contrast)` uses an
  `AffineConstraintsContrast` to build the affine functions of `w` and returns the
  truncation set as an array of disjoint intervals.
- Vectorized row solver: for each row, the sign interval `I_i`, then the roots of the
  quadratic (computed in the numerically stable form), then intersection with `I_i`. This
  gives `<= 2` intervals per row.
- Intersection of the rows' interval unions, by sweeping over sorted endpoints.
- `ratio_contrast_inference(constraints, ratio, direction, level)`: intervals and p-values for
  one contrast, mirroring `LassoInference._compute_inference`.

Changes to existing code:

- `bivariate_normal.py`: `compute_gaussian_conditional_stats` and `TruncBivariateNormal`
  accept arrays `L`, `U` of interval endpoints, summing probabilities and moments. Scalars
  behave exactly as now.

Tests (`tests/test_ratio_constraints.py`):

- the row solver against a brute-force grid search, including the two-piece case and rows
  with no solutions;
- the defaults (`C = 1`, `B = b`, no covariance) reproduce `AffineConstraintsContrast.get_interval`
  and the current `LassoInference` intervals and p-values;
- with an interval union of one interval, `TruncBivariateNormal` is unchanged; with several, it
  matches numerical integration;
- **calibration by simulation**: a randomized selection `A(Z + ω) <= B/C`, with `C` linear in
  `Z + ω` and signs conditioned on, then rejection sampling of the selection event. Pivots at
  the true parameter should be uniform (KS test), and pivots that treat `B/C` as fixed should
  be visibly miscalibrated.

## Phase 2: general user-facing API (deferred)

- `LassoInference` takes the extra inputs: per coordinate `B`, `C`, signs, and the
  covariances. The defaults reproduce the current behaviour exactly, which the existing tests
  will check.
- Return the truncation set (as intervals) in each contrast, for diagnostics.

## Phase 3: the uniLasso

Full-model coefficients (n > p): done.

- `python/lassoinf/unilasso.py`: `UniLassoInference` (a `LassoInference` whose targets are
  full-model coefficients and whose constraints are `RatioConstraints`), `unilasso_fit`
  (coordinate descent), and `unilasso_inference(X, y, lam)`.
- Tests (`tests/test_unilasso.py`): KKT of the fit; the truncation set against a brute-force
  evaluation of the selection event along `w` (recomputing every `C`, `D`, `b`); calibration
  by simulation for unireg and the uniLasso; the summary's estimates are the full-model least
  squares coefficients.
- Simulations (n = 60, p = 5, ρ = 0.5, κ = 1): pivots uniform for λ = 0, 0.25, 0.5 and 1 times
  √n (KS p 0.11 to 0.4), coverage 0.92 at the 90% level. Treating `b` as fixed is
  miscalibrated for unireg (KS p ≈ 0.0015) but not detectably so for the uniLasso in these
  designs.

Selected-model coefficients, next. The general approach:

- Express each lasso constraint row as an affine function of `w` plus `Σ_k M_ik D_k(w)`, with
  `D_k(w) = B_k(w) / C_k(w)`. `M` comes from `lasso_post_selection_constraints`. Only ratios
  whose `B_k` or `C_k` moves along `w` are kept; the rest fold into the constant.
- Generalize the row solver from one ratio to a few. On the sign interval `I`, which has no
  poles, multiply by the product of the moving `C_k` (known sign), and solve the polynomial
  inequality from its real roots (degree `#ratios + 1`: 2 or 3 for the well-specified
  uniLasso). Use a closed form for degree 2 and polynomial roots for higher degree.
- Many moving ratios per row (misspecified `Σ`) could make that ill-conditioned. If so, fall
  back to bracketing the sign changes on `w_obs ± K` sd. This is deferred until needed.

## Phase 4: R

Done for the full-model uniLasso, in pure R (`R_pkg/R/unilasso.R`), using the C++ contrasts
already exposed: `UniLassoInference`, `unilasso_inference`, `unilasso_fit`, and a union of
intervals in `TruncBivariateNormal`. Tests in `tests/testthat/test_unilasso.R`, vignette
`vignettes/unilasso.Rmd`. Porting the row solver to `cpp/` can wait until it is a bottleneck.

## Open questions

Resolved:

- uniLasso parametrization: `D_j = λ / |β̂_j^uni|`, with `B = λ s_C`, `C = β̂^uni`.
- `B` and `C` are computed from the selection data.
- Finding 3, for the well-specified uniLasso: at most two ratios per row (finding 5).

Open:

1. Targets other than individual coefficients, or a misspecified `Σ`: how far do we go with
   many ratios per row?
