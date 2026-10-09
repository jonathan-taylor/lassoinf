# Plan: selective inference with a data-dependent right-hand side

Branch: `data_dependent_penalty`. The math is in `docs/data_dependent_penalty.md`.

Goal: generalize the polyhedral lemma from `AX <= b` with `b` fixed to `AX <= B/C`, where `B`
and `C` are (asymptotically) jointly Gaussian with `X` and their signs are conditioned on. The
lasso (`B = D`, `C = 1`, no covariance) must remain the default, giving identical results.
The target application is the uniLasso.

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
   3.7 to 4.4 sd from the observed `w`, coming from a sum-of-ratios inactive row. So the code
   must handle unions, but in practice the extra pieces are rare and far in the tails.

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

## Phase 2: user-facing API

- `LassoInference` takes the extra inputs: per coordinate `B`, `C`, signs, and the
  covariances. The defaults reproduce the current behaviour exactly, which the existing tests
  will check.
- Return the truncation set (as intervals) in each contrast, for diagnostics.

## Phase 3: the lasso with data-dependent penalty factors

Depends on the decision about finding 3. The current proposal:

- Express each lasso constraint row as an affine function of `w` plus `Σ_k M_ik D_k(w)`, with
  `D_k(w) = B_k(w) / C_k(w)`. `M` comes from `lasso_post_selection_constraints` (columns
  `E` and the row's own inactive coordinate).
- Find each row's set on the sign interval `I`, which has no poles, by bracketing the sign
  changes of the rational function on a grid over `w_obs ± K` standard deviations, refined by
  `brentq`. Single-ratio rows use the closed form. Polynomial roots are an alternative for
  small `|E|`.
- uniLasso example and test: `C_j = β̂_j^uni` from the selection data, `B_j = λ s_C,j`, sign
  bounds from `s_C`. Check calibration by simulation.
- Docs: a uniLasso vignette.

## Phase 4: C++ and R

Port the row solver and the union-of-intervals truncated normal to `cpp/` (shared with R), and
expose them in `R_pkg`. This only happens once the Python API is settled.

## Open questions

1. **Finding 3.** Is the numerical route (a finite union of intervals per row, found by
   bracketing) acceptable for the lasso? Or did you have a reformulation in mind that keeps
   each row a single ratio, for example conditioning on more?
2. **uniLasso parametrization.** Is `D_j = λ / |β̂_j^uni|`, with `B = λ s_C` and `C = β̂^uni`,
   what you meant by "B = 1 and C are penalty factors roughly 1/cor"? That phrasing reads as
   `D = B/C = cor`, which is the reciprocal.
3. **Selection data.** In the randomized or carving setting, should `B` and `C` always be
   computed from the selection data, as finding 2 requires? Or do you also want the case where
   they come from the full data (the 2-d region)?
