---
jupytext:
  formats: md:myst,ipynb
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.1
kernelspec:
  name: python3
  display_name: Python 3 (ipykernel)
  language: python
---

# The polyhedral lemma with a data-dependent right-hand side

The polyhedral lemma of {cite}`LeeLasso`, as used throughout this package (see [](main.md)),
describes the law of a target $\hat{\theta}=\eta'Z$ conditional on a selection event
$\{AX \leq b\}$, where $X$ is Gaussian and $b$ is **fixed**. Here we let the right-hand side
depend on the data through a ratio:

$$
\left\{AX \leq B / C,\ \text{sign}(B) = s_B,\ \text{sign}(C) = s_C\right\},
$$

where the division and the inequality are taken row by row. $A$ is fixed; $B$ and $C$ are
(asymptotically) jointly Gaussian with $X$; and the signs of $B$ and $C$ are part of the
selection event, i.e. we also condition on them.

The motivating example is a lasso whose penalty factors are estimated from the data, such as
the uniLasso, where the penalty on feature $j$ is proportional to $1/|\hat{\beta}^{\text{uni}}_j|$,
the inverse of its univariate regression coefficient. The ordinary lasso is the special case
of $B$ constant and $C \equiv 1$.

+++

## The non-randomized case

Let $(X, B, C)$ be jointly Gaussian with the target $\hat{\theta} = \eta' X$, which has variance
$\sigma^2 = \text{Var}(\hat{\theta})$. As in {cite}`LeeLasso`, we decompose each vector into a
multiple of $\hat{\theta}$ plus a residual that is independent of it:

$$
\begin{aligned}
X &= N_X + \Gamma_X \hat{\theta}, & \Gamma_X &= \text{Cov}(X, \hat{\theta}) / \sigma^2, \\
B &= N_B + \Gamma_B \hat{\theta}, & \Gamma_B &= \text{Cov}(B, \hat{\theta}) / \sigma^2, \\
C &= N_C + \Gamma_C \hat{\theta}, & \Gamma_C &= \text{Cov}(C, \hat{\theta}) / \sigma^2.
\end{aligned}
$$

We condition on $N = (N_X, N_B, N_C)$. Compared with the usual lemma, the only new
inputs are $B$ and $C$ themselves and $\text{Cov}(B, \hat{\theta})$, $\text{Cov}(C, \hat{\theta})$.
These sit on top of $\text{Var}(\hat{\theta})$ and $\text{Cov}(X, \hat{\theta})$, which are already
required.

Fixing $N$ at its observed value, every quantity in the selection event is an affine function
of $t = \hat{\theta}$. For row $i$ write

$$
(AX)_i = u_i + a_i t, \qquad B_i = b_i + \beta_i t, \qquad C_i = c_i + \gamma_i t,
$$

with $u = A N_X$, $a = A \Gamma_X$, $b = N_B$, $\beta = \Gamma_B$, $c = N_C$, $\gamma = \Gamma_C$.

### One row

Row $i$ of the selection event is the set of $t$ with

$$
s_{B,i}(b_i + \beta_i t) > 0, \qquad s_{C,i}(c_i + \gamma_i t) > 0, \qquad
u_i + a_i t \leq \frac{b_i + \beta_i t}{c_i + \gamma_i t}.
$$

The two sign conditions each restrict $t$ to a half-line (or to all of $\mathbb{R}$, or to
nothing, when $\beta_i = 0$ or $\gamma_i = 0$). Their intersection is an interval $I_i$. On
$I_i$ the sign of $C_i$ is $s_{C,i}$, so multiplying the last inequality by $C_i$ gives the
equivalent condition

$$
s_{C,i}\, q_i(t) \leq 0, \qquad
q_i(t) = (c_i + \gamma_i t)(u_i + a_i t) - (b_i + \beta_i t)
       = \gamma_i a_i\, t^2 + (c_i a_i + \gamma_i u_i - \beta_i)\, t + (c_i u_i - b_i).
$$

This is a quadratic inequality in $t$. Its solution set depends on the sign of the leading
coefficient $\kappa_i = s_{C,i} \gamma_i a_i$:

* $\kappa_i > 0$: the set is the interval between the two roots (empty if there are no real
  roots). Intersected with $I_i$, row $i$ gives **one interval**.
* $\kappa_i < 0$: the set is the complement of the interval between the roots (all of
  $\mathbb{R}$ if there are no real roots). Intersected with $I_i$, row $i$ gives **at most
  two intervals**.
* $\kappa_i = 0$: the inequality is linear, and row $i$ gives one interval. In particular,
  with $\beta_i = \gamma_i = 0$ and $c_i = 1$ the row is $a_i t \leq b_i - u_i$, exactly the
  row of the usual polyhedral lemma.

So unlike the polyhedral case, a single row need not give an interval. Geometrically, on $I_i$
the map $t \mapsto B_i/C_i$ is one branch of a hyperbola (convex or concave, and monotone), and
the event asks that a line lie below it. A line can cross a convex branch twice.

*Example.* Take $B_i \equiv 1$, $C_i = t$ with $s_{C,i} = +1$, so $I_i = (0, \infty)$, and
$(AX)_i = 3 - t$. Then $3 - t \leq 1/t$ on $t > 0$ if and only if $t^2 - 3t + 1 \geq 0$,
so row $i$ holds on $(0, 0.382] \cup [2.618, \infty)$.

The sign conditions are what keep this tractable. Without them, $C_i$ could cross zero and
$B_i / C_i$ would have a pole inside the range of $t$. With them, every row is a union of at
most two intervals with endpoints in closed form. They are also natural in the examples:
in the uniLasso, the sign of $\hat{\beta}^{\text{uni}}_j$ fixes the sign constraint on
$\beta_j$.

### The truncation set

The selection event is the intersection over rows,

$$
S = \bigcap_i R_i, \qquad R_i = I_i \cap \{t : s_{C,i}\, q_i(t) \leq 0\},
$$

which is a finite union of disjoint intervals $S = \bigcup_k [l_k, r_k]$. Conditionally on $N$
and the selection event, $\hat{\theta}$ is $N(\theta, \sigma^2)$ truncated to $S$. The pivot is

$$
F_\theta(\hat{\theta}) =
\frac{\sum_k \left[\Phi\left(\frac{\min(r_k, \hat{\theta}) - \theta}{\sigma}\right) - \Phi\left(\frac{l_k - \theta}{\sigma}\right)\right]_+}
     {\sum_k \left[\Phi\left(\frac{r_k - \theta}{\sigma}\right) - \Phi\left(\frac{l_k - \theta}{\sigma}\right)\right]}
\sim \text{Unif}(0, 1),
$$

and intervals and $p$-values follow by inverting it in $\theta$ as usual.

+++

## The randomized case

In this package the selection event is a function of the randomized score $Z + \omega$, with
$\omega | Z \sim N(0, \bar{\Sigma})$ and target $\hat{\theta} = \eta'Z$ (see [](main.md)). There,
$(Z, \omega)$ is decomposed into independent pieces $(\hat{\theta}, N, \bar{\omega}, \bar{N})$, with
$\bar{\omega} = c'\omega$ of variance $\bar{s}^2$, and

$$
Z + \omega = N + \bar{N} + \Gamma \hat{\theta} + \bar{\Gamma} \bar{\omega},
\qquad \Gamma = \frac{\Sigma \eta}{\sigma^2}, \quad \bar{\Gamma} = \frac{\Sigma \eta}{\bar{s}^2}.
$$

Since $\Gamma \hat{\theta} = \bar{\Gamma} \cdot (\bar{s}^2/\sigma^2)\, \hat{\theta}$, the
selection data depend on $(\hat{\theta}, \bar{\omega})$ only through the scalar

$$
w = \frac{\bar{s}^2}{\sigma^2} \hat{\theta} + \bar{\omega},
\qquad Z + \omega = N + \bar{N} + \bar{\Gamma} w,
$$

and the event is an interval in $w$. The code computes it at $\hat{\theta} = 0$, which gives
the interval for $w$ directly, and `TruncBivariateNormal` gives the law of $\hat{\theta}$ given
$w$ in that interval.

**The same reduction holds for $B$ and $C$ when they are functions of the selection data**
$Z + \omega$, plus noise independent of $(Z, \omega)$. Then

$$
\text{Cov}(B, \hat{\theta}) = \text{Cov}(B, \bar{\omega}) = k_B,
$$

because $\text{Cov}(Z + \omega, \eta'Z) = \Sigma \eta = \bar{\Sigma} c = \text{Cov}(Z + \omega, c'\omega)$.
Regressing $B$ on the independent pair $(\hat{\theta}, \bar{\omega})$ gives

$$
B = N_B + \frac{k_B}{\sigma^2} \hat{\theta} + \frac{k_B}{\bar{s}^2} \bar{\omega}
  = N_B + \frac{k_B}{\bar{s}^2}\, w,
$$

and the same for $C$. So, conditionally on $(N, \bar{N}, N_B, N_C)$, everything in the selection
event is an affine function of $w$. The non-randomized construction then applies with $t$
replaced by $w$ and the coefficients

$$
a = A \bar{\Gamma}, \qquad \beta = k_B / \bar{s}^2, \qquad \gamma = k_C / \bar{s}^2.
$$

This yields a truncation set $S$ for $w$ that is a finite union of intervals. $\hat{\theta}$
and $\bar{\omega}$ remain independent Gaussians conditioned on $w \in S$. Since $S$ is a
disjoint union, every probability `TruncBivariateNormal` needs is a sum over its intervals,
e.g.

$$
P\left(\hat{\theta} > x,\ w \in S\right) = \sum_k P\left(\hat{\theta} > x,\ l_k \leq w \leq r_k\right),
$$

and the conditional moments are sums in the same way.

If $B$ or $C$ depends on $Z$ other than through $Z + \omega$, then
$\text{Cov}(B, \hat{\theta}) \neq \text{Cov}(B, \bar{\omega})$. The event then depends on
$(\hat{\theta}, \bar{\omega})$ through more than $w$: it is a region of the plane bounded by
conics. This is not covered here. For the uniLasso it means the univariate coefficients must
be computed from the same (randomized) data used for selection.

### Inputs

On top of what `LassoInference` already takes, the generalization needs, for the rows of the
constraint:

* the observed $B$ and $C$;
* the conditioned signs $s_B$ and $s_C$, defaulting to the observed signs. $s_{B,i} = 0$ means
  the sign of $B_i$ is not conditioned on, which is appropriate when $B_i$ is constant;
* $\text{Cov}(B, Z)$ and $\text{Cov}(C, Z)$, so that $k_B = \text{Cov}(B, Z)\eta$ and
  $k_C = \text{Cov}(C, Z)\eta$ for every target $\eta$. These default to zero.

The defaults, $C = 1$ and $\text{Cov}(B, Z) = \text{Cov}(C, Z) = 0$ with $B = b$, recover the
usual polyhedral lemma exactly.

+++

## The lasso with data-dependent penalty factors

For the weighted lasso the penalty is $\sum_j D_j |\beta_j|$, and the usual constraints
`lasso_post_selection_constraints` returns are $A(Z + \omega) \leq b$ with $b$ **linear** in $D$.
Writing $W = Q_{EE}^{-1}$, the active rows involve $c_E = W(\dots + D_E s_E)$, and the inactive
rows involve $D_j$ and $Q_{jE} c_E$. So

$$
b = b_0 + M D
$$

for a fixed matrix $M$. Here $M$ is dense in the active columns: every row depends on every
active penalty $D_E$. An inactive penalty $D_j$ appears only in the two rows for coordinate $j$.

Now let the penalty factors be data-dependent ratios, $D_k = B_k / C_k$, with the signs of
$B_k$, $C_k$ conditioned on. Then:

* **Inactive penalties enter as single ratios.** A row involving only $D_j$ for an inactive
  $j$ has exactly the form above. This covers all rows if $E$ is empty, or if the active
  penalties $D_E$ are fixed.
* **Active penalties enter every row, as a sum of ratios.** In general row $i$ reads
  $(AX)_i \leq b_{0,i} + \sum_k M_{ik} B_k/C_k$, and is not a single ratio.

The construction still goes through, with a polynomial in place of the quadratic. On the
interval $I$ where all the conditioned signs hold, no $C_k$ vanishes. Multiplying row $i$ by
$\prod_k |C_k(w)|$ over the ratios that appear in it gives a polynomial inequality of degree at
most $|E| + 2$. Each row is again a finite union of intervals, and so is $S$. What is lost is
the closed form: the roots must be found numerically, either as polynomial roots or by
bracketing the sign changes of the rational function directly on $I$. Bracketing is better
conditioned when $|E|$ is large. The density of $w$ is negligible outside a few standard
deviations of its observed value, so only a bounded part of $I$ needs searching.

### The uniLasso

The uniLasso fits univariate regressions $\hat{\beta}^{\text{uni}}_j$ and then a lasso with
penalty factors proportional to $1/|\hat{\beta}^{\text{uni}}_j|$ and sign constraints
$\text{sign}(\beta_j) \in \{0, \text{sign}(\hat{\beta}^{\text{uni}}_j)\}$. In the notation
above:

$$
C_j = \hat{\beta}^{\text{uni}}_j, \qquad s_{C,j} = \text{sign}(\hat{\beta}^{\text{uni}}_j), \qquad
B_j = \lambda\, s_{C,j}, \qquad D_j = B_j / C_j = \lambda / |\hat{\beta}^{\text{uni}}_j|.
$$

$B$ is constant, so $s_B = 0$ and $\text{Cov}(B, Z) = 0$. Conditioning on $s_C$ also fixes the
sign constraints, so the bounds $L$, $U$ are fixed given the event. For a Gaussian linear model
with score $Z = X'y$, $\hat{\beta}^{\text{uni}}_j = Z_j / \|X_j\|_2^2$ is linear in $Z$. When it
is computed from the selection data it is linear in $Z + \omega$, so the randomized reduction
applies exactly, with $\text{Cov}(C, Z) = \text{diag}(\|X_j\|_2^{-2})\, \Sigma$.

**How many intervals?** Here $B$ is constant, and on its sign branch $B_j/C_j = \lambda/|C_j(w)|$ is
positive, convex, and infinite at the pole. A row asking that a line lie below it can still
hold on two pieces. That doesn't happen when the line is proportional to $C_j(w)$: for an
inactive $j$ with $E$ empty, the score is $\|X_j\|_2^2 C_j$, and $k C \leq \lambda / C$ with
$C > 0$ is a single interval. A brute-force check of the full uniLasso event along $w$
(450 random instances, $p$ from 6 to 15, feature correlation up to $0.95$) found a single
interval in all but one instance. In that one, a second piece lay 3.7 to 4.4 standard
deviations from the observed $w$. It came from an inactive row involving all five active
penalties, and it ended at the point where another feature's univariate coefficient changes
sign. So a single interval is typical, but not guaranteed.

+++

## Summary

| | $b$ fixed (Lee et al.) | single ratio $B/C$ | lasso with $D = B/C$ |
|---|---|---|---|
| one row | interval | $\leq 2$ intervals, quadratic roots | finite union, polynomial of degree $\leq \lvert E\rvert + 2$ |
| truncation set | interval | finite union of intervals | finite union of intervals |
| extra inputs | none | $B$, $C$, $s_B$, $s_C$, $\text{Cov}(B, Z)$, $\text{Cov}(C, Z)$ | same, per coordinate |
| randomized case | reduces to $w$ | reduces to $w$ if $B$, $C$ are functions of $Z + \omega$ | same |
