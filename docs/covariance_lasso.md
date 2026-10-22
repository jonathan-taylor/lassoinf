---
jupytext:
  main_language: python
  cell_metadata_filter: -all
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

# Selection from an estimate and its covariance

Often all we have from a fitted model is an estimate $\hat\beta$ and its
estimated covariance $\hat\Sigma$: the MLE of a GLM with its inverse
information, a sandwich estimate, or a bootstrap covariance. To first
order,

$$\hat\beta \sim N(\beta, \Sigma).$$

`covariance_lasso_inference` selects coefficients from these two
ingredients alone, with the lasso in *covariance form*, and then reports
p-values, confidence intervals and selection-corrected point estimates
(selective MLEs) for the selected coefficients.

## The randomized lasso in covariance form

Selecting on a random fraction $\pi$ of the data would use an estimate
$\hat\beta^*$ with variance $\Sigma/\pi$. To first order it is
$\hat\beta^* = \hat\beta + \omega$, with $\omega$ independent of
$\hat\beta$ and

$$\omega \sim N\Big(0, \frac{1-\pi}{\pi}\Sigma\Big).$$

We draw such an $\omega$, so the selection imitates a split at
`prop_select` $=\pi$ without holding any data out, and solve

$$\hat b = \mathop{\mathrm{argmin}}_b \; \frac{1}{2}(\hat\beta^* - b)^T\Sigma^{-1}(\hat\beta^* - b) + \sum_j D_j |b_j|.$$

The selected set is $E = \{j: \hat b_j \neq 0\}$. In the score
coordinates of `LassoInference` this is $Q = \Sigma^{-1}$,
$Z = \Sigma^{-1}\hat\beta$ with $\mathrm{Var}(Z) = \Sigma^{-1}$, and
`scalar_noise` $= (1-\pi)/\pi$.

### The penalty

At $b = 0$ the lasso compares the score $S = \Sigma^{-1}\hat\beta^*$ with the penalty $D$.
The score has covariance $\Sigma^{-1}/\pi$, so the penalty standardizes it:

$$D_j = f_j\, t\, \sqrt{(\Sigma^{-1})_{jj}/\pi},$$

with $f_j$ the `penalty_factor`. Nothing is selected exactly when
$\max_j |\xi_j|/f_j \le t$, for the standardized score
$\xi_j = S_j / \sqrt{(\Sigma^{-1})_{jj}/\pi}$. This is the analog of `glmnet`'s
standardized $X^Ty$: in least squares, with $\Sigma = \sigma^2(X^TX)^{-1}$, the score is
$X^Ty^*/\sigma^2$ and $\xi_j$ is the marginal $z$-statistic of feature $j$. Unpenalized
coefficients are profiled out and excluded ones removed, so $\Sigma^{-1}$ is replaced by
the Schur complement of the unpenalized coefficients in it.

By default the threshold is

$$t = \texttt{lambda\_mult}\times E\,\|\xi\|_\infty$$

under the global null, estimated by Monte Carlo: a Bonferroni-like threshold, which for
$p$ independent coordinates grows like $\sqrt{2\log p}$. For uncorrelated estimates
$(\Sigma^{-1})_{jj} = 1/\mathrm{Var}(\hat\beta_j)$, and $j$ is selected when
$|\hat\beta^*_j|/\mathrm{SE}(\hat\beta^*_j) > f_j t$.

### Targets

`target='selected'` (the default) reports, for $j \in E$, the coefficient
of $j$ in the model of the selected coefficients: $\beta_E$ projected in
the $\Sigma^{-1}$ metric,
$\beta_{E\cdot} = (\Sigma^{-1})_{EE}^{-1}(\Sigma^{-1}\beta)_E$. For a GLM
this is, to first order, the coefficient in the GLM refit on $X_E$.
`target='full'` reports $\beta_j$ itself, estimated by $\hat\beta_j$.

The selective MLE maximizes the likelihood of the estimate of the target
given the selection. It corrects the estimate for having been selected,
pulling it towards zero for coefficients that were selected because they
looked large.

## A logistic regression with many small effects

```{code-cell} ipython3
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm

from lassoinf import covariance_lasso_inference
```

We simulate a logistic regression with $p = 60$ correlated features: 4
strong effects, 16 small ones, and 40 nulls.

```{code-cell} ipython3
rng = np.random.default_rng(0)
n, p, rho = 3000, 60, 0.3
R = np.linalg.cholesky(rho ** np.abs(np.subtract.outer(np.arange(p), np.arange(p))))
names = [f'x{j}' for j in range(p)]
beta = np.r_[[0.25, -0.2, 0.2, -0.15],
             rng.choice([-1, 1], 16) * rng.uniform(0.02, 0.06, 16),
             np.zeros(40)]
signal = pd.Series(['strong'] * 4 + ['small'] * 16 + ['null'] * 40, index=names)

def sim_data(n):
    X = rng.standard_normal((n, p)) @ R.T
    y = rng.binomial(1, 1 / (1 + np.exp(-(-0.3 + X @ beta))))
    return X, y

X, y = sim_data(n)
```

We fit the logistic regression and keep the slopes and their covariance.
Leaving the intercept out of $\hat\beta$ and $\hat\Sigma$ profiles it
out.

```{code-cell} ipython3
def glm_fit(X, y):
    fit = sm.GLM(y, sm.add_constant(X), family=sm.families.Binomial()).fit()
    beta_hat = pd.Series(fit.params[1:], index=names)
    cov_beta = pd.DataFrame(fit.cov_params()[1:, 1:], index=names, columns=names)
    return beta_hat, cov_beta

beta_hat, cov_beta = glm_fit(X, y)
(beta_hat / np.sqrt(np.diag(cov_beta))).abs().sort_values(ascending=False).head(10).round(2)
```

### Selection and inference

```{code-cell} ipython3
res = covariance_lasso_inference(beta_hat, cov_beta, prop_select=0.8, level=0.9, rng=1)
print(f'threshold t = {res.threshold:.2f}, selected: {list(res.selected)}')
```

The summary has the unadjusted estimate of each target and its standard
error, the selective MLE, and the selective interval and p-value:

```{code-cell} ipython3
Q = np.linalg.inv(cov_beta.values)
E = res.inference.E
summary = res.summary_.copy()
summary['target'] = np.linalg.solve(Q[np.ix_(E, E)], (Q @ beta)[E])
summary['signal'] = signal[res.selected]
summary.round(3)
```

`splitting_` has the estimate that is independent of the selection, the
analog of the estimate on the held-out fraction $1-\pi$ of a split. Its
standard errors are $1/\sqrt{1-\pi}$ times larger:

```{code-cell} ipython3
res.splitting_.round(3)
```

With `target='full'` the targets are the coefficients of the full model:

```{code-cell} ipython3
full = covariance_lasso_inference(beta_hat, cov_beta, prop_select=0.8, level=0.9,
                                  target='full', rng=1)
full_summary = full.summary_.assign(target=beta[full.inference.E], signal=signal[full.selected])
full_summary.round(3)
```

The same seed gives the same randomization, so the same selection.

### Penalty factors and excluded coefficients

`penalty_factor` and `exclude` work as in `glmnet`. Here `x0` is always
kept (unpenalized), `x1` is never considered, and the small effects are
penalized half as much:

```{code-cell} ipython3
pf = np.ones(p)
pf[0] = 0
pf[4:20] = 0.5
res_pf = covariance_lasso_inference(beta_hat, cov_beta, prop_select=0.8, level=0.9,
                                    exclude=['x1'], penalty_factor=pf, rng=1)
res_pf.summary_.assign(signal=signal[res_pf.selected]).round(3)
```

## Calibration

We repeat the experiment, recording for each selected coefficient
whether its nominal 90% interval covers the full-model coefficient
$\beta_j$, and comparing with the naive interval
$\hat\beta_j \pm 1.645\,\mathrm{SE}(\hat\beta_j)$ after the same selection
and with the split interval.

```{code-cell} ipython3
q = norm.ppf(0.95)
rows, errors = [], []
for _ in range(100):
    X, y = sim_data(n)
    beta_hat, cov_beta = glm_fit(X, y)
    res = covariance_lasso_inference(beta_hat, cov_beta, prop_select=0.8, level=0.9,
                                     target='full', rng=rng)
    E = res.inference.E
    s, sp = res.summary_, res.splitting_
    for k, j in enumerate(E):
        est, se = s['estimate'].iloc[k], s['std_err'].iloc[k]
        ci = {'naive': (est - q * se, est + q * se),
                     'split': (sp['lower_conf'].iloc[k], sp['upper_conf'].iloc[k]),
                     'selective': (s['lower_conf'].iloc[k], s['upper_conf'].iloc[k])}
        for method, (lo, hi) in ci.items():
            rows.append({'method': method, 'signal': signal.iloc[j],
                         'covered': lo <= beta[j] <= hi, 'length': hi - lo})
        # errors signed by the sign of the estimate: positive means away from 0
        sign = np.sign(est)
        errors.append({'signal': signal.iloc[j],
                       'unadjusted': sign * (est - beta[j]),
                       'selective MLE': sign * (s['mle'].iloc[k] - beta[j])})
intervals, errors = pd.DataFrame(rows), pd.DataFrame(errors)
```

```{code-cell} ipython3
(intervals.groupby(['method', 'signal'])
          .agg(intervals=('covered', 'size'), coverage=('covered', 'mean'),
               mean_length=('length', 'mean'))
          .round(3))
```

The naive intervals undercover the small effects and the nulls that were
selected: they were selected because their estimates happened to be
large. The selective and split intervals are calibrated. The selective
intervals use all of the data and are shorter than the split ones, most
of all for the strong effects, whose selection was never in doubt.

Selection also biases the unadjusted estimates away from zero. The
table has the average error of each estimate, signed by the sign of the
unadjusted estimate, so a positive error is a bias away from zero:

```{code-cell} ipython3
errors.groupby('signal')[['unadjusted', 'selective MLE']].mean().round(4)
```

The selective MLE removes most of that bias.
