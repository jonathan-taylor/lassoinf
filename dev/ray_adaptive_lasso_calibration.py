import sys, time, warnings
import numpy as np
from scipy.stats import kstest
sys.path.insert(0, 'tests')
from test_ray_selection import _adaptive_lasso_case
from lassoinf import AffineConstraints
from lassoinf.ray_selection import lasso_event, ray_contrast_inference
warnings.simplefilter('ignore')
kappa, n = float(sys.argv[1]), int(sys.argv[2])
rng = np.random.default_rng(int(10 * kappa) + n)
p, lam = 4, 3. * np.sqrt(n / 80)
beta = np.r_[0.35, -0.25, 0., 0.] * np.sqrt(80 / n)
piv, naive, cover, pieces, t0 = [], [], [], [], time.time()
while len(piv) < 500:
    Q, Z, Y, b, penalty = _adaptive_lasso_case(rng, n, p, beta, lam, kappa)
    E = np.nonzero(b)[0]
    if len(E) == 0:
        continue
    si = AffineConstraints(Z=Z, Z_noisy=Y, Q=Q, Q_noise=None, scalar_noise=kappa)
    event = lasso_event(b, Q @ b - Y, Q, penalty)
    D_obs = penalty(Y)
    fixed = lasso_event(b, Q @ b - Y, Q, lambda Y: D_obs)          # treats the penalties as fixed
    for k in E:
        eta = np.linalg.solve(Q, np.eye(p)[k])
        r = ray_contrast_inference(si, eta, event, level=0.9, n_sd=8, n_grid=801)
        piv.append(r.pivot(beta[k])); cover.append(r.lower_conf <= beta[k] <= r.upper_conf); pieces.append(len(r.intervals))
        naive.append(ray_contrast_inference(si, eta, fixed, level=0.9, n_sd=8, n_grid=801).pivot(beta[k]))
print(f'adaptive lasso kappa={kappa} n={n}: {len(piv)} targets, {time.time()-t0:.0f}s; KS p {kstest(piv, "uniform").pvalue:.3g}, '
      f'coverage {np.mean(cover):.3f}; penalties treated as fixed: KS p {kstest(naive, "uniform").pvalue:.3g}; pieces {np.bincount(pieces).tolist()}')
