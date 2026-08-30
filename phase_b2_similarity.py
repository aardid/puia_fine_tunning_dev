"""
Phase B.2: feature-space similarity between volcanoes.

Computes the Maximum Mean Discrepancy (MMD, RBF kernel with median heuristic)
between every pair of stations' training-window feature distributions, using
the local cache built by phase_b_local.py stage 1. Two matrices:
  - non-eruptive windows (label 0) — the bulk background behaviour
  - pre-eruptive windows (label 1) — the precursory signature (few samples!)

Also correlates each target's per-source MMD with the B.1 ablation AUC lifts.

Outputs (to forecasts/phase_b/):
  mmd_noneruptive.csv, mmd_preeruptive.csv, mmd_vs_lift.csv

Usage:
    python -u phase_b2_similarity.py
"""

import os
import sys
import logging
from fnmatch import fnmatch

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FORECAST_ROOT = os.path.join(SCRIPT_DIR, 'forecasts', 'phase_b')
LOCAL_CACHE = os.path.join(os.path.expanduser('~'), 'puia_local_cache')

STATIONS = ['WIZ', 'FWVZ', 'KRVZ', 'ONTA', 'SHW']
TARGETS = ['FWVZ', 'KRVZ']
drop_features = ['linear_trend_timewise', 'agg_linear_trend']

RNG = np.random.default_rng(0)
MAX_N = 1200          # subsample cap per station for the O(n^2) kernel
EPS = 1e-12


def apply_drop_features(fM):
    pats = [f'*__{d}__*' for d in drop_features]
    keep = [c for c in fM.columns
            if not any(fnmatch(c, p) for p in pats) and c not in drop_features]
    return fM[keep]


def load_station(sta):
    fM = pd.read_pickle(os.path.join(LOCAL_CACHE, f'{sta}_train_fM.pkl'))
    ys = pd.read_pickle(os.path.join(LOCAL_CACHE, f'{sta}_train_ys.pkl'))
    fM = apply_drop_features(fM)
    return fM, ys['label'].values > 0


def _sqdist(A, B):
    """Pairwise squared distances via the Gram-matrix identity (memory-safe:
    never materialises an (n, m, d) array)."""
    aa = (A ** 2).sum(1)[:, None]
    bb = (B ** 2).sum(1)[None, :]
    d2 = aa + bb - 2.0 * (A @ B.T)
    np.maximum(d2, 0., out=d2)
    return d2


def mmd_rbf(X, Y):
    """Unbiased MMD^2 with RBF kernel, bandwidth by median heuristic on the
    pooled sample. X, Y: (n,d), (m,d) float arrays, no NaN."""
    Z = np.vstack([X, Y])
    # median pairwise distance on a subsample for the bandwidth
    idx = RNG.choice(len(Z), size=min(len(Z), 800), replace=False)
    D = np.sqrt(_sqdist(Z[idx], Z[idx]))
    sigma = np.median(D[D > 0])
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = 1.0
    gamma = 1.0 / (2 * sigma ** 2)

    def k(A, B):
        return np.exp(-gamma * _sqdist(A, B))

    Kxx = k(X, X); Kyy = k(Y, Y); Kxy = k(X, Y)
    n, m = len(X), len(Y)
    np.fill_diagonal(Kxx, 0); np.fill_diagonal(Kyy, 0)
    mmd2 = (Kxx.sum() / (n * (n - 1)) + Kyy.sum() / (m * (m - 1))
            - 2 * Kxy.mean())
    return max(mmd2, 0.)


def main():
    os.makedirs(FORECAST_ROOT, exist_ok=True)
    log.info('Loading cached training matrices...')
    fMs, labels = {}, {}
    for sta in STATIONS:
        fMs[sta], labels[sta] = load_station(sta)
        log.info(f'  {sta}: {fMs[sta].shape}, positives {int(labels[sta].sum())}')

    # common feature set across all stations, complete cases only
    common = fMs[STATIONS[0]].columns
    for sta in STATIONS[1:]:
        common = common.intersection(fMs[sta].columns)
    log.info(f'Common features: {len(common)}')

    # per-station arrays with NaN columns removed globally
    nan_cols = set()
    for sta in STATIONS:
        sub = fMs[sta][common]
        nan_cols |= set(sub.columns[sub.isna().any() | np.isinf(sub).any()])
    feats = [c for c in common if c not in nan_cols]
    log.info(f'Complete-case features used: {len(feats)}')

    # global standardisation (pooled mean/std) so MMD is scale-free
    pooled = pd.concat([fMs[sta][feats] for sta in STATIONS])
    mu, sd = pooled.mean(), pooled.std().replace(0, 1)
    X = {sta: ((fMs[sta][feats] - mu) / sd).values for sta in STATIONS}

    results = {}
    for kind, sel in [('noneruptive', lambda s: ~labels[s]),
                      ('preeruptive', lambda s: labels[s])]:
        M = pd.DataFrame(index=STATIONS, columns=STATIONS, dtype=float)
        for i, a in enumerate(STATIONS):
            for b in STATIONS[i:]:
                Xa = X[a][sel(a)]
                Xb = X[b][sel(b)]
                if len(Xa) > MAX_N:
                    Xa = Xa[RNG.choice(len(Xa), MAX_N, replace=False)]
                if len(Xb) > MAX_N:
                    Xb = Xb[RNG.choice(len(Xb), MAX_N, replace=False)]
                if a == b:
                    # split-half self-MMD as the noise floor
                    half = len(Xa) // 2
                    v = mmd_rbf(Xa[:half], Xa[half:]) if half >= 5 else np.nan
                elif len(Xa) >= 4 and len(Xb) >= 4:  # n=4 estimates are very noisy
                    v = mmd_rbf(Xa, Xb)
                else:
                    v = np.nan
                M.loc[a, b] = v
                M.loc[b, a] = v
                log.info(f'  MMD^2[{kind}] {a}-{b} = {v if v==v else float("nan"):.4f} '
                         f'(n={len(Xa)},{len(Xb)})')
        out = os.path.join(FORECAST_ROOT, f'mmd_{kind}.csv')
        M.to_csv(out)
        results[kind] = M
        log.info(f'{kind} MMD matrix saved to {out}')

    # correlate with B.1 ablation lifts
    rows = []
    for target in TARGETS:
        abl = pd.read_csv(os.path.join(FORECAST_ROOT, f'ablation_{target}.csv'))
        abl = abl[abl.variant.str.startswith('no_')].copy()
        abl['source'] = abl.variant.str.replace('no_', '', regex=False)
        for _, r in abl.iterrows():
            rows.append({
                'target': target, 'source': r['source'],
                'auc_lift': r['auc_lift_of_dropped_source'],
                'mmd_noneruptive': results['noneruptive'].loc[target, r['source']],
                'mmd_preeruptive': results['preeruptive'].loc[target, r['source']],
            })
    df = pd.DataFrame(rows)
    for col in ['mmd_noneruptive', 'mmd_preeruptive']:
        ok = df[[col, 'auc_lift']].dropna()
        if len(ok) >= 3:
            rho = ok[col].corr(ok['auc_lift'], method='spearman')
            log.info(f'Spearman({col}, auc_lift) = {rho:.3f} (n={len(ok)})')
    out = os.path.join(FORECAST_ROOT, 'mmd_vs_lift.csv')
    df.to_csv(out, index=False)
    log.info(f'MMD-vs-lift table saved to {out}')


if __name__ == '__main__':
    main()
