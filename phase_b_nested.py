"""
Phase B nested evaluation: pool selection WITHOUT peeking at the test eruption.

The Phase B landscape maxima (FWVZ<-{SHW} 0.963, KRVZ<-{ONTA,SHW} 0.885) are
oracle-curated: the pool was chosen on the same eruptions it is scored on.
This script runs the honest protocol: for each target eruption e, choose the
best pool using ONLY the remaining eruptions (selection metric = eruption AUC
with e removed from the event set), then score the held-out e with the pool
chosen without it. Reports per-fold selections (stability!) and compares:

  - nested-curated (pool chosen per fold without the test eruption)
  - oracle-curated (pool chosen on all eruptions — the biased number)
  - fixed full pool, fixed no_WIZ pool (a priori references)

Fold score: eruption AUC with the single held-out eruption as the event set;
background (FP/TN) is the full non-eruptive record for all rows, identical
across pools, so folds differ only in TP behaviour and pool choice.

Usage:  python -u phase_b_nested.py
"""

import os
import sys
import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

_DAY = timedelta(days=1)
root = r'U:\Research\EruptionForecasting\eruptions'
DATA_DIR = os.path.join(root, 'data')
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FORECAST_ROOT = os.path.join(SCRIPT_DIR, 'forecasts', 'phase_b')

TARGETS = ['FWVZ', 'KRVZ']
window = 2.
ths = np.linspace(0, 1, num=101)


def parse_eruptions(sta):
    with open(os.path.join(DATA_DIR, f'{sta}_eruptive_periods.txt')) as fp:
        return [datetime.strptime(ln.rstrip(), '%Y %m %d %H %M %S')
                for ln in fp.readlines() if ln.strip()]


def pool_variants(target):
    """All pool variant dirs for this target from the Phase B forecasts."""
    out = {}
    prefix = f'{target}__'
    for d in sorted(os.listdir(FORECAST_ROOT)):
        p = os.path.join(FORECAST_ROOT, d, 'consensus_master.pkl')
        if d.startswith(prefix) and os.path.isfile(p):
            out[d[len(prefix):]] = p
    return out


def prep(master_path, tes):
    """Per-pool statistics: q95 of consensus in each eruption's 2-day
    pre-window, and the background consensus array (all pre-windows removed)."""
    con = pd.read_pickle(master_path)['consensus']
    q95, bg_mask = [], np.ones(len(con), dtype=bool)
    for te in tes:
        m = (con.index >= te - window * _DAY) & (con.index < te)
        sub = con[m]
        q95.append(sub.quantile(0.95) if len(sub) else 0.)
        bg_mask &= ~m
    return np.array(q95), con.values[bg_mask]


def auc_from(q95_events, bg):
    """Eruption AUC over the standard threshold grid (right-Riemann, matching
    run_phase_a step 4)."""
    tpr = [(q95_events >= th).mean() if len(q95_events) else 0. for th in ths]
    fpr = [(bg >= th).mean() for th in ths]
    auc = 0.
    for i in range(len(ths) - 1):
        auc += (fpr[i] - fpr[i + 1]) * tpr[i]
    return auc


def main():
    all_rows = []
    for target in TARGETS:
        tes = parse_eruptions(target)
        K = len(tes)
        pools = pool_variants(target)
        log.info(f'{target}: {K} eruptions, {len(pools)} pools')

        stats = {name: prep(path, tes) for name, path in pools.items()}

        # oracle selection (all eruptions) — the biased pick
        oracle_pool = max(pools, key=lambda p: auc_from(*stats[p]))
        log.info(f'  oracle pool (biased): {oracle_pool} '
                 f'(AUC_all = {auc_from(*stats[oracle_pool]):.4f})')

        fold_rows = []
        for i in range(K):
            others = [j for j in range(K) if j != i]
            # select on the other eruptions only
            sel_pool = max(pools, key=lambda p: auc_from(
                stats[p][0][others], stats[p][1]))
            # score the held-out eruption
            def fold_auc(pool):
                return auc_from(stats[pool][0][[i]], stats[pool][1])
            row = {
                'target': target, 'fold': i,
                'te': tes[i].strftime('%Y-%m-%d'),
                'selected_pool': sel_pool,
                'auc_nested': fold_auc(sel_pool),
                'auc_oracle': fold_auc(oracle_pool),
                'auc_full': fold_auc('full'),
                'auc_no_WIZ': fold_auc('no_WIZ'),
            }
            fold_rows.append(row)
            log.info(f'  fold {i} (te={row["te"]}): selects {sel_pool:20s} '
                     f'nested={row["auc_nested"]:.4f} oracle={row["auc_oracle"]:.4f} '
                     f'full={row["auc_full"]:.4f} no_WIZ={row["auc_no_WIZ"]:.4f}')

        df = pd.DataFrame(fold_rows)
        means = {c: df[c].mean() for c in
                 ['auc_nested', 'auc_oracle', 'auc_full', 'auc_no_WIZ']}
        log.info(f'  {target} MEANS: nested={means["auc_nested"]:.4f} '
                 f'oracle={means["auc_oracle"]:.4f} full={means["auc_full"]:.4f} '
                 f'no_WIZ={means["auc_no_WIZ"]:.4f}')
        all_rows.append(df)

    out = os.path.join(FORECAST_ROOT, 'nested_evaluation.csv')
    pd.concat(all_rows).to_csv(out, index=False)
    log.info(f'saved {out}')


if __name__ == '__main__':
    main()
