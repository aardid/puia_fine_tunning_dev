"""
Fill model-00 forecast coverage gaps (WIZ 2018-2020, KRVZ 2014-2019) without puia.

Crashed step-1 runs left the model-00 forecasts partial (forecast_complete()
passes if ANY consensus file exists). The cached feature matrices for the
missing years already contain full hi-res (10-min) coverage, so this script
replicates puia.model.forecast_models directly: load cached features, predict
with the 300 frozen model-00 trees, save consensus_{year}.pkl.

Notes:
- Per-tree forecast files (DecisionTreeClassifier_XXXX_{yr}.pkl) are NOT saved,
  only the consensus. A later forecast() with recalculate=False will simply
  re-predict those years (features are cached, so no tsfresh cost).
- Runs in any env with pandas/numpy/sklearn/joblib (no tsfresh needed). Trees
  were saved with sklearn 1.6.1; loading under a close version is fine.

Usage:
    python -u fill_model00_gaps.py
"""

import os
import sys
import time
import logging
import warnings
from glob import glob
from datetime import datetime

import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings('ignore')  # sklearn version / feature-name warnings

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

root = r'U:\Research\EruptionForecasting\eruptions'
FEAT_DIR = os.path.join(root, 'features')
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, 'models', 'cve_WIZ_FWVZ_KRVZ_ONTA_SHW', '00')
FORECAST_DIR = os.path.join(SCRIPT_DIR, 'forecasts', 'cve_WIZ_FWVZ_KRVZ_ONTA_SHW', '00')

data_streams = ['zsc2_rsamF', 'zsc2_mfF', 'zsc2_hfF', 'zsc2_dsarF']

# station -> (missing years, data window end clamp)
GAPS = {
    'WIZ':  ([2018, 2019, 2020], datetime(2020, 1, 31)),
    'KRVZ': ([2014, 2015, 2016, 2017, 2018, 2019], datetime(2019, 12, 31)),
}


def load_trees():
    """Load all model-00 trees and their per-tree feature lists once."""
    tree_files = sorted(glob(os.path.join(MODEL_DIR, 'DecisionTreeClassifier_*.pkl')))
    log.info(f'Loading {len(tree_files)} trees from {MODEL_DIR}...')
    trees = []
    for flp in tree_files:
        num = os.path.basename(flp).split('.')[0].split('_')[-1]
        for attempt in range(3):
            try:
                model = joblib.load(flp)
                break
            except OSError:
                if attempt < 2:
                    time.sleep(5)
                else:
                    raise
        with open(os.path.join(MODEL_DIR, f'{num}.fts')) as fp:
            fts = [' '.join(ln.rstrip().split()[1:]) for ln in fp.readlines()]
        trees.append((num, model, fts))
    log.info(f'  {len(trees)} trees loaded.')
    return trees


def load_feature_year(sta, yr, tf_clamp):
    """Assemble the 4-stream feature matrix for one station-year (10-min rows)."""
    fms = []
    for ds in data_streams:
        f = os.path.join(FEAT_DIR, f'fm_2.00w_{ds}_{sta}_{yr}.pkl')
        t0 = time.time()
        fm = pd.read_pickle(f)
        # restrict to this year (files can carry rows from other years) and
        # clamp to the pool data window
        lo, hi = datetime(yr, 1, 1), min(datetime(yr + 1, 1, 1), tf_clamp)
        fm = fm[(fm.index >= lo) & (fm.index <= hi)]
        fm = fm.loc[:, ~fm.columns.duplicated()]
        fm = fm[~fm.index.duplicated()]
        log.info(f'    {os.path.basename(f)}: {fm.shape[0]} rows in-year '
                 f'({time.time()-t0:.0f}s)')
        fms.append(fm)
    # align on common index across streams
    idx = fms[0].index
    for fm in fms[1:]:
        idx = idx.intersection(fm.index)
    fms = [fm.loc[idx] for fm in fms]
    fM = pd.concat(fms, axis=1, sort=False)
    fM = fM.loc[:, ~fM.columns.duplicated()]
    fM = fM.fillna(1.e-8)
    fM.sort_index(inplace=True)
    return fM


def check_coverage(fM, yr, tf_clamp):
    """Report 10-min coverage of the assembled matrix for sanity."""
    lo, hi = datetime(yr, 1, 1), min(datetime(yr + 1, 1, 1), tf_clamp)
    expected = int((hi - lo).total_seconds() / 600) + 1
    frac = len(fM) / expected if expected > 0 else 0.
    log.info(f'    coverage: {len(fM)}/{expected} 10-min windows ({frac:.1%})')
    if frac < 0.9:
        log.warning(f'    coverage below 90% for {yr} — consensus will be sparse there')


def forecast_year(sta, yr, tf_clamp, trees):
    out = os.path.join(FORECAST_DIR, sta, f'consensus_{yr}.pkl')
    if os.path.isfile(out):
        log.info(f'  {sta} {yr}: consensus already exists, skipping.')
        return
    log.info(f'  {sta} {yr}: loading features...')
    fM = load_feature_year(sta, yr, tf_clamp)
    if fM.shape[0] == 0:
        log.warning(f'  {sta} {yr}: no in-year rows in feature cache, skipping!')
        return
    check_coverage(fM, yr, tf_clamp)

    log.info(f'  {sta} {yr}: predicting with {len(trees)} trees...')
    t0 = time.time()
    total = np.zeros(fM.shape[0])
    for num, model, fts in trees:
        missing = [f for f in fts if f not in fM.columns]
        for f in missing:
            fM[f] = 0.
        total += model.predict(fM[fts]).astype(float)
    consensus = pd.DataFrame(total / len(trees), columns=['consensus'], index=fM.index)
    consensus.index.name = 'time'
    log.info(f'  {sta} {yr}: prediction done ({time.time()-t0:.0f}s), '
             f'consensus mean={consensus["consensus"].mean():.3f}')

    # atomic-ish save: write local then move is unreliable on U:, write direct with retry
    for attempt in range(3):
        try:
            consensus.to_pickle(out)
            pd.read_pickle(out)  # verify not truncated
            break
        except (OSError, EOFError):
            if attempt < 2:
                time.sleep(5)
            else:
                raise
    log.info(f'  {sta} {yr}: saved {out}')


if __name__ == '__main__':
    log.info('=' * 60)
    log.info('Filling model-00 forecast coverage gaps')
    log.info('=' * 60)
    trees = load_trees()
    for sta, (years, tf_clamp) in GAPS.items():
        log.info(f'Station {sta}: years {years}')
        for yr in years:
            forecast_year(sta, yr, tf_clamp, trees)
    log.info('ALL GAPS FILLED.')
    log.info('Next: delete _consensus_master_WIZ.pkl and _consensus_master_KRVZ.pkl, '
             'then re-run run_phase_a_steps34.py')
