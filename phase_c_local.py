"""
Phase C: SER/STRUT tree refinement of the generalized ensembles, local runner.

For each target (FWVZ, KRVZ) and each base ensemble (leave-target-out 'full'
pool and the WIZ-free 'no_WIZ' pool from Phase B), refines the 300 trees with
target-volcano data using STRUT and SER (puia/fine_tuning.py), then evaluates
with the Phase-A-style protocol:

- Background consensus over the whole target record comes from the ensemble
  refined on ALL target data (mirrors Phase A's model 00, which saw all
  eruptions when producing the background).
- Each eruption's window (te-1month .. te+4days) is spliced from the ensemble
  refined WITHOUT that eruption (target windows within +-1 month of te
  excluded from refinement) — the eruption is forecast out-of-sample.
- 'mix' = mean of the SER and STRUT master consensus per base.

Note the comparison caveat: Phase B pool AUCs never saw ANY target data, while
Phase C (like Phase A) has target background in-sample. Refinement uses the
local training cache from phase_b_local.py stage 1.

Stages (resumable):  refine -> forecast -> eval
Usage:
    python -u phase_c_local.py            # all stages
    python -u phase_c_local.py refine
"""

import os
import sys
import time
import pickle
import logging
import warnings
import importlib.util
from glob import glob
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

_DAY = timedelta(days=1)
_MONTH = timedelta(days=365.25 / 12)

# ============================================================
# Paths / configuration
# ============================================================
root = r'U:\Research\EruptionForecasting\eruptions'
DATA_DIR = os.path.join(root, 'data')
FEAT_DIR = os.path.join(root, 'features')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_ROOT_B = os.path.join(SCRIPT_DIR, 'models', 'phase_b')
FORECAST_ROOT_B = os.path.join(SCRIPT_DIR, 'forecasts', 'phase_b')
FORECAST_ROOT_C = os.path.join(SCRIPT_DIR, 'forecasts', 'phase_c')

LOCAL_CACHE = os.path.join(os.path.expanduser('~'), 'puia_local_cache')
REFINED_DIR = os.path.join(LOCAL_CACHE, 'phase_c_refined')

# import fine_tuning directly (puia/__init__ needs tsfresh, unavailable here)
_ft_path = os.path.join(SCRIPT_DIR, 'puia', 'fine_tuning.py')
_spec = importlib.util.spec_from_file_location('fine_tuning', _ft_path)
FT = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(FT)

data = {
    'WIZ':  ['2010-01-03', '2020-01-31'],
    'FWVZ': ['2006-01-01', '2015-12-31'],
    'KRVZ': ['2010-01-01', '2019-12-31'],
    'ONTA': ['2013-01-10', '2014-12-18'],
    'SHW':  ['2004-01-02', '2005-12-30'],
}
TARGETS = ['FWVZ', 'KRVZ']
BASES = ['full', 'no_WIZ']
# method name -> (fine_tuning method, resample_ratio). Round 1 ran plain
# strut/ser (negative result); round 2 tests regularised variants:
# leaf-only recalibration, threshold shrinkage, per-tree undersampling.
VARIANTS = {
    'strut': ('strut', None),
    'ser': ('ser', None),
    'leaf': ('leaf', None),
    'strut_shrink': ('strut_shrink', None),
    'strut_us': ('strut', 0.75),
    'ser_us': ('ser', 0.75),
}
METHODS = ['leaf', 'strut_shrink', 'strut_us', 'ser_us']
RESULTS_CSV = 'phase_c_results_variants.csv'
data_streams = ['zsc2_rsamF', 'zsc2_mfF', 'zsc2_hfF', 'zsc2_dsarF']
window = 2.
ths = np.linspace(0, 1, num=101)


def parse_eruptions(sta):
    with open(os.path.join(DATA_DIR, f'{sta}_eruptive_periods.txt')) as fp:
        return [datetime.strptime(ln.rstrip(), '%Y %m %d %H %M %S')
                for ln in fp.readlines() if ln.strip()]


def station_window(sta):
    return (datetime.strptime(data[sta][0], '%Y-%m-%d'),
            datetime.strptime(data[sta][1], '%Y-%m-%d'))


def load_base_trees(target, base):
    mdir = os.path.join(MODEL_ROOT_B, f'{target}__{base}')
    trees = []
    for flp in sorted(glob(os.path.join(mdir, 'DecisionTreeClassifier_*.pkl'))):
        num = os.path.basename(flp).split('.')[0].split('_')[-1]
        clf = joblib.load(flp)
        with open(os.path.join(mdir, f'{num}.fts')) as fp:
            fts = [' '.join(ln.rstrip().split()[1:]) for ln in fp.readlines()]
        trees.append((clf, fts))
    return trees


def refit_sets(target, fM_index, tes):
    """Refinement row-masks: 'all', plus 'loo{i}' excluding +-1 month of te_i."""
    t = pd.DatetimeIndex(fM_index)
    sets = {'all': np.ones(len(t), dtype=bool)}
    for i, te in enumerate(tes):
        sets[f'loo{i}'] = ~((t >= te - _MONTH) & (t <= te + _MONTH))
    return sets


# ============================================================
# Stage 1: refine ensembles
# ============================================================
def stage_refine():
    log.info('=' * 60)
    log.info('STAGE REFINE: SER/STRUT refinement of base ensembles')
    log.info('=' * 60)
    os.makedirs(REFINED_DIR, exist_ok=True)

    for target in TARGETS:
        fM = pd.read_pickle(os.path.join(LOCAL_CACHE, f'{target}_train_fM.pkl'))
        ys = pd.read_pickle(os.path.join(LOCAL_CACHE, f'{target}_train_ys.pkl'))
        y = ys['label'].values > 0
        tes = parse_eruptions(target)
        sets = refit_sets(target, fM.index, tes)
        log.info(f'{target}: {len(fM)} refinement windows, {int(y.sum())} positive, '
                 f'{len(tes)} eruptions')

        for base in BASES:
            trees = None
            for method in METHODS:
                for sname, mask in sets.items():
                    out = os.path.join(
                        REFINED_DIR, f'{target}__{base}__{method}__{sname}.pkl')
                    if os.path.isfile(out):
                        log.info(f'  {target}/{base}/{method}/{sname}: exists, skipping.')
                        continue
                    if trees is None:
                        t0 = time.time()
                        trees = load_base_trees(target, base)
                        log.info(f'  {target}/{base}: {len(trees)} base trees '
                                 f'loaded ({time.time()-t0:.0f}s)')
                        # sanity: extracted tree must reproduce sklearn predictions
                        clf0, fts0 = trees[0]
                        X0 = FT.build_X(fM.iloc[:1000], fts0)
                        agree = (FT.RefinableTree.from_sklearn(clf0).predict(X0)
                                 == clf0.predict(X0)).mean()
                        log.info(f'  extraction self-check agreement: {agree:.4f}')
                        if agree < 0.999:
                            raise RuntimeError('tree extraction mismatch!')
                    t0 = time.time()
                    ft_method, rr = VARIANTS[method]
                    refined = FT.refine_ensemble(
                        [(c, f) for c, f in trees],
                        fM.loc[mask], y[mask], ft_method, resample_ratio=rr)
                    roots = [(rt.root, fts) for rt, fts in refined]
                    with open(out, 'wb') as fp:
                        pickle.dump(roots, fp)
                    log.info(f'  {target}/{base}/{method}/{sname}: refined '
                             f'({time.time()-t0:.0f}s)')
    log.info('STAGE REFINE complete.')


# ============================================================
# Stage 2: forecast
# ============================================================
def load_refined(target, base, method, sname):
    f = os.path.join(REFINED_DIR, f'{target}__{base}__{method}__{sname}.pkl')
    with open(f, 'rb') as fp:
        roots = pickle.load(fp)
    return [(FT.RefinableTree(root), fts) for root, fts in roots]


def _read_pickle_retry(f, attempts=5, wait=30):
    """The U:\\ share drops out intermittently — retry with a long pause."""
    for a in range(attempts):
        try:
            return pd.read_pickle(f)
        except (OSError, EOFError, FileNotFoundError) as e:
            if a == attempts - 1:
                raise
            log.warning(f'    read failed ({e}), retry {a+1}/{attempts-1} '
                        f'in {wait}s: {os.path.basename(f)}')
            time.sleep(wait)


def load_hires_year(sta, yr, tf_clamp):
    fms = []
    for ds in data_streams:
        f = os.path.join(FEAT_DIR, f'fm_2.00w_{ds}_{sta}_{yr}.pkl')
        fm = _read_pickle_retry(f)
        lo, hi = datetime(yr, 1, 1), min(datetime(yr + 1, 1, 1), tf_clamp)
        fm = fm[(fm.index >= lo) & (fm.index <= hi)]
        fm = fm.loc[:, ~fm.columns.duplicated()]
        fm = fm[~fm.index.duplicated()]
        fms.append(fm)
    idx = fms[0].index
    for fm in fms[1:]:
        idx = idx.intersection(fm.index)
    fms = [fm.loc[idx] for fm in fms]
    fM = pd.concat(fms, axis=1, sort=False)
    fM = fM.loc[:, ~fM.columns.duplicated()]
    fM.sort_index(inplace=True)
    return fM


def stage_forecast():
    log.info('=' * 60)
    log.info('STAGE FORECAST: refined ensembles over target records')
    log.info('=' * 60)
    os.makedirs(FORECAST_ROOT_C, exist_ok=True)

    for target in TARGETS:
        ti, tf = station_window(target)
        tes = parse_eruptions(target)

        # ensembles to run: background ('all') + per-eruption LOO
        jobs = {}   # (base, method, sname) -> refined ensemble
        outs = {}   # (base, method, sname) -> output path
        for base in BASES:
            for method in METHODS:
                for sname in ['all'] + [f'loo{i}' for i in range(len(tes))]:
                    out = os.path.join(FORECAST_ROOT_C,
                                       f'{target}__{base}__{method}__{sname}.pkl')
                    if not os.path.isfile(out):
                        jobs[(base, method, sname)] = load_refined(
                            target, base, method, sname)
                        outs[(base, method, sname)] = out
        if not jobs:
            log.info(f'{target}: all forecasts exist, skipping.')
            continue
        log.info(f'{target}: forecasting {len(jobs)} refined ensembles')

        partial = {k: [] for k in jobs}
        for yr in range(ti.year, tf.year + 1):
            t0 = time.time()
            fM = load_hires_year(target, yr, tf)
            if fM.shape[0] == 0:
                continue
            for (base, method, sname), ens in jobs.items():
                if sname == 'all':
                    sub = fM
                else:
                    i = int(sname.replace('loo', ''))
                    te = tes[i]
                    sub = fM[(fM.index >= te - _MONTH) & (fM.index <= te + 4 * _DAY)]
                    if sub.shape[0] == 0:
                        continue
                con = FT.predict_consensus(ens, sub)
                partial[(base, method, sname)].append(
                    pd.DataFrame(con, columns=['consensus'], index=sub.index))
            log.info(f'  {target} {yr}: {fM.shape[0]} rows x {len(jobs)} ensembles '
                     f'({time.time()-t0:.0f}s)')
            del fM

        for k, frames in partial.items():
            if not frames:
                log.warning(f'  {target} {k}: no forecast rows!')
                continue
            con = pd.concat(frames)
            con = con[~con.index.duplicated()]
            con.sort_index(inplace=True)
            con.to_pickle(outs[k])
            log.info(f'  {target} {"/".join(k)}: saved ({len(con)} rows)')
    log.info('STAGE FORECAST complete.')


# ============================================================
# Stage 3: splice + evaluate
# ============================================================
def eruption_auc(consensus, tes):
    l_fpr, l_tpr = [], []
    for th in ths:
        c_tp = c_fn = c_tn = c_fp = 0
        con = consensus.copy()
        for te in tes:
            inds = (con.index < te - window * _DAY) | (con.index >= te)
            subset = con.loc[~inds]
            _max = subset.quantile(q=0.95)['consensus'] if len(subset) > 0 else 0.
            if _max >= th:
                c_tp += 288
            else:
                c_fn += 288
            con = con.loc[inds]
        idx_bool = con['consensus'] < th
        c_tn += len(con[idx_bool])
        c_fp += len(con[~idx_bool])
        l_tpr.append(c_tp / (c_tp + c_fn) if (c_tp + c_fn) > 0 else 0.)
        l_fpr.append(c_fp / (c_fp + c_tn) if (c_fp + c_tn) > 0 else 0.)
    auc = 0.
    for i in range(len(l_fpr) - 1):
        auc += (l_fpr[i] - l_fpr[i + 1]) * l_tpr[i]
    return auc


def splice_master(target, base, method, tes):
    bg = pd.read_pickle(os.path.join(
        FORECAST_ROOT_C, f'{target}__{base}__{method}__all.pkl'))
    master = bg.copy()
    for i, te in enumerate(tes):
        f = os.path.join(FORECAST_ROOT_C, f'{target}__{base}__{method}__loo{i}.pkl')
        loo = pd.read_pickle(f)
        idx = master.index
        l1 = idx.searchsorted(loo.index[0])
        l2 = idx.searchsorted(loo.index[-1])
        l1 = max(0, l1)
        l2 = min(len(idx) - 1, l2)
        master.drop(master.index[list(range(l1, l2 + 1))], inplace=True)
        master = pd.concat([master, loo])
        master = master[~master.index.duplicated()]
        master.sort_index(inplace=True)
    return master


def stage_eval():
    log.info('=' * 60)
    log.info('STAGE EVAL: Phase C results')
    log.info('=' * 60)

    rows = []
    for target in TARGETS:
        tes = parse_eruptions(target)
        for base in BASES:
            # unrefined reference (leave-volcano-out, from Phase B)
            ref = pd.read_pickle(os.path.join(
                FORECAST_ROOT_B, f'{target}__{base}', 'consensus_master.pkl'))
            auc_base = eruption_auc(ref, tes)
            rows.append({'target': target, 'base': base,
                         'method': 'base (no refinement)', 'auc': auc_base})
            log.info(f'{target}/{base}/base: AUC = {auc_base:.4f}')

            masters = {}
            for method in METHODS:
                m = splice_master(target, base, method, tes)
                m.to_pickle(os.path.join(
                    FORECAST_ROOT_C, f'{target}__{base}__{method}__master.pkl'))
                masters[method] = m
                auc = eruption_auc(m, tes)
                rows.append({'target': target, 'base': base,
                             'method': method, 'auc': auc})
                log.info(f'{target}/{base}/{method}: AUC = {auc:.4f}')

            # mix = mean of SER and STRUT consensus (only when both ran)
            if 'strut' in masters and 'ser' in masters:
                mix = masters['strut'].copy()
                common = mix.index.intersection(masters['ser'].index)
                mix = mix.loc[common]
                mix['consensus'] = (mix['consensus']
                                    + masters['ser'].loc[common, 'consensus']) / 2.
                auc = eruption_auc(mix, tes)
                rows.append({'target': target, 'base': base, 'method': 'mix',
                             'auc': auc})
                log.info(f'{target}/{base}/mix: AUC = {auc:.4f}')

    df = pd.DataFrame(rows)
    out = os.path.join(FORECAST_ROOT_C, RESULTS_CSV)
    df.to_csv(out, index=False)
    log.info(f'Results saved to {out}')
    log.info('STAGE EVAL complete.')


if __name__ == '__main__':
    stage = sys.argv[1] if len(sys.argv) > 1 else 'all'
    log.info(f'PHASE C (local) — stage: {stage}')
    if stage in ('all', 'refine'):
        stage_refine()
    if stage in ('all', 'forecast'):
        stage_forecast()
    if stage in ('all', 'eval'):
        stage_eval()
    log.info('PHASE C done.')
