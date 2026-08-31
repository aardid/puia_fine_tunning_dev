"""
Phase D: multi-source TrAdaBoost/TransferBoost, local runner.

For each target (FWVZ, KRVZ, WIZ): boost shallow decision trees on the
combined source-pool + target training windows, with instance weights that
automatically downweight unhelpful source data — the automatic analogue of
Phase B's hand-run pool curation. The per-source final weight fractions are
the key diagnostic: does boosting rediscover the Whakaari harm on its own?

Protocol mirrors Phase C: background consensus from the model fitted on all
target data; each eruption's window spliced from the model fitted WITHOUT it
(+-1 month of target data excluded) — eruptions are out-of-sample. Features:
complete-case columns common to all five stations (as in B.2), drop_features
applied.

Stages (resumable): fit -> forecast -> eval
Usage:  python -u phase_d_local.py
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
root = r'U:\Research\EruptionForecasting\eruptions'
DATA_DIR = os.path.join(root, 'data')
FEAT_DIR = os.path.join(root, 'features')
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FORECAST_ROOT_B = os.path.join(SCRIPT_DIR, 'forecasts', 'phase_b')
OUT_DIR = os.path.join(SCRIPT_DIR, 'forecasts', 'phase_d')
LOCAL_CACHE = os.path.join(os.path.expanduser('~'), 'puia_local_cache')
FIT_DIR = os.path.join(LOCAL_CACHE, 'phase_d_models')

_spec = importlib.util.spec_from_file_location(
    'fine_tuning', os.path.join(SCRIPT_DIR, 'puia', 'fine_tuning.py'))
FT = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(FT)
sys.modules['fine_tuning'] = FT   # make TransferBoostEnsemble picklable

STATIONS = ['WIZ', 'FWVZ', 'KRVZ', 'ONTA', 'SHW']
data = {
    'WIZ':  ['2010-01-03', '2020-01-31'],
    'FWVZ': ['2006-01-01', '2015-12-31'],
    'KRVZ': ['2010-01-01', '2019-12-31'],
}
TARGETS = ['FWVZ', 'KRVZ', 'WIZ']
BASE_DIRS = {'FWVZ': 'FWVZ__full', 'KRVZ': 'KRVZ__full', 'WIZ': 'WIZ__full'}
data_streams = ['zsc2_rsamF', 'zsc2_mfF', 'zsc2_hfF', 'zsc2_dsarF']
drop_features = ['linear_trend_timewise', 'agg_linear_trend']
window = 2.
ths = np.linspace(0, 1, num=101)
N_ITER = 40
MAX_DEPTH = 6


def read_pickle_retry(f, attempts=5, wait=30):
    for a in range(attempts):
        try:
            return pd.read_pickle(f)
        except (OSError, EOFError, FileNotFoundError) as e:
            if a == attempts - 1:
                raise
            log.warning(f'    read failed ({e}), retry {a+1}/{attempts-1} in {wait}s: '
                        f'{os.path.basename(f)}')
            time.sleep(wait)


def parse_eruptions(sta):
    with open(os.path.join(DATA_DIR, f'{sta}_eruptive_periods.txt')) as fp:
        return [datetime.strptime(ln.rstrip(), '%Y %m %d %H %M %S')
                for ln in fp.readlines() if ln.strip()]


def apply_drop_features(fM):
    from fnmatch import fnmatch
    pats = [f'*__{d}__*' for d in drop_features]
    keep = [c for c in fM.columns
            if not any(fnmatch(c, p) for p in pats) and c not in drop_features]
    return fM[keep]


def common_features(fMs):
    common = fMs[STATIONS[0]].columns
    for sta in STATIONS[1:]:
        common = common.intersection(fMs[sta].columns)
    bad = set()
    for sta in STATIONS:
        sub = fMs[sta][common]
        bad |= set(sub.columns[sub.isna().any() | np.isinf(sub).any()])
    return [c for c in common if c not in bad]


# ============================================================
# Stage 1: fit boosted ensembles
# ============================================================
def stage_fit():
    log.info('=' * 60)
    log.info('STAGE FIT: multi-source TrAdaBoost')
    log.info('=' * 60)
    os.makedirs(FIT_DIR, exist_ok=True)

    fMs = {s: apply_drop_features(
        pd.read_pickle(os.path.join(LOCAL_CACHE, f'{s}_train_fM.pkl')))
        for s in STATIONS}
    yss = {s: pd.read_pickle(os.path.join(LOCAL_CACHE, f'{s}_train_ys.pkl'))
           for s in STATIONS}
    feats = common_features(fMs)
    log.info(f'complete-case common features: {len(feats)}')

    for target in TARGETS:
        sources = [s for s in STATIONS if s != target]
        Xs = np.vstack([fMs[s][feats].values for s in sources])
        ys = np.concatenate([yss[s]['label'].values > 0 for s in sources])
        src_sta = np.concatenate([[s] * len(fMs[s]) for s in sources])
        tes = parse_eruptions(target)
        t_idx = pd.DatetimeIndex(fMs[target].index)
        Xt_all = fMs[target][feats].values
        yt_all = yss[target]['label'].values > 0

        sets = {'all': np.ones(len(t_idx), dtype=bool)}
        for i, te in enumerate(tes):
            sets[f'loo{i}'] = ~((t_idx >= te - _MONTH) & (t_idx <= te + _MONTH))

        for sname, mask in sets.items():
            out = os.path.join(FIT_DIR, f'{target}__tboost__{sname}.pkl')
            if os.path.isfile(out):
                log.info(f'  {target}/{sname}: exists, skipping.')
                continue
            t0 = time.time()
            tb = FT.TransferBoostEnsemble(n_iterations=N_ITER,
                                          max_depth=MAX_DEPTH)
            tb.fit(Xs, ys, src_sta, Xt_all[mask], yt_all[mask], feats)
            with open(out, 'wb') as fp:
                pickle.dump(tb, fp)
            wf = tb.src_weight_history[-1]
            log.info(f'  {target}/{sname}: fitted ({time.time()-t0:.0f}s); '
                     f'final source weight fractions: '
                     + ' '.join(f'{k}={v:.2f}' for k, v in sorted(wf.items())))
    log.info('STAGE FIT complete.')


# ============================================================
# Stage 2: forecast
# ============================================================
def load_year(sta, yr, tf_clamp):
    fms = []
    for ds in data_streams:
        f = os.path.join(FEAT_DIR, f'fm_2.00w_{ds}_{sta}_{yr}.pkl')
        fm = read_pickle_retry(f)
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
    fM = fM.fillna(1.e-8)
    fM.sort_index(inplace=True)
    return fM


def stage_forecast():
    log.info('=' * 60)
    log.info('STAGE FORECAST')
    log.info('=' * 60)
    os.makedirs(OUT_DIR, exist_ok=True)
    for target in TARGETS:
        ti = datetime.strptime(data[target][0], '%Y-%m-%d')
        tf = datetime.strptime(data[target][1], '%Y-%m-%d')
        tes = parse_eruptions(target)
        models = {}
        for sname in ['all'] + [f'loo{i}' for i in range(len(tes))]:
            with open(os.path.join(FIT_DIR, f'{target}__tboost__{sname}.pkl'),
                      'rb') as fp:
                models[sname] = pickle.load(fp)

        for yr in range(ti.year, tf.year + 1):
            out = os.path.join(OUT_DIR, f'{target}_{yr}.pkl')
            if os.path.isfile(out):
                log.info(f'  {target} {yr}: exists, skipping.')
                continue
            t0 = time.time()
            fM = load_year(target, yr, tf)
            if fM.shape[0] == 0:
                continue
            cons = {'all': models['all'].predict_score(fM)}
            frames = {'all': pd.Series(cons['all'], index=fM.index)}
            for i, te in enumerate(tes):
                sub = fM[(fM.index >= te - _MONTH) & (fM.index <= te + 4 * _DAY)]
                if sub.shape[0]:
                    frames[f'loo{i}'] = pd.Series(
                        models[f'loo{i}'].predict_score(sub), index=sub.index)
            pd.to_pickle(frames, out)
            log.info(f'  {target} {yr}: {fM.shape[0]} rows ({time.time()-t0:.0f}s)')
            del fM
    log.info('STAGE FORECAST complete.')


# ============================================================
# Stage 3: eval
# ============================================================
def eruption_auc(con, tes):
    q95, bg_mask = [], np.ones(len(con), dtype=bool)
    for te in tes:
        m = (con.index >= te - window * _DAY) & (con.index < te)
        q95.append(con[m].quantile(0.95) if m.sum() else 0.)
        bg_mask &= ~m
    q95 = np.array(q95)
    bg = con.values[bg_mask]
    tpr = [(q95 >= th).mean() for th in ths]
    fpr = [(bg >= th).mean() for th in ths]
    return sum((fpr[i] - fpr[i + 1]) * tpr[i] for i in range(len(ths) - 1))


def stage_eval():
    log.info('=' * 60)
    log.info('STAGE EVAL: Phase D results')
    log.info('=' * 60)
    rows = []
    for target in TARGETS:
        tes = parse_eruptions(target)
        base = read_pickle_retry(os.path.join(
            FORECAST_ROOT_B, BASE_DIRS[target], 'consensus_master.pkl'))['consensus']
        auc_base = eruption_auc(base, tes)
        rows.append({'target': target, 'method': 'base(full pool)',
                     'auc': auc_base})
        log.info(f'  {target}/base: AUC = {auc_base:.4f}')

        files = sorted(glob(os.path.join(OUT_DIR, f'{target}_*.pkl')))
        bg_parts, loo_parts = [], {i: [] for i in range(len(tes))}
        for f in files:
            frames = read_pickle_retry(f)
            bg_parts.append(frames['all'])
            for i in range(len(tes)):
                if f'loo{i}' in frames:
                    loo_parts[i].append(frames[f'loo{i}'])
        master = pd.concat(bg_parts)
        master = master[~master.index.duplicated()].sort_index()
        for i in range(len(tes)):
            if not loo_parts[i]:
                continue
            loo = pd.concat(loo_parts[i])
            loo = loo[~loo.index.duplicated()].sort_index()
            master = master.drop(master.index[
                (master.index >= loo.index[0]) & (master.index <= loo.index[-1])])
            master = pd.concat([master, loo])
            master = master[~master.index.duplicated()].sort_index()
        master = master.to_frame('consensus')['consensus']
        auc = eruption_auc(master, tes)
        rows.append({'target': target, 'method': 'tboost', 'auc': auc})
        log.info(f'  {target}/tboost: AUC = {auc:.4f}')

    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, 'phase_d_results.csv'),
                              index=False)
    log.info('STAGE EVAL complete.')


if __name__ == '__main__':
    stage = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if stage in ('all', 'fit'):
        stage_fit()
    if stage in ('all', 'forecast'):
        stage_forecast()
    if stage in ('all', 'eval'):
        stage_eval()
    log.info('PHASE D done.')
