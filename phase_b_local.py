"""
Phase B.1: per-source-volcano ablation, runnable on a local machine (no server,
no tsfresh, no puia import).

For each target volcano (FWVZ=Ruapehu, KRVZ=Tongariro), trains leave-one-
volcano-out ensembles: the full source pool minus the target ("full"), and one
variant additionally dropping each single source S. Forecasts the target's
whole record and computes the Phase-A-style eruption ROC/AUC per variant.
AUC_lift(S) = AUC_full - AUC_without_S ranks which sources help vs. harm.

Replicates puia.model.train_one_model exactly (same undersampler seeds,
Mann-Whitney feature selection, GridSearchCV grid) but reads training windows
straight from the cached feature-matrix pickles, so it runs in any env with
pandas/numpy/scipy/sklearn/imbalanced-learn.

Stages (resumable; each skips work already done):
  1. cache   — extract training-resolution rows (12-h grid) per station from
               the big cached year files on U:\\ into a local disk cache
  2. train   — train 300 trees per variant (10 variants)
  3. forecast— stream target hires features year by year, predict all
               variants, save per-variant master consensus, compute AUC table

Usage:
    python -u phase_b_local.py            # all stages
    python -u phase_b_local.py cache      # just stage 1
"""

import os
import sys
import time
import logging
import warnings
from glob import glob
from fnmatch import fnmatch
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import joblib
from scipy.stats import mannwhitneyu
from imblearn.under_sampling import RandomUnderSampler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, ShuffleSplit

warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

_DAY = timedelta(days=1)

# ============================================================
# Paths / configuration — keep in sync with run_phase_a_server.py
# ============================================================
root = r'U:\Research\EruptionForecasting\eruptions'
DATA_DIR = os.path.join(root, 'data')
FEAT_DIR = os.path.join(root, 'features')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_ROOT = os.path.join(SCRIPT_DIR, 'models', 'phase_b')
FORECAST_ROOT = os.path.join(SCRIPT_DIR, 'forecasts', 'phase_b')

# local (fast-disk) cache of training matrices; reused by later phases
LOCAL_CACHE = os.path.join(os.path.expanduser('~'), 'puia_local_cache')

data = {
    'WIZ':  ['2010-01-03', '2020-01-31'],
    'FWVZ': ['2006-01-01', '2015-12-31'],
    'KRVZ': ['2010-01-01', '2019-12-31'],
    'ONTA': ['2013-01-10', '2014-12-18'],
    'SHW':  ['2004-01-02', '2005-12-30'],
}
TARGETS = ['FWVZ', 'KRVZ']

window = 2.                      # days
overlap = 0.75
look_forward = 2.                # days
data_streams = ['zsc2_rsamF', 'zsc2_mfF', 'zsc2_hfF', 'zsc2_dsarF']
Ncl = 300
Nfts = 20
method = 0.75                    # undersampling ratio
random_seed = 0
drop_features = ['linear_trend_timewise', 'agg_linear_trend']

dtw = timedelta(days=window)
dto = timedelta(days=(1 - overlap) * window)   # 12 h training step

ths = np.linspace(0, 1, num=101)


def parse_eruptions(sta):
    with open(os.path.join(DATA_DIR, f'{sta}_eruptive_periods.txt')) as fp:
        return [datetime.strptime(ln.rstrip(), '%Y %m %d %H %M %S')
                for ln in fp.readlines() if ln.strip()]


def station_window(sta):
    ti = datetime.strptime(data[sta][0], '%Y-%m-%d')
    tf = datetime.strptime(data[sta][1], '%Y-%m-%d')
    return ti, tf


# ============================================================
# Stage 1: local cache of training matrices
# ============================================================
def training_index(sta):
    """The 12-h training window grid used by puia for this station."""
    ti, tf = station_window(sta)
    t0 = ti + dtw
    n = int((tf - t0) / dto) + 1
    return pd.DatetimeIndex([t0 + k * dto for k in range(n)])


def stage1_cache():
    log.info('=' * 60)
    log.info('STAGE 1: caching training matrices locally')
    log.info(f'  cache dir: {LOCAL_CACHE}')
    log.info('=' * 60)
    os.makedirs(LOCAL_CACHE, exist_ok=True)

    for sta in data.keys():
        out_fm = os.path.join(LOCAL_CACHE, f'{sta}_train_fM.pkl')
        out_ys = os.path.join(LOCAL_CACHE, f'{sta}_train_ys.pkl')
        if os.path.isfile(out_fm) and os.path.isfile(out_ys):
            log.info(f'  {sta}: cached, skipping.')
            continue

        ti, tf = station_window(sta)
        tidx = training_index(sta)
        log.info(f'  {sta}: {len(tidx)} training windows '
                 f'({tidx[0].date()} to {tidx[-1].date()})')

        stream_frames = []
        for ds in data_streams:
            year_frames = []
            for yr in range(ti.year, tf.year + 1):
                f = os.path.join(FEAT_DIR, f'fm_2.00w_{ds}_{sta}_{yr}.pkl')
                if not os.path.isfile(f):
                    log.warning(f'    MISSING {os.path.basename(f)}')
                    continue
                t0 = time.time()
                fm = pd.read_pickle(f)
                fm = fm[fm.index.isin(tidx)]
                fm = fm[~fm.index.duplicated()]
                year_frames.append(fm)
                log.info(f'    {os.path.basename(f)}: {fm.shape[0]} training rows '
                         f'({time.time()-t0:.0f}s)')
            sfm = pd.concat(year_frames, sort=False)
            sfm = sfm[~sfm.index.duplicated()]
            sfm.sort_index(inplace=True)
            stream_frames.append(sfm)

        fM = pd.concat(stream_frames, axis=1, sort=False)
        fM = fM.loc[:, ~fM.columns.duplicated()]
        fM.sort_index(inplace=True)

        # labels: eruption strictly within look_forward after window date
        tes = parse_eruptions(sta)
        lbl = np.zeros(len(fM), dtype=bool)
        for te in tes:
            lbl |= np.array([(0 < (te - t).total_seconds() / 86400. < look_forward)
                             for t in fM.index])
        ys = pd.DataFrame({'label': lbl.astype(float)}, index=fM.index)

        cover = len(fM) / len(tidx)
        log.info(f'  {sta}: fM {fM.shape}, positives {int(lbl.sum())}, '
                 f'grid coverage {cover:.1%}')
        if cover < 0.9:
            log.warning(f'  {sta}: training grid coverage below 90%!')

        fM.to_pickle(out_fm)
        ys.to_pickle(out_ys)
        log.info(f'  {sta}: cached.')

    log.info('STAGE 1 complete.')


# ============================================================
# Stage 2: train ablation variants
# ============================================================
def apply_drop_features(fM):
    pats = [f'*__{d}__*' for d in drop_features]
    keep = [c for c in fM.columns
            if not any(fnmatch(c, p) for p in pats) and c not in drop_features]
    return fM[keep]


def train_one_model_local(fM, ys, modeldir, random_state):
    """Faithful replica of puia.model.train_one_model (DT classifier),
    with the Mann-Whitney test vectorised for speed."""
    rus = RandomUnderSampler(sampling_strategy=method,
                             random_state=random_state + random_seed)
    fMt, yst = rus.fit_resample(fM, ys['label'])
    fMt = fMt.reset_index(drop=True)
    yst = pd.Series(np.asarray(yst) > 0, index=fMt.index, dtype=bool)

    fMt = fMt.replace([np.inf, -np.inf], np.nan)
    fMt = fMt.dropna(axis=1, how='any')
    fMt = fMt.loc[:, fMt.nunique() > 1].copy()

    mask = yst.values
    pos = fMt.loc[mask].values
    neg = fMt.loc[~mask].values
    try:
        _, pv = mannwhitneyu(pos, neg, alternative='two-sided', axis=0)
        pvals = pd.Series(pv, index=fMt.columns)
    except Exception:
        pvals = {}
        for col in fMt.columns:
            try:
                _, p = mannwhitneyu(fMt.loc[mask, col].values,
                                    fMt.loc[~mask, col].values,
                                    alternative='two-sided')
                pvals[col] = p
            except Exception:
                pvals[col] = 1.0
        pvals = pd.Series(pvals)
    pvals = pvals.fillna(1.0).sort_values(kind='mergesort')
    fts = pvals.index[:Nfts].tolist()
    pvs = pvals.values[:Nfts]
    fMt = fMt[fts]
    with open(os.path.join(modeldir, f'{random_state:04d}.fts'), 'w') as fp:
        for f, pv in zip(fts, pvs):
            fp.write('{:4.3e} {:s}\n'.format(pv, f))

    ss = ShuffleSplit(n_splits=5, test_size=0.25,
                      random_state=random_state + random_seed)
    model = DecisionTreeClassifier(class_weight='balanced')
    grid = {'max_depth': [3, 5, 7], 'criterion': ['gini', 'entropy'],
            'max_features': ['auto', 'sqrt', 'log2', None]}
    fl = os.path.join(modeldir, f'DecisionTreeClassifier_{random_state:04d}.pkl')
    if os.path.isfile(fl):
        return
    model_cv = GridSearchCV(model, grid, cv=ss, scoring='balanced_accuracy',
                            error_score=np.nan)
    model_cv.fit(fMt, yst)
    joblib.dump(model_cv.best_estimator_, fl, compress=3)


# B.3 curated pools, chosen from the B.1 single-source ablation lifts:
# FWVZ: KRVZ helps (+0.07), ONTA ~neutral, WIZ (-0.09) and SHW (-0.01) harm.
# KRVZ: SHW helps (+0.04), FWVZ/ONTA mildly harm, WIZ (-0.11) harms strongly.
CURATED = {
    'FWVZ': {
        'cur_KRVZ_ONTA': ['KRVZ', 'ONTA'],
        'cur_KRVZ': ['KRVZ'],
        'cur_KRVZ_ONTA_SHW': ['KRVZ', 'ONTA', 'SHW'],  # == no_WIZ, kept for naming clarity in B.3
    },
    'KRVZ': {
        'cur_SHW': ['SHW'],
        'cur_SHW_FWVZ': ['SHW', 'FWVZ'],
        'cur_SHW_FWVZ_ONTA': ['SHW', 'FWVZ', 'ONTA'],  # == no_WIZ
    },
}

# Source value is strongly non-additive (e.g. KRVZ: {SHW,FWVZ}=0.861 vs
# SHW=0.674, full=0.649), so map the full pool landscape: all non-empty
# source subsets. Names of already-trained variants are preserved above so
# their models are reused; the dedupe in variants_for skips those.
from itertools import combinations
for _target in ['FWVZ', 'KRVZ']:
    _sources = [s for s in data.keys() if s != _target]
    for _r in range(1, len(_sources) + 1):
        for _combo in combinations(_sources, _r):
            CURATED[_target][f'cur_{"_".join(_combo)}'] = list(_combo)


def variants_for(target):
    sources = [s for s in data.keys() if s != target]
    out = {'full': sources}
    for s in sources:
        out[f'no_{s}'] = [x for x in sources if x != s]
    # B.3 curated pools (skip aliases identical to an ablation variant)
    for vname, pool in CURATED.get(target, {}).items():
        if sorted(pool) not in [sorted(v) for v in out.values()]:
            out[vname] = pool
    return out


def stage2_train():
    log.info('=' * 60)
    log.info('STAGE 2: training ablation variants')
    log.info('=' * 60)

    fMs = {sta: pd.read_pickle(os.path.join(LOCAL_CACHE, f'{sta}_train_fM.pkl'))
           for sta in data.keys()}
    yss = {sta: pd.read_pickle(os.path.join(LOCAL_CACHE, f'{sta}_train_ys.pkl'))
           for sta in data.keys()}

    for target in TARGETS:
        for vname, pool in variants_for(target).items():
            mdir = os.path.join(MODEL_ROOT, f'{target}__{vname}')
            os.makedirs(mdir, exist_ok=True)
            done = len(glob(os.path.join(mdir, 'DecisionTreeClassifier_*.pkl')))
            if done >= Ncl:
                log.info(f'  {target}/{vname}: {done} trees exist, skipping.')
                continue
            log.info(f'  {target}/{vname}: pool={pool}')

            fM = pd.concat([fMs[s] for s in pool], axis=0, sort=False)
            ys = pd.concat([yss[s] for s in pool], axis=0)
            fM = fM.reset_index(drop=True)
            ys = ys.reset_index(drop=True)
            fM = apply_drop_features(fM)
            log.info(f'    fM {fM.shape}, positives {int(ys["label"].sum())}')

            t0 = time.time()
            fails = 0
            for i in range(Ncl):
                try:
                    train_one_model_local(fM, ys, mdir, i)
                except Exception as e:
                    fails += 1
                    if fails <= 3:
                        log.warning(f'    tree {i} failed: {e}')
                if (i + 1) % 50 == 0:
                    log.info(f'    {i+1}/{Ncl} trees ({time.time()-t0:.0f}s)')
            log.info(f'  {target}/{vname}: trained in {time.time()-t0:.0f}s'
                     + (f' ({fails} trees failed)' if fails else ''))

    log.info('STAGE 2 complete.')


# ============================================================
# Stage 3: forecast targets + ablation AUC table
# ============================================================
def load_variant_trees(target, vname):
    mdir = os.path.join(MODEL_ROOT, f'{target}__{vname}')
    trees = []
    for flp in sorted(glob(os.path.join(mdir, 'DecisionTreeClassifier_*.pkl'))):
        num = os.path.basename(flp).split('.')[0].split('_')[-1]
        model = joblib.load(flp)
        with open(os.path.join(mdir, f'{num}.fts')) as fp:
            fts = [' '.join(ln.rstrip().split()[1:]) for ln in fp.readlines()]
        trees.append((model, fts))
    return trees


def load_hires_year(sta, yr, tf_clamp):
    fms = []
    for ds in data_streams:
        f = os.path.join(FEAT_DIR, f'fm_2.00w_{ds}_{sta}_{yr}.pkl')
        fm = pd.read_pickle(f)
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


def predict_consensus(fM, trees):
    total = np.zeros(fM.shape[0])
    for model, fts in trees:
        missing = [f for f in fts if f not in fM.columns]
        for f in missing:
            fM[f] = 0.
        total += model.predict(fM[fts]).astype(float)
    return pd.DataFrame(total / len(trees), columns=['consensus'], index=fM.index)


def stage3_forecast_and_evaluate():
    log.info('=' * 60)
    log.info('STAGE 3: forecasting targets + ablation table')
    log.info('=' * 60)

    for target in TARGETS:
        ti, tf = station_window(target)
        vnames = list(variants_for(target).keys())
        # load all variants' trees once
        trees = {}
        for vname in vnames:
            master_out = os.path.join(FORECAST_ROOT, f'{target}__{vname}',
                                      'consensus_master.pkl')
            if os.path.isfile(master_out):
                continue
            trees[vname] = load_variant_trees(target, vname)
            log.info(f'  {target}/{vname}: {len(trees[vname])} trees loaded')
        if trees:
            partial = {v: [] for v in trees}
            for yr in range(ti.year, tf.year + 1):
                t0 = time.time()
                fM = load_hires_year(target, yr, tf)
                if fM.shape[0] == 0:
                    log.warning(f'  {target} {yr}: no hires rows!')
                    continue
                for vname in trees:
                    partial[vname].append(predict_consensus(fM, trees[vname]))
                log.info(f'  {target} {yr}: {fM.shape[0]} rows predicted for '
                         f'{len(trees)} variants ({time.time()-t0:.0f}s)')
                del fM
            for vname in trees:
                d = os.path.join(FORECAST_ROOT, f'{target}__{vname}')
                os.makedirs(d, exist_ok=True)
                master = pd.concat(partial[vname])
                master = master[~master.index.duplicated()]
                master.sort_index(inplace=True)
                master.to_pickle(os.path.join(d, 'consensus_master.pkl'))
                log.info(f'  {target}/{vname}: master consensus saved '
                         f'({len(master)} rows)')

        # evaluate
        results = []
        tes = parse_eruptions(target)
        for vname in vnames:
            master = pd.read_pickle(os.path.join(
                FORECAST_ROOT, f'{target}__{vname}', 'consensus_master.pkl'))
            auc = eruption_auc(master, tes)
            results.append({'target': target, 'variant': vname, 'auc': auc})
            log.info(f'  {target}/{vname}: AUC = {auc:.4f}')

        df = pd.DataFrame(results)
        full_auc = df.loc[df.variant == 'full', 'auc'].iloc[0]
        df['delta_vs_full'] = df.auc - full_auc
        # for no_S rows: positive lift means S helps (full beats pool without S)
        df['auc_lift_of_dropped_source'] = np.where(
            df.variant.str.startswith('no_'), full_auc - df.auc, np.nan)
        out_csv = os.path.join(FORECAST_ROOT, f'ablation_{target}.csv')
        df.to_csv(out_csv, index=False)
        log.info(f'  {target}: ablation table saved to {out_csv}')

    log.info('STAGE 3 complete.')


def eruption_auc(consensus, tes):
    """Phase-A step-4 style eruption ROC/AUC for a single station."""
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


if __name__ == '__main__':
    stage = sys.argv[1] if len(sys.argv) > 1 else 'all'
    log.info(f'PHASE B.1 (local) — stage: {stage}')
    if stage in ('all', 'cache'):
        stage1_cache()
    if stage in ('all', 'train'):
        stage2_train()
    if stage in ('all', 'forecast'):
        stage3_forecast_and_evaluate()
    log.info('PHASE B.1 done.')
