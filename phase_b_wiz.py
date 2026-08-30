"""
Phase B for WIZ (Whakaari) as target: full 15-subset source-pool landscape
over {FWVZ, KRVZ, ONTA, SHW}, plus nested (bias-free) pool selection over
WIZ's 5 eruptions.

11 of the 15 pools are reused from the FWVZ/KRVZ landscape runs (a pool
ensemble depends only on its source set, and WIZ is in none of them); only
the 4 pools containing both FWVZ and KRVZ are trained here (including WIZ's
'full' pool {FWVZ,KRVZ,ONTA,SHW}).

Stages (resumable): train -> forecast -> eval
Usage:  python -u phase_b_wiz.py
"""

import os
import sys
import time
import logging
import warnings
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

import phase_b_local as PB   # reuse config + training helpers (no side effects)

_DAY = timedelta(days=1)
root = r'U:\Research\EruptionForecasting\eruptions'
DATA_DIR = os.path.join(root, 'data')
FEAT_DIR = os.path.join(root, 'features')
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_ROOT = os.path.join(SCRIPT_DIR, 'models', 'phase_b')
FORECAST_ROOT_B = os.path.join(SCRIPT_DIR, 'forecasts', 'phase_b')
OUT_DIR = os.path.join(SCRIPT_DIR, 'forecasts', 'phase_b_wiz')

TARGET = 'WIZ'
data_streams = ['zsc2_rsamF', 'zsc2_mfF', 'zsc2_hfF', 'zsc2_dsarF']
window = 2.
ths = np.linspace(0, 1, num=101)

# pool name -> (source stations, model dir). None dir => train here.
POOLS = {
    'full':      (['FWVZ', 'KRVZ', 'ONTA', 'SHW'], None),
    'FKO':       (['FWVZ', 'KRVZ', 'ONTA'], None),
    'FKS':       (['FWVZ', 'KRVZ', 'SHW'], None),
    'FK':        (['FWVZ', 'KRVZ'], None),
    'KOS':       (['KRVZ', 'ONTA', 'SHW'], 'FWVZ__no_WIZ'),
    'FOS':       (['FWVZ', 'ONTA', 'SHW'], 'KRVZ__no_WIZ'),
    'FO':        (['FWVZ', 'ONTA'], 'KRVZ__cur_FWVZ_ONTA'),
    'FS':        (['FWVZ', 'SHW'], 'KRVZ__cur_SHW_FWVZ'),
    'KO':        (['KRVZ', 'ONTA'], 'FWVZ__cur_KRVZ_ONTA'),
    'KS':        (['KRVZ', 'SHW'], 'FWVZ__cur_KRVZ_SHW'),
    'OS':        (['ONTA', 'SHW'], 'FWVZ__cur_ONTA_SHW'),
    'F':         (['FWVZ'], 'KRVZ__cur_FWVZ'),
    'K':         (['KRVZ'], 'FWVZ__cur_KRVZ'),
    'O':         (['ONTA'], 'FWVZ__cur_ONTA'),
    'S':         (['SHW'], 'FWVZ__cur_SHW'),
}


def model_dir(name):
    pool, alias = POOLS[name]
    if alias is None:
        return os.path.join(MODEL_ROOT, f'WIZ__{name}')
    return os.path.join(MODEL_ROOT, alias)


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


# ============================================================
# Stage 1: train the 4 missing pools
# ============================================================
def stage_train():
    log.info('=' * 60)
    log.info('STAGE TRAIN: pools containing both FWVZ and KRVZ')
    log.info('=' * 60)
    fMs, yss = {}, {}
    for name, (pool, alias) in POOLS.items():
        if alias is not None:
            continue
        mdir = model_dir(name)
        os.makedirs(mdir, exist_ok=True)
        done = len(glob(os.path.join(mdir, 'DecisionTreeClassifier_*.pkl')))
        if done >= PB.Ncl:
            log.info(f'  WIZ__{name}: {done} trees exist, skipping.')
            continue
        for s in pool:
            if s not in fMs:
                fMs[s] = pd.read_pickle(os.path.join(PB.LOCAL_CACHE, f'{s}_train_fM.pkl'))
                yss[s] = pd.read_pickle(os.path.join(PB.LOCAL_CACHE, f'{s}_train_ys.pkl'))
        fM = pd.concat([fMs[s] for s in pool], axis=0, sort=False).reset_index(drop=True)
        ys = pd.concat([yss[s] for s in pool], axis=0).reset_index(drop=True)
        fM = PB.apply_drop_features(fM)
        log.info(f'  WIZ__{name}: pool={pool}, fM {fM.shape}, '
                 f'positives {int(ys["label"].sum())}')
        t0 = time.time()
        for i in range(PB.Ncl):
            PB.train_one_model_local(fM, ys, mdir, i)
            if (i + 1) % 100 == 0:
                log.info(f'    {i+1}/{PB.Ncl} trees ({time.time()-t0:.0f}s)')
        log.info(f'  WIZ__{name}: trained in {time.time()-t0:.0f}s')
    log.info('STAGE TRAIN complete.')


# ============================================================
# Stage 2: forecast WIZ with all 15 pools
# ============================================================
def load_trees(mdir):
    trees = []
    for flp in sorted(glob(os.path.join(mdir, 'DecisionTreeClassifier_*.pkl'))):
        num = os.path.basename(flp).split('.')[0].split('_')[-1]
        for a in range(3):
            try:
                clf = joblib.load(flp)
                break
            except OSError:
                if a == 2:
                    raise
                time.sleep(10)
        with open(os.path.join(mdir, f'{num}.fts')) as fp:
            fts = [' '.join(ln.rstrip().split()[1:]) for ln in fp.readlines()]
        trees.append((clf, fts))
    return trees


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


def predict_ensemble(trees, fM):
    total = np.zeros(fM.shape[0])
    for clf, fts in trees:
        missing = [f for f in fts if f not in fM.columns]
        for f in missing:
            fM[f] = 0.
        total += clf.predict(fM[fts]).astype(float)
    return total / len(trees)


def stage_forecast():
    log.info('=' * 60)
    log.info('STAGE FORECAST: WIZ record with 15 pools')
    log.info('=' * 60)
    os.makedirs(OUT_DIR, exist_ok=True)
    ti, tf = datetime(2010, 1, 3), datetime(2020, 1, 31)
    trees = {}
    for name in POOLS:
        t0 = time.time()
        trees[name] = load_trees(model_dir(name))
        log.info(f'  pool {name}: {len(trees[name])} trees ({time.time()-t0:.0f}s)')

    for yr in range(ti.year, tf.year + 1):
        out = os.path.join(OUT_DIR, f'WIZ_{yr}.pkl')
        if os.path.isfile(out):
            log.info(f'  WIZ {yr}: exists, skipping.')
            continue
        t0 = time.time()
        fM = load_year('WIZ', yr, tf)
        if fM.shape[0] == 0:
            log.warning(f'  WIZ {yr}: no rows!')
            continue
        df = pd.DataFrame({name: predict_ensemble(trees[name], fM)
                           for name in POOLS}, index=fM.index)
        df.to_pickle(out)
        log.info(f'  WIZ {yr}: {fM.shape[0]} rows x {len(POOLS)} pools '
                 f'({time.time()-t0:.0f}s)')
        del fM
    log.info('STAGE FORECAST complete.')


# ============================================================
# Stage 3: evaluate — landscape + nested selection
# ============================================================
def stage_eval():
    log.info('=' * 60)
    log.info('STAGE EVAL: WIZ pool landscape + nested selection')
    log.info('=' * 60)
    files = sorted(glob(os.path.join(OUT_DIR, 'WIZ_*.pkl')))
    df = pd.concat([read_pickle_retry(f) for f in files])
    df = df[~df.index.duplicated()]
    df.sort_index(inplace=True)
    tes = parse_eruptions('WIZ')
    log.info(f'  consensus rows: {len(df)}, eruptions: {len(tes)}')

    # per-pool stats: q95 in each eruption pre-window + background array
    stats = {}
    for name in POOLS:
        con = df[name]
        q95, bg_mask = [], np.ones(len(con), dtype=bool)
        for te in tes:
            m = (con.index >= te - window * _DAY) & (con.index < te)
            q95.append(con[m].quantile(0.95) if m.sum() else 0.)
            bg_mask &= ~m
        stats[name] = (np.array(q95), con.values[bg_mask])

    def auc_from(q95_ev, bg):
        tpr = [(q95_ev >= th).mean() if len(q95_ev) else 0. for th in ths]
        fpr = [(bg >= th).mean() for th in ths]
        return sum((fpr[i] - fpr[i + 1]) * tpr[i] for i in range(len(ths) - 1))

    rows = []
    for name in POOLS:
        auc = auc_from(*stats[name])
        rows.append({'pool': name, 'sources': '+'.join(POOLS[name][0]),
                     'auc': auc})
    res = pd.DataFrame(rows).sort_values('auc', ascending=False)
    for _, r in res.iterrows():
        log.info(f'  WIZ pool {r["pool"]:5s} ({r["sources"]:20s}): AUC = {r["auc"]:.4f}')
    full_auc = res.loc[res.pool == 'full', 'auc'].iloc[0]
    res['delta_vs_full'] = res.auc - full_auc

    # nested selection over 5 folds
    K = len(tes)
    fold_rows = []
    oracle = res.iloc[0]['pool']
    for i in range(K):
        others = [j for j in range(K) if j != i]
        sel = max(POOLS, key=lambda p: auc_from(stats[p][0][others], stats[p][1]))
        fold_rows.append({
            'fold': i, 'te': tes[i].strftime('%Y-%m-%d'), 'selected_pool': sel,
            'auc_nested': auc_from(stats[sel][0][[i]], stats[sel][1]),
            'auc_oracle': auc_from(stats[oracle][0][[i]], stats[oracle][1]),
            'auc_full': auc_from(stats['full'][0][[i]], stats['full'][1]),
        })
        r = fold_rows[-1]
        log.info(f'  fold {i} (te={r["te"]}): selects {sel:5s} '
                 f'nested={r["auc_nested"]:.4f} oracle={r["auc_oracle"]:.4f} '
                 f'full={r["auc_full"]:.4f}')
    fold_df = pd.DataFrame(fold_rows)
    log.info(f'  WIZ MEANS: nested={fold_df.auc_nested.mean():.4f} '
             f'oracle={fold_df.auc_oracle.mean():.4f} '
             f'full={fold_df.auc_full.mean():.4f}')

    res.to_csv(os.path.join(OUT_DIR, 'ablation_WIZ.csv'), index=False)
    fold_df.to_csv(os.path.join(OUT_DIR, 'nested_WIZ.csv'), index=False)

    # save the 'full' master where phase_c expects it
    d = os.path.join(FORECAST_ROOT_B, 'WIZ__full')
    os.makedirs(d, exist_ok=True)
    m = df[['full']].rename(columns={'full': 'consensus'})
    m.to_pickle(os.path.join(d, 'consensus_master.pkl'))
    log.info('STAGE EVAL complete.')


if __name__ == '__main__':
    stage = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if stage in ('all', 'train'):
        stage_train()
    if stage in ('all', 'forecast'):
        stage_forecast()
    if stage in ('all', 'eval'):
        stage_eval()
    log.info('PHASE B (WIZ target) done.')
