"""
Phase G: pseudo-prospective unified evaluation — the arbiter.

For each target eruption e (targets FWVZ, KRVZ, WIZ), each method may use
ONLY target-derived information from BEFORE e's forecast window
(window = te-1month .. te+4days; target-data cutoff = window start).
Source-volcano libraries are unrestricted (standard transfer setting);
target data, pool selection, and adaptation are strictly pre-cutoff.

Methods (as operational policies, with the no-prior-eruption fallback):
  full     leave-target-out full source pool (uses no target info)
  curated  pool picked from the 15-subset landscape by ranking on PRIOR
           eruptions + pre-cutoff background; fallback: full
  ser_us   full-pool base refined (per-tree undersampled SER) on pre-cutoff
           target data; fallback: base
  tboost   multi-source TrAdaBoost fit on sources + pre-cutoff target;
           fallback: full

Scoring (stricter than Phases A-D): pooled eruption AUC over the crisis
windows only — TP statistic = q95 of consensus in [te-2d, te); background =
the remaining window rows (unrest-elevated month, identical rows for every
method, all out-of-sample for the prospective models).

Stages (resumable): fit -> forecast -> eval
Usage:  python -u phase_g_local.py
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
root = r'U:\Research\EruptionForecasting\eruptions'
DATA_DIR = os.path.join(root, 'data')
FEAT_DIR = os.path.join(root, 'features')
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_ROOT = os.path.join(SCRIPT_DIR, 'models', 'phase_b')
FB = os.path.join(SCRIPT_DIR, 'forecasts', 'phase_b')
FBW = os.path.join(SCRIPT_DIR, 'forecasts', 'phase_b_wiz')
OUT_DIR = os.path.join(SCRIPT_DIR, 'forecasts', 'phase_g')
LOCAL_CACHE = os.path.join(os.path.expanduser('~'), 'puia_local_cache')
FIT_DIR = os.path.join(LOCAL_CACHE, 'phase_g_models')

_spec = importlib.util.spec_from_file_location(
    'fine_tuning', os.path.join(SCRIPT_DIR, 'puia', 'fine_tuning.py'))
FT = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(FT)
sys.modules['fine_tuning'] = FT

STATIONS = ['WIZ', 'FWVZ', 'KRVZ', 'ONTA', 'SHW']
TARGETS = ['FWVZ', 'KRVZ', 'WIZ']
BASE_DIR = {'FWVZ': 'FWVZ__full', 'KRVZ': 'KRVZ__full', 'WIZ': 'WIZ__full'}
data_streams = ['zsc2_rsamF', 'zsc2_mfF', 'zsc2_hfF', 'zsc2_dsarF']
drop_features = ['linear_trend_timewise', 'agg_linear_trend']
window = 2.
ths = np.linspace(0, 1, num=101)
N_ITER = 40
MAX_DEPTH = 6

# WIZ pool-name -> column in phase_b_wiz year files; FWVZ/KRVZ pools are dirs
WIZ_POOL_COLS = ['full', 'FKO', 'FKS', 'FK', 'KOS', 'FOS', 'FO', 'FS', 'KO',
                 'KS', 'OS', 'F', 'K', 'O', 'S']


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


def parse_eruptions(sta, attempts=5, wait=30):
    for a in range(attempts):
        try:
            with open(os.path.join(DATA_DIR, f'{sta}_eruptive_periods.txt')) as fp:
                return [datetime.strptime(ln.rstrip(), '%Y %m %d %H %M %S')
                        for ln in fp.readlines() if ln.strip()]
        except OSError as e:
            if a == attempts - 1:
                raise
            log.warning(f'  eruptive_periods read failed ({e}), retry in {wait}s')
            time.sleep(wait)


def pool_masters(target):
    """name -> consensus Series over the target's record (target-independent
    pool models, so all values are out-of-sample for the target)."""
    out = {}
    if target == 'WIZ':
        files = sorted(glob(os.path.join(FBW, 'WIZ_*.pkl')))
        df = pd.concat([read_pickle_retry(f) for f in files])
        df = df[~df.index.duplicated()].sort_index()
        for c in WIZ_POOL_COLS:
            out[c] = df[c]
    else:
        prefix = f'{target}__'
        for d in sorted(os.listdir(FB)):
            p = os.path.join(FB, d, 'consensus_master.pkl')
            if d.startswith(prefix) and os.path.isfile(p):
                out[d[len(prefix):]] = read_pickle_retry(p)['consensus']
    return out


def auc_from(q95_ev, bg):
    q95_ev = np.atleast_1d(q95_ev)
    tpr = [(q95_ev >= th).mean() for th in ths]
    fpr = [(bg >= th).mean() for th in ths]
    return sum((fpr[i] - fpr[i + 1]) * tpr[i] for i in range(len(ths) - 1))


def apply_drop_features(fM):
    from fnmatch import fnmatch
    pats = [f'*__{d}__*' for d in drop_features]
    keep = [c for c in fM.columns
            if not any(fnmatch(c, p) for p in pats) and c not in drop_features]
    return fM[keep]


def load_base_trees(dirname):
    mdir = os.path.join(MODEL_ROOT, dirname)
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


def common_features(fMs):
    common = fMs[STATIONS[0]].columns
    for sta in STATIONS[1:]:
        common = common.intersection(fMs[sta].columns)
    bad = set()
    for sta in STATIONS:
        sub = fMs[sta][common]
        bad |= set(sub.columns[sub.isna().any() | np.isinf(sub).any()])
    return [c for c in common if c not in bad]


def load_window_fM(sta, t0, t1):
    yrs = sorted({t0.year, t1.year})
    fms = []
    for ds in data_streams:
        parts = []
        for yr in yrs:
            f = os.path.join(FEAT_DIR, f'fm_2.00w_{ds}_{sta}_{yr}.pkl')
            fm = read_pickle_retry(f)
            parts.append(fm[(fm.index >= t0) & (fm.index <= t1)])
        fm = pd.concat(parts)
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


# ============================================================
# Stage 1: prospective fits (ser_us, tboost) per eruption
# ============================================================
def stage_fit():
    log.info('=' * 60)
    log.info('STAGE FIT: prospective ser_us + tboost per eruption')
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
        tes = parse_eruptions(target)
        sources = [s for s in STATIONS if s != target]
        t_idx = pd.DatetimeIndex(fMs[target].index)
        base_trees = None
        Xs = ys_src = src_sta = None

        for i, te in enumerate(tes):
            cutoff = te - _MONTH
            mask = t_idx < cutoff
            n_pos = int((yss[target]['label'].values[mask] > 0).sum())
            log.info(f'{target} e{i} (te={te.date()}): pre-cutoff rows '
                     f'{int(mask.sum())}, prior positives {n_pos}')
            if n_pos == 0:
                log.info(f'  no prior eruptions — fallback policies apply.')
                continue

            # --- ser_us ---
            out = os.path.join(FIT_DIR, f'{target}__e{i}__ser_us.pkl')
            if not os.path.isfile(out):
                if base_trees is None:
                    base_trees = load_base_trees(BASE_DIR[target])
                    log.info(f'  base trees loaded ({len(base_trees)})')
                t0 = time.time()
                refined = FT.refine_ensemble(
                    base_trees, fMs[target].loc[mask],
                    yss[target]['label'].values[mask] > 0,
                    'ser', resample_ratio=0.75)
                roots = [(rt.root, fts) for rt, fts in refined]
                with open(out, 'wb') as fp:
                    pickle.dump(roots, fp)
                log.info(f'  ser_us fitted ({time.time()-t0:.0f}s)')

            # --- tboost ---
            out = os.path.join(FIT_DIR, f'{target}__e{i}__tboost.pkl')
            if not os.path.isfile(out):
                if Xs is None:
                    Xs = np.vstack([fMs[s][feats].values for s in sources])
                    ys_src = np.concatenate(
                        [yss[s]['label'].values > 0 for s in sources])
                    src_sta = np.concatenate(
                        [[s] * len(fMs[s]) for s in sources])
                t0 = time.time()
                tb = FT.TransferBoostEnsemble(n_iterations=N_ITER,
                                              max_depth=MAX_DEPTH)
                tb.fit(Xs, ys_src, src_sta,
                       fMs[target][feats].values[mask],
                       yss[target]['label'].values[mask] > 0, feats)
                with open(out, 'wb') as fp:
                    pickle.dump(tb, fp)
                with open(out, 'rb') as fp:   # verify not truncated
                    pickle.load(fp)
                log.info(f'  tboost fitted ({time.time()-t0:.0f}s)')
        del base_trees, Xs
    log.info('STAGE FIT complete.')


# ============================================================
# Stage 2: forecast eruption windows
# ============================================================
def stage_forecast():
    log.info('=' * 60)
    log.info('STAGE FORECAST: eruption windows')
    log.info('=' * 60)
    os.makedirs(OUT_DIR, exist_ok=True)
    for target in TARGETS:
        tes = parse_eruptions(target)
        for i, te in enumerate(tes):
            out = os.path.join(OUT_DIR, f'{target}__e{i}.pkl')
            if os.path.isfile(out):
                log.info(f'{target} e{i}: exists, skipping.')
                continue
            t0w, t1w = te - _MONTH, te + 4 * _DAY
            t0 = time.time()
            fM = load_window_fM(target, t0w, t1w)
            res = {}
            f_ser = os.path.join(FIT_DIR, f'{target}__e{i}__ser_us.pkl')
            if os.path.isfile(f_ser):
                with open(f_ser, 'rb') as fp:
                    roots = pickle.load(fp)
                refined = [(FT.RefinableTree(r), fts) for r, fts in roots]
                res['ser_us'] = pd.Series(
                    FT.predict_consensus(refined, fM), index=fM.index)
            f_tb = os.path.join(FIT_DIR, f'{target}__e{i}__tboost.pkl')
            if os.path.isfile(f_tb):
                with open(f_tb, 'rb') as fp:
                    tb = pickle.load(fp)
                res['tboost'] = pd.Series(tb.predict_score(fM), index=fM.index)
            pd.to_pickle(res, out)
            log.info(f'{target} e{i}: {fM.shape[0]} window rows, methods '
                     f'{list(res)} ({time.time()-t0:.0f}s)')
            del fM
    log.info('STAGE FORECAST complete.')


# ============================================================
# Stage 3: evaluate
# ============================================================
def stage_eval():
    log.info('=' * 60)
    log.info('STAGE EVAL: pseudo-prospective comparison')
    log.info('=' * 60)
    rows, summary = [], []
    for target in TARGETS:
        tes = parse_eruptions(target)
        masters = pool_masters(target)
        log.info(f'{target}: {len(masters)} pool masters loaded')

        per_method = {m: {'q': [], 'bg': []}
                      for m in ['full', 'curated', 'ser_us', 'tboost']}
        for i, te in enumerate(tes):
            cutoff = te - _MONTH
            t0w, t1w = te - _MONTH, te + 4 * _DAY

            # prospective pool selection on prior eruptions + pre-cutoff bg
            prior = [t for t in tes if t < cutoff]
            if prior:
                def sel_metric(name):
                    con = masters[name]
                    q = [con[(con.index >= p - window * _DAY) &
                             (con.index < p)].quantile(0.95) for p in prior]
                    bgm = con.index < cutoff
                    for p in prior:
                        bgm &= ~((con.index >= p - window * _DAY) &
                                 (con.index < p))
                    return auc_from(np.array(q), con.values[bgm])
                sel = max(masters, key=sel_metric)
            else:
                sel = 'full'

            # window consensus per method
            win = {}
            win['full'] = masters['full'][(masters['full'].index >= t0w) &
                                          (masters['full'].index <= t1w)]
            win['curated'] = masters[sel][(masters[sel].index >= t0w) &
                                          (masters[sel].index <= t1w)]
            res = read_pickle_retry(os.path.join(OUT_DIR, f'{target}__e{i}.pkl'))
            win['ser_us'] = res.get('ser_us', win['full'])   # fallback
            win['tboost'] = res.get('tboost', win['full'])   # fallback

            for m, con in win.items():
                pre = con[(con.index >= te - window * _DAY) & (con.index < te)]
                bg = con[(con.index < te - window * _DAY) | (con.index >= te)]
                q = pre.quantile(0.95) if len(pre) else 0.
                per_method[m]['q'].append(q)
                per_method[m]['bg'].append(bg.values)
                rows.append({'target': target, 'eruption': i,
                             'te': te.strftime('%Y-%m-%d'), 'method': m,
                             'selected_pool': sel if m == 'curated' else '',
                             'pre_q95': q,
                             'bg_p95': float(np.quantile(bg.values, 0.95))})
            log.info(f'  e{i} (te={te.date()}): curated selects '
                     f'{sel:14s} pre-q95: ' +
                     ' '.join(f'{m}={per_method[m]["q"][-1]:.2f}'
                              for m in per_method))

        for m in per_method:
            auc = auc_from(np.array(per_method[m]['q']),
                           np.concatenate(per_method[m]['bg']))
            summary.append({'target': target, 'method': m, 'auc': auc,
                            'n_events': len(per_method[m]['q'])})
            log.info(f'  {target} PROSPECTIVE {m}: AUC = {auc:.4f}')

    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, 'phase_g_events.csv'),
                              index=False)
    pd.DataFrame(summary).to_csv(os.path.join(OUT_DIR, 'phase_g_summary.csv'),
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
    log.info('PHASE G done.')
