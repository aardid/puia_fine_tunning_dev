"""
Background-only (label-free) adaptation: can a generalized ensemble be
adapted to a target volcano using ONLY non-eruptive data — the deployment
case of a monitored volcano with no recorded eruptions?

For each target (FWVZ, KRVZ, WIZ), the base is its leave-target-out 'full'
pool ensemble. Adaptation data = the target's training-grid windows with ALL
eruption windows (+-1 month) excluded — no labels used anywhere. Evaluation
is on the real eruptions, exactly as in Phases B/C.

Methods:
  bg_strut  label-free STRUT: each internal node's threshold is re-set to
            the target-background quantile matching the fraction of SOURCE
            data the node routed left (preserves the tree's relative logic,
            re-anchors "loud" to the local noise floor). Leaves untouched.
  qmap      quantile mapping: each feature value is mapped target-background
            quantile -> source-pool value; unchanged trees then predict
            "in-distribution".

Also reports a pooled 3-volcano AUC with/without per-station background-
percentile calibration of the consensus (single shared alert threshold —
where calibration matters; per-station AUC is invariant to it).

Stages (resumable): fit+forecast -> eval
Usage:  python -u phase_c_bgonly.py
"""

import os
import sys
import time
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
FORECAST_ROOT_B = os.path.join(SCRIPT_DIR, 'forecasts', 'phase_b')
OUT_DIR = os.path.join(SCRIPT_DIR, 'forecasts', 'phase_c_bgonly')
LOCAL_CACHE = os.path.join(os.path.expanduser('~'), 'puia_local_cache')

_spec = importlib.util.spec_from_file_location(
    'fine_tuning', os.path.join(SCRIPT_DIR, 'puia', 'fine_tuning.py'))
FT = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(FT)

data = {
    'WIZ':  ['2010-01-03', '2020-01-31'],
    'FWVZ': ['2006-01-01', '2015-12-31'],
    'KRVZ': ['2010-01-01', '2019-12-31'],
}
# target -> (base model dir, source pool stations)
BASES = {
    'FWVZ': ('FWVZ__full', ['WIZ', 'KRVZ', 'ONTA', 'SHW']),
    'KRVZ': ('KRVZ__full', ['WIZ', 'FWVZ', 'ONTA', 'SHW']),
    'WIZ':  ('WIZ__full', ['FWVZ', 'KRVZ', 'ONTA', 'SHW']),
}
data_streams = ['zsc2_rsamF', 'zsc2_mfF', 'zsc2_hfF', 'zsc2_dsarF']
window = 2.
ths = np.linspace(0, 1, num=101)
MIN_T = 10   # min target-background rows at a node before re-anchoring


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


# ============================================================
# Label-free STRUT
# ============================================================
def bg_strut(node, Xs, Xt, min_t=MIN_T):
    """Re-anchor thresholds to target-background quantiles matching the
    source routing proportions. Source rows route by ORIGINAL thresholds
    (they define the reference), target rows by the NEW ones."""
    if node['leaf'] or len(Xs) == 0:
        return
    f, thr_old = node['feature'], node['threshold']
    p_left = float((Xs[:, f] <= thr_old).mean())
    if len(Xt) >= min_t and 0. < p_left < 1.:
        node['threshold'] = float(np.quantile(Xt[:, f], p_left))
    sl = Xs[:, f] <= thr_old
    tl = Xt[:, f] <= node['threshold']
    bg_strut(node['left'], Xs[sl], Xt[tl], min_t)
    bg_strut(node['right'], Xs[~sl], Xt[~tl], min_t)


# ============================================================
# Quantile mapping
# ============================================================
class QuantileMapper:
    """Per-feature monotone map: target-background quantile -> source value."""

    def __init__(self, src_df, tgt_df, features, max_n=20000, rng=None):
        rng = rng or np.random.default_rng(0)
        self.maps = {}
        for f in features:
            sv = src_df[f].dropna().values if f in src_df.columns else np.array([])
            tv = tgt_df[f].dropna().values if f in tgt_df.columns else np.array([])
            if len(sv) < 50 or len(tv) < 50:
                continue
            if len(sv) > max_n:
                sv = rng.choice(sv, max_n, replace=False)
            if len(tv) > max_n:
                tv = rng.choice(tv, max_n, replace=False)
            self.maps[f] = (np.sort(tv), np.sort(sv))

    def transform(self, fM):
        out = fM.copy()
        for f, (tq, sq) in self.maps.items():
            if f not in out.columns:
                continue
            cdf = np.searchsorted(tq, out[f].values, side='right') / len(tq)
            out[f] = np.interp(cdf, np.linspace(0, 1, len(sq)), sq)
        return out


# ============================================================
# Forecast helpers
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


def predict_sklearn(trees, fM):
    total = np.zeros(fM.shape[0])
    for clf, fts in trees:
        missing = [f for f in fts if f not in fM.columns]
        for f in missing:
            fM[f] = 0.
        total += clf.predict(fM[fts]).astype(float)
    return total / len(trees)


def predict_refined(refined, fM):
    total = np.zeros(len(fM))
    for rt, fts in refined:
        total += rt.predict(FT.build_X(fM, fts)).astype(float)
    return total / len(refined)


def stage_forecast():
    os.makedirs(OUT_DIR, exist_ok=True)
    for target, (base_dir, pool) in BASES.items():
        ti = datetime.strptime(data[target][0], '%Y-%m-%d')
        tf = datetime.strptime(data[target][1], '%Y-%m-%d')
        yrs = list(range(ti.year, tf.year + 1))
        if all(os.path.isfile(os.path.join(OUT_DIR, f'{target}_{y}.pkl'))
               for y in yrs):
            log.info(f'{target}: all years exist, skipping fit+forecast.')
            continue

        log.info(f'{target}: fitting background-only adaptations '
                 f'(base {base_dir}, pool {pool})')
        trees = load_base_trees(base_dir)
        log.info(f'  {len(trees)} base trees loaded')

        # source-pool training rows (for routing proportions and qmap source)
        src = pd.concat([pd.read_pickle(os.path.join(LOCAL_CACHE, f'{s}_train_fM.pkl'))
                         for s in pool], axis=0, sort=False)
        # target background rows: training grid minus all eruption windows
        tgt = pd.read_pickle(os.path.join(LOCAL_CACHE, f'{target}_train_fM.pkl'))
        tes = parse_eruptions(target)
        keep = np.ones(len(tgt), dtype=bool)
        tidx = pd.DatetimeIndex(tgt.index)
        for te in tes:
            keep &= ~((tidx >= te - _MONTH) & (tidx <= te + _MONTH))
        tgt_bg = tgt.loc[keep]
        log.info(f'  source rows {len(src)}, target background rows {len(tgt_bg)} '
                 f'({len(tgt)-len(tgt_bg)} eruption-window rows excluded)')

        # bg_strut: adapt each tree
        t0 = time.time()
        refined = []
        for clf, fts in trees:
            Xs = FT.build_X(src, fts)
            Xt = FT.build_X(tgt_bg, fts)
            rt = FT.RefinableTree.from_sklearn(clf)
            bg_strut(rt.root, Xs, Xt)
            refined.append((rt, fts))
        log.info(f'  bg_strut fitted ({time.time()-t0:.0f}s)')

        # qmap: per-feature maps over the union of tree features
        t0 = time.time()
        union = sorted({f for _, fts in trees for f in fts})
        qm = QuantileMapper(src, tgt_bg, union)
        log.info(f'  qmap fitted: {len(qm.maps)}/{len(union)} features mapped '
                 f'({time.time()-t0:.0f}s)')
        del src, tgt, tgt_bg

        for yr in yrs:
            out = os.path.join(OUT_DIR, f'{target}_{yr}.pkl')
            if os.path.isfile(out):
                log.info(f'  {target} {yr}: exists, skipping.')
                continue
            t0 = time.time()
            fM = load_year(target, yr, tf)
            if fM.shape[0] == 0:
                continue
            cons = {
                'bg_strut': predict_refined(refined, fM),
                'qmap': predict_sklearn(trees, qm.transform(fM)),
            }
            pd.DataFrame(cons, index=fM.index).to_pickle(out)
            log.info(f'  {target} {yr}: {fM.shape[0]} rows, 2 methods '
                     f'({time.time()-t0:.0f}s)')
            del fM
    log.info('STAGE FORECAST complete.')


# ============================================================
# Eval
# ============================================================
def prep_stats(con, tes):
    q95, bg_mask = [], np.ones(len(con), dtype=bool)
    for te in tes:
        m = (con.index >= te - window * _DAY) & (con.index < te)
        q95.append(con[m].quantile(0.95) if m.sum() else 0.)
        bg_mask &= ~m
    return np.array(q95), con.values[bg_mask]


def auc_from(q95_ev, bg, grid=None):
    grid = ths if grid is None else grid
    tpr = [(q95_ev >= th).mean() for th in grid]
    fpr = [(bg >= th).mean() for th in grid]
    return sum((fpr[i] - fpr[i + 1]) * tpr[i] for i in range(len(grid) - 1))


def stage_eval():
    log.info('=' * 60)
    log.info('STAGE EVAL: background-only adaptation')
    log.info('=' * 60)
    rows, pooled = [], {}
    for target, (base_dir, pool) in BASES.items():
        tes = parse_eruptions(target)
        base = read_pickle_retry(os.path.join(
            FORECAST_ROOT_B, base_dir, 'consensus_master.pkl'))['consensus']
        files = sorted(glob(os.path.join(OUT_DIR, f'{target}_*.pkl')))
        ad = pd.concat([read_pickle_retry(f) for f in files])
        ad = ad[~ad.index.duplicated()]
        ad.sort_index(inplace=True)

        stats = {'base': prep_stats(base, tes)}
        for m in ['bg_strut', 'qmap']:
            stats[m] = prep_stats(ad[m], tes)
        pooled[target] = stats
        for m, (q, bg) in stats.items():
            auc = auc_from(q, bg)
            rows.append({'target': target, 'method': m, 'auc': auc})
            log.info(f'  {target}/{m}: AUC = {auc:.4f}')

    # pooled 3-volcano AUC, raw vs per-station background-percentile calibrated
    log.info('  --- pooled (shared threshold across volcanoes) ---')
    for mode in ['raw', 'calibrated']:
        for m in ['base', 'bg_strut', 'qmap']:
            evs, bgs = [], []
            for target in BASES:
                q, bg = pooled[target][m]
                if mode == 'calibrated':
                    sbg = np.sort(bg)
                    q = np.searchsorted(sbg, q, side='right') / len(sbg)
                    bg = np.searchsorted(sbg, bg, side='right') / len(sbg)
                evs.append(q)
                bgs.append(bg)
            # weight background stations equally
            n = min(len(b) for b in bgs)
            rng = np.random.default_rng(0)
            bgs = [rng.choice(b, n, replace=False) for b in bgs]
            auc = auc_from(np.concatenate(evs), np.concatenate(bgs),
                           grid=np.linspace(0, 1, 201))
            rows.append({'target': 'POOLED_' + mode, 'method': m, 'auc': auc})
            log.info(f'  POOLED[{mode}]/{m}: AUC = {auc:.4f}')

    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, 'bgonly_results.csv'),
                              index=False)
    log.info(f'saved {os.path.join(OUT_DIR, "bgonly_results.csv")}')


if __name__ == '__main__':
    stage = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if stage in ('all', 'forecast'):
        stage_forecast()
    if stage in ('all', 'eval'):
        stage_eval()
    log.info('BACKGROUND-ONLY ADAPTATION done.')
