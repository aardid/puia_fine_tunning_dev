"""
Phase B external validation: apply the already-trained source-pool ensembles
to volcanoes OUTSIDE the phreatic pool — no retraining, pure out-of-sample.

The pool-curation rules ("drop WIZ", "small {ONTA,SHW}/{SHW} pools win") were
learned on FWVZ/KRVZ. Here they are tested on targets none of the ensembles
ever saw: matched WIZ-pairs isolate the Whakaari effect, and the curated
pools test whether tiny source sets generalise.

Ensembles reused (name -> trained model dir):
  all5        models/cve_WIZ_FWVZ_KRVZ_ONTA_SHW/00   {WIZ,FWVZ,KRVZ,ONTA,SHW}
  WKOS        models/phase_b/FWVZ__full              {WIZ,KRVZ,ONTA,SHW}
  KOS         models/phase_b/FWVZ__no_WIZ            {KRVZ,ONTA,SHW}
  WFOS        models/phase_b/KRVZ__full              {WIZ,FWVZ,ONTA,SHW}
  FOS         models/phase_b/KRVZ__no_WIZ            {FWVZ,ONTA,SHW}
  OS          models/phase_b/FWVZ__cur_ONTA_SHW      {ONTA,SHW}
  S           models/phase_b/FWVZ__cur_SHW           {SHW}

WIZ-effect contrasts: WKOS vs KOS, WFOS vs FOS (matched, differ only in WIZ).

Targets: stations with 4-stream hi-res (10-min) 2-day-window feature caches
and eruptions inside coverage. Eruptions outside consensus coverage are
excluded from the event set (not counted as misses).

Resumable per station-year. Usage:  python -u phase_b_external.py
"""

import os
import re
import sys
import time
import logging
import warnings
from glob import glob
from datetime import datetime, timedelta
from collections import defaultdict

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
root = r'U:\Research\EruptionForecasting\eruptions'
DATA_DIR = os.path.join(root, 'data')
FEAT_DIR = os.path.join(root, 'features')
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRIPT_DIR, 'forecasts', 'phase_b_external')

ENSEMBLES = {
    'all5': os.path.join(SCRIPT_DIR, 'models', 'cve_WIZ_FWVZ_KRVZ_ONTA_SHW', '00'),
    'WKOS': os.path.join(SCRIPT_DIR, 'models', 'phase_b', 'FWVZ__full'),
    'KOS':  os.path.join(SCRIPT_DIR, 'models', 'phase_b', 'FWVZ__no_WIZ'),
    'WFOS': os.path.join(SCRIPT_DIR, 'models', 'phase_b', 'KRVZ__full'),
    'FOS':  os.path.join(SCRIPT_DIR, 'models', 'phase_b', 'KRVZ__no_WIZ'),
    'OS':   os.path.join(SCRIPT_DIR, 'models', 'phase_b', 'FWVZ__cur_ONTA_SHW'),
    'S':    os.path.join(SCRIPT_DIR, 'models', 'phase_b', 'FWVZ__cur_SHW'),
}

TARGETS = ['PN7A', 'PVV', 'VNSS', 'BELO', 'COP', 'MBGH', 'VRLE', 'VTUN']

data_streams = ['zsc2_rsamF', 'zsc2_mfF', 'zsc2_hfF', 'zsc2_dsarF']
window = 2.
ths = np.linspace(0, 1, num=101)
HIRES_MIN_MB = 150


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


def hires_years(sta):
    per = defaultdict(dict)
    for ds in data_streams:
        for f in glob(os.path.join(FEAT_DIR, f'fm_2.00w_{ds}_{sta}_*.pkl')):
            yr = int(re.search(r'_(\d{4})\.pkl$', f).group(1))
            per[yr][ds] = os.path.getsize(f) // 1000000
    return sorted(y for y, d in per.items()
                  if len(d) == 4 and min(d.values()) >= HIRES_MIN_MB)


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


def load_year(sta, yr):
    fms = []
    for ds in data_streams:
        f = os.path.join(FEAT_DIR, f'fm_2.00w_{ds}_{sta}_{yr}.pkl')
        fm = read_pickle_retry(f)
        lo, hi = datetime(yr, 1, 1), datetime(yr + 1, 1, 1)
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
    os.makedirs(OUT_DIR, exist_ok=True)
    trees = {}
    for name, mdir in ENSEMBLES.items():
        t0 = time.time()
        trees[name] = load_trees(mdir)
        log.info(f'ensemble {name}: {len(trees[name])} trees ({time.time()-t0:.0f}s)')

    for sta in TARGETS:
        yrs = hires_years(sta)
        log.info(f'{sta}: hires years {yrs}')
        for yr in yrs:
            out = os.path.join(OUT_DIR, f'{sta}_{yr}.pkl')
            if os.path.isfile(out):
                log.info(f'  {sta} {yr}: exists, skipping.')
                continue
            t0 = time.time()
            fM = load_year(sta, yr)
            if fM.shape[0] == 0:
                log.warning(f'  {sta} {yr}: no rows!')
                continue
            cons = {}
            for name in ENSEMBLES:
                cons[name] = predict_ensemble(trees[name], fM)
            df = pd.DataFrame(cons, index=fM.index)
            df.to_pickle(out)
            log.info(f'  {sta} {yr}: {fM.shape[0]} rows x {len(ENSEMBLES)} ensembles '
                     f'({time.time()-t0:.0f}s)')
            del fM
    log.info('FORECAST complete.')


def eruption_auc(con, tes):
    """con: Series. Events limited to those with pre-window coverage."""
    q95, bg_mask = [], np.ones(len(con), dtype=bool)
    used = 0
    for te in tes:
        m = (con.index >= te - window * _DAY) & (con.index < te)
        if m.sum() > 0:
            q95.append(con[m].quantile(0.95))
            used += 1
        bg_mask &= ~m
    if not q95:
        return np.nan, 0
    q95 = np.array(q95)
    bg = con.values[bg_mask]
    auc = 0.
    tpr = [(q95 >= th).mean() for th in ths]
    fpr = [(bg >= th).mean() for th in ths]
    for i in range(len(ths) - 1):
        auc += (fpr[i] - fpr[i + 1]) * tpr[i]
    return auc, used


def stage_eval():
    rows = []
    for sta in TARGETS:
        files = sorted(glob(os.path.join(OUT_DIR, f'{sta}_*.pkl')))
        if not files:
            continue
        df = pd.concat([read_pickle_retry(f) for f in files])
        df = df[~df.index.duplicated()]
        df.sort_index(inplace=True)
        tes = parse_eruptions(sta)
        for name in ENSEMBLES:
            auc, used = eruption_auc(df[name], tes)
            rows.append({'target': sta, 'ensemble': name, 'auc': auc,
                         'n_events': used})
        r = {x['ensemble']: x['auc'] for x in rows if x['target'] == sta}
        log.info(f'{sta} (events={used}): ' +
                 ' '.join(f'{k}={v:.3f}' for k, v in r.items()))
        log.info(f'  WIZ effect (matched pairs): KOS-WKOS = {r["KOS"]-r["WKOS"]:+.3f}, '
                 f'FOS-WFOS = {r["FOS"]-r["WFOS"]:+.3f}')
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(OUT_DIR, 'external_validation.csv'), index=False)
    log.info(f'saved {os.path.join(OUT_DIR, "external_validation.csv")}')


if __name__ == '__main__':
    stage = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if stage in ('all', 'forecast'):
        stage_forecast()
    if stage in ('all', 'eval'):
        stage_eval()
    log.info('EXTERNAL VALIDATION done.')
