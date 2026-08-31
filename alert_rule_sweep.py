"""
Parameter sweep of the causal sustained-alert rule on the Ontake-only pool
consensus at Whakaari (full 2010-2020 record).

Rule parameters swept:
  q         trailing-background quantile      {0.95, 0.98, 0.99, 0.995, 0.999}
  lookback  trailing background length (days) {30, 60, 90, 180}
  medwin    consensus median window (hours)   {6, 12, 24, 48}
  sustain   min. exceedance duration (hours)  {6, 12, 24}

Scoring per combo (strictly causal thresholds, shifted 1 h):
  - detections of the 5 WIZ eruptions (episode overlapping final 10 days,
    onset before the eruption), lead time for Dec-2019
  - false alarms per year (episodes not overlapping any [te-10d, te+4d])
  - alert duty fraction

Output: forecasts/phase_b/alert_rule_sweep.csv
Usage:  python -u alert_rule_sweep.py
"""

import os
import sys
import time
import logging
from glob import glob
from datetime import datetime, timedelta
from itertools import product

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s',
                    datefmt='%H:%M:%S', handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger(__name__)

_DAY = timedelta(days=1)
REPO = os.path.dirname(os.path.abspath(__file__))
FBW = os.path.join(REPO, 'forecasts', 'phase_b_wiz')
DATA_DIR = r'U:\Research\EruptionForecasting\eruptions\data'

QS = [0.95, 0.98, 0.99, 0.995, 0.999]
LOOKBACK_D = [30, 60, 90, 180]
MEDWIN_H = [6, 12, 24, 48]
SUSTAIN_H = [6, 12, 24]
SEARCH_D, POST_D = 10, 4
TE_2019 = datetime(2019, 12, 9, 1, 11)


def parse_eruptions(sta):
    with open(os.path.join(DATA_DIR, f'{sta}_eruptive_periods.txt')) as fp:
        return [datetime.strptime(ln.rstrip(), '%Y %m %d %H %M %S')
                for ln in fp.readlines() if ln.strip()]


def episodes_from(active):
    eps, onset, run, need = [], None, 0, None
    return eps  # placeholder (replaced below)


def find_episodes(active, sustain):
    eps, onset, run = [], None, 0
    for t, a in active.items():
        if a:
            run += 1
            if run == sustain:
                onset = t - timedelta(hours=sustain - 1)
        else:
            if onset is not None:
                eps.append((onset, t))
                onset = None
            run = 0
    if onset is not None:
        eps.append((onset, active.index[-1]))
    return eps


def main():
    tes = parse_eruptions('WIZ')
    files = sorted(glob(os.path.join(FBW, 'WIZ_*.pkl')))
    df = pd.concat([pd.read_pickle(f) for f in files])
    df = df[~df.index.duplicated()].sort_index()
    con = df['O'].resample('1h').median().dropna()
    years = (con.index[-1] - con.index[0]).days / 365.25
    log.info(f'record: {con.index[0]} -> {con.index[-1]} ({years:.1f} yr), '
             f'{len(con)} hourly samples, {len(tes)} eruptions')

    log.info('precomputing medians and causal thresholds...')
    meds = {mw: con.rolling(mw, min_periods=max(mw // 2, 3)).median()
            for mw in MEDWIN_H}
    thrs = {}
    for lb, q in product(LOOKBACK_D, QS):
        t0 = time.time()
        thrs[(lb, q)] = con.rolling(lb * 24, min_periods=240).quantile(q).shift(1)
        log.info(f'  thr lookback={lb}d q={q} ({time.time()-t0:.0f}s)')

    rows = []
    for (lb, q), mw, su in product(product(LOOKBACK_D, QS), MEDWIN_H, SUSTAIN_H):
        thr = thrs[(lb, q)]
        active = (meds[mw] > thr) & thr.notna()
        eps = find_episodes(active, su)
        duty = sum((e - o).total_seconds() for o, e in eps) / \
            max((con.index[-1] - con.index[0]).total_seconds(), 1)
        fa = sum(1 for o, e in eps
                 if not any(o < te + POST_D * _DAY and e > te - SEARCH_D * _DAY
                            for te in tes))
        det, lead19 = 0, np.nan
        for te in tes:
            onsets = [o for o, e in eps
                      if o < te and e > te - SEARCH_D * _DAY
                      and o > te - 30 * _DAY]
            if onsets:
                det += 1
                if te == TE_2019:
                    lead19 = (te - min(onsets)).total_seconds() / 86400
        rows.append({'q': q, 'lookback_d': lb, 'medwin_h': mw, 'sustain_h': su,
                     'detected': det, 'lead_2019_d': lead19,
                     'false_alarms_per_year': fa / years,
                     'duty': duty, 'n_episodes': len(eps)})
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(REPO, 'forecasts', 'phase_b',
                            'alert_rule_sweep.csv'), index=False)

    hit19 = out[out.lead_2019_d.notna()]
    log.info(f'combos: {len(out)}; anticipating Dec-2019: {len(hit19)}')
    log.info('--- lowest false alarms while anticipating Dec-2019 ---')
    top = hit19.sort_values(['false_alarms_per_year', 'lead_2019_d'],
                            ascending=[True, False]).head(12)
    for _, r in top.iterrows():
        log.info(f'  q={r.q:<6} lb={r.lookback_d:>3.0f}d med={r.medwin_h:>2.0f}h '
                 f'sus={r.sustain_h:>2.0f}h  FA/yr={r.false_alarms_per_year:4.1f} '
                 f'lead2019={r.lead_2019_d:4.1f}d detected={r.detected:.0f}/5 '
                 f'duty={r.duty:.3f}')
    log.info('--- most eruptions detected (ties: fewest FA) ---')
    top2 = out.sort_values(['detected', 'false_alarms_per_year'],
                           ascending=[False, True]).head(8)
    for _, r in top2.iterrows():
        log.info(f'  q={r.q:<6} lb={r.lookback_d:>3.0f}d med={r.medwin_h:>2.0f}h '
                 f'sus={r.sustain_h:>2.0f}h  detected={r.detected:.0f}/5 '
                 f'FA/yr={r.false_alarms_per_year:4.1f} '
                 f'lead2019={r.lead_2019_d:4.1f}d')
    log.info('SWEEP done.')


if __name__ == '__main__':
    main()
