"""
Sustained-alert rule, evaluated causally over each target's full record.

Rule (operational, threshold-free in the tuning sense): at each hour, the
alert threshold is the 99th percentile of the model's own consensus over the
TRAILING 30 days (shifted 1 h — strictly causal). An alert episode begins
when the 12-hour rolling median exceeds that threshold for >= 6 consecutive
hours.

Scoring per (target, pool): an eruption is DETECTED if an episode overlaps
its final 10 days (lead = eruption time minus first onset in that span);
episodes not overlapping any [te-10d, te+4d] are FALSE ALARMS. Reported:
detections, mean lead (days), false alarms per year, alert duty fraction.

Output: forecasts/phase_b/alert_rule_table.csv
Usage:  python -u alert_rule_analysis.py
"""

import os
import sys
import time
import logging
from glob import glob
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s',
                    datefmt='%H:%M:%S', handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger(__name__)

_DAY = timedelta(days=1)
root = r'U:\Research\EruptionForecasting\eruptions'
DATA_DIR = os.path.join(root, 'data')
REPO = os.path.dirname(os.path.abspath(__file__))
FB = os.path.join(REPO, 'forecasts', 'phase_b')
FBW = os.path.join(REPO, 'forecasts', 'phase_b_wiz')

TARGETS = ['FWVZ', 'KRVZ', 'WIZ']
LOOKBACK_H = 720      # 30-day trailing background
Q = 0.99
MEDIAN_H = 12
SUSTAIN_H = 6
SEARCH_D = 10         # detection span before eruption
POST_D = 4            # post-eruption grace (not counted as false alarm)


def parse_eruptions(sta):
    with open(os.path.join(DATA_DIR, f'{sta}_eruptive_periods.txt')) as fp:
        return [datetime.strptime(ln.rstrip(), '%Y %m %d %H %M %S')
                for ln in fp.readlines() if ln.strip()]


def pool_series(target):
    out = {}
    if target == 'WIZ':
        files = sorted(glob(os.path.join(FBW, 'WIZ_*.pkl')))
        df = pd.concat([pd.read_pickle(f) for f in files])
        df = df[~df.index.duplicated()].sort_index()
        for c in df.columns:
            out[c] = df[c]
    else:
        prefix = f'{target}__'
        for d in sorted(os.listdir(FB)):
            p = os.path.join(FB, d, 'consensus_master.pkl')
            if d.startswith(prefix) and os.path.isfile(p):
                out[d[len(prefix):]] = pd.read_pickle(p)['consensus']
    return out


def episodes_of(con_1h):
    """Return list of (onset, end) alert episodes under the causal rule."""
    med = con_1h.rolling(MEDIAN_H, min_periods=MEDIAN_H // 2).median()
    thr = con_1h.rolling(LOOKBACK_H, min_periods=240).quantile(Q).shift(1)
    active = (med > thr) & thr.notna()
    eps = []
    onset = None
    run = 0
    for t, a in active.items():
        if a:
            run += 1
            if run == SUSTAIN_H:
                onset = t - timedelta(hours=SUSTAIN_H - 1)
        else:
            if onset is not None:
                eps.append((onset, t))
                onset = None
            run = 0
    if onset is not None:
        eps.append((onset, active.index[-1]))
    return eps


def main():
    rows = []
    for target in TARGETS:
        tes = parse_eruptions(target)
        pools = pool_series(target)
        log.info(f'{target}: {len(pools)} pools, {len(tes)} eruptions')
        for name, con in pools.items():
            t0 = time.time()
            con_1h = con.resample('1h').median().dropna()
            eps = episodes_of(con_1h)
            years = (con_1h.index[-1] - con_1h.index[0]).days / 365.25
            duty = sum((e - o).total_seconds() for o, e in eps) / \
                max((con_1h.index[-1] - con_1h.index[0]).total_seconds(), 1)
            detected, leads = 0, []
            fa = 0
            for o, e in eps:
                hit = False
                for te in tes:
                    if o < te and e > te - SEARCH_D * _DAY and o < te + POST_D * _DAY:
                        hit = True
                if not hit:
                    fa += 1
            for te in tes:
                onsets = [o for o, e in eps
                          if o < te and e > te - SEARCH_D * _DAY]
                onsets = [o for o in onsets if o > te - 30 * _DAY]
                if onsets:
                    detected += 1
                    leads.append((te - min(onsets)).total_seconds() / 86400)
            rows.append({
                'target': target, 'pool': name, 'n_eruptions': len(tes),
                'detected': detected,
                'mean_lead_days': float(np.mean(leads)) if leads else np.nan,
                'false_alarms_per_year': fa / years,
                'alert_duty_fraction': duty,
                'n_episodes': len(eps),
            })
            log.info(f'  {name:6s} detected {detected}/{len(tes)} '
                     f'lead={np.mean(leads) if leads else float("nan"):5.1f}d '
                     f'FA/yr={fa/years:4.1f} duty={duty:.3f} '
                     f'({time.time()-t0:.0f}s)')
    df = pd.DataFrame(rows)
    out = os.path.join(FB, 'alert_rule_table.csv')
    df.to_csv(out, index=False)
    log.info(f'saved {out}')


if __name__ == '__main__':
    main()
