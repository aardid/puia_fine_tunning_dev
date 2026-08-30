"""
Phase A, Steps 3-4 only: master consensus construction + ROC/AUC.

Standalone version of step3_master_consensus() / step4_compute_roc() from
run_phase_a_server.py that does NOT import puia (so it runs in any env with
pandas/numpy/matplotlib — no tsfresh required). Use this to (re)compute the
baseline metrics once the step 1-2 forecasts exist.

Usage:
    python -u run_phase_a_steps34.py
"""

import sys
import os
import logging
from datetime import timedelta
from glob import glob
import pandas as pd
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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
if sys.platform.startswith("linux"):
    root = r'/media/eruption_forecasting/eruptions'
elif sys.platform == "win32":
    root = r'U:\Research\EruptionForecasting\eruptions'
else:
    raise RuntimeError(f"Unsupported platform: {sys.platform}")

DATA_DIR = os.path.join(root, 'data')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FORECAST_DIR = os.path.join(SCRIPT_DIR, 'forecasts', 'cve_WIZ_FWVZ_KRVZ_ONTA_SHW')

data = {
    'WIZ':  ['2010-01-03', '2020-01-31'],
    'FWVZ': ['2006-01-01', '2015-12-31'],
    'KRVZ': ['2010-01-01', '2019-12-31'],
    'ONTA': ['2013-01-10', '2014-12-18'],
    'SHW':  ['2004-01-02', '2005-12-30'],
}
eruptions = {
    'WIZ':  [0, 1, 2, 3, 4],
    'FWVZ': [0, 1, 2],
    'KRVZ': [0, 1],
    'ONTA': [0],
    'SHW':  [0],
}

window = 2.
overlap = 0.75
look_forward = window
data_streams = ['zsc2_rsamF', 'zsc2_mfF', 'zsc2_hfF', 'zsc2_dsarF']
Ncl = 300
n_jobs = 30

ths = np.linspace(0, 1, num=101)


def datetimeify(t):
    """Minimal replacement for puia.utilities.datetimeify (string -> datetime).

    Handles the space-separated format used by *_eruptive_periods.txt
    ('2012 08 04 16 52 00') as well as ISO-style strings.
    """
    from datetime import datetime
    for fmt in ('%Y %m %d %H %M %S', '%Y-%m-%d %H:%M:%S', '%Y-%m-%d'):
        try:
            return datetime.strptime(t, fmt)
        except ValueError:
            pass
    return pd.to_datetime(t).to_pydatetime()


# ============================================================
# Step 3: Construct master consensus per station
# ============================================================
def step3_master_consensus():
    log.info('=' * 60)
    log.info('STEP 3: Constructing master consensus')
    log.info('=' * 60)

    for sta in data.keys():
        out_path = os.path.join(FORECAST_DIR, f'_consensus_master_{sta}.pkl')
        if os.path.isfile(out_path):
            log.info(f'  {sta} master consensus already exists, skipping.')
            continue

        log.info(f'  Building master consensus for {sta}...')

        # Load the model-00 forecasts for this station
        sta_path = os.path.join(FORECAST_DIR, '00', sta)
        con_files = sorted(glob(os.path.join(sta_path, '*consensus*.pkl')))
        consensus_master = pd.concat([pd.read_pickle(f) for f in con_files])
        consensus_master.sort_index(inplace=True)

        # Splice in LOO forecasts (consensus files live under the station subfolder)
        for erup in eruptions[sta]:
            loo_name = f'{sta}_{erup}'
            log.info(f'    Splicing {loo_name}...')
            loo_path = os.path.join(FORECAST_DIR, loo_name)
            loo_files = sorted(glob(os.path.join(loo_path, '*', '*consensus*.pkl')))
            if not loo_files:
                log.error(f'    No consensus file found for {loo_name}!')
                return
            consensus_erup = pd.concat([pd.read_pickle(f) for f in loo_files])
            consensus_erup.sort_index(inplace=True)

            idx = consensus_master.index
            l1 = idx.searchsorted(consensus_erup.index[0])
            l2 = idx.searchsorted(consensus_erup.index[-1])
            l1 = max(0, l1)
            l2 = min(len(idx) - 1, l2)
            consensus_master.drop(consensus_master.index[list(range(l1, l2 + 1))], inplace=True)
            consensus_master = pd.concat([consensus_master, consensus_erup])
            consensus_master.sort_index(inplace=True)

        consensus_master.to_pickle(out_path)
        log.info(f'  {sta} master consensus saved '
                 f'({consensus_master.index.min()} to {consensus_master.index.max()}, '
                 f'{len(consensus_master)} rows).')

    # Save run metadata
    os.makedirs(FORECAST_DIR, exist_ok=True)
    with open(os.path.join(FORECAST_DIR, 'readme.txt'), 'w') as f:
        f.write('data\n')
        for sta in data.keys():
            f.write(sta + '\t' + '\t'.join(data[sta]) + '\t')
        f.write('\n')
        f.write('eruptions\n')
        for sta in eruptions.keys():
            f.write(sta + '\t' + '\t'.join(str(e) for e in eruptions[sta]) + '\t')
        f.write('\n')
        f.write(f'window {window}\n')
        f.write(f'overlap {overlap}\n')
        f.write(f'look_forward {look_forward}\n')
        f.write(f'data_streams {data_streams}\n')
        f.write(f'Ncl {Ncl}\n')
        f.write(f'n_jobs {n_jobs}\n')

    log.info('STEP 3 complete.')


# ============================================================
# Step 4: Compute ROC and AUC
# ============================================================
def step4_compute_roc():
    log.info('=' * 60)
    log.info('STEP 4: Computing ROC/AUC')
    log.info('=' * 60)

    # Pre-load master consensus and eruption times once
    masters, tes_all = {}, {}
    for sta in data.keys():
        masters[sta] = pd.read_pickle(os.path.join(FORECAST_DIR, f'_consensus_master_{sta}.pkl'))
        fl_nm = os.path.join(DATA_DIR, sta + '_eruptive_periods.txt')
        with open(fl_nm, 'r') as fp:
            tes_all[sta] = [datetimeify(ln.rstrip()) for ln in fp.readlines()]

    l_fpr, l_tpr = [], []
    l_tp, l_fn, l_fp, l_tn = [], [], [], []
    l_dal, l_dal_non = [], []
    l_prec = []

    for j, th in enumerate(ths):
        c_tp, c_fn, c_tn, c_fp = 0, 0, 0, 0
        c_dal, c_dal_non = 0, 0

        for sta in data.keys():
            consensus = masters[sta].copy()

            for k, te in enumerate(tes_all[sta]):
                inds = (consensus.index < te - window * _DAY) | (consensus.index >= te)
                subset = consensus.loc[~inds]
                _max = subset.quantile(q=0.95)['consensus'] if len(subset) > 0 else 0.
                if _max >= th:
                    c_tp += 288
                else:
                    c_fn += 288
                consensus = consensus.loc[inds]

            idx_bool = consensus['consensus'] < th
            c_tn += len(consensus[idx_bool])
            c_fp += len(consensus[~idx_bool])

            consensus_aux = consensus.resample('2D').quantile(q=0.95)
            idx_bool = consensus_aux['consensus'] < th
            c_dal_non += len(consensus_aux[idx_bool]) * 2
            c_dal += len(consensus_aux[~idx_bool]) * 2

        tpr = c_tp / (c_tp + c_fn) if (c_tp + c_fn) > 0 else 0.
        fpr = c_fp / (c_fp + c_tn) if (c_fp + c_tn) > 0 else 0.
        prec = c_tp / (c_tp + c_fp) if (c_tp + c_fp) > 0 else 1.

        l_tp.append(c_tp)
        l_fn.append(c_fn)
        l_fp.append(c_fp)
        l_tn.append(c_tn)
        l_fpr.append(fpr)
        l_tpr.append(tpr)
        l_dal.append(c_dal)
        l_dal_non.append(c_dal_non)
        l_prec.append(prec)

    # AUC (trapezoidal)
    auc = 0.
    for i in range(len(l_fpr) - 1):
        dx = l_fpr[i] - l_fpr[i + 1]
        dy = l_tpr[i]
        auc += dx * dy
    log.info(f'AUC = {auc:.4f}')

    # Save performance metrics
    perf_file = os.path.join(FORECAST_DIR, 'perf_pars.csv')
    with open(perf_file, 'w') as f:
        f.write('threshold,TP,FN,FP,TN,ACCU,PREC,REC,FPR,TPR,DAL,NDAL\n')
        for th, c_tp, c_fn, c_fp, c_tn, fpr, tpr, dal, ndal, prec in zip(
                ths, l_tp, l_fn, l_fp, l_tn, l_fpr, l_tpr, l_dal, l_dal_non, l_prec):
            accu = (c_tp + c_tn) / (c_tp + c_fn + c_fp + c_tn) if (c_tp + c_fn + c_fp + c_tn) > 0 else 0.
            rec = c_tp / (c_tp + c_fn) if (c_tp + c_fn) > 0 else 0.
            f.write(f'{th:.3f},{c_tp},{c_fn},{c_fp},{c_tn},{accu:.4f},{prec:.4f},{rec:.4f},{fpr:.4f},{tpr:.4f},{dal},{ndal}\n')

    # Plot ROC
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.plot(l_fpr, l_tpr, 'b-', linewidth=2)
    for i, th in enumerate(ths):
        if th in [0.5, 0.6, 0.7, 0.8, 0.9]:
            ax.plot(l_fpr[i], l_tpr[i], 'ok')
            ax.text(l_fpr[i], l_tpr[i] + 0.02, f'{th:.1f}', fontsize=9)
    ax.plot([0, 1], [0, 1], 'k--', linewidth=0.8)
    ax.set_xlabel('False positive rate')
    ax.set_ylabel('True positive rate')
    ax.set_title(f'ROC: Phreatic Pool Leave-One-Eruption-Out (AUC={auc:.3f})')
    ax.set_ylim([0, 1.1])
    ax.legend([f'AUC = {auc:.3f}'])
    fig.savefig(os.path.join(FORECAST_DIR, 'roc_curve.png'), dpi=150)
    ax.set_xscale('log')
    fig.savefig(os.path.join(FORECAST_DIR, 'roc_curve_log.png'), dpi=150)
    plt.close(fig)

    log.info(f'ROC curve saved to {FORECAST_DIR}')
    log.info(f'Performance metrics saved to {perf_file}')
    return auc


if __name__ == '__main__':
    step3_master_consensus()
    auc = step4_compute_roc()
    log.info('=' * 60)
    log.info(f'PHASE A STEPS 3-4 COMPLETE. AUC = {auc:.4f}')
    log.info('=' * 60)
