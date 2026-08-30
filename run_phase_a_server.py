"""
Phase A: Leakage-audited baseline for the Marsden fine-tuning project.
Server version — designed for 30-core machine with resumability.

Runs the phreatic pool leave-one-eruption-out cross-validation using the
existing pipeline, then computes ROC/AUC metrics.

Usage (on Linux server):
    nohup python -u run_phase_a_server.py > run_phase_a_server.log 2>&1 &

    # Monitor progress:
    tail -f run_phase_a_server.log

    # Resume after crash — just re-run, it skips completed work:
    nohup python -u run_phase_a_server.py > run_phase_a_server.log 2>&1 &
"""

import sys
import os
import time
import logging
from datetime import timedelta
from glob import glob
import pandas as pd
import numpy as np
import pickle
import csv
import traceback

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from puia.model import MultiVolcanoForecastModel
from puia.data import SeismicData
from puia.utilities import datetimeify

# ============================================================
# Logging
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)
log = logging.getLogger(__name__)

_MONTH = timedelta(days=365.25/12)
_DAY = timedelta(days=1)

# ============================================================
# Paths — adjust for your server
# ============================================================
if sys.platform.startswith("linux"):
    root = r'/media/eruption_forecasting/eruptions'
elif sys.platform == "win32":
    root = r'U:\Research\EruptionForecasting\eruptions'
else:
    raise RuntimeError(f"Unsupported platform: {sys.platform}")

DATA_DIR = os.path.join(root, 'data')
FEAT_DIR = os.path.join(root, 'features')

# Models and forecasts stored locally (relative to this script)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, 'models', 'cve_WIZ_FWVZ_KRVZ_ONTA_SHW')
FORECAST_DIR = os.path.join(SCRIPT_DIR, 'forecasts', 'cve_WIZ_FWVZ_KRVZ_ONTA_SHW')

# ============================================================
# Configuration
# ============================================================
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
scales = None
drop_features = ['linear_trend_timewise', 'agg_linear_trend']

ths = np.linspace(0, 1, num=101)

# ============================================================
# Resumability helpers
# ============================================================
def model_trained(model_dir, root_name, ncl):
    """Check if a model has been fully trained (all Ncl trees exist)."""
    d = os.path.join(model_dir, root_name)
    if not os.path.isdir(d):
        return False
    fts_files = [f for f in os.listdir(d) if f.endswith('.fts')]
    return len(fts_files) >= ncl

def forecast_complete(forecast_dir, root_name, station):
    """Check if forecasts exist for a station under a given root."""
    d = os.path.join(forecast_dir, root_name, station)
    if not os.path.isdir(d):
        return False
    consensus_files = [f for f in os.listdir(d) if 'consensus' in f]
    return len(consensus_files) > 0

def loo_forecast_complete(forecast_dir, loo_name):
    """Check if the LOO forecast pkl exists.

    hires_forecast saves consensus files under a station subfolder:
    {forecast_dir}/{loo_name}/{station}/consensus_{year}.pkl
    """
    pattern = os.path.join(forecast_dir, loo_name, '*', '*consensus*.pkl')
    return len(glob(pattern)) > 0

# ============================================================
# Step 1: Train model 00 and forecast all stations
# ============================================================
def step1_train_and_forecast_all():
    log.info('=' * 60)
    log.info('STEP 1: Train model 00 (all eruptions) and forecast all stations')
    log.info('=' * 60)

    # Train
    if model_trained(MODEL_DIR, '00', Ncl):
        log.info('Model 00 already trained, skipping.')
    else:
        log.info('Training model 00...')
        fm = MultiVolcanoForecastModel(
            data=data, window=window, overlap=overlap, look_forward=look_forward,
            data_streams=data_streams, feature_dir=FEAT_DIR, data_dir=DATA_DIR,
            root='00', model_dir=MODEL_DIR, forecast_dir=FORECAST_DIR, scales=scales
        )
        exclude_dates = {sta: None for sta in eruptions.keys()}
        fm.train(drop_features=drop_features, retrain=False, Ncl=Ncl, n_jobs=n_jobs,
                 exclude_dates=exclude_dates)
        log.info('Model 00 training complete.')

    # Forecast each station
    for sta in data.keys():
        if forecast_complete(FORECAST_DIR, '00', sta):
            log.info(f'Model 00 forecast for {sta} already exists, skipping.')
            continue

        log.info(f'Forecasting model 00 for {sta}...')
        t0 = time.time()
        fm = MultiVolcanoForecastModel(
            data=data, window=window, overlap=overlap, look_forward=look_forward,
            data_streams=data_streams, feature_dir=FEAT_DIR, data_dir=DATA_DIR,
            root='00', model_dir=MODEL_DIR, forecast_dir=FORECAST_DIR, scales=scales
        )
        ti = datetimeify(data[sta][0])
        tf = datetimeify(data[sta][1])
        fm.hires_forecast(
            station=sta, ti=ti, tf=tf, recalculate=True, n_jobs=n_jobs,
            root=os.path.join('cve_WIZ_FWVZ_KRVZ_ONTA_SHW', '00', sta),
            threshold=1.0
        )
        log.info(f'Model 00 forecast for {sta} done in {time.time()-t0:.0f}s')

    log.info('STEP 1 complete.')

# ============================================================
# Step 2: Leave-one-eruption-out loop
# ============================================================
def step2_leave_one_out():
    log.info('=' * 60)
    log.info('STEP 2: Leave-one-eruption-out loop')
    log.info('=' * 60)

    total_eruptions = sum(len(v) for v in eruptions.values())
    done = 0

    for sta in data.keys():
        for erup in eruptions[sta]:
            loo_name = f'{sta}_{erup}'
            done += 1
            log.info(f'[{done}/{total_eruptions}] Station {sta}, eruption {erup}')

            # Check if already complete (both model trained and forecast done)
            if model_trained(MODEL_DIR, loo_name, Ncl) and loo_forecast_complete(FORECAST_DIR, loo_name):
                log.info(f'  {loo_name} already complete, skipping.')
                continue

            t0 = time.time()

            fm = MultiVolcanoForecastModel(
                data=data, window=window, overlap=overlap, look_forward=look_forward,
                data_streams=data_streams, feature_dir=FEAT_DIR, data_dir=DATA_DIR,
                root=loo_name, model_dir=MODEL_DIR, forecast_dir=FORECAST_DIR, scales=scales
            )

            # Build exclude_dates: hold out target eruption + any eruptions not in our pool
            te1 = fm.data[sta].tes[erup]
            exclude_dates = {sta: [[te1 - _MONTH, te1 + _MONTH]]}

            for _sta in eruptions.keys():
                if _sta != sta:
                    for _i in range(len(fm.data[_sta].tes)):
                        if _i not in eruptions[_sta]:
                            _te = fm.data[_sta].tes[_i]
                            if _sta in exclude_dates:
                                exclude_dates[_sta].append([_te - _MONTH, _te + _MONTH])
                            else:
                                exclude_dates[_sta] = [[_te - _MONTH, _te + _MONTH]]
            for _sta in eruptions.keys():
                if _sta not in exclude_dates:
                    exclude_dates[_sta] = None

            # Train (skip if already done)
            if model_trained(MODEL_DIR, loo_name, Ncl):
                log.info(f'  Model {loo_name} already trained.')
            else:
                log.info(f'  Training model {loo_name}...')
                fm.train(drop_features=drop_features, retrain=False, Ncl=Ncl, n_jobs=n_jobs,
                         exclude_dates=exclude_dates)
                log.info(f'  Training done.')

            # Forecast around the held-out eruption
            tf_fc = te1 + _DAY * 4
            ti_fc = te1 - _MONTH

            if loo_forecast_complete(FORECAST_DIR, loo_name):
                log.info(f'  Forecast {loo_name} already exists.')
            else:
                log.info(f'  Forecasting {loo_name} ({ti_fc.date()} to {tf_fc.date()})...')

                # Re-create model object for forecast (ensures clean state)
                fm = MultiVolcanoForecastModel(
                    data=data, window=window, overlap=overlap, look_forward=look_forward,
                    data_streams=data_streams, feature_dir=FEAT_DIR, data_dir=DATA_DIR,
                    root=loo_name, model_dir=MODEL_DIR, forecast_dir=FORECAST_DIR, scales=scales
                )

                # Forecast with plot
                fm.hires_forecast(
                    station=sta, ti=ti_fc, tf=tf_fc, recalculate=True, n_jobs=n_jobs, threshold=1.,
                    root=os.path.join('cve_WIZ_FWVZ_KRVZ_ONTA_SHW', loo_name, sta),
                    save=os.path.join(FORECAST_DIR, f'_fc_eruption_{loo_name}.png')
                )

                # Forecast without plot (for splicing — same call but no save)
                fm.hires_forecast(
                    station=sta, ti=ti_fc, tf=tf_fc, recalculate=True, n_jobs=n_jobs, threshold=1.,
                    root=os.path.join('cve_WIZ_FWVZ_KRVZ_ONTA_SHW', loo_name, sta)
                )
                log.info(f'  Forecast done.')

            elapsed = time.time() - t0
            log.info(f'  {loo_name} finished in {elapsed:.0f}s')

    log.info('STEP 2 complete.')

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
        con_files = sorted([f for f in os.listdir(sta_path) if 'consensus' in f])
        consensus_master = pd.concat([pd.read_pickle(os.path.join(sta_path, f)) for f in con_files])
        consensus_master.sort_index(inplace=True)

        # Splice in LOO forecasts
        for erup in eruptions[sta]:
            loo_name = f'{sta}_{erup}'
            log.info(f'    Splicing {loo_name}...')
            loo_path = os.path.join(FORECAST_DIR, loo_name)
            # consensus files live under the station subfolder, one per year
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
        log.info(f'  {sta} master consensus saved.')

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

    l_fpr, l_tpr = [], []
    l_tp, l_fn, l_fp, l_tn = [], [], [], []
    l_dal, l_dal_non = [], []
    l_prec = []

    for j, th in enumerate(ths):
        c_tp, c_fn, c_tn, c_fp = 0, 0, 0, 0
        c_dal, c_dal_non = 0, 0

        for sta in data.keys():
            consensus = pd.read_pickle(os.path.join(FORECAST_DIR, f'_consensus_master_{sta}.pkl'))

            fl_nm = os.path.join(DATA_DIR, sta + '_eruptive_periods.txt')
            with open(fl_nm, 'r') as fp:
                tes = [datetimeify(ln.rstrip()) for ln in fp.readlines()]

            for k, te in enumerate(tes):
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


# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    log.info('=' * 60)
    log.info('PHASE A: Phreatic Pool Leave-One-Eruption-Out Baseline')
    log.info(f'Server: {os.environ.get("COMPUTERNAME", os.environ.get("HOSTNAME", "unknown"))}')
    log.info(f'Cores: {n_jobs}')
    log.info(f'Data dir: {DATA_DIR}')
    log.info(f'Feature dir: {FEAT_DIR}')
    log.info(f'Model dir: {MODEL_DIR}')
    log.info(f'Forecast dir: {FORECAST_DIR}')
    log.info(f'Pool: {list(data.keys())} ({sum(len(v) for v in eruptions.values())} eruptions)')
    log.info('=' * 60)

    t_start = time.time()

    step1_train_and_forecast_all()
    step2_leave_one_out()
    step3_master_consensus()
    auc = step4_compute_roc()

    elapsed = time.time() - t_start
    log.info('=' * 60)
    log.info(f'PHASE A COMPLETE. AUC = {auc:.4f}')
    log.info(f'Total elapsed: {elapsed/3600:.1f} hours')
    log.info('=' * 60)
