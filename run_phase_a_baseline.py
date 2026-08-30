"""
Phase A: Leakage-audited baseline for the Marsden fine-tuning project.

Runs the phreatic pool leave-one-eruption-out cross-validation using the
existing pipeline, then computes ROC/AUC metrics.

This is the PLAN.md Phase A baseline -- establishes benchmark AUCs for
target volcanoes (FWVZ, KRVZ) before any fine-tuning.
"""

from datetime import timedelta
from puia.model import MultiVolcanoForecastModel
from puia.data import SeismicData
from puia.utilities import datetimeify, load_dataframe
from glob import glob
from sys import platform
import pandas as pd
import numpy as np
import os, shutil, json, pickle, csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_MONTH = timedelta(days=365.25/12)
_DAY = timedelta(days=1)

# set paths
if platform == "linux" or platform == "linux2":
    root = r'/media/eruption_forecasting/eruptions'
elif platform == "win32":
    root = r'U:\Research\EruptionForecasting\eruptions'

DATA_DIR = os.path.join(root, 'data')
FEAT_DIR = os.path.join(root, 'features')
MODEL_DIR = os.path.join(root, 'models')
FORECAST_DIR = os.path.join(root, 'forecasts')

# ============================================================
# Phreatic pool configuration
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

# model hyperparameters
window = 2.
overlap = 0.75
look_forward = window
data_streams = ['zsc2_rsamF', 'zsc2_mfF', 'zsc2_hfF', 'zsc2_dsarF']
Ncl = 300
n_jobs = 6  # adjust based on your machine
scales = None

# thresholds for ROC
ths = np.linspace(0, 1, num=101)

print(f'Window: {window}')
print(f'Overlap: {overlap}')
print(f'Data streams: {data_streams}')
print(f'Ncl: {Ncl}')
print(f'n_jobs: {n_jobs}')
print(f'Pool: {list(data.keys())}')


def run_leave_one_eruption_out():
    """Leave-one-eruption-out cross-validation for the phreatic pool."""
    model_dir = os.path.join('models', 'cve_' + '_'.join(data.keys()))
    forecast_dir = os.path.join('forecasts', 'cve_' + '_'.join(data.keys()))
    print(f'Model dir: {model_dir}')
    print(f'Forecast dir: {forecast_dir}')

    # Step 1: Train model '00' on ALL eruptions and forecast over whole periods
    erup = '00'
    print('\n=== Training model 00 (all eruptions) ===')
    fm = MultiVolcanoForecastModel(
        data=data, window=window, overlap=overlap, look_forward=look_forward,
        data_streams=data_streams, feature_dir=FEAT_DIR, data_dir=DATA_DIR,
        root=str(erup), model_dir=model_dir, forecast_dir=forecast_dir, scales=scales
    )
    drop_features = ['linear_trend_timewise', 'agg_linear_trend']
    exclude_dates = {sta: None for sta in eruptions.keys()}
    fm.train(drop_features=drop_features, retrain=True, Ncl=Ncl, n_jobs=n_jobs,
             exclude_dates=exclude_dates)

    for _sta in data.keys():
        tf = datetimeify(data[_sta][1])
        ti = datetimeify(data[_sta][0])
        fm.hires_forecast(
            station=_sta, ti=ti, tf=tf, recalculate=True, n_jobs=n_jobs,
            root=os.path.join('cve_' + '_'.join(data.keys()), str(erup), str(_sta)),
            threshold=1.0
        )

    # Step 2: For each eruption, retrain excluding it and forecast
    print('\n=== Leave-one-eruption-out loop ===')
    print(f'Eruptions: {eruptions}')
    for sta in data.keys():
        print(f'\nStation: {sta}')
        for erup in eruptions[sta]:
            print(f'  Eruption {erup}')
            fm = MultiVolcanoForecastModel(
                data=data, window=window, overlap=overlap, look_forward=look_forward,
                data_streams=data_streams, feature_dir=FEAT_DIR, data_dir=DATA_DIR,
                root=sta + '_' + str(erup), model_dir=model_dir, forecast_dir=forecast_dir,
                scales=scales
            )
            drop_features = ['linear_trend_timewise', 'agg_linear_trend']

            # Exclude the held-out eruption period from training
            te1 = fm.data[sta].tes[erup]
            exclude_dates = {sta: [[te1 - _MONTH, te1 + _MONTH]]}

            # Also exclude eruptions not in our eruptions dict (from .txt files)
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

            fm.train(drop_features=drop_features, retrain=True, Ncl=Ncl, n_jobs=n_jobs,
                     exclude_dates=exclude_dates)

            # Forecast around the held-out eruption (for plots)
            tf = te1 + _DAY * 4
            ti = te1 - _MONTH
            fm.hires_forecast(
                station=sta, ti=ti, tf=tf, recalculate=True, n_jobs=n_jobs, threshold=1.,
                root=os.path.join('cve_' + '_'.join(data.keys()), sta + '_' + str(erup)),
                save=os.path.join('forecasts', 'cve_' + '_'.join(data.keys()),
                                  '_fc_eruption_' + sta + '_' + str(erup) + '.png')
            )

            # Forecast for master consensus splicing
            tf = te1 + _DAY * 4
            ti = te1 - _MONTH
            fm.hires_forecast(
                station=sta, ti=ti, tf=tf, recalculate=True, n_jobs=n_jobs, threshold=1.,
                root=os.path.join('cve_' + '_'.join(data.keys()), sta + '_' + str(erup))
            )

    # Save CV info
    os.makedirs(os.path.join('forecasts', 'cve_' + '_'.join(data.keys())), exist_ok=True)
    with open(os.path.join('forecasts', 'cve_' + '_'.join(data.keys()), 'readme.txt'), 'w') as f:
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

    # Step 3: Construct master consensus per station
    print('\n=== Constructing master consensus ===')
    for sta in data.keys():
        print(f'  {sta}')
        _path = os.path.join(forecast_dir, '00', sta)
        _con = [x[2] for x in os.walk(_path)][0]
        _con = [x for x in _con if 'consensus' in x]
        _consensus_master = pd.concat([pd.read_pickle(os.path.join(_path, x)) for x in _con])
        _consensus_master.sort_index(inplace=True)

        for erup in eruptions[sta]:
            print(f'    Splicing eruption {erup}')
            _path_erup = os.path.join(forecast_dir, sta + '_' + str(erup))
            _consensus_erup = pd.read_pickle(glob(os.path.join(_path_erup, '*consensus*.pkl'))[0])

            # Use searchsorted instead of deprecated get_loc(method='nearest')
            idx = _consensus_master.index
            l1 = idx.searchsorted(_consensus_erup.index[0])
            l2 = idx.searchsorted(_consensus_erup.index[-1])
            l1 = max(0, l1)
            l2 = min(len(idx) - 1, l2)
            _consensus_master.drop(_consensus_master.index[list(range(l1, l2 + 1))], inplace=True)
            _consensus_master = pd.concat([_consensus_master, _consensus_erup])
            _consensus_master.sort_index(inplace=True)

        _consensus_master.to_pickle(os.path.join(forecast_dir, '_consensus_master_' + sta + '.pkl'))

    print('\n=== Master consensus files saved ===')
    return forecast_dir


def compute_roc(dir_path):
    """Compute ROC curve and AUC from master consensus files."""
    print(f'\n=== Computing ROC on {dir_path} ===')

    l_fpr, l_tpr = [], []
    l_tp, l_fn, l_fp, l_tn = [], [], [], []
    l_dal, l_dal_non = [], []
    l_prec = []

    for j, th in enumerate(ths):
        c_tp, c_fn, c_tn, c_fp = 0, 0, 0, 0
        c_dal, c_dal_non = 0, 0

        for sta in data.keys():
            _consensus = pd.read_pickle(os.path.join(dir_path, '_consensus_master_' + sta + '.pkl'))

            fl_nm = os.path.join(DATA_DIR, sta + '_eruptive_periods.txt')
            with open(fl_nm, 'r') as fp:
                tes = [datetimeify(ln.rstrip()) for ln in fp.readlines()]

            for k, te in enumerate(tes):
                inds = (_consensus.index < te - window * _DAY) | (_consensus.index >= te)
                _subset = _consensus.loc[~inds]
                if len(_subset) > 0:
                    _max = _subset.quantile(q=0.95)['consensus']
                else:
                    _max = 0.
                if _max >= th:
                    c_tp += 288
                else:
                    c_fn += 288
                _consensus = _consensus.loc[inds]

            _idx_bool = _consensus['consensus'] < th
            c_tn += len(_consensus[_idx_bool])
            c_fp += len(_consensus[~_idx_bool])

            _consensus_aux = _consensus.resample('2D').quantile(q=0.95)
            _idx_bool = _consensus_aux['consensus'] < th
            c_dal_non += len(_consensus_aux[_idx_bool]) * 2
            c_dal += len(_consensus_aux[~_idx_bool]) * 2

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

    # Compute AUC (trapezoidal)
    _auc = 0.
    for i in range(len(l_fpr) - 1):
        _dx = l_fpr[i] - l_fpr[i + 1]
        _dy = l_tpr[i]
        _auc += _dx * _dy
    print(f'AUC = {_auc:.4f}')

    # Save performance metrics
    os.makedirs(dir_path, exist_ok=True)
    perf_file = os.path.join(dir_path, 'perf_pars.csv')
    with open(perf_file, 'w') as f:
        f.write('threshold,TP,FN,FP,TN,ACCU,PREC,REC,FPR,TPR,DAL,NDAL\n')
        for th, c_tp, c_fn, c_fp, c_tn, fpr, tpr, dal, ndal, prec in zip(
                ths, l_tp, l_fn, l_fp, l_tn, l_fpr, l_tpr, l_dal, l_dal_non, l_prec):
            accu = (c_tp + c_tn) / (c_tp + c_fn + c_fp + c_tn) if (c_tp + c_fn + c_fp + c_tn) > 0 else 0.
            rec = c_tp / (c_tp + c_fn) if (c_tp + c_fn) > 0 else 0.
            f.write(f'{th:.3f},{c_tp},{c_fn},{c_fp},{c_tn},{accu:.4f},{prec:.4f},{rec:.4f},{fpr:.4f},{tpr:.4f},{dal},{ndal}\n')

    # Plot ROC curve
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.plot(l_fpr, l_tpr, 'b-', linewidth=2)
    for i, th in enumerate(ths):
        if th in [0.5, 0.6, 0.7, 0.8, 0.9]:
            ax.plot(l_fpr[i], l_tpr[i], 'ok')
            ax.text(l_fpr[i], l_tpr[i] + 0.02, f'{th:.1f}', fontsize=9)
    ax.set_xlabel('False positive rate')
    ax.set_ylabel('True positive rate')
    ax.set_title(f'ROC: Phreatic Pool Leave-One-Eruption-Out (AUC={_auc:.3f})')
    ax.set_ylim([0, 1.1])
    ax.legend([f'AUC = {_auc:.3f}'])
    fig.savefig(os.path.join(dir_path, 'roc_curve.png'), dpi=150)
    ax.set_xscale('log')
    fig.savefig(os.path.join(dir_path, 'roc_curve_log.png'), dpi=150)
    plt.close(fig)

    print(f'ROC saved to {dir_path}')
    print(f'Performance metrics saved to {perf_file}')
    return _auc


if __name__ == '__main__':
    print('=' * 60)
    print('PHASE A: Phreatic Pool Leave-One-Eruption-Out Baseline')
    print('=' * 60)

    forecast_dir = run_leave_one_eruption_out()
    auc = compute_roc(forecast_dir)

    print('\n' + '=' * 60)
    print(f'PHASE A COMPLETE. AUC = {auc:.4f}')
    print('=' * 60)
