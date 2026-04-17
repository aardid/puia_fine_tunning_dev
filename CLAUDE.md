# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PUIA ("volcanoes" in Maori) is a Python library for volcanic eruption forecasting using machine learning on seismic data. It implements the methodology from Dempsey et al. (2020, Nature Comms), Dempsey et al. (2022, Bull Volcanol), Ardid et al. (2022, Nature Comms), and Ardid & Dempsey et al. (2025, Nature Comms). The current development focus is **fine-tuning generalized (transfer-learning) eruption forecasting models to individual volcanoes**, as outlined in a Marsden Fund proposal (see `project/`).

## Key Concepts

- **Ergodic precursors**: Seismic patterns before eruptions that are shared across volcanoes, enabling transfer learning
- **tsfresh features**: ~700 time series features (statistical, spectral, autocorrelation) computed from sliding windows of seismic data
- **Consensus**: The ensemble mean of binary predictions from many decision trees (0-1 scale, not a probability)
- **Pseudo-prospective testing**: Train-test split that withholds eruptions to simulate real-time forecasting conditions
- **Data streams**: Frequency-band filtered seismic amplitudes — RSAM (2-5 Hz), MF (4.5-8 Hz), HF (8-16 Hz), DSAR (MF/HF ratio)
- **Transforms**: Applied as prefixes — `zsc2_rsamF` means z-score-normalized, earthquake-filtered RSAM

## ML Pipeline

```
SeismicData (10-min samples) → Transforms (zsc2, log, inv, diff)
  → Sliding windows (e.g. 48h, 75% overlap) → tsfresh features (~700 per stream)
  → RandomUnderSampler (balance eruptive/non-eruptive) → Feature selection (Mann-Whitney U, top N)
  → Train ensemble of decision trees (GridSearchCV + ShuffleSplit)
  → Forecast: consensus = mean(tree predictions)
  → Evaluate: ROC/AUC, Alert model (TP/FP/FN/TN), Forecast Skill Score
```

## Architecture

There are **two parallel APIs** that coexist:

### Legacy API (`puia/__init__.py`)
The original single-station `ForecastModel` from Dempsey 2020. Self-contained with its own feature extraction. Uses station codes directly.

### Modern API (`puia/model.py`)
The multi-volcano transfer learning architecture from Ardid et al. 2025:
- `ForecastModel` — single-station forecaster, delegates features to `Feature` class
- `MultiVolcanoForecastModel` — multi-station, takes `data={station: [ti, tf]}` dict
- `CombinedModel` — alternative multi-station model using `FeaturesMulti` for SVD/PCA analysis
- `ForecastTransLearn` (in `forecast.py`) — applies a trained model to a new (unseen) station

**Key relationships:**
- `model.py:ForecastModel` owns a `features.py:Feature` instance and one or more `data.py:SeismicData` instances
- `data.py:SeismicData` loads `{STATION}_seismic_data.csv` and `{STATION}_eruptive_periods.txt` from the data directory
- `features.py:Feature` handles window construction, tsfresh extraction, and caching by year
- `forecast.py:Forecast` wraps consensus output with labels and eruption times for evaluation
- `forecast.py:ROC`, `AlertModel`, `FSS` provide evaluation metrics

### Data flow between classes
```
SeismicData.df (DataFrame, 10-min index)
  → Feature._construct_windows() → Feature._extract_featuresX() [tsfresh]
  → Feature matrices cached as: features/fm_{window}w_{stream}_{station}_{year}.pkl
  → model.train() loads features, undersamples, selects top Nfts, trains Ncl trees
  → Models saved as: models/{root}/DT_{000..Ncl}.pkl
  → model.forecast() loads models, predicts on test features
  → Consensus saved as: forecasts/{root}/consensus.pkl
```

## External Data Paths

Data and features live **outside** this repo (configured in `paths.txt`):
- Data: `U:\Research\EruptionForecasting\eruptions\data` — `{STATION}_seismic_data.csv`, `{STATION}_eruptive_periods.txt`
- Features: `U:\Research\EruptionForecasting\eruptions\features` — cached feature matrices

Station configuration (networks, channels, clients) is hardcoded in `puia/__init__.py` in the `STATIONS` dict and in `puia/data.py`.

## Running the Code

### Dependencies
Python 3.x with: pandas, numpy, scipy, scikit-learn, tsfresh, imbalanced-learn, obspy, joblib, matplotlib, tqdm

### Running tests
```bash
python -m puia.tests
```

### Example: Train and forecast (from example scripts)
```python
from puia.model import MultiVolcanoForecastModel
from puia.utilities import datetimeify

data = {'YNM': ['2018-01-04', '2019-06-30']}
fm = MultiVolcanoForecastModel(
    data=data, window=2.0, overlap=0.75, look_forward=2.0,
    data_streams=['zsc2_rsamF','zsc2_dsarF','zsc2_mfF','zsc2_hfF'],
    root='my_model',
    data_dir=r'U:\Research\EruptionForecasting\eruptions\data',
    feature_dir=r'U:\Research\EruptionForecasting\eruptions\features',
)
fm.train(Ncl=300, Nfts=20, classifier='DT', n_jobs=6,
         drop_features=['linear_trend_timewise','agg_linear_trend'])
fm.hires_forecast(station='YNM', ti=datetimeify('2018-01-04'),
                  tf=datetimeify('2019-06-30'), threshold=0.8)
```

### Cross-validation
`cross_validation.py` implements leave-one-eruption-out CV: trains model '00' on all data, then for each eruption retrains excluding it, forecasts its period, and splices out-of-sample forecasts into a master consensus. `ROC_cross_validation()` computes performance metrics.

## Important Parameters

| Parameter | Typical | Notes |
|-----------|---------|-------|
| `window` | 1.0-2.0 days | Shorter = finer resolution but noisier features |
| `overlap` | 0.75 | Fraction of window overlap during training |
| `look_forward` | 1.0-2.0 days | Forecast horizon, usually matches `window` |
| `Ncl` | 100-500 | Number of decision trees in ensemble |
| `Nfts` | 10-100 | Top features selected per tree (by p-value) |
| `classifier` | `'DT'` | Decision tree consistently best; 7 options: SVM,KNN,DT,RF,NN,NB,LR |
| `method` | 0.75 | Undersampling ratio (minority/majority) |
| `data_streams` | `['zsc2_rsamF',...]` | `zsc2_*F` = log-space z-score + earthquake filter (recommended) |

## Current Development: Fine-Tuning Framework

The Marsden proposal (`project/26-UOC-086_0217205536840.pdf`) defines the next phase: fine-tuning generalized models to individual volcanoes (Ruapehu, Tongariro). Four strategies to implement:

1. **Sample-weighted loss** — re-weight training to emphasize local misfit during retraining
2. **Post-hoc tree reweighting** — adjust individual tree outputs based on local performance without retraining
3. **Stacking** — meta-learner on top of tree-level predictions to capture non-linear interactions
4. **Residual learning** — freeze generalized model, train additional trees on the local residual only

These would extend `MultiVolcanoForecastModel` in `puia/model.py`.

## Papers (in `papers/`)

- Dempsey et al. (2020) Nature Comms — foundational ML pipeline, Whakaari, 100 decision trees
- Ardid et al. (2022) Nature Comms — transferable DSAR precursors across NZ volcanoes
- Dempsey et al. (2022) Bull Volcanol — probabilistic forecasting, isotonic calibration, earthquake filtering, prospective testing
- Ardid & Dempsey et al. (2025) Nature Comms — ergodic precursors, transfer learning across 24 volcanoes, AUC~0.8
- Ardid et al. (2025) JGR:ML — Steamboat Geyser, template matching, 18-hr windows, isotonic regression
