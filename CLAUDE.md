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

Data and features live **outside** this repo on a network drive (configured in `paths.txt`):
- Data: `U:\Research\EruptionForecasting\eruptions\data` — `{STATION}_seismic_data.csv`, `{STATION}_eruptive_periods.txt`
- Features: `U:\Research\EruptionForecasting\eruptions\features` — cached feature matrices
- Models: `U:\Research\EruptionForecasting\eruptions\models` — trained model pkl files
- Forecasts: `U:\Research\EruptionForecasting\eruptions\forecasts` — forecast pkl files

Station configuration (networks, channels, clients) is hardcoded in `puia/__init__.py` in the `STATIONS` dict and in `puia/data.py`.

## Network Drive Resilience

The `U:\` network drive is **unreliable** — file reads and writes can fail with `OSError: [Errno 22] Invalid argument`, `WinError 59`, or `EOFError`. Both `load_dataframe` and `save_dataframe` in `utilities.py` have retry logic (3 attempts, 5s between retries). Feature loading in `features.py:_extract_features_single_scale` catches corrupted `.pkl` files, deletes them, and re-extracts. Forecast loading in `model.py:forecast` catches corrupted forecast files similarly.

Key issues to watch for:
- **Corrupted pkl files**: Network write failures produce truncated files that cause `EOFError` on next load
- **Missing features**: Some tsfresh features (FFT/CWT coefficients) may not be present in all years' feature matrices; `forecast_models` fills these with 0
- **Path separators**: Always use `os.path.normpath()` when constructing paths — mixing `/` and `\\` causes `FileNotFoundError` on Windows
- **Long operations**: Directory listings and file writes on `U:\` can take minutes; use timeouts and background tasks
- **Feature file sizes**: Each cached feature pickle is ~600 MB. The full features directory is ~646 GB. Loading all features for 5 stations across 4 streams is extremely I/O-bound on the network drive
- **Process stability**: Long-running scripts reading from `U:\` can die silently (no traceback) due to network timeouts or hangs. Always use `python -u` for unbuffered output and log to a file
- **File deletion**: `rm -rf` and `shutil.rmtree` on `U:\` are unreliable for large directories — prefer overwriting with `retrain=True` rather than deleting and recreating

## Running the Code

### Environment
Must use the `puia` conda environment: `conda activate puia` (Python 3.7, located at `C:\Program Files\Anaconda3\envs\puia`).

**WARNING**: Model `.pkl` files are numpy/sklearn-version-sensitive. Models saved in one environment (e.g., base conda with numpy 2.x) cannot be loaded in another (e.g., `puia` with numpy 1.x). If you hit `ModuleNotFoundError: No module named 'numpy._core'`, delete all model/forecast directories and retrain from scratch in the correct env.

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

### Phase A Baseline (`run_phase_a_server.py`)
Server version of the baseline with resumability — skips completed models and forecasts on restart.
Runs the phreatic pool (WIZ, FWVZ, KRVZ, ONTA, SHW — 12 eruptions total) leave-one-eruption-out cross-validation as the pre-fine-tuning benchmark. Steps:
1. Train model '00' on all eruptions, forecast all 5 stations
2. For each of 12 eruptions: retrain excluding it, forecast its period
3. Construct master consensus per station by splicing LOO forecasts
4. Compute ROC/AUC metrics

Uses `n_jobs=30` for 30-core server. Each LOO eruption takes ~1.5-2 hours. Full run takes ~24-30 hours.

Run on Windows server:
```powershell
conda activate puia
python -u run_phase_a_server.py > run_phase_a.log 2>&1
```

There is also `run_phase_a_baseline.py` (original non-server version, no resumability).

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

See `PLAN.md` for the full implementation plan. The Marsden proposal (`project/26-UOC-086_0217205536840.pdf`) defines fine-tuning generalized models to individual volcanoes (Ruapehu/FWVZ, Tongariro/KRVZ). Phases ordered cheapest-first with go/no-go gates:

- **Phase A**: Leakage-audited baseline (LOO cross-validation) — **DONE 2026-08-30, AUC = 0.8231**
- **Phase B**: Source selection / pool curation — **B.1 DONE 2026-08-30** (see below); B.2/B.3 running
- **Phase C**: SER/STRUT tree refinement — **DONE 2026-08-30, negative result** (see below)
- **Phase D**: TransferBoost — instance-weighted boosting (TrAdaBoost)
- **Phase E**: Logit residual model — small booster on generalized model's residuals (gated on C/D)
- **Phase F**: Stacking meta-learner (gated, likely skipped — too few eruptions)
- **Phase G**: Unified evaluation + learning curves (headline result)
- **Phase H**: Physical interpretation — universal vs. volcano-specific features

New code goes in `puia/fine_tuning.py` and `puia/evaluation.py`. **Do not modify `train_one_model()`** — add new entry points to preserve published results.

### Training Data Characteristics (phreatic pool)
- **24,830 windows** total across 5 stations with 2-day window, 0.75 overlap
- **48 positive** (eruptive) vs **24,782 negative** — extreme class imbalance (0.19% positive)
- **4,686 features** (4 streams × ~700 tsfresh features + multi-year duplicates)
- After `_drop_features`: **4,330 features**
- **35% NaN** cells overall; **3,213 of 4,686 columns** have at least one NaN (FFT/CWT coefficients missing in some years)
- After undersampling (method=0.75): **112 samples** (48 pos + 64 neg)
- `dropna(axis=1, how='any')` on 112 rows leaves ~1,400-2,300 columns depending on which rows are selected

### Phreatic Pool Configuration
```python
data = {
    'WIZ':  ['2010-01-03', '2020-01-31'],  # 5 eruptions
    'FWVZ': ['2006-01-01', '2015-12-31'],  # 3 eruptions (Ruapehu, primary target)
    'KRVZ': ['2010-01-01', '2019-12-31'],  # 2 eruptions (Tongariro)
    'ONTA': ['2013-01-10', '2014-12-18'],  # 1 eruption (Ontake)
    'SHW':  ['2004-01-02', '2005-12-30'],  # 1 eruption (St Helens)
}
```

## Bugs Fixed (uncommitted)

### Critical: Label alignment bug in `model.py:train_one_model` (line 137)
`RandomUnderSampler.fit_resample()` returns `yst` as a pandas Series preserving original indices (e.g., [5293, 12, 19847...]). After `fMt.reset_index(drop=True)` changes the feature matrix index to 0..N, constructing `pd.Series(yst > 0, index=fMt.index)` causes pandas to **reindex** the boolean series — mismatched indices become NaN, and `NaN.astype(bool)` silently converts to **True**. This made ~78% of training labels all-positive, producing degenerate depth-0 trees that predict "eruptive" everywhere (consensus saturated at 0.96-1.0, AUC≈0.53).

**Fix**: `yst = pd.Series(np.asarray(yst) > 0, index=fMt.index, dtype=bool)` — strips the old index before comparison. The legacy API (`__init__.py:1546`) uses `index=range(len(yst))` and was not affected.

**Lesson**: When `fit_resample` returns pandas objects, always convert to numpy before re-indexing. Watch for silent NaN→True conversions via `.astype(bool)`.

### Other fixes applied
- `RandomUnderSampler` API: positional `method` → `sampling_strategy=method` keyword (both `__init__.py` and `model.py`)
- Replaced broken `FeatureSelector` with direct Mann-Whitney U test for feature selection in `model.py:train_one_model`
- Network drive retry logic in `utilities.py:save_dataframe` and `load_dataframe` (3 attempts, 5s between)
- Corrupted pkl detection in `features.py:_extract_features_single_scale` and `model.py:forecast`
- Path normalization with `os.path.normpath()` in `ForecastModel` and `CombinedModel`
- Missing feature columns filled with 0 in `forecast_models` and `forecast_one_model`
- `zsc2_rsamF` data stream lookup added to `hires_forecast` plotting
- `reset_index(drop=True)` in `train()` to avoid duplicate label issues from multi-station concat
- `load_dataframe` pkl loading changed from manual `pickle.load` to `pd.read_pickle` (handles compression)

## Phase A Progress

Models and forecasts stored locally under `models/cve_WIZ_FWVZ_KRVZ_ONTA_SHW/` and `forecasts/cve_WIZ_FWVZ_KRVZ_ONTA_SHW/`.

| Step | Status |
|------|--------|
| Step 1: Model 00 + all station forecasts | Done (13 models × 301 .fts); coverage gaps filled 2026-08-30 via `fill_model00_gaps.py` |
| Step 2: LOO training + forecasting (12 eruptions) | Done (all 12 have consensus) |
| Step 3: Master consensus construction | **Done (2026-08-30)** — `_consensus_master_{sta}.pkl` |
| Step 4: ROC/AUC computation | **Done (2026-08-30)** — **AUC = 0.8231** (`perf_pars.csv`, `roc_curve.png`) |

**Baseline result (headline for Phase A)**: LOO AUC = 0.8231, consistent with the ~0.8 out-of-sample AUC in Ardid & Dempsey et al. (2025). At threshold 0.5: 8/12 eruptions detected (TPR 0.67, FPR 0.20); at 0.8: 4/12 (TPR 0.33, FPR 0.03). An earlier compute with incomplete WIZ/KRVZ coverage gave AUC 0.8151.

**Path bug fixed (2026-08-30)**: Step 3 and `loo_forecast_complete` globbed `{loo_name}/*consensus*.pkl`, but `hires_forecast` saves consensus under a station subfolder (`{loo_name}/{station}/consensus_{year}.pkl`), so Step 3 always aborted with "No consensus file found". Fixed in `run_phase_a_server.py`; steps 3-4 also available standalone (no tsfresh/puia import needed) in `run_phase_a_steps34.py`.

**Coverage gaps filled (2026-08-30)**: crashed step-1 runs had left model-00 forecasts partial (WIZ ended 2017, KRVZ ended 2013; `forecast_complete()` passes if *any* consensus file exists). `fill_model00_gaps.py` filled WIZ 2018-2020 and KRVZ 2014-2019 by predicting the frozen model-00 trees directly on the cached hi-res feature matrices (no tsfresh/puia needed; runs on a laptop against `U:\`). Per-tree forecast pkls were NOT written for these years, only the consensus — a later `forecast(recalculate=False)` will just re-predict them cheaply. **Residual caveat**: KRVZ 2014 dsarF features cover only ~26% of the year (~96 days) in the cache (`fm_2.00w_zsc2_dsarF_KRVZ_2014.pkl` is 87 MB vs ~320 MB for full years; the `puia_rep/features` dir has no KRVZ files), so KRVZ 2014 consensus is sparse — full coverage would need a tsfresh extraction on the server.

Log files: `run_phase_a_v7.log` (v7 died mid-WIZ_4 on network timeout), `run_phase_a_v9.log` (v9 died on numpy version mismatch). Models were retrained in the `puia` env to resolve compatibility.

## Phase B Progress (2026-08-30)

Phase B runs **entirely locally** via `phase_b_local.py` (no server, no tsfresh): stage 1 caches the 12-h training-grid rows from the big cached feature pickles into `C:\Users\aar135\puia_local_cache` (~25k windows × ~4.3k features, reusable by later phases); stage 2 trains leave-target-volcano-out ensembles replicating `train_one_model` (vectorised Mann-Whitney, same seeds/grid); stage 3 streams target hi-res features and computes eruption AUC per variant. `phase_b2_similarity.py` computes MMD feature-space similarity (B.2).

**B.1 single-source ablation results** (target forecast AUC; `forecasts/phase_b/ablation_{target}.csv`):

| Variant | FWVZ (Ruapehu) | KRVZ (Tongariro) |
|---------|------|------|
| full pool (all sources, no target) | 0.845 | 0.649 |
| no WIZ | **0.932** | **0.763** |
| no KRVZ / no FWVZ | 0.773 | 0.662 |
| no ONTA | 0.843 | 0.666 |
| no SHW | 0.857 | 0.606 |

**Key finding**: WIZ (Whakaari) — the source with the most eruptions (5) — *harms* transfer to both NZ targets: dropping it lifts FWVZ +0.087 and KRVZ +0.115. KRVZ helps FWVZ (+0.072 lift) and SHW helps KRVZ (+0.043). Caveat: with 3 (FWVZ) / 2 (KRVZ) target eruptions, TPR is coarse (steps of 1/3, 1/2); variant separation is mostly on the FPR side (~10 yrs non-eruptive data each). Note the leave-volcano-out full-pool AUCs (0.845/0.649) are not directly comparable to the Phase A pooled LOO AUC (0.823) — different test sets.

**B.3 results — exhaustive pool landscape, all 15 source subsets per target (2026-08-30)** (`forecasts/phase_b/ablation_{target}.csv`, `delta_vs_full` column):

FWVZ (Ruapehu), sorted: {SHW} **0.9625**, {ONTA,SHW} 0.9566, {KRVZ,ONTA,SHW} 0.9320, {KRVZ,SHW} 0.9028, {ONTA} 0.9006, {KRVZ,ONTA} 0.8940, {KRVZ} 0.8863 | {WIZ,KRVZ,ONTA} 0.8565, full 0.8446, {WIZ,KRVZ,SHW} 0.8431, {WIZ,KRVZ} 0.8389, {WIZ,ONTA,SHW} 0.7725, {WIZ,SHW} 0.7680, {WIZ,ONTA} 0.7615, {WIZ} 0.7553.

KRVZ (Tongariro), sorted: {ONTA,SHW} **0.8848**, {FWVZ,SHW} 0.8608, {FWVZ} 0.7713, {FWVZ,ONTA,SHW} 0.7632, {ONTA} 0.7621, {FWVZ,ONTA} 0.7551, {SHW} 0.6739 | {WIZ,FWVZ,SHW} 0.6659, {WIZ,ONTA,SHW} 0.6615, {WIZ,SHW} 0.6555, full 0.6485, {WIZ,FWVZ,ONTA} 0.6056, {WIZ,FWVZ} 0.6029, {WIZ,ONTA} 0.5990, {WIZ} 0.5170.

**Landscape findings**: (1) **Perfect WIZ separation for both targets** — every WIZ-free pool outperforms every WIZ-containing pool (FWVZ: ≥0.886 vs ≤0.857; KRVZ: ≥0.674 vs ≤0.666). Whakaari membership is the single dominant pool factor; e.g. {SHW}→FWVZ drops 0.963→0.768 when WIZ is added. (2) Best pools are small and foreign: FWVZ ← {SHW} alone (0.963); KRVZ ← {ONTA,SHW} (0.885) — no NZ volcano. (3) Source value is non-additive; single-source ablation reshuffles once WIZ is out. (4) Curation alone lifts FWVZ 0.845→0.963 and KRVZ 0.649→0.885 — per the B.3 gate, phases C-E now target a small residual. **Selection-bias caveat**: the max over 15 pools scored on 2-3 target eruptions is an optimistic (oracle-curated) upper bound; Phase G must use a nested/held-out protocol for headline claims.

**B.2 results (2026-08-30)** (`forecasts/phase_b/mmd_*.csv`): RBF-MMD² between stations' standardized training features (1296 complete-case common features). Non-eruptive: FWVZ↔KRVZ are the closest pair (0.012); WIZ is *closer* to both targets (0.017-0.025) than ONTA/SHW (0.03-0.06). **Feature-space similarity does NOT predict transfer value**: Spearman(MMD, ablation lift) = 0.05 (non-eruptive) / 0.19 (pre-eruptive, n=4 samples per station — very noisy), n=8 pairs. WIZ sits close to the targets in background feature space yet transfers worst — so pool curation cannot be shortcut by MMD screening in this pool; the harm is likely in precursor *dynamics*, not feature distributions.

## Phase C Progress (2026-08-30) — negative result

`puia/fine_tuning.py` implements SER/STRUT (Segev et al. 2016) with mutable dict-trees and vectorised prediction (extraction verified 100% faithful to sklearn). Simplifications: STRUT uses pure class-weighted Gini gain (no divergence-to-source term); unreachable nodes keep source structure. `phase_c_local.py` runs refine→forecast→eval with the Phase-A-mirror protocol: background from ensemble refined on all target data, each eruption spliced from the ensemble refined without it (±1 month) — eruptions are out-of-sample.

**Results (`forecasts/phase_c/phase_c_results.csv`): every refined variant is worse than its unrefined base.**

| target/base | base | STRUT | SER | mix |
|---|---|---|---|---|
| FWVZ/full | 0.845 | 0.519 | 0.704 | 0.527 |
| FWVZ/no_WIZ | 0.932 | 0.751 | 0.881 | 0.783 |
| KRVZ/full | 0.649 | 0.582 | 0.551 | 0.576 |
| KRVZ/no_WIZ | 0.763 | 0.554 | 0.468 | 0.519 |

**Diagnosis**: not an FPR explosion — the opposite. Refined ensembles' consensus collapses in range (base: 2.9% of background windows >0.8; STRUT/SER: 0%; SER mean consensus 0.025 vs base 0.352). With only 8-12 positive windows from 2-3 eruptions, refinement (SER especially, in-sample balanced-acc 0.99) memorises those eruptions' exact feature signatures; the held-out eruption doesn't match them, so refined trees rarely fire — the ergodic generality of the source ensemble is erased. Classic few-shot overfit / catastrophic forgetting, exactly what the Marsden plan flagged as the core risk.

**Interpretation for the paper**: with this few target eruptions, structural tree refinement destroys value while pool curation (Phase B) adds it — "choose your teachers, don't rewrite the lessons." Follow-up variants worth trying before closing C: (1) STRUT with the original divergence-to-source term or threshold shrinkage (interpolate old→new), (2) per-tree undersampled refinement sets to preserve ensemble diversity, (3) leaf-probability-only recalibration (no structural change — the mildest adaptation, close to tree-vote reweighting). Otherwise proceed to Phase D with C as the cautionary baseline.

## Papers (in `papers/`)

- Dempsey et al. (2020) Nature Comms — foundational ML pipeline, Whakaari, 100 decision trees
- Ardid et al. (2022) Nature Comms — transferable DSAR precursors across NZ volcanoes
- Dempsey et al. (2022) Bull Volcanol — probabilistic forecasting, isotonic calibration, earthquake filtering, prospective testing
- Ardid & Dempsey et al. (2025) Nature Comms — ergodic precursors, transfer learning across 24 volcanoes, AUC~0.8
- Ardid et al. (2025) JGR:ML — Steamboat Geyser, template matching, 18-hr windows, isotonic regression
