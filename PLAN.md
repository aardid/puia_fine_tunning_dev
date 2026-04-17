# Implementation Plan: Fine-Tuning Generalized Eruption Forecasting Models

This plan implements the Marsden Fund Fast-Start proposal: *"Understanding the Balance Between Universal and Volcano-Specific Controls on Pre-Eruptive Seismic Precursors at Data-Scarce Volcanoes"* (PI: Dr Alberto Ardid).

**Revision note**: Updated based on critical review addressing CV leakage, principled transfer-learning methods (SER/STRUT, TrAdaBoost), source-selection analysis, and nested CV for hyperparameter tuning.

---

## Target Volcanoes

| Station | Volcano   | Eruptions | Type     |
|---------|-----------|-----------|----------|
| FWVZ    | Ruapehu   | 3 (2006, 2007, 2009) | Phreatic |
| KRVZ    | Ruapehu   | 2 (2012-08, 2012-11) | Phreatic |
| RTZ     | Tongariro | 1 (2006)  | Phreatic |

**Challenge**: Ruapehu has 3-5 eruptions across two stations; Tongariro has only 1. RTZ with a single eruption cannot support fine-tuning evaluation -- it serves as the N=0 point on learning curves and as a leave-one-volcano-out test case only (see Phase G.3).

**Open question**: Are the two 2012 KRVZ eruptions (Aug, Nov) truly independent eruptions or one eruptive episode? If temporally coupled, treating them as independent CV folds violates independence. Needs geological assessment before Phase A.

---

## Guiding Principles

1. **Fix CV first, everything else after.** Any AUC comparison without a leakage-audited protocol is invalid.
2. **Nested CV for any hyperparameter.** Outer loop = leave-one-eruption-out; inner loop = leave-one-remaining-target-eruption-out for hyperparameter choice.
3. **Cheapest strategies first.** Source selection -> SER/STRUT tree refinement -> TrAdaBoost -> residual/stacking. Stop when improvement saturates.
4. **Every result reports AUC + calibration + CI + negative-transfer check** (AUC on non-target volcanoes to verify generalization is maintained).
5. **Do not modify `train_one_model()`.** Add new entry points (`train_transfer_model()` etc.) to preserve reproducibility of published Nat. Commun. results. Branch `fine-tuning-marsden` off `main`.
6. **Eruption-level inference, not window-level.** Metrics computed per eruption as the unit to avoid pseudo-replication from overlapping windows.

---

## Phase A: Leakage-Audited Baseline (was Phase 0)

**Goal**: Establish honest AUC benchmarks with a leakage-free CV protocol, reconciled against published numbers.

### A.1 CV protocol audit and fix

The current pipeline uses 48h windows with 75% overlap (36h shared between adjacent windows). Without temporal embargo, windows immediately adjacent to a held-out eruption's look-forward period share up to 36h of signal with test windows -- classical time-series leakage.

**Implement purged + embargoed leave-one-eruption-out:**
- Hold out contiguous segment [t_eruption - 14d, t_eruption + 2d] (tune based on feature autocorrelation)
- Remove all training windows whose 48h span touches the held-out segment
- Create `PurgedEruptionCV` splitter class in `puia/fine_tuning.py`

**For leave-one-volcano-out:** ensure no windows from the held-out volcano appear anywhere in training, including non-eruptive background (which could leak station-ID via environmental/instrumental signatures).

### A.2 Reconcile with published numbers
- Reproduce FWVZ, KRVZ, RTZ numbers from Ardid & Dempsey (2025) exactly using the original protocol
- Re-run with audited protocol; document delta
- If strict embargoing changes AUC (e.g. 0.85 -> 0.76), that's the honest baseline

### A.3 Define pools and run four baselines
1. **Phreatic pool, leave-one-eruption-out** (target eruption withheld, target station's other eruptions still in training)
2. **Phreatic pool, leave-one-volcano-out** (entire target station withheld -- the true data-scarce scenario, **primary baseline**)
3. **Magmatic pool, leave-one-volcano-out**
4. **Full (world) pool, leave-one-volcano-out**

Pool compositions from `cross_validation.py`:
- Phreatic: WIZ, FWVZ, KRVZ, ONTA, SHW (5 stations, 11 eruptions)
- Magmatic: VNSS, BELO, REF, AUH, CETU, GSTR, PVV, OKWR, SHW (9 stations, 14 eruptions)
- Full: union of above + VTUN, MBGH, VRLE (~15 stations)

Parameters: `window=2`, `overlap=0.75`, `look_forward=2`, `data_streams=['zsc2_rsamF','zsc2_mfF','zsc2_hfF','zsc2_dsarF']`, `Ncl=300`

### A.4 Metrics (used consistently across all phases)
- AUC + 95% bootstrap CI (1000 resamples over eruptions, not windows)
- Brier score and reliability diagram
- Hit rate at false-alarm rates = 5%, 10%, 25%
- Forecast lead-time distribution
- Forecast Skill Score (logarithmic, from Dempsey et al. 2022)

**Key files**: `cross_validation.py`, `puia/model.py:MultiVolcanoForecastModel`, `puia/forecast.py:MultiVolcanoROC`

**Deliverable**: Leakage-free baseline table with CIs; technical note documenting protocol differences from published pipeline.

---

## Phase B: Source Selection / Pool Curation (NEW)

**Rationale**: Before fine-tuning, ask whether the source pool is the problem. The transfer-learning literature (Pan & Yang 2009, Zhuang et al. 2020, TransferBoost) finds that *dropping dissimilar source tasks* often beats *fine-tuning on the target*. Ardid & Dempsey (2025) showed performance saturates at ~12 volcanoes, but also that not every source contributes equally.

### B.1 Per-source-volcano ablation
- For each source volcano S in the pool:
  - Train on {all sources} \ {S} \ {target}, forecast target
  - Train on {all sources} \ {target}, forecast target (full pool)
  - AUC_lift(S) = AUC_with_S - AUC_without_S
- Identifies which sources help vs. harm for each target
- Runs in O(|pool|) models -- computationally cheap

### B.2 Feature-space similarity analysis
- Compute Maximum Mean Discrepancy (MMD) or Wasserstein distance between:
  - Target volcano's non-eruptive feature distribution vs. each source's
  - Target pre-eruptive windows vs. source pre-eruptive windows
- Identify which sources are "close" to target in feature space
- Correlate with ablation results from B.1

### B.3 Curated-pool baseline
- Define curated pool: top-K most-helpful sources (positive ablation lift)
- Report curated-pool AUC as additional baseline
- **If curation alone closes most of the gap between leave-one-volcano-out and leave-one-eruption-out, subsequent phases target a smaller residual improvement**

**Deliverable**: Ranked source volcanoes by contribution to each target; MMD similarity matrix; curated-pool baseline AUC.

---

## Phase C: Tree-Structure Refinement -- SER/STRUT (replaces original Phase 2)

**Concept**: Instead of reweighting frozen trees (which can't fix splits chosen for the wrong distribution), *modify each tree's structure and thresholds* using target data. This is the principled version of "adapt the ensemble to the target."

**Background**: Segev et al. (2016) showed SER/STRUT outperform TrAdaBoost with DT base learners across datasets, especially under concept drift -- precisely the situation when moving from a diverse pool to a specific target.

### C.1 Implement SER/STRUT in `puia/fine_tuning.py`

```python
class TransferRandomForestRefiner:
    def __init__(self, base_model, method='strut'):
        # method: 'ser' (expand/reduce), 'strut' (threshold refinement), 'mix' (union)

    def refine(self, target_features, target_labels, min_target_samples=5):
        # SER: for each leaf, if target data improves by splitting, expand;
        #      if subtree is less predictive on target, prune
        # STRUT: keep tree topology; re-optimize split thresholds at each
        #        internal node using target data with combined divergence-gain
        #        + information-gain criterion

    def predict(self, features):
        # Consensus from refined forest
```

- Starting point: adapt from `Luke3D/TransferRandomForest` (GitHub) for scikit-learn DTs
- `min_target_samples_split` (5-10 windows) is the key hyperparameter -- tune via inner CV

### C.2 Regularization against target overfitting
- Require minimum number of target samples reaching a node before SER expansion or STRUT threshold change
- Inner-CV within target eruptions (excluding the held-out test eruption) for `min_target_samples`

### C.3 Negative-transfer guard
- After refinement, evaluate on held-out non-target volcanoes
- If refinement degrades their AUC beyond threshold, reject that refinement
- MIX forest (union of original + refined) provides automatic fallback

### C.4 Evaluation
- Nested CV: outer = leave-one-eruption-out on target; inner = remaining target eruptions for hyperparameters
- Report AUC lift over Phase A baseline and Phase B curated baseline, with bootstrap CIs

**Deliverable**: SER/STRUT refined ensembles for FWVZ, KRVZ; AUC improvement table; analysis of which features/thresholds get modified.

---

## Phase D: Instance-Weighted Boosting -- TrAdaBoost/TransferBoost (replaces original Phase 1)

**Concept**: Principled instance-level transfer learning. Source instances that are consistently misclassified by target-aware learners are progressively downweighted. This replaces ad hoc static sample weighting.

**Background**: TrAdaBoost (Dai et al. 2007) solves exactly this problem: lots of source data, little target data, distributional shift. TransferBoost (Eaton 2011) fixes TrAdaBoost's known weight-collapse failure mode and adds source-task-level weighting.

### D.1 Implement TransferBoost in `puia/fine_tuning.py`

```python
class TransferBoostEnsemble:
    def __init__(self, base_classifier='DT', max_depth=6, n_iterations=100):
        # Shallow DTs, not deep -- DTrBoost paper warns deep trees overfit
        # in boosting-transfer settings. Depth 4-8 is typical.

    def fit(self, source_features, source_labels, source_stations,
            target_features, target_labels):
        # Multi-source variant: each source volcano is a separate task
        # Automatically handles source selection within the boosting loop
        # Iteration count N is main hyperparameter (inner CV)

    def predict(self, features):
        # Weighted ensemble prediction
```

### D.2 Multi-source variant
- Treat each source volcano as a separate task (Multi-source TrAdaBoost)
- This subsumes Phase B's source-selection goal within the boosting framework
- Report "crossover point": how many target samples needed before transfer stops helping

### D.3 Evaluation
- Same nested CV protocol
- Explicit comparison to Phase A, B, and C
- Shallow DT base learners (depth 4-8), not the 300-tree RF

**Libraries**: `adapt-python` (has TrAdaBoost, TransferBoost, KMM); `Bin-Cao/TrAdaboost` as reference

**Deliverable**: TransferBoost AUCs; crossover-point analysis per target; source-weight evolution across boosting iterations.

---

## Phase E: Additive Logit-Residual Model (slimmed-down version of original Phase 4)

**Gate**: Only run if Phases C and D haven't saturated improvement.

**Concept**: Take the logit of the best Phase C/D forecast as a fixed offset; train a small shallow-tree booster on target data to predict the residual logit. Logit-space is mathematically cleaner than probability-space (avoids saturation compression and out-of-[0,1] predictions).

### E.1 Implementation in `puia/fine_tuning.py`

```python
class LogitResidualLearner:
    def __init__(self, base_model, n_residual_trees=20, max_depth=4, shrinkage=0.1):
        # Small ensemble, strong L2 shrinkage, shallow trees

    def fit(self, target_features, target_labels):
        # base_logit = logit(base_model.predict(target_features))
        # residual_logit = target_logit - base_logit
        # Train gradient-boosting-style model on residual_logit

    def predict(self, features):
        # final_logit = base_logit + alpha * residual_logit
        # return sigmoid(final_logit)
```

### E.2 Mixing coefficient
- `alpha` derived via cross-entropy minimization on inner-CV folds, not swept
- Few trees (10-30) with strong shrinkage to prevent overfitting

### E.3 Physical interpretation
- Feature importance from residual model = what the generalized model misses locally
- This is the primary Phase H interpretation material

**Deliverable**: Residual-corrected AUCs; residual-model feature importances.

---

## Phase F: Stacking Meta-Learner (conditional -- only if warranted)

**Decision gate**: If Phases C-E have closed >=70% of the gap between baseline and an oracle (e.g., the published volcano-specific benchmark from Ardid & Dempsey 2025), **skip this phase**. The complexity/overfitting risk isn't worth it at N<=5 eruptions.

If proceeding:
- Elastic net meta-learner on tree-level predictions
- Leave-one-target-eruption-out for meta-training
- **No overlap-based data augmentation** (overlapping windows are not independent samples)
- Only viable for FWVZ (3 eruptions) and KRVZ (2 eruptions), not RTZ

**Deliverable**: Stacking AUCs if phase is triggered; otherwise documented decision to skip.

---

## Phase G: Unified Evaluation and Significance (was Phase 5)

### G.1 Full comparison matrix
For each target station x each strategy (A baseline, B curated, C SER/STRUT, D TransferBoost, E residual, F stacking if run):
- AUC + 95% CI
- Brier score
- Hit rate @ 5%/10%/25% false alarm
- Forecast Skill Score
- Lead-time distribution

### G.2 Significance testing with honest N
- Permutation test across eruptions with Holm-Bonferroni correction across methods
- Report CI widths as a function of eruption count so readers understand inherent uncertainty
- Do NOT bootstrap within eruptions as if windows were independent

### G.3 Learning-curve / data-scarcity simulation (HEADLINE RESULT)
- For FWVZ (3 eruptions), run the full pipeline with 0, 1, 2, 3 eruptions of target data used for fine-tuning
- Show AUC-vs-target-eruption-count curve for each method
- RTZ serves as the N=0 point (leave-one-volcano-out only)
- **This is the most scientifically important figure**: it tells observatories "to get AUC X at your volcano, you need Y eruptions of local data"

### G.4 Negative-transfer audit
- Every fine-tuned model is re-evaluated on non-target volcanoes
- Any method that improves target AUC by Delta but degrades non-target AUC by >= Delta is flagged
- Report net pool-wide AUC alongside target-specific AUC

**Deliverable**: Summary table and figures for publication; learning-curve plot.

---

## Phase H: Physical Interpretation (was Phase 6)

### H.1 Universal vs. volcano-specific features
- **Universal features**: high importance in generalized model AND importance unchanged by fine-tuning on any target
- **Target-specific features**: importance significantly altered by fine-tuning in consistent direction across target eruptions
- Map to physical processes:
  - DSAR -> hydrothermal activity (Ardid et al. 2022)
  - RSAM -> general seismicity level
  - HF -> brittle fracture, shallow processes
  - MF -> intermediate-depth processes

### H.2 Fine-tuning-method-specific analysis
- Phase C (SER/STRUT): which split thresholds changed? What features gained/lost splits?
- Phase D (TransferBoost): which source volcanoes got upweighted/downweighted? Correlate with B.2 similarity
- Phase E (residual): which features dominate the residual correction?

### H.3 Temporal structure of corrections
- Are fine-tuning improvements concentrated in specific pre-eruptive phases (days vs. hours before)?
- Do corrections vary by frequency band (DSAR vs. RSAM vs. HF vs. MF)?

**Deliverable**: Feature attribution figures; narrative linking fine-tuning corrections to volcanic processes.

---

## Dependency Graph

```
A (leakage-audited baseline)
  |
  +-- B (source selection / pool curation)  <- cheap, first real result
  |     |
  |     +-- may reduce need for Phases C-F
  |
  +-- C (SER/STRUT tree refinement)  <- principled tree adaptation
  |
  +-- D (TransferBoost)  <- principled instance reweighting
  |
  +-- E (logit residual)  <- only if C/D haven't saturated
  |
  +-- F (stacking)  <- only if E hasn't saturated (gated)
        |
        v
    G (unified evaluation + learning curves)
        |
        v
    H (physical interpretation)
```

**Key difference from original plan**: strategies are ordered cheapest-first as a ladder, not parallel. Each builds on the previous and has explicit go/no-go gates.

---

## New Files to Create

| File | Purpose |
|------|---------|
| `puia/fine_tuning.py` | `PurgedEruptionCV`, `TransferRandomForestRefiner`, `TransferBoostEnsemble`, `LogitResidualLearner` |
| `puia/evaluation.py` | `evaluate_forecaster(model, eruptions, protocol, metrics)` -- shared harness for all phases |
| `fine_tuning_baseline.py` | Phase A/B evaluation scripts |
| `fine_tuning_ser_strut.py` | Phase C evaluation script |
| `fine_tuning_transfer_boost.py` | Phase D evaluation script |
| `fine_tuning_residual.py` | Phase E evaluation script |
| `evaluate_all_strategies.py` | Phase G unified comparison |
| `results/` | Output directory: one JSON per experiment run with config hash, git commit, all metrics |

## Existing Files -- NO modifications to core pipeline

| File | Note |
|------|------|
| `puia/model.py` | **Do not modify** `train_one_model()` or `train()` -- add new entry points only |
| `cross_validation.py` | Add leave-one-volcano-out mode as separate function; keep existing functions intact |

## External Libraries to Add

| Library | Purpose |
|---------|---------|
| `adapt` | TrAdaBoost, TransferBoost, KMM implementations |
| Reference: `Luke3D/TransferRandomForest` | Starting point for SER/STRUT (adapt to sklearn DTs) |

---

## Timeline Estimate

| Phase | Effort | Notes |
|-------|--------|-------|
| Phase A | 2-3 weeks | CV audit is critical path; must be right before anything else |
| Phase B | 1-2 weeks | Cheap ablation study; may reshape subsequent phases |
| Phase C | 2-3 weeks | SER/STRUT implementation and adaptation to sklearn DTs |
| Phase D | 2-3 weeks | TransferBoost implementation; multi-source variant |
| Phase E | 1-2 weeks | Only if C/D haven't saturated |
| Phase F | 1 week | Likely skipped; only if E hasn't saturated |
| Phase G | 2-3 weeks | Learning curves are the headline result |
| Phase H | 2-3 weeks | Physical interpretation for publication |

**Total**: ~14-20 weeks for full implementation and analysis.

---

## Key Risks and Mitigations

1. **CV leakage inflating all results** (the single most important risk)
   - Mitigation: Phase A is a hard gate. No downstream work starts until CV protocol is audited and embargo implemented.

2. **Too few eruptions for fine-tuning** (RTZ=1, KRVZ=2, FWVZ=3)
   - Mitigation: RTZ is N=0 point only. Nested CV prevents hyperparameter overfitting. Learning curve (Phase G.3) honestly reports what's achievable at each N.

3. **Negative transfer** -- fine-tuning degrades generalization
   - Mitigation: Phase B source-selection may solve the problem before fine-tuning. Phase G.4 audit flags any method that degrades non-target performance.

4. **Overfitting to target volcano**
   - Mitigation: Nested CV; strong regularization (shallow trees, shrinkage); negative-transfer guard in Phase C.3.

5. **Computational cost**
   - Mitigation: Pre-compute and cache all features per station-year once. Ladder structure (cheapest first, stop when saturated) avoids unnecessary compute. Estimate total budget before Phase C starts.

6. **Reproducibility of published results**
   - Mitigation: Branch `fine-tuning-marsden` off `main`. Never modify existing `train_one_model()`. Phase A.2 explicitly reconciles with published numbers.

---

## Open Questions Requiring Decisions

1. **Are the two 2012 KRVZ eruptions independent?** Affects CV validity. Needs geological assessment.
2. **Primary evaluation framing**: leave-one-volcano-out (true data-scarce) vs. leave-one-eruption-out (within-volcano)? The Marsden proposal implies the former.
3. **Whakaari 2019 eruption in source pool**: ethical/political sensitivity -- confirm approval for this research scope.
4. **Is the Multitimescale Template Matching pipeline (Ardid et al. 2024) in scope?** If yes, adds a phase between A and B.
5. **Compute resources**: laptop, workstation, or NeSI? Determines grid resolution for hyperparameter searches.
6. **Publication strategy**: one methodological paper (Phases A-D) + one interpretive paper (Phases E-H), or a single paper?
