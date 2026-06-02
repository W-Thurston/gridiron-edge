## Gridiron Edge -- PLAN

### Short-term working checklist. Pick up from where you left off.

### How This Document Fits

| Document | Relationship |
|----------|-------------|
| **ROADMAP.md** | Long-term direction. Tells you _which workstream_ to focus on. |
| **FEATURES.md** | Full feature catalog across all domains. Reference when building features. |
| **PLAN.md** (this file) | Concrete next steps. Check boxes, move completed items to CHANGELOG. |
| **CHANGELOG.md** | Completed work. Items move here from PLAN when finished. |
| **HANDOFF.md** | How things work. Update when the system changes. |

#### Status key

- [ ] -- Not started
- [~] -- In progress
- [x] -- Done (move to CHANGELOG)
- [!] -- Blocked (note blocker)

---

### Currently Active Workstreams

#### W2: Richer Game Model Outputs

_Extend existing models to produce spread, total, scores, and uncertainty bands._

**ROADMAP ref:** W2
**Unlocks:** W5 (Edge Engine), W8 (API), eventual frontend

##### Locked Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Architecture** | Post-processing step, not inside the model | Keeps model code clean and composable. Spread/bands/tier are derived from win_prob in `post_process.py`. Total is a separate regression model sharing the feature pipeline. |
| **Sigma source** | Per-model-version | rf_v3's underconfidence means its optimal sigma differs from xgb_v3 or logistic. Calibrating per-variant makes spread derivation as accurate as each model allows. |
| **Confidence tier method** | Uncertainty band width | More principled than probability cutoffs — a model with a tight 90% CI around 60% is genuinely more confident than one with a wide CI around 70%. Tier depends on Phase B (bands). |
| **Total model** | Separate regression model (C2) | Total points is genuinely different information from win probability. Same feature set + training harness, different target variable (combined score). |
| **Recalibration** | Evaluated, rejected by decision gate | rf_v3 well-calibrated on recent holdout data (ECE 0.036). Isotonic overfit the training partition (ECE → 0.000, Brier worsened on holdout). Infrastructure (4 functions, 14 tests) retained for future model versions. Root cause fix (TimeSeriesSplit) deferred to Phase 20f. |

##### Phase Ordering

```
Phase A  ──→  Phase A.5  ──→  Phase B  ──→  Phase D  ──→  Phase E
(spread +       (isotonic       (residuals,     (archive,      (validation,
 sigma cal)      recal rf_v3)    bands, tier)    pipeline)      docs)
                                    ↑
               Phase C  ────────────┘
              (total model — can start alongside A.5/B)
```

Phase C (total model) is independent of A.5 and B since it targets a separate regression
variable. It can run in parallel once Phase A establishes the post_process module structure.
It must land before Phase D wires everything into the archive.

---

##### Phase A: Spread Derivation + Per-Model Sigma Calibration ✅ DONE

_New file: `src/gridiron_edge/models/game_prediction/post_process.py`_

**Result:** 6 public functions, 33 tests, all passing.

- [x] **Created `post_process.py` with core pure functions:**
  - `win_prob_to_spread / spread_to_win_prob` — probit ↔ spread conversion
  - `calibrate_spread_sigma` — fit sigma per model via MSE minimization
  - `register_sigma / get_sigma` — per-model sigma registry with fallback
  - `enrich_predictions` — orchestrator that adds `model_spread` to predictions df
- [x] **Per-model sigma calibration (89,326 matched games, 13 model variants):**
  - Best: random_forest_v3 (sigma=13.97, spread MAE=9.92)
  - Range: 12.18 (elo_v2) to 20.35 (elo_v1)
  - Default league-wide sigma (13.86) was close for most models; per-model improves MAE by 0.01–0.04
  - elo_v1 outlier at 20.35 — probabilities too tightly clustered around 0.5
  - Vegas MAE: simpler models (Elo ~6.5) are closer to market than tree models (~7.5) — expected and arguably desirable (divergence = potential edge)
  - All 13 sigmas hardcoded in `_MODEL_SIGMAS`; TODO(W2-D) to wire into training harness
- [x] **Tests:** `tests/unit/models/test_post_process.py` (33 tests)
  - TestWinProbToSpread (8), TestSpreadToWinProb (5), TestGetSigma (5),
    TestCalibrateSigma (5), TestEnrichPredictions (10)

---

##### Phase A.5: Isotonic Recalibration of rf_v3 ✅ DONE (rejected)

_Evaluated second-pass isotonic recalibration. Decision gate rejected — rf_v3 is already well-calibrated on forward-looking data._

**Result:** 4 new functions, 14 new tests (47 total), calibrator **not saved**.

_Address known underconfidence before building downstream derivations on top._

###### Phase A.5 Result: Recalibration Rejected

**Training partition (overfit):**

| Metric | Before (raw) | After (recal) | Change |
|--------|-------------|---------------|--------|
| Brier | 0.1919 | 0.1834 | -0.008 (improved, but expected — fitting on training data) |
| ECE | 0.0771 | 0.0000 | -0.077 (perfect fit = memorization) |
| AUC | 0.7874 | 0.7904 | +0.003 |

**Holdout partition (decision gate):**

| Metric | Before (raw) | After (recal) | Verdict |
|--------|-------------|---------------|---------|
| Brier | 0.2136 | 0.2179 | ✗ Worse (+0.004) |
| ECE | **0.0365** | 0.0826 | ✗ Much worse (+0.046) |
| AUC | 0.7234 | 0.7223 | ✓ OK (within tolerance) |

**Key findings:**

1. **rf_v3 is already well-calibrated on recent data.** Holdout ECE of 0.036 is excellent. The "87% predicted → 95% actual" underconfidence was a full-backfill phenomenon averaged across 25 seasons of data, not reflective of forward-looking performance.
2. **The isotonic regression overfit.** Training ECE went to 0.000 (memorized), but on the holdout it pushed ECE from 0.036 to 0.083 — making calibration _worse_ by learning patterns from older seasons that don't hold for 2024-2025 / 2025-2026.
3. **Confidence tier analysis confirmed:** raw rf_v3 gaps on holdout were -0.009 (High), -0.011 (Moderate) — both excellent. After recalibration, Moderate gap blew out to +0.076.

**Decision:** Stay with raw rf_v3 probabilities. Sigma remains at 13.9732.

**Infrastructure retained:** `fit_recalibration`, `apply_recalibration`, `save_calibrator`, `load_calibrator` are ready for future use if a model version (e.g., after Phase 20f TimeSeriesSplit fix) shows holdout-confirmed miscalibration.


###### ⚠ Discovery: Temporal Leakage in Tree Model Training

During Phase A.5 design, we discovered that the existing tree model training
harness has temporal leakage in two places:

1. **Outer hyperparameter search:** `StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)` — `shuffle=True` destroys temporal ordering. Future season outcomes leak into folds that evaluate past seasons.
2. **Inner calibration:** `CalibratedClassifierCV(rf, method="isotonic", cv=3)` — uses sklearn's default `StratifiedKFold(3)`, which also ignores temporal ordering.

**Impact:**
- Model discrimination (AUC 0.774) is probably slightly inflated
- The first-pass isotonic calibration overfit to leaky CV folds, which likely explains the residual underconfidence — the calibrator learned a correction curve that doesn't fully generalize to truly forward-looking predictions
- Backfill evaluation numbers are honest (walk-forward season-by-season), so reported Brier/AUC reflect real out-of-sample performance
- This is a **training-time issue**, not an evaluation-time issue

**Resolution:**
- Immediate (Phase A.5): second-pass recalibration uses strict temporal splits — zero leakage
- Deferred (Phase 20f): fix the training harness itself (see Phase 20f section below)
- Immediate: ✅ Decision gate correctly rejected unnecessary recalibration

###### Phase A.5 Implementation

- [x] **Temporal-aware recalibration:**
  - `fit_recalibration(predicted_probs, actual_outcomes, seasons, *, holdout_seasons=2)` — fits `IsotonicRegression` on all seasons except the most recent N
  - Strict temporal split: fit on seasons ≤ 2022-2023, validate on 2023-2024 & 2024-2025
  - Guarantees zero leakage — calibrator never sees future outcomes
  - `apply_recalibration(probs, calibrator)` — applies fitted calibrator, clamps to (0.001, 0.999)
- [x] **Recalibrated probabilities replace originals:**
  - `enrich_predictions()` applies calibrator to `home_win_prob` and `away_win_prob` in place before deriving spread
  - No separate `home_win_prob_cal` column — the corrected value _is_ the prediction
- [x] **Calibrator persistence:**
  - `save_calibrator(calibrator, model_version, repo)` — saves to `data/models/{model_version}_cal/calibrator.joblib`
  - `load_calibrator(model_version, repo)` — loads calibrator, returns None if not found (graceful fallback)
  - Follows existing `ArtifactStore` directory convention
- [x] **Before/after evaluation report:**
  - Brier score (should improve — calibration is a Brier component)
  - ECE (should drop significantly)
  - AUC (should be unchanged — calibration doesn't affect discrimination)
  - Brier decomposition: reliability component should improve most
  - Confidence tier accuracy: re-check the 87% → 95% gap
  - Calibration curve: raw vs. recalibrated vs. perfect diagonal
- [x] **Re-run sigma calibration on recalibrated probabilities:**
  - The corrected probabilities will change the optimal sigma for rf_v3
  - Update `_MODEL_SIGMAS["random_forest_v3"]` with new value
- [x] **Decision gate:**
  - Accept if: Brier and ECE improve on the holdout seasons (truly out-of-sample) without degrading AUC
  - Reject if: calibrator overfits (improves on training seasons but not holdout)
  - If rejected: stay with raw rf_v3, accept conservative bias
- [x] **Tests:** ~15 new tests in `test_post_process.py`
  - TestFitRecalibration: fits on synthetic data, verifies monotonicity, respects temporal split
  - TestApplyRecalibration: output range (0,1), idempotent on perfectly calibrated input
  - TestSaveLoadCalibrator: round-trip via tmp_path
  - TestEnrichWithRecalibration: applies calibrator when present, skips gracefully when absent

---

##### Phase B: Historical Residuals, Margin STD, Uncertainty Bands + Confidence Tier

_Option C (historical residual-based intervals) — simplest, most defensible for V1._

- [ ] **Residual analysis utility in `post_process.py`:**
  - `compute_residuals(model_version)` — join archived predictions to actual outcomes, compute `predicted_margin - actual_margin`
  - `margin_std(model_version)` — standard deviation of residual distribution
  - Consider per-confidence-tier margin_std (tighter residuals for high-confidence games)
- [ ] **Uncertainty bands:**
  - `win_prob_lo, win_prob_hi` from `(model_spread ± z * margin_std)` converted back to probability space via probit
  - 90% credible interval: z = 1.645
  - Wider bands naturally emerge for lower-confidence predictions
- [ ] **Confidence tier from band width:**
  - Tier derived from `win_prob_hi - win_prob_lo`
  - Narrow band = High, wide band = Low
  - Empirically determine tier boundaries from the distribution of band widths
- [ ] **Integration into `enrich_predictions()`:**
  - Add `margin_std`, `win_prob_lo`, `win_prob_hi`, `confidence_tier`
- [ ] **Tests:**
  - Bands symmetric around point estimate
  - `win_prob_lo < win_prob < win_prob_hi` always holds
  - Band width increases as probability approaches 0.5
  - Known residual distributions produce expected intervals

---

##### Phase C: Total Points Regression Model + Projected Scores

_Separate model — total points is genuinely different information from win probability._

- [ ] **Total points regression model:**
  - New file: `models/game_prediction/total.py`
  - Target variable: `actual_total = home_score + away_score` (from games table)
  - Random Forest Regressor + XGBoost Regressor using expanded_107 feature set
  - Follows existing `_train_random_forest` / `_train_xgboost` pattern with MSE/MAE instead of Brier
  - Variant naming: `total_rf_v1`, `total_xgb_v1`
  - Register in `PredictorRegistry`
- [ ] **Projected scores derivation in `post_process.py`:**
  - `projected_home_score = (model_total + model_spread) / 2`
  - `projected_away_score = (model_total - model_spread) / 2`
- [ ] **Evaluation metrics for total model:**
  - MAE, RMSE against actual totals
  - Calibration: predicted vs. actual totals by bucket
  - Add to evaluation reporting
- [ ] **Tests:**
  - `home + away = total` and `home - away = spread` identities
  - Total model training smoke test (small dataset)
  - Evaluation metric correctness

---

##### Phase D: Archive Schema Extension + Pipeline Integration

_Wire everything together. Ensure backward compatibility._

- [ ] **Extend `_ARCHIVE_COLUMNS` in `archive.py`:**
  - New columns: `model_spread`, `model_total`, `projected_home_score`, `projected_away_score`, `win_prob_lo`, `win_prob_hi`, `confidence_tier`, `margin_std`
- [ ] **Schema versioning:**
  - Bump `CURRENT_SCHEMA_VERSION` in `features/manifest.py`
  - Migration logic: fill NaN for old predictions lacking new columns (same pattern as `is_backfilled` migration)
- [ ] **Update prediction pipeline:**
  - `_predict_historical_tree()` and `_predict_upcoming_tree()` call `enrich_predictions()` before returning
  - Similarly update logistic and elo prediction paths
  - Enrichment is model-agnostic — any predictor outputting `home_win_prob` gets full enrichment
- [ ] **Wire sigma calibration into training harness:**
  - After training completes, run `calibrate_spread_sigma()` on backfilled predictions
  - Store sigma in model artifact metadata (not hardcoded dict)
  - `get_sigma()` loads from artifact store, falls back to `_MODEL_SIGMAS` for models trained before this change
- [ ] **Wire recalibration fitting into training harness:**
  - After training + backfill, fit and save isotonic calibrator
  - Use temporal split (same pattern as Phase A.5)
  - Store calibrator alongside model artifact
- [ ] **Update `build_archive_rows()` in `archive.py`:**
  - Map new enrichment columns into archive rows
  - Handle absent enrichment columns gracefully (backward compat with older model versions)
- [ ] **CLI integration:**
  - `gridiron output predictions` displays spread, total, scores, tier in output table
  - `gridiron evaluate backfill` populates new columns for historical data
- [ ] **Tests:**
  - Archive round-trip with new columns
  - Backward compat: old archive loads without error, new columns are NaN
  - Full pipeline integration test: train → predict → enrich → archive → load

---

##### Phase E: Validation + Documentation

- [ ] **Validation report:**
  - Model spread vs. closing spread: MAE, correlation, bias
  - Model total vs. closing total: MAE, correlation
  - Uncertainty band coverage: % of actual outcomes within the 90% CI
  - Confidence tier accuracy: do High-confidence games win at higher rates?
- [ ] **Documentation updates:**
  - HANDOFF.md: new post_process module, updated archive schema, total model
  - FEATURES.md: document new output columns
  - PLAN.md: mark W2 phases complete
  - CHANGELOG.md: W2 completion entry

---

##### For reference: Estimated scope

| Phase | New/Modified Files | New Tests (est.) | Complexity | Status |
|-------|-------------------|------------------|------------|--------|
| A | `post_process.py` (new) | 33 | Low — pure math | ✅ Done |
| A.5 | `post_process.py` + evaluation | 14 | Low-Medium | ✅ Done (rejected) |
| B | `post_process.py` extension | ~10 | Low-Medium | Not started |
| C | `total.py` (new) | ~15 | Medium — new model | Not started |
| D | `archive.py`, `tree.py`, CLI (modify) | ~20 | Medium — integration | Not started |
| E | Validation scripts, docs | ~5 | Low | Not started |
| **Total** | **2 new + 4 modified** | **~98** | | |

---

### Parallel / Lower Priority

#### Phase 20f: Model Variant Infrastructure

_Can be interleaved with feature work. Not blocking._

**ROADMAP ref:** Ongoing

- [ ] **Fix temporal leakage in tree model training** ⚠ (discovered during W2 Phase A.5)
  - `StratifiedKFold(shuffle=True)` → `TimeSeriesSplit` or custom season-aware splitter in outer CV loop
  - `CalibratedClassifierCV(cv=3)` → `CalibratedClassifierCV(cv=TimeSeriesSplit(n_splits=3))` for inner isotonic calibration
  - Affects: `_train_random_forest()` and `_train_xgboost()` in `tree.py` (lines 290, 318, 343, 508, 618)
  - Impact: may fix the underconfidence issue at the source, making Phase A.5 recalibration unnecessary for future model versions
  - After fix: retrain all tree variants and compare Brier/ECE/AUC before and after
- [ ] Add model comparison reporting (side-by-side Brier/ECE across variants)
- [ ] Investigate ensemble approaches (blend logistic + tree predictions)
- [ ] Explore neural network variant (if complexity warrants)
- [ ] Add feature importance reporting per variant

---

### Architectural Debt / Housekeeping

_Items from the existing codebase that should be cleaned up when convenient._

- [ ] Review dataset registry for any stale or unused keys
- [ ] Ensure all tests pass after feature additions (run full test suite)
- [ ] Update HANDOFF.md after each significant change

---

### Backlog (Not Yet Active)

_These items are defined in ROADMAP.md but not yet broken into tasks. They'll be expanded here when their dependencies are met._

| Workstream | Blocked By | Notes |
|-----------|-----------|-------|
| W4: Player Data & First Props | None (W1 ✅) | W1 complete — can start immediately |
| W4.5: Scenario Engine | W2 + W4 | Can start Phase A (impact quantification) alongside W4 |
| W5: Edge Engine | W2 (W1 ✅, W3 ✅) | The convergence point — this is where it all comes together |
| W6: Portfolio & Bet Tracking | W5 (W1 ✅, W3 ✅) | Build after edge reports are working |
| W7: Multi-Book Odds | Odds source decision (W1 ✅, W3 ✅) | Deferred pending data source evaluation |
| W8: API Serving Layer | W2 + W5 | Backend must be producing useful data first |
| W9: Frontend | W8 | Prototype exists, wiring requires API |
| W10: Real-Time / Live | W5 + W7 + W8 | Most complex, least urgent |

---

### Changelog for This Document

| Date | Change |
|------|--------|
| 2026-05-30 | Rewrote PLAN.md to align with ROADMAP.md workstream structure. Added feature engineering priority matrix tasks. Added W1-W3 concrete tasks. |
| 2026-05-31 | Marked W0 and W1 as DONE. Updated backlog dependency table to reflect W1 completion. |
| 2026-05-31 | Marked W3 as DONE. Updated backlog dependency table to reflect W3 completion. Reordered Phase 20e before W2 (quick-win features first improves model before spread calibration). |
| 2026-06-01 | Phase 20e Priorities 1–7 + 14–15 marked DONE. Feature count 63 → 107. EPA_COLS 8 → 22. Three batches: rest diff + explosive (Batch 1), weather/venue verified (Batch 2), PBP efficiency (Batch 3). Added sack to PBP ingest. Next focus: W2. |
| 2026-06-01 | Pruned completed workstreams (W0, W1, W3, Phase 20e) to brief summary. Expanded W2 with full phased implementation plan (Phases A through E + A.5 recalibration). Locked decisions: per-model sigma, band-width confidence tiers, isotonic recal of rf_v3. |
| 2026-06-01 | Phase A complete (6 functions, 33 tests, 13 sigmas calibrated). Updated Phase A.5 with temporal leakage discovery: StratifiedKFold(shuffle=True) + CalibratedClassifierCV(cv=3) violate temporal ordering in tree.py training. Documented impact and resolution. Added temporal-aware recalibration design (strict season-based split). Added CV leakage fix to Phase 20f. Added decision: recalibrated probs replace originals. |
| 2026-06-01 | Phase A.5 complete. Decision gate rejected isotonic recalibration — rf_v3 well-calibrated on forward-looking data (holdout ECE 0.036, Brier worse after recal). Infrastructure (4 functions, 14 tests) retained for future use. Updated locked decisions table. |
