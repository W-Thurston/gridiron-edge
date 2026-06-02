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
| **Architecture** | Post-processing step, not inside the model | Keeps model code clean and composable. Spread/bands/tier are derived from win_prob in a new `post_process.py` module. Total is a separate regression model sharing the feature pipeline. |
| **Sigma source** | Per-model-version | rf_v3's underconfidence means its optimal sigma differs from xgb_v3 or logistic. Calibrating per-variant makes spread derivation as accurate as each model allows. |
| **Confidence tier method** | Uncertainty band width | More principled than probability cutoffs — a model with a tight 90% CI around 60% is genuinely more confident than one with a wide CI around 70%. Tier depends on Phase B (bands). |
| **Total model** | Separate regression model (C2) | Total points is genuinely different information from win probability. Same feature set + training harness, different target variable (combined score). |
| **Recalibration** | Isotonic recal of rf_v3 as Phase A.5 | Address known underconfidence (87% predicted → 95% actual) before deriving spreads and bands. All downstream outputs improve for free. |

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

##### Phase A: Spread Derivation + Per-Model Sigma Calibration

_New file: `src/gridiron_edge/models/game_prediction/post_process.py`_

- [ ] **Create `post_process.py` with core pure functions:**
  - `win_prob_to_spread(home_win_prob, sigma)` — probit inverse: `spread = sigma * Φ⁻¹(home_win_prob)`
  - `enrich_predictions(df)` — orchestrator that adds all post-processed columns (grows with each phase)
- [ ] **Per-model sigma calibration:**
  - Fit sigma empirically per model version: regress actual margin of victory against model win_prob using the probit link
  - Validate derived model_spread against historical closing spreads (from DK odds log)
  - Store calibrated sigma per model version as a constant or config
- [ ] **Tests:** `tests/unit/models/test_post_process.py`
  - Known-value tests: 50% → spread 0, 75% → spread ~9.3 (with default sigma)
  - Symmetry: `spread(p) = -spread(1-p)`
  - Edge cases: probabilities near 0 and 1

---

##### Phase A.5: Isotonic Recalibration of rf_v3

_Address known underconfidence before building downstream derivations on top._

- [ ] **Isotonic recalibration wrapper:**
  - `CalibratedClassifierCV(method='isotonic', cv=5)` on rf_v3
  - Store isotonic mapping as a separate artifact alongside the model
  - New variant: `random_forest_v3_cal` — uncalibrated rf_v3 stays for comparison
- [ ] **Before/after evaluation report:**
  - Brier score (should improve — calibration is a Brier component)
  - ECE (should drop significantly)
  - AUC (should be unchanged — calibration doesn't affect discrimination)
  - Calibration curve plot: raw vs. calibrated vs. perfect diagonal
  - Confidence tier accuracy: re-check the 87% → 95% gap
- [ ] **Decision gate:**
  - If calibration improves Brier and ECE without degrading AUC → `random_forest_v3_cal` becomes new best model
  - If it somehow hurts → stay with raw rf_v3, accept conservative bias
- [ ] **Tests:**
  - Calibrated probabilities are valid (0–1 range, monotonic with raw)
  - Calibrated model produces identical AUC to uncalibrated
  - Brier decomposition shows improved reliability component

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

| Phase | New/Modified Files | New Tests (est.) | Complexity |
|-------|-------------------|------------------|------------|
| A | `post_process.py` (new) | ~15 | Low — pure math |
| A.5 | `post_process.py` + evaluation | ~10 | Low-Medium |
| B | `post_process.py` extension | ~10 | Low-Medium |
| C | `total.py` (new) | ~15 | Medium — new model |
| D | `archive.py`, `tree.py`, CLI (modify) | ~20 | Medium — integration |
| E | Validation scripts, docs | ~5 | Low |
| **Total** | **2 new + 4 modified** | **~75** | |

---

### Parallel / Lower Priority

#### Phase 20f: Model Variant Infrastructure

_Can be interleaved with feature work. Not blocking._

**ROADMAP ref:** Ongoing

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
