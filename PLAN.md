# Gridiron Edge -- PLAN

## Short-term working checklist. Pick up from where you left off.

---

## How This Document Fits

| Document | Relationship |
|----------|-------------|
| **ROADMAP.md** | Long-term direction. Tells you *which workstream* to focus on. |
| **FEATURES.md** | Full feature catalog across all domains. Reference when building features. |
| **PLAN.md** (this file) | Concrete next steps. Check boxes, move completed items to CHANGELOG. |
| **CHANGELOG.md** | Completed work. Items move here from PLAN when finished. |
| **HANDOFF.md** | How things work. Update when the system changes. |

### Status key

- `[ ]` -- Not started
- `[~]` -- In progress
- `[x]` -- Done (move to CHANGELOG)
- `[!]` -- Blocked (note blocker)

---

## Currently Active Workstreams

These are the items to work on now. Ordered by value-density (most gain first).

---

#### W0: Test Framework Build-Out ✅ DONE

_Professional testing infrastructure established. Every new feature, module, or
workstream deliverable should include corresponding tests._

**Result:** 412 tests | 40% coverage | 0 failed | 0 deselected

##### All phases complete
- [x] Phase 0 — Foundation (directory restructure, hooks, fixtures, markers)
- [x] Phase 1 — Core & Datasets (60 tests)
- [x] Phase 2 — Missing Feature Tests (63 tests)
- [x] Phase 3 — Models & Evaluation (35 tests)
- [x] Phase 4 — Ingest, Transform, Sim (65 tests)
- [x] Phase 5 — Integration & E2E (28 tests)
- [x] Deferred items resolved (tune, diagnostics, slow test removal)

##### Deferred to later workstreams (not blocked, lower priority)
- [ ] `test_engine.py` / `test_playoffs.py` — numba sim kernels (sim workstream)
- [ ] Full `test_draftkings.py` — API mocking (odds workstream)
- [ ] `test_elo_predictor.py` — Elo predictor class (elo workstream)
- [ ] `test_epa_transform.py` — EPA ETL pipeline (data pipeline workstream)
- [ ] Migrate inline imports → top-level in existing test files (cosmetic)
- [ ] Migrate local `_make_*()` helpers → shared fixtures (cosmetic)

---

### W1: Quick Wins & Unblocking ✅ DONE

*Unblock before build. Trivial fixes that gate major downstream work.*

**ROADMAP ref:** W1

- [x] **Fix DK unicode minus bug**
     - File: `ingest/odds/draftkings.py` → `_norm_display_odds_american()`
     - Issue: Unicode minus (U+2212) not handled, broke odds parsing
     - Fix: Replace unicode minus with ASCII hyphen before int conversion
     - Unlocks: All downstream odds usage

- [x] **Build DK `game_id` resolver**
     - New module: `ingest/odds/_game_id.py`
     - Maps DraftKings event identifiers to canonical `YYYY_WW_AWAY_HOME` format
     - Reverse lookup from `NFLVERSE_SHORT_TO_LONG` with relocation handling
     - Supports both intermediate and wide DataFrame formats
     - Unlocks: Edge calculations, CLV, all market intelligence

- [x] **Validate end-to-end odds join**
     - Integration test: predictions_log + dk_odds join on game_id
     - Confirmed: 100% match rate on synthetic data, nulls surface for missing odds
     - Proves: Data spine is connected

**Result:** 25 new tests | All passing | Pushed to GitHub

---

#### W3: Market Intelligence Foundation ✅ DONE

_Build the market math package. Pure functions, no data dependencies._

**ROADMAP ref:** W3

**Result:** 64 new tests │ All passing │ Pushed to GitHub

##### Completed
- [x] Created `src/gridiron_edge/market/` package (`__init__.py`, `odds_math.py`, `kelly.py`)
- [x] `market/odds_math.py` — `american_to_decimal`, `american_to_implied_prob`,
      `decimal_to_american`, `hold_pct`, `no_vig` (power + additive methods)
- [x] `market/kelly.py` — `kelly_fraction` (full Kelly), `kelly_stake` (fractional
      Kelly, default quarter-Kelly)
- [x] Power devig via bisection — no scipy dependency
- [x] Unit tests: `test_odds_math.py` (42 tests), `test_kelly.py` (22 tests)

##### Deferred to later workstreams
- [ ] `market/consensus.py` — consensus line + best available (deferred to W7, pending multi-book data)

---

### Phase 20e: Feature Engineering (Continuous) — Priorities 1–7, 14–15 ✅ DONE

*Runs in parallel with everything else. Each feature follows the existing `FeatureSpec` + `Feature` protocol.*

**ROADMAP ref:** Ongoing / intersects W2, W4
**FEATURES.md ref:** Priority Matrix (top 15), Domains 1-6

**Result:** EPA_COLS 8 → 22 │ _EXPANDED_FEATURES 63 → 107 │ All passing │ Pushed to GitHub

##### Completed
- [x] **Batch 1:** Rest differential + explosive play rate (+8 model columns)
- [x] **Batch 2:** Weather & venue wiring — verified already complete in code
- [x] **Batch 3:** PBP efficiency features — pass/rush success rate splits,
      3rd-down conversion %, red zone TD %, turnover rate, sack rate (+36 model columns)
- [x] Added `sack` to `_KEEP_COLUMNS` in PBP ingest; re-ingested all seasons
- [x] Design pattern established: add metric to `_agg_side()` + add to `EPA_COLS` → auto-propagates

##### Deferred to later (Priorities 8–13, require new data sources or complex engineering)
- [ ] Passing efficiency (CPOE, air yards) — needs additional PBP column wiring
- [ ] Pace / play count — needs drive-level data
- [ ] Score differential / garbage time filtering — needs game-state context
- [ ] Penalty rate — low signal, low priority
- [ ] Special teams EPA — separate play type aggregation
- [ ] Coaching stability — needs external data source

#### For each new feature:

- [ ] Implement `FeatureSpec` with proper `deps`
- [ ] Add to `features/registry.py` with correct ordering
- [ ] Add to `features/pipeline.py` feature list
- [ ] Update manifest expected columns
- [ ] Write unit test
- [ ] Run full pipeline to validate no breakage
- [ ] Evaluate: does the feature improve Brier score / calibration on holdout?

---

### W2: Richer Game Model Outputs

*Extend existing models to produce spread, total, scores, and uncertainty bands.*

**ROADMAP ref:** W2

- [ ] **Design output enrichment approach**
     - Decision: derive spread/total inside the model, or as a post-processing step?
     - Recommendation: post-processing -- keeps model code clean
     - Write up approach in a design doc or HANDOFF.md section before implementing

- [ ] **Add `model_spread` derivation**
     - Method: win_prob -> spread via logistic inverse or calibration curve
     - Calibrate against historical closing spreads to validate accuracy
     - File: new `models/game_prediction/post_process.py`

- [ ] **Add `model_total` projection**
     - Method: requires offensive rating decomposition or separate regression
     - Input: team offensive + team defensive ratings -> combined expected points
     - This may require a new model or a calibration layer

- [ ] **Add projected scores** (`projected_home_score`, `projected_away_score`)
     - Derive from spread + total: `home = (total + spread) / 2`, `away = (total - spread) / 2`

- [ ] **Add uncertainty bands**
     - `home_win_prob_lo`, `home_win_prob_hi` (90% credible interval)
     - Option A: bootstrap resampling of model predictions
     - Option B: Monte Carlo sim engine per game (already have the engine)
     - Option C: historical residual-based intervals

- [ ] **Add `confidence_tier` to predictions**
     - Already exists in `evaluation/metrics.py` for analysis
     - Move classification logic to prediction time
     - Tiers: High (prob > 0.65 or < 0.35), Moderate (0.55-0.65 / 0.35-0.45), Low (0.45-0.55)
     - Or: derive from uncertainty band width

- [ ] **Add `margin_std`** (standard deviation of projected margin)

- [ ] **Extend archive schema**
     - New columns in prediction archive: model_spread, model_total, home_score, away_score, win_prob_lo, win_prob_hi, confidence_tier, margin_std
     - Bump `CURRENT_SCHEMA_VERSION` in manifest
     - Ensure backward compatibility with existing archived predictions

---

## Parallel / Lower Priority

### Phase 20f: Model Variant Infrastructure

*Can be interleaved with feature work. Not blocking.*

**ROADMAP ref:** Ongoing

- [ ] Add model comparison reporting (side-by-side Brier/ECE across variants)
- [ ] Investigate ensemble approaches (blend logistic + tree predictions)
- [ ] Explore neural network variant (if complexity warrants)
- [ ] Add feature importance reporting per variant

---

## Architectural Debt / Housekeeping

*Items from the existing codebase that should be cleaned up when convenient.*

- [x] ~~Verify weather feature is fully wired end-to-end~~ ✅ Confirmed during Phase 20e
- [x] ~~Confirm dome/neutral/altitude fields flow through the full pipeline~~ ✅ Confirmed during Phase 20e
- [ ] Review dataset registry for any stale or unused keys
- [ ] Ensure all tests pass after feature additions (run full test suite)
- [ ] Update HANDOFF.md after each significant change

---

## Backlog (Not Yet Active)

*These items are defined in ROADMAP.md but not yet broken into tasks. They'll be expanded here when their dependencies are met.*

| Workstream | Blocked By | Notes |
|---|---|---|
| W4: Player Data & First Props | None (W1 ✅) | W1 complete — can start immediately |
| W4.5: Scenario Engine | W2 + W4 | Can start Phase A (impact quantification) alongside W4 |
| W5: Edge Engine | W2 (W1 ✅, W3 ✅) | The convergence point — this is where it all comes together |
| W6: Portfolio & Bet Tracking | W5 (W1 ✅, W3 ✅) | Build after edge reports are working |
| W7: Multi-Book Odds | Odds source decision (W1 ✅, W3 ✅) | Deferred pending data source evaluation |
| W8: API Serving Layer | W2 + W5 | Backend must be producing useful data first |
| W9: Frontend | W8 | Prototype exists, wiring requires API |
| W10: Real-Time / Live | W5 + W7 + W8 | Most complex, least urgent |

---

## Changelog for This Document

| Date | Change |
|------|---------|
| 2026-05-30 | Rewrote PLAN.md to align with ROADMAP.md workstream structure. Added feature engineering priority matrix tasks. Added W1-W3 concrete tasks. |
| 2026-05-31 | Marked W0 and W1 as DONE. Updated backlog dependency table to reflect W1 completion. |
| 2026-05-31 | Marked W3 as DONE. Updated backlog dependency table to reflect W3 completion. Reordered Phase 20e before W2 (quick-win features first improves model before spread calibration). |
| 2026-06-01 | Phase 20e Priorities 1–7 + 14–15 marked DONE. Feature count 63 → 107. EPA_COLS 8 → 22. Three batches: rest diff + explosive (Batch 1), weather/venue verified (Batch 2), PBP efficiency (Batch 3). Added sack to PBP ingest. Next focus: W2. |
