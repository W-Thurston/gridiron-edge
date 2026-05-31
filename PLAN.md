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

#### W0: Test Framework Build-Out

_Establish professional testing infrastructure before building new features. Every new feature, module, or workstream deliverable should include corresponding tests._

**ROADMAP ref:** Cross-cutting / Code Quality

##### Phase 0 — Foundation ✅ DONE
- [x] Restructure `tests/` into `unit/`, `integration/`, `e2e/` subdirectories
- [x] Add root `conftest.py` with auto-markers by directory
- [x] Build `tests/fixtures/dataframes.py` — 9 centralized DataFrame factories
- [x] Build `tests/fixtures/repos.py` — composable `MiniRepoBuilder`
- [x] Update `tests/integration/conftest.py` to use `MiniRepoBuilder`
- [x] Add `.pre-commit-config.yaml` with pre-commit (unit) and pre-push (integration/e2e) hooks
- [x] Add pytest markers to `pyproject.toml` (unit, integration, e2e, slow, network)
- [x] Add coverage config to `pyproject.toml` (`fail_under = 40`, ratchet up)
- [x] Fix all drifted tests (home_field, weather, tree_models, features_pipeline)
- [x] Mark slow model training tests with `@pytest.mark.slow`

##### Phase 1 — Core & Datasets
- [ ] `unit/core/test_constants.py` — constants exist, expected values
- [ ] `unit/core/test_paths.py` — path construction, repo detection
- [ ] `unit/core/test_settings.py` — config loading, defaults
- [ ] `unit/datasets/test_registry.py` — `dataset_path()`, all keys resolve
- [ ] `unit/datasets/test_loaders.py` — CSV/Parquet load, missing file handling
- [ ] `unit/datasets/test_writers.py` — write + read roundtrip
- [ ] `unit/datasets/test_accessor.py` — delegation methods

##### Phase 2 — Missing Feature Tests
- [ ] `unit/features/test_divisional.py`
- [ ] `unit/features/test_epa.py`
- [ ] `unit/features/test_primetime.py`
- [ ] `unit/features/test_record.py`
- [ ] `unit/features/test_schedule_strength.py`
- [ ] `unit/features/test_venue_hfa.py`
- [ ] `unit/features/test_base.py` (FeatureSpec)
- [ ] `unit/features/test_registry.py` (validate_ordering)

##### Phase 3 — Models & Evaluation
- [ ] `unit/models/test_base.py` — Predictor/Trainable protocols
- [ ] `unit/models/test_registry.py` — variant registration
- [ ] `unit/models/test_artifact.py` — ArtifactStore save/load
- [ ] Rewrite `test_tree_models.py` training tests with mocked tiny estimators (replace @pytest.mark.slow)
- [ ] `unit/evaluation/test_backfill.py`, `test_select.py`, `test_tune.py`

##### Phase 4 — Ingest, Transform, Sim (mock-heavy)
- [ ] `unit/ingest/test_draftkings.py` — use dk_payload_fixture
- [ ] `unit/ingest/test_odds_store.py` — Parquet roundtrip
- [ ] `unit/transform/test_games_nflverse.py` — schema mapping
- [ ] `unit/sim/test_types.py`, `test_engine.py`, `test_playoffs.py`

##### Phase 5 — Integration & E2E
- [ ] `integration/test_dataset_roundtrip.py` — write → read → schema preserved
- [ ] `integration/test_model_train_predict.py` — train → save → load → predict → archive
- [ ] `e2e/test_cli_workflows.py` — full run-data-pipeline via CLI
- [ ] `e2e/test_prediction_pipeline.py` — ingest → model → evaluate (no CLI)

---

### W1: Quick Wins & Unblocking

*Unblock before build. These are trivial fixes that gate major downstream work.*

**ROADMAP ref:** W1

- [ ] **Fix DK unicode minus bug**
     - File: `ingest/odds/draftkings.py` -> `_norm_display_odds_american()`
     - Issue: Unicode minus (U+2212) not handled, breaks odds parsing
     - Fix: Replace unicode minus with ASCII hyphen before int conversion
     - Effort: < 30 minutes
     - Unlocks: All downstream odds usage

- [ ] **Build DK `game_id` resolver**
     - Map DraftKings event identifiers to canonical `YYYY_WW_AWAY_HOME` format
     - Approach options:
       a. Parse team names from DK event title + match to schedule by date/teams
       b. Build a lookup table from DK event metadata
     - Test: Join dk_odds_log to games table, verify match rate > 95%
     - Effort: 1-2 sessions
     - Unlocks: Edge calculations, CLV, all market intelligence

- [ ] **Validate end-to-end odds join**
     - After above fixes, run: load predictions_log + dk_odds_log, join on game_id
     - Confirm: predictions and odds align for the same games
     - This proves the data spine is connected

---

### Phase 20e: Feature Engineering (Continuous)

*Runs in parallel with everything else. Each feature follows the existing `FeatureSpec` + `Feature` protocol.*

**ROADMAP ref:** Ongoing / intersects W2, W4
**FEATURES.md ref:** Priority Matrix (top 15), Domains 1-6

#### Quick wins (existing data, minimal wiring)

- [ ] **Rest differential** (Domain 5)
     - `rest_diff = home_days_rest - away_days_rest`
     - Already have each team's rest -- just subtract
     - File: `features/team/rest.py` (extend existing)

- [ ] **Explosive play rate** (Domain 1)
     - % of plays gaining 20+ yds (pass) or 10+ yds (rush)
     - Source: PBP data (already ingested)
     - File: new `features/team/explosiveness.py`

#### Weather & venue wiring (highest priority -- data exists, features don't)

- [ ] **Wire weather into prediction features** (Domain 6, Priority #1)
     - OpenWeatherMap ingest exists and is idempotent
     - Need: feature module that reads weather data and produces model-ready columns
     - Columns: `temperature_f`, `wind_mph`, `precipitation_flag`
     - File: extend `features/team/weather.py` or verify it's fully wired

- [ ] **Wire dome/neutral/altitude into features** (Domain 5, Priority #2)
     - Already in schema (v3+) but not confirmed as prediction features
     - `is_dome`, `is_neutral_site`, `altitude_ft`
     - Indoor override: zero out weather features when dome=True
     - File: extend `features/team/venue_hfa.py`

#### Efficiency features from PBP (low cost, game model enrichment)

- [ ] **Success rate (pass/rush split)** (Domain 1, Priority #3)
     - % of plays with positive EPA, split by pass and rush
     - Rolling window (match EPA window convention)
     - File: new `features/team/success_rate.py`

- [ ] **3rd down conversion % (off + def)** (Domains 1+2, Priority #4)
     - Off: conversions / 3rd down attempts
     - Def: opponent conversions / opponent 3rd down attempts
     - Rolling window
     - File: new `features/team/third_down.py`

- [ ] **Red zone TD % (off + def)** (Domains 1+2, Priority #5)
     - Off: TDs / red zone trips
     - Def: opponent TDs / opponent red zone trips
     - File: new `features/team/red_zone.py`

- [ ] **Turnover differential / game** (Domain 3, Priority #6)
     - (Forced fumbles + INTs) - (Fumbles lost + INTs thrown) per game
     - Rolling window
     - File: new `features/team/turnovers.py`

- [ ] **Sack rate (off + def)** (Domains 1+2, Priority #7)
     - Off: sacks allowed / dropbacks
     - Def: sacks / opponent dropbacks
     - File: new `features/team/sacks.py`

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

### W3: Market Intelligence Foundation

*Build the market math package. Pure functions, no data dependencies.*

**ROADMAP ref:** W3

- [ ] **Create `src/gridiron_edge/market/` package**
     - `__init__.py`
     - Initial modules: `odds_math.py`, `kelly.py`

- [ ] **`market/odds_math.py`**
     - `american_to_decimal(odds: int) -> float`
     - `american_to_implied_prob(odds: int) -> float`
     - `decimal_to_american(dec: float) -> int`
     - `no_vig(odds_a: int, odds_b: int) -> tuple[float, float]`
       - Implement both multiplicative (power) method and additive method
       - Default to power method
     - `hold_pct(odds_a: int, odds_b: int) -> float`
     - Edge cases to handle: even odds (+100/-100), heavy favorites (> -500), positive both sides

- [ ] **`market/kelly.py`**
     - `kelly_fraction(model_prob: float, american_odds: int) -> float`
     - `kelly_stake(model_prob: float, american_odds: int, bankroll: float, fraction: float = 0.25) -> float`
     - Guard: return 0 if edge is negative (no bet)
     - Guard: cap at `fraction * full_kelly` (default quarter-Kelly)

- [ ] **Unit tests for all math functions**
     - Test known values (e.g., -110 -> implied 52.38%)
     - Test roundtrip: american -> decimal -> american
     - Test edge cases: +100, -100, -10000, +10000
     - Test Kelly with no edge (should return 0)
     - Test Kelly with large edge (should be capped)

- [ ] **`market/consensus.py`** (after W7 odds source decision)
     - `consensus_line(snapshots: list[LineSnapshot]) -> float`
     - `best_available(snapshots: list[LineSnapshot], side: str) -> tuple[str, float, int]`
     - Defer until multi-book data is available

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

- [ ] Verify weather feature is fully wired end-to-end (ingest -> transform -> feature -> model)
- [ ] Confirm dome/neutral/altitude fields flow through the full pipeline
- [ ] Review dataset registry for any stale or unused keys
- [ ] Ensure all tests pass after feature additions (run full test suite)
- [ ] Update HANDOFF.md after each significant change

---

## Backlog (Not Yet Active)

*These items are defined in ROADMAP.md but not yet broken into tasks. They'll be expanded here when their dependencies are met.*

| Workstream | Blocked By | Notes |
|------------|-----------|-------|
| W4: Player Data & First Props | W1 (for odds context) | Can start player data ingestion independently |
| W4.5: Scenario Engine | W2 + W4 | Can start Phase A (impact quantification) alongside W4 |
| W5: Edge Engine | W1 + W2 + W3 | The convergence point -- this is where it all comes together |
| W6: Portfolio & Bet Tracking | W1 + W3 + W5 | Build after edge reports are working |
| W7: Multi-Book Odds | W1 + W3 + odds source decision | Deferred pending data source evaluation |
| W8: API Serving Layer | W2 + W5 | Backend must be producing useful data first |
| W9: Frontend | W8 | Prototype exists, wiring requires API |
| W10: Real-Time / Live | W5 + W7 + W8 | Most complex, least urgent |

---

## Changelog for This Document

| Date | Change |
|------|---------|
| 2026-05-30 | Rewrote PLAN.md to align with ROADMAP.md workstream structure. Added feature engineering priority matrix tasks. Added W1-W3 concrete tasks. |
