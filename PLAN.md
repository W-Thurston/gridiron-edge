## Gridiron Edge — Development Plan

**Purpose:** single source of truth for _what to build next_ and _why_.
Updated at the start and close of every workstream.

| Document | Role |
|----------|------|
| **PLAN.md** (this file) | What is planned, what is active, what is deferred |
| **CHANGELOG.md** | What was built and when (completed workstream details) |
| **HANDOFF.md** | How the system works today (architecture, workflows, operations) |

#### Status key

| Tag | Meaning |
|-----|---------|
| Done | Done — details in CHANGELOG.md |
| In progress | In progress |
| Planned | Planned / blocked |
| Deferred | Deferred |

---

### Currently Active Workstream: W4 — Player Data & First Prop Models

**Goal:** Complete the player-level data layer, feature pipeline, and first prop
projection models. Achieve **M3** (first prop edge report).

**Design decisions (locked in):**

- **NaN strategy:** Drop rows with NaN for now. Mark every drop-site in code with
  `# TODO(nan): <reason>` for future audit. A dedicated research item is in the
  backlog to investigate data-driven imputation methods (see NaN Research Backlog
  below).
- **Model architecture:** One PropTrainer per prop type (e.g., QBPassYardsTrainer),
  mirroring how game models work — one model predicts the raw stat, post-processing
  derives P(over), lean, confidence tier.
- **Model types:** Champion/challenger across ElasticNet, Ridge, RandomForest,
  XGBoost (and potentially LightGBM). Same promotion gate pattern as game models
  (primary metric: MAE; guardrails: coverage, calibration).
- **Feature philosophy:** "Throw everything in, let the model decide." No manual
  feature selection up front. Systematic feature importance / elimination (e.g.,
  permutation importance, AIC/BIC) is a future workstream.
- **Prop odds:** Historical prop odds deferred. Upcoming-week prop odds via
  DraftKings ingest extension. Prop edge calculations will work once DK prop
  ingest is wired.
- **Stat families for V1:** QB passing yards, QB rushing yards, RB rushing yards,
  WR receiving yards, TE receiving yards (5 total).

---

#### Phase A — Player Data Foundation  ✅ Complete

| Step | Status | Deliverable |
|------|--------|-------------|
| A1: Player stats ingest | ✅ Done | `ingest/nflverse/player_stats.py` — nflreadpy, 1999–2024, ~5K rows/season, 42 cols. Stored at `data/raw/player_stats/player_stats_{season}.parquet`. |
| A2: Player game logs + game_id | ✅ Done | `transform/clean/player_stats.py` → `data/cleaned/player_game_logs.parquet` — 138,349 rows, 44 cols, 4,067 unique players, 0% game_id null, 0 duplicate (player_id, game_id). |
| A3: Rolling features | ✅ Done | `features/player/rolling.py` — `ROLLING_STAT_COLS` covering passing/rushing/receiving stats at L3/L6 windows. shift(1) for lookahead prevention. |
| A4: Matchup features | ✅ Done | `features/player/matchup.py` — 28 features (14 defensive-allowed stats × 2: L6 rolling avg + rank). Rankings: 1 = toughest, 32 = most generous. Joined via opponent_team. |

**Existing tests:**
`tests/unit/ingest/test_player_stats.py` (9 tests, 3 classes),
`tests/unit/transform/test_player_stats.py` (7 tests, 3 classes),
`tests/unit/features/test_player_rolling.py` (12 tests, 3 classes),
`tests/unit/features/test_player_matchup.py` (11 tests, 5 classes)

---

#### Phase A — Audit Findings

_Automated audit: 2026-06-05. Manual code review: same session.
B1 remediation: 2026-06-10. Final audit: **45 pass · 0 fail · 4 warn (non-blocking)**._
_Audit script: `scripts/audit_w4_phase_a.py`_

##### 🔴 Blocking Issues — All Resolved

**F1 — game_id null rate was 0.0007%** ✅ Resolved
- Was 1 row: Steve Bono, 1999 wk 9, `team=None`, `opponent_team=None`.
- Fix: Added `dropna(subset=["team", "opponent_team"])` guard in
  `clean_player_stats()` before the schedule join.

**F2 — game_id fixture bug in `test_player_matchup.py`** ✅ Resolved (false positive)
- Code review was based on garbled SharePoint `.py.txt` rendering.
  Actual fixture code was already correct.

**F3 — 46 duplicate (player_id, game_id) rows** ✅ Resolved (discovered during B1)
- Root cause: Schedule join mismatches — two different weeks of a player's
  data got assigned the same incorrect game_id (e.g., T.Duckett on ATL
  assigned `2002_03_NO_CHI`). The player's team didn't appear in the
  game_id at all.
- Fix: Added dedup in `clean_player_stats()` that drops all copies of
  duplicate `(player_id, game_id)` rows since the game_id is wrong for
  both. Removed 46 rows (23 mismatched pairs) out of 138K.

##### 🟡 Should-Fix Items — All Resolved

**W1 — `recent_team` column** ✅ Resolved
- Audit script expected `recent_team` but nflreadpy uses `team`. Fixed
  audit script. No source code change needed.

**W2 — 3 columns with >80% NaN** ✅ Resolved (analysis complete, no code change)
- Per-position NaN analysis confirmed these are **position-specific stats
  applied to all positions**. When filtered to the relevant position:
  - `passing_cpoe`: 91% overall → **27% for QBs** (manageable)
  - `passing_epa`: 87% overall → **>95% coverage for QBs across all seasons**
  - `pacr`: 87% overall → QB-only composite, similar to passing_epa
- No code changes needed. NaN rates are acceptable per-position.

**W3 — 4 broad `except Exception:` clauses** ✅ Resolved
- All 4 catch nflreadpy API calls (network, parse, schema errors).
  Broad catch is justified. Added justification comments to each.

**W4 — `base.py` is 551 lines** 📝 Noted for C1
- Consider splitting when adding multi-model support: `_types.py`,
  `_evaluate.py`, `base.py` (PropTrainer only).

**W5 — Verify `test_no_lookahead_week1` assertions** ✅ Resolved
- Test has proper `assert pd.isna()` on both mean and std for week 1.

**W6 — Add `# TODO(nan)` comments** ✅ Resolved
- Added to `passing_epa` and `passing_cpoe` in `_PASSING_STATS` in
  `rolling.py`.

##### 🟢 Non-Blocking Observations

**O1–O2:** Audit script bugs (import name mismatch, positional arg error).
Not source code bugs. Script has been partially fixed; remaining warns
are non-blocking.

**O3:** `target_share` and `air_yards_share` in `ROLLING_STAT_COLS` are
pre-computed shares from nflreadpy. Rolling averages of shares are
semantically valid. B2 will also compute shares from raw counts. Both
approaches retained — let the model decide.

**O4:** Prop model subclass tests have 5 tests each (adequate for
scaffolding, expand during Phase C).

**O5:** ElasticNet-only in prop subclasses. Expected — multi-model added
in C1.

**O6:** Non-skill positions (P, OT, DB, CB, LB) present in
player_game_logs. The `is_skill` column filters these for prop models.

##### NaN Landscape (from audit)

| Column | Overall NaN% | Per-Position NaN% | Notes |
|--------|-------------|-------------------|-------|
| `passing_cpoe` | 90.62% | QB: 27% | Older seasons + low-attempt games |
| `pacr` | 87.42% | QB: ~13% | Passer rating composite |
| `passing_epa` | 87.13% | QB: <5% | >95% coverage all seasons for QBs |
| `rushing_epa` | 58.73% | RB: 8%, QB: 17% | Players with 0 carries → no EPA |
| `wopr` | 25.09% | WR: 24%, TE: 23% | Weighted opportunity rating |
| `racr` | 21.53% | RB: 25%, WR: <5% | Receiver air conversion ratio |
| `target_share` | 19.95% | WR: 19%, TE: 18% | Non-receivers have no targets |
| `receiving_epa` | 19.63% | WR: <5%, TE: <5% | Non-receivers |
| `air_yards_share` | 15.15% | WR: 14%, TE: 13% | Non-receivers |

**Key insight:** When filtered to the relevant position, NaN rates are
manageable. The feature builder (B4) filters by position before training.

---

#### Phase B — Feature Pipeline Completion

##### B1: Audit & Stabilize Existing Features  ✅ Complete

**What was done (2026-06-10):**
1. Fixed F1: Added `dropna(subset=["team", "opponent_team"])` guard —
   removed 1 row (Steve Bono, 1999 wk9, team=None)
2. Fixed F3: Added `(player_id, game_id)` dedup — removed 46 rows
   (23 schedule join mismatch pairs)
3. Confirmed F2 was a false positive (SharePoint rendering artifact)
4. Ran per-position NaN analysis (W2) — confirmed NaN rates are
   position-specific noise, not data quality issues
5. Added justification comments to 4 broad except clauses (W3)
6. Verified `test_no_lookahead_week1` has proper assertions (W5)
7. Added `# TODO(nan)` comments to `passing_epa` and `passing_cpoe` (W6)
8. Fixed audit script `recent_team` → `team` expectation (W1)
9. Regenerated `player_game_logs.parquet`: 138,349 rows, 0 null game_ids,
   0 duplicate (player_id, game_id)
10. Final audit: **45 pass · 0 fail · 4 warn (non-blocking)**

##### B2: Usage Features  ✅ Complete

**What was done (2026-06-10):**
1. Created `features/player/usage.py` — 6 rolling usage features:
   `usage_{target,carry,touch}_share` × L{3,6}
2. Shares computed from raw counts (targets, carries), not nflreadpy
   pre-computed shares — gives the model both approaches
3. Division by zero produces 0.0 (not NaN)
4. Per-game share intermediates dropped — only rolling features exposed
5. Created `tests/unit/features/test_usage.py` — 16 tests, 5 classes
6. Snap % deferred — nflreadpy does not expose snap count data

**Features built:**
- `usage_target_share_L3`, `usage_target_share_L6` — player targets / team
  total targets (WR, TE)
- `usage_carry_share_L3`, `usage_carry_share_L6` — player carries / team
  total carries (RB)
- `usage_touch_share_L3`, `usage_touch_share_L6` — (targets + carries) /
  team total touches (all skill positions)

**Implementation pattern:**
- `_compute_team_totals()` → `_compute_per_game_shares()` →
  `_rolling_shares()` → `build_usage_features()`
- All rolling computations use `shift(1)` for lookahead prevention
- Season boundaries respected by default (`cross_season=False`)

##### B3: Game Context Features for Props  ✅ Complete

**What was done (2026-06-10):**
1. Created `features/player/game_context.py` — 6 game context features
   joined from `data/cleaned/NFL_wk_by_wk_cleaned.csv`
2. Features: `is_home`, `game_spread`, `over_under`, `implied_team_total`,
   `is_dome`, `rest_days`
3. No shift(1) needed — all features are known pre-game
4. Full team name → abbreviation mapping (37 entries, 1999–present)
5. Created `tests/unit/features/test_game_context.py` — 28 tests, 9 classes

**Features built:**
- `is_home` — derived from game_id format (4th segment = home team)
- `game_spread` — team perspective: favorite gets negative, underdog positive
- `over_under` — total points line from Vegas
- `implied_team_total` — `(over_under - game_spread) / 2`
- `is_dome` — ROOF in {dome, closed}
- `rest_days` — calendar days since team's previous game

**Key design decision:** These features are NOT shifted — spread, total,
dome, and rest are all known before kickoff and are legitimate predictors
at prediction time.


##### B4: Unified Prop Feature Builder  ✅ Complete

**What was done (2026-06-10):**
1. Created `features/player/builder.py` — single entry point
   `build_prop_features(position_filter=["QB"])` that chains all 4
   feature builders and returns a training-ready DataFrame
2. Created `features/player/_columns.py` — `PROP_FEATURE_COLS` built
   programmatically from component modules (stays in sync automatically)
3. Refactored all 4 builders (rolling, matchup, usage, game_context) to
   accept optional `df` parameter — enables single parquet load
4. Position filtering, NaN drop with `# TODO(nan)`, row count logging
5. Created `tests/unit/features/test_builder.py`

**Public API:**
```python
from gridiron_edge.features.player.builder import build_prop_features
df = build_prop_features(position_filter=["QB"])
```
**Pipeline flow:**
player_game_logs.parquet (loaded once)
    → build_player_rolling_features(df=...)   # ~46 rolling cols
    → build_matchup_features(df=...)          # 28 matchup cols
    → build_usage_features(df=...)            # 6 usage cols
    → build_game_context_features(df=...)     # 6 context cols
    → filter by position
    → dropna on feature columns
    → return training-ready DataFrame

---

#### Phase C — Prop Model Training  🔲 Planned

##### C1: Prop Trainer Framework + QB Passing Yards (First End-to-End Model) ✅ Complete

**What was done (2026-06-10):**
1. Rewired `_load_data()` to use `build_prop_features()` — single call
   replaces manual rolling+matchup+context chaining
2. Switched `train()` to `HOLDOUT_SEASONS` split (2023–2025 holdout)
3. Added position-aware NaN handling: features with >50% NaN for the
   filtered position are dropped (fixes 5,706 usable rows vs. 3)
4. Made `_build_features()` non-abstract (default no-op)
5. Deleted dead `_join_game_context()` and `_join_schedule_context()`
6. Fixed `_columns.py` matchup rank naming mismatch
7. Removed `dropna` from builder — deferred to trainer with position context

**First training results (qb_pass_yards ElasticNet):**
- Train: 5,706 rows (2009–2022), Holdout: 1,367 rows (2023–2025)
- MAE: 58.0 yards, RMSE: 72.6 yards, R²: 0.071
- 37/128 nonzero features after ElasticNet selection
- R² is low but expected for linear model on noisy player data —
  tree models (RF, XGBoost) will be added via champion/challenger

##### C2: Prop Output Enrichment (Post-Processing)

**Why:** Raw yard projections need market-facing outputs to calculate edges.

**New file:** `models/prop_prediction/post_process.py`

**Enrichment outputs per prediction:**
- `predicted_mean` — model point prediction
- `predicted_std` — `sqrt(model_residual_variance + player_rolling_std²)`.
  Model residual std captures systematic uncertainty; player's own std captures
  individual variance.
- `lo_90`, `hi_90` — `predicted_mean ± 1.645 * predicted_std`
- `p_over(line)` — `1 - Φ((line - predicted_mean) / predicted_std)` using normal
  CDF. Takes the line as input (from DK odds or user-specified).
- `lean` — "Over" if `p_over > 0.55`, "Under" if `p_over < 0.45`, "No Edge"
  otherwise (thresholds configurable)
- `confidence_tier` — based on `|p_over - 0.5|`: High (> 0.15), Moderate
  (0.08–0.15), Low (< 0.08)

**Parallels game model:** Just like `enrich_predictions()` takes raw win_prob and
adds spread/bands/tier, this takes raw predicted_yards and adds
distribution/lean/tier.

**New test:** `tests/unit/models/test_prop_post_process.py`

**Done when:** Every prop prediction row has all enrichment columns. Unit tests
verify P(over) math.

##### C3: Additional Prop Models

**Why:** With the pipeline validated on QB Pass Yards, extending is mechanical.

**Existing files to update (all scaffolded, need wiring to feature builder):**
1. `models/prop_prediction/qb_pass_yards.py` — QB Rushing Yards
   (`target_col='rushing_yards'`, `position_filter=['QB']`).
   **Note:** This needs its own file or a second PropModelSpec in the QB module.
   Evaluate whether to add `qb_rush_yards.py` or extend `qb_pass_yards.py` with
   a second spec.
2. `models/prop_prediction/rb_rush_yards.py` — RB Rushing Yards
   (`target_col='rushing_yards'`, `position_filter=['RB']`)
3. `models/prop_prediction/wr_rec_yards.py` — WR Receiving Yards
   (`target_col='receiving_yards'`, `position_filter=['WR']`)
4. `models/prop_prediction/te_rec_yards.py` — TE Receiving Yards
   (`target_col='receiving_yards'`, `position_filter=['TE']`)

**For each:**
- Wire to unified feature builder from B4
- Train all 4 model types (ElasticNet, Ridge, RF, XGB)
- Champion/challenger selects best
- Validate holdout MAE and prediction ranges

**Existing tests to update:**
`tests/unit/models/test_rb_rush_yards.py`,
`tests/unit/models/test_wr_rec_yards.py`,
`tests/unit/models/test_te_rec_yards.py`

**New file needed:** `models/prop_prediction/qb_rush_yards.py` (or decide on
multi-spec approach)

**Done when:** 5 prop model families trained, each with a champion model. Holdout
MAE table for all 5.

---

#### Phase D — Evaluation & Persistence  🔲 Planned

##### D1: Prop Evaluation Metrics

**Why:** Prop models need different eval metrics than game models.

**New file:** `evaluation/prop_metrics.py`

**Metrics:**
- **MAE / RMSE** — mean absolute error, root mean squared error vs actual
- **Hit rate** — given a line, % of "Over" leans that went over (and vice versa)
- **P(over) calibration** — when model says 70% over, does it go over ~70%?
  (calibration curve, reliability diagram)
- **Coverage** — do 90% prediction intervals contain ~90% of outcomes?
- **Bias** — `mean(predicted - actual)`, should be ~0
- **By-tier analysis** — MAE and hit rate broken down by confidence tier

**New test:** `tests/unit/evaluation/test_prop_metrics.py`

**Done when:** `prop_metrics.evaluate(predictions, actuals, lines)` returns a
structured report. Calibration plot can be generated.

##### D2: Prop Archive

**Why:** Persist prop predictions for tracking, CLV, and historical review.

**New file:** `evaluation/prop_archive.py`

**Schema:**
```
predicted_at, is_backfilled, season, week, game_id, game_date,
player_id, player_name, position, team,
stat_type, model_version,
predicted_mean, predicted_std, lo_90, hi_90,
line (nullable — only populated when odds available),
p_over (nullable), lean (nullable), confidence_tier
```

**Storage:** `data/output/props/prop_predictions_log.parquet`

**Pattern:** Same append-only, dedup-on `(game_id, player_id, stat_type,
model_version)` as game archive.

**New test:** `tests/unit/evaluation/test_prop_archive.py`

**Done when:** Prop predictions archive and load round-trip successfully.

---

#### Phase E — CLI & Integration  🔲 Planned

##### E1: Prop CLI

**Why:** User-facing deliverable — the thing you actually run on game day.

**New file:** `cli/props.py`

**Commands:**
- `gridiron props projections --week N --season YYYY` — table of all prop
  projections for the week
- `gridiron props backfill --model qb_pass_yards` — backfill historical prop
  predictions
- `gridiron props evaluate --model qb_pass_yards` — evaluation report for a
  prop model

**Output format (projections):**
```
Player       Pos  Stat         Proj   Lo90   Hi90   Line  P(Over)  Lean     Conf
P. Mahomes   QB   Pass Yards   278    198    358    274.5  0.52    No Edge  Low
L. Jackson   QB   Rush Yards    62     28     96     52.5  0.64    Over     Moderate
D. Henry     RB   Rush Yards    88     42    134     79.5  0.59    Over     Low
```

**Register in `cli/main.py`.**

**New tests:**
- `tests/integration/test_props_cli.py`
- `tests/e2e/test_prop_pipeline.py`

**Done when:** `gridiron props projections --week 12 --season 2024-2025` produces
a formatted table.

##### E2: DraftKings Prop Ingest Extension  🔲 Planned (Deferred — Upcoming Weeks Only)

**Why:** To calculate prop edges, we need prop lines from the market.

**Extend:** `ingest/odds/draftkings.py` (or new `ingest/odds/dk_props.py`)

**Scope:** Parse DK player prop markets (pass yards, rush yards, receiving yards
O/U). Map to `(player_id, stat_type, line, odds_over, odds_under)`. Store in
`data/odds/dk_prop_odds_log.parquet`.

**Note:** This is the last step because it requires understanding the DK prop API
response format. Historical prop odds remain deferred.

**Done when:** DK prop lines for the current week can be ingested and joined to
prop predictions.

---

#### Phase T — Testing  🔄 Parallel with all phases

Tests are built alongside each phase, not as a separate step at the end.

**Already existing (from Phase A):**

| File | Tier | Count |
|------|------|-------|
| `tests/unit/ingest/test_player_stats.py` | Unit | 9 tests, 3 classes |
| `tests/unit/transform/test_player_stats.py` | Unit | 7 tests, 3 classes |
| `tests/unit/features/test_player_rolling.py` | Unit | 12 tests, 3 classes |
| `tests/unit/features/test_player_matchup.py` | Unit | 11 tests, 5 classes |
| `tests/unit/features/test_usage.py` | Unit | 16 tests, 5 classes |
| `tests/unit/features/test_game_context.py` | Unit | 28 tests, 9 classes |
| `tests/unit/features/test_builder.py` | Unit | 20 tests, 2 classes |
| `tests/unit/models/test_prop_base.py` | Unit | 19 tests, 6 classes |
| `tests/unit/models/test_qb_pass_yards.py` | Unit | 5 tests, 1 class |
| `tests/unit/models/test_rb_rush_yards.py` | Unit | 5 tests, 1 class |
| `tests/unit/models/test_wr_rec_yards.py` | Unit | 5 tests, 1 class |
| `tests/unit/models/test_te_rec_yards.py` | Unit | 5 tests, 1 class |

**To create:**

| Phase | File | Tier |
|-------|------|------|
| B4 | `tests/integration/test_prop_feature_pipeline.py` | Integration |
| C2 | `tests/unit/models/test_prop_post_process.py` | Unit |
| D1 | `tests/unit/evaluation/test_prop_metrics.py` | Unit |
| D2 | `tests/unit/evaluation/test_prop_archive.py` | Unit |
| E1 | `tests/integration/test_props_cli.py` | Integration |
| E1 | `tests/e2e/test_prop_pipeline.py` | E2E |

**Also needed:** Add player-related DataFrame factories to
`tests/fixtures/dataframes.py` (e.g., `make_player_game_logs()`,
`make_player_features()`).

---

#### Dependency Graph

```
B1 (Audit & Stabilize) ✅
    │
    ▼
B2 (Usage Features) ✅
    │                                        T (Tests — parallel)
    ▼
B3 (Game Context Features) ✅
    │
    ▼
B4 (Unified Feature Builder) ✅
    │
    ├──────────────────────────────┐
    ▼                              ▼
C1 (QB Pass Yards + Framework) ✅  D1 (Prop Eval Metrics)
    │                              │
    ▼                              │
C2 (Post-Process Enrichment) ◄── YOU ARE HERE     │
    │                              │
    ▼                              │
C3 (4 More Prop Models)           │
    │                              │
    └──────────────┬───────────────┘
                   ▼
             D2 (Prop Archive)
                   │
                   ▼
             E1 (Prop CLI)
                   │
                   ▼
             E2 (DK Prop Ingest — last)
```

---

#### Files to Create

| File | Phase |
|------|-------|
| `models/prop_prediction/post_process.py` | C2 |
| `models/prop_prediction/qb_rush_yards.py` | C3 |
| `evaluation/prop_metrics.py` | D1 |
| `evaluation/prop_archive.py` | D2 |
| `cli/props.py` | E1 |
| `ingest/odds/dk_props.py` (or extend `draftkings.py`) | E2 |

#### Files to Modify

| File | Phase | Change |
|------|-------|--------|
| `models/prop_prediction/base.py` | C1 | Multi-model support, champion/challenger |
| `models/prop_prediction/qb_pass_yards.py` | C1 | Wire to feature builder, fix NaN |
| `models/prop_prediction/rb_rush_yards.py` | C3 | Wire to feature builder |
| `models/prop_prediction/wr_rec_yards.py` | C3 | Wire to feature builder |
| `models/prop_prediction/te_rec_yards.py` | C3 | Wire to feature builder |
| `cli/main.py` | E1 | Register props sub-app |
| `tests/fixtures/dataframes.py` | T | Add player DataFrame factories |

All paths relative to `src/gridiron_edge/` (source) or `tests/` (tests).

---

#### NaN Research Backlog  🔲 Deferred

**Problem:** Dropping NaN rows for early-season games (weeks 1–2 for L3, weeks
1–5 for L6) loses significant training data and makes early-season predictions
impossible.

**Research directions to explore:**
- **Bayesian shrinkage priors:** Use league-average or position-average as a
  prior, blend with observed data as games accumulate (empirical Bayes)
- **Seasonal carry-forward with decay:** Use prior season's final rolling average,
  decayed by a factor (e.g., 0.7) to reflect roster/scheme changes
- **Multiple imputation:** Generate multiple plausible feature sets, train on each,
  average predictions
- **Hierarchical models:** Player nested within team nested within position —
  partial pooling handles sparse data naturally
- **Missing-indicator pattern:** Add binary `is_imputed_*` flags so the model can
  learn to weight imputed features differently

**Data sources to investigate:** Does nflreadpy expose preseason stats? Can we use
combine/draft data as cold-start priors for rookies?

**When:** After W4 V1 is complete and evaluated.

---

### Parallel / Lower Priority

#### Weather Feature Integration (Deferred)

Deferred until OpenWeatherMap API key is reliably available and
weather features are validated against holdout data.

### Architectural Debt / Housekeeping

| Item | Notes |
|------|-------|
| `_DEFAULT_TOTAL_STD` hardcoded in `cli/edges.py` | Currently 13.17 (total model holdout RMSE). Wire into model metadata so it updates when a new total model is trained. |
| Historical edge validation / backtest | Deferred until season starts and real odds data is available. Run `gridiron edges clv` against a full season of archived predictions + odds to validate edge quality. |
| Schema migration helper | `archive.migrate_archive()` exists for pre-v2 archives. Can be removed once all archives are v2+. |
| Kelly adherence metric | `performance.kelly_adherence()` deferred — requires storing `recommended_stake` in the bet ledger schema. Add column to `_BET_COLUMNS` when implementing. |
| Balance display cosmetic | `balance_cmd` shows `$-100.00` for outflows instead of `-$100.00`. Fix sign formatting with `abs()` in `cli/betting.py`. |
| Recalibrate sigma/margin_std after retrain | The `_MODEL_SIGMAS` and `_MODEL_MARGIN_STDS` dicts in `post_process.py` still use values calibrated from the old versioned models. After a champion retrain, re-run sigma calibration and update the unversioned entries. |
| `ModelMetadata.holdout_brier` for regression | Repurposed for MAE in total model. Consider adding a generic `primary_metric` field. |

### Backlog

Workstream IDs match **ROADMAP.md** (authoritative numbering).

#### Completed

| ID | Workstream | Summary |
|----|-----------|---------|
| W1 | Quick Wins & Unblocking | DK unicode fix, game_id resolver, odds join validated |
| W2 | Richer Game Model Outputs | Spread, total, projected scores, bands, tiers, isotonic eval |
| W3 | Market Intelligence Foundation | odds_math.py, kelly.py — pure math, no data deps |
| W5 | Edge Engine | edge.py, recommendations.py, clv.py, CLI (report + clv) |
| W6 | Portfolio & Bet Tracking | ledger.py, bankroll.py, performance.py, CLI (8 commands) |

_Also completed (cross-cutting, not numbered in ROADMAP):_
- **Feature Engineering Expansion** — EPA_COLS 8→22→36, _EXPANDED_FEATURES 51→107→149
- **Test Framework Build-out** — Three-tier pyramid, auto-markers, shared fixtures
- **Champion/Challenger Refactor** — TimeSeriesSplit CV, gate-based promotion, 3
  unversioned champions (random_forest, xgboost, logistic)
- **Sigma & Confidence Tier Recalibration** — Probability-distance-based tiers
  replacing band-width thresholds

#### Planned

| ID | Workstream | Blocked by | Priority | Notes |
|----|-----------|------------|----------|-------|
| **W12** | Model Ensemble | Nothing | **High** | Combine elo + logistic + rf + xgb. Must beat xgboost Brier (0.218) via promotion gates. |
| **W4** | Player Data & First Prop Models | Nothing | **Active** | See active workstream above. |
| **W8** | API Serving Layer | Nothing | Medium | FastAPI endpoints for edges, games, portfolio → M5 |
| **W7** | Multi-Book Odds & Line Shopping | Odds source decision (§5.2) | Medium | Multi-book ingest, arb/middle detection → M4 |
| **W4.5** | Scenario Engine (What-If) | W4 | Medium | Injury impact modeling, usage redistribution |
| **W9** | Frontend | W8 | Lower | React/Next.js web UI → M5 |
| **W10** | Real-Time & Live Game | W7 + W8 | Lowest | Live win prob, live edges, hedge calculator → M6 |

### Changelog (PLAN.md edits only)

| Date | Change |
|------|--------|
| 2026-06-10 | **C1 complete.** Rewired PropTrainer to `build_prop_features()` + `HOLDOUT_SEASONS`. Position-aware NaN handling (>50% threshold). Deleted dead join methods. First QB pass yards model: MAE=58.0, RMSE=72.6, R²=0.071 (ElasticNet, 37/128 nonzero features, 5,706 train / 1,367 holdout rows). |
| 2026-06-10 | B4 complete. Phase B done. Created `features/player/builder.py` (unified entry point) and `features/player/_columns.py` (programmatic feature list). Refactored all 4 builders to accept optional df param for single-load pipeline. Created `tests/unit/features/test_builder.py`. Phase B (Feature Pipeline Completion) is now fully complete — C1 unblocked. |
| 2026-06-10 | **B3 complete.** Created `features/player/game_context.py` (6 features: is_home, game_spread, over_under, implied_team_total, is_dome, rest_days). Joined from games CSV, no shift needed (pre-game data). Created `tests/unit/features/test_game_context.py` (28 tests, 9 classes). |
| 2026-06-10 | **B2 complete.** Created `features/player/usage.py` (6 rolling usage features: target_share, carry_share, touch_share × L3/L6). Created `tests/unit/features/test_usage.py` (16 tests, 5 classes). Snap % deferred — nflreadpy does not expose snap count data. |
| 2026-06-10 | **B1 complete.** Remediated all audit findings: F1 (null team guard), F3 (dedup 46 schedule-join mismatches), W1–W6 resolved. F2 confirmed false positive. Per-position NaN analysis completed. Final audit: 45 pass, 0 fail, 4 warn (non-blocking). Player game logs: 138,349 rows, 4,067 players, 0 null game_ids, 0 duplicates. |
| 2026-06-05 | **Phase A audit completed.** Ran `audit_w4_phase_a.py` (43 pass, 2 fail, 4 warn). Code review of all 9 source files + 9 test files. Documented findings: 2 blocking (F1: game_id nulls, F2: fixture bug), 6 should-fix (W1–W6), 6 observations (O1–O6). NaN landscape table added. B1 updated to include audit remediation as first task. |
| 2026-06-05 | **W4 detailed implementation plan.** Added Phases A–E + T with step-level tasks, dependency graph, file inventories (create/modify), NaN research backlog. Reconciled against actual directory structure (136 dirs, 722 files). Confirmed Phase A complete, all 5 prop model files scaffolded with tests. Updated completed workstream summaries (added sigma recal, feature eng expansion to 149). |
| 2026-06-04 | **Complete rewrite.** Replaced stale W2-phase-detail PLAN with current state. Champion/challenger refactor complete. Removed resolved debt items (temporal CV, stale __pycache__). Updated backlog priorities (xgboost is new champion, W12 references its Brier). Removed W11 (live prediction pipeline already exists). |
| 2026-06-03 | W6 complete. W5 complete. Renumbered to match ROADMAP v2. |
| 2026-06-01 | W2, Feature Eng, Test Framework completed. |
| 2026-05-31 | Initial PLAN.md created with backlog. |
