# Gridiron Edge — Development Plan

> **Purpose:** single source of truth for *what to build next* and *why*.
> Updated at the start and close of every workstream.

| Document | Role |
|---|---|
| **PLAN.md** (this file) | What is planned, what is active, what is deferred |
| **CHANGELOG.md** | What was built and when (completed workstream details) |
| **HANDOFF.md** | How the system works today (architecture, workflows, operations) |

### Status key

| Tag | Meaning |
|---|---|
| Done | Done — details in CHANGELOG.md |
| In progress | In progress |
| Planned | Planned / blocked |
| Deferred | Deferred |

---

## Currently Active Workstreams

#### W4: Player Data & First Prop Models — In Progress

_Ingest player game logs, build rolling features, create first prop prediction models._
**ROADMAP ref:** W4  **Unlocks:** W4.5 (Scenario Engine), prop betting edges  **FEATURES.md ref:** Domains 4, 8

##### Data Source
- `nfl_data_py.import_weekly_data()` — pre-aggregated player-game stats (53 cols)
- Available 1999–present, ~5K rows/season, team codes match existing nflverse format
- Key columns: passing/rushing/receiving yards/TDs/EPA, target_share, air_yards_share, wopr, dakota, pacr, racr

##### Phase A: Player Data Foundation

| Step | Task | New files | Status |
|---|---|---|---|
| A1 | **Ingest** — fetch + cache weekly player stats per season to `data/raw/player_stats/` | `ingest/nflverse/player_stats.py` | Done |
| A2 | **Transform** — clean column names, standardize team codes, construct `game_id` via schedule join, filter skill positions, output `data/cleaned/player_game_logs.parquet` (138K rows, 44 cols, 4,068 unique players) | `transform/clean/player_stats.py` | Done |
| A3 | **Rolling features** — per-player rolling stats (L3, L6): mean + std dev for 23 stat columns. 92 rolling feature columns. `shift(1)` prevents lookahead. | `features/player/rolling.py` | Done |
| A4 | **Matchup features** — opponent defensive allowances by position group. 14 stats × 2 (L6 rolling avg + rank 1=toughest to 32=most generous) = 28 features. Joined via `opponent_team`. | `features/player/matchup.py` | Done |
| A5 | **Wire into pipeline** — add `fetch-player-stats` + `build-player-features` stages to CLI. Register in feature pipeline. | CLI + pipeline updates | Deferred (modules callable directly; CLI plumbing is low-risk) |

##### Phase B: First Prop Models

| Step | Task | New files | Status |
|---|---|---|---|
| B1 | **Prop model framework** — `PropTrainer` ABC, `PropModelSpec`, `PropModelMetadata`, `PropPrediction`, `evaluate_props()`, `_MIN_ATTEMPTS` volume filters, `_load_data()` with rolling + matchup + game context joins, TimeSeriesSplit training pipeline | `models/prop_prediction/base.py` | Done |
| B2 | **Fix prop model NaN crisis** — Position-aware `_feature_columns()` overrides, fix weather column names, exclude CPOE. See detailed fix plan below. | `base.py`, `rolling.py`, all 4 prop model files | Not started |
| B3 | **Train & evaluate all 4 prop models** — QB pass yards, RB rush yards, WR rec yards, TE rec yards. Compare ElasticNet baseline metrics. | All 4 prop model files | Not started |
| B4 | **Prop evaluation archive** — MAE/RMSE metrics, prop archive, comparison to book lines | `evaluation/prop_archive.py` | Not started |
| B5 | **Prop model tests** — Unit tests for position-specific `_feature_columns()` overrides, updated `UNIVERSAL_FEATURE_COLS` assertions | Test files | Not started |

##### B2 — Prop Model NaN Crisis: Diagnosis & Fix Plan

Diagnostic showed **0.0% row survival** (2/15,130) in QB pass yards model.
Three distinct bugs identified:

**Bug 1: Wrong weather column names in `base.py`**
- `_build_universal_features()` references `temp` and `wind`, but the
  pipeline produces `TEMP_F` and `WIND_SPEED_MPH`. Columns silently missing → NaN.
- **Fix:** Replace `"temp"` → `"TEMP_F"`, `"wind"` → `"WIND_SPEED_MPH"` in
  the game context block of `_build_universal_features()`.

**Bug 2: Position-irrelevant rolling stats cause mass NaN**
- Universal feature list includes all 23 rolling stat columns for every position.
  QBs almost never have `receiving_epa` (99.9% NaN), and WR/TE metrics
  (`target_share`, `air_yards_share`) leak into QB spec (28–36% NaN).
  `dropna` kills every row before ElasticNet sees the data.
- **Fix:** Export `PASSING_STATS`, `RUSHING_STATS`, `RECEIVING_STATS` from
  `rolling.py` (rename from private `_PASSING_STATS` etc.). Add
  `_build_position_features(stat_cols)` helper to `base.py`. Each model
  overrides `_feature_columns()`:
  - **QB:** `PASSING_STATS` (excl. `passing_cpoe`) + `RUSHING_STATS`
  - **WR:** `RECEIVING_STATS` + `RUSHING_STATS`
  - **RB:** `RUSHING_STATS` + `RECEIVING_STATS`
  - **TE:** `RECEIVING_STATS` + `RUSHING_STATS`

**Bug 3: CPOE has 26.8% NaN across all seasons**
- Already excluded from game model for same reason.
- **Fix:** Filter `passing_cpoe` out in QB `_feature_columns()` override.

**Acceptable NaN (not bugs):**
- Rolling cold-start (~8.9%): first games per player-season. Unavoidable.
- Matchup cold-start (~6%): first weeks of each season. Also unavoidable.
- Expected usable rows after fix: ~85–90%.

**Files to edit:**
- `src/gridiron_edge/features/player/rolling.py` — Rename `_PASSING_STATS` → `PASSING_STATS` (etc.)
- `src/gridiron_edge/models/prop_prediction/base.py` — Add `_build_position_features()`, fix column names
- `src/gridiron_edge/models/prop_prediction/qb_pass_yards.py` — Override `_feature_columns()`
- `src/gridiron_edge/models/prop_prediction/rb_rush_yards.py` — Override `_feature_columns()`
- `src/gridiron_edge/models/prop_prediction/wr_rec_yards.py` — Override `_feature_columns()`
- `src/gridiron_edge/models/prop_prediction/te_rec_yards.py` — Override `_feature_columns()`

**Verification script (run after B2):**
```bash
uv run python -c "
from gridiron_edge.models.prop_prediction.qb_pass_yards import QBPassYardsTrainer
trainer = QBPassYardsTrainer()
df = trainer._load_data()
features_df = trainer._build_features(df)
feature_cols = trainer._feature_columns()
target = trainer.spec.target_col
required = [*feature_cols, target]
nan_pcts = (features_df[required].isna().mean() * 100).sort_values(ascending=False)
high_nan = nan_pcts[nan_pcts > 0]
print(f'Columns with any NaN ({len(high_nan)} of {len(required)}):')
for col, pct in high_nan.items():
print(f'  {col:<50s}  {pct:>5.1f}%')
clean = features_df.dropna(subset=required)
print(f'Usable rows: {len(clean):,} / {len(features_df):,} ({len(clean)/len(features_df)*100:.1f}%)')
"
```

##### Locked Decisions

| Decision | Choice | Rationale |
|---|---|---|
| **Data source** | `nflreadpy.load_player_stats()` | Migrated from archived `nfl_data_py` (404s on 2025 data). Returns Polars DataFrames requiring `.to_pandas()`. Column names identical. |
| **Storage** | Per-season Parquet at `data/raw/player_stats/` | Same pattern as PBP ingest — idempotent, incremental |
| **Rolling windows** | L3 and L6 games | Short window captures form, longer window captures baseline. Same philosophy as team EPA rolling. |
| **First prop targets** | QB passing yards, RB rushing yards | Highest data volume, most stable signal, largest betting markets |
| **CPOE handling** | Excluded from game model EPA_COLS and prop model QB features (26.8% NaN from pre-CPOE era). Backlog item: investigate era-aware imputation or separate pre/post-2006 model variants. |

---

## Parallel / Lower Priority

---

## Architectural Debt / Housekeeping

| Item | Notes |
|---|---|
| `_DEFAULT_TOTAL_STD` hardcoded in `cli/edges.py` | Currently 13.17 (total model holdout RMSE). Wire into model metadata so it updates when a new total model is trained. |
| Historical edge validation / backtest | Deferred until season starts and real odds data is available. Run `gridiron edges clv` against a full season of archived predictions + odds to validate edge quality. |
| Schema migration helper | `archive.migrate_archive()` exists for pre-v2 archives. Can be removed once all archives are v2+. |
| Kelly adherence metric | `performance.kelly_adherence()` deferred — requires storing `recommended_stake` in the bet ledger schema. Add column to `_BET_COLUMNS` when implementing. |
| Balance display cosmetic | `balance_cmd` shows `$-100.00` for outflows instead of `-$100.00`. Fix sign formatting with `abs()` in `cli/betting.py`. |
| ModelMetadata.holdout_brier for regression | Repurposed for MAE in total model. Consider adding a generic `primary_metric` field. |
| CPOE imputation strategy | `passing_cpoe` has 26.8% NaN (structural, pre-2006 era). Currently excluded from all models. Investigate era-aware imputation, post-2006-only training, or composite metrics like `dakota`. |
| Retrain game models with expanded weather features | 6 new weather features added to `_GAME_FEATURES` (15 cols) and `_EXPANDED_FEATURES` (155 cols). Champions have not been retrained yet. Run `gridiron models train` for each champion and compare Brier scores. |

---

## Backlog

Workstream IDs match **ROADMAP.md** (authoritative numbering).

### Completed

| ID | Workstream | Summary |
|---|---|---|
| W1 | Quick Wins & Unblocking | DK unicode fix, game_id resolver, odds join validated |
| W2 | Richer Game Model Outputs | Spread, total, projected scores, bands, tiers, isotonic eval |
| W3 | Market Intelligence Foundation | odds_math.py, kelly.py — pure math, no data deps |
| W5 | Edge Engine | edge.py, recommendations.py, clv.py, CLI (report + clv) |
| W6 | Portfolio & Bet Tracking | ledger.py, bankroll.py, performance.py, CLI (8 commands) |

*Also completed (cross-cutting, not numbered in ROADMAP):*
- **Feature Engineering Expansion** — EPA_COLS 8→22→36, _EXPANDED_FEATURES 51→107→149→155. Includes sigma/margin_std recalibration, confidence tier rework (band-width → probability-distance), and weather feature expansion (6 new features: FEELS_LIKE_F, HUMIDITY_PCT, VISIBILITY_M, SNOW_FLAG, LOW_VIS_FLAG, WIND_CHILL_DELTA).
- **nflreadpy Migration** — Switched from archived `nfl_data_py` to `nflreadpy`. 2025 data now available. Fixed `load_schedules` type mismatch.
- **Test Framework Build-out** — Three-tier pyramid, auto-markers, shared fixtures
- **Champion/Challenger Refactor** — TimeSeriesSplit CV, gate-based promotion, 3 unversioned champions (random_forest, xgboost, logistic)

### Planned

| ID | Workstream | Blocked by | Priority | Notes |
|---|---|---|---|---|
| **W12** | Model Ensemble | Nothing | **High** | Combine elo + logistic + rf + xgb. Must beat xgboost Brier (0.218) via promotion gates. |
| **W8** | API Serving Layer | Nothing | Medium | FastAPI endpoints for edges, games, portfolio → M5 |
| **W7** | Multi-Book Odds & Line Shopping | Odds source decision (§5.2) | Medium | Multi-book ingest, arb/middle detection → M4 |
| **W4.5** | Scenario Engine (What-If) | W4 | Medium | Injury impact modeling, usage redistribution |
| **W9** | Frontend | W8 | Lower | React/Next.js web UI → M5 |
| **W10** | Real-Time & Live Game | W7 + W8 | Lowest | Live win prob, live edges, hedge calculator → M6 |

---

## Changelog (PLAN.md edits only)

| Date | Change |
|---|---|
| 2026-06-04 (late) | W4 Phase A complete (A1–A4). B1 (prop framework) complete. Weather feature expansion complete (6 new features, `_process_weather` refactored, Phase 20f removed). nflreadpy migration complete. Diagnosed prop model NaN crisis (0.0% row survival). Detailed B2 fix plan written. ElasticNet baseline results recorded before NaN fix: QB MAE=57.2/R²=0.105, RB MAE=25.0/R²=0.166, WR MAE=25.1/R²=0.204, TE MAE=18.4/R²=0.187 (from curated feature lists, before universal features). |
| 2026-06-04 | W4 activated. Added detailed Phase A (data foundation) and Phase B (first prop models) plans. Completed sigma/margin_std recalibration and feature engineering expansion (EPA_COLS 22→36, features 107→149). Removed resolved sigma debt item. Updated feature engineering cross-cutting summary. |
| 2026-06-04 | **Complete rewrite.** Replaced stale W2-phase-detail PLAN with current state. Champion/challenger refactor complete. Removed resolved debt items (temporal CV, stale __pycache__). Updated backlog priorities (xgboost is new champion, W12 references its Brier). Removed W11 (live prediction pipeline already exists). |
| 2026-06-03 | W6 complete. W5 complete. Renumbered to match ROADMAP v2. |
| 2026-06-01 | W2, Feature Eng, Test Framework completed. |
| 2026-05-31 | Initial PLAN.md created with backlog. |
