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
| A1 | **Ingest** — fetch + cache weekly player stats per season to `data/raw/player_stats/` | `ingest/nflverse/player_stats.py` | Not started |
| A2 | **Transform** — clean column names, standardize team codes, filter to regular+postseason, output `data/cleaned/player_game_logs.parquet` | `transform/clean/player_stats.py` | Not started |
| A3 | **Rolling features** — per-player rolling stats (L3, L6): mean, std dev. Covers FEATURES.md Priority Matrix items 9–10. | `features/player/rolling.py` | Not started |
| A4 | **Matchup features** — opponent defensive rank vs position group (Priority 12). | `features/player/matchup.py` | Not started |
| A5 | **Wire into pipeline** — add `fetch-player-stats` + `build-player-features` stages to CLI. Register in feature pipeline. | CLI + pipeline updates | Not started |

##### Phase B: First Prop Models

| Step | Task | New files | Status |
|---|---|---|---|
| B1 | **Prop model framework** — base class for continuous-target prop models, shared feature pipeline infra | `models/prop_prediction/base.py` | Not started |
| B2 | **QB passing yards** — first prop model: rolling stats + matchup + implied team total | `models/prop_prediction/qb_pass.py` | Not started |
| B3 | **RB rushing yards** — second prop model, same framework | `models/prop_prediction/rb_rush.py` | Not started |
| B4 | **Prop evaluation** — MAE/RMSE metrics, prop archive, comparison to book lines | `evaluation/prop_archive.py` | Not started |

##### Locked Decisions

| Decision | Choice | Rationale |
|---|---|---|
| **Data source** | `nfl_data_py.import_weekly_data()` | Pre-aggregated at player-game level, 53 columns, maintained by nflverse, available 1999–present |
| **Storage** | Per-season Parquet at `data/raw/player_stats/` | Same pattern as PBP ingest — idempotent, incremental |
| **Rolling windows** | L3 and L6 games | Short window captures form, longer window captures baseline. Same philosophy as team EPA rolling. |
| **First prop targets** | QB passing yards, RB rushing yards | Highest data volume, most stable signal, largest betting markets |
| **CPOE handling** | Excluded from game model EPA_COLS (26.8% NaN), available in epa_by_game for prop models | Structural NaN from pre-CPOE era. Will be used at player level via `dakota` composite. |

---

## Parallel / Lower Priority

### Phase 20f — Weather Feature Integration (Deferred)

Deferred until OpenWeatherMap API key is reliably available and
weather features are validated against holdout data.

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
- **Feature Engineering Expansion** — EPA_COLS 8→22→36, _EXPANDED_FEATURES 51→107→149. Includes sigma/margin_std recalibration and confidence tier rework (band-width → probability-distance).
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
| 2026-06-04 | W4 activated. Added detailed Phase A (data foundation) and Phase B (first prop models) plans. Completed sigma/margin_std recalibration and feature engineering expansion (EPA_COLS 22→36, features 107→149). Removed resolved sigma debt item. Updated feature engineering cross-cutting summary. |
| 2026-06-04 | **Complete rewrite.** Replaced stale W2-phase-detail PLAN with current state. Champion/challenger refactor complete. Removed resolved debt items (temporal CV, stale __pycache__). Updated backlog priorities (xgboost is new champion, W12 references its Brier). Removed W11 (live prediction pipeline already exists). |
| 2026-06-03 | W6 complete. W5 complete. Renumbered to match ROADMAP v2. |
| 2026-06-01 | W2, Feature Eng, Test Framework completed. |
| 2026-05-31 | Initial PLAN.md created with backlog. |
