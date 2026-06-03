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

No active workstream.  See **Backlog** for next candidates.

---

## Parallel / Lower Priority

### Phase 20f — Weather Feature Integration (Deferred)

Deferred until OpenWeatherMap API key is reliably available and
weather features are validated against holdout data.

---

## Architectural Debt / Housekeeping

| Item | Notes |
|---|---|
| Temporal leakage in tree model CV | StratifiedKFold(shuffle=True) doesn't respect time ordering. Isotonic recalibration infrastructure built (W2) but rf_v3 already well-calibrated (ECE 0.036), so calibrator was not saved. Revisit for future model versions. |
| Stale `__pycache__` after restructures | Clear with `find . -type d -name __pycache__ -exec rm -rf {} +` |
| `_DEFAULT_TOTAL_STD` hardcoded in `cli/edges.py` | Currently 13.17 (total model holdout RMSE). Wire into model metadata so it updates when a new total model is trained. |
| Historical edge validation / backtest | Deferred until season starts and real odds data is available. Run `gridiron edges clv` against a full season of archived predictions + odds to validate edge quality. |
| Schema migration helper | `archive.migrate_archive()` exists for pre-v2 archives. Can be removed once all archives are v2+. |
| Kelly adherence metric | `performance.kelly_adherence()` deferred — requires storing `recommended_stake` in the bet ledger schema. Add column to `_BET_COLUMNS` when implementing. |
| Balance display cosmetic | `balance_cmd` shows `$-100.00` for outflows instead of `-$100.00`. Fix sign formatting with `abs()` in `cli/betting.py`. |

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
- **Feature Engineering Expansion** — EPA_COLS 8→22, _EXPANDED_FEATURES 51→107, rf_v3/xgb_v3 trained
- **Test Framework Build-out** — Three-tier pyramid, auto-markers, shared fixtures, pre-commit/pre-push hooks

### Planned

| ID | Workstream | Blocked by | Priority | Notes |
|---|---|---|---|---|
| ~~W11~~ | ~~Live Prediction Pipeline~~ | — | — | Not needed — `output predictions` + `edges report` already covers this. Removed. |
| **W12** | Model Ensemble | Nothing | **High** | Combine elo + logistic + rf + xgb. Must beat rf_v3 Brier by ≥0.002. |
| **W4** | Player Data & First Prop Models | Nothing | Medium | Player-level features + QB/RB prop models → M3 |
| **W8** | API Serving Layer | Nothing | Medium | FastAPI endpoints for edges, games, portfolio → M5 |
| **W7** | Multi-Book Odds & Line Shopping | Odds source decision (§5.2) | Medium | Multi-book ingest, arb/middle detection → M4 |
| **W4.5** | Scenario Engine (What-If) | W4 | Medium | Injury impact modeling, usage redistribution |
| **W9** | Frontend | W8 | Lower | React/Next.js web UI → M5 |
| **W10** | Real-Time & Live Game | W7 + W8 | Lowest | Live win prob, live edges, hedge calculator → M6 |

---

## Changelog (PLAN.md edits only)

| Date | Change |
|---|---|
| 2026-06-03 | **Renumbered to match ROADMAP.md v2.** Added W11 (Live Prediction Pipeline) and W12 (Model Ensemble). Reconciled all IDs. Previous PLAN-only IDs (W3=FeatEng, W4=Tests, W7=LivePredict, W8=Ensemble, W9=MultiBook, W10=Props, W11=API, W12=Dashboard) retired in favor of ROADMAP-authoritative numbering. |
| 2026-06-03 | W6 (Portfolio & Bet Tracking) complete — moved to CHANGELOG. Added deferred items to Architectural Debt. |
| 2026-06-02 | W6 (Portfolio & Bet Tracking) activated. Phase A-E defined. |
| 2026-06-02 | W5 (Edge Engine) complete — moved to CHANGELOG. Updated backlog dependencies. |
| 2026-06-02 | W5 Edge Engine active — defined scope. |
| 2026-06-01 | W2, Feature Eng, Test Framework completed — moved to CHANGELOG. |
| 2026-05-31 | Initial PLAN.md created with backlog. |
