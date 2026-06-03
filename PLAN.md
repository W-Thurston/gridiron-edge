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
| ✅ | Done — details in CHANGELOG.md |
| 🔄 | In progress |
| 📋 | Planned / blocked |
| ⏸️ | Deferred |

---

## Currently Active Workstreams

No active workstream.  See **Backlog** for next candidates.

---

## Parallel / Lower Priority

### Phase 20f — Weather Feature Integration ⏸️

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

---

## Backlog

| ID | Workstream | Description | Blocked by | Priority |
|---|---|---|---|---|
| W1 | ✅ Odds Ingest & Joins | DraftKings API, game_id resolver, odds storage | — | — |
| W2 | ✅ Richer Game Model Outputs | Post-processing: spreads, bands, tiers, projected scores, isotonic recalibration | — | — |
| W3 | ✅ Feature Engineering Expansion | EPA_COLS 8→22, _EXPANDED_FEATURES 51→107, rf_v3/xgb_v3 trained | — | — |
| W4 | ✅ Test Framework Build-out | Three-tier tests, auto-markers, shared fixtures, pre-commit hooks, coverage config | — | — |
| W5 | ✅ Edge Engine | Edge calculation, recommendations, CLV, CLI commands | — | — |
| W6 | 📋 Simulation / Monte Carlo | Season-level simulations, playoff probability, draft position | None (W5 ✅) | High |
| W7 | 📋 Live Prediction Pipeline | Real-time pre-game predictions with current-week features | W3 | High |
| W8 | 📋 Model Ensemble | Combine elo + logistic + rf + xgb into a weighted ensemble | None (W2 ✅, W5 ✅) | High |
| W9 | 📋 Multi-Sportsbook Support | FanDuel, BetMGM ingestion alongside DraftKings | W1 | Medium |
| W10 | 📋 Automated Betting Dashboard | Web UI with live edges, CLV tracker, bankroll management | W7 + W8 (W5 ✅) | Medium |

---

## Changelog (PLAN.md edits only)

| Date | Change |
|---|---|
| 2026-06-02 | W5 (Edge Engine) complete — moved to CHANGELOG. Updated backlog dependencies. Added deferred items to Architectural Debt. |
| 2026-06-02 | W5 Edge Engine active — defined scope. |
| 2026-06-01 | W2, W3, W4 completed — moved to CHANGELOG. |
| 2026-05-31 | Initial PLAN.md created with W1–W10 backlog. |
