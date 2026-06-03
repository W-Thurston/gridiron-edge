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

| ID | Workstream | Description | Blocked by | Priority |
|---|---|---|---|---|
| W1 | Done: Odds Ingest & Joins | DraftKings API, game_id resolver, odds storage | — | — |
| W2 | Done: Richer Game Model Outputs | Post-processing: spreads, bands, tiers, projected scores, isotonic recalibration | — | — |
| W3 | Done: Feature Engineering Expansion | EPA_COLS 8->22, _EXPANDED_FEATURES 51->107, rf_v3/xgb_v3 trained | — | — |
| W4 | Done: Test Framework Build-out | Three-tier tests, auto-markers, shared fixtures, pre-commit hooks, coverage config | — | — |
| W5 | Done: Edge Engine | Edge calculation, recommendations, CLV, CLI commands | — | — |
| W6 | Done: Portfolio & Bet Tracking | Bet ledger, bankroll, performance analytics, CLI | — | — |
| W7 | Planned: Live Prediction Pipeline | Real-time pre-game predictions with current-week features | W3 | High |
| W8 | Planned: Model Ensemble | Combine elo + logistic + rf + xgb into a weighted ensemble | None (W2, W5 done) | High |
| W9 | Planned: Multi-Sportsbook Support | FanDuel, BetMGM ingestion alongside DraftKings | W1, odds source decision | Medium |
| W10 | Planned: Player Data & Props | Player-level features + first prop projection models | None | Medium |
| W11 | Planned: API Serving Layer | FastAPI endpoints for edges, games, portfolio | W5 + W6 done | Medium |
| W12 | Planned: Automated Betting Dashboard | Web UI with live edges, CLV tracker, bankroll management | W7 + W8 + W11 | Low |

---

## Changelog (PLAN.md edits only)

| Date | Change |
|---|---|
| 2026-06-03 | W6 (Portfolio & Bet Tracking) complete — moved to CHANGELOG. Added deferred items to Architectural Debt. |
| 2026-06-02 | W6 (Portfolio & Bet Tracking) activated. Phase A-E defined. |
| 2026-06-02 | W5 (Edge Engine) complete — moved to CHANGELOG. Updated backlog dependencies. |
| 2026-06-02 | W5 Edge Engine active — defined scope. |
| 2026-06-01 | W2, W3, W4 completed — moved to CHANGELOG. |
| 2026-05-31 | Initial PLAN.md created with W1-W10 backlog. |
