# Gridiron Edge - ROADMAP
## Long-Term Strategic Direction

---

## 0. How This Document Fits

| Document | Purpose | Updated When |
|---|---|---|
| **ROADMAP.md** (this file) | High-level long-term direction. Where we're headed and why. Workstreams, dependencies, architecture decisions. | When strategic direction changes or a workstream is completed/added. |
| **PLAN.md** | Short-term next steps. The current working checklist. | Every working session. |
| **CHANGELOG.md** | What's been completed. Items move here from PLAN.md when finished. | When work is completed. |
| **HANDOFF.md** | How things work right now. Architecture, conventions, commands, gotchas. | When the system changes meaningfully. |
| **DECISIONS.md** | Append-only log of architectural decisions made during workstreams. | When an architectural choice is locked. |
| **README.md** | Public-facing project overview. | When HANDOFF.md changes significantly. |

**Workflow:** ROADMAP tells you *what to work on next*. PLAN tells you *how to do it*. CHANGELOG proves *what's done*. HANDOFF explains *how it all works*.

---

## 1. Current State Summary

Gridiron Edge is a CLI-driven NFL analytics, modeling, and betting platform with a complete prediction-to-edge-to-bet-tracking pipeline. The architectural foundation is settled: composite identity, archive-driven CLI, vectorized data flows, canonical Elo simulator, three-tier test pyramid, and four composite workflows that wrap the full game-day operation.

### What's Working

| Area | Status | Key Assets |
|---|---|---|
| Data ingestion (nflverse) | ✅ Solid | Games, schedule, PBP, rosters |
| Data ingestion (weather) | ✅ Solid | OpenWeatherMap, idempotent |
| Data ingestion (odds) | ✅ Partial | DraftKings only; 403 bot detection is an active concern (tracked in PLAN.md) |
| Transform / clean layer | ✅ Solid | nflverse → canonical mappers |
| Dataset registry + I/O | ✅ Solid | Complete registry (20 keys), typed access, manifest validation |
| Feature engineering | ✅ Excellent (22 EPA + 107 total) | Elo, EPA, rest, travel, weather, venue, SoS, record, divisional, efficiency, situational |
| Feature pipeline + validation | ✅ Solid | Dependency ordering, schema versioning, incremental-build staleness detection |
| Vectorized data flows | ✅ Solid | Per-row apply patterns eliminated; cumsum-based streaks, masked merges |
| Elo ratings | ✅ Solid | Canonical simulator, parameterized divisor, fit/predict/table all share one source of truth |
| Game prediction models | ✅ Solid | Logistic / RF / XGB / Elo composite-key registry; first-class metric fields |
| Post-processing enrichment | ✅ Complete | Spread, total, projected scores, uncertainty bands, confidence tiers |
| Total points model | ✅ Complete | MAE 10.24, competitive with Vegas closing O/U |
| Calibration persistence | ✅ Complete | `_MODEL_SIGMAS` / `_MODEL_MARGIN_STDS` persisted to disk via calibration registry on `full-retrain` |
| Evaluation | ✅ Excellent | Brier, log loss, AUC, ECE, calibration, decomposition, confidence tiers, drift, heuristic warnings |
| Prediction archive | ✅ Solid | Append-only, composite identity, walk-forward backfill semantics |
| Monte Carlo simulation | ✅ Advanced | Season + playoffs, numba-optimized, divisor parameterized |
| Market math | ✅ Complete | odds_math, kelly, edge — pure functions, no data deps |
| Edge engine | ✅ Complete | edge, recommendations, clv — moneyline/spread/total edges, Kelly sizing, CLV analysis |
| Edge CLI | ✅ Complete | `gridiron edges report`, `gridiron edges clv` |
| Bet tracking | ✅ Complete | ledger.py, bankroll.py, performance.py — composite identity, decoupled bankroll |
| Betting CLI | ✅ Complete | 8 commands with calibration_health surfacing |
| Composite CLI workflows | ✅ Complete | `weekly-predict`, `post-week`, `full-retrain`, `verify` — shared stage abstraction, soft-fail semantics, dependency validation |
| Code quality | ✅ Excellent | Ruff, pyrefly, three-tier test pyramid, pre-commit + pre-push hooks |
| Testing infrastructure | ✅ Complete | 500+ tests, auto-markers, shared fixtures, MiniRepoBuilder |
| Player data ingestion | ✅ Solid | nflreadpy player game logs (1999–2024), 138K rows, 42 cols per row |
| Player feature engineering | ✅ Solid | Rolling stats (L3/L6), matchup (28 cols), usage (6 cols), game context (6 cols) |
| Player prop models | ✅ Solid (5 models) | ElasticNet + RF + XGB across 5 stat families with archive-driven champion selection |
| Prop post-processing | ✅ Complete | predicted_std, 90% intervals, P(over), lean, confidence tiers |
| Prop evaluation | ✅ Complete | Archive-driven; no retraining inside evaluation surfaces |
| Prop archive | ✅ Complete | Append-only Parquet, composite identity dedup, walk-forward semantics |
| Prop CLI | ✅ Complete | All commands archive- or artifact-driven, no retraining at use time |
| Audit remediation | ✅ Complete | Units 1-11 (~100 findings), 4 cross-cutting patterns, architectural cleanup |
| Deep code review (Tier 4 sweep) | ✅ Complete | 30 backlog items closed; 2 real bugs surfaced and fixed |

### What's Missing

| Area | Status | Impact |
|---|---|---|
| **API serving layer** | ❌ Not started | Everything is CLI + file output. Blocks visual QA of outputs and friend access. |
| **Frontend** | ❌ Not started | Prototype exists (Claude Design) but not wired. Required for M5. |
| Live prediction pipeline | ✅ Exists | `gridiron weekly-predict` is the working game-day workflow |
| Model ensemble | ❌ Not started | Using individual models only, no weighted combination |
| Multi-book odds ingestion | ❌ Not started | Can't compare books, can't shop lines |
| Player prop walk-forward backfill | ✅ Available | Per-(stat, algorithm) walk-forward backfill works |
| Injury/news feed | ❌ Not started | No injury data; blocks W4.5 scenario engine |
| Live game / real-time | ❌ Not started | No live state, odds, or win prob |

### Known Blockers

None at the workstream level. Two operational items tracked in PLAN.md:

- **DraftKings API 403 (bot detection):** `weekly-predict` soft-fails gracefully when this happens; historical odds ledger and game_id resolver work independently.
- **Walk-forward backfill on single-season expanded-feature windows:** Real bug, fix needed before W12 (Model Ensemble) can do a clean Brier comparison. Fix listed in PLAN.md "Real Bugs Surfaced."

---

## 2. Vision / End State

Gridiron Edge becomes a **complete NFL decision-support platform** that:

1. ✅ **Forecasts games** with projected spreads, totals, win probabilities, and uncertainty bands
2. ✅ **Projects player props** with full outcome distributions
3. ✅ **Compares model outputs against the betting market** to identify edges
4. **Shops lines** across multiple sportsbooks to find the best price
5. ✅ **Recommends stake sizing** using Kelly criterion and bankroll awareness
6. ✅ **Tracks betting performance** with CLV, ROI, P/L splits, and Kelly adherence
7. **Surfaces real-time information** (injuries, line moves, weather) that affects edge calculations
8. **Serves all of the above** through an API and eventually a frontend

The platform is built for personal use first (you + friends), with architecture that could support commercial access later.

**Sport scope:** NFL only for now. Architecture should be sport-agnostic where it costs less than 20% extra effort.

---

## 3. Prioritization Principles

Since this project is worked on in spare time with no fixed timelines, prioritization is driven by **value density** — what gives the most gain for the effort invested.

1. **Unblock before build.** If a trivial fix unblocks a major workstream, do it first.
2. **Complete the prediction → edge loop first.** ✅ Achieved (M1).
3. **Enrich existing models before building new ones.** Adding spread/total/uncertainty to existing game models is higher-value than building player prop models from scratch.
4. **Feature engineering is continuous.** It runs in parallel with everything else.
5. **Ship something usable.** Each workstream should produce an artifact you'd actually use on game day.
6. **Backend first, frontend later — but build a viewer when output volume outpaces eyeball-able CLI.** The frontend is also a verification surface. When the number of distinct outputs makes spot-checking from the CLI lossy, a UI becomes a QA tool, not just a presentation layer.
7. **Files are fine until they're not.** Stay with Parquet/CSV until multi-user access, concurrency, or query complexity forces a database.

---

## 4. Workstreams

Each workstream is a **major capability area** that can be broken into smaller tasks in PLAN.md. They are ordered by recommended priority (highest-gain-first), not by timeline.

### Completed Workstreams

#### W1: Quick Wins & Unblocking — ✅ COMPLETE
Fixed DK unicode minus bug, built game_id resolver, validated end-to-end odds joins. See CHANGELOG.md for details.

#### W2: Richer Game Model Outputs — ✅ COMPLETE
Spread derivation (probit + per-model sigma calibration), total points model, projected scores, 90% uncertainty bands, confidence tiers. See CHANGELOG.md for details.

#### W3: Market Intelligence Foundation — ✅ COMPLETE
Pure-math market package: odds_math.py, kelly.py. Power devig via bisection, no scipy dependency. See CHANGELOG.md for details.

#### W4: Player Data & First Prop Models — ✅ COMPLETE
Player game logs (nflreadpy, 1999–2024), 4 player feature modules, 5 prop models (QB pass/rush yards, RB rush yards, WR rec yards, TE rec yards), PropTrainer base class, post-processing enrichment, prop archive, prop CLI. M3 achieved. See CHANGELOG.md for details.

#### W5: Edge Engine — ✅ COMPLETE
Edge calculation (moneyline, spread, total), recommendations builder, CLV analysis, CLI commands. See CHANGELOG.md for details.

#### W6: Portfolio & Bet Tracking — ✅ COMPLETE
Bet ledger, bankroll management (decoupled), performance analytics, 8 CLI commands. Full round-trip validated. See CHANGELOG.md for details.

#### W3.5: Audit Remediation — ✅ COMPLETE
Closed ~100 findings from `audit_2026_06_18.md` across 11 audit units, plus 4 cross-cutting patterns (vectorization, polish, enums, registry completion). Major outcomes:

- **Identity unification:** `(model_name, model_type)` composite identity flows through every persistence layer.
- **Elo simulator:** Single canonical history simulator drives both the state table and the tuner.
- **Task-discriminated metadata:** Metrics live in a single dict on `BaseModelMetadata`.
- **Vectorization:** All audit-flagged per-row apply patterns eliminated.
- **Trainable protocol:** Reduced to `spec + is_trained`. Registry enforces consistency.
- **CLI ergonomics:** Stage staleness warnings, calibration health flags, archive-driven prop CLI, enum-based string constants.

See `AUDIT_REMEDIATION.md` for unit-by-unit closure detail and `DECISIONS.md` D1–D12 for architectural decisions.

#### W4.1: Composite CLI Workflows — ✅ COMPLETE
Four composite commands wrap related single-purpose commands into complete workflows:
- **`weekly-predict`** — game-day prep (refresh → predict → edges)
- **`post-week`** — archive + drift detection
- **`full-retrain`** — season-start refresh; persists calibration to disk
- **`verify`** — pre-commit quality checks

Shared infrastructure (`cli/_composites.py`) provides stage abstraction, dependency validation, soft-fail semantics, and consolidated summary rendering. M1.6 achieved.

#### W5.5: Deep Code Review + Test Suite Review (Tier 4 sweep) — ✅ COMPLETE
Multi-session opportunistic cleanup that closed 30 backlog items across CLI ergonomics, composite commands, dead code, documentation drift, exception narrowing, type cleanup, HTML escaping, season-label consistency, name mapping consolidation, calibration persistence, pipeline correctness, and incremental-build staleness detection. Two real bugs surfaced and fixed (XGBoost recalibration Pipeline feature-name warning, modeling-file stale-data preservation). Three items reclassified as future workstream candidates. TIER_4_BACKLOG.md retired; remaining items tracked in PLAN.md.

### Active Workstream

#### W8: API Serving Layer — 🟡 ACTIVE (next focus)

**Goal:** Expose analytics outputs through a REST API so a frontend (or other consumers) can access them, and so the full slate of outputs becomes visually verifiable in one place.

**Why it matters now:** This is both the bridge to a frontend *and* the next quality-assurance step. The CLI surfaces outputs one-at-a-time; a dashboard forces them side-by-side, which surfaces missing fields, schema drift, and silent bugs that the test suite is blind to. Doing this before W12 (Model Ensemble) or W4.5 (Scenario Engine) means any subsequent workstream automatically inherits a UI surface and a verification harness.

**Key deliverables:**
- Create `api/` package at `src/gridiron_edge/api/`
- Choose framework: **FastAPI** (lightweight, async, good docs, type-safe)
- Read-only first. No POST endpoints, no auth, no DB in scope for W8.
- Core endpoints:
  - `GET /games?week=12` — list games with model forecasts and edges
  - `GET /games/{game_id}` — game detail with fair values, team comparison
  - `GET /edges?week=12` — ranked edge table
  - `GET /teams` — power rankings
  - `GET /props?week=12` — top prop edges
  - `GET /portfolio/summary` — bankroll + performance (read-only)
- Data source: read from Parquet/CSV files. No database in W8.
- CORS configuration for frontend access

**Dependencies:** None. Fully unblocked.
**Unlocks:** W9 (Frontend), visual QA of full output set, M5 (with W9).
**Architecture notes:** Start with FastAPI reading Parquet files. The "files vs. database" decision (§5.1) is deferred until W9 reveals concrete query patterns that are awkward in pandas.

### Future Workstreams (ordered by current priority)

#### W9: Frontend — 🟢 PLANNED (immediately after W8)

**Goal:** Build a web UI that consumes the API and presents the analytics. Acts as the visual verification surface for everything the platform produces.

**Key deliverables:**
- Scaffold app (React/Vite or Next.js)
- Wire up to API endpoints
- Implement screens progressively as backend capabilities arrive:
  - Dashboard (games + edges)
  - Game Detail
  - Power Rankings
  - Player Props Explorer
  - Line Shopping (when W7 ships)
  - Bet Slip
  - Bankroll / Portfolio
  - News / Alerts
  - Settings
- The Claude Design prototype provides the visual spec.

**Dependencies:** W8.
**Unlocks:** M5 (friends can use it), output-driven prioritization of W12 vs W4.5 vs W7.

#### W12: Model Ensemble — 🟢 PLANNED

**Goal:** Combine elo, logistic, random forest, and XGBoost predictions into a weighted ensemble for better overall accuracy.

**Why it matters:** Individual models have different strengths. A well-tuned ensemble should beat any individual model on Brier score and AUC.

**Key deliverables:**
- Resolve the walk-forward backfill bug (single-season expanded-feature windows) first — it's a prerequisite for honest holdout comparison.
- Ensemble weighting strategy: Brier-weighted averaging, stacking (logistic meta-learner), simple rank averaging.
- Register ensemble predictor alongside individual models via `ModelRegistry`.
- Evaluation: must improve Brier by ≥0.002 over current champion to ship.
- Wire ensemble into prediction pipeline + edge report.

**Dependencies:** W2 ✅. Soft dependency on PLAN.md "Real Bugs" walk-forward backfill fix.
**Unlocks:** Better predictions for all downstream consumers; auto-surfaces in W8/W9 UI.

#### W4.5: Scenario / "What If" Engine — 🟢 PLANNED

**Goal:** Let a user ask "what if Mahomes is out?" or "what if KC is +120 instead of -110?" and see the propagated effects on predictions, edges, and recommended bets.

**Why deferred behind W8/W9 and W12:** Scenarios are most useful when their outputs can be visualized comparatively, and require an injury data source decision (§5.3) that is currently unresolved. Doing W8/W9 first means scenarios have a natural UI surface; doing W12 first means scenarios start from a stronger baseline.

**Phases:** player impact quantification → team adjustment → usage redistribution → conditional re-forecasting → CLI/API interface.

**Dependencies:** Injury data source decision (§5.3). Soft dependency on W8.
**Unlocks:** Real-time decision support during pregame and live windows.

#### W7: Multi-Book Odds & Line Shopping — 🟢 PLANNED

**Goal:** Ingest odds from multiple sportsbooks and build line-comparison tooling.

**Why deferred:** Lower value density than W8/W9/W12 until the existing single-book pipeline is visually verified.

**Key deliverables:**
- Odds source decision (see §5.2)
- Additional book ingest modules
- `market/line_shopping.py`: best_price, price_comparison_table, detect_arbitrage, detect_middles
- Line movement tracking and steam move detection
- CLI: `gridiron lines --week 12 --market spread`

**Dependencies:** W3 ✅, odds source decision (§5.2).
**Unlocks:** Better bet execution, arbitrage opportunities, M4.

#### W10: Real-Time & Live Game — 🟢 PLANNED (lowest priority)

**Goal:** Live game state ingestion, live win probability, live odds comparison, real-time alerts.

**Key deliverables:** Live game state ingest, live WP model, live odds ingest, live edge detection, hedge calculator, WebSocket API.

**Dependencies:** W7, W8.
**This is the most complex and least urgent workstream.** Not started until W7 and W8/W9 are solid.

### Cross-Cutting: Testing
**Testing runs in parallel with all workstreams.** Every new feature includes corresponding unit tests. Integration and e2e tests are added as cross-module workflows are built.

### Cross-Cutting: Feature Engineering
**Feature engineering is continuous.** Remaining FEATURES.md backlog (CPOE, pace, score differential, penalties, special teams, coaching) can be picked up alongside any workstream.

---

## 5. Architecture Decisions & Open Items

### 5.1 File Storage vs. Database

**Current:** Parquet + CSV, file-based, CLI-driven.

**Recommendation:** Stay with files through W8. Re-evaluate during W9 when concrete API query patterns emerge. Migrate to SQLite or PostgreSQL when:
- The API layer needs to serve concurrent requests
- The portfolio/bet ledger needs transactional integrity
- Query patterns require joins that are awkward in pandas
- Multi-user access is added

**Practical trigger:** If during W9 wiring you find yourself writing complex pandas merge chains in the API layer to answer a single request, that's the cue.

### 5.2 Odds Data Source

**Status:** Deferred. Current DK-only ingest is sufficient through W8/W9/W12.

**When needed:** Before W7 (Multi-Book Line Shopping) can begin.

| Source | Coverage | Props? | Cost | Notes |
|---|---|---|---|---|
| The Odds API | ~15 books | Limited | Free tier: 500 req/mo; paid: $20–$80/mo | Easy to start, good docs |
| Odds Jam | 20+ books | Yes | ~$40–$100/mo | Strong prop coverage |
| Pinnacle API | Pinnacle only | No (limited) | Free (with account) | Sharp book, useful as reference |
| Action Network | Major books | Yes | Varies | Requires investigation |
| DonBest | Comprehensive | Yes | Enterprise pricing | Likely overkill for V1 |
| Direct book APIs | Per-book | Varies | Free (with accounts) | High maintenance |

**Recommendation:** Start with **The Odds API** for multi-book game markets when W7 begins.

### 5.3 Injury Data Source

**Status:** Not yet addressed. **Blocks W4.5.**

**When needed:** Before W4.5 (Scenario Engine) can begin.

**Options:**
- ESPN API (free, has injury reports)
- nflverse injury data (if available in their data releases)
- Manual tracking (acceptable for V1 with a small number of games)
- Rotowire / Rotoworld feeds (may require scraping or API access)

### 5.4 Project Structure

Current packages, with W8/W9 additions noted:

- `src/gridiron_edge/`
  - `ingest/` ✅
  - `transform/` ✅
  - `datasets/` ✅
  - `features/` ✅ (team + player)
  - `models/` ✅ (game_prediction + prop_prediction + elo)
  - `ratings/` ✅
  - `sim/` ✅
  - `evaluation/` ✅
  - `viz/` ✅
  - `market/` ✅
  - `betting/` ✅
  - `cli/` ✅ (including `_composites.py`, `weekly_predict.py`, `post_week.py`, `full_retrain.py`, `verify.py`)
  - `core/` ✅
  - **`api/` — PLANNED (W8)**
  - **frontend lives outside `src/` — PLANNED (W9)**
  - `scenario/` — PLANNED (W4.5)

---

## 6. Dependency Graph

```

COMPLETED                                  ACTIVE / REMAINING
─────────                                  ──────────────────

W1 (Quick Wins) ✅
│
├─────────────────────┬───────────────────────────┐
▼                     ▼                           ▼
W2 (Model Outputs) ✅  W3 (Market Math) ✅          W4 (Player Data) ✅
│                     │                            │
└─────────┬───────────┘                            │
▼                                        │
W5 (Edge Engine) ✅ ◄──────── W4.5 (Scenario) ──┘
│                          ▲
┌────────┼─────────┐                │ (blocked: §5.3)
▼        ▼         ▼                │
W6 ✅   W3.5 ✅   W4.1 ✅            │
(Bet     (Audit)  (Composite          │
Track)            CLI)                │
│
W5.5 ✅                              │
(Deep Code Review)                   │
│
┌─── W8 (API) 🟡 ◄────┤
│      │              │
│      ▼              │
│   W9 (Frontend) ────┤
│                     │
│                     │
├─── W12 (Ensemble) ──┤
│                     │
├─── W7 (Multi-Book) ─┤
│      │              │
│      ▼              │
└─── W10 (Real-Time / Live)

```

**Current position:** All foundation workstreams complete. W8 (API Serving Layer) is the active workstream; PLAN.md carries its tier-by-tier design. The remaining workstreams are unblocked but deliberately ordered:

1. **W8 (API)** 🟡 active — prototype-driven, read-only, FastAPI + Pydantic v2.
2. **W9 (Frontend)** — sequential after W8.
3. After W9, pick between **W12 (Ensemble)**, **W4.5 (Scenario)** (pending §5.3), and **W7 (Multi-Book)** based on what the UI surfaces.
4. **W10 (Real-Time)** — deferred until everything else stabilizes.

---

## 7. What Success Looks Like (Milestones)

| Milestone | Description | Workstreams | Status |
|---|---|---|---|
| **M1: First actionable edge report** | Run `gridiron edges report --week 12` and get a ranked list of game edges with EV, Kelly stake, and best available book. | W1 + W2 + W3 + W5 | ✅ **ACHIEVED** |
| **M1.5: Weekly game-day predictions** | Run `gridiron output predictions` then `gridiron edges report` to see fresh edges. | Existing pipeline | ✅ **ACHIEVED** |
| **M1.6: One-command weekly workflow** | Run `gridiron weekly-predict` to do the full Thursday/Sunday prep in one command. | W4.1 | ✅ **ACHIEVED** |
| **M2: Know if the model makes money** | After a month of tracking bets, run `gridiron bet summary` and see your CLV, ROI, and record by confidence tier. | W6 | ✅ **ACHIEVED** |
| **M3: First prop edge** | Full prop evaluation report with accuracy, bias, coverage metrics. | W4 + W5 | ✅ **ACHIEVED** |
| **M4.5: Visual output verification** | Wire the Gridiron Edge frontend prototype to the API and walk the full surface. Every populated field is verified; every `null` field surfaces a backend gap that's tracked in §9. | W8 + W9 | 🟡 **NEXT** |
| **M4: Shop across 3+ books** | Run `gridiron lines --week 12` and see a cross-book comparison with best prices highlighted. | W7 | Planned |
| **M5: Friends can use it** | Stand up a web UI that your friends can access. Dashboard, game detail, edges. | W8 + W9 | Planned (delivered with M4.5 + auth) |
| **M6: Live game day experience** | Real-time win prob, live edges, hedge suggestions during a game. | W10 | Planned |

**M4.5 is the next north star.** It's both a usability milestone and a quality-assurance step: the act of seeing every output side-by-side will surface bugs the CLI hides.

---
## 9. Known Issues & Backlog

Items that are not active workstreams but need tracking. Sources: surfaced during W5.5 Tier 4 cleanup, and progressively added as W8 placeholders surface backend gaps.

### 9.1 Testing Infrastructure

| Item | Notes |
|---|---|
| Props e2e fit-load-predict tests | Originally deferred from W3. `tests/e2e/test_props_fit_load_predict.py` exists; confirm coverage matches game-side parity. |
| Composite commands don't have e2e tests | Unit tests cover stage definitions with mocks; e2e tests against real data would surface integration issues earlier. |
| Weather ingest happy-path integration test | Pre-existing bugs went undetected because there was no end-to-end test of the ingest pipeline. |
| Registry cold-start scenarios | Test additions for `build_prop_evaluation_df` integration when registry is empty at call time. |
| Performance baselines for tests | May need `pytest-benchmark` if runtime grows or regressions become a concern. |
| API layer tests (W8) | Three-tier coverage: unit (response models, handlers with mocked datasets, `_meta` correctness), integration (`MiniRepoBuilder`-backed), e2e (deferred to W9). |
| CLI test drift: `tests/e2e/test_cli_workflows.py::TestEvaluateSelectModelSmoke::test_empty_archive_exits_with_message` | Test asserts `False` on the empty-archive branch of `gridiron evaluate select-model`. Behavior or output of the command has changed without the test being updated. Likely accumulated drift during W4.1 / W5.5 CLI work. Low-impact fix; revisit when touching `cli/evaluate.py` next. |
| CLI test drift: `tests/integration/test_betting_cli.py::TestLogCommand::test_log_with_model_context` | Test asserts against `gridiron bet log` behavior but receives `Usage: root log [OPTIONS]` — the command help, not the executed result. Indicates a Typer signature or argument-structure change without test update. Low-impact fix; revisit when touching `cli/betting.py` next. |

### 9.2 Real Bugs

| Item | File | Notes |
|---|---|---|
| Walk-forward backfill produces no valid pipeline for single-season windows with expanded feature sets | `models/game_prediction/base.py::_run_hp_search` (root cause) and `evaluation/backfill.py::_walk_forward_one_season` (calling site) | Single-season walk-forward fails because filtered training data falls below `MIN_CV_TRAIN_ROWS` for expanded feature sets. Also: `_run_hp_search` does not forward `train_through_season` to `_prepare_window`. Fix options: (A) lower threshold for walk-forward, (B) fill expanded-feature NaN with neutral values, (C) force walk-forward to use combined feature set. **Soft-blocks W12 (clean Brier comparison).** |
| `GameModelMetadata` constructor rejects keyword arguments used by `tests/integration/test_model_train_predict.py::TestArtifactRoundtrip` | `models/base.py::GameModelMetadata` (likely) and `tests/integration/test_model_train_predict.py` | Four artifact round-trip tests fail with `TypeError: GameModelMetadata.__init__() got an unexpected keyword argument ...`. Surfaced when running the full pre-push test suite during W8 Tier 1 close-out. Likely a refactor-vs-test drift introduced during W3.7 (Game Model Refactor, 2026-06-18) when `BaseModelMetadata` / `GameModelMetadata` / `PropModelMetadata` were split. Fix is either updating the test fixtures to match the current metadata constructor or restoring the dropped keyword for backward compatibility — decide based on whether the dropped keyword is intentionally gone. |

### 9.3 Investigations

| Item | Notes |
|---|---|
| `CalibratedClassifierCV` uses `StratifiedKFold(shuffle=False)` | Not strictly time-ordered. Investigate `TimeSeriesSplit` switch and measure impact on calibration quality. May require backfill run for comparison. |

### 9.4 Operational

| Item | Notes |
|---|---|
| DraftKings odds endpoint returns 403 | Bot detection is more aggressive. Investigate headers, cookies, paid API alternatives. `weekly-predict` soft-fails gracefully. |
| Weather: missing stadium entries for 2026-2027 international games | 12 stadiums need lat/lon/altitude in `NFL_stadium_reference.csv`. Listed in HANDOFF.md. Data entry task. |
| Model calibration values pre-date current modeling file | `_MODEL_SIGMAS` and `_MODEL_MARGIN_STDS` hardcoded fallbacks calibrated against older modeling file. `full-retrain` composite now persists current values to disk via the calibration registry; next full-retrain run supersedes the fallbacks. |
| `verify --strict` not exercised in CI | Once a real CI surface exists, `gridiron verify --strict` should be the gate. |

### 9.5 Backend gaps surfaced by the prototype

Drives W8 Tier 2 work and post-W8 prioritization. These are gaps between what the prototype expects and what the platform currently produces. Items either get folded into W8 Tier 2, deferred here, or escalated to a future workstream.

| Item | Surfaced in (screen) | W8 disposition |
|---|---|---|
| Per-stat league-wide percentile ranking | Compare, Team Detail | W8 Tier 2 |
| Off/def rating decomposition | Team Rankings | W8 Tier 2 |
| Weekly Elo snapshot persistence | Team Detail rating trend, Projections delta | W8 Tier 2 |
| Opponent-allowed-by-position aggregation | Player vs Defense, Player Prop | W8 Tier 2 |
| Cohort splits per team (season, L4, home, away, vs winning, vs top-10) | Compare, Game Detail | W8 Tier 2 (limited splits ship; full set deferred) |
| Cohort splits per prop (indoor/outdoor, favored/underdog, vs top-10 def) | Player Prop | W8 Tier 2 (limited splits ship; full set deferred) |
| Feature attribution / per-factor prediction decomposition | Swing Factors, Explain waterfall, Prop reasoning | Deferred — future workstream (likely paired with W12 or W4.5) |
| Comparables retrieval (nearest-neighbor over historical games) | Explain comparables | Deferred — future workstream |
| Historical line movement | Line Drilldown chart | Deferred — folds into W7 |
| Game-day metadata (network, venue text, storylines) | Game cards, Game Detail header | Deferred — likely static config file rather than a workstream |
| Injury / lineup / news ingest | Game Detail injuries, News Wire | Blocked on ROADMAP §5.3 injury data source decision; unblocks W4.5 |
| Live game state ingest | Live screen | Blocks on W10 |
| WAR (Wins Above Replacement) per player | Team Detail top-players panel | Deferred — significant ML work, not currently prioritized |
| Multi-book odds | Line shopping, Prop shop sub-resource, Arbitrage / Middle tools | Blocks on W7 |

---

## 10. Changelog for This Document

| Date | Change |
|---|---|
| 2026-06-23 | **Document restructure.** PLAN.md now scoped to the active workstream only; future workstream candidates, real-bugs backlog, investigations, and operational items migrated to new ROADMAP.md §9 Known Issues & Backlog. Added backend-gaps-surfaced-by-prototype subsection to §9 with W8 Tier 2 disposition per item. M4.5 reworded to reflect prototype-driven verification framing. Current-position callout updated to mark W8 active. |
| 2026-06-23 | **Resync with PLAN.md.** Marked W4.1 (Composite CLI) and W5.5 (Deep Code Review / Tier 4 sweep) complete. M1.6 marked achieved. Set W8 (API) as active workstream with W9 (Frontend) sequential after. Added M4.5 milestone for visual output verification. Reordered future workstreams by current value-density: W8 → W9 → {W12, W4.5, W7} → W10. Added §5.3 as explicit blocker for W4.5. Updated §6 dependency graph and "Current position" callout. Added Principle 6.5 (frontend-as-verification-surface). Cleaned §1 to reflect composite CLI, calibration persistence, and pipeline staleness detection as shipped capabilities. |
| 2026-06-22 | **Workstream 5 (Tier 4 cleanup) complete.** 30 ambient hygiene items closed. Two real bugs fixed. Remaining items reclassified as workstream candidates. TIER_4_BACKLOG.md retired. |
| 2026-06-21 | **Audit remediation (W3.5) complete.** ~100 findings closed across 11 audit units, 4 cross-cutting patterns. W4.1 (Composite CLI Workflows) added. M1.6 milestone added. |
| 2026-06-10 | **W4 mostly complete. M3 achieved.** Player data pipeline, 5 prop models, post-processing enrichment, evaluation metrics, archive, CLI. |
| 2026-06-03 | Champion/challenger model refactor complete. Temporal CV fix (TimeSeriesSplit). 3 unversioned champions replace 10 versioned variants. M1.5 achieved. |
| 2026-06-03 | **v2 refresh.** Updated §1, marked W1–W6 complete in §4, added W11 (later removed) and W12, updated §5.4, redrew §6, marked M1/M2 achieved, added M1.5. |
| 2026-05-30 | Initial version — created from prototype review + gap analysis. |

***
