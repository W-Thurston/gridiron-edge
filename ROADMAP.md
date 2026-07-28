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
| Player data ingestion | ✅ Solid | nflreadpy player game logs (1999–2025), 138K rows, 42 cols per row |
| Player feature engineering | ✅ Solid | Rolling stats (L3/L6), matchup (28 cols), usage (6 cols), game context (6 cols) |
| Player prop models | ✅ Solid (5 models) | ElasticNet + RF + XGB across 5 stat families with archive-driven champion selection |
| Prop post-processing | ✅ Complete | predicted_std, 90% intervals, P(over), lean, confidence tiers |
| Prop evaluation | ✅ Complete | Archive-driven; no retraining inside evaluation surfaces |
| Prop archive | ✅ Complete | Append-only Parquet, composite identity dedup, walk-forward semantics |
| Prop CLI | ✅ Complete | All commands archive- or artifact-driven, no retraining at use time |
| Audit remediation | ✅ Complete | Units 1-11 (~100 findings), 4 cross-cutting patterns, architectural cleanup |
| Deep code review (Tier 4 sweep) | ✅ Complete | 30 backlog items closed; 2 real bugs surfaced and fixed |
| API serving layer (W8) | ✅ Complete | 16 endpoints, Pydantic-validated, field_status placeholder convention, champion resolution wired |
| Frontend app (W9) | ✅ Complete | Vite + React + TS; all screens render; typed openapi-fetch + React Query |
| Frontend fidelity arc (W9.5–W9.10) | ✅ Complete | Dashboard, GameDetail, Teams split-view, PlayerProp, Compare (both modes) rebuilt; primitives: Pill, WhyLink, TeamMark-w/colors, Spark, TeamHero, DistributionChart, RatingChart, BarChart; dev-panel highlight mode |
| Cohort splits (11 metrics) | ✅ Complete | Team cohort splits season/l4/home/away with off+def reciprocal pairs; per-prop situational splits; opponent-allowed by position |

### What's Missing

| Area | Status | Impact |
|---|---|---|
| Model ensemble | ❌ Not started | Individual models only; no weighted combination |
| Multi-book odds ingestion | ❌ Not started | Blocks line shopping, book selectors, real bet-slip odds, per-week book lines |
| Injury/news feed | ❌ Not started | Blocks W4.5 scenario engine + injury UI fields |
| Live game / real-time | ❌ Not started | No live state, odds, or win prob |
| Off/def rating decomposition | ❌ Not started | Blocks off/def ranking tabs, Compare Off/Def mini-stats |
| Frontend prototype-fidelity backlog | 🟡 Partial | §9.7 (backend gaps) + §9.8 (frontend polish); core screens done, PlayoffProjections + BetSlip rebuilds + blocked-on-data items remain |

### Known Blockers

None at the workstream level. Two operational items tracked in PLAN.md:

- **DraftKings API 403 (bot detection):** `weekly-predict` soft-fails gracefully when this happens; historical odds ledger and game_id resolver work independently.
- **Walk-forward backfill on single-season expanded-feature windows:** Real bug, fix needed before W12 (Model Ensemble) can do a clean Brier comparison. Fix tracked in ROADMAP §9.2.

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
8. ✅ **Serves all of the above** through an API and eventually a frontend

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

Full detail in CHANGELOG.md + DECISIONS.md. One-line summaries here.

| WS | Summary | Date | Milestone |
|---|---|---|---|
| W1 | Quick wins: DK unicode-minus fix, game_id resolver, odds joins | — | |
| W2 | Richer game outputs: spread, total model, uncertainty bands, tiers | — | |
| W3 | Market math: odds_math, kelly (pure-math, no scipy) | — | |
| W3.5 | Audit remediation: ~100 findings, composite identity, canonical Elo sim (D1–D12) | 2026-06-21 | |
| W4 | Player data + 5 prop models + archive + CLI | 2026-06-10 | M3 |
| W4.1 | Composite CLI: weekly-predict, post-week, full-retrain, verify | — | M1.6 |
| W5 | Edge engine: ML/spread/total edges, recommendations, CLV | — | M1 |
| W5.5 | Deep code review: 30 backlog items, 2 bugs fixed | 2026-06-22 | |
| W6 | Portfolio + bet tracking: ledger, bankroll, performance, 8 CLI cmds | — | M2 |
| W13 | Runtime champion resolution: manifest + resolver, `--model-type auto` (D21-adjacent) | 2026-07-01 | |
| W8 | API serving layer: 16 endpoints, field_status convention (D13–D21) | 2026-07-04 | |
| W9 | Frontend: Vite+React+TS, all screens render, openapi-fetch + React Query | 2026-07-03 | M4.5 |

**Frontend fidelity arc (W9.5–W9.10)** — rebuilt core screens to
prototype fidelity on a shared primitive set + the W9.8 highlight
discipline. Primitives: Pill, WhyLink, TeamMark-colors, Spark, TeamHero,
RatingChart, DistributionChart, BarChart, PendingChip, ComingSoonCard.

| WS | Summary | Date |
|---|---|---|
| W9.5 | Dashboard rebuild + 5 primitives + team-metadata CSV consolidation | 2026-07-04 |
| W9.6 | GameDetail full fidelity (hero, lines, win-prob, team-comparison) | 2026-07-07 |
| W9.7 | Teams split-view (`/teams?team=X`) + RatingChart | 2026-07-07 |
| W9.8 | Dev panel + pending-highlight mode (audit deferred → §9) | 2026-07-11 |
| W9.9 | PlayerProp rebuild + DistributionChart | 2026-07-07 |
| W9.10 | Compare both modes + backend B1–B4 + game_id fix + offseason-readiness | 2026-07-11 |

See CHANGELOG.md for details.


### Active Workstream

##### W9.11: Screen Completion — 🟡 ACTIVE

**Goal:** Finish the core-screen set by rebuilding PlayoffProjections and
BetSlip against verified real data and the shared primitive set established
during W9.5–W9.10, then complete the deferred pending-highlight audit across
the final frontend surface.

**Execution order:**

- **Tier 1 — PlayoffProjections rebuild — ✅ COMPLETE (2026-07-28).**
  Rebuilt the screen as a live counterpart to the original static playoff
  table: full-cell postseason probability heat matrix, accessible sorting,
  dependent conference/division filters, current Elo and record context,
  explicit Elo movement, as-of-week/run metadata, Week 1-aware unavailable
  treatment, and team-profile navigation. Composes the existing `/projections`
  and `/teams` static contracts; no request-time computation added.

- **Tier 2 — BetSlip rebuild — NEXT.** Rebuild the slip around verified existing
  probability, bankroll, stake, payout, and EV inputs. Target a Kelly suggestion
  card, bankroll-percentage indicator, EV summary, enhanced leg presentation,
  and quick-stake controls. Lock exact scope only after inspecting the current
  BetSlip, AppState, odds utilities, and edge-row contract.

- **Tier 0 — Pending-highlight audit sweep — DEFERRED CAPSTONE.** After both
  remaining screen rebuilds, walk every built screen with Highlight mode
  enabled. Classify each silently missing field as pending, blocked,
  unavailable, or defective; route it through the established field-status
  primitives and record larger follow-up gaps.

**Ready:** Tier 1 has no external dependency. Tier 2 requires a contract
verification pass before deep design; combined EV must not be assumed available
until the current edge/slip data path is confirmed.

**Unlocks:** Every core screen rebuilt and the frontend fidelity arc closed with
the pending-highlight discipline applied across the complete final surface.

**Deferred within W9.11:** True week-over-week playoff-probability movement
(requires persisted simulation snapshots); richer simulation provenance such as
model/config identity, random seed, and elapsed duration; SGP correlation
warnings, book selection, and line movement; anything requiring live or
multi-book data.

### Future Workstreams (ordered by current priority)

#### W12: Model Ensemble — 🟢 PLANNED

**Goal:** Combine elo, logistic, random forest, and XGBoost predictions into a weighted ensemble for better overall accuracy.

**Key deliverables:**
- Resolve the walk-forward backfill bug (single-season expanded-feature windows) first — it's a prerequisite for honest holdout comparison.
- Ensemble weighting strategy: Brier-weighted averaging, stacking (logistic meta-learner), simple rank averaging.
- Register ensemble predictor alongside individual models via `ModelRegistry`.
- Evaluation: must improve Brier by ≥0.002 over current champion to ship.
- Wire ensemble into prediction pipeline + edge report.

**Dependencies:** W2 ✅. Soft dependency on §9.2 walk-forward backfill bug fix.
**Unlocks:** Better predictions for all downstream consumers; auto-surfaces in W8 UI.

#### W4.5: Scenario / "What If" Engine — 🟢 PLANNED

**Goal:** Let a user ask "what if Mahomes is out?" or "what if KC is +120 instead of -110?" and see the propagated effects on predictions, edges, and recommended bets.

**Phases:** player impact quantification → team adjustment → usage redistribution → conditional re-forecasting → CLI/API interface.

**Dependencies:** Injury data source decision (§5.3). Soft dependency on W8.
**Unlocks:** Real-time decision support during pregame and live windows.

#### W7: Multi-Book Odds & Line Shopping — 🟢 PLANNED

**Goal:** Ingest odds from multiple sportsbooks and build line-comparison tooling.

**Key deliverables:**
- Odds source decision (see §5.2).
- Additional book ingest modules.
- `market/line_shopping.py`: best_price, price_comparison_table, detect_arbitrage, detect_middles.
- Line movement tracking and steam move detection.
- CLI: `gridiron lines --week 12 --market spread`.

**Dependencies:** W3 ✅, odds source decision (§5.2).
**Unlocks:** Better bet execution, arbitrage opportunities, M4.

#### W10: Real-Time & Live Game — 🟢 PLANNED (lowest priority)

**Goal:** Live game state ingestion, live win probability, live odds comparison, real-time alerts.

**Key deliverables:** Live game state ingest, live WP model, live odds ingest, live edge detection, hedge calculator, WebSocket API.

**Dependencies:** W7, W8.
**This is the most complex and least urgent workstream.** Not started until W7 and W8 are solid.

### Cross-Cutting: Testing
**Testing runs in parallel with all workstreams.** Every new feature includes corresponding unit tests. Integration and e2e tests are added as cross-module workflows are built.

### Cross-Cutting: Feature Engineering
**Feature engineering is continuous.** Remaining backlog (CPOE, pace, score differential, penalties, special teams, coaching) can be picked up alongside any workstream.

---

## 5. Architecture Decisions & Open Items

### 5.1 File Storage vs. Database

**Current:** Parquet + CSV, file-based, CLI-driven.

**Status:** Files held through W8/W9 as planned; no complex pandas merge chains emerged in the API layer (serializers read pre-computed artifacts per D21). **Decision: stay file-based.** Re-evaluate only if multi-user access, concurrency, or transactional bet-ledger integrity becomes a real requirement.

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
  - `api/` ✅ (W8)
  - frontend lives outside `src/` at `frontend/` ✅ (W9)
  - `scenario/` — PLANNED (W4.5)

---

## 6. Dependency Graph

```

COMPLETED                                    ACTIVE / REMAINING
─────────                                    ──────────────────

W1 (Quick Wins) ✅
│
├──────────┬───────────┐
▼          ▼           ▼
W2 ✅       W3 ✅        W4 ✅
│          │           │
└─────┬────┘           │
▼                       │
W5 (Edge Engine) ✅ ◄── W4.5 (Scenario) ──┘
│                       ▲
┌───────┼──────────┐    │ (blocked: §5.3)
▼       ▼          ▼    │
W6 ✅   W3.5 ✅    W4.1 ✅
(Bet    (Audit)   (Composite
Track)            CLI)
│
W5.5 ✅
(Deep Review)
│
W13 ✅
(Champion Resolution)
│
W8 (API) ✅
│
├── W9 (Frontend) ✅
│   └── W9.5–W9.9 ✅ (fidelity arc) · W9.10 ✅ (Compare)
│
├── W12 (Ensemble) 🟢 planned
│
├── W7 (Multi-Book) 🟢 planned
│
└── W10 (Real-Time) 🟢 deferred

```


**Current position:** W9.11 Tier 1 PlayoffProjections is complete and verified
against the real 32-team artifact.

- Next: Tier 2 — inspect and lock the current BetSlip probability, EV,
  bankroll, stake, and payout contracts before deep design.
- Final W9.11 capstone: Tier 0 — pending-highlight audit across the completed
  frontend surface.


---

## 7. What Success Looks Like (Milestones)

| Milestone | Description | Workstreams | Status |
|---|---|---|---|
| **M1: First actionable edge report** | Run `gridiron edges report --week 12` and get a ranked list of game edges with EV, Kelly stake, and best available book. | W1 + W2 + W3 + W5 | ✅ **ACHIEVED** |
| **M1.5: Weekly game-day predictions** | Run `gridiron output predictions` then `gridiron edges report` to see fresh edges. | Existing pipeline | ✅ **ACHIEVED** |
| **M1.6: One-command weekly workflow** | Run `gridiron weekly-predict` to do the full Thursday/Sunday prep in one command. | W4.1 | ✅ **ACHIEVED** |
| **M2: Know if the model makes money** | After a month of tracking bets, run `gridiron bet summary` and see your CLV, ROI, and record by confidence tier. | W6 | ✅ **ACHIEVED** |
| **M3: First prop edge** | Full prop evaluation report with accuracy, bias, coverage metrics. | W4 + W5 | ✅ **ACHIEVED** |
| **M4.5: Visual output verification** | Wire the Gridiron Edge frontend prototype to the API and walk the full surface. Every populated field is verified; every `null` field surfaces a backend gap that's tracked in §9. | W8 + W9 | ✅ **ACHIEVED** |
| **M4: Shop across 3+ books** | Run `gridiron lines --week 12` and see a cross-book comparison with best prices highlighted. | W7 | Planned |
| **M5: Friends can use it** | Stand up a web UI that your friends can access. Dashboard, game detail, edges. | W8 + W9 | Planned (delivered with M4.5 + auth) |
| **M6: Live game day experience** | Real-time win prob, live edges, hedge suggestions during a game. | W10 | Planned |

**M4.5 ✅ achieved.** The visual verification surface now exists and is actively used — the dev-panel highlight mode operationalizes it. Next milestone north star is **M4 (multi-book line shopping)** or **M5 (friends can use it, needs auth)**, both gated on W7 / auth respectively.

---
## 9. Known Issues & Backlog

Items that are not active workstreams but need tracking. Sources: surfaced during W5.5 Tier 4 cleanup, and progressively added as W8 placeholders surface backend gaps.
(§9.5 removed 2026-07-12 — superseded by §9.7/§9.8.)

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
| player_game_logs `game_id` misaligned with player rows | `transform/clean/player_stats.py::_join_game_id` | **FIXED (2026-07-11).** Root cause: merge-result Series assigned back onto a df with non-contiguous index (from upstream dropna), aligning by index label and scrambling game_id to same-week neighbors. Fix: `reset_index(drop=True)` before the 1:1 matchup-keyed merges so positional alignment holds; also derive trustworthy `is_home` from which join side matched. Regenerated player_game_logs + re-ran props compute-splits (which had aggregated against wrong game contexts). |
| clean-games clobbered full history with empty offseason result | transform/clean/games_nflverse.py `if df.empty` branch | **FIXED (2026-07-11).** weekly-predict scoped to an upcoming season fetched 0 completed games; the empty branch wrote a header-only CSV over NFL_wk_by_wk_cleaned.csv, wiping 1999→2025 (7277→1 rows). Restored via --all-years fetch+clean. Fix: empty branch refuses to overwrite populated existing history (warns, leaves intact); still writes empty schema on genuine first run. Extracted to _handle_empty_games helper. |
| Elo incremental fit crashed on empty games | ratings/elo/table.py::update_elo_state_incremental / build_elo_state_table_all_years | **FIXED (2026-07-11).** _build_years([]) → nfl_years[-1] IndexError when no completed games (offseason). Fix: empty-games short-circuit returns existing state unchanged; guarded the nfl_years access. |
| predict-week is elo-only; API champion→elo fallback for upcoming weeks | cli/weekly_predict.py::_stage_predict_week (elo-only by design); api/loaders.py::load_games_for_week + load_game (fallback shipped) | **RESOLVED via fallback (2026-07-11), deeper option deferred.** predict-week uses build_predictions_df (Elo-state-based) → archives win_prob under model_type="elo". The champion is logistic, so the games API (champion-first) found no rows for a freshly-predicted upcoming week → empty. Root cause: trained models predict from the modeling file (feature matrix), which is built from *completed* games only — so logistic/rf/xgb structurally cannot predict upcoming weeks; only Elo (carries forward, no feature matrix) can. Shipped fix: API resolves champion-first, falls back to elo when the champion has no rows for the (season, week). Correct by design — Elo is the right upcoming-week model (trained-model features like rolling EPA don't exist for a new season's Week 1). Deeper option (build an upcoming-week feature matrix so trained models predict upcoming weeks) = the §9 upcoming-feature-matrix note. |

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

---

### 9.6 D21 Deviations (compute-at-request-time)

Per D21, the API layer is a serialization boundary — every response reads from a pre-computed static artifact. Endpoints that currently violate this by computing at request time are tracked here for refactor.

| Item | Endpoint | Current behavior | Required refactor |
|---|---|---|---|
| `/model/performance` computes metrics at request time | `GET /model/performance` | Calls `build_evaluation_df` + `summarise` + scalar metric functions (Brier, log_loss, ECE, roc_auc, brier_decomposition) at request time. Real computation on the request path. | Add a batch job (post-retrain hook, or scheduled `evaluate write-summary` command) that computes the full summary and writes `data/output/evaluation/model_performance_summary.json`. Refactor the API endpoint to read that JSON and serialize. Opportunistic — expected during W8 Step 6 or in a mini-refactor after W13. |
| Runtime champion resolution (resolved by W13) | Multiple — cli/output.py hard-coded elo; API had no path forward | Consumers hard-coded specific (model_name, model_type) pairs, or would need to compare archived model outputs at request time. | **Resolved (W13 complete 2026-07-01).** Manifest at data/output/champions/champions.json; resolve_current_champion(model_name) reads from it. CLI consumers migrated to --model-type auto pattern. Elo-specific callsites (weekly_predict archive, output archive, evaluate tune, evaluate backfill defaults, ratings elo evaluate) are annotated as intentional. |
| API `season` type inconsistency | `/weeks/current` returns `season: int` (2026); `/games/{id}`, `/props/{id}` return `season: str` ("2026-2027"). Both valid but inconsistent. Surfaced by W9 during API client type generation. | Align to string form (matches games archive convention). Address opportunistically during W8 hygiene work. |
| Team-name convention: PFR-era codes vs. modern nflverse | Backend uses `KAN`, `JAC`, `LV` — legacy PFR conventions. Modern usage prefers `KC`, `JAX`. Not a bug — reference table was created deliberately — but surface area for confusion when frontend has to hardcode team lists. | Options: (a) leave as-is and require frontend/documentation to know the convention; (b) migrate to nflverse conventions repo-wide (all archives, references, tests); (c) expose a `/teams/names` API endpoint that returns the map so frontends don't guess. Not blocking. |
| MiniRepoBuilder default team-name-map inconsistent with archive data | `MiniRepoBuilder.with_teams_reference()` maps long team names to modern short codes (`KC`, `LAC`), but the rest of the codebase uses PFR-era short codes (`KAN`, `JAC`) in archives and CSVs. Tests that need to join across a team-name-map boundary must remember to use short codes matching the fixture's output, not the codebase's convention. | Either: (a) update the fixture to use PFR-era short codes to match the codebase; (b) make the mapping configurable per-test with a sensible default; (c) migrate the codebase to modern short codes (larger effort). Surfaces every time a test writes cross-CSV joins. |

### 9.7 W8 backend hygiene backlog (surfaced by prototype audit)

Gaps between what the frontend prototype expects and what our API returns today. Surfaced during the 2026-07-04 systematic prototype-vs-implementation audit. Items are prioritized by user-visible impact per the audit findings; P0 = blocks a screen from being usable, P1 = adds significant value to a partially-shipped screen, P2 = polish or nice-to-have.

**Not blocking any specific active workstream.** Items are pulled from this list as future work (per-screen or per-domain), not as a monolithic "W8 Tier 4" tier. Deferred items blocked on other workstreams reference those workstreams inline.

#### Team-related data gaps

| Item | Priority | Notes |
|---|---|---|
| Team `primary_color` + `secondary_color` (hex) | P0 | Used by TeamMark, TeamHero, chip fills across all screens |
| Team `city` / `name` split (currently one string) | P0 | Prototype renders as "Kansas City · Chiefs" with serif italic name |
| Team `conference` (AFC/NFC) | P0 | Used by conference filter pills, division labels |
| Team `division` letter (N/S/E/W) | P0 | Same |
| `TeamRankingRow.rank_delta` (week-over-week rank change) | P1 | Requires previous-week rank snapshot storage |
| `TeamProfile.ats_record` (season aggregate) | P1 | Requires odds join + season aggregation |
| `TeamProfile.ou_record` (season aggregate) | P1 | Same |
| `RecentResult` enriched with `spread_line`, `ats_result`, `total_line`, `total_result` | P1 | Requires per-game odds history |
| `TeamProfile.upcoming_games[]` with difficulty index | P1 | Difficulty derived from opponent rating |
| `TeamProfile.postseason_outlook` (compose from `/projections`) | P2 | Cross-endpoint composition |
| Rating history with uncertainty band `{week, rating, lo, hi}` | P2 | Requires Elo uncertainty estimation |
| Off/def rating decomposition | Blocked | Real modeling work; deferred workstream |
| Situational splits by down/quarter/etc. | Blocked | PBP-derived; deferred workstream |
| Top players by WAR | Blocked | Significant ML work; deferred workstream |

#### Game-related data gaps

| Item | Priority | Notes |
|---|---|---|
| `GameSummary.kick_time` (actual time-of-day, not just date) | P0 | Prototype renders "Sun · 4:25 PM ET" |
| `GameDetail.tv_network` | P1 | e.g., "CBS", "NBC", "ESPN" |
| `GameDetail.venue_text` | P1 | e.g., "M&T Bank Stadium · BAL" |
| `GameDetail.weather_text` | P1 | e.g., "Light rain · 41°F · Wind 11mph" |
| `GameDetail.top_edge` (compose from `/edges` filtered to game_id) | P1 | Top-EV edge for this specific game |
| `GameDetail.top_prop_edges[]` (compose from `/props` filtered to game_id) | P1 | 4-5 top prop edges for this game |
| `GameDetail.day_of_week_group` (for filter) | P2 | Enables day filter pills on GamesList |
| Game storyline text | Blocked | Requires generation; deferred |
| Multi-book market odds (spread/total/ML) | Blocked on W7 | Multi-book ingest |
| Live game state | Blocked on W10 | Live ingest |
| Injuries data | Blocked on §5.3 | Injury data source decision |
| Swing factors | Blocked on feature attribution | Deferred workstream |

##### Projections-related gaps

| Item | Priority | Notes |
|---|---:|---|
| Conference / division composition | P1 | Available through the existing shared team-metadata cache; do not duplicate static metadata into the simulation artifact. W9.11 Tier 1. |
| Elo delta contract and unavailable state | P1 | `/projections` computes current-week Elo minus prior same-season week. Rename the ambiguous response field to `elo_delta`, label it explicitly, and surface Week 1 as unavailable rather than a silent em dash. W9.11 Tier 1. |
| Current record | Deferred | Not present in `projections_summary.csv`; add only through a verified source and deliberate composition path. |
| Average-wins presentation | P1 | `AVG_WINS` is the canonical season-result projection. Do not manufacture a discrete projected record from an expectation. W9.11 Tier 1. |
| Simulation provenance | P2 | The current response supports season, simulation count, and computed time. Model/config identity, random seed, and elapsed duration require a richer metadata sidecar. |
| True week-over-week projection movement | Deferred | Requires persisted prior simulation summaries and like-for-like comparison. The existing delta is Elo movement, not playoff-probability movement. |


#### Prop-related gaps

| Item | Priority | Notes |
|---|---|---|
| `/players/{player_id}/history?stat=&limit=` — game log endpoint | P0 | Powers L6 sparkline in PlayersExplorer + 12-game history chart in PlayerProp ✅ shipped (B1)|
| Player season averages endpoint | P1 | For PlayerProp header stats (Pass yds/g, Rush yds/g) |
| `PropSummary.related_props[]` (via `?team=&exclude_prop_id=` filter) | P1 | For PlayerProp related props sidebar |
| Stat display names ("Pass Yds" vs "qb_pass_yards") | P2 | Frontend or backend mapping |
| Distribution shape parameters (skewness on ProjectionBlock) | P2 | For richer density curve rendering |
| Player jersey number + physical stats | P2 | Needs roster data |
| MVP odds / honors odds | P2 | External prop data |
| Prop injury status | Blocked on §5.3 | |
| Prop reasoning | Blocked on feature attribution | |
| Multi-book prop shopping | Blocked on W7 | |

#### Portfolio-related gaps

| Item | Priority | Notes |
|---|---|---|
| `/portfolio/summary?period=7d|30d|90d|ytd|all` | P0 | Powers period pills on Bankroll |
| `/portfolio/curve?horizon=21d&with_projection=true` | P1 | Powers projected band on curve |
| `bet.recommended_stake` field on ledger | P1 | Powers Kelly adherence dashboard |
| `bet.clv_computed_at` + populated CLV | P1 | Currently null in summary |
| `/portfolio/kelly-adherence` endpoint | P1 | Powers Kelly adherence card + distribution bars |
| `/portfolio/goals` CRUD endpoint | P2 | Powers Goals view |
| `/portfolio/tax` endpoint (deposit/withdrawal log) | P2 | Powers Tax & log view |
| Cashout value computation | Blocked on W10 | Requires live odds |
| Multi-sport support (if ever added) | Deferred | Not currently in scope |

#### Compare data expansion

| Item | Priority | Notes |
|---|---|---|
| ~12 additional team metrics per cohort (yardage, red zone %, third down %, pressure rate, takeaways) | P1 | Prototype has 24 metrics vs our 8 (Step 7 shipped 8) |
| Percentile per metric per cohort (not just for 4 stats) | P1 | Extends Step 2 percentile pass |
| `vs_winning` and `vs_top_10` cohorts | P1 | Extends Step 7 cohorts (currently 4: season/l4/home/away) (4 shipped; vs-winning/vs-losing/vs-top-10 still deferred — need opponent-record + self-ranking passes). |
| Matchup edges per dimension on `/compare/teams` | P1 | Derived from percentile diff per side |
| Special teams EPA metric | P2 | Requires PBP work |
| Penalties/game metric | P2 | Requires PBP work |

#### BetSlip / edges gaps

| Item | Priority | Notes |
|---|---|---|
| `EdgeRowShape.cover_prob` (or `model_prob`) exposed | P0 | Powers combined EV computation on parlays + Kelly stake suggestion |
| `/edges/correlations?game_id=` endpoint | P1 | Powers SGP correlation warning + parlay correlation matrix in Tools |
| Line movement history | Blocked on W7 | Powers "moved -0.5" indicators + live alerts |
| Book-level odds per game/market | Blocked on W7 | Powers book selector at bottom of slip |

#### Dashboard data gaps

| Item | Priority | Notes |
|---|---|---|
| `/model/performance/history?weeks=4` endpoint | P1 | Powers model performance sparkline on Dashboard |
| Rolling ROI windows (7d/30d) on portfolio summary | P1 | Powers "30d +24.3%" big number on Dashboard rail |
| Multi-sport switcher data | Deferred | Not currently in scope |

---

### 9.8 W9 frontend polish backlog

Remaining frontend gaps after the W9.5–W9.10 fidelity arc. Original audit 2026-07-04; **pruned 2026-07-12** to remove everything shipped in the arc. Priority: P0 = blocks screen use, P1 = significant value, P2 = polish. "Blocked" = waiting on a named workstream.

> Shipped in W9.5–W9.10 and removed from this list: all five original
> primitives (Pill, WhyLink, TeamMark-colors, Spark, TeamHero), the chart
> primitives (RatingChart, DistributionChart, BarChart), Dashboard
> sections, GameDetail rebuild, PlayerProp rebuild, Teams split-view,
> Compare both modes, split-view + composed-screen layout patterns.
> See ROADMAP §4 (arc) + HANDOFF §7 (primitive inventory).

#### Shared primitives (remaining)

| Item | Priority | Notes |
|---|---|---|
| `ProbBand` — generic WinProbBand w/ tick labels + color/height props | P1 | |
| `ConfPill` descriptive labels ("Higher confidence" vs "High") | P2 | |
| `Pct` — signed percentage renderer w/ pos/neg coloring | P2 | Currently inline |
| `Segmented` — mode/split switcher (extract from Compare's inline pills) | P2 | |

#### Chart components (remaining)

| Item | Priority | Notes |
|---|---|---|
| `BankrollCurve` w/ projected band | P1 | Dashed projection + uncertainty band + "Today" marker |
| `WinProbChart` — line w/ drive-event markers | Blocked on W10 | LiveGame |
| `LineMovementChart` — odds over time | Blocked on W7 | LineShopping drilldown |
| `Waterfall` + comparables retrieval | Blocked on feature attribution / nearest-neighbor retrieval | Explain screen |
| Correlation heat-map grid | Blocked on `/edges/correlations` | Tools + BetSlip SGP |


#### Per-screen remaining sections

**PlayoffProjections (/projections)** — W9.11 Tier 1 active

| Item | Priority | Notes |
|---|---:|---|
| HeatCell with sequential probability intensity | P0 | Fixed absolute 0–1 scale across all five postseason stages; numeric value remains primary. |
| Sortable headers with explicit active direction | P0 | Shared accessible primitive; default Win SB descending; nulls always last. |
| All / AFC / NFC filters | P1 | Reuse Pill and compose conference from shared team metadata. |
| Conference / division team context | P1 | Render compactly within team identity to preserve the centered layout. |
| Elo delta badge | P1 | Rename ambiguous contract field, label explicitly as Elo movement, and provide an unavailable state when no prior same-season week exists. |
| Simulation-run metadata | P1 | Season, simulation count, and computed time only. Do not fabricate model version, seed, or runtime. |
| Status and heat context | P2 | Explain probability intensity and the globally pending clinched/eliminated fields without repeating pending chips on every row. |
| Team-profile navigation | P2 | Follow the existing NavContext pattern and route to `/teams?team={abbr}`. |

**BetSlip (`/betslip`)** — not yet rebuilt

| Item | Priority | Notes |
|---|---|---|
| Kelly stake suggestion card w/ "Use" button | P0 | `utils/odds.ts` has `kelly()` |
| Bankroll % indicator on stake input | P0 | AppState has bankroll |
| EV row on payout summary | P0 | Needs combined model prob (`cover_prob`, §9.7 P0) |
| Round-robin mode (`choose(n,k)`) | P1 | Pure frontend |
| Teaser mode (±6/6.5/7) | P1 | Needs teaser pricing |
| LegCard enhanced (numbered, model comparison, EV/conf pills) | P1 | |
| Quick stake buttons | P1 | |
| SGP correlation warning | Blocked on `/edges/correlations` | |
| Live line-movement banner / book selector | Blocked on W7 | |

**GamesList (`/games`)** — table shipped; card layout pending

| Item | Priority | Notes |
|---|---|---|
| Rich card layout (kick, teams, spread/total/ML, WP, band, lean, actions) | P1 | |
| Filter pills (day, has-edge, primetime, weather) | P1 | |
| "+ Slip" button per row | P1 | |
| Network / weather-alert badges per row | Blocked | `tv_network` / `weather_text` backend |
| Time-of-day sort | Blocked | `kick_time` backend (§9.7 P0); date-sort shipped |

**PlayersExplorer (`/players`)** — table shipped; compare rail pending

| Item | Priority | Notes |
|---|---|---|
| Compare checkbox + rail (selected props) | P0 | |
| L6 sparkline column | P1 | **Unblocked by B1** (player-history); use `Spark` + `usePlayerHistory` |
| Colored stat/lean cells | Blocked on line context (odds) | |
| Sort by EV | P2 | |

**Tools (`/tools`)** — 3-tool grid shipped; 6-tab pending

| Item | Priority | Notes |
|---|---|---|
| Tab switcher for 6 tools | P0 | |
| Hedge calculator | P1 | Pure frontend math |
| Devig calculator | P1 | Pure frontend math |
| Slider component | P1 | Kelly + model tuning |
| Middle finder / Arbitrage table | Blocked on W7 | |

**Settings (`/settings`)** — single view; sidebar pending

| Item | Priority | Notes |
|---|---|---|
| Sidebar layout (8 sections) | P0 | |
| Data & export (CSV/PDF/delete) | P2 | |
| Display prefs (theme, density, tone) | P2 | Client-side |
| Connected books / alerts / model-tuning / limits | Blocked | OAuth / server-side pref storage |

**Onboarding (`/onboarding`)**

| Item | Priority | Notes |
|---|---|---|
| Tone preview step | P1 | Client-side |
| Progress bar, skip link, Kelly callout | P2 | |
| Sports selection / books connection | Blocked | Multi-sport not shipping / OAuth |

**GameDetail / PlayerProp / TeamsScreen / Compare** — rebuilt in the arc.
Remaining items are blocked-only:
- GameDetail: Swing factors (feature attribution), Injuries (§5.3), market-side lines (W7)
- PlayerProp: "why the model leans" (feature attribution), related-props sidebar (backend filter), line-shopping (W7)
- TeamsScreen: schedule-difficulty (upcoming_games backend), WAR top-players, off/def ranking tabs (off/def decomposition)
- Compare: vs-winning/losing/top-10 splits (opponent-record + self-ranking), book line + O/U coloring (W7), Change 6 sortable/drag rows (P2)

#### Cross-cutting patterns (remaining)

| Item | Priority | Notes |
|---|---|---|
| Sortable table headers (shared) | P1 | Projections, GamesList, PlayersExplorer, Bankroll |
| Slider input (shared) | P2 | Tools, Settings |

#### Priority summary (post-prune)

- **P0:** ~10 — mostly PlayoffProjections + BetSlip (the two unrebuilt screens)
- **P1:** ~20 — per-screen enhancements + remaining primitives/charts
- **P2:** ~10 — polish
- **Blocked:** ~20 — W7, W10, §5.3, feature attribution, OAuth

The two biggest remaining frontend chunks are **PlayoffProjections** and **BetSlip** rebuilds (neither touched in the W9.5–W9.10 arc). Everything else is per-screen polish or blocked on a named workstream.

### Deferred task: Pending-highlight audit sweep

After the next full-retrain pipeline run populates all backend data, walk every built screen with dev-panel Highlight mode ON. For each silently-missing element (shows blank/em-dash but doesn't light up), add a PendingChip / ComingSoonCard / field_status marker. Produces a punch-list of any larger gaps for follow-up.

Screens to walk: Dashboard, GamesList, GameDetail, TeamsScreen, PlayerProp, PlayersExplorer, PlayoffProjections, Compare, BetSlip, Bankroll.

Status: **Unblocked (2026-07-11)** — pipeline populated cohort splits, opponent-allowed, situational splits, projections. Ready to run. This is a natural next frontend task (capstone of the W9.8 highlight work).

### Future note: Upcoming-week feature matrix

Trained models (logistic/rf/xgb game models; all prop models) predict from the modeling file / prop feature tables, both built from *completed* games. So they cannot predict upcoming (unplayed) weeks — no feature rows exist. Consequences observed 2026-07-11 (offseason):
- Games serve **elo only** for upcoming weeks (WP populated; spread/total/projected-score null — those come from trained-model post-proc).
- **Props + edges empty** for upcoming weeks (prop models + odds absent).

To get trained-model projections for upcoming weeks: build an upcoming-week feature matrix (fold the upcoming schedule into build-features, compute per-game features for unplayed games — many rolling features are thin/undefined for Week 1 of a new season), then run the champion predict path + prop projections against it.

Optional / medium workstream. Elo-in-offseason is a reasonable default; this is only worth building if trained-model upcoming projections are wanted pre-season (more useful mid-season, where rolling features exist for the next unplayed week).

### Backlog (from 2026-07-06 audit)

- [ ] Games trainer: introduce `GamesTrainer.predict_with_meta(df, meta)` mirroring the prop-side pattern. Would enforce `meta.feature_columns` as the source of truth at predict time, making the fit/predict feature contract explicit rather than conventional. Callers in `evaluation/backfill.py` and any future game predictor path would use it.

- [ ] Games trainer: metadata records `feature_columns` from the spec, not from the actual fit. If `feature_fn` and `spec.feature_set.feature_names` ever disagree, the metadata lies about what the model was fit on. Verify they match at metadata-build time, or record from actual `x_train.columns`.

- [ ] Games trainer: audit `feature_fn` implementations for cross-slice dependencies. `_filter_for_walk_forward` re-splits pre-computed features, which assumes feature construction is purely row-local (no aggregate statistics over the training pool).

- [ ] Games trainer: `MIN_CV_TRAIN_ROWS = 4000` is a fixed default that walk-forward overrides. Consider scaling with training-pool size (e.g. `min(4000, N // 3)`) so callers don't need to know the guard exists.

- [ ] Prop trainer: 50% NaN threshold is porous at era boundaries (see passing_cpoe on qb_pass_yards). Consider tighter threshold, per-era feature sets, or imputation policy. Decide before making claims about prop dashboard historical coverage.

- [ ] `full-retrain --only` semantics: `--only` treats unlisted stages as "must not run", which blocks partial-resume workflows where completed stages aren't in the `--only` list. Consider making `--only` accept a dependency-satisfying stage, or improve the error to suggest `--assume-done`.

- [ ] Composite dependency graph: consider treating "artifact on disk" as satisfying a dependency, so partial resumes don't require explicit `--assume-done` flags.

- [ ] 2026 PBP fetch warning: `_stage_refresh_all_data` requests PBP for the current season and gets a warning that the source only supports through 2025. Clamp to the max available season instead of failing softly.

- [ ] `_parse_baseline_report` extracted zero-valued cells for the previous report's win_prob_logistic row, causing the delta to display as the raw current value with a `+` sign rather than a proper delta or an em-dash. Investigate whether the parser is misaligning columns or if 0.0 is being produced where None is expected.

### Tooling configuration hygiene

- **Ruff configuration cleanup — P2.** Remove the retired `ANN101` ignore and
  decide how repository-wide `PLR0917` should be handled. The current full
  Ruff run reports existing functions with more than five positional
  parameters across FastAPI/Typer entrypoints, Numba simulation kernels,
  serializers, domain functions, fixture builders, and patch-injected tests.
  Evaluate targeted per-file exclusions for framework- or JIT-constrained
  callsites versus a dedicated keyword-only/signature-refactor pass. These
  findings predate W9.11 and should not be remediated opportunistically inside
  frontend workstreams.

---

## 10. Changelog for This Document

| Date | Change |
|---|---|
| 2026-07-28 | **W9.11 Tier 1 complete.** Rebuilt PlayoffProjections with a full-cell probability heat matrix, accessible sorting, dependent conference/division selectors, current Elo and record context, explicit Elo-delta semantics, as-of-week metadata, Week 1-aware unavailable treatment, and team-profile navigation. Composed existing `/projections` and `/teams` contracts without adding request-time computation. Verified against the real 32-team artifact. |
| 2026-07-28 | **W9.11 Tier 1 design locked.** Moved PlayoffProjections ahead of the deferred audit sweep. Verified the seven-column simulation artifact, metadata sidecar, `/projections` schema/serializer/route, shared team metadata, Pill, custom routing, and design tokens. Preserved AVG_WINS as the canonical projection; clarified movement as Elo delta; limited run metadata to season/simulation count/computed time; locked HeatCell, sorting, conference filters, compact 920px table structure, and accessible team navigation. |
| 2026-07-28 | **W9.11 design resync.** Updated PlayoffProjections scope after verifying the real `/projections` response, `projections_summary.csv`, metadata sidecar, loader, and team-metadata path. Preserved `AVG_WINS` as the canonical projection; clarified trend as Elo delta; replaced speculative model metadata with truthful simulation-run metadata; deferred true probability movement and fabricated projected records. Marked BetSlip scope contract-dependent pending its deep-design verification pass. |
| 2026-07-12 | **W9.11 opened (Screen Completion).** Audit sweep → PlayoffProjections → BetSlip. Finishes the core-screen set. Ways-of-Working codified in PLAN.md. |
| 2026-07-12 | **ROADMAP trim + straggler fix.** Collapsed §4 completed workstreams to table (W1–W8, W13). Deleted §9.5 (superseded by §9.7/§9.8). Gutted §9.8 to genuine remainders (struck shipped W9.5–W9.10 items). Marked B1 player-history shipped in §9.7. Fixed 07-01→07-11 date stragglers; player logs 1999→2025; PLAN cross-ref. |
| 2026-07-12 | **Post-session cleanup + offseason-readiness findings.** Marked §9.2 clean-games clobber + Elo empty-games crash FIXED. Updated audit-sweep note to Unblocked. §1 frontend-arc → Complete (both Compare modes, BarChart); removed shipped player-history from What's Missing. §6 graph + Current position → W9.10 complete, between workstreams. Added findings: predict-week elo-only + champion→elo API fallback; upcoming-week feature-matrix future note. Marked B1–B4 shipped in §9.7. |
| 2026-07-11 | **W9.10 complete.** Compare Screen Rebuild — both modes shipped. New backend B1–B4 (player-history endpoint, 4-cohort opponent-allowed, defense-by-team, players roster). game_id scramble bug fixed at root in transform/clean/player_stats.py. |
| 2026-07-11 | **ROADMAP audit + cleanup.** Recorded W9.8 (dev panel) + W9.10 (Compare, active) which were missing. Updated §1 Working/Missing to post-frontend reality. Filled §4 Active with W9.10 + the Player-vs-Defense backend work (player-history endpoint + opponent-allowed splits expansion) as immediate next. Marked §5.1/§5.4 file-storage + api/frontend as shipped. Redrew §6 current position. Reframed M4.5 as achieved. |

***
