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
| API serving layer (W8) | ✅ Complete | 16 endpoints, Pydantic-validated, field_status placeholder convention, champion resolution wired |
| Frontend app (W9) | ✅ Complete | Vite + React + TS; all screens render; typed openapi-fetch + React Query |
| Frontend fidelity arc (W9.5–W9.10) | ✅ Mostly complete | Dashboard, GameDetail, Teams split-view, PlayerProp rebuilt; 5 shared primitives + DistributionChart + RatingChart; dev-panel highlight mode. Compare (W9.10) in progress. |
| Cohort splits (11 metrics) | ✅ Complete | Team cohort splits season/l4/home/away with off+def reciprocal pairs; per-prop situational splits; opponent-allowed by position |

### What's Missing

| Area | Status | Impact |
|---|---|---|
| Model ensemble | ❌ Not started | Individual models only; no weighted combination |
| Multi-book odds ingestion | ❌ Not started | Blocks line shopping, book selectors, real bet-slip odds, per-week book lines |
| Injury/news feed | ❌ Not started | Blocks W4.5 scenario engine + injury UI fields |
| Live game / real-time | ❌ Not started | No live state, odds, or win prob |
| Player game-history endpoint | ❌ Not started | Blocks Compare Player-vs-Defense bar chart, PlayerProp 12-game chart, PlayersExplorer L6 sparkline |
| Off/def rating decomposition | ❌ Not started | Blocks off/def ranking tabs, Compare Off/Def mini-stats |
| Frontend prototype-fidelity backlog | 🟡 Partial | ~180 catalogued items in §9.7/§9.8; core screens done, polish + blocked-on-data items remain |

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

#### W3.5: Audit Remediation — ✅ COMPLETE
Closed ~100 findings from `audit_2026_06_18.md` across 11 audit units, plus 4 cross-cutting patterns. Major outcomes:

- **Identity unification:** `(model_name, model_type)` composite identity flows through every persistence layer.
- **Elo simulator:** Single canonical history simulator drives both the state table and the tuner.
- **Task-discriminated metadata:** Metrics live in a single dict on `BaseModelMetadata`.
- **Vectorization:** All audit-flagged per-row apply patterns eliminated.
- **CLI ergonomics:** Stage staleness warnings, calibration health flags, archive-driven prop CLI, enum-based string constants.

See DECISIONS.md D1–D12 for architectural decisions.

#### W4: Player Data & First Prop Models — ✅ COMPLETE
Player game logs (nflreadpy, 1999–2024), 4 player feature modules, 5 prop models (QB pass/rush yards, RB rush yards, WR rec yards, TE rec yards), PropTrainer base class, post-processing enrichment, prop archive, prop CLI. M3 achieved. See CHANGELOG.md for details.

#### W4.1: Composite CLI Workflows — ✅ COMPLETE
Four composite commands wrap related single-purpose commands into complete workflows: `weekly-predict`, `post-week`, `full-retrain`, `verify`. Shared infrastructure in `cli/_composites.py` provides stage abstraction, dependency validation, soft-fail semantics, and consolidated summary rendering. M1.6 achieved.

#### W5: Edge Engine — ✅ COMPLETE
Edge calculation (moneyline, spread, total), recommendations builder, CLV analysis, CLI commands. See CHANGELOG.md for details.

#### W5.5: Deep Code Review + Test Suite Review — ✅ COMPLETE
Multi-session opportunistic cleanup that closed 30 backlog items and surfaced two real bugs that were fixed during the sweep. See CHANGELOG.md for details.

#### W6: Portfolio & Bet Tracking — ✅ COMPLETE
Bet ledger, bankroll management (decoupled), performance analytics, 8 CLI commands. Full round-trip validated. M2 achieved. See CHANGELOG.md for details.

#### W13: Runtime Champion Resolution — ✅ COMPLETE (2026-07-01)
Static manifest artifact at `data/output/champions/champions.json` written by `full-retrain`. `resolve_current_champion(model_name)` reads from it. CLI consumers migrated to `--model-type auto` pattern. Unblocks all downstream champion-only consumption paths.

**Scope:** manifest schema + reader API (Tier 1), writer + full-retrain integration + manual-override flags (Tier 2), CLI consumer migration + intentional-Elo annotations (Tier 3).

**Discovered as a mid-tier scope elevation from W8** when the API needed runtime champion resolution to serve `/games`, `/edges`, `/props`.

See CHANGELOG.md for details.

#### W9: Frontend — ✅ COMPLETE (2026-07-03)
Vite + React + TypeScript app at `frontend/` consuming the 16-endpoint API end-to-end. All 20 prototype-referenced screens render. 12 screens consume the API; 4 are blocked-screen placeholders with full blocker context; 4 are client-side (Onboarding, Settings, Tools, BetSlip).

**Architecture established:** Data flows via `openapi-fetch` typed client wrapped in per-endpoint React Query hooks. Three-Context state model (AppState, BetSlip, Nav) with localStorage persistence. Shared field-status primitives (`<PendingField />`, `<BlockedField />`, `<FieldValue />`) compose consistently. Consistent error UX via `<ErrorCard />` and global `<OfflineBanner />`.

**Unlocks:** M4.5 achieved (visual verification of full output set). W8 Tier 3 additive dataset priority now discoverable — the frontend has surfaced which pending/blocked states matter most.

#### W8: API Serving Layer — ✅ COMPLETE (2026-07-04)

**Goal:** Expose analytics outputs through a REST API so a frontend (or other consumers) can access them, and so the full slate of outputs becomes visually verifiable in one place.

**Delivered (2026-07-04):**
- ✅ 16 endpoints returning populated data with Pydantic-validated responses (Tier 2).
- ✅ 7 additive datasets computed by dedicated modules, persisted to `data/output/`, and consumed via loader→serializer→route pattern (Tier 3).
- ✅ Placeholder convention (D14) applied consistently: null + `_meta.field_status` for anything not yet populated.
- ✅ Champion resolution flows manifest → loader → serializer → route for game and prop endpoints.
- ✅ Testing infrastructure: `MiniRepoBuilder` extended with 4+ W8-specific methods; integration tests via FastAPI `dependency_overrides`.

**Remaining scope items** (each in a future workstream):
- Off/def rating decomposition (Elo variants or EPA-derived, real modeling work).
- Feature attribution / swing factors / prop reasoning (feature-attribution workstream).
- Injury data source and downstream fields (ROADMAP §5.3).
- Multi-book odds / line shopping fields (W7).
- Live game state fields (W10).
- PBP-derived aggregations for red_zone_rate_allowed and similar (future workstream).

**Unlocks:** W9 (Frontend) consumes end-to-end. Future workstreams (W12, W4.5, W7, W10) can proceed independently.

#### W9.5: Dashboard Rebuild + Cross-Cutting Primitives — ✅ COMPLETE (2026-07-04)

Small workstream between W8 close-out and next major work. Shipped 11
substeps: team metadata backend patch and CSV consolidation (Tier 1),
5 shared primitives (Pill, WhyLink, TeamMark-with-colors, Spark, TeamHero)
(Tier 2), 4 Dashboard sections + integration (Tier 3). Also consolidated
NFL_long_to_short_name.csv and NFL_conference_division.csv into
NFL_team_metadata.csv. See CHANGELOG.md for details.

#### W9.6: GameDetail Full Fidelity — ✅ COMPLETE (2026-07-07)

Rebuilt GameDetail from skeleton to prototype fidelity across 9
substeps. Full-width game header with team-colored TeamHero + kick +
model lean callout. Main column: Lines & Fair Value table (Model +
Recommendation rows populated), Win Probability card, Team Comparison
card (consumes Step 7c team_comparison field). Right rail: Top Prop
Edges card + placeholder cards for blocked sections (Swing Factors,
Injuries). See CHANGELOG.md for details.

#### W9.7: Teams Split-View Rebuild — ✅ COMPLETE (2026-07-07)

Restructured `/teams` and `/teams/:abbr` into consolidated split-view
screen at `/teams` with optional `?team=X` param. Left column rankings
table with 5-tab strip (Overall + 4 blocked); right column profile
with team hero band, rating chart, situational splits, recent results,
postseason outlook, and 2 blocked placeholders. New `RatingChart`
primitive supports inline W/L markers per week. See CHANGELOG.md for
details.

#### W9.9: PlayerProp Rebuild — ✅ COMPLETE (2026-07-07)

Rebuilt PlayerProp screen from skeleton to prototype fidelity across
8 substeps. Team-colored player hero band with prop summary callout
card on right side. Below: distribution chart (new primitive),
situational splits (Step 5 data), Player vs Defense table with WhyLink,
5 blocked ComingSoonCards. New `DistributionChart` primitive available
for W9.10 Compare. See CHANGELOG.md for details.

#### W9.8: Dev Panel + Pending Highlight Mode — ✅ COMPLETE (mechanism; audit deferred, 2026-07-01)

Floating bottom-right dev panel with a Highlight Pending & Blocked
toggle. When on, every pending/blocked element lights up orange so
backend gaps are visible during a visual pass.

**Delivered:** `DevPanelContext` + `--highlight` var (Tier 1);
`usePendingHighlight` hook, retrofit of `PendingField`/`BlockedField`,
consolidated `ComingSoonCard`, new `PendingChip` primitive (Tier 2).

**Deferred:** Tier 3 audit sweep — walking every screen in highlight
mode requires fully-populated backend data to distinguish frontend
gaps from unpopulated fields. Tracked in §9 deferred tasks.

#### Frontend Fidelity Arc (W9.5–W9.10) — framing note

W9.5 through W9.10 form a coherent push: rebuild each prototype screen
to fidelity, consuming a shared primitive set (Pill, WhyLink, TeamMark-
with-colors, Spark, TeamHero, DistributionChart, RatingChart) and the
W9.8 highlight discipline. W9.5–W9.9 complete; W9.10 (Compare) active.

See CHANGELOG.md for details.

### Active Workstream

#### W9.10: Compare Screen Rebuild — 🟡 ACTIVE

Two-mode matchup surface: Team vs Team + Player vs Defense.

**Team vs Team — ✅ complete.** Mode switcher, mirrored team pickers +
swap, cohort strip, narrative card, collapsible summary card, three
matchup cards with mirrored ranking-bar collision rows (offense value
↔ reciprocal defense-allowed, edge chips, descriptive sublabels,
title-style metric names), centered layout. Backed by an 11-metric
cohort_splits expansion (added def_pass_epa, def_third_down_pct,
def_redzone_td_pct so every offensive metric has its reciprocal).

**Player vs Defense — 🟡 redesign in progress.** Being reworked to
mirror Team-vs-Team: independent player / stat / team pickers, 7-split
strip (season/l4/home/away/vs-winning/vs-losing/vs-top-10), a per-game
bar chart (player's stat per game + team-allowed average line + book
line), and a "matchup, plainly" verdict card.

**Blocked on backend (immediate next work — Path C):**
1. `/players/{player_id}/history?stat=` endpoint — per-game stat values
   from `player_game_logs.parquet` (data exists; expose it). Powers the
   bar chart centerpiece. Also unblocks PlayerProp 12-game chart +
   PlayersExplorer L6 sparkline (§9.7 P0 item).
2. Expand `opponent_allowed` splits from 2 (season/l5) to 7
   (season/l4/home/away/vs-winning/vs-losing/vs-top-10) — mirrors the
   team_cohort_splits expansion pattern.

Book line + over/under bar coloring remain deferred (blocked on odds,
W7) — marked pending per highlight discipline.

**Deferred within W9.10:** Change 6 (sortable rows by category/edge +
drag-to-reorder) — P2, §9.8. Only build if missed.

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

**Status:** Files held through W8/W9 as planned; no complex pandas
merge chains emerged in the API layer (serializers read pre-computed
artifacts per D21). **Decision: stay file-based.** Re-evaluate only if
multi-user access, concurrency, or transactional bet-ledger integrity
becomes a real requirement.

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
│   └── W9.5–W9.9 ✅ (fidelity arc) · W9.10 🟡 (Compare, active)
│
├── W12 (Ensemble) 🟢 planned
│
├── W7 (Multi-Book) 🟢 planned
│
└── W10 (Real-Time) 🟢 deferred

```


**Current position:** Backend + API + frontend all shipped. Frontend fidelity arc (W9.5–W9.10) rebuilt the core screens; W9.10 (Compare) active with Player-vs-Defense blocked on two backend endpoints (player game-history + opponent-allowed splits expansion — the immediate next work). Path forward after W9.10:
- **Frontend polish backlog** (§9.7/§9.8) — pull P0/P1 items per-screen.
- **Pending-highlight audit sweep** (§9 deferred) — now unblocked (data populated).
- **W12 (Model Ensemble)** — soft-blocked on §9.2 walk-forward bug.
- **W4.5 (Scenario)** — blocked on §5.3 injury data.
- **W7 (Multi-Book)** — blocked on §5.2 odds source; unblocks many deferred UI items.
- **W10 (Real-Time)** — deferred.


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

**M4.5 ✅ achieved.** The visual verification surface now exists and is
actively used — the dev-panel highlight mode operationalizes it. Next
milestone north star is **M4 (multi-book line shopping)** or **M5
(friends can use it, needs auth)**, both gated on W7 / auth respectively.

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
| player_game_logs `game_id` misaligned with player rows | `transform/clean/player_stats.py::_join_game_id` | **FIXED (2026-07-01).** Root cause: merge-result Series assigned back onto a df with non-contiguous index (from upstream dropna), aligning by index label and scrambling game_id to same-week neighbors. Fix: `reset_index(drop=True)` before the 1:1 matchup-keyed merges so positional alignment holds; also derive trustworthy `is_home` from which join side matched. Regenerated player_game_logs + re-ran props compute-splits (which had aggregated against wrong game contexts). |

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

Gaps between what the frontend prototype expects and what our API
returns today. Surfaced during the 2026-07-04 systematic
prototype-vs-implementation audit. Items are prioritized by
user-visible impact per the audit findings; P0 = blocks a screen
from being usable, P1 = adds significant value to a partially-shipped
screen, P2 = polish or nice-to-have.

**Not blocking any specific active workstream.** Items are pulled from
this list as future work (per-screen or per-domain), not as a monolithic
"W8 Tier 4" tier. Deferred items blocked on other workstreams reference
those workstreams inline.

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
| Top players by WAR | Blocked | Significant ML work; ROADMAP §9.5 |

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

#### Projections-related gaps

| Item | Priority | Notes |
|---|---|---|
| `TeamProjectionRow.rating` (currently need cross-endpoint join) | P1 | Composed from `/teams` |
| `TeamProjectionRow.trend` (currently need cross-endpoint join) | P1 | Same |
| `TeamProjectionRow.current_record` | P1 | For "Curr." column |
| `TeamProjectionRow.projected_record` (derived from `avg_wins`) | P1 | Format like "14-3" |
| `TeamProjectionRow.conference` / `division` | P1 | For filter pills + Division column |
| `ProjectionsList.model_version` + `random_seed` on envelope | P2 | Metadata already in sidecar |

#### Prop-related gaps

| Item | Priority | Notes |
|---|---|---|
| `/players/{player_id}/history?stat=&limit=` — game log endpoint | P0 | Powers L6 sparkline in PlayersExplorer + 12-game history chart in PlayerProp |
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
| `vs_winning` and `vs_top_10` cohorts | P1 | Extends Step 7 cohorts (currently 4: season/l4/home/away) |
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

### 9.8 W9 frontend polish backlog (surfaced by prototype audit)

Frontend implementation gaps between prototype fidelity and current
shipped code. Same audit source (2026-07-04). Same priority framework.

Items grouped by:
1. Shared primitives (used across many screens)
2. Chart components (mostly new)
3. Chrome and layout patterns
4. Per-screen sections

#### Shared primitives (used across many screens)

| Item | Priority | Notes |
|---|---|---|
| `Pill` — shared filter toggle button | P0 | Every screen with filters inlines its own |
| `WhyLink` — explainability entry point (labeled + dot modes) | P0 | Missing entirely; used across ~10 screens in prototype |
| `TeamMark` with team primary color background | P0 | Currently uses `--bg-3` grey; big visual identity gap |
| `Spark` — generic sparkline | P1 | Currently only exists as team-scoped `RatingHistorySparkline` |
| `ProbBand` — generic version of WinProbBand with tick labels + color/height props | P1 | |
| `ConfPill` with descriptive labels ("Higher confidence" vs "High") | P1 | Currently just "High/Moderate/Low" |
| `Pct` — signed percentage renderer with pos/neg coloring | P2 | Frontend helper; currently formatted inline |
| `Segmented` — mode/split switcher (used by Compare, Tools, others) | P2 | |

#### Chart components (new)

| Item | Priority | Notes |
|---|---|---|
| `BankrollCurve` with projected band | P1 | Extends current BalanceCurve with dashed projection line + filled uncertainty band + "Today" marker |
| `DistributionChart` — SVG density curve overlay | P1 | Used by PlayerProp (projected distribution) and Explain (simulated outcomes) |
| `HistoryChart` — bar chart with hit/miss coloring | P1 | Used by PlayerProp (12-game history) |
| `GameLog` — SVG bar chart with book line + defense line overlays | P1 | Used by Compare player mode |
| `RatingChart` — line chart with uncertainty band + axis grid | P2 | Used by TeamProfile |
| `WinProbChart` — line chart with drive-event markers | Blocked on W10 | Used by LiveGame |
| `LineMovementChart` — line chart of odds over time | Blocked on W7 | Used by LineShopping drilldown |
| `Waterfall` — factor contribution visualization | Blocked on feature attribution | Used by Explain |
| Variance preview distribution bars | P2 | Used by Tools Kelly calculator |
| Correlation heat map grid | P2 | Used by Tools + BetSlip SGP mode |

#### Chrome and layout patterns

| Item | Priority | Notes |
|---|---|---|
| SubNav filter pills (day, category, market, position, etc.) | P0 | Currently inlined per screen; needs shared pattern |
| Split-view layout (e.g., TeamRankings + TeamProfile side-by-side) | P0 | Currently separate routes |
| Two-column layout (main / rail) — most screens need this | P0 | Currently single-column stacks on GameDetail, PlayerProp, Bankroll |
| Header band with hero identity + big number + action buttons | P1 | Used by Bankroll, GameDetail, PlayerProp, TeamProfile |
| `TeamHero` component (mark + city + name serif + record + rating) | P1 | Used by GameDetail, TeamProfile |
| Sidebar navigation with sub-sections (Settings, potentially Bankroll views) | P1 | Currently flat single view |

#### Per-screen missing sections

**Dashboard (`/today`)** — currently a debug scaffold; essentially unbuilt

| Item | Priority | Notes |
|---|---|---|
| Featured matchups grid (3 game cards) | P0 | Blocking primary landing page use |
| Model edges table with tab filters | P0 | Powers Model edges rail |
| Model performance rail (sparkline + big number) | P0 | Blocked on model performance history endpoint |
| Player prop edges rail (5-row compact list) | P0 | |
| Multi-sport SubNav pills | P2 | Data only NFL anyway |
| Remove API verification and field-status demo cards | P1 | Or move to `/debug` route |

**GameDetail (`/games/:id`)**

| Item | Priority | Notes |
|---|---|---|
| Two-column layout (65% main / 35% rail) | P0 | Currently single-column stack |
| Full-width game header with `TeamHero` + kick/venue/weather center + model lean + action buttons | P0 | Currently minimal header |
| Team comparison card that consumes `team_comparison` field (already populated via Step 7c) | P0 | Data ships; renderer doesn't |
| Lines & model fair value table | P1 | Partial (model side works, market side blocked on W7) |
| Win probability card with projected score + caveat callout | P1 | |
| Top prop edges card (compose from `/props` filtered to game_id) | P1 | |
| Swing factors card | Blocked on feature attribution | |
| Injuries card | Blocked on §5.3 | |
| "Add to bet slip" and "★ Track" buttons in header | P2 | Track blocked; slip integration works |

**PlayerProp (`/players/:id`)**

| Item | Priority | Notes |
|---|---|---|
| Player hero header (team-colored mark + serif italic name + season stats row) | P0 | Currently minimal card |
| Two-column layout | P0 | |
| Distribution chart (Gaussian density from mean + std) | P0 | Data exists |
| Situational splits card that consumes `situational_splits` field (already populated via Step 5) | P0 | Data ships; renderer doesn't |
| History bar chart (12-game log) | P1 | Blocked on player history endpoint |
| Related props sidebar | P1 | Blocked on related-props filter |
| "Why the model leans" reasoning column | Blocked on feature attribution | |
| Line shopping mini-table | Blocked on W7 | |

**TeamRankings + TeamProfile (`/teams`, `/teams/:abbr`)**

| Item | Priority | Notes |
|---|---|---|
| Split-view restructure (rankings + profile in one route) | P0 | Major restructure |
| Team hero band with team primary color background | P0 | |
| Rating chart with uncertainty band | P1 | Enhance current sparkline |
| Cohort splits table with colored percentile bars | P1 | Data exists via Step 7 |
| Schedule difficulty visualization (7 upcoming week blocks) | P1 | Blocked on upcoming_games backend |
| Postseason outlook rows | P1 | Blocked on postseason_outlook backend |
| Top players by WAR list with colored bars | Blocked on WAR | |
| Ranking table off/def tabs | Blocked on off/def decomposition | |

**Compare (`/compare`)** — largest single frontend gap

| Item | Priority | Notes |
|---|---|---|
| Mode switcher (Team vs Team / Player vs Defense) | P0 | |
| Split control strip (6 cohort pills) | P0 | Currently no cohort switcher |
| Three grouped sections layout ("When A has ball" / "When B has ball" / "Even footing") | P0 | Currently flat table |
| `TaleRow` with colored percentile bars + `AdvChip` edge indicator | P0 | Defines the visual identity |
| Enhanced `TeamPicker` with team colors + Off/Def mini-stat bars | P1 | |
| Auto-generated matchup narrative banner | P1 | Frontend computation on percentile diffs |
| Player vs Defense mode with `GameLog` chart | P1 | Blocked on player history endpoint |
| Drag-and-drop row reordering | P2 | Pure frontend |
| Sort by edge / sort by category buttons | P2 | |

**PlayoffProjections (`/projections`)**

| Item | Priority | Notes |
|---|---|---|
| `HeatCell` component with sequential color intensity per probability | P0 | Defines the screen's visual language |
| Sortable column headers with active-state UI | P0 | |
| Trend badge with colored background (green/red/neutral) | P1 | |
| Conference filter pills (AFC/NFC/All) in SubNav | P1 | |
| Model metadata top-right block (version, seed, run time) | P1 | Data available in sidecar |
| Heat scale gradient bar in footer legend | P2 | |
| Row click → team profile navigation | P2 | |

**BetSlip (`/betslip`)**

| Item | Priority | Notes |
|---|---|---|
| Kelly stake suggestion card with "Use" button | P0 | Data path exists (`utils/odds.ts` has `kelly()`) |
| Bankroll % indicator on stake input | P0 | AppStateContext has bankroll |
| EV row on payout summary | P0 | Needs combined model prob from legs |
| SGP mode + correlation warning | P1 | Blocked on `/edges/correlations` |
| Round-robin mode with subs count via `choose(n, k)` | P1 | Pure frontend math |
| Teaser mode with ±6/6.5/7 pt options | P1 | Needs teaser pricing logic |
| LegCard enhanced (numbered, model comparison, EV pill, conf pill) | P1 | |
| Quick stake buttons ($10/$25/$50/$100/$250) | P1 | |
| Live line-movement alert banner | Blocked on W7 | |
| Book selector at bottom of slip | Blocked on W7 | |

**GamesList (`/games`)**

| Item | Priority | Notes |
|---|---|---|
| Rich card layout instead of table row | P1 | Card shows: kick, network, weather, teams, spread/total/ML, WP, band, model lean, actions |
| Filter pills (day, has-edge, primetime, weather) | P1 | |
| Network badge per row | P1 | Blocked on `tv_network` backend |
| Weather alert indicator per row | P1 | Blocked on `weather_text` backend |
| "+ Slip" button per row | P1 | |

**PlayersExplorer (`/players`)**

| Item | Priority | Notes |
|---|---|---|
| Compare checkbox column with star toggle | P0 | Enables the compare rail |
| Compare rail on right side with selected props | P0 | |
| L6 sparkline column | P1 | Blocked on player history endpoint |
| Colored stat/lean cells (green OVER, red UNDER) | P1 | Blocked on line context |
| Filter pills in SubNav instead of inline FilterBar | P2 | |
| Sort by EV | P2 | |

**Tools (`/tools`)**

| Item | Priority | Notes |
|---|---|---|
| Tab switcher for 6 tools (Kelly, Hedge, Arb, Corr, Devig, Middle) | P0 | Currently 3-tool grid layout |
| Hedge calculator | P1 | Pure frontend math |
| Devig calculator | P1 | Pure frontend math |
| Middle finder (empty state until W7 lands) | P1 | Blocked on W7 |
| Slider component for percentage inputs | P1 | Used by Kelly + Model tuning |
| Variance preview distribution bars | P2 | Adds depth to Kelly output |
| Correlation heat map | Blocked on correlations endpoint | |
| Arbitrage finder table | Blocked on W7 | |

**Settings (`/settings`)**

| Item | Priority | Notes |
|---|---|---|
| Sidebar layout with 8 sections | P0 | Currently single view |
| Connected books section | P1 | Blocked on OAuth story (out of scope for now) |
| Alerts & notifications with 7 toggles + channel indicators | P1 | Blocked on server-side pref storage |
| Model tuning section with 5 sliders | P1 | Blocked on server-side model retuning endpoint |
| Responsible play limits section | P1 | Blocked on server-side enforcement |
| Data & export section (CSV, PDF, delete) | P2 | |
| Display preferences enhanced (theme, density, tone) | P2 | Client-side |

**Onboarding (`/onboarding`)**

| Item | Priority | Notes |
|---|---|---|
| Sports selection step + betting style tier | P1 | Missing step; multi-sport not shipping |
| Books connection step | P1 | Blocked on OAuth |
| Tone preview step | P1 | Client-side preference |
| Progress bar instead of dots | P2 | |
| Skip link (upper right) | P2 | |
| Kelly explanation callout on bankroll step | P2 | |

#### Cross-cutting interaction patterns

| Item | Priority | Notes |
|---|---|---|
| Row-click nav pattern (used across many list screens) | P0 | Partially wired |
| Filter pill component (shared) | P0 | See §9.8 Pill primitive |
| Sortable table headers (shared) | P1 | Used by Projections, GamesList, PlayersExplorer, Bankroll |
| Drag-and-drop reorder (used by Compare) | P2 | Pure frontend |
| Cohort/mode switcher pattern (used by Compare, Bankroll) | P2 | See Segmented primitive |
| Slider input (used by Tools, Settings) | P2 | |

#### Priority summary

- **P0 items:** ~35. Blocking meaningful use of one or more screens. Most valuable next work.
- **P1 items:** ~50. Adds significant value to partially-shipped screens.
- **P2 items:** ~30. Polish and nice-to-have.
- **Blocked items:** ~25. Waiting on named workstreams (W7, W10, §5.3, feature attribution).

Total estimated substeps to reach prototype fidelity across all
screens: **~80-100 substeps.**

Not all of these need to happen. This is a comprehensive backlog.
Future frontend workstreams (call them W9.5, W9.6, or per-screen
revamps) will pull from this list based on what's most valuable at
that moment.

### Deferred task: Pending-highlight audit sweep

After the next full-retrain pipeline run populates all backend data,
walk every built screen with dev-panel Highlight mode ON. For each
silently-missing element (shows blank/em-dash but doesn't light up),
add a PendingChip / ComingSoonCard / field_status marker. Produces a
punch-list of any larger gaps for follow-up.

Screens to walk: Dashboard, GamesList, GameDetail, TeamsScreen,
PlayerProp, PlayersExplorer, PlayoffProjections, Compare, BetSlip,
Bankroll.

Blocked on: full backend data population (next pipeline run).

### Backlog (from 2026-07-06 audit)

- [ ] Games trainer: introduce `GamesTrainer.predict_with_meta(df, meta)` mirroring
      the prop-side pattern. Would enforce `meta.feature_columns` as the source of
      truth at predict time, making the fit/predict feature contract explicit
      rather than conventional. Callers in `evaluation/backfill.py` and any future
      game predictor path would use it.

- [ ] Games trainer: metadata records `feature_columns` from the spec, not from
      the actual fit. If `feature_fn` and `spec.feature_set.feature_names` ever
      disagree, the metadata lies about what the model was fit on. Verify they
      match at metadata-build time, or record from actual `x_train.columns`.

- [ ] Games trainer: audit `feature_fn` implementations for cross-slice
      dependencies. `_filter_for_walk_forward` re-splits pre-computed features,
      which assumes feature construction is purely row-local (no aggregate
      statistics over the training pool).

- [ ] Games trainer: `MIN_CV_TRAIN_ROWS = 4000` is a fixed default that walk-forward
      overrides. Consider scaling with training-pool size (e.g. `min(4000, N // 3)`)
      so callers don't need to know the guard exists.

- [ ] Prop trainer: 50% NaN threshold is porous at era boundaries (see
      passing_cpoe on qb_pass_yards). Consider tighter threshold, per-era feature
      sets, or imputation policy. Decide before making claims about prop
      dashboard historical coverage.

- [ ] `full-retrain --only` semantics: `--only` treats unlisted stages as
      "must not run", which blocks partial-resume workflows where completed
      stages aren't in the `--only` list. Consider making `--only` accept a
      dependency-satisfying stage, or improve the error to suggest
      `--assume-done`.

- [ ] Composite dependency graph: consider treating "artifact on disk" as
      satisfying a dependency, so partial resumes don't require explicit
      `--assume-done` flags.

- [ ] 2026 PBP fetch warning: `_stage_refresh_all_data` requests PBP for the
      current season and gets a warning that the source only supports through
      2025. Clamp to the max available season instead of failing softly.

- [ ] `_parse_baseline_report` extracted zero-valued cells for the previous
  report's win_prob_logistic row, causing the delta to display as the raw
  current value with a `+` sign rather than a proper delta or an em-dash.
  Investigate whether the parser is misaligning columns or if 0.0 is being
  produced where None is expected.

---

## 10. Changelog for This Document

| Date | Change |
|---|---|
| 2026-07-11 | **ROADMAP audit + cleanup.** Recorded W9.8 (dev panel) + W9.10 (Compare, active) which were missing. Updated §1 Working/Missing to post-frontend reality. Filled §4 Active with W9.10 + the Player-vs-Defense backend work (player-history endpoint + opponent-allowed splits expansion) as immediate next. Marked §5.1/§5.4 file-storage + api/frontend as shipped. Redrew §6 current position. Reframed M4.5 as achieved. |

***
