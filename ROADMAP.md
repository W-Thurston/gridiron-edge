# Gridiron Edge — ROADMAP
## Long-Term Strategic Direction

---

## 0. How This Document Fits

| Document | Purpose | Updated When |
|---|---|---|
| **ROADMAP.md** (this file) | High-level long-term direction. Where we're headed and why. Workstreams, dependencies, architecture decisions. | When strategic direction changes or a workstream is completed/added. |
| **PLAN.md** | Short-term, nitty-gritty next steps. The current working checklist. Pick up from where you left off. | Every working session. |
| **CHANGELOG.md** | What's been completed. Items move here from PLAN.md when finished. | When work is completed. |
| **HANDOFF.md** | How things work right now. Architecture, conventions, commands, gotchas. | When the system changes meaningfully. |
| **README.md** | Public-facing project overview. | When HANDOFF.md changes significantly. |

**Workflow:** ROADMAP tells you *what to work on next*. PLAN tells you *how to do it*. CHANGELOG proves *what's done*. HANDOFF explains *how it all works*.

---

## 1. Current State Summary

Gridiron Edge is a CLI-driven NFL analytics, modeling, and betting platform with a complete prediction-to-edge-to-bet-tracking pipeline.

### What's Working

| Area | Status | Key Assets |
|---|---|---|
| Data ingestion (nflverse) | ✅ Solid | Games, schedule, PBP, rosters |
| Data ingestion (weather) | ✅ Solid | OpenWeatherMap, idempotent |
| Data ingestion (odds) | ✅ Partial | DraftKings only; schema supports multi-book |
| Transform / clean layer | ✅ Solid | nflverse → canonical mappers |
| Dataset registry + I/O | ✅ Solid | Typed keys, Parquet/CSV, manifest validation |
| Feature engineering | ✅ Excellent (22 EPA + 107 total) | Elo, EPA (22 metrics), rest, travel, weather, venue, SoS, record, divisional, efficiency splits, situational football |
| Feature pipeline + validation | ✅ Solid | Dependency ordering, schema versioning (v4) |
| Elo ratings | ✅ Solid | Parameterized, fit/predict/table |
| Game prediction models | ✅ Solid (8 variants) | Logistic (4) + Tree (4: RF, XGB), variant factory pattern |
| Post-processing enrichment | ✅ Complete (W2) | Spread, total, projected scores, uncertainty bands, confidence tiers |
| Total points model | ✅ Complete (W2) | total_rf_v1 — MAE 10.27, competitive with Vegas (MAE 3.11 vs closing O/U) |
| Evaluation | ✅ Excellent | Brier, log loss, AUC, ECE, calibration, decomposition, confidence tiers, drift |
| Prediction archive | ✅ Solid | Append-only, dedup, backfill-aware, 8 enrichment columns (v2 schema) |
| Monte Carlo simulation | ✅ Advanced | Season + playoffs, numba-optimized |
| Market math (W3) | ✅ Complete | odds_math.py, kelly.py — pure functions, no data deps |
| Edge engine (W5) | ✅ Complete | edge.py, recommendations.py, clv.py — moneyline/spread/total edges, Kelly sizing, CLV analysis |
| Edge CLI (W5) | ✅ Complete | `gridiron edges report`, `gridiron edges clv` — ranked edge tables, CSV export |
| Bet tracking (W6) | ✅ Complete | ledger.py, bankroll.py, performance.py — append-only Parquet, decoupled bankroll |
| Betting CLI (W6) | ✅ Complete | 8 commands: log, settle, list, summary, balance, export, deposit, withdraw |
| Code quality | ✅ Excellent | Ruff lint+format, pyrefly types, three-tier test pyramid, pre-commit + pre-push hooks |
| Testing infrastructure | ✅ Complete | 500+ tests, auto-markers, shared fixtures, MiniRepoBuilder |
| W1: Quick Wins & Unblocking | ✅ Done | Unicode minus fix, game_id resolver, odds join validated |

### What's Missing

| Area | Status | Impact |
|---|---|---|
| Live prediction pipeline | ❌ Not started | Can't generate predictions for upcoming games with current-week features |
| Model ensemble | ❌ Not started | Using individual models only, no weighted combination |
| Multi-book odds ingestion | ❌ Not started | Can't compare books, can't shop lines |
| Player-level data & features | ❌ Not started | No player entities, game logs, or player features |
| Player prop models | ❌ Not started | No prop projections |
| API serving layer | ❌ Not started | Everything is CLI + file output |
| Frontend | ❌ Not started | Prototype exists (Claude Design) but not wired |
| Injury/news feed | ❌ Not started | No injury data, no impact modeling |
| Live game / real-time | ❌ Not started | No live state, odds, or win prob |

### Known Blockers

None. All previously known blockers (DK unicode minus bug, game_id resolver) were resolved in W1.

---

## 2. Vision / End State

Gridiron Edge becomes a **complete NFL decision-support platform** that:

1. ✅ **Forecasts games** with projected spreads, totals, win probabilities, and uncertainty bands
2. **Projects player props** with full outcome distributions
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
2. **Complete the prediction → edge loop first.** ✅ Achieved (M1). A model that can't be compared to the market can't help you bet.
3. **Enrich existing models before building new ones.** Adding spread/total/uncertainty to existing game models is higher-value than building player prop models from scratch.
4. **Feature engineering is continuous.** It runs in parallel with everything else. Every new data source or insight is a potential feature.
5. **Ship something usable.** Each workstream should produce an artifact you'd actually use on game day, not just a technical capability.
6. **Backend first, frontend later.** Get the analytics engine right. The frontend is a presentation layer that plugs in later.
7. **Files are fine until they're not.** Stay with Parquet/CSV until multi-user access, concurrency, or query complexity forces a database.

---

## 4. Workstreams

Each workstream is a **major capability area** that can be broken into smaller tasks in PLAN.md. They are ordered by recommended priority (highest-gain-first), not by timeline.

### Completed Workstreams

#### W1: Quick Wins & Unblocking — ✅ COMPLETE

Fixed DK unicode minus bug, built game_id resolver, validated end-to-end odds joins. See CHANGELOG.md for details.

#### W2: Richer Game Model Outputs — ✅ COMPLETE

Spread derivation (probit + per-model sigma calibration), total points model (total_rf_v1), projected scores, 90% uncertainty bands, confidence tiers. Isotonic recalibration evaluated and rejected (rf_v3 already well-calibrated). See CHANGELOG.md for details.

#### W3: Market Intelligence Foundation — ✅ COMPLETE

Pure-math market package: odds_math.py (conversions, no-vig, hold), kelly.py (fractional Kelly staking). Power devig via bisection, no scipy dependency. See CHANGELOG.md for details.

#### W5: Edge Engine — ✅ COMPLETE

Edge calculation (moneyline, spread, total), recommendations builder, CLV analysis, CLI commands (`gridiron edges report`, `gridiron edges clv`). 94 tests. See CHANGELOG.md for details.

#### W6: Portfolio & Bet Tracking — ✅ COMPLETE

Bet ledger, bankroll management (decoupled), performance analytics (record, ROI, CLV, EV, streaks), 8 CLI commands. 86 tests. Full round-trip validated. See CHANGELOG.md for details.

### Future Workstreams

#### W4: Player Data & First Prop Models

**Goal:** Establish the player-level data layer and build the first 2–3 player prop projection models.

**Why it matters:** Player props are a huge betting market and a major differentiator. The architecture patterns for features and models already exist — this extends them to a new domain.

**Key deliverables:**

**Phase A — Player data foundation:**
- Add player_game_logs to dataset registry (from nflfastR or equivalent)
- Build ingest/nflverse/player_stats.py — per-player per-game stat lines
- Define player entity fields: player_id, name, position, team_id
- Build player-level feature modules at features/player/:
  - rolling_stats.py — rolling averages (L3, L6, L12) for key stats
  - matchup.py — opponent defensive rank in stat category
  - usage.py — snap pct, target share, carry share
  - splits.py — home/away, indoor/outdoor, vs. winning teams

**Phase B — First prop models:**
- Choose first stat families: **QB rushing yards**, **QB passing yards**, **RB rushing yards**
- Build models/player_prop/ following the same PredictorRegistry + variant factory pattern
- Prop model outputs: mean, median, lo_90, hi_90, P(over given line), lean, confidence_tier
- Build prop-specific evaluation metrics (hit rate, calibration of P(over))
- Build prop archive (following archive.py pattern)

**Dependencies:** None (W1 done).
**Unlocks:** Prop edge calculations (W5 extension), prop line shopping (W7 extension), M3.
**Architecture notes:** Follow the exact same FeatureSpec + Feature protocol for player features. Follow the exact same _make_*_variant() factory pattern for prop model variants. The current architecture was built well enough that this is **extension, not refactoring**.

#### W4.5: Scenario Engine (What-If Analysis)

**Goal:** Build a conditional forecasting layer that answers: "If Player X is out, what happens to the game, the team, and other players' projections?"

**Why it matters:** This is one of the most differentiating capabilities a sports analytics platform can have. Most public models treat injuries as binary noise. A scenario engine that cascades a single roster change through team ratings, game forecasts, usage redistribution, and prop re-forecasts provides genuinely unique analytical leverage.

**Key deliverables:**

**Phase A — Player impact quantification:**
- Player WAR estimates (wins above replacement) — can start rule-based by position tier, evolve to data-driven
- On/off EPA splits per player (requires PBP + snap data)
- Positional importance weights (QB > Edge > WR1 > RB1 > etc.)
- Backup quality rating (replacement player's estimated contribution)

**Phase B — Team adjustment layer:**
- scenario/team_adjustment.py:
  - Input: base team rating + list of absent players
  - Output: adjusted offensive/defensive rating
  - Method: WAR-weighted rating adjustment
  - Cumulative injury impact score (sum of WAR-weighted absences)
  - Re-derive game forecast (spread, total, win prob) from adjusted ratings

**Phase C — Usage redistribution:**
- scenario/usage_redistribution.py:
  - Carry share redistribution (if RB1 out, how do RB2/RB3 split?)
  - Target tree redistribution (if WR1 out, where do targets go?)
  - Snap share reallocation
  - Historical with/without data to calibrate redistribution patterns
  - Position-specific redistribution templates (starting point before data-driven)

**Phase D — Conditional re-forecasting:**
- Re-run prop models with adjusted usage inputs
- Re-run edge calculations with adjusted game/prop forecasts
- Produce delta report: "what changed and by how much"

**Phase E — CLI interface:**
- `gridiron scenario --game SF@BAL --out CMC`
- `gridiron scenario --game SF@BAL --out CMC --out Bosa`
- `gridiron scenario --game SF@BAL --compare`

**Dependencies:** W2 ✅, W4 (player data + prop models).
**Unlocks:** Injury-aware edge detection, smarter bet timing, prop market opportunities from roster changes.
**Architecture notes:** Create src/gridiron_edge/scenario/ package. This layer sits *on top of* the game and prop models — it calls them, it doesn't replace them. See **FEATURES.md Domain 9** for the full feature inventory.

#### W7: Multi-Book Odds & Line Shopping

**Goal:** Ingest odds from multiple sportsbooks and build line-comparison tooling.

**Why it matters:** Betting at the best available price is one of the simplest, most reliable ways to improve long-term ROI. It requires no model improvement — just market awareness.

**Key deliverables:**
- Select odds data source (see Section 5.2)
- Build additional book ingest modules or unified API ingest
- store.py schema already supports sportsbook column, so no schema changes needed
- market/line_shopping.py:
  - best_price(market_id, side) → (book, line, price)
  - price_comparison_table(game_id, market_type) → DataFrame
  - detect_arbitrage(snapshots) → list[ArbOpportunity]
  - detect_middles(snapshots) → list[MiddleOpportunity]
- Line movement tracking:
  - movement(market_id, hours=24) → DataFrame
  - Steam move detection (Pinnacle-first movement as signal)
- CLI: `gridiron lines --week 12 --market spread` → cross-book comparison table

**Dependencies:** W3 ✅, **odds source decision** (Section 5.2).
**Unlocks:** Better bet execution, arbitrage opportunities, steam move awareness, M4.

#### W8: API Serving Layer

**Goal:** Expose analytics outputs through a REST API so a frontend (or other consumers) can access them.

**Why it matters:** This is the bridge between the CLI-driven analytics engine and any UI or external consumer.

**Key deliverables:**
- Create api/ package at src/gridiron_edge/api/
- Choose framework: **FastAPI** (recommended: lightweight, async, good docs, type-safe)
- Core endpoints:
  - GET /games?week=12 — list games with model forecasts and edges
  - GET /games/{game_id} — game detail with fair values, team comparison
  - GET /edges?week=12 — ranked edge table
  - GET /teams — power rankings
  - GET /props?week=12 — top prop edges (when W4 is ready)
  - GET /lines?week=12&market=spread — cross-book line comparison (when W7 is ready)
  - GET /portfolio/summary — bankroll + performance
  - POST /bets — log a bet
- Data source: read from Parquet/CSV files initially. Swap to DB later if needed.
- CORS configuration for frontend access

**Dependencies:** W2 ✅ + W5 ✅ (forecast + edge data to serve). Fully unblocked.
**Unlocks:** W9 (Frontend), mobile access, friend access, M5.
**Architecture notes:** This is the point where file-based storage may start to feel limiting. Start with FastAPI reading Parquet files. Add a database when the API needs it, not before.

#### W9: Frontend

**Goal:** Build a web UI that consumes the API and presents the analytics.

**Key deliverables:**
- Scaffold app (React/Vite or Next.js)
- Wire up to API endpoints
- Implement screens progressively as backend capabilities arrive:
  - Dashboard (games + edges)
  - Game Detail
  - Power Rankings
  - Player Props Explorer
  - Line Shopping
  - Bet Slip
  - Bankroll / Portfolio
  - News / Alerts
  - Settings
- The Claude Design prototype provides the visual spec.

**Dependencies:** W8 (API). Can start scaffolding earlier but real wiring requires API.
**Unlocks:** Full platform experience, friend access, M5.

#### W10: Real-Time & Live Game

**Goal:** Add live game state ingestion, live win probability, live odds comparison, and real-time alerts.

**Key deliverables:**
- Live game state ingestion (score, clock, possession, down/distance)
- Live win probability model (state-space model trained on historical PBP)
- Live odds ingestion (every 30–60 seconds)
- Live edge detection (model fair vs. live market)
- Hedge calculator (given open pregame position, suggest live hedge)
- WebSocket API for real-time frontend updates

**Dependencies:** W5 ✅, W7, W8.
**This is the most complex and least urgent workstream.** It should not be started until W7 and W8 are solid.

#### ~~W11: Live Prediction Pipeline~~ — NOT NEEDED

The live prediction pipeline already exists. `gridiron output predictions` calls
`predict_upcoming()` on all registered models and archives results to
`predictions_log.parquet`. `gridiron edges report` reads from that archive.
The game-day workflow is simply: `run-data-pipeline` → `output predictions` →
`ingest dk-odds` → `edges report`. No new code required. See HANDOFF.md
operational checklist for the full sequence.

#### W12: Model Ensemble (NEW)

**Goal:** Combine elo, logistic, random forest, and XGBoost predictions into a weighted ensemble for better overall accuracy.

**Why it matters:** Individual models have different strengths — Elo captures long-term team quality, logistic models are well-calibrated, tree models capture non-linear interactions. A well-tuned ensemble should beat any individual model on Brier score and AUC.

**Key deliverables:**
- Ensemble weighting strategy: evaluate Brier-score-weighted averaging, stacking (logistic meta-learner), and simple rank averaging
- Register ensemble predictor alongside individual models via PredictorRegistry
- Evaluation: compare ensemble vs rf_v3 (current best) on holdout data — must improve Brier by ≥0.002 to ship
- Wire ensemble into prediction pipeline + edge report as default model
- Update archive schema if needed (model_version = "ensemble_v1")

**Dependencies:** W2 ✅ (enriched predictions from all models).
**Unlocks:** Better predictions for all downstream consumers (edges, bets, portfolio).

### Cross-Cutting: Testing

**Testing runs in parallel with all workstreams.** Every new feature, module, or workstream deliverable should include corresponding unit tests. Integration and e2e tests are added as cross-module workflows are built. See HANDOFF.md for testing architecture details.

### Cross-Cutting: Feature Engineering

**Feature engineering is continuous.** Remaining backlog from FEATURES.md (Priorities 8–13: CPOE, pace, score differential, penalties, special teams, coaching) can be picked up alongside any workstream. Each new feature follows the established pattern: add to _agg_side() + EPA_COLS → auto-propagates.

---

## 5. Architecture Decisions & Open Items

### 5.1 File Storage vs. Database

**Current:** Parquet + CSV, file-based, CLI-driven.

**Recommendation:** Stay with files through W11/W12. Migrate to SQLite or PostgreSQL when:
- The API layer (W8) needs to serve concurrent requests
- The portfolio/bet ledger needs transactional integrity
- Query patterns require joins that are awkward in pandas
- Multi-user access is added

**Practical trigger:** If you find yourself writing complex pandas merge chains in the API layer to answer a single request, it's time for a database.

**Migration path:** The Parquet schemas map cleanly to SQL tables. The append-only patterns (archive, ledger, odds store) are natural INSERT operations. Use Alembic for migrations from the start.

### 5.2 Odds Data Source

**Status:** Deferred to backlog. Current DK-only ingest is sufficient through W11/W12.

**When this decision is needed:** Before W7 (Multi-Book Line Shopping) can begin.

**Options to evaluate:**

| Source | Coverage | Props? | Cost | Notes |
|---|---|---|---|---|
| The Odds API | ~15 books | Limited | Free tier: 500 req/mo; paid: $20–$80/mo | Easy to start, good docs, REST |
| Odds Jam | 20+ books | Yes | ~$40–$100/mo | Strong prop coverage |
| Pinnacle API | Pinnacle only | No (limited) | Free (with account) | Sharp book, useful as reference |
| Action Network | Major books | Yes | Varies | Requires investigation |
| DonBest | Comprehensive | Yes | Enterprise pricing | Likely overkill for V1 |
| Direct book APIs | Per-book | Varies | Free (with accounts) | High maintenance, per-book parsing |

**Recommendation:** Start with **The Odds API** for multi-book game markets. Add a prop-specific source (Odds Jam or Action Network) when W4 reaches the point of prop edge calculations.

### 5.3 Injury Data Source

**Status:** Not yet addressed.

**When needed:** When injury impact modeling becomes a priority (W4.5 Scenario Engine).

**Options:**
- ESPN API (free, has injury reports)
- nflverse injury data (if available in their data releases)
- Manual tracking (acceptable for V1 with a small number of games)
- Rotowire / Rotoworld feeds (may require scraping or API access)

### 5.4 Project Structure

The existing module structure is clean. Current and proposed packages:
- `src/gridiron\_edge/`
- `ingest/`            # ✅ existing
- `transform/`          # ✅ existing
- `datasets/`           # ✅ existing
- `features/`           # ✅ existing
- `team/`               # ✅ existing (11 feature modules)
- `player/`             # PLANNED (W4)
- `models/`             # ✅ existing
- `game\_prediction/`   # ✅ existing (logistic, tree, post\_process, total, pipeline)
- `player\_prop/`       # PLANNED (W4)
- `ratings/`            # ✅ existing
- `sim/`                # ✅ existing
- `evaluation/`         # ✅ existing
- `viz/`                # ✅ existing
- `market/`             # ✅ BUILT (W3, W5)
- `odds\_math.py`       #   conversions, no-vig, hold
- `kelly.py`            #   fractional Kelly staking
- `edge.py`             #   moneyline/spread/total edge calculation
- `recommendations.py`  #   edge report builder, ranking
- `clv.py`              #   closing line value analysis
- `betting/`            # ✅ BUILT (W6)
- `ledger.py`           #   append-only bet log
- `bankroll.py`         #   transaction log, balance tracking
- `performance.py`      #   record, ROI, CLV, EV, streaks
- `scenario/`           # PLANNED (W4.5)
- `api/`                # PLANNED (W8)
- `main.py`
- `routes/`
- `core/`               # ✅ existing
- `cli/`                # ✅ existing

---

## 6. Dependency Graph

COMPLETED                              REMAINING
─────────                              ─────────

W1 (Quick Wins) ✅
 │
 ├─────────────────────┬───────────────────────────┐
 ▼                     ▼                           ▼
W2 (Model Outputs) ✅  W3 (Market Math) ✅          W4 (Player Data
 │                     │                              & Props)
 │                     │                              │
 └─────────┬───────────┘                              │
           ▼                                          │
  W5 (Edge Engine) ✅ ◄──────── W4.5 (Scenario) ─────┘
           │
  ┌────────┼─────────┐
  ▼        ▼         ▼
 W6 ✅    W7        W8 (API)
(Bet    (Line        │
Track)  Shopping)    ▼
          │        W9 (Frontend)
          │
          └─────┬──────────────────┐
                ▼                  │
          W10 (Real-Time / Live)   │
                                   │
                                   │
  W11 (Live Predict) ─── unblocked, independent
  W12 (Ensemble) ──────── unblocked, independent


**Current position:** W1–W6 complete. Three independent paths forward:
- **W11 (Live Prediction Pipeline)** — unblocked, highest urgency (needed for Week 1)
- **W12 (Model Ensemble)** — unblocked, improves all downstream predictions
- **W8 (API)** — unblocked, enables frontend (M5)

W4 (Player Data & Props) and W7 (Multi-Book Odds) are also unblocked but lower priority per the prioritization principles.

---

## 7. What Success Looks Like (Milestones)

These are not deadlines. They are recognizable moments where the system becomes meaningfully more useful.

| Milestone | Description | Workstreams | Status |
|---|---|---|---|
| **M1: First actionable edge report** | Run `gridiron edges report --week 12` and get a ranked list of game edges with EV, Kelly stake, and best available book. You'd trust it enough to bet. | W1 + W2 + W3 + W5 | ✅ **ACHIEVED** |
| **M2: Know if the model makes money** | After a month of tracking bets, run `gridiron bet summary` and see your CLV, ROI, and record by confidence tier. | W6 | ✅ **ACHIEVED** |
| **M1.5: Weekly game-day predictions** | Run `gridiron output predictions` on Thursday, then `gridiron edges report` to see fresh edges. | Existing pipeline | ✅ **ACHIEVED** (infrastructure exists) |
| **M3: First prop edge** | Run `gridiron props --week 12` and get a prop edge table for QB rush, QB pass, and RB rush. | W4 + W5 | Planned |
| **M4: Shop across 3+ books** | Run `gridiron lines --week 12` and see a cross-book comparison with best prices highlighted. | W7 | Planned |
| **M5: Friends can use it** | Stand up a web UI that your friends can access. Dashboard, game detail, edges. | W8 + W9 | Planned |
| **M6: Live game day experience** | Real-time win prob, live edges, hedge suggestions during a game. | W10 | Planned |

**M1.5 is the next north star.** M1 proved the system works on historical data. M1.5 makes it work on game day.

---

## 8. Changelog for This Document

| Date | Change |
|---|---|
| 2026-06-03 | Champion/challenger model refactor complete. Temporal CV fix (TimeSeriesSplit). 3 unversioned champions replace 10 versioned variants. W11 removed (already exists). M1.5 achieved. XGBoost is auto-selected champion (Brier 0.218). |
| 2026-06-03 | **v2 refresh.** Updated §1 (current state), marked W1–W6 complete in §4, added W11 (Live Prediction Pipeline) and W12 (Model Ensemble), updated §5.4 project structure to match built modules, redrew §6 dependency graph, marked M1/M2 achieved in §7, added M1.5 milestone. Reconciled with PLAN.md numbering. |
| 2026-05-30 | Initial version — created from prototype review + gap analysis vs. existing gridiron_edge codebase. |


***
