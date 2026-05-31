# Gridiron Edge — ROADMAP

## Long-Term Strategic Direction

***

## 0. How This Document Fits

| Document                   | Purpose                                                                                                        | Updated When                                                         |
| -------------------------- | -------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------- |
| **ROADMAP.md** (this file) | High-level long-term direction. Where we're headed and why. Workstreams, dependencies, architecture decisions. | When strategic direction changes or a workstream is completed/added. |
| **PLAN.md**                | Short-term, nitty-gritty next steps. The current working checklist. Pick up from where you left off.           | Every working session.                                               |
| **CHANGELOG.md**           | What's been completed. Items move here from PLAN.md when finished.                                             | When work is completed.                                              |
| **HANDOFF.md**             | How things work right now. Architecture, conventions, commands, gotchas.                                       | When the system changes meaningfully.                                |
| **README.md**              | Public-facing project overview.                                                                                | When HANDOFF.md changes significantly.                               |

**Workflow:** ROADMAP tells you *what to work on next*. PLAN tells you *how to do it*. CHANGELOG proves *what's done*. HANDOFF explains *how it all works*.

***

## 1. Current State Summary

Gridiron Edge is a CLI-driven NFL analytics and modeling platform with a strong data-to-prediction pipeline.

### What's Working

| Area                          | Status                | Key Assets                                                                                                    |
| ----------------------------- | --------------------- | ------------------------------------------------------------------------------------------------------------- |
| Data ingestion (nflverse)     | ✅ Solid               | Games, schedule, PBP, rosters                                                                               |
| Data ingestion (weather)      | ✅ Solid               | OpenWeatherMap, idempotent                                                                                  |
| Data ingestion (odds)         | ✅ Partial             | DraftKings only; schema supports multi-book                                                                 |
| Transform / clean layer       | ✅ Solid               | nflverse → canonical mappers                                                                                |
| Dataset registry + I/O        | ✅ Solid               | Typed keys, Parquet/CSV, manifest validation                                                                |
| Feature engineering           | ✅ Solid (11 features) | Elo, EPA, rest, travel, weather, venue, SoS, record, divisional                                             |
| Feature pipeline + validation | ✅ Solid               | Dependency ordering, schema versioning (v4)                                                                 |
| Elo ratings                   | ✅ Solid               | Parameterized, fit/predict/table                                                                            |
| Game prediction models        | ✅ Solid (8 variants)  | Logistic (4) + Tree (4: RF, XGB), variant factory pattern                                                   |
| Evaluation                    | ✅ Excellent           | Brier, log loss, AUC, ECE, calibration, decomposition, confidence tiers, drift                              |
| Prediction archive            | ✅ Solid               | Append-only, dedup, backfill-aware                                                                          |
| Monte Carlo simulation        | ✅ Advanced            | Season + playoffs, numba-optimized                                                                          |
| Code quality                  | ✅ Excellent           | Ruff lint+format, pyrefly types, three-tier test pyramid, pre-commit + pre-push hooks, coverage tracking    |
| Testing infrastructure        | ✅ Complete            | 412 tests, 40% coverage, auto-markers, shared fixtures, MiniRepoBuilder, 0 deselected                       |
| W1: Quick Wins & Unblocking   | ✅ Done                | Unicode minus fix, game_id resolver, odds join validated                                                    |

### What's Missing

| Area                             | Status        | Impact                                             |
| -------------------------------- | ------------- | -------------------------------------------------- |
| Multi-book odds ingestion        | ❌ Not started | Can't compare books, can't shop lines              |
| Edge/EV engine (model vs market) | ❌ Not started | Can't translate predictions into betting value     |
| Spread/total/score projections   | ❌ Not started | Models output win prob only, not spreads or totals |
| Uncertainty bands on predictions | ❌ Not started | Point estimates only, no credible intervals        |
| Player-level data & features     | ❌ Not started | No player entities, game logs, or player features  |
| Player prop models               | ❌ Not started | No prop projections                                |
| Portfolio / bet tracking         | ❌ Not started | No bet ledger, bankroll, CLV, P/L                  |
| API serving layer                | ❌ Not started | Everything is CLI + file output                    |
| Frontend                         | ❌ Not started | Prototype exists (separate) but not wired          |
| Injury/news feed                 | ❌ Not started | No injury data, no impact modeling                 |
| Live game / real-time            | ❌ Not started | No live state, odds, or win prob                   |

### Known Blockers

| Blocker                                               | Impact                                                 | Effort                 |
| ----------------------------------------------------- | ------------------------------------------------------ | ---------------------- |
| DK unicode minus bug in `_norm_display_odds_american` | Blocks all DK odds downstream use                      | Trivial (one-line fix) |
| DK `game_id` resolver                                 | Blocks joining odds to games for edge/CLV calculations | Small-medium           |


***

## 2. Vision / End State

Gridiron Edge becomes a **complete NFL decision-support platform** that:

1. **Forecasts games** with projected spreads, totals, win probabilities, and uncertainty bands
2. **Projects player props** with full outcome distributions
3. **Compares model outputs against the betting market** to identify edges
4. **Shops lines** across multiple sportsbooks to find the best price
5. **Recommends stake sizing** using Kelly criterion and bankroll awareness
6. **Tracks betting performance** with CLV, ROI, P/L splits, and Kelly adherence
7. **Surfaces real-time information** (injuries, line moves, weather) that affects edge calculations
8. **Serves all of the above** through an API and eventually a frontend

The platform is built for personal use first (you + friends), with architecture that could support commercial access later.

**Sport scope:** NFL only for now. Architecture should be sport-agnostic where it costs less than 20% extra effort.

***

## 3. Prioritization Principles

Since this project is worked on in spare time with no fixed timelines, prioritization is driven by **value density** — what gives the most gain for the effort invested.

1. **Unblock before build.** If a trivial fix unblocks a major workstream, do it first.
2. **Complete the prediction → edge loop first.** A model that can't be compared to the market can't help you bet. The #1 priority is closing the gap between a win probability and an actionable edge.
3. **Enrich existing models before building new ones.** Adding spread/total/uncertainty to existing game models is higher-value than building player prop models from scratch.
4. **Feature engineering is continuous.** It runs in parallel with everything else. Every new data source or insight is a potential feature.
5. **Ship something usable.** Each workstream should produce an artifact you'd actually use on game day, not just a technical capability.
6. **Backend first, frontend later.** Get the analytics engine right. The frontend is a presentation layer that plugs in later.
7. **Files are fine until they're not.** Stay with Parquet/CSV until multi-user access, concurrency, or query complexity forces a database.

***

## 4. Workstreams

Each workstream is a **major capability area** that can be broken into smaller tasks in PLAN.md. They are ordered by recommended priority (highest-gain-first), not by timeline.

***

### W1: Quick Wins & Unblocking

**Goal:** Remove known blockers and harvest low-effort improvements that unblock downstream work.

**Why it matters:** Two trivial fixes (DK unicode bug + game\_id resolver) currently block the entire market intelligence workstream. Fixing them unlocks W3 and W5.

**Key deliverables:**

* [ ] Fix DK unicode minus bug (`_norm_display_odds_american`)
* [ ] Build DK `game_id` resolver (map DK event identifiers to canonical `YYYY_WW_AWAY_HOME`)
* [ ] Validate that DK odds can be joined to games and predictions after fixes
* [ ] Any other low-effort blockers surfaced in PLAN.md

**Dependencies:** None — this is the starting point.

**Unlocks:** W3 (Market Intelligence), W5 (Edge Engine), W6 (Portfolio).

***

### W2: Richer Game Model Outputs

**Goal:** Extend existing game prediction models to produce spread, total, projected scores, and uncertainty bands — not just win probability.

**Why it matters:** Every downstream consumer (edge engine, UI, portfolio) expects richer outputs than a bare win probability. This is the highest-value model work because it leverages everything already built.

**Key deliverables:**

* [ ] Add `model_spread` derivation from win probability (logistic inverse or calibration curve)
* [ ] Add `model_total` projection (requires offensive rating decomposition or separate model)
* [ ] Add `projected_home_score` and `projected_away_score`
* [ ] Add simulation-based uncertainty bands (`home_win_prob_lo`, `home_win_prob_hi`)
  * Option A: Bootstrap resampling of model predictions
  * Option B: Use Monte Carlo sim engine to generate per-game credible intervals
* [ ] Add `confidence_tier` classification to prediction outputs (High / Moderate / Low)
  * Already exists in `metrics.py` for evaluation; move to prediction time
* [ ] Extend prediction archive schema to include new fields
* [ ] Add `margin_std` (standard deviation of projected margin) from historical residuals or simulation

**Dependencies:** None — uses existing models and sim engine.

**Unlocks:** W5 (Edge Engine needs fair spread/total to compare against market), W8 (API needs these fields), eventual frontend.

**Architecture notes:**

* The prediction archive schema (`archive.py`) currently stores `away_win_prob` and `home_win_prob`. Extend it with the new columns. Bump schema version.
* Consider whether spread/total derivation belongs in the model itself or in a post-processing step. Recommendation: post-processing step that takes win\_prob as input — keeps models clean and composable.

***

### W3: Market Intelligence Foundation

**Goal:** Build the core market math utilities and make existing odds data usable for edge calculations.

**Why it matters:** This is the bridge between "analytics project" and "betting platform." Without it, model outputs are interesting but not actionable.

**Key deliverables:**

* [ ] Create `market/` package at `src/gridiron_edge/market/`
* [ ] `market/odds_math.py` — pure math utilities:
  * `american_to_decimal(odds) → float`
  * `american_to_implied_prob(odds) → float`
  * `decimal_to_american(dec) → int`
  * `no_vig(odds_a, odds_b) → (fair_prob_a, fair_prob_b)` (power method / Shin method)
  * `hold_pct(odds_a, odds_b) → float`
* [ ] `market/consensus.py` — aggregate across books:
  * `consensus_line(snapshots) → float` (median or mean across books)
  * `best_available(snapshots, side) → (book_id, line, price)`
* [ ] `market/kelly.py` — stake sizing:
  * `kelly(model_prob, american_odds, fraction=0.25) → float`
  * `kelly_stake(model_prob, american_odds, bankroll, fraction=0.25) → float`
* [ ] Unit tests for all math functions (edge cases: even odds, heavy favorites, etc.)

**Dependencies:** W1 (DK bug fix + game\_id resolver) for real data validation.

**Unlocks:** W5 (Edge Engine), W6 (Portfolio — Kelly sizing), W7 (Line Shopping).

**Architecture notes:**

* These are **pure functions** with no data dependencies. They can be built and tested independently of any data source.
* This is the single most portable module in the system — it will be reused everywhere.

***

### W4: Player Data & First Prop Models

**Goal:** Establish the player-level data layer and build the first 2–3 player prop projection models.

**Why it matters:** Player props are a huge betting market and a major differentiator. The architecture patterns for features and models already exist — this extends them to a new domain.

**Key deliverables:**

**Phase A — Player data foundation:**

* [ ] Add `player_game_logs` to dataset registry (from nflfastR or equivalent)
* [ ] Build `ingest/nflverse/player_stats.py` — per-player per-game stat lines
* [ ] Define player entity fields: `player_id`, `name`, `position`, `team_id`
* [ ] Build player-level feature modules at `features/player/`:
  * `rolling_stats.py` — rolling averages (L3, L6, L12) for key stats
  * `matchup.py` — opponent defensive rank in stat category
  * `usage.py` — snap pct, target share, carry share
  * `splits.py` — home/away, indoor/outdoor, vs. winning teams

**Phase B — First prop models:**

* [ ] Choose first stat families: **QB rushing yards**, **QB passing yards**, **RB rushing yards**
* [ ] Build `models/player_prop/` following the same `PredictorRegistry` + variant factory pattern
* [ ] Prop model outputs: mean, median, lo\_90, hi\_90, P(over given line), lean, confidence\_tier
* [ ] Build prop-specific evaluation metrics (hit rate, calibration of P(over))
* [ ] Build prop archive (following `archive.py` pattern)

**Dependencies:** W1 (for data), feature pipeline patterns (already exist).

**Unlocks:** Prop edge calculations (W5 extension), prop line shopping (W7 extension), eventual prop UI.

**Architecture notes:**

* Follow the exact same `FeatureSpec` + `Feature` protocol for player features.
* Follow the exact same `_make_*_variant()` factory pattern for prop model variants.
* The current architecture was built well enough that this is **extension, not refactoring**.

***

### W4.5: Scenario Engine (What-If Analysis)

**Goal:** Build a conditional forecasting layer that answers: "If Player X is out, what happens to the game, the team, and other players' projections?"

**Why it matters:** This is one of the most differentiating capabilities a sports analytics platform can have. Most public models treat injuries as binary noise. A scenario engine that cascades a single roster change through team ratings, game forecasts, usage redistribution, and prop re-forecasts provides genuinely unique analytical leverage.

**Key deliverables:**

**Phase A -- Player impact quantification:**
- [ ] Player WAR estimates (wins above replacement) -- can start rule-based by position tier, evolve to data-driven
- [ ] On/off EPA splits per player (requires PBP + snap data)
- [ ] Positional importance weights (QB > Edge > WR1 > RB1 > etc.)
- [ ] Backup quality rating (replacement player's estimated contribution)

**Phase B -- Team adjustment layer:**
- [ ] `scenario/team_adjustment.py`:
     - Input: base team rating + list of absent players
     - Output: adjusted offensive/defensive rating
     - Method: WAR-weighted rating adjustment
- [ ] Cumulative injury impact score (sum of WAR-weighted absences)
- [ ] Re-derive game forecast (spread, total, win prob) from adjusted ratings

**Phase C -- Usage redistribution:**
- [ ] `scenario/usage_redistribution.py`:
     - Carry share redistribution (if RB1 out, how do RB2/RB3 split?)
     - Target tree redistribution (if WR1 out, where do targets go?)
     - Snap share reallocation
- [ ] Historical with/without data to calibrate redistribution patterns
- [ ] Position-specific redistribution templates (starting point before data-driven)

**Phase D -- Conditional re-forecasting:**
- [ ] Re-run prop models with adjusted usage inputs
- [ ] Re-run edge calculations with adjusted game/prop forecasts
- [ ] Produce delta report: "what changed and by how much"

**Phase E -- CLI interface:**
- [ ] `gridiron-edge scenario --game SF@BAL --out CMC`
     - Shows: adjusted win prob, adjusted spread, adjusted player props, new edges
- [ ] `gridiron-edge scenario --game SF@BAL --out CMC --out Bosa`
     - Multiple players out simultaneously
- [ ] `gridiron-edge scenario --game SF@BAL --compare`
     - Side-by-side: full strength vs current injury report

**Dependencies:** W2 (richer model outputs to re-derive), W4 (player data + prop models to re-forecast). Can start Phase A (impact quantification) in parallel with W4.

**Unlocks:** Injury-aware edge detection, smarter bet timing (bet before/after injury news), prop market opportunities from roster changes, narrative explanations for UI.

**Architecture notes:**
- Create `src/gridiron_edge/scenario/` package.
- This layer sits *on top of* the game and prop models -- it calls them, it doesn't replace them.
- Phase A can start with simple rule-based position tier tables. Data-driven WAR and on/off splits come later as player data matures.
- See **FEATURES.md Domain 9** for the full feature inventory that powers this workstream.

***

### W5: Edge Engine

**Goal:** Combine model forecasts with market prices to identify actionable edges, size bets, and rank opportunities.

**Why it matters:** This is where the system goes from "interesting analytics" to "helps you make money." It's the decision layer.

**Key deliverables:**

* [ ] `market/edge.py`:
  * `edge(model_prob, market_implied_prob) → float` (EV percentage)
  * `edge_spread(model_spread, market_spread) → float` (point difference)
  * `edge_total(model_total, market_total) → float`
* [ ] `market/recommendations.py`:
  * For each game: recommended side, EV, confidence, Kelly stake, best book
  * Ranked edge table (like the prototype's "Model edges" section)
* [ ] CLV calculation:
  * Join `predictions_log.parquet` to `dk_odds_log.parquet` at closing time
  * `clv(bet_line, closing_line) → float`
  * This leverages two of the strongest existing assets (archive + odds store)
* [ ] Weekly edge report CLI command:
  * `gridiron-edge edges --week 12` → prints ranked edges with EV, Kelly, best book
  * **This is the first artifact that makes the system usable on Sundays**

**Dependencies:** W1 (odds joinable), W2 (spread/total projections), W3 (market math).

**Unlocks:** W6 (Portfolio — bet tracking with edge context), W7 (Line Shopping), eventual frontend dashboard.

***

### W6: Portfolio & Bet Tracking

**Goal:** Build a bet ledger and bankroll tracking system to measure long-term performance.

**Why it matters:** Without tracking, you can't know if the system works. CLV, ROI, and Kelly adherence are the metrics that prove (or disprove) edge.

**Key deliverables:**

* [ ] Create `betting/` package at `src/gridiron_edge/betting/`
* [ ] `betting/ledger.py` — append-only bet log (following `archive.py` pattern):
  * Fields: bet\_id, game\_id, market\_type, side, line\_at\_bet, price\_at\_bet, stake, book, placed\_at, model\_prob\_at\_bet, model\_ev\_at\_bet, confidence\_tier
  * Settlement: status (open/won/lost/push), settled\_at, pnl, closing\_line, clv\_pct
* [ ] `betting/bankroll.py` — bankroll state:
  * Current balance
  * Balance history (time series)
  * Deposits / withdrawals log
* [ ] `betting/performance.py` — analytics:
  * Record (W-L-P) overall and by market type
  * ROI overall and by splits (market type, confidence tier, book, week)
  * CLV distribution
  * Kelly adherence (% of bets within ±20% of suggested stake)
* [ ] CLI commands:
  * `gridiron-edge bet log <side> <line> <price> <stake> <book>` — record a bet
  * `gridiron-edge bet settle <bet_id> <result>` — settle a bet
  * `gridiron-edge bet summary` — print performance dashboard
  * `gridiron-edge bet export --format csv` — export for tax/records

**Dependencies:** W1 (joinable odds for CLV), W3 (Kelly math), W5 (edge context on bets).

**Unlocks:** Performance evaluation loop (are the models actually making money?), eventual bankroll UI.

**Architecture notes:**

* Storage: Parquet is fine for V1. The append-only log pattern from `archive.py` works well here.
* When this moves to multi-user or API-served, migrate to SQLite or PostgreSQL. The ledger schema maps directly to a SQL table.
* **This is where the file-vs-database decision will eventually matter most.** For now, Parquet + CLI is sufficient.

***

### W7: Multi-Book Odds & Line Shopping

**Goal:** Ingest odds from multiple sportsbooks and build line-comparison tooling.

**Why it matters:** Betting at the best available price is one of the simplest, most reliable ways to improve long-term ROI. It requires no model improvement — just market awareness.

**Key deliverables:**

* [ ] Select odds data source (deferred discussion — see Section 5)
* [ ] Build additional book ingest modules or unified API ingest
  * `store.py` schema already supports `sportsbook` column, so no schema changes needed
* [ ] `market/line_shopping.py`:
  * `best_price(market_id, side) → (book, line, price)`
  * `price_comparison_table(game_id, market_type) → DataFrame`
  * `detect_arbitrage(snapshots) → list[ArbOpportunity]`
  * `detect_middles(snapshots) → list[MiddleOpportunity]`
* [ ] Line movement tracking:
  * `movement(market_id, hours=24) → DataFrame` (time series of line changes)
  * Steam move detection (Pinnacle-first movement as signal)
* [ ] CLI: `gridiron-edge lines --week 12 --market spread` → cross-book comparison table

**Dependencies:** W1 (DK fix), W3 (market math), odds source decision.

**Unlocks:** Better bet execution, arbitrage opportunities, steam move awareness.

**Architecture notes:**

* The odds source decision is the biggest dependency here. Options discussed in Section 5.
* Until the source is chosen, W3 (pure math) and W5 (edge using DK-only data) can proceed independently.

***

### W8: API Serving Layer

**Goal:** Expose analytics outputs through a REST API so a frontend (or other consumers) can access them.

**Why it matters:** This is the bridge between the CLI-driven analytics engine and any UI or external consumer.

**Key deliverables:**

* [ ] Create `api/` package at `src/gridiron_edge/api/`
* [ ] Choose framework: **FastAPI** (recommended: lightweight, async, good docs, type-safe)
* [ ] Core endpoints:
  * `GET /games?week=12` — list games with model forecasts and edges
  * `GET /games/{game_id}` — game detail with fair values, team comparison
  * `GET /edges?week=12` — ranked edge table
  * `GET /teams` — power rankings
  * `GET /props?week=12` — top prop edges (when W4 is ready)
  * `GET /lines?week=12&market=spread` — cross-book line comparison (when W7 is ready)
  * `GET /portfolio/summary` — bankroll + performance (when W6 is ready)
  * `POST /bets` — log a bet (when W6 is ready)
* [ ] Data source: read from Parquet/CSV files initially. Swap to DB later if needed.
* [ ] CORS configuration for frontend access

**Dependencies:** W2 + W5 (need forecast + edge data to serve). Can serve progressively — start with `/games` and `/edges`.

**Unlocks:** W9 (Frontend), mobile access, friend access, future commercial access.

**Architecture notes:**

* **This is the point where file-based storage may start to feel limiting.** If query patterns become complex (joins, filters, aggregations), consider migrating to SQLite or PostgreSQL.
* Recommendation: Start with FastAPI reading Parquet files. Add a database when the API needs it, not before.

***

### W9: Frontend

**Goal:** Build a React-based web UI that consumes the API and presents the analytics.

**Key deliverables:**

* [ ] Scaffold React app (Vite or Next.js)
* [ ] Wire up to API endpoints
* [ ] Implement screens progressively as backend capabilities arrive:
  1. Dashboard (games + edges)
  2. Game Detail
  3. Power Rankings
  4. Player Props Explorer
  5. Line Shopping
  6. Bet Slip
  7. Bankroll / Portfolio
  8. News / Alerts
  9. Settings
* [ ] The Claude Design prototype provides the complete visual spec. No design work needed.

**Dependencies:** W8 (API). Can start scaffolding earlier but real wiring requires API.

**Unlocks:** Full platform experience, friend access, commercial potential.

***

### W10: Real-Time & Live Game

**Goal:** Add live game state ingestion, live win probability, live odds comparison, and real-time alerts.

**Key deliverables:**

* [ ] Live game state ingestion (score, clock, possession, down/distance)
* [ ] Live win probability model (state-space model trained on historical PBP)
* [ ] Live odds ingestion (every 30–60 seconds)
* [ ] Live edge detection (model fair vs. live market)
* [ ] Hedge calculator (given open pregame position, suggest live hedge)
* [ ] WebSocket API for real-time frontend updates

**Dependencies:** W5, W7, W8 (the full stack must be working for live to make sense).

**This is the most complex and least urgent workstream.** It should not be started until W1–W7 are solid.

***

#### Cross-Cutting: Testing

**Testing (W0 in PLAN.md) runs in parallel with all workstreams.** Every new feature, module, or workstream deliverable should include corresponding unit tests. Integration and e2e tests are added as cross-module workflows are built. See HANDOFF.md for testing architecture details.

***

## 5. Architecture Decisions & Open Items

### 5.1 File Storage vs. Database

**Current:** Parquet + CSV, file-based, CLI-driven.

**Recommendation:** Stay with files through W1–W6. Migrate to SQLite or PostgreSQL when:

* The API layer (W8) needs to serve concurrent requests
* The portfolio/bet ledger needs transactional integrity
* Query patterns require joins that are awkward in pandas
* Multi-user access is added

**Practical trigger:** If you find yourself writing complex pandas merge chains in the API layer to answer a single request, it's time for a database.

**Migration path:** The Parquet schemas map cleanly to SQL tables. The append-only patterns (archive, ledger, odds store) are natural `INSERT` operations. Use Alembic for migrations from the start.

### 5.2 Odds Data Source

**Status:** Deferred to backlog. Current DK-only ingest is sufficient for W1–W5.

**When this decision is needed:** Before W7 (Multi-Book Line Shopping) can begin.

**Options to evaluate:**

| Source           | Coverage      | Props?       | Cost                                    | Notes                              |
| ---------------- | ------------- | ------------ | --------------------------------------- | ---------------------------------- |
| The Odds API     | \~15 books    | Limited      | Free tier: 500 req/mo; paid: $20–$80/mo | Easy to start, good docs, REST     |
| Odds Jam         | 20+ books     | Yes          | \~$40–$100/mo                           | Strong prop coverage               |
| Pinnacle API     | Pinnacle only | No (limited) | Free (with account)                     | Sharp book, useful as reference    |
| Action Network   | Major books   | Yes          | Varies                                  | Requires investigation             |
| DonBest          | Comprehensive | Yes          | Enterprise pricing                      | Likely overkill for V1             |
| Direct book APIs | Per-book      | Varies       | Free (with accounts)                    | High maintenance, per-book parsing |

**Recommendation for later discussion:** Start with **The Odds API** for multi-book game markets. Add a prop-specific source (Odds Jam or Action Network) when W4 reaches the point of prop edge calculations.

### 5.3 Injury Data Source

**Status:** Not yet addressed.

**When needed:** When injury impact modeling becomes a priority (likely alongside W2 or W4).

**Options:**

* ESPN API (free, has injury reports)
* nflverse injury data (if available in their data releases)
* Manual tracking (acceptable for V1 with a small number of games)
* Rotowire / Rotoworld feeds (may require scraping or API access)

### 5.4 Project Structure (Proposed Extensions)

The existing module structure is clean. Proposed new packages:

```
src/gridiron_edge/
  ingest/           # existing
  transform/        # existing
  datasets/         # existing
  features/         # existing
    team/           # existing
    player/         # NEW (W4)
  models/           # existing
    game_prediction/# existing
    player_prop/    # NEW (W4)
  ratings/          # existing
  sim/              # existing
  evaluation/       # existing
  viz/              # existing
  market/           # NEW (W3, W5, W7)
    odds_math.py
    consensus.py
    kelly.py
    edge.py
    recommendations.py
    line_shopping.py
  betting/          # NEW (W6)
    ledger.py
    bankroll.py
    performance.py
  api/              # NEW (W8)
    main.py
    routes/
  core/             # existing
  cli/              # existing
```

***

## 6. Dependency Graph

```
W1 (Quick Wins / Unblock)
 │
 ├──────────────────────────────────────────┬────────────────────────┐
 ▼                                          ▼                        ▼
W2 (Richer Model Outputs)            W3 (Market Math)          W4 (Player Data
 │                                     │                         & Props)
 │                                     │                          │
 └────────────┬────────────────────────┘                          │
              ▼                                                   │
         W5 (Edge Engine) <────────── W4.5 (Scenario Engine) ─────┘
              │
       ┌──────┼──────┐
       ▼      ▼      ▼
  W6 (Portfolio) W7 (Line  W8 (API)
       │     Shopping)   │
       │        │       │
       │        │       ▼
       │        │   W9 (Frontend)
       │        │
       └───┬────┘
           ▼
     W10 (Real-Time / Live)
```

**Key insight:** W1, W2, W3, and W4 can all be worked on in parallel (or interleaved). W5 is where they converge. Everything after W5 builds on a solid foundation.

**Feature engineering (Phase 20e) runs continuously in parallel with all workstreams.**

***

## 7. What Success Looks Like (Milestones)

These are not deadlines. They are recognizable moments where the system becomes meaningfully more useful.

| Milestone                             | Description                                                                                                                                          | Workstreams       |
| ------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------- |
| **M1: First actionable edge report**  | Run `gridiron-edge edges --week 12` and get a ranked list of game edges with EV, Kelly stake, and best available book. You'd trust it enough to bet. | W1 + W2 + W3 + W5 |
| **M2: Know if the model makes money** | After a month of tracking bets, run `gridiron-edge bet summary` and see your CLV, ROI, and record by confidence tier.                                | W6                |
| **M3: First prop edge**               | Run `gridiron-edge props --week 12` and get a prop edge table for QB rush, QB pass, and RB rush.                                                     | W4 + W5           |
| **M4: Shop across 3+ books**          | Run `gridiron-edge lines --week 12` and see a cross-book comparison with best prices highlighted.                                                    | W7                |
| **M5: Friends can use it**            | Stand up a web UI that your friends can access. Dashboard, game detail, edges.                                                                       | W8 + W9           |
| **M6: Live game day experience**      | Real-time win prob, live edges, hedge suggestions during a game.                                                                                     | W10               |

**M1 is the north star.** Everything else is valuable, but M1 is the moment the platform becomes a real tool.

***

## 8. Changelog for This Document

| Date       | Change                                                                                               |
| ---------- | ---------------------------------------------------------------------------------------------------- |
| 2026-05-30 | Initial version — created from prototype review + gap analysis vs. existing gridiron\_edge codebase. |

***
