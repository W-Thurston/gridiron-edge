# Gridiron Edge — Plan

What is being built next and why. See [HANDOFF.md](HANDOFF.md) for how everything works today and [CHANGELOG.md](CHANGELOG.md) for what has been completed.

**Status key:** `[x]` done · `[ ]` not started · `[~]` in progress

---

## Vision

Gridiron Edge is a long-term **football intelligence and betting research platform** — not just a prediction model.

| Layer | Description | Status |
|-------|-------------|--------|
| Data Platform | Ingest and store football reality | ✅ |
| Football State Representation | Represent what teams actually are | ✅ Phase 19 |
| Prediction Engine | What is likely to happen | ✅ Phase 20a–d |
| Model Evaluation | Which model should I use and why | ✅ Phase 20b |
| Model Reporting | Full characterisation of chosen model | ✅ Phase 20c |
| Tree-based Models | Non-linear signal capture (RF + XGBoost) | ✅ Phase 20d |
| Feature Engineering | Interactions, transforms, domain features | 🔜 Phase 20e |
| Model Variant Infrastructure | Programmatic model registration | 🔜 Phase 20f |
| Market Intelligence | What does Vegas think | 🔜 Phase 21 |
| Decision Engine | What should actually be bet | 🔜 Future |
| Explainability & Insights | Why does the model think this | 🔜 Future |

**Core philosophy:** Betting edge is discovered through measurement, infrastructure, and systematic disagreement with the market — not through chasing more complex models. Understand existing models deeply before building more complex ones.

---

## Architectural debt

Items that must be resolved before or during the phases they block.

| Item | Blocking | Fix |
|------|----------|-----|
| DK odds unicode minus bug | Phase 21 | Move `str.replace("\u2212", "-")` before `int()` cast in `_norm_display_odds_american` in `ingest/odds/draftkings.py` |
| Odds `game_id` resolver | Phase 21 | Map DK internal event IDs to canonical `YYYY_WW_AWAY_HOME` using team names + dates |
| Missing stadium coordinates | Ongoing | 12 new/renamed 2026-2027 stadia not in `NFL_stadium_reference.csv`; weather ingest skips affected games |
| Test coverage: `backfill.py` + `tune.py` | Ongoing | `archive.py`, `metrics.py`, `manifest.py` covered; backfill and tune still need tests |
| `off-season current_nfl_season()` | Ongoing | Returns `year - 1` when `month < 6`; must pass `--season` flags explicitly during off-season |

---

## Phase 20e — Feature engineering (in progress) 🔜

**Goal:** Add the domain features that most improve model discrimination beyond Elo + EPA.

**Category A — Venue and game context** (high signal, low implementation cost):
- [ ] Dome/outdoor flag (`IS_DOME`) — already in schema v3, confirm wired end-to-end
- [ ] Neutral site flag (`IS_NEUTRAL_SITE`) — already in schema v3
- [ ] Altitude (`ALTITUDE`) — stadium reference has it; confirm feature uses it
- [ ] Weather effects — integrate OWM data into prediction features (ingest exists; feature does not)
- [ ] Stadium attendance capacity — venue size as a proxy for home crowd noise

**Category B — Schedule stress** (medium signal, medium cost):
- [ ] Days rest (`DAYS_REST`) — already in schema v3, confirm wired end-to-end
- [ ] Short week flag (`SHORT_WEEK`) — already in schema v3
- [ ] Post-bye flag (`POST_BYE`) — already in schema v3
- [ ] Travel distance + timezone shift — metrics/travel already computes; confirm wired into features

**Category C — Team intelligence** (high signal, highest cost):
- [ ] QB Elo / QBR / passer rating as team-level features — requires QB-level PBP aggregation and starter tracking
- [ ] DVOA, CPOE, RYOE — advanced EPA derivatives, requires additional PBP work
- [ ] Win streak / loss streak / record (`WIN_STREAK`, `LOSS_STREAK`, `WIN_PCT`) — already in schema v3

**Acceptance criteria:**

```bash
uv run gridiron features model-inputs --all-years
uv run gridiron models train random_forest_v2
uv run gridiron models train xgboost_v2
uv run gridiron evaluate backfill --model-version random_forest_v2
uv run gridiron evaluate backfill --model-version xgboost_v2
uv run gridiron evaluate select-model
uv run gridiron evaluate report
uv run ruff check src/ --fix && uvx pyrefly check && uv run pytest
```

---

## Phase 20f — Model variant infrastructure 🔜

**Goal:** Eliminate per-variant class boilerplate in `tree.py` and `logistic.py`. Currently adding a model variant requires a new class body, re-export, `_make_*` function, and registration test. With frequent feature iteration this is unsustainable.

**Approach — programmatic factory (Option C):** `_register_rf_variant()` and `_register_xgb_variant()` factory functions produce and register a fully-typed class at module load time given just `name`, `description`, `feature_fn`, `feature_names`. One call per variant replaces ~35 lines of boilerplate.

**Why not Hydra + MLflow (Option A)?** Config-driven registration is the correct long-term direction but earns its keep only when tracking real bets and needing full experiment lineage. That is Phase 22+. Option C is the pragmatic bridge: same external interface, dramatically lower authoring cost, no new dependencies.

**Scope:**
- [ ] `_make_tree_variant(name, description, feature_fn, feature_names, model_type)` — produces + registers RF or XGBoost class
- [ ] `_make_logistic_variant(name, description, feature_fn, feature_names, elasticnet)` — produces + registers logistic class
- [ ] Replace all 8 hand-written class bodies (logistic v1-v4, RF v1-v2, XGB v1-v2) with factory calls
- [ ] `predictor.py` shim updated to import from factories
- [ ] `test_tree_models.py` updated to use registry lookups

**What does not change:** model version strings, `PredictorRegistry` interface, `ArtifactStore`, CLI commands, evaluation infrastructure.

**Acceptance criteria:**

```bash
uv run gridiron models list
uv run gridiron models train random_forest_v2
uv run gridiron evaluate report
uv run ruff check src/ --fix && uvx pyrefly check && uv run pytest
```

---

## Phase 21 — Market intelligence layer 🔜

**Prerequisite:** DK unicode minus bug fixed + odds `game_id` resolver built.

**Goal:** Systematically compare model predictions against the market line.

- [ ] Odds `game_id` resolver — map DK internal event IDs to canonical `YYYY_WW_AWAY_HOME`
- [ ] Historical odds database — spread, moneyline, totals; opening + closing lines
- [ ] Multi-sportsbook ingest — FanDuel, BetMGM alongside DraftKings
- [ ] Implied probability extraction with de-vig (power/Shin method)
- [ ] Edge calculation: `model_prob - implied_prob` per game
- [ ] Market disagreement analysis: where does the model consistently differ from consensus?
- [ ] CLI: `gridiron market edge --year --week`

---

## Phase 22 — Betting tracker + analytics 🔜

**Goal:** Close the loop from prediction to outcome tracking.

- [ ] Bet log CLI — record bets with stake, odds, market, sportsbook
- [ ] P&L tracking — ROI by model, season, market type, confidence tier
- [ ] `analytics/matchup_reports.py` — per-game matchup breakdown report
- [ ] `analytics/team_insights.py` — season-level team analytics and trends
- [ ] CLI: `gridiron bets log` + `gridiron bets summary`

---

## Backlog / Future

Items with long-term value but no near-term dependency:

**Simulation:**
- Per-stage playoff Monte Carlo — simulate each round independently; inject known results as postseason progresses
- Skip full-season sim after week 18 — use actual standings once regular season is complete
- Dynamic schedule updates — handle mid-season game relocations without manual file edits

**Infrastructure:**
- Config-driven model registration (Hydra + MLflow / Option A) — replace Option C once real bets are tracked and experiment lineage has P&L value. Prerequisite: Phase 22 operational.
- `datasets/registry.py` self-registration — replace flat dict with auto-discovery when dataset count grows past ~20
- Drive/possession simulation — expand Monte Carlo toward play-level outcome distributions
- Injury + roster intelligence — injury-adjusted team strength, player availability impacts
- Ensemble systems — stack Elo + EPA model + ML model with uncertainty estimates
- Public dashboards / automated reports
- Live game intelligence — real-time win probability updates (major architecture change)