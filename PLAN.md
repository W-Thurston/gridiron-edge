# Gridiron Edge — Plan

See [HANDOFF.md](HANDOFF.md) to run the project today.
See [CHANGELOG.md](CHANGELOG.md) for completed phase history and decisions.

**Status key:** `[x]` done · `[ ]` not started · `[~]` in progress

---

## Vision

Gridiron Edge is a long-term **football intelligence and betting research platform**.

| Layer | Status |
|-------|--------|
| Data Platform | ✅ Operational |
| Football State Representation | ✅ Phase 19 |
| Prediction Engine | ✅ Phase 20 — `random_forest_v2` production model |
| Model Evaluation + Reporting | ✅ Phase 20b/20c |
| Tree-based Models | ✅ Phase 20d |
| Feature Engineering | 🔜 Phase 20e (in progress) |
| Model Variant Infrastructure | ✅ Phase 20f |
| Market Intelligence | 🔜 Phase 21 |
| Decision Engine | 🔜 Future |
| Explainability & Insights | 🔜 Future |

**Core philosophy:** Betting edge is discovered through measurement, infrastructure,
and systematic disagreement with the market — not through chasing more complex models.
Understand existing models deeply before building more.

---

## Architecture

| Layer | Module |
|-------|--------|
| Game + schedule ingest | `gridiron_edge.ingest.nflverse` |
| PBP ingest | `gridiron_edge.ingest.nflverse.pbp` |
| Weather ingest | `gridiron_edge.ingest.pfr.collector_impl` (OpenWeatherMap) |
| Odds ingest + ledger | `gridiron_edge.ingest.odds` (DraftKings → Parquet) |
| Transform | `gridiron_edge.transform.clean` |
| Modeling features | `gridiron_edge.features.pipeline` (schema v3) |
| Elo ratings | `gridiron_edge.ratings.elo` |
| Season simulation | `gridiron_edge.sim` |
| Visualisation | `gridiron_edge.viz` |
| Evaluation | `gridiron_edge.evaluation` |
| Models | `gridiron_edge.models` (Predictor + Trainable protocols, ArtifactStore) |
| CLI | `uv run gridiron` (split across `cli/` sub-modules) |

---

## Current model rankings (holdout 2023–2026)

| model | Brier | ECE | AUC | auto-selected |
|---|---|---|---|---|
| random_forest_v2 | **0.21078** | 0.02108 | **0.71820** | ← production |
| random_forest_v1 | 0.21503 | 0.02836 | 0.70527 | |
| xgboost_v2 | 0.21865 | 0.02059 | 0.69200 | |
| xgboost_v1 | 0.21857 | 0.02093 | 0.69278 | |
| logistic_v3 | 0.22102 | **0.01606** | 0.68241 | |

---

## Phase 20e — Feature engineering 🔜 (in progress)

**Next batch (quick wins from existing data):**

- [ ] **Season stage** — early (wk 1–4), mid, late (wk 14–18), playoffs; from `WEEK_NUM`
- [ ] **Strength of schedule to date** — avg Elo of opponents faced this season
- [ ] **Strength of victory** — avg Elo of opponents beaten this season
- [ ] **Win/loss record** — wins and games played to date
- [ ] **Primetime flag** — MNF/SNF/TNF; from `GAMETIME` + `GAME_DAY_OF_WEEK`
- [ ] **Turnover differential (rolling)** — `TURNOVERS_WINNER/LOSER` already in games CSV

**Completed:**
- [x] Rest/schedule stress — `DAYS_REST`, `SHORT_WEEK`, `POST_BYE`
- [x] Weather — `IS_DOME`, `WIND_SPEED_MPH`, `TEMP_F`, `PRECIP_FLAG`
- [x] Travel — `KM_TRAVELED`, `TZ_SHIFT`, `ALTITUDE`, `IS_NEUTRAL_SITE`
- [x] Divisional flag — `IS_DIV_GAME`
- [x] Franchise HFA coefficient — `TEAM_A/B_FRANCHISE_HFA`

**After quick wins — feature brainstorm session (three tiers):**

1. **From data we already have** — all items above plus any others surfaced by
   evaluation analysis (season drift, early-season instability, etc.)
2. **Calculated from existing sources** — DVOA-style adjusted metrics, QB Elo,
   EPA extensions (CPOE, RYOE, weighted EPA)
3. **Requiring new data sources** — Next Gen Stats, line movement/CLV, injury
   reports, Vegas consensus

**Acceptance criteria (per batch):**
```bash
uv run gridiron features model-inputs --all-years
uv run gridiron models train random_forest_v2
uv run gridiron evaluate backfill --model-version random_forest_v2
uv run gridiron evaluate report
uv run ruff check src/ --fix && uvx pyrefly check && uv run pytest
```

---

## Phase 21 — Market intelligence layer 🔜

**Goal:** Systematically compare model predictions against the market.
**Prerequisite:** Odds `game_id` resolver (see architectural debt).

- [ ] Odds `game_id` resolver — map DK internal event IDs to canonical `YYYY_WW_AWAY_HOME`
- [ ] Historical odds database — spread, moneyline, totals, opening + closing lines
- [ ] Multi-sportsbook ingest — FanDuel, BetMGM alongside DraftKings
- [ ] Implied probability extraction with de-vig (power/Shin method)
- [ ] Edge calculation: `model_prob − implied_prob` per game
- [ ] Market disagreement analysis: where does model consistently differ from consensus?
- [ ] CLI: `gridiron market edge --year --week`

---

## Phase 22 — Betting tracker + analytics 🔜

**Goal:** Close the loop from prediction to outcome tracking.

- [ ] Bet log CLI — record bets with stake, odds, market, sportsbook
- [ ] P&L tracking — ROI by model, season, market type, confidence tier
- [ ] Matchup reports — per-game breakdown
- [ ] Team insights — season-level trends
- [ ] CLI: `gridiron bets log` + `gridiron bets summary`

---

## Architectural debt

| Item | Blocking at | Notes |
|------|------------|-------|
| Odds `game_id` resolver | Phase 21 | DK uses internal event IDs; join will fail silently without this |
| Test coverage for `backfill.py` + `tune.py` | Ongoing | `archive.py`, `metrics.py`, `manifest.py` covered; these two still need tests |
| `datasets/registry.py` self-registration | Long-term | Flat dict; low urgency until >20 datasets |

---

## Backlog

### Model features
- **Season-stage + quick-win features** — see Phase 20e above
- **QB intelligence** — QB Elo, QBR, passer rating; requires QB-level PBP aggregation and starter tracking
- **Advanced EPA metrics** — DVOA, CPOE, RYOE, weighted EPA
- **Stadium-level HFA coefficient** — separate coefficient per physical building; requires stadium open/close dates (name changes ≠ new stadium). Current franchise-level implementation in `venue_hfa.py` is the correct prior until dates are sourced.

### Infrastructure
- **Automated stadium reference updates** — `gridiron ingest nflverse-upcoming` currently warns when a stadium in the upcoming schedule is absent from `NFL_stadium_reference.csv`. Automate by geocoding new stadiums and appending to the reference rather than requiring manual curation.
- **Config-driven model registration (Hydra + MLflow)** — replaces the programmatic factory (Phase 20f) with full experiment tracking. Correct long-term direction; prerequisite is Phase 22 betting tracker operational (experiment lineage has no P&L value before then).
- **Ensemble systems** — stack Elo + EPA + ML with uncertainty estimates
- **Drive/possession simulation** — play-level Monte Carlo
- **Injury + roster intelligence** — injury-adjusted team strength
- **Live game intelligence** — real-time win probability (major architecture change)
- **Public dashboards / automated reports** — shareable weekly output

### Simulation improvements
- **Monte Carlo per-stage playoffs** — simulate each round independently so known results can be injected
- **Skip full-season sim after week 18** — use actual standings once regular season is complete
- **Dynamic schedule updates** — handle mid-season game relocations without manual file edits