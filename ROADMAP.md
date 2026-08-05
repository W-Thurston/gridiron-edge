# Gridiron Edge Roadmap

## Document Ownership

| Document | Purpose |
|---|---|
| `ROADMAP.md` | Genuine future capabilities, strategic priorities, and current limitations |
| `PLAN.md` | Active implementation checklist and completed-unit record |
| `HANDOFF.md` | Current operating system, commands, artifacts, and recovery guidance |
| `DECISIONS.md` | Append-only architectural decisions and supersession history |
| `CHANGELOG.md` | Dated implementation history |

The canonical weekly prediction, product, API, frontend, and verification architecture is implemented. Future work should build on the persisted-event and explicitly selected weekly-product contracts rather than restore retired archive, fallback, or request-time behavior.

## Current Platform State

Gridiron Edge currently provides:

- canonical one-row Away/Home game modeling;
- independent Win and Total model families;
- unversioned champion artifacts and a persisted champion manifest;
- model-specific weekly availability inspection and policy selection;
- immutable `live` and `backfilled` forecast events;
- immutable schedule-complete weekly products with explicit current selection;
- independent Win, Spread, Total, and projected-score readiness;
- source-neutral current-market storage and explicit edge diagnostics;
- completed-week closeout against exact selected live forecast events;
- archive-driven historical evaluation and champion comparison;
- a read-only serialization API;
- a generated-contract React frontend;
- schedule-first game presentation and truthful missing-data states;
- betting ledger, bankroll, performance, and local BetSlip decision support;
- season and playoff simulation;
- Python and frontend quality boundaries.

The successful 2026 Week 1 rehearsal produced complete Win, Spread, Total, projected-score, and provenance coverage for all scheduled games while market readiness remained independently blocked. Forecast PNG and HTML publication succeeded without requiring market prices.

## Strategic Priorities

Prioritize work by value density and architectural fit:

1. Preserve truthful persisted-state boundaries before adding breadth.
2. Resolve supported external data sources before building interfaces that depend on them.
3. Improve predictive quality only through honest time-ordered evaluation.
4. Keep the API a serialization boundary.
5. Keep unavailable, blocked, and analytical-empty states explicit.
6. Add product surface area only when the underlying data contract is real.
7. Continue using files until concurrency, transactional integrity, or query complexity requires a database.

## Future Work

### Supported Market Provider and Multi-Book Shopping

**Status:** Active program

**Goal:** establish a dependable supported market-data provider and add
cross-book execution tooling without coupling forecast publication to market
availability.

Current state:

- market storage is source-neutral;
- the nflverse schedule adapter can populate current game-market context when
  source data is available;
- the DraftKings adapter is legacy and unreliable because anti-bot responses
  can block access;
- weekly prediction consumes an existing current snapshot and does not fetch
  external prices;
- missing markets soft-fail only edge generation.

Program sequence:

1. **Provider and contract selection.** The Odds API v4 selected; normalized
   quote, freshness, identity, configuration, and failure boundaries locked.
2. **Source-neutral quote migration.** Separate provider from sportsbook,
   preserve provider event and update provenance, and make storage multi-book
   safe before live ingestion.
3. **Current provider adapter.** Ingest current and upcoming NFL moneyline,
   spread, and total quotes from The Odds API into the normalized store.
4. **Operational integration.** Add explicit refresh, freshness, coverage,
   same-book price evaluation, prediction-market joins, CLI, and API behavior
   while preserving forecast independence.
5. **Real-data frontend integration and audit.** Exercise Dashboard, Games,
   Game Detail, BetSlip, readiness, edge states, sportsbook provenance, and
   responsive presentation against real market responses.
6. **Multi-book shopping.** Add best-price comparison, book selection,
   arbitrage, middle detection, and the Line Shopping product surface.

Current and historical market data are separate workstreams within this
program. The current-market workstream comes first because it unlocks immediate
weekly operation and frontend usability. Historical archive and evaluation
follow after the provider and normalized quote contract are stable.

Current-market scope:

- documented supported API and secret/configuration boundary;
- sportsbook-level current and upcoming moneyline, spread, and total quotes;
- provider, book, event, market, outcome, line, price, and fetch provenance;
- canonical game identity resolution and unmatched-event diagnostics;
- source-neutral snapshot validation and atomic replacement;
- freshness, staleness, partial coverage, malformed response, rate-limit, and
  provider-failure states;
- `verify-week`, unified edge service, API, Dashboard, Games, Game Detail, and
  BetSlip integration.

Historical-market scope, planned separately:

- append-only timestamped quote storage;
- idempotent provider backfill and coverage reporting;
- opening, intermediate, and closing quote definitions;
- leakage-safe pre-kickoff quote selection;
- closing-line value, model-versus-market evaluation, line movement, strategy
  backtesting, and consensus policies;
- partitioning and retention based on observed provider volume.

The initial normalized quote contract must support both current snapshots and a
future historical archive, but historical ingestion and evaluation are not
acceptance requirements for the first current-provider implementation.

Do not fabricate production prices, collapse provider quotes to one book before
normalization, hide a network fetch inside `weekly-predict`, or treat the legacy
DraftKings adapter as a dependable recovery path.

### Model Ensemble

**Goal:** determine whether a time-ordered ensemble improves operational Win prediction enough to justify additional complexity.

Candidate approaches:

- Brier-weighted averaging;
- constrained blending;
- logistic stacking with time-ordered out-of-fold inputs;
- simple rank or probability averaging as a baseline.

Acceptance should require an honest historical comparison against the current champion, preserved calibration quality, complete upcoming-game feature coverage, deployable artifact metadata, availability inspection, and compatibility with the existing weekly policy and immutable event contracts.

An ensemble should register as another model identity. It must not compute dynamically in the API.

### Injury and News Data

**Goal:** add a reliable, timestamped source for player availability and material team news.

Required design work:

- choose a source and usage policy;
- preserve fetched-at and effective-at timestamps;
- distinguish reported, confirmed, and resolved status;
- map players and teams to canonical identities;
- define historical availability for honest evaluation;
- expose blocked or unavailable states when the source is incomplete.

This capability unlocks injury-aware game and prop presentation and is a prerequisite for credible personnel scenarios.

### Scenario Engine and Feature Attribution

**Goal:** answer bounded what-if and explanation questions without mutating production forecasts.

Potential scope:

- feature contribution or local explanation for persisted predictions;
- comparable historical games;
- controlled team-strength or player-availability adjustments;
- usage redistribution for player props;
- scenario-specific Win, Spread, Total, projected score, and edge calculations;
- explicit separation between persisted production output and hypothetical output.

Scenario computation should use an explicit request and response contract. It must not silently alter the selected weekly product or champion artifacts.

### Real-Time and Live Game Support

**Goal:** support in-game decision analysis.

Required foundations:

- live score, clock, down, distance, possession, and timeout state;
- timestamped live market data;
- a validated live win-probability model;
- live edge and hedge calculations;
- streaming or polling transport;
- strict freshness and stale-state presentation.

This remains lower priority than reliable pregame multi-book data and injury/news integration.

### Remaining API Batch-Artifact Boundaries

**Goal:** ensure every API endpoint serializes persisted artifacts rather than performing meaningful computation at request time.

Known candidate for verification:

- model-performance summaries should be confirmed as batch-produced artifacts; if still computed on request, add a batch writer and serialize its output.

For each candidate:

1. identify the current request-time computation;
2. define the persisted artifact schema and writer;
3. add freshness and provenance;
4. migrate loaders to read the artifact;
5. keep routes and serializers thin;
6. add parity tests before removing the old path.

Do not assume a listed historical deviation still exists. Verify it against current code before scheduling work.

### Frontend Product Enhancements

The core game-day and portfolio surfaces are functional. Remaining work should be pulled by real data availability and user value.

Potential enhancements:

- multi-book line-shopping views;
- injury and news presentation;
- scenario and explanation surfaces;
- line-movement and live-game charts;
- richer bankroll history and Kelly-adherence views;
- recorded-bet export and an explicitly designed recorded-bet write workflow;
- remaining table, layout, and accessibility polish;
- a real-data pending-state visual audit after all required backend artifacts are populated.

BetSlip remains a draft decision workspace. Any recorded-bet write workflow requires duplicate protection, bankroll transaction coupling, partial-failure semantics, and an explicit user action. It is not sportsbook execution.

### Model and Feature Research

Candidate research areas:

- offensive and defensive rating decomposition;
- coaching and coordinator effects;
- pace and neutral-situation tendencies;
- special-teams features;
- penalties, pressure, and situational efficiency;
- additional opponent-quality cohorts;
- richer prop distribution models;
- era-aware feature availability and imputation;
- calibrated uncertainty for ratings and projections.

Every new feature must preserve chronological construction, avoid leakage, and use empirical thresholds rather than arbitrary bins.

### Tooling and CI

Future tooling work:

- exercise `gridiron verify --strict` in a real CI surface;
- run the separate frontend lint, build, and test gates in CI;
- consider performance baselines if test or training runtime regresses;
- maintain generated OpenAPI and TypeScript contract checks;
- improve long-running composite resume diagnostics where needed;
- clamp current-season PBP requests to the maximum season published by the upstream source once that policy is defined;
- verify and repair any remaining baseline-report parser edge cases;
- review repository-wide lint exclusions only through dedicated, behavior-preserving work.

## Known Limitations

### Market data

A dependable supported long-term provider is not yet selected. Multi-book execution features remain blocked on that decision. The legacy DraftKings adapter may fail behind anti-bot responses.

### Injury, news, and live state

There is no integrated injury/news feed or live-game state. Related API and frontend fields must remain explicitly blocked.

### Scenario and explanation

Feature attribution, comparable-game retrieval, and what-if propagation are not implemented.

### Current-season PBP cadence

The upstream source may not publish the current season immediately. Pipeline refresh can warn while continuing with available historical feature state. A future cleanup may clamp requests to the latest published season.

### Postgame timing

`post-week` requires completed outcomes. Running it before games finish correctly exits nonzero and lists missing outcomes.

### Markets versus predictions

A selected weekly product can be prediction-ready while market readiness is blocked. Missing market data means no current edge result; it does not invalidate forecasts.

### File-backed architecture

Files remain appropriate for the current single-user workflow. Revisit this only for real multi-user concurrency, transactional guarantees, or query requirements.

## Prioritization Guidance

The next major work should normally be chosen from:

1. supported market provider and multi-book shopping;
2. model ensemble research;
3. injury/news source;
4. scenario engine and explanations;
5. remaining API batch-artifact migrations;
6. frontend enhancements unlocked by real data;
7. real-time and live-game support.

Before starting a new work item:

- verify the gap still exists in current code;
- add it to `PLAN.md` as a bounded execution unit;
- record any locked architectural choice in `DECISIONS.md`;
- update `HANDOFF.md` only after behavior ships;
- record completion in `CHANGELOG.md`.
