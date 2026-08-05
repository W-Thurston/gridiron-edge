# Gridiron Edge Handoff

This document describes the current operating system. Historical implementation details belong in `CHANGELOG.md`; future work belongs in `ROADMAP.md`; locked architecture belongs in `DECISIONS.md`; active execution checklists belong in `PLAN.md`.

## System Contract

Gridiron Edge is a file-backed NFL decision-support platform with a Python CLI, persisted model and evaluation artifacts, a read-only FastAPI service, and a generated-contract React frontend.

The game-prediction domain uses one canonical Away/Home-oriented row per game. Win models predict `HOME_WIN`; Away Win Probability is the complement. Total models independently predict `ACTUAL_TOTAL`. Differential features use Home minus Away. Runtime game prediction does not depend on doubled team-perspective rows, `TEAM_A`, `TEAM_B`, `HOME_FIELD`, `RESULT` as a target, implicit forecast recency, hidden Elo fallback, or request-time model execution.

The operational source of truth for a week is an explicitly selected immutable weekly product. Prediction readiness and market readiness are independent. A prediction-ready product remains valid when market data is missing, and forecast publication does not require sportsbook prices.

## Repository Layout

```text
src/gridiron_edge/
  api/                         read-only FastAPI schemas, loaders, serializers, routes
  betting/                     ledger, bankroll, settlement, performance
  cli/                         single-purpose and composite commands
  core/                        settings, logging, console helpers
  datasets/                    typed registry, loaders, writers
  evaluation/                  forecast events, backfills, metrics, selection, closeout
  features/                    canonical game and player feature pipelines
  ingest/                      nflverse, weather, player, and market adapters
  market/                      odds math, edge diagnostics, recommendations, CLV
  models/                      game, Elo, and prop model families
  ratings/                     Elo state and evaluation
  sim/                         season and playoff simulation
  transform/                   canonical cleaning and joins
  viz/                         persisted-output rendering
frontend/                      Vite, React, TypeScript, React Query, generated API client
tests/                         unit, integration, and end-to-end suites
data/                          registered input, artifact, and output storage
```

## Setup and Configuration

Use the repository-managed Python environment:

```bash
uv sync
```

Frontend dependencies:

```bash
cd frontend
pnpm install
cd ..
```

Local application loop:

```bash
# Terminal 1
uv run gridiron api serve

# Terminal 2
cd frontend
pnpm dev
```

The frontend uses the checked-in `api-schema.json` and generated `frontend/src/api/schema.ts`. After an API contract change:

```bash
uv run gridiron api export-schema
cd frontend
pnpm gen:api
pnpm build
cd ..
```

Do not hand-edit `frontend/src/api/schema.ts`.

## Canonical Data Pipeline

The implemented command is:

```bash
uv run gridiron run-data-pipeline
```

With no stage flags, all registered stages run:

```text
fetch-games
clean-games
fetch-upcoming
clean-upcoming
fetch-weather
build-epa
build-elo
build-features
```

`--skip` and `--only` are mutually exclusive. Current-market quotes are not part of this command and are refreshed explicitly under `gridiron ingest`.

Examples:

```bash
uv run gridiron run-data-pipeline --only build-features
uv run gridiron run-data-pipeline --skip fetch-weather
uv run gridiron run-data-pipeline --all-years --upcoming-season 2026 --fit-elo-all-years
```

Pipeline staleness checks resolve canonical input and output paths through the dataset registry. When a registered input is newer than an existing output, the command emits a nonfatal warning that the active stage will rebuild the stale output.

During the offseason, a completed-game fetch may contain no games for the upcoming season. `clean-games` refuses to overwrite populated historical data with an empty result. This protected state is expected.

## Dataset and Artifact Registry

Canonical dataset paths are owned by `src/gridiron_edge/datasets/registry.py`. Domain code should use `dataset_path()` rather than duplicate path strings.

Important artifacts:

```text
data/output/champions/champions.json
data/output/predictions/forecast_events.parquet
data/output/weekly_products/index.json
data/output/weekly_products/current.json
data/output/weekly_products/products/{product_id}.parquet
data/odds/odds_current.parquet
```

The repository remains intentionally file-backed. Revisit database storage only when multi-user concurrency, transactional integrity, or query complexity requires it.

## Game Models and Champion Manifest

Active game model identities are unversioned composite pairs:

```text
win_prob / elo
win_prob / logistic
win_prob / random_forest
win_prob / xgboost
total / random_forest
total / xgboost
```

Deployable artifacts live under:

```text
data/models/{model_name}/{model_type}/
```

The runtime champion manifest is:

```text
data/output/champions/champions.json
```

It is written by promotion workflows such as `full-retrain` and consumed as a static runtime artifact. The API does not compare model metrics or select champions at request time.

## Model Availability and Weekly Policy

Weekly availability is model-specific and truthful.

Elo requires complete exact-week Away and Home rating coverage. Trained models require readable artifact metadata, a persisted model file, exact model and task identity, agreement between artifact and current feature contracts, successful canonical feature construction, and complete required feature coverage for every scheduled game.

The weekly execution service:

1. scopes the rich upcoming schedule;
2. inspects available Win and Total candidates;
3. loads champion provenance;
4. resolves independent Win and Total policy decisions;
5. executes the exact selected registry identities;
6. requires exactly one valid prediction per scheduled game for each selected family before persistence.

Unavailable families do not execute and do not emit forecast events. There is no hidden Elo fallback.

## Forecast Event Contract

Forecast events are stored at:

```text
data/output/predictions/forecast_events.parquet
```

Each row is an immutable event identified by `event_id`. Multiple coherent events may coexist for the same game and model. Rewriting an identical event ID is idempotent; conflicting reuse is rejected.

Roles are explicit:

```text
live
backfilled
```

`live` events are generated by weekly operational prediction before kickoff. `backfilled` events are historical reconstructions for evaluation and champion comparison. Backfilled forecasts are not substitutes for forecasts issued live.

Selected Win and Total events from one weekly invocation share a run ID and UTC generation timestamp while preserving independent model identities.

## Immutable Weekly Products

Weekly products are stored under:

```text
data/output/weekly_products/
```

Layout:

```text
products/{product_id}.parquet   immutable validated products
index.json                      indexed product metadata
current.json                    explicit season-and-week selections
```

A product contains schedule-complete rows with independent Win, Spread, Total, and projected-score components plus their statuses and provenance.

Writing a product does not select it. `select_current_weekly_product()` explicitly selects an indexed product for one season and week. `load_current_weekly_product()` loads only that selection. Missing selection is an explicit error; consumers must not infer current state from file order or timestamps.

Spread derives from the exact selected Win event and its persisted calibration. Total values and uncertainty use the independently selected Total event and exact artifact metadata. Projected scores exist only when required Spread and Total point estimates are usable.

## Pregame Workflow

Run:

```bash
uv run gridiron weekly-predict --season 2026-2027 --week 1
```

Stages:

```text
ensure-data-fresh
predict-week
compose-weekly-product
verify-weekly-readiness
render-outputs
generate-edges
```

The command:

1. refreshes canonical data except out-of-band weather refresh;
2. resolves and executes policy-selected live Win and Total models;
3. persists immutable forecast events;
4. composes and explicitly selects a schedule-complete weekly product;
5. verifies selected-product prediction readiness;
6. publishes PNG and HTML forecast outputs;
7. evaluates edges against the existing source-neutral market snapshot.

Edge generation soft-fails when market data is unavailable. This does not invalidate prediction readiness or forecast publication.

`weekly-predict` supports `--skip` and `--only`. It does not support `--assume-done`.

## Market Data and Edge Diagnostics

The current snapshot is:

```text
data/odds/odds_current.parquet
```

Current The Odds API ingestion is explicit:

```bash
uv run gridiron ingest odds \
  --season 2026-2027 \
  --week 1
```
The command resolves its credential from --odds-api-key or ODDS_API_KEY, loads the canonical rich schedule, requests US NFL moneyline, spread, and total markets in American-odds format, preserves every returned sportsbook independently, appends the observation ledger, and atomically replaces the current snapshot after successful validation.

The command reports quote, game, and sportsbook counts plus provider quota metadata when returned. It is not invoked by weekly-predict, run-data-pipeline, post-week, full-retrain, or verification workflows. Request, HTTP, JSON, payload, empty-response, and zero-match failures preserve the existing quote artifacts.

Storage uses the canonical provider-aware quote contract through `write_current_odds_snapshot()` and `load_current_odds()`. The implemented nflverse consensus adapter records:

```text
provider=nflverse
sportsbook=null
```

The retired DraftKings adapter, resolver, and CLI command are absent. `weekly-predict` consumes an existing current snapshot and does not perform a network-dependent market fetch.

Edge diagnostics are authoritative result state. They distinguish:

```text
blocked
no calculable edges
no positive edges
positive edges filtered by min_ev
returned positive edges
```

Missing market data is not “No play.” `No play` is reserved for a completed evaluation with no positive edge. Blocked or analytically empty results remove stale scope-specific edge CSV output.

## Postgame Workflow

Run only after completed outcomes are available:

```bash
uv run gridiron post-week --season 2026-2027 --week 1
```

Stages:

```text
refresh-results
refresh-next-week-state
close-live-forecasts
```

The command refreshes outcomes, refreshes next-week schedule and feature state, and evaluates the exact `live` Win and Total events referenced by the selected weekly product. Missing components, events, or outcomes are reported explicitly. Incomplete closeout exits nonzero.

`post-week` does not run historical forecast backfill.

## Historical Backfills and Evaluation

Historical backfills write `backfilled` forecast events using honest time-ordered training cutoffs. They support model evaluation, champion comparison, and baseline reporting.

Do not combine live and backfilled roles when measuring operational forecast performance. Postgame closeout evaluates the selected product’s exact live event identities.

Prop evaluation and champion selection are likewise archive-driven. Prop projections require persisted deployable artifacts.

## Full Retrain Workflow

Run:

```bash
uv run gridiron full-retrain
```

Stages:

```text
refresh-all-data
backfill-game-models
backfill-prop-models
train-game-models
train-prop-models
refresh-calibrations
promote-champions
baseline-report
```

This is the heavy historical workflow and can run for hours. It supports model filters, `--skip`, `--only`, `--skip-prop-backfill`, and `--assume-done`.

Use `--assume-done` only when named prior stages completed and their required artifacts remain on disk. This option is not available on `weekly-predict` or `post-week`.

## API Serialization Boundary

The API is read-only and serializes persisted state. It does not:

- run model inference;
- resolve a weekly forecast by recency;
- compare champion metrics;
- fall back to Elo for Games endpoints;
- generate products or edges at request time.

Games are schedule-first. Scheduled games remain visible when prediction components are unavailable. Win, Spread, Total, and projected score are independent response blocks with independent readiness and provenance.

Unpopulated fields use `_meta.field_status`. Stable blocker slugs identify missing upstream capabilities, while semantic roadmap references avoid binding runtime responses to temporary workstream numbering.

OpenAPI is checked in as `api-schema.json`; frontend types are generated from it.

## Frontend Contract Workflow

The frontend uses Vite, React, TypeScript, React Query, and `openapi-fetch`. API-facing component types should derive from generated schemas:

```typescript
components["schemas"]["TypeName"]
```

After an API surface change:

```bash
uv run gridiron api export-schema
cd frontend
pnpm gen:api
pnpm lint
pnpm build
pnpm test:run
cd ..
```

Do not silently omit unavailable data. Use shared field-status, weekly-component, and edge-result presentation components. Team identities may arrive as canonical abbreviations or service-preserved long names and must resolve through shared team metadata.

Context modules intentionally colocate each Provider and matching hook. ESLint’s Fast Refresh export rule is narrowly disabled only for `src/context/*Context.tsx`; all other rules and files retain the standard configuration.

## BetSlip and Betting Ledger Boundaries

BetSlip is a local draft decision workspace. It does not place sportsbook wagers and is not the betting ledger.

A staged leg preserves immutable recommendation provenance separately from editable current odds, proposed stake, sportsbook, and notes. Current-price changes never rewrite the original recommendation snapshot. Missing price blocks price-dependent outputs rather than inventing odds.

The betting ledger records confirmed bets. Bankroll is managed separately and coordinated by the CLI. BetSlip identity is producer-independent so the same wager deduplicates across frontend surfaces.

## Verification Commands

Python and backend verification:

```bash
uv run gridiron verify
```

Default coverage:

- Ruff;
- Pyrefly;
- backend unit, integration, and end-to-end tests;
- external nflverse `fetch-games + clean-games` smoke check;
- model baseline comparison.

`--fast` skips e2e and smoke. `--very-thorough` adds slow tests. `--strict` converts smoke and missing-baseline soft failures into hard failures. This command does not run frontend gates or weekly readiness.

Selected-product operational readiness:

```bash
uv run gridiron verify-week --season 2026-2027 --week 1
```

It reports schedule, component, provenance, market, join, eligible-market, edge, and blocker state without modifying data or running inference.

Frontend gates:

```bash
cd frontend
pnpm lint
pnpm build
pnpm test:run
cd ..
```

## Operational Recovery

### Weekly prediction is blocked

1. Run `verify-week` for the exact season and week.
2. Inspect component and provenance blockers.
3. Repair missing artifacts, feature coverage, calibration, or selection inputs.
4. Rerun the coherent `weekly-predict` workflow.

Do not patch a selected product by inferring current events from recency.

### Market data is missing

The selected prediction product remains valid. Forecast PNG and HTML publication should succeed. Edge generation soft-fails with an explicit market blocker. Do not fabricate prices. Write a supported source-neutral market snapshot before rerunning edge generation.

### `post-week` is incomplete

Confirm completed outcomes exist for every scheduled game. Running `post-week` before games finish correctly exits nonzero and lists missing outcomes. Do not replace live events with historical backfills.

### Full retrain is interrupted

Use `full-retrain --help` to confirm exact stage names. Resume with `--only` plus `--assume-done` only for stages that actually completed and have valid artifacts on disk.

### Forecast rendering fails

Rendering consumes the selected product and must tolerate absent optional market columns. Fix the renderer or product contract, then rerun the coherent weekly workflow. A rerun creates a new immutable forecast run and product.

### Stale outputs

Prediction-readiness blockers remove stale forecast PNG and HTML outputs. Blocked or empty edge results remove stale edge CSV output. The explicitly selected weekly product remains the operational authority.

## Quality Gates

Preferred Python boundary:

```bash
uv run ruff check . --fix && \
uvx pyrefly check && \
uv run pytest -m "unit and not slow"
```

Frontend boundary:

```bash
cd frontend
pnpm lint
pnpm build
pnpm test:run
cd ..
```

Pre-commit runs Python lint, type checking, and unit tests. Pre-push adds integration and end-to-end tests.

## Known Limitations

- The Odds API v4 client, parser, and explicit current-market ingest command are implemented. Same-book recommendation evaluation, automatic refresh policy, broader operational integration, and multi-book shopping remain future work.
- Multi-book line shopping, arbitrage, middles, and book selectors remain future product work after supported ingestion and same-book evaluation are complete.
- Injury and news data are not integrated.
- Scenario analysis, feature attribution, and historical comparable retrieval remain future capabilities.
- Live game state, live odds, in-game win probability, and WebSocket updates are not implemented.
- Current-season PBP may be unavailable until the upstream source publishes it. Refresh can warn while continuing with the available historical feature state.
- Some API endpoints may still require batch-artifact refactors to fully satisfy the serialization-boundary design; track verified cases in `ROADMAP.md`.
- The project has never been live in production. Development-era schemas and artifacts do not require backward compatibility unless a current contract explicitly says otherwise.
