# Gridiron Edge Handoff

This document describes the current operating system. Historical implementation details belong in `CHANGELOG.md`; future work belongs in `ROADMAP.md`; locked architecture belongs in `DECISIONS.md`; active execution checklists belong in `PLAN.md`.

## System Contract

Gridiron Edge is a file-backed NFL decision-support platform with a Python CLI, persisted model and evaluation artifacts, a predominantly read-only FastAPI service with one narrow local wager-recording boundary, and a generated-contract React frontend.

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
data/odds/history/season={season}/week={week}.parquet
data/odds/collection_plans/current.json
data/odds/collection_plans/season={season}/week={week}.json
data/odds/collection_runs/season={season}/week={week}/scheduled_at={timestamp}/claim.json
data/odds/collection_runs/season={season}/week={week}/scheduled_at={timestamp}/result.json
data/output/candidate_issuance/issuances/{issuance_id}.json
data/output/recommendation_governance/schema=1/versions/{governance_id}.json
data/output/recommendation_policies/schema=1/{policy_id}.json
data/output/recommended_bet_results/schema=1/evaluations/{evaluation_id}.json
data/output/recommended_bet_results/schema=1/results/{result_id}.json
data/output/production_chain_preflight/schema=1/assessments/{preflight_id}.json
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

## Production Recommendation Chain

The production recommendation workflow is explicit and identity-addressed. It
does not select artifacts by modification time, infer recommendation state from
positive expected value, or place sportsbook wagers.

### Current 2026 Week 1 identities

```text
Candidate issuance:
278d60da4e2dc089ff7eb973620f49050f83de336034cbff0c8c1a097401ccff

Recommendation governance:
56757db59c2d04a55eb3f980299699403fdc982e4fe7ff4963f0898112f4824e

Recommendation policy:
9e2cc3363656366eae76ec0935f01ff201ce9c9784e2736936fd0af9ab0ab024

Recommended-bet evaluation:
8301fb74e1eaa10437376ff3b616aaa1efc3477944d1a8da0df94abd55de073c

Production-chain checkpoint:
acf50214f67aed1833e38f998685c3bde4f8f5489a3771f1e50adc319bb887fb
```

The Week 1 candidate issuance evaluated 1,680 canonical quote observations and
contains 698 candidates, 982 not-candidates, and zero unavailable rows. The
persisted recommendation evaluation contains 698 unavailable results and no
qualified, recommended, failed, or conflicting results.

### Issue candidates

```bash
uv run gridiron production-chain issue-candidates \
  --season 2026-2027 \
  --week 1 \
  --evaluated-at 2026-08-18T14:45:00+00:00 \
  --write
```

The command loads the explicitly selected weekly product, restricts immutable
forecast events to its exact product run, loads the canonical weekly quote
ledger, evaluates every exact observation, and optionally persists one
immutable issuance. It requires a timezone-aware UTC evaluation timestamp.

### Create and verify governance

Governance values are explicit inputs. The command supplies no hidden sizing or
exposure defaults.

```bash
uv run gridiron production-chain create-governance \
  --created-at 2026-08-18T15:30:00+00:00 \
  --fractional-kelly-multiplier 0.25 \
  --minimum-actionable-stake 5 \
  --stake-increment 1 \
  --stake-rounding down \
  --maximum-candidate-bankroll-fraction 0.02 \
  --maximum-game-bankroll-fraction 0.05 \
  --maximum-portfolio-bankroll-fraction 0.20 \
  --prohibit-opposing-positions \
  --correlation-check-mandatory \
  --exposure-eligible-status open \
  --write
```

```bash
uv run gridiron production-chain verify-governance \
  --governance-id 56757db59c2d04a55eb3f980299699403fdc982e4fe7ff4963f0898112f4824e
```

Governance identity is derived from governed content and excludes creation time.
Exact replay is idempotent. Different content cannot reuse an existing identity.

### Derive policy

```bash
uv run gridiron production-chain derive-policy \
  --issuance-id 278d60da4e2dc089ff7eb973620f49050f83de336034cbff0c8c1a097401ccff \
  --governance-id 56757db59c2d04a55eb3f980299699403fdc982e4fe7ff4963f0898112f4824e \
  --created-at 2026-08-18T15:45:00+00:00 \
  --write
```

Policy derivation consumes the exact issuance, governance, canonical quote
history and boundaries, cleaned outcomes, available closeout evidence, and
available settled-wager return evidence. Moneyline, Spread, and Total are
derived independently. Missing required evidence produces an unavailable family
policy rather than an invented threshold.

### Evaluate recommendations

```bash
uv run gridiron production-chain evaluate-recommendations \
  --issuance-id 278d60da4e2dc089ff7eb973620f49050f83de336034cbff0c8c1a097401ccff \
  --policy-id 9e2cc3363656366eae76ec0935f01ff201ce9c9784e2736936fd0af9ab0ab024 \
  --decision-at 2026-08-18T15:50:00+00:00 \
  --write
```

One immutable result is produced for every candidate row. The result preserves
exact offer, product, forecast, policy, checks, decision time, quote ages,
sizing, bankroll, portfolio, and correlation evidence. Historical
not-candidates and unavailable issuance rows are not duplicated into result
artifacts.

### Assess and verify the chain

```bash
uv run gridiron production-chain assess \
  --season 2026-2027 \
  --week 1 \
  --assessed-at 2026-08-18T15:50:00+00:00 \
  --write
```

```bash
uv run gridiron production-chain verify \
  --preflight-id acf50214f67aed1833e38f998685c3bde4f8f5489a3771f1e50adc319bb887fb
```

`assess` reads current repository evidence once. `verify` reads one exact stored
assessment and does not reassess mutable state.

Preflight classifies every component independently as available, incomplete,
unavailable, invalid, conflicting, or not yet eligible. Candidate issuance,
policy, and recommendation evidence are accepted only through strict artifact
readers and exact season, week, product, run, issuance, policy, and evaluation
relationships.

The current Week 1 assessment reports the selected product, forecast
provenance, quote snapshot, repeated quote history, selected collection plan,
candidate issuance, recommendation policy, recommendation results, backend
serialization, and frontend presentation as available for Moneyline, Spread,
and Total.

Collection execution remains not yet eligible until the first selected-plan
poll at `2026-09-08T12:00:00Z`. Completed outcomes, market closeout, CLV, and
realized performance remain not yet eligible before kickoff.

### Backend and frontend result delivery

The `/lines` and `/edges` loaders read persisted recommended-bet evaluations,
select the latest unambiguous explicit evaluation time for each exact offer, and
attach a result only when the full provider-aware offer identity matches.
Equal-time different results for one exact offer are rejected as conflicting.

The frontend mechanically presents the persisted lifecycle state:

```text
No persisted result       Candidate
qualified                 Qualified opportunity
recommended               Recommended
failed                    Failed qualification
unavailable               Recommendation unavailable
conflicting               Conflicting evidence
```

The Policy evidence disclosure displays persisted checks, policy identity,
evaluation time, exact offer provenance, forecast provenance, and persisted
suggested stake when present. The frontend does not calculate qualification,
recommendation state, or suggested stake.

### Postgame reassessment

After kickoff, production-chain assessment assembles postgame evidence once and
reuses the established owners:

```text
load_live_forecast_closeout()
close_candidate_issuance()
select_quote_history_boundaries()
evaluate_market_families()
load_games()
load_bets()
```

Completed outcomes are reconciled to the exact selected weekly product and live
forecast events. Candidate closeout requires the same provider, provider event,
sportsbook, game, market, and side and selects only the latest non-live quote
observed strictly before kickoff.

CLV kinds remain market-specific:

```text
Moneyline    moneyline_price
Spread       spread_points
Total        total_points
```

Realized performance requires uniquely attributed settled-wager evidence. If no
matching wager was recorded, return evidence remains unavailable rather than
zero.

## Quote Collection Worker Deployment

The supported recurring quote collector is a dedicated Raspberry Pi worker. It
executes explicitly selected weekly collection plans and does not choose or
infer a season or week.

### Validated worker

```text
Hardware: Raspberry Pi 4 Model B Rev 1.4
Architecture: aarch64
Memory: 8 GB
Storage: 2 TB SanDisk Extreme SSD
Boot: SSD-only
Stable storage transport: black USB 2 port
Repository: /home/thursty/apps/gridiron-edge
Python: /home/thursty/apps/gridiron-edge/.venv/bin/python
uv: /home/thursty/.local/bin/uv
```

The microSD installation is offline recovery media. The Raspberry Pi is not validated as a full API, frontend, model-training, or prediction appliance.

Repository-owned deployment assets
deploy/bin/install_quote_collection_worker.py
deploy/bin/verify_quote_collection_worker.py
deploy/systemd/gridiron-edge-collector.service
deploy/systemd/gridiron-edge-collector.timer
src/gridiron_edge/deployment/quote_collection_worker.py

Installed paths
/usr/local/libexec/gridiron-edge-collector
/etc/systemd/system/gridiron-edge-collector.service
/etc/systemd/system/gridiron-edge-collector.timer
/etc/gridiron-edge-collector.env


The environment file must be a regular root-owned, root-group mode-0600 file containing exactly one nonempty ODDS_API_KEY assignment.

Installation validates the secret assignment. Read-only verification checks only file type, owner, group, and mode and never opens the credential file.

Install or reinstall

Run from /home/thursty/apps/gridiron-edge:

sudo \
  /home/thursty/apps/gridiron-edge/.venv/bin/python \
  -B \
  deploy/bin/install_quote_collection_worker.py \
  --repository /home/thursty/apps/gridiron-edge \
  --user thursty \
  --group thursty \
  --uv-path /home/thursty/.local/bin/uv \
  --environment-file /etc/gridiron-edge-collector.env \
  --wrapper-path /usr/local/libexec/gridiron-edge-collector \
  --service-path /etc/systemd/system/gridiron-edge-collector.service \
  --timer-path /etc/systemd/system/gridiron-edge-collector.timer \
  --service-template deploy/systemd/gridiron-edge-collector.service \
  --timer-template deploy/systemd/gridiron-edge-collector.timer


Installation:

requires the rich upcoming schedule and global current-plan selection;
validates the complete staged deployment set before replacement;
preserves prior files and permission modes;
restores the prior deployment if systemd reload fails;
does not select or transfer a weekly plan;
does not enable the timer unless --enable-timer is supplied.

Activation is intentionally separate from installation.

Verify
sudo \
  /home/thursty/apps/gridiron-edge/.venv/bin/python \
  -B \
  deploy/bin/verify_quote_collection_worker.py \
  --repository /home/thursty/apps/gridiron-edge \
  --user thursty \
  --group thursty \
  --uv-path /home/thursty/.local/bin/uv \
  --environment-file /etc/gridiron-edge-collector.env \
  --wrapper-path /usr/local/libexec/gridiron-edge-collector \
  --service-path /etc/systemd/system/gridiron-edge-collector.service \
  --timer-path /etc/systemd/system/gridiron-edge-collector.timer


A healthy deployment reports Worker status: ready.

Verification covers:

repository, schedule, selection, and selected-plan resolution;
unresolved execution claims;
installed paths;
credential-file metadata;
secret assignment disclosure in deployed files and journal output;
systemd unit syntax;
timer enablement and activity;
clock synchronization;
root filesystem capacity;
Raspberry Pi throttling;
latest service result;
configured storage-error markers.
Service and timer semantics

The service is a non-root Type=oneshot unit with no automatic restart. The wrapper generates the current UTC evaluation timestamp and calls:

gridiron ingest execute-selected-odds-plan


with:

grace minutes: 15
minimum provider-credit reserve: 30
provider timeout seconds: 15


The timer wakes every five minutes. Due-time eligibility remains owned by the selected-plan executor. A wake that is not due exits successfully without provider access or execution artifacts.

The service is normally inactive (dead) between runs. The timer, not the service, is the persistent active unit.

Operate

Inspect state:

systemctl is-enabled gridiron-edge-collector.timer
systemctl is-active gridiron-edge-collector.timer
systemctl list-timers gridiron-edge-collector.timer --all
sudo systemctl status gridiron-edge-collector.service --no-pager
sudo journalctl -u gridiron-edge-collector.service -n 50 --no-pager


Run one managed execution:

sudo systemctl start gridiron-edge-collector.service


Pause recurring execution:

sudo systemctl disable --now gridiron-edge-collector.timer


Resume recurring execution:

sudo systemctl enable --now gridiron-edge-collector.timer


Disabling the timer does not delete plans, selections, claims, terminal results, or quote artifacts.

Weekly rollover

Generate and validate the new weekly plan on the reviewed development checkout. Transfer the rich upcoming schedule, scoped plan, and global current selection to the worker. Compare source and destination identities before enabling or continuing the timer.

The installer does not generate, select, or transfer weekly plans.

Verify after every plan rollover:

sudo \
  /home/thursty/apps/gridiron-edge/.venv/bin/python \
  -B \
  deploy/bin/verify_quote_collection_worker.py \
  --repository /home/thursty/apps/gridiron-edge \
  --user thursty \
  --group thursty \
  --uv-path /home/thursty/.local/bin/uv \
  --environment-file /etc/gridiron-edge-collector.env \
  --wrapper-path /usr/local/libexec/gridiron-edge-collector \
  --service-path /etc/systemd/system/gridiron-edge-collector.service \
  --timer-path /etc/systemd/system/gridiron-edge-collector.timer

Health checks
vcgencmd measure_temp
vcgencmd get_throttled
findmnt -no SOURCE,FSTYPE,AVAIL,TARGET /

sudo dmesg |
  grep -iE \
    'usb disconnect|reset|timeout|I/O error|capacity change|Synchronize Cache'


The validated healthy state had throttled=0x0, root storage on /dev/sda2, and no configured current USB disconnect, reset, timeout, I/O, capacity-change, or cache-synchronization errors.

The SSD must remain on the proven stable USB 2 path.

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

Prediction, market, edge, and recommendation reads serialize persisted state.
The explicit `POST /portfolio/bets` route is the narrow local write boundary for
recording a completed game-wager draft through the rollback-safe ledger and
bankroll owner. It does not place a sportsbook wager.

API request paths do not:

- run model inference;
- resolve a weekly forecast by recency;
- compare champion metrics;
- fall back to Elo for Games endpoints;
- derive recommendation policy or evaluate candidates;
- recalculate persisted Kelly sizing;
- generate products or edges at request time;
- accept trusted provider, model, expected-value, policy, or reference-offer
  evidence from the browser.

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

## Bet Slip and Betting Ledger Boundaries

Bet Slip version 4 is the only active draft contract. It is a local decision
workspace, not the betting ledger and not sportsbook execution.

Each game leg keeps three evidence categories separate:

```text
persistedRecommendation   immutable persisted recommendation evidence
edgeAnalytics             immutable analytical offer and model evidence
draft                     editable current odds, line, sportsbook, stake, note
```

Editing draft values never changes persisted recommendation state, exact
reference offer, policy identity, original expected value, checks, or persisted
suggested stake. Local bankroll and Kelly controls are transient what-if inputs.

A user may explicitly record a complete Moneyline, Spread, or Total draft
through the Portfolio API or betting CLI. Recommendation-backed recording sends
only the complete persisted result, evaluation, candidate, and policy identity
chain. The backend resolves trusted offer, forecast, model, expected-value, and
policy evidence from immutable artifacts.

Recorded terms may differ from the original recommendation reference terms.
Ledger and bankroll transaction writes share one rollback-safe domain operation.
If either write fails, prior artifacts are restored. A successful operation
records a wager in Gridiron Edge and does not place a sportsbook wager.

Manual and Candidate wagers may contain no recommendation identity. A
recommendation-backed wager requires the complete identity chain. Partial or
empty identity chains are rejected.

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

### Production-chain evidence is unavailable, invalid, or conflicting

1. Run `production-chain assess` for the exact season, week, and UTC assessment
   time without `--write`.
2. Inspect the component reason independently for Moneyline, Spread, and Total.
3. Verify the exact candidate issuance, governance, policy, evaluation, and
   selected-product identities through their owning strict readers or verify
   commands.
4. Do not delete a valid immutable artifact to make another artifact win by
   recency.
5. Multiple exact matching issuances, policies, or evaluations are conflicting
   unless an explicit selection contract is introduced.
6. After repairing the owning evidence, create a new assessment at an explicit
   UTC timestamp. Existing persisted assessments remain immutable.

### Recommendation results do not appear in the frontend

1. Confirm a recommended-bet evaluation exists for the requested season and
   week.
2. Confirm `/lines` or `/edges` returns a non-null `recommendation` for the exact
   offer.
3. Compare provider, provider event, sportsbook, game, market, side, fetch time,
   sportsbook update time, kickoff, live state, price, and line.
4. A null recommendation is presented as Candidate. A persisted unavailable
   result is presented as Recommendation unavailable.
5. Do not promote a Candidate by modifying frontend presentation logic.

### Collection execution is incomplete

Inspect `data/odds/collection_runs` for the exact scheduled poll. A `claim.json`
without `result.json` is unresolved and blocks automatic retry. Do not delete or
reclaim it automatically. Inspect the service journal and the immutable claim,
then resolve the failure deliberately.

Manual `gridiron ingest odds` runs may add historical observations but do not
satisfy selected-plan execution evidence.

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

- The Odds API v4 current-market path, provider-aware current and partitioned historical quote storage, selected-plan acquisition, exact-offer evaluation, Line Shopping, immutable candidate issuance, validated closeout and market-specific CLV contracts, recommendation governance, policy, persisted results, API attachment, frontend presentation, explicit local wager recording, and repository-owned Raspberry Pi worker deployment are implemented.
- The real 2026 Week 1 rehearsal has two fetch timestamps and complete repeated depth for current exact identities. The selected plan has not reached its first scheduled poll, and completed Week 1 outcomes, real latest-eligible closeouts, real CLV, and realized performance are not yet eligible.
- Current production policy remains unavailable because matured empirical outcome, closeout, and return evidence does not yet support an active threshold-selection method. Arbitrage, middles, movement interpretation, backtesting, and supported historical provider backfill remain future capabilities.
- Injury and news data are not integrated.
- Scenario analysis, feature attribution, and historical comparable retrieval remain future capabilities.
- Live game state, live odds, in-game win probability, and WebSocket updates are not implemented.
- Current-season PBP may be unavailable until the upstream source publishes it. Refresh can warn while continuing with the available historical feature state.
- Some API endpoints may still require batch-artifact refactors to fully satisfy the serialization-boundary design; track verified cases in `ROADMAP.md`.
- The project has never been live in production. Development-era schemas and artifacts do not require backward compatibility unless a current contract explicitly says otherwise.
