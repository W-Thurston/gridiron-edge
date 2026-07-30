# Gridiron Edge — Development Plan

> **Purpose:** a directed view of what we are currently building. PLAN.md is a
> working document, not a reference. It contains exactly one active workstream
> at a time, broken into tiers, each with its own design block that grows
> during the tier and collapses to a summary on completion.

## Where to find other information

| Document | Role |
|----------|------|
| **PLAN.md** (this file) | The active workstream and its in-flight design |
| **ROADMAP.md** | Long-term strategic direction, workstream inventory, prioritization, known issues & backlog |
| **CHANGELOG.md** | What was built and when (closed workstreams + tier summaries) |
| **HANDOFF.md** | How the system works today (architecture, workflows, operations) |
| **DECISIONS.md** | Architectural decisions made during workstreams |

## Status key

| Tag | Meaning |
|-----|---------|
| Active | Currently being worked |
| Designing | In design phase, not yet implementing |
| Complete | Success criteria met (summary retained inline) |
| Paused | Blocked on a dependency being addressed by another workstream |
| Re-scoped | Success criteria invalidated; replaced or dropped |

Workstream identifiers (W1, W2, …) match ROADMAP.md. They exist only inside the planning docs (PLAN, ROADMAP, DECISIONS, CHANGELOG) — never in source code, comments, or commit subjects.

## Ways of Working

How we operate in every session. A new thread should read this first.

1. **Confirm before building — never assume code exists or looks a certain way.** Before writing or modifying anything, verify the current state. Prefer over-confirming: it's cheaper to paste five files than to rebuild something that already existed and discover the drift later. Use grep to locate; ask the user to run commands or paste files. Assumptions are the #1 source of rework and format drift.

2. **Grep first, then read.** Locate with `grep -rn`, then request the specific file(s) or function(s). Ask the user to run the grep/curl and paste results rather than guessing at signatures, schemas, or data.

3. **Design before implementing, at two levels.**
   - **ROADMAP-level (high-level):** lock the workstream shape — what, why, tiers, success criteria — before touching code.
   - **Subsection-level (deep):** before each tier/substep, a focused design block with locked decisions, then implement.

4. **PLAN.md tracks the active work.** After ROADMAP design, expand the workstream into PLAN.md as a checklist. Check items off as completed.

5. **Commit small units as you go.** Each substep is its own commit with a clear message. Quality gates (`ruff` + `pyrefly` + unit tests, or `pnpm build && pnpm test:run` for frontend) pass before each commit.

6. **Section close-out ritual.** When a section/tier completes: clean its detail out of PLAN.md (collapse to a one-line summary or remove), check the item off in ROADMAP.md, and either continue to the next subsection or repeat the full design→plan→build→close loop for the next big chunk.

7. **Verify against real data.** After backend/data changes, confirm via curl or a Python one-liner against the actual artifact — don't trust the code path alone.

8. **Note on dates:** the assistant's system clock may report an earlier date than reality. Trust commit timestamps and the user; when in doubt, ask.

---

## Active Workstream: Schedule-Complete Game Prediction Product

### Objective

Build a static, schedule-complete weekly game prediction product for Moneyline,
Spread, and Total markets, with explicit model provenance, source-labeled market
data, immutable live forecasts, structured readiness diagnostics, and consistent
CLI, API, and frontend behavior.

The API remains a serialization boundary. Model inference, market ingestion,
prediction composition, edge calculation, and forecast selection occur before
API serialization.

### Motivation

The current runtime path is composed from incompatible contracts:

- `weekly-predict` generates and archives Elo win probabilities only.
- `weekly-predict --model-type` selects archive rows for edge generation but
  does not select the model used to generate weekly predictions.
- `/games` resolves the current win-probability champion and falls back to Elo.
- `/edges` and `gridiron edges report` resolve the champion without the same
  fallback.
- Total predictions are archived independently and are not composed with
  win-probability predictions.
- Edge generation reuses the win-model type when resolving total uncertainty.
- The prediction archive keeps only the latest row for each
  `(game_id, model_name, model_type)` and can replace live forecasts with later
  backfills.
- `post-week` performs a season-level walk-forward backfill rather than settling
  the immutable forecast issued before kickoff.
- DraftKings ingestion is no longer a reliable current source. The previously
  functional endpoint now appears to be blocked by Cloudflare human
  verification, and its undocumented URL format required periodic annual
  updates. The adapter should be preserved, but the production workflow must
  not depend on forcing it to work.
- nflverse upcoming schedule data already contains useful market context, but
  the current cleaning path discards it.
- Existing verification checks code health but not weekly prediction readiness,
  schedule coverage, market freshness, or prediction-to-market join coverage.

### Design Principles

1. Scheduled games are the denominator. Missing predictions or markets must not
   remove games from the weekly product.
2. Live forecasts and historical backtests are separate artifact roles.
3. Live forecast events are immutable.
4. Multiple forecast runs for the same game and model may coexist.
5. A selected current forecast is explicit, not implied by latest write order.
6. Win-probability and total models retain independent identities.
7. Week-specific model availability policy is explicit and serialized.
8. Market data records source and ingestion time.
9. Missing predictions, missing markets, join failures, incomplete markets, and
   no positive edges are distinct states.
10. Rendering is a pure consumer and must not generate or archive predictions.
11. CLI commands retain distinct operational intents but share domain services.
12. Tests accompany every implementation unit.
13. Working-unit labels belong in planning documentation only, not in source
    comments, docstrings, test names, or runtime identifiers.
14. Gridiron Edge has never been live and has no production compatibility
    obligations. New contracts may replace existing schemas, artifacts, commands,
    tests, and development-era behavior outright. Do not add migrations,
    compatibility readers, aliases, deprecation periods, or dual-write paths
    unless they serve a current development need.

---

## Intended CLI Responsibilities

### `gridiron run-data-pipeline`

Refresh foundational datasets, Elo state, and historical modeling inputs.

- DraftKings is removed from the default no-flags path.
- External market ingestion is explicit and source-neutral.
- Staleness checks use dataset-registry paths.
- Weather defaults and documented examples must agree.

### `gridiron weekly-predict`

Primary pregame workflow.

- Refresh required inputs.
- Resolve an availability-aware prediction policy.
- Produce a schedule-complete weekly product.
- Optionally attach source-labeled market data.
- Calculate edge results and diagnostics.
- Publish outputs.
- Verify weekly readiness.

The normal command should not generate one model and request another model's
archive rows for edge calculation.

### `gridiron output predictions`

Pure rendering/export command.

- Load an existing weekly product.
- Render requested formats.
- Do not run inference.
- Do not modify prediction storage.

### `gridiron post-week`

Completed-week closeout workflow.

- Refresh completed results.
- Join outcomes to immutable live forecasts.
- Evaluate the forecasts actually issued before kickoff.
- Refresh next-week state.
- Report incomplete closeout coverage.

Historical backfill does not belong in this command.

### `gridiron evaluate backfill`

Historical reconstruction and model-comparison workflow.

- Retain season-level walk-forward behavior for trained models.
- Retain chronological reconstruction for Elo.
- Store backfilled forecasts separately from live forecast events.
- Never replace live forecasts.

### `gridiron edges report`

Inspection/export command over the same weekly product and edge service used by
the API and `weekly-predict`.

### `gridiron verify`

Code-health verification.

### `gridiron verify-week`

Read-only operational readiness verification for one season and week.

---

## Implementation Units

### Unit 1: Define Forecast and Weekly Product Identity [Complete]

#### Completed

Defined immutable forecast-event, selected-forecast, and weekly-product
identities. Live and backfilled forecasts have distinct roles, while each
forecast event has its own event ID, shared invocation-level run ID, game and
model identity, and required UTC generation timestamp.

Selected-forecast references preserve the identity of the exact forecast event
chosen for downstream composition. Weekly-product identity records the product
ID, run ID, season, week, and UTC generation timestamp independently from
storage and selection behavior.

No archive persistence, deduplication, forecast selection, or product-storage
behavior was introduced.

#### Goal

Establish storage-independent identity contracts for immutable forecast events,
explicit forecast selection, and static weekly products before changing
persistence behavior.

#### Tests

- live and backfilled forecast roles are distinct;
- invalid role values are rejected;
- event identity is distinct from game and model identity;
- multiple events for the same game and model can have different event IDs;
- one run ID can group multiple forecast events;
- generated timestamps are required, timezone-aware, and UTC;
- selected forecasts reference exact event identities;
- weekly products retain product, run, season, week, and generation identity;
- identity contracts are immutable;
- identity contracts contain no storage or selection behavior.

#### Acceptance

Forecast events, selected forecasts, and weekly products have explicit,
immutable, storage-independent identities suitable for the event-preserving
storage and selection layers that follow.

---

### Unit 2: Preserve Multiple Forecast Events in Storage [Complete]

#### Completed

Introduced a strict, event-preserving Parquet store for immutable game forecast
events. Each forecast has its own event ID, while forecasts generated by one
invocation share a run ID, explicit live or backfilled role, and timezone-aware
UTC generation timestamp.

Added a pure composition boundary that maps canonical game prediction rows into
the forecast-event schema without performing inference or persistence.
Historical classification, regression, and Elo producers now return canonical
game identity and prediction values without embedding storage metadata.

Cut both game forecast writer paths over to immutable event storage:

- historical backfill invocations create distinct backfilled forecast runs;
- weekly Elo prediction invocations create distinct live forecast runs;
- repeated runs for the same game and model coexist;
- prior live and backfilled forecasts are never replaced or skipped;
- obsolete backfill overwrite semantics were removed;
- rendering retains its original display frame without creating additional
  forecast events;
- `output predictions` no longer writes prediction storage.

The development-era latest-write-wins archive is no longer written by game
forecast workflows. Its remaining read-only consumers and obsolete module will
be removed after explicit current-forecast selection is available, preventing
those consumers from interpreting multiple immutable events as one implicitly
selected forecast.

#### Goal

Replace game/model-pair deduplication with immutable forecast-event storage so
multiple live and historical reconstruction runs can coexist without deleting
or replacing prior forecasts.

#### Tests

- canonical schema validation rejects missing and unknown columns;
- required identity fields reject null and empty values;
- roles, week values, and UTC generation timestamps are validated;
- every forecast row receives a distinct event ID;
- all events from one invocation share a run ID and generation timestamp;
- identical event retries are idempotent;
- event ID reuse with different content is rejected;
- multiple live events for the same game and model coexist;
- live and backfilled events coexist;
- repeated backfill runs retain distinct run and event identities;
- canonical prediction composition preserves available forecast values;
- unavailable optional outputs remain null;
- source prediction frames are not mutated;
- historical classification, regression, and Elo producers retain game
  identity, team orientation, dates, and available prediction values;
- neutral-site orientation remains deterministic;
- weekly Elo output maps to canonical live events without mutating the display
  frame;
- separate weekly invocations receive separate run identities;
- rendering produces PNG and HTML outputs without writing forecast storage.

#### Acceptance

Game forecast persistence is event-preserving. No forecast is deleted, replaced,
or skipped merely because another event has the same game and model identity.
Live and backfilled invocations retain explicit, immutable event and run
identity, and game forecast rendering has no persistence side effects.

---

### Unit 3: Add Explicit Current-Forecast Selection [Complete]

#### Completed

Added pure, storage-independent selection for immutable forecast events.

Exact selection resolves `SelectedForecast` references by event ID, verifies
their game and model identity, preserves request order, and reports missing
references without substituting another event.

Run-scoped selection resolves one explicitly requested invocation, preserves
independent prediction families, rejects duplicate game and model identities
within a run, and returns deterministic output without treating ordering as
selection.

Candidate resolution represents selected, missing, and ambiguous states
explicitly. Live events exclude backfilled candidates for the same game and
model identity, but multiple eligible live runs remain ambiguous. When no live
event exists, a single backfilled event may be selected, while multiple
backfilled runs remain ambiguous.

No selector infers current state from generation time, event ID, input order,
Parquet order, or model priority. No model inference or persistence occurs
during selection.

#### Goal

Produce explicit selected-forecast views from immutable events without relying
on latest-write deduplication or implicit recency.

#### Tests

- exact event references select only the requested immutable event;
- reference order is preserved;
- missing references remain visible;
- duplicate references and identity conflicts are rejected;
- exact run selection excludes all other runs;
- missing runs return a canonical empty result;
- game and model identities are unique within a selected run;
- the same game and model may coexist across separate runs;
- win and total forecast families remain independently selectable;
- live candidates exclude matching backfilled candidates;
- newer backfills do not override live forecasts;
- multiple live runs remain ambiguous;
- multiple backfilled runs remain ambiguous when no live event exists;
- ambiguity is independent from timestamp and input order;
- candidate-resolution invariants are validated;
- selectors do not mutate source event frames;
- no selector performs model inference or storage writes.

#### Acceptance

Immutable forecast events can be selected by exact event identity, exact run
identity, or explicit candidate eligibility. Selected, missing, and ambiguous
states are machine-readable, and no current forecast is inferred from write
order, recency, or model priority.

---

### Unit 4: Define Weekly Readiness Diagnostics [Complete]

#### Completed

Added immutable weekly readiness contracts and a pure evaluator for schedule,
prediction, market, provenance, join, and edge coverage.

Readiness records the scheduled-game count; selected win-prediction, spread,
total, projected-score, and model-provenance coverage; market-covered games;
prediction-to-market matches; eligible game-market pairs; and positive-edge
count.

The evaluator distinguishes missing, partial, unmatched, and incomplete states
with machine-readable blockers. It scopes schedule, prediction, and market
inputs to the requested season and week, rejects duplicate game identities,
and validates required schemas before calculating readiness.

Eligible markets represent complete, calculable game-market pairs rather than
raw market-side rows. Moneyline requires both prices and a win probability;
spread requires a model spread, home line, and both prices; total requires a
model total, total line, and both prices.

Prediction and market artifact provenance is explicit. Unique UTC generation
and fetch timestamps are preserved, while missing or mixed provenance remains
visible rather than being resolved through recency. Market source is retained
only when exactly one non-empty source is present.

Zero positive edges is a valid analytical result and does not block readiness.
The evaluator performs no file I/O, forecast selection, prediction generation,
market ingestion, edge calculation, or input mutation.

#### Goal

Provide quantitative, machine-readable diagnostics that distinguish a valid
weekly analytical result from missing data, partial coverage, unmatched inputs,
incomplete markets, and unavailable provenance.

#### Tests

- complete weekly inputs produce a ready result with exact coverage counts;
- zero positive edges remains ready and is distinct from missing inputs;
- scheduled games remain the denominator for game-level coverage;
- sixteen scheduled games and fifteen predictions report partial coverage;
- missing and partial win, spread, total, projected-score, and provenance
  coverage are distinct;
- no predictions and no market data produce different blockers;
- zero prediction-to-market matches and incomplete markets are distinct;
- partial market coverage retains quantitative game and match counts;
- eligible markets count complete game-market pairs rather than raw sides;
- positive edges use the strict `ev > 0` rule;
- prediction and market artifact timestamps require timezone-aware UTC;
- missing prediction and market provenance remain visible;
- mixed market sources or fetch timestamps are reported as ambiguous;
- no timestamp or source is selected through recency;
- duplicate schedule and prediction game IDs are rejected;
- missing required input columns are rejected;
- non-empty edge inputs require an EV column;
- invalid scope and count relationships are rejected;
- readiness results are immutable;
- source DataFrames are not mutated.

#### Acceptance

Weekly readiness exposes exact game, prediction, market, match, eligibility,
provenance, and positive-edge counts with structured blockers. Missing data,
partial coverage, zero joins, incomplete markets, and valid zero-edge outcomes
cannot be confused with one another, and no readiness value is inferred from
filesystem metadata or implicit recency.

---

### Unit 5: Add Read-Only `verify-week` CLI [Complete]

#### Completed

Added a top-level `verify-week` command for read-only weekly operational
diagnostics, separate from the existing code-health `verify` workflow.

The command requires an NFL season and week, validates season continuity and
the supported week range, and optionally accepts an exact forecast run ID.
When a run ID is supplied, readiness uses only that invocation. Without one,
the current win-probability champion identifies the requested model type and
eligible forecast events are resolved without using recency, write order, or
event identity as an implicit tie-breaker.

The command reads the existing upcoming schedule, immutable forecast events,
current market snapshot, and persisted calibration metadata. It may calculate
an in-memory edge report from those existing inputs, but it does not fetch,
refresh, generate, enrich, render, or persist artifacts.

Output includes every schedule, prediction, model-output, provenance, market,
join, eligible-market, and positive-edge count. Prediction and market artifact
timestamps and market source are displayed when available. All readiness
blockers are printed using their machine-readable values.

Complete readiness exits successfully. Blocked readiness exits nonzero while
retaining the full diagnostic report. A valid result with zero positive edges
exits successfully.

The new command is registered independently from `verify`, and both appear in
top-level CLI help.

#### Goal

Expose weekly operational readiness independently from code-health verification
without fetching, modifying, or silently completing data.

#### Tests

- `verify-week` is registered on the top-level CLI;
- `verify` and `verify-week` both appear in top-level help;
- season and week are required;
- malformed and nonconsecutive season labels are rejected;
- weeks outside the supported range are rejected;
- an exact run ID is forwarded to run-scoped selection;
- complete readiness exits successfully;
- missing forecast selection exits nonzero;
- ambiguous forecast selection remains explicit;
- all diagnostic counts are rendered;
- prediction and market provenance are rendered;
- unavailable provenance is displayed explicitly;
- every blocker is rendered;
- valid zero-positive-edge readiness exits successfully;
- existing schedule, forecast, market, and calibration artifacts are read
  without mutation;
- no forecast, odds, archive, prediction-generation, enrichment, PNG, or HTML
  writer is imported by the command;
- read-only readiness assembly performs no writes.

#### Acceptance

Weekly operational readiness can be checked with `gridiron verify-week`
independently from code-health verification. The command reports complete
quantitative diagnostics and exits according to readiness without fetching,
generating, enriching, rendering, or modifying project data.

---

### Unit 6: Preserve Rich Upcoming Schedule Data [Complete]

#### Completed

Added a registry-backed, typed Parquet artifact for rich upcoming schedule
data while preserving the existing focused Elo schedule CSV as a compatibility
boundary.

The rich transform retains every unplayed nflverse schedule row and preserves
canonical season, week, game ID, kickoff date and time, away and home teams,
location and neutral-site context, stadium, roof, surface, divisional state,
team rest, available Moneyline, Spread, and Total fields, source identity, and
a shared timezone-aware UTC ingestion timestamp.

Optional venue, context, rest, and market fields remain nullable. Missing
Moneyline, Spread, Total, stadium, roof, surface, location, or rest values do
not remove scheduled games.

The existing Elo schedule is projected from the rich normalized rows and
retains its original eight-column schema, registered CSV path, loader, game
identity, and row coverage. Existing API and Elo-oriented consumers continue
using the focused schedule.

Weekly readiness verification now reads the rich schedule artifact and adapts
its canonical lowercase game identity into the readiness evaluator’s stable
schedule interface. It does not fall back silently to the focused Elo
artifact when the rich artifact is missing.

Both cleaned outputs are written through registered dataset writers. The rich
artifact path is defined only in the dataset registry, and its dedicated
loader does not fall back to the legacy schedule.

#### Goal

Create a model-ready, schedule-complete upcoming artifact without
destabilizing the focused Elo schedule consumer.

#### Tests

- the rich artifact resolves through one registered Parquet path;
- the rich loader does not fall back to the focused Elo CSV;
- nullable numeric values and UTC ingestion timestamps survive Parquet
  round-trip storage;
- every unplayed source schedule row survives rich cleaning;
- no scheduled game is dropped because market values are missing;
- venue, context, rest, and market fields remain nullable;
- canonical season, week, team names, and game IDs are preserved;
- neutral-site and location context are retained;
- one source and ingestion timestamp are shared across one build;
- empty source input produces stable typed rich and focused schemas;
- the focused Elo schedule retains its original columns and ordering;
- rich and focused artifacts contain identical canonical game-ID sets;
- completed source rows do not enter upcoming artifacts;
- both outputs use registered dataset writers;
- no duplicate rich artifact path exists outside the registry and its path
  assertion;
- `verify-week` reads the rich schedule and does not use a legacy fallback;
- existing API consumers continue using the focused schedule;
- all Unit 6-specific Ruff, Pyrefly, and pytest gates pass.

#### Acceptance

The repository has a typed, schedule-complete upcoming artifact suitable for
weekly composition and readiness diagnostics. Every unplayed source game is
preserved regardless of optional market coverage, while the focused Elo
schedule remains compatible and is derived from the same normalized rows.

---

### Unit 7: Fix Synthetic Upcoming Week 1 Elo Transition

#### Goal

Make upcoming Week 1 state deterministic and consistent with historical season
transition semantics.

#### Production files

- `src/gridiron_edge/ratings/elo/table.py`
- possibly `src/gridiron_edge/ratings/elo/simulator.py` if a shared transition
  helper is extracted

#### Test files

- update:
  `tests/unit/ratings/test_elo_table.py`
- update:
  `tests/unit/ratings/test_elo_core.py` only if shared math changes

#### Tests

- next season is derived from the latest historical season, not wall-clock year;
- the final postseason update is included;
- returning teams receive the intended offseason regression;
- the transition is reproducible from identical historical input;
- no arbitrary future weeks are fabricated;
- expansion behavior remains correct;
- existing historical pregame week semantics remain unchanged.

#### Acceptance

Upcoming Week 1 Elo uses a tested transition policy consistent with historical
evaluation semantics.

---

### Unit 8: Consolidate Elo Weekly Prediction Logic

#### Goal

Replace duplicate schedule-to-Elo joins with one domain function.

#### Production files

- `src/gridiron_edge/ratings/elo/predict.py`
- `src/gridiron_edge/viz/predictions.py`
- callers in `src/gridiron_edge/cli/`

#### Test files

- update Elo prediction unit tests
- update visualization prediction-builder tests
- update CLI tests for `ratings elo predict` and `output predictions`

#### Tests

- one function produces numeric away/home probabilities;
- schedule rows remain present when Elo is missing;
- missing Elo is represented through status rather than row deletion;
- neutral-site schedule identity is preserved;
- all callers receive the same schema;
- probability complements sum to one within tolerance.

#### Acceptance

There is one schedule-to-Elo prediction implementation.

---

### Unit 9: Define Availability-Aware Prediction Policy

#### Goal

Resolve prediction behavior by data availability without silently selecting a
model that cannot produce the requested week.

#### Production files

- new policy module under `src/gridiron_edge/models/game_prediction/`
- champion resolver integration as a read-only dependency

#### Test files

- create:
  `tests/unit/models/game_prediction/test_prediction_policy.py`

#### Tests

- Week 1 chooses only an eligible policy;
- in-season full-feature champion is selected only when required inputs exist;
- unavailable total prediction remains explicit;
- overrides are independently scoped to win and total models;
- policy output records rationale and model provenance;
- no API-layer inference or fallback is involved.

#### Acceptance

Model selection is explicit, availability-aware, and serializable.

---

### Unit 10: Build Schedule-Complete Win Prediction Component

#### Goal

Attach selected win probabilities to every scheduled game.

#### Production files

- new weekly product builder under
  `src/gridiron_edge/models/game_prediction/` or a dedicated domain package

#### Test files

- create:
  `tests/unit/models/game_prediction/test_weekly_win_product.py`

#### Tests

- output has exactly one row per scheduled game;
- missing win prediction does not remove the game;
- win-model identity is present for available predictions;
- live forecast event identity is preserved;
- selected forecast is explicit;
- probabilities and team orientation are validated;
- status is present for unavailable rows.

#### Acceptance

The weekly product has schedule-complete Moneyline probability coverage status.

---

### Unit 11: Attach Derived Spread Component

#### Goal

Derive model spread and spread uncertainty from the selected win component using
the correct win-model calibration identity.

#### Production files

- weekly product builder
- existing game post-processing module

#### Test files

- create or update:
  `tests/unit/models/game_prediction/test_weekly_spread_product.py`

#### Tests

- spread sign convention is fixed and documented;
- model spread uses the selected win model's calibration;
- no spread is fabricated when calibration is unavailable;
- spread provenance identifies the source win forecast and calibration;
- schedule completeness is preserved.

#### Acceptance

Every game has either a valid derived spread or an explicit blocker.

---

### Unit 12: Attach Independent Total Component

#### Goal

Compose selected total-model predictions without conflating total identity with
the win model.

#### Production files

- weekly product builder
- total prediction selector or loader

#### Test files

- create or update:
  `tests/unit/models/game_prediction/test_weekly_total_product.py`

#### Tests

- total champion resolves independently from the win champion;
- total uncertainty uses the total model identity;
- missing total prediction does not remove the game;
- total-model provenance is preserved;
- independent total archive rows compose correctly by game ID;
- no matching algorithm between win and total is assumed.

#### Acceptance

Total predictions are independently resolved and explicitly composed.

---

### Unit 13: Add Projected Scores and Product-Level Validation

#### Goal

Create projected scores only when required spread and total values exist.

#### Production files

- weekly product builder
- product validation module

#### Test files

- create or update:
  `tests/unit/models/game_prediction/test_weekly_game_product.py`

#### Tests

- projected scores reconcile to model total;
- projected score difference reconciles to spread convention;
- missing inputs produce explicit field status;
- invalid combinations are rejected;
- complete product remains one row per scheduled game.

#### Acceptance

The composed game product has internally coherent values and granular status.

---

### Unit 14: Persist the Static Weekly Game Product

#### Goal

Write and load an immutable, versioned weekly product.

#### Production files

- new storage module
- dataset registry entries
- relevant loaders and writers

#### Test files

- create:
  `tests/unit/models/game_prediction/test_weekly_product_store.py`
- create or update:
  `tests/integration/models/test_weekly_product_roundtrip.py`

#### Tests

- product round-trips without schema loss;
- run identity and generated timestamp survive;
- multiple runs created under the new contract coexist;
- selected current product is explicit;
- schema mismatch fails clearly;
- no model computation occurs during load.

#### Acceptance

API, CLI rendering, and edge calculation can consume one persisted weekly
product.

---

### Unit 15: Add Source-Labeled nflverse Market Adapter

#### Goal

Convert available nflverse schedule market fields into the generic market
contract.

#### Production files

- new adapter under `src/gridiron_edge/ingest/odds/` or renamed market package
- `src/gridiron_edge/ingest/odds/store.py`

#### Test files

- create:
  `tests/unit/ingest/odds/test_nflverse_adapter.py`
- update:
  `tests/integration/ingest/odds/test_odds_join.py`

#### Tests

- source is labeled `nflverse_schedule` or the locked equivalent;
- no DraftKings label is applied;
- ingestion timestamp is recorded;
- Moneyline, Spread, and Total sides normalize correctly;
- game IDs are validated against schedule truth;
- incomplete markets remain explicit;
- requested season/week scope is enforced.

#### Acceptance

Current market comparison no longer depends on the unreliable DraftKings pull.

---

### Unit 16: Reclassify DraftKings as Legacy Best-Effort Adapter

#### Goal

Preserve the historical adapter without presenting it as the default recovery
path.

#### Production files

- DraftKings adapter and CLI wrapper
- operational messages in API and frontend

#### Test files

- update DraftKings parser and CLI tests
- update API blocker-message tests
- update affected frontend empty-state tests

#### Tests

- failure does not block core data refresh;
- Cloudflare or non-JSON responses fail clearly;
- no bypass behavior is introduced;
- command help identifies the adapter as best-effort or legacy;
- normal workflow does not invoke it by default;
- stale recovery guidance is removed.

#### Acceptance

DraftKings code remains available but is not a production dependency.

---

### Unit 17: Build Unified Edge Diagnostics

#### Goal

Create structured coverage and result-state diagnostics before changing edge
math callers.

#### Production files

- `src/gridiron_edge/market/recommendations.py`
- new diagnostic types if appropriate

#### Test files

- update:
  `tests/unit/market/test_recommendations.py`
- create:
  `tests/unit/market/test_edge_diagnostics.py`

#### Tests

- no predictions;
- no market data;
- stale or wrong-scope market data;
- zero matched games;
- incomplete markets;
- no positive edges;
- positive edges;
- counts are derived from actual inputs;
- diagnostics retain win, total, and market provenance.

#### Acceptance

An empty edge table always has an explicit reason.

---

### Unit 18: Build Unified Weekly Edge Service

#### Goal

Use the persisted weekly product and source-labeled market snapshot from one
domain service.

#### Production files

- new service under `src/gridiron_edge/market/`
- existing recommendation functions remain pure math helpers

#### Test files

- create:
  `tests/unit/market/test_weekly_edge_service.py`
- update relevant odds-join integration tests

#### Tests

- schedule and game IDs align;
- Moneyline uses selected win probability;
- Spread uses derived spread and win-model calibration;
- Total uses independent total prediction and uncertainty;
- bankroll omission leaves dollar stake unavailable;
- explicit bankroll produces stake values;
- diagnostics remain correct after minimum-EV filtering.

#### Acceptance

CLI and API can consume the same edge result and diagnostics.

---

### Unit 19: Rewire `weekly-predict`

#### Goal

Make `weekly-predict` orchestrate the new domain services.

#### Production files

- `src/gridiron_edge/cli/weekly_predict.py`
- shared composite helpers only where generally reusable

#### Test files

- update:
  `tests/unit/cli/test_weekly_predict.py`
- update or add an end-to-end weekly workflow test

#### Tests

- prediction policy controls actual generated predictions;
- independently scoped overrides work if retained;
- missing market data does not fail prediction generation;
- default bankroll is absent;
- published outputs reference the persisted weekly product;
- readiness diagnostics appear in the result;
- `--skip` and `--only` examples match dependency rules;
- stale edge files are not presented as current.

#### Acceptance

One command produces a truthful, schedule-complete pregame product.

---

### Unit 20: Make `output predictions` a Pure Renderer

#### Goal

Render an existing weekly product without inference or storage mutation.

#### Production files

- `src/gridiron_edge/cli/output.py`
- `src/gridiron_edge/viz/predictions.py`

#### Test files

- update:
  `tests/unit/cli/test_output.py`
- update visualization tests

#### Tests

- rendering does not write forecast events;
- rendering does not change selected product state;
- missing product exits nonzero;
- supported formats are validated;
- repeated rendering is deterministic for the same product.

#### Acceptance

Output generation is a pure downstream operation.

---

### Unit 21: Rewire `edges report`

#### Goal

Use the unified edge service and return correct exit semantics.

#### Production files

- `src/gridiron_edge/cli/edges.py`

#### Test files

- update:
  `tests/unit/cli/test_edges.py`

#### Tests

- missing weekly product exits nonzero;
- missing market snapshot exits nonzero;
- zero joins exits nonzero;
- valid zero-positive-edge result exits successfully;
- `--format` is validated;
- bankroll and Kelly multiplier are validated at the CLI boundary;
- CSV output cannot silently remain stale.

#### Acceptance

Direct edge inspection matches `weekly-predict` and API behavior.

---

### Unit 22: Rebuild `post-week` Around Live Forecast Closeout

#### Goal

Evaluate immutable live forecasts rather than running historical backfill.

#### Production files

- `src/gridiron_edge/cli/post_week.py`
- new closeout service under `src/gridiron_edge/evaluation/`

#### Test files

- update:
  `tests/unit/cli/test_post_week.py`
- create:
  `tests/unit/evaluation/test_live_forecast_closeout.py`
- add an integration test for forecast-to-outcome joining

#### Tests

- exact requested week is closed;
- live forecasts are selected explicitly;
- backfilled forecasts are excluded;
- missing live forecasts remain visible;
- missing outcomes remain visible;
- schedule and forecast coverage counts reconcile;
- no historical backfill is invoked;
- next-week state refresh is tested independently;
- documented `--skip` and `--only` examples are executable.

#### Acceptance

`post-week` evaluates what the system actually forecast before kickoff.

---

### Unit 23: Harden Historical Backfill

#### Goal

Keep `evaluate backfill` as the explicit historical reconstruction workflow.

#### Production files

- `src/gridiron_edge/evaluation/backfill.py`
- `src/gridiron_edge/cli/evaluate.py`

#### Test files

- update:
  `tests/unit/evaluation/test_backfill.py`
- update:
  `tests/unit/cli/test_evaluate.py`

#### Tests

- mode is validated as an enum;
- live rows are never replaced;
- generated count and inserted count are distinguished;
- season-level walk-forward boundaries remain correct;
- zero generated rows are visible;
- classification and regression backfills retain task-appropriate provenance;
- skipped seasons are identified.

#### Acceptance

Historical backtests remain rigorous and cannot alter live forecast history.

---

### Unit 24: Make API Games Schedule-First

#### Goal

Serialize the persisted weekly product for every scheduled game.

#### Production files

- `src/gridiron_edge/api/loaders.py`
- game API schemas
- game routes

#### Test files

- update:
  `tests/unit/api/test_loaders.py`
- update game route tests
- update API integration tests

#### Tests

- every scheduled game is returned;
- missing predictions produce status rather than disappearance;
- no runtime model fallback occurs in the API;
- win and total provenance are serialized separately;
- spread sign documentation matches runtime behavior;
- valid scheduled game detail does not return 404 because prediction is missing.

#### Acceptance

`/games` is a pure schedule-complete serialization surface.

---

### Unit 25: Make API Edges Use Unified Service Results

#### Goal

Serialize shared edge rows and diagnostics.

#### Production files

- `src/gridiron_edge/api/loaders.py`
- `src/gridiron_edge/api/schemas/edges.py`
- `src/gridiron_edge/api/routes/edges.py`

#### Test files

- update edge schema, loader, and route tests

#### Tests

- sportsbook or source and ingestion timestamp are included;
- diagnostics distinguish all empty states;
- model and market provenance are exposed;
- edge strength enum matches implementation;
- API output matches direct CLI service output.

#### Acceptance

`/edges` no longer reconstructs model composition independently.

---

### Unit 26: Regenerate API Clients

#### Goal

Update generated contracts after API schemas stabilize.

#### Production files

- `api-schema.json`
- `frontend/src/api/schema.ts`

#### Test files

- generated-schema consistency checks
- frontend type-check/build coverage

#### Acceptance

Generated clients match the finalized API contract with no manual drift.

---

### Unit 27: Wire Frontend Weekly Status

#### Goal

Make missing or blocked weekly data visible across all game surfaces.

#### Production files

- Games list screen
- Game detail screen
- Dashboard featured matchups
- Dashboard model edges
- BetSlip edges table
- shared field-status components where appropriate

#### Test files

- update corresponding `*.test.tsx` files
- add focused tests for each blocker state

#### Tests

- no schedule rows disappear because prediction is unavailable;
- missing prediction is not shown as “No games found”;
- missing market data is not shown as “No play”;
- no positive edge remains a valid “No play” state;
- synthetic uncertainty bands are removed;
- market values render from real market context;
- pending and blocked states remain visibly identifiable.

#### Acceptance

The frontend reflects actual product readiness rather than silent nulls.

---

### Unit 28: Update Data Pipeline and Verification CLI Contracts

#### Goal

Correct remaining CLI semantics and documentation.

#### Production files

- `src/gridiron_edge/cli/main.py`
- `src/gridiron_edge/cli/verify.py`
- related help text and operational documentation

#### Test files

- update:
  `tests/unit/cli/test_main.py`
- update:
  `tests/unit/cli/test_verify.py`

#### Tests

- default pipeline does not require DraftKings;
- no-flags behavior matches documentation;
- staleness paths come from the registry;
- staleness warning wording is correct;
- smoke verification behavior matches its documentation;
- code verification remains separate from weekly readiness;
- frontend gates are either intentionally included or explicitly documented as
  outside this command.

#### Acceptance

CLI help, stage behavior, and exit semantics agree.

---

### Unit 29: Documentation and Operational Closeout

#### Goal

Document the final weekly lifecycle and remove obsolete guidance.

#### Documentation files

- `HANDOFF.md`
- `ROADMAP.md`
- `CHANGELOG.md`
- relevant architecture or decision documentation
- `PLAN.md`

#### Required updates

- live versus backfilled forecast roles;
- weekly product schema and storage path;
- model availability policy;
- supported market source;
- legacy DraftKings status;
- `weekly-predict` workflow;
- `post-week` workflow;
- `verify` versus `verify-week`;
- API serialization boundary;
- operational recovery guidance.

#### Acceptance

Documentation traces the implemented behavior, commands, artifacts, and known
limitations without stale work-unit nomenclature in runtime source.

---

## Quality Gates

Run focused gates after each unit.

### Python units

```bash
uv run ruff check <changed paths>
uvx pyrefly check
uv run pytest <focused test files> -q
```

Run the full relevant Python suite at contract boundaries:

after forecast storage and selection;
after weekly product persistence;
after edge service integration;
after CLI rewiring;
after API rewiring.
Frontend units
pnpm test
pnpm build


Use focused frontend tests during each UI unit and the full frontend suite after generated schema and screen wiring changes.

Final workstream gates
uv run ruff check .
uvx pyrefly check
uv run pytest -q
pnpm test
pnpm build


Also run a read-only verify-week check against a known fixture-backed season and week.

Out of Scope
Bypassing Cloudflare human verification.
Connecting to a sportsbook account or placing bets.
Treating DraftKings as the required current market source.
Rebuilding simulation and playoff internals unless a shared-contract change requires it.
Optimizing model champions for short-sample betting ROI.
Frontend visual redesign unrelated to truthful weekly product status.
Arbitrary multiweek future prediction before required state exists.
- Preserving compatibility with development-era prediction archives, generated
  artifacts, CLI behavior, or schemas that have never been used in production.


Definition of Done

The workstream is complete when:

Every scheduled game appears in the weekly product.
Live forecasts are immutable and coexist with historical backfills.
Current forecast selection is explicit.
Win and total model identities remain independent.
Week 1 and in-season prediction policies are explicit.
Current market data has source and ingestion provenance.
Edge results distinguish blockers from valid zero-edge outcomes.
weekly-predict, edges report, and /edges use one edge service.
post-week evaluates immutable live forecasts.
evaluate backfill remains the historical reconstruction workflow.
API routes serialize persisted products without runtime inference.
Frontend surfaces visibly distinguish pending, blocked, unavailable, and no-play states.
verify-week reports schedule, prediction, market, and edge coverage.
Python and frontend quality gates pass.
Operational documentation matches the implemented lifecycle.

---

## Paused Workstreams

#### W9.11 Tier 0: Final frontend audit — ⏸️ PAUSED

Paused in favor of W14 Game Prediction Season Readiness. Resume after the
moneyline, spread, total, odds-ingestion, edge, API, and focused game-day
frontend path is operationally verified.

The deferred BetSlip real-data review moves naturally into W14 Tier 7 because
that rehearsal should produce the real edge recommendations required to test
the staged-wager presentation.

---

## Changelog

| Date | Change |
|------|--------|
| 2026-07-30 | **W14 opened.** Paused W9.11 Tier 0 and made Game Prediction Season Readiness the active workstream. Locked the vertical-slice goal across upcoming-game features, moneyline/spread/total predictions, archives, odds ingestion, joins, edges, API contracts, Dashboard, Games, GameDetail, BetSlip, and a real weekly rehearsal. Tier 0 is a read-only current-state audit before implementation. |
| 2026-07-28 | Closed the PlayoffProjections navigation and Weekly Outcomes follow-up after real-data verification. |
| 2026-07-12 | **Doc-sync pass + PLAN reset.** Normalized planning docs after the frontend arc; PLAN reset to between-workstreams with a next-candidates list (audit sweep recommended). |
| 2026-07-11 | **W9.10 complete.** Both Compare modes shipped on backend B1–B4. Fixed game_id scramble, clean-games clobber, Elo empty-games crash; added champion→elo fallback + upcoming-Week-1 season resolver. |
| 2026-07-11 | **W9.10 status resync.** Team vs Team complete (6 alignment adjustments + 11-metric cohort_splits). Player vs Defense redesigned to independent pickers + bar chart + baseline verdict. |
| 2026-07-11 | **W9.10 design locked.** Two modes; highlight discipline baked in from W9.8. |
