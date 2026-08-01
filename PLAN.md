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

### Unit 7: Fix Synthetic Upcoming Week 1 Elo Transition [Complete]

#### Completed

Reworked synthetic upcoming Week 1 Elo generation to use the same deterministic
offseason transition policy as historical season boundaries.

Extracted a pure next-season transition helper from the Elo simulator.
Returning teams regress toward the mean using the configured offseason
regression fraction, and teams whose configured expansion season matches the
target season receive the configured expansion rating.

Historical season transitions continue using the same policy through the shared
helper. The Elo update formula, per-game prediction behavior, K-factor handling,
and historical pregame week semantics remain unchanged.

Synthetic Week 1 now derives its target season directly from the latest
historical season label. It no longer reads the current date or wall-clock
season.

The synthetic transition starts from each team’s rating at the final historical
season’s `max_week + 1` state. This is the postgame state produced after the
last played game, including the final postseason update, rather than the
pregame rating entering the final week.

Only the derived next season’s Week 1 state is added. No later weeks,
additional future seasons, or schedule-dependent states are fabricated.
Existing historical Elo rows are not modified.

#### Goal

Make upcoming Week 1 Elo deterministic and consistent with historical season
transition semantics.

#### Tests

- the next season label is derived from the latest historical season;
- malformed and nonconsecutive historical season labels are rejected;
- transition output is independent of the wall-clock year;
- the final postseason update is included through the final postgame state;
- the final week’s pregame rating is not reused as the next season rating;
- returning teams receive the configured offseason regression;
- historical and synthetic transitions use the same transition policy;
- identical historical inputs produce identical Week 1 output;
- only the derived next season’s Week 1 rows are created;
- no arbitrary future weeks or seasons are fabricated;
- expansion teams receive the configured expansion rating in their start
  season;
- expansion teams from other seasons are not added;
- transition inputs are not mutated;
- existing historical Elo rows remain unchanged;
- Elo core update and Python/Numba parity tests remain unchanged and green.

#### Acceptance

Upcoming Week 1 Elo is derived deterministically from the latest historical
season’s final postgame state, including the final postseason result. Returning
teams receive the established offseason regression exactly once, expansion
behavior remains explicit, and no wall-clock or arbitrary future-week behavior
affects the result.

---

### Unit 8: Consolidate Elo Weekly Prediction Logic [Complete]

#### Completed

Added one canonical, row-preserving schedule-to-Elo prediction function and
routed all weekly Elo prediction callers through it.

The shared domain function scopes the focused upcoming schedule to the
requested season and week, joins away and home Elo state by canonical team,
season, and week identity, and calculates numeric away and home win
probabilities for games with complete ratings.

Schedule truth remains authoritative. Every scoped scheduled game remains in
the result even when one or both Elo ratings are unavailable. Missing away,
home, or both ratings remain null and are represented through explicit
machine-readable prediction statuses. Missing Elo is never replaced silently
with an initial or fallback rating.

Ready predictions expose numeric away and home probabilities whose complements
sum to one within floating-point tolerance. Human-readable percentage columns
are added by one shared formatting adapter without recalculating or modifying
the numeric probabilities.

Duplicate Elo identities for the same team, season, and week are rejected
before joining. Schedule order, game identity, team identity, kickoff fields,
and neutral-site or other schedule context pass through unchanged. Input
schedule and Elo frames are not mutated.

The file-based Elo prediction entry point now only loads registered schedule
and Elo datasets before calling the domain function. CSV output adds formatted
percentage fields and writes the resulting prediction frame.

Visualization delegates Elo prediction assembly to the domain entry point and
contains no independent Elo merge, missing-rating filter, or probability
formula. Its private display-table builder remains responsible only for logos,
short display names, time labels, ordering, and separator rows.

The `ratings elo predict`, `output predictions`, and `weekly-predict` workflows
all route through the same schedule-to-Elo implementation. Canonical live
forecast conversion preserves scheduled rows and nullable Elo and probability
values without expanding the immutable forecast-event schema.

#### Goal

Replace duplicate schedule-to-Elo joins and probability calculations with one
domain implementation shared by ratings, visualization, output, and composite
weekly prediction workflows.

#### Tests

- one domain function produces away and home Elo predictions;
- ready probabilities are numeric;
- away and home probabilities sum to one within tolerance;
- every requested schedule row remains present;
- missing away Elo receives an explicit status;
- missing home Elo receives an explicit status;
- missing both Elo ratings receives an explicit status;
- missing ratings and probabilities remain null;
- missing Elo is not replaced with a default rating;
- neutral-site and schedule identity fields are preserved;
- requested season and week scoping remains exact;
- schedule input order is preserved;
- duplicate Elo team, season, and week identities are rejected;
- missing required schedule and Elo columns are rejected;
- schedule and Elo input frames are not mutated;
- registered schedule and Elo loaders feed the domain entry point;
- percentage formatting preserves numeric probabilities;
- percentage formatting preserves missing values;
- percentage formatting does not mutate the domain result;
- visualization delegates to the domain prediction entry point;
- visualization contains no independent Elo merge or probability calculation;
- visualization preserves missing-Elo rows and status;
- `ratings elo predict` delegates to the shared CSV output path;
- `output predictions` delegates to the shared visualization adapter;
- `weekly-predict` delegates to the shared visualization adapter;
- canonical live forecast conversion preserves rows with missing Elo values;
- CLI-required arguments remain enforced;
- existing output and weekly prediction behavior remains compatible.

#### Acceptance

There is one schedule-to-Elo prediction implementation. All weekly Elo callers
receive the same row-preserving schema, numeric probabilities, and explicit
missing-rating behavior. Visualization and CLI modules perform no independent
Elo joins, probability calculations, or missing-row deletion.

---

### Unit 9: Define Availability-Aware Prediction Policy [Complete]

#### Completed

Added an immutable, deterministic, and serializable game-prediction policy
that resolves win-probability and total-model decisions independently from
explicit weekly input availability.

Availability records the requested season and week together with whether Elo
state, full win-prediction features, and total-prediction features are
available. The policy does not infer availability from model preference,
champion status, the current date, API behavior, or attempted model execution.

A full-feature champion is selected only when its required inputs are
available. When full win-prediction features are unavailable but Elo state is
available, the policy selects Elo explicitly and records that rationale.
Missing total inputs remain an explicit unavailable total decision rather than
borrowing the selected win model or silently omitting the decision.

Win and total overrides are independently scoped. A win override does not
change total selection, and a total override does not change win selection.
Overrides remain availability-aware. An explicitly requested model whose
required inputs are unavailable is rejected as ineligible rather than silently
falling back to another model.

Selected decisions include model identity, selection source, stable rationale,
human-readable explanation, and serializable provenance. Champion provenance
preserves promotion timestamp, source run ID, and task-flexible metrics.
Policy-owned Elo and explicit overrides record their distinct selection
sources.

Added a read-only champion assembly boundary using the persistent champion
manifest. Win and total champion entries are resolved independently. An
explicit override skips champion lookup only for its own family. Missing
manifest entries become explicit unavailable decisions, while malformed
champion metadata remains an error rather than being disguised as ordinary
unavailability.

The pure resolver performs no filesystem access, dataset inspection, feature
generation, model execution, clock lookup, or API inference. The read-only
assembly performs no artifact loading, champion computation, promotion,
manifest writing, feature generation, or prediction execution.

#### Goal

Resolve prediction behavior from explicit data availability without silently
selecting a model that cannot produce the requested week.

#### Tests

- Week 1 selects Elo only when Elo state is available and full features are
  unavailable;
- a full-feature win champion is selected only when required inputs exist;
- a total champion is selected only when total inputs exist;
- unavailable total prediction remains explicit;
- missing Elo and unavailable full features produce an unavailable win
  decision;
- a missing champion is distinct from missing required inputs;
- win and total overrides are independently scoped;
- a win override does not change the total decision;
- a total override does not change the win decision;
- an ineligible override is rejected without fallback;
- an Elo override requires available Elo state;
- selected models include stable rationale and explanation;
- champion promotion timestamp, source run ID, metrics, and model identity are
  preserved as provenance;
- champion metrics serialize in deterministic key order;
- win and total champions resolve independently;
- a missing total champion does not alter the win decision;
- a win override skips only win champion lookup;
- a total override skips only total champion lookup;
- two overrides require no champion lookup;
- malformed champion metadata remains an error;
- policy contracts are immutable;
- identical inputs produce equal policy output;
- policy output is JSON serializable;
- the policy module imports no API or pandas layer;
- the policy performs no artifact loading, feature generation, prediction
  execution, champion promotion, or manifest writing.

#### Acceptance

Model selection is explicit, availability-aware, independently scoped by
prediction family, and serializable. Champion status does not imply that a
model can produce the requested week, unavailable outputs remain visible, and
no API-layer inference or silent fallback participates in the decision.

---

### Unit 10: Build Schedule-Complete Win Prediction Component [Complete]

#### Completed

Added a pure weekly win-product builder that attaches explicitly selected win
forecasts to rich schedule truth while preserving exactly one row per
scheduled game.

The builder scopes the supplied schedule to the requested season and week,
preserves source ordering and all schedule fields, and rejects duplicate
canonical game IDs. Neutral-site identity and other schedule context pass
through unchanged.

The Unit 9 prediction policy defines whether the win family is available and
which model identity is eligible. An unavailable win policy produces one
explicitly unavailable row for every scheduled game. A selected policy
requires forecast resolutions and events matching the selected win model
identity.

Forecast attachment uses explicit `ForecastCandidateResolution` values and
exact immutable event IDs. Selected events are never inferred from generation
time, storage order, role, write order, or model priority. Missing or absent
resolutions remain visible as missing forecast rows. Ambiguous resolutions
remain visible as ambiguous rows.

Available product rows preserve the selected forecast's model name, model
type, event ID, run ID, generation timestamp, operational role, and explicit
selection status.

Selected events must match schedule truth for season, week, game ID, away team,
and home team. Team orientation is validated rather than automatically
reversed or reinterpreted. The event model identity must also match the model
selected by the prediction policy.

Available predictions require both away and home win probabilities. Each
probability must fall within the closed zero-to-one range, and the pair must
sum to one within floating-point tolerance. Unavailable, missing, and
ambiguous rows retain null probability and forecast-provenance fields.

The component performs no model execution, champion resolution, feature
generation, forecast generation, forecast-store loading, filesystem access,
timestamp-based selection, or API-layer inference.

#### Goal

Attach selected win probabilities and immutable forecast provenance to every
scheduled game without silently dropping games or choosing forecasts.

#### Tests

- output contains exactly one row per scheduled game;
- output is scoped to the requested season and week;
- schedule source ordering is preserved;
- all supplied schedule fields pass through;
- duplicate scheduled game IDs are rejected;
- missing win forecasts do not remove scheduled games;
- missing forecasts retain null probabilities and provenance;
- ambiguous forecast selection remains explicit;
- unavailable win policy marks every scheduled game explicitly;
- unavailable policy cannot carry forecast resolutions;
- available predictions preserve model name and model type;
- live forecast event ID is preserved;
- forecast run ID is preserved;
- forecast generation timestamp is preserved;
- forecast operational role is preserved;
- selected forecast status is explicit;
- selected event identity must match its explicit resolution;
- selected model identity must match the prediction policy;
- forecast season and week must match product scope;
- forecast game ID must match schedule identity;
- away and home team orientation must match schedule truth;
- reversed orientation is rejected rather than silently corrected;
- available predictions require both probabilities;
- probabilities must remain between zero and one;
- away and home probabilities must sum to one within tolerance;
- neutral-site schedule identity is preserved;
- no model execution or champion resolution occurs;
- no forecast generation or storage read occurs;
- no timestamp, write-order, role, or model-priority selection occurs;
- no API-layer inference participates in composition.

#### Acceptance

The weekly win product provides schedule-complete Moneyline probability
coverage status. Every scheduled game remains visible, available predictions
carry explicit model and immutable forecast provenance, and unavailable,
missing, or ambiguous predictions retain machine-readable status without
silent selection or row deletion.

---

### Unit 11: Attach Derived Spread Component [Complete]

#### Completed

Added a schedule-complete spread component that derives model spread and spread
uncertainty from the selected weekly win forecast.

The component extends the Unit 10 win product without filtering, reordering, or
mutating its rows. Every scheduled game therefore retains either a valid
derived spread or an explicit spread blocker.

Spread derivation uses the existing probit conversion:

`model_spread = -sigma * Phi_inv(home_win_prob)`

The product follows the established NFL home-line convention. A negative model
spread means the home team is favored, a positive model spread means the away
team is favored, and zero represents pick'em.

Each available spread uses the exact calibration identity attached to the
selected win forecast through `win_model_name` and `win_model_type`. The spread
component does not resolve the current champion, use the total-model identity,
choose the newest calibration entry, apply a CLI default, or assume a specific
algorithm.

Added a strict persisted-calibration contract containing model name, model
type, composite calibration key, sigma, residual margin standard deviation,
and calibration update timestamp. Calibration is read from the existing
`game_model_calibration.json` registry by exact composite model key.

The strict calibration reader accepts only complete persisted records with
positive finite sigma and margin standard deviation values and a nonempty
update timestamp. Missing or incomplete records return no calibration. The
weekly spread component does not substitute the existing in-memory or
league-wide fallback values.

Existing fallback behavior in `get_sigma()` and `get_margin_std()` remains
unchanged for older consumers.

Sigma is used only for probability-to-spread conversion. Residual
`margin_std` is retained separately as `spread_uncertainty`; the two values are
not treated as interchangeable.

Available spread rows preserve a provenance chain from the derived spread to
the selected immutable win event and exact calibration record. Provenance
includes the source win event ID, win model name, win model type, persisted
calibration key, and calibration update timestamp.

Rows whose win component is unavailable receive `win_unavailable`. Rows with
an available win forecast but no complete exact calibration receive
`calibration_unavailable`. Blocked rows retain null spread, uncertainty, and
spread-provenance values.

#### Goal

Derive model spread and spread uncertainty from the selected win component
using the correct win-model calibration identity.

#### Tests

- negative model spread means the home team is favored;
- positive model spread means the away team is favored;
- the selected win model determines the calibration identity;
- different selected model calibrations produce different derived spreads;
- spread conversion uses persisted sigma;
- spread uncertainty uses persisted residual margin standard deviation;
- sigma and margin standard deviation remain distinct;
- missing calibration does not fabricate a spread;
- incomplete calibration does not fabricate a spread;
- no in-memory or league-wide fallback calibration is used;
- calibration model name must match the selected win forecast;
- calibration model type must match the selected win forecast;
- spread provenance identifies the source immutable win event;
- spread provenance identifies the selected win model;
- spread provenance identifies the persisted calibration key;
- spread provenance preserves the calibration update timestamp;
- unavailable win rows receive an explicit blocker;
- missing calibration rows receive an explicit blocker;
- blocked rows retain null spread and uncertainty;
- schedule row count is preserved;
- schedule row order is preserved;
- canonical game identity is preserved;
- neutral-site identity is preserved;
- the input weekly win product is not mutated;
- existing post-processing and fallback-getter tests remain green.

#### Acceptance

Every scheduled game has either a valid derived spread with residual spread
uncertainty and traceable source provenance, or an explicit blocker explaining
that the selected win forecast or its exact persisted calibration is
unavailable.

---

### Unit 12: Attach Independent Total Component [Complete]

#### Completed

Added an independent total component that attaches explicitly selected total
forecast events to the schedule-complete weekly product.

Total selection is controlled exclusively by the Unit 9 total policy decision.
The selected win model, derived spread model, and their algorithms do not
participate in total identity or selection. Win and total model algorithms may
differ without conflict.

Total forecasts attach only through explicit candidate resolutions and exact
immutable forecast event IDs. The component performs no selection by event
timestamp, storage order, forecast role, archive order, current champion, or
matching win algorithm.

Each selected total event must use the `total` model family and the model type
selected by the total policy. Event season, week, game ID, away team, and home
team must match the weekly product row. Reversed team orientation and
conflicting model identity are rejected rather than silently corrected.

Available total rows preserve the model-total point estimate, total model name,
total model type, immutable event ID, forecast run ID, generation timestamp,
operational role, and explicit selection status.

Added a strict total-uncertainty contract backed by the selected total model's
artifact metadata. Uncertainty uses the exact total model identity and reads
the positive finite holdout RMSE from `metrics["rmse"]`. The artifact training
timestamp is retained as uncertainty provenance.

The weekly total component does not use the existing `get_total_std()` default
fallback. If a valid total forecast exists but exact artifact uncertainty is
unavailable, the point estimate remains present while uncertainty remains null
and the row receives an explicit `uncertainty_unavailable` status.

Missing and ambiguous total forecasts remain visible without removing scheduled
games. An unavailable total policy produces one explicit unavailable total row
for every scheduled game.

The component preserves row count, ordering, canonical game identity, schedule
fields, neutral-site context, win fields, and spread fields.

#### Goal

Compose selected total-model predictions independently without conflating total
identity or uncertainty with the selected win model.

#### Tests

- total policy selection is independent from win policy selection;
- win and total algorithms may differ;
- total identity does not use the win model type;
- total identity does not use the spread model type;
- selected total events attach through exact immutable event IDs;
- total events compose by canonical game ID regardless of event ordering;
- available totals preserve the model-total point estimate;
- available totals preserve model name and model type;
- available totals preserve event ID and run ID;
- available totals preserve generation timestamp and operational role;
- total selection status is explicit;
- missing total prediction does not remove the game;
- ambiguous total selection does not remove the game;
- unavailable total policy preserves all games;
- selected total event season and week must match product scope;
- selected total event game ID must match schedule identity;
- selected total event team orientation must match schedule truth;
- selected total event model identity must match the total policy;
- model-total values must be present and finite;
- total uncertainty uses the selected total model identity;
- strict total uncertainty reads artifact `metrics["rmse"]`;
- total uncertainty preserves artifact `trained_at`;
- missing uncertainty does not remove the total point estimate;
- missing uncertainty remains explicit;
- no default total uncertainty is silently substituted;
- uncertainty identity mismatch is rejected;
- schedule row count and ordering are preserved;
- schedule, win, spread, and neutral-site fields remain unchanged.

#### Acceptance

Total predictions are independently resolved and explicitly composed. Every
scheduled game retains an explicit total state, available totals preserve
immutable forecast and artifact provenance, and neither model identity nor
uncertainty is inferred from the selected win component.

---

### Unit 13: Add Projected Scores and Product-Level Validation [Complete]

#### Completed

Added the final projected-score composition and product-level validation layers
for the schedule-complete weekly game product.

Projected scores are derived only when the spread component has an available
model spread and the total component has an available model-total point
estimate. A total row whose uncertainty is unavailable remains eligible for
score projection because the point estimate itself is still valid.

Projected scores use the established NFL home-line spread convention and the
existing domain equation:

`projected_home_score = (model_total - model_spread) / 2`

`projected_away_score = (model_total + model_spread) / 2`

The resulting values must reconcile to both upstream point estimates. Projected
home and away scores sum to the model total, and projected away score minus
projected home score equals the model spread. Equivalently, projected home
margin equals the negative model spread.

Added granular projected-score statuses for available scores, unavailable
spread, unavailable total, and simultaneous spread-and-total unavailability.
Blocked rows retain null projected score fields rather than fabricating partial
values.

The score attachment function copies the incoming weekly product, preserves
every row and existing field, and adds only projected-score status and projected
home and away scores.

Added final product validation across schedule, win, spread, total, and
projected-score components.

Schedule identity validation requires nonempty unique canonical game IDs.

Available win rows require finite complementary away and home probabilities,
selected win model identity, and immutable win event identity.

Available spread rows require a finite model spread, positive residual spread
uncertainty, source win event identity, selected win model identity, persisted
calibration key, and calibration update timestamp. Spread event and model
provenance must match the selected win component.

Unavailable spread rows must not contain model spread or spread uncertainty.

Available total rows require a finite model-total point estimate, total model
identity, immutable total event identity, positive total uncertainty, and
artifact training timestamp.

Rows with unavailable total uncertainty retain the valid total point estimate
and total forecast identity while requiring null uncertainty and uncertainty
provenance.

Unavailable total forecast rows must not contain model-total or uncertainty
values.

Available projected-score rows require finite home and away scores that
reconcile to the model total and model spread within floating-point tolerance.
Blocked projected-score rows must retain null score fields.

Validation returns a defensive copy and rejects inconsistent combinations. It
does not fill, coerce, infer, or repair invalid values.

#### Goal

Create projected scores only when required spread and total point estimates
exist, and validate internal coherence across the complete weekly game product.

#### Tests

- projected home and away scores sum to model total;
- projected away score minus projected home score equals model spread;
- negative model spread produces a higher projected home score;
- projected scores use the existing domain calculation;
- spread-unavailable rows receive explicit status;
- total-unavailable rows receive explicit status;
- rows missing both inputs receive combined explicit status;
- blocked projected-score rows retain null scores;
- unavailable total uncertainty does not block a valid score projection;
- available projected scores require finite spread and total values;
- available win probabilities must be finite and complementary;
- available win rows require model and immutable event identity;
- available spread rows require positive uncertainty;
- spread source event must match selected win event;
- spread model identity must match selected win model identity;
- unavailable spread rows cannot contain spread values;
- available total rows require total model and immutable event identity;
- available total uncertainty must be positive;
- unavailable total uncertainty requires null uncertainty values while
  preserving the total point estimate;
- unavailable total forecast rows cannot contain total values;
- projected scores that do not reconcile to total are rejected;
- projected scores that do not reconcile to spread are rejected;
- blocked rows containing projected scores are rejected;
- duplicate or blank game IDs are rejected;
- complete product remains one row per scheduled game;
- game ordering is preserved;
- schedule and neutral-site fields are preserved;
- input product is not repaired or silently coerced.

#### Acceptance

The composed weekly game product has internally coherent schedule, win, spread,
total, uncertainty, provenance, and projected-score values. Every scheduled
game remains present, projected scores exist only when their required point
estimates are available, and all missing or blocked values retain granular
machine-readable status.

---

### Unit 14: Persist the Static Weekly Game Product [Complete]

#### Completed

Added immutable, versioned persistence for validated weekly game products.

Each product run is stored as a separate Parquet artifact addressed by an
explicit product ID. Multiple products for the same season and week coexist
without overwrite or implicit precedence.

The store stamps every row with product schema version, product ID, product run
ID, and timezone-aware UTC generation timestamp. An atomic JSON index records
each product's scope, row count, exact ordered columns, generated timestamp,
and relative artifact path.

Loading by product ID validates the index schema, entry shape, artifact path,
artifact existence, exact column order, row count, product schema version,
product identity, run identity, generated timestamp, season, week, and complete
weekly game-product domain contract.

Identical rewrites are idempotent. Reusing a product ID with different content,
run identity, or generation timestamp is rejected. Missing artifacts, unindexed
artifacts, and other inconsistent storage states fail clearly.

Added a versioned current-product manifest keyed by canonical season and week.
Each selection records an explicit product ID and timezone-aware UTC selection
timestamp. Selecting current requires an indexed, loadable product whose scope
matches the requested season and week.

Writing a newer product does not alter current. Current changes only through an
explicit selection operation and is never inferred from timestamps, filenames,
modification times, run IDs, index ordering, or lexical product ID ordering.

Added standard dataset loader and writer wrappers for exact product loading,
current-product loading, immutable product writing, and explicit current
selection. These wrappers delegate to the product store and do not duplicate
storage or validation behavior.

Added integration coverage for two products in the same weekly scope. The test
writes and selects product A, writes product B without changing current,
explicitly changes current to B, and verifies that both products remain exactly
loadable with distinct coherent values.

The full persistence and loading workflow creates only weekly-product storage.
It performs no policy resolution, feature generation, model execution,
champion resolution, calibration loading, forecast selection, spread
derivation, total composition, projected-score calculation, or API inference.

#### Goal

Write and load immutable, versioned weekly game products through a static
serialization boundary that supports multiple runs and explicit current
selection.

#### Tests

- product round-trips without schema loss;
- exact ordered domain columns survive;
- nullable blocked values survive;
- product ID survives;
- product run ID survives;
- timezone-aware UTC generation timestamp survives;
- multiple products for the same season and week coexist;
- each coexisting product remains independently loadable;
- identical rewrites are idempotent;
- conflicting product content is rejected;
- conflicting run identity is rejected;
- conflicting generation timestamp is rejected;
- unsupported index schema fails clearly;
- artifact column mismatch fails clearly;
- row-count mismatch fails clearly;
- product identity mismatch fails clearly;
- season and week mismatch fail clearly;
- missing artifact fails clearly;
- unindexed artifact fails clearly;
- current selection is explicit;
- writing a newer product does not change current;
- current selection can be changed explicitly;
- missing current selection fails clearly;
- selection requires an indexed and loadable product;
- selection scope must match product scope;
- unsupported current-manifest schema fails clearly;
- dataset loaders delegate to product storage;
- dataset writers delegate to product storage;
- integration round trip preserves distinct product values;
- no unrelated compute or model artifacts are created;
- loading performs no prediction or model computation.

#### Acceptance

API, CLI rendering, and edge calculation can consume one persisted weekly game
product through the standard dataset boundary. Product runs are immutable and
versioned, multiple runs coexist, current selection is explicit, and loading is
strictly a validated serialization operation rather than a compute boundary.

---

### Unit 15: Add Source-Labeled nflverse Market Adapter [Complete]

#### Completed

Added a pure nflverse schedule-market adapter that converts rich upcoming
schedule fields into the generic long-format market contract.

Every adapted row is labeled `nflverse_schedule`. No DraftKings label, parser,
fetcher, game-ID resolver, or fallback behavior participates in adaptation.

The rich schedule's timezone-aware UTC ingestion timestamp is preserved as the
market `fetched_at` value. The adapter does not read the clock or substitute an
execution timestamp.

Canonical nflverse game IDs and away/home team orientation pass through
unchanged. Requested season and week scope is enforced, and duplicate or
incomplete schedule identity is rejected.

Each scheduled game produces six deterministic rows: away and home Moneyline,
away and home Spread, and Over and Under Total.

Moneyline prices map directly to away and home sides with null line values.
The nflverse `spread_line` uses home-team orientation, so the home side retains
the source line and the away side receives its additive inverse. Both Total
sides retain the source total line.

Incomplete markets remain explicit. Canonical side rows are retained with null
odds or line values rather than being omitted, defaulted, or fabricated.

Replaced the development-era DraftKings-specific market ledger and current
snapshot filenames with source-neutral `odds_log.parquet` and
`odds_current.parquet` artifacts. No compatibility reader, migration, alias,
fallback, or dual-write behavior was added for the retired paths.

Added strict validation and normalization for the shared long-format market
contract. Stored rows require canonical columns, nonempty source and market
identity, valid NFL week values, timezone-aware UTC ingestion timestamps, and
valid Moneyline, Spread, and Total side combinations.

Validation runs before current-snapshot writes and historical-ledger appends,
and again after snapshot and ledger loads. Nullable prices and lines survive
storage without omission or fabricated values.

Added integration coverage for adaptation, current-snapshot persistence, and
joining to schedule truth by canonical game ID. Complete markets, incomplete
markets, and unmatched scheduled games remain distinct. Source identity,
ingestion timestamp, normalized spread orientation, and Total values survive
the round trip.

#### Goal

Convert available nflverse schedule market fields into the generic market
contract without depending on the unreliable DraftKings pull.

#### Tests

- source is labeled `nflverse_schedule`;
- no DraftKings label is applied;
- rich-schedule ingestion timestamp is preserved as market fetch timestamp;
- timestamps require timezone-aware UTC values;
- mixed snapshot timestamps are rejected;
- Moneyline away and home sides normalize correctly;
- Spread away and home sides normalize correctly;
- nflverse home-line spread orientation is preserved;
- Total Over and Under sides normalize correctly;
- exactly six canonical market-side rows are emitted per scheduled game;
- incomplete markets retain explicit nullable rows;
- missing prices are not defaulted;
- missing lines are not inferred;
- canonical game IDs pass through unchanged;
- away and home team orientation passes through unchanged;
- duplicate scoped game IDs are rejected;
- requested season and week scope is enforced;
- non-nflverse source input is rejected;
- empty requested scope returns the canonical schema;
- generic market storage uses source-neutral paths;
- retired DraftKings-specific storage paths are not created;
- generic market schema and column order are validated;
- invalid market and side combinations are rejected;
- incomplete rows survive Parquet round trip;
- source and ingestion provenance survive persistence;
- complete, incomplete, and unmatched games remain distinct after joining;
- schedule truth remains the denominator under a left join;
- no DraftKings resolver, parser, or fetch path participates.

#### Acceptance

Current market comparison uses source-labeled nflverse schedule markets through
the generic market contract and source-neutral storage. Complete, incomplete,
and unmatched market states remain explicit, and the normal current-market path
does not depend on the unreliable DraftKings pull.

---

### Unit 16: Reclassify DraftKings as Legacy Best-Effort Adapter [Complete]

#### Completed

Reclassified DraftKings ingestion as an explicitly invoked legacy best-effort
adapter rather than a production dependency.

Added a dedicated adapter-unavailability error. HTTP failures, HTML or
human-verification responses, malformed JSON, non-object payloads, and malformed
event, market, or selection collections now fail with clear adapter-specific
messages.

No browser automation, cookie handling, retry mechanism, verification bypass,
or alternate endpoint workaround was introduced.

Valid payloads containing no current events remain distinct from unavailable
responses. Empty results do not fabricate persistence paths. Usable adapter
output continues through the generic source-neutral market store.

The explicit `gridiron ingest dk-odds` command remains available. Its help
identifies the adapter as legacy and best-effort, states that nflverse schedule
data supplies the supported current-market workflow, and explains that
DraftKings is not required by normal data refresh or weekly prediction.

Adapter failures exit nonzero. Valid empty results report that no files were
written. Successful runs report the generic market ledger and current-snapshot
paths.

Removed DraftKings ingestion from normal orchestration. `run-data-pipeline`
no longer contains an odds-fetch stage or imports the DraftKings adapter.
`weekly-predict` no longer imports DraftKings or contains a `fetch-odds` stage.

Weekly edge generation consumes the existing source-neutral current market
snapshot and depends only on prediction generation. Missing market data remains
a source-neutral soft failure and does not trigger an external adapter.

Removed stale odds-fetch references from other composite workflow descriptions
and stage sets.

Updated CLI and API recovery guidance to reference the current source-neutral
market snapshot and rich nflverse upcoming schedule. Removed references to the
nonexistent `gridiron ingest fetch-odds` command and the retired
`dk_odds_current.parquet` artifact.

Updated shared repository fixtures to write `odds_current.parquet` and aligned
integration fixtures with the required timezone-aware UTC market provenance
contract.

Updated the Line Shopping blocker to distinguish current nflverse schedule
markets from future multi-book sportsbook ingestion.

Legitimate DraftKings sportsbook identities, parser coverage, explicit legacy
command behavior, and stored book examples remain intact.

#### Goal

Preserve the historical DraftKings adapter without presenting it as the default
or required recovery path.

#### Tests

- valid DraftKings fixture JSON still parses;
- HTTP failures produce a clear adapter-specific error;
- non-JSON responses fail clearly;
- HTML and human-verification responses fail clearly;
- malformed JSON fails clearly;
- non-object payloads fail clearly;
- malformed expected collections fail clearly;
- valid empty payloads remain distinct from adapter failures;
- empty results do not fabricate artifact paths;
- no browser automation or verification bypass is introduced;
- explicit command help identifies the adapter as legacy and best-effort;
- explicit command help identifies nflverse as the supported market workflow;
- adapter failure exits nonzero;
- valid empty command result reports that no files were written;
- successful command output reports generic source-neutral artifacts;
- normal data pipeline has no external odds stage;
- normal data pipeline does not import or invoke DraftKings;
- weekly prediction has no external odds stage;
- weekly edge generation consumes an existing market snapshot;
- weekly edge generation does not invoke DraftKings;
- missing market output is source-neutral;
- post-week and full-retrain contain no retired odds stage references;
- CLI recovery guidance contains no nonexistent odds-ingestion command;
- API loader guidance contains no DraftKings-specific snapshot path;
- API missing-market errors reference the active generic snapshot path;
- shared fixtures write the active generic snapshot;
- integration market timestamps are timezone-aware UTC;
- frontend blocker copy identifies nflverse as the current game-market source;
- frontend blocker copy does not claim DraftKings is the current-only source;
- retired DraftKings-specific snapshot paths are not created;
- legitimate DraftKings book identities and explicit adapter tests remain.

#### Acceptance

DraftKings code remains available as an explicitly invoked legacy best-effort
adapter, but normal data refresh, weekly prediction, current market storage,
edge generation, API recovery guidance, shared fixtures, and frontend
operational messaging do not depend on or recommend it.

---

### Unit 17: Build Unified Edge Diagnostics [Complete]

#### Completed

Added immutable contracts for edge diagnostic blockers, terminal analytical
states, prediction and market provenance, weekly coverage diagnostics, and
recommendation results.

Defined explicit blockers for missing predictions, missing market data,
wrong-scope market data, stale market data, zero matched games, and incomplete
markets.

Defined terminal result states that distinguish blocked inputs, no calculable
edge rows, calculated rows with no positive expected value, and positive edge
rows.

Added validation for requested season and week, nonnegative counts,
prediction-to-market match bounds, eligible-market arithmetic, calculated and
positive row relationships, blocker uniqueness, and terminal-state
consistency.

Added deterministic JSON-compatible serialization for diagnostics and
provenance.

Implemented a pure evaluator that scopes prediction, market, calculated-edge,
and filtered-edge inputs to an explicit season and week.

The evaluator derives distinct prediction-game, market-game, and matched-game
counts from canonical game IDs. Duplicate input rows do not inflate coverage.

Added complete Moneyline, Spread, and Total counting using the existing
recommendation input semantics. Eligible-market count is derived from those
three complete market-family counts.

Missing market sides, prices, required lines, or corresponding model values
produce an explicit incomplete-market blocker.

Added deterministic optional market freshness evaluation through
caller-supplied `as_of` and `max_market_age` values. The evaluator does not read
the system clock, inspect file timestamps, or infer a freshness threshold.

Retained all recognized scoped win, total, weekly-product, market-source, and
market-timestamp provenance as sorted unique tuples. Mixed provenance is not
collapsed through recency or arbitrary selection.

Added a frozen recommendation result contract that pairs filtered edge rows
with diagnostics for the same weekly scope and validates that the returned row
count matches the diagnostic filtered-edge count.

Added `build_edge_result()` as a compatibility-preserving composition of the
existing edge report builder, ranking function, and diagnostic evaluator.

The existing DataFrame-returning recommendation functions and all edge math,
classification, Kelly sizing, and ranking behavior remain unchanged.

An empty result from the new recommendation operation now retains an explicit
blocker or terminal analytical state. A custom positive EV threshold may return
an empty table while preserving the fact that positive calculated edges existed
below that threshold.

The diagnostic and recommendation result paths do not mutate supplied frames,
execute models, ingest markets, select artifacts, read files, inspect file
timestamps, or access the system clock.

#### Goal

Create structured coverage and result-state diagnostics before changing edge
math callers.

#### Tests

- no predictions produce an explicit blocker;
- no market data produces an explicit blocker;
- simultaneous missing inputs retain both blockers;
- wrong-scope market data remains distinct from missing market data;
- explicit stale-market policy produces a stale blocker;
- staleness is not inferred without a supplied policy;
- zero matched games produce an explicit blocker;
- incomplete Moneyline, Spread, or Total coverage remains explicit;
- duplicate rows do not inflate distinct-game counts;
- complete market counts are derived from actual model and market values;
- no calculable edge rows produce an explicit terminal state;
- calculated rows with no positive EV produce an explicit terminal state;
- positive edge rows produce an explicit terminal state;
- a custom threshold may return no rows while retaining positive-edge counts;
- calculated, positive, and filtered counts are derived from actual inputs;
- win provenance is retained;
- total provenance is retained;
- weekly-product provenance is retained;
- all market sources and timestamps are retained;
- provenance values are deterministic, sorted, and unique;
- market timestamps require timezone-aware UTC values;
- diagnostic contracts are immutable and JSON serializable;
- recommendation result rows agree with diagnostic filtered-row counts;
- negative diagnostic minimum-EV thresholds are rejected;
- supplied DataFrames are not mutated;
- existing recommendation and ranking tests remain unchanged and green.

#### Acceptance

The new recommendation-level operation always pairs its edge table with
structured diagnostics. An empty edge table always has an explicit blocker,
analytical result state, or threshold explanation.

---

### Unit 18: Unify Weekly Edge Calculation

#### Completed

Established one persisted domain boundary for all current-week edge calculation.

Added structured edge diagnostics covering prediction availability, market
availability, market scope, market freshness, game-ID alignment, market
completeness, calculable rows, positive expected-value rows, filtered rows, and
artifact provenance.

Added a shared edge result containing ranked recommendation rows and structured
diagnostics.

Added a weekly edge service that loads only the explicitly selected immutable
weekly product and the current source-labeled market snapshot.

The service uses persisted selected win probabilities, derived Spread values,
independent Total predictions, and their persisted uncertainties. It does not
resolve champions, load prediction archives, execute models, recompute
predictions, or reload calibration artifacts.

Unavailable Spread or Total uncertainty disables only the corresponding market
family. Moneyline remains independently calculable.

Multiple persisted uncertainty values for one available market family are
rejected rather than selected, averaged, or replaced with defaults.

Added persisted round-trip coverage through immutable weekly-product storage,
explicit current-product selection, source-labeled market storage, canonical
game-ID alignment, recommendation calculation, bankroll sizing, provenance, and
minimum-EV filtering.

Migrated `gridiron edges report` to the shared service. Removed report-level
model selection, champion resolution, prediction-archive loading, direct market
loading, uncertainty lookup, edge construction, and ranking.

Changed report bankroll sizing from an implicit default to an optional input.
Unavailable dollar stakes render explicitly rather than as fabricated zero
values.

Migrated the `/edges` API loader and route to the shared edge result. Added
source-neutral API field-status mappings for missing weekly products, missing or
wrong-scope markets, stale markets, zero game-ID matches, and incomplete market
coverage.

Removed the retired odds-unavailable exception and its compatibility path.

Replaced API champion-and-archive integration fixtures with explicitly selected
weekly products and source-labeled market snapshots.

Added weekly-product composition to `weekly-predict`. The workflow now retains
the exact live Elo forecast run, builds a schedule-complete weekly product,
attaches derived Spread values, represents unavailable Total values explicitly,
writes the immutable product, and selects it as current before edge generation.

Removed the unrelated `weekly-predict --model-type` option. The current live
prediction stage produces Elo forecast events and now identifies itself
consistently as an Elo workflow.

Migrated `weekly-predict` edge generation to the shared service. Removed direct
prediction-archive loading, market loading, uncertainty lookup, recommendation
construction, and ranking from the workflow.

Migrated `verify-week` edge evaluation to the shared service while preserving
its independent read-only checks for schedule availability, forecast selection,
market coverage, and artifact provenance.

Audited all edge construction, artifact writing, current-market loading,
prediction-archive loading, uncertainty lookup, composite workflow, API, and
command-registration callsites.

Confirmed that all active current-week consumers use the shared service:

- `gridiron edges report`
- `GET /edges`
- `gridiron weekly-predict`
- `gridiron verify-week`

Confirmed that direct recommendation construction remains only in the domain
implementation, domain tests, and historical closing-line-value analysis.

Confirmed that remaining prediction-archive consumers support historical
calibration, game views, or historical CLV rather than current-week edge
calculation.

Confirmed that current edge CSV writing is limited to the standalone report
export and `weekly-predict`, both using rows returned by the shared service.

#### Goal

Ensure every current-week edge consumer uses the same explicitly selected
weekly product, source-labeled current market snapshot, recommendation
calculation, filtering behavior, bankroll semantics, provenance, and structured
diagnostics.

#### Tests

Covered diagnostic invariants, blocker precedence, market completeness,
freshness, provenance, optional market-family uncertainty, bankroll sizing,
minimum-EV filtering, immutable product selection, persisted product and market
round trips, CLI rendering, API field-status translation, weekly workflow
composition, verification behavior, and architecture dependency guards.

Verified that current-week consumers cannot restore direct prediction-archive,
market-loading, uncertainty-resolution, recommendation-construction, or ranking
dependencies.

All quality gates and tests pass.

#### Acceptance

All current-week edge surfaces consume one shared result derived from the
explicitly selected immutable weekly product and current source-labeled market
snapshot.

Historical CLV remains the intentional exception because it operates over
historical prediction and odds-ledger artifacts.

---

### Unit 19: Policy-Driven Weekly Prediction Orchestration

19.2h    Remove retired orientation and rebuild artifacts
  19.2h-A  Remove the inactive legacy pipeline entry points
  19.2h-B  Remove legacy feature registrations and implementations
  19.2h-C  Remove legacy modeling fixtures and generic artifact examples
  19.2h-D  Remove migration-only schema guards and stale terminology
  19.2h-E  Rebuild, retrain, and validate canonical artifacts
19.1b    Implement truthful availability inspection
19.3     Execute policy-selected models
19.4     Persist and select weekly product
19.5     Readiness and publication hardening
19.6     End-to-end orchestration acceptance

#### Goal

Make `weekly-predict` resolve prediction policy before model execution and
produce one truthful, schedule-complete pregame product whose forecast events,
model identities, readiness diagnostics, and published outputs are traceable to
the exact persisted weekly product.

Before policy-driven trained-model execution can be added, replace the
historical `TEAM_A` / `TEAM_B` modeling orientation with the canonical
`AWAY_TEAM` / `HOME_TEAM` game orientation already used by schedules, weekly
products, markets, APIs, and frontend surfaces.

#### Tests

- prediction availability is model-specific;
- explicit overrides are evaluated against the exact requested model identity;
- an ineligible override does not silently fall back;
- historical and upcoming model inputs share one stable home/away schema;
- one modeling row represents one game;
- Win targets and probabilities use the home-team perspective;
- differential features use `HOME - AWAY`;
- Total prediction remains independent from Win prediction;
- schedule-complete upcoming features can be built without orientation adapters;
- prediction policy controls the model implementations actually executed;
- forecast-event model identities match policy decisions;
- weekly-product component identities match selected forecast events;
- missing market data does not fail forecast or weekly-product publication;
- omitted bankroll leaves dollar stake unavailable;
- published outputs identify the exact weekly product;
- no-edge outcomes cannot leave stale files presented as current;
- valid `--skip` and `--only` paths respect dependency closure;
- one end-to-end workflow persists forecast events, selects the weekly product,
  evaluates readiness, and publishes only current outputs.

#### Acceptance

One command resolves prediction policy, executes the selected available Win and
Total model families, persists immutable forecast events, composes and
explicitly selects a schedule-complete weekly product, evaluates readiness, and
publishes only outputs traceable to that exact product.

Historical training rows, upcoming prediction rows, forecast events, weekly
products, and market joins use one stable home/away game orientation.

No game-prediction feature, model, predictor, or orchestration path requires
`TEAM_A` or `TEAM_B`.

Win models directly predict home win probability. Away win probability is the
complement.

All differential features use `HOME - AWAY`.

Unavailable components are represented explicitly and are never silently
omitted, recomputed from another model identity, or substituted from stale
artifacts.

### Unit 19.2a: Canonical Game Schema

#### Completed

Defined one canonical home/away-oriented schema for the game-prediction domain.

Locked identity columns to game, season, week, away team, and home team.

Defined optional game date and neutral-site identity fields.

Defined historical score and target columns using the home-team perspective.

Defined Win prediction outputs as home win probability and its away-team
complement.

Documented the existing home-oriented Spread convention, where a negative model
Spread means the home team is favored.

Established `HOME - AWAY` as the direction for every differential feature.

Added canonical helpers for Away, Home, and differential feature names.

Explicitly identified `TEAM_A`, `TEAM_B`, and `HOME_FIELD` as retired
orientation columns.

No feature, modeling, training, prediction, or artifact behavior changed in this
unit.

#### Goal

Establish one centrally defined home/away schema before migrating modeling rows,
features, predictors, tests, and development artifacts.

#### Tests

Verified canonical game identity, optional identity, scores, targets,
probabilities, prediction outputs, feature prefixes, differential naming, input
validation, and exclusion of retired orientation columns.

#### Acceptance

The future game-prediction schema has one documented source of truth and no
canonical field depends on `TEAM_A`, `TEAM_B`, or perspective-relative
`HOME_FIELD`.

### Unit 19.2b.1: Preserve Historical Home/Away Truth

#### Completed

Extended the cleaned historical games schema with explicit Away Team, Home Team,
Away Score, Home Score, and neutral-site status.

Preserved these values directly from nflverse schedule identity and score
fields. They are never reconstructed from winner/loser fields, game location,
game IDs, abbreviation reversal, or alphabetical ordering.

Retained existing winner/loser fields for historical consumers not yet migrated
to the canonical home/away schema.

Added validation for required columns, unique and nonempty game IDs, nonempty
team identities, distinct away and home teams, nonnegative scores, binary
neutral-site values, ties, winner and loser score reconciliation, and winning
team identity.

Updated favorite identity derivation to use the preserved home and away teams.

Updated the empty cleaned-games schema and added focused synthetic cleaner tests
covering home wins, away wins, neutral-site games, ties, empty first runs,
non-clobbering empty refreshes, duplicate game IDs, same-team rows, score
mismatches, and invalid tie state.

Restored the full nflverse raw history from 1999 through 2026 and regenerated
the cleaned historical artifact.

Validated 7,276 cleaned historical games with 7,182 standard-site games, 94
neutral-site games, and 15 ties.

Confirmed zero duplicate game IDs, same-team rows, negative scores, tie-state
mismatches, winner-score mismatches, loser-score mismatches, home-winner
identity mismatches, and away-winner identity mismatches.

Confirmed that the standard current-season refresh replaces only the requested
season while preserving all other seasons in the raw historical artifact.

All quality gates and tests pass.

#### Goal

Prevent historical home/away information from being discarded during cleaning
so all later modeling and prediction stages can consume explicit schedule
truth.

#### Tests

Covered home wins, away wins, neutral-site games, ties, empty output schema,
empty-refresh protection, duplicate IDs, invalid same-team rows, score
mismatches, tie mismatches, historical regeneration, and complete artifact
reconciliation.

#### Acceptance

The cleaned historical games artifact contains explicit, validated home/away
identity and scores for every completed game without downstream orientation
reconstruction.

### Unit 19.2b.1a: Safe Seasonal nflverse Refresh

#### Completed

Changed explicit nflverse season ingestion from destructive artifact replacement
to targeted season refresh.

`--season` now removes and replaces only the requested seasons while preserving
all other seasons in the shared raw historical artifact.

Multiple explicit seasons are sorted, deduplicated, fetched together, and
replaced atomically within the existing history.

The default current-season refresh now delegates to the same selected-season
refresh implementation.

`--all-years` remains the only command mode that intentionally replaces the raw
artifact with a selected historical range.

Updated CLI help to distinguish selected-season refresh behavior from
full-history replacement behavior.

Added a dedicated nflverse game-ingestion test module covering preservation of
unrequested seasons, multi-season replacement, sorting and deduplication,
invalid season rejection, new-artifact creation, duplicate game-ID rejection,
and single-season delegation.

All quality gates and tests pass.

#### Goal

Prevent targeted nflverse ingestion from silently collapsing the shared raw
historical artifact to only the requested season.

#### Tests

Covered targeted refresh, multi-season refresh, season normalization, invalid
input, first-write behavior, duplicate identities, and current-season
delegation.

#### Acceptance

Explicit season ingestion preserves all unrequested historical seasons, while
full-history ingestion remains the only intentional replacement operation.

### Unit 19.2b.2: Home/Away Modeling Table

#### Completed

Added a canonical one-row-per-game historical modeling-table builder.

The builder consumes only explicit Away Team, Home Team, Away Score, Home
Score, game identity, game date, and neutral-site fields from the cleaned
historical games artifact.

Derived a nullable Home Win target. Home wins are represented as one, away wins
as zero, and ties as null.

Defined Actual Margin as Home Score minus Away Score and Actual Total as the sum
of Away and Home scores.

Added strict validation for required columns, game identities, team identities,
duplicate game IDs, week numbers, scores, neutral-site values, and same-team
matchups.

Separated identity, week, score, neutral-site, and target validation into
focused helpers.

Ensured the builder does not mutate its input and returns chronologically stable
output.

Confirmed the new modeling schema contains no `TEAM_A`, `TEAM_B`, `HOME_FIELD`,
or `RESULT` columns.

Retained the existing symmetric builder as the active `build_model_inputs()`
input until the feature pipeline is migrated.

#### Goal

Create and validate the canonical one-row home/away modeling contract before
migrating feature generation and model execution.

#### Tests

Covered home wins, away wins, ties, neutral-site games, output schema, retired
column exclusion, ordering, input immutability, missing columns, duplicate game
IDs, null and empty identities, same-team rows, invalid weeks, null and negative
scores, and invalid neutral-site values.

All quality gates and tests pass.

#### Acceptance

A cleaned historical game can be converted into exactly one validated,
home/away-oriented modeling row without winner/loser reconstruction or
perspective-relative fields.

### Unit 19.2c.1: Canonical Home/Away Elo Feature

#### Completed

Added a canonical home/away Elo feature for one-row-per-game inputs.

The feature consumes stable Away Team, Home Team, season, and week identity and
produces Away Elo and Home Elo.

Joined Elo state independently for Away and Home using exact team, season, and
week identity.

Preserved every input row, its order, and unrelated columns.

Missing Away or Home ratings remain explicitly null and do not remove scheduled
games.

Ratings from another season or week do not satisfy the requested game identity.

Rejected malformed game schemas, malformed Elo schemas, and duplicate Elo
team-season-week identities.

Confirmed that feature execution does not mutate its input.

Registered the canonical implementation independently from the existing
TEAM_A/TEAM_B Elo feature.

Left the active feature sequence and current modeling pipeline unchanged.

#### Goal

Provide the first shared historical and upcoming team feature using the
canonical one-row home/away schema.

#### Tests

Covered registry identity, feature outputs, exact weekly joins, row and order
preservation, immutability, missing Away ratings, missing Home ratings,
other-week ratings, other-season ratings, duplicate Elo identities, missing
game columns, missing Elo columns, empty frames, and exclusion of retired
orientation names.

All quality gates and tests pass.

#### Acceptance

Historical and upcoming one-row game frames can receive explicit Away and Home
Elo ratings without TEAM_A, TEAM_B, HOME_FIELD, probability calculation, row
loss, or model execution.

### Unit 19.2c.2: Canonical Home/Away EPA Feature

#### Completed

Added a canonical home/away EPA feature for one-row-per-game inputs.

The feature produces paired Away and Home columns for all 36 existing EPA
metrics.

Reused the established rolling EPA implementation, preserving prior-game-only
shift behavior, configurable window length, cross-season history, playoff
exclusion, and null handling for unavailable metrics.

Joined Away and Home EPA independently using exact team, season, and week
identity.

Preserved every input game, original row order, and unrelated columns.

Missing team history remains explicitly null without removing games or affecting
the opposite team's available EPA values.

An unavailable EPA artifact produces the complete canonical EPA schema with null
values.

Rejected missing game identity columns, malformed nonempty EPA source schemas,
duplicate team-season-week identities, invalid season labels, and nonpositive
rolling windows.

Confirmed that the feature does not mutate its input.

Registered the canonical implementation independently from the existing
TEAM_A/TEAM_B EPA feature.

Left the active feature sequence, stored modeling pipeline, EPA-window tuning,
and legacy EPA implementation unchanged.

#### Goal

Provide schedule-complete pregame EPA features using the canonical one-row
home/away game orientation.

#### Tests

Covered registry identity, complete output schema, distinct Away and Home joins,
current-game anti-lookahead, unavailable EPA artifacts, missing Away history,
input preservation, duplicate EPA identities, invalid season labels, and
rolling-window validation.

All quality gates and tests pass.

#### Acceptance

Historical and upcoming one-row game frames can receive paired Away and Home
pregame EPA features without TEAM_A, TEAM_B, HOME_FIELD, lost games, or current
game leakage.

### Unit 19.2c.3: Canonical Home/Away Rest Features

#### Completed

Added canonical home/away rest features for one-row-per-game historical and
upcoming inputs.

The feature produces Away Days Rest, Home Days Rest, Away Short Week, Home Short
Week, Away Post Bye, Home Post Bye, and one Days Rest Differential.

Defined Days Rest Differential as Home Days Rest minus Away Days Rest.

Calculated rest from each target game's date against the latest completed game
date strictly before it.

This schedule-complete lookup supports upcoming games that do not yet exist in
completed historical data.

Prevented future-game leakage by excluding games on or after the target date.

Preserved the existing short-week threshold of fewer than six days and the
post-bye threshold of at least thirteen days.

Preserved cross-season elapsed rest rather than resetting the value at the
season boundary.

Missing team history or invalid target dates remain explicitly null, including
dependent flags and the differential.

Rejected malformed target schemas and duplicate historical team-game
identities.

Preserved input rows, unrelated columns, and caller immutability.

Registered the canonical implementation independently from the existing
TEAM_A/TEAM_B rest feature.

Left the active feature sequence and existing modeling pipeline unchanged.

#### Goal

Provide schedule-complete pregame rest features using stable Away and Home
identity without requiring target games to exist in completed history.

#### Tests

Covered registration, canonical output schema, distinct Away and Home values,
home-minus-away differential direction, future-game exclusion, short-week and
post-bye boundaries, missing history, cross-season elapsed rest, input
preservation, missing target columns, duplicate historical identities, and
exclusion of retired orientation names.

All quality gates and tests pass.

#### Acceptance

Historical and upcoming one-row game frames receive truthful Away and Home rest
features without TEAM_A, TEAM_B, HOME_FIELD, target-game archive dependency, or
future leakage.

### Unit 19.2c.4: Canonical Home/Away Record Features

#### Completed

Added canonical home/away season-record and streak features for
one-row-per-game historical and upcoming inputs.

The feature produces Away and Home wins, losses, win percentage, win streak,
and loss streak.

Calculated each team's state from completed games in the same season with week
numbers strictly earlier than the target week.

Target-week and future-week results cannot contribute to pregame features.

Records reset at the season boundary.

Preserved the existing tie convention: each tie contributes one-half win and
one-half loss and resets both win and loss streaks.

Teams without prior same-season history receive zero wins, zero losses, null win
percentage, and zero-valued streaks.

Derived result history directly from explicit Away Team, Home Team, Away Score,
and Home Score fields.

Rejected malformed target schemas, malformed historical schemas, duplicate game
IDs, and duplicate team-game identities.

Preserved target rows, unrelated columns, and caller immutability.

Registered the canonical implementation independently from the existing
TEAM_A/TEAM_B record feature.

Left the active feature sequence and existing modeling pipeline unchanged.

#### Goal

Provide schedule-complete pregame record and streak features using stable Away
and Home identity.

#### Tests

Covered registration, canonical output schema, distinct Away and Home records,
current-week exclusion, future-week exclusion, tie accounting, tie streak
reset, season reset, initial-state values, input preservation, missing target
columns, and duplicate historical identities.

All quality gates and tests pass.

#### Acceptance

Historical and upcoming one-row game frames receive truthful same-season Away
and Home records and streaks without TEAM_A, TEAM_B, HOME_FIELD, target-game
archive dependency, or current and future result leakage.

### Unit 19.2c.5: Canonical Home/Away Schedule-Strength Features

#### Completed

Added canonical home/away strength-of-schedule and strength-of-victory features
for one-row-per-game historical and upcoming inputs.

The feature produces Away SOS, Away SOV, Home SOS, and Home SOV.

Derived historical team-opponent results directly from explicit Away Team, Home
Team, Away Score, and Home Score fields.

Joined each opponent's Elo from the exact season and week of the historical
matchup.

Calculated target values from same-season games with week numbers strictly
earlier than the target week.

Target-week, future-week, and unplayed games cannot contribute to pregame
features.

SOS includes prior wins, losses, and ties with available opponent Elo.

SOV includes only outright wins with available opponent Elo. Ties are excluded
from SOV.

Historical games with unavailable opponent Elo are excluded from the applicable
average rather than assigned a default rating.

Teams without qualifying history receive explicit null SOS and SOV values.

Rejected malformed target, historical-game, and Elo schemas, duplicate
historical game IDs, and duplicate Elo team-season-week identities.

Preserved target rows, unrelated columns, and caller immutability.

Registered the canonical implementation independently from the existing
TEAM_A/TEAM_B schedule-strength feature.

Left the active feature sequence and existing modeling pipeline unchanged.

#### Goal

Provide schedule-complete pregame schedule-strength features using stable Away
and Home identity and historical opponent Elo from the original matchup week.

#### Tests

Covered registration, output schema, canonical Elo dependency, independent Away
and Home calculations, strength of victory, ties, current-week exclusion,
future-week exclusion, unplayed-game exclusion, missing opponent Elo, season
reset, initial null states, input preservation, malformed schemas, duplicate
game IDs, duplicate Elo identities, and exclusion of retired orientation names.

All quality gates and tests pass.

#### Acceptance

Historical and upcoming one-row game frames receive truthful Away and Home SOS
and SOV values without TEAM_A, TEAM_B, HOME_FIELD, target-game archive
dependency, current or future result leakage, or default substitution for
missing opponent Elo.

### Unit 19.2c.6: Canonical Home/Away Travel Features

#### Completed

Added canonical Away and Home travel features for one-row-per-game historical
and upcoming inputs.

The feature produces game-site altitude, Away and Home kilometers traveled, and
Away and Home timezone shifts.

Resolved the actual game venue internally by Game ID from historical games and
the rich upcoming schedule. The canonical modeling schema does not need to
carry stadium identity solely for travel calculation.

Joined actual venue coordinates by stadium name and franchise origins by team
and season.

Excluded special Alternate and International venue rows from franchise-origin
selection.

Allowed duplicate stadium aliases and franchise-season rows when they resolve
to the same coordinate tuple. Rejected conflicting venue or coordinate
identities.

Calculated both teams' travel from franchise-season home coordinates to the
actual game-site coordinates.

Standard home games naturally produce zero Home travel and timezone shift when
the actual venue matches the Home team's coordinates.

Neutral, international, relocated, and alternate-site games calculate travel
for both teams from their respective origins.

Missing venue or coordinate data remains explicitly null without removing the
target game or failing unrelated historical feature generation.

Added access to the rich upcoming schedule through the repository-scoped
dataset accessor. A missing rich upcoming artifact does not prevent historical
travel calculation.

Preserved target row order, unrelated columns, and caller immutability.

Registered the canonical implementation independently from the existing
TEAM_A/TEAM_B travel feature.

Left the active feature sequence, legacy travel implementation, and canonical
modeling schema unchanged.

#### Goal

Provide schedule-complete Away and Home travel features using the actual game
venue without requiring target rows to carry stadium metadata or use retired
orientation fields.

#### Tests

Covered registration, canonical outputs, standard home games, upcoming venue
resolution, historical venue resolution, neutral venues, unavailable rich
schedule data, missing venues, missing team origins, coordinate aliases,
conflicting venue identities, conflicting franchise and site coordinates, row
and column preservation, input immutability, missing target fields, and
exclusion of retired orientation names.

All scoped quality gates and tests pass.

#### Acceptance

Historical and upcoming one-row game frames receive truthful Away and Home
travel distance, timezone shift, and game-site altitude without TEAM_A,
TEAM_B, HOME_FIELD, venue assumptions, target-row stadium propagation, row
loss, or concealed unavailable states.

### Unit 19.2c.7: Canonical Home Franchise Advantage

#### Completed

Added a canonical Home franchise advantage feature for one-row-per-game
historical and upcoming inputs.

The feature produces one Home Franchise HFA value rather than mirrored Away and
Home coefficients.

Defined the coefficient as the Home franchise's prior home win rate minus the
prior league-average home win rate.

Calculated the coefficient from completed, non-neutral games strictly before
the target season and week.

Included all prior seasons and earlier weeks in the target season without using
target-week or future results.

Counted ties as one-half home wins.

Preserved the established minimum sample policy. Franchises with fewer than
twenty prior home games receive the intentional zero-valued league-average
prior.

Neutral target games receive zero Home Franchise HFA.

Derived historical home results directly from Home Team, Away Score, Home Score,
and neutral-site identity.

Excluded unplayed and historical neutral-site games from coefficient
calculation.

Rejected malformed target and historical schemas, duplicate historical game
IDs, invalid season labels, and invalid neutral-site values.

Preserved target columns, unrelated values, and caller immutability.

Registered the canonical implementation independently from the existing
TEAM_A/TEAM_B venue-HFA feature.

Left the active feature sequence and legacy implementation unchanged.

#### Goal

Provide one truthful pregame Home franchise advantage value using stable home
identity and only information available before the target game.

#### Tests

Covered registration, the single canonical output, coefficient calculation,
neutral targets, minimum sample behavior, tie accounting, historical neutral
games, current and future exclusion, earlier target-season games, prior-season
history, unplayed rows, empty history, input preservation, schema validation,
duplicate game IDs, invalid neutral state, invalid season labels, and exclusion
of retired orientation fields.

All scoped quality gates and tests pass.

#### Acceptance

Historical and upcoming one-row game frames receive one leakage-free Home
franchise advantage value without TEAM_A, TEAM_B, HOME_FIELD, WINNER, LOSER,
GAME_LOCATION, mirrored coefficients, target-week leakage, or future-result
leakage.

### Unit 19.2c.8: Canonical Game-Level Features

#### Completed

Added schedule-complete canonical divisional, primetime, dome, and weather
features for one-row-per-game historical and upcoming inputs.

Added a shared internal metadata resolver that combines completed historical
game metadata with the rich upcoming schedule by Game ID.

Normalized historical and upcoming field names while preserving one stable
canonical Game ID lookup.

Accepted identical overlapping metadata and selected a populated value when the
other source was unavailable.

Rejected conflicting non-null metadata for the same game and field.

Tolerated an unavailable rich upcoming artifact so historical feature
generation remains operational.

Added canonical divisional-game identity using historical DIV_GAME and upcoming
divisional metadata.

Added canonical primetime identity using historical and upcoming weekday and
kickoff-time metadata.

Preserved the established Monday, Sunday night, Thursday night, and Saturday
night primetime rules.

Represented missing, unknown, or malformed required primetime metadata as null
rather than silently classifying the game as non-primetime.

Added canonical dome and weather features using historical and upcoming roof
metadata plus weather observations keyed by Game ID.

Reused the established weather conversions, precipitation classifications,
visibility handling, derived fields, and controlled-environment defaults.

Known dome games receive controlled environmental values even when no weather
artifact is available.

Known outdoor games without weather coverage retain explicit null weather
values.

Missing roof metadata leaves dome state explicitly null rather than defaulting
to outdoor.

Preserved target rows, target order, unrelated columns, and caller
immutability.

Confirmed that no canonical replacement for perspective-relative HOME_FIELD is
needed. Stable Away Team, Home Team, and neutral-site identity already express
the canonical game orientation.

Registered the canonical implementations independently from the active legacy
features.

Left the active feature sequence, model columns, feature engineering, and
modeling pipeline unchanged.

#### Goal

Complete the canonical game-level feature family so historical and upcoming
one-row game inputs use the same schedule-complete metadata and weather path.

#### Tests

Covered shared historical and upcoming metadata resolution, identical
overlaps, null-versus-populated values, conflicting metadata, missing rich
schedule data, divisional state, primetime rules, nullable schedule metadata,
historical and upcoming roof state, weather conversions, dome defaults,
missing outdoor weather, missing roof metadata, row and order preservation,
input immutability, required Game ID validation, and exclusion of retired
orientation fields.

All scoped quality gates and tests pass.

#### Acceptance

Historical and upcoming one-row game frames receive divisional, primetime,
dome, and weather features through one canonical Game ID path without TEAM_A,
TEAM_B, HOME_FIELD, completed-history dependence for upcoming games, silent
metadata substitution, or row loss.

### Unit 19.2c.9: Canonical Feature Sequence and Output Contract

#### Completed

Defined one ordered canonical feature sequence for historical and upcoming
one-row game inputs.

The sequence includes canonical Elo, EPA, rest, record, schedule strength,
travel, Home franchise advantage, divisional, primetime, and weather features.

Validated the canonical sequence independently from the active legacy feature
sequence.

Confirmed that canonical schedule strength executes after canonical Elo, its
only declared feature dependency.

Added a canonical feature-column helper that derives the ordered output contract
directly from registered feature specifications.

Rejected duplicate output declarations across canonical features.

Verified that the declared canonical feature schema excludes TEAM_A, TEAM_B,
HOME_FIELD, RESULT, and prefixed retired-orientation columns.

Confirmed representative outputs from every canonical feature group are present
in the declared schema.

Verified sequence composition preserves one row per historical or upcoming
input game, canonical identity and target fields, input order, unrelated
columns, and caller immutability.

Verified unavailable feature outputs remain explicit null values without
dropping target games.

Left the active legacy feature sequence, model-input builder, model columns,
feature engineering, manifest schema version, and data version unchanged.

#### Goal

Define and validate the complete canonical feature sequence and its enriched
output contract before activating the one-row modeling pipeline.

#### Tests

Covered the exact canonical sequence, registry identity, dependency ordering,
Elo-before-schedule-strength ordering, ordered feature-spec output collection,
output uniqueness, retired-orientation exclusion, feature-group coverage,
duplicate-output rejection, legacy-sequence isolation, row and order
preservation, canonical identity and target preservation, input immutability,
unique output columns, and explicit null states.

The sequence-composition test uses controlled feature stubs to validate
orchestration and schema behavior. Individual canonical feature tests validate
the underlying calculations and dataset interactions.

All scoped quality gates and tests pass.

#### Acceptance

The project has one registered and dependency-valid canonical feature sequence
whose declared output schema is unique, schedule-complete, compatible with
one-row historical and upcoming game inputs, and free of TEAM_A, TEAM_B,
HOME_FIELD, and RESULT.

The active legacy feature pipeline and persisted development artifacts remain
unchanged pending canonical model-contract migration.

### Unit 19.2d: Activate One-Row Feature Pipeline

#### Completed

Defined canonical model-facing feature sets using stable Away and Home columns.

Defined all generated Elo and EPA differentials as Home minus Away.

Updated the feature-set contracts to 37 differential, 74 raw, 111 combined,
and 152 expanded columns while preserving the diff, raw, combined, and expanded
lookup keys.

Activated the canonical historical modeling-table builder and canonical feature
sequence.

The active modeling pipeline now produces one Away/Home-oriented base and
enriched row per completed game.

Changed persisted and incremental artifact identity to one row per Game ID.

Changed manifests to record the canonical feature sequence and feature-column
contract.

Replaced the data-version-only stale check with combined schema-version and
data-version compatibility validation.

Bumped the modeling schema version to 5 and data version to 2 so retired
development artifacts force a full canonical rebuild.

Updated shared fixtures, integration tests, and end-to-end tests for the
canonical source and artifact schemas.

Completed a full real-data canonical rebuild in 65.8 seconds.

The rebuilt artifacts contain 7,276 cleaned games, 7,276 base rows, and 7,276
enriched rows.

Both modeling artifacts contain unique Game IDs, unique columns, all required
canonical identity, target, and feature columns, and no TEAM_A, TEAM_B,
HOME_FIELD, or RESULT columns.

Validated Actual Margin as Home Score minus Away Score and Actual Total as Home
Score plus Away Score across the complete artifact.

Validated that the manifest contains the exact canonical feature sequence,
canonical feature-column contract, schema version 5, data version 2, and 7,276
rows.

Reviewed canonical feature null counts. Missing values remain concentrated in
truthfully unavailable rolling state, schedule strength, rest, and scheduling
metadata rather than causing row loss or default substitution.

#### Goal

Activate one canonical one-row-per-game historical modeling artifact and
feature pipeline.

#### Tests

Covered canonical model feature declarations, Home-minus-Away differential
direction, feature-set ordering and metadata, artifact-version compatibility,
manifest generation, one-row incremental identity, fixture consistency,
integration pipeline execution, and end-to-end artifact creation.

Completed a real full-history artifact rebuild and validated row counts, Game ID
uniqueness, column uniqueness, canonical schema completeness, target identities,
retired-orientation exclusion, manifest integrity, and feature null counts.

All quality gates, scoped tests, integration tests, end-to-end tests, and
real-data validation pass.

#### Acceptance

The active modeling build produces exactly one canonical Away/Home row for each
of 7,276 cleaned historical games, runs the canonical feature sequence,
persists the canonical schema and manifest, and contains no TEAM_A, TEAM_B,
HOME_FIELD, or RESULT dependency.

### Unit 19.2e: Migrate Win Model

#### Completed

Migrated the Win classification target from perspective-relative RESULT to
canonical HOME_WIN.

The positive classification class now means that the designated Home team won.

Excluded tied games through the nullable HOME_WIN target rather than the retired
RESULT value of 0.5.

Added explicit validation when canonical Win modeling data does not contain the
HOME_WIN target.

Preserved deterministic chronological training order by Year, Week Number,
Game Date, and Game ID.

Preserved feature and target alignment while filtering rows with unavailable
model features.

Migrated historical classification prediction assembly to one canonical row per
game.

Model positive-class probability now maps directly to Home Win Probability.
Away Win Probability is derived as its complement.

Removed classification prediction dependence on two-row selection, TEAM_A,
TEAM_B, HOME_FIELD, and neutral-site alphabetical orientation.

Canonical Away Team and Home Team identities pass directly from source rows to
prediction outputs, including neutral-site games.

Canonical Away Elo and Home Elo values pass directly to prediction outputs.

Migrated upcoming Win prediction to the canonical feature sequence.

Corrected numeric and formatted Home and Away probability orientation.

Corrected optional Total alignment after incomplete upcoming rows are filtered.

Added validation for duplicate Game IDs and misaligned probability or Total
prediction counts.

Preserved optional Total attachment, chronological output order, archive
columns, and caller immutability.

Left Total-model training, regression prediction assembly, alternate EPA-window
rebuilding, evaluation, retraining, and artifact regeneration for their
designated later units.

#### Goal

Train and predict Win probability directly from the canonical Home-team
perspective.

#### Tests

Covered canonical target selection, Home and Away outcomes, tie exclusion,
unavailable-feature exclusion, chronological preparation, holdout splitting,
missing-target validation, and input immutability.

Covered direct historical prediction cardinality, Home-positive-class mapping,
Away complement derivation, canonical team and Elo identity, neutral-site
identity preservation, chronological output, optional Total alignment,
duplicate Game IDs, prediction-count validation, and retired-orientation
exclusion.

Covered the upcoming classification lifecycle, canonical feature-sequence
execution, incomplete-row filtering, numeric and formatted probability
orientation, canonical Elo propagation, enrichment input, and Total index
alignment.

All scoped quality gates and tests pass.

#### Acceptance

Win models train on HOME_WIN and directly produce HOME_WIN_PROB, with
AWAY_WIN_PROB derived as its complement and no TEAM_A, TEAM_B, HOME_FIELD,
RESULT, two-row selection, or neutral-site alphabetical orientation.

### Unit 19.2f: Migrate Total Model

#### Completed

Migrated the Total regression target from reconstructed lowercase actual_total
to canonical ACTUAL_TOTAL.

Total training now reads ACTUAL_TOTAL directly from the persisted canonical
modeling artifact.

Removed Total-training dependence on the cleaned games dataset, PTS_WINNER,
PTS_LOSER, and a secondary Game ID target merge.

Preserved tied games because their combined score remains a valid Total target.

Added explicit validation when canonical Total modeling data does not contain
ACTUAL_TOTAL.

Preserved deterministic chronological training order by Year, Week Number,
Game Date, and Game ID.

Preserved feature and target alignment while filtering rows with unavailable
targets or model features.

Migrated historical Total prediction assembly to one canonical input row and
one canonical output row per game.

Migrated upcoming Total prediction to the canonical feature sequence.

Canonical Away Team and Home Team identities now pass directly to historical
and upcoming Total prediction outputs, including neutral-site games.

Removed regression prediction dependence on TEAM_A, TEAM_B, HOME_FIELD,
two-row grouping, perspective selection, and neutral-site alphabetical
orientation.

Added validation for duplicate Game IDs and prediction-count mismatches.

Removed opportunistic Total execution from historical and upcoming Win
prediction.

Removed the default Total model setting and the internal Win-to-Total prediction
helper.

Win predictors now produce only Win probabilities. Total predictors now produce
only model_total. Weekly-product orchestration remains responsible for selecting
and combining independent forecast events.

Preserved Total model registration under total_random_forest and total_xgboost.

#### Goal

Train and predict Total points directly from the canonical one-row game
perspective, independently from Win prediction.

#### Tests

Covered canonical ACTUAL_TOTAL target selection, persisted-target use, tie
retention, unavailable-target and unavailable-feature filtering, chronological
preparation, holdout splitting, missing-target validation, unique fixture
columns, and caller immutability.

Covered direct historical Total prediction cardinality, chronological
alignment, canonical Away and Home identity, neutral-site identity
preservation, output schema, duplicate Game ID rejection, prediction-count
validation, input immutability, and exclusion of retired orientation fields.

Covered the upcoming Total lifecycle, canonical feature-sequence execution,
incomplete-row filtering, model execution, model identity, and independent
model_total output.

Verified that predictor production code contains no TEAM_A, TEAM_B, or
HOME_FIELD references.

Verified that no Win-to-Total execution helper or default Total model setting
remains.

All scoped quality gates and tests pass.

#### Acceptance

Total models train on ACTUAL_TOTAL and independently produce one model_total per
canonical game without TEAM_A, TEAM_B, HOME_FIELD, winner/loser score
reconstruction, two-row selection, neutral-site orientation recovery, or Win
predictor side effects.

### Unit 19.2g: Migrate Tuning, Evaluation, and Retraining

#### Completed

Migrated alternate EPA-window rebuilding to the canonical Away/Home schema.

Alternate EPA windows delegate to HomeAwayEpaFeature so standard feature
generation and tuning share one implementation for rolling values, lookahead
prevention, source validation, missing-data behavior, and Away/Home joins.

Preserved the four-game fast path and replaced persisted four-game values when
another tuning window is requested.

Validated canonical walk-forward Win and Total dispatch.

Walk-forward evaluation trains from the canonical modeling artifact, filters
incomplete target-season rows, and predicts one canonical row per game.

Updated walk-forward row-sufficiency documentation for one row per game.

Migrated calibration refresh to canonical Actual Margin.

Calibration joins archived forecasts directly to the canonical modeling
artifact by Game ID and consumes signed Home-oriented Actual Margin without
winner/loser score reconstruction.

Added validation for missing Actual Margin and duplicate canonical Game IDs.

Migrated player situational Home and Away cohorts to direct Home Team and Away
Team identity.

Neutral-site situational splits preserve their designated schedule identity,
and unmapped player teams are classified as neither Home nor Away.

Migrated Elo tuning team-population construction to canonical Away Team and
Home Team identity.

WIN_OR_TIE remains only as the completed-game filter for Elo tuning and is not
used for orientation.

Updated evaluation manifest examples to canonical Away/Home identity and Elo
features.

Removed migrated execution-path dependence on TEAM_A, TEAM_B, HOME_FIELD,
RESULT, winner/loser score reconstruction, and lowercase actual_total.

#### Goal

Migrate alternate-window tuning, walk-forward evaluation, calibration, and full
retraining to the canonical one-row game contract.

#### Tests

Covered alternate EPA windows, lookahead prevention, missing EPA sources,
canonical target preservation, one-row identity, and input immutability.

Covered walk-forward Win and Total dispatch, positive-class orientation,
incomplete-feature filtering, cutoff arguments, and empty target seasons.

Covered canonical calibration margins, negative Home margins, missing targets,
duplicate Game IDs, and persisted calibration values.

Covered canonical situational Home and Away cohorts, neutral-site identity,
unknown team handling, duplicate game metadata, all existing cohort outputs,
and input immutability.

Covered canonical Elo tuning team populations, unplayed-game filtering, missing
source columns, empty identities, season grouping, and input immutability.

Covered canonical evaluation manifest examples and generic manifest validation.

All scoped quality gates and tests pass.

#### Acceptance

Tuning, evaluation, calibration, and retraining operate on canonical one-row
game data without TEAM_A, TEAM_B, HOME_FIELD, RESULT, winner/loser score
reconstruction, lowercase actual_total, or legacy probability orientation.

### Unit 19.2h: Re*ove Retired Orientation and Rebuil* Artifacts

#### Completed

Remove* the inactive legacy feature seque*ce from the modeling pipeline.

Re*oved the symmetric two-row-per-gam* base modeling-table builder.

The*pipeline now exposes one validated*canonical feature sequence and one*canonical one-row-per-game modelin*-table builder.

Removed legacy fe*ture-order validation from pipelin* import.

Updated feature manifest*and registry documentation to desc*ibe canonical
Away/Home feature na*es and outputs.

Preserved the exi*ting canonical sequence name becau*e it is already the stable
contrac* used across modeling, prediction,*tuning, and evaluation.

All scope* quality gates, canonical feature *ests, integration tests, and
end-t*-end prediction-pipeline tests pas*.

#### Goal

Remove the retired p*rspective-oriented feature archite*ture and rebuild all
game-model ar*ifacts from the canonical one-row *way/Home contract.

#### Tests

Le*acy pipeline entry-point removal a*d the active canonical pipeline ar*
covered.

Legacy feature implemen*ations, registrations, fixtures, s*hema guards, and
artifacts remain *o be removed.

#### Acceptance

Th* repository contains one canonical*game feature pipeline and no activ*
TEAM_A, TEAM_B, HOME_FIELD, RESUL*, two-row modeling, compatibility,*or retired
artifact dependency.


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
