# Gridiron Edge — Development Plan

> **Purpose:** the active implementation plan for the currently selected
> program and bounded unit. Completed program details belong in `CHANGELOG.md`,
> durable architecture in `DECISIONS.md`, current operations in `HANDOFF.md`,
> and future priorities in `ROADMAP.md`.

## Where to find other information

| Document | Role |
|----------|------|
| **PLAN.md** (this file) | The active program and its one bounded implementation unit |
| **ROADMAP.md** | Strategic priorities, genuine future capabilities, and known limitations |
| **CHANGELOG.md** | What was built and when |
| **HANDOFF.md** | How the system works today: architecture, workflows, operations, and recovery |
| **DECISIONS.md** | Append-only architectural decisions and supersession history |

## Ways of Working

These practices apply to every program and implementation unit. A new thread
should read this section before planning or modifying the repository.

1. **Confirm before building.**
   Never assume code, schemas, artifacts, commands, tests, or documentation
   exist or have a particular shape. Inspect the current repository state
   before proposing a change. Prefer a small read-only audit over implementation
   based on stale context.

2. **Locate first, then read the owning boundary.**
   Use targeted searches to identify the relevant files, functions, tests,
   commands, artifacts, and generated contracts. Read the exact owning
   boundaries before designing or editing them.

3. **Design at two levels before implementation.**
   - **Program level:** lock the capability, motivation, boundaries, sequence,
     dependencies, and success criteria in `ROADMAP.md`.
   - **Unit level:** add one bounded implementation unit to `PLAN.md` with its
     goal, design decisions, tests, and acceptance criteria before changing
     code.

4. **Keep one active bounded unit.**
   `PLAN.md` may retain a concise summary of completed programs, but detailed
    completed-unit records are removed during major program closeout. Only one
    implementation unit should be active at a time. New work starts only
    after it is selected from `ROADMAP.md` and scoped for execution.

5. **Use descriptive implementation language.**
   Program and unit identifiers belong in planning documents only. Source
   names, comments, docstrings, tests, artifacts, commands, and commit subjects
   should describe lasting domain behavior rather than when the behavior was
   added.

6. **Commit small coherent units.**
   Each completed unit should produce one focused commit with a Conventional
   Commit subject and a detailed bullet-list body covering implementation,
   tests, and documentation. The corresponding `PLAN.md` update belongs in the
   same commit as the unit's implementation.

7. **Run focused gates during implementation.**
   Run linting, type checking, and focused tests after each meaningful change.

   At a Python contract boundary, run:

   ```bash
   uv run ruff check . --fix && \
   uvx pyrefly check && \
   uv run pytest -m "unit and not slow"
   ```

   At a frontend contract boundary, run:

   ```bash
   cd frontend
   pnpm lint
   pnpm build
   pnpm test:run
   cd ..
   ```

   Run integration, end-to-end, external-source, network, or slow gates when
   the affected boundary requires them.

8. **Verify against real artifacts and responses.**
   After backend, data, model, API, or frontend integration changes, inspect
   the generated artifact or live response directly. Validate the relevant row
   counts, identities, uniqueness, coverage, provenance, representative
   values, timestamps, joins, and blocker states. Green tests do not replace
   real-data verification.

9. **Preserve generated-file ownership.**
   Regenerate checked-in schemas, clients, and derived contracts through their
   owning commands. Do not hand-edit generated artifacts.

10. **Close each unit completely.**
    After implementation and validation:
    - remove temporary migration and diagnostic scripts;
    - update affected operational and architectural documentation;
    - condense the completed `PLAN.md` unit to exactly these headings:
      `Completed`, `Goal`, `Files Added/Removed/Changed`, `Tests`, and
      `Acceptance`;
    - list every committed file added, removed, or changed, grouped by category
      with a concise description of its lasting responsibility or modification;
    - explicitly state `None` when an Added, Removed, or Changed category has no
      entries;
    - include the completed `PLAN.md` update in the same commit as the unit's
      implementation;
    - record durable architectural choices in `DECISIONS.md`;
    - record shipped behavior in `CHANGELOG.md`;
    - update `HANDOFF.md` only when the current operational contract changes;
    - update `ROADMAP.md` when future scope, sequencing, or priority changes;
    - verify that the staged file list agrees with the
      `Files Added/Removed/Changed` section;
    - inspect the staged diff before committing.

    Use this completed-unit structure:

    ```markdown
    #### Completed

    Concise description of the implemented behavior and resulting contract.

    #### Goal

    The lasting purpose of the unit.

    #### Files Added/Removed/Changed

    Added:
    - `path/to/new_file.py` - Lasting responsibility of the new file.
    - None.

    Changed:
    - `path/to/existing_file.py` - Behavioral or contract change.
    - `tests/path/test_file.py` - Regression or acceptance coverage.
    - None.

    Removed:
    - `path/to/retired_file.py` - Superseded responsibility that was removed.
    - None.

    #### Tests

    Focused tests, quality gates, integration checks, and real-data validation
    performed for the unit.

    #### Acceptance

    Concise statement proving the unit's intended contract is implemented,
    validated, documented, and ready for downstream use.
    ```

    Include only categories and files that reflect the committed unit scope.
    Do not list files that were merely inspected. Temporary scripts removed
    before the commit are not committed files and should not appear in the
    file-change inventory.

    Before committing, run:

    ```bash
    git diff --cached --name-status
    git diff --cached --stat
    git diff --cached --check
    ```

11. **Do not preserve development-era compatibility without a current need.**
    Gridiron Edge has never been live in production. Existing development
    schemas, artifacts, commands, tests, generated contracts, and historical
    behavior may be replaced when the active design requires it, unless a
    current contract explicitly requires compatibility.

12. **Use explicit dates and repository history.**
    Trust explicit user-provided dates, commit timestamps, and repository
    history. When dates conflict or are ambiguous, state the exact date being
    used rather than relying on relative wording.

## Planned Implementation Status

### Completed Program: Game Prediction Product

Units 1 through 29 are complete. The canonical weekly prediction, immutable
forecast-event, explicitly selected weekly-product, API serialization, frontend
readiness, and operational verification architecture is implemented and
validated through a real 2026 Week 1 rehearsal.

### Active Program: Betting Market Data

The active program will establish a supported betting-market source, preserve
source-neutral quote contracts, activate real edge and sportsbook workflows,
and enable a larger frontend review against real market data.

Current and upcoming market integration comes first. Historical market archive
and leakage-safe evaluation are a separate later workstream within the same
program. Both will share one normalized quote contract while retaining distinct
storage and operational semantics.

Only one bounded market unit is active at a time.

### Market Unit 1: Select a Supported Provider and Lock the Market Contract [Complete]

#### Completed

Selected The Odds API v4 as the supported provider for current and upcoming NFL
moneyline, spread, and total markets. The selection is based on its documented
NFL event feed, multiple-bookmaker payload, native American odds, event and
bookmaker timestamps, self-service access, and historical featured-market
availability from mid-2020.

Audited the existing source-neutral market store, nflverse schedule adapter,
legacy DraftKings adapter, weekly edge service, readiness diagnostics, CLV
ledger, settings, and ingest CLI. The audit found that the existing schema is a
single-source development contract: `sportsbook` currently conflates provider
and book identity, provider event and bookmaker update timestamps are absent,
and multi-book rows would be collapsed nondeterministically by the current
per-game pivot.

Locked a shared normalized quote contract for current observations and future
historical backfill. Current provider pulls will preserve all returned books and
append observed snapshots while atomically replacing the current snapshot.
Historical provider backfill, partitioning, closing-line policy, and
leakage-safe evaluation remain a separate later workstream.

Revised the Betting Market Data sequence so the normalized storage contract is
migrated before the provider adapter is implemented.

#### Goal

Choose a supported current-market provider and define the quote identity,
provenance, storage, freshness, failure, command, and downstream selection
contracts precisely enough to implement without guessing or coupling forecast
publication to network access.

#### Files Added/Removed/Changed

Added:
- None.

Changed:
- `PLAN.md` - Closed provider selection and opened the normalized quote-contract migration.
- `ROADMAP.md` - Refined the active program sequence around contract migration, provider ingestion, operational integration, frontend validation, and multi-book shopping.
- `DECISIONS.md` - Added D25 for The Odds API selection and the shared current/historical quote boundary.

Removed:
- None.

#### Tests

Reviewed official provider documentation for NFL coverage, featured markets,
American odds, multiple bookmakers, event and bookmaker timestamps, public
plans, and historical NFL featured-market availability. Compared the documented
capabilities with Odds-API.io, SportsDataIO, and the publicly available
Sportradar NFL material.

Audited the current odds schema, snapshot and ledger behavior, adapters, edge
pivot, freshness diagnostics, CLV consumers, settings, and CLI ownership.
Confirmed that the existing edge pivot groups by game rather than sportsbook
and therefore cannot safely consume a multi-book snapshot until the downstream
selection contract is migrated.

#### Acceptance

The Odds API v4 is the supported current and upcoming NFL market provider. The
normalized quote contract, source-versus-book provenance, observed-history
boundary, current-snapshot behavior, failure semantics, explicit ingest command,
and multi-book downstream requirements are locked in D25. Historical provider
backfill remains separate. Market Unit 2 can migrate storage and validation
without reopening provider selection or operational ownership.

---

### Market Unit 2: Migrate the Source-Neutral Quote Contract [Complete]

#### Completed

Replaced the development-era odds schema with the canonical 17-column
provider-aware quote contract. Separated upstream provider identity from the
sportsbook offering each price and added provider event, sportsbook update,
commence-time, and live-state provenance.

Rewrote generic quote storage around exact schema validation, canonical UTC
timestamps, row-level observation identity, atomic Parquet replacement, and
multi-book-safe persistence. Exact repeated observations are idempotent while
later observations, changed prices, changed lines, and distinct sportsbooks
remain independently representable.

Migrated the nflverse schedule adapter to truthful consensus provenance using
`provider=nflverse`, null sportsbook and provider-event identity, and explicit
pregame state. Preserved six market-side rows per game and the canonical spread
orientation.

Removed the retired DraftKings adapter, game resolver, ingest command, exports,
provider-specific generic-store conversion, fixtures, and tests.

Replaced ambiguous market-source provenance with explicit market providers and
sportsbooks across readiness, edge diagnostics, verify-week output, API schemas,
serializers, OpenAPI, generated TypeScript contracts, and related tests.

Regenerated the local current snapshot and observation ledger under the new
schema. The resulting artifacts contain 96 rows across 16 games with six rows
per game, truthful nflverse provenance, no fabricated sportsbooks, valid UTC
timestamp columns, zero spread-orientation violations, zero duplicate
observations, and idempotent exact reappend behavior.

#### Goal

Establish one provider-aware, multi-book-safe current quote contract before
implementing The Odds API client and operational ingestion workflow.

#### Tests

Ruff, Pyrefly, and the unit test boundary passed. Focused odds-store, nflverse
adapter, readiness, edge diagnostics, weekly edge service, CLI, API schema,
serializer, route, and odds-join tests passed.

OpenAPI and frontend TypeScript contracts were regenerated through their owning
commands. Frontend lint, production build, and all 344 frontend tests passed.

Real-artifact validation confirmed 96 rows across 16 games, six canonical
market-side rows per game, `provider=nflverse`, null sportsbook and
provider-native timestamps, pregame-only state, UTC timestamp dtypes, zero
spread-orientation violations, zero duplicate observations, and idempotent
ledger reappend behavior.

The full integration/e2e run exposed 13 failures outside the market-contract
path in select-model smoke output, team field-status metadata, compare-team
fixtures, and a stale weekly-product roundtrip fixture.

#### Acceptance

The nflverse adapter and generic market storage use the canonical
provider-aware quote contract. Provider and sportsbook provenance remain
distinct through domain, CLI, API, and generated frontend contracts. The
retired DraftKings path is absent.

A real current snapshot and observation ledger satisfy the locked schema,
identity, atomicity, orientation, and idempotency requirements. Market Unit 3
can implement The Odds API client and parser directly against this contract
without compatibility code or schema migration.

---

### Market Unit 3: Implement The Odds API Client and Parser [Complete]

#### Completed

Implemented The Odds API v4 client for current NFL featured markets using the
locked US region, moneyline, spread, total, American-odds, and ISO-timestamp
request contract.

Added strict request, response, quota-header, event, bookmaker, market, outcome,
numeric, timestamp, and schedule-matching validation. Preserved every returned
sportsbook independently and normalized matched pregame events directly into
the canonical provider-aware quote schema.

Added write-safe current ingestion. Request, HTTP, JSON, malformed-payload,
empty-response, and zero-match failures leave existing quote artifacts
unchanged. Successful ingestion appends observations and atomically replaces
the current snapshot.

Added `ODDS_API_KEY` configuration, explicit flag resolution, quota reporting,
and the isolated `gridiron ingest odds --season ... --week ...` command. Normal
prediction, retraining, post-week, verification, and data-pipeline workflows do
not perform provider network access.

Validated a live NFL Week 1 provider response containing 816 quotes across 16
canonical games and nine sportsbooks. The resulting snapshot contains complete
provider event, sportsbook, update-time, and commence-time provenance with no
duplicate current book-side rows.

#### Goal

Implement supported current NFL market ingestion directly against the canonical
quote contract without coupling forecast publication to network availability.

#### Tests

Ruff, Pyrefly, and the unit quality boundary passed. Parser, HTTP client,
settings, API-key resolution, write-safe ingestion, storage, nflverse
regression, command registration, help, validation, quota reporting, failure,
artifact-preservation, partial-coverage, and idempotency tests passed.

The live provider request cost three credits and returned 816 quotes, 16
provider events, 16 matched games, nine sportsbooks, and 144 game-book
combinations. Of those combinations, 120 offered all three requested market
families and 24 offered two. All emitted rows contained odds.

Live artifact validation confirmed the exact 17-column schema, UTC timestamp
dtypes, pregame-only rows, zero provider-event identity violations, zero
duplicate current sportsbook-side rows, and zero duplicate ledger observations.
The ledger contains 96 retained nflverse observations and 816 The Odds API
observations.

#### Acceptance

The explicit ingest command requests supported NFL featured markets, retains
all returned sportsbooks, matches usable pregame events to canonical games,
appends quote observations, atomically replaces the current snapshot, and
reports available provider quota metadata.

Missing configuration and all pre-write request, payload, parsing, and matching
failures preserve existing artifacts. Provider network access remains isolated
from normal weekly and composite workflows.

---

### Market Unit 4: Integrate Current Markets Operationally [Complete]

#### Completed
Preserved sportsbook-specific market offers through edge calculation, diagnostics, API, CLI, CSV, frontend selection, and Bet Slip staging. Added persisted all-or-selected sportsbook preferences, deterministic compact-offer selection, and sportsbook-specific Bet Slip v3 identities with immutable quote provenance.

#### Goal
Provide truthful multi-sportsbook edge recommendations without collapsing quote identity, while allowing users to control which sportsbooks are eligible across the frontend.

#### Tests
Validated sportsbook-aware recommendation generation, market-family diagnostics, serialization, API routes, CLI and CSV output, OpenAPI generation, Settings persistence, full-table filtering, deterministic compact selection, and Bet Slip v3 parsing and identity. Python quality gates and integration tests passed. Frontend lint, production build, and all 362 tests passed.

#### Acceptance
Each eligible sportsbook offer remains independently traceable by provider event, sportsbook, game, market, side, price, and timestamps. Users can select all or specific sportsbooks. Full tables retain eligible offers, compact surfaces select one deterministic best eligible offer, and matching wagers from different sportsbooks can coexist on the Bet Slip.

---

### Market Unit 5: Audit the Frontend Against Real Multi-Book Markets [Complete]

#### Completed
Audited the frontend against the real current multi-book market snapshot and corrected recommendation density, browser navigation, responsive presentation, and accessibility defects. Added a shared wager-family grouping contract, collapsed Model Edges and Available Edges to one best eligible offer per game-market-side family, preserved expandable sportsbook alternatives, repaired browser Back and Forward history, and made primary market surfaces responsive.

#### Goal
Validate the existing frontend market experience against real current multi-book data and correct presentation, state, traceability, responsive-layout, navigation, and accessibility defects without expanding into the separate Line Shopping product.

#### Tests
Audited 816 normalized quotes across 16 games and nine sportsbooks, producing 341 positive edge offers across 45 wager families. Validated deterministic best-offer grouping, consensus fallback, selected-sportsbook filtering, differing alternative lines and prices, sportsbook-specific Bet Slip staging, browser history traversal, direct-detail refresh behavior, responsive layouts, labeled table regions, explicit interaction controls, and cleanup of nested interactive table semantics. Frontend lint, TypeScript, production build, complete frontend tests, Python quality gates, and repository tests passed. Manual responsive and browser-navigation acceptance checks passed.

#### Acceptance
Recommendation surfaces show one best eligible sportsbook offer per wager family while preserving access to every eligible selected-book alternative. Model Edges limits by wager family rather than raw offer count. Expanded offers preserve their own sportsbook, line, price, EV, strength, and Bet Slip identity. Browser Back, Forward, direct links, and refresh retain route parameters. Dashboard, Settings, Game Detail, and market tables remain usable at standard and narrow widths. Interactive controls use explicit semantic buttons without nested row-button behavior. Line Shopping remains the dedicated future all-books-upfront comparison surface.

#### Files Added
- frontend/src/context/NavContext.test.tsx

#### Files Removed
- None

#### Files Changed
- frontend/src/App.css
- frontend/src/App.tsx
- frontend/src/components/betslip/EdgesTable.test.tsx
- frontend/src/components/betslip/EdgesTable.tsx
- frontend/src/components/dashboard/FeaturedMatchupsGrid.tsx
- frontend/src/components/dashboard/ModelEdgesTable.test.tsx
- frontend/src/components/dashboard/ModelEdgesTable.tsx
- frontend/src/context/NavContext.tsx
- frontend/src/index.css
- frontend/src/screens/Dashboard.tsx
- frontend/src/screens/GameDetail.tsx
- frontend/src/screens/Settings.tsx
- frontend/src/utils/sportsbookPreferences.test.ts
- frontend/src/utils/sportsbookPreferences.ts
- PLAN.md

---

### Market Unit 6: Build the Multi-Book Line Shopping Foundation [Completed]

#### Completed

Delivered the current slate-wide multi-book Line Shopping product with exhaustive
exact-offer model evaluation, selected-product guidance, deterministic market
comparison, chronological matchup ordering, persisted visual highlighting, and
accessible beginner-friendly explanations.

#### Goal

Provide a truthful comparison of every current sportsbook Moneyline, Spread, and
Total offer while preserving quote identity and keeping model probability,
expected value, playable thresholds, approval, and preferred-offer selection in
the backend.

#### Files Added/Removed/Changed

Added:
- close_market_unit6_docs.py
- frontend/src/components/primitives/ExplainTooltip.test.tsx
- frontend/src/components/primitives/ExplainTooltip.tsx
- src/gridiron_edge/api/serializers/lines.py
- src/gridiron_edge/market/line_shopping.py
- tests/integration/api/test_lines_routes.py
- tests/unit/api/test_serializers_lines.py
- tests/unit/market/test_line_shopping.py
- tests/unit/market/test_line_shopping_guidance.py

Removed:
- None

Changed:
- PLAN.md
- ROADMAP.md
- api-schema.json
- frontend/src/App.css
- frontend/src/api/hooks.ts
- frontend/src/context/AppStateContext.tsx
- frontend/src/screens/LineShopping.test.tsx
- frontend/src/screens/LineShopping.tsx
- src/gridiron_edge/api/app.py
- src/gridiron_edge/api/routes/lines.py
- src/gridiron_edge/api/schemas/_base.py
- src/gridiron_edge/api/schemas/lines.py
- src/gridiron_edge/api/serializers/teams.py
- tests/integration/api/test_api_contract.py
- tests/unit/api/test_app_routes.py
- tests/unit/api/test_schemas_lines.py
- tests/unit/api/test_serializers_teams.py

#### Tests

- Passed the full Python quality gates: Ruff, Pyrefly, and the unit test suite
  excluding slow tests.
- Passed the full frontend quality gates: ESLint, TypeScript production build,
  and all frontend tests.
- Verified the generated OpenAPI Line Shopping model-guidance contract and the
  regenerated typed frontend client.
- Validated the selected 16-game weekly product against the real 816-offer
  current snapshot: 96 outcome-guidance rows, 341 model-approved offers, and 61
  preferred approved offers.
- Validated the live Spread response: 16 games, 254 exact offers, 32 guidance
  rows, 99 approved offers, 20 preferred offers, one selected product identity,
  and a -110 reference price.
- Manually validated nine scope-wide sportsbook columns, BetMGM unavailable
  Spread cells, persisted highlight toggling, centered offer presentation,
  chronological kickoff display, and hover, focus, and tap explanations.

#### Acceptance

The Line Shopping product retains every exact sportsbook quote and its
provenance, classifies line and price quality independently, evaluates every
available offer against the explicitly selected weekly product, and preserves
negative-EV, break-even, unavailable-model, and partial-coverage states. Spread
and Total outcomes show continuous playable guidance at a documented -110
reference price, while Moneyline outcomes show model win probability and fair
American odds. Preferred approved offers preserve maximum-EV ties. The frontend
performs no probability or EV calculation, remains fully usable with highlights
disabled, orders matchups chronologically, displays Eastern kickoff times, and
provides accessible beginner-friendly explanations for wager outcomes, pushes,
American prices, model EV, and market classifications. Arbitrage, middle
detection, movement, and historical market evaluation remain deferred.

---

### Market Unit 7: Separate Prediction, Value, and Recommendation Semantics [Completed]

#### Completed

Delivered distinct Line Shopping semantics for model likelihood, exact-offer
value, and future recommendation qualification, with independently persisted
visual controls and value-specific presentation.

#### Goal

Prevent positive-EV offer highlighting from implying that the outcome is the
predicted winner or that Gridiron Edge recommends placing the wager.

#### Files Added/Removed/Changed

Added:
- None

Removed:
- None

Changed:
- PLAN.md
- ROADMAP.md
- frontend/src/App.css
- frontend/src/context/AppStateContext.tsx
- frontend/src/screens/LineShopping.test.tsx
- frontend/src/screens/LineShopping.tsx

#### Tests

- Passed frontend ESLint, the TypeScript production build, and the full frontend
  test suite.
- Passed focused ExplainTooltip and Line Shopping tests.
- Verified persisted nested display defaults, independent layer controls, and
  master-switch restoration without retaining the retired
  `lineShoppingHighlights` state.
- Verified a Moneyline model underdog can be a +EV candidate while the opposing
  side remains the model favorite.
- Manually reviewed Spread, Total, and Moneyline presentation and confirmed that
  recommendation styling should remain unavailable until its policy exists.

#### Acceptance

Line Shopping separately presents model favorite or underdog, +EV candidate,
preferred +EV offer, best line, and best exact-line price. Users can independently
control the visual layers, while the master Value highlights switch temporarily
suppresses offer decoration without resetting those choices. Value styling uses
a distinct teal treatment and the interface explicitly states that +EV is
neither the predicted winner nor a recommended wager. Existing exact-offer
evaluation, chronological ordering, sportsbook filtering, responsiveness, and
accessible explanations remain intact. Recommended-bet functionality remains
unavailable pending an empirically validated edge, reliability, freshness,
sizing, and portfolio-exposure policy.

---

### Market Unit 8: Establish Recommendation Qualification Diagnostics [Completed]

#### Completed

Established a pure, immutable recommendation-qualification diagnostic contract
for exact evaluated sportsbook offers. The contract reports what passed, failed,
or remains unavailable without assigning a qualified or recommended-bet state.

#### Goal

Create an explicit analytical boundary between a positive-EV candidate and any
future recommended-bet policy while preserving model, product, forecast, quote,
freshness, sizing, and unavailable-policy evidence.

#### Files Added/Removed/Changed

Added:
- src/gridiron_edge/market/qualification.py
- tests/unit/market/test_qualification.py

Removed:
- None

Changed:
- PLAN.md

#### Tests

- Passed focused Ruff, Pyrefly, and qualification unit-test gates.
- Passed the full Python quality gates and complete non-slow unit test suite.
- Validated frozen qualification contracts, canonical check ordering, and
  deterministic JSON-compatible serialization.
- Validated negative, break-even, missing-EV, model-unavailable, and
  uncertainty-unavailable offers as not candidates.
- Validated forecast event, run, model, game, season, week, and live-role
  provenance.
- Validated Moneyline and Spread against selected Win provenance and Total
  against selected Total provenance.
- Validated missing forecast provenance as unavailable rather than fabricated.
- Validated optional UTC quote-freshness evaluation with an inclusive cutoff.
- Validated sizing availability remains informational.
- Validated unavailable empirical edge, reliability, exposure, concentration,
  and correlation policies remain present in every candidate result.

#### Acceptance

Every exact evaluated sportsbook offer can produce deterministic qualification
diagnostics describing candidate eligibility, model availability,
selected-product provenance, immutable forecast provenance, quote identity,
timestamp evidence, optional freshness, and sizing availability. Positive-EV
offers remain qualification-unavailable while empirical edge, reliability, and
exposure policies are unavailable. No result is labeled qualified or
recommended, and no API or frontend recommendation state has been introduced.

---

### Market Unit 9: Retire Unsupported Historical Quote Interpretation [Completed]

#### Completed

Retired operational opening, closing, and closing-line-value behavior that was
not supported by the provider-aware historical quote evidence contract.
Historical quote rows remain available as observations, but no command or
settlement path interprets the first or last stored observation as a validated
market boundary.

Preserved the pure Moneyline, Spread, and Total CLV calculations for future use
with a separately validated, same-source, sportsbook-specific, pre-kickoff
quote-selection policy. Preserved nullable CLV fields and unavailable API states
without introducing a replacement historical-selection policy.

#### Goal

Remove unsupported historical quote interpretation while preserving pure CLV
math and the future data contract needed by a validated closeout workflow.

#### Files Added/Removed/Changed

Added:
- None

Removed:
- None

Changed:
- PLAN.md
- src/gridiron_edge/betting/ledger.py
- src/gridiron_edge/cli/betting.py
- src/gridiron_edge/cli/edges.py
- src/gridiron_edge/market/__init__.py
- src/gridiron_edge/market/clv.py
- tests/integration/test_edges_cli.py
- tests/unit/betting/test_ledger.py
- tests/unit/cli/test_edges.py
- tests/unit/market/test_clv.py
- tests/unit/market/test_weekly_edge_architecture.py

#### Tests

- Passed focused Ruff formatting and lint checks.
- Passed focused Pyrefly checks.
- Passed focused market, betting, CLI, API serializer, and integration tests.
- Passed the full Ruff, Pyrefly, and non-slow unit-test quality gates.
- Validated that `gridiron edges report` remains registered and available.
- Validated that `gridiron edges clv` is no longer registered.
- Validated that unsupported opening and closing selectors and the historical
  CLV report builder have no remaining source or test references.
- Validated that bet settlement records settlement results and PnL without
  loading historical quote observations.
- Validated that closing line, closing odds, and CLV remain null until a future
  validated closeout policy owns those fields.
- Validated pure probability-based and point-based CLV calculations.
- Validated CLV summaries consume only explicitly supplied validated values.

#### Acceptance

No operational path treats the first or last stored quote observation as a
validated opening or closing line. The edge CLI no longer offers unsupported
historical CLV analysis, and bet settlement no longer populates closing or CLV
fields from unvalidated observation selection. Pure scalar CLV math remains
available for a future same-source, sportsbook-specific, pre-kickoff closing
policy, while existing API and frontend fields truthfully remain unavailable.

---

### Market Unit 10: Stabilize Historical Quote Observation Evidence [Completed]

#### Completed

Hardened the existing provider-aware historical quote ledger with deterministic
observation ordering, exact-replay idempotence, same-fetch conflict detection,
and explicit retention of later unchanged, changed-price, and changed-line
observations.

Added pure historical coverage diagnostics that describe provider, sportsbook,
game, market-identity, fetch, timestamp, live-state, kickoff-metadata, and
repeated-observation depth without interpreting observations as opening,
closing, movement, CLV, backtest, or recommendation evidence.

Defined truthful provider-ingestion behavior when historical observation
persistence succeeds but current-snapshot replacement fails. The historical
observation remains recorded, the prior current snapshot remains protected by
its independent atomic-write boundary, and ingestion raises an explicit partial-
persistence error.

#### Goal

Harden the existing provider-aware quote observation ledger as the authoritative
historical market evidence source while preserving a strict separation between
stored observations and any future historical market interpretation.

#### Files Added/Removed/Changed

Added:
- src/gridiron_edge/market/history_coverage.py
- tests/unit/market/test_history_coverage.py

Removed:
- None

Changed:
- PLAN.md
- src/gridiron_edge/ingest/odds/store.py
- src/gridiron_edge/ingest/odds/the_odds_api.py
- tests/unit/ingest/odds/test_the_odds_api_ingest.py
- tests/unit/ingest/test_odds_store.py

#### Tests

- Passed focused Ruff formatting and lint checks.
- Passed focused Pyrefly checks.
- Passed focused quote-store, source-neutral storage, provider-parser,
  provider-ingest, historical-coverage, and ingest-CLI tests.
- Passed the full Ruff, Pyrefly, and non-slow unit-test quality gates.
- Validated exact complete-observation replay remains idempotent.
- Validated unchanged observations at later fetch timestamps remain distinct.
- Validated changed-price and changed-line observations at later fetch
  timestamps remain distinct.
- Validated identical same-fetch observations deduplicate.
- Validated conflicting same-fetch market-side observations are rejected.
- Validated invalid new observations leave an existing historical ledger
  unchanged.
- Validated persisted and filtered observation ordering is canonical.
- Validated nullable source-neutral provider-event, sportsbook, update-time, and
  kickoff provenance remains supported.
- Validated empty, unmatched, malformed, and failed provider pulls leave both
  quote artifacts unchanged.
- Validated snapshot replacement failure after successful observation append
  retains the new historical observation, preserves the prior snapshot, and
  raises an explicit partial-persistence error.
- Validated historical coverage for empty, single-fetch, repeated-fetch,
  multi-provider, multi-book, live, pregame, and missing-kickoff observations.
- Validated repeated unchanged observations count as temporal evidence without
  claiming line or price movement.
- Validated the real historical quote artifact reports its actual temporal
  evidence depth without interpreting observations as market movement.

#### Acceptance

The historical quote ledger deterministically preserves every supported exact
local observation, deduplicates exact replay, retains later observations,
rejects ambiguous same-fetch conflicts, and remains unchanged when validation
fails. Provider ingestion reports partial cross-file persistence truthfully if
history succeeds while current-snapshot replacement fails. Historical coverage
explicitly describes source, scope, timestamp, live-state, kickoff, and
repeated-fetch depth. No result is labeled opening, closing, movement, CLV,
backtest evidence, or recommendation evidence.

---

### Market Unit 11: Define Leakage-Safe Historical Quote Boundaries [Completed]

#### Completed

Implemented pure, provider-aware historical quote-boundary selection for every
exact source, provider-event, sportsbook, game, market, and side identity.

Each result preserves the earliest observed quote and selects the latest
eligible pregame quote only from non-live observations fetched strictly before
a consistent known kickoff. Missing kickoff evidence, conflicting kickoff
timestamps, and histories without an eligible pregame observation remain
explicit rather than being silently resolved.

Preserved observation count, distinct fetch count, and repeated temporal
evidence separately from boundary availability. One-fetch histories can expose
both observed boundaries while remaining visibly shallow and without implying
line movement, price movement, opening, closing, CLV, backtest, or
recommendation evidence.

#### Goal

Define deterministic, leakage-safe observed quote boundaries without merging
providers or sportsbooks and without introducing historical market
interpretation beyond the evidence stored in the canonical quote ledger.

#### Files Added/Removed/Changed

Added:
- src/gridiron_edge/market/history_boundaries.py
- tests/unit/market/test_history_boundaries.py

Removed:
- None

Changed:
- PLAN.md
- src/gridiron_edge/market/__init__.py

#### Tests

- Passed focused Ruff formatting and lint checks.
- Passed focused Pyrefly checks.
- Passed focused historical-boundary, history-coverage, and quote-store tests.
- Passed the full Ruff, Pyrefly, and non-slow unit-test quality gates.
- Validated empty history returns no boundary results.
- Validated one deterministic result is returned for every exact historical
  identity.
- Validated provider, provider event, sportsbook, game, market, and side remain
  part of every boundary identity.
- Validated source-neutral consensus and exact sportsbook histories are never
  merged.
- Validated earliest-observed selection is deterministic and independent of
  input row order.
- Validated latest-eligible-pregame selection includes only non-live rows
  fetched strictly before kickoff.
- Validated live observations are excluded regardless of timestamp.
- Validated observations fetched exactly at kickoff or after kickoff are
  excluded.
- Validated missing kickoff evidence reports kickoff unavailable without using
  date-only game data as a fallback.
- Validated conflicting non-null kickoff timestamps report kickoff conflict.
- Validated histories with no eligible pregame observation preserve their
  earliest-observed evidence.
- Validated one-fetch histories expose both observed boundaries while retaining
  a distinct-fetch count of one and no repeated temporal evidence.
- Validated repeated unchanged observations preserve repeated-fetch depth
  without claiming market movement.
- Validated selected observations preserve line, odds, local fetch time,
  sportsbook update time, kickoff, and live state.
- Validated boundary and selected-observation contracts are immutable.
- Validated the real historical quote artifact using the new boundary selector.

#### Acceptance

Every canonical historical quote identity produces a deterministic,
provider-aware boundary result. The result preserves the earliest observed
quote, selects the latest eligible pregame quote only from non-live observations
fetched strictly before a consistent known kickoff, and reports missing or
conflicting kickoff evidence explicitly. One-fetch histories remain visibly
shallow, consensus and sportsbook histories remain separate, and no result is
labeled opening, closing, movement, CLV, backtest evidence, or recommendation
evidence.

---

### Market Unit 12: Persist Exact Bet Reference Provenance [Completed]

#### Completed

Replaced the development-era bet-ledger contract with an exact reference-offer
provenance boundary that stores recorded wager terms independently from the
market observation that informed the wager.

Added nullable reference provider, provider event, sportsbook, market fetch
time, sportsbook update time, kickoff, American odds, and line fields. Manual
bets remain valid with explicitly absent reference provenance, while
reference-backed bets require a nonempty provider and timezone-aware UTC market
fetch timestamp.

Removed the stale CLV-enrichment descriptions and unused settlement CLV option.
Settlement continues to record results and PnL while preserving reference
provenance unchanged and leaving closing line, closing odds, and CLV unavailable
for a future validated closeout workflow.

#### Goal

Replace the development-era bet-ledger schema with a strict reference-offer
provenance contract that keeps actual wager terms separate from immutable market
evidence without introducing frontend submission, historical matching,
closeout, movement, CLV, backtest, or recommendation behavior.

#### Files Added/Removed/Changed

Added:
- None

Removed:
- None

Changed:
- PLAN.md
- src/gridiron_edge/betting/ledger.py
- src/gridiron_edge/cli/betting.py
- tests/unit/betting/test_ledger.py
- tests/unit/cli/test_betting.py

#### Tests

- Passed focused Ruff formatting and lint checks.
- Passed focused Pyrefly checks.
- Passed focused bet-ledger, betting-CLI, betting-performance, and portfolio
  serializer tests.
- Passed the full Ruff, Pyrefly, and non-slow unit-test quality gates.
- Validated the canonical ledger schema includes all reference-offer fields in
  deterministic order.
- Validated manual bets persist with every reference field null.
- Validated exact reference provider, provider event, sportsbook, market fetch
  time, sportsbook update time, kickoff, American odds, and line survive
  persistence.
- Validated reference-backed bets require a nonempty provider and timezone-aware
  UTC market fetch timestamp.
- Validated orphaned reference fields without a provider are rejected.
- Validated empty optional reference text values are rejected.
- Validated naive and non-UTC reference timestamps are rejected.
- Validated invalid, zero, and nonfinite reference American odds are rejected.
- Validated nonfinite reference lines are rejected.
- Validated actual wager sportsbook, odds, and line may differ from the reference
  offer without modifying either set of fields.
- Validated loading and filtering preserve reference provenance.
- Validated settlement preserves reference provenance unchanged.
- Validated malformed and stale persisted schemas are rejected before
  overwrite.
- Validated CLI-entered manual wagers continue to use null reference
  provenance.
- Validated the unused settlement CLV option is no longer registered.
- Validated stale CLV-enrichment descriptions are removed.
- Validated the new contract with a temporary persisted reference-backed wager
  whose actual FanDuel terms differed from its DraftKings reference offer.

#### Acceptance

The bet ledger stores actual wager terms independently from exact
reference-offer evidence. Reference-backed bets preserve provider,
provider-event, sportsbook, observation timestamps, kickoff, odds, and line,
while manual bets truthfully store null reference provenance. Invalid or
incomplete provenance is rejected, settlement preserves reference evidence
unchanged, and no API, frontend, historical matching, closeout, movement, CLV,
backtest, or recommendation behavior is introduced.

---

### Market Unit 13: Match Bets to Historical Reference Evidence [Completed]

#### Completed

Implemented a pure, immutable diagnostic contract that matches each recorded
bet's persisted reference-offer provenance to one exact canonical quote
observation.

Reference-backed bets are resolved using provider, provider event, sportsbook,
canonical game, market, side, and local market fetch timestamp. Nullable
provider-event and sportsbook identities use exact null-aware matching rather
than wildcard behavior.

After one exact observation is found, the matcher verifies sportsbook update
time, kickoff, American odds, and line. Missing observations, ambiguous
candidates, conflicting terms, manual bets, and successful matches remain
distinct explicit states. Actual wager terms do not participate in reference
matching and may differ from the immutable reference offer.

#### Goal

Verify that each recorded reference-backed wager identifies one exact canonical
historical quote observation without selecting closeout evidence or calculating
movement, CLV, backtest performance, qualification, or recommendation state.

#### Files Added/Removed/Changed

Added:
- src/gridiron_edge/market/bet_reference_matching.py
- tests/unit/market/test_bet_reference_matching.py

Removed:
- None

Changed:
- PLAN.md
- src/gridiron_edge/market/__init__.py

#### Tests

- Passed focused Ruff formatting and lint checks.
- Passed focused Pyrefly checks.
- Passed focused bet-reference matching, historical-boundary,
  history-coverage, and bet-ledger tests.
- Passed the full Ruff, Pyrefly, and non-slow unit-test quality gates.
- Validated empty bet input returns no match diagnostics.
- Validated manual bets return manual-bet status without attempting historical
  lookup.
- Validated exact reference-backed bets return matched status with an immutable
  selected quote observation.
- Validated provider, provider event, sportsbook, game, market, side, and local
  fetch timestamp all participate in exact identity.
- Validated nullable provider-event and sportsbook identities match only null
  values and are not treated as wildcards.
- Validated missing exact observations return observation-not-found status.
- Validated multiple exact candidates return ambiguous-observation status
  without arbitrary selection.
- Validated sportsbook update time, kickoff, American odds, and line conflicts
  are reported individually and together in deterministic field order.
- Validated null reference terms match only null quote terms.
- Validated actual wager sportsbook, odds, line, stake, and placement time do not
  participate in reference matching.
- Validated equivalent timezone-aware UTC timestamp representations match the
  same instant.
- Validated multiple results use deterministic bet-ID ordering independent of
  input order.
- Validated duplicate and empty bet IDs are rejected.
- Validated missing required bet or

---

### Market Unit 14: Scale Historical Quote Observation Storage [Completed]

#### Completed

Replaced the single ever-growing historical quote artifact with deterministic
season-and-week observation partitions.

Each append now updates only one bounded weekly partition while preserving exact
replay idempotence, later unchanged and changed observations, same-fetch
conflict detection, canonical ordering, complete validation, and atomic
replacement.

The public quote-storage boundary remains unchanged. Consumers continue using
`append_to_odds_ledger` and `load_odds_ledger` without depending on the physical
partition layout. Broad loading combines matching partitions without performing
cross-partition deduplication and orders results by season, week, and canonical
observation identity.

The operational current snapshot remains independently stored and replaced at
`data/odds/odds_current.parquet`.

#### Goal

Bound the physical cost of recurring quote acquisition by replacing the single
historical quote file with deterministic season-and-week partitions while
preserving all established observation evidence semantics.

#### Files Added/Removed/Changed

Added:
- None

Removed:
- None

Changed:
- PLAN.md
- src/gridiron_edge/ingest/odds/store.py
- tests/unit/cli/test_ingest_odds_cli.py
- tests/unit/ingest/odds/test_the_odds_api_ingest.py
- tests/unit/ingest/test_odds_store.py
- tests/unit/ingest/test_odds_store_source_neutral.py

#### Tests

- Passed focused Ruff formatting and lint checks.
- Passed focused Pyrefly checks.
- Passed focused quote-store, source-neutral storage, provider-client,
  provider-parser, provider-ingest, ingest-CLI, history-coverage,
  history-boundary, and bet-reference-matching tests.
- Passed the full Ruff, Pyrefly, and non-slow unit-test quality gates.
- Validated deterministic season-and-week partition paths.
- Validated one-scope append creates one weekly history partition.
- Validated mixed-season and mixed-week append input is rejected.
- Validated exact replay remains idempotent within one partition.
- Validated later unchanged, changed-price, and changed-line observations remain
  distinct.
- Validated same-fetch conflicts remain rejected.
- Validated invalid appends preserve the existing partition.
- Validated appending one week does not rewrite another week.
- Validated exact season-and-week loading reads one matching partition.
- Validated broader loading combines matching partitions without
  cross-partition deduplication.
- Validated broad results are ordered by season, week, and canonical observation
  order.
- Validated provider, sportsbook, market, season, and week filters remain
  supported.
- Validated missing history returns the canonical empty quote frame.
- Validated snapshot failure retains the newly persisted weekly history
  partition and preserves the prior current snapshot.
- Validated the current snapshot remains physically and behaviorally separate.
- Validated historical coverage, leakage-safe boundaries, and bet-reference
  matching continue consuming the public loader without physical-layout
  coupling.

#### Acceptance

Historical quote observations are stored in deterministic season-and-week
partitions, so one recurring collection rewrites only its bounded weekly scope.
Exact replay, later observations, provider identity, same-fetch conflict
detection, atomic replacement, canonical ordering, filtered loading, and broad
history loading remain truthful. The current snapshot remains separate, and no
scheduler, acquisition cadence, provider backfill, opening, closing, movement,
CLV, qualification, or recommendation behavior is introduced.

---

### Market Unit 15: Plan Weekly Quote Collection [Completed]

#### Completed

Implemented a scheduler-neutral weekly quote-collection planner that derives
explicit collection times from the canonical kickoff schedule for one selected
season and week.

The planner interprets nflverse game times in the `America/New_York` time zone,
converts exact schedule-derived kickoff instants to UTC, groups games by kickoff
window, and applies a configurable ramp guideline within a hard weekly poll
budget.

Added immutable policy, kickoff-group, planned-collection, and weekly-plan
contracts. Added versioned JSON persistence with semantic validation so
generated and deliberately edited plans use the same owning contract.

Added the explicit `gridiron ingest plan-odds` command. Plan generation requires
reproducible UTC `plan_start` and `created_at` inputs and performs no provider
request, quote persistence, current-snapshot replacement, or scheduler action.

#### Goal

Generate and validate deterministic, reviewable weekly collection plans from
actual NFL kickoff windows and a configurable provider budget without coupling
collection policy to deployment hardware or unattended execution.

#### Files Added/Removed/Changed

Added:
- src/gridiron_edge/market/collection_plan.py
- src/gridiron_edge/market/collection_plan_store.py
- tests/unit/cli/test_collection_plan_cli.py
- tests/unit/market/test_collection_plan.py
- tests/unit/market/test_collection_plan_store.py

Removed:
- None

Changed:
- PLAN.md
- src/gridiron_edge/cli/ingest.py
- src/gridiron_edge/market/__init__.py

#### Tests

- Passed focused Ruff formatting and lint checks.
- Passed focused Pyrefly checks.
- Passed focused collection-planning, plan-store, planning-CLI, odds-ingest,
  history-coverage, historical-boundary, and bet-reference-matching tests.
- Passed the full Ruff, Pyrefly, and non-slow unit-test quality gates.
- Validated explicit season, week, plan-start, and creation timestamps.
- Validated schedule kickoffs are interpreted in `America/New_York` and
  converted to UTC using a named time zone.
- Validated exact kickoff times form deterministic kickoff groups.
- Validated irregular weekday and multi-window schedules require no fixed
  Thursday, Sunday, or Monday assumptions.
- Validated baseline, approach, and near-kickoff candidates are generated from
  actual kickoff timestamps.
- Validated every planned collection precedes at least one remaining kickoff.
- Validated no collection is planned after all scoped games have started.
- Validated timestamps are unique and deterministically ordered.
- Validated plans never exceed their configured weekly poll limit.
- Validated projected credits equal planned polls multiplied by configured
  per-poll cost.
- Validated deterministic budget allocation and explicit omitted-candidate
  accounting.
- Validated unavailable schedule states remain explicit.
- Validated generated plans round-trip through versioned JSON unchanged.
- Validated manually edited plans use the same schema and semantic checks.
- Validated invalid, duplicate, post-kickoff, out-of-order, unknown-kickoff, and
  over-budget plans are rejected.
- Validated plan generation and validation do not mutate schedule inputs.
- Validated no provider client, quote store, current snapshot, or scheduler is
  invoked.
- Validated representative real 2026 schedule weeks through generated plan
  artifacts.

#### Acceptance

One explicit season-and-week schedule produces a deterministic, reviewable,
versioned collection plan based on actual UTC kickoff windows and a configurable
weekly provider budget. The plan explains every proposed collection, never
exceeds its poll or credit allowance, supports irregular NFL schedules, and can
be manually reviewed or edited under the same validation contract. No provider
request, quote persistence, scheduler deployment, Raspberry Pi dependency,
opening, closing, movement, CLV, qualification, or recommendation behavior is
introduced.

---

### Market Unit 16: Execute One Due Quote Collection [Complete]

#### Completed

Implemented a scheduler-neutral, single-shot execution boundary for one due
collection from a validated weekly quote-collection plan. Each invocation
evaluates at most one earliest unresolved poll at an explicit UTC timestamp and
preserves not-due, missed, previously claimed, completed, quota-blocked,
request-failed, ingest-failed, partially persisted, and successful outcomes as
explicit states.

Added deterministic, filename-safe receipt paths; atomic immutable execution
claims; semantically validated immutable terminal results; and loading of prior
results for due-state and last-known quota evaluation. Existing claims without
terminal results block automatic retry and require manual inspection.

Integrated execution with the established current-market ingest boundary so a
due poll makes no more than one provider request. Successful results preserve
quote, game, sportsbook, artifact-path, and provider-usage metadata. Added a
specific partial-persistence exception for successful historical append followed
by current-snapshot failure, allowing execution to record that outcome without
classifying failures from message text.

Added an explicit CLI command for executing one due collection with caller-
supplied season, week, evaluation time, grace period, quota reserve, timeout,
repository path, and API key inputs. The executor does not infer an active week,
mutate plans, install a scheduler, retry provider requests, or introduce
movement, CLV, qualification, or recommendation behavior.

#### Goal

Execute at most one due collection from one validated weekly quote-collection
plan, using atomic execution claims, explicit due-time evaluation, last-known
quota safeguards, immutable terminal results, and the existing provider-ingest
boundary. Preserve missed, claimed, blocked, failed, partially persisted, and
completed states without installing a scheduler or retrying provider requests
implicitly.

#### Files Added/Removed/Changed

Added:
- `src/gridiron_edge/market/collection_execution.py` - Added single-shot due-poll evaluation, quota prechecks, atomic claiming, provider execution, explicit terminal-state construction, and immutable result persistence.
- `src/gridiron_edge/market/collection_receipt_store.py` - Added versioned execution claim and terminal-result contracts, semantic validation, deterministic receipt paths, atomic writes, immutable result loading, and last-known quota lookup.
- `tests/unit/cli/test_collection_execution_cli.py` - Added CLI coverage for explicit collection execution inputs, output, and failure behavior.
- `tests/unit/market/test_collection_execution.py` - Added execution coverage for due-time boundaries, ordering, missed polls, prior claims and results, quota safeguards, provider outcomes, metadata preservation, and input immutability.
- `tests/unit/market/test_collection_receipt_store.py` - Added receipt-contract, path, atomicity, immutability, validation, loading, and quota-history coverage.

Changed:
- `PLAN.md` - Closed Market Unit 16 with its implemented scope, file inventory, tests, and acceptance result.
- `src/gridiron_edge/cli/ingest.py` - Added the explicit command boundary for executing one due quote collection.
- `src/gridiron_edge/ingest/odds/the_odds_api.py` - Added an explicit partial-persistence error when historical quote persistence succeeds but current-snapshot persistence fails.
- `src/gridiron_edge/market/__init__.py` - Exported the collection-execution and receipt-store interfaces.
- `tests/unit/ingest/odds/test_the_odds_api_ingest.py` - Added coverage for the explicit partial-persistence failure contract.

Removed:
- None.

#### Tests

Added unit coverage for unavailable plans, not-due evaluations, exact scheduled
times, grace-period behavior, the inclusive grace boundary, missed polls,
earliest-unresolved ordering, prevention of catch-up requests, completed polls,
existing claims without results, deterministic receipt paths, atomic claims,
immutable terminal results, semantic receipt validation, successful execution,
successful metadata preservation, unknown quota metadata, last-known quota
selection, quota reserve blocking, request failures, ingest failures, partial
persistence, secret-safe failure results, input immutability, and the explicit
CLI boundary.

Validated the complete Python project with:

- `uv run ruff check . --fix`
- `uvx pyrefly check src tests deploy/bin --search-path src`
- `uv run pytest -m "unit and not slow"`

All quality gates passed and all selected tests are green.

#### Acceptance

One invocation reads one validated weekly plan, evaluates at most one earliest
unresolved poll at an explicit UTC timestamp, atomically claims due work before
provider access, and invokes the established current-market ingest boundary no
more than once.

Missed polls are recorded without catch-up requests. Existing claims prevent
automatic retry. Completed polls are not executed again. Quota reserve blocking,
unknown quota state, request failure, ingest failure, partial persistence, and
successful completion remain distinct and durable. Successful results retain
collection counts, artifact paths, and provider quota metadata without
persisting API keys or unsafe provider content.

No automatic retry, catch-up polling, plan mutation, scheduler installation,
Raspberry Pi dependency, movement, CLV, qualification, or recommendation
behavior was introduced. Market Unit 16 is complete.

---

### Market Unit 17: Select the Active Quote Collection Plan [Complete]

#### Completed

Implemented an explicit, versioned current-selection boundary for one existing
validated weekly quote-collection plan. The global operational selection stores
only schema version, season, week, and an explicit UTC selection timestamp at
`data/odds/collection_plans/current.json`.

Selection validates the exact persisted plan before atomically replacing the
current-selection artifact. Selected-plan loading revalidates both the
selection and the referenced plan while deriving the plan path through the
existing deterministic season-and-week store. It does not duplicate the plan
path, search directories, inspect file recency, derive scope from the current
date, or generate or mutate a plan.

Added an explicit `select-odds-plan` command for selecting an existing reviewed
plan and an `execute-selected-odds-plan` command for resolving that selection.
The existing `execute-odds-plan` command remains available for exact-scope
manual and diagnostic execution.

Exact-plan and selected-plan execution now share one CLI orchestration boundary
and delegate all due-time, grace-period, quota, claim, receipt, provider, and
persistence behavior to the established single-shot executor. Selection and
selected-plan resolution do not access the provider.

#### Goal

Add an explicit, versioned current-selection boundary for one already persisted
and validated weekly quote-collection plan. Allow unattended operational tooling
to resolve the authorized season and week without deriving them from wall-clock
time, regenerating a plan, or modifying collection-policy and execution
semantics.

#### Files Added/Removed/Changed

Added:
- None.

Changed:
- `PLAN.md` - Closed Market Unit 17 with its implemented selection contract, file inventory, tests, and acceptance result.
- `src/gridiron_edge/cli/ingest.py` - Added explicit plan-selection and selected-plan execution commands and shared exact-plan and selected-plan CLI execution orchestration.
- `src/gridiron_edge/market/__init__.py` - Exported the current collection-plan selection, resolution, and path interfaces.
- `src/gridiron_edge/market/collection_plan_store.py` - Added the versioned global current-selection contract, deterministic selection path, atomic selection writer, strict reader, and selected-plan loader.
- `tests/unit/cli/test_collection_execution_cli.py` - Added coverage for resolving and executing the explicitly selected plan through the existing executor.
- `tests/unit/cli/test_collection_plan_cli.py` - Added coverage for explicit plan selection without provider access.
- `tests/unit/market/test_collection_plan_store.py` - Added selection-path, independence, persistence, loading, missing-state, schema, validation, and failed-replacement coverage.

Removed:
- None.

#### Tests

Added unit coverage for deterministic current-selection storage, independence
between plan persistence and selection, explicit selection of an existing
validated plan, selected-plan loading, exact selection schema, UTC selection
timestamps, missing referenced plans, missing current selection, unsupported
selection schema versions, malformed selection artifacts, and preservation of
the prior selection when a replacement attempt fails.

Added CLI coverage confirming that `select-odds-plan` selects an existing plan
without resolving provider credentials and that
`execute-selected-odds-plan` resolves the selected plan and invokes the
established single-shot executor exactly once. Existing exact-plan execution
remains covered and available.

Verified the public CLI surface:

- `gridiron ingest plan-odds`
- `gridiron ingest select-odds-plan`
- `gridiron ingest execute-odds-plan`
- `gridiron ingest execute-selected-odds-plan`

Validated the complete Python project with:

- `uv run ruff check . --fix`
- `uvx pyrefly check`
- `uv run pytest -m "unit and not slow"`

All quality gates passed and all selected tests are green.

#### Acceptance

One explicit operation selects one existing validated weekly quote-collection
plan through a versioned atomic global current-selection artifact. The artifact
contains only selection identity and selection time; the exact plan path remains
owned by the deterministic collection-plan store.

Selected-plan execution resolves and revalidates the selected plan before
delegating to the established single-shot executor. It does not infer season or
week, search for the newest plan, regenerate or mutate a plan, duplicate
execution logic, or access the provider during selection or resolution.

Missing, malformed, unsupported, missing-target, and scope-mismatched selection
states remain explicit. Exact-plan manual execution remains available. No
scheduler installation, automatic retry, catch-up polling, provider backfill,
movement, CLV, qualification, or recommendation behavior was introduced.
Market Unit 17 is complete.

---

### Market Unit 18: Codify the Quote Collection Worker Deployment [Completed]

#### Completed

2026-08-16

#### Goal

Make the validated Raspberry Pi quote-collection worker reproducible,
repository-owned, verifiable, transactional, and recoverable without changing
collection-plan, selection, due-time, quota, claim, receipt, provider-ingest,
historical interpretation, or recommendation semantics.

#### Files Added/Removed/Changed

Added:

- `deploy/bin/install_quote_collection_worker.py`
- `deploy/bin/verify_quote_collection_worker.py`
- `deploy/systemd/gridiron-edge-collector.service`
- `deploy/systemd/gridiron-edge-collector.timer`
- `src/gridiron_edge/deployment/__init__.py`
- `src/gridiron_edge/deployment/quote_collection_worker.py`
- `tests/unit/deployment/test_quote_collection_worker.py`

Changed:

- `CHANGELOG.md`
- `DECISIONS.md`
- `HANDOFF.md`
- `PLAN.md`
- `ROADMAP.md`

Removed:

- None.

#### Tests

- `uv run ruff check src/gridiron_edge/deployment tests/unit/deployment --fix`
  passed.
- `uvx pyrefly check src/gridiron_edge/deployment tests/unit/deployment`
  passed with zero errors.
- `uv run pytest tests/unit/deployment/test_quote_collection_worker.py`
  passed with 13 tests.
- Repository Ruff and the selected repository test suite passed.
- The corrected repository-wide `uvx pyrefly check` command was executed and
  reported 524 existing errors. Restoring that boundary is recorded separately
  in `ROADMAP.md`.
- Repository-owned installation completed on the target Raspberry Pi.
- `systemd-analyze verify` passed with exit status zero.
- Repository-owned verification reported `ready`.
- The selected 2026 Week 1 plan resolved with 34 planned polls and zero
  unresolved claims.
- The timer remained enabled and active with a five-minute cadence.
- Managed execution returned `not_due`, exited successfully, and created no
  execution or quote artifacts.
- The worker reported `throttled=0x0`, 46.2 C, root storage on the 2 TB SSD,
  and no configured current transport or storage error markers.

#### Acceptance

The quote-collection worker deployment is repository-owned, reproducible,
transactional, independently verifiable, and validated on the target Raspberry
Pi. The installed non-root oneshot service resolves the explicitly selected
weekly plan, generates a current UTC evaluation timestamp, loads the provider
credential from a protected root-owned environment file, and is triggered by
an enabled five-minute systemd timer.

Installation validates the complete staged deployment before replacement,
restores prior files and modes if systemd reload fails, and keeps explicit timer
activation separate. Read-only verification does not open the credential file.

Managed execution before the first planned poll returns `not_due`, exits
successfully, and creates no execution or quote artifacts. No collection
policy, due-time, quota, claim, receipt, provider-ingest, historical
interpretation, API, frontend, model, qualification, or recommendation behavior
was introduced.

---

### Market Unit 19: Restore the Repository-Wide Pyrefly Boundary [Completed]

#### Completed

2026-08-16

#### Goal

Restore the explicit repository-owned Pyrefly boundary as a truthful,
documented, and enforced zero-error repository quality gate without hiding
genuine production defects behind a blanket suppression baseline.

The canonical owned boundary is:

`uvx pyrefly check src tests deploy/bin --search-path src --search-path .`

Exploratory notebooks remain outside Ruff, Pyrefly, and automated test gates
because they are non-authoritative testing-ground artifacts.

#### Files Added/Removed/Changed

Added:

- None.

Changed:

- `PLAN.md`
- `pyproject.toml`
- `pyrefly.toml`
- `src/gridiron_edge/cli/evaluate.py`
- `src/gridiron_edge/features/team/elo.py`
- `src/gridiron_edge/models/base.py`
- `src/gridiron_edge/models/game_prediction/model.py`
- `tests/fixtures/repos.py`
- `tests/integration/api/test_edges_routes.py`
- `tests/integration/api/test_teams_routes.py`
- `tests/integration/test_dataset_roundtrip.py`
- `tests/integration/test_odds_join.py`
- `tests/unit/api/test_edges_route_diagnostics.py`
- `tests/unit/api/test_loader_player_history.py`
- `tests/unit/api/test_loader_players_list.py`
- `tests/unit/api/test_loaders.py`
- `tests/unit/api/test_schema_base.py`
- `tests/unit/api/test_schemas_comparables.py`
- `tests/unit/api/test_schemas_edges.py`
- `tests/unit/api/test_schemas_explain.py`
- `tests/unit/api/test_schemas_games.py`
- `tests/unit/api/test_schemas_injuries.py`
- `tests/unit/api/test_schemas_lines.py`
- `tests/unit/api/test_schemas_live.py`
- `tests/unit/api/test_schemas_model_performance.py`
- `tests/unit/api/test_schemas_news.py`
- `tests/unit/api/test_schemas_portfolio.py`
- `tests/unit/api/test_schemas_projections.py`
- `tests/unit/api/test_schemas_prop_reasoning.py`
- `tests/unit/api/test_schemas_prop_shop.py`
- `tests/unit/api/test_schemas_props.py`
- `tests/unit/api/test_schemas_swing_factors.py`
- `tests/unit/api/test_schemas_teams.py`
- `tests/unit/api/test_schemas_weeks.py`
- `tests/unit/api/test_serializers_compare.py`
- `tests/unit/api/test_serializers_model_performance.py`
- `tests/unit/api/test_serializers_projections.py`
- `tests/unit/betting/test_bankroll.py`
- `tests/unit/evaluation/test_archive.py`
- `tests/unit/evaluation/test_diagnostics.py`
- `tests/unit/evaluation/test_forecast_events.py`
- `tests/unit/evaluation/test_forecast_selection.py`
- `tests/unit/evaluation/test_forecast_store.py`
- `tests/unit/evaluation/test_manifest.py`
- `tests/unit/evaluation/test_percentiles.py`
- `tests/unit/evaluation/test_situational_splits.py`
- `tests/unit/features/test_canonical_feature_sequence.py`
- `tests/unit/features/test_home_away_elo_feature.py`
- `tests/unit/features/test_home_away_game_features.py`
- `tests/unit/features/test_player_matchup.py`
- `tests/unit/features/test_usage.py`
- `tests/unit/features/test_weather.py`
- `tests/unit/fixtures/test_game_modeling_dataframes.py`
- `tests/unit/ingest/odds/test_the_odds_api_parser.py`
- `tests/unit/market/test_recommendations.py`
- `tests/unit/metadata/test_stadium_sync.py`
- `tests/unit/models/game_prediction/test_weekly_game_product.py`
- `tests/unit/models/game_prediction/test_weekly_product_store.py`
- `tests/unit/models/game_prediction/test_weekly_spread_product.py`
- `tests/unit/models/test_artifact.py`
- `tests/unit/models/test_base.py`
- `tests/unit/models/test_epa_window.py`
- `tests/unit/models/test_games_model.py`
- `tests/unit/models/test_games_trainer.py`
- `tests/unit/models/test_metadata.py`
- `tests/unit/models/test_model_registry.py`
- `tests/unit/models/test_post_process.py`
- `tests/unit/models/test_prop_base.py`
- `tests/unit/ratings/test_elo_predict.py`
- `tests/unit/ratings/test_simulator.py`
- `tests/unit/sim/test_season.py`
- `tests/unit/transform/clean/test_schedule_nflverse.py`
- `tests/unit/viz/test_predictions.py`
- `tests/e2e/test_cli_workflows.py`

Removed:

- None.

#### Tests

- The initial explicit repository-owned Pyrefly boundary reported 431 errors
  across `src/`, `tests/`, and `deploy/bin/`.
- Production-source diagnostics were corrected before test-only diagnostics.
- Shared fixtures, API schemas, loader tests, serializer tests, model tests,
  feature tests, integration tests, and end-to-end CLI tests passed after their
  respective bounded corrections.
- Deliberately invalid Pydantic construction remained covered through dynamic
  runtime validation boundaries.
- Frozen-model mutation tests remained covered through narrowly documented
  dynamic mutation calls.
- Pandas slices, filters, datetime accessors, heterogeneous payloads, optional
  values, dataset identities, and protocol test doubles received explicit
  static contracts without changing their runtime behavior.
- `HomeAwayEloFeature` now declares the narrower Elo dataset dependency it
  actually consumes.
- Runtime-checkable model protocols and `GamesModel.spec` now agree on their
  structural contract.
- `uv run ruff check . --fix` passed.
- `uvx pyrefly check`
  passed with zero errors.
- `uv run pytest -m "unit and not slow"` passed.
- No temporary repository-update scripts remain at the repository root.

#### Acceptance

The explicit repository-owned Pyrefly boundary reports zero errors across
`src/`, `tests/`, and `deploy/bin/`. The enforced backlog was reduced from 431
errors to zero without introducing a blanket suppression baseline or excluding
maintained production, test, fixture, integration, end-to-end, or deployment
code.

Production source, tests, shared fixtures, and deployment administrative tools
are type-correct within the owned boundary. Repository and test import roots
resolve correctly. Exploratory notebooks remain explicitly outside automated
quality gates because they are non-authoritative testing-ground artifacts.

Deliberate negative tests retain their runtime validation behavior. Frozen-model
tests continue to verify immutability. Pandas typing boundaries are explicit.
Concrete production types were not weakened solely to accommodate loosely typed
test doubles. Feature dataset dependencies and runtime-checkable model protocols
now reflect the capabilities their implementations actually consume.

Ruff, the complete repository-wide Pyrefly gate, and the non-slow unit test
suite pass. No temporary update scripts remain in the repository.

No market candidate, closeout, CLV, recommendation-policy, API, frontend,
model-output, or operational worker behavior was introduced.

---

### Market Unit 20: Persist Immutable Pregame Candidate Issuance [Completed]

#### Completed

2026-08-16

#### Goal

Persist the exact candidate evidence evaluated before kickoff so later
qualification, recommendation, and outcome evaluation consume an immutable
historical decision artifact rather than reconstructing candidate state from
newer forecasts, products, market quotes, or policy.

The issuance boundary preserves the selected weekly product, referenced
forecast events, exact sportsbook quote, model probability, calculated expected
value, evaluation timestamp, and explicit issuance state.

#### Files Added/Removed/Changed

Added:

- `src/gridiron_edge/market/candidate_issuance.py`
- `src/gridiron_edge/market/candidate_issuance_store.py`
- `tests/unit/market/test_candidate_issuance.py`
- `tests/unit/market/test_candidate_issuance_evaluation.py`
- `tests/unit/market/test_candidate_issuance_store.py`

Changed:

- `PLAN.md`
- `src/gridiron_edge/market/__init__.py`

Removed:

- None.

#### Tests

- Deterministic SHA-256 issuance identity was verified for identical product,
  weekly scope, and evaluation context.
- Changing the explicit evaluation timestamp was verified to produce a distinct
  issuance identity.
- Non-UTC evaluation timestamps were rejected.
- Complete quote evidence was preserved, including provider, provider-event,
  sportsbook, canonical game, market, side, line, American price, fetch
  timestamp, sportsbook update timestamp, kickoff, and live state.
- Selected weekly-product identity, product-run identity, product generation
  timestamp, exact referenced forecast event, forecast run, role, generation
  timestamp, model name, and model type were preserved.
- Model probability, calculated expected value, explicit evaluation timestamp,
  state, and reason were preserved.
- Strictly positive expected value produced `candidate` with
  `positive_expected_value`.
- Negative and break-even expected value produced `not_candidate` with
  `expected_value_not_positive`.
- Missing kickoff, live quote evidence, quotes fetched at or after kickoff,
  missing selected forecasts, unavailable model evidence, and unavailable
  uncertainty remained explicit `unavailable` results.
- Issuance exactly at kickoff and after kickoff was rejected.
- Moneyline and spread issuance referenced the selected Win forecast event.
  Total issuance referenced the selected Total forecast event.
- Product and quote weekly-scope mismatch was rejected.
- Duplicate quote-observation identities were rejected.
- Reordered quote input produced identical deterministic issuance output.
- Product, forecast-event, and quote inputs were not mutated.
- Immutable JSON persistence round-tripped without evidence loss.
- Exact replay was idempotent and did not rewrite the artifact.
- Conflicting replay under the same deterministic issuance identity was
  rejected.
- Unsupported schema versions, malformed artifact keys, unexpected row keys,
  embedded identity conflicts, filename conflicts, nondeterministic row order,
  duplicate stored row identities, and unsafe issuance IDs were rejected.
- Focused Ruff checks passed.
- Focused Pyrefly checks passed with zero errors.
- All 29 candidate-issuance tests passed.
- `uv run ruff check . --fix` passed.
- `uvx pyrefly check` passed with zero errors.
- `uv run pytest -m "unit and not slow"` passed.
- No temporary Unit 20 update helpers remain in the repository.

#### Acceptance

A caller can evaluate every supplied exact sportsbook quote against one
explicitly selected immutable weekly product and its exact referenced forecast
events before kickoff. The result records each observation as `candidate`,
`not_candidate`, or `unavailable` with a deterministic evidence-only reason.

Each issuance row preserves provider, provider-event, sportsbook, canonical
game, market, side, line, American price, quote fetch timestamp, sportsbook
update timestamp, kickoff, live state, exact forecast-event identity, forecast
run, forecast role, forecast generation timestamp, model identity, model
probability, and calculated expected value. The issuance artifact also
preserves product identity, product-run identity, product generation timestamp,
weekly scope, and the explicit evaluation timestamp.

The deterministic issuance identity is derived from the schema version,
selected product identity, product-run identity, season, week, and evaluation
timestamp. Result rows and calculated values remain immutable artifact content,
so identical replay is idempotent while different content under the same
issuance identity is rejected.

Issuance at or after any known kickoff is rejected. Live observations and
observations fetched at or after kickoff cannot become candidates. The existing
quote-history definition of pregame eligibility remains authoritative:
`is_live` is false and `fetched_at` is strictly before kickoff.

Artifacts are persisted as deterministic JSON under the candidate-issuance
output root using exclusive creation without replacement. Existing artifacts,
embedded identities, serialized rows, and deterministic ordering are validated
when read.

Later qualification and evaluation work can consume the persisted artifact
without reopening current weekly-product selection, resolving newer forecast
events, querying newer quotes, recalculating model probability, or recalculating
expected value.

No candidate qualification, recommendation policy, staking, bankroll,
exposure, portfolio, settlement, closeout, CLV, CLI, API, frontend, or
operational-worker behavior was introduced.

---

### Market Unit 21: Implement Validated Market Closeout and CLV [Completed]

#### Completed

2026-08-16

#### Goal

Close immutable candidate issuance and recorded-wager reference evidence against
the correct latest eligible pregame quote observation without reconstructing
source identity, weakening exact-offer provenance, or treating storage order as
an implicit closing definition.

The closeout boundary requires exact provider-aware market identity, preserves
explicit unavailable and conflict states, and calculates CLV only when complete
validated closing-price or closing-line evidence exists.

#### Files Added/Removed/Changed

Added:

- `src/gridiron_edge/market/market_closeout.py`
- `tests/unit/market/test_market_closeout.py`

Changed:

- `PLAN.md`
- `src/gridiron_edge/market/__init__.py`

Removed:

- None.

#### Tests

- Exact provider, provider-event, sportsbook, canonical game, market, and side
  matching was verified.
- Provider, provider-event, sportsbook, game, market, and side mismatches were
  rejected as missing exact closeout identity.
- Reference and selected closeout line identities were preserved independently
  so valid point movement could be calculated.
- The maximum eligible fetch timestamp was selected independently of input and
  storage order.
- Only non-live observations fetched strictly before kickoff were eligible.
- Live observations, observations at kickoff, and post-kickoff observations
  could not displace valid pregame evidence.
- Missing kickoff and conflicting kickoff evidence remained explicit.
- Live-only, post-kickoff-only, and mixed unavailable history remained explicit.
- Duplicate observations at the maximum eligible fetch remained ambiguous.
- Conflicting observations at the maximum eligible fetch remained explicit
  conflicts.
- Missing exact closeout history remained explicit.
- Moneyline price CLV was calculated only when valid reference and closeout
  American prices existed.
- Moneyline price CLV used raw American-price implied probabilities and remained
  distinct from no-vig fair-probability analysis.
- Spread point CLV preserved the established Home and Away side orientation.
- Total point CLV preserved the established Over and Under side orientation.
- Missing reference price, closeout price, reference line, and closeout line
  produced explicit unavailable states with null CLV.
- Unavailable CLV was never represented as zero.
- Candidate issuance rows were closed directly from their immutable reference
  evidence without reopening current products, resolving forecast events, or
  recalculating original model probability or expected value.
- Candidate rows with invalid or live reference evidence could not produce CLV.
- Recorded wagers were passed through exact immutable reference matching before
  closeout.
- Manual, missing, ambiguous, and conflicting recorded-wager references mapped
  to explicit closeout states.
- Recorded-wager closeout preserved the original provider, provider-event,
  sportsbook, game, market, side, fetch time, sportsbook update time, kickoff,
  price, and line evidence.
- Candidate and recorded-wager results were deterministically ordered.
- Candidate issuance, wager, and quote inputs were not mutated.
- Result and reference contracts were immutable.
- Source inspection verified the closeout implementation does not use
  `iloc[-1]`, `tail(1)`, first-stored, last-stored, or opening-line semantics.
- Existing CLV and historical-boundary tests remained green.
- Focused Ruff checks passed.
- The configured repository-wide Pyrefly check passed with zero errors.
- Focused market closeout, CLV, historical-boundary, and reference-matching
  tests passed.
- `uv run ruff check . --fix` passed.
- `uvx pyrefly check` passed with zero errors.
- `uv run pytest -m "unit and not slow"` passed.
- No temporary Unit 21 update helpers remain in the repository.

#### Acceptance

A caller can close one immutable candidate issuance or recorded-wager reference
against canonical quote history using exact provider, provider-event,
sportsbook, game, market, and side identity.

Recorded wagers must first match their immutable reference observation,
including fetch timestamp, sportsbook update timestamp, kickoff, American
price, and line. Manual, missing, ambiguous, or conflicting references cannot
proceed to CLV calculation.

Closeout selects the observation at the maximum eligible fetch timestamp only
after requiring one unambiguous kickoff and filtering to non-live observations
strictly before kickoff. Input order and raw storage order cannot change the
result. The first or last stored observation is never interpreted as an opening
or closing definition.

Moneyline closeout calculates raw implied-probability price CLV only when valid
reference and closeout American prices exist. Spread and Total closeout
calculate side-oriented point CLV only when valid reference and closeout lines
exist. Reference and selected closeout terms remain preserved independently.
Missing calculations remain null with explicit unavailable states.

Candidate issuance is consumed directly as immutable historical evidence.
Recorded wagers reuse exact reference matching before closeout. Neither path
mutates the betting ledger, settlement fields, candidate issuance, quote
history, or input DataFrames.

No candidate qualification, recommendation policy, staking, bankroll,
exposure, portfolio, outcome grading, ROI policy, ledger mutation, closeout
persistence, CLI, API, frontend, or operational-worker behavior was introduced.

---

### Market Unit 22: Build Empirical Market-Family Evaluation [Completed]

#### Completed

2026-08-17

#### Goal

Evaluate immutable issued candidates independently for Moneyline, Spread, and
Total using matured outcome evidence, validated strictly pre-kickoff closeout
evidence, and exact settled-wager matches.

The evaluation boundary reports empirical candidate behavior, market-family
coverage, closeout and CLV evidence, and realized return without introducing
attribution, qualification thresholds, recommendation policy, staking policy,
or ledger mutation.

Realized-return evidence remains observational. It is derived only from exact
matches to settled wagers and preserves explicit available, unavailable, and
conflict states so missing evidence cannot be interpreted as zero return.

#### Files Added/Removed/Changed

Added:

- Market-family evaluation contracts, implementation, and test coverage.

Changed:

- `PLAN.md`
- Market-family report construction and realized-return evidence integration.
- Market-family evaluation unit, integration, and architecture tests.

Removed:

- `integrate_market_family_return_evidence_v2.py`
- Temporary Market Unit 22 implementation and repair helpers.

#### Tests

- Moneyline, Spread, and Total candidates were evaluated independently.
- Immutable candidate issuance was consumed directly without reopening current
  market products or recalculating original candidate evidence.
- Only candidates with matured outcome evidence contributed to empirical
  outcome cohorts.
- Candidates without matured outcomes remained explicitly outside empirical
  outcome calculations.
- Market-family reports preserved independent coverage for Moneyline, Spread,
  and Total.
- Validated closeout and CLV evidence was reported without weakening exact
  provider-aware market identity.
- Missing closeout evidence remained explicitly unavailable.
- Conflicting closeout evidence remained explicitly conflicting.
- Unavailable or conflicting closeout evidence did not contribute fabricated
  CLV values.
- Realized-return evidence was derived only from exact settled-wager matches.
- Family-specific realized-return coverage distinguished available,
  unavailable, and conflict states.
- Settled-wager matches preserved exact wager and immutable candidate identity.
- Empirical cohort mean return was populated from exact settled-wager matches.
- Empirical cohort aggregate return was populated from exact settled-wager
  matches.
- Cohorts without settled-wager evidence retained null mean and aggregate return
  metrics.
- Missing realized-return evidence was not represented as zero return.
- Conflicting realized-return evidence did not contribute to empirical return
  metrics.
- Market-family return evidence remained separate from outcome, closeout, and
  CLV coverage.
- Market families without settled-wager evidence remained reportable.
- Report construction remained deterministic.
- Evaluation inputs were not mutated.
- No outcome, closeout, CLV, or realized-return evidence was attributed to a
  particular model, feature, decision rule, or issuance cause.
- No empirical result was converted into a candidate qualification threshold.
- No empirical result was converted into recommendation policy.
- No empirical result changed wager eligibility, staking, bankroll, exposure,
  portfolio, or ledger behavior.
- Existing market issuance, exact-reference matching, closeout, CLV, wager
  settlement, and reporting tests remained green.
- AST validation passed.
- Syntax validation passed.
- `uv run ruff check . --fix` passed.
- `uvx pyrefly check` passed with zero errors.
- `uv run pytest -m "unit and not slow"` passed.
- The full test suite passed.
- No temporary Market Unit 22 helper files remain in the repository.

#### Acceptance

A caller can evaluate immutable issued candidates independently for Moneyline,
Spread, and Total without reopening current products, reconstructing source
identity, or recalculating the evidence that existed when each candidate was
issued.

Only candidates with matured outcome evidence contribute to empirical outcome
cohorts. Candidates without matured outcomes remain explicitly identifiable and
cannot silently affect empirical rates or aggregates.

Closeout and CLV evidence is consumed through the validated market closeout
boundary. Exact provider, provider-event, sportsbook, canonical game, market,
and side identity remains required. Unavailable and conflicting closeout
evidence remains explicit and cannot be interpreted as valid or zero-valued CLV.

Realized return is derived only from exact matches between immutable issued
candidates and settled wager evidence. Market-family reports distinguish
available, unavailable, and conflicting return coverage independently for
Moneyline, Spread, and Total.

When exact settled-wager evidence exists, empirical cohorts report mean return
and aggregate return from those exact matches. When no settled-wager evidence
exists, return metrics remain null. Missing, unavailable, or conflicting
evidence is never converted to zero return.

Outcome, closeout, CLV, and realized-return coverage remain distinct so the
presence of one evidence type does not imply the presence or validity of
another.

The evaluation is descriptive only. It does not attribute observed performance
to a model, feature, decision rule, or issuance cause. It does not establish
minimum sample sizes, evidence thresholds, candidate qualification rules,
recommendation policy, or approval criteria.

No candidate issuance, quote history, recorded wager, settlement evidence, or
input DataFrame is mutated.

No staking, bankroll, exposure, portfolio construction, ledger mutation,
candidate qualification, recommendation emission, recommendation persistence,
CLI, API, frontend, or operational-worker behavior was introduced.

---

### Market Unit 23: Lock Versioned Recommendation Policy [Completed]

#### Completed

2026-08-17

#### Goal

Derive, validate, and persist an immutable versioned policy that determines
whether one exact issued candidate may remain unqualified, become a qualified
opportunity, or become eligible for recommendation.

Policy derivation evaluates Moneyline, Spread, and Total independently from the
empirical evidence produced by Market Unit 22. Descriptive evidence does not
become a qualification threshold without a validated threshold-selection
method, and insufficient evidence remains an explicit valid policy result.

The policy boundary separates empirically derived qualification evidence from
governed operational inputs, evaluates exact candidate, freshness, bankroll,
sizing, duplicate, conflict, and exposure requirements deterministically, and
remains independent from API, frontend, CLI, ledger mutation, bankroll mutation,
and operational request paths.

#### Files Added/Removed/Changed

Added:

- `src/gridiron_edge/market/recommendation_policy.py`
- `src/gridiron_edge/market/recommendation_policy_store.py`
- `tests/unit/market/test_recommendation_policy.py`
- `tests/unit/market/test_recommendation_policy_evaluation.py`
- `tests/unit/market/test_recommendation_policy_store.py`

Changed:

- `PLAN.md`
- `src/gridiron_edge/market/candidate_issuance.py`
- `src/gridiron_edge/market/market_closeout.py`
- `tests/unit/market/test_candidate_issuance.py`
- `tests/unit/market/test_market_closeout.py`

Removed:

- Temporary Market Unit 23 implementation and correction helpers.

#### Tests

- One stable exact candidate-row identity was shared by candidate issuance,
  market closeout, and recommendation-policy evaluation.
- Existing closeout candidate-reference identities remained unchanged.
- Complete Market Unit 22 evidence received one canonical deterministic
  SHA-256 fingerprint.
- Governed operational inputs received a separate canonical deterministic
  SHA-256 fingerprint.
- Policy identity included schema, evidence, derivation, family-policy, and
  governance content while excluding observational `created_at` metadata.
- Repeated derivation from identical evidence and governance produced the same
  policy identity.
- Governance changes produced a different policy identity.
- Moneyline, Spread, and Total policies were derived independently.
- Descriptive quantile cohort boundaries were not promoted into qualification
  thresholds.
- Families with descriptive evidence but no validated threshold-selection
  method remained explicitly insufficient.
- Governed Kelly, stake, rounding, duplicate, conflict, and exposure values
  remained distinct from empirical evidence provenance.
- Invalid fractions, stake increments, exposure ordering, status ordering, and
  provenance were rejected.
- Exact immutable candidate references resolved within their parent issuance.
- Unknown or altered candidate references were rejected.
- Historical non-candidate and unavailable issuance states could not be
  promoted by recommendation policy.
- Explicit UTC decision time controlled recommendation quote freshness.
- Issuance quote age and decision quote age remained distinct evidence.
- Inactive family policies stopped before Kelly sizing and stake-dependent
  exposure evaluation.
- Mandatory unavailable or conflicting checks could not silently pass.
- Missing bankroll, portfolio, or mandatory correlation evidence prevented
  recommendation eligibility.
- Exact duplicate wagers and same-game opposing positions were detected.
- Portfolio snapshots rejected duplicate bet identities, future rows, invalid
  stakes, unsupported markets, and non-UTC timestamps.
- Full Kelly was calculated by the existing `kelly_fraction()` owner.
- Fractional-Kelly multiplication used the explicit persisted governance value.
- Raw, constrained, rounded, and actionable stake values remained distinct.
- Candidate, per-game, total portfolio, and explicit correlation capacities
  constrained proposed stakes deterministically.
- Minimum actionable stake and persisted rounding behavior were enforced.
- Policy checks were returned in stable deterministic order.
- Repeated evaluation produced identical immutable decisions.
- Candidate issuance, policy, bankroll, portfolio, and correlation inputs were
  not mutated.
- Recommendation policies round-tripped through strict versioned JSON.
- Policy paths were addressed by schema version and deterministic policy ID.
- Exact immutable artifact replay was idempotent.
- Conflicting content under an existing policy identity was rejected.
- Malformed top-level and nested keys were rejected.
- Unsupported schema versions, invalid fingerprints, mismatched policy IDs,
  unsafe paths, and filename-to-identity mismatches were rejected.
- No implicit current-policy selection artifact was introduced.
- Source inspection verified policy modules do not depend on API, CLI, mutable
  bankroll storage, or the betting ledger.
- Focused Ruff checks passed.
- The configured repository-wide Pyrefly check passed with zero errors.
- Focused candidate identity, policy derivation, evaluation, and persistence
  tests passed.
- `uv run ruff check . --fix` passed.
- `uvx pyrefly check` passed with zero errors.
- `uv run pytest -m "unit and not slow"` passed.
- No temporary Market Unit 23 helper files remain in the repository.

#### Acceptance

A caller can derive one immutable recommendation-policy artifact from one exact
Market Unit 22 evaluation report and explicit governed operational inputs.
Moneyline, Spread, and Total are evaluated independently, and evidence or
thresholds are never transferred between market families.

The complete Unit 22 report receives a canonical source-evidence fingerprint.
Governed operational inputs receive a separate fingerprint. Policy identity is
deterministic from schema, evidence, derivation method, independent family
policies, and governance content. Observational creation time does not alter
policy identity.

Current Unit 22 marginal empirical cohorts do not establish a validated joint
threshold-selection method. Market families with otherwise available
descriptive evidence therefore remain explicitly insufficient and contain no
invented expected-value, quote-age, observation-depth, CLV, or realized-return
thresholds.

Operational fractional-Kelly, stake increment, rounding, minimum actionable
stake, duplicate, opposing-position, bankroll, candidate exposure, per-game
exposure, portfolio exposure, and correlation requirements are explicit
versioned governed inputs. They are not represented as empirical Unit 22
findings.

One policy evaluation consumes an exact candidate row within its immutable
parent issuance, an explicit UTC decision time, and supplied immutable
bankroll, portfolio, and correlation evidence. It never reopens current market
products, reconstructs source identity, or recalculates the original issuance
evidence.

Recommendation freshness uses decision time minus the exact quote fetch time.
The historical issuance quote age remains preserved separately. Missing
kickoff, live evidence, at-kickoff or post-kickoff evidence, future timestamps,
and policy-age violations cannot pass freshness checks.

Mandatory checks have explicit passed, failed, unavailable, conflicting, or
not-applicable states. Failed mandatory checks prohibit promotion. Unavailable
or conflicting mandatory checks preserve insufficient evidence and are never
treated as passed.

Sizing reuses the existing full-Kelly arithmetic, applies the persisted
fractional multiplier, constrains the proposed amount by candidate, game,
portfolio, and explicit correlation capacities, applies persisted rounding,
and requires the persisted minimum actionable stake. Raw, constrained, rounded,
and actionable values remain independently auditable.

Exact duplicate exposure uses the shared immutable candidate reference.
Opposing positions use canonical game, market, and side evidence. Correlation is
never inferred from labels and requires separately supplied evidence when the
check is mandatory.

Policies persist under schema-versioned, identity-addressed immutable JSON
paths. Reads validate exact top-level and nested schemas, timestamps, enums,
provenance, fingerprints, canonical policy identity, and filename agreement.
Exact replay is idempotent, while conflicting content under one policy identity
is rejected.

Policy derivation and evaluation are deterministic and do not mutate candidate
issuance, policy artifacts, quote evidence, bankroll evidence, portfolio
evidence, the betting ledger, or input objects.

No production family policy was activated from insufficient evidence. Synthetic
active policies exist only in tests of evaluation mechanics and are not
represented as empirically derived policies.

No recommendation was emitted, persisted, published, displayed, transmitted,
or placed. No current-policy selector, policy activation command, staking
execution, sportsbook integration, automatic wagering, ledger mutation,
bankroll mutation, API route, frontend component, CLI command, notification, or
operational-worker behavior was introduced.

---

### Market Unit 24: Persist Recommended-Bet Results [Completed]

#### Completed

2026-08-17

#### Goal

Persist one immutable qualification and recommendation result for every exact
offer that enters recommendation-policy evaluation while preserving the full
candidate, selected-product, forecast, policy, check, sizing, bankroll,
portfolio, and correlation evidence used by the decision.

The result boundary distinguishes qualified, recommended, failed, unavailable,
and conflicting outcomes without rerunning qualification during persistence or
placing a sportsbook wager.

#### Files Added/Removed/Changed

Added:

- `src/gridiron_edge/market/recommended_bet_result.py`
- `src/gridiron_edge/market/recommended_bet_result_store.py`
- `tests/fixtures/recommended_bet_results.py`
- `tests/unit/market/test_recommended_bet_result.py`
- `tests/unit/market/test_recommended_bet_result_store.py`

Changed:

- `PLAN.md`
- `src/gridiron_edge/market/recommendation_policy.py`
- `tests/unit/market/test_recommendation_policy_evaluation.py`

Removed:

- Temporary Market Unit 24 implementation and correction helpers.

#### Tests

- Qualified opportunities remained distinct from recommendation-eligible
  results when empirical qualification passed but no actionable stake existed.
- Recommendation eligibility continued to require all mandatory checks and an
  actionable stake.
- One immutable result was produced for every historically issued candidate in
  canonical issuance order.
- Historical not-candidate and unavailable issuance rows were not duplicated
  into recommendation-result artifacts.
- Exact candidate, offer, selected-product, forecast-event, model, and policy
  provenance was preserved.
- Policy schema, evidence fingerprint, governance fingerprint, and derivation
  method were preserved.
- Decision time, issuance quote age, and decision quote age were preserved
  independently.
- Every ordered policy check and its mandatory classification, state, reason,
  observed value, and required value was preserved.
- Recommended, qualified, failed, unavailable, and conflicting persisted states
  mapped deterministically from the original policy decision.
- Full-Kelly, fractional-Kelly, raw, constrained, rounded, and actionable stake
  evidence remained distinct.
- Bankroll basis, portfolio snapshot identity, and correlation evidence were
  preserved when available.
- Inactive production-compatible policies produced immutable unavailable
  results with null actionable stake.
- Synthetic active policies remained limited to evaluation and persistence
  mechanics tests.
- Result identity was deterministic and changed when decision evidence changed.
- Evaluation identity preserved issuance, policy, decision time, and ordered
  result identities.
- Policy-to-decision provenance and quote-age mismatches were rejected.
- Multiple matching correlation groups were rejected as ambiguous.
- Individual results round-tripped through strict versioned JSON.
- Evaluation manifests round-tripped and resolved every referenced result by
  immutable identity.
- Exact result and evaluation replay was idempotent.
- Conflicting content under an existing result or evaluation identity was
  rejected.
- Malformed nested contracts, unsupported schema versions, unsafe identities,
  missing referenced results, duplicate manifest result IDs, identity changes,
  filename mismatches, and evaluation provenance disagreement were rejected.
- Store reads strictly reconstructed enum, timestamp, scalar, tuple, optional,
  and nested immutable evidence.
- Result, evaluation, policy, issuance, bankroll, portfolio, and correlation
  inputs were not mutated.
- Source inspection verified no API, CLI, mutable bankroll, betting-ledger, or
  implicit current-result selection dependency was introduced.
- Focused Ruff checks passed.
- The configured repository-wide Pyrefly check passed with zero errors.
- Focused recommendation-policy, recommended-bet result, and persistence tests
  passed.
- `uv run ruff check . --fix` passed.
- `uvx pyrefly check` passed with zero errors.
- `uv run pytest -m "unit and not slow"` passed.
- No temporary Market Unit 24 helper files remain in the repository.

#### Acceptance

A caller can evaluate every historically issued candidate in one immutable
candidate issuance against one exact recommendation policy and receive one
immutable result for every evaluated exact offer. Failed, unavailable,
conflicting, qualified, and recommended outcomes are retained rather than
filtered from the evaluation.

Each result preserves exact parent issuance and candidate-row identity together
with provider, provider event, sportsbook, canonical game, market, side, quote
fetch time, sportsbook update time, kickoff, live state, American price, and
line evidence.

Selected-product identity, product-run identity, forecast event, forecast run,
forecast role, forecast generation time, model identity, model probability,
expected value, policy version, source-evidence fingerprint, governance
fingerprint, and derivation method remain explicit provenance.

Decision time, issuance quote age, and decision quote age remain independently
preserved. Persistence does not call the clock, re-evaluate freshness, rerun
qualification, recalculate Kelly sizing, or reload mutable bankroll or portfolio
state.

Every policy check remains preserved in deterministic order with its stable
identifier, mandatory classification, state, reason, observed evidence, and
required value. Qualified opportunities remain distinct from recommendations.
Recommendation eligibility requires all mandatory recommendation checks and an
actionable stake.

When sizing was available, full-Kelly, fractional-Kelly, raw, constrained,
rounded, and actionable stake values remain independently auditable. When
evaluation stopped before sizing, unavailable values remain null.

Bankroll basis, portfolio snapshot identity and observation time, and explicit
correlation evidence are preserved when supplied. Correlation membership is
never inferred, and multiple matching groups are rejected.

Individual result and evaluation identities are deterministic from canonical
immutable evidence. Repeated construction and persistence from identical
evidence produces identical identities and exact idempotent replay.

Results and evaluation manifests persist under schema-versioned,
identity-addressed immutable paths. Reads validate exact nested schemas,
supported versions, UTC timestamps, enum and scalar values, candidate identity,
result identity, evaluation identity, ordered result references, and filename
agreement. Existing content cannot be replaced by different content under the
same identity.

A production result may validly remain unavailable because its market-family
policy is not active. Persistence never manufactures qualification or a
recommendation merely because an artifact is required.

No request-time qualification, current-result selector, Bet Slip creation,
wager recording, sportsbook integration, automatic wagering, ledger mutation,
bankroll mutation, API route, frontend component, CLI command, notification, or
operational-worker behavior was introduced. The system does not place a
sportsbook wager.

---

### Market Unit 25: Present and Record Recommended Wagers [Completed]

#### Completed

2026-08-17

#### Goal

Present persisted qualification and recommendation results across the product
without inferring recommendation state from analytical expected value, edge
strength, confidence, bankroll, Kelly sizing, or mutable frontend settings.

Preserve persisted recommendation evidence, immutable edge analytics, editable
Bet Slip draft terms, and transient local what-if calculations as separate
contracts. Allow a user to explicitly record a completed game-wager draft in
Gridiron Edge while preserving both the recorded terms and the original
recommendation and reference-offer evidence.

Recording must update the betting ledger and tracked bankroll through one
rollback-safe domain operation. It records a wager locally and does not place a
sportsbook wager.

#### Files Added/Removed/Changed

Added:

- `src/gridiron_edge/betting/recording.py`
- `tests/unit/betting/test_recording.py`
- `tests/integration/api/test_portfolio_routes.py`
- `frontend/src/components/recommendations/recommendationPresentation.ts`
- `frontend/src/components/recommendations/recommendationPresentation.test.ts`
- `frontend/src/components/recommendations/RecommendationStatus.tsx`
- `frontend/src/components/recommendations/RecommendationStatus.test.tsx`
- `frontend/src/components/recommendations/RecommendationDetails.tsx`
- `frontend/src/components/betslip/recordWager.ts`
- `frontend/src/components/betslip/recordWager.test.ts`

Changed:

- `PLAN.md`
- `api-schema.json`
- `src/gridiron_edge/api/loaders.py`
- `src/gridiron_edge/api/routes/portfolio.py`
- `src/gridiron_edge/api/schemas/portfolio.py`
- `src/gridiron_edge/api/serializers/portfolio.py`
- `src/gridiron_edge/betting/ledger.py`
- `src/gridiron_edge/cli/betting.py`
- `tests/unit/api/test_schemas_portfolio.py`
- `tests/unit/api/test_serializers_portfolio.py`
- `tests/unit/betting/test_ledger.py`
- `tests/unit/cli/test_betting.py`
- `frontend/package.json`
- `frontend/pnpm-lock.yaml`
- `frontend/tsconfig.app.json`
- `frontend/src/api/hooks.ts`
- `frontend/src/api/schema.ts`
- `frontend/src/components/betslip/BetLegCard.tsx`
- `frontend/src/components/betslip/BetLegCard.test.tsx`
- `frontend/src/components/betslip/EdgesTable.tsx`
- `frontend/src/components/betslip/EdgesTable.test.tsx`
- `frontend/src/components/betslip/SlipPanel.tsx`
- `frontend/src/components/betslip/SlipPanel.test.tsx`
- `frontend/src/components/dashboard/FeaturedMatchupsGrid.tsx`
- `frontend/src/components/dashboard/FeaturedMatchupsGrid.test.tsx`
- `frontend/src/components/dashboard/ModelEdgesTable.tsx`
- `frontend/src/components/dashboard/ModelEdgesTable.test.tsx`
- `frontend/src/context/BetSlipContext.tsx`
- `frontend/src/context/BetSlipContext.test.tsx`
- `frontend/src/screens/GameDetail.tsx`
- `frontend/src/screens/GameDetail.test.tsx`
- `frontend/src/screens/LineShopping.tsx`
- `frontend/src/screens/LineShopping.test.tsx`
- `frontend/src/utils/betLegs.ts`
- `frontend/src/utils/betLegs.test.ts`
- `frontend/src/utils/betSlipSummary.test.ts`
- `frontend/src/utils/sportsbookPreferences.test.ts`

Renamed:

- `tests/unit/api/serializers/test_recommendations.py` to
  `tests/unit/api/serializers/test_recommendation_serializers.py`

Removed:

- Temporary Market Unit 25 implementation and correction helpers.
- Bet Slip v3 production storage, parser, and immutable snapshot contracts.
- Request-time bankroll and Kelly sizing parameters from the `/edges` frontend
  contract.
- Retired immutable Bet Slip snapshot fields for Kelly fraction, Kelly stake,
  bankroll, and Kelly multiplier.

#### Tests

- Persisted recommendation presentation mapped qualified, recommended, failed,
  unavailable, and conflicting lifecycle states directly from persisted result
  state.
- Offers without an attached persisted result were presented as Candidate.
- Analytical expected value, edge strength, confidence, bankroll, local Kelly
  multiplier, and frontend settings did not manufacture a recommendation
  state.
- Persisted suggested stake was presented directly from the persisted
  recommendation result without frontend recalculation.
- Supporting, failed, unavailable, and conflicting policy checks remained
  distinct.
- Policy identity, policy schema version, evaluation time, exact offer
  provenance, forecast provenance, model identity, and selected-product
  identity remained available through shared presentation components.
- Line Shopping presented persisted policy state for each exact sportsbook
  offer while preserving existing model guidance, price, best-line, best-price,
  and preferred-offer behavior.
- Available Edges presented analytical strength and persisted policy state as
  separate columns.
- Positive analytical edges without persisted recommendation evidence remained
  Candidate rather than Recommended.
- Sportsbook offer grouping, alternative expansion, exact price and line
  preservation, filtering, and Bet Slip staging remained unchanged.
- Game Detail renamed the model callout to Top Analytical Edge and presented
  persisted policy state independently from side, sportsbook, odds, and
  expected value.
- Persisted suggested stake was displayed only when a persisted recommendation
  supplied it.
- The generated frontend schema included the recommendation presentation,
  policy-check, sizing, bankroll-basis, exact-offer, forecast, and policy
  provenance contracts.
- TypeScript was pinned to the generator-compatible 5.9 line after
  `openapi-typescript` failed with the TypeScript 7 compiler API.
- The checked-in OpenAPI artifact matched the live FastAPI application schema.
- Bet Slip version 4 replaced the version 3 development contract outright.
- Active Bet Slip storage used `hm-betslip-v4` and
  `hm-betslip-mode-v4`.
- Runtime parsing used `parseBetLegV4` and `parseBetLegsV4`.
- Version 1, version 2, and version 3 Bet Slip entries were rejected rather
  than migrated.
- Canonical game and prop wager identities remained deterministic.
- Bet Slip version 4 separated `persistedRecommendation`, `edgeAnalytics`, and
  editable `draft` evidence.
- Game legs copied attached persisted recommendation evidence without
  re-evaluation.
- Candidate game legs remained valid with null persisted recommendation
  evidence.
- Prop legs remained valid with null persisted recommendation evidence.
- Edge analytics preserved model identity, reference price, model probability,
  model value, market value, expected value, edge strength, provider,
  provider-event identity, sportsbook, quote timestamps, kickoff, and live
  state.
- Edge analytics did not persist reference Kelly fraction, reference Kelly
  stake, reference bankroll, or reference Kelly multiplier.
- Negative assertions explicitly verified that retired immutable sizing fields
  were absent from the version 4 analytical snapshot.
- Local tracked and what-if bankroll values remained transient inputs to draft
  analysis.
- Local Kelly multiplier remained a transient input to draft analysis.
- Current price, current expected value, full Kelly, multiplier-adjusted stake,
  proposed stake, payout, and profit remained draft-analysis values.
- Zero bankroll and zero Kelly multiplier remained valid local analysis inputs.
- Missing bankroll or Kelly multiplier continued to block only dollar sizing.
- Editing current odds, proposed stake, sportsbook, and note did not alter
  persisted recommendation or immutable analytical evidence.
- Bet Slip summaries preserved singles and parlay price, stake, payout, profit,
  incomplete-state, and correlation-caveat behavior.
- The betting ledger added canonical Unit 24 result, evaluation, candidate, and
  policy identity columns.
- Manual wagers persisted all Unit 24 recommendation identity columns as null.
- Recommendation-backed wagers persisted all four Unit 24 identities.
- Partial recommendation identity chains were rejected before persistence.
- Empty recommendation identity strings were rejected.
- Exact reference-offer provenance continued to require an explicit provider
  and observation timestamp.
- Reference provider, provider-event identity, sportsbook, fetch time,
  sportsbook update time, kickoff, American odds, and line round-tripped
  unchanged.
- Recorded odds, line, sportsbook, and stake remained separate from persisted
  reference-offer terms.
- Recorded terms were allowed to differ from the original recommendation
  reference terms.
- Model name and model type continued to form one complete optional identity.
- `record_wager()` validated wager identity, market and side compatibility,
  price, stake, sportsbook, and line before writing either artifact.
- Moneyline recording rejected a non-null line.
- Spread and Total recording required a finite line.
- Ledger and bankroll transaction artifacts were created together.
- The bankroll transaction referenced the generated ledger bet identity.
- Existing ledger and bankroll history were preserved when a new wager was
  recorded.
- A bankroll write failure removed a newly created ledger when neither artifact
  previously existed.
- A failed second write restored the original ledger and bankroll artifacts.
- Existing artifacts were restored through temporary-file replacement.
- Successful recording returned both the bet identity and bankroll transaction
  identity.
- Repeated recording commands created distinct bet identities because no
  idempotency-key contract was introduced.
- The betting CLI used the shared rollback-safe recording domain owner instead
  of independently writing the ledger and bankroll transaction log.
- CLI output used Wager recorded language.
- `POST /portfolio/bets` accepted recorded game-wager terms and an optional
  complete Unit 24 identity chain.
- API request models were frozen and rejected unknown fields.
- Partial Unit 24 identity chains were rejected by request validation.
- Unknown or non-unique recommendation result identity was rejected.
- Candidate and policy identity mismatches were rejected.
- Persisted recommendation game, market, and side mismatches were rejected by
  the recording domain.
- The API derived provider, exact offer, model, probability, expected value,
  and policy evidence from persisted Unit 24 artifacts rather than accepting
  those values from the browser.
- Manual API recording did not fabricate recommendation provenance.
- API recording returned HTTP 201 with the recorded BetRow and bankroll
  transaction identity.
- The returned bankroll transaction referenced the returned bet identity.
- The API response explicitly stated that no sportsbook wager was placed.
- Portfolio BetRow serialization exposed result, evaluation, candidate, and
  policy identities mechanically from the ledger.
- The typed frontend mutation submitted requests through
  `POST /portfolio/bets`.
- Successful frontend recording invalidated Portfolio summary, bets,
  transactions, and curve queries.
- A game draft could not be recorded without current American odds, a positive
  proposed stake, and a nonempty sportsbook.
- Candidate and manual drafts submitted null recommendation identities.
- Recommendation-backed drafts submitted only result, evaluation, candidate,
  and policy identities from persisted recommendation evidence.
- The frontend did not submit provider, model, expected value, checks, policy
  details, or reference-offer evidence.
- Edited draft odds, stake, and sportsbook were submitted as recorded terms
  rather than being replaced by persisted reference terms.
- The recording confirmation distinguished persisted reference evidence from
  the draft terms being recorded.
- The confirmation stated that the action records a wager in Gridiron Edge and
  does not place a sportsbook wager.
- Cancelled recording preserved the staged draft.
- Failed recording preserved the staged draft and displayed its error.
- Successful recording removed only the recorded draft and preserved other
  staged wagers.
- Successful recording displayed the API no-placement message.
- Prop drafts did not expose the game-wager recording action.
- Recommendation serializer tests retained mechanical lifecycle and sizing
  coverage after being renamed to avoid a duplicate pytest module basename.
- Python bytecode removal confirmed that the duplicate
  `test_recommendations.py` basename was a deterministic import collision rather
  than stale cache.
- Focused Ruff checks passed throughout implementation.
- The configured repository-wide Pyrefly check passed with zero errors.
- Focused market, betting, API schema, API serializer, API integration, CLI,
  Bet Slip, recommendation presentation, and frontend request-mapping tests
  passed.
- `uv run ruff check . --fix` passed.
- `uvx pyrefly check` passed with zero errors.
- `uv run pytest -m "unit and not slow"` passed.
- `pnpm --dir frontend run lint` passed.
- `pnpm --dir frontend run build` passed.
- `pnpm --dir frontend test:run` passed with 402 tests.
- The checked-in OpenAPI artifact matched the live FastAPI application
  contract.
- No temporary Market Unit 25 helper files remain in the repository.

#### Acceptance

Line Shopping, Available Edges, and Game Detail present persisted recommendation
state through one shared lifecycle mapping. Qualified, recommended, failed,
unavailable, and conflicting labels come only from persisted result evidence.
An exact offer without attached persisted evidence is a Candidate. Expected
value, edge strength, confidence, bankroll, Kelly sizing, or other analytical
evidence cannot manufacture recommendation state.

Persisted policy evidence remains mechanically auditable. Policy identity and
schema version, evaluation identity and time, exact offer provenance, selected
product and forecast provenance, model identity, ordered checks, sizing
evidence, bankroll basis, portfolio evidence, and persisted suggested stake are
presented without rerunning qualification or recomputing sizing.

Analytical value remains separate from policy state. Line Shopping continues to
show model guidance, line quality, price quality, and preferred positive-value
offers. Available Edges continues to show expected value and analytical edge
strength. Game Detail continues to select its top analytical edge. None of
these analytical views relabels an offer as Recommended without persisted
recommendation evidence.

Bet Slip version 4 is the only active draft contract. Previous development
versions are not read or migrated. Each staged wager preserves persisted
recommendation evidence separately from immutable edge analytics and editable
draft terms. Local bankroll and Kelly inputs remain transient what-if inputs
rather than immutable recommendation or analytical provenance.

Changing current odds, proposed stake, sportsbook, note, tracked bankroll,
what-if bankroll, or Kelly multiplier does not change persisted lifecycle
state, persisted suggested stake, policy identity, checks, exact reference
offer, model evidence, or original expected value.

A user can explicitly record a complete Moneyline, Spread, or Total game-wager
draft. The action requires current American odds, a positive proposed stake,
and a sportsbook. Props remain staged analytical drafts and are not submitted
to the game-wager recording endpoint.

Before recording, the interface separates persisted reference evidence from the
editable draft terms that will be recorded. The confirmation and resulting API
message state that Gridiron Edge records the wager locally and does not place a
sportsbook wager.

Recommendation-backed recording sends only the persisted result, evaluation,
candidate, and policy identity chain. The backend resolves exact offer, model,
expected value, and policy provenance from strict persisted Unit 24 artifacts.
The browser cannot supply or replace trusted provider, model, expected value,
check, policy, or reference-offer evidence.

Recorded terms remain independently auditable from recommendation reference
terms. A user may record a different current sportsbook, American price, line,
or stake without overwriting the original exact offer or persisted suggested
stake.

The recorded-wager domain validates the complete operation before persistence.
Ledger and bankroll transaction writes share one rollback-safe orchestration
boundary. If either write fails, existing ledger and bankroll artifacts are
restored. A failed operation does not leave an orphaned open wager or an
unreferenced bankroll transaction.

The ledger preserves the generated bet identity, recorded wager terms, exact
reference-offer evidence, complete Unit 24 recommendation identities, model
evidence, status, settlement evidence, and closeout fields in one strict
canonical schema.

A manual or Candidate wager may validly contain no Unit 24 recommendation
identity. Recommendation-backed wagers require a complete result, evaluation,
candidate, and policy identity chain. Partial or empty chains are rejected.

A successful API request returns the recorded BetRow, bankroll transaction
identity, and explicit no-placement message. The frontend removes only the
successfully recorded draft and refreshes Portfolio data. Cancellation,
validation failure, provenance mismatch, or persistence failure preserves the
staged draft.

The betting CLI and Portfolio API use the same rollback-safe recording domain
owner. Neither route performs independent ledger and bankroll writes. No
sportsbook integration, sportsbook authentication, automatic wagering, wager
placement, or background recording behavior was introduced. The system records
wagers in Gridiron Edge and does not place sportsbook wagers.

---

### Active Implementation

No bounded implementation unit is currently selected in this root plan.

Market Unit 26 remains active but calendar-gated in:

- `docs/programs/market-unit-26/PLAN.md`
- `docs/programs/market-unit-26/ROADMAP.md`

Select a new program from `ROADMAP.md`, add one bounded active unit below this notice, and preserve the one-active-unit rule within this root document.
