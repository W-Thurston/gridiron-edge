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
- `uvx pyrefly check`
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
