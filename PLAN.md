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

### Market Unit 4: Integrate Current Markets Operationally [Active]
