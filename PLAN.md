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

### Market Unit 2: Migrate the Source-Neutral Quote Contract [Active]

#### Goal

Replace the development-era odds row schema with the locked provider-aware,
multi-book quote contract while preserving truthful nflverse consensus rows and
preparing current observations for The Odds API adapter.

#### Locked Direction

- Add distinct `provider` and `provider_event_id` fields.
- Retain `sportsbook` as the actual offered-price book; allow it to be null for
  truthful consensus sources such as nflverse schedule markets.
- Add `commence_time`, `sportsbook_updated_at`, and `is_live` provenance.
- Preserve canonical `season`, `week`, `game_id`, `game_date`, teams, market,
  side, American odds, and line orientation.
- Current operational ingestion accepts pregame featured markets only. Live
  quotes are representable but excluded from weekly edge consumption until a
  live-market program exists.
- Define deterministic quote identity from provider, provider event, book,
  market, side, line, and source update time. `fetched_at` identifies the local
  observation.
- Preserve all books in storage. Do not choose a best book during ingestion.
- Continue appending current observations to the observed quote ledger and
  atomically replacing `odds_current.parquet` only after successful validation.
- A failed request, invalid payload, or zero usable matched games must not
  overwrite the current snapshot. Partial matched coverage may be written with
  explicit diagnostics; weekly readiness owns completeness classification.
- Replace the current game-only market pivot in a later operational unit. This
  unit changes storage and validation only.
- Development compatibility is not required. Existing local odds artifacts may
  be regenerated under the new schema.

#### Design Scope

- canonical column order, nullability, timestamp normalization, and validation;
- provider-aware observed-ledger idempotency;
- atomic current-snapshot replacement;
- nflverse schedule adapter migration;
- legacy DraftKings adapter migration or isolation behind the new schema;
- fixture and store-test migration;
- documented artifact reset for incompatible local Parquet files.

#### Tests

Add unit coverage for exact schema enforcement, source-versus-book provenance,
nullable consensus sportsbook, UTC timestamps, live-state validation,
deterministic identity, observed-ledger idempotency, atomic snapshot behavior,
and migrated adapter output. Run focused odds-store and nflverse-adapter tests,
then the Python quality boundary.

#### Acceptance

All current odds producers and storage functions use the provider-aware quote
contract. Existing incompatible artifacts have an explicit reset path. The
current snapshot can safely preserve multiple books without selecting one, and
the next unit can implement The Odds API client and parser directly against the
locked schema.

---
