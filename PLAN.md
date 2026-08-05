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

### Market Unit 1: Select a Supported Provider and Lock the Market Contract [Active]

#### Goal

Select a documented, supported provider for current and upcoming NFL moneyline,
spread, and total markets, and lock the normalized quote, freshness,
configuration, failure, and identity contracts before implementation.

#### Locked Direction

- Current and upcoming markets are the first operational workstream because
  they unlock weekly edges and real-data frontend validation.
- Historical market archive and market-aware evaluation remain a separate
  later workstream because timestamped quote history, opening and closing line
  definitions, backfill pagination, archive volume, and leakage-safe selection
  require independent design and acceptance.
- Provider evaluation must consider both current and historical capabilities,
  but lack of affordable historical access does not automatically block the
  current-market workstream.
- Current snapshots and future historical archives will normalize to the same
  quote contract while using different replacement, append, retention, and
  consumption semantics.
- `weekly-predict` remains a consumer of an existing source-neutral market
  snapshot. It will not hide a paid or network-dependent provider fetch inside
  forecast publication.
- Forecast publication remains independent from market availability. Provider
  or market failure may block edge calculation but must not invalidate a
  prediction-ready selected weekly product.
- The legacy DraftKings adapter remains best-effort and is not a candidate for
  the supported operational dependency.

#### Design Scope

Provider evaluation:
- documented and supported API;
- NFL moneyline, spread, and total availability;
- sportsbook-level prices and multiple-book coverage;
- event, team, market, outcome, and timestamp identity;
- preseason and regular-season coverage;
- freshness, update cadence, rate limits, pricing, and usage terms;
- historical access, player props, and live markets as comparative attributes.

Normalized quote contract:
- provider and sportsbook identity;
- provider event identity and canonical `game_id`;
- season, week, and commence time;
- market and outcome identity;
- line or point value where applicable;
- American odds without lossy conversion;
- fetch timestamp and source provenance;
- explicit pregame versus live state;
- deterministic quote identity and validation rules.

Operational boundaries:
- secrets and configuration ownership;
- explicit provider client and adapter responsibilities;
- current snapshot freshness and staleness policy;
- provider failure, partial coverage, malformed response, and rate-limit states;
- game identity resolution and unmatched-event diagnostics;
- compatibility with `data/odds/odds_current.parquet`, `verify-week`, unified
  edge calculation, API serialization, and frontend consumers.

#### Tests

This design unit will be accepted through documented provider evidence and a
contract review. No production provider code is added until the provider,
normalized schema, freshness policy, failure behavior, command boundary, and
current-versus-historical separation are locked.

#### Acceptance

A provider decision is recorded with verifiable coverage, pricing, rate-limit,
usage, and historical-access evidence. The normalized quote contract and
operational boundaries are specific enough to implement without guessing.
The next bounded unit can build the current-provider adapter without reopening
current-versus-historical scope or weekly-prediction ownership.

---
