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

### Completed

Weekly Prediction Input Integrity Unit 3: Require Verified Elo Lineage Before Weekly Prediction

### Goal

Persist exact source and output identity evidence whenever Elo is reconstructed,
then require that evidence at the pre-execution prediction-availability
boundary. Prevent Elo-only and trained Elo-dependent models from executing or
publishing forecast evidence when current games or Elo artifacts cannot be
authenticated against the recorded reconstruction.

### Files Added/Removed/Changed

Changed:

- `PLAN.md`
  - Activated and closed the bounded Unit 3 implementation.
  - Preserved affected-product disposition, immutable prediction-input
    evidence, corrected evaluation, and model explanation as later work.
- `ROADMAP.md`
  - Marked semantic Elo-lineage readiness complete while preserving Units 4
    through 6.
- `src/gridiron_edge/ratings/elo/fit.py`
  - Added lineage construction and persistence after successful Elo writing.
- `src/gridiron_edge/models/game_prediction/availability.py`
  - Required verified Elo lineage for Elo availability.
  - Derived trained-model lineage requirements from exact prediction feature
    contracts.
- `tests/integration/test_elo_fit.py`
  - Proved public Elo reconstruction writes matching lineage.
  - Proved repeated reconstruction preserves source and output content
    identities.
  - Proved rejected history preserves predecessor Elo and lineage artifacts.
- `tests/unit/models/game_prediction/test_availability.py`
  - Covered valid, missing, stale, malformed, and unsupported lineage behavior.
  - Proved all five current registered model contracts require Elo.
  - Proved future non-Elo contracts remain independently evaluable.
- `tests/unit/models/game_prediction/test_weekly_execution.py`
  - Proved unavailable prediction families stop before model execution.
- `tests/unit/cli/test_weekly_predict.py`
  - Proved unavailable and malformed lineage write no forecast events and cache
    no forecast-run identity.
- `DECISIONS.md`
  - Added D40 for verified Elo lineage before weekly prediction.
- `CHANGELOG.md`
  - Recorded the shipped behavior, tests, protected validation, and bounded
    scope.
- `HANDOFF.md`
  - Documented the lineage sidecar, operating procedure, fail-closed behavior,
    and remaining limitations.

Added:

- `src/gridiron_edge/ratings/elo/lineage.py`
  - Owns schema-1 lineage, exact byte identities, strict serialization and
    loading, safe path resolution, and current-artifact verification.
- `tests/unit/ratings/test_elo_lineage.py`
  - Covers strict lineage construction, persistence, loading, validation, and
    tamper detection.

Removed:

- None.

### Tests

Focused lineage, Elo integration, prediction availability, policy, weekly
execution, and weekly CLI tests passed.

Repository quality gates passed:

```text
uv run ruff check . --fix && \
uvx pyrefly check && \
uv run pytest -m "unit and not slow"
```

git diff --check passed.

Automated evidence proves:

schema-1 lineage is written after successful public Elo reconstruction;
exact persisted games and Elo bytes are hashed directly;
artifact paths remain repository-contained;
row counts, ordered columns, season scope, and latest weeks are recorded;
strict loading rejects malformed, unsupported, unsafe, or incorrectly typed evidence;
missing lineage and missing referenced artifacts are unavailable;
changed games or Elo content invalidates lineage;
changed rows, columns, or season scope invalidates lineage;
verified lineage does not override missing exact-week Elo coverage;
all five current registered game models directly require AWAY_ELO, HOME_ELO, and ELO_DIFF;
future models without Elo dependencies remain independently eligible;
blocked execution performs no model prediction;
unavailable or malformed lineage writes no forecast events.

Protected validation used only temporary copies and proved:

the games reference contained 7,292 rows from 1999-2000 through 2026-2027, latest completed week 1;
the games SHA-256 digest was 32c23ed2c75895f9e1466893020ed41ac4c261fe4c2cffb5cb372976c97a6efc;
the Elo reference contained 19,006 rows from 1999-2000 through 2026-2027, latest state week 2;
the Elo SHA-256 digest was 7818da246ca5c248f4cdad2341cf0f6cbd4753f6843357093821e2ab4e5fa861;
valid lineage made all six current availability facts true for the complete 16-game Week 2 schedule;
changed games, changed Elo, and missing lineage each made all six current availability facts false;
malformed lineage raised explicitly;
weekly execution stopped after feature-contract inspection and before model prediction;
temporary artifacts were restored to valid lineage;
working games, Elo, schedule, model, champion, forecast, and weekly-product artifacts remained read-only;
selection, market, edge, API, and frontend artifacts were not touched.
Acceptance

Every successful public Elo reconstruction now writes strict schema-1 evidence identifying the exact canonical games source and persisted Elo output.

Weekly prediction availability authenticates both current artifacts before policy resolution. Elo is available only when lineage verifies and every requested game has exact-week Away and Home state.

Missing or stale lineage blocks every current trained game model before prediction because all current feature contracts consume Elo. Malformed or unsupported lineage fails explicitly. Future non-Elo contracts remain independently evaluable.

Unavailable or malformed lineage produces no model prediction, forecast-event write, forecast-run identity, weekly-product composition, or selected weekly product.

The affected immutable 2026 Week 2 forecast events and weekly products remain unchanged. The unit introduced no retrospective prediction generation or selection, complete prediction-input persistence, corrected evaluation, API or frontend behavior, or explanation functionality.
