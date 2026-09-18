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

Weekly Prediction Input Integrity Unit 1: Preserve Historical Games During Weekly Refresh

### Goal

Ensure recurring explicit-season refresh updates the requested NFL season while
preserving every unrequested season in the canonical raw and cleaned games
artifacts. Prove the real shared pipeline seam selects the history-preserving
refresh boundary before downstream consumers run.

### Files Added/Removed/Changed

Changed:

- `PLAN.md`
  - Activated and closed the bounded history-preserving refresh unit.
  - Preserved later Elo, readiness, affected-product, provenance, evaluation,
    and explanation work as inactive.
- `ROADMAP.md`
  - Added the Weekly Prediction Input Integrity and Reproducibility corrective
    program and prioritization.
- `src/gridiron_edge/cli/main.py`
  - Recurring non-all-years execution now calls
    `fetch_nflverse_games_refresh()`.
  - Explicit-season recurring execution forwards the requested season.
  - No-season recurring execution delegates current-season resolution to the
    refresh wrapper.
  - All-years execution retains explicit replacement behavior.
- `tests/unit/cli/test_main.py`
  - Added direct shared-pipeline tests for full replacement, explicit-season
    recurring refresh, and inferred-current-season recurring refresh.
- `tests/unit/cli/test_weekly_predict.py`
  - Added direct freshness-stage coverage proving the composite supplies the
    explicit recurring-refresh contract.
- `src/gridiron_edge/ingest/nflverse/games.py`
  - Rejects an empty selected-season response before resolving or writing the
    registered raw artifact.
  - Preserves the existing artifact when the upstream fetch raises.
- `tests/unit/ingest/nflverse/test_games.py`
  - Proves upstream fetch failure preserves the existing raw artifact
    byte-for-byte.
  - Proves an empty response raises explicitly and preserves the existing raw
    artifact byte-for-byte.
- `DECISIONS.md`
  - Added D38, establishing ownership and behavior for replacement fetches,
    recurring season refreshes, and cleaned-history rebuilding.
- `CHANGELOG.md`
  - Recorded the corrected recurring refresh behavior, test evidence, protected
    validation, and bounded scope.
- `HANDOFF.md`
  - Documented the operating refresh contract, affected Week 2 evidence, and
    remaining corrective boundaries.


Added:

- None.

Removed:

- None.

### Tests

Focused tests passed:

```text
uv run pytest \
  tests/unit/ingest/nflverse/test_games.py \
  tests/unit/transform/test_games_nflverse.py \
  tests/unit/cli/test_main.py \
  tests/unit/cli/test_weekly_predict.py \
  tests/unit/cli/test_post_week.py
```
