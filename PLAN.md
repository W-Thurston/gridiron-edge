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

Weekly Prediction Input Integrity Unit 2: Validate and Rebuild Complete Elo History

### Goal

Replace the misleading incremental Elo lifecycle with one deterministic
full-history reconstruction contract. Reject incomplete canonical game history
before replacing Elo state, preserve the established pregame weekly-state
convention, and prove prior-season team strength survives into the next season
instead of resetting every team to the initial rating.

### Files Added/Removed/Changed

Changed:

- `PLAN.md`
  - Activated and closed the bounded Unit 2 implementation.
  - Preserved semantic readiness, affected-product disposition,
    prediction-input evidence, evaluation, and explanations as later work.
- `README.md`
  - Removed the retired `--fit-elo-all-years` option from the operating
    example.
- `ROADMAP.md`
  - Marked history-preserving refresh and complete-history Elo reconstruction
    complete while preserving the remaining corrective sequence.
- `src/gridiron_edge/ratings/elo/table.py`
  - Added canonical complete-history validation.
  - Required contiguous represented seasons beginning in 1999.
  - Allowed the latest represented season to remain partial.
  - Rejected malformed identities, duplicate games, invalid weeks, incomplete
    score pairs, negative scores, and incomplete historical coverage.
  - Invoked validation before Elo simulation.
  - Removed `update_elo_state_incremental()`.
- `src/gridiron_edge/ratings/elo/fit.py`
  - Replaced dual-mode fitting with one deterministic reconstruction.
  - Completed validation and simulation before writing Elo state.
- `src/gridiron_edge/cli/ratings.py`
  - Removed incremental and all-years Elo modes.
  - Exposed one complete-history fitting operation.
- `src/gridiron_edge/cli/main.py`
  - Removed the separate Elo fitting mode and pipeline option.
  - Made every active `build-elo` stage execute complete-history
    reconstruction.
- `src/gridiron_edge/cli/weekly_predict.py`
  - Removed the retired Elo-mode pipeline argument.
- `src/gridiron_edge/cli/post_week.py`
  - Removed the retired Elo-mode pipeline argument.
- `src/gridiron_edge/cli/full_retrain.py`
  - Removed the retired Elo-mode argument while retaining full-data pipeline
    scope.
- `src/gridiron_edge/cli/verify.py`
  - Removed the retired Elo-mode pipeline argument.
- `tests/fixtures/dataframes.py`
  - Added dedicated complete-history Elo games.
  - Extracted the existing two-game modeling input into a focused helper.
- `tests/fixtures/repos.py`
  - Added explicit complete-history Elo repository construction.
  - Preserved the default minimal games fixture.
- `tests/unit/ratings/test_elo_table.py`
  - Added complete-history validation coverage.
- `tests/integration/test_elo_fit.py`
  - Added deterministic reconstruction, valid output, prior-strength
    continuation, Week 1-to-Week 2 update, and predecessor-preservation tests.
- `tests/integration/test_features_pipeline.py`
  - Built Elo from valid complete history while preserving the focused
    two-game modeling input.
- `tests/e2e/test_prediction_pipeline.py`
  - Built Elo from valid complete history while preserving the original
    modeling-pipeline assertions.
- `tests/unit/cli/test_main.py`
  - Removed retired shared arguments and proved the obsolete pipeline option is
    absent.
- `tests/unit/cli/test_ratings.py`
  - Proved `ratings elo fit` exposes one complete-history operation.
- `tests/unit/cli/test_weekly_predict.py`
  - Updated the shared-pipeline invocation contract.
- `DECISIONS.md`
  - Added D39 for validated deterministic Elo reconstruction.
- `CHANGELOG.md`
  - Recorded behavior, tests, protected validation, and bounded scope.
- `HANDOFF.md`
  - Documented the current Elo operating contract and remaining limitations.

Added:

- None.

Removed:

- None.

### Tests

Focused Elo, simulator, integration, end-to-end, pipeline, and CLI tests passed.

Repository quality gates passed:

```text
uv run ruff check . --fix && \
uvx pyrefly check && \
uv run pytest -m "unit and not slow"
```


git diff --check passed.

Repository searches confirmed that update_elo_state_incremental, fit_elo_all_years, and --fit-elo-all-years are absent from active source. Unrelated incremental EPA, PBP, and feature behavior remains unchanged.

Protected validation used copies of the canonical games and Elo artifacts in a temporary repository. It proved:

7,292 canonical completed games covered contiguous seasons from 1999 through 2026;
reconstruction produced 19,006 Elo rows;
duplicate team-season-week identities remained zero;
Buffalo entered 2026 Week 1 at 1566.257299 and Week 2 at 1575.655543;
Detroit entered 2026 Week 1 at 1538.997399 and Week 2 at 1546.800161;
a second reconstruction produced identical output;
the exact 16-game 2026 Week 1-only frame was rejected;
the valid temporary Elo artifact remained byte-identical after rejection;
working-repository games and Elo artifacts remained unchanged;
no forecast, product, selection, model, market, edge, API, or frontend artifact changed.

Fixture-boundary verification confirmed that both focused modeling rows retain numeric Away and Home Elo values after Elo is built from the dedicated complete-history fixture.

Acceptance

Elo fitting now has one honest operational contract: deterministic reconstruction from validated complete canonical history. The false incremental function, fitting mode, CLI options, console language, and shared-pipeline argument are removed.

The reconstruction boundary rejects empty, malformed, duplicate, late-starting, and discontinuous history before replacing Elo state. A partial latest season remains valid. Validation and simulation failures preserve the existing Elo artifact.

Tests and protected validation prove prior-season strength survives offseason regression, Week 1 results update inherited state into Week 2, and the inspected reset-to-1500 failure mode cannot recur through the Elo fitting boundary.

The unit introduced no semantic weekly readiness, affected-product mutation, forecast regeneration, prediction-input persistence, model evaluation, calibration change, API or frontend behavior, or explanation functionality. The affected immutable 2026 Week 2 forecast events and weekly products remain unchanged.
