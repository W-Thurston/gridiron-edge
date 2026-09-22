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

### Active Program: Weekly Prediction Input Integrity and Reproducibility

Units 1 through 5 completed history-preserving weekly refresh, deterministic complete-history Elo reconstruction, exact persisted Elo-lineage enforcement, external disposition of the affected immutable 2026 Week 2 forecast evidence, and immutable prediction-input evidence for newly generated live weekly forecasts.

Corrected operational forecast generation and evaluation, model-quality assessment, and explanation evidence remain inactive future units.

Market Unit 26 remains active but calendar-gated in:

- `docs/programs/market-unit-26/PLAN.md`
- `docs/programs/market-unit-26/ROADMAP.md`

### Weekly Prediction Input Integrity Unit 5: Persist Immutable Input Evidence for New Live Weekly Forecasts [Completed September 21, 2026]

#### Completed

Implemented immutable, authenticated prediction-input evidence for every newly generated live weekly forecast event. Statistical and Elo execution now return predictions and exact computational evidence together. Weekly publication persists and authenticates immutable binary snapshots and family evidence before forecast events become durable.

Existing forecast events and weekly products remain unchanged. This unit does not retrofit historical evidence, regenerate the affected 2026 Week 2 forecasts, select a replacement product, perform corrected evaluation, add explanation evidence, or change API or frontend contracts.

#### Goal

Require every newly generated live weekly forecast event to have complete, immutable, strictly validated prediction-input evidence before persistence.

Evidence preserves the clean tracked Git revision, bounded source identities, required replay bytes, exact ordered statistical inputs, transformed estimator inputs, estimator and post-processing outputs, Elo formula inputs, final event outputs, and one-to-one binding to persisted forecast UUIDs.

#### Files Added/Removed/Changed

Added:

- `src/gridiron_edge/evaluation/prediction_input_evidence.py`
- `src/gridiron_edge/evaluation/prediction_input_evidence_store.py`
- `src/gridiron_edge/evaluation/prediction_input_sources.py`
- `src/gridiron_edge/models/game_prediction/prediction_execution.py`
- `tests/unit/cli/test_weekly_predict_publication.py`
- `tests/unit/evaluation/test_prediction_input_evidence.py`
- `tests/unit/evaluation/test_prediction_input_evidence_store.py`
- `tests/unit/evaluation/test_prediction_input_sources.py`
- `tests/unit/models/game_prediction/test_elo_prediction_execution.py`
- `tests/unit/models/game_prediction/test_games_model_evidence.py`
- `tests/unit/models/game_prediction/test_post_process_resolution.py`
- `tests/unit/models/game_prediction/test_prediction_execution.py`
- `tests/unit/models/game_prediction/test_statistical_prediction_execution.py`
- `tests/unit/models/game_prediction/test_weekly_execution_evidence.py`

Changed:

- `PLAN.md`
- `src/gridiron_edge/cli/weekly_predict.py`
- `src/gridiron_edge/models/artifact.py`
- `src/gridiron_edge/models/elo/model.py`
- `src/gridiron_edge/models/game_prediction/base.py`
- `src/gridiron_edge/models/game_prediction/model.py`
- `src/gridiron_edge/models/game_prediction/post_process.py`
- `src/gridiron_edge/models/game_prediction/weekly_execution.py`
- `tests/unit/cli/test_weekly_predict.py`
- `tests/unit/models/game_prediction/test_weekly_execution.py`
- `tests/unit/models/test_artifact.py`
- `tests/unit/models/test_games_model.py`
- `tests/unit/models/test_games_trainer.py`

Removed:

- None.

#### Tests

Passed Ruff, Pyrefly, Python compilation, the full non-slow unit suite, focused Unit 5 suites, and `git diff --check`.

Legacy-versus-evidence equivalence passed for all five statistical families: Logistic, Random Forest, and XGBoost Win; Random Forest and XGBoost Total.

Protected validation used a clean temporary repository and real 2026 Week 2 inputs. The default policy validated `win_prob / logistic` with `total / random_forest`; an override validated `win_prob / elo` with `total / random_forest`.

Protected validation proved clean revision resolution, exact source capture and recapture, complete family coverage, unique UUIDs, immutable snapshot and evidence publication, strict reload, idempotency, real statistical and Elo replay, source-drift rejection, tamper rejection, unrelated-event absence, and event persistence only after evidence success. Statistical replay used `rtol=0.0` and `atol=1e-12`.

Independent read-only review found no blocking correctness or safety defect. Requested statistical equivalence and XGBoost coverage were added and passed.

#### Acceptance

Every newly generated live weekly forecast event is bound to exactly one immutable, strictly validated family evidence artifact before event persistence.

Statistical evidence preserves exact source revision and identities, model and metadata bytes, scaler or explicit absence, external calibrator or explicit absence, ordered feature schema, raw and transformed inputs, raw and post-processed outputs, post-processing resolution, and final outputs.

Elo evidence preserves exact source revision and identities, Elo state and lineage bytes, lineage identities, Away and Home ratings, formula identity, divisor, probabilities, and final outputs.

Snapshots and evidence are immutable, create-only, strict under reload, idempotent under exact replay, and resistant to conflicting publication. Source inputs must remain byte-identical from capture through publication. Forecast events cannot persist until all evidence is durable and authenticated. Product composition remains downstream.

All pre-existing forecast events and weekly products remain byte-identical and readable without Unit 5 evidence. No corrected forecast, replacement product, automatic reselection, corrected evaluation, explanation artifact, API contract, or frontend surface was introduced.

Nonblocking follow-up: availability validates feature columns but does not yet validate persisted feature-set identity and modeling schema version as strictly as execution. Execution remains fail-closed.
