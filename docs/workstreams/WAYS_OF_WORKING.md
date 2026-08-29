# Gridiron Edge Ways of Working

## Purpose

These practices govern repository inspection, program planning,
implementation, validation, documentation, unit closure, and commits across
every Gridiron Edge workstream.

A new implementation or review thread must read this file before planning
or modifying the repository.

This file defines operating discipline. It does not own product purpose,
conceptual architecture, program scope, current workstream state, or the
active implementation unit.

## Authority and relationship to other artifacts

Authority remains:

1. `CONSTITUTION.md` — product purpose, audience, responsibility, and
   non-goals.
2. `VISION.md` — conceptual architecture and locked invariants.
3. `PROJECT_OUTLINE.md` — program capability sequence and workstream
   boundaries.
4. `DECISIONS.md` — durable architectural decisions.
5. Active workstream `ROADMAP.md` — inspected and approved workstream scope
   and sequence.
6. Active `PLAN.md` — exactly one active implementation unit.
7. `WAYS_OF_WORKING.md` — repository operating discipline.
8. `LONG_TERM_PROGRAM_GUIDE.md` — long-term progression and admission
   guidance.

`CONTEXT_SWITCH_PLAYBOOK.md` governs movement between threads.
`WAYS_OF_WORKING.md` governs how repository work is performed within those
threads.

Where this file conflicts with a higher-authority artifact, the
higher-authority artifact wins.

## Required practices

1. **Confirm before building.**
   Never assume code, schemas, artifacts, commands, tests, or documentation
   exist or have a particular shape. Inspect the current repository state
   before proposing a change. Prefer a small read-only audit over
   implementation based on stale context.

2. **Locate first, then read the owning boundary.**
   Use targeted searches to identify the relevant files, functions, tests,
   commands, artifacts, and generated contracts. Read the exact owning
   boundaries before designing or editing them.

3. **Design at two levels before implementation.**
   - **Program level:** lock the capability, motivation, boundaries,
     sequence, dependencies, and success criteria in `ROADMAP.md`.
   - **Unit level:** add one bounded implementation unit to `PLAN.md` with
     its goal, design decisions, tests, and acceptance criteria before
     changing code.

4. **Keep one active bounded unit.**
   `PLAN.md` may retain a concise summary of completed programs, but detailed
   completed-unit records are removed during major program closeout. Only
   one implementation unit should be active at a time. New work starts only
   after it is selected from `ROADMAP.md` and scoped for execution.

5. **Use descriptive implementation language.**
   Program and unit identifiers belong in planning documents only. Source
   names, comments, docstrings, tests, artifacts, commands, and commit
   subjects should describe lasting domain behavior rather than when the
   behavior was added.

6. **Commit small coherent units.**
   Each completed unit should produce one focused commit with a Conventional
   Commit subject and a detailed bullet-list body covering implementation,
   tests, and documentation. The corresponding `PLAN.md` update belongs in
   the same commit as the unit's implementation.

7. **Run focused gates during implementation.**
   Run linting, type checking, and focused tests after each meaningful
   change.

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
   the generated artifact or live response directly. Validate the relevant
   row counts, identities, uniqueness, coverage, provenance, representative
   values, timestamps, joins, and blocker states. Green tests do not replace
   real-data verification.

9. **Preserve generated-file ownership.**
   Regenerate checked-in schemas, clients, and derived contracts through
   their owning commands. Do not hand-edit generated artifacts.

10. **Close each unit completely.**
    After implementation and validation:

    - remove temporary migration and diagnostic scripts;
    - update affected operational and architectural documentation;
    - condense the completed `PLAN.md` unit to exactly these headings:
      `Completed`, `Goal`, `Files Added/Removed/Changed`, `Tests`, and
      `Acceptance`;
    - list every committed file added, removed, or changed, grouped by
      category with a concise description of its lasting responsibility or
      modification;
    - explicitly state `None` when an Added, Removed, or Changed category has
      no entries;
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

    Focused tests, quality gates, integration checks, and real-data
    validation performed for the unit.

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

## Applicability by work stage

### Inspection

The directly active practices are:

- confirm before building;
- locate first, then read the owning boundary;
- preserve the distinction between program and unit design;
- keep implementation inactive while inspection remains open;
- use lasting descriptive terminology;
- use explicit dates and repository history.

Inspection does not require an active `PLAN.md` unit. It prohibits one until
all required boundaries close and program scope is reconciled.

### Planning

The directly active practices are:

- design at program and unit levels;
- keep exactly one active bounded unit;
- use lasting implementation language;
- avoid unnecessary development-era compatibility;
- use explicit dates and repository history.

### Implementation

All required practices apply.

### Review

Reviewers enforce the practices applicable to the work stage. An inventory
review must not demand implementation-only closeout evidence, but it must
reject claims based on uninspected ownership or stale context.

### Closure

Closure particularly requires:

- one coherent Conventional Commit;
- applicable focused and full quality gates;
- real artifact or response verification;
- generated-file ownership;
- complete `PLAN.md` closeout;
- staged inventory and diff inspection;
- removal of temporary scripts;
- no unnecessary compatibility work;
- explicit dates and repository history.

## Review compliance

Each formal author-to-reviewer handoff includes a
`Ways-of-Working Compliance` disposition.

For inspection boundaries, review only stage-applicable rules. For
implementation and closure, review all twelve practices.
