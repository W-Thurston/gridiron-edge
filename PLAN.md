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

Units 1 through 3 corrected recurring history preservation, replaced the false
incremental Elo lifecycle with deterministic complete-history reconstruction,
and required exact persisted Elo lineage before weekly prediction.

Only Unit 4 is active. Immutable prediction-input evidence, corrected forecast
evaluation, and explanation evidence remain inactive future units.

Market Unit 26 remains active but calendar-gated in:

- `docs/programs/market-unit-26/PLAN.md`
- `docs/programs/market-unit-26/ROADMAP.md`

#### Completed

Weekly Prediction Input Integrity Unit 4 recorded and enforced one exact
known-defect disposition for the immutable 2026 Week 2 forecast evidence.

The disposition covers both affected live logistic Win runs, all 32 affected
forecast events, both affected weekly products, and the exact selected affected
product. It classifies Win probability and derived spread as known defective
because of incomplete Elo source history. Total evidence is not classified as
affected.

Operational use of an affected product is now blocked for weekly readiness,
weekly edge calculation, candidate issuance, and manual prediction rendering.
Historical forecast-event loading, weekly-product loading, Games API display,
and postgame closeout access remain available.

#### Goal

Preserve the original immutable 2026 Week 2 forecast events, weekly products,
index entries, and scoped current selection while recording one authenticated
known-defect disposition and preventing the affected products from being used
as operationally valid prediction evidence.

#### Files Added/Removed/Changed

Added:

- `src/gridiron_edge/evaluation/forecast_evidence_disposition.py`
  - Owns the schema-1 disposition contract, deterministic identity, strict
    validation, immutable-evidence authentication, product applicability, and
    shared operational-use enforcement.
- `src/gridiron_edge/evaluation/forecast_evidence_disposition_store.py`
  - Owns schema-versioned identity-addressed JSON persistence, exclusive
    immutable publication, exact replay, strict loading, canonical-path
    validation, and deterministic scope and product listing.
- `tests/unit/evaluation/test_forecast_evidence_disposition.py`
  - Proves domain identity, validation, evidence authentication, applicability,
    ambiguity handling, and operational rejection.
- `tests/unit/evaluation/test_forecast_evidence_disposition_store.py`
  - Proves immutable persistence, strict deserialization, canonical paths,
    deterministic listing, concurrent exact replay, and conflicting concurrent
    publication rejection.

Changed:

- `PLAN.md`
  - Closed Unit 4 with the implemented contract, complete file inventory,
    validation evidence, and acceptance result.
- `ROADMAP.md`
  - Marked affected-evidence disposition complete and advanced the prediction
    integrity program to immutable prediction-input evidence.
- `HANDOFF.md`
  - Documented the disposition artifact, current operational enforcement,
    recovery procedure, and intentionally unchanged historical and Games API
    paths.
- `DECISIONS.md`
  - Recorded the immutable external-disposition architecture and operational
    enforcement boundary.
- `CHANGELOG.md`
  - Recorded the shipped Unit 4 behavior, protected validation, real artifact,
    source preservation, and quality gates.
- `api-schema.json`
  - Regenerated the checked-in OpenAPI schema with the known-defective edge
    blocker.
- `frontend/src/components/field-status/edgeResultStatus.ts`
  - Added the exhaustive presentation message for known-defective forecast
    evidence.
- `frontend/src/components/field-status/edgeResultStatus.test.ts`
  - Proved the new blocker maps to its stable presentation message.
- `frontend/src/components/field-status/EdgeResultStatus.test.tsx`
  - Proved the React status component renders the known-defect message.
- `src/gridiron_edge/api/meta.py`
  - Added stable unavailable metadata for known-defective forecast evidence.
- `src/gridiron_edge/api/routes/edges.py`
  - Mapped blocked edge diagnostics to the new unavailable metadata.
- `src/gridiron_edge/cli/output.py`
  - Rejects affected selected products before display adaptation or PNG and
    HTML writes.
- `src/gridiron_edge/cli/production_chain.py`
  - Rejects affected selected products before forecast-event loading, quote
    loading, as-known quote derivation, candidate evaluation, or issuance
    persistence.
- `src/gridiron_edge/cli/verify_week.py`
  - Resolves exact dispositions for the selected product and adds the
    known-defective readiness blocker without replacing independent blockers.
- `src/gridiron_edge/evaluation/weekly_readiness.py`
  - Added the durable known-defective forecast-evidence prediction blocker.
- `src/gridiron_edge/market/edge_diagnostics.py`
  - Added the known-defective forecast-evidence edge blocker.
- `src/gridiron_edge/market/weekly_edge_service.py`
  - Returns an explicit blocked result with zero rows before market loading or
    edge calculation when the selected product is affected.
- `tests/unit/api/test_edges_route_diagnostics.py`
  - Proved stable API metadata mapping for the new blocker.
- `tests/unit/cli/test_output.py`
  - Proved rendering rejection order, exact disposition lookup, identity
    validation, undisposed behavior, and explicit ambiguity failure.
- `tests/unit/cli/test_production_chain_cli.py`
  - Proved candidate issuance stops before all downstream evidence reads and
    writes.
- `tests/unit/cli/test_verify_week.py`
  - Proved exact disposition lookup, additive blocker composition, undisposed
    behavior, identity validation, and explicit ambiguity failure.
- `tests/unit/evaluation/test_weekly_readiness.py`
  - Proved known-defective evidence blocks prediction readiness without
    independently blocking market readiness.
- `tests/unit/market/test_edge_diagnostics.py`
  - Proved the known-defective blocker is distinct from missing predictions.
- `tests/unit/market/test_weekly_edge_service.py`
  - Proved affected products return an explicit blocked result before market
    loading and edge calculation.

Removed:

- None.

The following ignored operational artifact was created through the public
store boundary and is intentionally not committed under the repository's
existing `data/` policy:

- `data/output/forecast_evidence_dispositions/schema=1/dispositions/05f36ea9f3f014c3ed2b9bd2f2586540189ac8fea3dcf5db3b354ad020565eeb.json`

#### Tests

Focused Unit 4 tests passed for:

- disposition contracts and evidence authentication;
- immutable disposition storage and strict loading;
- concurrent exact replay and conflicting publication;
- weekly readiness and blocker composition;
- weekly edge blocking;
- API edge metadata;
- candidate-issuance rejection order;
- prediction-rendering rejection order;
- generated frontend blocker presentation.

Repository quality gates passed:

```text
uv run ruff check . --fix
uvx pyrefly check
uv run pytest -m "unit and not slow"
```

Frontend contract and production gates passed:

```text
uv run gridiron api export-schema
cd frontend
pnpm gen:api
pnpm exec vitest run
pnpm build
cd ..
```


The checked-in OpenAPI schema consistency test passed.

Protected real-artifact validation used public loaders and persistence
boundaries in a temporary repository and confirmed:

- both affected runs contain exactly 16 unique live logistic Win events;
- all 32 affected event identities resolve exactly once;
- both affected weekly products contain exactly 16 rows;
- product and Win run identities match the affected runs;
- product Win event identities exactly match their corresponding run events;
- every available derived spread references its row's affected Win event;
- no Total event, run, model, type, or role identity is present;
- the current Week 2 selection identifies the expected affected product;
- disposition creation, authentication, persistence, strict reload, and exact
  replay succeed;
- tampered content fails canonical identity validation;
- the selected affected product is operationally rejected;
- an unrelated product remains eligible;
- readiness reports `known_defective_forecast_evidence`;
- edge calculation returns the explicit known-defect blocker and zero rows;
- candidate issuance stops before downstream reads or writes;
- prediction rendering stops before adaptation or output writes;
- copied and working source artifacts remain byte-identical.

The persisted disposition is:

```text
recorded_at:
2026-09-19T18:00:00+00:00

disposition_id:
05f36ea9f3f014c3ed2b9bd2f2586540189ac8fea3dcf5db3b354ad020565eeb

artifact SHA-256:
8529487ed56c5130eba032c2e8d1764b22e95f796b6336a32ac299266746318d
```

Protected source hashes remained unchanged:

```text
08038e6f8e295ab45d3b04b55bb7804b8b3a86f5a1edd34a71119635333d5a62
data/output/predictions/forecast_events.parquet

41b31a6c9e4941548c4a88c1daada151011365d70736b509a33286389861f7bd
data/output/weekly_products/index.json

820f7e5d09732d3b66024d5374b70db88aa1d3190162bac4bb56ff503f833b8b
data/output/weekly_products/current.json

e53f1f20fd8a9451d81e87ad4543f3a43f13a034a4ca0b0dad34b423460df8bc
first affected Week 2 product

1dcdc458dc70b3fce106d6f635fcce925256f1ef35269d3e36e6aac903524755
selected affected Week 2 product
```

Claude completed an independent read-only review and approved the unit for documentation closure after confirming the complete selected-product consumer inventory and additive readiness-blocker behavior.

#### Acceptance

One strict immutable schema-1 disposition records the complete known 2026 Week 2 incident across both affected live logistic Win runs, all 32 affected events, both affected weekly products, and the exact selected affected product.

The disposition classifies only Win probability and derived spread as known defective because of incomplete Elo source history. It does not classify Total evidence as affected.

Original forecast events, weekly-product Parquet artifacts, product index, and scoped current selection remain byte-identical and historically loadable. No corrected forecast, retrospective replacement product, automatic reselection, candidate issuance, edge artifact, or operational rendering was created.

The selected affected product is not prediction-ready, cannot calculate edges, cannot issue candidates, and cannot be freshly rendered as an operational prediction output. The operational artifact is persisted under the ignored data/output/ tree according to existing repository policy.

Unit 4 is implemented, validated against real evidence, independently reviewed, documented, and ready for downstream work.
