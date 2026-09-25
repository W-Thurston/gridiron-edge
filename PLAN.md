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
   uv run ruff format . && \
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

10. **Close each unit completely.** Follow `.claude/commands/closeout.md`
    (run as /closeout in Claude Code).

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

### Completed Program: Weekly Prediction Input Integrity and Reproducibility [Closed September 25, 2026]

Units 1 through 5 completed history-preserving weekly refresh, deterministic complete-history Elo reconstruction, exact persisted Elo-lineage enforcement, external disposition of the affected immutable 2026 Week 2 forecast evidence, and immutable prediction-input evidence for newly generated live weekly forecasts.

Corrected operational forecast generation and evaluation, model-quality assessment, and explanation evidence are now tracked as ROADMAP.md Tier 2 #4–7 and Tier 3 #8–10, under the active program below.

### Active Program: Foundation Completion (Tiers 1–4)

Resolves the correctness defects and documentation drift found while reconciling ROADMAP.md against the repository (Tiers 1–3), and completes calendar-driven market proof (Tier 4 / Market Unit 26, running in parallel). Full unit sequencing, the verified-defects table, and acceptance criteria live in ROADMAP.md's "Active Next Program" section.

**Gate:** ROADMAP Tier 5 #16 (the game-model feature program) does not start until every unit here completes and Market Unit 26 has closed Tier 4 #11, #12, and #14. Tier 4 #13 (recommendation-policy maturation) continues independently on its own calendar and does not gate Tier 5.

Market Unit 26's calendar-gated detail remains in:

- `docs/programs/market-unit-26/PLAN.md`
- `docs/programs/market-unit-26/ROADMAP.md`

#### Unit 1: Docs reconciliation [Completed September 25, 2026]

##### Completed

Brought `HANDOFF.md`, `CHANGELOG.md`, and `DECISIONS.md` back into agreement with verified current repository state: two previously undocumented shipped changes (the development forecast role and full-retrain's exact-run classification champion selection) now have `CHANGELOG.md` entries and `DECISIONS.md` records; seven stale or inaccurate `HANDOFF.md` passages were corrected in place against a live read of the code and data they describe, with no new sections added.

##### Goal

Prevent later Foundation Completion units (U2 onward) from being scoped or verified against documentation that no longer matches the repository.

##### Files Added/Removed/Changed

Added:
- None.

Changed:
- `CHANGELOG.md` - Added entries for the development forecast role (`2236adc`) and full-retrain's exact-run classification champion selection (`6932c77`), both shipped 2026-09-22 with no prior changelog record.
- `DECISIONS.md` - Added D43 (development forecast role) and D44 (exact-run classification champion selection); corrected D42's stale claim that the availability/execution metadata-parity follow-up was still outstanding.
- `HANDOFF.md` - Forecast Event Contract now lists `development` alongside `live`/`backfilled` with its actual eligibility and exclusions; the Canonical Data Pipeline and Known Limitations sections no longer list the shipped availability metadata preflight as outstanding; the Pregame Workflow section states that availability and execution now share one strict metadata contract instead of describing availability as a weaker preflight; the Week 2 known-defect section states that the current selection no longer matches the disposition, with a cross-reference to `ROADMAP.md`'s Canonical Artifact Decision; the Postgame Workflow section describes closeout as reporting the product's actual role rather than assuming `live`; the Full Retrain Workflow section describes exact-run calibration and champion ranking; the `recommended_bet_results` dataset path reads `schema=3`; the Production Recommendation Chain section states that three candidate-issuance, two governance, and three policy artifacts currently exist for Week 1 rather than presenting one canonical identity set, with a cross-reference to `ROADMAP.md` Track B Unit M3.

Removed:
- None.

##### Tests

Documentation only; no code, schema, or `data/` change. Verification performed: every corrected claim checked against a live read of the source file, code, or on-disk artifact it describes (`ForecastRole` enum, `availability.py`, `champion.py`, `live_forecast_closeout.py`, `current.json`, the forecast-evidence disposition, and the candidate-issuance/governance/policy/preflight directories); `git diff --check` clean on every commit in this unit.

##### Acceptance

`HANDOFF.md` describes only current behavior for every section touched; the development-role and exact-run-calibration changes are recorded in `CHANGELOG.md` and `DECISIONS.md`; the ROADMAP Tier 1 #1 reconciliation this unit covers is complete.

#### Unit 2: Evaluation correctness [Completed September 25, 2026]

##### Completed

Fixed three verified correctness defects (D2, D3, D4) in the shared game-model training and evaluation infrastructure, all upstream of any evaluation or regeneration work in later units. No model was retrained and no persisted artifact changed; every fix is verified against the real `data/modeling/modeling_file.parquet` read-only.

##### Goal

Make the walk-forward backfill, the gated promotion path, and the standard train/holdout split trustworthy before Tier 2 #5–7 build corrected evaluation evidence on top of them.

##### Files Added/Removed/Changed

Added:
- None.

Changed:
- `src/gridiron_edge/models/game_prediction/_features.py` - Added `_chronological_split_masks()`, a shared helper enforcing that training rows must be strictly before the earliest holdout season (not merely "not a holdout season"), so a season more recent than the holdout window is excluded from both pools instead of silently joining training (D4). `_prepare_data()` now returns two additional aligned `Series` (`year_train`, `year_hold`) giving each row's season label, co-indexed with that row's features.
- `src/gridiron_edge/models/game_prediction/total.py` - `_prepare_total_data()` uses the same shared `_chronological_split_masks()` helper (D4) and returns the same two additional aligned year `Series` (D2 prerequisite).
- `src/gridiron_edge/models/game_prediction/_epa_window.py` - `WindowData` gained `year_train`/`year_holdout` fields; `_get_cached_window_data()` threads them through from `_prepare_data()`.
- `src/gridiron_edge/models/game_prediction/base.py` - `_filter_for_walk_forward()` no longer looks up each row's season via a positional index label into a separately-sorted `df_reference` DataFrame (the source of the misattribution defect, D2); it now takes `year_train_orig`/`year_hold_orig` `Series` that are co-indexed with the feature matrices by construction, eliminating the entire class of index-alignment bugs. `_prepare_window()` updated to source and pass these through for both the classification (EPA-window cache) and regression (`_prepare_total_data`) branches.
- `src/gridiron_edge/cli/models.py` - `_apply_promotion_decision()` now branches on `challenger_meta.task`: a regression challenger (e.g. Total) is compared with `compare_regression_models()` (R²/coverage/MAE gates), never with the classification Brier/ECE/AUC gates that always rejected it because a regression metadata's Brier score is undefined (D3). Added `_regression_result_from_metadata()` helper.
- `tests/unit/models/test_win_training_data.py` - Updated 4 existing `_prepare_data()` unpacking sites for the new 8-tuple return; added `TestChronologicalSplitMasks` and an end-to-end test proving a season after the holdout window is excluded from both splits (D4).
- `tests/unit/models/test_total_training_data.py` - Updated 4 existing `_prepare_total_data()` unpacking sites for the new 8-tuple return; added the equivalent D4 end-to-end test.
- `tests/unit/models/test_games_trainer.py` - Added `TestFilterForWalkForward`: two tests proving correct season re-attribution using deliberately non-contiguous, gapped indices (the shape upstream row-dropping actually produces), including one row moving from the original holdout pool into the new training pool (D2).
- `tests/unit/cli/test_models.py` - Added `TestApplyPromotionDecisionRegressionTask`: a genuinely-better regression challenger is promoted, a worse one is rejected, and the printed comparison is confirmed to be the regression format, not the classification one (D3).

Removed:
- None.

##### Tests

Ruff, Pyrefly, and the full non-slow unit suite passed (4,010 tests, up from 4,001; the 9 new tests are listed above). Real-data validation: ran `_prepare_data()` (Win) and `_prepare_total_data()` (Total) directly against the current `data/modeling/modeling_file.parquet` — the in-progress `2026-2027` season now appears in neither `train_seasons` nor `holdout_seasons` for either task, and zero `2026-2027` rows appear in either split, confirming D4 on real data. Confirmed the pre-fix masking logic would have placed those same rows in training (`~year.isin(holdout)` evaluates `True` for `2026-2027`). No `data/` artifact was written; no model was retrained.

##### Acceptance

The walk-forward backfill attributes every row to its own season using data co-indexed with that row, never by re-deriving row identity from a separately-indexed reference; a Total (regression) challenger can be promoted through `gridiron models train` on its own gates; the standard train/holdout split never trains on a season more recent than the earliest holdout season. All three fixes are proven by tests that fail against the prior behavior and pass against the fix.

#### Unit 3: EPA-window prediction/serving parity [Completed September 25, 2026]

##### Completed

Fixed D1 (EPA-window train/serve skew): deployed Win models are tuned with EPA rolling windows of 6 (Logistic), 8 (Random Forest), and 6 (XGBoost), but live prediction, availability inspection, and walk-forward backfill always re-derived EPA features at the default window of 4 instead of each artifact's own validated window. Reused the existing training-side `_rebuild_features_with_window()` helper at all three consumption boundaries and recorded the window in prediction-input evidence's feature schema, bumping its schema version. No model was retrained and no persisted artifact or evidence changed; the fix is verified against the real current 2026 Week 3 upcoming schedule read-only.

##### Goal

Carry each artifact's own validated `epa_window` into live prediction, availability inspection, and walk-forward backfill so a model is always served the same EPA rolling window it was tuned and evaluated with.

##### Files Added/Removed/Changed

Added:
- None.

Changed:
- `src/gridiron_edge/models/game_prediction/model.py` - `_predict_upcoming_classification_with_evidence` and `_predict_upcoming_regression_with_evidence` (the only prediction path `weekly_execution.py` calls) now rebuild canonical EPA at the persisted artifact's own `epa_window` before feature extraction. Added `_validated_epa_window()`, applied inside `_validate_prediction_metadata()` (parallel to the existing `modeling_schema_version` check) and by `_prediction_feature_schema()`, which now threads `epa_window` into `create_prediction_feature_schema(...)`.
- `src/gridiron_edge/models/game_prediction/availability.py` - `_inspect_trained_model()` now validates `metadata.parameters["epa_window"]` as part of `metadata_contract_matches` (missing/malformed makes only that family unavailable, consistent with the existing metadata-preflight contract) and rebuilds a per-family EPA frame at that window before calling `feature_set.feature_fn(...)`, replacing the prior single window=4 frame shared across all five families.
- `src/gridiron_edge/evaluation/backfill.py` - `_walk_forward_one_season()` rebuilds the target season's EPA columns at that iteration's searched `epa_window` (from the freshly trained `meta.parameters`, default 4) before calling `feature_fn(target_df)`, so walk-forward predicts each target season with the same window the retrained model actually used.
- `src/gridiron_edge/evaluation/prediction_input_evidence.py` - Added `epa_window: int` to `PredictionFeatureSchema`, threaded through `prediction_feature_schema_id()`, `create_prediction_feature_schema()`, `_feature_schema_payload()`, and `_validate_feature_schema()` (positive-integer check, included in the hashed schema identity). Bumped `PREDICTION_INPUT_EVIDENCE_SCHEMA_VERSION` from 1 to 2.
- `src/gridiron_edge/evaluation/prediction_input_evidence_store.py` - `_feature_schema()` deserialization requires and reads the new `epa_window` key.
- `src/gridiron_edge/models/game_prediction/prediction_execution.py` - `_validate_statistical_feature_schema()` gained the matching `epa_window` positive-integer check and includes it when recomputing the expected schema identity.
- `src/gridiron_edge/models/game_prediction/_epa_window.py` - Docstring only: notes `_rebuild_features_with_window` is now also imported by prediction, availability, and backfill, not just training.
- `tests/unit/evaluation/test_prediction_input_evidence.py`, `tests/unit/models/game_prediction/test_statistical_prediction_execution.py`, `tests/unit/models/game_prediction/test_weekly_execution.py` - Updated the direct `create_prediction_feature_schema(...)` call sites to pass `epa_window=4`; added a schema-identity test proving two schemas differing only in `epa_window` produce different `schema_id`s.
- `tests/unit/models/game_prediction/test_availability.py` - `_metadata()` fixture now includes `epa_window: 4` by default; extended the stale-metadata parametrization with missing/`None`/bool/string/zero `epa_window` cases; added `test_each_family_is_rebuilt_at_its_own_epa_window` proving two families with different persisted windows each get their own rebuilt frame.
- `tests/unit/models/game_prediction/test_games_model_evidence.py` - Shared `_metadata()` fixture gained an `epa_window` parameter (default 4, preserving prior test behavior via the window-4 no-op fast path); added `epa_window=None/True/0` cases to `TestClassificationFailureOrdering`; added `test_non_default_epa_window_is_rebuilt_before_feature_extraction` to both the classification and regression evidence-execution test classes.
- `tests/unit/evaluation/test_backfill.py` - Added `test_target_season_features_use_the_searched_epa_window` proving the target season's features are rebuilt at the walk-forward iteration's searched window.

Removed:
- None.

##### Tests

Ruff, Pyrefly, and the full non-slow unit suite passed (4,023 tests, up from 4,010; the 13 new tests are listed above). Real-artifact validation (read-only, no `data/` writes): called `inspect_prediction_availability()` against the real current 2026 Week 3 upcoming schedule — all five statistical families and Elo remained available, matching pre-fix behavior. Directly rebuilt the real Week 3 enriched frame at windows 4, 6, and 8 and confirmed `AWAY_OFF_EPA_PER_PLAY`/`HOME_OFF_EPA_PER_PLAY` differ meaningfully across windows (e.g. Atlanta @ Green Bay: -0.232/-0.154 at window 4 vs. -0.109/-0.089 at window 6 vs. -0.105/-0.000 at window 8), confirming each family's own tuned window now reaches feature construction instead of the prior fixed default. No weekly forecast, backfill run, or evidence was generated as part of this unit.

##### Acceptance

Live prediction, availability inspection, and walk-forward backfill all rebuild canonical EPA at the exact artifact-specific window recorded in that model's persisted (or freshly trained, for backfill) metadata, never the default window of 4, unless an artifact's own window is 4. Missing or malformed `epa_window` metadata makes only the affected family unavailable, consistent with the existing metadata-preflight contract. Prediction-input evidence's feature schema now records the window used and its schema version is bumped to 2; no existing persisted evidence was modified, since nothing in the codebase re-reads previously persisted evidence outside the same write-then-verify command. D1 is proven fixed by tests that fail against the prior behavior (always window 4) and pass against the fix, plus real-data verification that per-family EPA values now differ by window.

### Statistical Availability Metadata Preflight Alignment [Completed September 22, 2026]

#### Completed

Aligned weekly statistical-model availability with the persisted metadata
contract already enforced by live prediction execution.

Availability now requires the exact registered feature-set identity, current
integer modeling schema version, and exact ordered feature columns before
declaring a statistical family available.

Stale but well-formed metadata makes only the affected family unavailable.
Malformed model identity, artifact kind, and task remain explicit errors.

#### Goal

Reject stale statistical artifacts during read-only availability inspection
before policy selection and model execution.

Preserve execution as an independent fail-closed validation boundary while
preventing availability from selecting an artifact that execution would reject.

#### Files Added/Removed/Changed

Changed:

- `PLAN.md`
  - Closed the statistical availability metadata preflight alignment unit.
- `src/gridiron_edge/models/game_prediction/availability.py`
  - Added exact feature-set identity and modeling schema version checks to the
    existing ordered feature-column availability contract.
- `tests/unit/models/game_prediction/test_availability.py`
  - Updated valid metadata fixtures and added missing, malformed, stale, and
    family-isolation coverage.

Removed:

- None.

#### Tests

Validation passed:

- focused Ruff checks;
- repository-wide Ruff checks;
- Pyrefly;
- Python compilation;
- 84 focused availability, policy, weekly-execution, and weekly-CLI tests;
- the full non-slow unit suite;
- `git diff --check`.

Focused coverage proves:

- valid current metadata remains available;
- missing or incorrect feature-set identity is unavailable;
- missing, null, Boolean, string, and stale modeling schema versions are
  unavailable;
- exact ordered feature columns remain required;
- stale metadata affects only the exact family;
- malformed identity, kind, and task remain explicit errors;
- availability does not deserialize estimators or scalers;
- execution remains an independent unchanged fail-closed boundary.

Protected real-artifact validation inspected the current 2026 Week 2 schedule
through the public availability boundary. Elo and all five statistical families
were available:

- `win_prob / logistic`;
- `win_prob / random_forest`;
- `win_prob / xgboost`;
- `total / random_forest`;
- `total / xgboost`.

A before-and-after SHA-256 comparison of every model, metadata, and scaler
artifact under `data/models/` produced no differences.

Independent read-only review confirmed all requested correctness, ordering,
scope, and non-mutation requirements and approved the unit for documentation
closure.

#### Acceptance

Availability requires exact persisted feature-set identity, current integer
modeling schema version, and exact ordered feature columns before declaring a
statistical family available.

Missing, malformed, or stale model-contract metadata makes only the affected
family unavailable before policy execution.

Malformed artifact identity, kind, and task remain explicit errors.

Availability rejects stale metadata before runtime feature construction and
does not deserialize estimators or scalers.

Live prediction execution retains its independent strict metadata validation
and remains fail-closed.

Current real statistical artifacts pass the tightened preflight without
regeneration or modification.

No model artifact, forecast, prediction-input evidence, weekly product, API,
frontend, or persisted schema contract changed.


### Development Forecast Role Foundation [Completed September 22, 2026]

#### Completed

Added an explicit development forecast role for retrospectively generated
canonical weekly fixtures without representing those fixtures as pre-kickoff
live issuance.

Development events can carry exact immutable prediction-input evidence,
participate in one role-coherent selected weekly product, and undergo exact
selected-event postgame closeout.

Live-only recommendation qualification and production-chain proof remain
isolated from development evidence. Spread production provenance now explicitly
inherits the live-role requirement from its source Win forecast.

#### Goal

Add a truthful forecast role for retrospective canonical development fixtures
while preserving the existing meanings and safety boundaries of live and
backfilled forecasts.

Keep the normal weekly prediction command live-only, provide a dedicated
development execution boundary, preserve existing persisted schemas and
evidence, and modify no operational artifacts.

#### Files Added/Removed/Changed

Changed:

- `PLAN.md`
  - Closed the development forecast role foundation after independent review.
- `src/gridiron_edge/evaluation/forecast_contracts.py`
  - Added `ForecastRole.DEVELOPMENT` and bounded selected-weekly role policies.
- `src/gridiron_edge/evaluation/forecast_events.py`
  - Updated role documentation for all supported forecast roles.
- `src/gridiron_edge/evaluation/prediction_input_evidence.py`
  - Authenticated exact live or development families while rejecting
    backfilled and mixed-role evidence.
- `src/gridiron_edge/evaluation/prediction_input_evidence_store.py`
  - Updated selected-weekly evidence documentation without changing storage.
- `src/gridiron_edge/evaluation/live_forecast_closeout.py`
  - Matched events against exact persisted product roles and exposed selected
    Win and Total roles in closeout results.
- `src/gridiron_edge/models/game_prediction/product_validation.py`
  - Allowed live or development products while enforcing one role across every
    available selected component.
- `src/gridiron_edge/models/game_prediction/weekly_execution.py`
  - Added distinct live and development wrappers over one role-aware execution
    boundary.
- `src/gridiron_edge/market/production_chain_preflight.py`
  - Required derived Spread production provenance to inherit the source Win
    live role.
- focused unit and integration tests
  - Added role, evidence, product, closeout, execution, qualification,
    production-proof, and disposition-preservation coverage.

Removed:

- None.

Operational artifacts changed:

- None.

#### Tests

Validation passed:

- repository-wide Ruff checks;
- Pyrefly;
- the full non-slow unit suite;
- focused selected-event closeout integration tests;
- focused production-chain repository integration tests;
- `git diff --check`.

Focused tests prove:

- development is distinct from live and backfilled;
- live and development events authenticate exact prediction-input evidence;
- backfilled and mixed-role evidence are rejected;
- weekly products enforce one role across all available games and families;
- development events close against completed outcomes using exact persisted
  product roles;
- live and development execution wrappers produce one coherent role across Win
  and Total;
- the weekly CLI continues to invoke only the live wrapper;
- development provenance fails recommendation qualification;
- Moneyline, Spread, and Total production proof remain live-only;
- development events cannot authenticate the original live Week 2 defect scope.

Protected read-only validation proved:

- 2026 Week 3 Logistic Win and Random Forest Total schema-1 evidence each
  authenticated against 16 live events;
- the existing Week 2 disposition authenticated successfully;
- the disposition applies to the selected affected Week 2 product and does not
  apply to the selected Week 3 product;
- before-and-after SHA-256 inventories of predictions, weekly products, and
  prediction-input evidence were byte-identical.

Independent read-only review found no blocking defects and approved closure
after reverting three unrelated type-coercion changes from the production-chain
file.

#### Acceptance

`ForecastRole.DEVELOPMENT` has a durable meaning distinct from live and
backfilled.

Live and development events can carry exact immutable prediction-input
evidence. Backfilled and mixed-role evidence cannot authenticate selected-weekly
claims.

Weekly products may use live or development events but require one coherent
role across every available selected component and game.

Selected-event closeout matches the exact role persisted by the weekly product
and reports Win and Total roles explicitly.

The existing weekly command remains live-only. Retrospective fixture generation
has a dedicated development execution boundary with no current production
caller.

Qualification, candidate, recommendation, and production-proof boundaries
remain live-only. Spread inherits that eligibility from its exact source Win
forecast.

The original Week 2 defect disposition and existing Week 3 schema-1 evidence
remain valid.

No persisted schema or operational artifact changed.
