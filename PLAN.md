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

#### Unit 4: Stale-artifact disposition and truncation-exposure verification [Completed September 25, 2026]

##### Completed

Deleted 24 pre-fix debugging artifacts (6 evaluation plots, 11 D4 training
logs, 7 legacy full-retrain reports) that reflected the D1–D4 pipeline bugs
U2/U3 just fixed and had no code reader; keeping them as a `full-retrain`
drift baseline would have corrupted the next comparison. Added a
repository-owned `gridiron evaluate prune-champions` command — rather than
hand-editing `champions.json` or invoking the existing full
`--write-manifest` re-promotion path, which re-ranks every family from
scratch and is Tier 2 #7's job (U9), not available yet — and used it to
remove the 5 champion-manifest entries with no backing trained artifact
(D6), preserving `win_prob` and `total` byte-for-byte. Verified the Week 1
percentile-ranking artifact was not computed from the D38 truncation reset:
its 32 distinct, non-tied `rating_pct` values are inconsistent with D38's
two-value (1510/1490) reset, which would force massive ties.

##### Goal

Apply dispositions to the stale-artifact inventory identified by the Tier 1
#3 audit before Tier 2 evaluation and regeneration work (U7 onward) builds on
top of it: remove pre-fix debugging output that reflects the D1–D4 pipeline
bugs U2/U3 just fixed, remove champion-manifest entries with no backing
artifact (D6), and verify the Week 1 percentile-ranking artifact was not
computed from the D38 history-truncation reset.

##### Files Added/Removed/Changed

Added:
- None.

Changed:
- `src/gridiron_edge/evaluation/champion_resolver.py` - Added `ChampionPruneResult` and `prune_orphaned_champions()`, which checks every manifest entry's `(model_name, model_type)` against `ArtifactStore.is_trained()` and rewrites the manifest via the existing `write_manifest()` preservation path, keeping only artifact-backed entries.
- `src/gridiron_edge/cli/evaluate.py` - Added the `evaluate prune-champions` command, reporting each removed entry or a no-op message.
- `tests/unit/evaluation/test_champion_resolver.py` - Added `TestPruneOrphanedChampions`: removes orphaned entries, preserves kept entries byte-for-byte including their own `source_run_id`, no-ops when nothing is orphaned, raises `ChampionNotFoundError` with no manifest, and treats a manifest `model_type` that doesn't match the actual trained artifact's type as orphaned.
- `tests/unit/cli/test_evaluate.py` - Added `TestPruneChampionsCommand`: covers the removed-entries and no-op console output paths through `CliRunner`.
- `HANDOFF.md` - Noted that `gridiron evaluate prune-champions` also writes the champion manifest, alongside the existing promotion workflows.
- `ROADMAP.md` - Marked D6's orphaned-manifest-entry portion resolved by this unit; left the calibration/manifest currency portion open for U9.
- `PLAN.md` - Closed out Foundation Completion Track A, Unit 4.

Removed (real `data/` artifacts, not version-controlled, no commit entry):
- `data/output/evaluation/win_prob_logistic/*.png` (6 files) - pre-fix evaluation plots.
- `data/output/d4_training_logs/*` (11 files, dated 2026-06-18) - pre-fix training/backfill logs.
- `data/output/reports/full-retrain-*.md` (7 files, dated 2026-06-22 through 2026-08-05) - pre-fix legacy full-retrain reports.
- 5 champion-manifest entries (`qb_pass_yards`, `qb_rush_yards`, `rb_rush_yards`, `te_rec_yards`, `wr_rec_yards`) removed from `data/output/champions/champions.json` via `gridiron evaluate prune-champions`.

##### Tests

Ruff, Pyrefly, and the full non-slow unit suite passed (4,030 tests, up from 4,023; the 9 new tests are listed above). Real-artifact validation: ran `gridiron evaluate prune-champions` against the real repository — it removed exactly the 5 entries the audit predicted and left `win_prob`/`total` byte-identical except `updated_at` (verified by direct read of the file before and after). Deleted the real 24 stale files and confirmed `find` over the three source directories returns nothing; confirmed `_stage_baseline_comparison` (the only reader of the reports directory) still soft-fails with `"no full-retrain report directory found"` rather than raising. Re-ran the full non-slow unit suite after the real `data/` deletions to confirm nothing depends on the removed files (4,030 passed, unchanged).

##### Acceptance

The 24 identified pre-fix debugging files no longer exist under `data/output/`. `champions.json` contains only `win_prob` and `total`, each byte-identical to its pre-prune entry except `updated_at`. The Week 1 percentile artifact is confirmed unaffected by D38 and was left unmodified. `gridiron evaluate prune-champions` exists as a repository-owned operation so this disposition never requires hand-editing the manifest again.

#### Unit 5: Repository-owned development-role forecast generation command [Completed September 25, 2026]

##### Completed

Added `gridiron generate-development-forecast`, a repository-owned command
that generates and persists one `development`-role forecast run (immutable
forecast events plus per-family prediction-input evidence) for any
already-retained season and week, including weeks already played, using only
data already on disk. It calls the existing
`execute_development_weekly_prediction_policy` wrapper rather than
duplicating role-aware execution logic, and does not compose a weekly
product, select a current product, or verify readiness — those remain
Unit 6's job when it regenerates the disposition-affected 2026 Week 2 state.

##### Goal

Add a repository-owned CLI command that generates and persists one
development-role forecast run (immutable forecast events plus per-family
prediction-input evidence) for an arbitrary already-retained season/week,
using only data already on disk. No external source is fetched. The command
calls the existing `execute_development_weekly_prediction_policy` wrapper
(added by the Development Forecast Role Foundation unit) rather than
duplicating its role-aware execution logic. Weekly-product composition,
current selection, and readiness verification remain out of scope for this
unit: Unit 6 performs those steps when it regenerates the disposition-affected
2026 Week 2 development state through this command.

##### Design decisions

- The live `weekly-predict` command sources its schedule from
  `data/cleaned/NFL_upcoming_schedule_rich.parquet`, which is fetch-derived
  and only ever holds unplayed games for the current season (verified: it
  currently holds only 2026-2027 weeks 3-18; weeks 1-2 have already dropped
  out). That file cannot supply an already-elapsed week, so the new command
  instead adapts the retained, append-only `data/cleaned/NFL_wk_by_wk_cleaned.csv`
  history (`games`) into the same 9-column rich-schedule shape
  (`season`/`week`/`game_id`/`game_day_of_week`/`game_date`/`game_time`/
  `away_team`/`home_team`/`neutral_site`) that `_scope_schedule` and
  `_build_elo_schedule` already require — verified by direct column
  comparison against `RICH_UPCOMING_COLUMNS` and `_RICH_SCHEDULE_COLUMNS`.
  This is the literal meaning of "scoped from retained history so no fetch
  is needed."
- No leakage risk from reusing "current" Elo/EPA state for an
  already-elapsed week: `NFL_Team_Elo.csv` stores one row per
  `(team, season, week)` representing the pre-game rating for that exact
  week, and the merge in `_merge_elo_predictions` joins on
  `(team, season, week)`, not on the newest row. Statistical feature
  construction (`run_features`) is likewise keyed by the schedule's own
  `WEEK_NUM`/`YEAR`, not wall-clock time. Requesting a past week therefore
  reconstructs that week's true pre-game state rather than leaking later
  weeks' results, verified by reading `NFL_Team_Elo.csv` directly (weeks
  1-3 present as distinct per-team rows for 2026-2027).
- `capture_prediction_source_artifacts`'s default `PREDICTION_SOURCE_SPECS`
  already lists `NFL_wk_by_wk_cleaned.csv` as a required bounded source
  (used by both roles today), so no change to the source-provenance
  contract is needed.
- Extract the three role-agnostic publication helpers currently private to
  `cli/weekly_predict.py` (`_require_execution_provenance`,
  `_publish_binary_snapshots`, `_publish_and_reload_evidence`) into a new
  shared module so the development command does not duplicate immutable
  publish/reload/authenticate logic. `weekly_predict.py` re-imports them
  under the same names so its existing tests (which patch
  `gridiron_edge.cli.weekly_predict._publish_binary_snapshots`, etc.) keep
  passing unmodified.
- New command name: `gridiron generate-development-forecast --season
  --week`. Deliberately not a flag on `weekly-predict`, preserving the
  Development Forecast Role Foundation unit's requirement that the normal
  weekly command stay live-only with a dedicated development boundary.
- No `--skip`/`--only`/composite-stage machinery: this is a single-purpose
  command (matching the `verify-week`/`post-week` pattern), not a multi-stage
  pipeline like `weekly-predict`.

##### Files Added/Removed/Changed

Added:
- `src/gridiron_edge/cli/_prediction_publication.py` - Extracted the three
  role-agnostic publication helpers (`_require_execution_provenance`,
  `_publish_binary_snapshots`, `_publish_and_reload_evidence`) out of
  `cli/weekly_predict.py` so both the live and development commands share one
  immutable publish/reload/authenticate path.
- `src/gridiron_edge/cli/development_forecast.py` - New
  `generate-development-forecast` command. `build_retained_history_schedule()`
  adapts the retained `games` dataset (`NFL_wk_by_wk_cleaned.csv`) into the
  9-column rich-schedule shape `_scope_schedule`/`_build_elo_schedule`
  require; `_generate_development_forecast()` resolves source revision and
  artifacts, builds the schedule, calls
  `execute_development_weekly_prediction_policy`, and publishes forecast
  events and evidence through the shared helpers above.
- `tests/unit/cli/test_development_forecast.py` - Column-shape coverage for
  the schedule adapter (required-column mapping, missing-column rejection,
  score-column exclusion) and fail-closed publication-order coverage
  mirroring `test_weekly_predict_publication.py` (revision, capture,
  schedule, run-id, execute, recapture, snapshots, evidence, events), plus
  CLI success/failure exit-code coverage.

Changed:
- `src/gridiron_edge/cli/weekly_predict.py` - Removed the three publication
  helpers now imported from `cli/_prediction_publication.py`; behavior
  unchanged. Existing tests continue to patch
  `gridiron_edge.cli.weekly_predict._publish_binary_snapshots` etc.
  unmodified, since the names are still bound in that module's namespace via
  the new import.
- `src/gridiron_edge/cli/main.py` - Registered
  `generate-development-forecast`.
- `HANDOFF.md` - Added a "Development Forecast Generation" workflow section
  documenting the command and the retained-history/no-leakage design
  rationale.
- `ROADMAP.md` - Marked Tier 2 #4's U5 portion complete; U6 (archive,
  regenerate Week 2, select, verify) remains open.
- `PLAN.md` - This unit record.

Removed:
- None.

##### Tests

Ruff, Pyrefly, and the full non-slow unit suite passed (4,040 tests, up from
4,030; the 10 new tests are listed above).

Real-artifact validation (read-only; no persisted artifact written): loaded
the real retained `games` dataset and confirmed
`build_retained_history_schedule()` correctly recovers both already-elapsed
2026-2027 weeks 1 and 2 (16 games each, including a correctly-flagged neutral-
site game) that no longer exist in the fetch-derived upcoming-schedule
snapshot. Called `execute_development_weekly_prediction_policy` directly
against this real retained-history schedule for both weeks: policy correctly
selected Logistic Win and Random Forest Total for each; all 32 events per
week carried `role=development`; per-team Elo ratings and win probabilities
differed correctly between week 1 and week 2 (matching `NFL_Team_Elo.csv`'s
one-row-per-team-per-season-per-week structure), confirming no leakage from
reusing "current" Elo/feature state for an already-elapsed week. This
verification used a synthetic `SourceRevision` and did not call
`resolve_clean_source_revision` or publish evidence, since this unit's own
changes leave the tracked worktree intentionally dirty until commit;
publish-path correctness (binary snapshot and evidence persistence, reload,
and re-authentication) is covered by the fail-closed unit tests instead, and
a first genuine on-disk development-role run is Unit 6's job when it
regenerates Week 2.

##### Acceptance

`gridiron generate-development-forecast` exists as a repository-owned command
that generates one development-role forecast run from retained history
alone, for any already-retained season and week including ones already
played, with no network fetch. It does not compose a weekly product, select
a current product, or verify readiness. The live `weekly-predict` command's
behavior and tests are unchanged except for the non-behavioral relocation of
shared publication helpers into `cli/_prediction_publication.py`.

#### Unit 6: Archive-by-documentation and regenerated Week 2 canonical state [Completed September 25, 2026]

##### Completed

Replaced the untracked ad hoc 2026 Week 2 `development`-role selection (D5)
with a freshly generated, repository-owned replacement produced with the
D1–D4 pipeline fixes (Units 2–3) applied. Recorded the archive boundary as
`DECISIONS.md` D45: weekly products and forecast events are immutable,
append/index-only stores with no delete path, and existing operational
guidance already forbids editing or deleting disposition-governed evidence,
so "archive" is this documented decision plus the store's existing
`select_current_weekly_product()` reselection — not a new store mechanism or
an "archived" flag. The two disposition-governed live Week 2 products and the
retired ad hoc product are all left physically unchanged. The replacement is
driven by a new repository-owned command, `gridiron regenerate-development-week`,
which generates, composes, selects, and reports readiness atomically in one
process — composition needs the full in-memory `PredictionPolicy` object
(availability, status, rationale, provenance), never persisted, so a separate
`--run-id`-based compose command would have had to fabricate policy metadata
that was never real. `generate-development-forecast` (Unit 5) keeps its exact
shipped contract unchanged (generation only, never touching `current.json`).
The live composition stage's core was factored into a shared, schedule-agnostic
helper (`cli/_weekly_product_composition.py`, the same extraction pattern
Unit 5 used for `_prediction_publication.py`); `weekly_predict.py`'s live,
upcoming-week-only contract is otherwise unchanged.

While executing this unit, found and fixed a second, narrower defect: weekly-
product composition and readiness verification both sourced the schedule
exclusively from the fetch-derived upcoming-schedule snapshot, which had
already dropped Week 2 (and any other elapsed week) by the time Unit 5
shipped — this made `verify-week` report a false `missing_schedule` blocker
regardless of the selected product's actual state. Both paths now fall back
to the retained-history schedule adaptation Unit 5 built, for any already-
elapsed week. A third, more serious defect surfaced only when the real
regeneration was first run: `resolve_forecast_candidates` never accounted for
`development`-role events at all (only `live`/`backfilled`), crashing with an
unhandled `IndexError` instead of resolving — fixed in a separate commit
since it is a distinct correctness defect in shared forecast-selection code,
not specific to this unit's own new files.

##### Goal

Execute ROADMAP.md's Canonical Artifact Decision for 2026 Week 2: archive the
disposition-affected live runs and the ad hoc development run through a
`DECISIONS.md`-recorded boundary, regenerate Week 2 through the corrected
pipeline, select it, and verify readiness, the API, and the frontend.

##### Files Added/Removed/Changed

Added:
- `src/gridiron_edge/cli/_weekly_product_composition.py` - Shared
  `compose_and_select_weekly_product()`, schedule/role-agnostic composition
  and explicit selection, factored out of `weekly_predict.py`.
- `tests/unit/cli/test_weekly_product_composition.py` - Empty-scope failure,
  a full composition/selection pass against a caller-supplied schedule, and
  the unavailable-family skip-resolution path.
- `tests/unit/cli/test_regenerate_development_week.py` - Success wiring
  (schedule/events/policy passed through unchanged), generation-failure, and
  composition-failure exit-code coverage.

Changed:
- `src/gridiron_edge/cli/weekly_predict.py` - `_stage_compose_weekly_product`
  is now a thin wrapper calling the shared helper; behavior and output are
  unchanged (proven by the existing dedicated stage tests, retargeted to the
  helper's new import locations).
- `src/gridiron_edge/cli/development_forecast.py` - `_generate_development_forecast`
  now returns `DevelopmentForecastResult` (run ID, generated-at, schedule, the
  full `WeeklyPredictionExecution`, artifacts) instead of a bare tuple, so a
  caller in the same process can reuse the real policy and events without
  reloading or reconstructing them; `generate_development_forecast_cmd`'s own
  output and behavior are unchanged. Added `regenerate_development_week_cmd`.
- `src/gridiron_edge/cli/verify_week.py` - `_load_schedule` now takes
  `season`/`week` and falls back to the retained-history schedule adaptation
  when the fetch-derived upcoming schedule has no rows for that scope.
- `src/gridiron_edge/cli/main.py` - Registered `regenerate-development-week`.
- `src/gridiron_edge/evaluation/forecast_selection.py` - `resolve_forecast_candidates`
  now prefers live, then development, then backfilled events (previously
  live-then-backfilled only; a development-only match crashed).
- `tests/unit/evaluation/test_forecast_selection.py` - Added development-role
  candidate-resolution coverage.
- `tests/unit/cli/test_verify_week.py` - Added
  `TestLoadScheduleRetainedHistoryFallback` (falls back correctly for an
  elapsed week; prefers the upcoming schedule when present; fails closed to
  the upcoming schedule if the fallback itself is unavailable).
- `tests/unit/cli/test_weekly_predict_product_stage.py` - Retargeted mock
  patch paths to the extracted helper's module; assertions unchanged.
- `tests/unit/cli/test_development_forecast.py` - Updated the two tests
  exercising `_generate_development_forecast`'s return shape for
  `DevelopmentForecastResult`.
- `DECISIONS.md` - Added D45.
- `CHANGELOG.md` - Added the 2026-09-25 entry.
- `HANDOFF.md` - Corrected the Week 2 known-defect section to describe the
  corrected current selection instead of the retired ad hoc one; documented
  `regenerate-development-week` and the schedule-fallback fix.
- `ROADMAP.md` - Marked Tier 1 #2 and Tier 2 #4's U6 portion complete; updated
  the Canonical Artifact Decision and Known Limitations sections to state the
  decision is executed, not just recorded.

Removed:
- None.

##### Tests

Ruff, Pyrefly, and the full non-slow unit suite passed (4,049 tests, up from
4,040; the 9 new tests are listed above). A follow-up commit fixing the
`resolve_forecast_candidates` defect above added 3 more tests and brought the
suite to 4,052.

Real-artifact validation (writes to `data/`, via the new owning command): ran
`gridiron regenerate-development-week --season 2026-2027 --week 2` against
the real repository once this unit's own commits landed (satisfying the
clean-tracked-worktree requirement, D42). It generated development-role run
`a9213428-5cca-43cf-bcae-d73f1ff155e8`, composed and selected
`weekly_2026_2027_wk02_a9213428-5cca-43cf-bcae-d73f1ff155e8` as the current
Week 2 product, and reported `prediction_ready: True`
(`market_ready: False`, blockers `missing_market_data`/
`market_scope_mismatch` — expected and independent of prediction readiness,
since Week 2 has already been played and the current market snapshot holds
only the current week's odds). Independently confirmed via
`gridiron verify-week --season 2026-2027 --week 2`: 16 scheduled games, 16
selected Win predictions, 16 spread values, 16 Total predictions, 16
projected scores, 16 complete-provenance rows.

Before/after SHA-256 comparison confirmed the two disposition-governed live
Week 2 products, the disposition artifact, and the retired ad hoc product are
all byte-identical (only `index.json`, additive, and `current.json`'s Week 2
entry changed). Started the real API server and confirmed
`GET /games?season=2026-2027&week=2` returns 200 with all 16 games'
Win/Spread/Total marked `available`, `role: "development"`, and the new
`run_id` — the frontend reads this same generated schema with no gating on
`role` anywhere in `frontend/src` (confirmed by source search), so no
frontend code change or separate browser check was needed for this
data-only unit.

##### Acceptance

`DECISIONS.md` D45 records the archive boundary. The two disposition-governed
live Week 2 products and their disposition are physically unchanged. The ad
hoc development selection is retired via explicit reselection, replaced by a
freshly generated, repository-owned development run produced with the
corrected pipeline. `gridiron verify-week` correctly evaluates readiness for
any already-retained week. API and frontend serialization is verified against
the corrected selection.

#### Unit 7: Comprehensive game-model evaluation metrics, cross-family report, and legacy-archive retirement [Completed September 25, 2026]

##### Completed

Implemented ROADMAP.md Tier 2 #5/#6's U7: added the model-quality metrics
`evaluation/metrics.py` was missing (Win calibration slope/intercept,
sharpness, season stability; Total median absolute error, interval
coverage, environment slices); added one new immutable, run-bound
`GameModelEvaluationReport` and CLI command (`gridiron evaluate
model-report`) covering all six game-model families on a verified common
game set; and retired every code consumer of the legacy prediction archive
(`evaluation/archive.py`), which is now deleted. While migrating consumers,
found and fixed a second defect: `champion.py::promote_champions()` (the
shared selector behind the manual `evaluate select-model --write-manifest`
and `props champion --write-manifest` flags) called a different, older,
archive-backed classification selector than `full-retrain` actually uses -
the two manifest-writing surfaces could disagree on which model wins. Fixed
by switching it to the same run-based selector, resolving each pair's
latest backfill run itself.

Real-artifact verification against the actual repository found that the six
families' *current* on-disk backfill runs do not share a common game set
(Elo's two runs cover 7,276 games; the other five families' most recent
runs share a different, identical 6,498-game set) - confirming this
report's fail-closed common-game-set guard is necessary, and that producing
a genuine six-family canonical report requires U8's corrected,
regenerated, aligned backfills, not this unit's job.

##### Goal

Add the missing model-quality metrics, add one cross-family immutable
evaluation report and CLI, and retire every consumer of the legacy
prediction archive in favor of the forecast-store-backed path - building
and proving the capability against whatever backfill runs exist today,
without regenerating backfills (U8) or touching calibration/the champion
manifest (U9).

##### Files Added/Removed/Changed

Added:
- `src/gridiron_edge/evaluation/game_model_evaluation_report.py` - Frozen
  `GameModelEvaluationReport`/`WinFamilyMetrics`/`TotalFamilyMetrics`
  contracts, content-addressed frame references, and canonical-identity
  validation, sibling to (not an extension of) `historical_backtest_report.py`.
- `src/gridiron_edge/evaluation/game_model_evaluation_report_builder.py` -
  `build_and_write_game_model_evaluation_report()`: evaluates all six
  families from caller-supplied exact run IDs, asserts common `GAME_ID`
  coverage (fails closed on mismatch), computes per-family metrics and a
  combined environment-slice table (dome/outdoor, cold, windy, joined from
  already-present `modeling_file.parquet` columns), and persists with exact
  replay verification.
- `src/gridiron_edge/evaluation/game_model_evaluation_report_store.py` /
  `_selection.py` / `_loader.py` - Immutable JSON+Parquet persistence,
  explicit current-selection, and strict deserialization, mirroring
  `historical_backtest_report_store.py`/`_selection.py`/`_loader.py`'s
  conventions.
- `tests/unit/evaluation/test_game_model_evaluation_report.py` /
  `_builder.py` / `_store.py` / `_selection.py` / `_loader.py` - Contract
  validation/tamper detection, full six-family build-and-replay plus the
  common-game-set-mismatch fail-closed path (proven against real backfill
  data), round-trip persistence, and selection/loading coverage.

Changed:
- `src/gridiron_edge/evaluation/metrics.py` - Added
  `calibration_slope_intercept` (unregularized logistic fit on
  `logit(clip(p))`, ties excluded), `sharpness`, `season_stability`,
  `median_absolute_error`, `interval_coverage` (residual-std diagnostic),
  `environment_slice_metrics`, and `build_forecast_run_total_evaluation_df`
  (the Total-task counterpart to the existing Win-only
  `build_forecast_run_evaluation_df`). Removed `build_evaluation_df()` and
  the `evaluation.archive` import; module docstring updated.
- `src/gridiron_edge/evaluation/select.py` - Added
  `registered_game_model_pairs`, `latest_backfilled_run_id`,
  `collect_latest_forecast_run_metrics`, `build_latest_run_evaluation_df`.
  Removed the legacy `collect_model_metrics`; `compute_report_data` now
  takes an explicit `run_id` instead of an archive-wide `season` filter
  (season is now a post-filter on the run's own games).
- `src/gridiron_edge/evaluation/champion.py` - Deleted the archive-backed
  `select_game_classification_champions`; `promote_champions()` now
  resolves each pair's latest backfill run and calls
  `select_game_classification_champions_from_runs`, the same selector
  `full-retrain` uses.
- `src/gridiron_edge/evaluation/diagnostics.py` - Docstrings updated to
  name the current builder function.
- `src/gridiron_edge/cli/evaluate.py` - `summary`/`calibration`/
  `diagnostics`/`select-model`/`report` migrated to each family's latest
  backfill run. Added `evaluate model-report` (six required exact-run-id
  options, `--select/--no-select`).
- `src/gridiron_edge/api/loaders.py` - `load_evaluation_df` now calls
  `build_latest_run_evaluation_df` instead of the legacy archive.
- `src/gridiron_edge/api/routes/model.py`,
  `src/gridiron_edge/api/schemas/model_performance.py`,
  `src/gridiron_edge/api/serializers/model_performance.py` - Comment/
  docstring updates only; `/model/performance`'s response contract, query
  parameters, and behavior are unchanged (see `DECISIONS.md` D46 for why a
  persisted-report redesign was considered and deferred).
- `api-schema.json`, `frontend/src/api/schema.ts` - Regenerated via
  `gridiron api export-schema` + `pnpm gen:api` (one-line description-text
  diff from the schema docstring change above; the generated TS client is
  byte-identical).
- `tests/unit/evaluation/test_metrics.py`, `test_select.py`,
  `test_champion.py`, `test_diagnostics.py` - Retargeted to the new
  functions/signatures; removed assertions about the deleted legacy path.
- `tests/unit/cli/test_evaluate.py`,
  `tests/unit/cli/test_props_champion_write_manifest.py`,
  `tests/unit/api/test_loaders.py` - Retargeted champion-selection and
  evaluation-loader mocks to the run-based functions; added
  `TestModelReportCommand` and `TestLoadEvaluationDf`.
- `tests/fixtures/helpers.py`, `tests/fixtures/repos.py` - Removed
  `assert_archive_schema_valid`/`with_predictions_archive`, dead after the
  archive's removal.
- `tests/integration/test_edges_cli.py` - Removed a dead, unused patch-path
  constant referencing the deleted archive module.
- `DECISIONS.md` - Added D46 (legacy-archive retirement and the two
  run-selection policies) and D47 (interval-coverage/calibration-slope
  methodology).
- `CHANGELOG.md`, `ROADMAP.md` - Recorded the shipped behavior; marked
  Tier 2 #5/#6's U7 portion complete.
- `HANDOFF.md` - Added a "Game-model evaluation" subsection describing the
  retired archive, the two run-selection policies, and the new report.

Removed:
- `src/gridiron_edge/evaluation/archive.py` - The append-only, overwriteable
  prediction log. Confirmed dead: `gridiron output predictions` never
  actually wrote to it despite the module's own docstring; every reader
  migrated to the forecast-store-backed path.
- `tests/unit/evaluation/test_archive.py`, `test_archive_schema.py`,
  `tests/integration/test_archive_roundtrip.py` - Dedicated coverage for
  the deleted module.

##### Tests

Ruff, Pyrefly, and the full non-slow unit suite passed (4,069 tests, net
+17 over the prior 4,052 after new coverage and deleted legacy tests).
`cd frontend && pnpm lint && pnpm build && pnpm test:run` passed (511
tests) after regenerating the client from the re-exported schema.

Real-artifact validation (read-only against the real repository; no `data/`
writes beyond ones this unit's own owning command performed):
- Ran the new report builder against the real repository's existing
  backfill runs: five families (Total RF/XGB, Win Logistic/RF/XGB) share an
  identical real 6,498-game set; Elo's two real runs cover a different,
  larger 7,276-game set. The builder's common-game-set check correctly
  raised, listing the exact mismatched games.
- Computed per-family metrics directly against the five aligned real runs:
  Logistic Win brier=0.223, ECE=0.018, AUC=0.677, calibration slope=0.92;
  Random Forest/XGBoost Total MAE≈10.8/10.9, 90%-nominal actual
  coverage≈0.90/0.90 - all sane, none NaN.
  Environment-slice metrics (dome/cold/windy) produced sensible per-slice
  breakdowns joined from the real `modeling_file.parquet`.
- Ran `gridiron evaluate model-report` against the real mismatched run IDs
  and confirmed the same fail-closed behavior end-to-end through the CLI.
- Ran `gridiron evaluate select-model`, `summary`, `calibration`, and
  `report` against the real repository (no mocking): correct per-family
  metrics, calibration tables, season-stability trend flags, and top-misses
  output, all sourced from real backfill runs.
- Started the real API server and called `/model/performance`; confirmed
  `roc_auc: null` (present on both old and new code paths against the real
  archive/backfill data - a pre-existing tie-handling gap in `roc_auc()`,
  unrelated to this unit) and otherwise sane `model_quality` values.
- `git diff --stat` confirmed `frontend/src/api/schema.ts` is byte-identical
  after `pnpm gen:api`, since only a schema description string changed.

##### Acceptance

`evaluation/metrics.py` computes calibration slope/intercept, sharpness,
and season stability for Win, and median absolute error, interval
coverage, and environment slices for Total, proven correct on constructed
fixtures and sane on real data. One new immutable, run-bound,
schema-versioned `GameModelEvaluationReport` exists covering all six
families evaluated on a verified common game set, with a CLI command that
fails closed on a real game-set mismatch. Every code consumer of the legacy
prediction archive is migrated to the forecast-store-backed path; the
archive module and its dedicated tests are removed. `promote_champions()`'s
classification selection now agrees with `full-retrain`'s. Champion
selection's ranking, gates, and persisted-metadata inputs are otherwise
unchanged. `/model/performance`'s contract, query parameters, and frontend
consumers are unchanged. A canonical six-family report over the corrected,
regenerated backfills remains U8's job.

#### Unit 8: Restored fixed-hyperparameter walk-forward retraining and the canonical six-family evaluation report [Completed September 25, 2026]

##### Completed

Restored `evaluation/backfill.py` walk-forward retraining to `DECISIONS.md`
D1's already-decided fixed-hyperparameter contract (each target season now
retrains with the currently-deployed champion's own persisted hyperparameters
and `epa_window`, instead of a fresh randomized HP search every season - a
regression from D1, not a new design choice), added current-model season-bound
export alignment for Elo, closed the current season's weather-data gap by
registering the previously-undiscoverable `gridiron ingest weather-backfill`
command, and regenerated all six game-model families' backfills on a verified
common 6,232-game set (seasons 2003-2004 through 2025-2026) to build the
canonical `GameModelEvaluationReport` U7 built the capability for.

##### Goal

Regenerate walk-forward and current-model backfills for all six game-model
families on a verified common game set, using the D1-D4 corrected pipeline,
produce the canonical `GameModelEvaluationReport` U7 built the capability
for, and resolve D7 (weather train/serve skew) for evaluation purposes.

##### Files Added/Removed/Changed

Added:
- `tests/unit/cli/test_ingest_weather_cli.py` - Coverage for `gridiron ingest
  weather` and the newly-registered `gridiron ingest weather-backfill`
  command wiring.

Changed:
- `src/gridiron_edge/models/game_prediction/base.py` - Added
  `GamesTrainer._fit_with_fixed_hyperparameters()` and a `fixed_hyperparameters`
  parameter on `train()`: fits one model directly from given hyperparameters
  and `epa_window`, with no per-call HP search loop.
- `src/gridiron_edge/evaluation/backfill.py` - Added
  `_load_champion_hyperparameters()` (loads the currently-deployed champion's
  persisted hyperparameters, stripping bookkeeping keys); walk-forward now
  calls `trainer.train(..., fixed_hyperparameters=...)` instead of
  `min_cv_train_rows=`. Current-model mode gained `start_season`/`end_season`
  post-filtering of exported predictions, with the underlying simulation
  always run over full history. `_validate_backfill_request` no longer
  rejects season bounds for current-model mode. Removed the now-dead
  `_WALK_FORWARD_MIN_CV_TRAIN_ROWS` constant and its stale contract comment.
- `src/gridiron_edge/cli/ingest.py` - Added the `weather-backfill` command,
  wiring the previously-unregistered `backfill_weather()` function.
- `src/gridiron_edge/ingest/weather/__init__.py` - Exported `backfill_weather`.
- `src/gridiron_edge/ingest/weather/backfill.py` - Corrected the module
  docstring's stale `--all-years` CLI example (that flag never existed).
- `tests/unit/models/test_games_trainer.py` - Added
  `TestFitWithFixedHyperparameters`: proves a single direct fit at the given
  hyperparameters/window for both tasks, and that `train(fixed_hyperparameters=...)`
  never calls `_run_hp_search`.
- `tests/unit/evaluation/test_backfill.py` - Added `TestLoadChampionHyperparameters`
  (strips only bookkeeping keys; raises when no champion artifact exists) and
  `TestCurrentModelSeasonBounds` (full history always simulated; only the
  requested season range is exported). Updated all `_walk_forward_one_season`
  call sites for the new `hyperparameters` parameter; replaced
  `test_current_model_rejects_season_bounds` with
  `test_current_model_accepts_season_bounds`.
- `HANDOFF.md` - Documented `gridiron ingest weather-backfill` alongside
  `fetch-weather`'s single-week limitation; documented walk-forward's
  fixed-hyperparameter retraining and current-model season-bound alignment
  in the "Historical Backfills and Evaluation" section.
- `DECISIONS.md` - Added D48 (walk-forward restored to D1's fixed-hyperparameter
  contract; current-model season-bound export alignment; the 2003-2004
  evaluation-window floor found while aligning all six families) and D49
  (D7 disposed as a closed data-completeness gap, not a live-serving
  architecture defect).
- `CHANGELOG.md` - Recorded the shipped behavior.
- `ROADMAP.md` - Marked Tier 2 #5/#6's U8 portion complete; marked D7 resolved
  in the verified-defects table with a cross-reference to D49.
- `PLAN.md` - This unit record.

Removed:
- None.

##### Tests

Ruff, Pyrefly, and the full non-slow unit suite passed (4,080 tests, up from
4,069; the net-new tests are listed above).

Real-artifact validation (writes to `data/`, via owning commands):
- The user ran `gridiron ingest weather-backfill --season-year 2026-2027`
  against the real repository: 32 games fetched, 0 failed. Confirmed
  `data/cleaned/NFL_wk_by_wk_w_weather.csv` grew from 7,276 to 7,308 rows and
  now includes 2026-2027 games.
- Ran all six `gridiron evaluate backfill` invocations against the real
  repository, bounded to `--start-season 2003-2004 --end-season 2025-2026`:
  Logistic/Random Forest/XGBoost Win and Random Forest/XGBoost Total
  (walk-forward) and Elo (current-model) each produced exactly 6,232 events.
  Combined walk-forward wall-clock was approximately 70 minutes (Logistic
  ~24 min, Random Forest Win ~23 min, XGBoost Win ~21 min, Random Forest
  Total ~1 min, XGBoost Total ~2 min), down from an estimated ~30 hours
  before the fix; Elo's current-model run took under 4 seconds.
- An initial run at the walk-forward default `--start-season 2002-2003`
  produced 6,498 events for the five ML families but 6,499 for Elo; traced
  the exact one-game difference to `2002_01_DAL_HOU`, the Houston Texans'
  franchise-inaugural game, which Elo can score (a new franchise gets a
  default starting rating) but the shared ML feature pipeline cannot
  (days-rest-family features have no prior game to reference). Confirmed via
  direct inspection of `modeling_file.parquet` and the persisted forecast
  events that no other row is affected anywhere in the 1999-2000 through
  2025-2026 range, then re-ran all six bounded to 2003-2004 onward to
  produce the aligned 6,232-game set.
- Ran `gridiron evaluate model-report` against the six aligned real run IDs:
  built successfully (report ID
  `5c2e1428f907741630e0e21d4d5c7991dfd541a9127f23e01c0d21a1ba8691fc`, 6,232
  games) with sane, non-null metrics for every family - Logistic Win
  brier=0.2209/ece=0.0167/auc=0.6848/calibration slope=1.01 (best-calibrated);
  Random Forest Win brier=0.2232/ece=0.0217; XGBoost Win brier=0.2251/
  ece=0.0194; Elo brier=0.2301/ece=0.0646 (uncalibrated, as expected); Random
  Forest Total mae=10.80/coverage=0.903; XGBoost Total mae=10.84/coverage=0.903
  (both against a 0.90 nominal interval). Selected as the current report.

##### Acceptance

Walk-forward backfill retrains each season using the deployed champion's own
fixed hyperparameters and `epa_window`, never a fresh per-season search,
proven by a test that fails against the old search-every-season behavior and
verified on the real repository. All six families regenerated over an
identical explicit season range, producing one verified common 6,232-game set.
`gridiron ingest weather-backfill` exists, was exercised against the real
2026-2027 gap, and `DECISIONS.md` D49 records D7's disposition as a closed
data-completeness gap with the residual pre-kickoff limitation explicitly
documented as out of scope. The canonical six-family `GameModelEvaluationReport`
was built and selected as current against the six aligned regenerated runs.

#### Unit 9: Reassess calibration and champion selection [Completed September 26, 2026]

##### Completed

Reassessed every game-model champion against U8's corrected, aligned
6,232-game evidence. Retrained all five trainable game-model pairs
through the gated `gridiron models train` comparison: four challengers
(`win_prob logistic/random_forest/xgboost`, `total xgboost`) failed
their promotion gates and left the existing champion in place;
`total random_forest` passed (MAE 10.41 → 10.40) and was promoted.
Refreshed Win calibration (`sigma`/`margin_std`) for all four win_prob
families from the aligned backfill runs, and promoted the champion
manifest — `win_prob` stays on `logistic`, `total` moves to the
freshly-retrained `random_forest`, and five previously-orphaned prop
champions (pruned for having no backing artifact in U4) were
repopulated from their own already-current archives. Verified all six
families remain feature-available for the current upcoming week
(2026-2027 Week 3) and regenerated the 2026-2027 Week 2 development
product end-to-end with the reassessed champions
(`prediction_ready: True`, 16/16 coverage on every readiness
dimension). Regenerated the baseline report, the first since U4 removed
the stale pre-fix reports.

While executing this unit, found and fixed a second, narrower defect in
a separate commit (`13abfb8`): `full-retrain`'s `refresh-calibrations`
and `promote-champions` stages read backfill run identities exclusively
from `ctx["game_backfill_run_ids"]`, populated only by the
same-invocation `backfill-game-models` stage. The module's own
documented `--only refresh-calibrations --only promote-champions`
resume example — and this unit's own need to reuse U8's
already-regenerated backfills without a fresh ~70-minute six-family
re-backfill — crashed on that path. Both stages now fall back to each
pair's latest already-persisted backfill run via
`latest_backfilled_run_id()`, mirroring `promote_champions()`'s existing
resume behavior.

##### Goal

Use only the U8-corrected evaluation evidence (the aligned 6,232-game
common set and the canonical `GameModelEvaluationReport`) to determine
whether the current game-model champions (`win_prob`, `total`) remain
justified; where warranted, refresh each deployed artifact, the Win
calibration registry, and the champion manifest, then verify the
reassessed champions still produce complete upcoming-game feature
coverage and a coherent weekly forecast, and regenerate the baseline
report.

##### Design decisions (pre-implementation audit)

- Current state confirming D6 is still open: `champions.json`'s
  `source_run_id` (`20260805_065153`, promoted 2026-08-04/05) and
  `game_model_calibration.json` (updated 2026-08-05) both predate every
  D1-D4 pipeline fix (U2/U3) and U8's regenerated, aligned backfills —
  neither has been touched since.
- Retrain the five trainable game-model pairs (`win_prob
  logistic/random_forest/xgboost`, `total random_forest/xgboost`) via
  `gridiron models train <name> <type>` (gated, not `--force`) against
  the current `modeling_file.parquet` (now including 2026-2027 weeks
  1-3) under the corrected D1-D4 `_prepare_data`/`_prepare_total_data`
  pipeline. This is required specifically because
  `select_game_regression_champions` (Total's cross-type selector)
  reads each model_type's own persisted holdout metrics directly from
  `ArtifactStore`, not from a backfill run — unlike win_prob's
  classification selector, which already reads U8's aligned backfill
  runs and needs no retrain to reflect corrected evidence. Elo is
  excluded (analytic, not `Trainable`, already current-model backfilled
  in U8).
- Refresh Win calibration (`sigma`/`margin_std`) from the same
  U8-aligned backfilled runs, reusing
  `full_retrain._stage_refresh_calibrations`'s exact read/merge/persist
  logic rather than duplicating it — invoked directly, not through the
  full `full-retrain` composite, since its other stages
  (`refresh-all-data`, `backfill-*`, `train-prop-models`) are out of
  scope or already done by U8.
- Promote the champion manifest via `gridiron evaluate select-model
  --write-manifest` (`write_champion_manifest` → `promote_champions`
  over the full catalog). Win_prob's classification ranking is already
  sourced from U8's aligned runs; Total's regression ranking now
  reflects the retrained artifacts above. Prop-family entries are
  refreshed from their own already-current archives as an accepted,
  unavoidable byproduct of the shared manifest-writing path — no new
  prop evaluation work is in scope for this unit.
- Verify complete upcoming-game feature coverage with the reassessed
  champions via the existing availability-inspection path (the same
  check `weekly-predict`/`verify-week` use), against data already on
  disk — no external fetch.
- Execute the weekly policy with the reassessed champions by
  regenerating a `development`-role forecast for the current upcoming
  week through the existing no-fetch commands
  (`gridiron generate-development-forecast` /
  `gridiron regenerate-development-week`, Units 5-6), not `gridiron
  weekly-predict` — its default stages call nflverse/Odds API refresh
  and current-odds edge generation, which are out of scope under
  "ask before running: external sources." This proves the reassessed
  champions produce a coherent forecast without touching the live
  production selection or any external source.
- Regenerate the baseline report via
  `full_retrain._stage_baseline_report`'s logic, producing the first
  `data/output/reports/full-retrain-*.md` snapshot since U4 removed the
  stale pre-fix reports; its delta table will correctly state no
  previous report exists.
- The canonical six-family `GameModelEvaluationReport` (U8) is not
  rebuilt here: it is keyed to specific backfill run IDs, none of which
  change when a deployed artifact is retrained (a train/holdout split,
  not a new walk-forward backfill run), so the currently-selected
  report remains valid corrected evidence throughout this unit.

##### Ask before running

This unit ran `gridiron models train` five times (each an internal
hyperparameter search plus fit) and rewrote the live-serving champion
manifest and calibration registry that weekly production inference
reads. Per `CLAUDE.md`'s ask-before-running gate, execution started only
after explicit user confirmation, even though each individual step is a
repository-owned command.

##### Files Added/Removed/Changed

Added:
- None.

Changed:
- `src/gridiron_edge/cli/full_retrain.py` - `_stage_refresh_calibrations`
  and `_stage_promote_champions` fall back to
  `latest_backfilled_run_id()` per pair when `ctx["game_backfill_run_ids"]`
  has no entry, instead of failing closed or raising.
- `tests/unit/cli/test_full_retrain.py` - Added one fallback test per
  stage proving the resume path succeeds against the latest on-disk
  backfill run when `ctx` omits `game_backfill_run_ids` entirely.
- `PLAN.md` - This unit record.

Removed:
- None.

Real `data/` artifacts regenerated via owning commands (not
version-controlled, no commit entry):
- `data/models/total/random_forest/` - retrained and promoted.
- `data/output/calibration/game_model_calibration.json` - all four
  win_prob families recalibrated from U8's aligned backfill runs.
- `data/output/champions/champions.json` - re-promoted; `total` now
  points at the retrained `random_forest` artifact, five orphaned prop
  entries repopulated.
- `data/output/reports/full-retrain-2026-09-25-235711.md` - new
  baseline report (first since U4).
- `data/output/weekly_products/` - 2026-2027 Week 2 development product
  regenerated (run `deb7ebd3-5fee-4242-b58a-d86db7e4bb53`) with the
  reassessed champions; the prior Week 2 development selection remains
  physically unchanged on disk per D45 (explicit reselection, no delete
  path).

##### Tests

Ruff, Pyrefly, and the full non-slow unit suite passed after the
`full_retrain.py` fallback fix (43/43 in `test_full_retrain.py`,
including the 2 new fallback tests).

Real-artifact validation (writes to `data/`, via owning commands):
- `gridiron models train` ×5: `win_prob logistic` challenger rejected
  (Brier 0.22109 → 0.22100, improvement below the gate's minimum);
  `win_prob random_forest` rejected (Brier worsened to 0.22218);
  `win_prob xgboost` rejected (Brier worsened to 0.22419);
  `total random_forest` **promoted** (MAE 10.41 → 10.40, R² 0.045 →
  0.043, both gates passed); `total xgboost` rejected (MAE worsened to
  10.46). Each challenger was trained under the corrected D1-D4
  `_prepare_data`/`_prepare_total_data` pipeline against the current
  `modeling_file.parquet` (now including 2026-2027 weeks 1-3).
- `gridiron full-retrain --only refresh-calibrations --assume-done
  backfill-game-models`: all four win_prob families recalibrated
  (`updated_at` refreshed to 2026-09-26) using the fallback fix against
  U8's real aligned backfill runs; `total_random_forest`/`total_xgboost`
  correctly skipped as "(not win_prob)".
- `gridiron evaluate select-model --write-manifest`: ranked all four
  win_prob families on the real aligned 6,218-game evaluation set
  (`win_prob_logistic` brier=0.22133/ece=0.01671/auc=0.68482, confirmed
  best) and wrote the full manifest — `win_prob` unchanged at
  `logistic`, `total` now `random_forest`, and `qb_pass_yards`,
  `qb_rush_yards`, `rb_rush_yards`, `wr_rec_yards`, `te_rec_yards`
  repopulated from their own real archives.
- Availability inspection (`inspect_prediction_availability`) against
  the real 2026-2027 Week 3 upcoming schedule: all six families
  (`elo`, `win_logistic`, `win_random_forest`, `win_xgboost`,
  `total_random_forest`, `total_xgboost`) reported available under the
  reassessed champions.
- `gridiron regenerate-development-week --season 2026-2027 --week 2`
  (Week 3 is not yet in retained history - only weeks 1-2 of 2026-2027
  have been played - so Week 2 was used to exercise the reassessed
  champions end-to-end): generated development run
  `deb7ebd3-5fee-4242-b58a-d86db7e4bb53`, composed and selected
  `weekly_2026_2027_wk02_deb7ebd3-5fee-4242-b58a-d86db7e4bb53`,
  `prediction_ready: True`. Independently confirmed via `gridiron
  verify-week --season 2026-2027 --week 2`: 16 scheduled games, 16
  selected Win predictions, 16 spread values, 16 Total predictions, 16
  projected scores, 16 complete-provenance rows;
  `market_ready: False` on `missing_market_data`/
  `market_scope_mismatch`, expected and independent of prediction
  readiness (Week 2 already played; the current market snapshot only
  covers the current week).
- `gridiron full-retrain --only baseline-report --assume-done
  promote-champions`: wrote the first baseline report since U4, correctly
  showing "no previous report found" for the delta table, `win_prob`
  metrics matching the unchanged champion, and `total` metrics matching
  the newly-promoted `random_forest` artifact (MAE 10.40, RMSE 13.31,
  R² 0.043).

##### Acceptance

Every game-model champion was reassessed against corrected evidence:
four of five retrained challengers were correctly rejected by the
existing promotion gates and one was correctly promoted. Win
calibration and the champion manifest (D6) no longer predate the
current model artifacts. The reassessed champions produce complete
upcoming-game feature coverage and a coherent, fully-ready weekly
forecast, verified end-to-end without any external-source call. The
baseline report reflects current reality. A real, verified defect in
`full-retrain`'s advertised resume workflow was found and fixed
separately, with tests proving the fix.

#### Unit 10: Persist Logistic explanation evidence [Completed September 26, 2026]

##### Completed

Implemented ROADMAP.md Tier 3 #8's U10: `gridiron evaluate explain-logistic
--run-id <run_id>` reconstructs and persists, in one immutable batch, exact
scaled-feature-by-coefficient contributions in log-odds space for every event
in that run's `win_prob`/`logistic` prediction-input evidence, reloading the
fitted estimator's `coef_`/`intercept_` from its exact immutable binary
snapshot rather than the live mutable model store. While performing this
unit's real-artifact validation, found and fixed a second, narrower defect
(D51): `prediction_input_evidence_store.py`'s scan-based lookups crashed the
entire scan on the first pre-U3, schema-version-1 evidence artifact they
encountered - both functions had no production caller until this unit's
builder became the first, so the defect was latent since U3 shipped.

##### Goal

Generate and persist, in batch, exact scaled-feature-by-coefficient
contributions in log-odds space for every `win_prob`/`logistic` statistical
event already captured in one run's prediction-input evidence, bound to that
evidence's exact model, scaler, feature schema, and forecast-event
identities, with the reconstructed estimator output verified against the
persisted `raw_estimator_output` within a strict tolerance. No request-time
inference, no API or frontend change (U11's job), and no tree-family
attribution (Tier 3 #9's job, gated on a resolved approach).

##### Files Added/Removed/Changed

Added:
- `src/gridiron_edge/evaluation/logistic_explanation_evidence.py` - Frozen
  `LogisticExplanationBatch`/`LogisticExplanationEvent`/
  `LogisticFeatureContribution` contracts, canonical SHA-256 identity, and
  reconciliation/invariant validation (contribution arithmetic,
  reconstructed-log-odds/probability consistency, tolerance-bound agreement
  with `raw_estimator_output`).
- `src/gridiron_edge/evaluation/logistic_explanation_evidence_builder.py` -
  `build_and_write_logistic_explanation_batch()`: resolves one run's
  `win_prob`/`logistic` persisted-estimator evidence, reloads the model from
  its immutable binary snapshot, computes and persists contributions, and
  exact-replay-verifies the write.
- `src/gridiron_edge/evaluation/logistic_explanation_evidence_store.py` -
  Immutable JSON persistence
  (`data/output/logistic_explanations/schema=1/batches/{batch_id}.json`)
  mirroring `prediction_input_evidence_store.py`'s atomic
  create-only/idempotent-replay write path; `find_logistic_explanation_by_event`.
- `tests/unit/evaluation/test_logistic_explanation_evidence.py` /
  `_builder.py` / `_store.py` - Contract validation/tamper detection,
  real reconstruction and fail-closed reconciliation coverage, and
  round-trip/idempotent-replay/lookup coverage.

Changed:
- `src/gridiron_edge/cli/evaluate.py` - Added `evaluate explain-logistic`.
- `src/gridiron_edge/evaluation/prediction_input_evidence_store.py` -
  `_all_evidence()` now skips any artifact whose embedded `schema_version`
  is not current (logging a warning) instead of failing the whole scan;
  added `_embedded_schema_version()`. `read_prediction_input_evidence` on a
  known path is unchanged.
- `tests/unit/evaluation/test_prediction_input_evidence_store.py` - Added
  `test_scan_skips_earlier_schema_version_artifacts`.
- `tests/unit/cli/test_evaluate.py` - Added `TestExplainLogisticCommand`.
- `DECISIONS.md` - Added D50 (reconstruction design) and D51 (scan-tolerance
  fix).
- `CHANGELOG.md` - Recorded the shipped behavior.
- `HANDOFF.md` - Added a "Logistic Explanation Evidence" subsection.
- `ROADMAP.md` - Marked Tier 3 #8's U10 portion complete.

Removed:
- None.

##### Tests

Ruff, Pyrefly, and the full non-slow unit suite passed (4,122 tests, up from
4,080; the ~41 new tests are listed above).

Real-artifact validation (writes to `data/`, via the new owning command
only): ran `gridiron evaluate explain-logistic --run-id
deb7ebd3-5fee-4242-b58a-d86db7e4bb53` (the current 2026-2027 Week 2
development run) against the real repository. All 16 `win_prob`/`logistic`
events reconstructed with a maximum reconciliation error of `1.11e-16` -
nine orders of magnitude tighter than the required `1e-9` tolerance.
Inspecting one persisted event directly confirmed a coherent, interpretable
decomposition (`ELO_DIFF` and `OFF_PASS_EPA_DIFF` as the two largest
contributions for `2026_02_CAR_ATL`). Before the scan-tolerance fix, this
same command raised on any `--run-id` because the store's real four legacy
pre-U3 evidence artifacts crashed the underlying scan; confirmed fixed by
rerunning the same command after the fix, which correctly logged four
"skipping" warnings for the legacy artifacts and resolved the target run's
evidence.

##### Acceptance

Every `win_prob`/`logistic` statistical event in a given run's
prediction-input evidence has a persisted, immutable log-odds explanation
record bound to that run's exact evidence, model, and scaler identity — not
the live mutable model store. Each record's contributions plus intercept
reconstruct the evidence's own persisted `raw_estimator_output` within a
documented strict tolerance, proven exact (not merely tolerance-passing) on
real data; a violation fails the build rather than persisting
silently-wrong evidence. No request-time model inference is introduced
anywhere; the new CLI command is the only write path. Random Forest/XGBoost
attribution and the `/explain` API contract change remain explicitly out of
scope. The prediction-input evidence scan-based lookups are usable against
the real repository's permanent legacy-schema artifacts for the first time.

#### Unit 11: `/explain` log-odds contract and field-status wiring [Completed September 26, 2026]

##### Completed

Implemented ROADMAP.md Tier 3 #8's U11: `GET /games/{game_id}/explain` now
resolves `headline_win_prob` and `factors` from the game's selected weekly
product row and, for a Logistic champion, U10's persisted Logistic
explanation store, instead of unconditionally returning a
`scenario_engine`-blocked null shape. `ExplainFactor.delta` (a
never-wired, mislabeled "percentage-point contribution") is replaced with
`log_odds_contribution`, `coefficient`, and `transformed_value`. `band`,
`distribution`, and `market_implied` are unchanged, still blocked
`scenario_engine`. While performing real-artifact validation, found that
the real explanation store already has two immutable batches claiming the
same events for the currently selected run — a legitimate consequence of
the store's create-only, no-current-selection design (D50) — and fixed the
route to treat that as a distinct blocked state
(`ambiguous_explanation_evidence`) instead of letting the store's
fail-closed `ValueError` become an unhandled 500.

##### Goal

Change the `/explain` contract from percentage-point deltas to log-odds
contributions, regenerate the frontend client, and serialize persisted
Logistic explanation evidence through `_meta.field_status` wherever it
exists — with no request-time inference and no ExplainPage UI work (still
gated on the scenario engine / comparable-game work in Tier 3 #9–10).

##### Design decisions

- Three-way `factors` unavailability rule keyed on the selected weekly
  product row: `win_status != "available"` → `NO_WIN_FORECAST` (new,
  mirrors the existing `NO_ELO_FORECAST` family-specific slug); Win
  champion isn't Logistic → the existing `Blocker.FEATURE_ATTRIBUTION`
  (same slug `props.py`/`games.py` already use, Tier 3 #9 unresolved);
  Logistic with no batch found yet → `NO_EXPLANATION_EVIDENCE` (new — the
  CLI is manual, not run per weekly forecast). `headline_win_prob` is
  populated whenever `win_status == "available"`, independent of the
  champion's model type or explanation-evidence state, since it is just the
  already-persisted `home_win_prob`.
- Real-artifact validation surfaced a fourth state:
  `find_logistic_explanation_by_event` raises when more than one batch
  claims an event, and the real store already has exactly that. Gave the
  store a dedicated `AmbiguousLogisticExplanationError(ValueError)` instead
  of a bare `ValueError` so the route can catch it specifically without
  masking a genuinely malformed-artifact error as the same condition;
  mapped to a new `Unavailable.AMBIGUOUS_EXPLANATION_EVIDENCE` slug via the
  same route-catches-a-domain-exception pattern `props.py` already uses for
  `ChampionNotFoundError`. See `DECISIONS.md` D52.
- `ExplainFactor` gained `coefficient`/`transformed_value` (already computed
  and persisted by U10) alongside the renamed `log_odds_contribution`; the
  unused `description` field was dropped. No backward compatibility
  preserved — the prior shape was dev-era and never served real data.
- The intercept is serialized as its own leading `ExplainFactor` with
  `is_baseline=True` rather than a new top-level `GameExplain` field,
  reusing a field the schema already carried for exactly this purpose.
- No human-readable feature-label mapping exists anywhere in the repo, so
  `label` is the raw feature name — not invented text.
- `frontend/src/screens/ExplainPage.tsx` and `WhyLink.tsx` are unchanged:
  both already document that unblocking the page's content is separate,
  future work from this contract change.

##### Files Added/Removed/Changed

Added:
- `src/gridiron_edge/api/serializers/explain.py` - `serialize_game_explain()`
  builds the response and owns all `_meta.field_status` construction per
  D18, including the three-way `factors` rule and the always-blocked
  scenario-engine fields.
- `tests/unit/api/test_serializers_explain.py` - Coverage for all five
  `factors` states (win-unavailable, non-Logistic champion, no evidence yet,
  ambiguous evidence, evidence found) plus the always-blocked fields.
- `tests/unit/api/test_routes_explain.py` - `TestClient` coverage: 404 on
  unknown `game_id`; 200 with real factors; non-Logistic-champion,
  win-unavailable, and ambiguous-evidence blocked states, each via mocking
  the route's own loader functions (mirroring
  `test_edges_route_diagnostics.py`'s pattern).

Changed:
- `src/gridiron_edge/api/schemas/explain.py` - `ExplainFactor`: removed
  `delta`/`description`; added `log_odds_contribution`, `coefficient`,
  `transformed_value`.
- `src/gridiron_edge/api/routes/explain.py` - Real resolution logic
  (`load_game` → 404 or branch on `win_status`/`win_model_type` →
  optionally `load_logistic_explanation_for_event`, catching
  `AmbiguousLogisticExplanationError`) replacing the unconditional stub.
- `src/gridiron_edge/api/loaders.py` - Added
  `load_logistic_explanation_for_event()`, wrapping
  `find_logistic_explanation_by_event` with `repo=settings.repo_root` per
  D19.
- `src/gridiron_edge/api/meta.py` - Added `Unavailable.NO_WIN_FORECAST`,
  `NO_EXPLANATION_EVIDENCE`, `AMBIGUOUS_EXPLANATION_EVIDENCE`.
- `src/gridiron_edge/evaluation/logistic_explanation_evidence_store.py` -
  `find_logistic_explanation_by_event` now raises
  `AmbiguousLogisticExplanationError` (a `ValueError` subclass) instead of a
  bare `ValueError` for the multiple-claims case.
- `tests/unit/api/test_schemas_explain.py` - Updated for the renamed/added
  `ExplainFactor` fields.
- `tests/unit/api/test_app_routes.py` - Removed `/games/{game_id}/explain`
  from the generic always-200-stub smoke table; it now legitimately 404s on
  an unknown `game_id`, covered instead by the dedicated route test file.
- `api-schema.json` - Regenerated via `gridiron api export-schema`.
- `HANDOFF.md` - Added a `/games/{game_id}/explain` subsection under
  "Logistic Explanation Evidence" documenting the four resolution states.
- `CHANGELOG.md`, `DECISIONS.md` (D52) - This unit's record.

Removed:
- None.

##### Tests

Ruff, Pyrefly, and the full non-slow unit suite passed (4,133 tests, up from
4,122; the 11 new tests are listed above). Frontend gate passed:
`pnpm gen:api` regenerated `frontend/src/api/schema.ts` (gitignored, not
checked in) from the updated `api-schema.json`; `pnpm lint`, `pnpm build`
(148 modules, no reference to the removed `delta`/`description` fields
anywhere in the frontend), and `pnpm test:run` (511 tests) all passed
unchanged.

Real-artifact validation (read-only, no `data/` writes — this unit adds no
writer): called `GET /games/{game_id}/explain` through the real FastAPI app
against the real repository. `2026_02_CAR_ATL` (from the selected
2026-2027 Week 2 product, run `deb7ebd3-5fee-4242-b58a-d86db7e4bb53`)
returned `headline_win_prob=0.5541168882599108` (the persisted
`home_win_prob`) with `factors` blocked `ambiguous_explanation_evidence` -
confirmed by direct inspection that the real store holds two batches
(`36265a5f8b1a...`, `e2c399f98809...`) both covering the same 16 events for
that run, so all 16 real games in the current selection hit this path, not
the populated-factors path. An unknown `game_id` returned 404. The
non-Logistic-champion, win-unavailable, and single-batch-found states do
not currently exist in the real repository's data, so they are proven by
the fixture-based unit and route tests instead.

##### Acceptance

`GET /games/{game_id}/explain` serializes only persisted state — the
selected weekly product row and, when applicable, the persisted Logistic
explanation store — with no request-time inference. `ExplainFactor`
carries log-odds contributions (`coefficient * transformed_value`), not the
prior mislabeled percentage-point `delta`. Every non-populated `factors`
state (win unavailable, non-Logistic champion, no evidence yet, ambiguous
evidence) is marked through `_meta.field_status` with a slug distinguishing
its cause; none produces a 500. `band`, `distribution`, and
`market_implied` remain unchanged. `api-schema.json` and the generated
frontend client reflect the new contract; no frontend UI behavior changed.
ROADMAP.md Tier 3 #8 (U10 + U11) is now fully complete.

#### Unit 12: Tree-model attribution decision [Completed September 26, 2026]

##### Completed

Resolved ROADMAP.md Tier 3 #9 as a documented decision rather than an
implementation: audited the actual calibration architecture behind both Win
tree model types and recorded why neither gets an attribution mechanism
now, plus the precondition that would unblock one later. No source code
changed — `/games/{game_id}/explain`'s existing `FEATURE_ATTRIBUTION`
blocking behavior for non-Logistic champions (D52/U11) is unchanged.

##### Goal

Decide, from the real on-disk model and calibration architecture rather
than assumption, whether Tree SHAP or another exact-attribution method is
buildable now for the Win family's tree champions (Random Forest,
XGBoost), and record that decision so it does not stay silently open.

##### Files Added/Removed/Changed

Added:
- None.

Changed:
- `DECISIONS.md` (D53) - Records that Random Forest/classification is
  unconditionally wrapped in `CalibratedClassifierCV(isotonic)`
  (`models/game_prediction/base.py:171-184`), making exact attribution
  against its raw output permanently unable to reconstruct the actually-
  served calibrated probability; that XGBoost/classification is only
  *conditionally* recalibrated post-training when holdout ECE exceeds
  0.025 (`base.py:1038-1085`) — the current on-disk `win_prob`/`xgboost`
  artifact has `calibration_applied: false` (ECE 0.0245), so it is
  currently uncalibrated, but that is a per-training-run outcome, not an
  architectural guarantee; and that neither tree type is the current Win
  champion (`champions.json`: `win_prob` = `logistic`), so the blocked
  branch has no real consumer today. Records the unresolved precondition
  for building this later: attribute in pre-calibration margin space and
  surface the isotonic remap as an explicit, non-attributed adjustment,
  with per-artifact branching on `calibration_applied` to fail closed.
- `HANDOFF.md` - Reworded the two `/games/{game_id}/explain` references to
  Tier 3 #9 from "unresolved" to point at D53's recorded decision, since
  the roadmap item is now resolved even though the served behavior
  (blocked `feature_attribution`) is unchanged.
- `ROADMAP.md` - Tier 3 #9's Track A implementation note marked
  `[Complete: documented decision, D53]`.
- `CHANGELOG.md` - This unit's record.

Removed:
- None.

##### Tests

No source code changed, so no test suite run beyond confirming this by
inspection. Verified the decision's factual claims directly against the
repository rather than trusting ROADMAP's prior framing: read
`models/game_prediction/base.py`'s `_create_model` (calibration wrapping)
and post-training recalibration branch; inspected
`data/models/win_prob/random_forest/metadata.json` and
`.../xgboost/metadata.json` for real ECE values and the `calibration_applied`
flag; inspected `data/output/champions/champions.json` to confirm the
current Win champion is `logistic`, not a tree model; and re-read
`api/serializers/explain.py` to confirm the `FEATURE_ATTRIBUTION` blocking
path this decision leaves unchanged.

##### Acceptance

ROADMAP.md Tier 3 #9 has a recorded, evidence-grounded decision
(`DECISIONS.md` D53) rather than being silently open: no attribution is
built for either Win tree model type; the reason is specific to each
type's actual calibration behavior, not a blanket "trees are hard"
assumption; and the precondition for revisiting it later is stated
explicitly. `/games/{game_id}/explain` behavior is unchanged and still
correct under this decision.

#### Unit 13: Comparable-games historical corpus and retrieval evidence [Completed September 26, 2026]

##### Completed

Implemented ROADMAP.md Tier 3 #10's backend half: a new immutable
historical feature-vector corpus (`gridiron evaluate build-comparable-corpus`)
covering 7,271 historical `win_prob`/`logistic` games, and a new per-run
comparable-games retrieval evidence store
(`gridiron evaluate find-comparables --run-id <run_id>`) that matches each
event's already-persisted feature vector against the corpus by Euclidean
distance in the champion's own standardized feature space, excludes the
query's own game, filters to an empirically-derived distance threshold, and
persists real recorded outcomes plus the top contributing features per
match. No `/games/{game_id}/comparables` API or frontend change — that
route stays on its existing `Blocker.COMPARABLES` null shape until a future
unit (**Unit 14**, mirroring how U10/U11 split Logistic explanation into a
backend unit and an API unit). While implementing the idempotent-identity
design both new stores share, found and fixed a real bug: comparing full
encoded bytes on replay (mirroring `LogisticExplanationBatch`'s pattern)
incorrectly rejected a genuine no-op rebuild, since `generated_at` legitimately
differs between runs but isn't part of either artifact's identity. See
`DECISIONS.md` D54.

##### Goal

Build comparable-game retrieval scoped to `win_prob`/`logistic`: define
similarity from the champion's own authenticated model inputs, prevent
future-information leakage, explain why games are comparable, and derive
the retrieval threshold empirically — as a persisted, batch-computed
artifact pair, not request-time computation.

##### Design decisions

- Similarity space is the exact `combined_111` feature vector the deployed
  `win_prob`/logistic champion trains on, scaled by that champion's own
  fitted `StandardScaler` — reusing `_rebuild_features_with_window` (D1/U3)
  and `FEATURE_SETS["combined"]` rather than writing new feature-engineering
  code. Leakage prevention is inherited structurally from that
  already-trusted pipeline (every row's features are pre-kickoff by
  construction), not a temporal restriction between the query game and
  candidate pool — a 2026 game can legitimately be compared against a 2003
  or a 2025 historical game.
- The corpus is a new artifact pair, not an extension of
  `prediction_input_evidence` (scoped to selected weekly forecasts only)
  or the historical backtest evidence store (persists prediction outputs,
  not feature vectors). Bound to one exact
  `(model_content_digest, scaler_content_digest)` pair computed from the
  live `data/models/win_prob/logistic/` artifact at build time; retrieval
  fails closed if a run's own evidence was produced by a different
  snapshot (a stale corpus after a champion retrain), never silently
  comparing incompatible scaled spaces.
- Historical `VEGAS_LINE`/`FAVORITED` are joined from
  `data/cleaned/NFL_wk_by_wk_cleaned.csv` — already complete for the full
  historical range, so no Tier 6 #19 market-data dependency.
- Distance metric is plain Euclidean distance in the already-standardized
  space (no cosine/Mahalanobis justification needed once features share
  roughly unit variance); the threshold is the 90th percentile of
  leave-one-out nearest-neighbor distances computed once at corpus-build
  time, not a chosen constant. "Why comparable" is the top 5
  highest-squared-difference feature names per match — raw names, no
  invented human-readable labels, same precedent as D52.
- Not affected by D53's tree-calibration problem: nearest-neighbor distance
  doesn't decompose a calibrated decision function, so this technique would
  work regardless of champion model type — it stays Logistic-scoped only
  because that's the current champion and the corpus is snapshot-bound, not
  because of a calibration blocker.
- Both `corpus_id` and `batch_id` exclude `generated_at` from their
  content-addressed identity, so rebuilding from unchanged inputs is a true
  no-op instead of producing a second artifact claiming the same content —
  a deliberate improvement on `LogisticExplanationBatch`'s precedent, which
  is why D52 needed `AmbiguousLogisticExplanationError` in the first place.
  `AmbiguousComparableGamesError` still exists for
  `find_comparable_games_by_event`, but now specifically means "this event
  was matched against two different corpus generations," a genuine
  staleness signal rather than a duplicate-run artifact.
- Real-artifact validation surfaced that idempotent identity alone isn't
  sufficient: the store's own "reject a replay with different content"
  check must compare the embedded identity field, not full encoded bytes,
  since a legitimately-idempotent rebuild still produces a different
  `generated_at`. Fixed in both new stores. See `DECISIONS.md` D54.

##### Files Added/Removed/Changed

Added:
- `src/gridiron_edge/evaluation/comparable_games_corpus.py` - Immutable
  corpus contracts: identity, validation, and the NaN-tolerant frame content
  digest.
- `src/gridiron_edge/evaluation/comparable_games_corpus_store.py` - JSON
  manifest plus Parquet frame persistence; idempotent replay; `find_latest`
  by model identity.
- `src/gridiron_edge/evaluation/comparable_games_corpus_builder.py` -
  Replays `_rebuild_features_with_window`/`FEATURE_SETS["combined"]` over
  every historical game, scales with the live champion's scaler, derives
  the empirical distance threshold, and persists the corpus.
- `src/gridiron_edge/evaluation/comparable_games_evidence.py` - Immutable
  per-event retrieval batch contracts: identity, validation (contiguous
  ranks, non-decreasing distance, threshold-bounded matches), payload.
- `src/gridiron_edge/evaluation/comparable_games_evidence_store.py` - JSON
  batch persistence; idempotent replay; `find_comparable_games_by_event`
  with `AmbiguousComparableGamesError` for genuine cross-corpus staleness.
- `src/gridiron_edge/evaluation/comparable_games_evidence_builder.py` -
  Matches each run event against the corpus by Euclidean distance,
  self-excludes, threshold-filters, ranks, and computes real outcomes plus
  top contributing features.
- `tests/unit/evaluation/test_comparable_games_corpus.py`,
  `test_comparable_games_corpus_store.py`,
  `test_comparable_games_evidence.py`, `test_comparable_games_evidence_store.py`,
  `test_comparable_games_evidence_builder.py` - Contract, persistence, and
  ranking/threshold/outcome coverage (40 tests).

Changed:
- `src/gridiron_edge/cli/evaluate.py` - Added `build-comparable-corpus` and
  `find-comparables` commands.
- `DECISIONS.md` (D54), `CHANGELOG.md`, `ROADMAP.md` - This unit's record;
  Track A's completion gate widened from U1–U13 to U1–U14 to name the
  deferred API-wiring unit explicitly.

Removed:
- None.

##### Tests

Ruff, Pyrefly, and the full non-slow unit suite passed (4,173 tests, up
from 4,133; the 40 new tests are listed above).
`comparable_games_corpus_builder.py`'s real orchestration (live
`ArtifactStore` reads, `load_modeling_file`, historical CSV join) is
covered by real-artifact validation below rather than a fixture-based unit
test, since faithfully mocking it would require reconstructing the live
modeling file's full 125-column schema; `comparable_games_evidence_builder.py`
is fully unit-tested against a small synthetic corpus and evidence pair
(ranking, self-exclusion, threshold filtering, real-outcome computation,
idempotent rerun, stale-corpus rejection, no-corpus-yet rejection).

Real-artifact validation (writes to `data/output/comparable_games_corpus/`
and `data/output/comparable_games/`, both new stores this unit owns):
`gridiron evaluate build-comparable-corpus` against the real repository
produced a 7,271-game corpus (of 7,308 total; 37 excluded for incomplete
features or a tied game), `corpus_id` `15567f84e1340c749f7bf70f52018bfe68873a24d35570ab7e6b3927bec971bd`,
distance threshold (p90) 9.2768, leave-one-out median 8.2393 — reproduced
identically on a second run (idempotent). `gridiron evaluate find-comparables
--run-id deb7ebd3-5fee-4242-b58a-d86db7e4bb53` (the selected 2026-2027 Week
2 product) matched all 16 real games; sample sizes honestly ranged from 0
(`2026_02_GB_NYJ`, `2026_02_MIN_CHI` — genuinely unusual matchups) to the
requested cap of 20 (`2026_02_DET_BUF`, `2026_02_LV_LAC`), reproduced
identically on rerun. Inspected one real match
(`2026_02_CAR_ATL`'s closest comparable, the 2006 NFC Wild Card
Seahawks–Cowboys game) and confirmed the persisted score, favorite, cover
outcome, and top contributing feature names are all correct against the
real historical record.

##### Acceptance

`gridiron evaluate build-comparable-corpus` and
`gridiron evaluate find-comparables` persist immutable, content-addressed
evidence with no request-time computation; both are idempotent on rerun
against unchanged inputs. Similarity is defined over the champion's own
authenticated feature space, leakage is prevented structurally, "why
comparable" is named per real feature, and the retrieval threshold is
derived empirically rather than chosen. `/games/{game_id}/comparables`
remains correctly blocked until Unit 14 wires the API contract.

#### Unit 14: `/comparables` contract wiring [Completed September 26, 2026]

##### Completed

Implemented ROADMAP.md Tier 3 #10's remaining half: `GET
/games/{game_id}/comparables` now resolves `comparables`, `sample_size`,
`favorite_win_rate`, and `favorite_cover_rate` from the game's selected
weekly product row and, for a Logistic champion, U13's persisted
comparable-games retrieval evidence — instead of an unconditional
`Blocker.COMPARABLES`-blocked null shape. `ComparableGame`'s prior
speculative, never-wired fields (`date_label`, `favorite`, `underdog`,
`line`, `final_score`, `note`) are replaced with the persisted
`ComparableGameMatch`'s own fields directly. This closes Tier 3 #10 and,
with it, Track A (U1–U14) in full.

##### Goal

Wire `/games/{game_id}/comparables` to serve U13's persisted evidence with
no request-time computation, mirroring U11's `/explain` contract change,
and regenerate the frontend client.

##### Design decisions

- Four-way `comparables` unavailability rule mirroring D52's `/explain`
  shape exactly: `win_status != "available"` → reused `NO_WIN_FORECAST`;
  Win champion isn't Logistic → reused `Blocker.COMPARABLES` (no corpus
  exists for that model type yet, per U13's scope, not a calibration
  problem); more than one batch claims the event
  (`AmbiguousComparableGamesError`) → new `AMBIGUOUS_COMPARABLE_EVIDENCE`;
  Logistic with no batch yet → new `NO_COMPARABLE_EVIDENCE`. All four
  response fields are blocked or populated together — unlike `/explain`,
  there is no independent "headline" field here.
- `ComparableGame` redesigned to expose `game_id`, `rank`, `distance`,
  `season`, `week`, `game_date`, `away_team`, `home_team`, `away_score`,
  `home_score`, `favorite_team`, `spread_magnitude`, `favorite_won`,
  `favorite_covered`, and `top_contributing_features` directly — no
  server-generated display strings (`date_label`, `line`, `final_score`).
  Formatting real values into presentation text is a frontend concern,
  same reasoning D52 applied to feature labels.
- An empty `matches` tuple serializes as `comparables: []`, `sample_size:
  0` — not a blocked state. It is real evidence ("nothing was similar
  enough"), not missing evidence; two of the real selected Week 2 games
  (`2026_02_GB_NYJ`, `2026_02_MIN_CHI`) exercise this path today.
- `/games/sf-bal/comparables` removed from `test_app_routes.py`'s
  generic always-200 smoke table: an unknown `game_id` now legitimately
  404s, exactly as U11 did for `/explain`, covered instead by a dedicated
  route test file.
- See `DECISIONS.md` D55.

##### Files Added/Removed/Changed

Added:
- `src/gridiron_edge/api/serializers/comparables.py` -
  `serialize_game_comparables()` builds the response and owns all
  `_meta.field_status` construction per D18, including the four-way rule.
- `tests/unit/api/test_serializers_comparables.py` - Coverage for all five
  states (win-unavailable, non-Logistic champion, ambiguous evidence, no
  evidence yet, populated including the honest-empty case).
- `tests/unit/api/test_routes_comparables.py` - `TestClient` coverage: 404
  on unknown `game_id`; 200 with real comparables; non-Logistic-champion,
  win-unavailable, ambiguous-evidence, and no-evidence-yet blocked states,
  mocking the route's own loader functions.

Changed:
- `src/gridiron_edge/api/schemas/comparables.py` - `ComparableGame`:
  removed `date_label`/`favorite`/`underdog`/`line`/`final_score`/`note`;
  added `game_id`, `rank`, `distance`, `season`, `week`, `game_date`,
  `away_team`, `home_team`, `away_score`, `home_score`,
  `top_contributing_features` (new `ComparableFactor` element type).
- `src/gridiron_edge/api/routes/comparables.py` - Real resolution logic
  (`load_game` → 404 or branch on `win_status`/`win_model_type` →
  optionally `load_comparable_games_for_event`, catching
  `AmbiguousComparableGamesError`) replacing the unconditional stub.
- `src/gridiron_edge/api/loaders.py` - Added
  `load_comparable_games_for_event()`, wrapping
  `find_comparable_games_by_event` with `repo=settings.repo_root` per D19.
- `src/gridiron_edge/api/meta.py` - Added
  `Unavailable.NO_COMPARABLE_EVIDENCE`, `AMBIGUOUS_COMPARABLE_EVIDENCE`.
- `src/gridiron_edge/api/app.py` - Updated the `comparables` tag
  description to reflect the real contract instead of "blocked."
- `tests/unit/api/test_schemas_comparables.py` - Rewritten for the new
  `ComparableGame`/`ComparableFactor` shape.
- `tests/unit/api/test_app_routes.py` - Removed
  `/games/{game_id}/comparables` from the generic always-200 smoke table.
- `api-schema.json` - Regenerated via `gridiron api export-schema`.
- `HANDOFF.md` - `/games/{game_id}/comparables` subsection now describes
  the real four-state contract instead of "not yet wired."
- `CHANGELOG.md`, `DECISIONS.md` (D55) - This unit's record.
- `ROADMAP.md` - Tier 3 #10 marked fully complete; Track A gate confirmed
  at U1–U14, all closed.

Removed:
- None.

##### Tests

Ruff, Pyrefly, and the full non-slow unit suite passed (4,188 tests, up
from 4,173; the 15 net new/changed tests are listed above). Frontend gate
passed: `pnpm gen:api` regenerated `frontend/src/api/schema.ts` (gitignored)
from the updated `api-schema.json`; `pnpm lint`, `pnpm build` (148 modules,
no reference to the removed display-string fields anywhere in the
frontend), and `pnpm test:run` (511 tests) all passed unchanged — no
frontend UI consumes this contract yet.

Real-artifact validation (read-only, no `data/` writes): called `GET
/games/{game_id}/comparables` through the real FastAPI app against the
real repository. `2026_02_CAR_ATL` (selected 2026-2027 Week 2 product)
returned 6 real comparables (`favorite_win_rate=1.0`,
`favorite_cover_rate=0.167`), matching U13's CLI validation exactly.
`2026_02_GB_NYJ` returned `comparables: []`, `sample_size: 0`, with no
blocked `_meta.field_status` entry — the honest-empty path, confirmed live
rather than only in fixtures. An unknown `game_id` returned 404.

##### Acceptance

`GET /games/{game_id}/comparables` serializes only persisted state — the
selected weekly product row and, when applicable, U13's persisted
comparable-games evidence — with no request-time computation.
`ComparableGame` carries real match fields, not the prior unwired display
strings. Every non-populated state (win unavailable, non-Logistic
champion, no evidence yet, ambiguous evidence) is marked through
`_meta.field_status` with a distinguishing slug; none produces a 500. An
honestly-empty result is distinguished from a blocked one. `api-schema.json`
and the generated frontend client reflect the new contract; no frontend UI
behavior changed. ROADMAP.md Tier 3 #10 (U13 + U14) is now fully complete,
and Track A (U1–U14) is complete.

#### Unit 15: Logistic explanation batch identity fix and duplicate disposition [Completed September 26, 2026]

##### Completed

Fixed the real defect D52 found but deferred: `LogisticExplanationBatch`'s
identity included `generated_at`, so two `explain-logistic` invocations
over the same run's evidence produced two distinct, otherwise-identical
batches both legitimately claiming the same events — exactly what the real
store had, blocking all 16 real 2026-2027 Week 2 games behind
`ambiguous_explanation_evidence` instead of serving actual factors. Fixed
the identity to exclude `generated_at` (mirroring D54's identical fix for
comparable-games evidence), found and fixed a second related bug in the
builder's own post-write self-check that the first fix exposed, disposed
of the two real duplicate batches (deleted, not archived in place — this
store has no selection layer to redirect), and regenerated one clean batch.
All 16 real games now resolve to populated `factors` through the real API.
See `DECISIONS.md` D56.

##### Goal

Make `explain-logistic` reruns idempotent (no second batch for unchanged
evidence), and unblock the real store's actual duplicate-batch condition
so `/explain` serves real data for the currently selected week instead of
a permanently-blocked state.

##### Design decisions

- `logistic_explanation_batch_id`/`_identity_payload` no longer take or
  hash `generated_at`; `logistic_explanation_batch_payload` still records
  it in the stored JSON (informational), just outside the identity
  computation. Exact mirror of D54's `comparable_games_batch_id` fix.
- `write_logistic_explanation_batch`'s "reject a replay with different
  content" check now compares the embedded `batch_id` on an existing file,
  not full encoded bytes — the same fix already applied twice to the
  comparable-games stores (D54), applied here for the same reason.
- Found while exercising the fix against real data, not anticipated up
  front: `logistic_explanation_evidence_builder.py`'s own post-write
  verification did `if stored != batch: raise` (full dataclass equality),
  which fails on every genuinely idempotent rerun now that `generated_at`
  legitimately differs between the freshly-constructed `batch` and the
  `stored` object read back from a prior run. Changed to compare
  `batch_id` only; the build result now returns the actually-stored batch,
  not the transient one. Neither comparable-games builder has this bug —
  neither performs a full-equality post-write self-check, so D54 never
  exercised this path.
- Disposition of the two real duplicate batches: deleted, not archived in
  place per D45's precedent, since (a) this store has no selection pointer
  for "archived" to redirect away from — every file in the directory is
  scanned unconditionally — and (b) both old files' embedded `batch_id`
  would fail the new identity validation as corrupt regardless, since they
  were hashed under the prior scheme. Confirmed byte-identical in substance
  before deleting (same run/evidence/digests/all 16 events), so nothing
  unique was lost. `prediction_input_evidence` (the primary evidence) was
  never touched.

##### Files Added/Removed/Changed

Added:
- None.

Changed:
- `src/gridiron_edge/evaluation/logistic_explanation_evidence.py` -
  Identity payload excludes `generated_at`.
- `src/gridiron_edge/evaluation/logistic_explanation_evidence_store.py` -
  Replay check compares embedded identity, not full bytes; updated
  `AmbiguousLogisticExplanationError` docstring.
- `src/gridiron_edge/evaluation/logistic_explanation_evidence_builder.py` -
  Post-write verification compares `batch_id`, not full equality; returns
  the actually-stored batch.
- `tests/unit/evaluation/test_logistic_explanation_evidence.py`,
  `test_logistic_explanation_evidence_store.py`,
  `test_logistic_explanation_evidence_builder.py` - Added
  identity-excludes-`generated_at`, idempotent-rewrite, and
  idempotent-rerun-with-a-different-timestamp coverage (the last of which
  reproduces the builder bug found above).
- `DECISIONS.md` (D56), `HANDOFF.md`, `CHANGELOG.md`, `ROADMAP.md` - This
  unit's record.

Removed:
- `data/output/logistic_explanations/schema=1/batches/36265a5f8b1a791a433cdc2b3c64c075db8e6a9d254381e80471588f371011a9.json`,
  `.../e2c399f9880929524e833b9014730e87675127ff769896067fdd3c13428afcd8.json`
  - Confirmed byte-identical accidental duplicates from before this fix;
    deleted per the disposition above, replaced by one freshly-regenerated
    batch under the fixed identity scheme.

##### Tests

Ruff, Pyrefly, and the full non-slow unit suite passed (4,191 tests, up
from 4,188; the 3 new tests are listed above).

Real-artifact validation (writes to `data/output/logistic_explanations/`,
the store this unit owns): deleted the two confirmed-identical stale
batches, ran `gridiron evaluate explain-logistic --run-id
deb7ebd3-5fee-4242-b58a-d86db7e4bb53` to regenerate one clean batch
(`bc98cdb5...`, max reconciliation error 1.11e-16, matching the original
build's precision exactly), and confirmed re-running the same command a
second time is a true no-op (same `batch_id`, no error) — the exact
scenario that previously raised "does not exactly replay input" once the
identity fix was in place without the builder fix. Called
`GET /games/{game_id}/explain` through the real FastAPI app for all 16 real
games in the selected 2026-2027 Week 2 product: every one now returns
populated `factors` with no `_meta.field_status` entry for that field,
where all 16 previously returned `ambiguous_explanation_evidence`.

##### Acceptance

Re-running `explain-logistic` against unchanged evidence never produces a
second batch. All 16 real Week 2 games serve real `factors` through the
live API. The two disposed duplicate batches are gone, not edited in
place, with the disposition and its rationale recorded in `DECISIONS.md`
D56; `prediction_input_evidence` is untouched.

#### Unit 16: `ExplainPage` — factor waterfall and comparables, no scenario controls [Completed September 26, 2026]

##### Completed

Rebuilt `ExplainPage` from a hardcoded `BlockedScreen` into a real,
data-driven page over U10/U11's Logistic explanation evidence and
U13/U14's comparable-games evidence — the first frontend consumer of
either contract. Shows the real factor waterfall (sorted by magnitude,
top 8 with a "show more" disclosure) and the real comparable-games table
(with an honest distinct empty state), while the credible band, outcome
distribution, and market comparison stay explicit `ComingSoonCard`
placeholders pending the scenario engine (Tier 7, unbuilt). Verified live
in a headless browser against the real dev server and API, not just
component tests. See `DECISIONS.md` D57.

##### Goal

Give the persisted explanation and comparable-games evidence (Tier 3 #8,
#10) a real frontend surface, scoped to what the backend actually
supports today — no fabricated sensitivity controls, distribution, or
market comparison.

##### Design decisions

- New hooks `useExplain(gameId)`/`useComparables(gameId)` in
  `api/hooks.ts`, exact shape of the existing `useGame` (react-query +
  generated client, `enabled: gameId !== null`).
- Header data (team names, game date) comes from the existing `useGame`
  hook — `GameExplain` itself carries no team/date fields, so the page
  composes three independent queries for one screen.
- Factors: intercept pinned first, remaining 111 sorted by descending
  `|log_odds_contribution|`, top 8 shown with a "Show N more factors"
  toggle — implements the collapsible-waterfall request made earlier this
  session; no data is dropped, only the default view is bounded.
- Comparables: an empty `comparables: []` renders a distinct plain-text
  "no historical game was similar enough" message, never conflated with a
  blocked `_meta.field_status` state.
- Every field renders through the existing `FieldValue`/`BlockedField`
  components — no new field-status presentation code needed, and no field
  is silently hidden when unavailable.
- `TeamMark` already resolves team identity by either abbreviation or long
  name (`useTeamByAbbr` checks both), so `ComparableGame`'s persisted long
  team names render correctly with zero new resolution code.
- `route.params.gameId` was already threaded in everywhere `WhyLink` is
  used, so no navigation changes were needed.
- See `DECISIONS.md` D57 for the full rationale and alternatives
  considered (why not the full mockup-style cumulative waterfall bar; why
  not hide the scenario-dependent sections entirely).

##### Files Added/Removed/Changed

Added:
- `frontend/src/screens/ExplainPage.test.tsx` - Component coverage: loading
  state, populated factors/comparables, magnitude sorting and the
  show-more/show-fewer toggle, blocked-factors placeholder, honestly-empty
  comparables, and a populated comparable row's real outcome fields.

Changed:
- `frontend/src/screens/ExplainPage.tsx` - Full rewrite from
  `BlockedScreen` to a real page (header, factor waterfall, comparables
  table, scenario-dependent `ComingSoonCard`s).
- `frontend/src/api/hooks.ts` - Added `useExplain`, `useComparables`.
- `DECISIONS.md` (D57), `CHANGELOG.md`, `ROADMAP.md` - This unit's record.

Removed:
- None.

##### Tests

`pnpm lint`, `pnpm build` (149 modules, zero type errors against the
regenerated `schema.ts`), and `pnpm test:run` (517 tests, up from 511 — 6
new) all passed.

Real-browser validation (headless Chromium against the real
`gridiron api serve` and `pnpm dev`, not a mock): navigated to
`#/explain?gameId=2026_02_CAR_ATL` — real headline win probability
(55.4%), 112 real factor rows (8 shown, "Show 103 more factors" expands to
all, "Show fewer factors" collapses back), 6 real comparable games with
correct team colors/scores/favorite/won/cover, and all three
scenario-dependent cards showing "Not yet available." Navigated to
`#/explain?gameId=2026_02_GB_NYJ` (a genuinely unusual matchup) — real
factors populated normally, comparables section shows the honest-empty
message instead of a blocked state. Zero console errors on either page.
Screenshots captured for both.

##### Acceptance

`ExplainPage` shows real factors and real comparables for a real game with
both evidence types generated, verified live in a browser against the real
API — not fabricated, not silently hidden when unavailable. Unavailable
and honestly-empty states are visually distinct. The scenario-dependent
section is clearly marked "not yet available," not omitted or faked.
ROADMAP.md Tier 3 #10 and Tier 8 #23 both now have a shipped frontend
surface for explanation and comparable-games evidence; the scenario engine
(band/distribution/market comparison) remains explicitly future work.

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
