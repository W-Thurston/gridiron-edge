# Gridiron Edge - Changelog

### 2026-09-26 - `/explain` log-odds contract and field-status wiring

#### Changed

- `GET /games/{game_id}/explain` now resolves `headline_win_prob` and
  `factors` from the game's selected weekly product row and (for a Logistic
  champion) the persisted Logistic explanation store, instead of always
  returning a `scenario_engine`-blocked null shape. Unknown `game_id` now
  returns 404. `band`, `distribution`, and `market_implied` are unchanged,
  still blocked `scenario_engine`.
- `ExplainFactor.delta` (documented as a "percentage-point contribution,"
  never wired to real data) is replaced with `log_odds_contribution`; added
  `coefficient` and `transformed_value`; removed the unused `description`
  field. No backward compatibility preserved. `api-schema.json` and the
  generated frontend client are regenerated.
- Added `Unavailable.NO_WIN_FORECAST`, `NO_EXPLANATION_EVIDENCE`, and
  `AMBIGUOUS_EXPLANATION_EVIDENCE` field-status slugs.
- `logistic_explanation_evidence_store.py`'s `find_logistic_explanation_by_event`
  now raises a dedicated `AmbiguousLogisticExplanationError(ValueError)`
  instead of a bare `ValueError` when more than one persisted batch claims
  the same event, so the route can catch it specifically rather than
  masking a genuinely malformed-artifact `ValueError`. See `DECISIONS.md`
  D52.

#### Real-artifact validation

Called `GET /games/{game_id}/explain` through the real FastAPI app against
the real repository. The selected 2026-2027 Week 2 product's `2026_02_CAR_ATL`
returned `headline_win_prob=0.5541...` (the persisted `home_win_prob`) with
`factors` blocked `ambiguous_explanation_evidence` - the real store already
has two immutable batches both covering all 16 events of that product's run,
which `find_logistic_explanation_by_event` correctly refuses to resolve
silently. An unknown `game_id` returned 404. No `data/` artifact changed;
this unit is read-only at the API layer.

### 2026-09-26 - Persisted Logistic explanation evidence

#### Added

- `gridiron evaluate explain-logistic --run-id <run_id>`: reconstructs and
  persists, in one immutable batch, exact scaled-feature-by-coefficient
  contributions in log-odds space for every `win_prob`/`logistic` event in
  that run's prediction-input evidence. Reloads the fitted estimator's
  `coef_`/`intercept_` from its exact immutable binary snapshot, never the
  live mutable model store; performs no feature construction, scaling, or
  model inference of its own. Each event's reconstructed probability must
  reconcile with the evidence's own persisted `raw_estimator_output` within
  `abs_tol=1e-9` or the build fails closed.
- `evaluation/logistic_explanation_evidence.py`,
  `logistic_explanation_evidence_builder.py`,
  `logistic_explanation_evidence_store.py`: the contract, builder, and
  immutable JSON store, sibling to `prediction_input_evidence.py` and
  `game_model_evaluation_report.py`. Stored at
  `data/output/logistic_explanations/schema=1/batches/{batch_id}.json`, with
  no "current" selection concept - batches are keyed by run and source
  evidence, not superseded.

#### Fixed

- `prediction_input_evidence_store.py`'s scan-based lookups
  (`list_prediction_input_evidence_by_run`, `find_prediction_input_evidence_by_event`)
  crashed the entire scan on the first prediction-input evidence artifact
  whose embedded `schema_version` predates the current one - both functions
  were unusable against the real repository since U3's schema-2 bump, since
  neither had a production caller until this unit's builder became the
  first. They now skip earlier-schema artifacts with a logged warning; a
  direct `read_prediction_input_evidence` call remains exactly as strict as
  before. See `DECISIONS.md` D51.

#### Real-artifact validation

Ran `gridiron evaluate explain-logistic` against the real current
2026-2027 Week 2 development run (`deb7ebd3-5fee-4242-b58a-d86db7e4bb53`):
reconstructed all 16 `win_prob`/`logistic` events with a maximum
reconciliation error of `1.11e-16` against the persisted
`raw_estimator_output`.

---

### 2026-09-26 - Reassessed game-model champions and calibration against corrected evidence

#### Changed

- Retrained all five trainable game-model pairs (`win_prob
  logistic/random_forest/xgboost`, `total random_forest/xgboost`) through
  the gated `gridiron models train` comparison against U8's corrected
  pipeline and current `modeling_file.parquet`. `total_random_forest`
  passed its promotion gates (MAE 10.41 → 10.40) and was promoted; the
  other four challengers were correctly rejected, leaving their champions
  unchanged.
- Refreshed Win calibration (`sigma`/`margin_std`) for all four win_prob
  families from U8's aligned backfill runs.
- Re-promoted the champion manifest: `win_prob` unchanged at `logistic`,
  `total` now `random_forest`; repopulated five prop-family entries
  (`qb_pass_yards`, `qb_rush_yards`, `rb_rush_yards`, `wr_rec_yards`,
  `te_rec_yards`) that U4 had pruned for lacking a backing artifact, now
  backed by their own current archives.
- Regenerated the 2026-2027 Week 2 development product with the
  reassessed champions and confirmed `prediction_ready: True` with
  complete 16/16 coverage; regenerated the baseline report (the first
  since U4 removed the stale pre-fix reports).

#### Fixed

- `full-retrain`'s `refresh-calibrations` and `promote-champions` stages
  crashed when resumed without a same-session `backfill-game-models` run
  (the module's own documented `--only refresh-calibrations --only
  promote-champions` example), since `ctx["game_backfill_run_ids"]` is
  only populated by that stage. Both now fall back to each pair's latest
  already-persisted backfill run, matching `promote_champions()`'s
  existing resume behavior.

---

### 2026-09-25 - Restored fixed-hyperparameter walk-forward retraining; closed the current-season weather gap; regenerated the canonical six-family evaluation report

#### Added

- `gridiron ingest weather-backfill`: registers the previously-documented-but-
  never-wired `backfill_weather()` gap-filling function as a CLI command.
  Walks every completed game not already present in the observed weather
  archive and fetches the remainder from OpenWeatherMap's historical
  timemachine endpoint, skipping games already on disk; supports
  `--season-year`, `--dry-run`, and `--max-calls`.
- `models/game_prediction/base.py::GamesTrainer._fit_with_fixed_hyperparameters()`
  and `train(..., fixed_hyperparameters=...)`: fits one model directly from
  given hyperparameters and `epa_window`, with no per-call HP search.
- `evaluation/backfill.py::_load_champion_hyperparameters()`: loads the
  currently-deployed champion's own persisted hyperparameters (stripping
  bookkeeping keys) for walk-forward to retrain each season with.
- `start_season`/`end_season` support for current-model backfill: post-filters
  exported predictions by season after the full chronological simulation
  runs, letting Elo's exported game set align with the walk-forward families'.

#### Changed

- `evaluation/backfill.py`: walk-forward retraining now uses each family's
  currently-deployed champion's fixed hyperparameters and `epa_window`
  instead of re-running a full randomized HP search on every season -
  restoring conformance with `DECISIONS.md` D1, which already specified
  fixed-hyperparameter walk-forward but had silently drifted to per-season
  search. Cuts the five ML families' combined regeneration time from an
  estimated ~30 hours to roughly 70 minutes on the real repository.
- Regenerated all six game-model families' walk-forward/current-model
  backfills over the aligned 2003-2004 through 2025-2026 range (one season
  later than the walk-forward default, since 2002-2003 contains the Houston
  Texans' franchise-inaugural game, which the shared ML feature pipeline
  cannot score but Elo can - see D48) and built a new canonical
  `GameModelEvaluationReport` over the resulting verified 6,232-game common
  set, superseding the mismatched runs U7 flagged (Elo 7,276 games vs. 6,498
  for the other five).

#### Fixed

- Closed the 2026-2027 season's weather-data gap (32 games, weeks 1-2) via
  `gridiron ingest weather-backfill`; see `DECISIONS.md` D49 for the full D7
  disposition.

---

### 2026-09-25 - Comprehensive game-model evaluation metrics, cross-family report, and legacy-archive retirement

#### Added

- `evaluation/metrics.py`: Win calibration slope/intercept (unregularized
  logistic fit on `logit(p)`, ties excluded), sharpness (variance of
  predicted probabilities), and season stability (stdev of per-season
  Brier); Total median absolute error, residual-std interval coverage, and
  environment-slice metrics (dome/outdoor, cold, windy); a Total-task
  evaluation-frame builder (`build_forecast_run_total_evaluation_df`) - the
  prior forecast-store path only supported Win.
- A new immutable, run-bound, schema-versioned `GameModelEvaluationReport`
  covering all six game-model families (Win: Elo, Logistic, Random Forest,
  XGBoost; Total: Random Forest, XGBoost) on a verified common game set,
  content-addressed at `data/output/game_model_evaluation/`, sibling to (not
  an extension of) the existing betting/ROI-focused
  `HistoricalBacktestReport`. New `gridiron evaluate model-report` command
  requires an exact backfill run_id per family and fails closed if the six
  families' game sets disagree.
- `evaluation/select.py::latest_backfilled_run_id()` /
  `collect_latest_forecast_run_metrics()` / `build_latest_run_evaluation_df()`
  - single-invocation "most recent persisted run per family" resolution used
    by the CLI and API's best-effort evaluation surfaces.

#### Removed

- `evaluation/archive.py` (the append-only, overwriteable prediction log) and
  `evaluation/metrics.py::build_evaluation_df()`, its only reader. Confirmed
  `gridiron output predictions` never actually wrote to this archive despite
  the module's own docstring - the archive had no writer left in production,
  only readers, all migrated to the forecast-store-backed path.
- `evaluation/champion.py::select_game_classification_champions()` (legacy,
  archive-backed) - `promote_champions()`'s manual `--write-manifest` path
  now calls the same run-based `select_game_classification_champions_from_runs()`
  selector `full-retrain` already used, resolving each pair's latest
  backfill run itself. The two manifest-writing surfaces could previously
  disagree on which model wins.

#### Changed

- `cli/evaluate.py`'s `summary`/`calibration`/`diagnostics`/`select-model`/
  `report` commands and `api/loaders.py::load_evaluation_df`
  (`/model/performance`) now source each family's latest immutable backfill
  run instead of the legacy archive. `/model/performance`'s response
  contract, query parameters, and the frontend screens that read it are
  unchanged - see `DECISIONS.md` D46 for why a persisted-report redesign of
  that route was considered and deferred.

#### Validation

- Ruff, Pyrefly, and the full non-slow unit suite passed (4,069 tests).
  `pnpm lint`, `pnpm build`, and `pnpm test:run` (511 tests) passed after
  regenerating `api-schema.json` (a one-line description-text diff from a
  docstring change) and confirming the generated frontend client
  (`frontend/src/api/schema.ts`) is byte-identical.
- Real-artifact validation: the new metrics and report builder were run
  against the real repository's existing backfill runs. Confirmed the six
  families' *current* on-disk backfill runs do not actually share a common
  game set (Elo's two runs cover 7,276 games; the other five families' most
  recent runs share an identical, different 6,498-game set) - the new
  report's fail-closed common-game-set check correctly rejects this real
  mismatch. Verified per-family metric values against the five aligned real
  runs (e.g. Logistic Win: brier=0.223, calibration slope=0.92, AUC=0.68;
  Random Forest Total: MAE=10.8, 90%-nominal actual coverage=0.90).
  `gridiron evaluate select-model`/`summary`/`calibration`/`report` and the
  live `/model/performance` route were exercised against real data.

#### Scope

A canonical `GameModelEvaluationReport` built from all six families on the
corrected, regenerated walk-forward backfills - the real six-family success
case - is ROADMAP.md Tier 2 #5/#6's other half (Unit 8), since today's real
backfill runs don't share a common game set. This unit builds and proves the
capability; U8 regenerates the aligned backfills and produces the canonical
report from them.

### 2026-09-25 - Resolved development-role forecast candidates

#### Fixed

- `resolve_forecast_candidates` preferred `live` events over `backfilled` ones but never accounted for `development` events (added by D43). A match against development-role events fell through both role filters to an empty frame and crashed with an unhandled `IndexError` on `.iloc[0]` instead of resolving. Extended the role-preference chain to live, then development, then backfilled.

#### Validation

- Added coverage: a lone development event resolves; live still excludes a same-game development candidate; development still excludes a same-game backfilled candidate.
- Ruff, Pyrefly, and the full non-slow unit suite passed (4,052 tests, up from 4,049).

#### Scope

Found while executing Foundation Completion Track A, Unit 6 (below): the first real caller of weekly-product composition against `development`-role events. No other caller of `resolve_forecast_candidates` passes development-role events today.

### 2026-09-25 - Regenerated and reselected the canonical Week 2 development state (D45)

#### Added

- `gridiron regenerate-development-week --season --week`: generates one `development`-role forecast run from retained history, composes a weekly product from it, explicitly selects it as the current product for that season and week, and reports readiness — one atomic action, sharing the real in-memory prediction policy rather than reconstructing it after the fact.
- A shared, schedule-agnostic weekly-product composition helper (`cli/_weekly_product_composition.py`), factored out of the live `weekly-predict` pipeline so composition is not duplicated between the live and development paths.

#### Fixed

- Weekly-product composition and readiness verification both sourced the schedule exclusively from the fetch-derived upcoming-schedule snapshot, which only ever holds the current season's not-yet-played weeks. Once a week drops out of it, readiness verification reported a false `missing_schedule` blocker regardless of the selected product's actual state. Both paths now fall back to the retained-history schedule adaptation for any already-elapsed week.
- `resolve_forecast_candidates` never accounted for `development`-role events (only `live` and `backfilled`); a match against development-role events crashed with an unhandled `IndexError` instead of resolving. Found when the real Week 2 regeneration was first run. Fixed separately (see the 2026-09-25 forecast-candidate-resolution entry above).

#### Changed

- Retired the untracked ad hoc `development`-role Week 2 selection (D5) by explicit reselection to a freshly generated, repository-owned replacement produced with the D1-D4 pipeline fixes already applied. The two disposition-governed live Week 2 runs remain unchanged, still governed by their existing schema-1 disposition.

#### Validation

- Ruff, Pyrefly, and the full non-slow unit suite passed.
- Added focused coverage for the schedule fallback, the extracted composition helper, and the new command's success/failure paths; regression-tested that the live `weekly-predict` composition path's exact call chain and output are unchanged.
- Real-artifact validation: see `PLAN.md`'s Foundation Completion Track A, Unit 6 record.

#### Scope

This changes weekly-product composition and readiness verification for any already-retained week, and adds one new CLI command. It does not change `weekly-predict`'s live-only, upcoming-week-only contract, the API, or the frontend.

### 2026-09-22 - Preserved exact full-retrain lineage (D44)

#### Changed

- `full-retrain` retains immutable forecast run identities for every game-model backfill, instead of allowing later stages to reference backfilled evidence by convenience or modification time.
- Win-model calibration refresh now uses exclusively the exact backfilled forecast runs produced by the same invocation.
- Classification champion ranking uses explicit immutable-run evidence: each candidate model type's metrics come from one specific backfill `run_id`, summed across Brier, ECE, and AUC rank.
- Total champion selection continues to use each freshly trained artifact's own persisted holdout metadata, unchanged.
- Legacy archive-based evaluation is preserved for non-authoritative reporting surfaces only.
- Missing, empty, mismatched, duplicate, or malformed lineage now fails closed rather than silently substituting a different run.

#### Validation

- Added focused evaluation, champion-selection, and full-retrain orchestration tests.
- Ruff, Pyrefly, focused tests, and the full unit suite passed.

#### Scope

This changes only `full-retrain`'s classification champion-selection and calibration-refresh stages. It does not change the gated `gridiron models train` comparison path, forecast events, weekly products, the API, or the frontend.

### 2026-09-22 - Added development forecast role (D43)

#### Added

- Added `ForecastRole.DEVELOPMENT`, a role for retrospectively generated canonical weekly fixtures, distinct from `live` (pre-kickoff operational issuance) and `backfilled` (historical evaluation reconstructions).
- `development` events can carry exact immutable prediction-input evidence and participate in one role-coherent selected weekly product, on the same terms as `live`.
- Selected-event postgame closeout authenticates against the exact role the weekly product actually carries and reports Win and Total roles explicitly, instead of assuming `live`.
- Added a dedicated `development` weekly execution boundary (`execute_development_weekly_prediction_policy`), separate from the live wrapper the normal weekly command uses.

#### Changed

- Weekly products now require one coherent role (`live` or `development`) across every available selected component and game; mixed-role products are rejected.
- Derived Spread production provenance explicitly inherits the live-role requirement from its source Win forecast.

#### Validation

- Added role, evidence, product, closeout, execution, qualification, production-proof, and disposition-preservation coverage.
- Confirmed live-only recommendation qualification, candidate issuance, and production-chain proof remain isolated from `development` evidence.
- Confirmed the existing Week 2 known-defect disposition and Week 3 schema-1 evidence remained valid and unaffected.
- Before-and-after SHA-256 inventories of forecast events, weekly products, and prediction-input evidence were byte-identical.
- Ruff, Pyrefly, the full unit suite, and focused integration tests passed.

#### Scope

The normal weekly prediction command (`weekly-predict`) remains live-only; it does not call the new development wrapper. No production caller of the development boundary exists yet — see `ROADMAP.md` Foundation Completion Track A, Unit U5, for the repository-owned generation command this role still needs.

### 2026-09-22 - Aligned statistical availability with execution metadata validation

#### Changed

- Weekly statistical-model availability now requires the persisted feature-set
  identity to match the exact registered prediction feature contract.
- Availability now requires the persisted modeling schema version to be a
  non-Boolean integer equal to the current modeling schema version.
- The existing exact ordered feature-column requirement remains part of the
  same metadata eligibility contract.
- Missing or stale contract metadata makes only the affected statistical family
  unavailable before policy selection.
- Malformed model identity, artifact kind, and task remain explicit errors.
- Live prediction execution retains its independent strict metadata validation.

#### Validation

- Added coverage for missing and incorrect feature-set identity.
- Added coverage for missing, null, Boolean, string, and stale modeling schema
  versions.
- Added family-isolation coverage proving stale Logistic metadata does not make
  a valid Total family unavailable.
- Preserved coverage proving availability does not deserialize estimators or
  scalers.
- Focused availability, policy, weekly-execution, and weekly-CLI validation
  passed with 84 tests.
- Ruff, Pyrefly, Python compilation, the full non-slow unit suite, and
  `git diff --check` passed.
- Real 2026 Week 2 availability inspection reported Elo and all five current
  statistical families available.
- Before-and-after SHA-256 inventories for model, metadata, and scaler artifacts
  were identical.
- Independent read-only review approved the unit for closure.

#### Scope

This change affects only read-only statistical availability preflight. It does
not modify estimators, metadata artifacts, prediction execution, forecast
events, prediction-input evidence, weekly products, APIs, frontend contracts,
or persisted schemas.

### 2026-09-21 - Persisted immutable prediction-input evidence for new live weekly forecasts (Prediction Input Integrity Unit 5)

#### Added

- Added schema-1 family evidence for statistical and Elo live execution.
- Added deterministic evidence identity, bounded source references, feature-schema identity, exact event-level inputs and outputs, content-addressed binary snapshots, identity-addressed evidence persistence, deterministic lookup, and source-drift rejection.
- Added evidence-aware execution results for statistical and Elo families.

#### Changed

- Weekly prediction now requires a clean tracked Git revision before selected-family execution.
- Statistical execution validates model, task, feature-set, modeling schema, and ordered feature columns before estimator invocation.
- Deployable metadata now records the authoritative feature-set name.
- Weekly execution returns evidence without persistence; the CLI publishes snapshots and evidence, strictly reloads them, and only then persists events.
- Existing forecast-event and weekly-product schemas remain unchanged.

#### Validation

- Domain, store, execution, weekly orchestration, and CLI tests cover strict validation, immutable publication, concurrency, tamper rejection, exact family coverage, and failure ordering.
- Legacy-versus-evidence output equivalence passed for all five statistical families.
- Protected real-artifact validation used 2026 Week 2 in a clean temporary repository. Default Logistic Win plus Random Forest Total and Elo Win override plus Random Forest Total both passed.
- Statistical snapshot replay matched raw outputs with `rtol=0.0` and `atol=1e-12`.
- Independent review found no blocking correctness or safety defect. Requested equivalence and XGBoost coverage were added and passed.
- Ruff, Pyrefly, compilation, focused suites, the full non-slow unit suite, and `git diff --check` pass.

#### Operational evidence

- Elo reconstruction restored the ignored lineage sidecar while preserving Elo CSV digest `7818da246ca5c248f4cdad2341cf0f6cbd4753f6843357093821e2ab4e5fa861`.
- Canonical games retained digest `32c23ed2c75895f9e1466893020ed41ac4c261fe4c2cffb5cb372976c97a6efc`.
- Five ignored statistical artifacts were regenerated through the repository-owned training stage and now contain current feature-set names, schema version 5, and exact feature columns.
- Protected forecast events, both affected Week 2 products, `index.json`, and `current.json` remained byte-identical.

#### Follow-up

Availability checks feature columns but not persisted feature-set identity or modeling schema version as strictly as execution. Execution remains fail-closed, so this is nonblocking.

#### Scope

Applies only to newly generated live weekly events. Existing live and backfilled events remain unchanged. No corrected Week 2 forecast, product replacement, corrected evaluation, explanation evidence, API change, or frontend change was introduced.

See `DECISIONS.md` D42.

### 2026-09-19 - Recorded and enforced affected forecast evidence disposition (Prediction Input Integrity Unit 4)

#### Added

- Added a strict schema-1 forecast-evidence disposition contract with one
  supported known-defect classification for incomplete Elo source history.
- Added deterministic SHA-256 identity over the complete canonical disposition
  payload.
- Added strict validation for schema fields, enum values, UTC timestamps,
  season and week scope, digests, sorted unique identity collections, selected
  product membership, and embedded identity.
- Added authentication of exact affected forecast runs, events, weekly
  products, derived-spread provenance, Total-evidence absence, and scoped
  current selection.
- Added identity-addressed immutable JSON persistence under
  `data/output/forecast_evidence_dispositions/schema=1/dispositions/`.
- Added create-only atomic publication, idempotent exact replay, strict loading,
  canonical-path validation, deterministic listing, and concurrent conflict
  protection.
- Added `known_defective_forecast_evidence` to weekly readiness, edge
  diagnostics, API unavailable metadata, generated OpenAPI, and frontend edge
  status presentation.

#### Changed

- Weekly readiness now resolves exact dispositions for the selected product and
  adds the known-defective blocker without replacing independent schedule,
  prediction, or market blockers.
- Weekly edge calculation now returns an explicit blocked result with zero rows
  before market loading or edge calculation when the selected product is known
  defective.
- Candidate issuance now rejects an affected selected product before loading
  forecast events, loading quote history, deriving an as-known quote set,
  evaluating candidates, or writing an issuance.
- Manual prediction rendering now rejects an affected selected product before
  display adaptation or PNG and HTML writes.
- The edges API now mechanically maps the new edge blocker to stable unavailable
  metadata.
- The generated frontend edge-blocker type and exhaustive presentation mapping
  now include known-defective forecast evidence.
- Low-level forecast-event and weekly-product loaders remain unchanged for
  historical audit and postgame evaluation.

#### Operational evidence

- Recorded one immutable disposition at
  `2026-09-19T18:00:00+00:00`.
- Disposition identity:
  `05f36ea9f3f014c3ed2b9bd2f2586540189ac8fea3dcf5db3b354ad020565eeb`.
- Persisted artifact SHA-256:
  `8529487ed56c5130eba032c2e8d1764b22e95f796b6336a32ac299266746318d`.
- The disposition covers two affected live logistic Win runs, all 32 affected
  forecast events, both affected weekly products, and the exact selected
  affected product.
- The affected components are Win probability and derived spread.
- No Total event, run, model, type, or role evidence is classified as affected.
- The disposition artifact is operationally persisted under the ignored
  `data/` tree and is intentionally not committed under current repository
  policy.

#### Validation

- Domain tests prove deterministic identity, strict validation, evidence
  authentication, applicability, operational rejection, and ambiguity errors.
- Store tests prove round-trip persistence, strict schema loading, canonical
  path enforcement, exact replay, concurrent exact replay, and rejection of
  conflicting concurrent publication without overwrite.
- Readiness tests prove the blocker makes `ready` and `prediction_ready` false
  while market readiness remains independently derived.
- Edge tests prove the affected product returns a distinct known-defect blocked
  result with zero rows and no market loading or edge calculation.
- Candidate-issuance tests prove rejection occurs before every downstream read
  or write.
- Rendering tests prove rejection occurs before display adaptation and output
  writes.
- API and frontend tests prove stable blocker mapping and presentation.
- Protected real-artifact validation authenticated both runs, all 32 events,
  both products, and the scoped selection through public loaders and a
  temporary repository.
- Protected validation proved strict reload, exact replay, tamper rejection,
  operational blocking, unrelated-product eligibility, and byte-identical
  source preservation.
- The working-repository disposition was then persisted through the same public
  store boundary.
- Real `verify-week` output reports
  `known_defective_forecast_evidence` alongside independent existing blockers.
- Real weekly edge calculation returns `blocked`, the exact known-defect
  blocker, and zero rows.
- Forecast events, both affected product Parquet files, `index.json`, and
  `current.json` retained their protected SHA-256 identities.
- Ruff, Pyrefly, the full non-slow unit suite, focused Unit 4 suites, OpenAPI
  schema consistency, one-shot frontend tests, and the frontend production
  build pass.
- An independent read-only review confirmed the complete operational consumer
  inventory and approved Unit 4 for documentation closure.

#### Scope

This unit records and enforces the known defect in the original immutable 2026
Week 2 Win probability and derived-spread evidence. It does not generate a
corrected forecast, create or select a replacement product, persist complete
prediction-input rows, run corrected model-performance evaluation, add model
explanations, or change the Games API contract.

The Games API and frontend continue to serialize and display the affected
selected product without disposition metadata. Postgame closeout continues to
read that product as immutable historical evidence of what was actually
selected and predicted. The disposition governs readiness, edge calculation,
candidate issuance, and manual prediction rendering.

See `DECISIONS.md` D41.

## 2026-09-18 - Verified Elo lineage before weekly prediction (Prediction Input Integrity Unit 3)

### Added

- Added strict schema-1 Elo lineage at
  `data/cleaned/NFL_Team_Elo.metadata.json`.
- Lineage records the exact repository-relative path, persisted-byte SHA-256
  digest, row count, ordered columns, season scope, and latest week for the
  canonical games source and reconstructed Elo output.
- Added strict serialization and loading rules for schema fields, timestamps,
  digests, counts, columns, season labels, and repository-contained paths.
- Added current-artifact verification for valid, missing, stale, malformed, and
  unsupported lineage states.

### Changed

- Every successful `fit_elo()` execution now writes lineage after persisting the
  reconstructed Elo CSV.
- Weekly prediction availability now authenticates current games and Elo
  artifacts before policy resolution and model execution.
- Elo-only prediction requires both verified lineage and exact-week Away and
  Home Elo coverage.
- Trained-model lineage requirements are derived from each registered model's
  exact prediction feature contract.
- All five current trained game models are confirmed to require `AWAY_ELO`,
  `HOME_ELO`, and `ELO_DIFF`.
- Missing or stale lineage makes every current trained Win and Total model
  unavailable.
- Malformed or unsupported lineage remains an explicit error.
- A future model without an Elo dependency remains independently eligible when
  its own required inputs are complete.

### Validation

- Unit coverage proves exact-byte digest calculation, deterministic lineage
  serialization, strict schema loading, safe path resolution, and explicit
  type and timestamp validation.
- Unit coverage proves that missing lineage and missing referenced artifacts
  are unavailable.
- Unit coverage proves that changed games content, changed Elo content, changed
  row counts, changed columns, and changed season scope invalidate lineage.
- Unit coverage proves malformed JSON, unsupported schemas, malformed digests,
  invalid timestamps, unsafe paths, invalid counts, and unexpected fields raise
  explicitly.
- Integration coverage proves successful public Elo reconstruction writes
  matching lineage and repeated reconstruction preserves source and output
  content identities.
- Real-registry coverage proves all five current trained game-model feature
  contracts directly require `AWAY_ELO`, `HOME_ELO`, and `ELO_DIFF`.
- Availability coverage proves verified lineage does not override missing
  exact-week Elo coverage and that a synthetic non-Elo model contract remains
  independently eligible.
- Weekly-execution coverage proves unavailable families stop before model
  registry execution lookup.
- Weekly CLI coverage proves unavailable and malformed lineage write no forecast
  events and cache no forecast-run identity.
- Ruff, Pyrefly, and the full non-slow unit test suite pass.
- Protected validation wrote and strictly loaded schema-1 lineage in a temporary
  repository.
- Protected lineage identified 7,292 canonical games from `1999-2000` through
  `2026-2027`, latest completed week 1, with SHA-256 digest
  `32c23ed2c75895f9e1466893020ed41ac4c261fe4c2cffb5cb372976c97a6efc`.
- Protected lineage identified 19,006 Elo rows from `1999-2000` through
  `2026-2027`, latest state week 2, with SHA-256 digest
  `7818da246ca5c248f4cdad2341cf0f6cbd4753f6843357093821e2ab4e5fa861`.
- Valid lineage made Elo and all five current trained model contracts available
  for the complete 16-game 2026 Week 2 schedule.
- Changing one games value, changing one Elo value, or removing lineage made all
  six availability facts false.
- Malformed lineage raised explicitly.
- Blocked weekly execution stopped after model-contract inspection and before
  model prediction.
- Working games, Elo, schedule, model, champion, forecast, and weekly-product
  artifacts remained read-only.
- Selection, market, edge, API, and frontend artifacts were not touched.

### Scope

This unit authenticates persisted Elo lineage before weekly model execution. It
does not change affected Week 2 artifacts, generate or select retrospective
forecasts, persist complete prediction inputs, evaluate corrected predictions,
alter API or frontend contracts, or add model explanations.

See `DECISIONS.md` D40.

## 2026-09-18 - Validated complete-history Elo reconstruction (Prediction Input Integrity Unit 2)

### Fixed

- Replaced the false incremental Elo lifecycle with one deterministic
  reconstruction from validated complete canonical game history.
- Elo reconstruction now rejects history that is empty, malformed, duplicated,
  late-starting, or missing intermediate seasons before replacing the
  registered Elo artifact.
- The specific 2026 Week 1-only input shape that caused the inspected Week 2
  reset is now rejected because it begins after the canonical 1999 history
  floor.
- Validation or simulation failure occurs before the Elo writer is called, so
  an existing Elo artifact remains unchanged.

### Changed

- Removed `update_elo_state_incremental()`.
- Removed the `all_years` parameter from `fit_elo()`.
- Removed `fit_elo_all_years` from shared pipeline orchestration and all
  composite callers.
- Removed `--fit-elo-all-years` from `run-data-pipeline`.
- Removed incremental and all-years modes from `ratings elo fit`.
- `ratings elo fit` and every `build-elo` pipeline stage now execute the same
  complete-history reconstruction contract.
- `run-data-pipeline --all-years` continues to govern source-data and modeling
  scope, not a separate Elo mode.
- Added dedicated complete-history Elo fixtures while preserving the original
  two-game modeling-fixture behavior.

### Validation

- Complete-history input validation covers required fields, empty history,
  duplicate and missing game identities, invalid team identities, identical
  Away and Home teams, malformed season labels, late-starting history, missing
  intermediate seasons, invalid weeks, incomplete score pairs, unplayed games,
  and negative scores.
- Integration coverage proves deterministic reconstruction, valid canonical
  output, inherited prior-season strength, Week 1-to-Week 2 updates, and
  byte-identical predecessor preservation after partial-history rejection.
- CLI coverage proves that `ratings elo fit` exposes one reconstruction
  operation and retired mode flags are absent.
- Feature-pipeline and prediction-pipeline tests use complete history to build
  Elo while retaining their focused two-game modeling inputs. Both modeling
  rows have complete numeric Elo coverage.
- Ruff, Pyrefly, and the full non-slow unit test suite pass.
- Relevant Elo, simulator, integration, end-to-end, pipeline, and composite CLI
  tests pass.
- Protected validation reconstructed 19,006 Elo rows from 7,292 canonical
  completed games spanning 1999 through 2026, with zero duplicate
  team-season-week identities.
- Protected validation reproduced Buffalo Week 1 and Week 2 ratings of
  `1566.257299` and `1575.655543`.
- Protected validation reproduced Detroit Week 1 and Week 2 ratings of
  `1538.997399` and `1546.800161`.
- A deterministic second reconstruction produced identical output.
- The exact 16-row 2026 Week 1-only input was rejected.
- The valid temporary Elo artifact remained byte-identical after rejection.
- No working-repository data, forecast, product, selection, model, market, API,
  or frontend artifact changed during protected validation.

### Scope

This unit corrects Elo reconstruction and source-history validation. It does
not add semantic weekly readiness, change affected Week 2 products, regenerate
forecasts, persist exact prediction inputs, change model evaluation, alter API
or frontend contracts, or add model explanations.

See `DECISIONS.md` D39.

## 2026-09-18 - History-preserving weekly games refresh (Prediction Input Integrity Unit 1)

### Fixed

- Corrected the shared data-pipeline fetch selection so recurring
  non-all-years execution refreshes the requested nflverse season through
  `fetch_nflverse_games_refresh()` rather than replacing the registered raw
  games artifact with `fetch_nflverse_games(seasons=[...])`.
- Explicit-season weekly refresh now preserves every unrequested historical
  season while replacing the requested season exactly once.
- The explicit all-years path retains its full-replacement behavior.

### Added

- Shared-pipeline regression coverage proves:
  - all-years execution uses the explicit replacement boundary;
  - explicit-season recurring execution uses the history-preserving refresh
    boundary with the requested season;
  - recurring execution without an explicit season delegates current-season
    resolution to the history-preserving refresh wrapper.
- Direct `weekly-predict` stage coverage proves that the composite supplies the
  explicit recurring-refresh arguments exercised by the corrected shared
  branch.
- Unchanged `post-week` regression coverage confirms compatibility with the
  corrected shared pipeline boundary.

### Validation

- Focused nflverse ingest, games cleaning, shared CLI, `weekly-predict`, and
  `post-week` tests pass.
- Explicit failure tests prove that an upstream fetch exception and an empty
  selected-season response both preserve the existing raw artifact
  byte-for-byte. Empty responses now fail before any artifact write.
- Ruff, Pyrefly, and the full non-slow unit test suite pass.
- Protected validation exercised the public
  `fetch_nflverse_games_refresh(season=2026)` wrapper in a temporary
  repository.
- Protected validation preserved 7,276 unrequested raw rows across 27
  unrequested seasons, replaced the complete 272-row 2026 target slice,
  restored deliberately removed game `2026_18_TEN_HOU`, and retained zero
  duplicate raw game IDs.
- The protected source contained 28 total seasons from 1999 through 2026.
- Cleaning the refreshed protected artifact produced 7,292 unique completed
  games across every season from `1999-2000` through `2026-2027`, with zero
  duplicate cleaned game IDs.
- No Elo, model prediction, forecast-event, weekly-product, selection, edge, or
  frontend workflow ran during protected validation.

### Scope

This correction preserves historical games during recurring refresh. It does
not redesign Elo reconstruction, change semantic model readiness, alter or
reselect affected 2026 Week 2 forecast evidence, persist prediction feature
rows, change API or frontend contracts, or add model explanations.

See `DECISIONS.md` D38.

## 2026-08-26 - Small API and documentation cleanup (WS2 Unit 7)

### Fixed
- `api/routes/portfolio.py::get_portfolio_splits`'s empty-ledger branch now
  delegates to `serialize_splits(pd.DataFrame(), dimension)` instead of
  directly constructing `PortfolioSplits(...)`, restoring the
  `_meta.field_status`/`NO_SPLIT_DATA` metadata the direct-construction
  path was silently omitting — a confirmed D18 violation (serializers, not
  route handlers, own `_meta.field_status` construction).
- `api/serializers/portfolio.py`'s module docstring no longer misattributes
  its no-I/O, already-loaded-DataFrames behavior to D19, which actually
  governs `api/loaders.py`'s settings-threading — a different concern. The
  factual description is retained without an incorrect citation.

### Changed
- Removed development-era "Unit 22"/"Unit 24" phase-naming from seven
  confirmed instances across `recommendation_policy.py`,
  `betting/ledger.py` (three instances, not one — `log_bet`'s own `Args:`
  block carried two previously-undercounted mentions),
  `api/schemas/portfolio.py`, `api/schemas/recommendations.py`, and one
  test function name in `test_recommended_bet_result.py`. Each instance
  was reviewed individually and replaced with accurate domain language, not
  a blind find-and-replace.
- `api-schema.json` and `frontend/src/api/schema.ts` regenerated: item 1's
  route-behavior fix changed the live OpenAPI contract for the empty-splits
  response.

### Process note
Phase-naming cleanup (item 3) was deliberately held blocked mid-unit until
fresh, current source for `recommendation_policy.py` and `betting/ledger.py`
was supplied, rather than proceeding from a months-old Boundary 2
characterization — consistent with the correction discipline this
workstream established in Units 2 and 4. A stray copy/paste path-header
comment, carried into the diff from how a file was pasted during design,
was caught during review and removed before commit.

### Verification
- Ruff, Pyrefly, and the full unit test suite pass, including the OpenAPI
  schema-parity test after regeneration.
- Frontend `pnpm gen:api && pnpm build` passes.

### Workstream 2 status
This closes the seven-unit sequence originally planned in ROADMAP.md
(persistence hardening: Units 1-3; identity-evolution: Unit 4; capability
protocol: Unit 5; attribution ownership: Unit 6; this cleanup: Unit 7).
Two items remain deliberately open per Unit 5's own closure (D35, D36):
forward-impact discoverability and general validity/invalidation beyond
Unit 4's single-field precedent. Workstream 2 does not close until those
are addressed by a future, evidence-gated unit.

## 2026-08-26 - Attribution-operation ownership (WS2 Unit 6)

### Changed
- `market_family_evaluation.py::_closeout_matches` now re-derives the
  canonical candidate reference via `candidate_issuance_row_id(issuance.issuance_id, row)`
  and compares it directly against the persisted reference's `reference_id`,
  instead of checking an issuance-ID prefix plus 11 individually-compared
  materialized fields (every field the v1 hash payload covers, checked
  redundantly, without ever confirming the digest suffix itself matched).
  `reference_kickoff` — the one field outside the hash payload — is
  retained as a separate, explicit check.
- `test_market_family_evaluation.py`'s `_closeout` fixture helper now
  constructs genuine digest-backed references instead of a fake test
  suffix (`f"{issuance.issuance_id}:test-{row.market}-{row.side}"`), which
  the corrected function would otherwise (correctly) reject.

### Added
- `test_closeout_with_mismatched_digest_but_matching_fields_does_not_match`
  — proves a reference with correct materialized fields but a forged
  digest suffix is no longer attributed, closing the specific ambiguity
  this unit fixes.
- `DECISIONS.md` D37 — formally names and classifies seven
  reference-attribution operations (previously six, per Boundary 4;
  `bet_reference_matching.py::match_bet_references` confirmed this session
  as a genuine seventh operation, not a duplicate). Splits them into
  canonical authentication (digest-backed, 1:1) and structural attribution
  (group/aggregate, no digest) families, and documents why
  `_history_matches`/`_wager_return_for_row` correctly have no equivalent
  gap to `_closeout_matches`'s — they answer structurally different
  questions.

### Verification
- Full source of `market_closeout.py`, `bet_reference_matching.py`,
  `market_family_evaluation.py`, and all three focused test files read in
  full before any change (correcting an earlier reliance on a months-old
  Boundary 4 characterization).
- Ruff, Pyrefly, and the full unit test suite pass. All pre-existing
  `test_market_family_evaluation.py` tests pass unchanged, confirming none
  of them depended on the fake suffix's specific content — only on
  downstream evaluation behavior.

### See also
`DECISIONS.md` D37 — full seven-operation taxonomy and the rationale for
the `_closeout_matches` correction.

## 2026-08-26 - Identity-evolution contract for candidate references (WS2 Unit 4)

### Changed
- `candidate_issuance_row_id` gained an independently owned derivation
  version, separate from `CandidateIssuance`'s own artifact schema.
  Version 1 is the exact pre-existing algorithm, extracted verbatim with no
  version marker in its hashed payload — the version selects which
  implementation runs, it is not an input to that implementation's digest,
  so every reference explicitly derived under v1 remains stable.
- Added `UnsupportedCandidateReferenceVersionError` (a `ValueError`
  subtype), propagated directly and uncaught, so a recorded
  reference-derivation version with no known implementation is
  structurally distinct from ordinary evidence-content corruption.
- `RECOMMENDED_BET_RESULT_SCHEMA_VERSION` incremented from 1 to 2.
  `RecommendedBetResult` gains `candidate_reference_derivation_version`,
  recorded at construction time. `validate_recommended_bet_result`
  dispatches re-derivation through the recorded version, not a hardcoded
  current-version comparison.
- `recommended_bet_result_store.py`, `market_closeout.py`,
  `recommendation_policy.py`, and `tests/fixtures/recommended_bet_results.py`
  required no code changes — confirmed by direct repository search
  (`rg -n 'RecommendedBetResult\(' src tests frontend`), not assumed.
- Recommendation API offer provenance now exposes
  `candidate_reference_derivation_version`, copied mechanically by the
  serializer. `api-schema.json` and the frontend's generated `schema.ts`
  regenerated to match; one frontend test fixture updated for the new
  required field.

### Verification (two full review rounds)
- First round: confirmed core design; caught two docstring lint
  violations, the stale OpenAPI snapshot, and a frontend fixture that
  failed to compile against the regenerated contract.
- Second round: found the initial closure diff incomplete —
  `test_recommended_bet_result_store.py` (schema-2 round-trip, namespace,
  missing-field rejection, and schema-1-shaped rejection tests) was
  missing entirely; no direct proof existed that no test/fixture
  constructs the dataclass positionally (resolved via repository search);
  the Python serializer had no direct runtime assertion (added); one new
  evaluation-identity test was found to duplicate existing coverage
  without proving what its name claimed and was replaced with a correctly
  narrow identity-function test; the dispatcher docstring was found too
  broad and narrowed; invalid-version test coverage was expanded to
  include non-integer runtime values; a missing trailing newline in the
  regenerated `api-schema.json` was caught and corrected.
- Ruff, Pyrefly, the full Python unit test suite, and the frontend
  TypeScript build all pass after both rounds' corrections.
- Real-data regeneration: all schema-1 development artifacts regenerated
  under schema 2 via the production-chain CLI against real 2026-2027
  season week 1 data (1,680 quote observations, 698 issued candidates, all
  698 results passing validation with zero errors); old `schema=1/` tree
  deleted. All 698 regenerated results were `result_state=unavailable` (no
  historical outcome data available in this environment) — this exercises
  the schema/dispatch machinery end-to-end but not the reference-mismatch
  validation branches, which remain covered by dedicated unit tests.

### See also
`DECISIONS.md` D31 — records the independently-versioned reference
derivation, the schema-2 increment, the replace-and-regenerate policy, and
revisit triggers.

## 2026-08-25 - Bet-ledger writer coordination (WS2 Unit 3)

### Changed
- `betting/ledger.py` gained a module-level `threading.RLock` (`_LEDGER_LOCK`),
  held across the complete read-modify-write sequence in `log_bet` and
  `settle_bet`.
- `betting/recording.py::record_wager` holds the same lock across its full
  snapshot/ledger-write/bankroll-write/restore sequence, so a rollback
  cannot discard a concurrently-completed ledger mutation from another
  thread. An `RLock` (not a plain `Lock`) was required because
  `record_wager` calls `log_bet` internally while already holding the lock.
- Module docstring states the coordination boundary explicitly: this lock
  coordinates threads within one process only. It provides no protection
  across multiple worker processes or a CLI invocation running alongside
  the API.

### Evidence basis (not assumed)
- No existing locking utility or `DECISIONS.md` entry governs ledger
  concurrency (confirmed via owner-run local `rg`/`grep`).
- The CLI bet commands are never used; the API runs as a single process
  (owner-confirmed).
- `api/routes/portfolio.py::record_portfolio_bet` is a sync `def`, routed by
  FastAPI/Starlette through a thread pool — confirmed by this project's own
  earlier traceback showing `run_in_threadpool` → `anyio.to_thread.run_sync`.
  Two near-simultaneous requests genuinely execute as two threads of one
  process, which is the actual, evidenced risk this unit closes.

### Verification
- Ruff, Pyrefly, and the full unit test suite pass.
- New tests: two overlapping `log_bet` calls both survive; a `log_bet`
  overlapping a `settle_bet` both survive; a failed `record_wager` rollback
  does not discard a concurrent, independently-completed ledger mutation.
  Each test induces a real race window (a monkeypatched read-delay) that
  would cause a lost update without the lock.
- All Unit 1 (immutable-artifact publication) and Unit 2 (ledger atomic
  publication) behavior, plus all pre-existing ledger/settlement/schema/
  single-call-rollback tests, pass unchanged.

### See also
`DECISIONS.md` D27 — records the chosen intra-process locking mechanism,
its evidence basis, and the explicit condition under which it must be
revisited.

## 2026-08-25 - Bet-ledger atomic publication (WS2 Unit 2)

### Changed
- `betting/ledger.py::_write_ledger` now stages every complete ledger
  rewrite to a colocated temporary file and publishes via `os.replace`,
  instead of calling `to_parquet` directly on the canonical path.
  Direct-path writing was confirmed (by tracing pandas' pyarrow write path
  and reproducing empirically) to truncate the destination to zero bytes
  synchronously before any row is serialized — so any interruption during
  any write previously destroyed the entire prior ledger, not just the row
  being added. An interruption at any point now leaves the prior valid
  ledger completely unchanged, whether or not a ledger already existed.
- Module docstring corrected from "append-only" to an accurate description
  of the mutable-ledger contract (settlement mutates existing rows; the
  complete file is rewritten and republished on every write), and now
  explicitly discloses that this store does not coordinate overlapping
  writers.

### Explicitly deferred, not silently dropped
- **Writer coordination ("Guarantee B")** — two overlapping `log_bet`/
  `settle_bet` calls, or a `recording.py` compensating rollback racing a
  concurrent write, can still silently lose an update. An earlier draft of
  this unit incorrectly claimed this was solved by atomic publication alone
  and was corrected before implementation. Tracing `recording.py::record_wager`
  confirmed any future fix cannot be scoped to `_write_ledger` alone: its
  rollback can discard a second writer's completed mutation regardless of
  per-write atomicity. Recorded as an open item for a future unit, requiring
  its own design (locking, optimistic concurrency, or an enforced
  single-writer boundary) and its own `DECISIONS.md` entry.

### Verification
- Ruff, Pyrefly, and the full unit test suite pass.
- New tests prove: an existing ledger survives a temporary-serialization
  failure and a pre-publication failure byte-for-byte; a first-ever write
  that fails during serialization leaves the canonical path absent (not
  corrupt); a successful write fully replaces the destination with exact
  schema and no leaked temporary file.
- `recording.py`'s existing compensating-rollback tests and all pre-existing
  ledger tests (schema, model identity, recommendation provenance,
  reference-offer provenance, settlement, filtering) pass unchanged.

## 2026-08-25 - Immutable artifact publication hardening (WS2 Unit 1)

### Changed
- Five immutable JSON persistence modules (recommendation policy,
  recommendation governance, production-chain preflight, recommended-bet
  result/evaluation, candidate issuance) now publish via create-only,
  atomically visible linking (`os.link`), matching the proven pattern
  already shipped in `collection_receipt_store.py` (WS1 Unit 4), instead of
  `Path.replace()`, which unconditionally overwrites an existing
  destination.
- `FileExistsError` from the publish attempt is now the authoritative
  concurrency signal (a competing writer's destination can appear at any
  time up to the publish call), replacing a `path.exists()` pre-check that
  left a race window between check and replace.
- `candidate_issuance_store.py` gained a colocated temporary-file
  serialization stage; it previously wrote directly into its final path,
  which was create-only but not crash-atomic.
- Each store's pre-existing replay-equality contract (byte-serialized
  comparison for governance, preflight, and recommended-result; reconstructed
  domain-object comparison for policy and candidate issuance) is unchanged —
  only the publish mechanism was hardened.

### Investigated, not changed
- `collection_plan_store.py`'s two writers (`select_current_collection_plan`,
  `write_collection_plan`) were classified against their real callers and
  tests and confirmed to be intentionally replaceable scoped state, not
  create-once identity-addressed artifacts. Excluded from this unit with no
  runtime behavior change; corrects an earlier inspection overclassification.

### Verification
- Ruff, Pyrefly, and the full unit test suite pass.
- New race-specific tests per hardened store prove the actual defect closed:
  a competing writer's conflicting content is never overwritten and the
  store's existing conflict error fires; identical concurrent content is
  accepted as an idempotent replay; a pre-publication failure leaves neither
  a partial destination nor a leaked temporary file. Candidate issuance
  additionally has a temporary-serialization-failure test confirming
  `os.link` is never reached if serialization itself fails.
- No store's round-trip, idempotent-replay, or conflicting-replay tests
  changed in observable behavior.


## 2026-08-24 - Quote observation workstream complete

All four planned units for the quote-observation workstream are complete:
point-in-time cutoff-visible retrieval, exact candidate-reference identity,
truthful coverage counts, and collection claim/receipt lifecycle robustness. The
existing quote-observation substrate required no wholesale replacement; every
verified gap the initial inspection found is closed. Store multi-writer safety,
provider-label/event stability, and the descriptive/event-time trust boundary
remain documented, deliberately deferred open items.

## 2026-08-24 - Collection claim and receipt lifecycle robustness

### Changed
- A lost claim-creation race (two collection processes contending for the same
  planned poll) now resolves as the existing CLAIMED outcome instead of an
  uncaught FileExistsError crash.
- Claim and result publication is now crash-atomic as well as create-only: both
  write to a temporary file beside the destination, then publish through a hard
  link, which raises on an existing destination and never exposes a partially
  serialized file.
- An unexpected exception during ingestion after a claim is created is now
  recorded as an explicit UNEXPECTED_FAILURE terminal result rather than leaving
  the claim unresolved with no record.

### Verification
- Ruff, Pyrefly, and the unit test suite pass, including new coverage for the true
  lost-claim-creation race, an unexpected post-claim failure producing a persisted
  terminal result, and an interrupted serialization leaving no partial or leaked
  temporary file.
- No automatic retry, reclaim, lease, or expiry of an already-unresolved claim was
  introduced; the case that cannot be truthfully recorded (the terminal write
  itself failing) remains a surfaced, degraded, unresolved claim, unchanged.

## 2026-08-24 - Truthful quote-history coverage counts

### Changed
- Quote-history coverage now reports a genuine pregame count. `pregame_observation_count`
  previously equalled non-live rows and never compared `fetched_at` to `commence_time`,
  so a non-live observation collected at or after kickoff was mis-reported as pregame.
  It now counts only non-live observations with a known kickoff and
  `fetched_at < commence_time`.

### Added
- `non_live_at_or_after_kickoff_observation_count` on the coverage contract: non-live
  observations with a known kickoff collected at or after it, so late evidence is
  surfaced rather than hidden. `live_observation_count` and `missing_commence_time_count`
  retain their meaning and are independent diagnostics that may overlap (a live row
  with a missing kickoff increments both); the counts do not partition the rows.

### Verification
- Ruff, Pyrefly, and the unit test suite pass.
- Real Week 1 ledger (read-only, checksum-guarded): row_count 1680, pregame 1680,
  non_live_at_or_after_kickoff 0, live 0, missing_commence 0; source parquet SHA-256
  unchanged. Every observation is genuinely pregame there because ingest excludes
  started and live events; the corrected count therefore equals row_count with no
  regression, and the late/missing-kickoff behavior is proven by focused tests.

## 2026-08-24 - Candidate reference exact over canonical observation identity

### Changed
- The cross-artifact candidate reference now hashes the complete canonical
  observation identity, including `sportsbook_updated_at` and `is_live`, which were
  previously omitted. The reference is now injective over the canonical observation
  identity: two canonically-distinct observations can no longer collapse to the same
  reference. The external `issuance_id:sha256` reference shape and issuance scope are
  unchanged; recommendation policy, recommended-bet result, and market-closeout
  consumers re-derive the reference from issuance rows that already carry both fields,
  so no consumer required modification.

### Verification
- Ruff, Pyrefly, and the unit test suite pass. The candidate, policy, result, and
  closeout suites exercise the cross-artifact re-derivation end to end.
- Embedded prior references exist only in git-ignored development-state output
  (`data/output/recommended_bet_results/`) and regenerate through the production-chain
  commands. No committed artifact depends on the reference.

## 2026-08-24 - Point-in-time quote evidence retrieval

### Added
- Cutoff-visible quote-evidence retrieval (`as_known_at`) that returns canonical
  quote observations whose system-known time (`fetched_at`) is at or before an
  inclusive, UTC-validated decision cutoff. Validates input through the canonical
  quote contract, never mutates the input, returns a fresh canonically-ordered frame,
  and returns the canonical empty quote frame when nothing is visible.

### Changed
- `issue-candidates` now applies the cutoff-visible operation to the loaded quote
  ledger before candidate issuance, so no observation learned after the declared
  evaluation time can enter an issuance. System-known visibility (`fetched_at <=
  cutoff`) is kept separate from pregame eligibility (`is_live is False and
  fetched_at < commence_time`); the observed history selector is reused by
  composition and is unchanged.

### Verification
- Ruff, Pyrefly, and the unit test suite pass.
- Real Week 1 ledger (read-only, checksum-guarded): cutoff 2026-08-18 14:30:00 UTC
  reduced 1,680 observations to 840 visible, retaining only fetched_at
  2026-08-18 14:23:18.347996 UTC; canonical schema and ordering verified; source
  parquet SHA-256 unchanged.

## 2026-08-18 - Production recommendation-chain rehearsal

### Added

- Added immutable recommendation-governance ownership with deterministic
  content identity, strict validation, identity-addressed JSON persistence, and
  exact replay protection.
- Added explicit production-chain commands for creating and verifying
  governance, issuing candidates, deriving recommendation policy, and evaluating
  recommended-bet results from exact persisted identities.
- Added a real-repository candidate issuance integration boundary.
- Added focused governance domain, persistence, CLI, exact preflight matching,
  collection-execution assessment, and postgame-assembly coverage.

### Production orchestration

- Issued one immutable 2026 Week 1 candidate artifact from the explicitly
  selected weekly product, its exact forecast run, and the canonical weekly
  quote ledger.
- Evaluated 1,680 historical quote observations and persisted 698 candidates,
  982 not-candidates, and zero unavailable rows.
- Persisted explicit recommendation governance and derived one exact policy.
- Preserved Moneyline, Spread, and Total policy states as unavailable because
  completed outcomes, validated market closeouts, and settled-wager return
  evidence do not yet exist.
- Evaluated all 698 candidates against the exact policy and persisted 698
  unavailable recommended-bet results with no actionable stakes.
- Kept local wager recording explicit and optional. No sportsbook wager was
  placed.

### Production-chain proof

- Added strict production-chain assessment and immutable checkpoint persistence
  for selected product, forecast provenance, current quotes, repeated quote
  history, selected collection plan, planned execution, candidate issuance,
  recommendation policy, recommendation results, backend serialization,
  frontend presentation, optional recorded wagers, outcomes, closeout, CLV, and
  realized performance.
- Replaced generic artifact-directory checks with strict readers and exact
  season, week, product, run, issuance, policy, and evaluation identity matching.
- Added explicit unavailable, incomplete, invalid, conflicting, available, and
  not-yet-eligible proof states without selecting artifacts by file recency.
- Grounded collection-execution assessment in the selected plan, existing
  due-state evaluator, and immutable claim and terminal-result receipts.
- Added shared postgame evidence assembly that reuses selected-product outcome
  reconciliation, exact candidate closeout, market-specific CLV,
  historical-boundary, market-family evaluation, cleaned-game, and optional
  settled-wager owners once per assessment.
- Preserved pre-kickoff short-circuit behavior so future outcome, closeout, CLV,
  and performance evidence is not loaded or claimed before eligibility.

### Real-data validation

- Performed two live The Odds API ingestions of 840 quotes each across 16 games
  and nine sportsbooks.
- Verified the canonical Week 1 ledger contains 1,680 observations at two
  distinct UTC fetch timestamps.
- Verified all exact identities have repeated depth two: 274 Moneyline, 282
  Spread, and 284 Total identities.
- Persisted candidate issuance
  `278d60da4e2dc089ff7eb973620f49050f83de336034cbff0c8c1a097401ccff`.
- Persisted recommendation governance
  `56757db59c2d04a55eb3f980299699403fdc982e4fe7ff4963f0898112f4824e`.
- Persisted recommendation policy
  `9e2cc3363656366eae76ec0935f01ff201ce9c9784e2736936fd0af9ab0ab024`.
- Persisted recommended-bet evaluation
  `8301fb74e1eaa10437376ff3b616aaa1efc3477944d1a8da0df94abd55de073c`.
- Persisted production-chain checkpoint
  `acf50214f67aed1833e38f998685c3bde4f8f5489a3771f1e50adc319bb887fb`.
- Verified the API attaches recommendation results only to exact matching
  offers.
- Verified Line Shopping and Game Detail render Recommendation unavailable and
  expandable Policy evidence from persisted results.

### Verification

- Repository Ruff passed.
- The configured repository-wide Pyrefly boundary passed with zero errors.
- Focused and repository unit and integration tests passed.
- Frontend recommendation presentation remained mechanically driven by
  persisted states; no frontend recommendation calculation was added.

### Current scope

- Market Unit 26 remains active.
- The pregame and recommendation middle chain is proven independently for
  Moneyline, Spread, and Total.
- Selected-plan execution remains not yet eligible until the first planned poll
  at `2026-09-08T12:00:00Z`.
- Completed outcomes, validated closeout, market-specific CLV, and realized
  performance remain future real-week acceptance evidence.

## 2026-08-16 - Quote collection worker deployment

### Added

- Added repository-owned systemd service and timer templates for the selected
  weekly quote-collection plan.
- Added explicit administrative entry points for installation and read-only
  verification under `deploy/bin/`.
- Added a generated invocation wrapper that resolves no season or week and
  creates a current UTC evaluation timestamp for every service invocation.
- Added a deployment package with explicit path, identity, credential,
  selection, schedule, systemd, artifact, journal, clock, throttling, and
  storage-health checks.
- Added explicit installation and activation error types.
- Added 13 focused deployment tests.

### Deployment safety

- Validate the complete staged service and timer before replacing deployed
  files.
- Require destination parent directories rather than silently creating
  system-level paths.
- Preserve prior deployment bytes and permission modes.
- Restore the previous deployment and reload systemd if the installation reload
  fails.
- Keep timer activation as a separate post-install operation; activation
  failure does not roll back a valid installation.
- Validate exactly one nonempty API-key assignment during installation.
- Keep read-only verification from opening the protected credential file.
- Keep credentials out of repository files, command arguments, generated units,
  wrapper contents, and journal output.

### Raspberry Pi validation

- Installed the repository-owned deployment on the Raspberry Pi 4 quote worker.
- Verified the service and timer with `systemd-analyze verify`.
- Verified the selected 2026 Week 1 plan with 34 planned polls.
- Verified the timer is enabled and active with a five-minute cadence.
- Verified repeated managed executions return `not_due` with exit status zero
  before the first planned poll.
- Verified no execution claim, terminal result, quote-history, or current
  snapshot artifact was created by `not_due`.
- Verified the environment file remains root-owned mode 0600.
- Verified the service runs as the non-root deployment user.
- Verified the worker reports `ready`.
- Verified root storage on the 2 TB SSD, temperature of 46.2 C,
  `throttled=0x0`, and no configured current USB or storage error markers.

### Verification

- Focused Ruff passed.
- Focused Pyrefly passed with zero errors.
- All 13 deployment tests passed.
- Repository Ruff and the selected repository test suite passed.
- Corrected the canonical repository type-check command to
  `uvx pyrefly check`.
- The corrected repository-wide type check reported 524 existing errors.
  Restoring that complete boundary is recorded separately in `ROADMAP.md`.

### Scope

- No collection policy, due-time, quota, claim, receipt, provider-ingest,
  historical interpretation, API, frontend, model, qualification, or
  recommendation behavior changed.

## 2026-08-05 - Weekly prediction architecture and operational closeout

### Added

- Immutable game forecast events with explicit `live` and `backfilled` roles, event identity, invocation run identity, UTC generation time, model identity, and schedule scope.
- Immutable schedule-complete weekly products with indexed storage and explicit season-and-week current selection.
- Model-specific weekly availability inspection and independent policy selection for Win and Total families.
- Source-neutral current-market storage, authoritative edge diagnostics, and explicit prediction-versus-market readiness.
- Schedule-first API game responses with independent Win, Spread, Total, and projected-score component status and provenance.
- Shared frontend presentation for weekly component readiness and edge result states.
- Registry-driven pipeline staleness checks and aligned verification command contracts.

### Changed

- Migrated game prediction to one canonical Away/Home-oriented row per game using `HOME_WIN`, independent `ACTUAL_TOTAL`, and Home-minus-Away differentials.
- Replaced mutable or recency-selected weekly runtime behavior with immutable events and explicitly selected weekly products.
- Updated `weekly-predict` to execute policy-selected live Win and Total models, publish forecast output, and soft-fail only edge generation when markets are unavailable.
- Updated `post-week` to evaluate the exact selected live forecast events after completed outcomes are available.
- Kept the API as a serialization boundary and removed current Games-path dependence on champion resolution or Elo fallback.
- Regenerated OpenAPI and frontend TypeScript contracts and replaced runtime workstream references with stable semantic blocker references.
- Rewrote `HANDOFF.md` as the current operational guide and pruned `ROADMAP.md` to genuine future work and limitations.

### Operational validation

- Completed historical backfills and deployable training for Win Logistic, Win Random Forest, Win XGBoost, Total Random Forest, and Total XGBoost under the canonical feature contract.
- Refreshed calibrations, promoted champions from the complete comparison set, and regenerated the baseline report.
- Completed a real 2026 Week 1 rehearsal with 16 scheduled games, complete Win, Spread, Total, projected-score, and provenance coverage, immutable forecast and product artifacts, and successful PNG and HTML publication.
- Confirmed missing market data remains an explicit independent blocker and soft-fails only edge generation.
- Confirmed frontend lint, production build, and all 344 frontend tests pass.


What has been built and when. Newest first.

### Market data

- Replaced the development odds schema with a provider-aware 17-column quote
  contract separating upstream provider identity from offered-price sportsbook
  identity.
- Added provider event ID, sportsbook update timestamp, commence time, and live
  state provenance with strict UTC and market-side validation.
- Reworked current snapshot and observation-ledger persistence around atomic
  Parquet replacement and row-level idempotency while preserving multiple
  sportsbooks and distinct observations.
- Migrated nflverse schedule markets to truthful consensus provenance with
  `provider=nflverse`, null sportsbook, and explicit pregame state.
- Removed the retired DraftKings adapter, event resolver, ingest command,
  exports, fixtures, tests, and generic wide-to-long conversion.
- Replaced `market_source` and `market_sources` with explicit
  `market_providers` and `market_sportsbooks` across readiness, diagnostics,
  CLI, API, OpenAPI, and generated TypeScript contracts.
- Regenerated current market artifacts with 96 rows across 16 games and
  validated six rows per game, UTC timestamp columns, spread orientation,
  observation uniqueness, and exact-reappend idempotency.
- Added The Odds API v4 client and parser for current NFL moneyline, spread, and
  total markets using US bookmakers, American odds, and ISO timestamps.
- Added strict provider payload, quota-header, event, sportsbook, market,
  outcome, timestamp, and canonical schedule-matching validation.
- Added write-safe current-market ingestion that preserves existing artifacts
  after request, response, payload, parsing, empty-result, and zero-match
  failures.
- Added `ODDS_API_KEY` configuration and the explicit
  `gridiron ingest odds --season ... --week ...` command with quota reporting.
- Validated a live response containing 816 quotes across 16 games and nine
  sportsbooks with zero provider-event identity violations, current book-side
  duplicates, or duplicate ledger observations.

---
## 2026-07-29 — BetSlip decision-support rebuild

Rebuilt BetSlip from a prototype staging panel into a provenance-aware,
model-informed wager-shortlisting and what-if analysis workspace.

### Edge and sizing API contract

- Added required `american_odds` to edge-report rows and preserved the exact
  offered price used by the model calculation.
- Kept `market_value` semantically distinct from sportsbook price.
- Added response-level bankroll and Kelly-multiplier provenance.
- Removed the hidden `$1,000` bankroll default from `/edges`.
- Made bankroll optional for edge generation:
  - EV and full-Kelly fraction remain available without a bankroll;
  - `kelly_stake` remains null when no dollar-sizing basis is supplied.
- Preserved zero as a valid bankroll.
- Constrained bankroll to nonnegative values and Kelly multiplier to `[0, 1]`.
- Regenerated OpenAPI and TypeScript contracts.

### BetLeg v2

- Added a versioned discriminated union for game wagers and prop
  interests/wagers.
- Added canonical producer-independent IDs for game and prop selections.
- Separated immutable recommendation provenance from editable draft inputs.
- Preserved:
  - reference price;
  - model identity and probability;
  - reference EV and edge strength;
  - full-Kelly fraction;
  - reference Kelly stake;
  - reference bankroll and multiplier basis.
- Added editable current odds, proposed stake, sportsbook, and notes.
- Represented props without fabricated prices.
- Added strict runtime parsing and pure constructors from generated API types.
- Added guarded helpers for current EV, implied probability, break-even price,
  price quality, Kelly, suggested stake, payout, and profit.

### Persistence and producer migration

- Added validated v2 local persistence for legs and single/parlay mode.
- Added a separate versioned sizing preference.
- Rejected malformed persisted legs individually.
- Ignored untrusted legacy prototype storage.
- Migrated all live producers:
  - Available Edges;
  - Dashboard Featured Matchups;
  - Dashboard Model Edges;
  - Dashboard Prop Edges;
  - GameDetail Model Lean;
  - GameDetail Top Prop Edges;
  - PlayerProp.
- Removed production placeholder `-110` prices.
- Removed prop type escapes.
- Removed producer-specific wager IDs.
- Prevented missing or `No Edge` prop leans from defaulting to Over.
- Preserved producer source as metadata rather than wager identity.

### Bankroll provenance

- Added tracked and explicit what-if bankroll modes.
- Preferred `/portfolio/summary.bankroll` for tracked sizing.
- Preserved an empty bankroll ledger as a valid `$0.00` tracked bankroll.
- Kept unavailable portfolio data distinct from a zero bankroll.
- Prevented silent fallback between tracked, what-if, and legacy AppState
  values.
- Added a persisted Kelly multiplier with quarter-Kelly as the default.
- Shared one resolved sizing basis between `/edges` and staged-wager analysis.
- Relabeled the legacy AppState bankroll as a standalone calculator value in
  Settings and Onboarding.

### Decision-support presentation

- Replaced compact rows with game and prop decision cards.
- Added immutable reference values beside editable current values.
- Added current EV, break-even price, price-status, full-Kelly, suggested stake,
  proposed stake, payout, and profit.
- Added optional manual sportsbook and draft-note fields.
- Added truthful unavailable states for missing price, probability, bankroll,
  and stake inputs.
- Added complete/incomplete singles summaries using each leg's proposed stake.
- Added separate parlay stake, quoted combined odds, payout, and profit.
- Kept combined parlay probability, EV, and Kelly unavailable because
  correlation is not modeled.
- Added an explicit parlay-correlation caveat.
- Retained remove-leg and clear-slip actions.
- Explicitly stated that Gridiron Edge does not place sportsbook wagers.

### Responsive behavior and accessibility

- Added a responsive two-column decision workspace that stacks at narrow
  widths.
- Added a keyboard-focusable horizontal-scroll region for Available Edges.
- Added table caption, column scopes, row scopes, and canonical row keys.
- Added wager-specific Add and remove labels.
- Added unique per-leg IDs and labels for odds, stake, sportsbook, and notes.
- Added pressed-state semantics for wager and bankroll modes.
- Added politely announced aggregate-summary updates.
- Added structured empty-slip guidance.

### Verification

- Added focused coverage for:
  - exact price and sizing serialization;
  - canonical IDs and parser rejection;
  - persistence and malformed-storage recovery;
  - producer-independent deduplication;
  - current-price EV and Kelly behavior;
  - tracked, what-if, zero, unavailable, and invalid bankroll states;
  - priced games and unpriced props;
  - complete and incomplete singles;
  - complete and incomplete parlays;
  - no-execution behavior;
  - responsive classes and accessibility semantics.
- Backend and frontend quality gates pass.
- A staged-wager real-data visual review remains deferred because `/edges`
  currently returns no available recommendations. No synthetic production edge
  path or fabricated wager data was introduced for that review.

### Deferred

- Recorded-bet write API and ledger coupling.
- `Record Bet` frontend workflow.
- Draft-slip and recorded-bet export.
- Multi-book odds ingestion and line shopping.
- Real-data staged-wager visual verification when edge recommendations are
  available.

## 2026-07-28 — PlayoffProjections navigation and Weekly Outcomes

Extended the PlayoffProjections rebuild with explicit navigation from Team
Rankings and a league-wide weekly schedule-probability matrix.

### Navigation

- Added a shared Team Rankings / Playoff Projections sibling switcher.
- Added the switcher above both screens with active-page semantics and route
  coverage.
- Preserved `/teams?team={abbr}` navigation from both projections views.

### Weekly Outcomes API

- Added `GET /projections/grid`.
- Added frozen Pydantic schemas for the response, team rows, and weekly states.
- Added a static source container and loader for:
  - `data/output/temp/season_grid.csv`;
  - the cleaned upcoming schedule;
  - completed regular-season games;
  - unified long-name / abbreviation mappings.
- Added hand-written serialization for played, projected, bye, and unavailable
  states.
- Added opponent, home/away perspective, game ID, date, time, weekly win
  probability, actual W/L/T result, and `completed_through_week`.
- Determined byes from schedule membership rather than treating a
  `Wxx_WIN_P == 0` artifact value as a bye or loss.
- Added `no_schedule_data` field-status metadata.
- Regenerated `api-schema.json` and `frontend/src/api/schema.ts`.

### Weekly Outcomes frontend

- Added the `useProjectionGrid` React Query hook.
- Added `WinProbabilityCell`, a full-table-cell primitive using a fixed
  diverging red-neutral-green scale centered at 50%.
- Added Playoff Chances / Weekly Outcomes local views.
- Added Week 1–18 rows with:
  - grouped Played Games / Projected Games headers;
  - a clear played/projected boundary;
  - explicit BYE cells;
  - a sticky Team column;
  - shared conference and dependent division filters.
- Preserved filters while switching local views.
- Reused one team-identity implementation across Playoff Chances and Weekly
  Outcomes.
- Added sortable Team and Week 1–18 headers.
- Week sorting cycles from highest chance to win, to lowest chance to win, to
  default team-name order.
- BYE and unavailable rows remain last in both probability directions.
- Equal probabilities use team name as a deterministic tiebreaker.
- Weekly sorting is applied after conference/division filtering and remains
  independent from the Playoff Chances sort.

### Matchup details and accessibility

- Added pointer-hover and keyboard-focus matchup details.
- Portaled tooltips to `document.body` so the horizontal-scroll container does
  not clip them.
- Added viewport clamping at top, left, and right edges.
- Added responsive tooltip width for long team names.
- Formatted tooltips as three centered rows:
  - team matchup;
  - week, date, and time;
  - projected chance to win or played result.
- Kept numeric percentages, explicit BYE labels, grouped headers, and
  accessible names as non-color encodings.

### Verification

- Verified the real response contains 32 teams and 18 weekly entries per team
  for the 2026–2027 preseason artifact.
- Verified `completed_through_week = 0` in the preseason state.
- Verified scheduled games include opponent, venue perspective, date, time,
  and probability.
- Verified Arizona Week 14 is a confirmed BYE with null probability rather
  than a zero-percent game.
- Verified conference and division filtering, local-view state persistence,
  played/projected grouping, sticky team identity, tooltip edge behavior,
  keyboard access, narrow-width scrolling, and long matchup names.
- Backend quality gates, frontend build, focused tests, and the full frontend
  test suite pass.
- Verified ascending/descending weekly probability sorting, three-state reset,
  bye-last behavior, filtered sorting, accessible sort metadata, and sort-state
  persistence across local views.

### Deferred

- Probability-cell texture remains deferred pending color-vision review.

## 2026-07-28 — PlayoffProjections rebuild

Rebuilt the playoff-projections screen as a live, interactive counterpart to
the original static playoff table.

### API contract

- Renamed `week_over_week_delta` to the semantically explicit `elo_delta`.
- Added `items.elo_delta = no_prior_snapshot` status metadata when no usable
  prior same-season Elo snapshot exists.
- Preserved partial-null behavior without incorrectly marking the entire field
  unavailable.
- Regenerated `api-schema.json` and the TypeScript API schema.

### Frontend

- Added `HeatCell`, using a fixed absolute 0–1 probability scale and full-cell
  heat coloring.
- Added accessible `SortableHeader` with explicit active direction.
- Added sortable columns for team, current Elo, Elo delta, average wins, and
  every postseason probability stage.
- Added dependent conference and division selectors.
- Composed current Elo, current record, conference, division, colors, and
  as-of week from the existing `/teams` response.
- Added season, as-of-week, simulation-count, and computed-time context.
- Replaced repeated Week 1 Elo warnings with quiet row placeholders and one
  explanatory legend caveat.
- Preserved warnings for unexpected missing Elo deltas after Week 1.
- Added team navigation to `/teams?team={abbr}`.
- Preserved all postseason stages at narrow widths through horizontal overflow.

### Verification

- Verified all 32 teams against the real Week 1 projections artifact.
- Verified 16-team conference filters and four-team division filters.
- Verified sorting, filter resets, team navigation, continuous heat treatment,
  Week 1 Elo handling, narrow-width behavior, and highlight mode.
- Frontend build and test suite pass.
- Targeted projections backend tests, Ruff checks, and Pyrefly pass.


## 2026-07-01 — W9.10 Compare Screen Rebuild

Two-mode matchup surface at `/compare`. Both modes prototype-aligned
against real, pipeline-populated data.

### Team vs Team

- Mode switcher (Pill, URL-synced `?mode=`)
- Mirrored team pickers (name→logo / logo→name, inward-facing) + swap
  button, width-constrained + centered
- Cohort strip (Season/L4/Home/Away)
- Separate cards: floating pickers → narrative → collapsible summary
  (centered team-left/right) → three matchup cards
- Matchup rows: 5-column center-aligned, mirrored rank-fill bars
  (rank 1 = full, filling toward center), edge chips (arrow + team +
  descriptor), value+rank inline, descriptive sublabels, title-style
  metric names
- Auto-narrative card (biggest collision per direction from rank diffs)
- 11-metric cohort_splits expansion: added def_pass_epa,
  def_third_down_pct, def_redzone_td_pct so every offensive metric has
  its reciprocal defensive-allowed pair

### Player vs Defense

- Independent player / stat-category / team selection (retired the
  prop_id model)
- Searchable player combobox (client-filtered `/players` roster)
- Stat category derived from player position (each option carries B1
  statKey + B3 stat_type)
- Team dropdown (all 32, independent)
- 7-split strip: Season/L4/Home/Away live + vs-Winning/Losing/Top-10
  pending (non-clickable, highlight-marked)
- Per-game bar chart (new BarChart primitive): player's stat as bars
  (B1) + team split-average as a solid reference line (B3) that moves
  with the split while bars stay static
- "Matchup, plainly" verdict card: big avg-allowed, rank-as-context
  line, baseline-driven verdict (defense-allowed vs player's own
  average → Favorable/Tough/Neutral, lean over/under), quantified delta
- By-split comparison table: player avg (from bars, per split) vs
  defense-allowed + Def rank across 4 live splits; 3 pending rows

### Backend (Path C — built to unblock the frontend)

- **B1** — `GET /players/{id}/history?stat=&season=&limit=`: per-game
  stat series from player_game_logs. **Also fixed a root-cause game_id
  scramble** — `_join_game_id` assigned merge-result Series onto a
  non-contiguous index (upstream dropna), scrambling game_id to same-
  week neighbors. Fix: reset_index before the 1:1 merges; derive
  trustworthy is_home. Regenerated logs; re-ran props compute-splits
  (had aggregated against wrong game contexts).
- **B2** — opponent_allowed expanded {season, l5} → {season, l4, home,
  away}. Home/away is the DEFENSE's perspective (inverse of offensive
  player is_home).
- **B3** — `GET /defense/{team}/allowed?stat_type=`: per-team allowed
  aggregates (all cohorts), keyed on arbitrary team (independent-team
  picker needs this; /compare/player is prop-keyed).
- **B4** — `GET /players?season=`: skill-player roster for the picker,
  deduped to latest team.

### New primitives

- **BarChart** — SVG bars + Y-grid + solid horizontal reference line
  with value tag. Reusable (PlayerProp game-log chart is a future
  consumer).

DistributionChart retired from Compare (BarChart replaced it); still
used by PlayerProp.

### Prototype-alignment adjustments (Team mode)

Six post-build refinements per side-by-side review: keep cards (not
floating), mirrored inward team icons, narrative as separate card,
collapsible summary card, centered ranking-bar comparison rows, three
separate matchup cards. Plus: value+rank inline, descriptive sublabels,
title-style center metric names, shortened fill bars, centered page
(max-width), grid centering within cards.

### Deferred

- Book line + O/U bar coloring — odds (W7); legend PendingChip, no
  fake line
- vs-winning / vs-losing / vs-top-10 splits — 3 pending pills + table
  rows (medium backend: opponent-record + self-ranking)
- Change 6 (sortable matchup rows by category/edge + drag-reorder) —
  P2, §9.8; build only if missed

### Substep arc

Tier 1 (mode switcher) → Tier 2 (Team vs Team + 6 alignment
adjustments) → backend B1–B4 (+ game_id fix) → C1 (pickers/strip) →
C2 (bar chart) → C3 (verdict + table) → C4 (cleanup + close-out).

## 2026-07-07 — W9.9 PlayerProp Rebuild

Rebuilt PlayerProp screen (`/players/:propId`) from skeleton with 6
ComingSoonCards to prototype fidelity across 8 substeps in 4 tiers.
Consumes existing data from Step 5 (situational splits) + Step 6
(player vs defense) + game + team metadata endpoints. New
DistributionChart primitive extractable to Compare screen (W9.10).

### Shipped

**Tier 1 — Layout restructure (1 substep):**
- Rebuild GameDetail-style skeleton: full-width hero placeholder +
  single-column content below
- Preserve existing content in placeholder cards (`HeroPlaceholder`,
  `DistributionPlaceholder`, `SectionPlaceholder`)
- Player vs Defense table unchanged
- Blocked ComingSoonCards reduced from 6 to 5 (Situational Splits
  removed from grid — will be real card in Tier 3b)

**Tier 2 — Hero header (2 substeps):**
- 2a: Player hero band with team-colored gradient (180deg, 30% mix →
  var(--bg-1)), TeamMark 56px, breadcrumb, big serif player name
- 2b: Prop summary callout — distinct card with green accent border,
  stat label with game context "MON vs SF", big em-dash for pending
  line, model mean + range on flex row, pending markers for
  confidence + EV
- "+ Bet slip" button outside the summary card

**Tier 3 — Content sections (3 substeps):**
- 3a: `DistributionChart` primitive at `components/primitives/`. SVG
  Gaussian density curve with 90% credible band shading, dashed
  vertical mean marker with value label, x-axis endpoints for
  lo_90/hi_90. Line marker slot (unused; pending). Responsive width
  via viewBox.
- 3b: `SituationalSplitsCard` consuming Step 5 `situational_splits`
  field. 8 cohorts in canonical order (Season / Last 4 / Home /
  Away / Favored / Underdog / Indoor / Outdoor). Format: "X.X avg ·
  N games". Empty state when field null (many props currently
  pending).
- 3c: Player vs Defense polish — WhyLink dot in header (info tone,
  kind: prop_defense), table headers restyled with uppercase small
  letter-spacing to match card language.

**Tier 4 — Placeholders + cleanup (2 substeps):**
- 4a: Change ComingSoonCards grid from 2-col to 3-col (`repeat(3, 1fr)`)
  for better use of horizontal space (5 cards fit as 3 + 2 rows).
- 4b: Cleanup verification — no dead code, no unused imports.

### Architecture consumed

**All 5 W9.5 primitives:**
- `TeamMark` — hero band + prop summary
- `Pill` — not used in PlayerProp; deferred
- `WhyLink` — dot variant on Player vs Defense card
- `Spark` — not used; new dedicated `DistributionChart` primitive
- `TeamHero` — not composed inline; hero band custom-built

### New primitive

**DistributionChart** — at `components/primitives/DistributionChart.tsx`.
- Renders Gaussian PDF from mean + std
- 90% credible band shading (filled area between lo/hi)
- Dashed vertical marker at mean + value label above
- Endpoint labels for lo_90 + hi_90
- Line marker slot (currently unused; awaits odds data)
- Fallback: renders "No distribution data available" when mean or std
  unavailable
- Responsive width via SVG viewBox

Will pay dividends in Compare screen (W9.10) Player vs Defense mode.

### New helper

**`utils/props.ts`** — extracted `formatStatType` and
`formatStatTypeShort` helpers. Same slug → display mapping used across
Dashboard PropEdgesRail, GameDetail TopPropEdgesCard, and PlayerProp.
Dashboard and GameDetail migrations from inline helper deferred to
follow-up cleanup.

### Backend gaps that surfaced

Same as previously identified in ROADMAP §9.7:
- Line context data (line, lean, confidence_tier, p_over) — blocked
  on odds join
- Situational splits data pending for many props
- Player game history endpoint — not consumed yet in W9.9
- Related props filter endpoint — not consumed yet

### Design tension noted

Prototype's PlayerProp had a right-side rail with "Why the model
leans", line shopping mini-table, and related props sidebar. Our
narrower app width doesn't accommodate a rail without cramping.
Consistent with W9.7 lesson — same design constraint applies.
Deferred right-rail elements to blocked ComingSoonCards on our
main content flow.

### Test coverage

Existing 60 tests continue passing. No new tests added for PlayerProp
components — coverage would be integration-level. Primitive tests
from W9.5 provide indirect coverage of building blocks.

### What's not shipped

- No player game history chart (backend endpoint blocked, §9.7)
- No line shopping mini-table (W7 blocked)
- No related props sidebar (backend filter blocked, §9.7)
- No "Why the model leans" reasoning (blocked on feature attribution)

### Next

Between workstreams. Options: W9.8 (backend enablers batch),
W9.10 (Compare rebuild — largest remaining screen gap), or another
polish sweep.

## 2026-07-07 — W9.7 Teams Split-View Rebuild

Restructured `/teams` and `/teams/:abbr` from two separate screens
into a single split-view at `/teams` with optional `?team=X` param.
Left column shows the league rankings; right column shows the
selected team's profile. Clicking a rankings row updates the right
pane without navigation, preserving ranking context across team
browsing.

### Shipped

**Tier 1 — Route restructure (1 substep):**
- Consolidated `TeamRankings` and `TeamProfile` into single
  `TeamsScreen.tsx`
- Auto-select #1 team silently when no `?team=` param
- Row click updates URL param via `navigate("/teams", {team: abbr})`
- Router routes /teams to TeamsScreen (both with and without team param)

**Tier 2 — Left column (2 substeps):**
- Enhanced rankings table with hover state on rows and selection
  highlight
- Trend column with signed colored pill (green/red/dim)
- 5-tab strip (Overall / Offense / Defense / ATS / Net Rating)
  via Pill primitive
- Overall (default) renders 32 teams sorted by rating
- Offense / Defense / ATS / Net Rating render `BlockedTabState`
  with §9.7 backend reference

**Tier 3 — Right column sections (4 substeps):**
- **Team hero band** — team-colored vertical gradient
  (180deg, 30% mix top → var(--bg-1) bottom). TeamMark (56px) left,
  breadcrumb (conf/div · rank · season · through week) above serif
  italic team name, inline hero stats (Record/Rank/Rating)
- **Rating chart** — new `RatingChart` primitive with Y-axis grid +
  rating labels, dots at each data point, X-axis week labels every
  ~4-5 weeks, inline W/L markers per week (green W below line, red
  L above line)
- **Situational Splits card** — Pill cohort switcher (Season/L4/Home/
  Away), 8 metrics (off/def EPA, breakdowns, situational percentages,
  turnover diff) from `cohort_splits` field
- **Recent Results** — existing `RecentResultsStrip` (unchanged)
- **Postseason Outlook** — composed from `/projections`, 5 rows
  (Make Playoffs, Reach Divisional, Reach Conf. Championship, Reach
  Super Bowl, Win Super Bowl) with colored progress bars mapping
  probability to fill width

**Tier 4 — Placeholders + cleanup (2 substeps):**
- Schedule Difficulty placeholder (blocker: `schedule_difficulty`,
  roadmap `§9.7`)
- Top Players placeholder (blocked)
- Deleted `TeamRankings.tsx` and `TeamProfile.tsx` files
- Deleted commented dead code (`ProfileCell`, `InlineFieldStatus`)

**Substep 4c — Polish sweep (7 adjustments in one substep):**
- Left column narrower (5fr / 11fr split); gap between columns
- Hero band aligned within profile column (removed negative margin
  after realizing our container width doesn't match prototype's
  full-screen assumption)
- Green 3px left border on selected rankings row
- Rankings subheader "Wk N · model v4.2"
- Single-column layout beneath hero (reverted from failed 80/20
  attempt; prototype's 80/20 works at ~1400px, ours at ~800px
  didn't fit)
- Postseason outlook: colored progress bars per row
- Rating chart W/L markers moved from X-axis text to inline
  (below line for W, above for L)

### Architecture consumed

**All 5 W9.5 primitives:**
- `TeamMark` — throughout (colored via cache)
- `Pill` — rankings tabs + cohort switcher
- `WhyLink` — not used in Teams; deferred
- `Spark` — not used; new dedicated `RatingChart` primitive instead
- `TeamHero` — not used inline; hero band composition doesn't fit
  its API. Composed directly in TeamsScreen.

### New primitive

**RatingChart** — SVG line chart with:
- Responsive width via viewBox
- Y-axis grid + rating value labels
- Line + data point dots
- X-axis labels every 4-5 weeks
- Optional recentResults prop for inline W/L markers

Location: `frontend/src/components/primitives/RatingChart.tsx`

### Helpers established

- `stripCityPrefix` — same pattern as GameDetail (backend returns
  full name, we strip city prefix to render "New England Patriots"
  as "New England _Patriots_" italic split)
- `expandDivisionLetter` — N/S/E/W → North/South/East/West
- `formatSeason` — "2025-2026" → "2025"

### Composition patterns

- Split-view route: single URL, optional query param drives selection
- Cross-endpoint composition: `/teams` + `/projections` joined
  client-side by team_abbr for postseason outlook
- Blocked-state tabs on `Pill` primitive: tabs remain clickable and
  show blocker messaging when selected, consistent with `field_status`
  pattern

### What surfaced

**Design tension:** Prototype uses ~1400px full-screen layout; our
app is centered ~800px. Two-column layouts that work in prototype
(e.g., 80/20 rating chart + narrow rail) don't fit our width. Reverted
to single-column below hero band. Documented for future workstreams —
prototype fidelity work will always need this constraint check.

### Test coverage

Existing 59 tests continue passing. No new tests added for TeamsScreen
components — coverage would be integration-level. Primitive tests from
W9.5 provide indirect coverage of building blocks.

### What's not shipped

Preserved as `ScheduleDifficultyPlaceholder` and Top Players
`ScaffoldCard`:
- Schedule Difficulty (blocker: `upcoming_games` backend enrichment)
- Top Players (blocker: WAR feature attribution)

Not consumed:
- WhyLink primitive (opportunity for future explainability affordance
  on rating chart or team stats)

### Backend gaps that surfaced

None new. Same gaps as previously identified in ROADMAP §9.7:
- `upcoming_games` enrichment for Schedule Difficulty
- WAR data for Top Players
- Off/def rating decomposition for Offense/Defense/Net Rating tabs
- Cumulative ATS record for ATS tab
- Enriched RecentResult with spread/ATS/O-U (not consumed in W9.7)

### Next

Between workstreams. Options: W9.8 (backend enablers), W9.9
(PlayerProp rebuild), or something else.

## 2026-07-07 — W9.6 GameDetail Full Fidelity

Rebuilt GameDetail (`/games/:id`) from skeleton with 5 coming-soon
cards to prototype fidelity across 9 substeps in 4 tiers. Uses all
5 primitives shipped in W9.5.

### Shipped

**Tier 1 — Layout restructure (1 substep):**
- Full-width header slot + 2-column grid (3fr main / 2fr rail)
- Preserved existing prediction data in placeholder cards during transition
- All old flat layout removed

**Tier 2 — Header composition (2 substeps):**
- Team hero header: two TeamHero components (right-oriented away,
  left-oriented home) framing center block with kick date + "at" +
  venue/weather placeholders
- Model lean callout composed from `/edges` filtered to game_id:
  recommendation + EV% + confidence tier + WhyLink dot + slip button
- Empty state "No model edge" when no edges available for game

**Tier 3 — Main column cards (3 substeps):**
- **Lines & Model Fair Value table** — 3 rows × 3 columns:
  - Market row: em-dashes (blocked on W7)
  - Gridiron Edge fair row: spread + total + moneyline (probability
    → American via new `probToAmerican()` helper in utils/odds.ts)
  - Recommendation row: top edge per market from `/edges` filter;
    highlighted with green tint
- **Win Probability card** — 2 columns:
  - Left: two prob bands with team label + big % + range label
  - Right: projected score display + margin string
  - Away band derived from home band (1 - home_hi/lo)
- **Team Comparison card** — 8 metrics × 4 cohorts:
  - Season / Last 4 / Home / Away tabs via Pill primitive
  - Simple 3-column layout (away value / metric / home value)
  - Green + bold coloring on winning team per metric
  - "Open full comparison →" button navigates to Compare with team
    abbrevs prefilled
  - Consumes `team_comparison` field from Step 7c

**Tier 4 — Right rail + cleanup (3 substeps):**
- **Top Prop Edges card** — compact right-rail list:
  - Filters `/props` by game_id (client-side)
  - Sort by predicted_mean descending, take top 4
  - Each row: player + TeamMark + position + confidence tier +
    stat + lean + line + model value + WhyLink dot + slip button
  - Row click → PlayerProp; "See all N →" shows count of props
- Placeholder integration: Swing Factors + Injuries remain as
  ComingSoonCard (blocked on named workstreams)
- Cleanup: deleted SectionPlaceholder dead code

### Architecture consumed

- **All 5 W9.5 primitives:** TeamHero (heavy — 4 usages), Pill
  (Team Comparison cohort tabs), WhyLink (model lean + prop edges),
  TeamMark (throughout, colored via cache), Spark (not directly
  used in GameDetail — future win prob chart candidate)

### New helpers

- `stripCityPrefix()` — removes "Kansas City " prefix from
  "Kansas City Chiefs" when city and name are exposed separately.
  Otherwise displays "Kansas City Kansas City Chiefs".
- `probToAmerican()` in `utils/odds.ts` — win probability to
  American odds using standard formula: `prob >= 0.5` (favored)
  gives negative American; `prob < 0.5` gives positive American.
- `formatKickLabel()` — game_date → "SUN · FEB 8" (mono uppercase)
- `formatMargin()` — model_spread + team names → "TEAM by X.X"
- `formatSpreadDisplay()`, `formatTotalDisplay()`, `formatMLDisplay()`
  — two-line stacks per Lines table cell
- `formatStatType()` — "qb_pass_yards" → "Pass Yds" for prop rows

### Composition patterns established

- Header + right-side callout wrapped in outer flex container with
  space-between
- Composed cards from multiple endpoints (game, edges, props) via
  React Query
- Client-side filtering by game_id for cross-endpoint composition
  (no backend filter params needed)
- Bet slip integration via `useBetSlip.add()` with placeholder -110
  odds (real odds arrive with W7)

### Test coverage

Existing 59 tests continue passing. No new tests added for GameDetail
sub-components — coverage would be integration-level (real API data
in browser). Primitive tests from W9.5 provide indirect coverage of
building blocks.

### What's not shipped

Preserved as `ComingSoonCard` placeholders for future work:
- Swing Factors (blocked on feature attribution workstream)
- Injuries (blocked on §5.3 injury data source)

No `Spark` usage in GameDetail. Future win probability chart or
prop distribution overlay could use it.

### Backend gaps that surfaced

None new. Same gaps as previously identified in ROADMAP §9.7:
- `kick_time` not exposed (game_date only)
- `venue_text`, `weather_text` pending
- `market_spread`, `market_total`, moneyline market lines blocked on W7

### Next

Between workstreams. Options: W9.7 (Teams split-view), W9.8 (backend
frontend-enablers), W9.9 (PlayerProp rebuild), or another slice of
polish work.

## 2026-07-06 — W9.5 Frontend Polish (Dashboard Rebuild + Cross-Cutting Primitives)

Small workstream between W8 close-out and next major work. Focused on
two things: rebuild the Dashboard (unusable debug scaffolding from
W9 Tier 1) into a real landing page, and ship 5 cross-cutting shared
primitives identified in the prototype audit.

### Shipped

**Tier 1 — Backend patch (1 substep):**
- Added `city`, `conference`, `division`, `primary_color`, `secondary_color`
  fields to `TeamRankingRow` and `TeamProfile` schemas.
- Consolidated `NFL_long_to_short_name.csv` and `NFL_conference_division.csv`
  into unified `NFL_team_metadata.csv`.
- Registry migrations: `teams_long_short` + `divisions` → `team_metadata`.
- Sim, API, CLI, and test fixture consumers all migrated.

**Tier 2 — Cross-cutting primitives (5 substeps):**
- **Pill:** shared filter toggle button.
- **WhyLink:** explainability affordance (labeled and dot variants),
  navigates to `/explain` with subject params.
- **TeamMark:** refactor with team primary color background via
  React Query cache; falls back to grey when unavailable.
- **Spark:** generic sparkline generalized from `RatingHistorySparkline`.
- **TeamHero:** composed team identity block (team-colored mark + city +
  name + record + rating), left/right orientation.

**Tier 3 — Dashboard sections (5 substeps):**
- **FeaturedMatchupsGrid:** 3-card top row. Composes `/edges` (ranked
  by EV) + `/games` (predictions). Uses TeamHero, WinProbBand, WhyLink,
  bet slip integration.
- **ModelEdgesTable:** sortable ranked table with 4 filter tabs
  (All/Spread/Total/Moneyline). Uses Pill primitive.
- **PropEdgesRail:** 5-row compact list sorted by predicted mean
  descending. WhyLink and slip integration.
- **ModelPerformanceRail:** Spark-based sparkline + all-time ROI +
  W-L-P record + bankroll CTA. Consumes `/portfolio/summary` and
  `/portfolio/curve`.
- **Dashboard integration:** 2-row grid layout. Removed API loop
  verification card and field-status demo (moved to git history only).

### Architecture established

- **`components/primitives/` folder** for cross-domain shared components.
  Future workstreams pull from here.
- **`components/dashboard/` folder** for Dashboard-specific sections.
  Pattern reusable for other screen rebuilds.
- **Team color hook (`useTeamMetadata`, `useTeamByAbbr`):** React Query
  cache with 5-minute stale time. All screens using `TeamMark` benefit.
- **Placeholder odds (-110):** when adding edges to bet slip from Dashboard,
  we use -110 as placeholder odds until W7 multi-book lands.

### Test coverage

59 tests total across primitives + Dashboard sections. Vitest + React
Testing Library. Every primitive has its own unit test file.

### What's not shipped (surfaced during work)

- Rolling 7d/30d ROI windows on `/portfolio/summary` — tracked in
  ROADMAP §9.7. Currently shown as "All-Time" honestly.
- Real market odds — bet slip integration uses placeholder -110. Blocked
  on W7 (multi-book odds).
- Prop leg semantic — `add()` currently encodes prop as "prop" market
  type; BetLeg schema might need refinement for prop legs specifically.

### Next

Between workstreams. Options: W12 (Model Ensemble), W7 (Multi-Book Odds),
W4.5 (Scenario Engine, blocked on §5.3), or another frontend polish
sweep pulling from §9.7/§9.8 backlogs.

## 2026-07-04 — W8 Tier 3: Additive Datasets (7 additives, 15+ substeps)

Seven-step tier closing out W8 Tier 3. Each additive is a small feature
engineering module + persistence artifact + loader + serializer wiring
that populates one or more previously-scaffolded `field_status: pending`
fields on the API.

### Additives shipped

- **Step 1 (2026-07-03):** `week_over_week_delta` on `/projections`. No new module — reads directly from Elo state table.
- **Step 2 (2026-07-03):** Per-team percentile ranking pass (4 stats: elo, avg_wins, make_playoffs, win_sb). Populates `/teams`, `/teams/{abbr}`, `/compare/teams`. New module `evaluation/percentiles.py`.
- **Step 3 (2026-07-03):** `trend` on `/teams` and `/teams/{abbr}`. Reused `compute_elo_deltas` from Step 1.
- **Step 4 (2026-07-03):** `n_simulations` on `/projections`. New `projections_metadata.json` sidecar written by `run_full_simulation`.
- **Step 5 (2026-07-03):** Per-player situational splits (8 cohorts) on `/props/{prop_id}`. Joins player game logs to games CSV. New module `evaluation/situational_splits.py`.
- **Step 6 (2026-07-04):** Opponent-allowed-by-position defense rows on `/compare/player/{prop_id}` (3 of 4 rows; `red_zone_rate_allowed` deferred). New module `evaluation/opponent_allowed.py`.
- **Step 7 (2026-07-04):** Team cohort splits (4 cohorts × 8 metrics) on `/compare/teams`, `/teams/{abbr}`, `/games/{game_id}`. New module `evaluation/team_cohort_splits.py` and new `gridiron teams` CLI subcommand.

### New CLI subcommands

- `gridiron sim compute-percentiles`
- `gridiron props compute-splits`
- `gridiron props compute-opponent-allowed`
- `gridiron teams compute-cohort-splits`

### New persistence artifacts

- `data/output/rankings/percentiles/percentiles_{season}_wk{NN}.parquet`
- `data/output/rankings/team_cohort_splits.parquet`
- `data/output/props/situational_splits/{stat_type}.parquet`
- `data/output/props/opponent_allowed.parquet`
- `data/output/temp/projections_metadata.json`

### Test-fixture inconsistencies discovered

- `MiniRepoBuilder.with_teams_reference()` produces modern short codes (`KC`, `LAC`, `BUF`, `MIA`) but rest of codebase uses PFR-era codes (`KAN`, `JAC`). Not blocking. Captured in ROADMAP §9.6.

### Remaining not-shipped

- Off/def rating decomposition (real modeling work).
- Various `field_status: pending` fields blocked on named workstreams.

### Next

W8 workstream closed. Between-workstreams pause. Available next workstreams: W12 (Ensemble), W4.5 (Scenario, blocked on §5.3), W7 (Multi-Book, blocked on §5.2), W10 (Real-Time, deferred).

## 2026-07-03 — W9 Frontend (20-screen React app consuming the API)

Three-tier workstream shipping a complete React frontend. Consumes
the 16-endpoint W8 Tier 2 API surface end-to-end. Every prototype
screen renders with real data where populated, structured
`field_status` where scaffolded, and consistent error UX everywhere.

### Shipped

**Tier 1 — Client infrastructure (7 substeps):**
- Vite + React + TypeScript scaffolding with Geist font loading and
  OKLCH dark theme port.
- Chrome components: TopNav, SubNav, Breadcrumb.
- Three React Contexts (AppState, BetSlip, Nav) with
  localStorage/sessionStorage persistence.
- `openapi-fetch` typed API client from checked-in schema.
- React Query with per-endpoint hooks and query key namespacing.

**Tier 2 — Populated screens (7 substeps + 1 pre-substep):**
- Pre-substep 2.0: Field-status primitives (`<PendingField />`,
  `<BlockedField />`, `<FieldValue />`).
- 2a: Games (GamesList + GameDetail).
- 2b: Teams (TeamRankings + TeamProfile).
- 2c: Projections (PlayoffProjections).
- 2d: Players/Props (PlayersExplorer + PlayerProp).
- 2e: Compare (ComparePage) with URL-synced state for bookmarking.
- 2f: Bankroll consuming 5 `/portfolio/*` endpoints in parallel.
- 2g: BetSlip staging bets from `/edges`.

**Tier 3 — Blocked screens + polish (6 substeps):**
- 3a: BlockedScreen for LineShopping, LiveGame, NewsWire, ExplainPage.
- 3b: Settings, Onboarding, Tools (client-side, no API).
- 3c: Aesthetic identity documented.
- 3d: Vitest smoke tests.
- 3e: Keyboard accessibility sweep.
- 3f: ErrorCard + OfflineBanner.

### Architecture established

- **Data flow:** endpoint → generated TypeScript type → React Query hook
  → screen component → shared primitive (FieldValue, ErrorCard, etc.).
- **State:** three Contexts (Nav, BetSlip, AppState) persisted client-side.
- **Styling:** OKLCH dark theme via CSS variables. Cards for grouping,
  monospace for numerics, serif for editorial emphasis.
- **Testing:** Vitest + React Testing Library. Smoke coverage of
  critical paths.
- **Accessibility:** Semantic HTML, focus indicators, ARIA labels on
  icon-only controls.

### W8 backend hygiene items surfaced

- `season` type inconsistency (int vs string across endpoints).
  Captured in ROADMAP §9.6.
- Team abbreviation convention (KAN/JAC vs KC/JAX). Not blocking, not
  tracked as an issue.
- `evaluate select-model --write-manifest` display bug: "Persist
  manifest" step appeared after the summary block. Fixed as a small
  W8 patch during 2f verification.

### Next

W8 Tier 3 designing. Now that W9 has surfaced which pending/blocked
states appear on real screens, the additive dataset priority can be
decided empirically rather than speculatively. Most likely first
additive: per-stat percentile ranking (drives compare screen rank
columns and team detail rank fields) OR opponent-allowed-by-position
(drives the entire defense side of PlayerProp).

## 2026-07-02 — W8 Tier 2: Direct-Serialization Endpoints (16 endpoints populated)

Eight-step tier closing out Tier 2 of the API serving layer. Every
prototype-referenced URL returns a 200 with a Pydantic-validated
shape. Fields not yet populated are marked with structured
`_meta.field_status` per D14.

### Endpoints populated

Step 1 (2026-07-01): `/weeks/current` and all `/portfolio/*`.
Step 2 (2026-07-01): `/model/performance` (composed metrics endpoint).
Step 3 (2026-07-01): `/teams` and `/teams/{abbr}`.
Step 4 (2026-07-01): `/projections`.
Step 5 (2026-07-01): `/games` and `/games/{game_id}`.
Step 6 (2026-07-01): `/edges`.
Step 7 (2026-07-01): `/props` and `/props/{prop_id}`.
Step 8 (2026-07-01): `/compare/teams` and `/compare/player/{prop_id}`.

### Architecture established

- **Loader pattern (`api/loaders.py`):** pandas DataFrames in, dicts
  out, explicit `settings.repo_root` threading (D19).
- **Schema pattern (`api/schemas/*.py`):** Pydantic v2, `frozen=True`,
  `extra="forbid"`, nullable defensive fields, `_meta` envelope
  via `BaseResponse` / `BaseListResponse`.
- **Serializer pattern (`api/serializers/*.py`):** hand-written per
  D17, owns `_meta.field_status` construction per D18.
- **Route pattern (`api/routes/*.py`):** FastAPI, exception
  translation (`ChampionNotFoundError` → `NO_CHAMPION_MANIFEST`,
  `OddsUnavailableError` → `NO_ODDS_AVAILABLE`), lazy scope resolution
  (Step 7d learning).
- **Testing pattern:** `MiniRepoBuilder` extended with four
  W8-specific methods (`with_champion_manifest`,
  `with_predictions_archive`, `with_odds_snapshot`,
  `with_teams_reference`); integration tests via FastAPI
  `dependency_overrides`.

### New `Unavailable` slugs registered

`NO_CHAMPION_MANIFEST` (Step 5d), `NO_ODDS_AVAILABLE` (Step 6d),
`OPPONENT_ALLOWED_BY_POSITION` (Step 8b).

### New `api/` modules

- `api/exceptions.py` — API-surface data-state exceptions from
  loaders to routes.
- `api/_prop_id.py` — Shared `decode_prop_id` helper used by
  `/props/{prop_id}` and `/compare/player/{prop_id}`.

### Field_status scaffolding

Fields not yet populated ship with structured `_meta.field_status`
metadata. Categories:

- **Pending backend work:** kick, venue, weather (games); line, p_over,
  lean, confidence_tier (props); schedule_difficulty,
  playoff_probability, cohort_splits, percentile_ranks (teams
  compare); recent_form, situational_splits, historical_vs_opponent
  (props/compare).
- **Blocked on named workstreams:** swing_factors, prop_reasoning
  (feature attribution); injuries, injury_status (§5.3);
  multi_book_shopping (W7); off_rating, def_rating (Tier 3);
  trend (weekly Elo snapshot); avg_allowed, rank_against_position,
  last_5_games_avg, red_zone_rate_allowed (opponent aggregation).

### Tests

- Per-route integration test file in `tests/integration/api/` for
  each populated endpoint cluster (games, edges, props, compare).
- Per-schema unit test file in `tests/unit/api/` for each new schema.
- Per-serializer unit test file for each new serializer.

### Next

Tier 3 (additive datasets) designing. Kickoff waits for W9
(Frontend) feedback to identify which additive to build first.
W9 unblocked and ready to start.

## 2026-07-01 — W13 Tier 3: CLI Consumer Refactor (W13 workstream complete)

Four-step tier migrating CLI consumers to use the champion manifest
via the ``--model-type auto`` sentinel pattern. Closes W13 as a
workstream.

### Shipped
- ``cli/_composites.py::resolve_win_prob_model_type`` — helper for
  the ``"auto"`` sentinel. Reads the manifest; passes explicit values
  through; raises ``typer.BadParameter`` on missing manifest with an
  actionable message.
- ``cli/weekly_predict.py`` — ``--model-type`` default flipped from
  ``"random_forest"`` to ``"auto"``. Resolution happens after
  Typer/user-input validation.
- ``cli/edges.py`` — both ``report`` and ``clv`` migrated with the
  same pattern.
- Intentional Elo callsites annotated with comments explaining why
  they aren't migrated:
  * ``cli/weekly_predict.py::_stage_predict_week`` (archives
    ``build_predictions_df`` output, which is Elo-based).
  * ``cli/output.py::output_predictions`` (same pattern).
  * ``cli/evaluate.py::evaluate_tune`` — both ``--apply`` branches
    (tune is Elo-specific by design).
  * ``cli/evaluate.py::evaluate_backfill`` — CLI defaults are
    historical convenience, not a champion pick.
  * ``cli/ratings.py::elo_evaluate`` — Elo command by name.

### Tests
- ``tests/unit/cli/test_composite.py`` — added
  ``TestResolveWinProbModelType`` (4 tests).
- ``tests/unit/cli/test_weekly_predict.py`` — added
  ``TestModelTypeResolution`` (3 tests). Existing
  ``test_runs_all_stages_when_all_succeed`` updated to pass
  ``--model-type random_forest`` explicitly.
- ``tests/unit/cli/test_edges.py`` — new file with
  ``TestReportModelTypeResolution`` and ``TestClvModelTypeResolution``
  (3 tests each).
- ``tests/integration/test_edges_cli.py`` — six existing tests
  updated to pass ``--model-type random_forest`` explicitly.

### Scope note

The original W13 handoff paragraph identified "8 hard-coded
callsites." Categorization during Tier 3 design revealed that only
3 were user-facing CLI defaults that should resolve to the champion
(``weekly_predict``, ``edges report``, ``edges clv``). The other 5
were:

- Provenance labels for Elo-based predictions (correct as-is).
- Elo-specific by design (correct as-is).
- Historical CLI convenience defaults (kept for backward compat;
  users pass explicit values in practice).

All 5 got explanatory comments instead of refactors.

### W13 workstream summary (Tiers 1–3, all shipped 2026-07-01)

- **Tier 1:** manifest schema + reader API (``champion_resolver.py``).
- **Tier 2:** writer + full-retrain integration + manual-override
  CLI flags (9 steps).
- **Tier 3:** CLI consumer migration (4 steps).

### Next
W8 (API Serving Layer) resumes at Tier 2 Step 5. Runtime champion
resolution now available for the ``/games``, ``/games/{id}``,
``/games/{id}/predictions``, ``/edges``, and ``/props/{prop_id}``
endpoints via ``resolve_current_champion``.

## 2026-07-01 — W13 Tier 2: Runtime Champion Manifest (Writer + Full-Retrain Integration)

Nine-step tier closing out the writer half of Runtime Champion Resolution.
The static manifest at ``data/output/champions/champions.json`` is now
populated by the ``promote-champions`` stage in ``full-retrain`` and by
optional ``--write-manifest`` flags on ``evaluate select-model`` and
``props champion``. All champion decisions across CLI and stage
surfaces share the same code path.

### Shipped
- ``champion_resolver.write_manifest`` — atomic write via
  ``os.replace``; preservation semantics for per-entry ``source_run_id``.
- ``evaluation.champion.select_game_classification_champions`` —
  wraps ``select.py``'s ``collect_model_metrics`` + ``rank_models``
  on Brier / ECE / AUC.
- ``evaluation.champion.select_game_regression_champions`` — reads
  ``ArtifactStore`` metadata; picks lowest MAE; tie-breaks to
  ``random_forest``.
- ``evaluation.champion.select_prop_champion_for_family`` and
  ``select_prop_champions_all_families`` — iterate ``PropModelType``,
  build ``RegressionModelResult`` per algorithm from the prop archive,
  delegate to existing ``select_prop_champion``.
- ``evaluation.champion.build_prop_champion_candidates`` — shared
  per-algorithm evaluation helper reused by the selector and by
  ``props champion`` for terminal display.
- ``evaluation.champion.promote_champions`` — pure function combining
  the three selectors + manifest merge + write. Returns a
  ``PromoteChampionsResult`` with fresh, preserved, and warning fields.
- ``cli/full_retrain.py::_stage_promote_champions`` — thin adapter over
  ``promote_champions``. Depends on ``refresh-calibrations`` only;
  runtime order still places it after ``backfill-prop-models`` when
  both are active. ``baseline-report`` re-wired to depend on
  ``promote-champions``.
- ``cli/full_retrain.py::_stage_baseline_report`` — appends a Current
  Champions bullet-list block above the Game Models table. Format
  chosen so the existing markdown-table delta parser ignores it.
- ``cli/_composites.py::write_champion_manifest`` — shared helper for
  the manual-override CLI flags.
- ``cli/evaluate.py::evaluate_select_model`` — new ``--write-manifest``
  flag. Runs the full catalog through ``promote_champions``.
- ``cli/props.py::champion_cmd`` — new ``--write-manifest`` flag.
  Refactored inline per-algorithm loop to use
  ``build_prop_champion_candidates``.
- ``models/catalog.py`` — new module. Single source of truth for
  ``GAME_MODEL_PAIRS``, ``PROP_STAT_FAMILIES``, ``PROP_ALGORITHMS``.
  Used by both ``full_retrain.py`` and the manual-override flags.

### Decisions

No new architectural decisions locked at the D-level; all decisions
were within the Tier 2 design phase and are captured in PLAN.md's
inline "How" block for the tier.

### Tests

- ``tests/unit/evaluation/test_champion_resolver.py`` — 8 tests for
  ``write_manifest`` (schema, atomic write, preservation semantics,
  roundtrip, empty writes, defensive copy).
- ``tests/unit/evaluation/test_champion.py`` — added
  ``TestSelectGameRegressionChampions`` (9 tests),
  ``TestSelectGameClassificationChampions`` (9 tests),
  ``TestSelectPropChampionForFamily`` (5 tests),
  ``TestSelectPropChampionsAllFamilies`` (3 tests),
  ``TestPromoteChampions`` (3 tests),
  ``TestBuildPropChampionCandidates`` (3 tests).
- ``tests/unit/cli/test_full_retrain.py`` — added
  ``TestStagePromoteChampions`` (5 tests), extended ``TestStageList``
  (2 new tests), extended ``TestBaselineReportStage`` (3 new tests),
  updated ``TestCommandInvocation`` for the new stage.
- ``tests/unit/cli/test_evaluate.py`` — new file with
  ``TestSelectModelWriteManifestFlag`` (3 tests).
- ``tests/unit/cli/test_props_champion_write_manifest.py`` — new file
  with ``TestPropsChampionWriteManifestFlag`` (3 tests).

### Next
Tier 3: refactor the 8 hard-coded ``model_name="win_prob",
model_type="elo"`` callsites across ``weekly_predict.py``, ``output.py``,
``edges.py``, ``evaluate.py`` to use ``resolve_current_champion``.
Confirm ``ratings.py``'s intentional Elo usage stays as-is with a
comment.

## 2026-06-27 — W8 Tier 1: API Skeleton and Blocked-Endpoint Stubs

First tier of W8. FastAPI app skeleton at `src/gridiron_edge/api/` plus
12 blocked-endpoint stubs matching the prototype URL inventory. Each
returns 200 with a structurally valid null response carrying registered
blocker slugs. Reachable via `gridiron api serve`.

### Shipped

- `api/app.py` — FastAPI factory with OpenAPI tag inventory and CORS.
- `api/meta.py` — `ResponseMeta`, `Blocker` registry, `Unavailable` slugs.
- `api/schemas/_base.py` — `BaseResponse` / `BaseListResponse` with
  `_meta` envelope.
- `api/deps.py` — `SettingsDep` / `DataPathResolverDep` shared
  dependencies with a single override seam.
- Twelve stubbed routes for Tier 3-blocked endpoints (comparables,
  explain, injuries, lines, live, model, news, prop_reasoning,
  prop_shop, swing_factors, plus placeholder shapes for teams,
  projections). Each returns a structurally valid null response
  with `_meta.field_status` populated.

### Architectural decisions

- **D14:** Placeholder convention — null field + `_meta.field_status`
  entry with either the literal string "pending" or a `BlockedStatus`
  object naming a stable blocker slug.
- **D16:** Every Tier 3 route uses a slug registered in
  `Blocker.all_slugs()`; consistency test enforces this.

### Tests

- Integration tests confirm all 12 endpoints reachable via
  `gridiron api serve`, return the expected status codes, and carry
  valid `_meta.field_status` metadata where scaffolded.

### Next

Tier 2 (populated endpoints): fill in the 16 endpoints the prototype
consumes with real data. Establish loader/schema/serializer/route
pattern.

## 2026-06-22 — Workstream 5: Tier 4 Cleanup Sweep

### Summary

Multi-session opportunistic cleanup that closed 30 items from the Tier 4 backlog and surfaced two real bugs that were promoted to fixes during the sweep. The Tier 4 backlog is retired; remaining items moved to PLAN.md as workstream candidates.

### Highlights

**CLI ergonomics (4 items):**
- `bet summary` now renders `calibration_health` and `ev_vs_actual_gap` from the existing summary dict
- `models info win_prob elo` now directs analytic-model users to `evaluate summary` instead of suggesting training
- `props train-and-save` exposed as a CLI command to produce persisted artifacts for projections
- CLI season-label inconsistency resolved — `props backfill` now accepts both `2023` and `"2023-2024"` formats

**Composite commands (5 items):**
- `weekly-predict` renders top-edge preview from ranked edge report
- `full-retrain` generates timestamped baseline reports with delta-vs-previous tables
- `verify` baseline-comparison now actually compares metrics against the latest full-retrain
- `verify` composite-key parser correctly handles multi-token model types (e.g., `random_forest`)
- `full-retrain` calibration values persist to disk at `data/output/calibration/game_model_calibration.json`
- `post-week` drift threshold extracted to a named constant

**Dead code removal (5 items):**
- `_shared.py` re-export shim
- `_game_location` helper (logic was inlined into the cleaner code path)
- `_EPA_RELIABLE_FROM` constant
- `UNIVERSAL_FEATURE_COLS` and the related test fixtures
- `PropPrediction` dataclass (vestigial pre-DataFrame design)
- `max_mae_tolerance` field in `RegressionPromotionGates` (defined but never enforced)

**Documentation drift (3 items):**
- Stale `WS2` / `D1` / `D3` workstream markers removed from production source
- Schema version comments referencing v2/v3 replaced with version-neutral language
- Phase markers like `(existing)` / `(new)` audit complete; surviving instances refer to runtime state, not project history
- HTML escaping added to `viz/predictions.py::render_predictions_html`

**Architecture (4 items):**
- `_TEAM_CODE_MAP` historical abbreviation mapping consolidated into `core/constants.py::TEAM_CODE_NORMALIZATION`
- `run-data-pipeline` retained as a data-layer primitive (intentionally not refactored to composite form)
- `repos.py::with_epa_by_game` routed through the shared `_write()` helper for registry consistency
- Inline imports in composite CLI files: lightweight imports moved to module top, heavy imports (matplotlib, sklearn-touching, prediction pipeline, prop registry) kept inline for fast `gridiron --help` startup

**Type safety and error handling (2 items):**
- Exception narrowing in viz/predictions.py (GAMETIME parsing), ingest/odds/draftkings.py (float coercion), cli/betting.py (odds ledger load). Broad catches retained where defensive
- `# pyrefly: ignore` and `Any` annotation audit complete. Most existing suppressions are legitimate workarounds for known stub limitations; further type work deferred

### Real bugs surfaced and fixed

**XGBoost recalibration Pipeline feature-name warning:** The `CalibratedClassifierCV` Pipeline in the XGBoost post-training calibration branch was fitted on a DataFrame and predicted on `.values` arrays, producing sklearn `UserWarning` at every predict call. Fix: fit and predict on `.values` arrays consistently with the rest of the codebase.

**Modeling file stale-data preservation:** Investigation of a row-count discrepancy between feature sets revealed that the incremental build mode of `build_model_inputs()` was silently preserving stale weather data for ~12,000 historical rows. Weather data was missing for 1999-2010 seasons in the modeling file despite the weather source data being complete and the WeatherFeature implementation being correct. Root cause: incremental builds only recomputed features for new GAME_IDs, leaving older rows with values from whenever they were first computed. Fix: added `data_version` field to the modeling manifest; pipeline detects mismatch and forces a full rebuild with a warning. Convention documented for future bug fixes.

### What this enables

- Tree-based game models can now train on the rebuilt modeling file with 9,920 training rows (up from 5,705, a 74% increase), thanks to historical weather data now being available
- Future feature implementation bug fixes will trigger automatic full rebuilds via `data_version` bumps, preventing silent stale data
- The composite CLI workflows produce richer terminal output (top edges in weekly-predict, drift health in post-week, baseline diffs in full-retrain, real metric comparison in verify)
- Disk-backed calibration values persist across `full-retrain` runs

### What's deferred to future workstreams

- **Testing infrastructure** (5 items): props e2e tests, composite commands e2e tests, weather ingest integration test, registry cold-start scenarios, performance baselines
- **Real bug** (1 item): Walk-forward backfill produces no valid pipeline for single-season windows with expanded feature sets
- **Investigation** (1 item): `CalibratedClassifierCV` shuffle=False → TimeSeriesSplit comparison
- **Operational** (4 items): DraftKings 403, stadium coverage data entry, calibration refresh after next full-retrain, `verify --strict` CI gate

### Files retired

- `TIER_4_BACKLOG.md` — replaced by PLAN.md's "Future Workstream Candidates" section

## 2026-06-18 — Workstream 2: Game Model Refactor

### Added
- `BaseModelMetadata` shared metadata type with `GameModelMetadata` and `PropModelMetadata` subclasses.
- `GamesTrainer` + spec subclasses (`WinProbTrainer`, `TotalTrainer`) for unified game model training.
- `GamesPredictor` base class with five composite-key registrations.
- Composite registry keys (e.g. `win_prob_random_forest`) replacing flat keys.
- Nested artifact path scheme `data/models/{model_name}/{model_type}/`.
- Elo migrated to `win_prob_elo` composite registration.

### Changed
- All classification metrics (Brier, ECE, AUC, log_loss, accuracy) are now first-class fields on `GameModelMetadata`.
- Prediction archive schema migrated from `model_version` to `(model_name, model_type)`.
- CLI commands use `(model_name, model_type)` pair throughout.

### Removed
- `models/game_prediction/tree.py`, `logistic.py`, `pipeline.py`.
- Free functions `train_total_model`, `load_total_model`, `predict_total`.
- Flat registry keys: `logistic`, `random_forest`, `xgboost`.
- Legacy `LogisticPredictor`, `RandomForestPredictor`, `XGBoostPredictor` re-exports.
- `EloV1Predictor`, `EloV2Predictor`, `EloV3Predictor` (collapsed into `WinProbEloPredictor`).
- `evaluation/archive.py::migrate_archive` (no longer needed).

## 2026-06-17 — Workstream 1: Champion/Challenger for Props

- **Prop model factory pattern** (`PropModelType` enum: elasticnet,
  random_forest, xgboost) with `_create_model()` factory and
  `_get_param_grid()` providing per-algorithm HP search spaces
  (ElasticNet: 25 combos, RandomForest: 36, XGBoost: 54).
- **Spec-only subclasses**: all 5 prop trainers (`qb_pass_yards`,
  `qb_rush_yards`, `rb_rush_yards`, `wr_rec_yards`, `te_rec_yards`)
  reduced to ~15-20 lines each. Shared `_fit()`, `_predict()`,
  and `train(model_type=)` consolidated in `PropTrainer` base.
- **`clip_lo` / `clip_hi` on `PropModelSpec`**: spec-driven prediction
  clipping (0.0 floor; per-position ceilings of 200-600 yards).
- **`model_type` field on `PropModelMetadata`**: artifact tracking.
- **Generalized champion/challenger gates** in `evaluation/champion.py`:
  - Classification path (game models) renamed for symmetry:
    `ClassificationPromotionGates`, `ClassificationComparisonResult`,
    `extract_classification_metrics`, `compare_classification_models`,
    `format_classification_comparison`.
  - Regression path (prop models): `RegressionPromotionGates` (R² > 0,
    coverage in [0.85, 0.97]), `RegressionModelResult`,
    `RegressionComparisonResult`, `compare_regression_models()`,
    `select_prop_champion()` (lowest MAE among eligible, ElasticNet
    fallback per Decision #11), `format_regression_comparison()`.
- **CLI enhancements** in `cli/props.py`:
  - `evaluate --model-type {elasticnet,random_forest,xgboost}`
  - New `champion` command - trains all 3 types, compares, selects
  - `console.header()` / `step()` / `console.summary()` parity with
    game model CLI; tqdm bars match game model styling
    (ncols=88, colour="cyan", live best-metric postfix).
- **Tests**: 15 prop champion tests (factory, grids, clips), 16
  regression champion gate tests, 5 CLI structure tests, plus
  updates to existing classification tests for renamed symbols.

### Validated

- All 15 prop model trainings completed via
  `gridiron props champion --model all`.
- ElasticNet selected as champion in 5/5 stat families.
- `qb_rush_yards` triggered fallback policy (no model passes R²>0
  guardrail) - known limitation; feature work deferred to later WS.

### Changed

- `cli/models.py`: updated to use renamed classification symbols
  (`ClassificationComparisonResult`, `compare_classification_models`,
  `extract_classification_metrics`).

## 2026-06-10 - W4: Player Data & First Prop Models - Mostly Complete

Complete player-level data pipeline, 5 trained prop models, post-processing
enrichment, evaluation metrics, archive, and CLI. M3 milestone achieved.

##### Player data foundation (Phase A)
- **nflreadpy migration:** Switched from archived nfl_data_py to nflreadpy.
  Key API changes: import_weekly_data() → load_player_stats().to_pandas().
  nflreadpy returns Polars DataFrames requiring .to_pandas() conversion.
- **Player stats ingest:** 26 seasons (1999–2024), ~5K rows/season,
  42 columns per player-game row. Stored at data/raw/player_stats/.
- **Player stats cleaning:** Dropped rows with null game_id (1 row,
  1999 week 9), deduplicated 46 schedule-join mismatch rows.

##### Player feature engineering (Phase B)
- **Rolling features (features/player/rolling.py):** L3 and L6 rolling
  mean + std for 23 stat columns (~46 features). Shift(1) prevents
  lookahead. Position-specific stat columns.
- **Matchup features (features/player/matchup.py):** 28 features -
  14 defensive-allowed stats × 2 (L6 rolling average + rank).
  Rankings: 1=toughest, 32=most generous. Joined via opponent_team.
- **Usage features (features/player/usage.py):** 6 features -
  target_share, carry_share, touch_share × L3/L6 windows.
- **Game context features (features/player/game_context.py):** 6 features
  from cleaned games CSV - is_home, game_spread, over_under,
  implied_team_total, is_dome, rest_days. No shift(1) needed -
  these are known pre-game.
- **Unified builder (features/player/builder.py):**
  build_prop_features(position_filter=["QB"]) chains all 4 builders
  with single parquet load. NaN handling deferred to trainer.
- **Programmatic feature list (features/player/_columns.py):**
  PROP_FEATURE_COLS built from component modules - stays in sync.

##### Prop model training (Phase C)
- **PropTrainer base class (models/prop_prediction/base.py):**
  - _load_data() calls build_prop_features()
  - train() uses HOLDOUT_SEASONS (2023–2025) - consistent with game models
  - Position-aware NaN handling: features with >50% NaN for the filtered
    position are dropped before training
  - _feature_columns() returns PROP_FEATURE_COLS
  - _build_features() non-abstract - default returns df as-is
  - Deleted dead _join_game_context() and _join_schedule_context()
- **5 trained models (ElasticNet baselines):**

| Model | Train | Holdout | MAE | RMSE | R² | Nonzero |
|-------|-------|---------|-----|------|----|---------|
| qb_pass_yards | 5,706 | 1,367 | 58.0 | 72.6 | 0.071 | 37/128 |
| qb_rush_yards | 1,434 | 468 | 16.4 | 20.2 | 0.090 | 52/128 |
| rb_rush_yards | 10,023 | 2,001 | 25.0 | 32.3 | 0.168 | 16/124 |
| wr_rec_yards | 23,831 | 4,535 | 25.1 | 32.9 | 0.203 | 55/120 |
| te_rec_yards | 10,087 | 2,052 | 18.3 | 24.2 | 0.188 | 58/120 |

##### Post-processing enrichment (Phase C2)
- **models/prop_prediction/post_process.py:** Pure function architecture.
  - predicted_std = sqrt(model_rmse² + player_L3_std²)
  - 90% prediction intervals, lo_90 clipped at 0
  - P(over) = 1 - Φ((line - mean) / std), Normal CDF for V1
  - Lean: Over (>0.55), Under (<0.45), No Edge
  - Confidence tier: High (|p-0.5|>0.15), Moderate (>0.08), Low
  - Line input optional - NaN line → NaN p_over/lean/tier
  - TARGET_STD_MAP maps model names to rolling std columns

##### Evaluation metrics (Phase D1)
- **evaluation/prop_metrics.py:** 6 metric functions + orchestrator.
  - AccuracyMetrics: MAE, RMSE, R², median AE
  - BiasMetrics: mean error, % over-predicted
  - CoverageMetrics: actual vs nominal coverage, interval width
  - CalibrationMetrics: P(over) reliability diagram, MACE
  - HitRateMetrics: Over/Under/overall, push exclusion
  - TierMetrics: per-tier MAE, hit rate, |p_over - 0.5|
  - PropEvalReport.print_summary() formatted output
  - Graceful degradation: accuracy/bias always; others when data available

##### Prop archive (Phase D2)
- **evaluation/prop_archive.py:** Append-only Parquet.
  - 19-column schema, dedup on (game_id, player_id, stat_type, model_version)
  - Metadata: predicted_at, is_backfilled, model_version
  - Optional filters on load: stat_type, season

##### Prop CLI (Phase E1)
- **cli/props.py:** 3 commands registered as gridiron props.
  - gridiron props evaluate --model qb_pass_yards
  - gridiron props projections [--model all] [--top 20]
  - gridiron props backfill --model qb_pass_yards
  - Lazy trainer registry for fast --help
  - _train_and_enrich() shared helper

##### First CLI evaluation (qb_pass_yards)
- MAE: 63.4, RMSE: 80.6, R²: 0.118, Median AE: 51.8, N: 1,433
- Bias: +9.7 (over-predicting), 52.8% over-predicted
- Coverage: 93.8% (nominal 90%), interval width: 323.6

##### Key design decisions
- Position-aware NaN handling (>50% threshold per position)
- predicted_std combines model RMSE + player L3 rolling std
- No shift(1) on game context features (known pre-game)
- Builder does NOT dropna - trainer handles with position context
- PROP_FEATURE_COLS built programmatically from component modules
- Spread derived from FAVORITED + abs(VEGAS_LINE)
- Normal CDF for P(over) V1 - upgradeable to empirical later
- Lean/tier thresholds consistent with game model post-processing

##### Deferred
- E2: DraftKings prop odds ingest
- Champion/challenger for props (RF, XGBoost)
- Integration/E2E tests for prop pipeline
- Snap % features (nflreadpy doesn't expose snap counts)

##### Tests added: ~90 new
- test_prop_post_process.py (26 tests, 7 classes)
- test_prop_archive.py (16 tests, 3 classes)
- test_prop_metrics.py (23 tests, 7 classes)
- test_builder.py (unit tests for unified builder)
- test_qb_rush_yards.py (5 tests)
- Updates to test_qb_pass_yards, test_rb_rush_yards, test_wr_rec_yards,
  test_te_rec_yards (PROP_FEATURE_COLS migration)

##### Files added

| File |
|------|
| src/gridiron_edge/features/player/builder.py |
| src/gridiron_edge/features/player/_columns.py |
| src/gridiron_edge/features/player/game_context.py |
| src/gridiron_edge/features/player/usage.py |
| src/gridiron_edge/models/prop_prediction/post_process.py |
| src/gridiron_edge/models/prop_prediction/qb_rush_yards.py |
| src/gridiron_edge/evaluation/prop_metrics.py |
| src/gridiron_edge/evaluation/prop_archive.py |
| src/gridiron_edge/cli/props.py |
| tests/unit/features/test_builder.py |
| tests/unit/models/test_prop_post_process.py |
| tests/unit/models/test_qb_rush_yards.py |
| tests/unit/evaluation/test_prop_metrics.py |
| tests/unit/evaluation/test_prop_archive.py |

##### Files modified

| File | Change |
|------|--------|
| src/gridiron_edge/models/prop_prediction/base.py | Rewired to build_prop_features, HOLDOUT_SEASONS, position-aware NaN |
| src/gridiron_edge/models/prop_prediction/qb_pass_yards.py | Removed _build_features override |
| src/gridiron_edge/models/prop_prediction/rb_rush_yards.py | Removed _build_features override |
| src/gridiron_edge/models/prop_prediction/wr_rec_yards.py | Removed _build_features override |
| src/gridiron_edge/models/prop_prediction/te_rec_yards.py | Removed _build_features override |
| src/gridiron_edge/features/player/rolling.py | Added optional df param |
| src/gridiron_edge/features/player/matchup.py | Added optional df param, fixed line length |
| src/gridiron_edge/cli/main.py | Registered props_app |

##### Summary
- **9 new source files**, 8 modified
- **5 new test files**, 4 modified, **~90 new tests**
- All quality gates green: ruff, pyrefly, pytest

## 2026-06-04 - Sigma/Margin_std Recalibration & Versioned Model Cleanup - Complete

Recalibrated spread derivation parameters and confidence tiers after
TimeSeriesSplit retrain. Cleaned all vestiges of old versioned model names.

##### Sigma/margin_std recalibration
- Calibrated on holdout seasons (2023–2025) using existing
  calibrate_spread_sigma() infrastructure
- random_forest: sigma 13.97→10.63, margin_std 12.85→13.54
- xgboost: sigma 13.95→11.43, margin_std 13.44→13.34
- logistic: sigma 12.75→11.99, margin_std 13.53→13.29
- Spread ranges compressed (e.g. RF [-43, 16] → [-33, 12])
- Old sigmas were inflated because StratifiedKFold CV pushed
  models toward overconfident probabilities

##### Confidence tier rework
- Old approach: band_width (win_prob_hi - win_prob_lo) thresholds
  at 0.65/0.82. With honest margin_std, band_width was nearly
  constant (~0.95 for all games), making tiers useless (98.7% Low)
- New approach: probability distance from 0.5, folded to favorite
  side. Thresholds: >= 0.70 High, >= 0.60 Moderate, else Low
- Uses max(prob, 1-prob) to avoid IEEE 754 subtraction artifacts
- Validated win rates: High ~80%, Moderate ~65%, Low ~54%
- Distribution: ~23% High / ~30% Moderate / ~47% Low

##### Versioned model cleanup
- Removed all versioned entries (rf_v1–v3, xgb_v1–v3, logistic_v1–v4,
  elo_v1) from _MODEL_SIGMAS and _MODEL_MARGIN_STDS dicts
- Cleaned prediction archive of ~90K old versioned-model rows
- Updated all versioned model references in tests, docstrings,
  comments, and diagnostics colors to unversioned champion names
- total_rf_v1 references intentionally preserved (not part of
  champion/challenger system)

##### Scripts added
- scripts/recalibrate_sigma.py - holdout sigma/margin_std calibration
- scripts/clean_archive.py - archive cleanup for deprecated model versions

##### Files changed

| Action | File |
|---|---|
| Modified | src/gridiron_edge/models/game_prediction/post_process.py |
| Modified | src/gridiron_edge/evaluation/diagnostics.py |
| Modified | src/gridiron_edge/evaluation/backfill.py |
| Modified | src/gridiron_edge/models/artifact.py |
| Modified | src/gridiron_edge/models/base.py |
| Modified | src/gridiron_edge/cli/evaluate.py |
| Modified | tests/unit/models/test_post_process.py |
| Modified | tests/unit/models/test_pipeline.py |
| Modified | tests/unit/market/test_recommendations.py |
| Modified | tests/integration/test_edges_cli.py |
| Modified | tests/integration/test_betting_cli.py |
| Modified | PLAN.md |
| Modified | HANDOFF.md |
| Added | scripts/recalibrate_sigma.py |
| Added | scripts/clean_archive.py |

##### Summary
- **2 new scripts**, 6 source files modified, 5 test files modified
- Resolves "Recalibrate sigma/margin_std after retrain" debt item

## 2026-06-04 - Champion/Challenger Model Refactor - Complete

Replaced versioned model variants with a champion/challenger system and
fixed temporal CV leakage in all model families.

#### Temporal CV fix
- `_features.py`: added chronological sort (`sort_values(["YEAR", "WEEK_NUM"])`)
  in `_prepare_data` so TimeSeriesSplit respects temporal ordering
- `tree.py`: switched RF and XGB from `StratifiedKFold(shuffle=True)` to
  `TimeSeriesSplit(n_splits=5)` for hyperparameter search CV
- `logistic.py`: switched `LogisticRegressionCV` from default 5-fold to
  explicit `TimeSeriesSplit(n_splits=5)` fold list
- `_features.py`: added `MIN_CV_TRAIN_ROWS = 4000` constant - early
  TimeSeriesSplit folds with <4000 rows are skipped during HP search
  to avoid undersized training sets biasing toward conservative HPs

#### Champion/challenger promotion system
- New module: `evaluation/champion.py`
  - `PromotionCriteria`: gate thresholds (Brier ≥ 0.002 improvement,
    ECE ≤ 0.01 degradation, AUC ≤ 0.01 degradation)
  - `ComparisonResult`: full comparison outcome with per-gate results
  - `compare_models()`: runs all gates, returns verdict
  - `format_comparison()`: human-readable metric table with ✅/❌ gates
  - `extract_metrics()`: standardised metric dict from ModelMetadata
- 13 unit tests (`tests/unit/evaluation/test_champion.py`)

#### Simplified model registry
- Replaced 10 versioned registrations with 3 unversioned champions:
  `random_forest`, `xgboost`, `logistic`
- Old versioned names (rf_v1–v3, xgb_v1–v3, logistic_v1–v4) removed
  from PredictorRegistry
- Versioned names retained only in `post_process.py` sigma/margin_std
  dicts for backward compatibility with old prediction archives
- Default model in `cli/edges.py` changed from `random_forest_v3` to
  `random_forest`
- Updated `diagnostics.py` model colors, `predictor.py` docstrings,
  `__init__.py` docstrings, `artifact.py` examples

#### CLI updates (`cli/models.py`)
- `gridiron models train <name>`: auto-compares challenger vs champion
  using promotion gates. First training auto-saves as champion.
  Backup/restore on rejection.
- `--force`: promote despite failed gates
- `--no-promote`: train and compare without replacing champion
- `gridiron models info <name>`: shows all 5 holdout metrics
- Removed `--overwrite` flag (replaced by auto-compare flow)

#### All training functions now store 5 holdout metrics
- Brier, ECE, AUC, log loss, accuracy stored in `parameters` dict
- RF: added `expected_calibration_error`, `roc_auc`, `log_loss`, `accuracy`
- XGB: added `roc_auc`, `log_loss`, `accuracy` (ECE already existed)
- Logistic: added all 4 (none existed previously)

#### Retrained champions (honest temporal CV metrics)

| Model | Brier | ECE | AUC | Accuracy | Notes |
|---|---|---|---|---|---|
| xgboost | 0.218 | 0.014 | 0.691 | 64.0% | 🏆 Auto-selected champion |
| random_forest | 0.220 | 0.013 | 0.702 | 64.3% | Best calibration |
| logistic | 0.225 | 0.017 | 0.683 | 63.5% | |
| elo_v2 (baseline) | 0.227 | 0.073 | 0.676 | 62.2% | All ML models beat Elo |

Note: metrics are lower than old rf_v3 (Brier 0.195, AUC 0.774) because
the old StratifiedKFold CV inflated HP selection. The new numbers are the
honest ones. Calibration (ECE) improved dramatically (0.036 → 0.013).

#### Files changed

| Action | File |
|---|---|
| Added | `src/gridiron_edge/evaluation/champion.py` |
| Modified | `src/gridiron_edge/models/game_prediction/tree.py` |
| Modified | `src/gridiron_edge/models/game_prediction/logistic.py` |
| Modified | `src/gridiron_edge/models/game_prediction/_features.py` |
| Modified | `src/gridiron_edge/models/game_prediction/post_process.py` |
| Modified | `src/gridiron_edge/models/game_prediction/predictor.py` |
| Modified | `src/gridiron_edge/models/game_prediction/__init__.py` |
| Modified | `src/gridiron_edge/models/artifact.py` |
| Modified | `src/gridiron_edge/cli/models.py` |
| Modified | `src/gridiron_edge/cli/edges.py` |
| Modified | `src/gridiron_edge/evaluation/diagnostics.py` |
| Added | `tests/unit/evaluation/test_champion.py` |
| Modified | `tests/unit/models/test_tree_models.py` |
| Modified | `tests/integration/test_edges_cli.py` |
| Modified | `tests/unit/market/test_recommendations.py` |

#### Summary
- **1 new source file**, 10 modified
- **1 new test file**, 3 modified, **13 new tests**
- All quality gates green: ruff, pyrefly, pytest

## 2026-06-03 - W6: Portfolio & Bet Tracking - Complete

The feedback loop - track bets, measure performance, prove (or disprove)
the system works.  The M2 milestone.  Builds on W5 (edge context for
bets), W3 (market math for PnL), and W1 (odds ledger for CLV on
settlement).

#### Bet ledger (`betting/ledger.py`)
- Append-only Parquet ledger following the `archive.py` pattern
- 20-column schema: bet context (game, market, side, odds, stake, book),
  model context (version, prob, EV, strength, tier), settlement
  (status, settled_at, pnl, closing_line, closing_odds, clv)
- `compute_pnl()`: pure function - won = stake × (decimal_odds − 1),
  lost = −stake, push/open = 0
- `log_bet()`: generate UUID, append row with status "open", return bet_id
- `settle_bet()`: validate open, compute PnL, optionally compute CLV
  from odds ledger (ML = probability-based, spread/total = point-based)
- `load_bets()`: load with filters (status, season, week, market_type, book)
- Fixed pandas FutureWarning: `dropna(axis=1, how="all")` + `reindex` for concat
- Fixed pandas FutureWarning: `pd.to_datetime()` cast before `settled_at` assignment
- 24 unit tests (`tests/unit/betting/test_ledger.py`)

#### Bankroll management (`betting/bankroll.py`)
- Decoupled from ledger - CLI orchestrates both
- Transaction types: deposit, withdraw, bet_placed, bet_settled
- Sign convention: deposits/settlements = positive, withdrawals/bets = negative
- `deposit()` / `withdraw()`: record cash movements (positive amounts only)
- `record_bet_placed(stake)`: record stake leaving bankroll
- `record_bet_settled(stake, pnl)`: record gross return (stake + pnl)
  - won: stake + profit, lost: 0, push: stake
- `current_balance()`: sum of all signed transactions
- `balance_history()`: running balance DataFrame with cumulative sum
- `load_transactions()`: load with optional txn_type filter
- Same `dropna` + `reindex` concat pattern as ledger
- 23 unit tests (`tests/unit/betting/test_bankroll.py`)

#### Performance analytics (`betting/performance.py`)
- Pure DataFrame-in, results-out - no I/O
- `record()`: W-L-P counts, win_pct (pushes excluded from denominator),
  optional `split_by` for grouping
- `roi()`: total_staked, total_pnl, roi_pct, optional `split_by`
- `clv_summary()`: mean/median CLV, % positive, n_bets
- `ev_analysis()`: mean_ev_at_bet, mean_actual_roi, ev_vs_actual_gap
- `streak_analysis()`: current streak (±), longest W/L streaks,
  pushes break streaks
- `summary()`: combined dashboard dict calling all of the above
- Kelly adherence deferred (requires `recommended_stake` in ledger schema)
- 22 unit tests (`tests/unit/betting/test_performance.py`)

#### CLI (`cli/betting.py`)
- 8 commands registered as `gridiron bet` in `cli/main.py`
- `gridiron bet log`: record bet → `log_bet()` + `record_bet_placed()`
- `gridiron bet settle <id> <result>`: settle → `settle_bet()` +
  `record_bet_settled()`, optional CLV via `--with-clv/--no-clv`
- `gridiron bet list`: show bets with optional status/market filters
- `gridiron bet summary`: performance dashboard with optional `--split-by`
- `gridiron bet balance`: current balance + recent transaction history
- `gridiron bet export`: CSV export to `data/output/bets/`
- `gridiron bet deposit <amount>`: add funds
- `gridiron bet withdraw <amount>`: remove funds
- Graceful error handling throughout (not found, already settled, invalid amount)
- 17 integration tests (`tests/integration/test_betting_cli.py`)

#### Manual validation
- Full round-trip verified: deposit → log → list → settle → summary →
  balance → export → withdraw
- Math verified: deposit $1000, bet $100 at −150, won → PnL +$66.67,
  balance $1066.67. Second bet $50 spread, lost → balance $1016.67.
  Withdraw $200 → balance $816.67. All correct.

#### Files changed
| Action | File |
|---|---|
| Added | `src/gridiron_edge/betting/__init__.py` |
| Added | `src/gridiron_edge/betting/ledger.py` |
| Added | `src/gridiron_edge/betting/bankroll.py` |
| Added | `src/gridiron_edge/betting/performance.py` |
| Added | `src/gridiron_edge/cli/betting.py` |
| Modified | `src/gridiron_edge/cli/main.py` (import + register `betting_app`) |
| Added | `tests/unit/betting/__init__.py` |
| Added | `tests/unit/betting/test_ledger.py` |
| Added | `tests/unit/betting/test_bankroll.py` |
| Added | `tests/unit/betting/test_performance.py` |
| Added | `tests/integration/test_betting_cli.py` |

#### Summary
- **4 new source files**, 1 modified
- **4 new test files**, **86 new tests** (24 + 23 + 22 + 17)
- `betting/` package: 3 modules (ledger, bankroll, performance)
- All quality gates green: ruff, pyrefly, pytest

## 2026-06-02 - W5: Edge Engine - Complete

The convergence point - model predictions meet market prices to surface
betting edges.  Builds on W1 (odds ingest & joins), W2 (enriched
predictions with spreads/bands/tiers), and W3 (market math in
odds_math/kelly).

#### Edge calculation core (`market/edge.py`)
- Pure scalar functions, no I/O - follows the `odds_math.py` / `kelly.py` leaf pattern
- 3 frozen dataclasses: `MoneylineEdge`, `SpreadEdge`, `TotalEdge`
- `expected_value()`: EV = model_prob * decimal_odds - 1
- `moneyline_edge()`: no-vig debiases market, returns +EV side or None
- `spread_cover_prob()`: probit P(home covers) via calibrated `margin_std`
- `spread_edge()`: cover prob -> EV -> Kelly -> +EV side or None
- `total_cover_prob()`: probit P(over) via total model residual std
- `total_edge()`: over/under prob -> EV -> Kelly -> +EV side or None
- `classify_edge_strength()`: EV -> strong (>=5%) / moderate (2-5%) / lean (0-2%) / no_edge
- 37 unit tests (`tests/unit/market/test_edge.py`)

#### Edge report builder (`market/recommendations.py`)
- `pivot_odds_to_wide()`: long-format odds -> one row per game (handles duplicate fetches via groupby/last)
- `join_predictions_to_odds()`: inner-join predictions <-> wide odds on `game_id` (auto-pivots long odds)
- `compute_game_edges()`: single game -> list of edges across all available markets, graceful NaN handling
- `build_edge_report()`: full orchestrator -> 18-column report DataFrame
  - Kelly stake = bankroll * kelly_multiplier * kelly_frac (capped at bankroll * kelly_multiplier)
  - `classify_edge_strength()` applied to every row
- `rank_edges()`: filter to `ev > min_ev`, sort descending
- 21 unit tests (`tests/unit/market/test_recommendations.py`)

#### Closing Line Value (`market/clv.py`)
- `closing_line_value()`: probability-based CLV = (close_prob - bet_prob) / bet_prob
- `spread_clv()`: point-based CLV for spread bets (home: bet - close; away: close - bet)
- `total_clv()`: point-based CLV for total bets (over: close - bet; under: bet - close)
- `extract_opening_odds()` / `extract_closing_odds()`: first / last pull per (game_id, market, side) from ledger
- `build_clv_report()`: augments edge report with `opening_value`, `closing_value`, `clv` columns
- `summarize_clv()`: mean, median, pct positive, edge count
- Reuses `pivot_odds_to_wide` from `recommendations.py` via `_pivot_and_suffix()` (DRY)
- 30 unit tests (`tests/unit/market/test_clv.py`)

#### CLI (`cli/edges.py`)
- `gridiron edges report --week N --season YYYY-YYYY`
  - Loads prediction archive + current odds -> builds edge report -> ranks by EV
  - Rich console table: color-coded EV (green/yellow/dim), Kelly stakes, confidence tiers
  - CSV export via `--format csv` to `data/output/edges/`
  - Options: `--model-version`, `--bankroll`, `--kelly-multiplier`, `--min-ev`
- `gridiron edges clv --season YYYY-YYYY`
  - Loads predictions + full odds ledger -> builds edge report -> computes CLV -> summary stats
- Graceful empty-data handling throughout (no predictions, no odds, no edges)
- Registered in `cli/main.py` as `edges_app`
- 6 integration tests (`tests/integration/test_edges_cli.py`)

#### Files changed
| Action | File |
|---|---|
| Added | `src/gridiron_edge/market/edge.py` |
| Added | `src/gridiron_edge/market/recommendations.py` |
| Added | `src/gridiron_edge/market/clv.py` |
| Added | `src/gridiron_edge/cli/edges.py` |
| Modified | `src/gridiron_edge/cli/main.py` (import + register `edges_app`) |
| Modified | `src/gridiron_edge/market/__init__.py` (re-exports) |
| Added | `tests/unit/market/test_edge.py` |
| Added | `tests/unit/market/test_recommendations.py` |
| Added | `tests/unit/market/test_clv.py` |
| Added | `tests/integration/test_edges_cli.py` |

#### Summary
- **4 new source files**, 2 modified
- **4 new test files**, **94 new tests** (37 + 21 + 30 + 6)
- `market/` package: 5 modules (odds_math, kelly, edge, recommendations, clv)
- All quality gates green: ruff, pyrefly, pytest

## 2026-06-02 - W2: Richer Game Model Outputs - Complete

Extended game prediction models to produce spread, total, projected scores,
uncertainty bands, and confidence tiers - not just win probability.

#### Post-processing enrichment (`post_process.py`)
- **Spread derivation:** probit link with per-model sigma calibration (13 variants)
  - Best: random_forest_v3 (sigma=13.97, spread MAE vs Vegas=3.16, r=0.80)
- **Isotonic recalibration:** infrastructure built, decision gate rejected for rf_v3
  (holdout ECE 0.036 already excellent; recalibration worsened it to 0.083)
- **Uncertainty bands:** 90% credible intervals via spread ± z*margin_std → probit
  - Per-model margin_std registry (best: rf_v3 at 12.85, worst: elo_v1 at 13.89)
- **Confidence tiers:** band width → High (<0.65) / Moderate (0.65–0.82) / Low (≥0.82)
  - Validated: High 96.8%, Moderate 86.8%, Low 64.0% favored-team win rate
- **Projected scores:** home = (total - spread) / 2, away = (total + spread) / 2
  - Home MAE: 6.95, Away MAE: 6.74, near-zero bias

#### Total points model (`total.py`)
- Random Forest regressor targeting actual_total = PTS_WINNER + PTS_LOSER
- Uses same 107-feature expanded set as win models
- TimeSeriesSplit CV (not KFold) to avoid temporal leakage
- total_rf_v1 trained: holdout MAE=10.27, RMSE=13.17 (n=1,467)
- Competitive with Vegas closing totals (model MAE 3.11 vs closing O/U, r=0.64)

#### Prediction pipeline (`pipeline.py`)
- Composable orchestrator: load → predict (win) → predict (total) → build rows → enrich
- `predict_games()` replaces monolithic `_predict_historical_tree()` internals
- `build_game_predictions()` maps raw model output to game-level rows
- All model families (elo, logistic, tree) now produce enriched predictions

#### Archive schema extension (`archive.py`)
- 8 new columns: model_spread, model_total, projected_home_score,
  projected_away_score, margin_std, win_prob_lo, win_prob_hi, confidence_tier
- Backward compatible: old archives load with NaN fill for missing columns

#### Validation report (rf_v3 vs Vegas)
| Metric | Value |
|--------|-------|
| Spread MAE vs closing line | 3.16 |
| Spread correlation | 0.80 |
| Total MAE vs closing O/U | 3.11 |
| Total correlation | 0.64 |
| Home score MAE | 6.95 |
| Away score MAE | 6.74 |
| High confidence fav win% | 96.8% |
| Moderate confidence fav win% | 86.8% |
| Low confidence fav win% | 64.0% |

**Note:** VEGAS_LINE uses opposite sign convention from model_spread
(positive = home favored vs negative = home favored). Documented in HANDOFF.md.

#### Phase reference cleanup
Scrubbed all Phase A/B/C/D/E/20c/20d/20e and W2 references from source and
test files. Replaced with descriptive terminology. PLAN.md and CHANGELOG.md
retain historical phase references since they are historical records.

#### Tests added: 44 new (total ~456)
- test_post_process.py: 33 → 55 (bands, tiers, enrichment)
- test_total.py: 11 (projected scores, enrichment with total)
- test_pipeline.py: 7 (build_game_predictions)
- test_archive_schema.py: 4 (schema extension, backward compat)

## 2026-06-01 - Phase 20e Feature Engineering Complete

Completed Priorities 1-7 + 14-15 across three batches:
- Batch 1: Rest differential + explosive play rate (+8 columns)
- Batch 3: PBP efficiency (success splits, 3rd down, red zone,
  turnovers, sack rate) (+36 columns)
- Batch 2/15: Weather & venue wiring verified already complete

Feature count: _EXPANDED_FEATURES 16 -> 107. EPA_COLS 8 -> 22.
Model features now cover EPA, efficiency splits, explosiveness,
situational football, turnovers, pass rush, rest, weather, venue.

Remaining Phase 20e backlog: Priorities 8-13 (CPOE, pace, score
differential, penalties, special teams, coaching). These require
additional PBP columns or external data sources.

Next active workstream: W2 (Richer Game Model Outputs).

## W3: Market Intelligence Foundation - 2026-05-31

### New package: `market/`
- Pure-math leaf package at `src/gridiron_edge/market/` - no data dependencies,
  no pandas, no I/O

### `market/odds_math.py`
- `american_to_decimal()`: American → decimal odds conversion
- `american_to_implied_prob()`: American → raw implied probability (includes vig)
- `decimal_to_american()`: decimal → American; even-money normalises to +100
- `hold_pct()`: bookmaker overround for two-way markets
- `no_vig()`: fair probabilities via power method (default) or additive rescaling
- `_power_devig()`: bisection solver for `raw_a^k + raw_b^k = 1` - no scipy

### `market/kelly.py`
- `kelly_fraction()`: full-Kelly optimal fraction; returns 0 when edge ≤ 0
- `kelly_stake()`: dollar amount using fractional Kelly (default quarter-Kelly)
- Input validation: probability must be in (0, 1), bankroll ≥ 0, fraction in [0, 1]

### Tests added (64)
- `test_odds_math.py` (42) - conversions, roundtrips, extreme odds (±10000),
  hold percentage, no-vig additive vs power, sums-to-one, fair-probs-not-above-raw
- `test_kelly.py` (22) - positive/negative/zero edge, fractional staking,
  zero bankroll, guard rails on probability/bankroll/fraction

### Deferred
- `market/consensus.py` - deferred until multi-book data available (W7)

## W1: Quick Wins & Unblocking - 2026-05-31

### DK Unicode Minus Fix
- `ingest/odds/draftkings.py` → `_norm_display_odds_american()`: handle Unicode
  minus (U+2212) before `isdigit()` check and `int()` conversion. DraftKings API
  returns `"−150"` with U+2212 instead of ASCII hyphen; this caused `ValueError`
  on all negative odds parsing.

### DK `game_id` Resolver
- New module: `ingest/odds/_game_id.py`
- `team_long_to_short()`: reverse lookup from `NFLVERSE_SHORT_TO_LONG`, with
  historical relocation codes (`OAK`, `SD`, `STL`) deprioritized so current
  codes (`LV`, `LAC`, `LA`) always win
- `build_game_id()`: constructs canonical `YYYY_WW_AWAY_HOME` format
- `resolve_dk_game_ids()`: vectorized column addition supporting both
  intermediate (`home_team`/`away_team`) and wide (`team`/`opponent`/`location`)
  DataFrame formats

### End-to-End Odds Join Validation
- Integration test confirms predictions ↔ odds join on `game_id` at 100% match
  rate on synthetic data, with left-join surfacing unmatched games as nulls

### Tests added (25)
- `test_draftkings_parse.py` (9) - Unicode minus, positive, int/float passthrough,
  fallback keys, non-numeric string, missing keys
- `test_game_id.py` (13) - team lookup, all 32 teams resolve, build_game_id format,
  week padding, unknown teams → None, both DataFrame formats, column preservation
- `test_odds_join.py` (3) - canonical format validation, inner join match rate,
  left join null surfacing


## W0 Complete: Test Framework Build-Out - 2026-05-31

### Summary
Professional three-tier testing infrastructure (unit → integration → e2e)
with automated quality gates, shared fixtures, and 412 tests at 40% coverage.

### Phases completed
- **Phase 0** - Foundation: directory restructure, auto-markers, shared fixtures,
  pre-commit/pre-push hooks, coverage config
- **Phase 1** - Core & Datasets: 60 tests covering constants, paths, settings,
  registry, loaders, writers, accessor
- **Phase 2** - Missing Features: 63 tests covering all 11 feature modules,
  feature registry, FeatureSpec protocol
- **Phase 3** - Models & Evaluation: 35 tests covering Predictor/Trainable
  protocols, model registry, artifact store, backfill, select, tune, diagnostics
- **Phase 4** - Ingest, Transform, Sim: 65 tests covering odds store, nflverse
  helpers, sim types/constants, geo/haversine, DK fixture validation
- **Phase 5** - Integration & E2E: 28 tests covering dataset roundtrips,
  artifact roundtrips, CLI workflows, full prediction pipeline via MiniRepoBuilder
- **Deferred resolution** - Added test_tune.py (16 tests), test_diagnostics.py
  (8 tests), removed slow training tests that exercised sklearn/xgboost internals

### Coverage baseline
- 412 tests, 0 failed, 0 deselected
- 40.04% line coverage (threshold: 40%, ratchet up over time)
- Core business logic (features, datasets, evaluation) at 80-100%
- Sim, viz, CLI, and model training code deferred to respective workstreams

### Deferred test areas (to be added with respective workstreams)
- Numba sim kernels: `test_engine.py`, `test_playoffs.py` (sim workstream)
- DK API mocking: full `test_draftkings.py` (odds workstream)
- Elo predictor: `test_elo_predictor.py` (elo workstream)
- Transform ETL: `test_epa_transform.py` (data pipeline workstream)
- Cosmetic: migrate inline imports → top-level; migrate local helpers → shared fixtures


### Test Framework Build-Out - 2026-05-31

Established professional three-tier testing infrastructure.

**Test directory restructure**
- Restructured `tests/` into `unit/`, `integration/`, `e2e/` subdirectories
- Tests auto-tagged by directory via `pytest_collection_modifyitems` in root conftest - no manual `@pytest.mark` decorators needed
- Existing tests moved to `tests/unit/` with zero import changes required

**Shared fixtures**
- `tests/fixtures/dataframes.py` - 9 centralized DataFrame factories: `make_games`, `make_modeling_rows`, `make_stadiums`, `make_elo_state`, `make_epa_by_game`, `make_weather_enriched`, `make_eval_df`, `make_predictions`, `make_accessor`
- `tests/fixtures/repos.py` - composable `MiniRepoBuilder` class (builder pattern: `.with_games().with_stadiums().with_elo_state().build()`)
- Replaces duplicated `_make_games()`, `_make_eval_df()`, `mini_repo` patterns across 8+ test files

**Pre-commit / pre-push hooks:**
- Added `.pre-commit-config.yaml` with two stages:
  - `pre-commit`: ruff lint + format, pyrefly type check, unit tests
  - `pre-push`: integration + e2e tests
- Installed via `pre-commit install` + `pre-commit install --hook-type pre-push`
- Safety valve: `|| test $? -eq 5` allows commits during incremental marker migration

**Pytest configuration:**
- Added markers to `pyproject.toml`: `unit`, `integration`, `e2e`, `slow`, `network`
- `--strict-markers` enforced - no typos in marker names
- Coverage config added: `fail_under = 60`, `show_missing = true`

**Fixed drifted tests**
- `test_home_field_feature`: `GAME_LOCATION` `"NULL_VALUE"` → `"H"` (aligned with constants consolidation)
- `test_weather`: `_make_modeling_row` returns DataFrame not dict; `test_null_value_string_gives_nan` assertion updated
- `test_tree_models`: imports updated for `_epa_window` module extraction (`_rebuild_features_with_window`, `_EPA_WINDOW_OPTIONS`)
- `test_features_pipeline`: `pd.read_csv` → `pd.read_parquet` for `modeling_base`/`modeling_full`
- Model training tests (`TestRandomForestV1Training`, `TestXGBoostV1Training`) marked `@pytest.mark.slow` (~15min each)

**Tooling**
- `mirror_repo_to_sharepoint.py` - mirrors repo to SharePoint-synced folder for Copilot indexing. Copies `.py` files as `.py.txt` with SOURCE headers; preserves `.md`/`.json`/`.yaml` as-is. Supports `--clean`, `--dry-run`, `--extra-ext`.


## Thermonuclear Code Quality Review - 2026-05-30

Eight review batches across the full codebase, followed by six implementation passes and full pipeline validation. All changes committed in four atomic commits.

### Pass 1+2 - Constants consolidation + Elo engine

**Constants - single source of truth in `core/constants.py`:**
- `HOME_GAME_LOCATION = "H"`, `AWAY_WIN_LOCATION = "@"`, `HOLDOUT_SEASONS`, `EXPANSION_TEAMS` - all previously defined independently in 2–4 files each
- Retired the PFR-era `"NULL_VALUE"` home-game sentinel → `"H"` for `GAME_LOCATION`; `""` for all missing data fields (GAMETIME, STADIUM, ROOF, SURFACE, GAME_DATE, GAME_DAY_OF_WEEK) across the transform layer
- All consumers updated: `venue_hfa`, `home_field`, `record`, `primetime`, `backfill`, `tune`, `elo/predictor`, `metrics`, `schedule_nflverse`, `games_nflverse`, `_nflverse_common`
- Deleted dead placeholder packages: `datasets/contracts/`, `analytics/`, `config/`

**Elo engine - parameterised divisor:**
- `ratings/elo/core.py`: `elo_win_probability(divisor=DEFAULT_ELO_DIVISOR)` and `update_elo(divisor=)` - divisor no longer hardcoded to 480
- `EloTableConfig` gains `divisor: float = 480.0`; `_build_elo_dict` passes it through
- `tune.py`: `_win_prob` deleted - `_simulate_and_score` delegates to `core.elo_win_probability`
- `SimulationConfig` gains `divisor: float = 480.0`; numba `_elo_win_prob`/`_elo_update` in `sim/_engine.py` accept divisor as a parameter
- `gridiron sim run` gains `--divisor` flag

### Batch 1-8 code review fixes

Individual file-level fixes from all 8 review batches:
- `DatasetSpec`: dropped redundant `key` field (14 instantiations updated)
- `FeatureRegistry`: duplicate-name guard + descriptive `KeyError` in `register()`/`get()`
- `features/team/epa.py`: vectorised inner EPA rolling loop; extracted `_join_team_epa` helper; `EPA_COLS` made public
- `ratings/elo/table.py`: deleted backwards-compat alias `update_elo_state_table_incremental`
- `evaluation/diagnostics.py`: filled `_MODEL_COLORS` gaps for logistic_v4, random_forest_v1/v2, xgboost_v2
- `evaluation/metrics.py`: removed duplicate `_archive_path` and `load_prediction_log` - now imports from `archive.py`
- `viz/excel.py` → `viz/rankings.py`: renamed; `cli/output.py` updated
- `metrics/travel/geo.py`: `Tude` type alias renamed to `CoordinateValue`
- `backfill.py`, `tune.py`, `metrics.py`: local `_AWAY_WIN_LOCATION` definitions removed, imported from `core.constants`

### Pass 3 - File decomposition

**`sim/season.py`** (1235 lines) split into three files:
- `sim/_types.py` - constants, all config dataclasses (`SimulationConfig`, `SimPaths`, `TeamIndex`, `ScheduleArrays`, `SimulationResults`), `_log_phase`, `format_record`. Pure-data leaf - no I/O, no numba.
- `sim/_engine.py` - numba kernels: `_elo_win_prob`, `_elo_update`, `apply_actuals_to_matrices`, `simulate_remaining_regular_season`, `precompute_game_counts`
- `sim/season.py` - data loading, output builders, `run_full_simulation` (~734 lines)
- `sim/__init__.py` - public API re-exports; sync assertions validate `playoffs.py` constants match `_types.py` at import time
- `viz/charts.py` - import updated from `sim.season` → `sim._types`

**`models/game_prediction/_shared.py`** (333 lines) split:
- `_columns.py` - schema version, all column lists, `FeatureSet` dataclass; pure-data leaf
- `_features.py` - feature engineering functions, `FEATURE_SETS` dict, `_prepare_data`, `_is_trained`
- `_shared.py` - thin re-export shim (33 lines)
- `logistic.py` and `tree.py` updated to import from new modules directly

**`models/game_prediction/tree.py`** (984 lines):
- `_epa_window.py` extracted - `_EPA_RAW_COLS`, `_EPA_COL_MAP`, `_EPA_WINDOW_OPTIONS`, `WindowData` NamedTuple, `_rebuild_features_with_window`, `_get_cached_window_data`
- `tree.py` reduced to 820 lines

**Final line counts:** no file exceeds 820 lines. `playoffs.py` ↔ `_types.py` constant sync is machine-checked at import time.

### Pass 4 - Feature dependency enforcement

- `features/base.py`: `FeatureSpec` gains `depends_on: Sequence[str] = ()` field
- `features/registry.py`: `validate_ordering(feature_names)` - raises `ValueError` at import time if ordering violates any `depends_on` constraint
- `features/pipeline.py`: calls `validate_ordering(FEATURES)` at module level
- Dependencies declared: `travel` → `home_field`; `venue_hfa` → `travel`; `schedule_strength` → `team_elo`

### Pass 5 - CLI stage-list pattern

- `cli/main.py`: 10 boolean flags replaced with `--skip STAGE` / `--only STAGE` repeatable options
- `ALL_STAGES` defines the canonical stage vocabulary: `fetch-games`, `clean-games`, `fetch-upcoming`, `clean-upcoming`, `fetch-weather`, `fetch-odds`, `build-epa`, `build-elo`, `build-features`
- Dead `build-epa` stage fixed - was declared but never executed
- `PLR0912`/`PLR0915` suppressions moved to `_run_pipeline_stages` where they belong; `run_data_pipeline` is now clean
- `evaluation/select.py` introduced - `collect_model_metrics`, `rank_models`, `compute_report_data` extracted from `cli/evaluate.py`

### Pass 6 - Archive schema migration

- `evaluation/archive.py`: `is_backfilled: bool` column added to schema; `build_archive_rows` and `append_to_prediction_log` gain `is_backfilled` parameter; `write_archive_rows` and `load_prediction_log` backward-compatible; `migrate_archive()` added
- `models/elo/predictor.py`: `_BACKFILL_TS` constant deleted; predictions use actual timestamp + `is_backfilled=True`
- `logistic.py`, `tree.py`: inline `datetime(1970, 1, 1)` sentinels replaced with actual timestamp + `is_backfilled=True`

### Post-commit fixes

- `ingest/weather/openweather.py` - `fetch_weather` now reads existing `weather_enriched.csv` and fetches only games not already enriched. Idempotent - safe to re-run without duplicating rows.
- `sim/season.py` - `run_full_simulation` raises `FileNotFoundError` with actionable message when the upcoming schedule CSV is empty, instead of a cryptic `IndexError`.

---

## Phase 20d - Tree-based models (RF + XGBoost)

- `models/game_prediction/tree.py` - Random Forest and XGBoost variants registered alongside logistic models
- `models/game_prediction/logistic.py` - v3 and v4 logistic variants added
- `PredictorRegistry` - `register` + `get` + `trainable_names()` pattern generalised
- `evaluation/tune.py` - hyperparameter grid search for Elo K/divisor and EPA window
- `evaluation/diagnostics.py` - calibration plots, model comparison charts

---

## Phase 20c - Model reporting

- `evaluation/select.py` - `select_model` + `generate_report` pipeline
- `cli/evaluate.py` - `evaluate report`, `evaluate select-model`, `evaluate calibration` commands
- Full model characterisation: Brier score, log loss, calibration, accuracy per season

---

## Phase 20b - Model evaluation infrastructure

- `evaluation/metrics.py` - Brier score, log loss, calibration table, accuracy
- `evaluation/backfill.py` - `backfill_model(model_version)` covering all registered models
- `evaluation/archive.py` - append-only prediction log at `predictions_log.parquet`
- `cli/evaluate.py` - `evaluate backfill`, `evaluate summary` commands

---

## Phase 20a - Prediction engine

- `models/game_prediction/logistic.py` - logistic v1 + v2 registered predictors
- `models/base.py` - `Predictor` + `Trainable` protocols
- `models/registry.py` - `PredictorRegistry`
- `models/artifact.py` - `ArtifactStore` (joblib-based)
- `cli/models.py` - `models train`, `models list` commands

---

## Phase 19 - Football state representation (EPA, rest, travel, records)

- `features/team/epa.py` - rolling EPA features from PBP data
- `features/team/rest.py` - days rest, short week, post-bye flags
- `features/team/travel.py` - km traveled, timezone shift
- `features/team/record.py` - win/loss/tie record, win streak
- `features/team/schedule_strength.py` - SOS, SOV
- `ingest/nflverse/pbp.py` - play-by-play ingestion
- `transform/clean/epa.py` - PBP → game-level EPA aggregation
- Schema v3 modeling file with all Phase 19 features

---

## Phase 18 - Evaluation infrastructure

- Prediction archive - append-only Parquet log
- `evaluation/metrics.py` - Brier score, log loss, calibration, accuracy
- `evaluation/backfill.py` - generic backfill covering all registered models
- `evaluation/tune.py` - Elo parameter grid search
- `datasets/manifest.py` - schema versioning for modeling files

---

## Phase 15-17 - Excel retirement, Scrapy retirement, dead code removal

- `ingest/odds/` - DraftKings odds ingest + append-only Parquet ledger
- `ingest/odds/store.py` - long-format odds storage with dedup
- `viz/predictions.py` - weekly matchup PNG + static HTML (migrated from notebook)
- `viz/rankings.py` - Elo rankings CSV (was Excel)
- Scrapy / PFR scraper fully deleted
- Dead stub files removed; all ruff/pyrefly gates passing

---

## Phase 13-14 - nflverse migration + console system

- Replaced PFR/Scrapy with `nfl_data_py` - bypasses Cloudflare
- `ingest/nflverse/` - game + schedule + upcoming ingestion
- `transform/clean/games_nflverse.py` + `schedule_nflverse.py` - canonical schema mappers
- `core/console.py` - timed step context manager, header/summary banners, verbose mode
- `core/logging.py` - WARNING in compact mode, DEBUG in verbose

---

## Phases 1-12 - Core refactor + tooling

Original migration from `data_pipelines/` + `model_pipelines/` + `utils/` into `src/gridiron_edge/`. uv migration, Ruff + Pyrefly quality gates, Google-style docstrings, full type annotation pass. See git history for full detail.

---

## Multi-book Line Shopping and model guidance

- Added exact multi-book Moneyline, Spread, and Total comparison with independent
  best-line and exact-line best-price classification.
- Added exhaustive selected-product probability and expected-value evaluation
  for every current quote, including unavailable, break-even, and negative-EV
  states.
- Added continuous Spread and Total playable thresholds at a documented -110
  reference price and fair Moneyline probabilities and prices.
- Added model-approved and preferred-offer classifications with maximum-EV tie
  preservation and selected-product provenance.
- Added the `/lines` API contract, generated OpenAPI ownership, and typed frontend
  integration.
- Added persisted toggleable highlighting, chronological matchup ordering,
  Eastern kickoff formatting, responsive comparison presentation, and preserved
  raw-market usability when highlighting is disabled.
- Added accessible hover, focus, and tap explanations for wager outcomes, pushes,
  American prices, model EV, and market classifications.
- Preserved exact quote identity, partial sportsbook coverage, and unavailable
  sportsbook or model states without fabricating coverage.
