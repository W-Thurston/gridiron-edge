# Gridiron Edge — Development Plan

> **Purpose:** single source of truth for *what to build next* and *why*.
> Updated at the start and close of every workstream.

| Document | Role |
|----------|------|
| **PLAN.md** (this file) | What is planned, what is active, what is deferred |
| **CHANGELOG.md** | What was built and when (completed workstream details) |
| **HANDOFF.md** | How the system works today (architecture, workflows, operations) |
| **ROADMAP.md** | Long-term strategic direction, workstream inventory, architecture decisions |

#### Status key

| Tag | Meaning |
|-----|---------|
| Done | Done — details in CHANGELOG.md |
| In progress | In progress |
| Planned | Planned / blocked |
| Deferred | Deferred |

---

## High-Level Priority Order

Agreed 2026-06-10. See ROADMAP.md for full workstream inventory.

| # | Workstream | Status |
|---|-----------|--------|
| 1 | Champion/Challenger for Props (RF + XGBoost) | Done |
| 2 | Game Model Refactor (align to props pattern) | Done |
| 3 | Integration & E2E Tests | Done |
| 4 | Deep Code Review + Test Suite Review | Planned |
| 5 | Scenario / "What If" Engine (W4.5) | Planned |
| 6 | API & Frontend (W8 + W9) | Planned |
| 7 | All External Odds (DK props, historical, line shopping) | Planned |
| 8 | Evaluate remaining work | Planned |

Two code reviews planned: one after workstream 3 (before building more),
one near completion of all workstreams. A separate dedicated test suite
review is included as part of workstream 4.

---

## Workstream 1: Champion/Challenger for Props (RF + XGBoost)

**Goal:** Add RandomForest and XGBoost as prop model types alongside
ElasticNet, build a champion selection system, and improve R² across all
5 stat families.

**Why it matters:** ElasticNet R² ranges from 0.071 (QB pass) to 0.203
(WR rec). Tree models capture nonlinear interactions ElasticNet cannot.
This is the single highest-impact improvement available.

**ROADMAP ref:** W4 extension.

## W4 Holdout Baselines (post-Unit-1)

Re-baselined 2026-06-19 with TimeSeriesSplit inner CV. Numbers below are
from the canonical training holdout (PropTrainer.train() output, not the
champion_cmd evaluate_prop_model display). Pre-Unit-1 baselines retained
in commit history for reference; the new fix produces near-identical
numbers per D2 in DECISIONS.md.

| Stat | ElasticNet MAE | RF MAE | XGB MAE | ElasticNet R² | Champion |
|------|----------------|--------|---------|---------------|----------|
| qb_pass_yards | 58.1 | 58.3 | 58.0 | 0.071 | elasticnet |
| qb_rush_yards | 16.4 | 16.5 | 16.4 | 0.099 | elasticnet (no model passed R²>0 gate) |
| rb_rush_yards | 25.0 | 25.0 | 24.9 | 0.168 | elasticnet |
| wr_rec_yards | 25.1 | 25.0 | 25.1 | 0.206 | random_forest |
| te_rec_yards | 18.3 | 18.3 | 18.3 | 0.187 | random_forest |

### Locked Decisions

| # | Decision | Choice | Rationale |
|---|----------|--------|-----------|
| 1 | Multi-model architecture | Factory in base class, `model_type` param on `train()` | Prop subclasses are spec-only — no training logic divergence across model types |
| 2 | Model types | ElasticNet, RandomForest, XGBoost | Ridge is subsumed by ElasticNet (l1_ratio=0) |
| 3 | Scaling | StandardScaler for ElasticNet, None for trees | Tree models don't need feature scaling |
| 4 | HP search strategy | Grid search evaluated on holdout set | Consistent with current approach; HOLDOUT_SEASONS split provides temporal validation |
| 5 | ElasticNet grid | 5 alpha × 5 l1_ratio = 25 combos | Current grid, keep unchanged |
| 6 | RF grid | `n_estimators` ∈ [100, 300, 500], `max_depth` ∈ [8, 12, 16, None], `min_samples_leaf` ∈ [5, 10, 20] = 36 combos | Standard, fast |
| 7 | XGB grid | `n_estimators` ∈ [100, 300, 500], `max_depth` ∈ [4, 6, 8], `learning_rate` ∈ [0.01, 0.05, 0.1], `subsample` ∈ [0.8, 1.0] = 54 combos | Standard, fast |
| 8 | Promotion primary metric | Lowest holdout MAE | Direct measure of prediction accuracy for regression |
| 9 | Guardrail 1 | R² > 0 | Must beat mean baseline |
| 10 | Guardrail 2 | Coverage ∈ [85%, 97%] for 90% nominal interval | Uncertainty estimates not wildly off |
| 11 | Fallback champion | ElasticNet if no model passes guardrails | Known stable baseline |
| 12 | Model persistence | Retrain on demand (no artifacts) | Same as current; artifact storage is a future optimization |
| 13 | Clip ranges | Per-spec via `clip_lo`, `clip_hi` on `PropModelSpec` | Centralized, not scattered in predict methods |
| 14 | Champion comparison | External function in `evaluation/champion.py`, generalized for both classification and regression | Same separation of concerns as game models |
| 15 | Comparison architecture | Single generalized `compare_models()` supporting both Brier (classification) and MAE (regression) gates | Unified champion/challenger pattern across all models |
| 16 | Selection loop location | Outside the trainer (CLI or utility function) | The trainer trains, the evaluator judges |

### Phase A: Prop Model Factory — Steps

#### A1: Add PropModelType enum and factory to base.py

New enum:

    class PropModelType(str, Enum):
        ELASTICNET = "elasticnet"
        RANDOM_FOREST = "random_forest"
        XGBOOST = "xgboost"

New factory method in PropTrainer:

    def _create_model(self, model_type: PropModelType, params: dict) -> tuple[Any, StandardScaler | None]

Returns `(model_instance, scaler_or_none)`. ElasticNet gets a
StandardScaler; RF and XGB get None.

New HP grid method:

    def _get_param_grid(self, model_type: PropModelType) -> list[dict]

Returns the parameter grid for the given model type.

Modified `train()` signature:

    def train(self, *, model_type: str = "elasticnet", repo: Path | None = None) -> PropModelMetadata

Internally calls `_create_model()`, iterates over `_get_param_grid()`,
evaluates each on holdout, selects best, retrains on full training set.

New `_fit()` in base (**no longer abstract**):
- Iterates over param grid with tqdm progress bar
- Trains model + optional scaler with each param combo
- Evaluates MAE on holdout
- Selects best params
- Retrains on full training set with best params

New `_predict()` in base (**no longer abstract**):
- Applies scaler if present
- Calls `model.predict()`
- Clips to `spec.clip_lo` / `spec.clip_hi`

| Status | Step |
|--------|------|
| Not started | A1 |

#### A2: Add clip_lo / clip_hi to PropModelSpec

Update `PropModelSpec` dataclass:

    @dataclass(frozen=True)
    class PropModelSpec:
        name: str
        target_col: str
        position_filter: list[str]
        description: str
        clip_lo: float = 0.0
        clip_hi: float = 600.0

Clip ranges per stat family:

| Model | clip_hi |
|-------|---------|
| qb_pass_yards | 600 |
| qb_rush_yards | 200 |
| rb_rush_yards | 250 |
| wr_rec_yards | 300 |
| te_rec_yards | 250 |

Update all 5 subclass specs to set their clip_hi values.

| Status | Step |
|--------|------|
| Not started | A2 |

#### A3: Strip _fit() / _predict() / _scaler / _model from all 5 subclasses

Each subclass becomes ~15-20 lines: just the `spec` property.
All training + prediction logic lives in the base class.

Subclasses affected:
- `qb_pass_yards.py`
- `qb_rush_yards.py`
- `rb_rush_yards.py`
- `wr_rec_yards.py`
- `te_rec_yards.py`

| Status | Step |
|--------|------|
| Not started | A3 (depends on A1) |

#### A4: Add model_type to PropModelMetadata

Ensure `PropModelMetadata` (or its equivalent) stores `model_type` so
the champion comparison function knows which model type was used.
This should align with what `evaluation/champion.py` expects.

| Status | Step |
|--------|------|
| Not started | A4 |

### Phase B: Generalized Champion/Challenger Gates — Steps

#### B1: Generalize evaluation/champion.py for regression

Currently supports classification only (Brier, ECE, AUC).

Add regression support:

    @dataclass(frozen=True)
    class RegressionPromotionGates:
        max_mae_tolerance: float = 0.0   # challenger MAE must be <= champion MAE
        min_r2: float = 0.0              # must beat mean baseline
        min_coverage: float = 0.85       # 90% interval coverage floor
        max_coverage: float = 0.97       # 90% interval coverage ceiling

Design approach: generalize `compare_models()` to accept a `mode`
parameter ("classification" or "regression") or auto-detect based on
which metrics are present in the metadata. The function selects the
appropriate gate system.

Single `ComparisonResult` return type — same structure for both.

Existing `PromotionGates` (Brier/ECE/AUC) stays for classification.
New `RegressionPromotionGates` added for regression.

    def compare_models(
        champion: ModelMetadata | PropModelMetadata,
        challenger: ModelMetadata | PropModelMetadata,
        *,
        mode: str = "classification",
        classification_gates: PromotionGates = PromotionGates(),
        regression_gates: RegressionPromotionGates = RegressionPromotionGates(),
    ) -> ComparisonResult

`format_comparison()` also generalized to show the right metrics
and gate labels depending on mode.

| Status | Step |
|--------|------|
| Not started | B1 |

#### B2: Champion selection utility function

New function (in `evaluation/champion.py` or `evaluation/prop_champion.py`):

    def select_prop_champion(
        results: list[PropModelMetadata],
        gates: RegressionPromotionGates = RegressionPromotionGates(),
    ) -> tuple[PropModelMetadata, str]

Logic:
1. Filter to models passing all guardrails (R² > 0, coverage in range)
2. Among eligible models, select lowest MAE
3. If no model passes guardrails, return ElasticNet as fallback
4. Return `(champion_metadata, comparison_table_string)`

| Status | Step |
|--------|------|
| Not started | B2 (depends on B1) |

### Phase C: CLI Integration — Steps

#### C1: Add --model-type option to evaluate command

    gridiron props evaluate --model qb_pass_yards --model-type xgboost

If `--model-type` not specified, uses champion model type (default
elasticnet until champion selection is run).

| Status | Step |
|--------|------|
| Not started | C1 |

#### C2: Add champion command

    gridiron props champion --model qb_pass_yards
    gridiron props champion --model all

Trains all 3 model types for the specified stat family (or all 5
families), shows comparison table, reports champion:

    🏈 Champion/Challenger: qb_pass_yards

      Model Type     MAE    RMSE      R²  Coverage  Status
      ─────────────────────────────────────────────────────
      elasticnet    58.0    72.6   0.071    93.8%   Eligible
      random_forest 52.1    66.3   0.142    91.2%   ★ CHAMPION
      xgboost       53.4    67.8   0.128    90.5%   Eligible

| Status | Step |
|--------|------|
| Not started | C2 (depends on A1–A4, B1–B2) |

#### C3: Train all 15 models and validate

Run `gridiron props champion --model all`. Paste results into this
document and CHANGELOG.md. Verify:
- All 15 models train without error
- At least one non-ElasticNet model wins per stat family
- Coverage stays in [85%, 97%] for all models
- R² improves for QB pass (currently 0.071)

| Status | Step |
|--------|------|
| Not started | C3 (depends on C2) |

### Phase A/B/C: Test Plan

**File:** `tests/unit/models/test_prop_champion.py`

| Test | What It Validates |
|------|-------------------|
| `test_model_type_enum_values` | 3 values: elasticnet, random_forest, xgboost |
| `test_create_model_elasticnet` | Returns (ElasticNet, StandardScaler) |
| `test_create_model_rf` | Returns (RandomForestRegressor, None) |
| `test_create_model_xgb` | Returns (XGBRegressor, None) |
| `test_param_grid_elasticnet` | 25 combos |
| `test_param_grid_rf` | 36 combos |
| `test_param_grid_xgb` | 54 combos |
| `test_clip_ranges_from_spec` | Predictions clipped to spec.clip_lo/clip_hi |

**File:** `tests/unit/evaluation/test_champion.py` (extend existing)

| Test | What It Validates |
|------|-------------------|
| `test_regression_gates_default_values` | max_mae_tolerance=0.0, min_r2=0.0, coverage=[0.85, 0.97] |
| `test_compare_models_regression_mode` | Correct gates applied in regression mode |
| `test_regression_champion_selects_lowest_mae` | Given 3 results, picks lowest MAE |
| `test_regression_guardrail_r2` | R² <= 0 model excluded |
| `test_regression_guardrail_coverage_low` | Coverage < 85% excluded |
| `test_regression_guardrail_coverage_high` | Coverage > 97% excluded |
| `test_regression_fallback_elasticnet` | If no model passes, ElasticNet wins |
| `test_format_comparison_regression` | Output contains MAE, R², Coverage labels |
| `test_classification_mode_unchanged` | Existing Brier/ECE/AUC gates still work |

**Updates to existing test files:**
- `test_qb_pass_yards.py` etc.: Remove tests for `_fit()` / `_predict()` (now in base). Verify spec has clip_lo/clip_hi.
- `test_prop_base.py` (new or extend): Model type factory tests.

### Phase A/B/C: Files Inventory

| Action | File |
|--------|------|
| **Modify** | `models/prop_prediction/base.py` — PropModelType enum, factory, HP grids, train(model_type=) |
| **Modify** | `models/prop_prediction/qb_pass_yards.py` — spec-only, remove _fit/_predict/_scaler/_model |
| **Modify** | `models/prop_prediction/qb_rush_yards.py` — same |
| **Modify** | `models/prop_prediction/rb_rush_yards.py` — same |
| **Modify** | `models/prop_prediction/wr_rec_yards.py` — same |
| **Modify** | `models/prop_prediction/te_rec_yards.py` — same |
| **Modify** | `evaluation/champion.py` — RegressionPromotionGates, generalized compare_models(), select_prop_champion() |
| **Modify** | `cli/props.py` — --model-type option, champion command |
| **Create** | `tests/unit/models/test_prop_champion.py` |
| **Modify** | `tests/unit/evaluation/test_champion.py` — regression gate tests |
| **Modify** | `tests/unit/models/test_qb_pass_yards.py` — remove _fit tests, add clip spec |
| **Modify** | `tests/unit/models/test_qb_rush_yards.py` — same |
| **Modify** | `tests/unit/models/test_rb_rush_yards.py` — same |
| **Modify** | `tests/unit/models/test_wr_rec_yards.py` — same |
| **Modify** | `tests/unit/models/test_te_rec_yards.py` — same |

---

### Workstream 2: Game Model Refactor

**Goal:** Refactor `models/game_prediction/` to mirror the `models/prop_prediction/` architecture established in Workstream 1. Eliminate the dynamic-class variant factories, unify the metadata schema, fold the `total` regression model into the same framework as `win_prob`, and align field names so the codebase has one coherent model structure.

**Status:** **Done** (closed 2026-06-19). Details in CHANGELOG.md.

#### Final Results

D5 baseline verification against pre-WS2 Brier targets:

| Model | WS1 Baseline | D5 Actual | Delta | Verdict |
|---|---|---|---|---|
| `win_prob_logistic` | 0.225 | 0.22153 | -0.003 | Improved ✅ |
| `win_prob_random_forest` | 0.220 | 0.21191 | -0.008 | Improved ✅ |
| `win_prob_xgboost` | 0.218 | 0.21598 | -0.002 | Improved ✅ |
| `win_prob_elo` | n/a | 0.23143 | — | Baseline established |
| `total_random_forest` | n/a | MAE 10.24 / RMSE 13.12 / R² 0.056 | — | Baseline established |
| `total_xgboost` | n/a | MAE 10.35 / RMSE 13.19 / R² 0.046 | — | Baseline established |

All three classification Brier baselines exceeded their pre-WS2 targets. Total models established their first formal baselines.

#### Architecture delivered

- `BaseModelMetadata` shared metadata base with `GameModelMetadata` and `PropModelMetadata` subclasses.
- `GamesTrainer` ABC with `WinProbTrainer` and `TotalTrainer` spec-only subclasses.
- `GamesPredictor` base with five thin composite-key subclasses (one per `(model_name, model_type)` pair).
- `WinProbEloPredictor` consolidates the three legacy Elo variants (`elo_v1` / `v2` / `v3`) into one composite-key registration.
- Composite registry keys: `win_prob_logistic`, `win_prob_random_forest`, `win_prob_xgboost`, `win_prob_elo`, `total_random_forest`, `total_xgboost`.
- Nested artifact path scheme: `data/models/{model_name}/{model_type}/`.
- Prediction archive schema migrated from `model_version` (single column) to `model_name` + `model_type` (two columns).
- All classification metrics promoted from `parameters` dict to first-class fields on `GameModelMetadata`.

#### Phased delivery (closed)

| Phase | Description | Status |
|---|---|---|
| D1a | Shared metadata + subclasses | Done |
| D1b | `ArtifactStore` generalization + nested paths | Done |
| D2a | `GamesTrainer` + spec subclasses (alongside legacy) | Done |
| D2b.1 | `GamesPredictor` + composite-key registrations | Done |
| D2b.2 | Archive schema migration + all callers | Done |
| D2b.3 | Legacy code deleted | Done |
| Elo-WS2 | Elo migrated to `win_prob_elo` composite key | Done |
| D3 | Metrics promoted to first-class fields | Done |
| D4 | All 5 game models retrained successfully | Done |
| D5 | Brier baselines verified; CHANGELOG updated | Done |

#### Bugs surfaced during D4+D5 verification (fixed)

- Season string conversion bug in `_features.py` / `total.py` — modeling file already stores seasons as `"YYYY-YYYY"` strings; the incorrect int→str conversion was crashing every training run.
- `GamesTrainer._run_hp_search` was calling `.set_params(**sampled)` on `CalibratedClassifierCV`, which doesn't expose RF hyperparameters. Fixed via `_apply_params()` helper that uses the `estimator__` prefix when wrapping.
- `_parse_composite_key` (in three files) was using `str.partition("_")` to split keys, which broke for any model_name containing underscores (i.e. all `win_prob_*`). Fixed via prefix matching against `get_known_model_names()` from `predictor.py` as single source of truth.
- `evaluation/select.py::collect_model_metrics` was failing on regression models whose `away_win_prob` column is NaN-only. Added explicit skip.
- Scaler was being saved alongside the logistic model but never applied at predict time. Logistic predictions were severely overconfident (std 0.485 vs ~0.15 expected, archive Brier 0.355 vs 0.222 training Brier). Fixed by loading and applying scaler in all four classification + regression predict methods plus `_maybe_predict_totals`.

#### Known follow-ups (post-WS2 scope)

| Item | Notes | When |
|---|---|---|
| Logistic feature names lost across joblib round-trip | sklearn `UserWarning`, cosmetic only; predict path uses `.values` arrays. | Code review (W4) |
| `CalibratedClassifierCV` inner CV uses `StratifiedKFold(shuffle=False)` | Not strictly time-ordered. Matches WS1 baseline approach. Investigate switching to `TimeSeriesSplit`; expect baseline shift. | Dedicated calibration workstream |
| Logistic `n_games` (7,008) vs tree models (5,705) | Different feature_set (`combined_70` vs `expanded_107`) → different NaN drop patterns. Investigate which expanded columns drive unnecessary row drops. | Code review (W4) |
| Elo sigma + margin_std values | `_MODEL_SIGMAS[("win_prob", "elo")] = 13.60` and `_MODEL_MARGIN_STDS[("win_prob", "elo")] = 13.89` carry pre-WS2 calibration values. Refresh after next archive regeneration. | Sigma calibration follow-up |
| No end-to-end fit-load-predict integration test | The scaler-not-applied bug only surfaced in production-data verification. Worth adding to prevent regression. | Workstream 3 (E5) |

---

### Workstream 3: Integration & End-to-End Tests

**Goal:** Build a comprehensive integration and end-to-end test layer that exercises the full lifecycle of the prediction pipeline. Catches the kinds of bugs that surfaced during WS2 D5 verification — specifically the scaler-not-applied-at-predict-time issue that only manifested when the scaler was loaded from disk and applied to production-shaped data. Covers both game and prop model paths.

**Status:** **Active** (started 2026-06-18).

#### Why this workstream now

The WS2 D5 verification phase exposed a real test gap. Training reported Brier 0.223 for `win_prob_logistic`, but archive-derived Brier was 0.355 — a ~3 hour triage cycle that would have been caught overnight by a fit-load-predict integration test. The pattern across WS2 was that unit tests caught structural issues but missed integration bugs. WS3 closes that gap.

#### Goals

- Cover the full fit → load → predict → archive → evaluate lifecycle for all 5 game models + Elo.
- Cover the full train → predict → archive lifecycle for all 5 prop models.
- Exercise the actual sklearn estimators (no stubs) with minimized HP grids for tractability.
- Build shared test infrastructure (fixtures + helpers) that benefits future workstreams.

#### Sub-workstreams

#### Sub-workstreams

| Phase | Description | Status |
|---|---|---|
| E0 | Shared infrastructure | Done |
| E1-games | Fit-load-predict for 5 game models + Elo | Done |
| E1-props | Fit-load-predict for 5 prop models | Deferred |
| E2 | Archive round-trip (games) | Done |
| E3-games | Backfill flow end-to-end | Done |
| E3-props | Prop archive flow | Deferred |
| E4 | CLI workflow smoke tests | Done |
| E5 | Scaler-application regression test | Done (folded into E1-games) |

#### Decisions locked

- **Speed budget:** under 5 minutes total. E1 and E5 marked `@pytest.mark.slow`, run on PR but not on every commit.
- **Test infrastructure:** extend existing `tests/fixtures/dataframes.py` and `MiniRepoBuilder` in `tests/fixtures/repos.py`. New `tests/fixtures/helpers.py` holds shared assertion + context-manager helpers.
- **sklearn coverage:** real estimators end-to-end with minimized HP grids via context-manager patching of `_get_param_grid` and `_n_iter_for`. Catches integration bugs the stub approach would miss.
- **Scope:** WS2 game models + all 5 props. No new tests for the props if WS1-style tests already exist; WS3 adds the integration layer prop-side too so future refactors don't regress them.
- **No Brier baseline assertions:** integration tests verify plumbing correctness. Model quality is verified through D5-style production runs.
- **Module-scoped tiny-model fixture:** E2/E3/E4 reuse a `@pytest.fixture(scope="module")` tiny trained model rather than re-training between every test. Trades ~30s once per module for tighter test runtimes.

#### File plan

| # | File | Action |
|---|---|---|
| 1 | `tests/fixtures/dataframes.py` | Add `make_games_modeling_df`, `make_props_modeling_df`, `make_modeling_manifest` |
| 2 | `tests/fixtures/repos.py` | Add `with_modeling_file`, `with_epa_by_game` cleanup, `with_player_stats` |
| 3 | `tests/fixtures/helpers.py` | New: `patch_minimal_param_grid`, `assert_archive_schema_valid`, `assert_predictions_reasonable` |
| 4 | `tests/e2e/test_games_fit_load_predict.py` | New: 5 tests (logistic, RF, XGB classification; RF, XGB regression) + Elo predict-only + scaler regression |
| 5 | `tests/e2e/test_props_fit_load_predict.py` | New: 5 tests (QB pass, QB rush, RB rush, WR rec, TE rec) |
| 6 | `tests/integration/test_games_backfill.py` | New: 4 tests (writes, overwrite, skip-without-overwrite, evaluation join) |
| 7 | `tests/integration/test_props_archive.py` | New: 2 tests (write to log, load with filters) |
| 8 | `tests/integration/test_archive_roundtrip.py` | New: 3 tests (games round-trip, props round-trip, multi-column dedup) |
| 9 | `tests/integration/test_cli_workflows.py` | Extend: 4 CliRunner smoke tests for models train, evaluate backfill, evaluate select-model, models list |

#### Definition of done

- All 9 files apply cleanly.
- `uv run pytest -m "unit and not slow"` continues to pass without slowdown.
- `uv run pytest -m "integration"` runs in under 3 minutes.
- `uv run pytest -m "slow"` runs in under 5 minutes.
- Each test marked appropriately (`unit`, `integration`, `slow`, `e2e`).
- The scaler bug from WS2 D5 is reproducible via `test_win_prob_logistic_fit_load_predict` (would fail if reintroduced).

#### Known follow-ups (post-workstream)

- Props e2e fit-load-predict tests deferred. The prop feature pipeline
  reads from multiple synthetic DataFrames with distinct column
  expectations from downstream builders (game_context, rolling, matchup,
  usage). Reactive iteration approach proved too expensive; needs
  upfront fixture design study. Track as future workstream session: read
  the prop builders end-to-end and ship one fixture extension that
  covers all required columns. Once that lands, prop archive integration
  tests follow naturally.
- repos.py::with_epa_by_game still bypasses the _write helper
  unnecessarily. The helper already handles parquet correctly; the
  bypass is a copy-paste leftover. One-line cleanup pending.
- Logistic feature names lost across joblib round-trip (sklearn
  UserWarning, cosmetic; predict path uses .values arrays). Same issue
  flagged in Workstream 2 follow-ups.
- Performance baselines not established for tests. If runtime grows in
  future workstreams, may need a pytest-benchmark pass.

---

## Dependency Graph

    Workstream 1: Champion/Challenger for Props
        │
        ├── A1: PropModelType enum + factory in base.py
        │
        ├── A2: clip_lo/clip_hi in PropModelSpec
        │
        ├── A3: Strip _fit/_predict from 5 subclasses
        │       (depends on A1)
        │
        ├── A4: model_type in PropModelMetadata
        │
        ├── B1: Generalize evaluation/champion.py
        │       (independent of A*)
        │
        ├── B2: select_prop_champion() utility
        │       (depends on B1)
        │
        ├── C1: CLI --model-type option
        │       (depends on A1)
        │
        ├── C2: CLI champion command
        │       (depends on A1–A4, B1–B2)
        │
        └── C3: Train all 15 models, validate
                (depends on C2)

    Workstream 2: Game Model Refactor
        │
        ├── D1: Assess shared metadata
        │       (depends on W1 A4)
        │
        ├── D2: Replace _make_tree_variant()
        │       (depends on D1)
        │
        ├── D3: Wire into generalized champion.py
        │       (depends on W1 B1, D2)
        │
        ├── D4: Run full game model test suite
        │       (depends on D2, D3)
        │
        └── D5: Retrain champions, verify metrics
                (depends on D4)

    Workstream 3: Integration & E2E Tests
        │
        ├── E1: Player DataFrame factories
        │
        ├── E2: Prop repo fixture
        │       (depends on E1)
        │
        ├── E3: Feature pipeline integration tests
        │       (depends on E2)
        │
        ├── E4: CLI integration tests
        │       (depends on E2, W1 C2)
        │
        ├── E5: E2E tests
        │       (depends on E2)
        │
        └── E6: Full suite validation
                (depends on E3–E5)

    Order: W1 complete → W2 complete → W3

---

## Future Workstreams (High-Level)

### Workstream 4: Deep Code Review + Test Suite Review

Two-part review session:
1. **Code review:** Pattern consistency across game + prop models, CLI
   output formatting parity, naming conventions, dead code, import
   hygiene, docstring completeness.
2. **Test suite review:** Edge case coverage audit, fixture quality,
   test isolation, missing negative tests, coverage ratchet assessment.

Detailed plan created when workstreams 1–3 complete.

### Audit Remediation
Tracks the systematic remediation of findings from
`audit_2026_06_18.md`. Per-unit progress, files touched, and
re-baseline outcomes are maintained in `AUDIT_REMEDIATION.md`.
Architectural decisions made during remediation are documented in
`DECISIONS.md`.

**Approach:** Path B — parallel tracks. Track 1 (Units 1-4)
addresses leakage and user-visible bugs sequentially. Track 2
(Unit 5) runs in parallel after Unit 1 completes and unifies the
Predictor Registry. Tier 2 sequential work (Units 6-10) follows
once both tracks close. Tier 3 surgical fixes (Unit 11) wrap up
the high-and-medium severity items. Tier 4 hygiene is ambient
cleanup as files are touched.

#### Status

See `AUDIT_REMEDIATION.md` for current unit and re-baseline log.
Latest completion:
- Unit 9 completed (2026-06-20).
- Collapsed eleven NaN-filled holdout metric fields across
  GameModelMetadata and PropModelMetadata into a single
  task-discriminated `metrics: dict[str, float]` on
  BaseModelMetadata.
- Schema version bumped to 3; legacy artifacts migrate
  silently on read.
- gridiron models list / info now display task-appropriate
  metrics.
- Champion/challenger comparator and prop CLI both consume
  the new dict surface.
#### Unit summary

| Unit | Status | Findings closed | Re-baseline? |
|------|--------|-----------------|--------------|
| 1    | ✅ Complete | prop_base/C1, C2 | Yes (prop) |
| 1b   | ✅ Complete | game_base/H1, H2 | No (below noise floor) |
| 1c   | ✅ Complete | rolling/H1, partial cli_props/C2 + M3 | No (numerical equivalence verified) |
| 2    | ✅ Complete  | walk-forward backfill | No |
| 3    | ✅ Complete  | diagnostics, store, models, predictor | No |
| 4    | ✅ Complete  | travel perf, Elo drift | Yes (Elo) |
| 5    | ✅ Complete | Predictor Registry unification + prop archive identity migration | No |
| 6a   | ✅ Complete | Betting ledger identity migration | No |
| 6b   | ✅ Complete | Artifact discriminator cleanup | No |
| 7a   | ✅ Complete | Canonical prop evaluation join | No |
| 7b   | ✅ Complete | Prop walk-forward backfill | Yes (prop) |
| 7c   | ✅ Complete | Artifact-loading prop CLI commands | No |
| 8    | ✅ Complete | Elo engine unification | No (parity test in place) |
| 9    | ✅ Complete | Task-discriminated metadata | No |
| 10   | Pending  | Trainable Protocol decision | No |
| 11   | Pending  | Tier 3 surgical | Per fix |
| Tier 4 | Ongoing | Documentation, dead code, naming | No |

#### Completion criteria

Workstream 5 is complete when:
- All units (1-11) are marked complete in `AUDIT_REMEDIATION.md`
- Tier 4 is reduced to ambient background cleanup
- `audit_2026_06_18.md` is annotated with closure status per finding
- `AUDIT_REMEDIATION.md` is archived to `code_reviews/`

### Workstream 5: Scenario / "What If" Engine (W4.5)

See ROADMAP.md W4.5 for full description. Now unblocked by W4 completion.
Five phases: player impact quantification → team adjustment → usage
redistribution → conditional re-forecasting → CLI interface.

### Workstream 6: API & Frontend (W8 + W9)

FastAPI serving layer + React/Next.js frontend consuming it. See
ROADMAP.md W8/W9.

### Workstream 7: All External Odds

DraftKings prop odds ingest (E2 from W4), historical odds data, multi-book
line shopping (W7). Requires odds source decision (ROADMAP.md §5.2).

### Workstream 8: Evaluate Remaining Work

Assessment of what's left: model ensemble (W12), real-time/live (W10),
feature engineering backlog, NaN research, architectural debt.

---

## Architectural Debt / Housekeeping

| Item | Notes | When |
|------|-------|------|
| `_DEFAULT_TOTAL_STD` hardcoded in `cli/edges.py` | Currently 13.17 (total model holdout RMSE). Wire into model metadata. | Code review (W4) |
| base.py line count | Currently ~550 lines. Will grow with factory methods. Monitor and split if >700. | Code review (W4) |
| Backport factory pattern to game models | Covered by Workstream 2 (D2). | W2 |
| `PROP_FEATURE_COLS` vs `_EXPANDED_FEATURES` naming | Ensure consistent naming convention across game/prop feature lists. | Code review (W4) |

---

## NaN Research Backlog (Deferred)

Current strategy: drop rows with NaN, with `# TODO(nan)` markers at each
drop site. Future investigation items:

- Bayesian shrinkage priors for early-season rolling stats
- Seasonal carry-forward (use last season's final L6 as prior for week 1)
- Multiple imputation for missing game context features
- Missing-indicator pattern (add `feature_is_missing` binary columns)
- Rookie cold-start with draft capital / combine data

Best done after model architecture is stable (post workstream 2).

---

## Changelog

| Date | Change |
|------|--------|
## Changelog

| Date | Change |
|------|--------|
| 2026-06-19 | **Workstream 2 closed.** Game model refactor complete. All 5 WS2 game models retrained on new infrastructure. D5 Brier baselines verified — all three classification champions improved vs pre-WS2 (logistic 0.222 / RF 0.212 / XGB 0.216). Five major architectural improvements shipped: composite-key registry, nested artifact paths, unified GamesTrainer/GamesPredictor, archive schema migration, first-class metric fields. Workstream 3 (Integration & E2E Tests) is now active. |
| 2026-06-10 | **Full rewrite.** New priority order: champion/challenger → game model refactor → integration tests → code review → scenario engine → API/frontend → external odds → evaluate. Three workstreams planned in detail (champion/challenger, game model refactor, integration tests). All design decisions locked. Generalized champion.py (option B) chosen for unified classification + regression gates. |
