## Gridiron Edge — Development Plan

**Purpose:** single source of truth for _what to build next_ and _why_.
Updated at the start and close of every workstream.

| Document | Role |
|----------|------|
| **PLAN.md** (this file) | What is planned, what is active, what is deferred |
| **CHANGELOG.md** | What was built and when (completed workstream details) |
| **HANDOFF.md** | How the system works today (architecture, workflows, operations) |

#### Status key

| Tag | Meaning |
|-----|---------|
| Done | Done — details in CHANGELOG.md |
| In progress | In progress |
| Planned | Planned / blocked |
| Deferred | Deferred |

---

### Currently Active Workstream: W4 — Player Data & First Prop Models

**Goal:** Complete the player-level data layer, feature pipeline, and first prop
projection models. Achieve **M3** (first prop edge report).

**Design decisions (locked in):**

- **NaN strategy:** Drop rows with NaN for now. Mark every drop-site in code with
  `# TODO(nan): <reason>` for future audit. A dedicated research item is in the
  backlog to investigate data-driven imputation methods (see NaN Research Backlog
  below).
- **Model architecture:** One PropTrainer per prop type (e.g., QBPassYardsTrainer),
  mirroring how game models work — one model predicts the raw stat, post-processing
  derives P(over), lean, confidence tier.
- **Model types:** Champion/challenger across ElasticNet, Ridge, RandomForest,
  XGBoost (and potentially LightGBM). Same promotion gate pattern as game models
  (primary metric: MAE; guardrails: coverage, calibration).
- **Feature philosophy:** "Throw everything in, let the model decide." No manual
  feature selection up front. Systematic feature importance / elimination (e.g.,
  permutation importance, AIC/BIC) is a future workstream.
- **Prop odds:** Historical prop odds deferred. Upcoming-week prop odds via
  DraftKings ingest extension. Prop edge calculations will work once DK prop
  ingest is wired.
- **Stat families for V1:** QB passing yards, QB rushing yards, RB rushing yards,
  WR receiving yards, TE receiving yards (5 total).

---

#### Phase A — Player Data Foundation  ✅ Complete

| Step | Status | Deliverable |
|------|--------|-------------|
| A1: Player stats ingest | ✅ Done | `ingest/nflverse/player_stats.py` — nflreadpy, 1999–2024, ~5K rows/season, 42 cols. Stored at `data/raw/player_stats/player_stats_{season}.parquet`. |
| A2: Player game logs + game_id | ✅ Done | `transform/clean/player_stats.py` → `data/cleaned/player_game_logs.parquet` — 138K rows, 44 cols, 4,068 unique players, 100% game_id non-null. |
| A3: Rolling features | ✅ Done | `features/player/rolling.py` — `ROLLING_STAT_COLS` covering passing/rushing/receiving stats at L3/L6 windows. shift(1) for lookahead prevention. |
| A4: Matchup features | ✅ Done | `features/player/matchup.py` — 28 features (14 defensive-allowed stats × 2: L6 rolling avg + rank). Rankings: 1 = toughest, 32 = most generous. Joined via opponent_team. |

**Existing tests:**
`tests/unit/ingest/test_player_stats.py` (9 tests, 3 classes),
`tests/unit/transform/test_player_stats.py` (7 tests, 3 classes),
`tests/unit/features/test_player_rolling.py` (12 tests, 3 classes),
`tests/unit/features/test_player_matchup.py` (11 tests, 5 classes)

---

#### Phase A — Audit Findings (Fix Before Phase B)

_Automated audit run: 2026-06-05. Manual code review: same session._
_Audit script: `scripts/audit_w4_phase_a.py`_

**Pre-Phase-B gate:** All items marked 🔴 must be resolved before starting B1.
Items marked 🟡 should be addressed during B1 (Audit & Stabilize). Items marked
🟢 are non-blocking observations to keep in mind.

##### Automated Audit Results: 43 pass · 2 fail · 4 warn

##### 🔴 Blocking Issues

**F1 — game_id null rate is 0.0007%, not 0%**
- File: `data/cleaned/player_game_logs.parquet`
- Detail: ~97 rows have null game_id (out of 138,368). These are likely
  games that didn't join to the schedule lookup (international games,
  preseason, or edge cases).
- Action: Investigate which rows have null game_id (`df[df.game_id.isna()]`).
  Either fix the schedule join or drop these rows in `clean_player_stats()`
  with a logged warning.

**F2 — game_id fixture bug in `test_player_matchup.py`**
- File: `tests/unit/features/test_player_matchup.py`
- Detail: In `_make_player_logs()`, KC rows use `"2024KC_LV"` while LV rows
  use `"2024_KC_LV"` (inconsistent). Also, game_id doesn't vary by week
  (no `{week}` in the f-string), so all weeks share the same game_id.
- Action: Change both to `f"2024_{week:02d}_KC_LV"` for consistency.

##### 🟡 Should Fix During B1

**W1 — `recent_team` column not present in raw data**
- File: `data/raw/player_stats/` parquet files
- Detail: Audit expected `recent_team` but nflreadpy uses `team` instead.
  The audit script's expectation was wrong, not the data. However, verify
  that `team` is the correct column used everywhere downstream.
- Action: Confirm `team` column is used consistently. No code change needed
  unless `recent_team` is referenced elsewhere.

**W2 — 3 columns with >80% NaN in player_game_logs**
- `passing_cpoe`: 90.62% NaN
- `pacr` (passer rating): 87.42% NaN
- `passing_epa`: 87.13% NaN
- Detail: These are QB-only stats. For non-QBs (75%+ of rows), they're
  naturally NaN. At the QB level, NaN rates will be much lower but still
  significant for CPOE (~27% of QB rows). `passing_epa` being 87% NaN is
  surprising — investigate whether this is a nflreadpy data gap for older
  seasons or a position-filtering issue.
- Action: During B1, run NaN analysis filtered to QBs only. Add
  `# TODO(nan)` comments in `rolling.py` for `passing_cpoe` and
  `passing_epa` in `_PASSING_STATS`.

**W3 — 4 bare/broad `except Exception:` clauses**
- `ingest/nflverse/player_stats.py` L149
- `transform/clean/player_stats.py` L62, L68
- `models/prop_prediction/base.py` L415
- Action: Review each. Replace with specific exception types where possible,
  or add `# noqa` with justification if the broad catch is intentional
  (e.g., protecting against unknown nflreadpy errors).

**W4 — `base.py` is 551 lines**
- File: `models/prop_prediction/base.py`
- Detail: Contains PropModelSpec, PropModelMetadata, PropPrediction,
  evaluate_props, _MIN_ATTEMPTS, UNIVERSAL_FEATURE_COLS,
  _build_universal_features, and the full PropTrainer ABC.
- Action: Consider splitting during C1 when adding multi-model support.
  Natural split: `_types.py` (dataclasses + specs), `_evaluate.py`
  (evaluate_props), `base.py` (PropTrainer only).

**W5 — Verify `test_no_lookahead_week1` has real assertions**
- File: `tests/unit/features/test_player_rolling.py`
- Detail: The uploaded content showed the test calling `_compute_rolling()`
  but the assertion was cut off. The audit found 12 test functions total,
  so the test likely has assertions — but verify.
- Action: Open the test file locally and confirm the assertion checks that
  week 1 rolling features are NaN.

**W6 — Add `# TODO(nan)` comments to rolling.py**
- File: `features/player/rolling.py`
- Detail: `passing_cpoe` (90.6% NaN overall, ~27% for QBs) and
  `passing_epa` (87.1% overall) in `_PASSING_STATS` will generate rolling
  features that are heavily NaN.
- Action: Add comments:
  ```python
  "passing_epa",      # TODO(nan): 87% NaN overall (QB-only stat, older seasons may lack EPA)
  "passing_cpoe",     # TODO(nan): 91% NaN overall (~27% for QBs). Evaluate during feature importance.
  ```

##### 🟢 Non-Blocking Observations

**O1 — Audit script function name mismatch (A3 live test)**
- The audit script tried to import `build_rolling_features` but the actual
  function is `build_player_rolling_features`. This is an audit script bug,
  not a source code bug. The fixed script is at `scripts/audit_w4_phase_a.py`.

**O2 — Audit script positional arg error (A4 live test)**
- The audit script called `build_matchup_features(df)` but the function
  uses keyword-only arguments (`build_matchup_features(*, window=..., repo=...)`).
  This is an audit script bug. The function loads its own data internally.

**O3 — `target_share` and `air_yards_share` in ROLLING_STAT_COLS**
- These are already share metrics from nflreadpy. Computing rolling averages
  of shares is semantically valid (tells you average recent usage share).
  When we build `usage.py` (B2), we'll also compute shares from raw counts.
  Both approaches are valid — let the model decide which is more predictive.

**O4 — Prop model subclass tests have 5 tests each (not 1)**
- Initial code review based on truncated uploads suggested only 1 test per
  file. The audit confirmed 5 test functions per file. Still thin for
  production code, but adequate for scaffolding. Expand during Phase C.

**O5 — ElasticNet-only in all prop model subclasses**
- Expected. Multi-model support (RF, XGBoost, Ridge) will be added in C1
  when the PropTrainer base class is updated.

**O6 — Non-skill positions in player_game_logs**
- Position distribution shows P (402), OT (185), DB (91), CB (86), LB (82).
  The `is_skill` column should filter these out for prop models, but verify
  during B1 that the filter is applied correctly.

##### NaN Landscape (from audit)

| Column | Overall NaN% | Notes |
|--------|-------------|-------|
| passing_cpoe | 90.62% | QB-only, nflreadpy may not have for older seasons |
| pacr | 87.42% | Passer rating composite — QB-only |
| passing_epa | 87.13% | QB-only, **investigate older season coverage** |
| rushing_epa | 58.73% | Many players have 0 carries → no EPA |
| wopr | 25.09% | WR-only weighted opportunity rating |
| racr | 21.53% | Receiver air conversion ratio |
| target_share | 19.95% | Non-receivers have no targets |
| receiving_epa | 19.63% | Non-receivers |
| air_yards_share | 15.15% | Non-receivers |
| position | 0.01% | ~14 rows with missing position |
| game_id | 0.00% | ~97 rows (the F1 issue above) |

**Key insight:** Most high-NaN columns are **position-specific stats applied
to all positions**. When filtered to the relevant position (e.g., QBs for
passing_cpoe), NaN rates will drop significantly. The B1 audit should
recompute NaN rates per position group to get the true picture.

---

#### Phase B — Feature Pipeline Completion  🔲 Planned

##### B1: Audit & Stabilize Existing Features

**Why:** Prop model training failed with NaN issues last session. Must confirm the
feature matrix is clean before building on top of it.

**Tasks:**
1. **Fix F1 and F2** from audit findings above (blocking)
2. **Address W1–W6** from audit findings (non-blocking but do now)
3. Load `player_game_logs.parquet` — verify schema (44 cols), row count (~138K),
   game_id coverage (100% after F1 fix), is_skill distribution
4. Run rolling feature builder — check NaN rates per column per window (L3/L6).
   **Run NaN analysis per position group** (not just overall) to get the true picture.
   Document expected NaN patterns (e.g., L3 → NaN for weeks 1–2 of each season)
5. Run matchup feature builder — verify 28 features join cleanly, check NaN rates
6. Clean up any duplicate/inefficient conditional logic flagged last session
7. Add `# TODO(nan): <reason>` comments at every `dropna()` or NaN-producing site

**Done when:** A single script can produce the full player feature DataFrame and
print a NaN report. All NaN rates are documented and understood. All 🔴 and 🟡
audit items resolved. Re-run `scripts/audit_w4_phase_a.py` → 0 failures.

##### B2: Usage Features

**Why:** Target share and carry share are 🔴 High-signal features (FEATURES.md
Domain 8). Volume is the #1 driver of counting stats.

**New file:** `features/player/usage.py`

**Features to build:**
- `target_share_L3`, `target_share_L6` — player targets / team total targets
  (WR, TE)
- `carry_share_L3`, `carry_share_L6` — player carries / team total carries (RB)
- `touch_share_L3`, `touch_share_L6` — (targets + carries) / team total plays
  (all skill positions)
- Snap % — **only if** nflreadpy exposes snap count data. Check column
  availability first. If unavailable, defer and document.

**Implementation pattern:**
- Compute team-level totals per game from `player_game_logs` (sum targets/carries
  by team + game)
- Join back to player rows, compute share
- Apply rolling windows with `shift(1)` for lookahead prevention

**New test:** `tests/unit/features/test_usage.py`

**Done when:** Usage features join cleanly to the player feature DataFrame. NaN
rates documented.

##### B3: Game Context Features for Props

**Why:** A player's stat line is heavily influenced by game script. Big favorites
run the ball; big underdogs throw. Implied team total sets the volume ceiling.

**New file:** `features/player/game_context.py`

**Features to build:**
- `implied_team_total` — `(total ± spread) / 2` per team per game. For historical
  games, use VEGAS_LINE and actual total. For upcoming games, use model predictions.
- `game_spread` — the spread from the team's perspective (negative = favored).
  Game script proxy.
- `is_home` — binary, from player_game_logs
- `is_dome` — binary, from stadium reference (already exists in team features)
- `rest_days` — from schedule data (already exists in team features, needs joining
  to player rows)

**Key detail:** These are *game-level* features that get joined to player rows via
`(game_id, team)`. Most of this data already exists in the team feature pipeline —
this step is primarily about **wiring** it to the player feature DataFrame rather
than recomputing.

**New test:** `tests/unit/features/test_game_context.py`

**Done when:** Game context features join to player rows. Each player-game row has
team total, spread, home/dome/rest context.

##### B4: Unified Prop Feature Builder

**Why:** Need a single entry point that assembles all player features into the
training-ready DataFrame — the prop equivalent of `build_model_inputs()`.

**New files:**
- `features/player/builder.py` — orchestrator
- `features/player/_columns.py` — feature column definitions (equivalent of
  `models/game_prediction/_columns.py`)

**Responsibilities of `builder.py`:**
1. Load `player_game_logs.parquet`
2. Join rolling features (from `rolling.py`)
3. Join matchup features (from `matchup.py`)
4. Join usage features (from `usage.py`)
5. Join game context features (from `game_context.py`)
6. Accept `position_filter` argument (e.g., `['QB']`, `['RB']`, `['WR', 'TE']`)
7. Return one DataFrame: one row per player-game with all features + target columns
8. Apply `dropna()` on feature columns (with `# TODO(nan)` annotation)
9. Log: total rows, rows dropped to NaN, final usable rows, feature count

**Temporal safety audit:** Verify every feature uses `shift(1)` or equivalent. No
lookahead. Add an assertion or test that no feature column correlates > 0.99 with
the target (which would indicate leakage).

**New tests:**
- `tests/unit/features/test_builder.py`
- `tests/integration/test_prop_feature_pipeline.py`

**Done when:** `build_prop_features(position_filter=['QB'])` returns a clean
DataFrame. `build_prop_features(position_filter=['RB'])` also works. Feature
count and NaN report printed.

---

#### Phase C — Prop Model Training  🔲 Planned

##### C1: Prop Trainer Framework + QB Passing Yards (First End-to-End Model)

**Why:** Validates the entire pipeline from raw data through trained model. Get one
model working perfectly before scaling.

**Existing files to update:**
- `models/prop_prediction/base.py` — PropTrainer base class (scaffolded; needs
  multi-model support, champion/challenger wiring)
- `models/prop_prediction/qb_pass_yards.py` — QBPassYardsTrainer (scaffolded;
  needs connection to new feature builder, NaN fix)

**PropTrainer base class should provide:**
- `train(position_filter, target_col, repo)` → trains model, saves artifact
- `_prepare_data(df, position_filter, target_col)` → filters position, drops NaN,
  splits on HOLDOUT_SEASONS
- TimeSeriesSplit CV (consistent with game models)
- Champion/challenger promotion via existing `evaluation/champion.py` pattern
  (adapt gates for regression: primary = MAE, guardrails = coverage, calibration)
- Support multiple model types: ElasticNet, Ridge, RF, XGBoost via factory or
  config

**Training flow:**
```
build_prop_features(position_filter=['QB'])
    → _prepare_data(target_col='passing_yards')
    → TimeSeriesSplit CV with HP search
    → Retrain best params on full train set
    → ArtifactStore.save('qb_pass_yards_elasticnet', model, metadata)
    → Champion/challenger comparison
```

**Validation checks:**
- Predictions on holdout are in reasonable range (QB pass yards: ~100–450)
- MAE is reasonable (target: < 60 yards)
- No obvious systematic bias (plot predicted vs actual)

**Existing tests to update:**
`tests/unit/models/test_prop_base.py`,
`tests/unit/models/test_qb_pass_yards.py`

**Done when:** QBPassYardsTrainer trains successfully on all 4 model types
(ElasticNet, Ridge, RF, XGB). Best model selected as champion. Holdout MAE
reported.

##### C2: Prop Output Enrichment (Post-Processing)

**Why:** Raw yard projections need market-facing outputs to calculate edges.

**New file:** `models/prop_prediction/post_process.py`

**Enrichment outputs per prediction:**
- `predicted_mean` — model point prediction
- `predicted_std` — `sqrt(model_residual_variance + player_rolling_std²)`.
  Model residual std captures systematic uncertainty; player's own std captures
  individual variance.
- `lo_90`, `hi_90` — `predicted_mean ± 1.645 * predicted_std`
- `p_over(line)` — `1 - Φ((line - predicted_mean) / predicted_std)` using normal
  CDF. Takes the line as input (from DK odds or user-specified).
- `lean` — "Over" if `p_over > 0.55`, "Under" if `p_over < 0.45`, "No Edge"
  otherwise (thresholds configurable)
- `confidence_tier` — based on `|p_over - 0.5|`: High (> 0.15), Moderate
  (0.08–0.15), Low (< 0.08)

**Parallels game model:** Just like `enrich_predictions()` takes raw win_prob and
adds spread/bands/tier, this takes raw predicted_yards and adds
distribution/lean/tier.

**New test:** `tests/unit/models/test_prop_post_process.py`

**Done when:** Every prop prediction row has all enrichment columns. Unit tests
verify P(over) math.

##### C3: Additional Prop Models

**Why:** With the pipeline validated on QB Pass Yards, extending is mechanical.

**Existing files to update (all scaffolded, need wiring to feature builder):**
1. `models/prop_prediction/qb_pass_yards.py` — QB Rushing Yards
   (`target_col='rushing_yards'`, `position_filter=['QB']`).
   **Note:** This needs its own file or a second PropModelSpec in the QB module.
   Evaluate whether to add `qb_rush_yards.py` or extend `qb_pass_yards.py` with
   a second spec.
2. `models/prop_prediction/rb_rush_yards.py` — RB Rushing Yards
   (`target_col='rushing_yards'`, `position_filter=['RB']`)
3. `models/prop_prediction/wr_rec_yards.py` — WR Receiving Yards
   (`target_col='receiving_yards'`, `position_filter=['WR']`)
4. `models/prop_prediction/te_rec_yards.py` — TE Receiving Yards
   (`target_col='receiving_yards'`, `position_filter=['TE']`)

**For each:**
- Wire to unified feature builder from B4
- Train all 4 model types (ElasticNet, Ridge, RF, XGB)
- Champion/challenger selects best
- Validate holdout MAE and prediction ranges

**Existing tests to update:**
`tests/unit/models/test_rb_rush_yards.py`,
`tests/unit/models/test_wr_rec_yards.py`,
`tests/unit/models/test_te_rec_yards.py`

**New file needed:** `models/prop_prediction/qb_rush_yards.py` (or decide on
multi-spec approach)

**Done when:** 5 prop model families trained, each with a champion model. Holdout
MAE table for all 5.

---

#### Phase D — Evaluation & Persistence  🔲 Planned

##### D1: Prop Evaluation Metrics

**Why:** Prop models need different eval metrics than game models.

**New file:** `evaluation/prop_metrics.py`

**Metrics:**
- **MAE / RMSE** — mean absolute error, root mean squared error vs actual
- **Hit rate** — given a line, % of "Over" leans that went over (and vice versa)
- **P(over) calibration** — when model says 70% over, does it go over ~70%?
  (calibration curve, reliability diagram)
- **Coverage** — do 90% prediction intervals contain ~90% of outcomes?
- **Bias** — `mean(predicted - actual)`, should be ~0
- **By-tier analysis** — MAE and hit rate broken down by confidence tier

**New test:** `tests/unit/evaluation/test_prop_metrics.py`

**Done when:** `prop_metrics.evaluate(predictions, actuals, lines)` returns a
structured report. Calibration plot can be generated.

##### D2: Prop Archive

**Why:** Persist prop predictions for tracking, CLV, and historical review.

**New file:** `evaluation/prop_archive.py`

**Schema:**
```
predicted_at, is_backfilled, season, week, game_id, game_date,
player_id, player_name, position, team,
stat_type, model_version,
predicted_mean, predicted_std, lo_90, hi_90,
line (nullable — only populated when odds available),
p_over (nullable), lean (nullable), confidence_tier
```

**Storage:** `data/output/props/prop_predictions_log.parquet`

**Pattern:** Same append-only, dedup-on `(game_id, player_id, stat_type,
model_version)` as game archive.

**New test:** `tests/unit/evaluation/test_prop_archive.py`

**Done when:** Prop predictions archive and load round-trip successfully.

---

#### Phase E — CLI & Integration  🔲 Planned

##### E1: Prop CLI

**Why:** User-facing deliverable — the thing you actually run on game day.

**New file:** `cli/props.py`

**Commands:**
- `gridiron props projections --week N --season YYYY` — table of all prop
  projections for the week
- `gridiron props backfill --model qb_pass_yards` — backfill historical prop
  predictions
- `gridiron props evaluate --model qb_pass_yards` — evaluation report for a
  prop model

**Output format (projections):**
```
Player       Pos  Stat         Proj   Lo90   Hi90   Line  P(Over)  Lean     Conf
P. Mahomes   QB   Pass Yards   278    198    358    274.5  0.52    No Edge  Low
L. Jackson   QB   Rush Yards    62     28     96     52.5  0.64    Over     Moderate
D. Henry     RB   Rush Yards    88     42    134     79.5  0.59    Over     Low
```

**Register in `cli/main.py`.**

**New tests:**
- `tests/integration/test_props_cli.py`
- `tests/e2e/test_prop_pipeline.py`

**Done when:** `gridiron props projections --week 12 --season 2024-2025` produces
a formatted table.

##### E2: DraftKings Prop Ingest Extension  🔲 Planned (Deferred — Upcoming Weeks Only)

**Why:** To calculate prop edges, we need prop lines from the market.

**Extend:** `ingest/odds/draftkings.py` (or new `ingest/odds/dk_props.py`)

**Scope:** Parse DK player prop markets (pass yards, rush yards, receiving yards
O/U). Map to `(player_id, stat_type, line, odds_over, odds_under)`. Store in
`data/odds/dk_prop_odds_log.parquet`.

**Note:** This is the last step because it requires understanding the DK prop API
response format. Historical prop odds remain deferred.

**Done when:** DK prop lines for the current week can be ingested and joined to
prop predictions.

---

#### Phase T — Testing  🔄 Parallel with all phases

Tests are built alongside each phase, not as a separate step at the end.

**Already existing (from Phase A):**

| File | Tier | Count |
|------|------|-------|
| `tests/unit/ingest/test_player_stats.py` | Unit | 9 tests, 3 classes |
| `tests/unit/transform/test_player_stats.py` | Unit | 7 tests, 3 classes |
| `tests/unit/features/test_player_rolling.py` | Unit | 12 tests, 3 classes |
| `tests/unit/features/test_player_matchup.py` | Unit | 11 tests, 5 classes |
| `tests/unit/models/test_prop_base.py` | Unit | 19 tests, 6 classes |
| `tests/unit/models/test_qb_pass_yards.py` | Unit | 5 tests, 1 class |
| `tests/unit/models/test_rb_rush_yards.py` | Unit | 5 tests, 1 class |
| `tests/unit/models/test_wr_rec_yards.py` | Unit | 5 tests, 1 class |
| `tests/unit/models/test_te_rec_yards.py` | Unit | 5 tests, 1 class |

**To create:**

| Phase | File | Tier |
|-------|------|------|
| B2 | `tests/unit/features/test_usage.py` | Unit |
| B3 | `tests/unit/features/test_game_context.py` | Unit |
| B4 | `tests/unit/features/test_builder.py` | Unit |
| B4 | `tests/integration/test_prop_feature_pipeline.py` | Integration |
| C2 | `tests/unit/models/test_prop_post_process.py` | Unit |
| D1 | `tests/unit/evaluation/test_prop_metrics.py` | Unit |
| D2 | `tests/unit/evaluation/test_prop_archive.py` | Unit |
| E1 | `tests/integration/test_props_cli.py` | Integration |
| E1 | `tests/e2e/test_prop_pipeline.py` | E2E |

**Also needed:** Add player-related DataFrame factories to
`tests/fixtures/dataframes.py` (e.g., `make_player_game_logs()`,
`make_player_features()`).

---

#### Dependency Graph

```
B1 (Audit Remediation + Stabilize)
    │
    ▼
B2 (Usage Features)                              T (Tests — parallel)
    │
    ▼
B3 (Game Context Features)
    │
    ▼
B4 (Unified Feature Builder)
    │
    ├──────────────────────────────┐
    ▼                              ▼
C1 (QB Pass Yards + Framework)   D1 (Prop Eval Metrics)
    │                              │
    ▼                              │
C2 (Post-Process Enrichment)      │
    │                              │
    ▼                              │
C3 (4 More Prop Models)           │
    │                              │
    └──────────────┬───────────────┘
                   ▼
             D2 (Prop Archive)
                   │
                   ▼
             E1 (Prop CLI)
                   │
                   ▼
             E2 (DK Prop Ingest — last)
```

---

#### Files to Create

| File | Phase |
|------|-------|
| `features/player/usage.py` | B2 |
| `features/player/game_context.py` | B3 |
| `features/player/builder.py` | B4 |
| `features/player/_columns.py` | B4 |
| `models/prop_prediction/post_process.py` | C2 |
| `models/prop_prediction/qb_rush_yards.py` | C3 |
| `evaluation/prop_metrics.py` | D1 |
| `evaluation/prop_archive.py` | D2 |
| `cli/props.py` | E1 |
| `ingest/odds/dk_props.py` (or extend `draftkings.py`) | E2 |

#### Files to Modify

| File | Phase | Change |
|------|-------|--------|
| `models/prop_prediction/base.py` | C1 | Multi-model support, champion/challenger |
| `models/prop_prediction/qb_pass_yards.py` | C1 | Wire to feature builder, fix NaN |
| `models/prop_prediction/rb_rush_yards.py` | C3 | Wire to feature builder |
| `models/prop_prediction/wr_rec_yards.py` | C3 | Wire to feature builder |
| `models/prop_prediction/te_rec_yards.py` | C3 | Wire to feature builder |
| `cli/main.py` | E1 | Register props sub-app |
| `tests/fixtures/dataframes.py` | T | Add player DataFrame factories |

All paths relative to `src/gridiron_edge/` (source) or `tests/` (tests).

---

#### NaN Research Backlog  🔲 Deferred

**Problem:** Dropping NaN rows for early-season games (weeks 1–2 for L3, weeks
1–5 for L6) loses significant training data and makes early-season predictions
impossible.

**Research directions to explore:**
- **Bayesian shrinkage priors:** Use league-average or position-average as a
  prior, blend with observed data as games accumulate (empirical Bayes)
- **Seasonal carry-forward with decay:** Use prior season's final rolling average,
  decayed by a factor (e.g., 0.7) to reflect roster/scheme changes
- **Multiple imputation:** Generate multiple plausible feature sets, train on each,
  average predictions
- **Hierarchical models:** Player nested within team nested within position —
  partial pooling handles sparse data naturally
- **Missing-indicator pattern:** Add binary `is_imputed_*` flags so the model can
  learn to weight imputed features differently

**Data sources to investigate:** Does nflreadpy expose preseason stats? Can we use
combine/draft data as cold-start priors for rookies?

**When:** After W4 V1 is complete and evaluated.

---

### Parallel / Lower Priority

#### Weather Feature Integration (Deferred)

Deferred until OpenWeatherMap API key is reliably available and
weather features are validated against holdout data.

### Architectural Debt / Housekeeping

| Item | Notes |
|------|-------|
| `_DEFAULT_TOTAL_STD` hardcoded in `cli/edges.py` | Currently 13.17 (total model holdout RMSE). Wire into model metadata so it updates when a new total model is trained. |
| Historical edge validation / backtest | Deferred until season starts and real odds data is available. Run `gridiron edges clv` against a full season of archived predictions + odds to validate edge quality. |
| Schema migration helper | `archive.migrate_archive()` exists for pre-v2 archives. Can be removed once all archives are v2+. |
| Kelly adherence metric | `performance.kelly_adherence()` deferred — requires storing `recommended_stake` in the bet ledger schema. Add column to `_BET_COLUMNS` when implementing. |
| Balance display cosmetic | `balance_cmd` shows `$-100.00` for outflows instead of `-$100.00`. Fix sign formatting with `abs()` in `cli/betting.py`. |
| Recalibrate sigma/margin_std after retrain | The `_MODEL_SIGMAS` and `_MODEL_MARGIN_STDS` dicts in `post_process.py` still use values calibrated from the old versioned models. After a champion retrain, re-run sigma calibration and update the unversioned entries. |
| `ModelMetadata.holdout_brier` for regression | Repurposed for MAE in total model. Consider adding a generic `primary_metric` field. |

### Backlog

Workstream IDs match **ROADMAP.md** (authoritative numbering).

#### Completed

| ID | Workstream | Summary |
|----|-----------|---------|
| W1 | Quick Wins & Unblocking | DK unicode fix, game_id resolver, odds join validated |
| W2 | Richer Game Model Outputs | Spread, total, projected scores, bands, tiers, isotonic eval |
| W3 | Market Intelligence Foundation | odds_math.py, kelly.py — pure math, no data deps |
| W5 | Edge Engine | edge.py, recommendations.py, clv.py, CLI (report + clv) |
| W6 | Portfolio & Bet Tracking | ledger.py, bankroll.py, performance.py, CLI (8 commands) |

_Also completed (cross-cutting, not numbered in ROADMAP):_
- **Feature Engineering Expansion** — EPA_COLS 8→22→36, _EXPANDED_FEATURES 51→107→149
- **Test Framework Build-out** — Three-tier pyramid, auto-markers, shared fixtures
- **Champion/Challenger Refactor** — TimeSeriesSplit CV, gate-based promotion, 3
  unversioned champions (random_forest, xgboost, logistic)
- **Sigma & Confidence Tier Recalibration** — Probability-distance-based tiers
  replacing band-width thresholds

#### Planned

| ID | Workstream | Blocked by | Priority | Notes |
|----|-----------|------------|----------|-------|
| **W12** | Model Ensemble | Nothing | **High** | Combine elo + logistic + rf + xgb. Must beat xgboost Brier (0.218) via promotion gates. |
| **W4** | Player Data & First Prop Models | Nothing | **Active** | See active workstream above. |
| **W8** | API Serving Layer | Nothing | Medium | FastAPI endpoints for edges, games, portfolio → M5 |
| **W7** | Multi-Book Odds & Line Shopping | Odds source decision (§5.2) | Medium | Multi-book ingest, arb/middle detection → M4 |
| **W4.5** | Scenario Engine (What-If) | W4 | Medium | Injury impact modeling, usage redistribution |
| **W9** | Frontend | W8 | Lower | React/Next.js web UI → M5 |
| **W10** | Real-Time & Live Game | W7 + W8 | Lowest | Live win prob, live edges, hedge calculator → M6 |

### Changelog (PLAN.md edits only)

| Date | Change |
|------|--------|
| 2026-06-05 | **Phase A audit completed.** Ran `audit_w4_phase_a.py` (43 pass, 2 fail, 4 warn). Code review of all 9 source files + 9 test files. Documented findings: 2 blocking (F1: game_id nulls, F2: fixture bug), 6 should-fix (W1–W6), 6 observations (O1–O6). NaN landscape table added. B1 updated to include audit remediation as first task. |
| 2026-06-05 | **W4 detailed implementation plan.** Added Phases A–E + T with step-level tasks, dependency graph, file inventories (create/modify), NaN research backlog. Reconciled against actual directory structure (136 dirs, 722 files). Confirmed Phase A complete, all 5 prop model files scaffolded with tests. Updated completed workstream summaries (added sigma recal, feature eng expansion to 149). |
| 2026-06-04 | **Complete rewrite.** Replaced stale W2-phase-detail PLAN with current state. Champion/challenger refactor complete. Removed resolved debt items (temporal CV, stale __pycache__). Updated backlog priorities (xgboost is new champion, W12 references its Brier). Removed W11 (live prediction pipeline already exists). |
| 2026-06-03 | W6 complete. W5 complete. Renumbered to match ROADMAP v2. |
| 2026-06-01 | W2, Feature Eng, Test Framework completed. |
| 2026-05-31 | Initial PLAN.md created with backlog. |
