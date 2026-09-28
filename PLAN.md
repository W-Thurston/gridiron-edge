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

### Active Program: Game-Model Feature Program (ROADMAP Tier 5 #16), Phase A

No unit is currently active. Units 1–5 (batches A1–A5) closed below.

### Unit 5 — Phase A Batch A5: Rating Ensemble Inputs [Completed September 28, 2026]

#### Completed

Wired seven rating-ensemble suffixes (14 raw and 7 differential columns) into
the canonical Win/Total feature set: team Glicko-2 rating and RD, Massey,
Colley and Keener ratings, and a starting-QB rating with its prior start
count. Every family is recomputed in memory from complete game history rather
than persisted (`DECISIONS.md` D69): the games CSV is already digest-pinned,
and a truncated history raises instead of silently changing ratings, which is
the lineage-equivalent guard for path-dependent ratings. Constants were fit on
1999-2022 only (D70). A game beyond the last completed game receives the last
state carried forward, so no value is null at any horizon, including a
season's first week. Glicko-2 volatility was declined on a pre-declared gate
(cross-snapshot std 0.00003 against 0.002). Verified against regenerated real
data and a full replay of the upcoming schedule. Promotion is deferred to
Unit 7, per D62.

#### Goal

Close the four A5 rows of the Game-Model Build Queue (`FEATURES.md`), and
settle program rule 5 for each rating family: which are persisted with Elo-
equivalent lineage and which are deterministic recomputations, with
leakage-safe, live-available values.

#### Files Added/Removed/Changed

Added:
- `src/gridiron_edge/ratings/history.py` - shared chronology helpers and
  complete-history validation for recomputed ratings.
- `src/gridiron_edge/ratings/glicko2.py` - Glicko-2 step and full-history
  replay with as-of lookup and frozen lookahead.
- `src/gridiron_edge/ratings/qb_elo.py` - additive team + QB Elo replay with
  as-of QB lookup.
- `src/gridiron_edge/ratings/linear_ratings.py` - weighted Massey, Colley, and
  Keener solvers over a per-cutoff game window.
- `src/gridiron_edge/features/team/rating_ensemble.py` - registers
  `home_away_glicko2`, `home_away_linear_ratings`, and `home_away_qb_elo`.
- `tests/fixtures/rating_history.py` - compact complete-history builder for
  rating tests.
- `tests/unit/ratings/test_glicko2.py`, `test_linear_ratings.py`,
  `test_qb_elo.py`, `tests/unit/features/test_rating_ensemble.py` - algorithm,
  leakage, lookahead, and fail-closed coverage.

Changed:
- `src/gridiron_edge/features/team/qb_profile.py` - resolved-starter helper is
  public; no behavior change.
- `src/gridiron_edge/features/pipeline.py` - registers the three features at
  the end of `CANONICAL_FEATURES`.
- `src/gridiron_edge/features/manifest.py` - schema 11 to 12, data 8 to 9.
- `src/gridiron_edge/models/game_prediction/_columns.py`, `_features.py` -
  rating-ensemble suffix list; the diff builder loops it.
- `tests/unit/features/test_canonical_feature_sequence.py`,
  `tests/unit/features/test_qb_profile.py`,
  `tests/unit/models/test_game_feature_sets.py` - updated sequence, rename,
  and counts (raw 138, diff 69, combined 207, expanded 260).
- `FEATURES.md` - four rows marked Done, A5 summary, counts, changelog.
- `ROADMAP.md` - Unit 5 closed; two known limitations recorded.
- `DECISIONS.md` - D69, D70.
- `CHANGELOG.md` - this unit's entry.

Removed:
- None.

Regenerated (not committed; `data/` is gitignored):
- `data/modeling/modeling_file.parquet` and manifest via
  `gridiron features model-inputs --all-years`.

#### Tests

Validation passed:
- 45 algorithm tests (Glicko-2 reproduces Glickman's published example) and
  19 feature tests, plus updated count tests; ruff, pyrefly, and the full
  non-slow unit suite (4,465 tests). One real bug was found and fixed by a
  test: Keener's power iteration oscillated for two teams (a zero-diagonal
  matrix has eigenvalues plus and minus lambda); an identity shift fixed it.

Real-data validation:
- Modeling file: `schema_version 12`, `data_version 9`, 201 columns (+14),
  7,308 rows with identical `GAME_ID`s, every pre-existing column identical,
  zero nulls in the new columns in every season and week.
- Sanity: 2007 Patriots top the Glicko-2 ranking; Spearman against Elo 0.89
  to 0.99 across sample seasons.
- Leakage on real data at cutoffs 2010 week 5, 2024 week 10, and 2026 week 2:
  truncating history at the cutoff, and scrambling every later score, leave
  all rows through the cutoff (including the cutoff week) exactly equal.
  Stored columns equal a fresh recompute and repeated runs are identical.
- Live-availability replay (program rule 3): all 226 upcoming games (weeks
  3-18) through the full `CANONICAL_FEATURES` pipeline: zero NaNs in all 14
  new columns at every horizon, and each team's ratings are identical across
  horizons. Week 3 has zero NaNs across all 260 expanded features; Elo and
  strength of victory remain NaN beyond next week (pre-existing, recorded in
  ROADMAP).
- Cost: about 0.6 s per weekly build, 3.5 s for all 7,308 rows.
- Findings for Unit 7: `GLICKO2_RATING`'s differential correlates 0.976 with
  `ELO_DIFF` and `GLICKO2_RD` correlates -0.96 with `WEEK_NUM`. Batch A3's QB
  features excluded all 2026 starts because `player_game_logs` ended at 2025
  (recorded in ROADMAP). Player logs were then refreshed by hand (2026
  weeks 1-3, pre-2026 rows identical) and the modeling file regenerated: only
  the eight A3 QB EPA/CPOE/experience columns changed, on the 16 games of 2026
  week 2, and the 226-game replay still shows zero NaNs beyond the known
  Elo/strength-of-victory limitation. No pipeline stage refreshes the logs.

#### Acceptance

All four A5 rows are wired with leakage-safe, live-available values, except
Glicko-2 volatility, which is declined with a recorded measurement. Each
family's persist-or-recompute choice and its lineage-equivalent guard are
recorded in D69, the fitted constants in D70 with no holdout data, and the
modeling file is regenerated at schema 12. No Elo artifact, lineage file, or
existing column changed. The replay shows zero NaNs at every horizon and the
Python gate passes. Promotion remains Unit 7's (D62).

### Unit 4 — Phase A Batch A4: Situational Context [Completed September 28, 2026]

#### Completed

Wired twelve Domain 5/6/11/12 catalog rows into the canonical Win/Total
feature set: season week number and indoor override (status corrections),
back-to-back road games, body-clock kickoff hour, field surface, a continuous
league scoring level, wind speed bins, form trend, Pythagorean win % and luck
gap, one-score record, and special-teams and punt/return EPA. No PBP
re-download and no new sidecar were needed. Reconciling the indoor-override
row found a real train/serve skew: 755 historical retractable-stadium games
were classified outdoor in training while the same stadiums are classified
covered when serving live; they are now covered (`DECISIONS.md` D67).
Kickoffs are excluded from special teams on structural-break evidence (D66).
Verified against regenerated real data and a full replay of the upcoming
schedule. Promotion is deferred to Unit 7, per D62.

#### Goal

Close the Domain 5/6/11/12 Game-Model Build Queue rows in `FEATURES.md` A4 and
correct the two whose recorded status was wrong, using data already on disk,
with leakage-safe and pre-kickoff-available definitions recorded where a
choice cannot be re-derived from code (D66-D68).

#### Files Added/Removed/Changed

Added:
- `src/gridiron_edge/features/team/surface.py` - `IS_ARTIFICIAL_TURF` from
  schedule surface, filled from the stadium reference at (stadium, season);
  unknown labels raise.
- `src/gridiron_edge/features/team/body_clock.py` - each team's home-clock
  kickoff hour, evaluated at the game date.
- `src/gridiron_edge/features/team/road_trips.py` - schedule-based
  back-to-back road flags.
- `src/gridiron_edge/features/team/league_environment.py` - season-to-date
  league points per team-game blended with the prior season.
- `tests/unit/features/test_surface.py` (16), `test_body_clock.py` (15),
  `test_road_trips.py` (12), `test_league_environment.py` (10).

Changed:
- `src/gridiron_edge/core/enums.py` - `COVERED_STADIUMS` includes the per-game
  `open`/`closed` retractable states (D67).
- `src/gridiron_edge/features/team/weather.py` - `WIND_SPEED_BIN`; covered
  classification and climatology follow the corrected enum.
- `src/gridiron_edge/features/team/record.py` - form trend, Pythagorean
  win %/luck gap, one-score record from the same result history.
- `src/gridiron_edge/transform/clean/epa.py` - `_agg_special_teams`; merges
  special-teams EPA before the short-code mapping (D64).
- `src/gridiron_edge/features/team/epa.py` - `st_epa_per_play`,
  `punt_epa_per_play` in `EPA_COLS`.
- `src/gridiron_edge/features/pipeline.py` - registers the four new features.
- `src/gridiron_edge/features/manifest.py` - schema 10 to 11, data 7 to 8.
- `src/gridiron_edge/models/game_prediction/_columns.py`, `_features.py` -
  new game, team, and season-performance columns; the diff builder loops the
  new suffix list.
- `tests/unit/core/test_enums.py`, `test_weather.py`,
  `test_home_away_game_features.py`, `test_home_away_record_feature.py`,
  `tests/unit/features/test_epa.py`, `test_canonical_feature_sequence.py`,
  `tests/unit/models/test_game_feature_sets.py`,
  `tests/unit/transform/clean/test_epa.py` - new coverage and updated counts.
- `FEATURES.md` - twelve rows corrected/marked Done, summary and coverage.
- `ROADMAP.md` - Unit 4 closed; two known limitations recorded.
- `DECISIONS.md` - D66, D67, D68.
- `CHANGELOG.md` - this unit's entry.

Removed:
- None.

Regenerated (not committed; `data/` is gitignored):
- `data/cleaned/epa_by_game.parquet` via `gridiron transform aggregate-epa`;
  `data/modeling/modeling_file.parquet` and manifest via
  `gridiron features model-inputs --all-years`.

#### Tests

Validation passed:
- 53 new tests across the four new modules, plus new cases in the extended
  weather, record, and special-teams suites; ruff, pyrefly, and the full non-slow unit suite
  (4,401 tests). One test failed before the fix and exposed a live bug: a
  season with no completed games (first week of a new season) had no
  prior-season lookup and returned NaN.

Real-data validation:
- `epa_by_game.parquet` regenerated: 14,610 rows and 7,305 games unchanged,
  every pre-existing column identical, special-teams nulls under 1% per
  season, DET-KC 2023 punt EPA matches a hand calculation.
- Modeling file: `schema_version 11`, `data_version 8`, 187 columns, 7,308
  rows. Only weather columns changed among pre-existing ones (755 games);
  `IS_DOME` flipped for exactly those 755. Body-clock nulls are exactly the
  259 games of 1999; Arizona reads 10.0 in September and 11.0 in winter.
- Training-row loss versus schema 10: 15 games in the expanded pool, all
  1999 Monday games with no recorded `gametime` (D68). Logistic's `combined`
  pool is unchanged.
- Live-availability replay (program rule 3): all 226 upcoming games (weeks
  3-18) through the full `CANONICAL_FEATURES` pipeline: zero NaNs across all
  29 new columns at every horizon. Week 3 has zero NaNs across all 239
  expanded features; Elo and strength of victory are NaN beyond next week
  (pre-existing, recorded in ROADMAP).

#### Acceptance

Every A4 row is built with leakage-safe, live-available values or corrected
with an accurate recorded status; the retractable-roof skew is fixed and its
effect measured; special teams exclude kickoffs on recorded evidence; the
modeling file is regenerated at schema 11; the replay shows zero NaNs in the
new columns; decisions are recorded; the Python gate passes. The unit
deviates from its scoped plan in two recorded ways: back-to-back road lives in
a new `road_trips.py` module rather than extending `rest.py` (whose
completed-only history cannot see unplayed prior games), and the training-row
loss is 15 games rather than zero. Promotion remains deferred to Unit 7 (D62).

### Unit 3 — Phase A Batch A3: QB-and-Coach Sidecar [Completed September 28, 2026]

#### Completed

Wired eight Domain 4/10 features — QB EPA/play (career + rolling), QB CPOE
(rolling), a QB-change flag, QB experience, head-coach win % (career), HC
tenure, and a new-coaching-staff flag — into the canonical Win/Total
feature set. Built a new QB/coach identity sidecar per program rule 5,
since starting-QB and head-coach identity are nflverse schedule fields the
existing `games`/`schedule_upcoming_rich` clean steps don't carry.
Starting-QB identity for unplayed games uses a carry-forward-most-recent-
known-starter policy (`DECISIONS.md` D65), verified at 100% coverage
against the real current upcoming schedule. Verified against regenerated
real data and a full replay of the current upcoming schedule. Promotion is
deferred to Unit 7, per D62.

#### Goal

Close seven `❌` Domain 4/10 Game-Model Build Queue rows (`FEATURES.md`
A3), establishing the first Phase A batch that reads new nflverse
**schedule** fields not present in any existing cleaned dataset, and the
program's first genuinely new missingness policy (carry-forward starter
identity) rather than reusing D60's existing era-gap pattern outright.

#### Files Added/Removed/Changed

Added:
- `src/gridiron_edge/transform/clean/qb_coach_sidecar.py` -
  `build_qb_coach_sidecar()`: combines completed rows from
  `games_raw_nflverse` and unplayed rows from
  `schedule_upcoming_raw_nflverse` (the same two raw artifacts `games`/
  `schedule_upcoming_rich` already read) into the new QB/coach identity
  sidecar. No new external fetch.
- `src/gridiron_edge/features/team/qb_profile.py` - Resolves each side's
  starting QB from the sidecar (carry-forward policy, D65), then derives
  career/rolling EPA and CPOE, experience, and a change flag from
  `player_game_logs`. Registers `home_away_qb_profile`.
- `src/gridiron_edge/features/team/coach_profile.py` - Career (not
  same-season) coach win % from the sidecar joined to completed-game
  results; tenure and a new-staff flag from consecutive-season
  (team, coach) continuity. Registers `home_away_coach_profile`.
- `tests/unit/transform/clean/test_qb_coach_sidecar.py` (7 tests),
  `tests/unit/features/test_qb_profile.py` (21 tests),
  `tests/unit/features/test_coach_profile.py` (18 tests).

Changed:
- `src/gridiron_edge/datasets/registry.py` - New `qb_coach_sidecar`
  `DatasetKey`/`DatasetSpec`.
- `src/gridiron_edge/datasets/loaders.py` / `accessor.py` - Added
  `qb_coach_sidecar()` and `player_game_logs()` (the latter was missing
  despite `player_game_logs` already being a registered dataset key).
- `src/gridiron_edge/transform/clean/__init__.py` /
  `src/gridiron_edge/cli/transform.py` - Exported `build_qb_coach_sidecar`
  and added `gridiron transform clean-qb-coach-sidecar`.
- `src/gridiron_edge/cli/main.py` - `run-data-pipeline` gained a
  `clean-qb-coach-sidecar` stage between `clean-upcoming` and
  `fetch-weather`, so the standard pipeline keeps the sidecar current
  rather than requiring a separate manual step.
- `src/gridiron_edge/cli/post_week.py` - Next-week-state refresh also
  runs the new stage.
- `src/gridiron_edge/models/game_prediction/_columns.py` - Added
  `_QB_PROFILE_SUFFIXES`/`_COACH_PROFILE_SUFFIXES` (Elo-pattern manual
  lists, per D63) to `_RAW_FEATURES`/`_DIFF_FEATURES`; added the two
  boolean flags directly to `_TEAM_FEATURES` (no diff column, matching
  `AWAY_SHORT_WEEK`/`HOME_SHORT_WEEK`).
- `src/gridiron_edge/models/game_prediction/_features.py` - Fixed
  `_make_diff_features` to loop over the two new suffix lists — the same
  gap Unit 2 hit (declared `*_DIFF` columns without computing them),
  caught this time before the unit closed.
- `src/gridiron_edge/features/pipeline.py` - Registered
  `home_away_qb_profile`/`home_away_coach_profile` in
  `CANONICAL_FEATURES`.
- `src/gridiron_edge/features/manifest.py` - `CURRENT_SCHEMA_VERSION`
  9→10, `CURRENT_DATA_VERSION` 6→7.
- `src/gridiron_edge/evaluation/prediction_input_sources.py` - Added the
  new sidecar to `PREDICTION_SOURCE_SPECS` (`required=False`, matching
  `epa_by_game.parquet`'s existing entry) — program rule 5 names this
  requirement specifically for new schedule-derived sidecars.
- `tests/unit/features/test_canonical_feature_sequence.py`,
  `tests/unit/models/test_game_feature_sets.py` - Updated the hardcoded
  `CANONICAL_FEATURES` sequence and feature counts (110/55/165/16/30/211).
- `tests/unit/datasets/test_registry.py`,
  `tests/unit/evaluation/test_prediction_input_sources.py` - Updated
  hardcoded dataset-key and source-spec counts/contents.
- `tests/unit/cli/test_main.py`, `tests/unit/cli/test_weekly_predict.py`,
  `tests/unit/cli/test_post_week.py` - Updated hardcoded pipeline
  stage sets for the new `clean-qb-coach-sidecar` stage.
- `FEATURES.md` - Domain 4/10 status corrections; Game-Model Build Queue
  and Summary Statistics updates; changelog entries.
- `ROADMAP.md` - Unit 3 marked closed in the Phase A unit sequence.
- `HANDOFF.md` - Canonical Data Pipeline stage list.
- `DECISIONS.md` - Added D65 (carry-forward starter policy).
- `CHANGELOG.md` - This unit's entry.

Removed:
- None.

Regenerated (not committed; `data/` is gitignored):
- `data/cleaned/NFL_qb_coach_sidecar.parquet` (new) via
  `gridiron transform clean-qb-coach-sidecar`.
- `data/modeling/modeling_file.parquet` and its manifest via
  `gridiron features model-inputs --all-years`.

#### Tests

Validation passed:
- 46 new focused tests across the sidecar transform and the two new
  features;
- repository-wide Ruff and Pyrefly;
- the full non-slow unit suite (4,322 tests).

Real-data validation:
- `data/raw/NFL_wk_by_wk_nflverse.parquet` inspected directly: starting-QB
  and head-coach identity are 0% null for every completed game 1999–2025;
  345 of 348 distinct QB IDs resolve against `player_game_logs`; CPOE is
  100% null before 2006, the same boundary as team-level CPOE.
- The modeling file regenerated in full: manifest shows `schema_version:
  10`, `data_version: 7`, 166 columns, row count unchanged (7,308).
- Live-availability replay (program rule 3) against the real 2026-2027
  upcoming schedule (weeks 3-18, 226 scheduled games), run through the
  full `CANONICAL_FEATURES` pipeline: zero NaNs across all eighteen new
  columns for every scheduled game — carrying forward each team's most
  recent known starter gives 100% coverage, and every carried-forward
  QB has prior `player_game_logs` history.

#### Acceptance

QB EPA/CPOE, QB change/experience, and head-coach win %/tenure/new-staff
are correctly computed, leakage-safe, and available in the canonical
Win/Total feature set. Regenerated data and the live-availability replay
confirm the new columns are guaranteed pre-kickoff values for every
currently scheduled game, including the QB-identity-dependent columns via
the carry-forward policy recorded in `DECISIONS.md` D65. The new sidecar
is registered in the prediction-input evidence source inventory, and the
standard `run-data-pipeline`/`post-week` commands keep it current.
Promotion (the gated Brier/MAE comparison against the current champion)
remains explicitly deferred to Unit 7 per D62 and is not part of this
unit's acceptance bar.

### Unit 2 — Phase A Batch A2: Opponent-Adjusted EPA and Unit Interactions [Completed September 28, 2026]

#### Completed

Wired opponent-adjusted offensive and defensive EPA/play (pass + rush
split) into the canonical Win/Total feature set via a ridge regression on
play-level PBP — the first canonical feature to read play-level data
directly rather than an already-aggregated intermediate — plus two
features built on top of it: unit efficiency differentials (a
matchup-level pass/rush edge) and the market-blind variant of projected
total plays. Verified against regenerated real data and a full replay of
the current upcoming schedule. Promotion is deferred to Unit 7, per the
same wiring/promotion split as Unit 1 (D62).

#### Goal

Close four open Phase A2 catalog rows (Domain 1/2 opponent-adjusted
EPA/play, Domain 13 unit efficiency differentials and projected total
plays), establishing the first play-level (not game-aggregate) feature and
the first cross-team matchup feature in the canonical pipeline.

#### Files Added/Removed/Changed

Added:
- `src/gridiron_edge/features/team/opponent_adjusted_epa.py` - Ridge
  regression on play-level PBP (offense + defense terms per team, season-
  to-date, fit fresh per as-of cutoff) and the registered
  `HomeAwayOpponentAdjustedEpaFeature`.
- `src/gridiron_edge/features/team/unit_efficiency.py` - Matchup-level
  feature: each side's opponent-adjusted offense against the specific
  defense it faces this game.
- `src/gridiron_edge/features/team/projected_total_plays.py` - Market-
  blind combinator: both teams' own rolling pace, summed.
- `tests/unit/features/test_opponent_adjusted_epa.py` (23 tests),
  `test_unit_efficiency.py` (7 tests), `test_projected_total_plays.py` (6
  tests) - Coverage for the three new features above.

Changed:
- `src/gridiron_edge/datasets/loaders.py` / `accessor.py` - Added
  `pbp()`/`load_pbp()` with optional `seasons`/`columns` filtering (PBP is
  ~540MB, unlike every other zero-arg accessor).
- `src/gridiron_edge/models/game_prediction/_columns.py` - Added the new
  metrics to `_RAW_FEATURES`/`_DIFF_FEATURES`/`_GAME_FEATURES` via
  `_OPPONENT_ADJUSTED_EPA_SUFFIXES`/`_UNIT_EFFICIENCY_SUFFIXES`/
  `PROJECTED_TOTAL_PLAYS` (Elo-pattern, not `EPA_COLS`, since these are
  already-final values, not raw per-game stats).
- `src/gridiron_edge/models/game_prediction/_features.py` - Fixed
  `_make_diff_features` to loop over the two new suffix lists too (it
  declared the new `*_DIFF` columns without computing them — a real gap
  the existing test suite caught immediately).
- `src/gridiron_edge/features/pipeline.py` - Registered
  `home_away_opponent_adjusted_epa`, `unit_efficiency_differentials`, and
  `projected_total_plays` in `CANONICAL_FEATURES`.
- `src/gridiron_edge/features/manifest.py` - `CURRENT_SCHEMA_VERSION`
  6→9, `CURRENT_DATA_VERSION` 3→6 across three sub-batch bumps.
- `tests/unit/datasets/test_accessor.py` / `test_loaders.py` - Delegation
  tests for the new `pbp()` accessor/loader.
- `tests/unit/features/test_canonical_feature_sequence.py` - Updated the
  hardcoded `CANONICAL_FEATURES` sequence.
- `tests/unit/models/test_game_feature_sets.py` - Updated hardcoded
  feature counts and the `_canonical_frame()` fixture.
- `FEATURES.md` - Domain 1/2/13 status corrections; Game-Model Build
  Queue and Summary Statistics updates; changelog entries.
- `ROADMAP.md` - Unit 2 marked closed in the Phase A unit sequence.
- `DECISIONS.md` - Added D63 (recompute-vs-persist methodology) and D64
  (short-to-long team-code mapping).
- `CHANGELOG.md` - This unit's entry.

Removed:
- None.

Regenerated (not committed; `data/` is gitignored):
- `data/modeling/modeling_file.csv` and its manifest via
  `gridiron features model-inputs --all-years` (`epa_by_game.parquet` was
  untouched by this unit).

#### Tests

Validation passed:
- 36 new focused tests across the three new features, plus 4 new
  `pbp()` delegation tests;
- repository-wide Ruff and Pyrefly;
- the full non-slow unit suite (4,276 tests).

Real-data validation:
- The modeling file regenerated in full: `schema_version: 9`,
  `data_version: 6`, 148 columns, row count unchanged (7,308).
- Live-availability replay (program rule 3) against the real 2026-2027
  upcoming schedule (weeks 3-18, 226 scheduled games): zero NaNs across
  all thirteen new columns for every scheduled game — the season-to-date
  regression design already has enough data by week 3 for every team.

Two real defects surfaced and were fixed before this unit closed, both
caught by inspection rather than assumed correctness:
- A short-team-code-to-long-name mapping gap (`DECISIONS.md` D64) that
  would have silently produced all-`NaN` output for every game.
- `UnitEfficiencyDifferentialsFeature` originally persisted its two
  `_DIFF` columns directly into the base modeling file, unlike every
  sibling feature (Elo, opponent-adjusted EPA), where `_DIFF` columns are
  always derived later by `_make_diff_features` and never stored in the
  base file. Caught by inspecting the regenerated manifest directly (150
  columns instead of the expected 148), not by a failing test.

#### Acceptance

Opponent-adjusted EPA, unit efficiency differentials, and projected total
plays are correctly computed, leakage-safe, and available in the canonical
Win/Total feature set, consistent with every sibling feature's wiring
convention after the mid-implementation correction. Regenerated data and
the live-availability replay confirm the new columns are guaranteed
pre-kickoff values for every currently scheduled game. Promotion (the
gated Brier/MAE comparison against the current champion) remains
explicitly deferred to Unit 7 per D62 and is not part of this unit's
acceptance bar.

### Unit 1 — Phase A Batch A1: Construction Fixes [Completed September 28, 2026]

#### Completed

Wired CPOE (`off_cpoe`/`def_cpoe`), a paired pre-2007 era-gap availability
indicator for it, and a new offensive fumble-rate feature into the
canonical Win/Total feature set, and corrected a stale catalog status for
an already-built defensive interception-rate feature. Verified against
regenerated real data and a full replay of the current upcoming schedule.
Promotion — the gated retrain that decides whether these features are
adopted into a champion — is deferred to a new consolidated-sweep unit
(Unit 7), per a same-day revision to the program's wiring/promotion rules
(`DECISIONS.md` D62).

#### Goal

Close two genuinely open Phase A1 catalog rows (CPOE, offensive fumble
rate) and correct one stale row (defensive interception rate), establishing
the wiring half of the "wire now, promote later" pattern the rest of
Phase A's units will follow.

#### Files Added/Removed/Changed

Added:
- `tests/unit/transform/clean/test_epa.py` - Fumble-rate aggregation
  coverage; this transform module had no prior test coverage.

Changed:
- `src/gridiron_edge/transform/clean/epa.py` - Added `off_fumble_rate`/
  `def_fumble_rate` to `_agg_side` (fumbles lost per offensive scrimmage
  play); updated the output-schema docstring.
- `src/gridiron_edge/features/team/epa.py` - Added `off_cpoe`, `def_cpoe`,
  `off_cpoe_available`, `def_cpoe_available`, `off_fumble_rate` to
  `EPA_COLS` (36→41); added `_apply_cpoe_availability_policy` (D60's
  fill-plus-indicator era-gap handling) wired into
  `HomeAwayEpaFeature.compute()`.
- `src/gridiron_edge/features/manifest.py` - Bumped `CURRENT_SCHEMA_VERSION`
  5→6 and `CURRENT_DATA_VERSION` 2→3.
- `src/gridiron_edge/models/game_prediction/_features.py` - `FEATURE_SETS`
  names now derive from the schema version (D61) instead of a hand-counted
  column total.
- `tests/unit/features/test_epa.py` - Updated `EPA_COLS` count/membership;
  added CPOE-availability-policy coverage and an explicit no-new-leakage
  test for the new columns.
- `tests/unit/models/test_epa_window.py` - Added the new columns to the
  realistic `synthetic_epa_by_game` fixture.
- `tests/unit/models/test_game_feature_sets.py` - Updated hardcoded feature
  counts (84/42/126/167) and `FeatureSet` name assertions.
- `FEATURES.md` - Status corrections (CPOE, Fumble rate/off, INT rate/def
  in Part I); Game-Model Build Queue and Summary Statistics updates (Part
  III); changelog entries.
- `ROADMAP.md` - Program rules 1/2 revised to decouple wiring from
  promotion; Phase A unit sequence expanded from 6 to 7 units.
- `DECISIONS.md` - Added D60 (CPOE era-gap policy + fumble-rate
  denominator), D61 (`FeatureSet` naming), D62 (wiring/promotion
  decoupling).
- `CHANGELOG.md` - This unit's entry.

Removed:
- None.

Regenerated (not committed; `data/` is gitignored):
- `data/cleaned/epa_by_game.parquet` via `gridiron transform aggregate-epa`.
- `data/modeling/modeling_file.csv` and its manifest via
  `gridiron features model-inputs --all-years`.

#### Tests

Validation passed:
- focused fumble-rate, CPOE-availability-policy, feature-count, and
  new-column leakage-boundary tests;
- repository-wide Ruff and Pyrefly;
- the full non-slow unit suite (4,236 tests).

Real-data validation:
- `data/cleaned/epa_by_game.parquet` regenerated from cached PBP: same
  14,610 rows; `off_fumble_rate`/`def_fumble_rate` 0% null with a realistic
  ~0.7-0.8% per-play rate; `off_cpoe` pre-2006 nullity unchanged.
- The modeling file regenerated in full: manifest shows `schema_version: 6`,
  `data_version: 3`, 135 columns including the new CPOE and fumble-rate
  fields.
- Live-availability replay (program rule 3) against the real 2026-2027
  upcoming schedule (weeks 3-18, 226 scheduled games): zero NaNs across all
  five new columns for every scheduled game. Read-only — no file under
  `data/` was modified by the replay itself.

#### Acceptance

CPOE and offensive fumble rate are correctly computed, era-gap-aware (D60),
and available in the canonical Win/Total feature set with no new leakage
surface; the defensive interception-rate catalog status now matches the
actual, already-built aggregation. Regenerated data and the live-
availability replay confirm the new columns are guaranteed pre-kickoff
values for every currently scheduled game. Promotion (the gated Brier/MAE
comparison against the current champion) is explicitly deferred to Unit 7
per the revised program rules (D62) and is not part of this unit's
acceptance bar.

### Completed Program: Weekly Prediction Input Integrity and Reproducibility [Closed September 25, 2026]

Units 1 through 5 completed history-preserving weekly refresh, deterministic complete-history Elo reconstruction, exact persisted Elo-lineage enforcement, external disposition of the affected immutable 2026 Week 2 forecast evidence, and immutable prediction-input evidence for newly generated live weekly forecasts.

Corrected operational forecast generation and evaluation, model-quality assessment, and explanation evidence are now tracked as ROADMAP.md Tier 2 #4–7 and Tier 3 #8–10, under the active program below.

### Completed Program: Foundation Completion (Tiers 1–4) [Closed September 27, 2026]

Two coordinated tracks. Track A (Tiers 1–3, Units 1–16, sequential) resolved
every verified correctness defect (D1–D5, D8; D6–D7 previously resolved
in-flight), replaced the canonical Week 2 development state, rebuilt
historical game-model evaluation on a corrected pipeline, reassessed
calibration and champion selection, and persisted and served both Logistic
explanation evidence and comparable-game retrieval through `ExplainPage`.
Track B (Tier 4 / Market Unit 26, calendar-driven) proved the complete
production recommendation chain for real 2026 Week 1 data, independently for
Moneyline, Spread, and Total, and fixed three defects found along the way
(a duplicate candidate-issuance conflict, a postgame timestamp-dedup crash,
and a broken cross-game closeout-kickoff check).

Every item in ROADMAP.md's Foundation Completion acceptance checklist is
met. Full per-unit implementation detail lives in git commit history (each
unit's own commit) and `DECISIONS.md` D43–D59; Market Unit 26's complete
closing record lives in `docs/programs/market-unit-26/PLAN.md`.

No unit is currently active. The next program must be deliberately selected
from `ROADMAP.md` — Tier 5 #16 (the game-model feature program) is now
unblocked.

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
