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

Active unit: Unit 7 (screening and promotion), below. Units 1–5 (batches A1–A5) and the player-log refresh pipeline unit closed below. Unit 6 (market-aware variant) is deliberately not built.

### Unit 7 — Phase A Screening and Promotion [Active, scoped September 28, 2026]

#### Goal

Make the adopt/reject call for batches A1–A5 (`DECISIONS.md` D62) with one
consolidated incremental sweep: baseline (schema 5 composition), +A1,
+A1+A2, … +A1..A5, in wiring order. Each batch's attributable delta is
measured against the step immediately before it. Win covers Logistic, Random
Forest and XGBoost (Elo is a constant reference row); Total covers Random
Forest and XGBoost. Rejected batches and dead-weight columns are removed by
one forward schema bump at the end. Results and reasons go in `FEATURES.md`.

#### Verified current state (September 28, 2026)

- Modeling file: schema 12, data 9, 7,308 rows, 201 columns. Data is current
  through 2026 Week 2; 2026 rows are outside both training (seasons before
  2023) and the holdout (`HOLDOUT_SEASONS`, 2023-24 to 2025-26), so freshness
  does not affect the sweep.
- Feature counts by step (combined for Logistic / expanded for RF, XGBoost,
  Total): baseline 111/152, A1 126/167, A2 144/186, A3 165/211, A4 186/239,
  A5 207/260. Baseline counts match the on-disk champion metadata.
- Hyperparameter search: `TimeSeriesSplit` 5 folds, seed 42, sampled combos
  RF Win 50, XGBoost Win 75, RF/XGBoost Total 50 each; Logistic searches the
  six EPA windows with an inner `LogisticRegressionCV`. Sampled combos depend
  only on grid and seed, so every step draws the same ones.
- Every persisted champion artifact is schema 5 (`combined_111`,
  `expanded_152`) and already fails the metadata preflight (D62). The
  manifest champions are Logistic (Win) and Random Forest (Total).
- Existing gates: `compare_classification_models` (Brier >= 0.002, ECE
  <= +0.01, AUC <= -0.01) has no bootstrap, per-season or log-loss check;
  `compare_regression_models` needs R² > 0 and any MAE improvement. No Total
  MAE/RMSE/bias threshold exists anywhere (Tier 2 reassessment set none).
- Row loss by step on the expanded set (holdout games of 854 Win / 855
  Total): baseline 854/855, A2 806/807, A3 797/798, A4 and A5 unchanged. A2's
  adjusted EPA is NaN for every Week 1 game (441 of 448 NaN rows; all 48
  holdout Week 1 games) because D63 uses season-to-date only, while baseline
  EPA is never NaN in the holdout. A3's career QB EPA and coach win % are NaN
  on 24 holdout games (15 in Week 1). Program rule 3's replay covered only
  Weeks 3-18, so it never exercised a season opener.
- A4 changed pre-existing values (D67: `IS_DOME` and weather for 755
  retractable-roof games), so a column-masked baseline still carries that fix.
- The trainer recalibrates XGBoost from holdout ECE (`base.py`), and walk-
  forward backfill reuses the schema-5 champion hyperparameters. The sweep
  can use neither.

#### Design decisions

1. **Harness:** a reusable, read-only `gridiron evaluate feature-sweep`
   command over a new `evaluation/feature_sweep.py`, built on the lower-level
   search and fit helpers, never `GamesTrainer.train` with persistence. It
   never calls `ArtifactStore.save`, never writes `data/models`, champions or
   any prediction artifact, and writes nothing under `data/`.
2. **Artifacts:** one consolidated Markdown report at
   `docs/reports/phase-a-feature-sweep.md` plus a per-game holdout prediction
   JSON beside it. The report is a dated record, not an input to any pipeline.
3. **Steps and columns:** each step's columns are the exact schema-5..12
   compositions reconstructed from the wiring commits, selected by name from
   the current modeling file. D67 is treated as a bug fix present in every
   step, including baseline, and recorded as a caveat.
4. **Primary comparison is row-matched:** the training pool and holdout are
   the intersection of rows valid at all six steps (expected 797 Win / 798
   Total holdout games), identical for every step and model. A secondary
   as-shipped reading refits with the tuned hyperparameters on each step's own
   pool to record training-row loss.
5. **Hyperparameters and seeds:** each step runs its own search on training
   data only, with identical seeded combos and unchanged budgets. A
   fixed-hyperparameter sensitivity reading uses the baseline's tuned values.
6. **Holdout scoring:** hyperparameters are frozen after the pre-2023 search;
   models then retrain expanding through each holdout season (D48-style)
   to produce per-season predictions. Nothing is selected on the holdout.
7. **Calibration:** Random Forest is always isotonic-wrapped (D53). XGBoost
   gets production's isotonic wrap when its out-of-fold *training* ECE
   exceeds production's 0.025 threshold, decided once per step from the
   pre-holdout search and frozen; never from holdout ECE. The Elo reference
   is an `ELO_DIFF`-only logistic fit on the same rows (the simulated Elo
   model is not re-run, since Elo inputs are identical at every step).
8. **Metrics:** Win: Brier, log loss, ECE, AUC, calibration slope/intercept,
   per-season Brier. Total: MAE, RMSE, bias, median absolute error, per-season
   MAE. Deltas use a paired game-bootstrap (fixed seed, 90% interval).
9. **Win bar (from ROADMAP):** Brier improves >= 0.002, bootstrap 90% lower
   bound above zero, holding in 2 of 3 seasons; ECE worsens <= 0.01; log loss
   not worse.
10. **Total bar (locked here, before any result exists):** MAE improves
    >= 0.09 (about 0.9% of MAE, the relative size of 0.002 Brier), bootstrap
    90% bound clear of zero, in 2 of 3 seasons; RMSE not worse; bias shifts
    by no more than 0.25.
11. **Cross-family attribution:** columns are shared, so a batch is kept if it
    clears the bar for at least one family and no family shows significant
    harm (bootstrap 90% bound worse than +0.001 Brier or +0.05 MAE). Per-model
    feature sets are out of scope.
12. **Week-1 missingness (A2, A3):** the sweep runs as-is on the matched
    games. A batch that clears the bar but leaves Week 1 NaN is "adopt only if
    fixed": it is recorded, and the fix (prior-season carry-in or fill, a new
    D-entry, its own schema bump and replay including a season opener) is a
    separate follow-on unit, not done here. Such a batch stays wired until
    that unit decides.
13. **Column-level suspects** (FORM_TREND, GLICKO2_RD, GLICKO2_RATING diff,
    collinear rating diffs in Logistic's set, QB features): drop-one-group
    ablations from the full A1-A5 model, decided on a pre-holdout
    walk-forward window (2019-2022). The holdout figure is printed for
    confirmation only and cannot change a decision. Glicko-2 volatility
    stays declined (D70).
14. **Multiple comparisons:** the holdout was used before for champion
    selection, and the bar is tested across several families and steps. The
    report states the number of comparisons and flags any pass that misses
    the pre-holdout window's direction.
15. **Champions and schema:** the sweep overwrites no champion. One forward
    schema bump removes rejected batches and columns at the end, with the
    full non-slow gate. Retraining and promoting champions under the final
    schema is a separate follow-on unit, since retraining earlier would bake
    the rejected `feature_set_name` into new artifacts.
16. **Leakage:** no hyperparameter, calibration or column decision reads
    holdout data; tests assert the harness's decision inputs exclude it.
17. **Noise-floor evidence:** (a) a placebo batch at each step (random-noise
    columns as wide as the real batch, fixed seed) run through the same
    pipeline; a real batch must clearly exceed the placebo delta spread to be
    credited. (b) Tree models rerun with 3 seeds; the report gives each
    delta's seed spread. Either can veto a marginal pass; neither can rescue a
    fail.
18. **Reference models:** Win: constant home-win rate and an `ELO_DIFF`-only
    logistic, fit on the same training rows; Total: constant league-mean
    predictor. Reported on the matched holdout in every step's table, with the
    same metrics and bootstrap.
19. **Additional diagnostics:** an early-season slice (Weeks 1-3 versus the
    rest) for every step; a leakage smell test flagging new columns with
    implausibly high univariate AUC/correlation on training data; a per-column
    train-versus-holdout drift check (PSI); Logistic condition number, VIF and
    coefficient stability across the expanding refits; Holm correction beside
    the raw bootstrap bounds. Evidence only; none changes a bar.
20. **Frontend comparison is out of scope here:** showing the reference and
    Elo comparisons in the frontend needs a persisted immutable evidence
    product, an API contract change and generated-contract regeneration
    (`ROADMAP.md`, frontend enhancements). This unit's reference outputs are
    shaped so that product can reuse them, but persists nothing under `data/`.

#### Implementation plan

1. Sub-step 1: step definitions and matched-row builder, with tests for
   composition per step and identical holdout games across steps.
2. Sub-step 2: metrics and paired-bootstrap deltas, per-season splits, and
   the adopt/reject evaluator for both bars, with unit tests on synthetic
   data.
3. Sub-step 3: search, expanding holdout refit and calibration rule; the
   reference models, placebo columns and seed reruns; a timed single-step
   pilot on the baseline, reported to the user with a runtime estimate before
   the full run.
4. Sub-step 4: report writer and CLI; then, after the user approves, the
   full sweep and the column-level ablations.
5. Sub-step 5: record decisions in `FEATURES.md` (and a D-entry for the bars
   and attribution rules), one forward schema bump for removals, gate, and
   closeout.

#### Tests

- Unit: step compositions, matched-row invariants, metric and bootstrap
  correctness, gate evaluator edge cases, no-holdout-in-decisions, no writes
  outside the report path, CLI wiring.
- Real-data verification: inspect actual holdout counts, paired-bootstrap
  intervals, ECE, log loss and per-season numbers in the report; reproduce
  the baseline reading against the schema-5 champion metrics as a sanity
  check; confirm no champion or prediction artifact changed (hashes before
  and after).
- Python gate after the schema bump.

#### Progress and first results (2026-09-29)

Sub-steps 1-4 are built and the full sweep ran (5.5 hours, five families, six
steps each; champion artifacts hash-identical before and after; nothing
written under `data/`).

Added (uncommitted): `evaluation/feature_sweep_steps.py`, `_rows.py`,
`_metrics.py`, `_gates.py`, `_fit.py`, `_run.py`, `_driver.py`, `_report.py`,
`_diagnostics.py`, `_ablation.py`, `_columns.py`; the `gridiron evaluate
feature-sweep` command in `cli/evaluate.py`; and tests
`tests/unit/evaluation/test_feature_sweep_*.py`. Outputs (untracked):
`docs/reports/phase-a-feature-sweep.md` and
`docs/reports/phase-a-feature-sweep-predictions.json`.

Result (`DECISIONS.md` D71, `FEATURES.md`): no batch clears the bar in any
family; every 90% interval is as wide as the bars, so the reading is "not
demonstrated", not "harmful". Baseline to the full set: Win Brier +0.0006
(Logistic), +0.0026 (Random Forest), +0.0020 (XGBoost); Total MAE +0.040
(Random Forest), +0.039 (XGBoost); no Holm-adjusted p below 0.51. A2 and A3
also leave scheduled games without a value (64 and 27 rows from the holdout
onward, mostly Week 1). No leakage flags; `LEAGUE_PPG_TO_DATE` and the CPOE
columns drift (PSI above 0.25); Logistic's combined set is rank-deficient.

Deviations and caveats recorded: the Elo reference is an `ELO_DIFF`-only
logistic; very short EPA windows impute undefined rates with training medians;
Logistic's saga solver hit `max_iter` repeatedly at production settings; RF's
isotonic wrap yields 0/1 probabilities that spike RF log loss on two steps, so
"log loss not worse" is noisy for RF.

Open: the adopt/reject call and any forward schema bump are on hold pending
the user's decision and stage 2 below. Records done: D71, `FEATURES.md`.
Remaining: `HANDOFF.md` entry for the command, the removal bump (once, at the
end, with the full non-slow gate), closeout.

#### Stage 2 - model-driven feature selection (scoped, awaiting review)

Goal: test whether letting each model select its own columns from the wired
pool yields a leaner model at equal accuracy, and, if so, name that subset as
the model's feature set. Motivation: models already regularize, so the upside
of pruning is bounded near the sweep's effect sizes (about 0.002 Brier); the
value is a smaller, live-safer feature set, judged against the champions.

Design decisions:

21. **Selection inside training only.** On the matched pre-2023 rows, with
    time-ordered CV. Logistic: columns with non-zero L1/elastic-net
    coefficients that are stable across folds. Trees: columns with stable
    permutation importance on the CV validation folds. Frozen per model type;
    the holdout never informs a choice.
22. **Placebo control.** Inject random-noise columns into the pool and report
    how many survive selection; a procedure that keeps noise is not credited.
23. **Pool.** All wired columns (schema 12), unchanged, so the pool matches what
    the sweep measured. Survivors are flagged for Week 1 `NaN` risk; the fix is
    the conditional follow-on below.
24. **Gate.** Non-inferiority to both the full set and the baseline, plus
    fewer columns, with the same bootstrap and season checks. A Brier or MAE
    improvement bar is reported but not required; requiring it reruns the
    power problem.
25. **More power, without contaminating it.** The 3-season holdout has too few
    games to resolve effects this small, so the *procedure* is also validated
    on a wider walk-forward window. A subset selected on pre-2023 data cannot
    be scored on 2003-2022, because its selection has seen those seasons.
    Two cutoffs keep it clean: (a) an early selection using only seasons before
    2013 is scored walk-forward on 2013-2025 (about 3,400 matched games) with
    fixed hyperparameters, against the full set and the baseline; (b) the
    deployable subset is selected on all pre-2023 matched rows and confirmed
    once on the 2023-2025 holdout. Non-inferiority means the paired-bootstrap
    lower bound of the improvement is above minus the harm margin (0.001 Brier,
    0.045 MAE). Hyperparameters are frozen from the full-set search and shared
    by all arms; that favors no arm, but is noted.
26. **Schema.** Selected subsets are per-model `FeatureSet`s. The modeling file
    stays at schema 12 until the final subsets are known; the single forward
    bump comes after, with names that record the selection.

Implementation plan: (6) selection procedures with an in-procedure placebo
control and unit tests on synthetic data **[built 2026-09-29: L1 stability
selection with the one-standard-error rule for Logistic; shadow-feature
permutation selection for trees]**; (7) wide walk-forward evaluator with
non-inferiority, unit-tested **[built 2026-09-29]**; (7b) stage driver and
report joining both to the sweep checkpoints; (8) a timed pilot on one family,
reported before the full run; (9) the overnight run (user approval required),
report, decisions in `FEATURES.md` and a D-entry.

Tests: leakage guards (selection never reads holdout labels; assertion on
decision inputs), placebo survival counts, stability across folds, wide
walk-forward alignment with the matched games, and real-data spot checks of the
selected subsets, counts and row loss.

Acceptance: each model type has a frozen, named subset evaluated on the
matched holdout and the wide walk-forward; the noise-survival rate is
reported; the non-inferiority call is recorded; no champion or prediction
artifact changes; the Python gate passes.

#### Stage 2 results and Stage 3 - per-batch attribution on the wide window (pre-declared 2026-09-30, before it ran)

Stage 2 ran on 2026-09-30 (`docs/reports/phase-a-feature-selection.md`). The
selected subsets kept 5-24 columns with zero placebo survivors in the final
selections, but none passed non-inferiority to the full set, and three of five
families were significantly worse on the wide window (2013-2025, about 3,300
games). The same window showed the full A1-A5 set beating baseline: Total
Random Forest +0.046 MAE [+0.006, +0.082], Total XGBoost +0.049
[+0.001, +0.095], Win XGBoost +0.0033 Brier [+0.0011, +0.0054], Win Random
Forest +0.0018 [-0.0001, +0.0036], Win Logistic +0.0017 [-0.0006, +0.0038].
Pruning is therefore not pursued; the question becomes which batches carry the
gain.

Stage 3 scores each batch against the step before it on the wide window using
each step's own frozen hyperparameters from the sweep (no new searches, refits
only). The original holdout bars were sized for about 800 games and their
effect floors cannot be met even by the whole bundle on Total (0.046 against
0.09), so the criterion is revised here, before results, and the revision is
recorded as such. A batch is adopted when all hold for at least one family:

- point improvement is at least half the original floor: 0.001 Brier (Win) or
  0.045 MAE (Total);
- the Benjamini-Hochberg adjusted one-sided p-value, across all 25
  batch-by-family comparisons, is at most 0.10;
- the batch improves at least 60% of the wide-window seasons.

It is rejected when no family passes, or when any family is significantly
harmed (the whole 90% interval worse than 0.001 Brier or 0.05 MAE). A batch
that would be adopted but leaves scheduled games without a value is "adopt only
if fixed". The placebo and seed checks are not repeated on the wide window; the
BH correction and the 60% season rule are the noise guards. Unlike the holdout
bars this is a data-informed revision, motivated by power, not by any batch's
result, and the report states so.

#### Stage 3 results and decision (2026-09-30, `DECISIONS.md` D72)

Stage 3 ran (`docs/reports/phase-a-feature-attribution.md`). Under the criterion
recorded above: A3 (QB and coach) **adopt only if fixed** (improves all five
families; passes Total Random Forest and Win XGBoost); A1, A2, A4 **reject**;
A5 **reject (near miss)**. The composition becomes baseline plus A3.

Remaining work for this unit, in order:
1. **[Built 2026-09-30, `DECISIONS.md` D73]** The fill for A3's incomplete cells:
   debut priors for known starters and coaches with no history (QB EPA per
   attempt -0.1293, coach win share 0.4055, fit on 2000-2022), unresolved
   identities stay missing.
2. **[Code done 2026-09-30; schema 13 / data 10]** One forward bump removing A1,
   A2, A4, A5 from the canonical set (expanded 177 columns, combined 132; code
   retained, unregistered); unit suite 4,561 passed. Modeling inputs regenerated
   2026-09-30 (`gridiron features model-inputs --all-years`, 191 s): 7,324 rows
   in the same order, 201 to 154 columns, only previously-missing cells changed
   in the six filled columns (one value differs by 3e-17), and zero missing
   values in the 177 declared features from 2023 onward, including every Week
   1. Live replay of the upcoming schedule: week 4 has no missing value in any
   feature; weeks 5-18 are missing only `ELO_DIFF`, `AWAY_ELO`, `HOME_ELO` and
   strength of victory (the known next-week-only limitation), none in the kept
   columns.
3. Confirm baseline plus A3 against baseline directly (search and wide-window
   plus holdout refits), since A3 was measured with A1 and A2 present.
4. Retrain and promote champions under the final schema (approval required;
   this also unblocks `weekly-predict`).
5. `HANDOFF.md` entry, full non-slow gate, closeout.

#### Stage 4 - confirmation of the final composition (pre-declared 2026-09-30, before it ran)

A3 was attributed with A1 and A2 present, so the final composition (baseline
plus A3, schema 13) is checked against baseline directly before promotion. To
stay cheap it reuses the sweep's frozen hyperparameters (the baseline step's
for the baseline arm; the old A3 step's for the final arm) and does refits
only, on the regenerated matched games, which now include season openers
because the fill removes their missing values. It reports the wide window
(2013-2025), the holdout (2023-2025) and a Week 1 slice. **Confirmed** when no
family is significantly worse than baseline on either window (the whole 90%
interval worse than 0.001 Brier or 0.05 MAE) and at least 4 of 5 wide-window
point improvements are positive. A failure reopens the composition before any
retrain; a pass releases the game-models-only retrain
(`full-retrain --skip-prop-backfill --skip refresh-all-data`, props are not
touched by the schema).

**Stage 4 result (2026-09-30, `docs/reports/phase-a-final-confirmation.md`):
CONFIRMED.** Baseline plus the QB-and-coach batch beats baseline in all five
families on the wide window (3,550-3,562 matched games, now including season
openers), with every 90% interval excluding zero: Total Random Forest +0.055
MAE, Total XGBoost +0.040, Win Logistic +0.0015 Brier, Win Random Forest
+0.0021, Win XGBoost +0.0027. Holdout point improvements are all positive with
intervals including zero (expected at about 850 games). Week 1 games: positive
in four of five families (Win Logistic -0.0007, not significant, 204 games).
The game-models-only retrain is released.

#### Follow-on - incomplete-value fill for A3 (built; see D73)

Only for columns that survive stage 2 (or are otherwise kept) and leave
scheduled games without a value. Design constraint: fill `NaN` cells only, so
every existing non-`NaN` value is unchanged and the sweep and stage 2 results
still apply. Adjusted EPA: a prior-season carry-in shrunk toward zero for a
season's first week. Career QB EPA/CPOE and head-coach win %: an indicator plus
fill, per the D60 policy. Requires a missingness `DECISIONS.md` entry (program
rule 4), a correction note to D63, regenerated modeling inputs (a `data/`
write, approval required), and a live replay that includes a season opener.
This is its own bounded unit and starts only after this one closes.

#### Approvals required from the user

The full sweep ran on approval (option A). Still required: stage 2's overnight
run; any external-source call (the week 3 nflverse refresh is separate); any
write to `data/`; the removal bump; commit.

#### Acceptance

The sweep reports every step on identical matched games with the bars above;
each of A1-A5 has a recorded adopt, reject or adopt-only-if-fixed call with
reasons in `FEATURES.md`; column-level suspects have recorded outcomes;
rejected batches and columns are removed by one forward schema bump; no
champion or prediction artifact changed; the Python gate passes.

### Pipeline Unit: Player Game Log Refresh Stages [Completed September 28, 2026]

#### Completed

`data/cleaned/player_game_logs.parquet` now refreshes with the weekly data
update. Two new pipeline stages, `fetch-player-stats` and `clean-player-stats`,
run ahead of `build-features` in `run-data-pipeline`, `post-week`'s
`refresh-next-week-state`, and `weekly-predict`. The weekly fetch re-fetches
the resolved season on every run, since its file grows through the year, and a
fetch that fails or returns no rows can no longer replace a cached season
file; the stage reports that outcome instead of failing, because a new season
has no data before its first game. `gridiron ingest player-stats` and
`gridiron transform clean-player-stats` run the steps individually. The
cleaning step is unchanged and still needs network access for its schedule
lookup.

#### Goal

Remove the gap that left the logs, and every feature built from them, lagging
the games file by every week since the last manual run: the fetcher had no
command or stage and skipped any cached season, so the current season could
never refresh.

#### Files Added/Removed/Changed

Added:
- `tests/unit/cli/test_player_stats_commands.py` - coverage for the two new
  commands.

Changed:
- `src/gridiron_edge/ingest/nflverse/player_stats.py` - `force` option; a
  failed or empty fetch never replaces a cached season file.
- `src/gridiron_edge/cli/main.py` - the two stages in `ALL_STAGES` and
  `_run_pipeline_stages`, with a visible detail when no data is returned.
- `src/gridiron_edge/cli/post_week.py` - the next-week-state refresh runs both
  stages.
- `src/gridiron_edge/cli/ingest.py`, `transform.py` - `ingest player-stats`
  and `transform clean-player-stats`.
- `tests/unit/ingest/test_player_stats.py` - force, skip-when-cached, and
  never-degrade-the-cache coverage.
- `tests/unit/cli/test_main.py`, `test_post_week.py`, `test_weekly_predict.py`
  - stage order and set assertions, and stage execution.
- `HANDOFF.md` - stage list, refresh semantics, and the network note.
- `ROADMAP.md` - removed the resolved known limitation.
- `CHANGELOG.md` - this unit's entry.
- `PLAN.md` - this record.

Removed:
- None.

Regenerated (not committed; `data/` is gitignored):
- `data/raw/player_stats/player_stats_2026.parquet` and
  `data/cleaned/player_game_logs.parquet`.

#### Tests

Validation passed:
- 12 new tests plus updated stage-set assertions; ruff, pyrefly, and the full
  non-slow unit suite (4,479 tests).

Real-data validation:
- Ran `gridiron run-data-pipeline --only fetch-player-stats --only
  clean-player-stats` against nflverse: both stages completed in 2.3 s. The raw
  2026 file (3,274 rows, weeks 1-3) and cleaned logs (137,307 rows) were
  rewritten with contents identical to the earlier manual refresh, since no
  game has finished since; all 136,384 pre-2026 rows are identical, columns
  are unchanged, and there are no duplicate (player, game) rows.

#### Acceptance

The weekly, post-week, and predict entry points refresh player logs before
building features, a failed or empty fetch cannot degrade the cache, the stages
and commands are documented, the known limitation is removed, and the Python
gate passes.

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
  (recorded in ROADMAP, since resolved by the player-log refresh unit above). Player logs were then refreshed by hand (2026
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
