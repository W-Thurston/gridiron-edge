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

### Active Program: Weekly Recap (ROADMAP W15) [Scoped October 4, 2026]

**Units 23 and 24 completed October 4, 2026, and Unit 26 October 5, 2026 (`CHANGELOG.md`); next: Unit 25 -
Starting-QB section, not yet active. Unit 26 was taken ahead of it on 2026-10-05 because week 4 was not finished
(2 of 16 games scored in the local raw file; Monday ATL @ NO unplayed; no closeout receipt); start Unit 25 once
week 4's closeout receipt exists.** Sequence:
Unit 23 (artifact and game forecasts), Unit 24 (API and page), Unit 25 (starting QBs, the former
Unit 16), Unit 26 (weekly-cycle hook), Unit 27 (season roll-up), Unit 28 (edges, recommendations and
wagers). One unit is active at a time. Every unit is scoped below so a new thread can pick it up;
only the active one may change code. Evidence below was read from this checkout on 2026-10-04
(read-only; one throwaway probe script, not kept). Items marked *unverified* were not checked.

**Picking up a unit in a new thread.**
1. Read CLAUDE.md, "Ways of Working" above, this program section (goal, audit, decisions, shared
   contracts) and the unit's own section. The unit's "Start here" list names the files to read first.
2. Confirm the previous unit is committed (`git log`; CHANGELOG.md) and that the unit's "Depends on" holds.
3. Re-check the audit facts the unit relies on: data changes weekly (a new week closes every Tuesday), so
   counts quoted here are as of 2026-10-04.
4. Settle the unit's "Decide at start" items with the user before writing code, and record the answers in
   the unit section.
5. Mark the unit `[Active]` here and the previous one `[Completed]` (with a pointer to CHANGELOG), and
   follow /closeout at the end. Ask before any `data/` write, any commit, and anything in CLAUDE.md's
   "Ask before running" list.

**Goal.** One page that answers "how did we do last week?" for every projection: each game's win
probability, spread, total and projected score against the result and against the closing market; the
assumed starting QBs against who played; the edges and recommendations published, with their outcomes
and closing-line comparison; and recorded wagers. Season aggregates sit beside the per-game table.
Unavailable pieces are shown with an explicit status, never hidden. The recap is an immutable artifact
built by a CLI command from persisted state, served read-only, and shown on one page. Nothing is scored
at request time.

#### Audit: what exists and what the recap can reuse

- **Finished weeks with a selected product: 1, 2 and 3, not zero.** Each has a selected product and
  16 of 16 outcomes in the cleaned games file. Week 1 (`a57bb984`) and week 3 (`f2150cb7`) are `live`;
  week 2 (`deb7ebd3`) is `development` (regenerated after the fact, not issued before kickoff). All three
  are product schema 1, so none carries a starter record. Week 4 is in progress (2 of 16 scored on
  2026-10-04; closes from Tuesday 2026-10-06 06:00 local). Week 5 is the first week whose every row will
  carry a starter record.
- **The selected week 4 product is schema 2 with mixed rows.** `8bd40f8d` (the cycle refresh of
  2026-10-04T19:36Z) carries 9 rows from run `81248588` (a schema-1 run; those rows read
  `starting_qb_status = sidecar_unavailable`, although the run simply predates the record) and 6 rows
  of its own with a full starter record. It has 15 rows: **PIT @ CLE (Thursday 2026-10-01, final 24-27)
  is in no product from `81248588` on.** Its last pre-kickoff forecast is in non-selected product
  `94803328` (2026-10-01T12:39Z); product `37997002` was generated 2026-10-02T03:48Z, after kickoff, and
  is labeled `live` (the defect `DECISIONS.md` D83 fixed for later weeks). The recap must show this game
  as not in the selected product, and must not reach into a non-selected product to fill it.
- **The closeout is reusable as is, but it is not persisted and it counts product rows, not scheduled
  games.** `evaluation/live_forecast_closeout.close_live_forecasts` is pure: it matches each row to its
  own event and run (carried rows included, D83) and returns a per-game reconciliation frame (win
  probability, projected home margin, model total, scores, actual margin and total, tie handling,
  evaluable flags) plus Brier, log loss, accuracy, MAE, RMSE and bias. `post-week` and the cycle only
  print it. Run read-only on 2026-10-04 it reconciles weeks 1-3 completely (win Brier 0.237, 0.228,
  0.246; spread MAE 11.87, 12.14, 9.18; total MAE 12.70, 11.93, 11.81) and reports week 4 as 15
  scheduled games, missing PIT @ CLE.
- **Schedule and outcome sources.** The full schedule (including unplayed games) is the raw nflverse
  games file (`games_raw_nflverse`); the pure helpers `week_completion` and `completed_game_ids` that
  read it live in `cli/week_cycle.py`. The cleaned games file holds only completed games, with the
  closeout's canonical scores and the closing lines `VEGAS_LINE` (expected home margin) and `OVER_UNDER`.
  For all 50 completed 2026 games these equal nflverse's post-game `spread_line` and `total_line`.
- **Closing market convention.** `evaluation/market_agreement.py` (D85) turns `VEGAS_LINE` into a home
  win probability with a normal margin (`MARGIN_STD = 13.5`) and compares totals with `OVER_UNDER`. The
  recap uses the same convention so a week reads on the same scale as the market-agreement report. Probe
  (ad hoc moneyline de-vig, not the D85 convention; for orientation only): model against market win
  Brier 0.237/0.214, 0.228/0.231, 0.246/0.254; total MAE 12.70/11.69, 11.93/10.69, 11.81/10.94.
- **Market at forecast time is already in the product.** Each row carries nflverse's lines at
  composition (`spread_line`, `total_line`, moneylines, `source`, `ingested_at`), so the movement from
  forecast to close needs no new data.
- **The quote ledger is too patchy to be the closing source.** Week 1: 34 snapshots, 16 games; week 2:
  1 snapshot; week 3: 12 snapshots but 1 game (the local manual snapshot was lost, D78); week 4: 17
  snapshots, 15 games. It stays the source for candidate and wager CLV (`market/market_closeout.py`),
  not for the per-game market comparison.
- **Published edges have no immutable record.** `weekly-predict`'s edge stage overwrites
  `data/output/edges/edges_{season}_wk{WW}.csv` and deletes it when blocked (the directory is empty now).
  The only immutable edge evidence is the production chain, run once, for week 1: candidate issuance
  `e945987f` (698 candidates), recommendation evaluation `100a70a8` (698 `unavailable` results) and
  preflight `cf860776`. The cycle's week 4 edge receipts record counts only ("45 edges (top EV 27.1%)" at
  21:00Z; `model_edges_unvalidated` from 22:00Z). So weeks 2-4 read "not recorded", and later weeks read
  "withheld" (D85).
- **Recorded wagers exist.** `bet_ledger.parquet` holds 40 imported DraftKings wagers ($1 stakes):
  week 1, 17 settled; week 2, 15 settled and 1 still `open` after the week finished; 7 without a
  `game_id`. None is linked to a recommendation and none has CLV. `betting/performance.py` (`record`,
  `roi`, `clv_summary`) and `market_closeout.close_recorded_wagers` are the owners to reuse.
- **Starter evidence.** Actual starters are in the raw nflverse games file (`away_qb_id`, `home_qb_id`;
  16 of 16 for weeks 1-4). The digest-verified sidecar copy
  (`data/output/qb_sidecar_captures/sidecar_sha256_ee9c8a73...parquet`, 16 week 4 rows with projected id,
  source and as-of) covers the 9 carried week 4 rows.
- **Evaluation artifacts and the performance page are stale and mostly not served.** `/performance`
  (not in the top nav; reached from the dashboard rail) shows the historical backtest report selected
  2026-08-18 and is framed as betting (net wins, units, ROI). The six-family `GameModelEvaluationReport`
  (2026-09-26: calibration slope and intercept, ECE, sharpness, AUC, season stability, total interval
  coverage, environment slices) is served nowhere. Both predate the 2026-10-01 backfills and champions.
  `/model/performance` runs `summarise` at request time (ROADMAP Tier 8 #22). These belong to the new
  analysis workstream (ROADMAP W22), not to the recap.
- **Storage pattern.** Immutable stores define their own roots (for example
  `evaluation/game_model_evaluation_report_store.py`: JSON manifest plus Parquet frames, frame digests,
  atomic write, replay accepted, a reused identity with different content rejected) and an explicit
  `current.json` selection. Identity either includes the build time (the model report: a rerun is always
  new) or excludes it (recommendation governance: a rerun is an exact replay). The dataset registry
  (`datasets/registry.py`) lists datasets and output directories.
- **Frontend.** Weekly components already carry explicit per-family statuses rendered by
  `field-status/WeeklyComponentValue`; `_meta.field_status` (`pending` or a registered blocker, D14,
  D18) covers fields not built yet. The recap reuses both.

#### Program decisions (taken with the user, 2026-10-04)

1. The game-forecast section comes first, not the starter scorecard: it can be verified now against
   three complete weeks, while the starter section has 6 recorded rows until week 5 finishes.
2. Every week with a selected product gets a recap, whatever its forecast role. The role is recorded per
   week and per row, aggregates include every week, and a live-only subtotal is always available. Nothing
   is dropped; everything is reproducible from the recorded identities.
3. The broader analysis direction is recorded as ROADMAP W22 (ranked candidates with evidence) and is
   not scoped into units here.

#### Shared contracts (every unit of this program)

1. **Persisted state only.** The recap is built by a CLI command; the API only reads and serializes the
   selected artifact (verifying its digests); nothing is scored, recomputed or generated at request time.
   The builder reads only the week's *selected* product, never a non-selected one, and never chooses by
   recency.
2. **Orientation.** One row per game, Away/Home; win means `HOME_WIN`; every differential is home minus
   away.
3. **Closing market (D85's convention).** The closing lines are the cleaned games file's `VEGAS_LINE`
   (expected home margin) and `OVER_UNDER`; the market home win probability is Φ(`VEGAS_LINE` /
   `MARGIN_STD`), importing `MARGIN_STD` from `evaluation/market_agreement.py` rather than copying it.
   The product's own `spread_line`/`total_line` are the market *at forecast time*. Every model metric is
   shown beside the market's.
4. **Identity and replay.** Each artifact's id is the SHA-256 of its canonical content excluding
   `built_at` (the recommendation-governance pattern). Unchanged inputs replay to the same id and the
   stored files are kept byte for byte; a reused id with different content is rejected; changed inputs
   (a stat correction, a newly selected product, a ledger settlement, a builder change) give a new id, and
   old artifacts stay. `current.json` holds one explicit selection per season and week (and, from Unit
   27, per season); the build command selects what it built.
5. **One weekly artifact, growing by section.** A section unit (25, 28) adds its section to the recap,
   bumps the recap schema version, and rebuilds and reselects every finished week (local and
   deterministic, but it writes `data/`, so ask first). The loader accepts only the current schema; no
   compatibility shims (Ways of Working 11).
6. **Explicit status for every value.** Each family, section and side carries a status from a Python
   `StrEnum` that the API exports as a schema enum; the frontend renders every non-available status with
   explicit text through the shared field-status components. A missing piece is a row with a status,
   never a dropped row.
7. **Every week, every role.** A week with a selected product is recapped whatever its forecast role; the
   role is recorded per week and per row; aggregates cover every week and always offer a live-only
   subtotal.
8. **No thresholds without a record.** No pass or fail criterion is introduced unless a DECISIONS entry
   declares it and says whether it was set before or after seeing the data.
9. **Naming.** Code, tests and artifacts use domain names and cite DECISIONS D-numbers, never unit or
   ROADMAP numbers. Unit 23 records the recap's contract as D86; later units add an entry only for a new
   durable decision.

#### Unit sequence

| Unit | Scope | Depends on | Verifiable on |
|---|---|---|---|
| 23 | Artifact, command, game-forecast section | none | weeks 1-3 now |
| 24 | API routes and the `/recap` page | 23 | weeks 1-3 |
| 25 | Starting-QB section | 24 | week 4 after Tuesday 2026-10-06; week 5 after about 2026-10-13 |
| 26 | Weekly-cycle and `post-week` hook | 23 (ideally 25) | dry run; week 5's unattended closeout |
| 27 | Season roll-up | 24 | weeks 1-4 |
| 28 | Edges, recommendations and wagers | 24 (27 for season wager totals) | weeks 1, 2 and 4 |

Unit 26 must land before week 5's closeout window opens (the first 06:00 local at least 4 h after week 5's
last kickoff, about Tuesday 2026-10-13) for week 5 to be recapped by the cycle; otherwise build it by hand.

#### Unit 23 - Weekly Recap Artifact and Game-Forecast Section [Completed October 4, 2026; `DECISIONS.md` D86]

Closed; the completed record is under "Unit 23" below. The scoping record follows.

**Goal.** After a week's results are in, one command builds an immutable recap of every scheduled game's
forecast against the result and against the closing market, from persisted state only, and selects it.
A rerun on unchanged inputs is an exact replay.

**Depends on.** Nothing. Weeks 1-3 are complete and selected.

**Start here.** `evaluation/live_forecast_closeout.py` (the reconciliation this reuses);
`evaluation/game_model_evaluation_report_store.py` and `_selection.py` (store and `current.json`
pattern); `market/recommendation_governance.py` (`recommendation_governance_id`: identity excluding the
creation time); `evaluation/market_agreement.py` (`MARGIN_STD`, `_joined`); `cli/week_cycle.py`
(`week_completion`, `completed_game_ids`, tested in `tests/unit/cli/test_week_cycle.py`);
`datasets/registry.py` and `datasets/loaders.py` (`load_games`, `load_current_weekly_product`);
`evaluation/forecast_store.load_forecast_events`; `cli/evaluate.py` (`model-report` and
`market-agreement` for command shape); HANDOFF "Immutable Weekly Products", "Postgame Workflow",
"Dataset and Artifact Registry"; DECISIONS D24, D77, D83, D85.

**Files.**
- `evaluation/week_schedule.py` (new): `week_completion` and `completed_game_ids` moved from
  `cli/week_cycle.py`, plus the scoped regular-season schedule frame; the CLI imports them; their tests
  move to `tests/unit/evaluation/test_week_schedule.py`.
- `evaluation/weekly_recap.py` (new, pure): the status enums, the manifest and metric dataclasses,
  `build_weekly_recap(...)` returning the manifest and game frame, the identity function and validation.
- `evaluation/weekly_recap_store.py` (new): write, verify and load; `select_current_weekly_recap`,
  `load_current_weekly_recap(season, week)`; root from the registry key `weekly_recaps`.
- `cli/evaluate.py`: the `weekly-recap` command (thin: load inputs, build, write, select, print).
- `datasets/registry.py`: `weekly_recaps` -> `data/output/weekly_recaps`.
- Tests: `tests/unit/evaluation/test_weekly_recap.py`, `test_weekly_recap_store.py`,
  `tests/unit/cli/test_evaluate.py` (command).

**Design.**
1. **Command.** `gridiron evaluate weekly-recap --season S --week W [--dry-run]`. It refuses, naming
   the reason: a week with any unplayed regular-season game (listing them); no selected product; a game
   whose scores differ between the raw and cleaned games files. `--dry-run` builds and prints without
   writing. Output: the week's metrics, model beside market, the not-in-product games, and the previous
   and new selection (new, replayed, or reselected).
2. **Scope is the schedule, not the product.** Rows are the week's regular-season games from the raw
   nflverse file. A scheduled game missing from the selected product is a row with status
   `not_in_product`.
3. **Reuse, do not recompute.** Event matching, outcomes, ties and the forecast values come from
   `close_live_forecasts`' reconciliation frame; the recap joins the schedule, the product's
   market-at-forecast columns and the closing lines onto it. The recap's model metrics over in-product
   games equal the closeout's (asserted in tests). The closeout itself is not changed (see
   "Not in this unit").
4. **Game row** (one per game): identity and kickoff; `product_row_status`; the row's forecast role, run
   id, `generated_at` and whether it is carried (run differs from the product run); per family (win,
   spread, total, projected score) the forecast value and a status (`available`, `forecast_missing`,
   `not_in_product`, and `outcome_tie` for win); market at forecast time (the product's `spread_line` as
   home margin, `total_line`, `ingested_at`); the closing home margin and total and the closing home win
   probability, with `closing_line_missing` when absent; the market-implied team scores ((total ± margin)
   / 2); the outcome (scores, margin, total, home win, null on a tie); and per family the model and market
   errors and which was closer (`model`, `market`, `tie`).
5. **Week metrics**, each with its count, model and market side by side: win Brier, log loss and accuracy;
   spread MAE, RMSE and bias against the closing margin; total MAE, RMSE and bias against the closing total;
   projected team score MAE against the market-implied team scores; and the "closer to the result" counts.
   No pass or fail threshold: a week is 16 games.
6. **Artifact.** `data/output/weekly_recaps/schema=1/recaps/{recap_id}.json` (manifest: season, week,
   product id, run, carried runs, product schema version, roles, digests of the scoped schedule and
   outcome rows, the conventions used, counts, week metrics, the game-frame reference with row count,
   columns and content digest, `built_at`) and `schema=1/games/{recap_id}.parquet`. Identity, replay and
   selection follow shared contract 4.

**Decided at start (2026-10-04, from the audit; the unit had no open items).**
1. Week 2's `development` product has no `spread_line`, `total_line` or `ingested_at`: its rows read
   `not_recorded` for the market at forecast time; the closing comparison is unaffected.
2. A selected component whose forecast event does not match refuses the week instead of reading
   `forecast_missing` (two immutable stores disagree; weeks 1-4 all match).
3. Model and market are each scored over their own available games, with counts, so the model side equals the
   closeout; "closer" counts use games both sides forecast.
4. The projected-score metric is the model's team-score MAE against the actual scores beside the market-implied
   team scores' MAE against the actual scores.
5. The game frame is named by its content digest (`games/{content_digest}.parquet`), not the recap id, since the
   recap id hashes the frame reference. A played game missing from the cleaned file refuses like a score conflict.

**Tests.** Pure builder on synthetic weeks: every family available; a scheduled game not in the product;
`forecast_missing`; a tie; a missing closing line; a carried row; the sign conventions against a hand case
(model spread to home margin, `VEGAS_LINE` as home margin, implied team scores); metrics against hand values;
the market probability. Identity excludes `built_at`; store replay keeps the file byte for byte; a conflict
is rejected; selection round trip. Command refusals (unplayed game, no product, score conflict) and the
dry run, with stubbed loaders. The moved helpers keep their tests.

**Real-data verification (writes `data/output/weekly_recaps/` through the command; ask first).** Build
weeks 1-3; hand-check one game per week (each family, both markets); confirm the model metrics equal the
closeout's (week 1 win Brier 0.2372, week 3 spread MAE 9.18, as read on 2026-10-04); rerun and confirm the
same identity with files unchanged; week 2 shows its `development` role; week 4 is refused while games are
unplayed. After Tuesday's closeout, week 4 builds with PIT @ CLE `not_in_product` and 9 carried rows.

**Not in this unit.** The API and page (Unit 24); starters, edges, wagers and season aggregates (Units
25, 27, 28); the cycle hook (Unit 26). The closeout keeps counting product rows: making it count scheduled
games would fail Tuesday's week 4 closeout (PIT @ CLE) and retry hourly, so it is recorded in ROADMAP W23
instead and the recap reports the gap.

**Acceptance.** Weeks 1-3 build from persisted state; every scheduled game appears with an explicit
status; metrics match the closeout and a hand check; a rerun replays; HANDOFF (a "Weekly Recap" section
and the registry list), DECISIONS D86 (identity, schedule scope, closing-market convention, every role
included), CHANGELOG and this record are updated.

#### Unit 24 - Weekly Recap API and Page [Completed October 4, 2026]

Closed; the completed record is under "Unit 24" below. The scoping record follows.

**Goal.** Serve the selected weekly recaps read-only and show them on one page that answers "how did we do
last week?" for the game forecasts, with every unavailable value explicit.

**Depends on.** Unit 23 committed; recaps for weeks 1-3 selected.

**Start here.** `api/routes/model.py` (`/historical-performance`: a verified persisted report, 404 and 500
mapping) and `api/routes/edges.py`; `api/loaders.py` (`load_current_model_performance_report`);
`api/schemas/_base.py`, `api/meta.py` (`ResponseMeta`, `with_pending`, `Unavailable`), a hand-written
serializer such as `api/serializers/games.py`; `api/app.py` (router registration) and
`tests/unit/api/test_app_routes.py`; `frontend/src/api/hooks.ts`, `frontend/src/Router.tsx`,
`components/chrome/TopNav.tsx`, `context/NavContext.tsx` (route params), `screens/ModelPerformance.tsx`
(a persisted-report screen), `components/field-status/WeeklyComponentValue.tsx` and `PendingField.tsx`,
`frontend/src/design-decisions.md`; HANDOFF "API Serialization Boundary" and "Frontend Contract
Workflow"; DECISIONS D14, D16-D18, D21.

**Design.**
1. `api/routes/recaps.py`: `GET /recaps` lists the selected recaps (season, week, recap id, `built_at`,
   product id, roles, scheduled count, available count per family), verifying each manifest;
   `GET /recaps/{season}/{week}` returns the manifest summary, week metrics (model and market with
   counts) and the game rows. Both read only through `weekly_recap_store`. No selection for the week is a
   404 whose message names the command; a verification failure is a 500 (the `model.py` pattern).
2. `api/schemas/recaps.py` and `api/serializers/recaps.py` (hand-written, D17). Status enums from the
   recap module become schema enums. The response carries `starting_qbs`, `edges`, `wagers` and `season`
   as null fields marked `pending` in `_meta.field_status` (D14) until their units fill them.
3. Frontend: `screens/WeeklyRecap.tsx` at `/recap` (with `season` and `week` params), a "Recap" nav item,
   hooks `useRecaps` and `useWeeklyRecap`. The default week is the highest (season, week) among the
   selected recaps, which is ordered by week, not by build time. Layout: a header (week, product id, a role
   badge that names `development` explicitly, `built_at`, and a one-line statement of the closing-line
   convention); one model-against-market card per family with counts and the "closer" tally; the per-game
   table (matchup, kickoff, result, then per family the model value, market at forecast, closing market,
   errors and which was closer), each non-available value through `WeeklyComponentValue` with explicit
   text (`not_in_product`: "No forecast in the selected product"); each game links to its game page; the
   pending sections render as pending cards. No charts (Unit 27).
4. Regenerate contracts: `uv run gridiron api export-schema`, `cd frontend && pnpm gen:api`, then the
   frontend gate.

**Decided at start (2026-10-04, with the user).**
1. The nav item reads "Last Week" and sits right after "Games"; the route stays `/recap`.
2. No dashboard "last week" card in this unit.

**Tests.** Routes: selected, missing (404), verification failure (500), schema enums, pending meta;
serializer status and null handling; the route registry test. Frontend: the screen renders a recap-shaped
fixture, every status message, the `development` badge, week switching, and the empty state.

**Real-data verification.** `uv run gridiron api serve`; read `/recaps` and `/recaps/2026-2027/3` and
compare three values per family with the artifact; run the frontend (`pnpm dev`, or the run skill) and look
at weeks 1-3 (and week 4 once built), including the `development` badge and any not-in-product row.

**Acceptance.** Both routes serialize only the persisted recap; the page shows every scheduled game and
every metric beside the market with explicit statuses; contracts regenerated, not hand-edited; gates pass;
HANDOFF and CHANGELOG updated.

#### Unit 25 - Starting-QB Section [Scoped October 4, 2026; not active]

**Goal.** Score what the forecast assumed about each starting quarterback against who started, beside
nflverse's pre-game value and carry-forward, inside the recap. This is the former Unit 16 (its design below
is the starting point).

**Depends on.** Unit 24 committed (this unit extends the API and page together); week 4 complete (after
Tuesday 2026-10-06) for the first recorded rows.

**Start here.** The Unit 16 section below; DECISIONS D65, D75, D76, D77; the product's starter columns
(`starting_qb_status`, and per side `qb_id`, `qb_name`, `qb_source`, `qb_projection_as_of`,
`qb_nflverse_id`, `qb_carried_forward_id`, `qb_differs_from_carry_forward`); the raw nflverse games file's
`away_qb_id`/`home_qb_id` for completed games (the source of D75's baseline); `features/team/qb_profile.py`
(`resolve_starting_qb_provenance`); `evaluation/starting_qb_projection.py` (backtest scoring and the
carry-forward baseline); the game API's `starting_qb` block and
`frontend/src/components/games/StartingQuarterbacksCard.tsx` (status texts to reuse);
`data/output/qb_sidecar_captures/` and the prediction-input evidence store (sidecar digest).

**Design.**
1. Per game and side: the assumed id, name, source and as-of; nflverse's id at composition; the
   carry-forward id; the actual starter's id and name; and whether each of the three predictors was right.
   Per-side status: `available`, `not_recorded` (the row carries no record: a schema-1 product, or a row
   whose `starting_qb_status` is `sidecar_unavailable`), `starter_unresolved`, `not_in_product`,
   `actual_missing`.
2. Week counts: slots scored, correct per predictor, and the misses (team, assumed, nflverse,
   carry-forward, actual, source, as-of). Rows without a record are counted and named, not dropped.
3. Recap schema bump and rebuild (shared contract 5); API section replaces the pending field; the page
   gets a starters section with the week's counts and a misses table.
4. No threshold (the Unit 16 design, point 3). The cumulative roll-up is Unit 27's.

**Decide at start.** (a) Week 4's 9 carried rows: score them from the digest-verified sidecar copy, labeled
`sidecar_capture` (recommended: its digest `ee9c8a73...` is the one run `81248588`'s prediction-input
evidence names, so it is the projection that forecast used, and the copy is refused on a mismatch), or
leave them `not_recorded`. (b) Confirm the raw nflverse `*_qb_id` of a completed game as "who started"
(recommended: it is D75's baseline source; spot-check weeks 1-4 against player game logs first).

**Tests.** Synthetic weeks with each predictor right and wrong, a mid-week change, `not_recorded` and
`not_in_product` rows named, a capture digest mismatch refused, the counts.

**Real-data verification.** Weeks 1-3 read `not_recorded` throughout; week 4 has 6 recorded games (plus the
9 carried per decision a) and PIT @ CLE `not_in_product`; reproduce one team-week by hand; rerun replays.
Week 5 (about 2026-10-13) is the first full week; record that check in CHANGELOG when it is done.

**Acceptance.** Starter accuracy for each predictor per week, every slot with a status, misses listed;
gates pass; HANDOFF, CHANGELOG, and DECISIONS if decision (a) adopts the capture.

#### Unit 26 - Build the Recap at Closeout [Completed October 5, 2026]

Closed; the completed record is under "Unit 26" below. The scoping record follows.

**Goal.** The weekly cycle's closeout (and `post-week`) builds and selects the finished week's recap, so no
week depends on someone remembering to run the command.

**Depends on.** Unit 23 committed (ideally Unit 25, so the recap the cycle builds is complete). Due before
week 5's closeout window (about Tuesday 2026-10-13).

**Start here.** `cli/post_week.py` (`_build_stages`), `cli/week_cycle.py` (`build_close_stages`),
`cli/_composites.py` (`soft_fail` semantics, the summary), `weekly_cycle/runner.py` (`closeout`: the receipt's
`stages`, `warnings`, `report`), `weekly_cycle/decision.py` (a week is closed by a `done` closeout receipt);
tests in `tests/unit/cli/test_post_week.py`, `tests/unit/cli/test_week_cycle.py` and
`tests/unit/weekly_cycle/`; HANDOFF "Weekly Cycle" and "Postgame Workflow"; DECISIONS D82; ROADMAP W20 and
W21.

**Design.**
1. A `build-weekly-recap` stage after `close-live-forecasts` in `post-week`'s stages (and so in the
   cycle's close stages), depending on it, calling the same build-write-select function as the command.
2. `soft_fail=True`: a recap failure (including its own refusal of an incomplete week closed with
   `--allow-incomplete`) shows in the stage results, warnings and summary, but the closeout still counts
   as closed and the next week's forecast proceeds.
3. The stage adds "weekly recap <id> (new | replayed)" to the cycle report, so the receipt records which
   recap the closeout selected. `post-week --only build-weekly-recap` rebuilds by hand.
4. Once Unit 27 exists, that unit adds the season roll-up rebuild to the same stage.

**Decide at start.** Soft versus hard failure (recommended: soft, for the reason above); check how
`week-cycle status` shows a soft-failed stage, so a failure is visible (an alert channel is W20, out of
scope).

**Decided at start (2026-10-05, with the user).**
1. Soft failure, as recommended.
2. Visibility: `week-cycle status` printed only a closeout's detail (`week N closed`); soft-failed stages
   were only in the receipt JSON (already true of `check-collector`). A done closeout's detail now names
   them: `week 4 closed; soft-failed: build-weekly-recap`. The recap id goes in the receipt's report.
3. Dependency: a declared `depends_on` would make `post-week --only build-weekly-recap` fail the
   dependency check, and the recap calls `close_live_forecasts` itself rather than reading the stage's
   output. So in `post-week` the stage is ordered after `close-live-forecasts` without declaring it (like
   every `post-week` stage); in the cycle it is gated on `verify-week-data-current` like its siblings. A
   hard failure of `close-live-forecasts` still stops the run before it.
4. If this lands before week 4's closeout, the cycle builds and selects week 4's first recap (schema 1)
   unattended; the user accepted that (Unit 25 rebuilds it at its schema anyway).

**Tests.** Stage order and dependency; a soft failure leaves the closeout `done`; the report line; `--only`;
the runner's receipt carries it (stubbed ports).

**Real-data verification.** `uv run gridiron week-cycle --dry-run`; `post-week --only build-weekly-recap
--week 4 --season 2026-2027` (writes; ask first) replays the recap built by hand (same id). The live check is
week 5's unattended closeout: read its receipt and the selection afterwards.

**Acceptance.** A closeout builds and selects the recap, the receipt names it, a failure is visible without
blocking the cycle; gates pass; HANDOFF "Weekly Cycle" and "Postgame Workflow", CHANGELOG updated.

#### Unit 27 - Season Roll-Up [Scoped October 4, 2026; not active]

**Goal.** Aggregates across weeks beside the per-game table: how the model and the market have done week by
week and season to date, and how well calibrated each is.

**Depends on.** Unit 24 committed; recaps for weeks 1-4 selected. Uses Unit 25's section if present.

**Start here.** Unit 23's `weekly_recap.py` and store; `evaluation/metrics.py` and `evaluation/diagnostics.py`
(existing calibration and ECE helpers: reuse, do not reimplement; grep `calibration`, `ece`);
`evaluation/game_model_evaluation_report.py` (metric names); `evaluation/market_agreement.py`;
`frontend/src/components/model-performance/PerformanceChart.tsx`; load the dataviz skill before any chart.

**Design.**
1. A season artifact under `weekly_recaps/schema=N/seasons/` (manifest plus a per-week frame and a
   calibration frame), built only from the selected weekly recaps' game frames, never from products. Its
   identity covers the season and the exact weekly recap ids it read (shared contract 4); `current.json`
   selects one per season. `gridiron evaluate weekly-recap` rebuilds it after each weekly build, and a
   `--season S --season-only` form rebuilds it alone.
2. Contents: per week the role, counts, model and market metrics and closer counts; season-to-date values
   after each week; season totals with a live-only subtotal; the per-game paired model-minus-market
   difference (Brier, absolute error) with its mean and standard error; model and market reliability over
   fixed probability bins with counts and ECE; finished weeks with no selected recap listed as `not_built`;
   starter accuracy to date when Unit 25's section exists.
3. API `GET /recaps/{season}/summary`; the page's season panel replaces the pending field: per-week model and
   market series (Brier, spread and total MAE), season to date, the reliability plot with bin counts, and a
   live-only toggle.

**Decide at start.** The bins (recommended: ten equal-width probability bins, declared in the unit record
before the data are read); whether to show the standard error as an interval; chart forms (from the dataviz
skill).

**Tests.** Synthetic weeks with hand values for the per-week, cumulative and paired metrics; calibration
counts; the live-only subtotal; identity tracks the weekly recap ids; `not_built` weeks; API and page.

**Real-data verification.** Weeks 1-4: each week's row equals its weekly recap; season to date by hand for
one metric; the live-only subtotal excludes week 2; rerun replays.

**Acceptance.** Season aggregates built from the weekly recaps only, beside the market, with counts; gates
pass; HANDOFF, CHANGELOG, and DECISIONS for the bins if they become a lasting convention.

#### Unit 28 - Edges, Recommendations and Wagers [Scoped October 4, 2026; not active]

**Goal.** Say, for each week, what the system published as edges and recommendations and how they compared
with the closing line and the result, and how the recorded wagers did, without recomputing any edge.

**Depends on.** Unit 24 committed (Unit 27 for season wager totals).

**Start here.** HANDOFF "Production Recommendation Chain" (including "Postgame reassessment") and "Bet Slip
and Betting Ledger Boundaries"; `market/production_chain_preflight.py` (finds the week's issuance by the
selected product id, around line 546, and its evaluations, around lines 1165-1233);
`market/candidate_issuance_store.py`, `market/recommended_bet_result_store.py`,
`api/loaders.load_recommended_bet_results_for_week` (the latest unambiguous evaluation rule);
`market/market_closeout.py` (`close_candidate_issuance`, `close_recorded_wagers`),
`market/candidate_outcome.py`, `market/history_boundaries.py`, `ingest/odds/store.load_odds_ledger`;
`betting/ledger.load_bets`, `betting/performance.py` (`record`, `roi`, `clv_summary`);
`market/edge_validation.py` (D85) and `weekly_cycle/receipts.py` (edge receipts); DECISIONS D27, D59, D85;
ROADMAP W23 items 3 and 5.

**Design.**
1. Edge evidence per week, one status: `production_chain` (an issuance exists for the week's selected
   product), `withheld_unvalidated` (the week's cycle edge receipts record `model_edges_unvalidated`), or
   `not_recorded` (otherwise: the edge CSV is mutable, W23). No edge is recomputed.
2. For a chain week: the issuance, policy and evaluation ids; candidate and not-candidate counts;
   recommendation states by the latest unambiguous evaluation; candidate closeout through
   `close_candidate_issuance` on the week's ledger, giving CLV by family (median and share positive) and
   candidate outcomes (won, lost, push).
3. Wagers: the week's bets by `game_id`: count, staked, P&L, ROI and record (through `betting/performance`);
   an `open` bet on a finished game is `unsettled`; CLV through `close_recorded_wagers` where its reference
   matches, otherwise `clv_unavailable` (the imports carry no reference); the linked recommendation when
   present. Bets without a `game_id` go to the season roll-up as an unattributed count. The scoped ledger
   rows are digested, so a later settlement gives a new recap identity on rebuild.
4. Recap schema bump and rebuild; API sections and page sections replace the pending fields.

**Decide at start.** (a) Whether candidates (which were never recommendations) get a hypothetical result
(recommended: CLV and outcome counts only, no hypothetical P&L). (b) How a week with both published and
withheld edge receipts reads (week 4: "45 edges" at 20:00Z and 21:00Z, withheld from 22:00Z; recommended:
the status of the latest receipt before the week's last kickoff, with the receipt outcomes listed).
(c) Whether the Bankroll screen should link to the recap.

**Tests.** Each edge status; chain counts and closeout on a synthetic issuance and ledger; wager record,
`unsettled`, `clv_unavailable`, unattributed; identity changes when a bet settles.

**Real-data verification.** Week 1: the chain (698 candidates, all `unavailable` recommendations), candidate
CLV from the 34-snapshot ledger, 17 settled wagers. Week 2: 15 settled and 1 `unsettled`; edges
`not_recorded`. Week 4: edge status per decision (b). Hand-check one candidate's CLV and one week's P&L.

**Acceptance.** Every week states what was published (or that it was withheld or not recorded) and how
recorded wagers did, with nothing recomputed; gates pass; HANDOFF, CHANGELOG updated.

### Program: Game-Model Feature Program (ROADMAP Tier 5 #16), Phase A

Active unit: none (the active program is the Weekly Recap, above). Units 12-15 and 17-20 are complete (the program below); Unit 16 is reframed as part of the weekly recap (ROADMAP W15). Unit 11 (weekly readiness coverage counts only scheduled games), Unit 10 (projected starting QB in live serving, `DECISIONS.md` D76), Unit 9 (projected starting-QB backtest), Unit 8 (neutral-site handling in the Logistic Win model, ROADMAP Tier 5 #15a), Unit 7 (screening and promotion), Units 1–5 (batches A1–A5) and the player-log refresh pipeline unit are closed below. Unit 6 (market-aware variant) is deliberately not built.

### Program: Trust in Published Forecasts and Edges (ROADMAP Next Workstreams W1-W3, W14) [In progress, scoped October 3, 2026]

Active unit: none (W16 is done as Unit 21; W15 is the active Weekly Recap program above). Units 12-15 and 17-22 are complete (Unit 14's first live rollover is pending Monday night). The program is a sequence of bounded units, run one
at a time in this order: **Unit 12 (W2, done)**, **Unit 13 (collector pull)**, **Unit 14 (collector rollover)**, **Unit 15 (W1)**, **Unit 16 (W3)**. Evidence for every
statement below was read from the persisted artifacts on 2026-10-03; items marked *unverified*
were not.

**Why this order.** W2 first because every `weekly-predict` run before it ships writes a product
that cannot record its assumed starters, and W3 cannot score a week that has no record (the next
live run is week 5). W1 is independent and could swap with W2; it is second because it needs the
Pi decision and the Pi evidence. W3 last because it needs W2's persisted record and a finished
week to score.

#### Verified state that drives the design

- **Selected week 4 product is `weekly_2026_2027_wk04_81248588-b472-4ccd-9748-5f5525c6e844`**
  (`current.json`, generated 2026-10-03T16:41Z). Five week-4 products are on disk (not four), and
  none is named `25c6e844`; the request's id does not match this checkout.
- **Stale market (W1).** `odds_current.parquet` has one `fetched_at`, 2026-09-30T12:38:29Z, from
  `ingest odds` (3.2 days before the selected product). `evaluate_edge_diagnostics` supports
  `as_of` and `max_market_age`, but all four callers (`weekly-predict` edge stage, `verify-week`,
  `gridiron edges`, the API loader) call `build_weekly_edge_result` without them, so the blocker
  can never fire. `verify-week` prints "Market fetched at 2026-09-30T12:38:29" next to "Ready, No
  blockers"; the frontend Line Shopping screen shows the timestamp with no age or flag; the API
  edge field carries `market_fetched_at` but no age. The week 4 edge result on that snapshot was 337 positive-EV rows
  of 403 eligible, top IND @ WAS moneyline at +40.7% EV (see the fresh-quote result below).
- **Why the Pi evidence is missing here.** `GRIDIRON_COLLECTOR_HOST` is unset, no
  `gridiron-edge-collector-sync` user timer is installed, and the newest local collection receipt
  is 2026-09-14T22:20Z. Local ledger weeks 2, 3 and 4 hold exactly one manual `ingest odds`
  snapshot each; only week 1 has collector data (34 snapshots, 16 games).
- **The Pi is running but collecting nothing for week 4** (read from the Pi by the user on
  2026-10-03, pasted into the session; not independently verified). The collector timer fires every
  5 minutes and each run reports "completed", but `current.json` still selects week 3 (selected
  2026-09-28T01:13Z), only plans for weeks 1 and 3 exist (none for week 2 or week 4), and the Pi's
  newest week 3 receipt is `scheduled_at=2026-09-28T231329Z` (12 runs; week 1 has 34). The Pi's
  `odds_current.parquet` is from 2026-09-28 17:15 MDT, older than this machine's manual
  2026-09-30 snapshot. So no weekly rollover was run for week 4 (`gridiron ops
  rollover-collector-week`), and the runs since 09-28 poll nothing. Consequence for the design:
  a pull alone cannot make week 4's snapshot fresh, and the freshness gate must be what notices
  an idle collector. It also means the dense ledger evidence is one week (week 1) plus a short week
  3, not the several weeks assumed earlier.
- **Pi rollover and pull, 2026-10-03 (run by the user; outputs pasted into the session).** The
  week 4 plan was pushed to the Pi and selected there (`current.json` week 4, selected
  2026-10-03T19:25Z; 28 polls, last 2026-10-05T22:20Z), and the second pull below confirms the Pi
  then collected week 4 (a run receipt scheduled 19:25:59Z and a 19:30:38Z snapshot). The first polls of the plan had already passed. The pull then
  replaced two local files with the Pi's copies: `odds_current.parquet` is now the Pi's week 3
  snapshot (52 rows, 2026-09-28T23:15Z; no week 4 rows), and
  `history/.../week=03/observations.parquet` now holds the Pi's 12 snapshots (636 rows). **The
  local-only 2026-09-22 week 3 manual snapshot (850 rows) is gone and is not recoverable from this
  checkout.** Cause: `build_rsync_command` uses `rsync -az` with no `--ignore-existing`, so the
  documented "one-way and additive only" pull overwrites a same-path file whose local content
  differs (HANDOFF.md says it never deletes; it does not promise to keep local content, but the
  ledger is documented as append-only evidence). The 2026-09-30 week 4 snapshot survives in
  `history/.../week=04`. Until the Pi writes week 4 quotes and they are pulled, week 4 edges are
  blocked as market wrong scope, which is the honest state. Unit 13 fixes the pull so it merges
  append-only ledger files and never replaces local observations.
- **Fresh-quote result, 2026-10-03 (second pull, after the Pi polled week 4).** The Pi polled week 4
  at 2026-10-03T19:30:38Z; `odds_current.parquet` now has 810 rows for 15 games at that time, and
  `verify-week` reads "Market fetched at 2026-10-03T19:30:38" and Ready. **Fresh quotes did not
  remove the Washington edge: IND @ WAS home moneyline is now +48.2% EV (market 33.1% at FanDuel
  +180, model 52.9%), up from +40.7%.** LA @ PHI rose from +33.3% to +46.0%; positive-edge rows
  stayed 337 of 405 eligible. So the stale line was not what produced the Washington edge, and the
  earlier framing of it as a stale-quote artifact is withdrawn: it is a real model-versus-market
  disagreement. W1 is still needed (the 3.2-day-old snapshot was served as current with no flag),
  but the Washington disagreement is a model/input question, and the assumed Washington starter
  (Mariota in place of Daniels, D76) is the first input to examine, which is what Unit 12's record
  makes inspectable. This is not evidence the model is wrong or right: no outcome is known.
  The second pull also replaced the local 2026-09-30 week 4 snapshot file with the Pi's (now one
  snapshot, 19:30Z); that snapshot, the one the selected product's edges had used, is gone too.
- **Unpersisted starter (W2).** The product has 62 columns and none mention quarterbacks. The
  selected product's prediction-input evidence records the QB/coach sidecar's SHA-256 and size
  (`ee9c8a73...`, 151,937 bytes), which still match the file on disk today, plus raw and
  transformed feature values, but no ids or sources, and the digest cannot recover a file once the
  sidecar is rewritten. The depth-chart and injury caches are not in the evidence's nine source
  artifacts. The API schema has no quarterback field. For week 4 the sidecar projects 15 of 16
  games (the played Thursday game has none); in 4 of 30 team slots the live starter differs from
  the last actual starter (WAS, TB, CHI, SEA).
- **A third source exists and nothing records it.** `NFL_upcoming_schedule_nflverse.parquet` now
  carries `away_qb_id`/`home_qb_id` for weeks 4 and 5 (15 of 15 each, 0 for week 6 on), and
  `RESOLVED = projected else forward-fill(raw)` already uses it. `DECISIONS.md` D65 recorded those
  fields as null for every unplayed game on 2026-09-28; the file was refreshed today and its prior
  state is gone. The nflverse value and the projection agree in 28 of 30 week 4 team slots
  (they disagree for SEA and CHI, both home), so the scorecard has a real question to settle. *Unverified*:
  what nflverse's pre-game value is derived from, and when it began appearing.
- **Baseline for scoring.** Carry-forward matched the actual starter in 57 of 64 team-games
  (89.1%) in 2026 weeks 2-3, consistent with D75's 88.4%. A week is 30-32 slots, so one week's
  accuracy has a standard error of roughly 5 points.
- **The week 4 assumption record is perishable.** Week 4's projection exists only in the current
  sidecar, whose digest still matches the selected product's evidence. Any sidecar rebuild after
  week 4 games are played rewrites it. A hash-verified copy is an approved-write step (see open
  questions) and bridges until W2 ships.
- **Product schema is strict.** `_load_indexed_artifact` rejects any artifact whose
  `product_schema_version` is not the single current value, so a bump makes the 16 existing
  products (weeks 1-4, including the Week 1 production-chain inputs) unloadable unless the loader
  accepts both. *Unverified* which consumers of those products would break.

#### Unit 12 - Persist and Show the Assumed Starting QB (W2) [Completed October 3, 2026]

Closed; the completed record is under "Unit 12" below.

#### Unit 13 - Safe and Automatic Collector Pull [Completed October 3, 2026]

Closed; the completed record is under "Unit 13" below.

#### Unit 14 - Automatic Weekly Collector Rollover on the Pi [Completed October 3, 2026]

Closed in code; the completed record is under "Unit 14" below. Deploying it to the Pi is a user step
and is listed there.

#### Unit 15 - Market Freshness (W1) [Completed October 4, 2026]

Closed; the completed record is under "Unit 15" below.

#### Unit 17 - Weekly Cycle Command [Completed October 4, 2026]

Closed; the completed record is under "Unit 17" below.

#### Unit 18 - Pin the Week 1 Preflight Integration Test (ROADMAP W17) [Completed October 4, 2026]

Closed; the completed record is under "Unit 18" below.

#### Unit 19 - Weekly Cycle State Machine and Hourly Timer [Completed October 4, 2026; `DECISIONS.md` D82]

Closed in code; the completed record is under "Unit 19" below. Installing the timer is a user step.

#### Unit 20 - Per-Game Forecast Freeze (first slice of ROADMAP W6) [Completed October 4, 2026; `DECISIONS.md` D83]

Closed; the completed record is under "Unit 20" below.

#### Unit 21 - Market Readiness Over Games That Have Not Started (ROADMAP W16) [Completed October 4, 2026; `DECISIONS.md` D84]

Closed; the completed record is under "Unit 21" below. The scoping record follows.

**Goal.** Judge market coverage only over games that can still be quoted, so a week in progress is not
reported as `partial_market_coverage` because the collector stops quoting started games.

**Audit (read from the code).** The blocker is raised in one place, `weekly_readiness._market_coverage_blockers`
(`market_game_count < scheduled_count`), fed by `evaluate_weekly_readiness`, whose only caller is
`verify_week.load_weekly_readiness` (also used by `weekly-predict` and `week-cycle` reports). No API route or
schema carries it. The pure function receives a schedule of `season/week/game_id` only, so it cannot know
which games started. Prediction coverage must keep counting every scheduled game (a started game keeps its
carried forecast, `DECISIONS.md` D83). Observed 2026-10-04: week 4 reported 6 of 14 games with market data.

**Design.**
1. The readiness schedule gains an optional boolean `started` column, set by `verify_week` from each game's
   kickoff against an `as_of` clock (default now). A game whose kickoff cannot be read counts as not started.
2. Market coverage (games with market data, prediction-market matches, eligible markets, and the coverage
   blockers) is judged over the scheduled games that have not started. `WeeklyReadiness` records
   `unstarted_game_count` so the report can say "6 of 6 not yet started". Quotes for started games are
   ignored, not counted as coverage.
3. When every game has started (a finished week, or Monday night), the unchanged behavior applies: coverage is
   judged over all scheduled games, so an elapsed week with no quotes still reports its missing market.
4. Prediction coverage and all prediction blockers are unchanged. No API, schema or stored-artifact change.

**Tests.** Pure readiness: all unstarted games quoted passes while started games are unquoted; an unquoted
unstarted game still reports partial coverage; quotes on started games do not count; all-started falls back to
all scheduled; no `started` column is the old behavior; counts and invariants. `verify_week` marks started
games from kickoff and `as_of`. Real data: `verify-week` for week 4 against this checkout (read-only).

**Acceptance.** A week in progress with every unstarted game quoted is market ready; one missing quote still
blocks; finished weeks behave as before.

#### Unit 22 - Withhold Unvalidated Model Edges and Measure Market Agreement (ROADMAP W18) [Completed October 4, 2026; `DECISIONS.md` D85]

Closed; the completed record is under "Unit 22" below. The scoping record follows.

**Goal.** Stop presenting raw model-versus-market EV as an edge until the model is shown to beat the market,
keep edges off games that have started, and make the evidence a durable, repeatable command.

**Evidence (2026-10-04, throwaway script, walk-forward backfills 2013-2025 against closing lines).** Win
(Logistic, 3,550 games): Brier 0.2198 against 0.2117 for the market-implied probability; where the model
sat more than 10 points above the market the home team won 45% (model 0.62, market 0.47); more than 10
below, 60% (model 0.46, market 0.61). Total (Random Forest, 3,562 games): MAE 10.76 against 10.51 for the
line; 4-6 points under the line the over hit 49% (model 0.35). Week 4 showed 149 of 162 markets positive EV
with a median of 9-16%: the EV is arithmetically consistent and its input is not predictive.

**Audit (read from the code).** Edge rows come from one seam, `weekly_edge_service.build_weekly_edge_result`
(callers: `edges report`, the `weekly-predict` edge stage, `verify-week`, the API loader). It already returns
a blocked result with a named diagnostic blocker (`market_stale` is the precedent): rows are empty, the API
serializes the blocker and the frontend shows it through `EdgeResultStatus`/`edgeResultStatus.ts`. It does
not exclude games that have started (`commence_time` is carried, never used). Recommendations and Kelly
stakes only exist on edge rows. `model_key` on every row is the game's Win model key, so Total rows are
mislabeled (provenance is correct in `diagnostics.provenance`); left for a later change.

**Design.**
1. `gridiron evaluate market-agreement` (`evaluation/market_agreement.py`): for each backfilled family
   (latest backfill run), over 2013 onward, compare the model with the closing market (the cleaned games
   file's `VEGAS_LINE`, the expected home margin, and `OVER_UNDER`): Brier or MAE against the market, the
   calibration slope of the outcome on the model-minus-market gap with its 95% interval (slope 0: the
   disagreement carries no information; 1: the model is right), and realized rates by gap bin. Writes
   `docs/reports/market-agreement.md`; reads local data only. The validation rule is recorded before it is
   applied: raw model EV is trustworthy only if the slope's 95% interval excludes 0 and includes 1. The
   thresholds above were chosen after seeing the data.
2. `market/edge_validation.py` holds one policy constant, `RAW_MODEL_EDGES_VALIDATED = False`, with the
   reason and the report name. `build_weekly_edge_result` returns a blocked result with a new diagnostic
   blocker `model_edges_unvalidated` (no rows, no recommendations, no stakes) unless the caller passes
   `allow_unvalidated=True`. Only `gridiron edges report --allow-unvalidated` does, and it prints a
   warning; the API loader, the weekly-predict stage and the cycle never do. Flipping the constant needs a
   DECISIONS entry and a passing report.
3. Games that have kicked off are excluded: the service drops quote rows whose `commence_time` is at or
   before `as_of`, so counts, rows and diagnostics cover only games that can still be bet.
4. API: the new blocker value appears in the generated schema (`export-schema`, `gen:api`); the frontend
   adds its message to `edgeResultStatus.ts` and its tests. No other contract change.
5. Not in this unit: a disagreement view without EV, a market-anchored probability (W7), the `model_key`
   label.

**Tests.** The pure agreement tables (slope, intervals, bins, Brier, MAE) on synthetic data with a known
answer; the report text; the command refuses missing data. Edge service: blocked by default with the new
blocker, rows with the override, started games excluded, no change for other blockers. CLI flag and warning.
API serializer and schema enum; frontend message and tests. Real data: run the command (read-only inputs,
writes the report), `edges report` blocked, `verify-week` unchanged.

**Acceptance.** No edge, recommendation or stake is shown for a model that has not passed the recorded
rule, the reason is visible wherever edges would be, started games produce no edges, and the evidence can be
regenerated with one command.

#### Unit 16 - Weekly Starter Projection Scorecard (W3) [Reframed October 4, 2026; becomes Unit 25 of the Weekly Recap program, ROADMAP W15]

The user wants one page for last week's performance covering every projection, not only the starting QB.
The starter scorecard is therefore the quarterback section of that recap and no longer a standalone unit.
Its design below is the starting point for Unit 25. It will be adapted to the recap artifact there: a
section of the weekly recap rather than its own artifact, and the cumulative roll-up moves to Unit 27.

**Goal.** After a week's games, score what the live system assumed against the actual starters,
and persist the result.

**Design.**
1. `gridiron evaluate starter-scorecard --season S --week W` reads the selected product's
   persisted record (Unit 12), the actual starters in `NFL_wk_by_wk_nflverse.parquet`, and scores
   three predictors per team slot: the assumed starter, the nflverse schedule value recorded at
   composition, and carry-forward. Refuses a week with unplayed games or a product without a
   record (`not_recorded`); the Thursday game already played at composition is excluded as "not
   assumed". Week 4 predates the record but its digest-verified sidecar copy
   (`data/output/qb_sidecar_captures/`) carries the same three values and could be scored as a one-off.
2. Output: counts (slots scored, correct per predictor), per-team miss rows (team, assumed,
   nflverse, carry-forward, actual, source, as-of), and a cumulative roll-up across scored weeks.
   Persisted as an immutable, identity-addressed artifact under `data/output/` through the
   dataset registry; a rerun for the same product replays or is rejected, never overwritten.
3. No pass/fail threshold is set now. One week is about 30 slots (standard error roughly 5
   points) and the early-week behavior D75 could not test is exactly what is being observed; any
   decision criterion is declared in `DECISIONS.md` before the roll-up passes a declared sample
   size, and the entry says whether it was set before or after the data.

**Tests.** Synthetic weeks covering each predictor right and wrong, a substitution, unplayed
refusal, missing record refusal, replay and conflict, per-team detail. Real data: score week 5
after it is played.

**Acceptance.** The scorecard reproduces by hand for one team-week; a second run is idempotent;
rows missing a record are named, not dropped; documentation updated.

#### Decisions taken (delegated to the assistant by the user, October 3, 2026)

The user confirmed the Pi collector is running and deferred the design choices. Each is a
recommendation adopted as the plan, to be recorded in `DECISIONS.md` when its unit lands.

1. **Pi sync:** run the weekly `rollover-collector-week` step (the Pi is idle on week 3) and use
   `gridiron ops pull-collector-evidence` as the path from the Pi's ledger to the
   local snapshot; `ingest odds` stays the manual fallback. Needs the Pi's host
   (`GRIDIRON_COLLECTOR_HOST`) and an approved pull before Unit 13's evidence step. Installing the
   sync timer is a machine-configuration change and stays the user's call.
2. **Over-age snapshot:** a blocker for edges, recommendations and Kelly stakes; market readiness
   reports `stale_market_data`; prediction readiness is untouched.
3. **Product schema:** version 2, with the loader accepting exactly {1, 2} and the API reporting
   `not_recorded` for version 1 products.
4. **Week 4 sidecar copy:** worth doing now as a hash-verified copy, but it writes `data/` and
   waits for explicit approval.
5. **Depth-chart and injury cache digests in the evidence:** out of Unit 12's scope. The product
   record plus the sidecar digest bind the starter to the run; widening the evidence schema is a
   separate unit if the projection inputs ever need to be replayed.
6. **Staleness clock:** the server injects the clock at each boundary (including the API
   dependency) to enforce the blocker; the frontend displays age computed from the served
   `market_fetched_at`.

### Unit 26 — Build the Recap at Closeout [Completed October 5, 2026]

#### Completed

`post-week` and the weekly cycle's closeout end with a `build-weekly-recap` stage that builds and selects the
finished week's recap through `evaluation/weekly_recap_build.py`, the same load-build and store-select functions
`gridiron evaluate weekly-recap` now calls, so both share refusals and the new/replayed/reselected rules. The stage
is a soft failure: a refusal returns an unsuccessful result (`recap refused: <reason>`), and any other error is
caught by the composite; either way it is listed as soft-failed with the reason in the warnings, the closeout
receipt is `done`, and the next week's forecast proceeds. On success it adds `weekly recap <id> (new | replayed |
reselected)` to the cycle report, which the closeout receipt stores and `post-week` now prints after its summary.
In `post-week` it declares no dependency (so `--only build-weekly-recap` runs it alone); in the cycle it is gated
on `verify-week-data-current`, after `close-live-forecasts` and before `check-collector`. A cycle run that succeeds
with soft-failed stages names them in its receipt detail (`week 4 closed; soft-failed: build-weekly-recap`), the
line `week-cycle status` prints.

#### Goal

Every finished week gets its recap built and selected by the closeout, so none depends on someone remembering to
run the command, and a recap failure is visible without blocking the cycle.

#### Files Added/Removed/Changed

Added:
- `src/gridiron_edge/evaluation/weekly_recap_build.py` - load the persisted inputs and build a week's recap (with the command's refusals); store and select it, reporting new, replayed or reselected.

Changed:
- `src/gridiron_edge/cli/evaluate.py` - `weekly-recap` calls the shared build and store-select functions.
- `src/gridiron_edge/cli/post_week.py` - the soft `build-weekly-recap` stage; report lines printed after the summary; help text.
- `src/gridiron_edge/cli/week_cycle.py` - the recap stage in the close stages; a successful run's detail names its soft-failed stages.
- `tests/unit/cli/test_post_week.py` - stage list and soft flag; the stage builds, reports and replays; a refusal writes nothing; `--only` by hand; a refusal is a warning with exit 0.
- `tests/unit/cli/test_week_cycle.py` - close-stage order, gate and soft flag; receipt detail with and without soft failures.
- `tests/unit/weekly_cycle/test_weekly_cycle_runner.py` - a soft-failed recap still closes the week and the forecast follows; the receipt records the recap report line.
- `HANDOFF.md` - "Postgame Workflow", "Weekly Recap" and "Weekly Cycle" describe the stage and soft-failure visibility.
- `CHANGELOG.md` - the shipped behavior.
- `ROADMAP.md` - W15 progress.
- `PLAN.md` - this record, the decisions taken at start, and the program status.

Removed:
- None.

#### Tests

Focused: `test_post_week.py`, `test_week_cycle.py`, `tests/unit/weekly_cycle/`, `test_evaluate.py` (125 passed,
the existing `weekly-recap` command tests unchanged). Python gate: ruff check and format, pyrefly 0 errors,
`pytest -m "unit and not slow"` 4,906 passed. Real data, with the recap store checksummed before and after: `week-cycle --dry-run` idle for week 4; `evaluate
weekly-recap --week 3 --dry-run` rebuilds the selected `315255ae...` ("already stored"); week 4 refused on 14
unplayed games; `post-week --only build-weekly-recap` for week 3 replays `315255ae...` (exit 0), and for week 4
soft-fails with the 14 games listed (exit 0); every recap file and `current.json` unchanged. The plan's week 4
replay check and the live check wait for week 4's closeout (from Tuesday 2026-10-06 06:00 local): read its
receipt's report and the selection afterwards.

#### Acceptance

A closeout builds and selects the recap through the same path as the command, the receipt names it, and a failure
shows as soft-failed in the receipt detail, warnings and `status` without blocking the closeout or the next
forecast; gates pass; HANDOFF, CHANGELOG, ROADMAP and this record are updated.

### Unit 24 — Weekly Recap API and Page [Completed October 4, 2026]

#### Completed

`GET /recaps` lists every selected weekly recap ordered by season and week (recap id, `built_at`, `selected_at`,
product, forecast role, distinct row roles, scheduled and in-product counts, available count per family);
`GET /recaps/{season}/{week}` serves the manifest summary, conventions, counts, week metrics (model and market with
counts and closer tallies) and one row per scheduled game, grouped into forecast provenance, result, market at
forecast, closing market and the four families. Both read only through `weekly_recap_store` (identity and frame
digest verified): an unselected week is a 404 naming the build command, a missing or tampered selected recap a 500,
a week below 1 a 422. The recap's status enums and `ForecastRole` are schema enums. `starting_qbs`, `edges`,
`wagers` and `season_to_date` (renamed from the scoped `season`, which would collide with the season string) are
null and `pending` in `_meta.field_status`. The frontend "Last Week" nav item (after "Games") opens `/recap`: a
week picker defaulting to the highest selected (season, week), a header with a role badge that names
`development` explicitly and the closing-line convention, one model-against-market card per family, and a
per-game table with a family toggle (Win, Spread, Total, Projected score) that keeps the table readable; every
non-available value is explicit text, each matchup links to its game page, and the unbuilt sections render as
pending cards. No dashboard card (decided at start).

#### Goal

Serve the selected weekly recaps read-only and show "how did we do last week?" for the game forecasts on one page,
with every unavailable value explicit.

#### Files Added/Removed/Changed

Added:
- `src/gridiron_edge/api/routes/recaps.py` - the two read-only recap routes and their 404/500 mapping.
- `src/gridiron_edge/api/schemas/recaps.py` - the recap list and week response schemas, with status enums.
- `src/gridiron_edge/api/serializers/recaps.py` - hand-written serialization of the stored manifest and game frame, pending sections.
- `tests/unit/api/test_routes_recaps.py` - list, detail, 404, 422, 500 on tampering and missing files, row statuses, pending meta, schema enums.
- `frontend/src/screens/WeeklyRecap.tsx` - the "Last Week" page.
- `frontend/src/screens/WeeklyRecap.test.tsx` - default week, role badge, metric cards, every status message, week switching, game links, pending sections, empty state.
- `frontend/src/components/field-status/recapStatus.ts` - recap status and role messages.

Changed:
- `src/gridiron_edge/evaluation/weekly_recap_store.py` - `list_current_weekly_recap_selections`; each selection checked against its key.
- `src/gridiron_edge/api/app.py` - the `recaps` router and OpenAPI tag.
- `tests/unit/evaluation/test_weekly_recap_store.py` - listing order, empty listing, a mis-keyed selection.
- `tests/unit/api/test_app.py` - the `recaps` tag.
- `api-schema.json` - regenerated (additive: two paths, recap schemas and enums).
- `frontend/src/api/hooks.ts` - `useRecaps`, `useWeeklyRecap`.
- `frontend/src/Router.tsx` - the `/recap` route.
- `frontend/src/components/chrome/TopNav.tsx` - the "Last Week" nav item.
- `frontend/src/App.css` - recap table styles.
- `HANDOFF.md` - serving and page paragraph in "Weekly Recap".
- `CHANGELOG.md` - the shipped behavior.
- `ROADMAP.md` - W15 progress.
- `PLAN.md` - this record and the decisions taken at start.

Removed:
- None.

#### Tests

Python gate: ruff, pyrefly, `pytest -m "unit and not slow"` (4,898 passed; the one failure, the checked-in schema
check, ran before the schema export and passes after it). Frontend gate: lint, build, `test:run` (539 passed).
Real data against the running API: `/recaps` lists weeks 1-3 (week 2 `development`, 16 of 16 available per
family); `/recaps/2026-2027/3` metrics equal the stored manifest exactly (spread MAE 9.18, win Brier 0.2458) and
all 16 games match the stored frame on every family value checked (0 mismatches); week 2 rows read
`not_recorded` for the market at forecast time; week 4 is a 404 naming the command. The page, driven in headless
Chromium against the dev server: "Last Week" active, weeks 1-3 switch, the week 2 development badge, all four family
tables, a game link to `#/games?gameId=2026_01_NE_SEA`, no console errors.

#### Acceptance

Both routes serialize only the persisted, verified recap; the page shows every scheduled game and every metric
beside the market with explicit statuses; contracts were regenerated, not hand-edited; gates pass; HANDOFF and
CHANGELOG are updated. Week 4 appears once its recap is built after Tuesday's results.

### Unit 23 — Weekly Recap Artifact and Game-Forecast Section [Completed October 4, 2026]

#### Completed

`gridiron evaluate weekly-recap --season S --week W [--dry-run]` builds an immutable recap of a finished week
from persisted state only (`DECISIONS.md` D86): one row per regular-season game of the raw schedule (a game the
selected product lacks reads `not_in_product`), each with its forecast role, run and carried flag, and the win
probability, spread, total and projected team scores against the result, the closing market (`VEGAS_LINE`,
`OVER_UNDER`, Phi(`VEGAS_LINE` / 13.5), implied team scores) and the market at forecast time. Every missing value
carries a status (`forecast_missing`, `outcome_tie`, `not_recorded`, `closing_line_missing`). Week metrics put
model beside market with counts, plus "closer to the result" counts; event matching, outcomes and ties come from
`close_live_forecasts`, so model metrics equal the closeout's. The command refuses an unplayed game, a missing
product, a raw/cleaned score disagreement, an unscheduled product game and an unmatched forecast event. The
recap is stored as a JSON manifest plus a content-addressed Parquet game frame under the registered
`weekly_recaps` root; its identity excludes `built_at`, so a rerun replays byte for byte, and `current.json`
selects one recap per season and week. `week_completion` and `completed_game_ids` moved to
`evaluation/week_schedule.py`. Weeks 1-3 are built and selected.

#### Goal

Answer "how did we do last week?" for every game forecast with a durable, reproducible artifact that later units
serve, extend by section and roll up by season.

#### Files Added/Removed/Changed

Added:
- `src/gridiron_edge/evaluation/week_schedule.py` - one week's regular-season schedule from the raw games file, and the completion helpers moved from the cycle CLI.
- `src/gridiron_edge/evaluation/weekly_recap.py` - status enums, manifest and metric dataclasses, the pure builder, identity, validation and JSON round trip.
- `src/gridiron_edge/evaluation/weekly_recap_store.py` - write with replay, verified load, explicit per-week selection.
- `tests/fixtures/weekly_recap.py` - a synthetic finished week covering every status.
- `tests/unit/evaluation/test_week_schedule.py` - the moved helper tests and the scoped schedule.
- `tests/unit/evaluation/test_weekly_recap.py` - statuses, sign conventions, hand-checked metrics, equality with the closeout, identity, refusals.
- `tests/unit/evaluation/test_weekly_recap_store.py` - round trip, byte-for-byte replay, conflicts and tampering, selection.

Changed:
- `src/gridiron_edge/cli/evaluate.py` - the `weekly-recap` command.
- `src/gridiron_edge/cli/week_cycle.py` - imports the moved schedule helpers.
- `src/gridiron_edge/datasets/registry.py` - `weekly_recaps` -> `data/output/weekly_recaps`.
- `tests/unit/cli/test_evaluate.py` - command refusals, dry run, write, replay and reselection.
- `tests/unit/cli/test_week_cycle.py` - the moved helper tests removed.
- `tests/unit/datasets/test_registry.py` - the new registry key.
- `DECISIONS.md` - D86.
- `HANDOFF.md` - "Weekly Recap" section and the registry list.
- `CHANGELOG.md` - the shipped behavior.
- `ROADMAP.md` - W15 progress.
- `PLAN.md` - this record and the decisions taken at start.

Removed:
- None.

#### Tests

Python gate: ruff, pyrefly, `pytest -m "unit and not slow"` (4,885 selected; the two registry assertions that
pinned the key list were updated and pass). Real data: dry runs, then builds of weeks 1-3 (ids `7d7e402d`,
`7a89618f`, `315255ae`); model metrics equal the closeout's (week 1 win Brier 0.2372, week 3 spread MAE 9.18);
one game per week recomputed by hand from the product row and cleaned games file with no mismatch (BAL @ IND,
PHI @ TEN, BAL @ DAL); a rerun of all three replayed with every stored file byte-identical; week 2 reads
`development` with the market at forecast time `not_recorded`; week 4 is refused with its 14 unplayed games. An
in-memory week 4 with placeholder results gave PIT @ CLE `not_in_product` and 9 rows carried from `81248588`.
Week 4's real build waits for Tuesday's results.

#### Acceptance

Weeks 1-3 are built from persisted state and selected, every scheduled game appears with explicit statuses, the
metrics match the closeout and a hand check, a rerun replays exactly, and the contract is recorded in D86 and
HANDOFF for Unit 24 to serve.

### Test Environment Independence from Terminal Color Settings [Completed October 3, 2026]

#### Completed

`tests/conftest.py` removes `FORCE_COLOR`, `PY_COLORS` and `GITHUB_ACTIONS` from the environment and
sets `NO_COLOR=1` before any code under test is imported. Typer reads those variables when
`typer.rich_utils` is first imported and renders colored, differently wrapped help and errors, which
broke eight assertions in `tests/unit/cli` whenever one was set. No source change.

#### Goal

Make the unit gate and the commit hooks give the same result whatever the shell's color settings.

#### Files Added/Removed/Changed

Added:
- None.

Changed:
- `tests/conftest.py` - clears color-forcing variables and sets `NO_COLOR` before imports.
- `ROADMAP.md` - W13 marked done.
- `CHANGELOG.md` - shipped behavior.
- `PLAN.md` - this record.

Removed:
- None.

#### Tests

The eight failing tests reproduced with `FORCE_COLOR=3`, then passed under `FORCE_COLOR=3`, under
`FORCE_COLOR=1 GITHUB_ACTIONS=true PY_COLORS=1`, and with none set (`tests/unit/cli`, 432 passed each).
Python gate; full unit suite run with `FORCE_COLOR=3` set.

#### Acceptance

`uv run pytest -m "unit and not slow"` passes with and without the color variables set, so commits no
longer need `env -u FORCE_COLOR`.

### Unit 22 — Withhold Unvalidated Model Edges and Measure Market Agreement [Completed October 4, 2026]

#### Completed

`gridiron evaluate market-agreement` tests each backfilled Win and Total family against the closing line
(model versus market score, the calibration slope of the outcome on the model-minus-market gap with its 95%
interval, realized outcomes by gap bin) and writes `docs/reports/market-agreement.md`. Run on this checkout:
no family passes the recorded rule (slopes -0.15 to 0.12, every interval includes 0; Elo's is significantly
negative; the market scores better in every family). Accordingly `weekly_edge_service.build_weekly_edge_result`
returns a blocked result with the new diagnostic blocker `model_edges_unvalidated` (no rows,
recommendations or stakes) wherever edges are produced, controlled by
`market/edge_validation.RAW_MODEL_EDGES_VALIDATED = False`; an input blocker takes priority, readiness is
unaffected, and only `edges report --allow-unvalidated` shows the raw rows (with a warning). The service
also ignores quotes for games that have kicked off. The blocker flows through the API meta/route, the
regenerated schema and the frontend message. Not done: a no-EV disagreement view, a market-anchored
probability (ROADMAP W7), and the `model_key` label on Total rows.

#### Goal

Stop presenting model-versus-market EV that the evidence says is noise, keep edges off started games, and
make the evidence repeatable.

#### Files Added/Removed/Changed

Added:
- `src/gridiron_edge/evaluation/market_agreement.py` - agreement tables, calibration slope, Markdown report.
- `src/gridiron_edge/market/edge_validation.py` - the validation switch and its explanation.
- `tests/unit/evaluation/test_market_agreement.py` - slope, bins, scores and report on synthetic data with known answers.
- `docs/reports/market-agreement.md` - generated evidence (a dated record, not a pipeline input).

Changed:
- `src/gridiron_edge/market/weekly_edge_service.py` - started-game quote exclusion and the withholding gate.
- `src/gridiron_edge/market/edge_diagnostics.py` - `model_edges_unvalidated` blocker.
- `src/gridiron_edge/cli/evaluate.py` - the `market-agreement` command.
- `src/gridiron_edge/cli/edges.py` - `--allow-unvalidated`, warning and blocker message.
- `src/gridiron_edge/cli/verify_week.py` - the new blocker maps to no readiness blocker.
- `src/gridiron_edge/api/meta.py`, `src/gridiron_edge/api/routes/edges.py` - field-status reason for the blocker.
- `api-schema.json` - regenerated (blocker enum value).
- `frontend/src/components/field-status/edgeResultStatus.ts` - message for the blocker.
- `frontend/src/components/field-status/edgeResultStatus.test.ts`, `frontend/src/components/dashboard/ModelEdgesTable.test.tsx`, `frontend/src/components/betslip/EdgesTable.test.tsx` - blocker coverage.
- `tests/unit/market/test_weekly_edge_service.py` - gate, override, priority, started-game exclusion.
- `tests/unit/cli/test_edges.py`, `tests/unit/cli/test_edges_report_service.py` - flag and call signature.
- `tests/unit/api/test_edges_route_diagnostics.py` - blocker-to-field-status mapping.
- `tests/integration/api/test_edges_routes.py`, `tests/integration/test_odds_join.py` - edge calculation tests assume validated edges.
- `DECISIONS.md` - D85.
- `HANDOFF.md` - the withheld-edges contract.
- `CHANGELOG.md` - the shipped behavior.
- `ROADMAP.md` - W18 done, W7 note.
- `PLAN.md` - this record.

Removed:
- None.

#### Tests

New unit tests for each case in the design, including that the validator validates a model that is truly
informative (so the gate is not hard-wired). Python gate run; frontend gate run (lint, build, 530 tests). The
edge-related integration tests were run once (452 passed); the full integration and e2e suites were not.
Real data: the command ran against the local backfills and closing lines and wrote the report.

#### Acceptance

No edge, recommendation or stake is shown while no model family passes the recorded rule, the reason is
visible wherever edges would be, started games produce no edges, and the evidence regenerates with one command.

### Unit 21 — Market Readiness Over Games That Have Not Started [Completed October 4, 2026]

#### Completed

`verify-week` judges market coverage over the scheduled games that have not kicked off (`DECISIONS.md` D84).
The readiness schedule carries an optional `started` flag set from each game's kickoff and an `as_of` clock;
games with market data, prediction-market matches, eligible markets and the coverage blockers use only
unstarted games, and quotes for started games do not count. Prediction coverage still counts every scheduled
game. When all games have started the market is judged over all of them. `WeeklyReadiness` gains
`unstarted_game_count` and the report prints "N of M not yet started". Against this checkout, week 4
mid-week moved from `partial_market_coverage` (6 of 14) to ready (2 of 2 unstarted games quoted). Not done:
the edge calculation still includes started games (ROADMAP W18), so the eligible-market and positive-edge
counts shown still span them.

#### Goal

Report a week in progress honestly: coverage of what can still be quoted, not a permanent partial-coverage
blocker caused by games that have started.

#### Files Added/Removed/Changed

Added:
- None.

Changed:
- `src/gridiron_edge/evaluation/weekly_readiness.py` - market scope is the unstarted games; `unstarted_game_count`.
- `src/gridiron_edge/cli/verify_week.py` - marks started games from kickoff and `as_of`; prints the scope.
- `tests/unit/evaluation/test_weekly_readiness.py` - market readiness over unstarted games, fallbacks and invariant.
- `tests/unit/cli/test_verify_week.py` - kickoff marking and report wording.
- `DECISIONS.md` - D84.
- `HANDOFF.md` - readiness contract.
- `CHANGELOG.md` - the shipped behavior.
- `ROADMAP.md` - W16 done; W18 added.
- `PLAN.md` - this record.

Removed:
- None.

#### Tests

Unit tests for each case in the design (see Files). Python gate run. Real data, read-only: `verify-week` for
week 4 at 2026-10-04T20:34Z reported 14 scheduled, 2 not yet started, market data 2 of 2, market ready.
Not run: integration/e2e suites.

#### Acceptance

A week in progress with every unstarted game quoted is market ready; an unquoted unstarted game still
blocks; finished weeks behave as before.

### Unit 20 — A Game Moves From Upcoming to Played Without Losing Its Forecast [Completed October 4, 2026]

#### Completed

A mid-week refresh no longer loses or rewrites a game's forecast (`DECISIONS.md` D83). `weekly-predict`
forecasts only the target week's games that have not kicked off. Composition takes the week's current
product and keeps its row, unchanged, for every game that has started or whose result has landed (so
Thursday's game survives Friday's run), and takes the new run's events for the rest; a started game that
never had a forecast stays `forecast_missing`. The product's `product_run_id` is the composing run and the
earlier runs its rows cite are declared as `carried_run_ids` (index record and write identity); storage and
the production-chain preflight accept only the product run or a declared carried run and reject an unused
declared one. Composition checks the carried events still exist in the forecast store and rejects a `live`
row generated after its game's kickoff. The closeout already scored each row by its own event and run, so it
is unchanged and now covered by a mixed-run test. Rendered outputs come from the persisted product so
carried games stay shown. The Unit 19 watch refreshes at any point in the week. The product schema version
is unchanged and weeks 1-4 products load as before. Not done: market coverage in readiness still counts
started games (ROADMAP W16).

#### Goal

Let a game transition from forecast to played while its last pre-kickoff forecast is preserved, scored and
shown, however late in the week the pipeline runs.

#### Files Added/Removed/Changed

Added:
- None.

Changed:
- `src/gridiron_edge/evaluation/forecast_contracts.py` - `WeeklyProductIdentity.carried_run_ids` with validation.
- `src/gridiron_edge/models/game_prediction/weekly_product_store.py` - carried runs in the index record; rows must cite the product run or a declared carried run; `cited_run_ids`.
- `src/gridiron_edge/models/game_prediction/weekly_starting_qb.py` - `unrecorded_starting_qb_record` for carried schema-1 rows.
- `src/gridiron_edge/cli/_weekly_product_composition.py` - carry rule (`carried_game_ids`), row carrying, carried-event check, pregame guard, carried runs on the identity.
- `src/gridiron_edge/cli/weekly_predict.py` - forecasts only unstarted games, passes the previous product to composition, renders from the persisted product.
- `src/gridiron_edge/market/production_chain_preflight.py` - forecast provenance accepts declared carried runs.
- `src/gridiron_edge/weekly_cycle/runner.py` - the watch refreshes at any point in the week.
- `tests/unit/models/game_prediction/test_weekly_product_store.py` - carried-run storage coverage.
- `tests/unit/market/test_production_chain_preflight.py` - carried-run provenance coverage.
- `tests/unit/cli/test_weekly_product_composition.py` - carry rule, pregame guard, refresh composition coverage.
- `tests/unit/cli/test_weekly_predict.py`, `tests/unit/cli/test_weekly_predict_publication.py`, `tests/unit/cli/test_weekly_predict_product_stage.py` - unstarted-schedule filter and fixtures with kickoff times.
- `tests/unit/evaluation/test_live_forecast_closeout.py` - a refreshed product scores each game on its own run.
- `tests/unit/weekly_cycle/test_weekly_cycle_runner.py` - refresh after the first kickoff.
- `DECISIONS.md` - D83.
- `HANDOFF.md` - weekly-predict and Weekly Cycle contracts.
- `CHANGELOG.md` - the shipped behavior.
- `ROADMAP.md` - W6 first slice done.
- `PLAN.md` - this record.

Removed:
- None.

#### Tests

New unit tests for the carry rule (started, finished and upcoming games), the pregame guard, carried-row
shaping, composition of a refresh with real validation of carried events, storage and preflight acceptance
and rejection of carried runs, the unstarted-schedule filter, closeout over a mixed-run product, and the
watch refreshing after a kickoff. Python gate run. Real data, read-only: all 16 persisted products load
under the new check; week 4's selected product has every live forecast before its game's kickoff; a refresh
at 2026-10-04 22:00Z would keep 13 of its 15 rows. A subset-schedule execution of the real prediction
policy returned the same probabilities as the full-week run for the shared games (home win probability
difference 0.0). Not run: a real `weekly-predict` refresh, integration/e2e suites, frontend gate (no API
contract change).

#### Acceptance

After a refresh, every started or finished game keeps its last pre-kickoff forecast, scored and shown;
unstarted games take the new forecast; no live forecast is generated after its game's kickoff.

### Unit 19 — Weekly Cycle State Machine and Hourly Timer [Completed October 4, 2026]

#### Completed

`gridiron week-cycle` is one idempotent, state-driven command (`DECISIONS.md` D82) that replaces the
`close` and `forecast` subcommands. It derives the target week (`select_target_week`) and does the first step
that is due: close the finished week (from the first 06:00 local after its last game is over, retried hourly
until every result is in and the rebuilt player stats, play-by-play, EPA, game logs and modeling file hold the week's games), forecast the target week when it has no current product, else the starter watch.
The watch refreshes the depth charts and injuries, rebuilds the starter resolution in memory from the cached
inputs, and compares each unstarted game's side with the starters recorded in the current weekly product, at
a cadence set by hours to the next unstarted kickoff (24/12/6/3/1 h). A change refreshes the forecast only
while no game of the week has started and at most once an hour; afterwards it records "starter changed,
refresh waiting". A dirty tracked tree records a blocked receipt and skips the heavy step. Every attempt
writes an immutable receipt under `data/output/week_cycle/season=S/week=WW/` (the receipts are the state
store, and start the forward-timestamped observation ledger with content digests of the upstream files). A
tick also rebuilds the edge report when a newer quote snapshot or product has arrived. `status` and
`--dry-run` read local state only; a lock stops overlapping runs; a user systemd hourly timer template is
provided. The first forecast of a week uses nflverse's published starter or carry-forward when no injury rows
exist yet (no injury-report gate). Limits: the schedule is refreshed at closeout and at each forecast
refresh, not by the watch; a week closed by hand with `post-week` has no closeout receipt, so the cycle
closes it once more when the next week has no forecast; the cadence tiers were chosen before any evidence
about how often nflverse refreshes these files.

#### Goal

Make the weekly workflow run by itself while staying runnable by hand, with every decision recorded.

#### Files Added/Removed/Changed

Added:
- `src/gridiron_edge/weekly_cycle/__init__.py` - package marker.
- `src/gridiron_edge/weekly_cycle/cadence.py` - watch tiers, retry and tick-slack rules, closeout window.
- `src/gridiron_edge/weekly_cycle/receipts.py` - immutable cycle receipts: schema, no-overwrite write, read, lookup.
- `src/gridiron_edge/weekly_cycle/decision.py` - pure decision of which step is due from schedule and receipts.
- `src/gridiron_edge/weekly_cycle/starter_watch.py` - starter comparison, per-team chart evidence, upstream digests.
- `src/gridiron_edge/weekly_cycle/runner.py` - runs the due steps through injectable ports and writes receipts.
- `deploy/systemd/gridiron-edge-week-cycle.service` - one-shot service running `gridiron week-cycle`.
- `deploy/systemd/gridiron-edge-week-cycle.timer` - hourly, persistent timer.
- `tests/unit/weekly_cycle/test_weekly_cycle_decision.py` - cadence and decision coverage.
- `tests/unit/weekly_cycle/test_weekly_cycle_receipts_and_watch.py` - receipt and starter-comparison coverage.
- `tests/unit/weekly_cycle/test_weekly_cycle_runner.py` - state machine against stubbed effects.

Changed:
- `src/gridiron_edge/cli/week_cycle.py` - `close`/`forecast` replaced by `week-cycle`, `--dry-run` and `status`; new `verify-week-data-current` closeout gate; real ports, lock, stage lists reused (injury gate, worktree stage and next-step stage removed).
- `src/gridiron_edge/cli/post_week.py` - closing hint points at `week-cycle`.
- `src/gridiron_edge/transform/clean/qb_coach_sidecar.py` - `build_qb_coach_sidecar_frame` split out so the watch resolves starters without writing.
- `tests/unit/cli/test_week_cycle.py` - stage lists, command, status, lock, override and timer-template coverage.
- `tests/unit/cli/test_post_week.py` - hint expectation.
- `HANDOFF.md` - Weekly Cycle section rewritten, timer install documented.
- `DECISIONS.md` - D82 implementation notes.
- `CHANGELOG.md` - the shipped behavior.
- `ROADMAP.md` - W4 started, Unit 20 sequencing.
- `PLAN.md` - this record.

Removed:
- None.

#### Tests

64 new unit tests plus the reworked CLI tests: tier boundaries, slack, closeout window, each decision branch,
receipt immutability, starter comparison, and the runner for waiting/failed/blocked closeout and forecast,
watch with and without change before and after kickoff, dirty tree, debounce, schema-1 product, fetch failure,
edge refresh, the data-current gate and its coverage helpers. Python gate run (ruff, format, pyrefly, unit suite). Real data, read-only: `status` and
simulated clocks against this checkout (target week, closeout window, season end); the watch observation
against the live caches; the data-current gate against the live artifacts (week 3: 16 of 16 games in all five artifacts, no gaps); a synthetic week 5 product recorded from the stored sidecar compared 30 sides with
0 changes, confirming the in-memory resolution equals the persisted one. Not run: any real `week-cycle`
execution, integration/e2e suites, timer installation.

#### Acceptance

Running `week-cycle` by hand does the right next thing or says why not; closeout, forecast and watch are
state-driven from receipts; every check leaves an immutable receipt; no forecast is rewritten after the
week's first kickoff. Installing and watching the timer is the next user step.

### Unit 18 — Week 1 Preflight Integration Test Independent of Operational State [Completed October 4, 2026]

#### Completed

`test_historical_week_one_evidence_is_classified_truthfully` asserted that the Week 1 `quote_snapshot` held
18 observations at one timestamp. The component is read from the current quote snapshot, which operations
replace as the season progresses (it has held a later week since 2026-09-30), so the test failed on the
untouched `HEAD` and would have blocked `pre-push`. The test now derives its expectation from what the
current snapshot holds for Week 1 at run time: the component must be available with exactly those rows and
fetch times when there are any, and unavailable with "No exact market observation exists." when there are
none. Its assertions about the immutable Week 1 evidence (history, candidate issuance, policy, recommendation
result) are unchanged. No source change.

#### Goal

Keep the real-repository preflight test about how the chain classifies evidence, not about which week the
snapshot happens to hold today.

#### Files Added/Removed/Changed

Added:
- None.

Changed:
- `tests/integration/market/test_production_chain_preflight_repository.py` - the quote-snapshot expectation follows the current snapshot's Week 1 rows instead of a frozen count.
- `ROADMAP.md` - W17 marked done.
- `CHANGELOG.md` - the fix.
- `PLAN.md` - this record.

Removed:
- None.

#### Tests

The test failed on the untouched `HEAD` before the change and passes after it. The full integration suite and
the Python gate were run. Not run: e2e.

#### Acceptance

The test passes whichever week the current snapshot holds, and still fails if the preflight misclassifies
the snapshot (available without matching rows, or unavailable with them).

### Unit 17 — Weekly Cycle Command [Completed October 4, 2026]

#### Completed

`gridiron week-cycle` runs the weekly rhythm as one command with two phases (`DECISIONS.md` D81), composed
from the existing composite stages. `close --season S --week N` (Tuesday) runs `refresh-results`, a new
`verify-week-complete` gate (every scheduled regular-season game of week N has a result, else it stops or
closes anyway with `--allow-incomplete`), the `post-week` stages gated on it, a soft `check-collector` (do
the pulled rollover receipts show the worker rolled to week N+1; how recent is the collector sync) and a
closing hint. `forecast --season S --week M` (Friday) runs `check-clean-worktree`, `check-injury-report`
(refreshes the cache and requires week M reports, else it stops or proceeds with `--allow-early`), the
`weekly-predict` stages and a soft `report-forecast` (starter sources and the games that differ from
carry-forward, prediction and market readiness, market age), then the top-edges preview. The week is never
inferred; `--skip`, `--only` and `--bankroll` behave as in the commands it composes. `weekly_predict_context`
is extracted so both share one stage context, and `post-week`'s hint now points at `week-cycle forecast` and
no longer names `ingest odds`.

#### Goal

Give the weekly rhythm one entry point per day, with the Friday phase waiting for the injury report the
projected starters need, instead of two commands and a hint that had gone stale.

#### Files Added/Removed/Changed

Added:
- `src/gridiron_edge/cli/week_cycle.py` - the `week-cycle` sub-app: the completeness, collector, worktree and injury checks, the forecast summary, and the `close` and `forecast` stage assembly.
- `tests/unit/cli/test_week_cycle.py` - the checks, stage assembly and ordering, and both commands end to end with stubbed stages.

Changed:
- `src/gridiron_edge/cli/main.py` - registers `week-cycle`.
- `src/gridiron_edge/cli/weekly_predict.py` - `weekly_predict_context` extracted for reuse.
- `src/gridiron_edge/cli/post_week.py` - the closing hint.
- `tests/unit/cli/test_post_week.py` - the new hint.
- `DECISIONS.md` - D81.
- `HANDOFF.md` - the weekly cycle section.
- `CHANGELOG.md` - shipped behavior.
- `ROADMAP.md` - W15 note.
- `PLAN.md` - this record.

Removed:
- None.

#### Tests

New unit tests cover the completeness count (regular season only), the injury-report rule (regular season
reports of the week), the git worktree check against a real temporary repository, collector receipts and sync
status in every state, the starter summary with and without the record, `close` and `forecast` stage order,
dependencies and soft stages, the completeness gate and injury check including their override flags, and both
commands end to end with stubbed stages (exit codes, flag passing, `--only`, a bad season). Python gate.
Real data, read-only against this checkout: week 4 reads 1 of 16 games played so `close` would refuse; week 5
has no published injury report so `forecast` would stop; the collector check reports no week 5 rollover
receipt yet and a sync 2 minutes old; the worktree check names the uncommitted files. Not run: a live `close`
or `forecast` (they refresh data from nflverse, write products and select one), which are approved
operational steps.

#### Acceptance

Tuesday, `week-cycle close` leaves the week closed, next-week state refreshed and a clear statement of
whether the collector rolled; Friday, `week-cycle forecast` publishes the forecast or stops with the reason
and prints what was assumed and how fresh the market is. The unit is verified by tests and read-only checks;
the first real runs are Tuesday 2026-10-06 (`close --week 4`) and the Friday injury report
(`forecast --week 5`).

### Unit 15 — Market Freshness [Completed October 4, 2026]

#### Completed

Quote snapshots older than `MAX_MARKET_AGE` (24 hours, provisional; `DECISIONS.md` D80) can no longer
produce edges. Every boundary that computes them (the `weekly-predict` edge stage, `verify-week`,
`gridiron edges report`, the API) now gets the clock and the limit together from `freshness_inputs()`, so
the existing `market_stale` blocker fires: edges, recommendations and Kelly stakes are withheld, market
readiness reports `stale_market_data`, and prediction readiness is untouched (`verify-week` prints both,
the snapshot's age against the limit, and the collector sync's last outcome). The API serves a `freshness`
block with `/edges` and `/lines` (the clock comes from `clock_dependency`); `/lines` keeps showing stale
quotes but withholds price guidance. The game page, edge tables and Line Shopping show a `MarketAsOf`
note with the snapshot time, its age and what is withheld. The limit comes from the new
`gridiron evaluate quote-movement` (`docs/reports/quote-movement.md`): the rule gave 48 hours, borderline
against the 24-48 h bin, so the lower edge was taken; both the rule and the margin were chosen after seeing
the data and the weeks collected from now on are the out-of-sample check. Also fixed: quote timestamps
with and without fractional seconds in one ledger could not be loaded or merged together, and a duplicate
shadowed `_edge_stage_detail` was removed. The rendered forecast PNG and HTML are unchanged.

#### Goal

Make an old market snapshot visible everywhere a quote or an edge appears and impossible to turn into
an edge, with a limit that comes from measured drift rather than a guess.

#### Files Added/Removed/Changed

Added:
- `src/gridiron_edge/market/market_freshness.py` - the one age limit, the clock-and-limit inputs every edge boundary uses, and age formatting.
- `src/gridiron_edge/evaluation/quote_movement.py` - quote drift by age, book disagreement, the age-limit rule and its report.
- `src/gridiron_edge/api/schemas/freshness.py` - the `MarketFreshness` response block.
- `src/gridiron_edge/api/serializers/_freshness.py` - builds that block from the server clock.
- `docs/reports/quote-movement.md` - the generated drift report the limit is derived from.
- `frontend/src/components/field-status/marketAge.ts` - snapshot age against the served limit.
- `frontend/src/components/field-status/MarketAsOf.tsx` - the shared "market as of" note with the stale state.
- `frontend/src/components/field-status/MarketAsOf.test.tsx` - age rules and the three displayed states.
- `tests/unit/market/test_market_freshness.py` - the limit, clock inputs and age labels.
- `tests/unit/evaluation/test_quote_movement.py` - drift, disagreement, the limit rule, borderline handling and the report.
- `tests/unit/api/test_market_freshness_routes.py` - the freshness block, a blocked stale result and guidance withheld for stale lines.

Changed:
- `src/gridiron_edge/cli/weekly_predict.py`, `src/gridiron_edge/cli/verify_week.py`, `src/gridiron_edge/cli/edges.py`, `src/gridiron_edge/api/loaders.py` - supply the clock and limit to the edge service; the edge stage names the newest fetch, `verify-week` and `edges report` show the age, `verify-week` shows both readiness states and the sync status; the shadowed duplicate removed.
- `src/gridiron_edge/api/deps.py`, `src/gridiron_edge/api/routes/edges.py`, `src/gridiron_edge/api/routes/lines.py` - the clock dependency and the stale handling.
- `src/gridiron_edge/api/schemas/edges.py`, `src/gridiron_edge/api/schemas/lines.py`, `src/gridiron_edge/api/serializers/edges.py`, `src/gridiron_edge/api/serializers/lines.py` - the `freshness` field.
- `src/gridiron_edge/cli/evaluate.py` - `evaluate quote-movement`.
- `src/gridiron_edge/ingest/odds/store.py` - ISO 8601 timestamp parsing.
- `api-schema.json` - regenerated by `gridiron api export-schema`.
- `frontend/src/components/betslip/EdgesTable.tsx`, `frontend/src/components/dashboard/ModelEdgesTable.tsx`, `frontend/src/screens/GameDetail.tsx`, `frontend/src/screens/LineShopping.tsx` - show the market age.
- `frontend/src/components/betslip/EdgesTable.test.tsx`, `frontend/src/components/dashboard/FeaturedMatchupsGrid.test.tsx`, `frontend/src/components/dashboard/ModelEdgesTable.test.tsx`, `frontend/src/screens/GameDetail.test.tsx`, `frontend/src/screens/LineShopping.test.tsx` - fixtures carry the freshness block.
- `tests/unit/api/test_loaders.py`, `tests/unit/api/test_schemas_edges.py`, `tests/unit/api/test_schemas_lines.py`, `tests/unit/api/test_serializers_edges.py`, `tests/unit/api/test_serializers_lines.py`, `tests/unit/cli/test_edges.py`, `tests/unit/cli/test_edges_report_service.py`, `tests/unit/cli/test_verify_week.py`, `tests/unit/cli/test_weekly_predict.py`, `tests/unit/ingest/test_odds_store.py` - the clock and limit supplied at each boundary, the age and sync output, the stale stage detail, and the timestamp regression.
- `tests/integration/api/test_edges_routes.py`, `tests/integration/api/test_lines_routes.py` - fixtures pin the API clock an hour after their snapshots.
- `DECISIONS.md` - D80.
- `HANDOFF.md` - market freshness.
- `CHANGELOG.md` - shipped behavior.
- `ROADMAP.md` - W1 status and the new readiness watch item.
- `PLAN.md` - this record and the weekly recap reframing of Unit 16.

Removed:
- None.

#### Tests

New unit tests cover the limit constant and clock inputs, age labels at the boundary, the drift and
disagreement measures, pairs after kickoff ignored, the limit rule, borderline handling, no limit with too
little data, the report, the freshness block and clock passed through the edges route, a stale result
blocked with its own reason, fresh, stale and boundary line responses, the verify-week age and sync lines,
the edge report age line, the stale stage detail, the frontend age rules and states, and two regression
tests that fail without the timestamp fix. Python gate and frontend gate (lint, build, tests);
`export-schema` and `gen:api` run. Real data: `evaluate quote-movement` on the ledger (48 games, weeks 1,
3 and 4) reproduced the table above; `verify-week` for week 4 printed the age and sync status. Integration: `tests/integration/api` passed (173) and the other integration suites passed (81, 1 skipped)
except `test_historical_week_one_evidence_is_classified_truthfully`, which also fails on the untouched `HEAD`:
it reads the live `odds_current.parquet` and expects Week 1's 18 observations, which it has not held since
the 2026-09-30 week 4 snapshot (ROADMAP W17). Not run: e2e, and the browser.

#### Acceptance

No edge, recommendation or Kelly stake is produced from a snapshot over the limit at any boundary; the age
is visible on every surface that shows quotes or edges and a stale state says what is withheld; the limit
and its post-hoc criterion are recorded with the data; the gates pass and the documentation is current.
The limit is provisional until the out-of-sample weeks are checked.

### Unit 14 — Automatic Weekly Collector Rollover on the Pi [Completed October 3, 2026]

#### Completed

`execute-selected-odds-plan --auto-rollover` runs a rollover decision before the poll
(`DECISIONS.md` D79). The target week is the earliest (season, week) in the rich schedule with a game
whose UTC kickoff is still ahead; while the selection equals it nothing happens and there is no network
call. When it differs the schedule is refreshed from nflverse (a failure keeps the existing schedule and
is recorded), the target is recomputed, and the week's existing plan is selected or a default-policy plan
is built from now and selected; a plan that is not `available` is never selected. Failures, invalid
schedules or plans and a completed season are debounced for an hour, a rollover error never stops the
poll, and every due decision writes an immutable receipt under `data/odds/collection_rollovers/`, which
the collector pull merges as immutable evidence. `gridiron ops auto-rollover` reports (default) or applies
the same decision. The installer-rendered wrapper passes the flag. The code was then deployed to the Pi by the user.

#### Goal

Make the Pi select and start collecting each week's plan itself, so a forgotten manual rollover cannot
leave the worker polling a finished week.

#### Files Added/Removed/Changed

Added:
- `src/gridiron_edge/market/collection_rollover.py` - the target-week rule, the rollover decision, receipts and debounce.
- `tests/unit/market/test_collection_rollover.py` - target rule, due and not-due, existing and built plans, refresh failure, season complete, invalid input, debounce, dry run.

Changed:
- `src/gridiron_edge/cli/ingest.py` - `--auto-rollover` on `execute-selected-odds-plan`; failures never stop the poll.
- `src/gridiron_edge/cli/ops.py` - `ops auto-rollover` (dry run by default).
- `src/gridiron_edge/deployment/quote_collection_worker.py` - the rendered wrapper passes `--auto-rollover`.
- `src/gridiron_edge/deployment/collector_merge.py` - rollover receipts are immutable evidence.
- `tests/unit/cli/test_collection_execution_cli.py`, `tests/unit/cli/test_ops_cli.py`, `tests/unit/deployment/test_quote_collection_worker.py`, `tests/unit/deployment/test_collector_merge.py` - the flag and its failure isolation, the ops command, the wrapper, receipt merging.
- `DECISIONS.md` - D79.
- `HANDOFF.md` - the worker's automatic rollover.
- `CHANGELOG.md` - shipped behavior.
- `ROADMAP.md` - W14 progress.
- `PLAN.md` - this record.

Removed:
- None.

#### Tests

New unit tests cover the target rule (a Monday game still ahead, season end, a naive clock rejected), a
not-due tick making no refresh call or receipt, a due rollover that refreshes, builds, selects and writes a
receipt, idempotence, a missing selection, an existing plan selected without being rebuilt, a refresh
failure keeping the old schedule, a refresh that changes kickoffs being reported, season complete and a
missing or invalid schedule debounced for an hour, a dry run writing nothing, the CLI flag with and
without a rollover failure, the ops command, the wrapper, and receipt merging. Python gate. Real data:
dry runs against this checkout's schedule chose week 4 until Monday's kickoff, week 5 afterward, week 12
on 2026-11-24 and season complete on 2027-01-12. Deployed to the Pi and checked there (flag present, dry run, a clean tick). Not verified: a live
rollover (the next boundary is Monday 2026-10-05 after 20:15 ET).

#### Acceptance

Code, tests and documentation are complete. The code is deployed to the Pi (five files copied by `rsync`,
wrapper reinstalled and verified ready): the Pi's wrapper carries `--auto-rollover`, its dry run against its
own schedule returned `not_due` for week 4, and the next five-minute tick completed with no rollover error.
Still pending, and not verified: the first live rollover after Monday 2026-10-05's last kickoff. Confirm then
that the Pi's `current.json` names week 5, a receipt exists under `data/odds/collection_rollovers/`, and the
pulled ledger shows week 5 quotes.

### Unit 13 — Safe and Automatic Collector Pull [Completed October 3, 2026]

#### Completed

`gridiron ops pull-collector-evidence` no longer mirrors the worker's tree over the local one. It
stages the worker's `data/odds` and merges it (`DECISIONS.md` D78): weekly observation partitions are
unioned and never lose a local row, `odds_current.parquet` takes the snapshot with the later
`fetched_at`, collection plans and run receipts are copied only when absent (a differing one is a
reported conflict that fails the pull and leaves the local file), and `collection_plans/current.json`
is never pulled. `ingest odds` and the pull share an exclusive `data/odds` write lock, and each pull
records `data/odds/sync_status.json`. `--unattended --identity` pulls with a dedicated read-only key
without prompting; `gridiron ops collector-key-line` prints the worker's restricted `authorized_keys`
line (`restrict,command="rrsync -ro <repo>/data/odds"`), and the sync service template runs the
unattended pull. The dedicated key `~/.ssh/gridiron_collector_ro` was generated on this machine.

#### Goal

Keep this machine's copy of the Pi's quote ledger and current snapshot current without anyone
running a command, and make it impossible for a pull to destroy local evidence.

#### Files Added/Removed/Changed

Added:
- `src/gridiron_edge/deployment/collector_merge.py` - the per-file-family, evidence-preserving merge of a staged worker tree.
- `tests/unit/deployment/test_collector_merge.py` - merge coverage: unions, never losing rows, snapshots, immutables, pointer, dry run, lock.

Changed:
- `src/gridiron_edge/deployment/collector_sync.py` - staged pull and merge, unattended targets, restricted key line, sync status.
- `src/gridiron_edge/ingest/odds/store.py` - the shared write lock and the observation-partition merge.
- `src/gridiron_edge/ingest/odds/the_odds_api.py` - ingestion writes under the lock.
- `src/gridiron_edge/cli/ops.py` - `--unattended`, merge reporting, `collector-key-line`.
- `deploy/systemd/gridiron-edge-collector-sync.service` - runs the unattended pull with the dedicated key.
- `tests/unit/deployment/test_collector_sync.py`, `tests/unit/cli/test_ops_cli.py` - staged pull, unattended command, key line, status and conflict behavior.
- `DECISIONS.md` - D78.
- `HANDOFF.md` - merge rules, unattended key and timer.
- `CHANGELOG.md` - shipped behavior.
- `ROADMAP.md` - W14 progress.
- `PLAN.md` - this record and the revised proposed program.

Removed:
- None.

#### Tests

New unit tests cover unions with disjoint, overlapping and identical rows, a repeat pull changing
nothing, a conflicting observation leaving the local file untouched, the later-snapshot rule, immutable
copy-if-absent and conflict reporting, the unpulled pointer, dry run, the exclusive lock, the staged
command (default and unattended), key-line validation, and status recording including a failure keeping
the previous success time. Python gate. Real data: an identical staged copy of the real `data/odds`
merged into another copy left every file byte-identical (0 rows added, 0 conflicts). Operational: a dry run and a real unattended pull against the Pi, then the installed user timer, all succeeded.

#### Acceptance

The merge never loses local evidence and is idempotent; an unattended pull with the dedicated read-only
key ran against the Pi without a prompt (a real pull on 2026-10-03 added 1,620 week 4 rows and 4 files, 810
to 2,430 rows, with every other partition unchanged); the installed user timer fired on its own and
recorded a success in `data/odds/sync_status.json`; documentation is current and the Python gate passes.
Not verified: that the key cannot read outside `data/odds` (checked by design of `rrsync -ro`, not by
attempting it).

### Unit 12 — Persist and Show the Assumed Starting QB [Completed October 3, 2026]

#### Completed

Each weekly product now records, per game and side, the assumed starting QB: id, name, source
(`depth_chart_report`, `nflverse_schedule`, `carried_forward`), the depth chart snapshot time, the
nflverse schedule's own value at composition, the carry-forward baseline, whether the assumed
starter differs from it, and a per-game `starting_qb_status` (`available`, `starter_unresolved`,
`sidecar_unavailable`). The record is built by `resolve_starting_qb_provenance`, the same
computation that feeds the QB features, and only from a sidecar whose SHA-256 matches the run's
prediction-input evidence; composition fails if the sidecar changed. Product schema 2 carries the
record; schema 1 products still load and the API serves them as `not_recorded`. The store stamps
version 2 when the record is present and version 1 otherwise, and a version and its columns must
agree. `/games` and `/games/{game_id}` serve a required `starting_qb` block and the game page
shows it, with explicit text for unresolved and not-recorded states (`DECISIONS.md` D77). The
depth chart and injury cache digests stay out of the evidence. No product was regenerated; the
selected week 4 product is version 1 and reads `not_recorded`.

#### Goal

Make what the forecast assumed about each team's starting quarterback persisted, reproducible and
visible, so it can be inspected now and scored against actual starters later.

#### Files Added/Removed/Changed

Added:
- `src/gridiron_edge/models/game_prediction/weekly_starting_qb.py` - the product's starting-QB record: columns, statuses, attachment from the resolver, validation.
- `tests/unit/models/game_prediction/test_weekly_starting_qb.py` - record sources, statuses, validation, schema 1 and 2 storage and mismatch rejection.
- `frontend/src/components/games/StartingQuarterbacksCard.tsx` - game-page card for the persisted starters and their non-available states.
- `frontend/src/components/games/StartingQuarterbacksCard.test.tsx` - card coverage for recorded, not-recorded and unresolved states.

Changed:
- `src/gridiron_edge/features/team/qb_profile.py` - `resolve_starting_qb_provenance` returns source, snapshot time and baselines; `resolve_starting_qb_history` is a view of it.
- `src/gridiron_edge/models/game_prediction/product_validation.py` - validates the starting-QB record when present.
- `src/gridiron_edge/models/game_prediction/weekly_product_store.py` - schema 2, supported versions {1, 2}, version stamped by record presence, version and columns must agree, separate index version.
- `src/gridiron_edge/cli/_weekly_product_composition.py` - evidence-digest-bound sidecar loading, name lookup and record attachment at composition.
- `src/gridiron_edge/api/schemas/games.py`, `src/gridiron_edge/api/serializers/games.py` - the `starting_qb` block, `not_recorded` for schema 1 products.
- `api-schema.json` - regenerated by `gridiron api export-schema`.
- `frontend/src/screens/GameDetail.tsx` - renders the card.
- `frontend/src/components/field-status/weeklyComponentStatus.ts` - status and source copy for starting QBs.
- `tests/unit/api/test_schemas_games.py`, `tests/unit/api/test_serializers_games.py` - block schema and serialization, including `not_recorded`.
- `tests/unit/cli/test_weekly_product_composition.py`, `tests/unit/cli/test_weekly_predict_product_stage.py` - digest binding and the record on the composed product.
- `tests/unit/models/game_prediction/test_weekly_product_store.py` - round trip of a recorded product.
- `frontend/src/components/dashboard/FeaturedMatchupsGrid.test.tsx`, `frontend/src/components/games/gameCardData.test.ts`, `frontend/src/screens/GameDetail.test.tsx`, `frontend/src/screens/GamesList.test.tsx` - fixtures carry the required block.
- `DECISIONS.md` - D77.
- `HANDOFF.md` - the product record and schema versions.
- `CHANGELOG.md` - shipped behavior.
- `ROADMAP.md` - W2 status.
- `PLAN.md` - this record, and the proposed program for Units 13 and 14.

Removed:
- None.

#### Tests

New unit tests cover each source (projected, nflverse, carried forward), unresolved and missing-sidecar
statuses, parity with the feature resolver, validation failures, schema 1 and 2 round trips, version and
column disagreement and unsupported versions, evidence-digest binding, and the API and frontend states.
Python gate: ruff, pyrefly 0 errors, 4,647 unit tests passed. Frontend gate: lint, build, 520 tests passed.
`export-schema` and `gen:api` run. Real data: on the digest-verified copy of the week 4 sidecar all 15
games resolve with names, the record's ids equal the feature resolver's, and four home slots differ from
carry-forward (WAS, TB, CHI, SEA); the selected week 4 product reads `not_recorded`. Not verified: a real
`weekly-predict` composition and the page in a browser.

#### Acceptance

A freshly composed product carries a validated record whose ids equal the features' ids and which cannot
be built from a changed sidecar; version 1 products load unchanged and are reported as not recorded; the API
and page show source, snapshot time and difference or an explicit unavailable state; generated contracts
come from their commands; the gates pass and documentation is current. Regenerating week 4 is a separate
approved step.

### Unit 11 — Weekly Readiness Coverage Counts Only Scheduled Games [Completed October 3, 2026]

#### Completed

`load_weekly_readiness` (`cli/verify_week.py`) no longer passes the edge diagnostics' game counts
(`market_game_count`, `matched_game_count`) into `evaluate_weekly_readiness`, which derives them
from the schedule-scoped markets and predictions it already receives. Edge diagnostics count every
game in the odds snapshot and product, including games already played, so once a week's first game
was played they exceeded the scheduled count and readiness raised `market_game_count must not
exceed scheduled_game_count`, failing every `weekly-predict` rerun after that point. Normal weeks
are unchanged (the derived and diagnostic counts are equal when every market game is scheduled);
`eligible_market_count` and `positive_edge_count` still come from the edge result.

#### Goal

Make verification and `weekly-predict` work for in-week reruns after a game of the week has been
played.

#### Files Added/Removed/Changed

Added:
- None.

Changed:
- `src/gridiron_edge/cli/verify_week.py` - game coverage counts derived from schedule-scoped frames, not edge diagnostics.
- `tests/unit/cli/test_verify_week.py` - regression test with a played game still in the snapshot and product.
- `CHANGELOG.md` - shipped behavior.
- `PLAN.md` - this record.

Removed:
- None.

#### Tests

The regression test (schedule of one game, snapshot and product of two, diagnostics reporting two
of each) fails on the previous code and passes now; the existing readiness, `verify-week` and
`weekly-predict` tests pass (125). Python gate. Real data: `gridiron verify-week --season 2026-2027
--week 4` reported Ready with no blockers (15 scheduled, 15 with market data), where the same
state previously raised.

#### Acceptance

Readiness verification succeeds for a week with a played game still present in the odds snapshot
and product; normal-week output is unchanged. The week 4 rerun is a separate approved operational
step.

### Unit 10 — Projected Starting QB in Live Serving [Completed October 3, 2026]

#### Completed

The live QB features now use a projected starting QB. The sidecar build
(`transform/clean/qb_coach_sidecar.py`) projects each unplayed game's starter from the
cached depth chart and injury reports (the Unit 9 resolver: top quarterback, skipping anyone
reported Out or Doubtful) for the earliest unplayed week that has a published injury report,
and records `AWAY_/HOME_QB_ID_PROJECTED`, `_QB_PROJECTION_SOURCE` and `_QB_PROJECTION_AS_OF`
(the chart snapshot time). `resolve_starting_qb_history` uses a projected id for that game's own
row, keeps later weeks anchored to the last known starter, and derives the change flag from the
resolved ids; played games and training rows are unchanged. Missing caches, no report or no
chart fall back to today's behavior. The `fetch-depth-charts` and `fetch-injuries` stages refresh
both caches ahead of the sidecar build in `run-data-pipeline`, `post-week` and `weekly-predict`;
a failed fetch is reported, not fatal. The resolver moved to `features/team/projected_starter.py`
and the Unit 9 report reproduces exactly. No schema, data-version, model or champion change
(`DECISIONS.md` D76). On the real data (a temporary copy; nothing applied to the real sidecar),
16 week 4 games were projected from the 2026-10-03T12:27Z snapshot and three teams' starters
changed (Chicago, Seattle, Washington). Applying the projection to the live sidecar and
regenerating the week 4 product are separate approved steps.

#### Goal

Make the live QB features describe the quarterback most likely to play, using the Unit 9 resolver
for the current week, without changing training rows, the model, or the behavior for any game
the projection cannot cover.

#### Files Added/Removed/Changed

Added:
- `src/gridiron_edge/features/team/projected_starter.py` - the pure projected-starter resolver (candidate list with its inputs) and the timestamped-chart helpers, shared by serving and the backtest.
- `tests/unit/features/test_projected_starter.py` - resolver and snapshot-helper coverage (moved with the resolver).
- `tests/unit/transform/clean/test_qb_coach_sidecar_projection.py` - projection eligibility and fail-open behavior, sidecar build, and `resolve_starting_qb_history` override semantics.

Changed:
- `src/gridiron_edge/transform/clean/qb_coach_sidecar.py` - projection step and the six projection columns.
- `src/gridiron_edge/features/team/qb_profile.py` - projected id replaces the resolved id for its own game; later weeks stay anchored.
- `src/gridiron_edge/evaluation/starting_qb_projection.py` - imports the shared resolver.
- `src/gridiron_edge/cli/main.py`, `src/gridiron_edge/cli/post_week.py` - `fetch-depth-charts` and `fetch-injuries` stages.
- `tests/unit/features/test_qb_profile.py` - projected-starter parity and change-flag tests.
- `tests/unit/evaluation/test_starting_qb_projection.py`, `tests/unit/cli/test_main.py`, `tests/unit/cli/test_post_week.py`, `tests/unit/cli/test_weekly_predict.py` - resolver tests moved out; stage lists and the new stages covered.
- `DECISIONS.md` - D76.
- `HANDOFF.md` - the caches, the projection and the new stages.
- `CHANGELOG.md` - shipped behavior.
- `ROADMAP.md` - integration status.
- `PLAN.md` - this record.

Removed:
- None.

#### Tests

New unit tests: projection only for the earliest unplayed week with a report, none without a
report, snapshots, injuries or a chart, Out and Doubtful skipped, `AS_OF` recorded; the sidecar
build writes projections for unplayed games only and leaves them null without caches;
`resolve_starting_qb_history` uses a projected id for its own row, keeps later weeks anchored,
leaves played rows and absent-column output identical, and derives the change flag from resolved
ids; a projected quarterback receives that quarterback's own profile features (parity with the
history experiment's feature builder); the new pipeline stages run before the sidecar build,
force-refresh the resolved season, fetch full history under `--all-years`, and tolerate an empty
fetch. Python gate: ruff, pyrefly 0 errors, 4,629 unit tests passed. Real data: the Unit 9 report
reproduced byte for byte in its body after the refactor; on a temporary copy of the data the
rebuilt sidecar's original columns equalled the live sidecar's, 16 week 4 games were projected,
no played game was, and played-game resolved starters were unchanged.

#### Acceptance

For a week with a published report, the live QB features use the projected starter for every
eligible team and today's value for the rest; played-game rows and feature columns are unchanged;
the projection and its snapshot time are in the sidecar; the pipeline refreshes both caches
without failing on a bad fetch; Unit 9's numbers reproduce; documentation is current; the Python
gate passes. Applying it to the live sidecar and week 4 remains an approved operational step.

### Unit 9 — Projected Starting-QB Baseline Backtest [Completed October 3, 2026]

#### Completed

nflverse depth charts (2001 onward) and injury reports (2009 onward) are cached as-is
under `data/raw/depth_charts/` and `data/raw/injuries/` by `gridiron ingest
depth-charts` and `gridiron ingest injuries` (mirroring `ingest player-stats`; a failed
or empty fetch never replaces a cached file; nothing in the weekly pipeline reads them
yet). `gridiron evaluate starting-qb-projection` scores a rules-based projected starter
(depth chart top quarterback, skipping anyone reported Out or Doubtful) against
carry-forward (`DECISIONS.md` D65) on 8,309 team-games, 2009-2024, with a leakage
guard and a 2025 snapshot timing check, and writes
`docs/reports/starting-qb-projection.md`. The resolver is a pure function that returns
the ordered candidate list with its inputs. Result (D75): 92.7% against 88.4% on every
week; after week 1, 92.3% against 90.3% (+1.92 points), 0.08 under the pre-declared 2.0
bar, which was an unprincipled round number chosen after an exploratory run (the claim
that it was not tuned is withdrawn). A downstream test then fed the win model QB
features for the previous starter, the resolver's pick and the actual starter,
walk-forward 2013-2024: the resolver gains +0.00112 Brier [+0.00051, +0.00172] after
week 1 (83% of seasons), capturing about 85% of the +0.00131 that perfect knowledge
would give, which meets the D72 standard. That standard was applied after the first
bar missed. The next unit integrates the projected starter into the QB features.

#### Goal

Establish, from history, whether a projected starting QB built from the depth chart
and weekly availability beats carry-forward, and whether the difference is worth
integrating, before any model, schema or serving change.

#### Files Added/Removed/Changed

Added:
- `src/gridiron_edge/ingest/nflverse/weekly_availability.py` - as-is per-season cache of depth charts and injury reports, with loaders.
- `src/gridiron_edge/evaluation/starting_qb_projection.py` - pure projected-starter resolver, team-game frame, scoring, season-stratified bootstrap, bar and leakage guard, snapshot timing.
- `src/gridiron_edge/evaluation/starting_qb_projection_report.py` - Markdown report builder.
- `docs/reports/starting-qb-projection.md` - dated backtest record (not a pipeline input).
- `tests/unit/ingest/nflverse/test_weekly_availability.py`, `tests/unit/evaluation/test_starting_qb_projection.py`, `tests/unit/cli/test_evaluate_starting_qb_projection.py` - ingestion, resolver, scoring, guard, timing, report and command coverage.

Changed:
- `src/gridiron_edge/cli/ingest.py` - `ingest depth-charts` and `ingest injuries`.
- `src/gridiron_edge/cli/evaluate.py` - `evaluate starting-qb-projection`.
- `DECISIONS.md` - D75.
- `HANDOFF.md` - depth chart and injury caches and the backtest command.
- `CHANGELOG.md` - shipped behavior.
- `ROADMAP.md` - injury audit follow-up, Tier 7 reuse note, first-observed/current odds display idea.
- `PLAN.md` - this record.

Removed:
- None.

#### Tests

New unit tests: ingestion (files written unchanged to the right directory, cached seasons
skipped, failed or empty fetches leave the cache byte-identical, first seasons), resolver
rules (skip Out and Doubtful, Questionable only in the variant, duplicates, ties, empty
chart), scoring, bootstrap reproducibility, the bar and the leakage guard, 2025 snapshot
timing with and without report timestamps, report content, and the command writing only
the report. They found and fixed two edge cases (scoring an empty frame; a report with no
misses). Python gate: ruff, pyrefly 0 errors, 4,605 unit tests passed with color off (eight
existing CLI help-text tests fail when `FORCE_COLOR` is set, independent of this unit).
Real data: the command reproduced the exploratory accuracies (88.4%, 89.8%, 92.7%) and ran on
the cached files; the downstream estimate rebuilt actual-starter features that matched the
modeling file exactly (maximum difference 0.0).

#### Acceptance

The report states each rule's accuracy with intervals by season and era, the error
breakdown, the 2025 timing evidence and the leakage check; the bar was applied as
written and missed narrowly; the downstream test was run and the call recorded in D75
with its post-hoc status; no model, schema, champion or prediction artifact changed;
the only `data/` writes were the approved raw caches.

### Unit 8 — Neutral-Site Handling in the Logistic Win Model [Completed October 1, 2026]

#### Completed

The Logistic Win model no longer applies home-field advantage at neutral-site
games. `IS_NEUTRAL_SITE` is the last column of the Logistic `combined` set (133
columns; the tree `expanded` set keeps 177 without a duplicate). The Logistic
trainer fits on non-neutral games only, so the column is constant zero in the
fit, then pins the neutral coefficient to the negative intercept and fails if the
column was not constant zero. Neutral games stay in the holdout and every metric.
The pinned coefficient is persisted, so log-odds reconstruction and `/explain`
contributions are unchanged in method. Random Forest, XGBoost and Total are
unchanged in behavior (audit evidence in D74). Schema 14 / data 11; modeling
inputs regenerated, game champions retrained and promoted (run
`20261001_133022`: Win Logistic stays champion, holdout Brier 0.2203 against
0.2196; Total XGBoost unchanged), comparable-games corpus rebuilt. The structural
choice rests on the prior that a neutral site carries no home-field advantage:
walk-forward intervals include zero and the 21-game neutral holdout slice scored
worse (-0.021 Brier, interval includes zero; D74). The week 4 product was
regenerated and reselected after the retrain (IND-WAS 0.594 to 0.518); the earlier
products stay on disk.

#### Goal

Remove the phantom home-field term the Logistic Win model applied at neutral
sites (ROADMAP Tier 5 #15a), in training and serving, without changing the method
for non-neutral games.

#### Files Added/Removed/Changed

Added:
- `tests/unit/models/test_logistic_neutral_site.py` - neutral-excluded fit, coefficient pin, reconstruction, fail-closed and tree no-op coverage.

Changed:
- `src/gridiron_edge/models/game_prediction/_columns.py`, `_features.py` - neutral flag in the Logistic `combined` set; de-duplicated tree `expanded` set.
- `src/gridiron_edge/models/game_prediction/base.py` - non-neutral-only Logistic fit and the post-fit neutral coefficient pin in both final-fit paths.
- `src/gridiron_edge/features/manifest.py` - schema 14 / data 11.
- `tests/unit/models/test_game_feature_sets.py`, `tests/unit/evaluation/test_feature_sweep_steps.py` - updated column counts.
- `DECISIONS.md` - D74.
- `HANDOFF.md` - neutral-site handling in the Logistic Win model.
- `CHANGELOG.md` - shipped behavior and retrain result.
- `ROADMAP.md` - #15a status and the neutral-site Known Limitation.
- `PLAN.md` - this record.

Removed:
- None.

#### Tests

Nine unit tests for the trainer and feature sets; full Python gate (ruff, pyrefly
apart from the known untracked-work `gcp/bq.py` error, 4,577 unit tests passed).
Real-data checks: the persisted Logistic artifact has neutral coefficient
-0.3091 equal to the negative intercept, scaler mean 0 and scale 1; holdout
comparison against the schema-13 Logistic by neutral and non-neutral slice;
modeling file at schema 14 / data 11 with unchanged shape (7,324 x 154); corpus
rebuilt over 7,287 games.

#### Acceptance

Neutral-site games get no home-field term in the Logistic Win model; non-neutral
predictions are unchanged in method with a holdout Brier change of -0.0002;
reconstruction and explanation stay exact; schema 14 champions are promoted and
the corpus rebuilt; the week 4 product is regenerated and selected;
documentation is current.

### Unit 7 — Phase A Screening and Promotion [Completed October 1, 2026]

#### Completed

Batches A1–A5 were screened with a read-only harness and only the QB-and-coach
batch (A3) was kept, conditional on a fill. `gridiron evaluate feature-sweep`,
`feature-selection`, `feature-attribution` and `feature-confirmation` score
feature changes walk-forward with paired-bootstrap intervals and persist nothing
under `data/`. The 2026-09-28 holdout sweep (about 800 games) could not resolve
effects of this size and rejected every batch (D71). Model-driven selection
(stage 2) kept 5-24 columns with no placebo survivors but was inferior to the
full set, so pruning was dropped. Per-batch attribution on the wide window
(2013-2025, about 3,300 games, criterion pre-declared before it ran) adopted A3
only if fixed and rejected A1, A2, A4 and A5 (D72). The fill gives a known
starter or first-time head coach with no history a debut prior; an unresolved
identity stays missing and still disables the family (D73). Schema 13 / data 10
forward-removed the rejected batches (combined 132, expanded 177 columns).
Final-composition confirmation beat baseline in all five families on the wide
window with every 90% interval excluding zero. Game champions were retrained and
promoted 2026-10-01 (Win Logistic Brier 0.2196, was 0.2211; Total XGBoost MAE
10.364, was 10.403) and `weekly-predict` for week 4 ran. The default walk-forward
backfill now stops at the last holdout season. Retrain surfaced a pre-existing
neutral-site defect in the Logistic Win model (ROADMAP Tier 5 #15a).

#### Goal

Make the adopt/reject call for the Phase A batches on honest out-of-sample
evidence, keep only what survives, and leave the canonical feature set, champions
and live availability consistent under one forward schema.

#### Files Added/Removed/Changed

Added:
- `src/gridiron_edge/evaluation/feature_sweep_steps.py`, `_rows.py`, `_metrics.py`, `_gates.py`, `_fit.py`, `_run.py`, `_driver.py`, `_report.py`, `_diagnostics.py`, `_ablation.py`, `_columns.py` - sweep harness: step compositions, matched rows, metrics and bootstrap, adopt/reject gates, search and expanding refits, references, diagnostics, ablations, reporting.
- `src/gridiron_edge/evaluation/feature_sweep_selection.py`, `_stage2.py`, `_wide.py` - model-driven selection with placebo control and the wide-window non-inferiority evaluator.
- `src/gridiron_edge/evaluation/feature_sweep_stage3.py` - per-batch attribution on the wide window under the pre-declared criterion.
- `src/gridiron_edge/evaluation/feature_sweep_confirm.py` - confirmation of the final composition against baseline.
- `tests/unit/evaluation/test_feature_sweep_*.py` (steps, rows, metrics, gates, run, driver, report, diagnostics, selection, stage2, stage3, wide, confirm) and `tests/unit/cli/test_evaluate_feature_sweep.py` - harness, leakage-guard and CLI coverage.
- `docs/reports/phase-a-feature-sweep.md`, `phase-a-feature-sweep-predictions.json`, `phase-a-feature-selection.md`, `phase-a-feature-selection-subsets.json`, `phase-a-feature-attribution.md`, `phase-a-final-confirmation.md` - dated screening records (not pipeline inputs).

Changed:
- `src/gridiron_edge/cli/evaluate.py` - the four feature-screening commands.
- `src/gridiron_edge/features/manifest.py`, `features/pipeline.py`, `features/team/epa.py`, `features/team/qb_profile.py`, `features/team/coach_profile.py` - schema 13 / data 10: rejected batches unregistered, debut-prior fill for the QB-and-coach columns.
- `src/gridiron_edge/models/game_prediction/_columns.py`, `_features.py` - canonical column lists and `FeatureSet` composition for schema 13.
- `src/gridiron_edge/evaluation/backfill.py` - default walk-forward end season is the last holdout season.
- `tests/unit/features/test_canonical_feature_sequence.py`, `test_epa.py`, `test_qb_profile.py`, `test_coach_profile.py`, `tests/unit/models/test_game_feature_sets.py`, `tests/unit/evaluation/test_backfill.py` - schema 13 composition, fill and backfill coverage.
- `DECISIONS.md` - D71 (sweep bars and result), D72 (wide-window attribution supersedes the holdout bars), D73 (debut prior; schema 13).
- `FEATURES.md` - recorded batch outcomes and reasons.
- `HANDOFF.md` - "Feature Screening Commands" and the schema 13 fill contract.
- `CHANGELOG.md` - Unit 7 entry (added in the closeout commit).
- `ROADMAP.md` - Unit 7 closed, Tier 5 #15a and the neutral-site Known Limitation (closeout commit).
- `PLAN.md` - this record.

Removed:
- None.

#### Tests

Unit suite for the harness (composition per step, identical matched games,
metric and bootstrap correctness, gate edge cases, no holdout in decision
inputs, no writes outside report paths, CLI wiring); schema 13 composition and
fill tests; Python gate green at each commit (unit suite 4,561 passed after the
schema bump). Real-data checks: the full holdout sweep (5.5 hours) left champion
artifacts hash-identical and wrote nothing under `data/`; modeling inputs
regenerated (7,324 rows, 201 to 154 columns, only previously-missing cells
changed, zero missing values in the 177 declared features from 2023 onward);
live replay of the upcoming schedule had no missing value in week 4 features;
final confirmation on 3,550-3,562 matched wide-window games; retrain,
backfill, calibration and promotion ran at 4 cores and the week 4 product
validated.

#### Acceptance

Each of A1–A5 has a recorded adopt, reject or adopt-only-if-fixed call with
reasons (D71, D72, `FEATURES.md`). The kept batch is wired with its fill under
one forward schema bump (D73), confirmed against baseline, and promoted through
the retrained champions. No batch decision read holdout data. Documentation is
current. The neutral-site defect found at the end is tracked separately as
Tier 5 #15a and is not part of this unit's contract.

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
