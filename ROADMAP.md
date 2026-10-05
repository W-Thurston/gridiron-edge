# Gridiron Edge Roadmap

## Document Ownership

| Document | Purpose |
|---|---|
| `ROADMAP.md` | Strategic priorities, ordered remaining work, genuine future capabilities, and current limitations |
| `PLAN.md` | One active bounded implementation unit and concise completed-unit record |
| `HANDOFF.md` | Current operating system, commands, artifacts, and recovery guidance |
| `DECISIONS.md` | Append-only architectural decisions and supersession history |
| `CHANGELOG.md` | Dated implementation history |

## Project Status and Compatibility Policy

Gridiron Edge has never been live in production. Development-era schemas, artifacts, commands, tests, generated contracts, and historical behavior may be replaced when a cleaner or more correct current design requires it. Backward compatibility is required only when a current contract explicitly requires it.

Development artifacts are not production records by default. Preserve an artifact only when it remains useful as canonical evidence, a regression fixture, or an intentionally supported contract. Otherwise, archive, replace, or regenerate it through the current repository-owned boundaries.

The canonical weekly prediction, product, API, frontend, and verification architecture is implemented. Future work should build on persisted forecast events, explicit weekly-product selection, verified input lineage, strict model availability, and immutable prediction-input evidence rather than restore retired archive, fallback, or request-time computation behavior.

## Current Platform

Gridiron Edge currently provides:

- canonical one-row Away/Home game modeling;
- independent Win and Total model families;
- unversioned champion artifacts and a persisted champion manifest;
- model-specific weekly availability inspection and policy selection;
- history-preserving recurring game refresh;
- deterministic complete-history Elo reconstruction;
- verified Elo lineage before weekly prediction;
- strict statistical metadata preflight for feature-set identity, modeling schema version, and ordered feature columns;
- immutable live and backfilled forecast events;
- immutable prediction-input evidence for newly generated live weekly forecasts;
- immutable schedule-complete weekly products with explicit current selection;
- independent Win, Spread, Total, and projected-score readiness;
- source-neutral current-market storage and explicit edge diagnostics;
- completed-week closeout against exact selected live forecast events;
- archive-driven historical evaluation and champion comparison;
- a read-only serialization API;
- a generated-contract React frontend;
- schedule-first game presentation and truthful missing-data states;
- betting ledger, bankroll, performance, and local BetSlip decision support;
- season and playoff simulation;
- Python and frontend quality boundaries.

The 2026 Week 1 rehearsal produced complete Win, Spread, Total, projected-score, and provenance coverage for all scheduled games while market readiness remained independently blocked. Forecast PNG and HTML publication succeeded without requiring market prices.

## Strategic Principles

Prioritize work by value density, evidence quality, and architectural fit:

1. Treat current repository evidence as authoritative and verify gaps before planning implementation.
2. Prefer a clean current contract over development-era compatibility.
3. Preserve chronology and prevent leakage in every evaluation and feature pipeline.
4. Improve predictive quality only through honest time-ordered evidence.
5. Keep the API a serialization boundary.
6. Keep unavailable, blocked, conflicting, and analytically empty states explicit.
7. Add product surfaces only when the underlying data and artifact contracts are real.
8. Derive thresholds and categories empirically rather than from intuition.
9. Continue using files until concurrency, transactional integrity, or query complexity creates a demonstrated need for a database.
10. Keep calendar-gated operational proof separate from implementation backlog.

## Active Next Program

**Active since 2026-10-04: the Weekly Recap (W15), `PLAN.md` Units 23-28.** The paragraph below records
the feature program's state, which is paused, not closed.

**Previously selected: Tier 5 #16, the game-model feature program, Phase A.** Foundation Completion (below) closed 2026-09-27 with every acceptance criterion met and lifted the gate on Tier 5 #16. Phase A is now sequenced into units below (Tier 5 #16, "Phase A unit sequence"); Units 1–5 (batches A1–A5) are wired (schema v12) with promotion deferred to Unit 7, per the revised wiring/promotion split (`DECISIONS.md` D62). Unit 7 closed 2026-10-01 (only the QB-and-coach batch kept, schema 13, D72/D73). The neutral-site fix (Tier 5 #15a, Unit 8) closed 2026-10-02; Unit 6 (batch A6, market-aware variant) must be deliberately scoped into `PLAN.md` before work begins.

### Next Workstreams (added 2026-10-03)

Candidate work identified after Units 8-11 (neutral-site fix, projected starting QB, readiness fix). Sizes are rough (S, M, L). Items W1-W3 are recommended to be scoped together as one program: they close the trust gap in what is published today and complete the starting-QB loop. Each still needs its own `PLAN.md` unit before code (Ways of Working). Items cross-reference the tiers below that own the same scope.

**Trust in what is published now**

- **W1. Market freshness gate [S]. [Implemented as Unit 15, `DECISIONS.md` D80; the 24 h limit is provisional until more weeks are collected, then rerun `gridiron evaluate quote-movement`.]** The week 4 edge report (2026-10-03) was computed against a quote snapshot fetched 2026-09-30T12:38Z, three days earlier, and nothing flagged it. `edge_diagnostics` already supports a stale-market blocker (`EdgeDiagnosticBlocker.MARKET_STALE`, driven by optional `as_of` and `max_market_age`), but `weekly-predict`'s edge stage and `verify-week` do not pass them. The Raspberry Pi collector's observation ledger is not on the development machine, so how fresh quotes reach the edge stage is undecided. Scope: choose and implement the path from the Pi's ledger (or an explicit `gridiron ingest odds`) to the current snapshot; require a maximum quote age for edges, readiness and the API field status, with the threshold derived from observed line movement, not intuition (Strategic Principle 8); show "market as of" in the edge report, rendered output and frontend. Decide with the user: sync method and what an over-age snapshot does (blocker versus visible warning). Market readiness stays independent of prediction readiness. Owns the stale-quote part of Tier 4 #11-#13.
- **W2. Show the assumed starting QB [S-M]. [Implemented as Unit 12, `DECISIONS.md` D77; week 5 is the first product to carry it.]** Unit 10 (`DECISIONS.md` D76) sets each side's live starter from a projection recorded only in the QB/coach sidecar, which is rewritten on every build, so the assumed starter is not persisted with the forecast and is not shown anywhere. Scope: capture, at product composition, each side's assumed starter id and name, its source (`depth_chart_report`, the nflverse schedule value, or carry-forward), the depth chart snapshot time, and whether it differs from the carried-forward starter; persist it in the immutable weekly product (a product schema version bump, with the API schema regenerated through `gridiron api export-schema` and `pnpm gen:api`); serve it through the read-only API; show it on the game page with the shared field-status components so a missing projection is never hidden. Audit first how much of this prediction-input evidence already records.
- **W3. Weekly projection scorecard [S]. [Done 2026-10-05 as the quarterback section of the weekly recap, W15 / `DECISIONS.md` D87.]** After each week's games, score the resolver against nflverse's schedule value and carry-forward on the real starters, with per-team detail for misses, and persist the result. This validates the live behavior, builds the evidence base for later changes (including the early-week behavior D75 could not test), and feeds the Unit 9 backtest. Depends on W2's persisted record of what was assumed at prediction time (the sidecar cannot supply it).

**Weekly recap**

- **W15. Weekly recap page [L]. [Scoped 2026-10-04 as the active program, `PLAN.md` Units 23-28: artifact and game forecasts, API and page, starting QBs, cycle hook, season roll-up, edges and wagers. The audit there found three finished weeks to verify against (1-3, week 2 `development`), that the closeout is pure but unpersisted, and that published edges have no immutable record. Unit 23 (the recap artifact, `gridiron evaluate weekly-recap` and the game-forecast section, `DECISIONS.md` D86) done 2026-10-04; weeks 1-3 built. Unit 24 (`GET /recaps`, `GET /recaps/{season}/{week}` and the "Last Week" page at `/recap`) done 2026-10-04. Unit 26 (the `post-week` and `week-cycle` closeout build and select the recap, a soft stage) done 2026-10-05, ahead of Unit 25 while week 4 finished. Unit 25 (the starting-QB section, `DECISIONS.md` D87, W3) done 2026-10-05; weeks 1-3 rebuilt at schema 2; week 4 is first scored at its closeout. Unit 27 (the season recap, `GET /recaps/{season}/summary` and the page's season panel, `DECISIONS.md` D88) done 2026-10-05 on weeks 1-3; week 4's closeout rolls it up again. Next: Unit 28 (edges, recommendations and wagers). Follow-ons scoped 2026-10-05: Unit 29 (the cycle retries a soft-failed recap; W21) and Unit 30 (alerts; W20).]** One page that answers "how did we do last week?" for every projection, not only the starting QB: each game's win probability, spread, total and projected score against the result; the assumed starting QBs against who played (W3); the edges and recommendations the system published, with their outcomes and how they compared with the closing lines; and recorded wagers' realized performance where any exist. Aggregates across weeks (calibration, error, running totals) sit beside the per-game table, and unavailable pieces are shown, never hidden. Shape: an immutable weekly recap artifact built by a CLI command after the week's results are in (the postgame workflow already refreshes outcomes and closes the Win and Total events), served read-only through the API, shown on one frontend page using the shared field-status components. Audit first what already exists (`post-week` closeout, model-performance routes and evaluation artifacts, `recommended_bet_results`, the production recommendation chain) so the recap reuses it instead of recomputing; no scoring happens at request time. Needs W2's persisted starter record (done) and a finished week with a recorded product (week 5 is the first). Sequenced after W1; W3 is its first section.

**Analysis, not only betting (added 2026-10-04)**

- **W22. Analysis surfaces [L].** The user wants the frontend to be about analysis as well as betting. The
  recap audit (2026-10-04) found evidence that is computed or persisted but not shown, or shown stale. Ranked
  by value against cost; each needs its own audit and `PLAN.md` unit, and every view serializes persisted
  artifacts only (no request-time scoring):
  1. *Fresh, served model-quality evaluation [M].* The six-family `GameModelEvaluationReport`
     (calibration slope and intercept, ECE, sharpness, AUC, season stability, total interval coverage,
     environment slices) is served nowhere. The `/performance` page shows the historical backtest selected
     2026-08-18, framed as betting (net wins, units, ROI) and absent from the top nav. Both reports predate the
     2026-10-01 backfills and champions. Regenerate them through their owning commands (local compute; ask
     first), serve the quality report, reframe the page around model quality with the naive references of
     Tier 8 #24, and move `/model/performance`'s request-time `summarise` to a persisted artifact (Tier 8 #22).
  2. *Model against market [M].* The market-agreement report (D85: gap bins, slopes, Brier and MAE
     against the closing line) exists only as markdown. Serving its tables answers "where does the model
     disagree with the market, and who was right". It pairs with W19's per-game disagreement view.
  3. *In-week forecast history [M].* Week 4 has six products. How each game's forecast moved over the
     week, and why (a starter change, a refresh), is recorded but not viewable. W6 owns the "as of" contract.
  4. *Starter-watch timeline [S-M].* The cycle's immutable watch receipts record every depth-chart and
     injury check per team with content digests. A built read model would show what changed each week.
  5. *Team efficiency trends [M].* `epa_by_game.parquet` (offense and defense EPA per game) is not served
     by any route. Elo history is served only as current state and deltas (*unverified* in detail).
  6. *Line movement [M].* The quote ledger supports first-observed, current and closing views (Tier 8
     #24, W7's display idea). It is dense only for week 1 and week 4 so far, so it waits for a few weeks of
     the Pi collector.
  7. *Prop model accuracy [M].* Prop projections are logged (`prop_predictions_log.parquet`) and prop
     backfills are evaluated (`evaluation/prop_metrics.py`), but no route serves their accuracy.
  8. *Research evidence pages [S-M].* Feature attribution, the starter-projection backtest and quote
     movement exist as markdown reports. Showing them needs structured artifacts, not markdown parsing.

- **W23. Findings from the recap audit [S].** (1) The closeout (`close_live_forecasts`) counts product
  rows, not scheduled games: week 4 reads 15 games and misses PIT @ CLE, which is in no product from
  `81248588` on. Making it count the schedule would fail that week's closeout, so decide how a closeout
  treats a scheduled game with no selected forecast (an explicit `not_in_product` count that does not fail
  the close is the likely answer). (2) Rows carried from a schema-1 run read
  `starting_qb_status = sidecar_unavailable`; the true state is "not recorded". (3) The weekly edge CSV is
  overwritten and deleted when blocked, so the edges published during a week leave no immutable record.
  Persist them if raw edges are ever re-enabled (D85). (4) The full-retrain report's "Delta vs Previous"
  table shows each full value as its delta when no previous report exists. (5) One week 2 wager is still
  `open` after its game finished (a ledger settlement, the user's step).

**In-week operation (the longer-term goal: forecasts updated continuously as live data arrives)**

- **W14. Automatic collector operation [M].** A manual weekly rollover and pull failed in practice (week 4 never rolled over; the first pull overwrote two local ledger files). Make the pull merge-safe and unattended with a read-only dedicated key, and let the Pi roll its own plan each week. Revisits `DECISIONS.md` D58 (both revisit triggers have fired). Owns the operational half of W1; sequenced before the freshness gate. The merge-safe, unattended pull is implemented (Unit 13, `DECISIONS.md` D78, pending operational setup); the Pi-side automatic rollover is implemented and deployed (Unit 14, `DECISIONS.md` D79; first live rollover pending Monday 2026-10-05 night).
- **W4. Forward-timestamped observation ledger [M]. [Started by the Unit 19 watch receipts (done), which record every depth-chart and injury check with content digests of the upstream files; what remains is reading them back to set the cadence and to evaluate early-week projections.]** nflverse's 2025 and later injury files carry no modification time, and earlier seasons carry a placeholder, so no as-of injury state can be reconstructed. Record our own `observed_at` for each depth chart and injury fetch in an append-only, immutable store. This cannot be recovered afterwards, and it is what allows early-week projections to be evaluated honestly. Related: Tier 6 #18.
- **W5. QB override command [S-M].** An immutable, timestamped manual observation (team, week, quarterback, reason, observer) that the sidecar build consumes ahead of the projection, for evidence no source carries (a coach's hint, who took first-team reps). Writes only through a CLI command, never an API route (`POST /portfolio/bets` stays the only write route). Must keep the resolver pure and record the override as the projection source.
- **W6. Versioned in-week forecasts [L]. [First slice done as Unit 20, `DECISIONS.md` D83: a refresh keeps every started or finished game's forecast; what remains is the explicit "as of" contract, API serving of earlier versions, and a product-history view.]** Each `weekly-predict` rerun now creates another immutable product (week 4 has four). Needs an explicit "as of" contract: what a forecast version is, how the current one is selected without inferring by recency, how the API serves it, and how products, evidence and edges relate. The scenario engine (Tier 7) reuses it.

**Model improvement**

- **W7. Market-aware variant [L].** Tier 5 #16 Unit 6. Closing lines are on disk back to 1999 (moneylines from about 2009); opening lines and movement need The Odds API historical endpoint (paid, from 2020). Display idea: first-observed and current odds per market, from the existing quote ledger, labeled "first observed", not "opening". Its value is anchoring (it reduces false edges); it does not find independent edges. `gridiron evaluate market-agreement` (D85) shows why it matters: no current family's disagreement with the closing market predicts outcomes, so edges stay withheld until a market-anchored probability passes the same rule.
- **W8. Other positions' availability [L].** The Unit 9 backtest method extends to other positions, but the effect size and a replacement-value model are unproven beyond quarterback. Tier 6 #18.
- **W9. Phase B features [L].** Tier 5 #16 Phase B; needs an nflverse play-by-play re-download (ask first).
- **W10. Ensembles [M].** Tier 5 #15; expect gains near 0.002 Brier.

**Watch items and hygiene**

- **W11. Neutral-site choice:** revisit `DECISIONS.md` D74 after weeks 6, 7, 9, 10 and 11 add five more neutral games (the 21-game 2023-2025 neutral holdout slice scored worse, interval including zero).
- **W12. Recommendation-policy maturation:** Tier 4 #13; accumulates weekly.
- **W16. Readiness counts games that have started [S]. [Done as Unit 21, `DECISIONS.md` D84.]** `verify-week` reports `partial_market_coverage` once any game of the week has kicked off, because the collector stops quoting started games (seen 2026-10-04, after IND @ WAS). Like the played-game fix in Unit 11, market coverage should be judged over games that have not started. Not yet scoped.
- **W18. Edges for games that have started [S-M]. [Done as Unit 22, `DECISIONS.md` D85: started games' quotes are ignored and raw model edges are withheld; the `model_key` label and a no-EV disagreement view remain as W19.]** The edge calculation, the edge CSV, the API and the readiness eligible-market and positive-edge counts still include games that have kicked off, priced from their last pre-kickoff quotes (seen 2026-10-04: 149 positive edges across 6 games while 4 of them were under way). Decide whether edges, recommendations and Kelly stakes should be withheld for a started game, and show it with the shared field-status components rather than hiding it. Related: W16, `DECISIONS.md` D84. Separately, audit why 149 of 162 eligible markets were positive EV (median EV 9-16%) before trusting the edge report.
- **W17. Integration test coupled to live quote state [S]. [Done as Unit 18.]** `tests/integration/market/test_production_chain_preflight_repository.py::test_historical_week_one_evidence_is_classified_truthfully` reads the real repository and asserts the Week 1 `quote_snapshot` (18 observations) from `odds_current.parquet`, which now holds the current week; it has failed since the 2026-09-30 week 4 snapshot and fails on the untouched `HEAD` (found 2026-10-04). Pin its evidence to an immutable fixture, or assess only history, so operations cannot break it; `pre-push` runs the integration suites.
- **W19. Show model-versus-market disagreement without EV, and fix the `model_key` label [S-M].** Follow-ups to Unit 22 (`DECISIONS.md` D85). While model edges are withheld the model's view of each game is still worth seeing: show its probability, spread and total beside the market's, labeled "model-market disagreement, historically not predictive" with the shared field-status components, and no EV, edge strength, recommendation or stake. That needs a new API shape (the edge row requires `ev`) and frontend work. Separately, every edge row's `model_key` is the game's Win model key, so Total rows are mislabeled (the diagnostics' provenance is correct); derive the key per market family. Rerun `gridiron evaluate market-agreement` as backfills and forecasts accumulate, and flip `RAW_MODEL_EDGES_VALIDATED` only with a DECISIONS entry and a passing report.
- **W20. Where the unattended weekly cycle runs [S].** The hourly timer (Unit 19, `DECISIONS.md` D82) runs from the main checkout, and a dirty tracked tree blocks the forecast and refresh steps and records a `blocked` receipt (exit 0, shown only by `week-cycle status`). Decide between a dedicated clean "runner" checkout (its own `data/`, kept current from the main checkout, so development never blocks the cycle) and staying on the main checkout, and whether a blocked step should alert somewhere visible instead of only a receipt. Undecided as of 2026-10-04. The alert half is scoped as `PLAN.md` Unit 30 (2026-10-05): alerts on failed, blocked, long-waiting and unretried soft-failed steps, deduplicated, with the channel decided at its start; the runner-checkout half stays open.
- **W21. Weekly-cycle limits found while building it [S].** (1) The starter watch refreshes depth charts and injuries but not the schedule, so a flexed game's kickoff is stale for the cadence and the kickoff-based rules until the next forecast or closeout refreshes it; refresh the upcoming schedule on a watch tick when it is cheap, or on a daily cadence. (2) A week closed by hand with `post-week` leaves no closeout receipt, so the cycle closes it once more when the next week has no forecast; let `post-week` write the same receipt. (3) The cadence tiers (24/12/6/3/1 h) were chosen before any evidence about how often nflverse refreshes these files; tune them from the watch receipts (W4). (4) A soft-failed closeout stage counts the closeout as done and is never retried, so a recap the closeout could not build waits for a manual `post-week --only build-weekly-recap`; scoped as `PLAN.md` Unit 29 (2026-10-05).
- **W13. Test environment fragility [S]. [Done: `tests/conftest.py` clears terminal color variables before anything under test is imported.]**

### Foundation Completion (Tiers 1–4) [Closed 2026-09-27]

**Goal:** resolve the correctness defects and state drift found while reconciling this roadmap against the repository, establish one corrected evaluation and champion-selection foundation, and complete calendar-driven market proof — all before the next feature or evaluation program (Tier 5 #16) begins.

This supersedes the "Repository State and Canonical Artifact Reconciliation" program named in the 2026-09-24 revision; reconciliation is now Unit 1 below rather than a separate program.

Two coordinated tracks. Track A is sequential (each unit depends on the last). Track B runs in parallel on the NFL calendar and does not block Track A.

#### Verified defects (resolve before any evaluation or regeneration runs on top of them)

| # | Defect | Evidence |
|---|---|---|
| D1 | EPA-window train/serve skew: deployed Win models are tuned with EPA rolling windows of 6 (Logistic), 8 (Random Forest), and 6 (XGBoost), but live and backfill feature construction always uses the default window of 4. | Model metadata `epa_window`; `features/team/epa.py:33,362`; `run_features` constructs every feature with no arguments (`features/registry.py:69,87`); `evaluation/backfill.py:401`. |
| D2 | Walk-forward leakage: after tied games are dropped and the index is reset, the walk-forward season lookup is misaligned against the original frame, so roughly the first 14 games of each target season are labeled as the prior season and trained on. | `models/game_prediction/_features.py:169-185` vs. `models/game_prediction/base.py:433` (called with `df_reference=df` at `:815`). |
| D3 | Total models can never be promoted: `gridiron models train` always runs the classification comparison gates against a Total model, which always rejects it. | `cli/models.py:87`; `evaluation/champion.py:72-74,189`. |
| D4 | Holdout is not chronological: deployed models train through the in-progress 2026-2027 season while `HOLDOUT_SEASONS` is fixed at 2023-24 through 2025-26. | `core/constants.py:17`; `models/game_prediction/_features.py:191`. |
| D5 | The selected 2026 Week 2 product was reselected outside any repository-owned command to an untracked `development`-role run; the known-defect disposition no longer names the current selection, so its authentication would now raise and readiness/edge blocking no longer applies. | `data/output/weekly_products/current.json`; `evaluation/forecast_evidence_disposition.py:267-311`. |
| D6 | Calibration and the champion manifest (2026-08-05) predate the current model artifacts (2026-09-21). **Resolved by U4 and U9:** U4 removed the manifest's five orphaned prop entries (no corresponding artifact) via `gridiron evaluate prune-champions`; U9 reassessed every game-model champion against U8's corrected evidence (promoting a retrained `total_random_forest`), refreshed Win calibration from the aligned backfill runs, and re-promoted the manifest (repopulating the five prop families U4 had pruned, now backed by real archives). | `data/output/calibration/`, `data/output/champions/champions.json`. |
| D7 | Weather train/serve skew: training rows use observed OpenWeatherMap backfill; live predictions use stadium/league-month climatology. **Resolved by U8:** the underlying cause was a closed data-completeness gap (the current 2026-2027 season had zero observed weather rows because `fetch_weather` never backfills a missed week), not a live-serving architecture defect; walk-forward evaluation was never exposed to it (D4 already excludes the in-progress season). See D49. | `features/team/weather.py:323`. |
| D8 | Candidate issuance does not reject non-live forecast roles or forecasts generated after `evaluated_at`. **Resolved:** `issue_pregame_candidates` now rejects any non-`live` forecast role and any forecast generated after the issuance's own `evaluated_at` (`DECISIONS.md` D59). | `market/candidate_issuance.py` `_reject_non_live_or_late_events`; `tests/unit/market/test_candidate_issuance_evaluation.py`. |

#### Track A — Tiers 1–3 (sequential) [Complete: U1–U14]

- **Tier 1 #1 — Reconcile roadmap, repository state, and canonical artifacts.**
  Implementation: **U1**, docs only. Record the development forecast role and the exact-run calibration/champion-ranking change in `CHANGELOG.md`/`DECISIONS.md`; record the ad hoc Week 2 reselection; correct stale `HANDOFF.md` text (forecast roles, the metadata-preflight follow-up note, Week 2 blocking language, role-aware postgame closeout, full-retrain description, the `recommended_bet_results` schema path, and Market Unit 26 identities); give every roadmap item its reconciled status.
- **Tier 1 #2 — Decide and implement the canonical Week 2 development state.**
  **Decision: replace** (see below). Implementation: **U6** **[Complete]**.
- **Tier 1 #3 — Audit truncated-history and stale-metadata exposure.**
  Implementation: **U2** (fix D2 walk-forward leakage and D3 Total promotion, with an explicit chronological evaluation-period policy resolving D4; code only, no data changes), **U3** (fix D1 EPA-window parity: carry each artifact's validated `epa_window` into prediction, availability, and backfill feature construction; record the window in the prediction-input evidence feature schema, which bumps its schema version; target completion before the first Track B proof week's forecast), **U4** (apply dispositions to the stale-artifact inventory: archive or delete pre-fix evaluation plots, training logs, and legacy full-retrain reports; remove champion-manifest entries with no artifact; verify the Week 1 percentile rankings were not reset by the truncation).
- **Tier 2 #4 — Generate one clean canonical weekly forecast.**
  Implementation: **U5** (add a repository-owned development-role generation command over the existing development execution wrapper, scoped from retained history so no fetch is needed) **[Complete: `gridiron generate-development-forecast`]** and **U6** (archive the disposition-affected Week 2 live runs and the ad hoc development run through a `DECISIONS.md`-recorded archive boundary, regenerate Week 2 through U5 with the U2/U3 fixes applied, select it, and verify readiness, the API, and the frontend) **[Complete: `gridiron regenerate-development-week`, D45]**.
- **Tier 2 #5 — Rebuild historical game-model evaluation** and **Tier 2 #6 — Evaluate all current game-model families.**
  Implementation: **U7** (add the still-missing metrics — Win calibration slope/intercept, sharpness, season stability; Total median absolute error, interval coverage, environment slices — plus one immutable run-bound evaluation report and CLI over all six families on a common game set, and retire the consumers still reading the legacy prediction archive) **[Complete: `gridiron evaluate model-report`, D46, D47]** and **U8** (regenerate walk-forward backfills for all families on the corrected pipeline, produce the report, and resolve the D7 weather-skew disposition for evaluation purposes) **[Complete: D7 resolved, D48, D49]** - U7 confirmed the six families' current on-disk backfill runs do not share a common game set (Elo's runs cover 7,276 games vs. 6,498 for the other five); U8 restored walk-forward's D1 fixed-hyperparameter contract, added current-model season-bound export alignment, and built the canonical six-family `GameModelEvaluationReport` over a verified 6,232-game common set (seasons 2003-2004 through 2025-2026).
- **Tier 2 #7 — Reassess calibration and champion selection.**
  Implementation: **U9** (promote through the gated comparison using corrected evidence, refresh calibration, promote the manifest, regenerate baseline and per-run performance reports, and verify upcoming-game coverage with the reassessed champions) **[Complete: `total_random_forest` promoted after a genuine gate pass; `win_prob` unchanged after four rejected challengers; calibration and manifest re-promoted from U8's aligned evidence]**.
- **Tier 3 #8 — Persist Logistic explanation evidence.**
  Implementation: **U10** (scaled-feature-by-coefficient contributions in log-odds space, reconciled to the persisted estimator output within tolerance, bound to the exact evidence/event/run identities and model/scaler digests already captured in prediction-input evidence; immutable batch store and CLI) **[Complete: `gridiron evaluate explain-logistic`, D50; found and fixed a prediction-input-evidence scan defect, D51]**, **U11** (change the `/explain` contract from percentage-point deltas to log-odds contributions, regenerate the frontend client, and surface it through field-status) **[Complete: D52; found and fixed an additional gap — the real explanation store already has two batches claiming the same events, so the route treats that as a distinct blocked state (`ambiguous_explanation_evidence`) instead of a 500]**, and **U15** (fix the root cause of U11's real duplicate-batch finding: `batch_id` included `generated_at`, making every rerun a new artifact; excluded it, mirroring D54's identical fix for comparable-games evidence, and disposed of the two real duplicates) **[Complete: D56; all 16 real Week 2 games now serve real `factors` instead of `ambiguous_explanation_evidence`]**.
- **Tier 3 #9 — Evaluate tree-model attribution only if justified.**
  Implementation: **U12** **[Complete: documented decision, `DECISIONS.md` D53]**. Random Forest is unconditionally wrapped in isotonic calibration, so exact attribution against its raw output can never reconstruct the served probability. XGBoost is only conditionally recalibrated per training run based on holdout ECE — the current on-disk artifact is uncalibrated, so its margin-space contributions would be exact today, but that is not a fixed property of the model type. Neither tree type is the current Win champion (`logistic` is, unchanged through U9), so nothing is built; the precondition for building this later (pre-calibration-space attribution plus per-artifact branching on calibration status) is recorded but not resolved.
- **Tier 3 #10 — Build comparable-game retrieval.**
  Implementation: **U13** (backend: a new historical feature-vector corpus for every `win_prob`/`logistic` game, scaled in the champion's own standardized feature space, plus a per-run retrieval-evidence batch matching each event against it by Euclidean distance with an empirically-derived threshold; immutable stores and CLI, no API change) **[Complete: `gridiron evaluate build-comparable-corpus`/`find-comparables`, D54]** and **U14** (wire `/games/{game_id}/comparables` to serve the persisted retrieval evidence, mirroring U11's `/explain` contract change) **[Complete: D55]**.

#### Track B — Tier 4 / Market Unit 26 (parallel, calendar-driven) [Closed 2026-09-27]

M1 through M4 are done. The Raspberry Pi worker's own evidence was inventoried directly over SSH (M1); `gridiron ops pull-collector-evidence` (M2) now syncs its immutable `data/odds` — quote history, collection plans, collection-run claim/result receipts — into this repository, one-way and additive-only, with `gridiron ops rollover-collector-week` handling the paired weekly plan-rollover push. D8 and the real Week 1 triple-candidate-issuance conflict are both resolved (M3): the conflict traced to three content-identical re-evaluations from repeated development runs, of which only one was ever carried through to persisted results — the two orphans and their two orphaned policies were deleted after confirming nothing referenced them, and `issue-candidates`/`evaluate-recommendations` now refuse to create a second immutable artifact for the same exact scope, so it cannot recur. The real 2026 Week 1 weekly proof cadence (M4, Tier 4 #11/#12/#14) is complete: the persisted production-chain checkpoint (`cf860776f0d4bf16804551e4b9ccaa12c91f394d2a9aa1a0dd5877d1bec1eb54`) resolves every component `AVAILABLE` for Moneyline, Spread, and Total except `recorded_wager`/`realized_performance`, both explicitly and validly `UNAVAILABLE` because no wager was recorded. Full detail: `docs/programs/market-unit-26/PLAN.md`.

- **M5 — Tier 4 #13, recommendation-policy maturation.** Still open, on its own calendar, independent of the closed items above: a new design unit defining a threshold-selection method and an explicit sample-size rule, then accumulating evidence weekly. Matured-policy evidence takes most of a season to accumulate; this does not gate Tier 5.

#### Feature-program gate

ROADMAP Tier 5 #16 (the game-model feature program) does not begin until Track A (U1–U14) is complete and Track B has closed Tier 4 #11, #12, and #14. **Both are now done — the gate is lifted.** Tier 4 #13 continues independently on its own calendar and does not gate Tier 5.

#### Acceptance — all met

- [x] every defect in the table above has a resolution recorded in code, tests, or an explicit `DECISIONS.md` entry (D1–D8 all resolved);
- [x] `HANDOFF.md` describes only current behavior;
- [x] the Week 2 decision is executed, not just recorded;
- [x] historical evaluation, calibration, and the champion manifest are regenerated from the corrected pipeline;
- [x] Logistic explanation evidence is persisted and served;
- [x] Market Unit 26 has closed #11, #12, and #14, with #13 continuing on its own calendar;
- [x] `PLAN.md` named exactly one bounded active unit at a time throughout.

## Canonical Artifact Decision: 2026 Week 2 Development State — Resolved and executed (U6)

Two Week 2 Logistic Win forecast runs were generated from an Elo state rebuilt from only the 16 completed Week 1 games. A schema-1 disposition recorded the affected events and products as known defective; the current Week 2 selection had since drifted to an untracked `development`-role run outside that disposition's scope (see D5 above).

**Decision: replace (executed).** The defective events, affected products, and the disposition itself remain physically unchanged and permanently on disk — none of the three has a delete path, and operational guidance already forbids editing or deleting disposition-governed evidence, so "archive" is the `DECISIONS.md`-recorded boundary D45, not a store change. The ad hoc selection was retired by ordinary explicit reselection to a freshly generated, coherent Week 2 development fixture, produced through the current history-preserving, lineage-verified, metadata-strict, evidence-aware weekly path (Unit 5's development-generation command, applied after Units 2–3 fix D1–D4) and composed/selected through Unit 6's new `gridiron regenerate-development-week` command. API and frontend serialization was verified against the corrected selected product.

The disposition implementation and its tests remain a supported capability even though the specific Week 2 operational artifact it originally named is retired — a future known-defect event would still need to be governed the same way.

## Ordered Remaining Work

### Tier 1: Repository and Artifact Integrity

#### 1. Reconcile roadmap, repository state, and canonical artifacts

Complete the active reconciliation program above and produce the authoritative classification of shipped, stale, gated, partial, and missing capabilities.

#### 2. Decide and implement the canonical Week 2 development state

Execute the retain-or-replace decision. If replacing the state, use repository-owned loaders, stores, and generation boundaries rather than hand-editing artifacts.

#### 3. Audit truncated-history and stale-metadata exposure

Classify modeling inputs, historical evaluations, backfills, forecast closeout, production preflight, API loaders, player game-context consumers, and other derived artifacts by whether they were created while canonical game history was truncated or model metadata was stale.

For each affected artifact, choose one disposition:

- verified current;
- regenerate;
- replace;
- archive;
- delete;
- retain only as an explicit regression fixture.

Because compatibility is not required, deletion and correct regeneration are acceptable when they produce a cleaner current state.

### Tier 2: Corrected Prediction and Evaluation Foundation

#### 4. Generate one clean canonical weekly forecast

Exercise the complete current path:

```text
complete retained history
-> deterministic Elo reconstruction
-> verified Elo lineage
-> strict statistical availability
-> policy resolution
-> evidence-aware execution
-> immutable input snapshots and family evidence
-> forecast events
-> weekly product
-> explicit current selection
-> readiness verification
-> API and frontend serialization
```

The run may be identified honestly as development validation rather than historically live evidence.

#### 5. Rebuild historical game-model evaluation

Inspect whether current backfills, prediction archives, evaluation summaries, and baseline reports were built from complete and valid inputs. Rebuild uncertain artifacts using leakage-safe, time-ordered boundaries.

Required outcomes:

- authoritative evaluation periods and exclusions;
- explicit chronological training and holdout boundaries;
- regenerated backfilled predictions where needed;
- independent classification and regression evaluation;
- regenerated baseline reports from corrected evidence.

#### 6. Evaluate all current game-model families

Win families:

- Elo;
- Logistic;
- Random Forest;
- XGBoost.

Win evaluation should include Brier score, log loss, calibration error, calibration slope and intercept when supported, accuracy as secondary context, probability sharpness, season-level stability, and reliability by confidence band.

Total families:

- Random Forest;
- XGBoost.

Total evaluation should include MAE, RMSE, median absolute error, bias, interval coverage when uncertainty is emitted, season-level stability, and performance by relevant game environment.

#### 7. Reassess calibration and champion selection

Use only corrected evaluation evidence to determine whether current champions remain justified. When warranted:

- regenerate calibration artifacts;
- promote corrected champions through the static manifest boundary;
- verify complete upcoming-game feature coverage;
- execute the weekly policy with the selected champions;
- regenerate dependent baseline and operational artifacts.

### Tier 3: Explainability and Analytical Trust

#### 8. Persist Logistic explanation evidence

Prefer exact scaled-feature-by-coefficient contributions in log-odds space as the first explanation method.

Required behavior:

- bind explanations to exact model, scaler, feature schema, transformed inputs, and forecast identity;
- preserve the intercept explicitly;
- reconstruct the persisted estimator output within a strict tolerance;
- generate and persist explanations in batch;
- serialize persisted explanation artifacts through the API;
- perform no request-time model inference;
- distinguish contribution from causality.

#### 9. Evaluate tree-model attribution only if justified [Resolved: not built — see `DECISIONS.md` D53]

After Logistic explanation evidence is complete, assess Tree SHAP or another appropriate method for tree champions. Require exact model-byte and feature-schema binding, consistency tests, and persisted outputs.

Assessed and not built: Random Forest's Win classifier is unconditionally wrapped in isotonic calibration with no closed-form decomposition, so exact attribution against its raw output cannot satisfy the required reconstruction test. XGBoost's Win classifier is only conditionally recalibrated per training run (holdout ECE threshold); its current on-disk artifact is uncalibrated, so exact margin-space attribution would be reasonable for it today, but an attribution mechanism cannot assume that holds for every future retrain. Neither tree type is the current Win champion, so there is no consumer to validate a mechanism against yet. Revisit only once a resolved approach exists for attributing through the calibration boundary (D53's stated precondition).

#### 10. Build comparable-game retrieval [Complete: U13 backend (`DECISIONS.md` D54), U14 API wiring (D55)]

Define similarity from authenticated model inputs, prevent future-information leakage, explain why games are comparable, and derive any thresholds empirically.

Done: a historical feature-vector corpus in the `win_prob`/logistic champion's own standardized space, Euclidean-distance retrieval per run event with an empirically-derived (leave-one-out, p90) distance threshold, real recorded outcomes, and named top-contributing features — all immutable, batch-computed, and idempotent on rebuild. `GET /games/{game_id}/comparables` serves this evidence directly, with an honest empty result (not a blocked state) when a matchup has no sufficiently similar historical game. Frontend UI consumption remains future work (Tier 8 #23/#24).

### Tier 4: Market and Recommendation Proof

#### 11. Complete selected-plan quote collection evidence

Complete the calendar-gated selected-plan executions, verify claim and result completeness, accumulate repeated real observations, validate worker synchronization, and resolve any incomplete claims deliberately.

#### 12. Complete market closeout and CLV

Produce market-specific closing evidence for Moneyline, Spread, and Total using exact provider-aware offer identities and latest eligible non-live quotes before kickoff. Preserve missing and ambiguous close states explicitly.

#### 13. Mature recommendation policy from empirical evidence

Require sufficient completed outcomes, repeated quote depth, validated closeout, CLV samples, settled returns, and explicit sample sufficiency before deriving actionable thresholds. Do not invent a policy when evidence is incomplete.

#### 14. Complete end-to-end production-chain proof

Prove the complete chain for Moneyline, Spread, and Total:

```text
forecast
-> quote history
-> candidate issuance
-> policy
-> recommendation evaluation
-> recorded wager
-> closeout
-> CLV
-> realized return
-> performance report
```

### Tier 5: Model and Feature Improvement

#### 15. Research model ensembles

Evaluate Brier-weighted averaging, constrained blending, time-ordered Logistic stacking, and simple probability averaging against corrected champions. Require honest out-of-sample gains, preserved calibration, deployable metadata, complete feature coverage, and no API-time computation.

#### 15a. Fix neutral-site handling in the Win models [Closed 2026-10-02: Unit 8, `DECISIONS.md` D74]

The Logistic Win model applied its home-field intercept at neutral-site games (week 4 London game: 59.4% for the nominal Home side against a 36.9% market). Fixed structurally in schema 14 (neutral coefficient pinned to the negative intercept), the game champions were retrained, and the week 4 product was regenerated and reselected (IND-WAS 51.8%, spread -0.57). The trees and Total were audited and left unchanged (D74). Remaining watch item: revisit the structural choice when more neutral games accumulate (the 21-game 2023-2025 neutral holdout slice scored worse, interval includes zero). Even with the fix, week 4 Washington is about 52% against a 37% market, so a large gap there is not a neutral-site effect.

#### 16. Game-model feature program

**Depends on:** the Foundation Completion feature-program gate above (Track A units U1–U14 complete; Track B has closed Tier 4 #11, #12, and #14). Scope is game models only — Win (`HOME_WIN`: Elo, Logistic, Random Forest, XGBoost) and Total (`ACTUAL_TOTAL`: Random Forest, XGBoost). Prop-model features are Tier 5 #17. Feature-level detail, adopted status, and exclusions live in `FEATURES.md` Part III, "Game-Model Build Queue"; this entry owns the program rules and phase ordering.

**Program rules**

1. **Screen before promoting.** Evaluate each candidate out of band, with candidate columns joined onto the modeling file by `GAME_ID`. Only adopted batches enter `FeatureRegistry`/`CANONICAL_FEATURES`; a rejected feature never costs a *net* schema bump. **Wiring and promotion are decoupled (revised 2026-09-28, `DECISIONS.md` D62):** a batch is wired into `EPA_COLS`/`CANONICAL_FEATURES` with its own schema bump as soon as it is built and tested, so several batches can accumulate without each one blocking on a full retrain. The gated `gridiron models train` comparison that actually decides adoption runs later, through one consolidated incremental sweep (see the Phase A screening & promotion unit below) — not immediately after each wiring unit. A batch the sweep rejects is removed by a subsequent *forward* schema bump, never a rewind of an already-published schema version.
2. **One batch per unit.** Each batch gets one new `FeatureSet` name and one schema or data version bump (`features/manifest.py`) when it is wired. Promotion happens only through the gated `gridiron models train` comparison run by the consolidated sweep, never through full-retrain promotion, which has no minimum-improvement gate.
3. **Live availability.** A feature may enter a live feature set only if it has a guaranteed pre-kickoff value for every scheduled game, proven by replaying archived upcoming-schedule inputs. Any NaN feature disables that family for the entire week.
4. **Missingness decisions.** Era-gapped features (for example CPOE before 2006, moneylines before 2007, Next Gen Stats from 2016, PFR advanced stats from 2018) need a recorded `DECISIONS.md` entry choosing a training-window or indicator-plus-fill policy before adoption.
5. **New inputs stay out of cleaned games.** `data/cleaned/NFL_wk_by_wk_cleaned.csv`'s exact bytes and columns are pinned by Elo lineage; new schedule fields go in a separate sidecar dataset, added to the prediction-input evidence source inventory. Any persisted rating system needs lineage treatment equivalent to Elo's, recorded in `DECISIONS.md`.
6. **As-of timing and leakage.** Every feature is built only from what is knowable pre-kickoff, with an explicit leakage test.

**Acceptance per adopted batch**, measured under the Tier 2 corrected walk-forward protocol (same holdout seasons, same games, same hyperparameter budget, fixed seeds, baseline rebuilt without the batch):

- **Win:** Brier score improves by at least 0.002, with the paired game-bootstrap 90% lower bound above zero, holding in at least 2 of 3 holdout seasons; ECE worsens by no more than 0.01; log loss is not worse.
- **Total:** MAE improves by at least 0.09 with the paired-bootstrap 90% lower bound above zero, in at least 2 of 3 holdout seasons; RMSE not worse; absolute bias shift at most 0.25 (locked before the first sweep, `DECISIONS.md` D71; Tier 2 #7 set no numeric threshold).
- **Operational:** the live-availability replay passes, training-row loss is recorded, and a rejected feature's status and reason are recorded in `FEATURES.md`.

**Phase A — free, data already on disk.** Sequenced into units below, one batch per unit (program rule 2). Batches are independent of each other except where a unit's own dependency is noted; the order follows `FEATURES.md`'s existing A1–A7 listing (cheapest/least-entangled first) rather than a hard dependency chain like Foundation Completion's Track A.

**Phase A unit sequence (adopted 2026-09-28; sequencing and promotion model revised 2026-09-28, `DECISIONS.md` D62):**

Units 1–6 each wire one batch into `EPA_COLS`/`CANONICAL_FEATURES` with its own schema bump. Per the revised rule 1, a wiring unit closes once its columns are built, tested, and the live-availability replay passes (program rule 3) — it does not individually retrain or promote. Unit 7 makes the promote/reject call for whatever batches have accumulated by the time it runs, via one consolidated incremental sweep; it does not have to wait for Units 2–6 to all exist first.

- **Unit 1 — A1, construction fixes.** CPOE (Domain 1, currently ⚠️ Partial), fumble rate/offense (Domain 3, ❌), and a correction to the stale INT-rate/defense classification (already built, mis-marked ❌). **Closed: wired, tested, live-availability replay passed (schema v6, `DECISIONS.md` D60/D61) — promotion deferred to Unit 7.**
- **Unit 2 — A2, opponent-adjusted EPA and unit interactions.** Opponent-adjusted offensive and defensive EPA/play (pass + rush), unit efficiency differentials built on it, and the market-blind variant of projected total plays. **Closed: wired, tested, live-availability replay passed (schema v9, `DECISIONS.md` D63/D64) — promotion deferred to Unit 7.**
- **Unit 3 — A3, QB-and-coach sidecar.** QB EPA (career + rolling), CPOE rolling, QB change flag, QB experience, head-coach win % and tenure, new-coaching-staff flag. **Closed: wired, tested, live-availability replay passed (schema v10, `DECISIONS.md` D65) — promotion deferred to Unit 7.**
- **Unit 4 — A4, situational context.** Season week number and indoor-override status corrections, back-to-back road games, body-clock kickoff, field surface, league scoring environment/rule era, wind speed bins, season week performance, Pythagorean luck gap, one-score game record, special-teams and punt/return EPA. **Closed: wired, tested, live-availability replay passed (schema v11, `DECISIONS.md` D66–D68) — promotion deferred to Unit 7.** Also fixed a train/serve roof-classification skew (D67).
- **Unit 5 — A5, rating ensemble inputs.** QB Elo/rating, team Glicko-2 (rating + RD + volatility), Keener/Colley/Massey ratings, rating uncertainty as a feature. **Closed: wired, tested, live-availability replay passed (schema v12, `DECISIONS.md` D69/D70) — promotion deferred to Unit 7.** All four families are recomputed from complete history rather than persisted; Glicko-2 volatility was declined on a pre-declared variance gate.
- **Unit 6 — A6, market-aware variant.** Consensus closing spread/total, no-vig market win probability, historical closing line as a prior, season ATS/O-U record, performance as favorite/underdog — as a separate model identity from the market-blind champion, not a drop-in feature set. Needs its own `DECISIONS.md` entries per program rule 4 before adoption. Display idea (user, 2026-10-03): show two odds columns per market, *first observed* and *current*, from the existing quote ledger (`data/odds/history`); label the first one "first observed", not "opening", unless a true opening source is added. **Source audit (2026-10-02):** training data for closing lines already exists: `data/raw/NFL_wk_by_wk_nflverse.parquet` has `spread_line` and `total_line` for every game since 1999 and moneylines and spread/total odds from about 2009 (5,343 games), and the lines behave like unbiased closing consensus (positive spread means Home favored; margin minus spread averages 0.085; overround 2.7%). The exact capture time is not stated, so confirm "closing" before relying on it. The no-vig moneyline scores Brier 0.2114 over 2013-2025 (3,561 games) against about 0.2204 for the Logistic walk-forward on similar games (game sets not identical), a gap several times larger than the effects screened in Phase A. Opening lines and movement are not on disk; The Odds API historical endpoint (paid plans only; from 2020-06-06 at 10-minute snapshots, 5-minute from 2022-09; featured markets from the start; cost 10 x markets x regions credits per request) could supply about six seasons. Design constraint: a model trained on closing lines must be served late in the week (or trained at the serving lead time) to avoid train/serve skew, and a market-aware model's edges against the market shrink by construction, so its value is anchoring, not independent edges.
- **Unit 7 — Phase A screening and promotion.** Builds a one-off incremental walk-forward sweep — baseline, then baseline+A1, baseline+A1+A2, and so on in wiring order — reusing the existing chronological holdout split and `compare_classification_models`/`compare_regression_models` gates, so each batch's own attributable Brier/MAE delta is measured against the batch immediately before it, not just against the pre-Phase-A baseline (preserving the "baseline rebuilt without the batch" acceptance methodology while amortizing the walk-forward cost across every wired batch at once). Produces one consolidated report; promotes whichever batches clear the acceptance bar (recorded in `FEATURES.md`) and forward-removes (a new schema bump, never a rewind) any that don't. Runs whenever enough units have accumulated to be worth evaluating together, not necessarily only after Unit 6. **Closed 2026-10-01: the holdout sweep could not resolve the effects (D71); wide-window attribution kept only A3 with a debut-prior fill and forward-removed A1, A2, A4, A5 (schema 13, D72/D73); champions retrained and promoted.**
- **A7 is not sequenced as a mainline unit.** Playoff/elimination context and seeding-locked risk affect too few holdout games (~48) to reliably clear the acceptance bar; assess later only if enough holdout data accumulates, mirroring the Tier 3 #9 precedent of recording a documented non-build decision rather than building on thin evidence.

Phase B and Phase C units are numbered when their phase starts (Phase B needs an nflverse re-download; ask before running. Phase C costs money). Pre-numbering them now would outrun what program rule 1 ("screen before promoting") can verify before Phase A closes.

**Phase B — free, needs an nflverse re-download** (ask before running): re-ingest play-by-play with the columns needed for a win-probability-based garbage-time filter, quarterback-attributed rolling EPA and CPOE, drive-level features, turnover luck, and penalty yards; depth charts for a live projected starter, only if the sidecar's pre-kickoff QB-identity coverage proves insufficient; pressure rate calibrated between historical participation data and in-season PFR advanced stats.

**Phase C — costs money, last:** live forecast weather (fixes the training/serving weather skew properly); market timing features built from Odds API history going forward (depends on Tier 4 #11 and #19); paid benchmark comparisons (FTN DVOA, PFF) as reference only, not model inputs.

Excluded from this program, with reasons recorded in `FEATURES.md`: injury-dependent features move to Tier 6 #18; player-value and roster/usage features (WAR, on/off splits, backup quality, usage redistribution) move to Tier 7; player-level rating systems (OpenSkill) move to the Research Backlog.

Every feature must preserve chronology, avoid leakage, and use empirical thresholds.

#### 17. Improve prop models and expand supported prop families

Potential work includes richer distribution models, target-specific uncertainty, playing-time and usage treatment, injury-dependent projections, corrected historical evaluation, additional prop families, and calibrated Over/Under probabilities.

### Tier 6: External Data Foundations

#### 18. Add injury and news ingestion

Choose a reliable source and usage policy, preserve fetched-at and effective-at timestamps, distinguish reported, confirmed, and resolved status, map players and teams canonically, retain historical availability, and expose incomplete coverage explicitly.

**Source audit (2026-10-02, read-only; one season each of 2009 and 2024 downloaded to a scratch directory, nothing written to `data/`).** nflverse `injuries` (free, same release family the repository already uses) is small and usable but coarser than the roadmap wording assumes:
- Size: 84 KB (2009) and 139 KB (2024) of parquet per season; 15+ seasons is on the order of 2 MB. Storage is not a constraint, and no streaming or multi-worker ingestion is needed for this source.
- Granularity: one row per player, team and week (two duplicates in 2024), the last report of the week, not a within-week series. 2024 `date_modified` is mostly Friday (4,818 of 6,215 rows), so it reflects the final practice and game-status report; Wednesday and Thursday history and Sunday inactives are not retained. 2009 `date_modified` is a single placeholder value (99.6% null), so early seasons have no usable timestamps. Features can therefore only be "status as of the final weekly report", which means predictions that use them must run Friday evening or later.
- Vocabulary changes by era: 2009 uses `Probable` (retired later); 2024 has no game status for half of the rows (practice-only entries). Only players on the report appear; healthy players are absent.
- QB signal is present (2024: 201 quarterback rows, 14 `Out`, 8 `Doubtful`), joinable by `gsis_id` to the existing resolved-starter logic (`DECISIONS.md` D65).
- Depth charts (nflverse, back to 2001) changed data source after 2024, so the 2025+ format differs; treat as a separate, later source.
- Follow-up (2026-10-03): nflverse's 2025 and 2026 injury files drop `date_modified`, and 2025+ depth charts are timestamped snapshots (about daily). The starting-QB projection (Unit 9, `DECISIONS.md` D75) uses the depth chart plus the weekly report; its integration into the live QB features is built (Unit 10, `DECISIONS.md` D76). Injury ingestion beyond availability status, news, and a forward timestamped ledger remain here.
- Decision pending with the user: whether weekly prediction moves to late week so these features can be used without train/serve skew, or the model trains on a fixed late-week as-of state and serves only then.

#### 19. Add supported historical market backfill

Use the source-neutral provider contract with exact observation timestamps, sportsbook identity, kickoff and live-state validation, and honest coverage gaps. Do not invent opening, closing, or movement interpretations.

### Tier 7: Scenario Analysis

#### 20. Build bounded what-if computation

Reuse the pure projected-starter candidate list and the single projected-starter recompute function built for live serving (`DECISIONS.md` D75) for "player absence" and quarterback swaps; do not build a second path. Support explicit hypothetical adjustments such as team strength, player absence, usage redistribution, environment, alternate lines, or alternate model selection without mutating production forecasts or current weekly products.

#### 21. Persist scenario request and result contracts

Bind every scenario to an exact base forecast and explicit modifications:

```text
scenario request
-> exact base forecast identity
-> explicit modifications
-> computation version
-> result
-> comparison with base forecast
```

### Tier 8: API and Frontend Completion

#### 22. Verify remaining API batch-artifact boundaries

Inspect each candidate endpoint before scheduling work. Move meaningful request-time computation to persisted batch artifacts, retain thin loaders and serializers, and add parity tests before removing prior behavior.

The known candidate is model-performance summary delivery, but the current implementation must be inspected before treating it as a gap.

#### 23. Add explanation and scenario surfaces [Partial: explanation + comparables shipped, U16]

After backend evidence exists, expose explanations, comparable games, and scenarios with exact provenance and truthful unavailable states.

Done: `ExplainPage` renders the real factor waterfall (sorted by magnitude, collapsed beyond the top 8 with a "show more" disclosure) and the real comparable-games table, both served from persisted evidence with no request-time computation; a genuinely-empty comparables result is shown distinctly from a blocked one. Not done: the credible band, outcome distribution, and market comparison remain `ComingSoonCard` placeholders — they depend on the scenario engine (Tier 7), which is unbuilt.

#### 24. Complete evidence-backed frontend enhancements

Potential enhancements:

- multi-book line-shopping views;
- injury and news presentation;
- explanation and scenario surfaces;
- line-movement charts;
- richer bankroll history and Kelly-adherence views;
- recorded-bet export;
- remaining table, layout, and accessibility polish;
- a real-data pending-state visual audit;
- model-performance comparison against naive references (constant home-win rate for Win, constant league mean for Total) and Elo, alongside the champions. The references are produced by the Phase A sweep's harness as persisted, immutable evidence; the API serializes that persisted evidence only, never computing a reference at request time.

### Tier 9: Live and Real-Time Support

#### 25. Add live game state

Foundations include score, clock, down, distance, possession, timeouts, timestamped live markets, freshness rules, and streaming or polling transport.

#### 26. Develop and validate in-game models

Require time-indexed training data, a validated live win-probability model, stale-state behavior, live edge and hedge evaluation, and operational transport. This remains lower priority than complete pregame foundations.

### Tier 10: Tooling and Operational Hardening

#### 27. Strengthen CI and quality automation

Potential work:

- exercise `gridiron verify --strict` in a real CI surface;
- run separate frontend lint, build, and test gates in CI;
- maintain generated OpenAPI and TypeScript contract checks;
- establish performance baselines where useful;
- improve long-running composite resume diagnostics;
- verify remaining baseline-report parser edge cases;
- review lint exclusions only through dedicated behavior-preserving work.

Repository-wide Ruff and Pyrefly are currently active quality gates. Do not describe their baseline as future work unless a verified gap reappears.

#### 28. Define upstream data-cadence behavior

Define truthful handling for unpublished current-season PBP, offseason empty schedules, upstream season lag, weather availability, provider quotas, and stale operational artifacts.

#### 29. Revisit storage architecture only when justified

Retain file-backed storage until demonstrated multi-user concurrency, transaction, or query requirements justify a database.

## Calendar-Gated Work

### Market Unit 26 [Closed 2026-09-27]

Market Unit 26 closed 2026-09-27: real 2026 Week 1 selected-plan execution, repeated quote coverage, completed outcomes, validated closeout and CLV, and complete Moneyline/Spread/Total production-chain proof are all done. Its detailed record remains in:

- `docs/programs/market-unit-26/PLAN.md`
- `docs/programs/market-unit-26/ROADMAP.md`

What remains open on Unit 26's own calendar, not gating anything else: maturing empirical recommendation thresholds (Tier 4 #13) from accumulating weekly evidence, and realized performance, which stays explicitly unavailable until a wager is recorded.

The root roadmap may proceed independently while that evidence matures. Calendar-gated work must not occupy the one active root implementation unit unless the required real-world evidence is available.

## Research Backlog

Research should not displace integrity, evaluation, or calendar-eligible operational proof.

Candidate research areas:

- ensemble methods;
- additional game-model features;
- richer prop distributions;
- calibration-transparent tree-model attribution: pre-calibration TreeSHAP /
  `pred_contribs` plus an explicit, separately-labeled calibration-adjustment
  step, rather than attempting to reconstruct the calibrated output directly
  (see `DECISIONS.md` D53, which found no resolved approach exists yet);
- explainability as an explicit champion-selection criterion, not only an
  accuracy/calibration gate — and whether already-promoted opaque champions
  (for example `total`/random_forest) should be reconsidered under it;
- Explainable Boosting Machines (EBM) or other GAM-style additive models as
  an accuracy-vs-explainability bridge for future Win, Total, and prop
  model families — potentially exact per-prediction decomposition (like
  Logistic) without Random Forest/XGBoost's calibration-boundary problem;
- supported historical market data;
- scenario methods;
- live model design.

Research results do not become operational behavior without time-ordered evaluation, deployment contracts, availability handling, persisted provenance, and explicit acceptance criteria.

## Known Limitations

### Canonical Week 2 state: replaced (U6, closed)

The originally disposition-affected Week 2 Logistic Win runs consumed incomplete-history Elo inputs and remain governed unchanged by their existing disposition. The untracked ad hoc `development`-role selection that later replaced them (D5) has been retired and replaced by a freshly generated, repository-owned `development`-role run, composed and selected through `gridiron regenerate-development-week` (see D45). The Games API and frontend now serve this corrected selection.

### Historical evaluation may require regeneration

Collateral impact from the truncated-history period has not yet been fully classified. Historical predictions, evaluation archives, backfills, closeout artifacts, preflight evidence, and derived datasets must not be treated as corrected until verified or regenerated.

### Recommendation-policy maturation is ongoing (Tier 4 #13)

Market Unit 26 closed 2026-09-27: real 2026 Week 1 selected-plan quote
coverage, validated closeout, CLV, and complete Moneyline/Spread/Total
production-chain proof are done (see `docs/programs/market-unit-26/PLAN.md`).
What remains open, on its own calendar and not gating anything else: maturing
empirical recommendation thresholds from accumulating weekly evidence, and
realized performance, which stays explicitly unavailable until a wager is
recorded (optional, not a gap).

### Timezone shift is evaluated at the build date, not the game date

`features/team/travel.py`'s `_build_tz_cache` computes UTC offsets at
`datetime.now(UTC)`, so `AWAY_TZ_SHIFT`/`HOME_TZ_SHIFT` for Arizona (no DST)
pairings vary with the day the modeling file is built, which is a
reproducibility defect. The body-clock feature (Unit 4, `DECISIONS.md` D68)
deliberately evaluates offsets at each game's date and does not inherit it.
Fixing the older feature changes existing values, so it needs its own small
unit with a data-version bump.

### Rating-ensemble lookahead is frozen, not simulated

Glicko-2, the QB rating, and Massey/Colley/Keener give a game beyond the next
week the state after the last completed game (`DECISIONS.md` D69). That is
unbiased in expectation and never null, but it understates how far ratings
will move over a long horizon. The same carry-forward approach could remove
Elo's next-week-only limitation below; that would be its own unit, since it
changes the Elo artifact contract.

### Elo, strength of victory, and rest are next-week only

Replaying the full upcoming schedule (Unit 4) shows Elo and strength of
victory are NaN for weeks beyond the next one, and `rest.py` measures rest
from the last completed game. All three are correct for next-week serving,
which is what the live path uses; none has been verified for longer
lookahead. The batch A4 columns are exact at any horizon.

### Injury, news, and live state are unavailable

There is no integrated injury/news feed or live-game state. Dependent API and frontend fields must remain explicitly blocked.

### What-if scenario evidence is unavailable

What-if propagation and the scenario engine (Tier 7) are not implemented. `ExplainPage` shows this honestly as three "not yet available" placeholders (credible band, outcome distribution, market comparison) rather than fabricated interactivity. Persisted feature attribution and comparable-game retrieval are both complete and served through the real `ExplainPage` UI (U13/U14/U16, `DECISIONS.md` D54/D55/D57).

### Champion selection does not yet gate on explainability

`evaluation/champion.py`'s promotion gates currently select on predictive and calibration metrics only. As of the current champion manifest, `total`/random_forest and two prop champions (`qb_pass_yards`, `wr_rec_yards`) have no per-prediction attribution, and none is required for promotion. Whether explainability should become an explicit champion-selection criterion — and whether existing opaque champions should be reconsidered under it — is open; see the Research Backlog.

### Current-season PBP may lag

The upstream source may not publish the current season immediately. Pipeline refresh may warn while continuing with available historical feature state. A bounded cleanup should define and implement the repository policy for clamping requests to supported seasons.

### Postgame work requires completed outcomes

`post-week` correctly exits nonzero and lists missing outcomes when run before all scoped games finish.

### Markets and predictions remain independent

A selected weekly product can be prediction-ready while market readiness is blocked. Missing market data means no current edge result; it does not invalidate a valid forecast.

### File-backed architecture is intentional

Files remain appropriate for the current single-user workflow. Revisit this only for demonstrated concurrency, transactional, or query requirements.

## Prioritization Guidance

The next major work should normally be chosen in this order:

1. repository state and canonical artifact reconciliation;
2. canonical Week 2 retain-or-replace decision;
3. truncated-history and stale-artifact audit;
4. one clean canonical weekly forecast;
5. corrected historical evaluation and backfills;
6. calibration and champion reassessment;
7. persisted Logistic explanation evidence;
8. calendar-eligible market collection, closeout, CLV, and realized-return proof;
9. recommendation-policy maturation;
10. ensemble and feature research;
11. injury and news data;
12. bounded scenario analysis;
13. remaining API batch-artifact verification;
14. evidence-backed frontend surfaces;
15. prop-model improvements;
16. CI, operational cadence, and source handling;
17. live-game support after pregame foundations are complete.

Item 9 runs on its own calendar in parallel with items 1–8 and 10 (Foundation Completion Track B, above) and no longer gates item 10: the game-model feature program (Tier 5 #16) starts once items 1–7 are complete and calendar-eligible market proof (item 8, specifically Tier 4 #11/#12/#14) has closed, without waiting for a matured recommendation policy.

Before starting a new work item:

- verify the gap against current code and artifacts;
- add exactly one bounded active unit to `PLAN.md`;
- record locked architectural choices in `DECISIONS.md` only when a new durable decision is made;
- update `HANDOFF.md` only after behavior ships;
- record completed behavior in `CHANGELOG.md`;
- remove or reclassify the roadmap item when the unit closes.
