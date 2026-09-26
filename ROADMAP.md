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

### Foundation Completion (Tiers 1–4)

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
| D8 | Candidate issuance does not reject non-live forecast roles or forecasts generated after `evaluated_at`. | `cli/production_chain.py:404-480`. |

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

#### Track B — Tier 4 / Market Unit 26 (parallel, calendar-driven)

No selected-plan poll has completed from this workstation; the Week 1 plan's polls are all past due. The Raspberry Pi worker has been polling independently all season, but nothing in this repository can currently read its evidence back, and its stored contents beyond the latest quote snapshot are unconfirmed.

- **M1 — Pi evidence inventory.** Read-only, done with the operator: confirm what the worker has persisted (quote-history partitions, claims, results, timer/journal state, credit usage) before deciding what Weeks 1–3 evidence is recoverable.
- **M2 — Worker evidence sync**, closing part of **Tier 4 #11**: a repository-owned pull (e.g. over SSH/rsync) that copies immutable quote-history partitions, claims, and results into local `data/`, verifying identities and digests, refusing conflicting overwrites, and idempotent on repeat; paired with a plan-rollover procedure that compares identities after every transfer.
- **M3 — Chain guards and the Week 1 issuance conflict**: resolve D8 (issuance must reject non-live and late-generated forecasts), and resolve Week 1's existing triple-candidate-issuance conflict (archive the extras or add an explicit `DECISIONS.md`-recorded selection rule).
- **M4 — Weekly proof cadence** (Tier 4 #11, #12, #14): forecast, plan, roll the plan to the worker, issue candidates before the first kickoff, optionally record one BetSlip wager against a candidate, fetch outcomes, sync per M2, and run production-chain assessment — repeated weekly starting from the first week M1 confirms is usable. Close Unit 26's #11/#12/#14 with the recommendation policy recorded as explicitly unavailable.
- **M5 — Tier 4 #13, recommendation-policy maturation.** A new design unit defining a threshold-selection method and an explicit sample-size rule, then accumulating evidence weekly. This runs independently and is not required for the Tier 5 gate below — matured-policy evidence take most of a season to accumulate.

#### Feature-program gate

ROADMAP Tier 5 #16 (the game-model feature program) does not begin until Track A (U1–U14) is complete and Track B has closed Tier 4 #11, #12, and #14. Track A is now complete; the gate remains closed on Track B. Tier 4 #13 continues independently and does not gate Tier 5.

#### Acceptance

- every defect in the table above has a resolution recorded in code, tests, or an explicit `DECISIONS.md` entry;
- `HANDOFF.md` describes only current behavior;
- the Week 2 decision is executed, not just recorded;
- historical evaluation, calibration, and the champion manifest are regenerated from the corrected pipeline;
- Logistic explanation evidence is persisted and served;
- Market Unit 26 has closed #11, #12, and #14, with #13 continuing on its own calendar;
- `PLAN.md` names exactly one bounded active unit at a time throughout.

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

#### 16. Game-model feature program

**Depends on:** the Foundation Completion feature-program gate above (Track A units U1–U14 complete; Track B has closed Tier 4 #11, #12, and #14). Scope is game models only — Win (`HOME_WIN`: Elo, Logistic, Random Forest, XGBoost) and Total (`ACTUAL_TOTAL`: Random Forest, XGBoost). Prop-model features are Tier 5 #17. Feature-level detail, adopted status, and exclusions live in `FEATURES.md` Part III, "Game-Model Build Queue"; this entry owns the program rules and phase ordering.

**Program rules**

1. **Screen before promoting.** Evaluate each candidate out of band, with candidate columns joined onto the modeling file by `GAME_ID`. Only adopted batches enter `FeatureRegistry`/`CANONICAL_FEATURES`; a rejected feature never costs a schema bump.
2. **One batch per unit.** Each batch gets one new `FeatureSet` name, one schema or data version bump (`features/manifest.py`), and one retrain through the gated `gridiron models train` comparison — never through full-retrain promotion, which has no minimum-improvement gate.
3. **Live availability.** A feature may enter a live feature set only if it has a guaranteed pre-kickoff value for every scheduled game, proven by replaying archived upcoming-schedule inputs. Any NaN feature disables that family for the entire week.
4. **Missingness decisions.** Era-gapped features (for example CPOE before 2006, moneylines before 2007, Next Gen Stats from 2016, PFR advanced stats from 2018) need a recorded `DECISIONS.md` entry choosing a training-window or indicator-plus-fill policy before adoption.
5. **New inputs stay out of cleaned games.** `data/cleaned/NFL_wk_by_wk_cleaned.csv`'s exact bytes and columns are pinned by Elo lineage; new schedule fields go in a separate sidecar dataset, added to the prediction-input evidence source inventory. Any persisted rating system needs lineage treatment equivalent to Elo's, recorded in `DECISIONS.md`.
6. **As-of timing and leakage.** Every feature is built only from what is knowable pre-kickoff, with an explicit leakage test.

**Acceptance per adopted batch**, measured under the Tier 2 corrected walk-forward protocol (same holdout seasons, same games, same hyperparameter budget, fixed seeds, baseline rebuilt without the batch):

- **Win:** Brier score improves by at least 0.002, with the paired game-bootstrap 90% lower bound above zero, holding in at least 2 of 3 holdout seasons; ECE worsens by no more than 0.01; log loss is not worse.
- **Total:** meets the MAE/RMSE/bias thresholds set by Tier 2 #7's reassessment.
- **Operational:** the live-availability replay passes, training-row loss is recorded, and a rejected feature's status and reason are recorded in `FEATURES.md`.

**Phase A — free, data already on disk:** construction fixes (a garbage-time filter, a CPOE missingness policy, team `qb_epa`, defensive INT and fumble-lost rates); opponent-adjusted EPA and the unit-efficiency differentials built on it; a QB-and-coach sidecar built from the raw schedule (starting-QB identity and change flag, QB experience, coach tenure); situational context (body-clock kickoff, back-to-back road games, surface, a continuous league scoring environment, Pythagorean luck gap, special-teams and punt/return EPA, wind bins and a cold flag, season week number, a win-streak ablation); ensemble rating inputs (SRS, Massey, Colley, Keener, team Glicko-2); and a market-aware variant (closing spread/total, no-vig win probability) as a separate model identity from the market-blind champion.

**Phase B — free, needs an nflverse re-download** (ask before running): re-ingest play-by-play with the columns needed for a win-probability-based garbage-time filter, quarterback-attributed rolling EPA and CPOE, drive-level features, turnover luck, and penalty yards; depth charts for a live projected starter, only if the sidecar's pre-kickoff QB-identity coverage proves insufficient; pressure rate calibrated between historical participation data and in-season PFR advanced stats.

**Phase C — costs money, last:** live forecast weather (fixes the training/serving weather skew properly); market timing features built from Odds API history going forward (depends on Tier 4 #11 and #19); paid benchmark comparisons (FTN DVOA, PFF) as reference only, not model inputs.

Excluded from this program, with reasons recorded in `FEATURES.md`: injury-dependent features move to Tier 6 #18; player-value and roster/usage features (WAR, on/off splits, backup quality, usage redistribution) move to Tier 7; player-level rating systems (OpenSkill) move to the Research Backlog.

Every feature must preserve chronology, avoid leakage, and use empirical thresholds.

#### 17. Improve prop models and expand supported prop families

Potential work includes richer distribution models, target-specific uncertainty, playing-time and usage treatment, injury-dependent projections, corrected historical evaluation, additional prop families, and calibrated Over/Under probabilities.

### Tier 6: External Data Foundations

#### 18. Add injury and news ingestion

Choose a reliable source and usage policy, preserve fetched-at and effective-at timestamps, distinguish reported, confirmed, and resolved status, map players and teams canonically, retain historical availability, and expose incomplete coverage explicitly.

#### 19. Add supported historical market backfill

Use the source-neutral provider contract with exact observation timestamps, sportsbook identity, kickoff and live-state validation, and honest coverage gaps. Do not invent opening, closing, or movement interpretations.

### Tier 7: Scenario Analysis

#### 20. Build bounded what-if computation

Support explicit hypothetical adjustments such as team strength, player absence, usage redistribution, environment, alternate lines, or alternate model selection without mutating production forecasts or current weekly products.

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
- a real-data pending-state visual audit.

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

### Market Unit 26

Market Unit 26 remains active but calendar-gated. Its detailed plan and roadmap remain in:

- `docs/programs/market-unit-26/PLAN.md`
- `docs/programs/market-unit-26/ROADMAP.md`

Remaining proof includes selected-plan execution, repeated quote coverage, completed outcomes, validated closeout and CLV, empirical recommendation thresholds, realized performance, and complete Moneyline, Spread, and Total production-chain acceptance.

The root roadmap may proceed independently while this evidence matures. Calendar-gated work must not occupy the one active root implementation unit unless the required real-world evidence is available.

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

### Market proof is incomplete

Repeated selected-plan quote coverage, validated closeout and CLV, empirical recommendation thresholds, recommendation product integration, realized performance, and complete Moneyline, Spread, and Total proof remain incomplete or calendar-gated.

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
