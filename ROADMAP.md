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

### Repository State and Canonical Artifact Reconciliation

**Goal:** establish one verified, authoritative inventory of the current platform, canonical artifacts, stale development state, calendar-gated work, and remaining implementation backlog before beginning another large feature or evaluation program.

This program begins with read-only inspection and does not assume that development-era artifacts must survive.

Required work:

1. Verify current code capabilities and artifact state against repository evidence.
2. Classify each roadmap item as complete, active next, calendar-gated, planned, research, deferred, or rejected.
3. Remove completed work from future-work sections and eliminate duplicate or stale backlog language.
4. Identify artifacts produced while cleaned game history was truncated or while model metadata was stale.
5. Determine which affected artifacts were subsequently rebuilt and which remain questionable.
6. Decide whether the defective 2026 Week 2 development state should remain as a regression fixture or be replaced with corrected canonical development artifacts.
7. Define one canonical path for corrected forecast generation, evaluation, calibration, and champion reassessment.
8. Leave `PLAN.md` with exactly one bounded next implementation unit after reconciliation.

Acceptance:

- this roadmap contains one complete ordered backlog;
- `HANDOFF.md` describes only current behavior;
- completed capabilities are not presented as future work;
- future capabilities are not presented as implemented;
- every remaining item has a clear status and dependency;
- the Week 2 development state has an explicit retain-or-replace decision;
- stale or questionable artifacts have a delete, archive, verify, or regenerate disposition;
- calendar-gated market proof remains separate from implementation work;
- the next bounded implementation unit is evidence-based and recorded in `PLAN.md`.

## Canonical Artifact Decision Required

### 2026 Week 2 Development State

Two Week 2 Logistic Win forecast runs were generated from an Elo state rebuilt from only the 16 completed Week 1 games. A schema-1 disposition currently records the affected events and products as known defective.

Because the project has never been live, preservation is not mandatory merely because these artifacts exist. The reconciliation program must choose one of the following explicitly.

#### Retain as a regression fixture

Retain the events, products, selection, and disposition because they provide useful end-to-end evidence for known-defect governance, readiness blocking, and downstream enforcement. If retained, documentation must describe them as an intentional development regression fixture rather than an untouchable production record.

#### Replace with corrected canonical development state

Archive or remove the defective events, affected products, selection, and operational disposition, then regenerate a coherent Week 2 development fixture through the current history-preserving, lineage-verified, metadata-strict, evidence-aware weekly path. Verify API and frontend serialization against the corrected selected product.

**Recommended default:** replace the defective operational state unless the regression-fixture value is judged greater than the continuing conceptual overhead. The disposition implementation and tests may remain as a supported capability even if the specific Week 2 operational artifact is retired.

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

#### 9. Evaluate tree-model attribution only if justified

After Logistic explanation evidence is complete, assess Tree SHAP or another appropriate method for tree champions. Require exact model-byte and feature-schema binding, consistency tests, and persisted outputs.

#### 10. Build comparable-game retrieval

Define similarity from authenticated model inputs, prevent future-information leakage, explain why games are comparable, and derive any thresholds empirically.

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

#### 16. Research game-model features

Candidate areas:

- offensive and defensive strength decomposition;
- coaching and coordinator effects;
- pace and neutral-situation tendencies;
- special teams;
- penalties, pressure, and situational efficiency;
- opponent-quality cohorts;
- calibrated uncertainty.

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

#### 23. Add explanation and scenario surfaces

After backend evidence exists, expose explanations, comparable games, and scenarios with exact provenance and truthful unavailable states.

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
- comparable-game retrieval;
- tree-model attribution;
- supported historical market data;
- scenario methods;
- live model design.

Research results do not become operational behavior without time-ordered evaluation, deployment contracts, availability handling, persisted provenance, and explicit acceptance criteria.

## Known Limitations

### Canonical Week 2 state is unresolved

The selected Week 2 development product is currently governed by a known-defect disposition because its Logistic Win predictions consumed incomplete-history Elo inputs. The reconciliation program must decide whether to retain this state as a regression fixture or replace it with corrected canonical development artifacts.

Until that decision is executed, readiness, edge calculation, candidate issuance, and manual rendering remain blocked for the affected selection under the current contract. The Games API, frontend, and postgame paths may still expose the persisted development artifact according to their existing boundaries.

### Historical evaluation may require regeneration

Collateral impact from the truncated-history period has not yet been fully classified. Historical predictions, evaluation archives, backfills, closeout artifacts, preflight evidence, and derived datasets must not be treated as corrected until verified or regenerated.

### Market proof is incomplete

Repeated selected-plan quote coverage, validated closeout and CLV, empirical recommendation thresholds, recommendation product integration, realized performance, and complete Moneyline, Spread, and Total proof remain incomplete or calendar-gated.

### Injury, news, and live state are unavailable

There is no integrated injury/news feed or live-game state. Dependent API and frontend fields must remain explicitly blocked.

### Scenario and explanation evidence are unavailable

Persisted feature attribution, comparable-game retrieval, and what-if propagation are not implemented.

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

Before starting a new work item:

- verify the gap against current code and artifacts;
- add exactly one bounded active unit to `PLAN.md`;
- record locked architectural choices in `DECISIONS.md` only when a new durable decision is made;
- update `HANDOFF.md` only after behavior ships;
- record completed behavior in `CHANGELOG.md`;
- remove or reclassify the roadmap item when the unit closes.
