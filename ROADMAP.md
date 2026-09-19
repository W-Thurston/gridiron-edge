# Gridiron Edge Roadmap

## Document Ownership

| Document | Purpose |
|---|---|
| `ROADMAP.md` | Genuine future capabilities, strategic priorities, and current limitations |
| `PLAN.md` | Active implementation checklist and completed-unit record |
| `HANDOFF.md` | Current operating system, commands, artifacts, and recovery guidance |
| `DECISIONS.md` | Append-only architectural decisions and supersession history |
| `CHANGELOG.md` | Dated implementation history |

The canonical weekly prediction, product, API, frontend, and verification architecture is implemented. Future work should build on the persisted-event and explicitly selected weekly-product contracts rather than restore retired archive, fallback, or request-time behavior.

## Current Platform State

Gridiron Edge currently provides:

- canonical one-row Away/Home game modeling;
- independent Win and Total model families;
- unversioned champion artifacts and a persisted champion manifest;
- model-specific weekly availability inspection and policy selection;
- immutable `live` and `backfilled` forecast events;
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

The successful 2026 Week 1 rehearsal produced complete Win, Spread, Total, projected-score, and provenance coverage for all scheduled games while market readiness remained independently blocked. Forecast PNG and HTML publication succeeded without requiring market prices.

## Strategic Priorities

Prioritize work by value density and architectural fit:

1. Correct weekly prediction input integrity and historical continuity before further model-quality, explanation, or product-surface work.
2. Preserve truthful persisted-state boundaries before adding breadth.
3. Resolve supported external data sources before building interfaces that depend on them.
4. Improve predictive quality only through honest time-ordered evaluation.
5. Keep the API a serialization boundary.
6. Keep unavailable, blocked, and analytical-empty states explicit.
7. Add product surface area only when the underlying data contract is real.
8. Continue using files until concurrency, transactional integrity, or query complexity requires a database.

## Future Work

### Weekly Prediction Input Integrity and Reproducibility

**Goal:** ensure every weekly forecast is generated from history-preserving, semantically validated, and reproducible model inputs before it can become the explicitly selected weekly product.

A read-only inspection completed on September 18, 2026 confirmed that both immutable 2026 Week 2 Win forecast runs used an Elo state rebuilt from only the 16 completed Week 1 games. The weekly explicit-season path selected `fetch_nflverse_games(seasons=[season])`, which replaced retained raw history; `clean_nflverse_games()` then replaced cleaned history from that nonempty partial artifact; and `update_elo_state_incremental()` ignored existing Elo state and rebuilt every team from the 1500 initial rating. The resulting Week 2 `{1490, 1510}` Elo pattern was reproduced exactly from the Week 1-only games frame. All 16 Week 2 logistic Win predictions used the reset Elo values through `AWAY_ELO`, `HOME_ELO`, and `ELO_DIFF`.

Corrective work proceeds as separate bounded units. Units 1 through 4 are
complete; immutable prediction-input evidence is next. Corrected operational
forecast generation and evaluation must follow reproducible input evidence,
and explanation work remains later.

1. **Preserve games history during weekly refresh. Completed September 18, 2026.**
  Recurring refresh now uses the history-preserving nflverse boundary,
   rejects empty selected-season responses, and preserves unrequested seasons.
2. **Validate and rebuild complete Elo history. Completed September 18, 2026.**
  Elo fitting now has one deterministic reconstruction contract,
   validates contiguous history from 1999, rejects partial-history resets, and
   preserves the predecessor artifact when validation or simulation fails.
3. **Require verified Elo lineage before weekly prediction. Completed September 18, 2026.**
  Every successful Elo reconstruction now records exact
   games and Elo artifact identities. Weekly availability authenticates both
   artifacts before policy resolution, blocks every current Elo-dependent model
   when lineage is missing or stale, and fails explicitly for malformed
   evidence before forecast persistence.
4. **Resolve affected forecast and product status. Completed September 19,
   2026.** One authenticated immutable schema-1 disposition now records both
   affected Week 2 live logistic Win runs, all 32 affected events, both
   affected weekly products, and the exact selected affected product. The
   disposition classifies Win probability and derived spread as known defective
   because of incomplete Elo source history and blocks readiness, edge
   calculation, candidate issuance, and manual rendering without modifying or
   reselecting historical evidence.
5. **Persist immutable prediction-input evidence.** Link each forecast event to the exact ordered feature row, feature-schema identity, source-artifact hashes, model and scaler identities, optional calibrator identity, and generation timestamp.
6. **Add persisted logistic explanation evidence.** After corrected and reproducible forecasts exist, prefer exact scaled-feature-by-coefficient contributions in log-odds space over Tree SHAP for the logistic champion. Explanations must reconstruct the persisted model output and remain separate from API-time computation.

Acceptance for this program requires history-preserving weekly refresh,
deterministic Elo continuity, verified source and output lineage, explicit
disposition of affected Week 2 artifacts, computationally reproducible forecast
inputs, corrected operational evidence, and an out-of-sample quality assessment
of the corrected path before explanation surfaces are enabled. The first four
requirements are complete.

Potential collateral impact beyond the confirmed Week 2 Elo and Win products remains unconfirmed. Modeling-input construction, historical evaluation, backfill, forecast closeout, production preflight, API loaders, and player game-context consumers should be classified by whether they ran while the cleaned games artifact was truncated. That scope check must not be represented as confirmed impact without runtime or artifact evidence.

#### Parked Production Proof: Market Unit 26

Market Unit 26 remains active but calendar-gated. Its implemented platform, real 2026 Week 1 rehearsal evidence, persisted identities, remaining selected-plan execution, postgame closeout, CLV, realized-performance acceptance, and associated follow-on market capabilities are maintained in:

- `docs/programs/market-unit-26/PLAN.md`
- `docs/programs/market-unit-26/ROADMAP.md`

The complete pre-split planning documents are preserved under `docs/archive/market-program-through-unit-26/`. Root roadmap prioritization may proceed independently while Unit 26 waits for scheduled collection and completed-game evidence.

### Model Ensemble

**Goal:** determine whether a time-ordered ensemble improves operational Win prediction enough to justify additional complexity.

Candidate approaches:

- Brier-weighted averaging;
- constrained blending;
- logistic stacking with time-ordered out-of-fold inputs;
- simple rank or probability averaging as a baseline.

Acceptance should require an honest historical comparison against the current champion, preserved calibration quality, complete upcoming-game feature coverage, deployable artifact metadata, availability inspection, and compatibility with the existing weekly policy and immutable event contracts.

An ensemble should register as another model identity. It must not compute dynamically in the API.

### Injury and News Data

**Goal:** add a reliable, timestamped source for player availability and material team news.

Required design work:

- choose a source and usage policy;
- preserve fetched-at and effective-at timestamps;
- distinguish reported, confirmed, and resolved status;
- map players and teams to canonical identities;
- define historical availability for honest evaluation;
- expose blocked or unavailable states when the source is incomplete.

This capability unlocks injury-aware game and prop presentation and is a prerequisite for credible personnel scenarios.

### Scenario Engine and Feature Attribution

**Goal:** answer bounded what-if and explanation questions without mutating production forecasts.

Potential scope:

- feature contribution or local explanation for persisted predictions;
- comparable historical games;
- controlled team-strength or player-availability adjustments;
- usage redistribution for player props;
- scenario-specific Win, Spread, Total, projected score, and edge calculations;
- explicit separation between persisted production output and hypothetical output.

Scenario computation should use an explicit request and response contract. It must not silently alter the selected weekly product or champion artifacts.

### Real-Time and Live Game Support

**Goal:** support in-game decision analysis.

Required foundations:

- live score, clock, down, distance, possession, and timeout state;
- timestamped live market data;
- a validated live win-probability model;
- live edge and hedge calculations;
- streaming or polling transport;
- strict freshness and stale-state presentation.

This remains lower priority than reliable pregame multi-book data and injury/news integration.

### Remaining API Batch-Artifact Boundaries

**Goal:** ensure every API endpoint serializes persisted artifacts rather than performing meaningful computation at request time.

Known candidate for verification:

- model-performance summaries should be confirmed as batch-produced artifacts; if still computed on request, add a batch writer and serialize its output.

For each candidate:

1. identify the current request-time computation;
2. define the persisted artifact schema and writer;
3. add freshness and provenance;
4. migrate loaders to read the artifact;
5. keep routes and serializers thin;
6. add parity tests before removing the old path.

Do not assume a listed historical deviation still exists. Verify it against current code before scheduling work.

### Frontend Product Enhancements

The core game-day and portfolio surfaces are functional. Remaining work should be pulled by real data availability and user value.

Potential enhancements:

- multi-book line-shopping views;
- injury and news presentation;
- scenario and explanation surfaces;
- line-movement and live-game charts;
- richer bankroll history and Kelly-adherence views;
- recorded-bet export and an explicitly designed recorded-bet write workflow;
- remaining table, layout, and accessibility polish;
- a real-data pending-state visual audit after all required backend artifacts are populated.

BetSlip remains a draft decision workspace. Any recorded-bet write workflow requires duplicate protection, bankroll transaction coupling, partial-failure semantics, and an explicit user action. It is not sportsbook execution.

### Model and Feature Research

Candidate research areas:

- offensive and defensive rating decomposition;
- coaching and coordinator effects;
- pace and neutral-situation tendencies;
- special-teams features;
- penalties, pressure, and situational efficiency;
- additional opponent-quality cohorts;
- richer prop distribution models;
- era-aware feature availability and imputation;
- calibrated uncertainty for ratings and projections.

Every new feature must preserve chronological construction, avoid leakage, and use empirical thresholds rather than arbitrary bins.

### Tooling and CI

Future tooling work:

- restore the intended repository-wide Pyrefly boundary using
  `uvx pyrefly check`;
- define the production, test, script, and exploratory-notebook type-check
  scope explicitly;
- correct repository and test import roots before treating missing test-fixture
  imports as source defects;
- triage configuration failures, production-source findings, shared fixture
  annotations, negative validation tests, Pandas inference limitations, and
  exploratory notebook diagnostics separately;
- establish and enforce a zero-error repository-wide baseline without
  suppressing genuine production defects;
- preserve focused Pyrefly checks during bounded implementation units while the
  repository-wide baseline is being restored;
- exercise `gridiron verify --strict` in a real CI surface;
- run the separate frontend lint, build, and test gates in CI;
- consider performance baselines if test or training runtime regresses;
- maintain generated OpenAPI and TypeScript contract checks;
- improve long-running composite resume diagnostics where needed;
- clamp current-season PBP requests to the maximum season published by the
  upstream source once that policy is defined;
- verify and repair any remaining baseline-report parser edge cases;
- review repository-wide lint exclusions only through dedicated,
  behavior-preserving work.

## Known Limitations

#### Weekly prediction input integrity

The original immutable 2026 Week 2 live logistic Win evidence is formally
classified by schema-1 disposition
`05f36ea9f3f014c3ed2b9bd2f2586540189ac8fea3dcf5db3b354ad020565eeb`
as known defective because its Elo inputs came from incomplete source history.

The affected selected product is blocked from prediction readiness, weekly edge
calculation, candidate issuance, and manual prediction rendering. Original
forecast events, weekly products, index entries, and the scoped selection remain
unchanged and historically loadable.

The Games API and frontend continue to serialize the persisted Week 2 Win
probability and derived spread without disposition metadata. Postgame closeout
also continues to read the affected product as historical evidence of what was
actually selected. These paths are intentionally outside the Unit 4 operational
enforcement boundary.

Exact computational reproduction remains incomplete because forecast events do
not preserve the complete ordered feature row and immutable source, model,
scaler, and optional calibrator identities. Immutable prediction-input evidence
is the next bounded unit. Corrected operational forecasts and model-quality
evaluation must follow that evidence boundary. Explanation surfaces remain
later work.

### Market data

The Odds API v4 client, parser, provider-aware quote contract, partitioned
historical observations, explicit ingest command, sportsbook-specific offer
evaluation, operational edge integration, frontend sportsbook preferences,
Line Shopping, Bet Slip quote identity, kickoff-aware collection planning,
single-shot execution, and active-plan selection are implemented.

A Raspberry Pi quote-collection worker is running through a systemd timer.
Repository-owned deployment assets, installation verification, monitoring,
recovery, and operational artifact synchronization remain active work.
Repeated real quote coverage, validated closeout and CLV, empirical
recommendation thresholds, recommendation product integration, and full
Moneyline, Spread, and Total production proof remain incomplete.

### Injury, news, and live state

There is no integrated injury/news feed or live-game state. Related API and frontend fields must remain explicitly blocked.

### Scenario and explanation

Feature attribution, comparable-game retrieval, and what-if propagation are not implemented.

### Current-season PBP cadence

The upstream source may not publish the current season immediately. Pipeline refresh can warn while continuing with available historical feature state. A future cleanup may clamp requests to the latest published season.

### Postgame timing

`post-week` requires completed outcomes. Running it before games finish correctly exits nonzero and lists missing outcomes.

### Markets versus predictions

A selected weekly product can be prediction-ready while market readiness is blocked. Missing market data means no current edge result; it does not invalidate forecasts.

### File-backed architecture

Files remain appropriate for the current single-user workflow. Revisit this only for real multi-user concurrency, transactional guarantees, or query requirements.

## Prioritization Guidance

The next major work should normally be chosen from:

1. weekly prediction input integrity and reproducibility, next adding immutable
  prediction-input evidence after completing history preservation, complete Elo
  reconstruction, verified lineage, and affected-evidence disposition;
2. supported market provider and multi-book shopping;
3. model ensemble research;
4. injury/news source;
5. scenario engine and explanations;
6. remaining API batch-artifact migrations;
7. frontend enhancements unlocked by real data;
8. real-time and live-game support.

Before starting a new work item:

- verify the gap still exists in current code;
- add it to `PLAN.md` as a bounded execution unit;
- record any locked architectural choice in `DECISIONS.md`;
- update `HANDOFF.md` only after behavior ships;
- record completion in `CHANGELOG.md`.
