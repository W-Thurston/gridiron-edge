# Market Unit 26 Roadmap

**Status:** Active production-proof program.

**Historical snapshot:** `docs/archive/market-program-through-unit-26/ROADMAP.md`.

This document preserves the implemented market platform, the active real-week proof, remaining calendar-gated acceptance, and genuine follow-on market capabilities associated with Market Unit 26. It does not control unrelated future programs in the root `ROADMAP.md`.

---

### Supported Market Provider and Multi-Book Shopping

**Status:** Active program, with Market Unit 26 as the only active bounded unit.

**Goal:** operate a truthful provider-aware, multi-book recommendation product
whose persisted evidence can be followed from selected forecasts and exact
pregame quotes through candidate issuance, policy evaluation, presentation,
optional local wager recording, closeout, CLV, and realized performance.

#### Implemented Platform

The supported current-market and recommendation foundations are implemented:

- The Odds API v4 is the supported current and upcoming NFL market provider.
- The canonical quote contract preserves provider, provider event, sportsbook,
  game, market, side, price, line, local fetch time, sportsbook update time,
  kickoff, and live state.
- Current snapshots and append-preserved season-and-week quote history have
  separate storage and operational semantics.
- Exact replay is idempotent, same-fetch conflicts are rejected, later
  observations remain distinct, and historical loading is deterministic.
- Leakage-safe historical boundaries preserve earliest observed and latest
  eligible non-live pregame evidence independently for each exact
  provider-aware market identity.
- Current ingestion, weekly collection planning, active-plan selection,
  single-shot due execution, immutable claims and results, quota safeguards,
  and partial-persistence reporting are implemented.
- The Raspberry Pi systemd worker deployment, installation, verification,
  secret handling, monitoring, rollback, and recovery assets are
  repository-owned and validated.
- Sportsbook-specific Moneyline, Spread, and Total offers remain independently
  traceable through edge calculation, diagnostics, API, CLI, CSV, Line
  Shopping, Game Detail, Available Edges, and Bet Slip staging.
- Line Shopping preserves every exact offer, classifies line and price quality,
  evaluates exact-offer expected value against the selected weekly product, and
  keeps model likelihood, analytical value, and recommendation semantics
  separate.
- Immutable pregame candidate issuance preserves the exact product, forecast,
  offer, model probability, expected value, evaluation time, state, and reason.
- Exact candidate and recorded-wager references can be closed against the
  latest eligible pregame observation from the same provider, provider event,
  sportsbook, game, market, and side.
- Moneyline price CLV, Spread point CLV, and Total point CLV are calculated only
  from complete validated closeout terms.
- Empirical Moneyline, Spread, and Total evaluation reports outcomes, closeout,
  CLV, observation depth, quote age, cohorts, and exact settled-wager return
  evidence without converting descriptive evidence into intuitive thresholds.
- Immutable recommendation governance, deterministic policy derivation,
  mandatory policy checks, Kelly sizing, bankroll and portfolio constraints,
  and immutable recommended-bet result persistence are implemented.
- Persisted recommendation states are serialized mechanically and presented in
  Line Shopping, Available Edges, Game Detail, and Bet Slip without request-time
  qualification or frontend decision calculation.
- Immediate model recommendations and governed policy results are separate
contracts. The backend owns one model-recommended direction and one globally
preferred exact offer per available game-market. Moneyline uses the higher win
probability; Spread and Total use the strongest exact modeled cover probability.
- Line Shopping presents one recommended exact offer as a green cell, with
Recommended bet as the only highlight enabled by default. +EV, best line, best
price, and model favorite remain independent optional analytical layers.
- Selected-sportsbook filtering preserves the backend-recommended direction and
moves the green highlight to the best remaining exact offer on that side. It
never switches to the opposing side merely because the preferred sportsbook was
deselected.
- Governed qualification, sizing, exposure, checks, and persisted result
artifacts remain immutable audit evidence. Their current unavailable state does
not prevent immediate pregame model recommendations.
- A user may explicitly record a wager locally through one rollback-safe ledger
  and bankroll operation. Gridiron Edge does not place sportsbook wagers.
- Production-chain preflight validates exact selected-product, quote-history,
  collection-plan, candidate, policy, result, backend, frontend, collection
  execution, outcome, closeout, CLV, and performance evidence independently by
  market family.
- The repository-wide Ruff, Pyrefly, Python test, and frontend quality
  boundaries are restored and enforced.

#### Active Proof: Market Unit 26

The active unit is proving the complete production recommendation chain for one
real completed NFL week independently for Moneyline, Spread, and Total.

The real 2026 Week 1 rehearsal has already established:

- one explicitly selected 16-game weekly product with complete selected
  forecast provenance;
- a current 840-row multi-book quote snapshot across 16 games and nine
  sportsbooks;
- a canonical 1,680-row weekly quote ledger at two distinct UTC fetch
  timestamps;
- repeated depth of two for all 274 Moneyline, 282 Spread, and 284 Total exact
  historical identities;
- immutable candidate issuance across all 1,680 observations, containing 698
  candidates, 982 not-candidates, and zero unavailable rows;
- 228 Moneyline, 226 Spread, and 244 Total candidates;
- explicit immutable recommendation governance;
- one persisted family-specific recommendation policy whose three families are
  unavailable because required completed-outcome, closeout, and return evidence
  is not yet available;
- 698 persisted recommended-bet results, all explicitly unavailable and none
  qualified, recommended, failed, or conflicting;
- 698 persisted governed recommended-bet results, all explicitly unavailable
and none qualified, recommended, failed, or conflicting;
- 48 immediate model recommendations across 16 games and Moneyline, Spread, and
Total, with exactly one recommendation per game-market and no duplicates;
- generated API and frontend contracts preserving the globally recommended
exact offer and every offer on the recommended side;
- one-green-cell Line Shopping presentation, selected-sportsbook fallback on
the same recommended direction, and no recommendation badges or Policy evidence
dropdowns on primary betting surfaces;
- strict preflight resolution of the exact candidate issuance, policy, and
  recommendation evaluation;
- shared postgame evidence assembly through existing outcome, closeout, CLV,
  historical-boundary, market-family evaluation, cleaned-game, and optional
  settled-wager owners.

The persisted rehearsal identities are maintained in the active `PLAN.md` unit.

#### Remaining Unit 26 Acceptance

As of the 2026-09-27 checkpoint
(`cf860776f0d4bf16804551e4b9ccaa12c91f394d2a9aa1a0dd5877d1bec1eb54`; full
detail in the active `PLAN.md` unit), items 1-6 below are done against real
2026 Week 1 evidence. The active unit remains open only on item 7 and a
deliberate decision to close:

1. **Done.** Executed due polls from the selected 2026 Week 1 collection plan
   through the repository-owned worker and preserved immutable claim and
   terminal-result receipts for all 34 scheduled polls, confirmed directly on
   the Raspberry Pi. Pulling that evidence onto this dev workspace (new
   `gridiron ops pull-collector-evidence` command) made it visible here too;
   `collection_execution` is `AVAILABLE`.
2. **Done.** Refreshed cleaned completed-game outcomes after Week 1 games
   finished and reconciled them to the exact selected weekly product and
   forecast events; `completed_outcome` is `AVAILABLE`.
3. **Done.** Closed exact issued candidates against the latest eligible
   non-live quote observed strictly before kickoff from the same provider,
   provider event, sportsbook, game, market, and side; `market_closeout` is
   `AVAILABLE`.
4. **Done.** Validated Moneyline price CLV, Spread point CLV, and Total point
   CLV independently; `clv` is `AVAILABLE` for all three market families.
5. Evaluate realized performance only from uniquely attributed settled-wager
   evidence. No recorded wager is required; absent wager evidence must remain
   explicitly unavailable rather than zero. Currently, validly,
   `UNAVAILABLE` -- no wager has been recorded.
6. **Done.** Persisted a chronological production-chain assessment at an
   explicit UTC timestamp (`2026-09-27T23:51:51.131315+00:00`); exact replay
   verified without reassessing mutable repository state.
7. Complete a final real-data frontend presentation review now that outcome,
   closeout, and CLV evidence exists. Presentation cleanup must not change
   persisted recommendation semantics.

Market Unit 26 closes only after a real completed week satisfies independent
Moneyline, Spread, and Total proof or records an explicit evidence-backed
unavailable state for a component that cannot validly become available. Every
component now satisfies that bar for the real 2026 Week 1 rehearsal --
item 7's frontend review and a deliberate decision to run closeout are what
remain.

#### Genuine Follow-On Market Capabilities

The following remain future capabilities after Unit 26:

- supported provider historical backfill if operational collection does not
  produce sufficient depth for empirical calibration;
- empirically validated joint threshold-selection methods and later governed
  policy calibration from matured Moneyline, Spread, and Total evidence, without
  making that maturity a prerequisite for immediate model recommendations;
- line and price movement interpretation beyond validated observed boundaries;
- strategy backtesting with strict chronological evidence controls;
- arbitrage detection with exact executable book, line, price, timing, and stake
  constraints;
- middle detection with explicit paired-line and settlement semantics;
- consensus or cross-book reference policies that do not weaken exact-offer
  provenance;
- frontend density, layout, evidence-detail, and accessibility cleanup informed
  by the real recommendation and postgame evidence experience;
- optional API and frontend appliance hosting on the Raspberry Pi as a separate
  deployment decision.

Current snapshots and historical evidence remain separate workstreams sharing
one normalized quote contract. Forecast publication stays independent from
provider access. Missing market or recommendation evidence remains explicit and
does not invalidate the selected weekly prediction product.

Do not fabricate production prices, collapse provider quotes before
normalization, infer recommendation state from positive expected value, hide a
network fetch inside weekly prediction, treat manual ingestion as selected-plan
execution, or use the retired DraftKings adapter as a dependable recovery path.
