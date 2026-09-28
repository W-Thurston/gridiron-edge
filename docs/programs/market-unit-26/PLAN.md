# Market Unit 26 Plan

**Status:** Active, calendar-gated production proof.

**Canonical root documents:** `PLAN.md` and `ROADMAP.md` are available for other active work.

**Historical snapshot:** `docs/archive/market-program-through-unit-26/PLAN.md`.

This document owns the active execution checklist, completed implementation evidence, persisted rehearsal identities, validation record, and final acceptance conditions for Market Unit 26. Update this file whenever Unit 26 evidence changes.

---

### Market Unit 26: Prove the Full Production Recommendation Chain

#### Goal

Prove the complete production recommendation chain for one real completed NFL
week independently for Moneyline, Spread, and Total.

Each market-family proof must begin with one explicitly selected weekly product
and exact forecast provenance, continue through repeated timestamped pregame
quote observations, immutable candidate issuance, versioned qualification and
recommendation policy evaluation, and persisted recommendation or explicit
unavailability, and remain visible through backend serialization and frontend
presentation.

Where an explicit wager is recorded, the proof must preserve duplicate
protection, ledger provenance, and bankroll transaction semantics without
placing a sportsbook wager.

Each proof must finish with a completed game outcome, validated same-provider
and same-sportsbook closeout evidence, market-appropriate closing-line value,
realized performance, and one reproducible chronological audit record.

Moneyline, Spread, and Total require separate empirical acceptance. Evidence
for one market family cannot satisfy another.

#### Implemented Evidence Chain

The real 2026 Week 1 rehearsal now proves the complete pregame and
recommendation middle chain for Moneyline, Spread, and Total.

The explicitly selected weekly product and its immutable forecast provenance
resolve successfully for all 16 scheduled games. The current quote snapshot is
available, and the canonical weekly quote ledger contains 1,680 observations at
two distinct UTC fetch timestamps. Every exact historical identity has repeated
observation depth: 274 Moneyline identities, 282 Spread identities, and 284
Total identities each have two distinct fetch timestamps.

Explicit candidate issuance evaluated every canonical historical observation
before kickoff and persisted one immutable artifact with 698 candidates, 982
not-candidates, and zero unavailable rows. Candidate counts are independent by
market family: 228 Moneyline, 226 Spread, and 244 Total.

Explicit recommendation governance is persisted under a deterministic
content-derived identity. A recommendation policy was derived from the exact
candidate issuance, governed inputs, historical boundaries, current outcome
availability, and available closeout and return evidence. Moneyline, Spread,
and Total each remain explicitly unavailable because required completed-outcome,
closeout, and return evidence does not yet exist.

The exact issuance was evaluated against the exact persisted policy. All 698
candidate rows received immutable recommended-bet results. Every result is
truthfully unavailable, with no qualified, recommended, failed, or conflicting
results and no actionable stake.

The product now separates an immediate model-recommended bet from the stricter
governed qualification result. The backend selects one model-recommended
direction and one globally preferred exact offer for every available game and
market. Moneyline follows the higher modeled win probability. Spread and Total
select the exact wager with the strongest modeled cover probability. Stable
line, price, and exact-offer ordering resolve remaining ties.

The real Week 1 response contains 48 model recommendations across 16 games and
three market families, with exactly one recommendation per game and market and
zero duplicate game-market selections. Line Shopping renders the recommended
exact offer as one green cell. Recommendation badges, repeated lifecycle labels,
persisted suggested-stake callouts, and Policy evidence disclosures were removed
from the primary betting surfaces.

Sportsbook preferences do not change the backend-recommended direction. The
backend marks every exact offer on that direction, while the frontend selects
one best visible offer after applying the user's sportsbook filter. If the
globally preferred sportsbook is deselected, the green highlight moves to the
best remaining offer on the same side. If no selected sportsbook offers that
side, the interface shows no green recommendation rather than switching to the
opposing side.

The immutable governed policy and recommended-bet result artifacts remain a
separate qualification, sizing, exposure, and audit boundary. Their current
Week 1 results remain unavailable because mature empirical outcome, closeout,
CLV, and return evidence does not yet support an active threshold-selection
method. That governed unavailability no longer prevents the model from providing
an immediate pregame recommendation.

Production-chain preflight now validates candidate issuance, recommendation
policy, and recommendation evaluation through strict artifact readers and exact
identity relationships. It reports malformed evidence as invalid and ambiguous
matching evidence as conflicting rather than selecting the latest file by
recency.

Collection-execution readiness now uses the explicitly selected collection plan,
the existing due-state evaluator, and immutable claim and result receipts. The
two manual August 18 quote ingestions establish historical observation depth but
do not count as execution of the selected plan.

Postgame proof assembly is wired once per preflight assessment through the
existing selected-product outcome closeout, exact candidate market closeout,
market-specific CLV, historical-boundary, market-family evaluation, cleaned-game,
and optional settled-wager owners. Before the earliest kickoff, those owners are
short-circuited and all postgame states remain not yet eligible.

#### Persisted Rehearsal Identities

- Candidate issuance:
  `e945987f2903435ac8c798ea5085bd5a39d3ffe2a7741cf5123cabf221e427c0`
- Recommendation governance:
  `56757db59c2d04a55eb3f980299699403fdc982e4fe7ff4963f0898112f4824e`
- Recommendation policy:
  `33255cc82cced0c438fd067ff30261cc4344cfbec2c72f7914124b8a9d3ccb6f`
- Recommended-bet evaluation:
  `100a70a86b73d19c55d5e810355529c553cf690a324ab108bf62ee06ab13b709`
- Persisted production-chain checkpoint:
  `cf860776f0d4bf16804551e4b9ccaa12c91f394d2a9aa1a0dd5877d1bec1eb54`
  (assessed 2026-09-27T23:51:51Z, superseding both the stale 2026-08-18
  checkpoint that referenced now-deleted evidence and the intermediate
  2026-09-27T21:50:23Z checkpoint recorded before collector evidence was
  pulled onto this machine).

**2026-09-27 correction.** A fresh read-only `production-chain assess` found
`candidate_issuance` in state `CONFLICTING`: three immutable issuances existed
for the one selected Week 1 product (evaluated 2026-08-24, 08-26, and 08-27),
none matching the original 2026-08-18 issuance this table used to record. Row
comparison confirmed all three were content-identical re-evaluations of
unchanged forecast and quote-history state -- accidental duplicate
`issue-candidates --write` invocations during development, not a genuine
evidence disagreement. Reference tracing showed only one of the three
(`e945987f...`, 2026-08-27) was ever carried through `derive-policy` and
`evaluate-recommendations` into the 698 persisted `recommended_bet_results`
this document's Implemented Evidence Chain describes; the other two issuances
and their two corresponding orphaned policies
(`69afb8463075b3779134da5b3a603df5eb2bc374211af637943be8535379a4cd`,
`f82b5c6aa1e023d9850b302e8b17179ad32ead7f29553d1407ade02201954c33`) were
unreferenced by anything downstream and were deleted. `issue-candidates` and
`evaluate-recommendations` now refuse `--write` when an issuance or
evaluation already exists for the exact same product or issuance scope, so a
repeat invocation cannot recreate this state.

Resolving the conflict let a fresh assessment reach the postgame-evidence
assembly code for the first time since Week 1 games completed, exposing a
second, independent defect: `_postgame_family_from_evidence` in
`production_chain_preflight.py` built each family's `market_closeout` and
`clv` component timestamps as `sorted(...)` over one `closeout_fetched_at`
value per closeout candidate, without deduplication. Because one
quote-collection poll fetches every game, sportsbook, and market at once,
many closeout candidates legitimately share the same fetch timestamp, so the
component's own sorted-and-unique invariant (`_validate_component`) rejected
the result and crashed the assessment. This was dormant rather than new: the
candidate-issuance conflict (and, before that, the pre-kickoff "not yet
eligible" short-circuit) had always prevented this code from running. Fixed
by deduplicating both timestamp collections through a set before sorting;
regression coverage lives in
`tests/integration/market/test_production_chain_preflight_repository.py`
(updated to assert the corrected `AVAILABLE` states) and the existing unit
suite for `production_chain_preflight.py`.

With both defects fixed, an assessment
(`59330c6a70b85f02b77969fa105299fe72fcd3b9fa15da07dbffbef696d724df`, assessed
2026-09-27T21:50:23Z) resolved `completed_outcome`, `market_closeout`, and
`clv` to `AVAILABLE` for Moneyline, Spread, and Total -- real postgame proof
that was never reachable before that day. `quote_snapshot` was `UNAVAILABLE`
(the global current-snapshot pointer had moved on to Week 3 on this machine)
and `collection_execution` was `INCOMPLETE`, because this dev workspace's own
`data/odds` had no `collection_runs` receipts -- the worker's `data/` is a
separate, gitignored store on the Raspberry Pi.

Checking the Pi directly (manually, over SSH) confirmed all 34 scheduled
Week 1 polls resolved cleanly: every `scheduled_at=*/` folder under
`data/odds/collection_runs/season=2026-2027/week=01/` had both `claim.json`
and a terminal `result.json`, `verify_quote_collection_worker.py` reported
`unresolved_claims: 0`, and the timer had run every five minutes,
uninterrupted, since 2026-08-19. The evidence existed; this workspace simply
couldn't see it. `gridiron ops pull-collector-evidence` was added (one-way,
additive-only `rsync` pull of the Pi's `data/odds` into this workspace's
`data/odds`; never touches `data/output`; see HANDOFF.md's "Collector
Evidence Sync (Dev Machine)" section) and run for real, pulling all 34
receipts, the full Week 1 quote-history ledger, and the Pi's current
snapshot.

Reassessing with that data reached the postgame closeout code with real,
multi-kickoff data for the first time and immediately hit a third defect:
`_validate_component`'s `market_closeout` check compared
`closeout.timestamps[-1]` -- the latest fetch time across *every game in the
family* -- against `closeout.kickoff`, a single arbitrary game's kickoff
(`candidates[0]`). Any week with staggered kickoffs (which is every real NFL
week) has a late game's legitimate pre-kickoff closeout fetch land after an
early game's kickoff, so this cross-game comparison was guaranteed to raise
`Closeout observation must precede kickoff.` on real data. `market_closeout.py`
already guarantees the real per-game invariant correctly (`close_market_reference`
only selects observations with `fetched_at < that reference's own kickoff`);
the broken aggregate re-check in `_validate_component` was removed, and a
correct per-candidate assertion was added in `_postgame_family_from_evidence`
instead, where each candidate's own fetch/kickoff pair is still available.
Regression coverage: `test_postgame_closeout_allows_staggered_kickoffs_across_games`
and `test_postgame_closeout_rejects_observation_at_or_after_its_own_kickoff`
in `tests/unit/market/test_production_chain_preflight.py`.

With all three defects fixed and real collector evidence in place, the
persisted checkpoint above
(`cf860776f0d4bf16804551e4b9ccaa12c91f394d2a9aa1a0dd5877d1bec1eb54`, assessed
2026-09-27T23:51:51Z) resolves every component to `AVAILABLE` for Moneyline,
Spread, and Total except `recorded_wager` and `realized_performance`, which
are validly `UNAVAILABLE` (`no_settled_wager_evidence`) because no wager has
ever been recorded -- exactly the optional, non-blocking terminal state the
design calls for.

#### Current Preflight State

As of the persisted checkpoint
(`cf860776f0d4bf16804551e4b9ccaa12c91f394d2a9aa1a0dd5877d1bec1eb54`, assessed
2026-09-27T23:51:51Z), every component is `AVAILABLE` independently for
Moneyline, Spread, and Total:

- selected weekly product;
- exact forecast provenance;
- current quote snapshot and repeated canonical quote history;
- explicitly selected collection plan;
- selected-plan collection execution (validated terminal receipts for all 34
  scheduled Week 1 polls);
- exact immutable candidate issuance;
- exact persisted recommendation policy;
- exact persisted recommendation evaluation and governed results;
- one backend-owned model-recommended direction and exact offer per game-market;
- 48 real Week 1 model recommendations with no duplicate game-market selection;
- selected-sportsbook fallback within the same recommended direction;
- generated API and frontend contracts for exact-offer and recommended-side evidence;
- one-green-cell Line Shopping presentation with Recommended bet as the only
  default highlight;
- governed policy evidence retained separately from primary presentation;
- completed game outcomes, reconciled to the selected weekly product;
- latest-eligible pregame market closeout;
- market-specific CLV.

The only components not `AVAILABLE` are `recorded_wager` and
`realized_performance`, both explicitly and validly `UNAVAILABLE`
(`no_settled_wager_evidence`) because no wager has ever been recorded.
Gridiron Edge records wagers locally only through explicit user action and
does not place sportsbook wagers; this is optional evidence, not a blocker.

#### Remaining Implementation and Operational Work

1. **Done (2026-09-27).** Persisted a final production-chain checkpoint
   (`cf860776f0d4bf16804551e4b9ccaa12c91f394d2a9aa1a0dd5877d1bec1eb54`)
   against the corrected identities above and the pulled collector evidence,
   superseding both the stale 2026-08-18 checkpoint and the intermediate
   2026-09-27T21:50:23Z checkpoint.
2. **Done (2026-09-27).** Confirmed directly on the Raspberry Pi: all 34
   scheduled Week 1 polls have both `claim.json` and a terminal `result.json`
   under `data/odds/collection_runs/season=2026-2027/week=01/`, zero
   unresolved. `verify_quote_collection_worker.py` reports
   `unresolved_claims: 0` and `latest_service_result: success`; the timer has
   run every five minutes, uninterrupted, since 2026-08-19. One open item:
   the verify script reports overall `Worker status: degraded` on a single
   `warning: storage_health: capacity change` -- not blocking, not yet
   explained, worth a look but not a Unit 26 acceptance blocker.
3. **Done (2026-09-27).** The repository-owned worker executed the selected
   Week 1 plan and persisted immutable claim and terminal-result receipts for
   all 34 due polls (see item 2).
4. **Done (2026-09-27).** Reassessed collection execution from the exact
   receipts, pulled onto this workspace via the new
   `gridiron ops pull-collector-evidence` (one-way, additive-only `rsync` pull
   of the Pi's `data/odds` into this workspace's `data/odds`; never touches
   `data/output`; see HANDOFF.md's "Collector Evidence Sync (Dev Machine)"
   section). `collection_execution` is `AVAILABLE` for all three market
   families in the persisted checkpoint above. An optional `systemd --user`
   timer (`deploy/systemd/gridiron-edge-collector-sync.{service,timer}`) can
   automate future pulls; it still needs a passphrase-free credential path
   (an unlocked `ssh-agent` reachable from the systemd user session, or a
   dedicated passphrase-less key) before it can run unattended, since the
   current key prompts interactively.
5. **Done (2026-09-27).** Refreshed cleaned completed-game outcomes reconcile
   to the selected weekly product; `completed_outcome` is `AVAILABLE` for all
   three market families in the persisted checkpoint above.
6. **Done (2026-09-27).** Every exact candidate reference evaluated against
   the canonical quote ledger using latest-eligible, non-live, strictly
   pre-kickoff evidence; `market_closeout` is `AVAILABLE` for all three
   market families.
7. **Done (2026-09-27).** Moneyline price CLV, Spread point CLV, and Total
   point CLV verified independently, with no cross-family substitution; `clv`
   is `AVAILABLE` for all three market families.
8. Evaluate realized performance only from uniquely attributed settled-wager
   evidence. Preserve unavailable return evidence when no wager was recorded.
   Currently valid and `UNAVAILABLE` (`no_settled_wager_evidence`) because no
   wager has been recorded; this only changes if one is.
9. **Done (2026-09-27).** The checkpoint in item 1 is exactly this postgame
   assessment, persisted at an explicit UTC timestamp
   (`2026-09-27T23:51:51.131315+00:00`); `production-chain verify` reads it
   back without repository reassessment.
10. Complete the final frontend review now that real postgame evidence
    (completed outcomes, closeout, CLV) is available. Presentation cleanup
    may improve density and readability but must not change persisted
    recommendation semantics. Still open -- not yet done.
11. Close Market Unit 26 only after one real completed week satisfies independent
    Moneyline, Spread, and Total acceptance or records an explicit evidence-backed
    unavailable state for a component that cannot validly become available. As
    of the checkpoint above, every component is `AVAILABLE` except the
    validly optional `recorded_wager`/`realized_performance` (item 8) --
    acceptance now appears satisfied pending item 10's frontend review and a
    deliberate decision to close the unit.

#### Tests and Real-Data Validation to Date

- Repository Ruff, Pyrefly, focused unit, integration, API, and frontend gates
  pass for the implemented candidate, governance, policy, recommendation,
  preflight, collection-execution, and postgame-assembly boundaries.
- Two live The Odds API ingestions returned 840 quotes each across 16 games and
  nine sportsbooks and produced a 1,680-row canonical Week 1 history partition.
- Repeated-history validation confirmed two timestamps for every current exact
  Moneyline, Spread, and Total identity.
- The real candidate issuance evaluated all 1,680 historical observations and
  persisted 698 candidates with zero unavailable rows.
- The real policy preserved unavailable family states because completed
  outcomes, closeouts, and returns are not yet available.
- The real recommendation evaluation persisted 698 unavailable results and zero
  recommended results.
- The real Line Shopping response produced 48 immediate model recommendations:
one for each of 16 games across Moneyline, Spread, and Total, with a maximum of
one recommendation per game-market and zero duplicate game-market selections.
- Real-data verification confirmed that Moneyline recommendation direction
follows the higher win probability, including Seattle over New England, the
Rams over San Francisco, and Pittsburgh over Atlanta in the current Week 1
product.
- Frontend verification confirmed that Recommended bet is the only highlight
enabled by default in a clean browser state, one green exact-offer cell presents
the model recommendation, and +EV, best-line, best-price, and model-favorite
layers remain independently optional.
- Sportsbook-filter coverage confirmed that deselecting the globally preferred
book moves the green recommendation to the best remaining offer on the same
backend-recommended side without manufacturing an opposing recommendation.
- Recommendation badges, Policy evidence dropdowns, and primary-surface
persisted suggested-stake callouts remain absent while the underlying governed
audit contracts remain available.
- Exact production-chain verification reads persisted assessments without
  reassessing mutable repository evidence.
- The selected collection plan contains 34 planned polls across six kickoff
  groups, begins at `2026-09-08T12:00:00Z`. All 34 resolved with terminal
  claim/result receipts, confirmed directly on the Raspberry Pi and, after
  pulling that evidence onto this workspace, in a real assessment:
  `collection_execution` reads `AVAILABLE` for all three market families.
- `issue-candidates --write` and `evaluate-recommendations --write` now refuse
  a second immutable artifact for the same exact product or issuance scope
  (`test_issue_candidates_write_rejects_second_issuance_for_same_product`,
  `test_evaluate_recommendations_write_rejects_second_evaluation_for_same_issuance`
  in `tests/unit/cli/test_production_chain_cli.py`), closing the gap that
  produced the 2026-09-27 duplicate-issuance correction above.
- `_postgame_family_from_evidence`'s `market_closeout` and `clv` timestamp
  collections are deduplicated before sorting, fixing a crash
  (`Component timestamps must be sorted and unique.`) that only manifested
  once real closeout candidates sharing a fetch timestamp reached this code
  for the first time.
- The `market_closeout` validation's broken cross-game timestamp check
  (comparing the latest fetch time across every game in the family against
  one arbitrary game's kickoff) was removed and replaced with a correct
  per-candidate assertion in `_postgame_family_from_evidence`
  (`test_postgame_closeout_allows_staggered_kickoffs_across_games`,
  `test_postgame_closeout_rejects_observation_at_or_after_its_own_kickoff` in
  `tests/unit/market/test_production_chain_preflight.py`).
  `tests/integration/market/test_production_chain_preflight_repository.py`
  asserts the real, corrected `AVAILABLE` states against real repository
  data throughout.
- The new `gridiron ops pull-collector-evidence` command and its
  `collector_sync.py` implementation have 13 unit tests covering the exact
  `rsync` invocation built (with/without an identity file, trailing-slash
  handling, `data/output` never touched, `--dry-run`) and CLI wiring
  (`tests/unit/deployment/test_collector_sync.py`,
  `tests/unit/cli/test_ops_cli.py`).
- The final persisted production-chain checkpoint
  (`cf860776f0d4bf16804551e4b9ccaa12c91f394d2a9aa1a0dd5877d1bec1eb54`,
  assessed 2026-09-27T23:51:51Z) confirms real `collection_execution`,
  `completed_outcome`, `market_closeout`, and `clv` evidence for Moneyline,
  Spread, and Total from the real 2026 Week 1 rehearsal.

#### Acceptance

Market Unit 26's evidence chain is now fully proven for the real 2026 Week 1
rehearsal. As of the checkpoint above, every component is `AVAILABLE` for
Moneyline, Spread, and Total independently -- selected product, forecast
provenance, quote snapshot and history, collection plan and execution,
candidate issuance, recommendation policy and result, backend and frontend
serialization, completed outcomes, market closeout, and CLV. The only
non-`AVAILABLE` components, `recorded_wager` and `realized_performance`, are
explicitly and validly `UNAVAILABLE` (`no_settled_wager_evidence`) because no
wager has been recorded -- exactly the optional terminal state the design
calls for, not a gap.

The pregame and recommendation middle chain is implemented and exercised with
real 2026 Week 1 market data. The product provides one immediate
backend-owned model recommendation per available game-market before kickoff,
selects one best visible exact offer within the user's sportsbook preferences,
and presents it as one green cell without badges or policy disclosures.

The stricter governed policy chain remains persisted under exact immutable
identities, strictly revalidated by production-chain preflight, and separate
from the immediate model recommendation.

What remains before formally closing the unit: item 10's final frontend
review (density/readability cleanup against the now-real postgame evidence;
must not change persisted recommendation semantics), and a deliberate
decision to run closeout. Nothing found in this pass indicates the chain
itself is incomplete -- the three defects above were all diagnostic/tooling
gaps (duplicate issuance artifacts, a validation crash, a data-visibility
gap between the Pi and this workspace), not missing evidence.
