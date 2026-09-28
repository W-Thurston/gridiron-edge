# Market Unit 26 Plan

**Status:** Closed 2026-09-27.

**Canonical root documents:** `PLAN.md` and `ROADMAP.md` are available for other active work.

**Historical snapshot:** `docs/archive/market-program-through-unit-26/PLAN.md`.

This document owns the closing implementation evidence, persisted identities, validation record, and acceptance conditions for Market Unit 26.

---

### Market Unit 26: Prove the Full Production Recommendation Chain [Closed 2026-09-27]

#### Completed

Proved the complete production recommendation chain for one real completed
NFL week (2026 Week 1) independently for Moneyline, Spread, and Total, and
fixed three real defects discovered while assembling that proof:

1. **Duplicate candidate-issuance conflict.** Three content-identical
   candidate-issuance artifacts existed for the one selected Week 1 product
   (accidental duplicate `issue-candidates --write` runs during
   development), which made `production-chain assess` report
   `candidate_issuance: CONFLICTING` and cascade to `UNAVAILABLE` for every
   downstream component. Row comparison confirmed the duplicates were
   genuinely redundant (not a data disagreement); reference tracing found
   the one issuance actually carried through to persisted
   `recommended_bet_results`. The two orphaned issuances and their two
   orphaned policies were deleted after confirming nothing referenced them.
   `issue-candidates` and `evaluate-recommendations` now refuse `--write`
   when an artifact already exists for the exact same product or issuance
   scope, so this cannot recur.
2. **Postgame timestamp-dedup crash.** `_postgame_family_from_evidence`
   built `market_closeout`/`clv` timestamp collections via `sorted(...)`
   over one `closeout_fetched_at` value per candidate without
   deduplication. Since one quote-collection poll fetches every game at
   once, many candidates legitimately share a fetch timestamp, so the
   component's own sorted-and-unique validation crashed the assessment the
   first time real closeout candidates reached this code.
3. **Broken cross-game closeout-kickoff check.** A separate validation
   compared the latest closeout fetch time across *every game in a market
   family* against *one arbitrary game's* kickoff — guaranteed to fail on
   any week with staggered kickoffs (every real NFL week). The real
   per-game invariant was already correctly enforced upstream
   (`close_market_reference` only selects observations fetched before that
   exact game's own kickoff); the broken aggregate check was removed and
   replaced with a correct per-candidate assertion.

Separately, resolved ROADMAP.md's D8 (candidate issuance accepted a
non-`live` forecast role or a forecast generated after the issuance's own
`evaluated_at` with no validation) — a defensive gap unrelated to the three
defects above, found while reviewing what closing Unit 26 would also close
at the program level.

The Raspberry Pi worker's collector evidence — previously invisible to this
workspace, since the worker's `data/` is a separate, gitignored store — is
now pullable on demand via the new `gridiron ops pull-collector-evidence`,
confirming all 34 scheduled Week 1 polls resolved with terminal
claim/result receipts, zero unresolved. `gridiron ops rollover-collector-week`
was added as the paired weekly plan-rollover command and validated for real
against Week 3.

#### Goal

Prove the complete production recommendation chain for one real completed
NFL week independently for Moneyline, Spread, and Total: one explicitly
selected weekly product and forecast provenance, repeated timestamped
pregame quote observations, immutable candidate issuance, versioned policy
evaluation, persisted recommendation or explicit unavailability, backend
and frontend serialization, a completed game outcome, validated closeout,
market-appropriate CLV, realized performance where a wager exists, and one
reproducible chronological audit record — with Moneyline, Spread, and Total
requiring separate empirical acceptance.

#### Files Added/Removed/Changed

Added:
- `src/gridiron_edge/deployment/collector_sync.py` - One-way, additive-only `rsync` pull of the deployed worker's `data/odds`, and per-file push (schedule, plan, current-selection pointer, pointer last) for weekly rollover.
- `src/gridiron_edge/cli/ops.py` - `gridiron ops pull-collector-evidence` and `gridiron ops rollover-collector-week` commands.
- `deploy/systemd/gridiron-edge-collector-sync.service` - Optional `systemd --user` service for automating future collector-evidence pulls (not yet enabled; see `DECISIONS.md` D58).
- `deploy/systemd/gridiron-edge-collector-sync.timer` - Paired timer, 15-minute interval with startup catch-up.
- `tests/unit/deployment/test_collector_sync.py` - Coverage for the pull/push `rsync` command construction, including the pointer-last push ordering.
- `tests/unit/cli/test_ops_cli.py` - CLI-level coverage for both `ops` commands.
- `tests/unit/market/test_candidate_issuance_evaluation.py` additions - D8 guard regression tests (non-live role rejected, late-generated forecast rejected).

Changed:
- `src/gridiron_edge/market/production_chain_preflight.py` - Deduplicated `market_closeout`/`clv` timestamp collections before sorting; removed the broken cross-game closeout-kickoff check and added a correct per-candidate assertion in `_postgame_family_from_evidence`.
- `src/gridiron_edge/cli/production_chain.py` - `issue-candidates`/`evaluate-recommendations --write` refuse a second immutable artifact for the same exact scope.
- `src/gridiron_edge/market/candidate_issuance.py` - `issue_pregame_candidates` rejects a non-`live` forecast role or a forecast generated after `evaluated_at` (resolves D8; `DECISIONS.md` D59).
- `src/gridiron_edge/deployment/quote_collection_worker.py` - `storage_health` no longer flags `zram`/`loop` virtual devices' benign boot-time capacity-change messages as a fault.
- `src/gridiron_edge/cli/main.py` - Registered the new `ops` sub-app.
- `tests/unit/cli/test_production_chain_cli.py`, `tests/unit/market/test_production_chain_preflight.py`, `tests/unit/deployment/test_quote_collection_worker.py` - Regression coverage for the three defects and the storage_health fix.
- `tests/integration/market/test_production_chain_preflight_repository.py` - Asserts the real, corrected `AVAILABLE` states against real repository data throughout the closing evidence chain.
- `tests/integration/market/test_candidate_issuance_repository.py` - Row-count assertion tracks real ledger growth instead of a frozen snapshot.
- `tests/integration/api/test_api_contract.py`, `tests/e2e/test_cli_workflows.py` - Two pre-existing, unrelated test failures (a never-valid placeholder game_id; stale expected message text) found and fixed while getting the accumulated 31-commit push green — not Market Unit 26 defects, but blocking the same push.
- `HANDOFF.md` - Rewrote the stale multi-artifact "2026 Week 1 identities" section with the real, resolved single-chain identities and current command examples; corrected stale pre-kickoff "not yet eligible" language; added "Collector Evidence Sync (Dev Machine)" and rewrote "Weekly rollover"; documented the `storage_health` fix and incident.
- `ROADMAP.md` - D8 marked resolved; Track B / Tier 4 #11, #12, #14 and the feature-program gate marked closed; Foundation Completion's acceptance checklist and the program itself closed; "Market proof is incomplete" and "Calendar-Gated Work" limitation sections updated.
- Root `PLAN.md` - Condensed the completed Foundation Completion program (Units 1–16, Track A and B) to a summary, per this repository's major-program-closeout convention; cleared the active-unit slot.
- `DECISIONS.md` - Added D58 (Pi git/passphrase-automation deferrals) and D59 (D8 resolution).
- `CHANGELOG.md` - Added the 2026-09-27 entry for this unit's shipped behavior.
- This file (`docs/programs/market-unit-26/PLAN.md`) - Condensed to this closeout structure.
- `docs/programs/market-unit-26/ROADMAP.md` - Marked the program's proof closed; Unit 26 Acceptance items resolved or validly closed; item 7 (frontend review) reclassified as a Genuine Follow-On capability rather than a closing blocker.

Removed: None.

#### Tests

Ruff, Pyrefly, and the full non-slow unit suite passed throughout (4,226
tests at closing). The full integration+e2e suite (pre-push hook scope)
passed clean (284 passed, 1 unrelated skip) after fixing the two pre-existing
failures noted above. Real-data validation, performed directly against
production infrastructure rather than fixtures:

- Confirmed on the deployed Raspberry Pi worker over SSH: all 34 scheduled
  Week 1 polls resolved with terminal `claim.json`/`result.json` receipts,
  zero unresolved; the timer ran every five minutes, uninterrupted, since
  2026-08-19; `verify_quote_collection_worker.py` reported `Worker status:
  ready` after the `storage_health` fix was deployed there.
- Pulled that evidence onto this workspace via the new
  `gridiron ops pull-collector-evidence` and persisted a fresh
  production-chain checkpoint resolving every component `AVAILABLE`.
- Ran `gridiron ops rollover-collector-week` for real against Week 3 (2
  remaining games, 11 polls, 33 credits), confirmed the push landed
  correctly, and confirmed via the worker's own verify script that it was
  selected and is collecting.

#### Acceptance

The real 2026 Week 1 production-chain checkpoint
(`cf860776f0d4bf16804551e4b9ccaa12c91f394d2a9aa1a0dd5877d1bec1eb54`, assessed
2026-09-27T23:51:51Z) resolves every component to `AVAILABLE` for
Moneyline, Spread, and Total independently: selected product, forecast
provenance, quote snapshot and history, collection plan and execution,
candidate issuance, recommendation policy and result, backend and frontend
serialization, completed outcomes, market closeout, and CLV. The only
non-`AVAILABLE` components, `recorded_wager` and `realized_performance`,
are explicitly and validly `UNAVAILABLE` because no wager has been
recorded — the optional terminal state the design calls for, not a gap.

Persisted closing identities:

- Candidate issuance: `e945987f2903435ac8c798ea5085bd5a39d3ffe2a7741cf5123cabf221e427c0`
- Recommendation governance: `56757db59c2d04a55eb3f980299699403fdc982e4fe7ff4963f0898112f4824e`
- Recommendation policy: `33255cc82cced0c438fd067ff30261cc4344cfbec2c72f7914124b8a9d3ccb6f`
- Recommended-bet evaluation: `100a70a86b73d19c55d5e810355529c553cf690a324ab108bf62ee06ab13b709`
- Production-chain checkpoint: `cf860776f0d4bf16804551e4b9ccaa12c91f394d2a9aa1a0dd5877d1bec1eb54`

Market Unit 26 closes: a real completed week satisfies independent
Moneyline, Spread, and Total proof, with every component either `AVAILABLE`
or an explicit evidence-backed unavailable state that cannot validly become
available otherwise. A final frontend density/readability review remains
open, tracked as a Genuine Follow-On Market Capability in this program's
`ROADMAP.md` — presentation polish, not evidence, and not a closing
blocker. Recommendation-policy maturation (Tier 4 #13) continues
independently on its own calendar.
