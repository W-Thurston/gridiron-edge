# Gridiron Edge — Development Plan

> **Purpose:** a directed view of what we are currently building. PLAN.md is a
> working document, not a reference. It contains exactly one active workstream
> at a time, broken into tiers, each with its own design block that grows
> during the tier and collapses to a summary on completion.

## Where to find other information

| Document | Role |
|----------|------|
| **PLAN.md** (this file) | The active workstream and its in-flight design |
| **ROADMAP.md** | Long-term strategic direction, workstream inventory, prioritization, known issues & backlog |
| **CHANGELOG.md** | What was built and when (closed workstreams + tier summaries) |
| **HANDOFF.md** | How the system works today (architecture, workflows, operations) |
| **DECISIONS.md** | Architectural decisions made during workstreams |

## Status key

| Tag | Meaning |
|-----|---------|
| Active | Currently being worked |
| Designing | In design phase, not yet implementing |
| Complete | Success criteria met (summary retained inline) |
| Paused | Blocked on a dependency being addressed by another workstream |
| Re-scoped | Success criteria invalidated; replaced or dropped |

Workstream identifiers (W1, W2, …) match ROADMAP.md. They exist only inside the planning docs (PLAN, ROADMAP, DECISIONS, CHANGELOG) — never in source code, comments, or commit subjects.

## Ways of Working

How we operate in every session. A new thread should read this first.

1. **Confirm before building — never assume code exists or looks a certain way.** Before writing or modifying anything, verify the current state. Prefer over-confirming: it's cheaper to paste five files than to rebuild something that already existed and discover the drift later. Use grep to locate; ask the user to run commands or paste files. Assumptions are the #1 source of rework and format drift.

2. **Grep first, then read.** Locate with `grep -rn`, then request the specific file(s) or function(s). Ask the user to run the grep/curl and paste results rather than guessing at signatures, schemas, or data.

3. **Design before implementing, at two levels.**
   - **ROADMAP-level (high-level):** lock the workstream shape — what, why, tiers, success criteria — before touching code.
   - **Subsection-level (deep):** before each tier/substep, a focused design block with locked decisions, then implement.

4. **PLAN.md tracks the active work.** After ROADMAP design, expand the workstream into PLAN.md as a checklist. Check items off as completed.

5. **Commit small units as you go.** Each substep is its own commit with a clear message. Quality gates (`ruff` + `pyrefly` + unit tests, or `pnpm build && pnpm test:run` for frontend) pass before each commit.

6. **Section close-out ritual.** When a section/tier completes: clean its detail out of PLAN.md (collapse to a one-line summary or remove), check the item off in ROADMAP.md, and either continue to the next subsection or repeat the full design→plan→build→close loop for the next big chunk.

7. **Verify against real data.** After backend/data changes, confirm via curl or a Python one-liner against the actual artifact — don't trust the code path alone.

8. **Note on dates:** the assistant's system clock may report an earlier date than reality. Trust commit timestamps and the user; when in doubt, ask.

---

### Current Workstream — W9.11 Tier 2: BetSlip decision-support rebuild

**Status:** Designing

## Purpose

Rebuild BetSlip as a model-informed wager-shortlisting, odds-comparison,
bet-sizing, and recordkeeping workspace.

BetSlip is not a sportsbook execution interface. Gridiron Edge will not connect
to a sportsbook account or claim to place wagers. The interface helps the user:

- collect bets that appear interesting based on model outputs;
- preserve the model and reference-market context behind each selection;
- manually enter or update the odds currently available;
- understand how changed odds affect modeled EV, break-even price, payout, and
  Kelly sizing;
- allocate proposed stakes against an explicit bankroll basis;
- optionally record bets actually made outside Gridiron Edge;
- review P&L and export staged or recorded data.

## Non-goals

- No sportsbook authentication.
- No sportsbook account or balance integration.
- No direct wager placement.
- No claim that a wager is “safe.”
- No fabricated sportsbook price.
- No automatic line shopping until multi-book odds ingestion exists.
- No combined parlay probability, EV, or Kelly without correlation-aware model
  support.
- No implied execution state from adding a leg to the slip.
- No fake Place Bet or Log Bet action before a write contract exists.

## Product language

Use model- and calculation-specific language:

- `Reference price`
- `Current entered price`
- `Model break-even price`
- `Minimum acceptable price`
- `Positive modeled EV at X or better`
- `Current price no longer has positive modeled EV`
- `Model-suggested Kelly sizing`

Do not use:

- `Safe to bet`
- `Guaranteed`
- `Place Bet` unless a real execution capability is intentionally introduced
- `Logged` or `Saved` unless durable persistence succeeded

## Locked architecture

### Draft slip versus recorded bet

The draft BetSlip is temporary decision state:

- selected wagers;
- editable current odds;
- proposed stakes;
- what-if calculations;
- mode and display preferences;
- optional notes.

A recorded bet is durable portfolio history:

- the price actually taken;
- actual stake;
- manually entered sportsbook if desired;
- recorded timestamp;
- open/settled state;
- result;
- realized payout and P&L;
- closing-line context when available.

Draft state and recorded history must remain separate.

### Price provenance

Every leg distinguishes three price concepts:

1. `referenceAmericanOdds`
   - Exact price used by the edge engine when the recommendation was created.
   - Immutable recommendation provenance.
   - Null when no verified price exists.

2. `currentAmericanOdds`
   - User-editable what-if price currently available.
   - Drives current EV, payout, profit, break-even comparison, and Kelly sizing.
   - Initially equals the reference price when one exists.
   - Null until entered when no verified reference price exists.

3. `placedAmericanOdds`
   - Price the user records as actually taken.
   - Exists only on a recorded bet, not an uncommitted draft leg.

Changing the current price must never rewrite the reference price or reference
recommendation metrics.

### Bankroll provenance

The bankroll basis used for recommendations and calculations must be explicit.

Candidate sources:

- backend bankroll ledger via `/portfolio/summary`;
- user-entered calculator bankroll;
- locally configured legacy AppState bankroll.

These sources must not be silently conflated.

The backend ledger bankroll is the preferred source for tracked portfolio
sizing. A manually entered what-if bankroll may be supported but must be labeled
as calculator input.

### Kelly semantics

- `kellyFraction` is full Kelly.
- A visible multiplier such as quarter-Kelly converts full Kelly to a proposed
  sizing fraction.
- Recommended dollar stake must identify the bankroll and multiplier on which
  it is based.
- Editing current odds recalculates the current Kelly fraction and proposed
  stake.
- User-entered stake remains distinct from the model-suggested stake.
- Kelly output is advisory model math, not a claim of safety.

### Singles and parlays

Singles:

- calculate payout and profit only from a verified or user-entered current
  price;
- calculate current modeled EV only when model probability and current price
  are available;
- calculate Kelly only when model probability, current price, bankroll, and
  multiplier are available.

Parlays:

- calculate quoted combined payout only when every leg has a current price;
- do not calculate combined model probability;
- do not calculate combined EV;
- do not calculate combined Kelly;
- display that leg correlation is not modeled;
- retain per-leg model and sizing context.

### Missing-data discipline

- Missing price is a visible unavailable state, never a default `-110`.
- Missing line is not zero.
- Missing probability blocks EV and Kelly calculations.
- Missing bankroll blocks dollar sizing but not Kelly fraction.
- Prop lines and prices remain visibly pending until prop odds ingestion lands.
- Props may still be collected as interesting model projections without being
  represented as priced wagers.

---

## Verified current limitations

- BetSlip is persisted in unvalidated localStorage.
- Existing IDs describe producer screens rather than canonical wagers.
- The same wager can be added multiple times from different screens.
- The declared leg type supports only moneyline, spread, and total.
- Prop producers bypass the type with `market: "prop" as never`.
- Several producers fabricate `-110`.
- `market_value` is not the offered American price:
  - moneyline: no-vig market probability;
  - spread: home-team market spread;
  - total: market total.
- The edge engine retains the real price as `edge.odds`, but the edge report
  currently discards it.
- BetSlip drops model value/probability, EV, edge strength, Kelly fraction,
  Kelly stake, model identity, and recommendation context.
- Stake is local to `SlipPanel` and resets on remount.
- AppState bankroll, `/edges` bankroll, and backend ledger bankroll are
  currently separate concepts.
- `/edges` defaults to a $1,000 bankroll and 0.25 Kelly multiplier unless
  explicit query values are supplied.
- The portfolio API is currently read-only.
- Line-shopping and prop-shop APIs remain blocked on odds ingestion.
- The current interface has no sportsbook execution action.

---

## Tier 2 implementation plan

### A. Preserve reference price and sizing context in `/edges` — ✅ COMPLETE

- [x] Add `american_odds` to the edge report columns.
- [x] Populate `american_odds` from `edge.odds` for moneyline, spread, and total edges.
- [x] Preserve existing `market_value` semantics.
- [x] Document the distinction between market context and offered price.
- [x] Add required `american_odds` to `EdgeRow`.
- [x] Add response-level `bankroll` to `EdgeList`.
- [x] Add response-level `kelly_multiplier` to `EdgeList`.
- [x] Echo sizing context for populated and blocked empty responses.
- [x] Add recommendation-report coverage for all three market types.
- [x] Verify positive and negative American price preservation.
- [x] Verify market value and American price are not conflated.
- [x] Add serializer and route coverage.
- [x] Regenerate OpenAPI and TypeScript contracts.
- [x] Run backend and frontend contract gates.
- [x] Commit the edge-price contract unit.

The edge report now preserves `edge.odds` as required `american_odds`, while `market_value` retains its market-specific meaning: no-vig probability for moneyline, home-team spread for spread, and market total for total. `/edges` also returns the bankroll and Kelly multiplier used to calculate `kelly_stake`, including on blocked and legitimate-empty responses.


### B. Define the BetLeg v2 domain model

- [ ] Add a discriminated union:
  - game wager;
  - prop interest/wager.
- [ ] Separate immutable recommendation provenance from editable draft inputs.
- [ ] Add canonical game-wager IDs.
- [ ] Add canonical prop-wager IDs.
- [ ] Keep producer/source metadata outside canonical identity.
- [ ] Add `referenceAmericanOdds`.
- [ ] Add editable `currentAmericanOdds`.
- [ ] Add nullable sportsbook text for manual entry only.
- [ ] Add model identity.
- [ ] Add model probability or applicable fair-value context.
- [ ] Add reference EV.
- [ ] Add reference edge strength.
- [ ] Add reference full-Kelly fraction.
- [ ] Add reference Kelly stake and its bankroll/multiplier basis.
- [ ] Add game-specific market, side, line, game ID, and teams.
- [ ] Add prop-specific prop ID, game ID, player, team, stat type, side, line,
      and projection mean.
- [ ] Represent unpriced props truthfully.
- [ ] Add pure constructors from generated edge and prop API shapes.
- [ ] Add pure current-price calculation helpers.
- [ ] Add constructor, canonical-ID, and calculation tests.
- [ ] Commit the BetLeg v2 pure-foundation unit.

### C. Add validated draft persistence

- [ ] Introduce a versioned storage key.
- [ ] Add runtime parsing for persisted v2 legs.
- [ ] Reject malformed legs.
- [ ] Do not treat prototype placeholder prices as verified reference prices.
- [ ] Decide whether to discard or visibly reset legacy v1 state.
- [ ] Persist local mode.
- [ ] Decide whether editable current odds persist.
- [ ] Decide whether proposed stakes persist.
- [ ] Preserve canonical deduplication after reload.
- [ ] Add storage parse, rejection, persistence, and deduplication tests.
- [ ] Commit the validated-context unit.

### D. Migrate all leg producers

- [ ] Migrate BetSlip `EdgesTable`.
- [ ] Migrate Dashboard Featured Matchups.
- [ ] Migrate Dashboard Model Edges.
- [ ] Migrate Dashboard Prop Edges.
- [ ] Migrate GameDetail Model Lean.
- [ ] Migrate GameDetail Top Prop Edges.
- [ ] Migrate PlayerProp.
- [ ] Remove all fabricated `-110` values.
- [ ] Remove all `market: "prop" as never` workarounds.
- [ ] Remove producer-specific wager IDs.
- [ ] Preserve producer source as metadata.
- [ ] Verify adding the same wager from different screens deduplicates.
- [ ] Verify game and prop selections remain distinguishable.
- [ ] Add producer integration coverage.
- [ ] Commit the producer-migration unit.

### E. Lock current-price and calculation behavior

- [ ] Add editable current American odds per leg.
- [ ] Preserve the immutable reference price beside the editable current price.
- [ ] Calculate implied probability from current price.
- [ ] Calculate current modeled EV when probability and price are available.
- [ ] Calculate model break-even American price.
- [ ] Show whether the current price remains positive modeled EV.
- [ ] Recalculate full-Kelly fraction from current price.
- [ ] Apply a visible editable Kelly multiplier.
- [ ] Calculate proposed Kelly dollars from the explicit bankroll basis.
- [ ] Keep user-entered stake distinct from suggested Kelly stake.
- [ ] Calculate single-leg payout and profit.
- [ ] Block price-dependent outputs when current price is unavailable.
- [ ] Add parity tests against backend odds/EV/Kelly examples.
- [ ] Commit the calculation-model unit.

### F. Lock bankroll behavior

- [ ] Use `/portfolio/summary.bankroll` as the preferred tracked bankroll.
- [ ] Define behavior when the ledger bankroll is unavailable.
- [ ] Decide whether to allow a clearly labeled what-if bankroll override.
- [ ] Pass explicit bankroll and Kelly multiplier to `/edges`.
- [ ] Ensure `kelly_stake` and displayed bankroll use the same basis.
- [ ] Relabel or remove the legacy AppState bankroll to prevent ambiguity.
- [ ] Surface bankroll source in the BetSlip.
- [ ] Add bankroll-loading, fallback, and mismatch tests.
- [ ] Commit the bankroll-contract unit.

### G. Rebuild the BetSlip presentation

- [ ] Preserve the Available Edges / BetSlip decision-workspace structure where
      useful.
- [ ] Add clear empty-state guidance.
- [ ] Render game and prop legs with distinct identity treatments.
- [ ] Show reference price.
- [ ] Add editable current price.
- [ ] Show price-unavailable state without substituting a price.
- [ ] Show model probability or fair value.
- [ ] Show model break-even/minimum acceptable price.
- [ ] Show reference EV and current what-if EV.
- [ ] Show edge strength.
- [ ] Show full-Kelly fraction.
- [ ] Show multiplier-adjusted suggested stake.
- [ ] Add editable proposed stake.
- [ ] Show payout and profit only when calculable.
- [ ] Explain why a calculation is unavailable.
- [ ] Retain remove-leg and clear-slip actions.
- [ ] Preserve singles/parlay mode.
- [ ] Add parlay correlation caveat.
- [ ] Do not render a fake Place Bet action.
- [ ] Add responsive/narrow-width behavior.
- [ ] Add keyboard and accessible-label coverage.
- [ ] Commit the BetSlip presentation unit.

### H. Optional recorded-bet workflow

This section requires separate backend design and is not required to complete
the visual/calculation rebuild.

- [ ] Inspect the betting-ledger append contract.
- [ ] Define a frontend-safe recorded-bet request.
- [ ] Add duplicate protection.
- [ ] Define bankroll transaction coupling.
- [ ] Define partial failure semantics.
- [ ] Add a write API only if the contract is intentionally approved.
- [ ] Label the action `Record Bet`, not `Place Bet`.
- [ ] Capture placed odds, stake, optional sportsbook, and timestamp.
- [ ] Refresh portfolio queries after successful recording.
- [ ] Keep sportsbook execution outside Gridiron Edge.

### I. P&L and export

- [ ] Preserve existing portfolio P&L as the historical source.
- [ ] Add CSV export for recorded bets if the ledger-write workflow lands.
- [ ] Consider draft-slip CSV export for manual recordkeeping.
- [ ] Include canonical wager ID, market, side, line, reference price, current or
      placed price, stake, model probability, EV, Kelly context, and status.
- [ ] Do not export inferred execution data.
- [ ] Add export-format tests.

### J. Verification and close-out

- [ ] Verify a priced moneyline edge.
- [ ] Verify a priced spread edge.
- [ ] Verify a priced total edge.
- [ ] Verify an unpriced prop interest.
- [ ] Verify manual odds adjustment changes EV and Kelly.
- [ ] Verify crossing the break-even threshold changes the visible state.
- [ ] Verify unpriced legs block payout and Kelly dollars.
- [ ] Verify a fully priced parlay calculates payout.
- [ ] Verify an incompletely priced parlay blocks combined payout.
- [ ] Verify no combined parlay EV/Kelly is shown.
- [ ] Verify canonical deduplication across producer screens.
- [ ] Verify draft persistence and invalid-storage recovery.
- [ ] Verify tracked-bankroll provenance.
- [ ] Verify CSV output if included.
- [ ] Complete a real-data visual pass.
- [ ] Run backend and frontend quality gates.
- [ ] Close Tier 2 documentation.

## Success criteria

Tier 2 is complete when:

- no producer fabricates a sportsbook price;
- every staged leg has canonical identity and explicit provenance;
- game and prop selections are represented without type escapes;
- current odds can be edited without changing recommendation history;
- threshold, EV, Kelly, payout, and profit outputs are shown only when their
  required inputs are available;
- suggested stake identifies its bankroll and Kelly-multiplier basis;
- parlay output does not imply correlation-aware model support;
- the UI does not imply sportsbook execution;
- the same wager deduplicates across producer screens;
- local persistence is versioned and validated;
- the real-data visual review is clean;
- focused and full quality gates pass.

---

## Paused Workstreams

_(none currently paused)_

---

#### Tier 1 — PlayoffProjections rebuild — ✅ COMPLETE (2026-07-28)

Rebuilt `/projections` as a sortable, filterable simulation-probability
surface grounded in the existing projections and team-state contracts.

Shipped:

- Full-cell fixed-scale HeatCell probability matrix across all five postseason
  stages.
- Accessible SortableHeader primitive with active direction and nulls-last
  behavior.
- All-conference / AFC / NFC filtering with a dependent
  North / South / East / West division selector.
- Current Elo, one-week Elo adjustment, current record, average wins, and
  conference/division context.
- Season, as-of week, simulation count, and computed-time metadata.
- Explicit `elo_delta` API naming in place of the ambiguous
  `week_over_week_delta`.
- Expected Week 1 Elo-delta absence shown quietly with one explanatory legend
  caveat; unexpected later-season absence remains highlight-aware.
- Team-profile navigation through the existing NavContext route.
- Horizontal overflow preserving every postseason stage on narrow viewports.
- Comprehensive primitive, API-contract, serializer, route, and screen tests.

Verified against the real 32-team Week 1 artifact. All targeted backend gates,
frontend build, and frontend tests pass. Repository-wide Ruff still reports
pre-existing PLR0917 findings outside this workstream.

#### Tier 1 follow-up — Projections navigation and weekly outcomes — ✅ COMPLETE (2026-07-28)

Extended the PlayoffProjections rebuild with explicit sibling navigation
between Team Rankings and Playoff Projections and a league-wide Weekly
Outcomes view.

Shipped:

- Shared Team Rankings / Playoff Projections navigation on both screens.
- Static `GET /projections/grid` contract composed from the weekly simulation
  grid, schedule, completed results, and unified team mappings.
- Explicit played, projected, bye, and unavailable weekly states.
- Full-cell Week 1–18 win-probability matrix using a fixed diverging
  red-neutral-green scale centered at 50%.
- Grouped Played Games / Projected Games headers with a clear transition
  boundary.
- Sticky team column using the same team identity and surface treatment as the
  Playoff Chances view.
- Conference and dependent division filtering shared across both projections
  views, with selections preserved while switching views.
- Viewport-portaled matchup tooltips with top/left/right edge clamping,
  responsive width, and centered matchup, schedule, and outcome rows.
- Pointer-hover and keyboard-focus access to opponent, home/away perspective,
  week, date, time, probability, state, and actual result.
- Explicit BYE treatment distinct from a zero-percent game.
- Generated OpenAPI and TypeScript contracts for the new endpoint.
- Sortable Team and Week 1–18 columns, with three-state weekly sorting
  (highest probability, lowest probability, then team-name order) and
  bye/unavailable rows always last.

Verified against the real 32-team 2026–2027 preseason artifact. Backend and
frontend quality gates pass, and the final real-data visual review is clean.

Probability-cell texture remains deferred pending color-vision review because
numeric percentages, explicit bye states, grouped headers, and accessible
labels already provide non-color encodings.

---

## Changelog

| Date | Change |
|------|--------|
| 2026-07-28 | Closed the PlayoffProjections navigation and Weekly Outcomes follow-up after real-data verification. |
| 2026-07-12 | **Doc-sync pass + PLAN reset.** Normalized planning docs after the frontend arc; PLAN reset to between-workstreams with a next-candidates list (audit sweep recommended). |
| 2026-07-11 | **W9.10 complete.** Both Compare modes shipped on backend B1–B4. Fixed game_id scramble, clean-games clobber, Elo empty-games crash; added champion→elo fallback + upcoming-Week-1 season resolver. |
| 2026-07-11 | **W9.10 status resync.** Team vs Team complete (6 alignment adjustments + 11-metric cohort_splits). Player vs Defense redesigned to independent pickers + bar chart + baseline verdict. |
| 2026-07-11 | **W9.10 design locked.** Two modes; highlight discipline baked in from W9.8. |
