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


### B. Define the BetLeg v2 domain model — ✅ COMPLETE

- [x] Add a discriminated union:
  - game wager;
  - prop interest/wager.
- [x] Separate immutable recommendation provenance from editable draft inputs.
- [x] Add canonical game-wager IDs.
- [x] Add canonical prop-wager IDs.
- [x] Keep producer/source metadata outside canonical identity.
- [x] Add `referenceAmericanOdds`.
- [x] Add editable `currentAmericanOdds`.
- [x] Add nullable sportsbook text for manual entry only.
- [x] Add model identity.
- [x] Add model probability or applicable fair-value context.
- [x] Add reference EV.
- [x] Add reference edge strength.
- [x] Add reference full-Kelly fraction.
- [x] Add reference Kelly stake and its bankroll/multiplier basis.
- [x] Add game-specific market, side, line, game ID, and teams.
- [x] Add prop-specific prop ID, game ID, player, team, stat type, side, line,
      and projection mean.
- [x] Represent unpriced props truthfully.
- [x] Add pure constructors from generated edge and prop API shapes.
- [x] Add pure current-price calculation helpers.
- [x] Add strict v2 runtime parsers.
- [x] Add constructor, canonical-ID, calculation, and parser tests.
- [x] Commit the BetLeg v2 pure-foundation unit.

BetLeg v2 is a versioned discriminated union for game wagers and prop
interests. Canonical IDs describe the wager rather than the producer screen.
Immutable recommendation provenance—including reference price, model context,
EV, and Kelly sizing basis—is stored separately from editable draft odds,
stake, sportsbook, and notes. Props remain explicitly unpriced until real odds
exist. Pure helpers provide guarded EV, Kelly, payout, profit, break-even, and
price-quality calculations, while strict parsers reject legacy or malformed
persisted state.

### C. Add validated draft persistence — ✅ COMPLETE

- [x] Introduce versioned leg and mode storage keys.
- [x] Add strict runtime parsing for persisted v2 legs.
- [x] Reject malformed legs individually.
- [x] Ignore legacy prototype state rather than treating placeholder prices as verified.
- [x] Preserve valid single/parlay mode.
- [x] Validate runtime additions defensively.
- [x] Deduplicate by canonical wager ID.
- [x] Add loading, rejection, add, remove, clear, persistence, mode, and deduplication coverage.

The live context now reads and writes validated v2 draft state. Legacy
prototype storage is intentionally ignored because it may contain fabricated
prices, undeclared prop markets, incorrect game identifiers, and
producer-specific IDs.

### D. Migrate all leg producers — ✅ COMPLETE

- [x] Migrate BetSlip `EdgesTable`.
- [x] Migrate Dashboard Featured Matchups.
- [x] Migrate Dashboard Model Edges.
- [x] Migrate Dashboard Prop Edges.
- [x] Migrate GameDetail Model Lean.
- [x] Migrate GameDetail Top Prop Edges.
- [x] Migrate PlayerProp.
- [x] Remove all fabricated `-110` values.
- [x] Remove all `market: "prop" as never` workarounds.
- [x] Remove producer-specific wager IDs.
- [x] Preserve producer source as metadata.
- [x] Prevent missing or `No Edge` prop leans from defaulting to Over.
- [x] Verify source-independent canonical deduplication.
- [x] Adapt the existing slip panel to render v2 game and prop legs.
- [x] Block aggregate payout and combined-odds output when any leg is unpriced.
- [x] Run focused and full frontend quality gates.

All live producers now create canonical v2 legs through shared constructors.
Game edges retain the exact price and sizing context returned by `/edges`.
Props retain their actual game/player/projection context and remain explicitly
unpriced until real prop odds exist. The current SlipPanel is compatibility-safe
but has not yet received the final decision-support redesign.

### E. Lock current-price and calculation behavior — ✅ COMPLETE

- [x] Add editable current American odds per leg.
- [x] Preserve the immutable reference price beside the editable current price.
- [x] Calculate implied probability from current price.
- [x] Calculate current modeled EV when probability and price are available.
- [x] Calculate model break-even American price.
- [x] Show whether the current price remains positive modeled EV.
- [x] Recalculate full-Kelly fraction from current price.
- [x] Apply an explicit Kelly multiplier as a calculation input.
- [x] Calculate proposed Kelly dollars from an explicit bankroll basis.
- [x] Keep user-entered stake distinct from suggested Kelly stake.
- [x] Calculate single-leg payout and profit.
- [x] Block price-dependent outputs when current price is unavailable.
- [x] Preserve price and payout calculations while blocking EV/Kelly when model probability is unavailable.
- [x] Add draft-update validation and persistence coverage.
- [x] Add parity tests against established odds, EV, and Kelly examples.
- [x] Commit the calculation-model unit.

BetSlip draft state now supports validated edits to current odds, proposed
stake, sportsbook, and notes without changing immutable recommendation
provenance. A pure leg-analysis helper derives reference and current
calculations, model break-even price, current-price acceptability, full Kelly,
multiplier-adjusted suggested dollars, payout, and profit. Bankroll and Kelly
multiplier remain explicit inputs so their authoritative source can be resolved
independently in Section F.

### F. Lock bankroll behavior — ✅ COMPLETE

- [x] Use `/portfolio/summary.bankroll` as the preferred tracked bankroll.
- [x] Define behavior when the ledger bankroll is unavailable.
- [x] Decide whether to allow a clearly labeled what-if bankroll override.
- [x] Pass explicit bankroll and Kelly multiplier to `/edges`.
- [x] Ensure `kelly_stake` and displayed bankroll use the same basis.
- [x] Relabel or remove the legacy AppState bankroll to prevent ambiguity.
- [x] Surface bankroll source in the BetSlip.
- [x] Add bankroll-loading, fallback, and mismatch tests.
- [x] Commit the bankroll-contract unit.

#### F1. Require an explicit bankroll for edge dollar sizing — ✅ COMPLETE

- [x] Remove the hidden `$1,000` bankroll default from `/edges`.
- [x] Allow bankroll to be omitted explicitly.
- [x] Preserve edge rows, EV, and full-Kelly fraction when bankroll is omitted.
- [x] Return `kelly_stake = null` when no bankroll basis is supplied.
- [x] Preserve zero as a valid bankroll with zero-dollar Kelly sizing.
- [x] Constrain bankroll to nonnegative values at the HTTP boundary.
- [x] Constrain Kelly multiplier to `[0, 1]` at the HTTP boundary.
- [x] Validate bankroll and multiplier again at the report boundary.
- [x] Echo null bankroll provenance through populated and blocked responses.
- [x] Add report and route coverage for omitted, zero, and invalid sizing inputs.
- [x] Regenerate OpenAPI and TypeScript contracts.
- [x] Run backend and frontend quality gates.
- [x] Commit the nullable-sizing contract unit.

`/edges` no longer substitutes a hidden `$1,000` sizing basis. When bankroll is
omitted, recommendations still expose model economics and full-Kelly fraction,
but dollar Kelly stake remains explicitly unavailable. A supplied zero bankroll
is preserved as a valid tracked state rather than interpreted as missing.

#### F2a. Define the BetSlip sizing preference model — ✅ COMPLETE

- [x] Add a versioned sizing-preference contract.
- [x] Default to tracked-bankroll mode.
- [x] Add an explicit what-if bankroll mode.
- [x] Preserve zero as a valid tracked or what-if bankroll.
- [x] Add an explicit Kelly multiplier constrained to `[0, 1]`.
- [x] Default to quarter-Kelly (`0.25`).
- [x] Resolve bankroll amount with explicit tracked, what-if, or unavailable provenance.
- [x] Prevent tracked mode from silently falling back to a what-if bankroll.
- [x] Prevent what-if mode from silently falling back to the tracked bankroll.
- [x] Reject malformed persisted sizing state.
- [x] Reject invalid updates while preserving the prior valid preference.
- [x] Add focused parser, resolution, update, boundary, and default tests.
- [x] Run the frontend build and full test suite.
- [x] Commit the pure sizing-preference unit.

The pure sizing model now distinguishes tracked, what-if, and unavailable
bankroll sources. It preserves zero as a valid amount, constrains Kelly
multipliers to `[0, 1]`, defaults to quarter-Kelly, and prevents silent source
fallback. React persistence, portfolio loading, and BetSlip integration remain
isolated to the next unit.

#### F2b. Add persisted tracked/what-if sizing orchestration — ✅ COMPLETE

- [x] Load the tracked bankroll from `/portfolio/summary`.
- [x] Preserve zero as a valid tracked bankroll.
- [x] Load and validate `hm-betslip-sizing-v1`.
- [x] Fall back to the default tracked quarter-Kelly preference for malformed persisted state.
- [x] Resolve tracked, what-if, or unavailable bankroll provenance.
- [x] Expose tracked-bankroll loading and error state.
- [x] Add validated mode, what-if bankroll, and Kelly-multiplier updates.
- [x] Persist valid sizing-preference changes.
- [x] Prevent silent source fallback.
- [x] Keep the legacy AppState bankroll outside BetSlip sizing resolution.
- [x] Add focused loading, persistence, source-resolution, update, error, and zero-value coverage.
- [x] Run the frontend build and full test suite.
- [x] Commit the sizing-hook unit.

The query-aware sizing hook now combines the tracked portfolio bankroll with
the versioned tracked/what-if preference. It exposes the effective bankroll,
its explicit provenance, the Kelly multiplier, and tracked loading/error state
without consulting the legacy AppState bankroll. Live BetSlip integration
remains isolated to F2c.

#### F2c. Integrate sizing provenance into the live BetSlip — ✅ COMPLETE

- [x] Extend `useEdges()` with optional bankroll and Kelly-multiplier query parameters.
- [x] Resolve sizing once at the BetSlip screen boundary.
- [x] Pass the same effective bankroll and Kelly multiplier to edge generation and slip analysis.
- [x] Omit bankroll from `/edges` requests when sizing is unavailable.
- [x] Preserve zero as an explicit bankroll.
- [x] Preserve backend-returned bankroll and multiplier as immutable recommendation provenance on newly staged legs.
- [x] Surface tracked, what-if, and unavailable bankroll sources.
- [x] Add minimal tracked/what-if source controls.
- [x] Add a validated what-if bankroll input.
- [x] Add an explicit Kelly-multiplier selector.
- [x] Surface tracked-bankroll loading and error states.
- [x] Keep the legacy AppState bankroll outside the BetSlip sizing path.
- [x] Relabel AppState bankroll as a standalone calculator value in Settings and Onboarding.
- [x] Add focused screen, edge-query, sizing-hook, and Settings coverage.
- [x] Verify no hidden bankroll fallback or substitution remains.
- [x] Run the frontend build, focused tests, and full test suite.
- [x] Commit the live sizing-integration unit.

The BetSlip now resolves one explicit sizing basis and shares it between edge
generation and staged-wager analysis. Tracked portfolio bankroll is preferred,
what-if sizing requires an intentional override, and unavailable sizing remains
unavailable without falling back to AppState or a hidden default. Newly staged
recommendations retain the exact bankroll and Kelly multiplier returned by the
backend.

### G. Rebuild the BetSlip presentation — 🔵 ACTIVE

#### G1. Rebuild staged-wager decision cards — ✅ COMPLETE

- [x] Replace compact game-only rows with discriminated game and prop cards.
- [x] Show immutable reference price beside editable current price.
- [x] Show model identity and probability.
- [x] Show model break-even price.
- [x] Show positive-EV threshold status.
- [x] Show reference and recalculated current EV.
- [x] Show edge strength.
- [x] Show current full-Kelly fraction.
- [x] Show multiplier-adjusted suggested stake.
- [x] Add editable proposed stake.
- [x] Show single-leg payout and profit when calculable.
- [x] Represent unpriced prop reference data explicitly.
- [x] Add optional manual sportsbook and draft-note inputs.
- [x] Route all editable values through validated draft updates.
- [x] Preserve remove-leg behavior.
- [x] Add focused game, prop, editing, unavailable-state, and accessibility coverage.
- [x] Run the frontend build, focused tests, and full suite.
- [x] Commit the staged-wager card unit.

Each staged wager is now a decision-support card rather than a compact display
row. The card separates immutable reference recommendation context from editable
current odds and proposed stake, then derives current EV, Kelly, break-even,
payout, and profit through the pure analysis contract. Props remain visibly
unpriced until a current price is entered.

#### G2a. Define aggregate single/parlay summary behavior — ✅ COMPLETE

- [x] Add a pure discriminated single/parlay summary model.
- [x] Require current odds and proposed stake for every single leg.
- [x] Sum complete single-leg stakes, payouts, and profits.
- [x] Block all aggregate single economics when any leg is incomplete.
- [x] Preserve zero as a valid proposed stake.
- [x] Require current odds for every parlay leg.
- [x] Require a separate explicit parlay stake.
- [x] Calculate quoted combined decimal and American parlay odds.
- [x] Calculate quoted parlay payout and profit.
- [x] Ignore per-leg proposed stakes in parlay mode.
- [x] Preserve zero as a valid parlay stake.
- [x] Keep combined parlay probability, EV, and Kelly explicitly unavailable.
- [x] Add empty, complete, incomplete, zero-value, and mixed game/prop coverage.
- [x] Run the frontend build and full test suite.
- [x] Commit the aggregate-summary model unit.

The aggregate summary model now reports single-wager economics only when every
staged leg has current odds and a proposed stake. Parlay mode uses a separate
stake and calculates quoted combined odds, payout, and profit only when every
leg is priced. Combined parlay model probability, EV, and Kelly remain
explicitly unavailable because leg correlation is not modeled.

#### G2b. Rebuild aggregate summaries and execution boundary — ✅ COMPLETE

- [x] Replace compatibility calculations with the pure aggregate-summary model.
- [x] Use each leg's proposed stake in singles mode.
- [x] Require every single leg to have current odds and proposed stake.
- [x] Show total proposed stake, potential payout, and potential profit only when the entire singles set is complete.
- [x] Explain missing current-price and proposed-stake inputs.
- [x] Add a separate parlay-stake input.
- [x] Ignore per-leg proposed stakes in parlay mode.
- [x] Require current odds for every parlay leg.
- [x] Show quoted combined odds, payout, and profit only when the parlay is complete.
- [x] Explain missing parlay-price and parlay-stake inputs.
- [x] Add the parlay-correlation caveat.
- [x] Keep combined parlay probability, EV, and Kelly unavailable.
- [x] Retain remove-leg and clear-slip actions.
- [x] State explicitly that Gridiron Edge does not place sportsbook wagers.
- [x] Verify that no execution action is rendered.
- [x] Add focused singles, parlay, incomplete-input, stake-editing, caveat, clear-slip, and no-execution coverage.
- [x] Run the frontend build, focused tests, and full suite.
- [x] Commit the aggregate-summary presentation unit.

The BetSlip now derives all aggregate economics from the pure summary model.
Singles use each leg's proposed stake and remain incomplete until every staged
wager has current odds and a stake. Parlays use a separate stake and expose
only quoted combined odds, payout, and profit. Combined model probability, EV,
and Kelly remain unavailable because leg correlation is not modeled. The
interface explicitly states that Gridiron Edge does not execute sportsbook
wagers.

#### G3a. Polish the responsive edge workspace — ✅ COMPLETE

- [x] Add a stable responsive class for the BetSlip decision workspace.
- [x] Preserve the two-column Available Edges / Bet Slip layout on wide screens.
- [x] Stack the two workspace panels at narrow widths.
- [x] Allow workspace children to shrink without forcing page overflow.
- [x] Reduce card padding on small screens.
- [x] Add a horizontally scrollable Available Edges table region.
- [x] Preserve readable table columns on narrow screens.
- [x] Add an accessible table caption.
- [x] Add column-header scopes.
- [x] Add row-header semantics to matchup cells.
- [x] Replace array-index row keys with canonical wager IDs.
- [x] Add wager-specific accessible labels to Add actions.
- [x] Add an announced loading status.
- [x] Add concise staging guidance.
- [x] Improve empty-state guidance while preserving pending and blocked states.
- [x] Add focused layout, table-semantics, action-label, and empty-state tests.
- [x] Run the frontend build and full test suite.
- [x] Commit the responsive edge-workspace unit.

The BetSlip route now uses a CSS-driven responsive workspace that preserves the
two-column decision layout on wide screens and stacks the panels at narrow
widths. Available Edges remains readable through a focusable horizontal-scroll
region and now exposes complete table semantics, canonical row identity,
wager-specific Add labels, and clearer staging guidance.

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
