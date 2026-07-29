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

### Current Workstream — W9.11 Tier 0: Final frontend audit

**Status:** Designing

## Purpose

Complete the deferred pending-highlight and final frontend consistency audit
across the now-complete core-screen set.

The audit should identify visible missing, pending, blocked, or misleading
states without reopening completed screen implementations unless a bounded,
reproducible defect is found.

## W9.11 completed tiers

### Tier 1 — PlayoffProjections rebuild — ✅ COMPLETE (2026-07-28)

Rebuilt PlayoffProjections around the verified static simulation and team-state
contracts. Added sortable postseason probabilities, conference/division
filters, Elo context, simulation metadata, team navigation, sibling navigation,
and the Week 1–18 Weekly Outcomes matrix.

### Tier 2 — BetSlip decision-support rebuild — ✅ COMPLETE (2026-07-29)

Rebuilt BetSlip as a model-informed wager-shortlisting and what-if analysis
workspace.

Delivered:

- exact reference-price and sizing provenance from `/edges`;
- nullable bankroll input for dollar sizing, with no hidden `$1,000` default;
- versioned game and prop BetLeg variants;
- canonical producer-independent wager identity;
- immutable recommendation snapshots separated from editable draft state;
- validated v2 persistence and recovery from malformed or legacy state;
- shared constructors across every live game and prop producer;
- truthful unpriced prop interests with no fabricated sportsbook odds;
- editable current odds, proposed stake, sportsbook, and notes;
- guarded current EV, break-even, Kelly, payout, and profit calculations;
- tracked portfolio and explicit what-if bankroll modes;
- explicit bankroll-source and Kelly-multiplier provenance;
- complete/incomplete singles aggregation;
- quoted parlay odds, payout, and profit without unsupported combined model EV,
  probability, or Kelly;
- responsive Available Edges and BetSlip layouts;
- wager-specific accessible labels, pressed-state semantics, and live summary
  updates;
- explicit language that Gridiron Edge does not place sportsbook wagers.

Backend and frontend quality gates passed throughout the rebuild.

A staged-wager real-data visual review remains deferred because `/edges`
currently returns no available recommendations. Synthetic production data,
fabricated prices, and permanent demo paths were intentionally not introduced
to bypass that operational data state. Automated tests cover priced game legs,
unpriced props, current-price edits, threshold behavior, Kelly sizing, singles,
parlays, incomplete states, responsive classes, and accessibility semantics.

Optional future work not required for Tier 2 completion:

- a deliberately designed recorded-bet write API;
- a `Record Bet` action coupled to the ledger and bankroll transaction model;
- recorded-bet or draft-slip CSV export;
- multi-book odds and line shopping.

These items belong in `ROADMAP.md`, not the active Tier 2 plan.

## Tier 0 implementation plan

### A. Audit scope

- [ ] Walk every completed frontend route with pending highlighting disabled.
- [ ] Repeat the route walk with pending highlighting enabled.
- [ ] Verify pending, blocked, and unavailable states remain visibly distinct.
- [ ] Verify no populated field is incorrectly highlighted as pending.
- [ ] Verify no missing field is silently omitted where a visible state is
      required.
- [ ] Verify shared primitives remain visually and semantically consistent.
- [ ] Record only reproducible defects or intentionally deferred data gaps.

### B. Core-screen coverage

- [ ] Dashboard.
- [ ] Games list.
- [ ] Game detail.
- [ ] Team rankings and team profile.
- [ ] PlayerProp.
- [ ] Players explorer.
- [ ] Playoff Chances and Weekly Outcomes.
- [ ] Compare, both modes.
- [ ] BetSlip empty and blocked-data states.
- [ ] Bankroll and portfolio screens.
- [ ] Tools, Settings, and Onboarding.

### C. Deferred real-data verification

- [ ] Complete the staged-wager BetSlip visual review when `/edges` returns at
      least one real recommendation.
- [ ] Verify a priced game wager from the live edge response.
- [ ] Verify an unpriced prop interest and manually entered current odds.
- [ ] Verify tracked and what-if sizing against real portfolio state.
- [ ] Verify complete and incomplete single/parlay summaries.

The BetSlip review is operationally blocked by the current absence of available
edge recommendations. It does not block beginning the broader Tier 0 audit.

### D. Close-out

- [ ] Resolve bounded frontend defects found by the audit.
- [ ] Move newly identified backend or data gaps to `ROADMAP.md`.
- [ ] Run frontend quality gates.
- [ ] Update `CHANGELOG.md`, `HANDOFF.md`, and `ROADMAP.md`.
- [ ] Close W9.11.

## Success criteria

Tier 0 is complete when:

- every completed frontend route has been reviewed with pending highlighting on
  and off;
- missing, blocked, pending, and populated states are represented truthfully;
- no audit-only working terminology leaks into production code;
- bounded presentation defects discovered during the audit are resolved;
- data-dependent gaps are recorded in the roadmap without fabricated UI data;
- the staged-wager BetSlip review is completed when real edge data becomes
  available, or remains explicitly documented as an operational verification
  blocker;
- focused and full frontend quality gates pass.

---

## Paused Workstreams

_(none currently paused)_

---

## Changelog

| Date | Change |
|------|--------|
| 2026-07-28 | Closed the PlayoffProjections navigation and Weekly Outcomes follow-up after real-data verification. |
| 2026-07-12 | **Doc-sync pass + PLAN reset.** Normalized planning docs after the frontend arc; PLAN reset to between-workstreams with a next-candidates list (audit sweep recommended). |
| 2026-07-11 | **W9.10 complete.** Both Compare modes shipped on backend B1–B4. Fixed game_id scramble, clean-games clobber, Elo empty-games crash; added champion→elo fallback + upcoming-Week-1 season resolver. |
| 2026-07-11 | **W9.10 status resync.** Team vs Team complete (6 alignment adjustments + 11-metric cohort_splits). Player vs Defense redesigned to independent pickers + bar chart + baseline verdict. |
| 2026-07-11 | **W9.10 design locked.** Two modes; highlight discipline baked in from W9.8. |
