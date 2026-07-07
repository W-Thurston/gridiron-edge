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

---

### Current Workstream: W9.6 — GameDetail Full Fidelity

**Status:** Designing.

### What we are building

Rebuilt GameDetail (`/games/:id`) from the current skeleton
(header + prediction cell grid + 5 coming-soon cards) to prototype
fidelity. Ships 5 new working sections and preserves 2 as
placeholders (blocked on named workstreams).

Uses all 5 primitives shipped in W9.5 (`Pill`, `WhyLink`, `TeamMark`,
`Spark`, `TeamHero`).

### Why we are building it now

GameDetail is the most-visited screen after Dashboard (every "click
game" from Dashboard, GamesList, or Compare lands here). Currently
renders as: minimal header + flat prediction cell grid + 5
placeholder cards that say "not yet available." Data ships for most
of it — `team_comparison` field from Step 7c, prop edges from
`/props`, and predictions from `/games/:id`. Just needs rendering.

Compound value: consumes all 5 primitives shipped in W9.5, showing
the primitives' payoff clearly.

#### Success criteria

- Two-column layout replaces single-column stack.
- Full-width game header renders with TeamHero for both teams,
  center block with kick + venue + weather placeholders, and model
  lean callout (from `/edges`).
- Lines & model fair value table with 3 rows (Market / Model /
  Recommendation).
- Win probability card with 2 prob bands + projected score.
- Team comparison card with 4 cohort tabs (Season / L4 / Home /
  Away) — consumes existing Step 7c data.
- Top prop edges card with 4-5 filtered rows from `/props`.
- Swing factors and Injuries remain as `<ComingSoonCard>` with
  proper `field_status` badges.
- All quality gates pass.

### Locked architectural decisions

| Decision | Choice |
|---|---|
| Layout | Full-width header + 3fr/2fr 2-col grid below |
| Kick time | Date only from `game_date` (no time-of-day backend data yet) |
| Weather / venue | Em-dash placeholder (blocked; consistent with elsewhere) |
| Model lean callout | Compose from `/edges` filtered to game |
| Market row on lines table | Show pending markers |
| Team comparison cohorts | 4 tabs (Season/L4/Home/Away) via Pill primitive |
| Team comparison row rendering | Simple 3-col (away value / metric / home value) with color-coded percentile — no colored bars for v1 |
| Bet slip integration | Add recommended edge as leg with -110 placeholder odds |
| Right rail | Top prop edges (rendered) + Swing factors (placeholder) + Injuries (placeholder) |
| Track button | Skip (no tracking system) |
| Existing prediction card | Replace with new composed cards |

### Prerequisite

None. All primitives from W9.5 are shipped.

### Tiers

**Tier 1 — Layout restructure (1 substep).**

Rebuild the layout skeleton with full-width header slot and 2-col
grid below. No new components yet; move existing prediction cells
into placeholder cards. Verifies structural change without semantic
change.

**Tier 2 — Header composition (2 substeps).**

Build the full-width game header:
- 2a: Team hero header with TeamHero for both teams, center block
  with placeholders.
- 2b: Model lean callout (compose from /edges).

**Tier 3 — Main column cards (3 substeps).**

Build the 3 main column sections:
- 3a: Lines & model fair value table.
- 3b: Win probability card with projected score.
- 3c: Team comparison card with cohort tabs.

**Tier 4 — Right rail + integration (3 substeps).**

Complete the right rail and integrate:
- 4a: Top prop edges card.
- 4b: Placeholder integration for blocked sections.
- 4c: Final integration cleanup — old prediction card removed,
  layout wired, tests updated.

### Timeline

Total: 9 substeps. Not tied to calendar; natural cadence.

### Success artifacts

By workstream close:

- GameDetail renders with 5 new working sections
- 2 placeholders remain (Swing factors, Injuries) with clear blocker
  messaging
- All 5 W9.5 primitives consumed (TeamHero heavily; Pill in comparison
  tabs; WhyLink in header + prop edges; Spark not directly here — for
  future win prob chart)
- Two-column layout established as reusable pattern for future screen
  rebuilds

Tier design blocks are drafted at the start of each step.

---

## Paused Workstreams

_(none currently paused)_

---

## Changelog

| Date | Change |
|------|--------|
| 2026-07-06 | **W9.6 GameDetail Full Fidelity design.** Locked. 9 substeps across 4 tiers. Layout restructure + header composition + main column cards (lines table, win prob, team comparison) + right rail (prop edges + placeholders). Uses all 5 primitives from W9.5. |
