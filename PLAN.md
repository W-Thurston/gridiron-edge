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

### Current Workstream: W9.5 — Dashboard Rebuild + Cross-Cutting Primitives

**Status:** Designing.

### What we are building

A rebuilt Dashboard (currently a debug scaffold shipped in Substep
2.0 during W9 Tier 1) that renders as the actual primary landing
page: featured matchups grid, model edges table with tab filters,
model performance rail with sparkline, and player prop edges rail.

Alongside Dashboard: five cross-cutting frontend primitives that
unblock every other future frontend workstream — `Pill` (filter
toggle), `WhyLink` (explainability affordance), `TeamMark`-with-team-
colors (visual identity), `Spark` (generic sparkline), and
`TeamHero` (composed team identity block).

Small W8 patch upfront: add `primary_color`, `secondary_color`,
`conference`, and `division` fields to `TeamRankingRow` and
`TeamProfile` schemas so primitives that consume team identity have
real data.

### Why we are building it now

Two motivations:

1. **Dashboard is unusable.** The primary landing page currently shows
   an API loop verification card and a field-status primitives demo
   table — debug scaffolding shipped in W9 Tier 1 and never built out.
   Every visit to `/today` renders as broken/incomplete.

2. **Primitives unblock everything.** Shared components identified in
   the prototype audit (`Pill`, `WhyLink`, `TeamMark`-with-colors,
   `Spark`, `TeamHero`) are used across 10+ screens. Building them in
   isolation with proper APIs makes every future frontend workstream
   faster and more consistent.

#### Success criteria

- Dashboard `/today` renders four working sections: featured matchups
  grid (3 games), model edges table with 5 filter tabs, model
  performance rail with sparkline + big number, prop edges rail (5 rows).
- Five primitives exist in `components/primitives/` with unit tests.
- `TeamMark` renders team primary color background where team metadata
  is available.
- `WhyLink` navigates to `/explain?subject=...` with proper parameters.
- Backend fields (`primary_color`, `secondary_color`, `conference`,
  `division`) populate on `TeamRankingRow` and `TeamProfile` for all
  32 teams.
- Existing API verification card and field-status demo removed from
  Dashboard (moved to `/debug` route or deleted).
- All quality gates pass.

### Locked architectural decisions

| Decision | Choice |
|---|---|
| Order of substeps | Backend patch → Primitives → Dashboard sections |
| Backend patch strategy | Single commit at start; ships colors + conf + div |
| Featured matchups when market lines blocked | Model-only render; market side blank |
| Model performance rail with limited data | All-time ROI as big number; mark 7d/30d as pending |
| Multi-sport pills | Skip; NFL-only for now |
| Primitives location | `frontend/src/components/primitives/` (new folder) |
| Team metadata source | New reference CSV: `data/cleaned/NFL_team_metadata.csv` |
| Test coverage | Vitest smoke tests per primitive; integration tests for Dashboard sections |
| Existing dashboard content | Move to `/debug` route (preserved for future work) |

### Prerequisite

None. Backend patch is part of Tier 1.

### Tiers

**Tier 1 — Backend patch (1 substep).**

Add team metadata fields to schemas and populate from new reference
CSV. Unblocks primitive work.

**Tier 2 — Shared primitives (5 substeps).**

Build five primitives in `components/primitives/`:
- `Pill` — filter toggle
- `WhyLink` — explainability affordance
- `TeamMark` (refactor) — with team primary color
- `Spark` — generic sparkline (renamed from `RatingHistorySparkline`)
- `TeamHero` — composed team identity block

**Tier 3 — Dashboard sections (5 substeps).**

Build four working sections + integration:
- `FeaturedMatchupsGrid` — 3 game cards
- `ModelEdgesTable` — table with 5 filter tabs
- `PropEdgesRail` — 5-row compact list
- `ModelPerformanceRail` — card with sparkline + big number
- Dashboard integration — wire sections into 2-column layout;
  remove debug demo content

### Disconfirming evidence

- **If backend patch reveals data mismatches** (some teams missing
  colors, city/name split inconsistent), we handle per-team with
  defaults and log for follow-up. Not blocking.
- **If `WhyLink` navigation reveals `/explain` needs specific
  parameters we haven't documented**, we add them as we go. `/explain`
  is currently a `<BlockedScreen />` — the WhyLink shipping is what
  makes the navigation-to-blocked-screen pattern meaningful.
- **If `Spark` refactor breaks existing `RatingHistorySparkline`
  usages** (TeamProfile), we ship both temporarily and migrate. Not
  blocking.
- **If Dashboard's model performance rail has no historical data**
  (fresh installs), sparkline renders as flat or empty. Mark as
  pending. Acceptable.
- **If `/props?limit=5&sort=ev_desc` doesn't exist**, add query params
  to `/props` endpoint as a substep 3c prerequisite. Small.

### Timeline

Total: 11 substeps. Not tied to calendar; work at natural cadence.
Estimated ~1-2 weeks of active work at normal substep rhythm.

### Success artifacts

By workstream close:

- Dashboard renders as a real landing page with 4 working sections
- 5 primitives live in `components/primitives/` with unit tests
- Backend patch shipped with team colors + conf + div
- 4 new API-consuming Dashboard sections
- Frontend visual identity (team primary colors) shipped
- Path to explainability (`WhyLink`) established even though `/explain`
  itself remains blocked

Ready for close-out and next-workstream decision.

Tier design blocks are drafted at the start of each step.

---

## Paused Workstreams

_(none currently paused)_

---

## Changelog

| Date | Change |
|------|--------|
| **W9.5 complete.** Dashboard Rebuild + Cross-Cutting Primitives shipped in 11 substeps across 3 tiers. Tier 1: team metadata backend patch + CSV consolidation. Tier 2: 5 shared primitives (Pill, WhyLink, TeamMark-with-colors, Spark, TeamHero). Tier 3: 4 Dashboard sections + integration. Debug scaffolding removed. |
| 2026-07-04 | **W9.5 Dashboard Rebuild + Cross-Cutting Primitives design.** Locked. Total 11 substeps across 3 tiers. Tier 1: backend patch adds team colors + conference + division. Tier 2: 5 primitives (`Pill`, `WhyLink`, `TeamMark`-with-colors, `Spark`, `TeamHero`). Tier 3: 5 Dashboard sections + integration. Featured matchups model-only until W7 lands. |
