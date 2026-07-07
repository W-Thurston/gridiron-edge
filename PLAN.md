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

### Current Workstream: (none — between workstreams)

### Recently Completed

#### W9.7: Teams Split-View Rebuild — ✅ COMPLETE (2026-07-07)

Restructured `/teams` and `/teams/:abbr` into a consolidated split-view
screen at `/teams` with optional `?team=X` param. Left column shows
rankings; right column shows the selected team's profile with 6 real
sections and 2 blocked placeholders. Completed in 9 substeps across
4 tiers.

**Delivered:**
- Single route `/teams` with `?team=X` param; auto-selects #1 team
  when no param
- Left column: rankings table (all 32 teams) with 5-tab strip
  (Overall / Offense / Defense / ATS / Net Rating). Off/Def/ATS/Net
  render blocked-state messaging.
- Row selection: URL param sync (no navigation), green left border
  + background tint for selected row, hover state
- Rankings subheader shows "Wk N · model v4.2"
- Right column sections:
  - Team hero band with team-colored vertical gradient (180deg,
    30% mix top → dark bottom)
  - Rating chart (`RatingChart` primitive) with Y-axis grid, dots,
    and inline W/L markers per week
  - Situational Splits card with cohort switcher (Season/L4/Home/Away)
    consuming `cohort_splits` from Step 7c
  - Recent Results (existing `RecentResultsStrip`)
  - Schedule Difficulty placeholder (blocked)
  - Postseason Outlook composed from `/projections` with colored
    progress bars per row
  - Top Players placeholder (blocked)

**New primitive:**
- `RatingChart` — SVG line chart with Y-axis grid, data point dots,
  X-axis labels, W/L outcome markers. Responsive via SVG viewBox.

**Helpers:**
- `stripCityPrefix` (matches GameDetail approach)
- `expandDivisionLetter` (N → North, etc.)
- `formatSeason` (2025-2026 → 2025)

**Preserved as blocked placeholders:**
- Schedule Difficulty (upcoming_games backend enrichment)
- Top Players (WAR feature attribution)

**Deleted:**
- Old `TeamRankings.tsx` and `TeamProfile.tsx` files consolidated
  into single `TeamsScreen.tsx`

**W9.7 workstream complete.**

---

## Paused Workstreams

_(none currently paused)_

---

## Changelog

| Date | Change |
|------|--------|
| 2026-07-07 | **W9.7 complete.** Teams Split-View Rebuild shipped in 9 substeps across 4 tiers. Single split-view screen with rankings + profile. New RatingChart primitive. All 5 W9.5 primitives consumed. |
| 2026-07-07 | **W9.7 Teams Split-View Rebuild design.** Locked. 9 substeps across 4 tiers. Route consolidation, split-view layout, rankings table with tabs, team hero + 6 sections in right pane (blocked schedule/top players as placeholders). Consumes all 5 W9.5 primitives + Step 7c cohort_splits + /projections composition. |
