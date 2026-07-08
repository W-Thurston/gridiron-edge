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

### Current Workstream: W9.9 — PlayerProp Rebuild

**Status:** Designing.

### What we are building

Rebuild `PlayerProp` screen (`/players/:propId`) from its current
skeleton (identity card + 4-cell projection + 6 ComingSoonCards) to
prototype fidelity. Full-width hero with player identity + prop
summary callout; single column below with distribution chart,
situational splits, player vs defense, and blocked placeholders.

Sections rendered:
- **Hero band:** team-colored gradient, big player mark, serif italic
  name, inline identity stats
- **Prop summary callout:** right side of hero — stat, line
  placeholder, model mean, range, EV placeholder, slip button
- **Below hero:** distribution chart (new primitive), situational
  splits (Step 5 data), player vs defense (existing), blocked-section
  placeholders

**New primitive:**
- `DistributionChart` — SVG density curve for prop distributions.
  Renders Gaussian from mean + std. Extractable to Compare screen
  (W9.10) later.

### Why we are building it now

Two motivations:

1. **Pattern continuity.** W9.6 (GameDetail) and W9.7 (Teams
   split-view) established the "screen rebuild consuming primitives +
   backend composition" pattern. W9.9 continues the pattern before
   we tackle Compare (W9.10) which is larger.

2. **Data ready.** Situational splits from Step 5 render cleanly.
   Player vs Defense (Step 6) mostly ships. Projection block (mean,
   std, bounds) supports Gaussian distribution rendering. Same
   "data exists, renderer doesn't" opportunity as Team Comparison
   in W9.6.

3. **Compound benefit.** DistributionChart primitive pays dividends
   in W9.10 Compare (Player vs Defense mode). Building here means
   Compare's substep count drops.

#### Success criteria

- PlayerProp `/players/:propId` renders as hero + single column
  layout
- Full-width hero with team-colored gradient, player mark, serif
  italic name, inline identity stats
- Prop summary callout on right side of hero with stat, line
  placeholder, model mean, range, EV placeholder, slip button
- Distribution chart shows density curve from `predicted_mean` +
  `predicted_std`
- Situational splits card renders 8 cohorts from Step 5
  `situational_splits` field
- Player vs Defense table renders 4 projection + 3 defense rows
- Blocked sections shipped as ComingSoonCards
- All quality gates pass

### Locked architectural decisions

| Decision | Choice |
|---|---|
| Layout | Full-width hero + single column below (matches W9.7 width lesson) |
| Hero band | Team-colored gradient (same pattern as TeamsScreen) |
| Prop summary | Right side of hero (with placeholders for blocked line/EV) |
| Distribution chart | New primitive; Gaussian from mean + std |
| Distribution shading | Skip over/under coloring (blocked on line data) |
| Situational splits | 2-column list (label / value + sample size) |
| Player vs Defense | Preserve existing structure, minor polish |
| ComingSoonCards | Keep for blocked field_status fields |
| WhyLink | Dot variants on Distribution + Player vs Defense headers |
| Column layout | Single column below hero (prototype's 2-col doesn't fit our width) |

### Prerequisite

None. All backend data ships from existing endpoints. All W9.5
primitives available.

### Tiers

**Tier 1 — Layout restructure (1 substep).**

Rebuild layout skeleton: full-width hero + single column below.
Preserve existing content in placeholder cards during transition.

**Tier 2 — Hero header (2 substeps).**

- 2a: Player hero band with team-colored gradient, big player mark,
  serif italic name, inline identity stats
- 2b: Prop summary callout on right side of hero

**Tier 3 — Content sections (3 substeps).**

- 3a: New `DistributionChart` primitive + integrate
- 3b: Situational Splits card consuming Step 5 data
- 3c: Player vs Defense polish (existing table + primitive
  consistency)

**Tier 4 — Placeholders + cleanup (2 substeps).**

- 4a: ComingSoonCards for blocked sections
- 4b: Cleanup + integration verification

### Disconfirming evidence

- **If situational_splits data structure differs from expected
  shape** (some props may not have all 8 cohorts populated), we
  render only populated cohorts and show "not available" for the
  rest.
- **If prop team abbreviation doesn't have colors in team metadata**
  (some edge cases), fall back to grey. Standard `TeamMark` pattern.
- **If distribution chart renders poorly for very small std values**
  (< 5), we accept the visual and note as backlog item.
- **If Player vs Defense structure change causes test breakage**,
  scope adjustment during that substep.
- **If width constraints force us to shrink some hero components**,
  adjust proportions rather than layout — same lesson as W9.7.

### Timeline

Total: 8 substeps. Not tied to calendar; natural cadence.

### Success artifacts

By workstream close:

- PlayerProp renders as real screen with 4 populated sections + 4-5
  blocked placeholders
- New `DistributionChart` primitive available for W9.10 Compare
- All 5 W9.5 primitives consumed (heavy Pill + TeamMark; some WhyLink)
- Established prop-screen composition patterns
- Cleanup: no stale imports, dead code, or unused helpers

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
| 2026-07-07 | **W9.9 PlayerProp Rebuild design.** Locked. 8 substeps across 4 tiers. Full-width hero + single column below. New DistributionChart primitive. Consumes situational_splits (Step 5) + player vs defense (Step 6). Column layout single (not 2-col — width constraint lesson from W9.7). |
| 2026-07-07 | **W9.7 complete.** Teams Split-View Rebuild shipped in 9 substeps across 4 tiers. Single split-view screen with rankings + profile. New RatingChart primitive. All 5 W9.5 primitives consumed. |
| 2026-07-07 | **W9.7 Teams Split-View Rebuild design.** Locked. 9 substeps across 4 tiers. Route consolidation, split-view layout, rankings table with tabs, team hero + 6 sections in right pane (blocked schedule/top players as placeholders). Consumes all 5 W9.5 primitives + Step 7c cohort_splits + /projections composition. |
