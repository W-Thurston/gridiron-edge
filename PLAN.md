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

#### W9.9: PlayerProp Rebuild — ✅ COMPLETE (2026-07-07)

Rebuilt PlayerProp (`/players/:propId`) from skeleton (identity card
+ 4-cell projection + 6 ComingSoonCards) to prototype fidelity in 8
substeps across 4 tiers.

**Delivered:**
- Full-width player hero band with team-colored gradient (matches
  TeamsScreen pattern)
- Prop summary callout card on right side of hero: distinct card with
  green accent border, stat label with game context ("MON vs SF"),
  big em-dash for pending line, model mean + range on flex row,
  pending markers for confidence + EV
- "+ Bet slip" button outside the summary card
- Distribution chart primitive (new): SVG Gaussian density curve
  with 90% credible band shading, mean marker + label, x-axis
  endpoints
- Situational Splits card consuming Step 5 data: 8 cohorts in
  canonical order with "X.X avg · N games" format
- Polished Player vs Defense table with WhyLink dot in header
- 5 blocked ComingSoonCards in 3-column grid

**New primitive:**
- `DistributionChart` — SVG probability density chart. Renders
  Gaussian PDF from mean + std with 90% band shading, mean marker,
  x-axis endpoints. Extractable to Compare screen (W9.10).

**Helpers:**
- `getOpponentFromGameId` — parse game_id string (inline, not
  primitive)
- `formatSeason` — "2025-2026" → "2025" (shared with TeamsScreen)
- `formatStatType` + `formatStatTypeShort` — extracted to
  `utils/props.ts` for reuse across Dashboard/GameDetail/PlayerProp

**Preserved as blocked placeholders:**
- Historical vs Opponent (pending)
- Recent Form (pending)
- Injury Status (blocked on §5.3)
- Prop Reasoning (blocked on feature attribution)
- Multi-Book Shopping (blocked on W7)

**Established pattern:**
- Composed player screens now consume prop + game + team metadata in
  a single flow
- Distribution chart primitive establishes the shape for future
  probability visualizations (Compare's Player vs Defense mode)

**W9.9 workstream complete.**

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
| 2026-07-07 | **W9.9 complete.** PlayerProp Rebuild across 8 substeps in 4 tiers. New DistributionChart primitive. Consumes situational_splits (Step 5) + prop + game + team metadata. Composed screen pattern established for future prop-related work. |
| 2026-07-07 | **W9.9 PlayerProp Rebuild design.** Locked. 8 substeps across 4 tiers. Full-width hero + single column below. New DistributionChart primitive. Consumes situational_splits (Step 5) + player vs defense (Step 6). Column layout single (not 2-col — width constraint lesson from W9.7). |
| 2026-07-07 | **W9.7 complete.** Teams Split-View Rebuild shipped in 9 substeps across 4 tiers. Single split-view screen with rankings + profile. New RatingChart primitive. All 5 W9.5 primitives consumed. |
| 2026-07-07 | **W9.7 Teams Split-View Rebuild design.** Locked. 9 substeps across 4 tiers. Route consolidation, split-view layout, rankings table with tabs, team hero + 6 sections in right pane (blocked schedule/top players as placeholders). Consumes all 5 W9.5 primitives + Step 7c cohort_splits + /projections composition. |
