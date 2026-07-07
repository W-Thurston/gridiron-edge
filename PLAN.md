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

### Current Workstream: W9.7 — Teams Split-View Rebuild

**Status:** Designing.

### What we are building

Restructure `/teams` and `/teams/:abbr` into a single split-view
screen at `/teams` with an optional `?team=X` param. Left column
shows the league table (all 32 teams); right column shows the
selected team's profile. Clicking a table row updates the right
pane without navigation, preserving ranking context.

Sections rendered:
- **Left column:** Rankings table with Overall/Offense/Defense/ATS/Net
  Rating tabs (Off/Def blocked as placeholders)
- **Right column:** Team hero band with primary color + 3 stat hero
  row + rating chart + cohort splits + recent results + postseason
  outlook

Blocked sections shipped as `ComingSoonCard` for consistency:
- Schedule Difficulty (backend blocked)
- Top Players by WAR (blocked in §9.7)

Old `/teams/:abbr` route redirects to `/teams?team=abbr` for
backward compatibility.

### Why we are building it now

Two motivations:

1. **UX gap.** Current two-route pattern forces back-navigation to
   compare teams. Users lose ranking context. Split-view is the
   natural affordance for a "browse-and-inspect" flow.

2. **Data ready.** All the data we need ships from existing
   endpoints:
   - `/teams` for rankings
   - `/teams/{abbr}` for team profile with rating_history + record + recent_results + cohort_splits (via Step 7c)
   - `/projections` for postseason outlook (composed client-side)
   - Team metadata (colors, city, name) via W9.5 primitives

   Rebuild leverages every primitive from W9.5 (Pill, WhyLink,
   TeamMark, Spark, TeamHero).

#### Success criteria

- Route `/teams` renders split-view with rankings on left, profile
  on right
- Auto-selects #1 ranked team when no `?team=X` param
- Row click updates URL and right pane (no navigation)
- Rankings table shows all 32 teams with tabs for view mode
- Team hero band uses team primary color as background
- Rating chart, cohort splits, recent results, postseason outlook
  all populated from existing data
- Old `/teams/:abbr` route redirects to `/teams?team=abbr`
- Schedule Difficulty + Top Players remain as ComingSoonCards
- Old TeamRankings + TeamProfile files deleted (consolidated into
  single `TeamsScreen.tsx`)
- All quality gates pass

### Locked architectural decisions

| Decision | Choice |
|---|---|
| Route structure | Single `/teams` with optional `?team=X`; old `/teams/:abbr` redirects |
| Default team | Auto-select #1 ranked |
| Split ratio | 40 / 60 (left / right) |
| Left column rows | All 32 teams, scrollable |
| Rankings tabs | Overall (default) / Offense (blocked) / Defense (blocked) / ATS (blocked) / Net rating |
| Rating chart | Existing Spark primitive; no uncertainty band (not shipped) |
| Cohort splits layout | 3-column with cohort switcher via Pill (matches GameDetail Team Comparison pattern) |
| Recent results | Show what backend has (opponent + score + W/L result) |
| Hero stats | Record + Rank + Rating (3 stats big) |
| Team hero band background | `color-mix` primary_color at 15% |
| Column sorting | Skip (backlog item; sort by rating desc default) |
| File consolidation | Single `TeamsScreen.tsx` file |

### Prerequisite

None. All backend data already ships. All W9.5 primitives available.

### Tiers

**Tier 1 — Route restructure (1 substep).**

Combine `TeamRankings` and `TeamProfile` into a single component at
`/teams`. Auto-select #1 team when no `?team=X` param. Old
`/teams/:abbr` route added to redirect.

**Tier 2 — Left column (2 substeps).**

- 2a: Rankings table with row selection and URL sync (no navigation)
- 2b: Rankings tabs (Overall/Offense/Defense/ATS/Net rating); Off/Def
  blocked as pill placeholders

**Tier 3 — Right column sections (4 substeps).**

- 3a: Team hero band with primary color + 3 hero stats
- 3b: Rating chart with Spark primitive
- 3c: Cohort splits section (3-column, Pill cohort switcher, 8 metrics)
- 3d: Recent results + postseason outlook (two smaller cards)

**Tier 4 — Blocked placeholders + integration (2 substeps).**

- 4a: Schedule Difficulty + Top Players as ComingSoonCards
- 4b: Final integration cleanup (delete old files, verify redirects,
  ensure tests pass)

### Disconfirming evidence

- **If team primary color at 15% mix is too dark for readability** on
  hero band, adjust to 8-10% or use a different tint.
- **If /projections filtering by team_abbr adds latency**, consider
  cross-endpoint caching or client-side pre-filtering.
- **If rankings table row height is too tall to fit 32 rows without
  scrolling**, tighten `padding` on rows.
- **If Old `/teams/:abbr` redirect creates flicker**, we accept it or
  handle in nav context.
- **If backend `recent_results` field doesn't have enough context to
  render (e.g., missing opponent short abbr)**, we display what's
  available (opponent long name + score + result letter).

### Timeline

Total: 9 substeps. Not tied to calendar; natural cadence.

### Success artifacts

By workstream close:

- `TeamsScreen.tsx` renders as split-view
- 32-team ranking table with tab structure
- Right pane populated with 6 real sections + 2 blocked placeholders
- All 5 W9.5 primitives consumed (Pill, WhyLink, TeamMark, Spark, TeamHero)
- Backward-compatible route handling for old `/teams/:abbr` URLs
- Cleanup: `TeamRankings.tsx` + `TeamProfile.tsx` deleted

---

## Paused Workstreams

_(none currently paused)_

---

## Changelog

| Date | Change |
|------|--------|
| 2026-07-07 | **W9.7 Teams Split-View Rebuild design.** Locked. 9 substeps across 4 tiers. Route consolidation, split-view layout, rankings table with tabs, team hero + 6 sections in right pane (blocked schedule/top players as placeholders). Consumes all 5 W9.5 primitives + Step 7c cohort_splits + /projections composition. |
