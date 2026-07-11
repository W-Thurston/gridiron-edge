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

### Current Workstream: W9.10 — Compare Screen Rebuild

**Status:** Designing.

### What we are building

Rebuild the Compare screen (`/compare`) from its current flat 3-column
stat table to a two-mode matchup surface: Team vs Team and Player vs
Defense. Mode switcher at top; cohort control strip; grouped matchup
sections; auto-generated narrative banner.

We won't reach full prototype fidelity (its 24-stat × 6-cohort universe
needs backend expansion). But we ship a significantly better Compare
with existing data: Step 7 cohort splits (8 metrics × 4 cohorts) drive
Team vs Team; Step 6 opponent-allowed + W9.9's DistributionChart drive
Player vs Defense.

Every pending/blocked element uses the W9.8 highlight primitives, so
Compare auto-participates in the deferred audit sweep.

### Why we are building it now

Compare is the largest single-screen gap in the app and the most
conceptually distinct — it embodies the platform's "see the raw
materials, draw your own conclusion" philosophy. Two use cases
(team matchups, player-vs-defense) share the screen deliberately.

Building it now:
- Consumes W9.9's DistributionChart primitive (Player vs Defense mode)
- Consumes Step 6 + Step 7 data already shipping
- Applies the highlight discipline from W9.8 throughout
- Closes the last major screen-rebuild before we'd need real backend
  expansion for further fidelity

#### Success criteria

- Mode switcher: Team vs Team / Player vs Defense (Pill-based)
- URL syncs `?mode=team|player`
- **Team vs Team mode:**
  - Enhanced team pickers (team colors + rating + record) with swap button
  - Cohort control strip (Season/L4/Home/Away)
  - Grouped matchup sections: "When A has ball" / "When B has ball" /
    "Even footing" consuming Step 7 cohort splits
  - Auto-generated narrative banner computed from stat differences
- **Player vs Defense mode:**
  - Player + defense pickers
  - DistributionChart for the prop's projection
  - Player-vs-Defense stat rows (existing table)
- All pending/blocked elements use highlight primitives
- All quality gates pass

### Locked architectural decisions

| Decision | Choice |
|---|---|
| Modes | Both Team vs Team + Player vs Defense |
| Layout | Prototype's sectional structure, stacked single-column (width lesson from W9.7/W9.9) |
| Mode switcher | Pill-based, URL-synced `?mode=` |
| Team vs Team grouping | "When A has ball" / "When B has ball" / "Even footing" |
| Team pickers | Enhanced with colors + rating + record + swap button |
| Cohort switcher | Pill row, 4 cohorts (Season/L4/Home/Away) |
| Narrative banner | Computed dynamically from cohort split differences |
| Player vs Defense chart | DistributionChart primitive (from W9.9) |
| Player vs Defense table | Preserve existing structure |
| Drag-reorder rows | Skip (deferred polish) |
| Highlight discipline | All pending/blocked via W9.8 primitives |

### Prerequisite

None. Step 6 + Step 7 data ships. DistributionChart primitive available.
All W9.5 primitives + W9.8 highlight mechanism in place.

### Tiers

**Tier 1 — Restructure + mode switcher (1 substep).**

- 1a: Layout restructure. Mode switcher (Pill) with URL sync. Retain
  existing content in placeholder sections during rebuild.

**Tier 2 — Team vs Team mode (4 substeps).**

- 2a: Enhanced team pickers (colors + rating + record) + swap button.
- 2b: Cohort control strip (4 Pills).
- 2c: Grouped matchup sections consuming Step 7 cohort splits.
- 2d: Auto-generated narrative banner computed from data.

**Tier 3 — Player vs Defense mode (2 substeps).**

- 3a: Player + defense pickers.
- 3b: DistributionChart integration + player-vs-defense stat rows.

**Tier 4 — Cleanup + close-out (1 substep).**

- 4a: Cleanup + workstream close-out prep.

### Disconfirming evidence

- **Backend cohort splits are only 8 metrics, not the prototype's 24.**
  "When A has ball" sections will have fewer rows. Render what we have;
  mark absent metric categories with highlight primitives if we want
  them visible as gaps.
- **Defensive-side matchup data:** Step 7 gives each team its own
  off/def metrics, but not opponent-specific matchup pairs. "When A has
  ball" compares A's offense vs B's defense using the two teams' cohort
  splits — works, but isn't a true opponent-adjusted collision.
- **Player vs Defense mode needs a player picker sourced from `/props`.**
  If the props list is large, picker UX may need search/filter. Start
  simple (dropdown), enhance if needed.
- **Width constraint:** prototype's player mode is a 2-column
  (game-log hero + verdict rail). Ours stacks single-column. Same lesson
  as W9.7/W9.9.
- **Auto-narrative could produce awkward phrasing** for edge cases (ties,
  missing data). Guard with fallback text.

### Timeline

Total: ~8 substeps. Not tied to calendar.

### Success artifacts

By workstream close:
- Compare renders two working modes
- Enhanced team pickers with visual identity
- Grouped matchup sections + auto-narrative
- Player vs Defense mode with DistributionChart
- All W9.5 primitives + DistributionChart + highlight primitives consumed
- Compare no longer the biggest screen gap

---

## Paused Workstreams

_(none currently paused)_

---

## Changelog

| Date | Change |
|------|--------|
| 2026-07-01 | **W9.10 Compare Screen Rebuild design.** Locked. ~8 substeps across 4 tiers. Two modes (Team vs Team + Player vs Defense). Team mode: enhanced pickers, cohort strip, grouped matchup sections, auto-narrative. Player mode: DistributionChart + defense stat rows. Highlight discipline baked in from W9.8. |
