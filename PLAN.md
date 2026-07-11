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

**Status:** Active. Team vs Team complete; Player vs Defense redesign
paused on backend endpoints (the immediate next work).

### What we are building

Two-mode matchup surface at `/compare`: Team vs Team and Player vs
Defense. Mode switcher, cohort/split controls, and prototype-aligned
layout consuming cohort splits, opponent-allowed, and player history.

### Progress

**Tier 1 — Restructure + mode switcher — ✅ complete.**
Pill mode switcher, URL sync (`?mode=team|player`), layout skeleton.

**Tier 2 — Team vs Team mode — ✅ complete (incl. prototype alignment).**
- Enhanced team pickers with team colors + rating + record + swap;
  mirrored inward-facing identity (name→logo / logo→name), width-
  constrained and centered
- Cohort strip (Season/L4/Home/Away)
- Separate cards: floating pickers → narrative card → collapsible
  summary card (centered team-left/team-right) → three matchup cards
- Matchup rows: 5-column center-aligned with mirrored rank-fill bars
  (rank 1 = full, fills toward center), edge chips (arrow + team +
  descriptor), value+rank inline, descriptive sublabels ("Rush EPA /
  play" vs "Rush EPA allowed / play"), title-style metric names
  ("Run efficiency")
- Auto-narrative card computing biggest collision per direction from
  rank differentials
- Backed by an 11-metric cohort_splits expansion: added def_pass_epa,
  def_third_down_pct, def_redzone_td_pct so every offensive metric has
  its reciprocal defensive-allowed pair

**Tier 3 — Player vs Defense mode — 🟡 redesign, paused on backend.**
Redesigned to mirror Team vs Team (per 2026-07-01 spec + prototype
screenshot). Independent player / stat-category / team pickers; a
7-split strip (season / l4 / home / away / vs-winning / vs-losing /
vs-top-10); a per-game bar chart as the centerpiece; a "matchup,
plainly" verdict card; a comparison table.

**Tier 4 — Cleanup + close-out — pending.**

### Immediate next work — Tier B (backend, Path C)

Player vs Defense's centerpiece (per-game bar chart) and split options
are blocked on backend artifacts that don't exist yet. Build these two
endpoints/expansions first, then resume the frontend rebuild:

**B1 — Player game-history endpoint.**
`GET /players/{player_id}/history?stat=<stat>&limit=<n>` returning
per-game values `[{week, value, opponent, game_id}, ...]` for the
season. Data already exists in `player_game_logs.parquet` — this
exposes it. Powers the bar chart. Also unblocks §9.7 P0 items
(PlayerProp 12-game chart, PlayersExplorer L6 sparkline).

**B2 — Opponent-allowed splits expansion.**
Expand `opponent_allowed` aggregation from 2 splits (season, l5) to 7
(season / l4 / home / away / vs-winning / vs-losing / vs-top-10),
mirroring the team_cohort_splits expansion pattern. Powers the split
strip's team-allowed average line.

### Then — Tier C (Player vs Defense frontend rebuild)

Rebuild the mode against B1/B2 data:
- Pickers: player + stat-category (same row, left card) + team (right
  card), mirroring Team-vs-Team card layout
- Split strip (7 splits via Pill)
- **Bar chart** (replaces the DistributionChart in this mode): player's
  chosen stat per game across the season (bars don't change with split);
  team's split-average as a solid horizontal line; book line as dashed
  (pending — blocked on odds); bar over/under coloring (pending — odds)
- **"The matchup, plainly"** verdict card: team allows X to the position,
  ranks Y of 32, favorable/unfavorable vs our projection
- Comparison table (keep current; consider centering to match Team mode)

### Locked architectural decisions

| Decision | Choice |
|---|---|
| Modes | Both Team vs Team + Player vs Defense |
| Layout | Sectional cards, stacked single-column (width lesson W9.7/W9.9) |
| Mode switcher | Pill-based, URL-synced `?mode=` |
| Team vs Team | Complete: mirrored pickers, cohort strip, narrative + summary + 3 matchup cards, ranking-bar collision rows |
| Cohort metrics | 11-metric cohort_splits (offense + reciprocal defense pairs) |
| Player-vs-Defense pickers | Independent player / stat-category / team (not a single prop_id) |
| Player-vs-Defense splits | 7 (season/l4/home/away/vs-winning/vs-losing/vs-top-10) |
| Player-vs-Defense centerpiece | Per-game bar chart (not DistributionChart) — bars = player's stat per game; solid line = team-allowed split avg; dashed line = book line (pending) |
| Player selection source | Derive from `/props` (dedupe by player); stat categories filtered to what the player has |
| Book line + O/U coloring | Deferred (blocked on odds, W7); marked pending per highlight discipline |
| Drag-reorder + sort (Change 6) | Deferred (P2, §9.8) |

### Blocked / deferred within W9.10

- **B1/B2 backend** — immediate next work; everything in Tier C waits on it
- **Book line + over/under bar coloring** — blocked on odds (W7); render as pending
- **Change 6 (sort by category/edge + drag-reorder)** — P2, deferred; build only if missed

### Disconfirming evidence

- **Player-history data completeness:** `player_game_logs.parquet` was
  last refreshed Jun 10 (stale vs Jul 5 game-side data). Bar chart may
  miss the final playoff weeks for some players. Acceptable for
  verification; note if completeness matters for real use.
- **7-split expansion cost:** vs-winning / vs-losing / vs-top-10 require
  opponent-record and opponent-rank context at aggregation time — more
  than the simple home/away partition. Confirm the source data supports
  it before promising all 7; ship the subset we can compute and mark
  the rest pending.
- **Stat-category → stat_type mapping:** QBs have pass + rush; RBs have
  rush + rec. Picker filters to categories the player actually has data
  for.
- **Width constraint:** single-column, matching the Team-vs-Team layout;
  Compare page centered at max-width.

### Timeline

Team vs Team complete. Remaining: B1 + B2 (backend) → Tier C (frontend
rebuild) → Tier 4 (cleanup + close-out). Roughly 6–9 substeps.

---

## Paused Workstreams

_(none currently paused)_

---

## Changelog

| Date | Change |
|------|--------|
| 2026-07-11 | **W9.10 status resync.** Team vs Team complete (incl. six prototype-alignment adjustments + 11-metric cohort_splits expansion). Player vs Defense redesigned to mirror Team mode (independent player/stat/team pickers, 7-split strip, per-game bar chart centerpiece, "matchup plainly" card). Paused on backend: B1 player-history endpoint + B2 opponent-allowed splits expansion (Path C). Book line / O-U coloring deferred (odds); Change 6 deferred (P2). |
| 2026-07-11 | **W9.10 Compare Screen Rebuild design.** Locked. ~8 substeps across 4 tiers. Two modes (Team vs Team + Player vs Defense). Team mode: enhanced pickers, cohort strip, grouped matchup sections, auto-narrative. Player mode: DistributionChart + defense stat rows. Highlight discipline baked in from W9.8. |
