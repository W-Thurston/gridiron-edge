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

### Current Workstream header to (none — between workstreams)

---

## Paused Workstreams

_(none currently paused)_

---

#### W9.10: Compare Screen Rebuild — ✅ COMPLETE (2026-07-01)

Two-mode matchup surface. **Team vs Team** and **Player vs Defense**,
both prototype-aligned against real data.

**Team vs Team:** mirrored team pickers + swap, cohort strip, narrative
card, collapsible summary card, three matchup cards with mirrored
ranking-bar collision rows (offense value ↔ reciprocal defense-allowed,
edge chips, descriptive sublabels, title-style metric names). Backed by
an 11-metric cohort_splits expansion (added def_pass_epa,
def_third_down_pct, def_redzone_td_pct for reciprocal pairs).

**Player vs Defense:** independent player / stat-category / team pickers
(searchable player combobox), 7-split strip (4 live + 3 pending), a
per-game bar chart (player's stat as bars + team split-average as a
moving reference line), and a "matchup, plainly" verdict card +
by-split comparison table. Verdict is baseline-driven (defense-allowed
vs player's own average → lean over/under), with rank as general
context.

**Backend built to unblock it (Path C):**
- B1: `/players/{id}/history` (per-game series). Also fixed a root-cause
  game_id scramble in player_game_logs (`_join_game_id` index
  misalignment) and derived trustworthy is_home.
- B2: opponent_allowed expanded to 4 cohorts (season/l4/home/away,
  defense-perspective).
- B3: `/defense/{team}/allowed` (per-team allowed, all cohorts) for
  arbitrary-team selection.
- B4: `/players` roster list for the picker.

**New primitives:** BarChart (bars + reference line). DistributionChart
retired from Compare (still used by PlayerProp).

**Deferred:** book line + O/U bar coloring (odds, W7); vs-winning/
vs-losing/vs-top-10 splits (3 pending pills); Change 6 (sortable + drag
matchup rows, P2). All marked pending per highlight discipline.

**W9.10 complete.** Both Compare modes shipped; Compare no longer the
biggest screen gap.

---

## Changelog

| Date | Change |
|------|--------|
| 2026-07-11 | **W9.10 complete.** Both Compare modes shipped. Player-vs-Defense rebuilt with independent pickers + per-game bar chart + baseline-driven verdict, on new backend B1–B4 (player-history, 4-cohort opponent-allowed, defense-by-team, players roster). Fixed game_id scramble at root. |
| 2026-07-11 | **W9.10 status resync.** Team vs Team complete (incl. six prototype-alignment adjustments + 11-metric cohort_splits expansion). Player vs Defense redesigned to mirror Team mode (independent player/stat/team pickers, 7-split strip, per-game bar chart centerpiece, "matchup plainly" card). Paused on backend: B1 player-history endpoint + B2 opponent-allowed splits expansion (Path C). Book line / O-U coloring deferred (odds); Change 6 deferred (P2). |
| 2026-07-11 | **W9.10 Compare Screen Rebuild design.** Locked. ~8 substeps across 4 tiers. Two modes (Team vs Team + Player vs Defense). Team mode: enhanced pickers, cohort strip, grouped matchup sections, auto-narrative. Player mode: DistributionChart + defense stat rows. Highlight discipline baked in from W9.8. |
