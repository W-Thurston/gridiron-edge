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

### Current Workstream: W8 — API Serving Layer (Tier 3)

**Status:** Designing.

### What we are building

Tier 3 additive datasets that populate the `field_status: pending` and
`field_status: blocked` fields currently surfaced by W8 Tier 2's 16
endpoints. Which additives ship first, and in what order, is driven
by W9 feedback — the frontend surfaced which pending states most
impact the UX.

### Why we are building it

W8 Tier 2 shipped 16 endpoints with roughly 20% of prototype-referenced
fields populated. The remaining 80% are scaffolded — the shape exists,
the data doesn't. Tier 3 fills in the data, endpoint by endpoint,
prioritized by which additive dataset unlocks the most UI value per
unit of backend work.

### Prerequisite: prioritization

Before design begins, decide which additive dataset ships first. The
inventory:

| Addition | Populates | User-facing impact (from W9) |
|---|---|---|
| Per-stat league-wide percentile ranking pass | Compare screen rank columns, Team Detail rank fields | TBD |
| Off/def rating decomposition | Team Rankings off/def split | TBD |
| Weekly Elo snapshot persistence | Team rating-history endpoint, projections week-over-week delta | TBD |
| Opponent-allowed-by-position aggregation | Player vs Defense view, Player Prop matchup section | TBD |
| Limited cohort splits (season, L4, home, away) per team | Game Detail split tabs, Compare splits | TBD |
| Limited cohort splits (indoor/outdoor, favored/underdog) per prop | Player Prop situational splits | TBD |
| Prior-week projection snapshot for delta | Projections 1-week change column | TBD |

Rate each additive for user-facing impact based on what you saw
during W9 exploration. Highest impact goes first.

### Design (to be filled in when prioritization locks)

The Tier 3 design phase produces:
- Which additive dataset ships first as Step 1.
- Substep breakdown (mirrors Tier 2's rhythm: design → loader → schema
  → serializer → route → integration test).
- Locked architectural decisions for the additive itself.

Tier design blocks are drafted at the start of each step.

---

## Paused Workstreams

_(none currently paused)_

---

## Changelog

| Date | Change |
|------|--------|
| 2026-07-03 | **W9 Frontend complete.** Vite + React + TypeScript app consuming the 16-endpoint API. Three tiers: client infrastructure, populated screens (12 API-consuming), blocked screens + polish (4 blocked, 4 client-side). Every prototype-referenced URL renders. Every `field_status` scaffolded field surfaces its state via `<PendingField />` / `<BlockedField />`. Consistent error UX via `<ErrorCard />` and global `<OfflineBanner />`. Details in CHANGELOG.md. |
| 2026-07-01 | **W8 API Serving Layer Tier 2 complete.** 16 endpoints returning populated data with Pydantic-validated responses. Champion resolution threads through loader → serializer → route. Placeholder convention (D14) applied consistently via `_meta.field_status`. Details in CHANGELOG.md. |
| 2026-07-01 | **W13 Runtime Champion Resolution complete.** Static manifest artifact at `data/output/champions/champions.json` written by `full-retrain`. `resolve_current_champion(model_name)` reads from it. CLI consumers migrated to `--model-type auto` pattern. Unblocks all downstream champion-only consumption paths. Details in CHANGELOG.md. |
| 2026-06-23 | PLAN.md restructured to focus on the active workstream only. Future workstream candidates, real-bugs backlog, investigations, and operational items migrated to ROADMAP.md §9. |
