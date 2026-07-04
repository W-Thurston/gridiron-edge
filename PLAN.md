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

### Design — Tier 3

**Step 1 — Populate `week_over_week_delta` on `/projections`.** 🟡 Active.

**What we are building:** The `TeamProjectionRow.week_over_week_delta`
field on the `/projections` API response is currently declared but never
populated (returns null with no `field_status` marker). Compute the
value as the Elo rating change per team from the prior NFL week within
the same season, and populate the field.

**Why we are building it now:** Small, clean opening step. Populates a
field the frontend already renders (in the "1w Δ" column). Real
user-visible impact — the column moves from em-dashes to signed values
showing Seattle gained 8.4 Elo, Kansas City lost 3.2, etc.
Establishes the "Tier 3 step" rhythm on a low-risk scope: no new
artifact, no new schema field, no new module.

**Why we're not building a snapshot mechanism:** Initial design assumed
we'd need per-week snapshots to compute deltas. Investigation revealed
the Elo state table (`data/cleaned/NFL_Team_Elo.csv`) already stores
one row per (team, season, week). Prior-week Elo is a direct lookup.
No snapshot infrastructure needed.

#### Success criteria

- `/projections` response populates `week_over_week_delta` for every
  team where prior-week Elo exists in the state table.
- Value semantic: `current_elo - prior_week_elo`. Positive means Elo
  went up.
- Week 1 (no prior week within the season): field is null. Frontend
  renders em-dash.
- `pnpm build`, `uv run gridiron verify` both pass.

#### Locked decisions

| Decision | Choice | Rationale |
|---|---|---|
| Storage | No new artifact | Elo state table already has weekly granularity |
| Delta metric | Elo change | Direct measure of "power shift" per team |
| Week 1 handling | Null | Semantically clearer than playoff-final delta (only 2 teams change) |
| Loader shape | Extend `load_projections_summary_df` (or add a composed loader that joins) | Loader owns computation per D19 |
| Serializer change | Populate field that's already in the schema | No schema change |

#### Substep breakdown

**Substep 1a — Populate `week_over_week_delta` on `/projections`.** ✅ Complete (2026-07-03).

Shipped in one commit:
- `compute_elo_deltas` helper in `api/loaders.py`. Converts long team names in Elo state to short codes via `load_team_name_map` before returning delta rows keyed on short abbreviation.
- `load_projections_summary_df` joins delta column into projections CSV data.
- Serializer populates `week_over_week_delta` from the DataFrame column. `NO_PRIOR_SNAPSHOT` blocker removed from `field_status`.
- Week 1 → null (em-dash) per design.

Test-fixture inconsistency surfaced during integration testing:
`MiniRepoBuilder.with_teams_reference()` produces modern short codes (`KC`) while other fixtures use PFR-era codes (`KAN`). Captured for ROADMAP §9.6 as future backend hygiene.

**Step 1 — Complete (2026-07-03).**

#### Disconfirming evidence

- **If `current_nfl_season()` and the Elo state's latest `NFL_YEAR`
  disagree** (e.g., off-season month issue), delta computation may
  return null even when prior data exists. Add a diagnostic log so
  future debugging is easy.
- **If the Elo state uses `NFL_TEAM` short codes but projections CSV
  uses `TEAM` short codes with different conventions** (e.g., legacy
  vs modern abbreviations), the join fails silently and returns null
  for every team. Verify both use the same convention. If not, add a
  team-name-map join step (`load_team_name_map`).

Tier design blocks are drafted at the start of each step.

---

## Paused Workstreams

_(none currently paused)_

---

## Changelog

| Date | Change |
|------|--------|
| 2026-07-03 | **W8 Tier 3 Step 1 complete.** `week_over_week_delta` field on `/projections` now populated with per-team Elo delta from prior NFL week. No new artifact — reads directly from the existing Elo state table. First Tier 3 additive shipped. |
| 2026-07-03 | **W8 Tier 3 Step 1 design.** Prior-week projection delta populates via existing Elo state table. No snapshot mechanism needed — `NFL_Team_Elo.csv` already stores weekly Elo per team. Single substep to update the projections loader and serializer. Week 1 → null (em-dash) per user preference over playoff-final delta which reads as "nothing happened" for 30 of 32 teams. |
| 2026-07-03 | **W9 Frontend complete.** Vite + React + TypeScript app consuming the 16-endpoint API. Three tiers: client infrastructure, populated screens (12 API-consuming), blocked screens + polish (4 blocked, 4 client-side). Every prototype-referenced URL renders. Every `field_status` scaffolded field surfaces its state via `<PendingField />` / `<BlockedField />`. Consistent error UX via `<ErrorCard />` and global `<OfflineBanner />`. Details in CHANGELOG.md. |
| 2026-07-01 | **W8 API Serving Layer Tier 2 complete.** 16 endpoints returning populated data with Pydantic-validated responses. Champion resolution threads through loader → serializer → route. Placeholder convention (D14) applied consistently via `_meta.field_status`. Details in CHANGELOG.md. |
| 2026-07-01 | **W13 Runtime Champion Resolution complete.** Static manifest artifact at `data/output/champions/champions.json` written by `full-retrain`. `resolve_current_champion(model_name)` reads from it. CLI consumers migrated to `--model-type auto` pattern. Unblocks all downstream champion-only consumption paths. Details in CHANGELOG.md. |
| 2026-06-23 | PLAN.md restructured to focus on the active workstream only. Future workstream candidates, real-bugs backlog, investigations, and operational items migrated to ROADMAP.md §9. |
