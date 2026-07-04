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

**Tier 3 — Additive datasets.** ✅ Complete (2026-07-04).

Seven additives shipped across 15+ substeps, populating scaffolded
`field_status` fields across the frontend surface:

- **Step 1:** `week_over_week_delta` on `/projections` — Elo change from Elo state table.
- **Step 2:** Percentile ranking pass — 4 stats across `/teams`, `/teams/{abbr}`, `/compare/teams`.
- **Step 3:** `trend` on `/teams` and `/teams/{abbr}` — reused compute_elo_deltas helper.
- **Step 4:** `n_simulations` on `/projections` — new metadata sidecar written by sim.
- **Step 5:** `situational_splits` on `/props/{prop_id}` — per-player, 8 cohorts, from player game logs + games CSV.
- **Step 6:** Defense-side rows on `/compare/player/{prop_id}` — per-opponent-position aggregates from player game logs. `red_zone_rate_allowed` remains blocked pending PBP-derived aggregation.
- **Step 7:** Team cohort splits on `/compare/teams`, `/teams/{abbr}`, `/games/{game_id}` — 4 cohorts × 8 metrics from EPA data.

**New CLI subcommand apps:**
- `gridiron sim compute-percentiles` (Step 2)
- `gridiron props compute-splits` (Step 5)
- `gridiron props compute-opponent-allowed` (Step 6)
- `gridiron teams compute-cohort-splits` (Step 7) — new `gridiron teams` app

**New artifacts under `data/output/`:**
- `rankings/percentiles/percentiles_{season}_wk{NN}.parquet`
- `rankings/team_cohort_splits.parquet`
- `props/situational_splits/{stat_type}.parquet`
- `props/opponent_allowed.parquet`
- `temp/projections_metadata.json`

**Remaining not-shipped from original inventory:**
- Off/def rating decomposition — real modeling work; deferred to future workstream if pursued.

**Remaining field_status: pending fields** are all blocked on named workstreams (feature attribution, injury data source, multi-book odds, PBP-derived aggregations). Not additive-dataset work.

**W8 workstream complete.** Tier 1 (skeleton + stubs), Tier 2 (16 populated endpoints), Tier 3 (7 additive datasets) all shipped.

Tier design blocks are drafted at the start of each step.

---

## Paused Workstreams

_(none currently paused)_

---

## Changelog

| Date | Change |
|------|--------|
| 2026-07-04 | **W8 Tier 3 complete. W8 workstream closed.** Seven additive datasets shipped populating scaffolded fields across the frontend API surface. New CLI subcommand `gridiron teams`. Five new persistence artifacts. Remaining field_status: pending fields blocked on named workstreams. |
| 2026-07-04 | **W8 workstream closed.** Full W8 (API Serving Layer) shipped across three tiers: skeleton + blocked stubs (Tier 1), populated endpoints (Tier 2, 16 endpoints), additive datasets (Tier 3, 7 additives). Consumed end-to-end by W9 (Frontend). Now between workstreams. |
| 2026-07-04 | **W8 Tier 3 Step 7 complete.** Team cohort splits: 8 metrics × 4 cohorts per team from EPA data. Populates `/compare/teams` (new `cohort_splits` field), `/teams/{abbr}` (rename `situational_splits` → `cohort_splits`), and `/games/{game_id}` (populate `team_comparison`). New `gridiron teams` CLI subcommand app. |
| 2026-07-04 | **W8 Tier 3 Step 7 design (revised).** Team cohort splits: 8 metrics × 4 cohorts per team, from `epa_by_game.parquet`. Populates 3 endpoints: `/compare/teams` (new `cohort_splits` field), `/teams/{abbr}` (rename `situational_splits` → `cohort_splits`), `/games/{game_id}` (populate `team_comparison`). New `gridiron teams` CLI subcommand. Three substeps. |
| 2026-07-04 | **W8 Tier 3 Step 6 complete.** Opponent-allowed-by-position aggregations for `/compare/player/{prop_id}`. 3 of 4 defense-side rows populate from the artifact; `red_zone_rate_allowed` remains blocked. `resolve_opponent_from_game_id` helper added to `_prop_id.py`. |
| 2026-07-04 | **W8 Tier 3 Step 6 design.** Opponent-allowed-by-position: per-defense aggregations of stat allowed to each position across season + l5 cohorts. Populates 3 defense-side rows on `/compare/player/{prop_id}`. Two substeps: computation module + CLI (6a), loader + serializer (6b). red_zone_rate_allowed deferred pending PBP-derived aggregation. |
| 2026-07-04 | **W8 Tier 3 Step 5 complete.** Situational splits computed by joining player game logs to games CSV; 8 cohorts (season, home/away, favored/underdog, indoor/outdoor, l4). Per-stat-type Parquet artifacts consumed by `/props/{prop_id}`. First real feature-engineering module in Tier 3. |
| 2026-07-04 | **W8 Tier 3 Step 5 design.** Prop cohort splits for 8 cohorts (season, home, away, favored, underdog, indoor, outdoor, l4). Data joined from player_game_logs + games CSV on game_id. Per-stat-type Parquet artifacts at `data/output/props/situational_splits/`. Two substeps: computation module + CLI (5a), loader + serializer (5b). |
| 2026-07-04 | **W8 Tier 3 Step 4 complete.** `n_simulations` on `/projections` populated via new `projections_metadata.json` sidecar. Backwards compatible — legacy projections without sidecar leave the field null. |
| 2026-07-04 | **W8 Tier 3 Step 3 complete.** `trend` field on `/teams` and `/teams/{abbr}` populated via reused `compute_elo_deltas` from Step 1. Smaller substep than 1 or 2 due to helper reuse. |
| 2026-07-04 | **W8 Tier 3 Step 3 design.** Populate `trend` field on `/teams` and `/teams/{abbr}` with per-team Elo change from prior NFL week. Same shape as Step 1's `week_over_week_delta` on projections. Single substep. |
| 2026-07-04 | **W8 Tier 3 Step 2 complete.** Per-team percentile ranking pass shipped across `/teams`, `/teams/{abbr}`, and `/compare/teams`. New `evaluation/percentiles.py` module + persistence artifact at `data/output/rankings/percentiles/`. Wired into `sim run` and exposed via `gridiron sim compute-percentiles` for standalone use. |
| 2026-07-04 | **W8 Tier 3 Step 2 design.** Per-team percentile ranking pass for 4 stats (Elo, avg_wins, make_playoffs, win_sb). Three substeps: computation module (2a), loader + `/teams` endpoints (2b), `/compare/teams` percentile fields (2c). Aggregate `percentile_ranks` scaffold row on `/compare/teams` replaced with per-row percentiles on rankable stat rows. Frontend consumes `pct` values via `rankColor()` and bar-width formulas already in the prototype. |
| 2026-07-04 | **W8 Tier 3 Step 1 complete.** `week_over_week_delta` field on `/projections` now populated with per-team Elo delta from prior NFL week. No new artifact — reads directly from the existing Elo state table. First Tier 3 additive shipped. |
| 2026-07-04 | **W8 Tier 3 Step 1 design.** Prior-week projection delta populates via existing Elo state table. No snapshot mechanism needed — `NFL_Team_Elo.csv` already stores weekly Elo per team. Single substep to update the projections loader and serializer. Week 1 → null (em-dash) per user preference over playoff-final delta which reads as "nothing happened" for 30 of 32 teams. |
| 2026-07-03 | **W9 Frontend complete.** Vite + React + TypeScript app consuming the 16-endpoint API. Three tiers: client infrastructure, populated screens (12 API-consuming), blocked screens + polish (4 blocked, 4 client-side). Every prototype-referenced URL renders. Every `field_status` scaffolded field surfaces its state via `<PendingField />` / `<BlockedField />`. Consistent error UX via `<ErrorCard />` and global `<OfflineBanner />`. Details in CHANGELOG.md. |
| 2026-07-01 | **W8 API Serving Layer Tier 2 complete.** 16 endpoints returning populated data with Pydantic-validated responses. Champion resolution threads through loader → serializer → route. Placeholder convention (D14) applied consistently via `_meta.field_status`. Details in CHANGELOG.md. |
| 2026-07-01 | **W13 Runtime Champion Resolution complete.** Static manifest artifact at `data/output/champions/champions.json` written by `full-retrain`. `resolve_current_champion(model_name)` reads from it. CLI consumers migrated to `--model-type auto` pattern. Unblocks all downstream champion-only consumption paths. Details in CHANGELOG.md. |
| 2026-06-23 | PLAN.md restructured to focus on the active workstream only. Future workstream candidates, real-bugs backlog, investigations, and operational items migrated to ROADMAP.md §9. |
