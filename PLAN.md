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

**Step 1 — Populate `week_over_week_delta` on `/projections`.** ✅ Complete (2026-07-04).

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

**Substep 1a — Populate `week_over_week_delta` on `/projections`.** ✅ Complete (2026-07-04).

Shipped in one commit:
- `compute_elo_deltas` helper in `api/loaders.py`. Converts long team names in Elo state to short codes via `load_team_name_map` before returning delta rows keyed on short abbreviation.
- `load_projections_summary_df` joins delta column into projections CSV data.
- Serializer populates `week_over_week_delta` from the DataFrame column. `NO_PRIOR_SNAPSHOT` blocker removed from `field_status`.
- Week 1 → null (em-dash) per design.

Test-fixture inconsistency surfaced during integration testing:
`MiniRepoBuilder.with_teams_reference()` produces modern short codes (`KC`) while other fixtures use PFR-era codes (`KAN`). Captured for ROADMAP §9.6 as future backend hygiene.

**Step 1 — Complete (2026-07-04).**

**Step 2 — Per-team percentile ranking pass.** ✅ Complete (2026-07-04).

Shipped in three substeps:
- **2a:** `evaluation/percentiles.py` module + persistence. Computes per-team percentiles for 4 stats (rating, avg_wins, make_playoffs, win_sb), writes to `data/output/rankings/percentiles/`. Wired into `sim run` as a final step; standalone `gridiron sim compute-percentiles` CLI command added.
- **2b:** Loader + populate `/teams` and `/teams/{abbr}`. Four percentile fields added to `TeamRankingRow` and `TeamProfile` schemas. Empty artifact → null fields.
- **2c:** Populate `/compare/teams` percentile fields. `team_a_pct` and `team_b_pct` added to `StatRow`. Populated on 4 rankable stat rows. Aggregate `percentile_ranks` scaffold row removed; `avg_wins` and `win_sb` rows added; `playoff_probability` renamed to `make_playoffs`.

**Step 2 — Complete (2026-07-04).**

**Step 3 — Populate `trend` field on `/teams` and `/teams/{abbr}`.** ✅ Complete (2026-07-04).

Reused `compute_elo_deltas` from Step 1 to compute per-team Elo change
from the prior NFL week within the same season. Serializers populate
the `trend` field on both `TeamRankingRow` and `TeamProfile`. Removed
`NO_PRIOR_SNAPSHOT` blocker on trend fields. Week 1 → null.

Single substep — smaller than Steps 1 and 2 given the reuse.

**Step 3 — Complete (2026-07-04).**

**Step 4 — Populate `n_simulations` on `/projections`.** ✅ Complete (2026-07-04).

New metadata sidecar `projections_metadata.json` written alongside the
projections CSV in `run_full_simulation`. Contains `n_simulations` and
`computed_at`. `load_projections_summary_df` return tuple grew from
`(df, mtime)` to `(df, mtime, n_simulations)`. Serializer accepts and
populates the field. Backwards compatible — legacy projections without
sidecar leave the field null.

Single substep.

**Step 4 — Complete (2026-07-04).**

**Step 5 — Populate `situational_splits` on `/props/{prop_id}`.** ✅ Complete (2026-07-04).

Shipped in two substeps:
- **5a:** `evaluation/situational_splits.py` module. Joins player game logs to games CSV on game_id, partitions by 8 cohorts (season, home, away, favored, underdog, indoor, outdoor,l4), aggregates sample_size + mean_value per (player_id, cohort). CLI command `gridiron props compute-splits`. Per-stat-type Parquet artifact at `data/output/props/situational_splits/{stat_type}.parquet`.
- **5b:** Loader `load_prop_situational_splits` reads the artifact and filters to player_id. Serializer populates `situational_splits` as nested dict of cohort → {sample_size, mean_value}. Conditional field_status: pending when artifact missing, no marker when populated (even if empty for the player).

**Step 5 — Complete (2026-07-04).**

**Step 6 — Populate defense-side rows on `/compare/player/{prop_id}`.** ✅ Complete (2026-07-04).

Shipped in two substeps:
- **6a:** `evaluation/opponent_allowed.py` module.
Computes per- (opponent_team, position, stat_type, cohort) aggregates: mean allowed, sample size, rank against position (1 = stingiest). Two cohorts:
  season, l5. CLI command `gridiron props compute-opponent-allowed`.
  Single Parquet artifact at `data/output/props/opponent_allowed.parquet`.
- **6b:** Loader `load_opponent_allowed_for_prop` reads artifact and filters to (opponent, position, stat_type). New `_prop_id.py` helper `resolve_opponent_from_game_id` parses game_id string to determine opponent. Serializer populates 3 of 4 defense-side rows
  (`avg_allowed`, `rank_against_position`, `last_5_games_avg`).
  Conditional blocker: rows blocked when artifact missing, populated
  otherwise. `red_zone_rate_allowed` remains blocked (PBP-derived,
  out of scope).

**Step 6 — Complete (2026-07-04).**

**Step 7 — Team cohort splits.** 🟡 Active.

#### What we are building

Precomputed per-(team, cohort) EPA and efficiency aggregations for
the current season. Populates cohort_splits fields on three
endpoints: `/compare/teams`, `/teams/{abbr}`, and `/games/{game_id}`.

Eight metrics per (team, cohort):
- `off_epa_per_play`, `def_epa_per_play` (unit strengths)
- `off_pass_epa`, `off_rush_epa`, `def_rush_epa` (breakdown)
- `off_third_down_pct`, `off_redzone_td_pct` (situational)
- `turnover_diff` (composite: off_turnover_rate - def_turnover_rate)

Four cohorts: season, l4, home, away.

#### Why we are building it now

Fills currently-blocked space on three major screens: Compare
(team-vs-team stat rows), Team Profile (situational splits card),
and Game Detail (team comparison card). Uses existing
`epa_by_game.parquet` data — no new joins needed.

Symmetric with Step 5's prop cohort splits at team level.

#### Success criteria

- New computation module `evaluation/team_cohort_splits.py`.
- New CLI command `gridiron teams compute-cohort-splits`.
- Artifact at `data/output/rankings/team_cohort_splits.parquet`.
- `/compare/teams` populates `cohort_splits` for both teams.
- `/teams/{abbr}` populates renamed `cohort_splits` field.
- `/games/{game_id}` populates `team_comparison` field with cohort
  splits for both teams playing.
- All quality gates pass.

#### Locked decisions

| Decision | Choice | Rationale |
|---|---|---|
| Data source | `epa_by_game.parquet` (existing) | All 8 metrics from single source; no joins needed |
| Metrics (8) | off_epa_per_play, off_pass_epa, off_rush_epa, def_epa_per_play, def_rush_epa, off_third_down_pct, off_redzone_td_pct, turnover_diff | Matches prototype Team Comparison rows |
| Cohorts | season, l4, home, away | Matches Step 5's cohort model |
| Season scope | Current season only | Same as Step 5 |
| Team names | Long names in EPA → short codes via `load_teams_long_short` | Same pattern as Step 5 |
| Home/away | Parse from game_id string (e.g. "2024_01_KC_LAC" → KC away, LAC home) | Consistent with Step 6 |
| Rank direction | Off metrics: rank 1 = highest. Def metrics + turnover_diff: rank 1 = best (lowest def; highest turnover_diff). | Reflects "better team" for both directions |
| Artifact | `data/output/rankings/team_cohort_splits.parquet` (long format) | Small (~128 rows) |
| CLI location | New `gridiron teams` subcommand app in `cli/teams.py` | No team commands exist today |
| Schema additions | `CompareTeamsResponse.cohort_splits: dict[str, dict] | None`; populate `GameDetail.team_comparison` (existing field) | Additive on compare; renaming needed for TeamProfile |
| Schema rename | `TeamProfile.situational_splits` → `cohort_splits` | Consistent naming across endpoints |
| When computed | Standalone command; wire into `full-retrain` as follow-up | Same as Step 6 |

#### Disconfirming evidence

- **Prototype Team Comparison uses "vs winning teams" cohort** — this
  would be a 5th cohort. Skip for T7 (deferred to future step or Tier 4
  polish).
- **Prototype shows "Run def yds/g" not `def_rush_epa`** — the yardage
  metric isn't directly available in EPA data. Use `def_rush_epa` as a
  proxy for run defense strength.
- **Prototype's `turnover_diff` shows +0.4 per game** — this is per-game
  turnover margin. My proposed `turnover_diff` metric is a rate difference
  (0-1 scale); may want to scale to per-game (× off_plays or × games).
  Simpler: leave as rate difference and let frontend format.
- **Frontend consuming `situational_splits` on TeamProfile** will need
  updating when renamed to `cohort_splits`. Small frontend change.
- **`GameDetail.team_comparison` is currently a broader concept** in the
  frontend prototype (includes labels, bar rendering, etc.). Populating
  it with cohort_splits is one interpretation; frontend may need format
  adjustment. Defer that adjustment; the schema will hold the data
  either way.

#### Substep breakdown

**Substep 7a — Team cohort splits computation module + CLI.**
- New file `src/gridiron_edge/evaluation/team_cohort_splits.py`.
- Public functions:
  - `compute_team_cohort_splits(epa_df, long_to_short)` → DataFrame
    with columns: team_abbr, cohort, off_epa_per_play, off_pass_epa,
    off_rush_epa, def_epa_per_play, def_rush_epa, off_third_down_pct,
    off_redzone_td_pct, turnover_diff, sample_size, plus 8 rank columns.
  - `write_team_cohort_splits(df, repo)` → Path.
  - `load_team_cohort_splits(repo)` → DataFrame.
- New `cli/teams.py` with `teams_app` and `compute_cohort_splits_cmd`.
- Register `teams_app` in `cli/main.py`.
- Unit tests: cohort partitioning, rank direction (off vs def),
  team name conversion, empty inputs, missing metric columns,
  turnover_diff computation.

**Substep 7b — Loader + populate `/compare/teams` and `/teams/{abbr}`.**
- Rename `TeamProfile.situational_splits` → `cohort_splits` in schema.
- Add `cohort_splits: dict[str, dict] | None` field on
  `CompareTeamsResponse`.
- New loader `load_team_cohort_splits_df(settings)`.
- Helper `format_cohort_splits_for_team(df, team_abbr)` → nested dict.
- Update serializers.
- Remove pending markers when populated.
- Integration tests.

**Substep 7c — Populate `team_comparison` on `/games/{game_id}`.**
- Update `serialize_game_detail` to populate `team_comparison` with
  cohort splits for both teams.
- Remove pending marker when populated.
- Integration test.

Tier design blocks are drafted at the start of each step.

---

## Paused Workstreams

_(none currently paused)_

---

## Changelog

| Date | Change |
|------|--------|
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
