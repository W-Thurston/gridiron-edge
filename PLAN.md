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

**Step 6 — Populate defense-side rows on `/compare/player/{prop_id}`.** 🟡 Active.

#### What we are building

Precomputed per-(opponent_team, position, stat_type) aggregations —
mean stat allowed and rank against the position — for the current
season. Populates 3 of the 4 currently-blocked defense-side rows on
the compare/player response: `avg_allowed`, `rank_against_position`,
`last_5_games_avg`. Leaves `red_zone_rate_allowed` pending (requires
PBP data not derived here).

#### Why we are building it now

Biggest UX unlock on the /compare/player screen. 4 of 8 stat rows
currently render "not available" — this step populates 3 of them.
Same computation pattern as Step 5 (situational splits), so the
substep breakdown mirrors Step 5's rhythm.

#### Success criteria

- New computation module `evaluation/opponent_allowed.py` produces
  per-(opponent_team, position, stat_type, cohort) rows.
- Two cohorts computed: `season` and `l5` (last 5 games rolling).
- New CLI command `gridiron props compute-opponent-allowed` writes
  the artifact to `data/output/props/opponent_allowed.parquet`.
- `/compare/player/{prop_id}` populates the 3 defense-side rows for
  which we have data. `red_zone_rate_allowed` stays pending.
- `field_status: blocked/OPPONENT_ALLOWED_BY_POSITION` marker removed
  from the 3 populated rows.
- All quality gates pass.

#### Locked decisions

| Decision | Choice | Rationale |
|---|---|---|
| Cohorts | `season` (all games in current season) + `l5` (last 5 games rolling) | Matches prototype's avg_allowed + last_5_games_avg rows |
| Attribution | Per game, sum stat from all offensive players with matching position. Average across games. | Handles rare multi-QB games; broader than "starter only" |
| Season scope | Current season only (max NFL_YEAR from player_game_logs) | Simplest; matches Step 5's approach |
| When computed | Standalone CLI command `gridiron props compute-opponent-allowed`. Wire into `full-retrain` as follow-up | Small, standalone command; can be automated later |
| `red_zone_rate_allowed` | Defer to future step | Requires PBP-derived aggregation; out of scope for W8 Tier 3 |
| Team codes | Modern short codes (KC, LAC, JAX) matching player_game_logs and prop archive | player_game_logs.opponent_team already uses these conventions |
| Artifact location | `data/output/props/opponent_allowed.parquet` — single file | Small (~1,280 rows); no partitioning needed |
| Rank convention | 1 = stingiest (lowest avg allowed) to 32 = most generous. Ranked within (position, stat_type, cohort). | Matches prototype's convention |
| Opponent lookup at request time | Parse from game_id string (e.g. "2026_01_KC_LAC" + player_team=KC → opponent=LAC) | No games CSV lookup needed |

#### Disconfirming evidence

- **Team code mismatch (already noted in ROADMAP §9.6):** player_game_logs
  uses modern short codes (KC, LAC, JAX); the `NFL_long_to_short_name.csv`
  reference table uses PFR-era codes (KAN, JAC). Since this step reads
  directly from player_game_logs and doesn't cross a naming-map
  boundary, we can use modern codes throughout without issue. But test
  fixtures should be consistent to avoid confusion.
- **Small sample sizes early in the season:** if the season has 3 weeks
  played, `l5` cohort will only have 3 games. Handle gracefully (return
  sample_size < 5).
- **`is_skill` filter unclear:** player_game_logs has an `is_skill`
  column. Should we filter to only skill players when attributing stats?
  Defer to Substep 6a — for now, sum all players with matching position
  regardless of `is_skill`.

#### Substep breakdown

**Substep 6a — Opponent-allowed computation module + CLI.**
- New file `src/gridiron_edge/evaluation/opponent_allowed.py`.
- Public functions:
  - `compute_opponent_allowed(player_game_logs)` — returns DataFrame
    with columns `opponent_team, position, stat_type, cohort,
    avg_allowed, sample_size, rank_against_position`.
  - `write_opponent_allowed(df, repo)` — persist to Parquet.
  - `load_opponent_allowed(repo)` — read the artifact.
- CLI command `gridiron props compute-opponent-allowed` in `cli/props.py`.
- Unit tests: cohort partitioning, rank computation, per-position sum,
  empty inputs, multiple stat_types.

**Substep 6b — Loader + populate defense-side rows on `/compare/player/{prop_id}`.**
- New loader `load_opponent_allowed_for_prop(settings, opponent_team,
  position, stat_type)` — returns dict of cohort → aggregates.
- Small helper `resolve_opponent_from_game_id(game_id, player_team)` in
  `api/_prop_id.py` — parses game_id string, returns the team code that
  isn't the player's team.
- Update `serialize_compare_player` and its route:
  - Determine opponent via `resolve_opponent_from_game_id`.
  - Load opponent-allowed lookup.
  - Populate 3 defense-side rows.
  - Remove `OPPONENT_ALLOWED_BY_POSITION` blocker for those 3 rows.
- Integration test.

Tier design blocks are drafted at the start of each step.

---

## Paused Workstreams

_(none currently paused)_

---

## Changelog

| Date | Change |
|------|--------|
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
