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

### Current Workstream: W8 — API Serving Layer

### What we are building

A read-only REST API that exposes every analytics output Gridiron Edge produces, shaped to match the Gridiron Edge frontend prototype end-to-end. Every screen in the prototype gets the endpoints it needs; every field that the backend can populate today returns real data; every field the backend can't yet produce returns `null` accompanied by metadata describing why.

The API is implemented as a FastAPI app with Pydantic v2 response models, mounted as an `api/` package and served via `gridiron api serve`. Pydantic is confined to the `api/` boundary — domain code (models, evaluation, market, features) stays pandas/dataclass-shaped.

### Why we are building it

1. **Verification surface.** The CLI surfaces outputs one at a time. The frontend prototype puts ~19 screens worth of outputs side by side. Wiring the prototype to the API is the next quality-assurance step.
2. **Frontend unblock.** W9 (Frontend) cannot start until there is an API to wire to.
3. **Roadmap discovery.** The placeholder fields the API ships form a structured, observable inventory of "what's missing" that drives ROADMAP §9 prioritization.

#### Success criteria

- Every endpoint in the prototype-driven inventory returns a 200 response with a valid Pydantic-validated shape.
- Every populated endpoint returns real data for ≥80% of its fields, with the remainder explicitly marked in `_meta.field_status`.
- Every Tier 3 endpoint returns `null` fields with structured `_meta.field_status` entries naming a blocker.
- `gridiron api serve` starts the API and surfaces OpenAPI docs at `/docs`.
- Test coverage: unit (response model shape + `_meta` correctness), integration (`MiniRepoBuilder`-backed), with e2e deferred to W9.
- All quality gates pass.

### Disconfirming evidence

- **If the archive's schema drifts** during a Tier 2 step (e.g., new columns, renamed columns), loaders assuming the current shape break silently. Substep verification should include a `.columns.tolist()` check against the live archive before writing loaders.
- **If a Tier 3 additive dataset lands earlier than expected** (e.g., an injuries feed becomes available mid-Tier 2), a Tier 2 endpoint may want to consume it opportunistically rather than shipping with `field_status: blocked`. Decide case-by-case; the default is "ship blocked and revisit."
- **If Pydantic v2 shape validation catches loader inconsistencies** — for example, a `float | None` field receiving `NaN` — the loader is the fix site, not the schema. Track any such catches during Tier 2 to inform Tier 3 hardening.

### Locked architectural decisions

| Decision | Choice |
|---|---|
| Framework | **FastAPI** |
| Validation / response models | **Pydantic v2** |
| Pydantic scope | **API boundary only** |
| Data source | **Parquet/CSV via existing dataset registry** — pre-computed static artifacts per D21 |
| Serve command | **`gridiron api serve`** |
| Endpoint coverage | **Full prototype shape, no cuts** |
| Placeholder convention | **`null` + `_meta.field_status`** |
| Placeholder granularity | **Field-level** |

### Tiers

**Tier 1 — Skeleton + blocked-endpoint stubs.** ✅ Complete (2026-06-27).
FastAPI app skeleton, `_meta` envelope plumbing, twelve endpoints returning 200 with structurally valid null responses carrying registered blocker slugs. `/docs` groups by domain.

**Tier 2 — Direct-serialization endpoints.** 🟡 Active. Steps 1–4 complete (10 endpoints populated). Step 5 (games) in progress.

| Step | Scope | Status |
|---|---|---|
| 1 | `/weeks/current` + all `/portfolio/*` | ✅ Complete (2026-07-01) |
| 2 | `/model/performance` | ✅ Complete (2026-07-01) |
| 3 | `/teams` + `/teams/{abbr}` | ✅ Complete (2026-07-01) |
| 4 | `/projections` | ✅ Complete (2026-07-01) |
| 5 | `/games`, `/games/{id}` | ✅ Complete (2026-07-02) |
| 6 | `/edges` | ✅ Complete (2026-07-02) |
| 7 | `/props`, `/props/{prop_id}` | ✅ Complete (2026-07-02) |
| 8 | `/compare/teams`, `/compare/player/{prop_id}` | 🟡 Active |

#### Step 5 — `/games`, `/games/{id}` ✅ Complete (2026-07-02)

Shipped in four substeps: loader with champion resolution and archive filtering,
schemas with pending/blocked field scaffolding, serializer with field_status
population, routes with `ChampionNotFoundError` translation to structured D14
metadata. `MiniRepoBuilder` extended with `with_champion_manifest` and
`with_predictions_archive` helpers. `/games/{game_id}/predictions` dropped
per YAGNI.

#### Step 6 — `/edges` ✅ Complete (2026-07-02)

Shipped in four substeps: loader with champion resolution, odds join,
and edge computation via `market.recommendations.build_edge_report`;
schemas with nullable `point_edge` and `cover_prob` for moneyline rows;
serializer with NaN normalization at the Pydantic boundary; route with
translation for `ChampionNotFoundError` → `NO_CHAMPION_MANIFEST` and
`OddsUnavailableError` → `NO_ODDS_AVAILABLE`. `MiniRepoBuilder` extended
with `with_odds_snapshot`. `api/exceptions.py` introduced for API-surface
data-state signals; `OddsUnavailableError` is its first entry.

#### Step 7 — `/props`, `/props/{prop_id}` ✅ Complete (2026-07-02)

Shipped in four substeps: loader with per-family champion resolution
via `resolve_current_champion` iterated across `PROP_STAT_FAMILIES`;
schemas with `ProjectionBlock` and `LineBlock` clusters, plus scaffolded
fields for historical/situational/reasoning/injury/recent-form/prop-shop;
serializer with `prop_id` composition (`{game_id}__{player_id}__{stat_type}`),
season int → string normalization, and field_status marking; routes with
`_decode_prop_id` helper and asymmetric exception translation (list:
`ChampionNotFoundError` → 200 empty; detail: `ChampionNotFoundError`
→ 200 with projection and line_context null and field_status blocked).

`_resolve_scope` refactored to lazy: only reads `NFL_wk_by_wk_cleaned.csv`
when a default is actually needed (bug fix, previously eager). Same lazy
pattern should be applied to `_resolve_scope` in `games.py`, `edges.py`,
and `teams.py` as a follow-up (tracked in D19 audit backlog).

MiniRepoBuilder gained no new methods; test-side helpers
(`_write_prop_manifest`, `_write_prop_archive`) inline in
`test_props_routes.py` for now.

#### Step 8 — `/compare/teams`, `/compare/player/{prop_id}` design

**Scope:** Read-only comparison endpoints. Both are heavily blocked by
Tier 3 additive datasets (per-stat percentile ranking, off/def
decomposition, opponent-allowed-by-position aggregation, cohort
splits). T2 ships the ~20% of fields we can populate from existing
data, with everything else scaffolded via `field_status`. Rationale:
close the API surface completely so W9 (Frontend) sees a 200 for
every prototype-referenced URL, even where most fields are pending.

**Substeps:**

- **8a — `/compare/teams`.** Loader reuses `load_elo_state_df` and
  `load_games_df` for team ratings and records. Schema uses a
  list-of-stat-rows shape (`stats: [{ label, team_a, team_b, unit }, ...]`)
  matching the prototype's compare-view table. Serializer builds the
  row list, populating `rating`, `rank`, `wins`, `losses`, `ties` from
  real data; blocking `off_rating`, `def_rating`, `trend`, cohort
  splits, and per-stat percentile ranks with `field_status: pending`
  or `field_status: blocked` slugs.
- **8b — `/compare/player/{prop_id}`.** Loader calls `load_prop` for
  the prop-side data (already in the archive). Defense-side data is
  entirely blocked pending opponent-allowed-by-position aggregation.
  Route decodes `prop_id` via the same `_decode_prop_id` helper as
  `/props/{prop_id}`. Schemas mirror the compare-teams row-list shape
  but with a projection vs defense-context axis instead of team-a vs
  team-b.

**Field scope (locked):**

Populated in `/compare/teams` response:
- `season`, `team_a`, `team_b` (echoed back)
- `stats` list:
  - `rating` (Elo) — populated per team from `load_elo_state_df`.
  - `record.wins`, `record.losses`, `record.ties` — populated from
    `load_games_df` for the season.
  - `rank` — populated per team from ranked Elo.

Scaffolded with `field_status`:
- `off_rating`, `def_rating` per team → **blocked**, slug
  `OFF_DEF_DECOMPOSITION` (already registered in `Unavailable`).
- `trend` per team → **blocked**, slug `NO_PRIOR_SNAPSHOT` (already
  registered).
- `schedule_difficulty`, `playoff_probability` per team → **pending**
  (Tier 3 additive).
- `cohort_splits` (season, L4, home, away) per team → **pending**
  (Tier 3 additive).
- Per-stat percentile ranks → **pending** (Tier 3 additive).

Populated in `/compare/player/{prop_id}` response:
- `prop_id`, `game_id`, `season`, `week`, `player_id`, `player_name`,
  `position`, `team`, `stat_type`, `model_key` (from `load_prop`)
- `projection` block: `predicted_mean`, `predicted_std`, `lo_90`,
  `hi_90` (from archive; same as `/props/{prop_id}`)

Scaffolded with `field_status`:
- `defense_context` block → **entirely blocked** pending
  opponent-allowed-by-position aggregation. New slug
  `OPPONENT_ALLOWED_BY_POSITION` scheduled for `Unavailable`.
- `line_context` — same treatment as `/props/{prop_id}` (pending on
  odds-join at prediction time).
- `situational_splits` for the player → **pending** (Tier 3
  additive).

**Filter model:**

- `/compare/teams?team_a=KC&team_b=LAC&season=`: `team_a` and `team_b`
  required. Unknown abbreviations → 404. `season` optional, defaults
  from lazy `_resolve_scope`.
- `/compare/player/{prop_id}`: no query params beyond the path. Same
  `prop_id` decode + 404 semantics as `/props/{prop_id}`.

**Champion-, team-, and prop-missing behavior:**

| State | Response |
|---|---|
| `team_a` or `team_b` unknown abbreviation | 404 |
| Season lookup fails (games CSV missing) | 500 (unchanged from `/teams/{abbr}` pattern) |
| `prop_id` malformed on `/compare/player/{prop_id}` | 404 |
| Champion for prop stat_type missing | 200 with projection null and `field_status.projection: blocked/NO_CHAMPION_MANIFEST` (mirrors `/props/{prop_id}`) |
| Prop not in archive | 404 |

**Design decisions locked:**

| Decision | Choice | Rationale |
|---|---|---|
| Framing | Full endpoints with 20% populated + 80% scaffolded | "Full prototype shape, no cuts" success criterion. W9 sees 200 for every URL. |
| `/compare/teams` response shape | List of stat rows | Matches prototype's compare-table visual. Extension-friendly. |
| `/compare/player/{prop_id}` response shape | Row-list mirror of `/compare/teams` | Consistent client rendering. |
| Loader reuse | Reuse `load_elo_state_df`, `load_games_df`, `load_prop` | No new loader machinery. Comparison is a serialization concern. |
| New `Unavailable` slug | `OPPONENT_ALLOWED_BY_POSITION` in 8b | Registers the blocker so its field_status has a stable identity. |
| `prop_id` decode | Reuse `_decode_prop_id` from `/props` | DRY. |

**Disconfirming evidence to watch for:**

- **If the row-list shape doesn't cleanly serialize** (e.g., needing
  discriminated unions on `label` to enforce which cells populate),
  fall back to a discriminated tagged-union schema. Not expected.
- **If the frontend needs sorted stats or grouped sections
  (Offense/Defense/Special Teams)**, the response might need a
  `groups: [{group_label, stats: [...]}]` structure. Ship flat first;
  frontend can group client-side.
- **If defense_context on `/compare/player/{prop_id}` is fully
  redundant with a future dedicated endpoint** (e.g., `/defense/{team}`
  when that lands in Tier 3), we may want to link rather than embed.
  Not a T2 concern.

**Tier 3 — Backend additions.** Designing. Additions inventory unchanged from original plan.

| Addition | Populates |
|---|---|
| Per-stat league-wide percentile ranking pass | Compare screen rank columns, Team Detail rank fields |
| Off/def rating decomposition | Team Rankings off/def split |
| Weekly Elo snapshot persistence | Team rating-history endpoint, projections week-over-week delta |
| Opponent-allowed-by-position aggregation | Player vs Defense view, Player Prop matchup section |
| Limited cohort splits (season, L4, home, away) per team | Game Detail split tabs, Compare splits |
| Limited cohort splits (indoor/outdoor, favored/underdog) per prop | Player Prop situational splits |
| Prior-week projection snapshot for delta | Projections 1-week change column |

---

## Changelog

| Date | Change |
|------|--------|
| 2026-07-02 | **W8 Tier 2 Step 8 design.** Inline design block added for `/compare/teams` and `/compare/player/{prop_id}`. Two substeps: `/compare/teams` (team-vs-team stat row list) and `/compare/player/{prop_id}` (projection vs defense context). Framing A locked — full endpoints with ~20% populated fields from existing data, ~80% scaffolded via `field_status` pending Tier 3 additive datasets. `OPPONENT_ALLOWED_BY_POSITION` new slug scheduled for `Unavailable`. Response shape: list of stat rows matching the prototype's compare-table visual. |
| 2026-07-02 | **W8 Tier 2 Step 7 complete.** `/props` and `/props/{prop_id}` shipped in four substeps (loader, schemas, serializer, routes). Per-family champion resolution iterates `PROP_STAT_FAMILIES`; each family independent. `prop_id` composite `{game_id}__{player_id}__{stat_type}`. Line-derived fields (`line`, `p_over`, `lean`, `confidence_tier`) 100% null in T2, scaffolded as `field_status: pending` — odds-join at prediction time not yet implemented. `_resolve_scope` refactored to lazy read of games CSV. Endpoints populated so far: 14. |
| 2026-07-02 | **W8 Tier 2 Step 7 design.** Inline design block added for `/props` and `/props/{prop_id}`. Four substeps: loader with per-family champion resolution (7a), schemas with ProjectionBlock and scaffolded line-dependent fields (7b), serializer (7c), routes with `prop_id` decoding (7d). Archive verification revealed `line`, `p_over`, `lean`, `confidence_tier` are 100% null in the archive today; all four scaffolded as `field_status: pending`. `prop_id` composite locked as `{game_id}__{player_id}__{stat_type}`. Season string normalized to int for archive read. |
| 2026-07-02 | **W8 Tier 2 Step 6 complete.** `/edges` shipped in four substeps (loader, schemas, serializer, route). `api/exceptions.py` introduced with `OddsUnavailableError` for loader → route data-state signaling. `MiniRepoBuilder` gained `with_odds_snapshot`. `NO_ODDS_AVAILABLE` registered in `Unavailable`. Two-exception route pattern (`ChampionNotFoundError` and `OddsUnavailableError` translating to distinct `field_status` slugs) validated end-to-end. Endpoints populated so far: 12. |
| 2026-07-02 | **W8 Tier 2 Step 6 design.** Inline design block added for `/edges`. Four substeps: loader (6a), schemas (6b), serializer (6c), route (6d). Field scope, filter model, and both loader-signaled data-state exceptions (`ChampionNotFoundError`, `OddsUnavailableError`) locked. `NO_ODDS_AVAILABLE` slug scheduled for `Unavailable`. `api/exceptions.py` scheduled as a new module. |
| 2026-07-02 | **W8 Tier 2 Step 5 complete.** `/games` and `/games/{id}` shipped in four substeps (loader, schemas, serializer, routes). Champion resolution threads from manifest through loader to Pydantic response. `MiniRepoBuilder` gained `with_champion_manifest` and `with_predictions_archive` methods. `NO_CHAMPION_MANIFEST` slug registered in `Unavailable`. Two lessons applied for future integration tests: `dependency_overrides` keys on the exact function inside `Depends(...)`, and D19 `repo_root` threading needs an audit sweep across `api/loaders.py`. Endpoints populated so far: 12. |
| 2026-07-01 | **W8 resumed; Tier 2 Step 5 in progress.** PLAN.md restructured: W13 complete block removed (moved to CHANGELOG), W8 promoted from Paused to Current Workstream. Step 5 rescoped: `/games` and `/games/{id}` (dropped `/games/{game_id}/predictions` per YAGNI). Substep 5a (games loader in `api/loaders.py`) complete: `load_games_for_week` and `load_game` resolve the current win_prob champion, filter the prediction archive, and translate to API-facing shape. |
| 2026-07-01 | **W13 Tier 3 complete. W13 workstream closed.** Four steps: resolve_win_prob_model_type helper, weekly_predict refactor, edges refactor (both report and clv), intentional-Elo annotations. Actual scope was 3 CLI-option default sites (not 8, per original handoff estimate); the other 5 sites were provenance labels or intentional Elo usage and got comments instead of refactors. W8 (API) unpauses; Tier 2 Step 5 (game endpoints) is now unblocked. |
| 2026-07-01 | **W13 Tier 2 complete.** Nine steps shipped: manifest writer, three selectors, full-retrain integration, baseline-report annotation, two --write-manifest CLI flags, and the champion_cmd refactor. All champion decisions across CLI and stage surfaces share the same code path. Central catalog at gridiron_edge.models.catalog is now the single source of truth for model pairs and prop families. Tier 3 (CLI consumer refactor) begins. |
| 2026-07-01 | W13 workstream definition locked. Scope: persist the champion decision that `evaluate select-model` and `select_prop_champion` already compute; expose via `resolve_current_champion(model_name)`; hook the write into `full-retrain` as a new stage; refactor 8 hard-coded CLI callsites. Three tiers: manifest+resolver, writer+integration, consumer refactor. Tier 1 verification to follow. |
| 2026-07-01 | **W8 paused; W13 opened.** W8 Tier 2 Step 5 pre-planning discovered no runtime champion resolution for game models. Per D21 (API is a serialization boundary, not a compute boundary), the champion decision must be a static artifact. Elevated to W13 (Runtime Champion Resolution) as a new workstream. W8 pauses in Tier 2 at Step 4; resumes when W13 closes. Design phase for W13 to follow. |
| 2026-07-01 | Tier 2 Step 4 complete: /projections. Reads Monte Carlo season/playoff projections CSV; returns 32-team ranking with staleness timestamp. One new Unavailable slug (NO_PROJECTIONS_DATA). Endpoints populated so far: 10. |
| 2026-07-01 | Tier 2 Step 3 complete: /teams and /teams/{abbr}. First multi-source endpoint composition — Elo state + games records + team name normalization. Two new Unavailable slugs (NO_PRIOR_SNAPSHOT, OFF_DEF_DECOMPOSITION). Introduced resolve_current_season_week as a shared loader. Endpoints populated so far: 9. |
| 2026-07-01 | Tier 2 Step 2 complete: /model/performance. Combines model prediction quality (via evaluation/metrics.py) with betting performance (via betting/performance.py) into a single nested response. Two new Unavailable slugs (NO_EVALUATION_DATA, SINGLE_CLASS_OUTCOME) for data-limit fields. Endpoints populated so far: 7. **Note: violates D21 by computing at request time; deferred refactor tracked in ROADMAP §9.6.** |
| 2026-07-01 | Tier 2 Step 1 complete: /weeks/current + /portfolio/{summary,bets,curve,transactions,splits}. Introduced api/loaders.py, api/serializers/ package. D19 records explicit repo_root threading; D20 extends placeholder convention with Unavailable slugs for data-limit and missing-query-param cases. |
| 2026-06-27 | Tier 2 design phase complete. Inline "How" block expanded with three-layer architecture (loaders → serializers → routes), 8-step implementation order, locked decisions D17 (per-endpoint serializers) and D18 (serializer-owned field_status), and the inventory of pending fields expected to surface during the tier. |
| 2026-06-27 | Tier 1 complete. Skeleton + blocked-endpoint stubs shipped. 12 endpoints reachable via `gridiron api serve` with structurally valid null responses carrying registered blocker slugs. Integration tests lock round-trip parity and field_status completeness. |
| 2026-06-26 | Tier 1 wiring verified end-to-end. All 12 endpoints reachable via `gridiron api serve`. |
| 2026-06-23 | W8 Tier 1 design phase complete. Four-layer architecture (meta → schemas/_base → app/deps → Tier 3 routes), module layout, locked decisions per D16, and 8-step implementation order. |
| 2026-06-23 | PLAN.md restructured to focus on the active workstream only. Future workstream candidates, real-bugs backlog, investigations, and operational items migrated to ROADMAP.md §9. W8 (API Serving Layer) set as active workstream. |
