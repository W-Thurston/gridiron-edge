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
| 5 | `/games`, `/games/{id}` | ✅ Complete (2026-07-01) |
| 6 | `/edges` | ✅ Complete (2026-07-01) |
| 7 | `/props`, `/props/{prop_id}` | 🟡 Active |
| 8 | `/compare/teams`, `/compare/player/{prop_id}` | Not started |

#### Step 5 — `/games`, `/games/{id}` ✅ Complete (2026-07-01)

Shipped in four substeps: loader with champion resolution and archive filtering,
schemas with pending/blocked field scaffolding, serializer with field_status
population, routes with `ChampionNotFoundError` translation to structured D14
metadata. `MiniRepoBuilder` extended with `with_champion_manifest` and
`with_predictions_archive` helpers. `/games/{game_id}/predictions` dropped
per YAGNI.

#### Step 6 — `/edges` ✅ Complete (2026-07-01)

Shipped in four substeps: loader with champion resolution, odds join,
and edge computation via `market.recommendations.build_edge_report`;
schemas with nullable `point_edge` and `cover_prob` for moneyline rows;
serializer with NaN normalization at the Pydantic boundary; route with
translation for `ChampionNotFoundError` → `NO_CHAMPION_MANIFEST` and
`OddsUnavailableError` → `NO_ODDS_AVAILABLE`. `MiniRepoBuilder` extended
with `with_odds_snapshot`. `api/exceptions.py` introduced for API-surface
data-state signals; `OddsUnavailableError` is its first entry.

#### Step 7 — `/props`, `/props/{prop_id}` design

**Scope:** Read-only endpoints exposing per-player prop projections
from the current champion for each stat family. `/props` lists edges
across families for a (season, week); `/props/{prop_id}` returns
detail for one player-game-stat combination.

**Substeps:**

- **7a — Props loader.** Two functions in `api/loaders.py`:
  `load_props_for_week` (list) and `load_prop` (detail). Both iterate
  the registered prop stat families, resolve each family's champion via
  `resolve_current_champion(family)`, and filter the prop archive to
  the champion. Families without a resolved champion are silently
  skipped. Season parameter is normalized: API accepts
  `"2026-2027"` and passes the leading int (`2026`) to `load_prop_archive`.
- **7b — Props schemas.** `PropSummary`, `PropDetail`, `PropList` in
  `api/schemas/props.py`. `ProjectionBlock` clusters champion output
  (`predicted_mean`, `predicted_std`, `lo_90`, `hi_90`). Line-dependent
  fields (`line`, `p_over`, `lean`, `confidence_tier`) are nullable
  and scaffolded on `PropDetail` with `field_status: pending`
  (odds-join not yet implemented at prediction time). Additional
  scaffolds on `PropDetail`: historical vs opponent, situational
  splits, prop reasoning, injury status, recent form, multi-book
  shopping.
- **7c — Props serializer.** `api/serializers/props.py` with
  `serialize_prop_summary`, `serialize_props_list`, and
  `serialize_prop_detail`. `_row_to_projection` clusters the four
  champion fields; NaN normalization at the Pydantic boundary via
  `_none_if_nan`.
- **7d — Props routes.** `api/routes/props.py` with `GET /props` and
  `GET /props/{prop_id}`. `prop_id` decoded as
  `{game_id}__{player_id}__{stat_type}` (double-underscore separator).
  Empty-champion state translates to 200 with
  `field_status.items: blocked/NO_CHAMPION_MANIFEST` when zero families
  resolved; empty archive returns 200 with no field_status. Malformed
  `prop_id` or unknown prop returns 404.

**Field scope (locked):**

Populated in `PropSummary`:
- `prop_id` (composite: `{game_id}__{player_id}__{stat_type}`)
- `game_id`, `season` (string form: `"2026-2027"`), `week`
- `player_id`, `player_name`, `position`, `team`
- `stat_type`, `model_key` (composite `{model_name}_{model_type}`)
- Projection: `predicted_mean`, `predicted_std`, `lo_90`, `hi_90`

Scaffolded (field_status: pending) on both `PropSummary` and `PropDetail`:
- `line` — pending: odds-join not yet run at prediction time. Depends
  on odds ingest running before or after prediction generation
  (implementation choice for a future backend workstream).
- `p_over`, `lean` — pending: derived from `line`; blocked on `line`.
- `confidence_tier` — pending: derived from `p_over` in current
  post-processing; blocked on `line`.

Scaffolded on `PropDetail` only (list rows stay lean):
- `historical_vs_opponent` → **pending** (opponent-adjusted matchup
  aggregation; ROADMAP §9 Tier 3 additive)
- `situational_splits` → **pending** (indoor/outdoor,
  favored/underdog, home/away splits per prop; ROADMAP §9 Tier 3
  additive)
- `prop_reasoning` → **blocked on feature attribution workstream**
- `injuries` → **blocked on ROADMAP §5.3**
- `recent_form` → **pending** (last-N-games aggregation)
- `multi_book_shopping` → **blocked on multi-book odds ingest (W7)**

**Filter model:**

`/props?season=&week=&stat_type=&position=&min_p_over=`

- `season`, `week`: default from `resolve_current_season_week()`.
  Season passed as `"2026-2027"` and normalized to leading int for
  archive read.
- `stat_type`: default null (all families). When set, only that
  family's champion is resolved and its rows returned.
- `position`: default null (all positions — implicitly filtered by
  `stat_type` when both are passed).
- `min_p_over`: default null (no filter). Only meaningful when `line`
  populates. Reserved for future backend work; ignored today.

**Champion- and archive-missing behavior:**

| State | Response |
|---|---|
| Manifest missing for all families | 200, `items: []`, `field_status.items: blocked/NO_CHAMPION_MANIFEST` |
| Manifest present but no families have entries | 200, `items: []`, `field_status.items: blocked/NO_CHAMPION_MANIFEST` |
| Some families have champion, others don't | 200, list of resolved families (silent skip) |
| Archive empty for resolved champions | 200, `items: []`, no `field_status` (legitimate) |
| No rows match (season, week) | 200, `items: []`, no `field_status` (legitimate) |
| `prop_id` malformed | 404 |
| `prop_id` not in archive | 404 |

**Design decisions locked:**

| Decision | Choice | Rationale |
|---|---|---|
| Champion resolution granularity | Per-stat-family, iterated in loader | Matches how `promote-champions` writes per-family entries. Each family independent. |
| `prop_id` format | `{game_id}__{player_id}__{stat_type}` | Composite matches archive dedup key minus (model_name, model_type). Double-underscore safe from single-underscore game_id/stat_type. |
| Season normalization | API accepts string ("2026-2027"), loader passes int (2026) to archive | Bridges API convention with archive convention. |
| Line-dependent fields | All null in T2 with `field_status: pending` | Archive today doesn't populate line at prediction time. Not blocked on a workstream — just deferred backend work. |
| Missing-family behavior | Silent skip | Families are independent domains. Frontend can call `list_current_champions` separately to see what's registered. |
| Detail exception translation | 404 for both malformed and unknown | Same convention as `/games/{game_id}`. |

**Disconfirming evidence to watch for:**

- **If the frontend needs to know which families have no champion,**
  we may need to add a `field_status.families` list or similar. Not
  scoped today.
- **If `predicted_std`, `lo_90`, `hi_90` are sometimes null in the
  archive** (edge case: early implementation before uncertainty
  quantification landed), the schema needs to accommodate it.
  Verified today: 100% populated for the existing 1,433 rows.
- **If `stat_type` becomes a plural filter** (e.g., `?stat_type=qb_pass_yards,qb_rush_yards`),
  the query parsing changes. Not in T2 scope.

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
