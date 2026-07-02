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

### Current Workstream: W9 — Frontend

### What we are building

A React web app that consumes the Gridiron Edge REST API end-to-end.
Wires the frontend prototype (20 screens, 3 Contexts, OKLCH dark theme)
to the live 16-endpoint API from W8 Tier 2. Every populated field
renders real data; every `field_status` marker gets a consistent visual
treatment; blocked screens render "coming soon" placeholders.

Handoff package (README.md + `source/`) from the design workstream is
the reference. Prototype is HTML/JSX with in-browser Babel and mock
data; W9 recreates the designs in Vite + React + TypeScript with real
API integration.

### Why we are building it

1. **Verification of the API.** W8 Tier 2 built against the prototype
   as a shape reference. Wiring the real frontend to the real API is
   the next verification loop — surfaces schema mismatches, missing
   fields, unexpected `field_status` states.
2. **Roadmap discovery.** Which `field_status` states frustrate users,
   which additive Tier 3 datasets unlock the most value — discovered
   only by wiring it up. W8 Tier 3 kickoff waits for this signal.
3. **First user-facing surface.** All prior workstreams built backend
   value. W9 surfaces that value visually.

#### Success criteria

- Local dev loop works: `gridiron api serve` in one terminal, `pnpm
  dev` (or equivalent) in another; all 20 screens render.
- Every populated field on every screen shows real API data.
- Every `field_status: pending` field renders a consistent pending
  state.
- Every `field_status: blocked` field renders a "not available" state
  with the blocker slug + roadmap reference visible in tooltip.
- Four entirely-blocked screens (LineShopping, LiveGame, NewsWire,
  ExplainPage) render full-screen "coming soon" cards.
- OpenAPI schema is checked in and versionable; `gridiron api
  export-schema` command exists.
- Quality gates: `pnpm typecheck`, `pnpm test`, `pnpm build` pass.

### Disconfirming evidence

- **If schema shape mismatches surface,** those are W8 bugs. Track
  and fix in W8, not W9.
- **If a screen requires an endpoint that doesn't exist,** the
  prototype evolved after W8 Tier 2 scoping or the endpoint inventory
  was wrong. Add a small W8 patch step; do not shoehorn into W9.
- **If `field_status` state proliferation makes screens unusable,**
  the placeholder convention (D14) may need refinement. Consider
  screen-level "under construction" banners in place of per-field
  markers. Discover during Tier 2.
- **If OKLCH color rendering has browser compat issues** (older
  browsers not on the target list), a fallback palette may be needed.
- **If the three-Context state model doesn't scale** with React
  Query's async data, adopt Zustand or Jotai. Prototype state is
  synchronous only.

### Locked architectural decisions

| Decision | Choice | Rationale |
|---|---|---|
| Framework | Vite + React + TypeScript | Prototype README default; portable from JSX |
| Styling | CSS variables + CSS Modules | Preserves the OKLCH token table 1:1 |
| State | Three React Contexts (Nav, BetSlip, AppState) | Match prototype; low migration risk |
| Data fetching | React Query (TanStack Query) | Standard for API-backed React; handles loading/error/cache |
| API client | Generated from OpenAPI schema | Reproducible, typed, versionable |
| Schema source | Checked-in `api-schema.json` via `gridiron api export-schema` | Reproducible, no server dep during frontend build |
| Routing | Hash-based (`#/route`) | Match prototype; no server config needed |
| Deployment | Local dev only for W9 | Verification workstream, not productization |
| Testing | Vitest + React Testing Library (unit); Playwright (e2e, optional) | Standard React stack |
| Field_status: pending | Placeholder + info badge, `--ink-4` faint text | Distinct from blocked; preserves layout |
| Field_status: blocked | "Not available" state, tooltip shows blocker + roadmap | Distinct from pending; user affordance for "not soon" |
| Entirely-blocked screens | Full-screen "coming soon" card | Better UX than broken shell |

### Prerequisite: OpenAPI schema export command

Not a W9 deliverable. Small W8 patch: add
`gridiron api export-schema [--output api-schema.json]` that serializes
the FastAPI app's OpenAPI spec to a checked-in JSON file. W9 build
consumes it via the client generator. Estimated one commit; can land
during W9 Tier 1 or as a standalone W8 patch step.

### Tiers

**Tier 1 — Client infrastructure.**

Deliverable: local dev server serving an empty shell that can hit the
API and render one screen (Dashboard) end-to-end.

Substeps:
1. Vite + React + TypeScript scaffolding, Geist font loading, base
   `styles.css` port.
2. Chrome components (TopNav, SubNav, Breadcrumb, Frame).
3. Three Contexts (AppState, BetSlip, Nav) with localStorage /
   sessionStorage persistence.
4. `gridiron api export-schema` command (W8 patch).
5. API client generation from checked-in OpenAPI schema.
6. React Query setup with base client, loading/error states.
7. Dashboard route wired to `/weeks/current` + `/games?week=` —
   proves the loop works.

**Tier 2 — Populated screens.**

Deliverable: all 12 API-consuming screens render real data.

Substeps grouped by domain:
1. Games (GamesList + GameDetail) → `/games`, `/games/{id}`.
2. Teams (TeamRankings + TeamProfile) → `/teams`, `/teams/{abbr}`.
3. Projections → `/projections`.
4. Players / Props (PlayersExplorer + PlayerProp) → `/props`,
   `/props/{prop_id}`, `/compare/player/{prop_id}`.
5. Compare (ComparePage) → `/compare/teams`.
6. Bankroll → `/portfolio/*`.
7. BetSlip → client-side + `/edges` integration for staging.

**Tier 3 — Blocked screens + polish.**

Deliverable: 20-screen complete UI.

Substeps:
1. Blocked-screen placeholders (LineShopping, LiveGame, NewsWire,
   ExplainPage).
2. Client-side screens (Onboarding, Settings, Tools).
3. Aesthetic variants decision (Terminal / Fintech / Editorial or none).
4. Screen-level integration testing.
5. A11y sweep.
6. Error states (network failures, backend down).

Tier design blocks are drafted at the start of each tier.

---

## Paused Workstreams

### W8 — API Serving Layer

**Status:** Paused (Tier 2 complete; Tier 3 pending W9 feedback).

**Where we stopped:** Tier 2 complete (2026-07-01). 16 endpoints
populated. Every prototype-referenced URL returns a 200 with a
Pydantic-validated shape. Fields not yet populated are marked with
structured `field_status`.

**How this resumes:** When W9 identifies which Tier 3 additive dataset
provides the most user value. Frontend feedback drives which of:

- Per-stat league-wide percentile ranking pass
- Off/def rating decomposition
- Opponent-allowed-by-position aggregation
- Cohort splits (season/L4/home/away, indoor/outdoor)
- Weekly Elo snapshot persistence for trend fields
- Prior-week projection snapshot for delta

...to build first. When W9 signals a priority, W8 Tier 3 opens with that
additive scoped as the first step. Substeps mirror Tier 2's rhythm
(design → loader → schema → serializer → route → integration test).

#### Tier 2 summary retained inline

**Tier 2 — Direct-serialization endpoints.** ✅ Complete (2026-07-01).
Eight steps shipped. 16 endpoints populated with real data. Every
prototype-referenced URL returns a 200 with a validated Pydantic shape.
Fields not yet populated (per Tier 3 additive datasets) are marked with
structured `field_status` metadata per D14.

#### Tier 3 additions inventory (unchanged)

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
| 2026-07-02 | **W9 design phase complete.** Design block locked: Vite + React + TypeScript, CSS variables + CSS Modules, three-Context state model matching the prototype, React Query for API data, generated API client from checked-in OpenAPI schema (via new `gridiron api export-schema` W8 patch). Three tiers: client infrastructure (7 substeps), populated screens (7 substeps grouped by domain), blocked screens + polish (6 substeps). Field_status rendering strategy locked: pending as placeholder + info badge; blocked as "not available" state with tooltip; entirely-blocked screens as full-screen "coming soon" cards. |
| 2026-07-02 | **W9 promoted to Current Workstream.** W8 Tier 2 complete; W8 Tier 3 paused pending W9 signal on which additive dataset to build first. W9 design phase to follow — likely 3-tier structure covering client infrastructure, populated screens, and `field_status`-driven blocked-state rendering. |
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
