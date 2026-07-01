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
| Complete | Tier success criteria met (summary retained inline) |
| Re-scoped | Tier success criteria invalidated; replaced or dropped |

Workstream identifiers (W1, W2, …) match ROADMAP.md. They exist only inside the planning docs (PLAN, ROADMAP, DECISIONS, CHANGELOG) — never in source code, comments, or commit subjects.

---

## Current Workstream: W8 — API Serving Layer

### What we are building

A read-only REST API that exposes every analytics output Gridiron Edge produces, shaped to match the Gridiron Edge frontend prototype end-to-end. Every screen in the prototype gets the endpoints it needs; every field that the backend can populate today returns real data; every field the backend can't yet produce returns `null` accompanied by metadata describing why.

The API is implemented as a FastAPI app with Pydantic v2 response models, mounted as a new `api/` package and served via `gridiron api serve`. Pydantic is confined to the `api/` boundary — domain code (models, evaluation, market, features) stays pandas/dataclass-shaped.

### Why we are building it

Three reinforcing reasons:

1. **Verification surface.** The CLI surfaces outputs one-at-a-time. The frontend prototype puts ~19 screens worth of outputs side-by-side. Wiring the prototype to the API is the next quality-assurance step — every populated field gets verified by inspection, every placeholder field becomes a visible roadmap signal. The act of building this *is* the verification pass that the CLI cannot perform.
2. **Frontend unblock.** W9 (Frontend) cannot start until there is an API to wire to. W8 is the gating dependency for M5 (friends can use it).
3. **Roadmap discovery.** The placeholder fields the API ships will form a structured, observable inventory of "what's missing" that drives ROADMAP §9 (Known Issues & Backlog) prioritization. We learn what's worth building next by seeing what's empty in the UI.

### Success criteria

- Every endpoint in the prototype-driven inventory (Tier 1 + Tier 2 + Tier 3) returns a 200 response with a valid Pydantic-validated shape.
- Every Tier 1 endpoint returns real data for ≥80% of its fields, with the remainder explicitly marked in `_meta.field_status`.
- Every Tier 2 endpoint has a documented backend addition shipped or explicitly deferred with a Tier 2 → Tier 3 demotion.
- Every Tier 3 endpoint returns `null` fields with structured `_meta.field_status` entries naming a blocker.
- `gridiron api serve` starts the API and surfaces OpenAPI docs at `/docs`.
- Test coverage: unit (response model shape + `_meta` correctness), integration (`MiniRepoBuilder`-backed), with e2e deferred to W9.
- All quality gates pass (ruff, pyrefly, pytest).

### Disconfirming evidence

What would tell us this workstream is the wrong thing to build (or scoped wrong) before completion?

- If during Tier 1 we discover that the dataset registry can't cheaply serve the shapes the prototype expects without significant new aggregation work, the "files-first" architecture decision may be wrong and we'd need to revisit ROADMAP §5.1 (file vs. database) before continuing.
- If during Tier 2 the placeholder convention (`null` + `_meta.field_status`) proves too noisy in practice — frontend consumers find it hard to reason about, or `_meta` blocks balloon — we'd revisit the convention before committing the full surface.
- If the prototype's expected shapes drift substantially during the workstream (you decide a screen should look fundamentally different), Tier 1 endpoints may need to be re-shaped rather than incrementally extended.

### Locked architectural decisions

| Decision | Choice | Rationale |
|---|---|---|
| Framework | **FastAPI** | Async, type-driven, free OpenAPI docs at `/docs`. |
| Validation / response models | **Pydantic v2** | Native FastAPI integration. Heavy dep accepted, scoped to `api/` only. |
| Pydantic scope | **API boundary only** | Domain code stays pandas/dataclass-shaped. Pydantic lives in `api/schemas/`. |
| Data source | **Parquet/CSV via existing dataset registry** | No DB in W8. Revisit during W9 if hot paths or query complexity demand it (ROADMAP §5.1 trigger). |
| Serve command | **`gridiron api serve`** | Consistent with rest of CLI surface; wraps `uvicorn`. |
| Endpoint coverage | **Full prototype shape, no cuts** | Every screen gets the endpoints it needs. Unsupported data is field-level placeholders, not missing endpoints. |
| Placeholder convention | **`null` + `_meta.field_status`** | Unpopulated fields return `null` with an entry in the response's `_meta.field_status` dict (`"pending"` for backend work; `{status, blocker, roadmap}` for upstream-workstream blockers). Revisit if noisy. |
| Placeholder granularity | **Field-level** | One status per field. Section-level rollup deferred unless duplication becomes a problem. |

### Open design questions (resolved during tier design, not at workstream start)

1. Endpoint shape — flat vs. nested. `GET /games/{id}` returns full nested team + edge + prop data, or flat with separate `/games/{id}/edges` calls? Lean flat-and-separate.
2. Caching layer. Registry-direct per request, or thin in-memory cache keyed on (dataset, modified-time)? Default: registry-direct. Likely first hit: percentile rankings for Compare and Game Detail.
3. Pagination / filtering conventions. Query-param-based filtering is obvious; pagination probably unnecessary at NFL scale. Document the stance.
4. Error model. Pydantic ValidationError → 422 is automatic. Project-level convention needed for domain errors (missing week, unknown model). HTTP codes + plain JSON vs. structured envelope.

---

## Tiers

### Tier 1 — Skeleton + blocked-endpoint stubs

**Status:** Complete (2026-06-27)

#### Summary

Shipped the FastAPI app skeleton, the `_meta` envelope plumbing, and every blocked endpoint returning its null-shape response with structured `_meta.field_status` entries.

Concretely:

- **`api/meta.py`** — `FieldStatus` discriminated union (`"pending"` literal or `BlockedStatus` object), `ResponseMeta` envelope with `with_pending`/`with_blocked` builders, `Blocker` registry of `(slug, roadmap)` tuples matching ROADMAP §9.5.
- **`api/schemas/_base.py`** — `BaseResponse` (frozen, `extra="forbid"`) with `response_meta` aliased to `_meta` on the wire; `BaseListResponse[T]` generic for list-shaped endpoints.
- **`api/app.py` + `api/deps.py`** — `create_app()` factory, OpenAPI tag inventory, dataset registry and settings dependencies, permissive dev CORS.
- **`cli/api.py`** — `gridiron api serve` wrapping uvicorn.
- **Nine route files** under `api/routes/`, each owning one blocker domain: `lines.py`, `live.py`, `news.py`, `injuries.py`, `explain.py`, `swing_factors.py`, `comparables.py`, `prop_shop.py`, `prop_reasoning.py`.
- **Nine schema files** under `api/schemas/` describing the prototype-expected response shape for each route file.

Twelve endpoints currently return 200 with structurally valid null responses carrying registered blocker slugs. `/docs` groups endpoints by domain. Quality gates green; 175+ unit and integration tests.

One implementation friction worth noting for Tier 2: Pyrefly does not model Pydantic's `populate_by_name=True` runtime behavior, so route construction sites use `# pyrefly: ignore[unexpected-keyword]` when passing `response_meta=...`. This is the same "legitimate workaround for known stub limitations" pattern documented in W5.5. If we change the alias convention in Tier 2, the suppressions can be removed.

Tests in `tests/integration/api/test_api_contract.py` lock in two cross-cutting contracts: every response round-trips through its declared model, and every null field has a corresponding `_meta.field_status` entry (D14 enforced, not just convention).

---

### Tier 2 — Direct-serialization endpoints

**Status:** Active

#### What we are building

The Tier 1 (direct-serialization) endpoints, each reading from the dataset registry and serializing through a Pydantic response model. This is the bulk of the populated surface area.

#### Why

This is where the API stops being a skeleton and starts being a verification surface. Every endpoint wired here returns real model output and exposes any field-level gaps that the data pipeline didn't anticipate. Each endpoint is independent — partial completion still ships a coherent API.

#### Success criteria

- All Tier 1 endpoints (per the inventory below) return real data for ≥80% of their fields.
- Remaining fields are explicitly marked `pending` in `_meta.field_status`.
- Integration tests against `MiniRepoBuilder` fixtures cover each endpoint's happy path.
- No Pydantic imports outside `api/`.
- Quality gates green.

#### Disconfirming evidence

- If serializing more than ~3 endpoints requires reaching across multiple domain modules in awkward ways, the dataset registry abstraction may be insufficient for API consumption and we'd revisit before wiring the rest.
- If the response time for the heavier endpoints (Edges, Projections) is unacceptable on a single request, we pull caching forward from Tier 3.

#### How

**Architecture: thin loader layer + per-endpoint serializers + thin routes.**

1. **`api/loaders.py`** — thin wrappers around the existing dataset loaders (`datasets/loaders.py`, `evaluation/archive.py`, `evaluation/prop_archive.py`, betting modules, etc.). Each wrapper takes `Settings` as input and returns a DataFrame or domain object. This is the single seam where caching gets added if a hot path emerges; routes never import from outside `api/` for data access.
2. **`api/serializers/`** — one function per endpoint. Each takes loader output and returns the Pydantic response model from `api/schemas/`. Per D17, serializers are hand-written (5–15 lines each) rather than driven by a column-mapping engine. Per D18, the serializer owns the construction of `_meta.field_status` for fields it can and cannot populate.
3. **`api/routes/` Tier 2 files** — small FastAPI handlers. Each pulls the loader through `api/deps.py` dependency injection, calls the serializer, returns the result. Routes are thin enough that most are 5–10 lines.

**Module layout additions:**

```
src/gridiron\_edge/api/
├── loaders.py                      # NEW
├── routes/
│   ├── weeks.py                    # NEW
│   ├── games.py                    # NEW
│   ├── edges.py                    # NEW
│   ├── teams.py                    # NEW
│   ├── projections.py              # NEW
│   ├── props.py                    # NEW
│   ├── portfolio.py                # NEW
│   ├── model.py                    # NEW
│   └── compare.py                  # NEW
├── schemas/
│   ├── weeks.py                    # NEW
│   ├── games.py                    # NEW
│   ├── edges.py                    # NEW
│   ├── teams.py                    # NEW
│   ├── projections.py              # NEW
│   ├── props.py                    # NEW
│   ├── portfolio.py                # NEW
│   ├── model\_performance.py        # NEW
│   └── compare.py                  # NEW
└── serializers/                    # NEW package
├── **init**.py
└── <one file per route file>
```

Tests mirror this layout under `tests/unit/api/` (per-schema, per-serializer, per-loader) and `tests/integration/api/` (the existing `test_api_contract.py` extends to cover the new endpoints).

**Locked design decisions** (full rationale in DECISIONS.md D17 and D18):

- D17: per-endpoint hand-written serializers rather than reflection-driven mapping. Boilerplate is acceptable; transparency pays off when columns get renamed.
- D18: serializers own `_meta.field_status` construction. Routes stay thin; pending-field knowledge lives with the code that has the most context.

**Open design questions resolved during implementation:**

1. Time-window semantics on `/portfolio/curve?period=` — calendar days or activity days? Decide during the portfolio step.
2. Default behavior of `/games` and `/edges` without a `?week=` param — current week vs. full season. Decide during the games step.
3. `/compare` percentile-rank caching — compute per-request or cache. Decide during the compare step based on measured response time.

**Implementation order (eight steps, simplest-first):**

| Step | Endpoints | Why this position |
|---|---|---|
| 1 | `/weeks/current`, all `/portfolio/*` | Smallest data shape, no model integration. Proves the loader-serializer-route pattern end-to-end. |
| 2 | `/model/performance` | Reuses portfolio machinery for model-quality metrics. Small extension. |
| 3 | `/teams`, `/teams/{abbr}` | Introduces the multi-source pattern (Elo + records + schedule). |
| 4 | `/projections` | Single source (Monte Carlo CSV output). Tests CSV-shaped serialization. |
| 5 | `/games`, `/games/{id}`, `/games/{id}/predictions` | Multi-source: predictions archive + schedule + edges. Composite identity flows through. |
| 6 | `/edges` | Builds on Step 5 understanding. Per-week CSVs. |
| 7 | `/props`, `/props/{prop_id}` | Parallel to games but reads from prop archive. |
| 8 | `/compare/teams`, `/compare/player/{prop_id}` | Most novel aggregation (percentile ranks, opponent-allowed-by-position). May add backend computations. |

Each step begins with a verification mini-block confirming loader signatures and column names against the actual codebase, then proceeds through schema → serializer (with unit tests) → loader wrapper (with unit tests) → route → integration test extension.

**Quality gates per step:**

```
uv run ruff check . --fix && uvx pyrefly check && uv run pytest tests/unit/api tests/integration/api -v
```

Plus a live smoke test on each new endpoint, hand-spotted against the underlying dataset to confirm the serializer produces correct values. Hand-spotting is the actual verification work that Tier 2 exists to do — it's what makes the API a verification surface rather than just a new abstraction layer.

**Pending placeholders we expect to surface** (informs ROADMAP §9.5 refinement during the tier):

| Endpoint | Pending field | Reason | Resolution path |
|---|---|---|---|
| `/games/{id}` | `injuries` | No injury data source | Blocked on §5.3 |
| `/games/{id}` | `swing_factors` | No feature attribution | Blocked on feature attribution |
| `/games/{id}` | `storyline`, `network` | No game metadata source | Pending |
| `/teams/{abbr}` | `off_rating`, `def_rating` | Composite-only rating today | Pending — may add in tier |
| `/teams/{abbr}` | `splits.l4`, `splits.vs_winning` | No arbitrary cohort splits | Pending |
| `/projections` | `week_over_week_delta` | No prior-week snapshot | Pending — may add in tier |
| `/props/{id}` | `splits.indoor`, `splits.vs_top10_def` | No prop cohort splits | Pending |
| `/compare/teams` | `percentile_ranks` | No league-wide percentile computation | Pending — should add in tier |
| `/compare/player/{id}` | `defense_allowed_by_position` | No aggregation today | Pending — should add in tier |

Fields marked "may add in tier" or "should add in tier" are candidates for backend additions during Tier 2. Each is decided per-step based on cost; items that prove larger than expected get demoted to ROADMAP §9 with explicit rationale.

---

### Tier 3 — Tier 2 backend additions

**Status:** Designing

#### What we are building

The small backend additions needed to populate fields that Tier 1 endpoints currently mark `pending`. Each addition is a discrete, well-scoped piece of work with a clear "field X on endpoint Y populates" success signal.

#### Why

These are the items where the prototype reveals a backend gap that isn't a workstream-sized blocker but isn't free either. Doing them inside W8 keeps the verification feedback loop tight — we see the field stop being null in the same workstream we noticed it was null.

#### Success criteria

- Each Tier 2 → populated transition is observable: a specific field on a specific endpoint moves from `null + _meta.pending` to a real value, with a test covering the transition.
- Backend additions that prove larger than expected get demoted to ROADMAP §9 (Known Issues & Backlog) with explicit rationale, not silently dropped.
- Quality gates green.

#### Disconfirming evidence

- If two or more Tier 3 items turn out to be hidden multi-day projects rather than focused additions, the scoping was wrong — pause and re-evaluate which items belong in W8 vs. later workstreams.
- If the percentile-ranking computation pass (likely the biggest Tier 3 item) doesn't fit cleanly into the existing feature pipeline, that's a signal the API may need its own derived-data layer, which would be a meaningful architectural shift.

#### How

High-level; expands during the design phase.

Tier 2 backend additions inventory (each populates fields on Tier 1 endpoints):

| Addition | Populates |
|---|---|
| Per-stat league-wide percentile ranking pass | Compare screen rank columns, Team Detail rank fields |
| Off/def rating decomposition (currently composite-only) | Team Rankings off/def split |
| Weekly Elo snapshot persistence | Team rating-history endpoint, projections week-over-week delta |
| Opponent-allowed-by-position aggregation | Player vs Defense view, Player Prop matchup section |
| Limited cohort splits (season, L4, home, away) per team | Game Detail split tabs, Compare splits |
| Limited cohort splits (indoor/outdoor, favored/underdog) per prop | Player Prop situational splits |
| Prior-week projection snapshot for delta | Projections 1-week change column |

Per-item design + decision on demotion to ROADMAP §9 happens during this tier's design phase.

---

## Changelog

| Date | Change |
|------|--------|
| 2026-07-01 | Tier 2 Step 2 complete: /model/performance. Combines model prediction quality (via evaluation/metrics.py) with betting performance (via betting/performance.py) into a single nested response. Two new Unavailable slugs (NO_EVALUATION_DATA, SINGLE_CLASS_OUTCOME) for data-limit fields. Endpoints populated so far: 7. |
| 2026-07-01 | Tier 2 Step 1 complete: /weeks/current + /portfolio/{summary,bets,curve,transactions,splits}. Introduced api/loaders.py, api/serializers/ package. D19 records explicit repo_root threading; D20 extends placeholder convention with Unavailable slugs for data-limit and missing-query-param cases. |
| 2026-06-27 | Tier 2 design phase complete. Inline "How" block expanded with three-layer architecture (loaders → serializers → routes), 8-step implementation order, locked decisions D17 (per-endpoint serializers) and D18 (serializer-owned field_status), and the inventory of pending fields expected to surface during the tier. Ready for Step 1 (weeks + portfolio). |
| 2026-06-27 | Tier 1 complete. Skeleton + blocked-endpoint stubs shipped: api/meta.py, api/schemas/_base.py, api/app.py + api/deps.py, cli/api.py, 9 route files, 9 schema files. 12 endpoints reachable via `gridiron api serve` with structurally valid null responses carrying registered blocker slugs. Integration tests lock round-trip parity and field_status completeness. Tier 2 (direct-serialization endpoints) now active. |
| 2026-06-26 | Tier 1 wiring verified end-to-end. All 12 endpoints reachable via `gridiron api serve`; response shapes carry `_meta.field_status` with registered blocker slugs. |
| 2026-06-23 | W8 Tier 1 design phase complete. Inline "How" block expanded with four-layer architecture (meta → schemas/_base → app/deps → Tier 3 routes), module layout, locked decisions per D16, and 8-step implementation order. Ready for Step 1 (`api/meta.py`). |
| 2026-06-23 | PLAN.md restructured to focus on the active workstream only. Migrated future workstream candidates, real-bugs backlog, investigations, and operational items to ROADMAP.md §9 (Known Issues & Backlog). Completed-workstream history moves to CHANGELOG.md responsibility. PLAN.md now contains exactly one active workstream at a time, broken into tiers, each with What / Why / Success / Disconfirming evidence / How blocks. W8 (API Serving Layer) set as active workstream with Tier 1 / 2 / 3 in Designing status. |
