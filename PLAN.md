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

### Tier 1 — Skeleton + Tier 3 stubs

**Status:** Designed, ready for implementation

#### What we are building

The FastAPI app skeleton, the `_meta` envelope plumbing, and every Tier 3 endpoint returning its blocked-field shape. No backend wiring yet — pure schema work.

#### Why

Fastest path to a working API surface. Lets W9 frontend integration begin immediately against the full endpoint inventory, even though most Tier 1/2 endpoints will return mostly-null shapes at this point. Forces the placeholder convention into shape under real use before committing to it for populated endpoints.

#### Success criteria

- `gridiron api serve` starts a FastAPI app exposing the full endpoint inventory.
- `_meta.field_status` envelope works end-to-end through a `BaseResponse` Pydantic model that all responses inherit.
- All Tier 3 endpoints return 200 with a fully-null shape and a populated `_meta.field_status` block naming the upstream blocker.
- OpenAPI docs at `/docs` render every endpoint with its full schema and group Tier 3 sub-resources under their own tags.
- Unit tests cover `_meta` envelope construction and Tier 3 blocked-field shapes.
- Integration test confirms every Tier 3 endpoint returns 200 with a non-empty `_meta.field_status` referencing a registered blocker slug.

#### Disconfirming evidence

- If the `_meta` envelope feels structurally wrong as soon as it has to carry real status entries — i.e., the developer ergonomics of constructing it are bad — revisit the placeholder convention before Tier 2.
- If Pydantic v2 model composition doesn't cleanly support the `BaseResponse` + nested `_meta` pattern (e.g., generic list responses with frozen parents misbehave), revisit the validation library decision.
- If the `Blocker` registry pattern proves awkward (e.g., we keep adding tuples), promote to an Enum or a registered-class pattern.

#### How

**Architecture: four-layer bottom-up build.**

1. **`api/meta.py`** — `FieldStatus` discriminated union (`"pending"` literal or `BlockedStatus` object), `ResponseMeta` envelope with `with_pending` / `with_blocked` builder methods, `Blocker` registry of `(slug, roadmap_ref)` tuples matching ROADMAP §9.5.
2. **`api/schemas/_base.py`** — `BaseResponse` Pydantic model (frozen, `extra="forbid"`) with optional `response_meta: ResponseMeta | None` aliased to `_meta` on the wire. `BaseListResponse[T]` generic for list-shaped endpoints carrying `items: list[T]` and `total: int | None`.
3. **`api/app.py` + `api/deps.py`** — FastAPI app factory with per-domain OpenAPI tagging, permissive CORS for dev, dataset registry as a cached dependency. `cli/api.py` adds `gridiron api serve` wrapping uvicorn.
4. **Tier 3 routes** — one file per blocker domain (per D16). Each file is small (~20-40 lines) and follows a uniform template: define the prototype-expected shape with all fields `Optional = None`, return a constructed response with `_meta.field_status` listing every null field's blocker.

**Module layout:**

```
src/gridiron\_edge/api/
├── app.py
├── deps.py
├── meta.py
├── routes/
│   ├── # Tier 1 — populated during Tier 2
│   ├── weeks.py, games.py, edges.py, teams.py,
│   ├── projections.py, props.py, portfolio.py,
│   ├── compare.py, model.py
│   │
│   └── # Tier 3 — blocked, ships in Tier 1
│       ├── lines.py            → MULTI\_BOOK (W7)
│       ├── live.py             → LIVE\_STATE (W10)
│       ├── news.py             → NEWS\_INGEST
│       ├── injuries.py         → INJURY\_DATA (§5.3)
│       ├── explain.py          → scenario\_engine (W4.5)
│       ├── swing\_factors.py    → FEATURE\_ATTRIBUTION
│       ├── comparables.py      → COMPARABLES
│       ├── prop\_shop.py        → MULTI\_BOOK (W7)
│       └── prop\_reasoning.py   → FEATURE\_ATTRIBUTION
└── schemas/
├── \_base.py     # BaseResponse, BaseListResponse\[T]
└── <one file per route file>
```

**Locked design decisions** (full rationale in DECISIONS.md D16):

- List endpoints surface blocked-list state through `_meta.field_status["items"]`, not a separate envelope field. Uniform with scalar/object field blocking.
- Tier 3 sub-resource endpoints get their own route files grouped by blocker domain. When a blocker clears, the unblock work is a single-file diff.

**Implementation order:**

1. `api/meta.py` + unit tests for `FieldStatus`, `ResponseMeta`, `Blocker` registry.
2. `api/schemas/_base.py` + unit tests for `BaseResponse`, `BaseListResponse[T]`, alias handling.
3. `api/deps.py` + `api/app.py` (no routers yet) + `create_app()` smoke test.
4. `cli/api.py` + `gridiron api serve --help` smoke test + wire into `cli/main.py`.
5. Tier 3 schema files (one per route file).
6. Tier 3 route files following the uniform template.
7. Wire Tier 3 routes into `app.py`; walk `/docs`; confirm every endpoint renders with the right OpenAPI tag.
8. Integration test using `TestClient` — assert every Tier 3 endpoint returns 200, has non-empty `_meta.field_status`, and references a `Blocker.*` slug.

Quality gates after each step: `ruff check . --fix && uvx pyrefly check && uv run pytest -m "unit and not slow" -v`.

#### Open questions deferred to Tier 2

- Singular vs. plural list keys (currently `items: list[T]`; revisit if generic shape proves awkward).
- `extra="forbid"` on `BaseResponse` (currently locked; revisit if early refactors make it painful).
- Frozen `BaseResponse` + list field mutation semantics (Pydantic v2 doesn't auto-freeze nested lists; verify during Step 2 unit tests).

---

### Tier 2 — Tier 1 endpoint wiring

**Status:** Designing

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

High-level; expands during the design phase.

Tier 1 endpoint inventory:

```
GET /weeks/current
GET /games?season=\&week=
GET /games/{game\_id}
GET /games/{game\_id}/predictions
GET /edges?season=\&week=\&market=
GET /teams
GET /teams/{abbr}
GET /projections
GET /props?season=\&week=\&position=\&stat=
GET /props/{prop\_id}
GET /portfolio/summary
GET /portfolio/bets?status=
GET /portfolio/curve?period=
GET /portfolio/transactions
GET /portfolio/splits?dimension=
GET /model/performance?period=
```

Implementation order to be set during the design phase. Likely: portfolio first (simplest data shapes, exercises the envelope), then games, then teams, then projections, then props, then edges.

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
| 2026-06-23 | W8 Tier 1 design phase complete. Inline "How" block expanded with four-layer architecture (meta → schemas/_base → app/deps → Tier 3 routes), module layout, locked decisions per D16, and 8-step implementation order. Ready for Step 1 (`api/meta.py`). |
| 2026-06-23 | PLAN.md restructured to focus on the active workstream only. Migrated future workstream candidates, real-bugs backlog, investigations, and operational items to ROADMAP.md §9 (Known Issues & Backlog). Completed-workstream history moves to CHANGELOG.md responsibility. PLAN.md now contains exactly one active workstream at a time, broken into tiers, each with What / Why / Success / Disconfirming evidence / How blocks. W8 (API Serving Layer) set as active workstream with Tier 1 / 2 / 3 in Designing status. |
