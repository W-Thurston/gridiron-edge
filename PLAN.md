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

## Current Workstream: W13 — Runtime Champion Resolution

### What we are building

Persist the champion decision that `evaluate select-model` (games) and `select_prop_champion` (props) already compute at command time. The persisted decision lives in a static manifest at `data/output/champions/champions.json`. A resolver API (`evaluation/champion_resolver.py::resolve_current_champion(model_name)`) reads it. `full-retrain` writes it as a new stage between `refresh-calibrations` and `baseline-report`. Eight hard-coded CLI callsites get refactored to use the resolver instead of literal `("win_prob", "elo")` pairs.

The champion algorithm itself does not change. What changes is that the decision becomes a persistent artifact instead of a terminal-only recommendation.

### Why we are building it

Three reasons:

1. **W8 Tier 2 is blocked without it.** `/games`, `/games/{id}`, and downstream endpoints need to serve champion-only outputs per D21. The champion decision must be a pre-computed artifact, not a request-time computation.
2. **Eight CLI callsites currently hard-code `("win_prob", "elo")`.** This is technical debt from before the composite-key infrastructure was complete. Now that `select-model` produces the real answer, consumers should read it.
3. **W3.6/W3.7 built the composite-identity foundation but stopped short of persisting the model_name-level decision.** W13 finishes that job.

#### Success criteria
- ✅ ``data/output/champions/champions.json`` exists after full-retrain
  completes, with entries for every model_name that has ≥1 trained
  model_type.
- ✅ ``resolve_current_champion(model_name)`` returns the current
  ``(model_name, model_type)`` pair from the manifest.
- ✅ Missing manifest or missing entry raises ``ChampionNotFoundError``;
  documented and caught in downstream consumers.
- ✅ New full-retrain stage ``promote-champions`` runs the three
  selectors in write mode.
- 🟡 All 8 CLI hard-coded sites use ``resolve_current_champion()``
  (Tier 3).
- ✅ Existing tests pass; new tests cover the manifest read/write and
  the resolver's error path.
- ✅ All quality gates pass.

### Disconfirming evidence

- **If `select-model` needs substantial refactoring to be invoked programmatically** (currently a Typer command that prints to terminal), the write path is harder than "flag on existing command." Tier 2 pre-verification will show whether the ranking logic in `evaluation/select.py` is already factored cleanly enough to reuse.
- **If the prop side needs its own `select-model`-style ranking command** (it currently invokes `select_prop_champion` inline during `props champion`, which prints rather than persists), we may need to build the prop persistence path as a separate task.
- **If more than the 8 known callsites hard-code models,** the Tier 3 refactor scope grows. An audit during Tier 3 will surface any others.

### Locked architectural decisions

| Decision | Choice | Rationale |
|---|---|---|
| Champion is a static artifact | **Yes** | D21 — API is a serialization boundary. |
| Manifest format | **JSON at `data/output/champions/champions.json`** | Small, human-inspectable, matches `data/output/` convention. |
| Selection algorithm | **Reuse `evaluate select-model`'s composite rank (games) and `select_prop_champion` (props)** | Both already exist and are used. W13 does not touch algorithms. |
| Read API | **`evaluation/champion_resolver.py::resolve_current_champion(model_name)`** | Central import point for CLI + API consumers. |
| Missing-entry handling | **`ChampionNotFoundError` exception** | Explicit; downstream callers catch and degrade. |
| Write trigger | **New `full-retrain` stage between `refresh-calibrations` and `baseline-report`** | The retrain pipeline is the natural moment. Baseline report can then reference champion picks. |
| Manual write path | **Extend `evaluate select-model` with `--write-manifest` flag** for game side; add equivalent to `props champion` for prop side | Users can update the manifest without a full retrain when needed. |
| Prop and game share manifest | **Yes — single `champions.json` with entries for both** | Uniform read API. Frontend and CLI don't care which is which; they resolve by `model_name`. |

### Open design questions (resolved during tier design)

1. **Manifest schema fields.** Minimum: `{model_name: {model_type, promoted_at, source_run_id}}`. Should promotion metrics be embedded for audit, or just `source_run_id`? Decide during Tier 1.
2. **How does `select-model` get invoked programmatically from `full-retrain`?** Options: extract the ranking logic into a plain function callable from both the Typer command and the new stage, or invoke the Typer command as a subprocess. Decide during Tier 2 after inspecting `evaluation/select.py`.
3. **Prop side ranking:** `props champion` invokes `select_prop_champion` inline and prints. Do we extend it with `--write-manifest`, or build a new `evaluate promote-prop-champion` command? Decide during Tier 2.
4. **`cli/ratings.py` line 74 uses `("win_prob", "elo")` intentionally** (it's the Elo ratings command). Confirm during Tier 3 refactor that this stays as-is with a comment, or gets migrated to the resolver with an elo-specific opt-in.

### Tiers

**Tier 1 — Manifest schema + resolver API.** ✅ Complete (2026-07-01).
Shipped ``champion_resolver.py`` with ``read_manifest``,
``resolve_current_champion``, ``resolve_current_champion_with_metadata``,
``list_current_champions``, and ``ChampionNotFoundError``. Manifest
schema documented inline. Reader-only; writer path deferred to Tier 2.

**Tier 2 — Manifest writer and full-retrain integration.** ✅ Complete (2026-07-01).
Shipped in nine steps:

1. ``write_manifest`` primitive with atomic writes and preservation
   semantics for ``source_run_id`` (entries with existing ``source_run_id``
   keep it; new entries get the caller's).
2. ``select_game_regression_champions`` — reads ``ArtifactStore`` metadata,
   picks lowest holdout MAE, tie-breaks to ``random_forest``.
3. ``select_game_classification_champions`` — wraps ``collect_model_metrics``
   + ``rank_models`` from ``evaluation/select.py`` on Brier/ECE/AUC.
4. ``select_prop_champion_for_family`` and
   ``select_prop_champions_all_families`` — build ``RegressionModelResult``
   list from the prop archive, delegate to existing ``select_prop_champion``.
5. ``_stage_promote_champions`` in ``cli/full_retrain.py`` between
   ``refresh-calibrations`` and ``baseline-report``. Preserves prior
   manifest entries for families outside the current subset.
6. Baseline report annotates current champions above the metrics table.
   Bullet-list format so the delta-table parser ignores it.
7. ``evaluate select-model --write-manifest`` — manual override for the
   game side. Uses the shared ``promote_champions`` pure function
   extracted in this step.
8. ``props champion --write-manifest`` — parity for the prop side.
   Both flags call the shared ``write_champion_manifest`` helper in
   ``cli/_composites.py``.
9. ``champion_cmd`` refactored to use ``build_prop_champion_candidates``,
   sharing evaluation logic with the manifest writer.

Central catalog at ``gridiron_edge.models.catalog`` is the single source
of truth for ``GAME_MODEL_PAIRS``, ``PROP_STAT_FAMILIES``, and
``PROP_ALGORITHMS``. Used by both ``full_retrain.py`` and the
``--write-manifest`` CLI flags.

Locked decisions: manifest at ``data/output/champions/champions.json``;
selectors live in ``evaluation/champion.py``; game-regression ties
broken by preferring ``random_forest``; subset semantics preserve
untouched families in partial retrains; ``promote-champions`` depends
only on ``refresh-calibrations`` (not on ``backfill-prop-models``) so
``--skip-prop-backfill`` continues to work.

**Tier 3 — CLI consumer refactor.** 🟡 Active.
Replace 8 hard-coded ``model_name="win_prob", model_type="elo"``
callsites with ``resolve_current_champion("win_prob")`` calls in:

* ``cli/weekly_predict.py`` lines 96, 172
* ``cli/output.py`` line 52
* ``cli/edges.py`` lines 74, 161
* ``cli/evaluate.py`` lines 356, 374
* ``cli/ratings.py`` line 74 — confirm this stays as-is with a comment
  (intentional elo usage for the ratings command)

Tier design block drafted at the start of the tier.

---

## Paused Workstreams

### W8 — API Serving Layer

**Status:** Paused pending W13 completion.

**Where we stopped:** Tier 2 Step 4 complete (`/projections` shipped, 10 endpoints populated). Step 5 (`/games`, `/games/{id}`, `/games/{id}/predictions`) requires runtime champion resolution to serve pre-computed champion-only outputs per D21. W13 addresses this dependency.

**How this resumes:** When W13 closes, Tier 2 status returns from Paused to Active. Step 5 verification and design proceed with `resolve_current_champion()` as a locked given, and the remaining steps (5–8) continue on the original ordering.

#### What we are building (unchanged)

A read-only REST API that exposes every analytics output Gridiron Edge produces, shaped to match the Gridiron Edge frontend prototype end-to-end. Every screen in the prototype gets the endpoints it needs; every field that the backend can populate today returns real data; every field the backend can't yet produce returns `null` accompanied by metadata describing why.

The API is implemented as a FastAPI app with Pydantic v2 response models, mounted as a new `api/` package and served via `gridiron api serve`. Pydantic is confined to the `api/` boundary — domain code (models, evaluation, market, features) stays pandas/dataclass-shaped.

#### Why we are building it (unchanged)

1. **Verification surface.** The CLI surfaces outputs one-at-a-time. The frontend prototype puts ~19 screens worth of outputs side-by-side. Wiring the prototype to the API is the next quality-assurance step.
2. **Frontend unblock.** W9 (Frontend) cannot start until there is an API to wire to.
3. **Roadmap discovery.** The placeholder fields the API ships form a structured, observable inventory of "what's missing" that drives ROADMAP §9 prioritization.

#### Success criteria (unchanged)

- Every endpoint in the prototype-driven inventory returns a 200 response with a valid Pydantic-validated shape.
- Every populated endpoint returns real data for ≥80% of its fields, with the remainder explicitly marked in `_meta.field_status`.
- Every Tier 3 endpoint returns `null` fields with structured `_meta.field_status` entries naming a blocker.
- `gridiron api serve` starts the API and surfaces OpenAPI docs at `/docs`.
- Test coverage: unit (response model shape + `_meta` correctness), integration (`MiniRepoBuilder`-backed), with e2e deferred to W9.
- All quality gates pass.

#### Locked architectural decisions (unchanged)

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

#### Tiers

##### Tier 1 — Skeleton + blocked-endpoint stubs

**Status:** Complete (2026-06-27)

Shipped the FastAPI app skeleton, the `_meta` envelope plumbing, and every blocked endpoint returning its null-shape response with structured `_meta.field_status` entries. Twelve endpoints returning 200 with structurally valid null responses. `/docs` groups by domain. See CHANGELOG entry for full detail.

##### Tier 2 — Direct-serialization endpoints

**Status:** Paused (Steps 1–4 complete; Steps 5–8 paused pending W13)

Endpoints populated so far: 10.

**Step 1 — `/weeks/current` + all `/portfolio/*` (Complete, 2026-07-01)**
Six endpoints. Introduced `api/loaders.py`, `api/serializers/` package. D19 (explicit `repo_root` threading), D20 (`Unavailable` slug family for data-limit and missing-query-param cases) recorded.

**Step 2 — `/model/performance` (Complete, 2026-07-01)**
Nested response combining model prediction quality (via `evaluation/metrics.py`) with betting performance (via `betting/performance.py`). Two new `Unavailable` slugs (`NO_EVALUATION_DATA`, `SINGLE_CLASS_OUTCOME`).

Note: This endpoint computes metrics at request time, which violates D21. Tracked in ROADMAP §9.6 for refactor to a pre-computed summary artifact.

**Step 3 — `/teams` + `/teams/{abbr}` (Complete, 2026-07-01)**
First multi-source composition — Elo state + games records + team name normalization. Two new `Unavailable` slugs (`NO_PRIOR_SNAPSHOT`, `OFF_DEF_DECOMPOSITION`). Introduced `resolve_current_season_week` as a shared loader helper.

**Step 4 — `/projections` (Complete, 2026-07-01)**
Reads Monte Carlo season/playoff projections CSV; returns 32-team ranking with staleness timestamp. One new `Unavailable` slug (`NO_PROJECTIONS_DATA`).

**Step 5 — `/games`, `/games/{id}`, `/games/{id}/predictions` — PAUSED**
Discovered during pre-verification that:
1. The predictions archive holds every model per game (audit trail pattern).
2. No runtime champion resolution exists to identify which model's output is authoritative.
3. Per D21, the API cannot compute this at request time.
4. Per user direction, averaging across models is off the table (would obscure identity, pre-empt W12).

Pausing W8 to complete W13 (Runtime Champion Resolution). Step 5 resumes when W13 closes.

**Steps 6–8 — Paused pending W13.**
- Step 6 (`/edges`) — needs champion resolution to identify which predictions produced the edges.
- Step 7 (`/props`, `/props/{prop_id}`) — same, for prop models.
- Step 8 (`/compare/teams`, `/compare/player/{prop_id}`) — no direct dependency on W13, but sequenced after Steps 5–7 in the original plan.

##### Tier 3 — Backend additions

**Status:** Designing (not yet started; waits for Tier 2 to finish)

Unchanged from original design. Additions inventory:

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
