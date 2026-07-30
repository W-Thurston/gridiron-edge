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

## Ways of Working

How we operate in every session. A new thread should read this first.

1. **Confirm before building — never assume code exists or looks a certain way.** Before writing or modifying anything, verify the current state. Prefer over-confirming: it's cheaper to paste five files than to rebuild something that already existed and discover the drift later. Use grep to locate; ask the user to run commands or paste files. Assumptions are the #1 source of rework and format drift.

2. **Grep first, then read.** Locate with `grep -rn`, then request the specific file(s) or function(s). Ask the user to run the grep/curl and paste results rather than guessing at signatures, schemas, or data.

3. **Design before implementing, at two levels.**
   - **ROADMAP-level (high-level):** lock the workstream shape — what, why, tiers, success criteria — before touching code.
   - **Subsection-level (deep):** before each tier/substep, a focused design block with locked decisions, then implement.

4. **PLAN.md tracks the active work.** After ROADMAP design, expand the workstream into PLAN.md as a checklist. Check items off as completed.

5. **Commit small units as you go.** Each substep is its own commit with a clear message. Quality gates (`ruff` + `pyrefly` + unit tests, or `pnpm build && pnpm test:run` for frontend) pass before each commit.

6. **Section close-out ritual.** When a section/tier completes: clean its detail out of PLAN.md (collapse to a one-line summary or remove), check the item off in ROADMAP.md, and either continue to the next subsection or repeat the full design→plan→build→close loop for the next big chunk.

7. **Verify against real data.** After backend/data changes, confirm via curl or a Python one-liner against the actual artifact — don't trust the code path alone.

8. **Note on dates:** the assistant's system clock may report an earlier date than reality. Trust commit timestamps and the user; when in doubt, ask.

---

## Current Workstream — W14: Game Prediction Season Readiness

**Status:** Designing

### Purpose

Make the game-prediction vertical slice operational before the NFL season.

For a real upcoming week, Gridiron Edge should be able to:

1. refresh the required schedule, game, feature, model, and market data;
2. produce trustworthy moneyline, spread, and total forecasts;
3. preserve model and artifact provenance;
4. retrieve and normalize current game-market odds;
5. join predictions to odds without sign, identity, or freshness ambiguity;
6. generate truthful moneyline, spread, and total edges;
7. expose prediction, market, and edge states through the API;
8. render the complete weekly decision loop in Dashboard, Games, GameDetail,
   and BetSlip.

This workstream prioritizes a usable game-day product path over a broad
frontend audit. The frontend remains a verification surface, but frontend work
is limited to the screens and contracts required by the game-prediction loop.

### Product outcome

The season-readiness path should answer, for every upcoming game:

#### Moneyline

- What is the model win probability?
- Which model and artifact produced it?
- What is the fair American price?
- What market price is currently available?
- What is the modeled EV?
- Is dollar Kelly sizing available, and from which bankroll basis?

#### Spread

- What is the projected margin or model spread?
- Is spread produced by a trained artifact, post-processing, or another
  derivation?
- What market spread and price are currently available?
- What is the point edge?
- What is the cover probability?
- What is the modeled EV?

#### Total

- What is the projected total?
- Which model and artifact produced it?
- What market total and price are currently available?
- What is the point edge?
- What are the Over and Under probabilities?
- What is the modeled EV?

### Non-goals

- No broad frontend pending-highlight audit.
- No frontend polish outside Dashboard, Games, GameDetail, and BetSlip unless
  required by a shared contract.
- No prop-model readiness work.
- No injury, news, or live-game data.
- No sportsbook authentication or wager placement.
- No multi-book line-shopping interface in the initial season-readiness path.
- No new model ensemble until the current individual-model path is verified.
- No fabricated odds, predictions, edges, or production demo data.
- No assumption that spread is a standalone trained model until the current
  implementation is inspected.

### Locked principles

#### One vertical slice

The unit of readiness is not an isolated model, API endpoint, or screen. The
unit is the complete path:

```text
upcoming schedule
  → upcoming-game features
  → moneyline / spread / total predictions
  → prediction archive
  → odds pull and normalized snapshot
  → prediction-to-market join
  → edge report
  → API serialization
  → Dashboard / Games / GameDetail / BetSlip
```

Static artifact boundary

The API remains a serialization boundary. Prediction, evaluation, odds, coverage, and edge artifacts must be produced before the API serves them.

No false empty states

The system must distinguish:

no upcoming schedule;
prediction unavailable;
partial prediction coverage;
odds pull failed;
odds snapshot stale;
market missing;
prediction-to-odds join failed;
successful complete inputs with no positive edge.

No edges available is a valid model result only when prediction and market coverage are sufficient and the join completed successfully.

Provenance and freshness

Every user-visible prediction or market comparison should be attributable to:

season and week;
game ID;
market;
model name and model type;
source artifact or archived prediction;
prediction generation time when available;
odds source and snapshot time when available;
bankroll and Kelly multiplier when dollar sizing is shown.
Existing workflows first

Use and harden the existing:

weekly-predict;
prediction archive;
champion manifest and resolver;
DraftKings ingest and odds store;
recommendation report;
edge report;
API serializers;
frontend consumers.

Do not create parallel season-readiness commands or duplicate prediction paths unless the current architecture cannot support the required behavior.

Verified starting state

The repository currently contains:

trained win_prob artifacts for logistic, random forest, and XGBoost;
trained total artifacts for random forest and XGBoost;
no separate spread-model artifact visible under data/models;
game prediction modules for win probability, total, and post-processing;
current calibration and champion manifests;
a Week 1 Elo prediction artifact and prediction archive;
a Week 1 edge CSV;
DraftKings ingest, game-ID resolution, and odds-store modules;
an empty data/odds directory in the supplied repository tree;
market math and moneyline/spread/total recommendation code;
game, edge, and portfolio API contracts;
Dashboard, Games, GameDetail, and BetSlip frontend consumers;
existing unit, integration, and end-to-end coverage across these layers.

The roadmap also records a critical architectural limitation: trained game models currently consume a modeling file built from completed games, so the normal upcoming-week path only archives Elo predictions. Building and validating an upcoming-week feature matrix is therefore part of the season-readiness investigation rather than an optional future note.

Tier 0 — Current-state audit and dependency map — 🔵 ACTIVE
A. Prediction-path inspection
 Inspect the win-probability model registry, artifacts, metadata, champion manifest entry, prediction path, archive schema, and API selection.
 Inspect the total model registry, artifacts, metadata, champion manifest entry, prediction path, archive schema, and API selection.
 Determine exactly how projected margin and spread outputs are produced.
 Confirm whether spread has a trained target or is derived through post-processing.
 Trace projected scores, uncertainty, confidence tiers, cover probability, and Over/Under probability to their authoritative sources.
 Verify model feature columns against artifact metadata.
 Verify calibration data corresponds to the current modeling artifacts.
 Identify every intentional Elo fallback and every place where trained model output is expected.
B. Upcoming-week input inspection
 Inspect how the cleaned upcoming schedule enters prediction workflows.
 Inspect completed-game and upcoming-game boundaries in feature generation.
 Confirm why the current modeling file contains no upcoming-game rows.
 Inventory features that can be calculated for an unplayed game.
 Inventory features that require prior in-season observations.
 Define truthful Week 1 behavior for unavailable rolling features.
 Define mid-season behavior for next-week rolling features.
 Determine whether one upcoming-game feature matrix can serve moneyline, spread, and total prediction.
 Verify feature-schema and stale-artifact handling requirements.
C. Archive and evaluation inspection
 Inspect game prediction archive identity and deduplication.
 Verify moneyline, spread/margin, and total fields retained per archived row.
 Verify model name and model type provenance.
 Inspect current champion-selection metrics for win_prob.
 Inspect current champion-selection metrics for total.
 Determine the appropriate historical readiness metrics for spread.
 Verify walk-forward coverage available for market-relative evaluation.
 Identify whether historical odds can be joined to archive rows for moneyline, spread, and total evaluation.
D. Odds-path inspection
 Inspect the current DraftKings request contract.
 Reproduce and characterize the current 403 behavior.
 Inspect parsing for moneyline, spread, and total markets.
 Verify American-price preservation and Unicode-minus normalization.
 Verify spread sign and home/away conventions.
 Verify total-line and Over/Under side conventions.
 Verify canonical game-ID resolution.
 Inspect current snapshot and historical-ledger write behavior.
 Determine why data/odds is empty in the supplied repository state.
 Determine whether a reliable single-source game-market pull can be restored.
 Evaluate alternative odds sources only after the current pull path is fully understood.
E. Join and edge inspection
 Trace prediction-to-odds joins for all three game markets.
 Verify season, week, game-ID, market, side, line, and price keys.
 Verify moneyline no-vig and implied-probability semantics.
 Verify spread point-edge and cover-probability semantics.
 Verify total point-edge and Over/Under probability semantics.
 Verify positive and negative American prices are preserved.
 Verify missing markets remain distinct from no-positive-edge results.
 Inspect the existing Week 1 edge CSV and identify its input provenance.
 Trace populated, blocked, and legitimate-empty /edges responses.
F. Workflow inspection
 Trace every stage of weekly-predict.
 Record its required inputs and produced artifacts.
 Verify which failures are soft and which are hard.
 Verify stage dependencies and partial-resume behavior.
 Inspect verify coverage for predictions, odds, joins, and edges.
 Inspect existing unit, integration, and end-to-end workflow tests.
 Identify missing season-readiness diagnostics and integration coverage.
G. Focused frontend inspection
 Trace game-prediction fields consumed by Dashboard.
 Trace game-prediction and market fields consumed by Games.
 Trace moneyline, spread, total, and edge fields consumed by GameDetail.
 Confirm BetSlip accepts real moneyline, spread, and total edge rows.
 Inventory fields that conflate unavailable predictions, unavailable markets, stale odds, failed joins, and legitimate no-edge states.
 Identify the minimum frontend changes required for a truthful live weekly rehearsal.
Tier 0 deliverable

Produce one verified dependency map containing:

authoritative source for each moneyline, spread, and total output;
artifact and archive path for each output;
upcoming-week feature requirements;
calibration and evaluation dependencies;
odds pull, parser, normalization, and storage path;
join and edge-generation path;
API and frontend consumers;
concrete missing or defective links;
proposed implementation order for the remaining tiers.

No production implementation should begin until this dependency map is complete.

Planned implementation tiers

The exact implementation detail will be locked after Tier 0 verifies the current contracts.

Tier 1 — Upcoming-game prediction foundation

Build or repair the shared upcoming-game feature and prediction path needed to produce moneyline, spread, and total outputs for real unplayed games.

Tier 2 — Market-specific readiness and evaluation

Verify the predictive and market-relative quality of moneyline, spread, and total outputs. Repair calibration, bias, or probability-conversion defects identified by evidence.

Tier 3 — Odds ingestion reliability and coverage diagnostics

Make the initial game-market odds source operationally reliable enough for the season-readiness path. Add freshness, market coverage, and failure diagnostics.

Tier 4 — Prediction-to-market join and edge readiness

Verify canonical joins, line and side conventions, market probabilities, EV, Kelly, and explicit distinctions between missing inputs and legitimate no-edge results.

Tier 5 — API season-readiness contract

Expose sufficient artifact, prediction, market, freshness, coverage, and blocker metadata for the focused frontend path without computing at request time.

Tier 6 — Focused game-day frontend

Harden only Dashboard, Games, GameDetail, and BetSlip for real moneyline, spread, and total use.

Tier 7 — Live weekly rehearsal

Run the complete weekly workflow against real upcoming-game and odds data. Verify artifacts, APIs, focused frontend screens, and real BetSlip staging.

Success criteria

W14 is complete when:

every scheduled game in the target upcoming week has an explicit prediction state for moneyline, spread, and total;
unavailable market outputs are explained rather than silently omitted;
moneyline, spread, and total outputs identify their authoritative model or derivation;
predicted outputs are archived with season, week, game, model, and generation provenance;
odds pull success, failure, freshness, and market coverage are observable;
moneyline, spread, and total lines and prices normalize to canonical game and side conventions;
prediction-to-market coverage is reported by game and market;
/edges distinguishes input failure, partial coverage, join failure, legitimate empty results, and populated recommendations;
the existing weekly workflow can produce or explicitly diagnose every required artifact;
Dashboard, Games, GameDetail, and BetSlip present the same underlying prediction and market state consistently;
at least one real weekly rehearsal is completed without fabricated data;
backend and frontend quality gates pass;
W9.11 Tier 0 remains paused until this vertical slice is operationally verified.

---

## Paused Workstreams

#### W9.11 Tier 0: Final frontend audit — ⏸️ PAUSED

Paused in favor of W14 Game Prediction Season Readiness. Resume after the
moneyline, spread, total, odds-ingestion, edge, API, and focused game-day
frontend path is operationally verified.

The deferred BetSlip real-data review moves naturally into W14 Tier 7 because
that rehearsal should produce the real edge recommendations required to test
the staged-wager presentation.

---

## Changelog

| Date | Change |
|------|--------|
| 2026-07-30 | **W14 opened.** Paused W9.11 Tier 0 and made Game Prediction Season Readiness the active workstream. Locked the vertical-slice goal across upcoming-game features, moneyline/spread/total predictions, archives, odds ingestion, joins, edges, API contracts, Dashboard, Games, GameDetail, BetSlip, and a real weekly rehearsal. Tier 0 is a read-only current-state audit before implementation. |
| 2026-07-28 | Closed the PlayoffProjections navigation and Weekly Outcomes follow-up after real-data verification. |
| 2026-07-12 | **Doc-sync pass + PLAN reset.** Normalized planning docs after the frontend arc; PLAN reset to between-workstreams with a next-candidates list (audit sweep recommended). |
| 2026-07-11 | **W9.10 complete.** Both Compare modes shipped on backend B1–B4. Fixed game_id scramble, clean-games clobber, Elo empty-games crash; added champion→elo fallback + upcoming-Week-1 season resolver. |
| 2026-07-11 | **W9.10 status resync.** Team vs Team complete (6 alignment adjustments + 11-metric cohort_splits). Player vs Defense redesigned to independent pickers + bar chart + baseline verdict. |
| 2026-07-11 | **W9.10 design locked.** Two modes; highlight discipline baked in from W9.8. |
