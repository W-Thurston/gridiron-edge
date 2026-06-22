# Gridiron Edge - Development Plan

> **Purpose:** single source of truth for *what to build next* and *why*.
> Updated at the start and close of every workstream.

| Document | Role |
|----------|------|
| **PLAN.md** (this file) | What is planned, what is active, what is deferred |
| **CHANGELOG.md** | What was built and when (completed workstream details) |
| **HANDOFF.md** | How the system works today (architecture, workflows, operations) |
| **ROADMAP.md** | Long-term strategic direction, workstream inventory, architecture decisions |
| **DECISIONS.md** | Architectural decisions made during workstreams |

#### Status key

| Tag | Meaning |
|-----|---------|
| Done | Done - details in CHANGELOG.md |
| In progress | In progress |
| Planned | Planned / blocked |
| Deferred | Deferred |

---

## High-Level Priority Order

| # | Workstream | Status |
|---|-----------|--------|
| 1 | Champion/Challenger for Props (RF + XGBoost) | Done |
| 2 | Game Model Refactor (align to props pattern) | Done |
| 3 | Integration & E2E Tests | Done |
| 3.5 | Audit Remediation (Units 1-11) | Done |
| 4 | Composite CLI Workflows | Done |
| 5 | Deep Code Review + Test Suite Review | Done |
| 6 | Scenario / "What If" Engine (W4.5) | Planned (next) |
| 7 | API & Frontend (W8 + W9) | Planned |
| 8 | All External Odds (DK props, historical, line shopping) | Planned |
| 9 | Evaluate remaining work | Planned |

See `ROADMAP.md` for the full strategic context behind each workstream.

---

## Completed Workstreams

The first four workstreams (W1 through W3.5) closed substantial architectural and remediation work. Full details for each in `CHANGELOG.md`.

| Workstream | Closed | Summary |
|---|---|---|
| W1: Champion/Challenger for Props | 2026-06-04 | Added RF + XGBoost as prop model types alongside ElasticNet, unified champion/challenger pattern across all prop models, generalized `evaluation/champion.py` for both classification and regression gates. |
| W2: Game Model Refactor | 2026-06-19 | Refactored `models/game_prediction/` to mirror prop pattern. Composite-key registry, nested artifact paths, `GamesTrainer`/`GamesPredictor` unification, prediction archive schema migration from `model_version` to `(model_name, model_type)`. All 5 game models retrained; baselines improved or matched pre-WS2 targets. |
| W3: Integration & E2E Tests | 2026-06-20 | Three-tier test pyramid with auto-applied pytest markers. Shared fixtures, `MiniRepoBuilder` for integration repos. 500+ tests total. Game-side fit-load-predict integration tests; prop-side deferred. |
| W3.5: Audit Remediation (Units 1-11) | 2026-06-21 | Closed ~100 findings from `audit_2026_06_18.md` across 11 units plus 4 cross-cutting patterns. Architectural cleanup: canonical Elo simulator, identity unification, task-discriminated metadata, enforced trainability contracts, vectorized data pipelines, dataset registry completion. See `AUDIT_REMEDIATION.md` and `DECISIONS.md` (D1–D12) for full detail. |
| W4: Composite CLI Workflows | 2026-06-21 | Four composite commands wrap existing single-purpose commands into complete workflows. `weekly-predict` for game-day prep, `post-week` for archive + drift detection, `full-retrain` for season-start refresh, `verify` for pre-commit quality checks. Shared infrastructure (`cli/_composites.py`) provides stage abstraction, dependency validation, soft-fail semantics, and consolidated summary rendering. |
| W5: Deep Code Review + Test Suite Review (Tier 4 sweep) | 2026-06-22 | Multi-session opportunistic cleanup that closed 30 backlog items across CLI ergonomics, composite commands, dead code, documentation drift, exception narrowing, type cleanup, HTML escaping, season-label consistency, name mapping consolidation, calibration persistence, pipeline correctness, and incremental-build staleness detection. The review surfaced two real bugs (XGBoost recalibration Pipeline feature-name warning and modeling-file stale-data preservation) and added the pipeline staleness detector to prevent the latter from recurring. Three items reclassified as future workstream candidates rather than ambient hygiene. |

---

---

## Future Workstream Candidates (from former Tier 4 backlog)

Items identified during the Workstream 5 cleanup that are real work but don't fit "opportunistic ambient cleanup." These need scoping and prioritization before they become active workstreams.

### Testing Infrastructure

| Item | Notes |
|---|---|
| Props e2e fit-load-predict tests | Deferred from WS3. Needs prop fixture design study before implementation. |
| Composite commands don't have e2e tests | Unit tests cover stage definitions and orchestration with mocks; e2e tests against real data would surface integration issues earlier. |
| Weather ingest happy-path integration test | Pre-existing bugs went undetected because there was no end-to-end test of the ingest pipeline. |
| Registry cold-start scenarios | Test additions for `build_prop_evaluation_df` integration in conditions where the registry is empty at call time. |
| Performance baselines for tests | May need pytest-benchmark pass if runtime grows or regressions become a concern. |

### Real Bugs Surfaced During Tier 4 Cleanup

| Item | File | Notes |
|---|---|---|
| Walk-forward backfill produces no valid pipeline for single-season windows with expanded feature sets | `models/game_prediction/base.py::_run_hp_search` (root cause) and `evaluation/backfill.py::_walk_forward_one_season` (calling site) | Single-season walk-forward fails because filtered training data falls below MIN_CV_TRAIN_ROWS for expanded feature sets. Also surfaced: `_run_hp_search` does not forward `train_through_season` to `_prepare_window`. Fix needs choice between (A) lower threshold for walk-forward, (B) fill expanded-feature NaN with neutral values, or (C) force walk-forward to use combined feature set. |

### Investigations

| Item | Notes |
|---|---|
| `CalibratedClassifierCV` uses `StratifiedKFold(shuffle=False)` | Not strictly time-ordered. Investigate `TimeSeriesSplit` switch and measure impact on calibration quality. May require backfill run for comparison. |

### Operational

| Item | Notes |
|---|---|
| DraftKings odds endpoint returns 403 | Bot detection has gotten more aggressive. Investigate headers, cookies, paid API alternatives. `weekly-predict` soft-fails gracefully when this happens. |
| Weather: missing stadium entries for 2026-2027 international games | 12 stadiums need lat/lon/altitude in `NFL_stadium_reference.csv`. Listed in HANDOFF.md. Data entry task. |
| Model calibration values pre-date current modeling file | `_MODEL_SIGMAS` and `_MODEL_MARGIN_STDS` hardcoded fallbacks were calibrated against older modeling file. The `full-retrain` composite now persists current values to disk via the calibration registry; the next full-retrain run will supersede the hardcoded fallbacks. |
| `verify --strict` not exercised in CI | Once a real CI surface exists, `gridiron verify --strict` should be the gate. |

---

---

## Current Focus: Workstream 6 - Scenario / "What If" Engine (W4.5)

**Goal:** Build a scenario engine that lets a user ask "what if Mahomes is out?" or "what if KC is +120 instead of -110?" and see the propagated effects on predictions, edges, and recommended bets.

**Status:** Planning. See ROADMAP.md W4.5 for the original scope. The architectural foundation needed for this (composite identity, archive-driven CLI, prop integration spine) is now in place after Workstreams 3.5 and 4.

**Initial design questions:**
- Should scenarios be persisted or ephemeral?
- How granular should player impact be (binary out vs. snap percentage)?
- Should the CLI surface a single `gridiron scenario` command or a sub-app?

Detailed plan to follow when work begins.

---

---

## Future Workstreams

### Workstream 5: Deep Code Review + Test Suite Review

Two-part review session:
1. **Code review:** Pattern consistency across game + prop models, CLI output formatting parity, naming conventions, dead code, import hygiene, docstring completeness.
2. **Test suite review:** Edge case coverage audit, fixture quality, test isolation, missing negative tests, coverage ratchet assessment, props integration test gap from WS3.

Detailed plan created when Workstream 4 closes.

### Workstream 6: Scenario / "What If" Engine (W4.5)

See `ROADMAP.md` W4.5 for full description. Now unblocked by audit work and prop pipeline. Five phases: player impact quantification → team adjustment → usage redistribution → conditional re-forecasting → CLI interface.

### Workstream 7: API & Frontend (W8 + W9)

FastAPI serving layer + React/Next.js frontend consuming it. See `ROADMAP.md` W8/W9. Unblocked by current architectural state.

### Workstream 8: All External Odds

DraftKings prop odds ingest, historical odds data, multi-book line shopping. Requires odds source decision (`ROADMAP.md` §5.2).

### Workstream 9: Evaluate Remaining Work

Assessment of what's left: model ensemble (W12), real-time/live (W10), feature engineering backlog, NaN research, architectural debt.

---

## NaN Research Backlog (Deferred)

Current strategy: drop rows with NaN, with `# TODO(nan)` markers at each drop site. Future investigation items:

- Bayesian shrinkage priors for early-season rolling stats
- Seasonal carry-forward (use last season's final L6 as prior for week 1)
- Multiple imputation for missing game context features
- Missing-indicator pattern (add `feature_is_missing` binary columns)
- Rookie cold-start with draft capital / combine data

Best done after model architecture is stable (achieved post-WS2).

---

## Changelog

| Date | Change |
|------|--------|
| 2026-06-22 | **Workstream 5 (Tier 4 cleanup sweep) closed.** 30 backlog items closed across CLI ergonomics, composite commands, dead code, documentation drift, exception narrowing, type cleanup, HTML escaping, season-label consistency, name mapping consolidation, calibration persistence, pipeline correctness, and incremental-build staleness detection. Two real bugs surfaced and fixed. Remaining open items reclassified as workstream candidates and moved to PLAN.md sections (Testing Infrastructure, Real Bugs, Investigations, Operational). TIER_4_BACKLOG.md retired. Workstream 6 (Scenario Engine) becomes the next planned focus. |
| 2026-06-21 | **Workstream 4 (Composite CLI Workflows) closed.** Four composite commands (`weekly-predict`, `post-week`, `full-retrain`, `verify`) wrap related single-purpose commands into complete workflows. Shared infrastructure handles stage abstraction, dependency validation, soft-fail semantics, and summary rendering. Workstream 5 (Deep Code Review + Test Suite Review) is the next planned focus. |
| 2026-06-21 | **Audit remediation complete (Workstream 3.5).** All 11 audit units closed. ~100 findings closed across architectural, correctness, performance, and stylistic dimensions. Composite CLI workflows (Workstream 4) added as the next focused workstream. Document restructured to reflect current focus rather than completed history. |
| 2026-06-19 | **Workstream 2 closed.** Game model refactor complete. Workstream 3 (Integration & E2E Tests) becomes active. |
| 2026-06-10 | **Full rewrite.** Priority order locked: champion/challenger → game model refactor → integration tests → code review → scenario engine → API/frontend → external odds → evaluate. |
