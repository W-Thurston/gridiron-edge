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
| **TIER_4_BACKLOG.md** | Ambient hygiene items handled opportunistically as files are touched |

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
| 6 | Scenario / "What If" Engine (W4.5) | Planned |
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

---

## Current Focus: Workstream 5 - Deep Code Review + Test Suite Review

**Goal:** Two-part review session covering:

1. **Code review:** Pattern consistency across game + prop models, CLI output formatting parity, naming conventions, dead code, import hygiene, docstring completeness.
2. **Test suite review:** Edge case coverage audit, fixture quality, test isolation, missing negative tests, coverage ratchet assessment, props integration test gap from WS3.

**Status:** Planning. The first pass should consume the items in `TIER_4_BACKLOG.md` that involve files we'd naturally touch during the review.

**Pre-work:**
- Refresh `TIER_4_BACKLOG.md` since several items there are about file-level concerns we'd be reviewing anyway.
- Decide on scope: full audit (multiple sessions) vs targeted (one session).

Detailed plan to follow.

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
| 2026-06-21 | **Workstream 4 (Composite CLI Workflows) closed.** Four composite commands (`weekly-predict`, `post-week`, `full-retrain`, `verify`) wrap related single-purpose commands into complete workflows. Shared infrastructure handles stage abstraction, dependency validation, soft-fail semantics, and summary rendering. Workstream 5 (Deep Code Review + Test Suite Review) is the next planned focus. |
| 2026-06-21 | **Audit remediation complete (Workstream 3.5).** All 11 audit units closed. ~100 findings closed across architectural, correctness, performance, and stylistic dimensions. Composite CLI workflows (Workstream 4) added as the next focused workstream. Document restructured to reflect current focus rather than completed history. |
| 2026-06-19 | **Workstream 2 closed.** Game model refactor complete. Workstream 3 (Integration & E2E Tests) becomes active. |
| 2026-06-10 | **Full rewrite.** Priority order locked: champion/challenger → game model refactor → integration tests → code review → scenario engine → API/frontend → external odds → evaluate. |
