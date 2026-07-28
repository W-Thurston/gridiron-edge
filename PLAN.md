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

## Current Workstream

### W9.11 — Screen Completion

#### Tier 1 — PlayoffProjections rebuild — ✅ COMPLETE (2026-07-28)

Rebuilt `/projections` as a sortable, filterable simulation-probability
surface grounded in the existing projections and team-state contracts.

Shipped:

- Full-cell fixed-scale HeatCell probability matrix across all five postseason
  stages.
- Accessible SortableHeader primitive with active direction and nulls-last
  behavior.
- All-conference / AFC / NFC filtering with a dependent
  North / South / East / West division selector.
- Current Elo, one-week Elo adjustment, current record, average wins, and
  conference/division context.
- Season, as-of week, simulation count, and computed-time metadata.
- Explicit `elo_delta` API naming in place of the ambiguous
  `week_over_week_delta`.
- Expected Week 1 Elo-delta absence shown quietly with one explanatory legend
  caveat; unexpected later-season absence remains highlight-aware.
- Team-profile navigation through the existing NavContext route.
- Horizontal overflow preserving every postseason stage on narrow viewports.
- Comprehensive primitive, API-contract, serializer, route, and screen tests.

Verified against the real 32-team Week 1 artifact. All targeted backend gates,
frontend build, and frontend tests pass. Repository-wide Ruff still reports
pre-existing PLR0917 findings outside this workstream.

#### Tier 1 follow-up — Projections navigation and weekly outcomes — ACTIVE

**Goal**

Connect Team Rankings and Playoff Projections as sibling views, then add a
league-wide weekly schedule grid that displays each team's chance to win every
regular-season game.

**Locked decisions**

- Add a shared Team Rankings / Playoff Projections switcher to both screens.
- Add Weekly Outcomes as a local view within Playoff Projections.
- Weekly rows are teams; columns are Weeks 1–18.
- Scheduled-game cells display win probability, not W/L labels.
- Weekly probability color uses a diverging red → neutral → green scale
  centered at 50%.
- Bye weeks render as explicit neutral BYE cells.
- Played and projected games are divided through grouped headers and a
  stronger vertical boundary.
- Tooltips use the row team's perspective: `vs.` for home, `at` for away.
- Tooltip details include opponent, week, date, time, played/projected state,
  and actual result when available.
- Conference and dependent division filtering is shared across projections
  views.
- The grid is a static artifact composition; the API does not run simulation
  or perform new model computation.

##### A. Teams / projections sibling navigation

- [ ] Add shared `TeamViewSwitcher`.
- [ ] Add focused navigation and active-state tests.
- [ ] Add the switcher above `TeamsScreen`.
- [ ] Add the switcher above `PlayoffProjections`.
- [ ] Run `pnpm build && pnpm test:run`.
- [ ] Commit the sibling-navigation unit.

##### B. Weekly-grid API contract

- [ ] Inspect completed-games columns and current outcome conventions.
- [ ] Verify weekly probability behavior for played weeks, projected weeks,
      ties, and byes.
- [ ] Add projections-grid schemas.
- [ ] Add static loader composition from season grid, schedule, actuals, and
      team mappings.
- [ ] Add hand-written serializer and `_meta.field_status` behavior.
- [ ] Add `/projections/grid` route.
- [ ] Test played win/loss/tie, projected game, bye, and missing data.
- [ ] Regenerate `api-schema.json` and the TypeScript client schema.
- [ ] Verify the real preseason response.
- [ ] Run backend and frontend contract gates.
- [ ] Commit the grid-contract unit.

##### C. Weekly-grid frontend

- [ ] Add full-cell `WinProbabilityCell` with a fixed diverging scale.
- [ ] Add accessible hover/focus game-detail treatment.
- [ ] Add Playoff Chances / Weekly Outcomes local view selection.
- [ ] Add grouped Played Games / Projected Games headers.
- [ ] Add the played/projected boundary.
- [ ] Add sticky or visually anchored Team column if needed after visual review.
- [ ] Render BYE distinctly from a zero-percent game.
- [ ] Reuse conference and dependent division filtering.
- [ ] Test all-projected preseason and mixed played/projected fixtures.
- [ ] Run `pnpm build && pnpm test:run`.
- [ ] Complete a real-data visual pass.
- [ ] Re-close Tier 1 documentation.

---

## Paused Workstreams

_(none currently paused)_

---

#### W9.10: Compare Screen Rebuild — ✅ COMPLETE (2026-07-01)

Two-mode matchup surface. **Team vs Team** and **Player vs Defense**, both prototype-aligned against real data.

**Team vs Team:** mirrored team pickers + swap, cohort strip, narrative card, collapsible summary card, three matchup cards with mirrored ranking-bar collision rows (offense value ↔ reciprocal defense-allowed, edge chips, descriptive sublabels, title-style metric names). Backed by an 11-metric cohort_splits expansion (added def_pass_epa, def_third_down_pct, def_redzone_td_pct for reciprocal pairs).

**Player vs Defense:** independent player / stat-category / team pickers (searchable player combobox), 7-split strip (4 live + 3 pending), a per-game bar chart (player's stat as bars + team split-average as a moving reference line), and a "matchup, plainly" verdict card + by-split comparison table. Verdict is baseline-driven (defense-allowed vs player's own average → lean over/under), with rank as general context.

**Backend built to unblock it (Path C):**
- B1: `/players/{id}/history` (per-game series). Also fixed a root-cause game_id scramble in player_game_logs (`_join_game_id` index misalignment) and derived trustworthy is_home.
- B2: opponent_allowed expanded to 4 cohorts (season/l4/home/away, defense-perspective).
- B3: `/defense/{team}/allowed` (per-team allowed, all cohorts) for arbitrary-team selection.
- B4: `/players` roster list for the picker.

**New primitives:** BarChart (bars + reference line). DistributionChart retired from Compare (still used by PlayerProp).

**Deferred:** book line + O/U bar coloring (odds, W7); vs-winning/ vs-losing/vs-top-10 splits (3 pending pills); Change 6 (sortable + drag matchup rows, P2). All marked pending per highlight discipline.

**W9.10 complete.** Both Compare modes shipped; Compare no longer the biggest screen gap.

---

## Changelog

| Date | Change |
|------|--------|
| 2026-07-12 | **Doc-sync pass + PLAN reset.** Normalized planning docs after the frontend arc; PLAN reset to between-workstreams with a next-candidates list (audit sweep recommended). |
| 2026-07-11 | **W9.10 complete.** Both Compare modes shipped on backend B1–B4. Fixed game_id scramble, clean-games clobber, Elo empty-games crash; added champion→elo fallback + upcoming-Week-1 season resolver. |
| 2026-07-11 | **W9.10 status resync.** Team vs Team complete (6 alignment adjustments + 11-metric cohort_splits). Player vs Defense redesigned to independent pickers + bar chart + baseline verdict. |
| 2026-07-11 | **W9.10 design locked.** Two modes; highlight discipline baked in from W9.8. |
