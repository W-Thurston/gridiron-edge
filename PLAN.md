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

#### Tier 1 — PlayoffProjections rebuild — ACTIVE

**Goal**

Rebuild `/projections` as a compact, sortable simulation-probability surface using the verified projection artifact and shared team metadata. Preserve the semantics of the real data, fit the established centered layout, and make every missing state explicit.

**Verified inputs**

- `projections_summary.csv` contains 32 teams and only:
  `TEAM`, `AVG_WINS`, `P_MAKE_PLAYOFFS`, `P_REACH_DIV`, `P_REACH_CONF`, `P_REACH_SB`, `P_WIN_SB`.
- `/projections` additionally exposes season, computed time, simulation count, Elo movement, and pending clinched/eliminated fields.
- Team conference, division, city/name, and colors are available through the shared `/teams` metadata cache.
- No existing HeatCell or sortable-header primitive exists.
- Routing is custom through NavContext.
- Tier 0 pending-highlight audit is deferred until after both remaining screen rebuilds.

**Locked decisions**

- Keep `AVG_WINS` as average wins; do not derive a discrete projected record.
- Rename `week_over_week_delta` to `elo_delta`.
- Treat the delta as Elo rating-point movement, never probability movement.
- Default sort: Win SB descending. Nulls always last.
- Visible columns: Team, Avg Wins, Playoffs, Div. Round, Conf. Champ., Make SB, Win SB.
- Fold conference/division, Elo delta, and real status pills into team identity.
- Use a fixed 0–1 heat scale across all probability cells.
- Display only truthful run metadata: season, simulation count, computed time.
- Preserve every postseason stage on narrow screens through horizontal overflow.
- Follow the existing NavContext navigation pattern for team-profile links.

##### 1. Documentation synchronization

- [ ] Update ROADMAP §4 W9.11 execution order to Tier 1 → Tier 2 → deferred Tier 0.
- [ ] Correct ROADMAP §9.7 projections data assumptions.
- [ ] Replace ROADMAP §9.8 PlayoffProjections remainder with the locked scope.
- [ ] Update ROADMAP §6 current position.
- [ ] Add the 2026-07-28 ROADMAP document-changelog entry.
- [ ] Commit the ROADMAP/PLAN design synchronization as one documentation-only unit.

##### 2. Contract semantics and status metadata

- [x] Grep all exact uses of `week_over_week_delta`.
- [x] Inspect `api/meta.py` for the established unavailable classification.
- [x] Rename the public projection field to `elo_delta` across schema, serializer, generated client, frontend consumer, and tests.
- [x] Correct stale serializer comments about `n_simulations` and universal delta population.
- [x] Mark `items.elo_delta` unavailable when the projection response contains no usable prior-week Elo deltas.
- [x] Preserve populated-delta behavior without an unavailable marker.
- [x] Define and test the partial-null fallback without silently rendering null.
- [x] Run backend quality gates:
      `ruff`, `pyrefly`, and targeted projections tests.
- [x] Verify the real `/projections` response with curl.
- [x] Commit the contract/status correction as one unit.

##### 3. Shared table primitives

- [x] Add `HeatCell`.
- [x] Use a fixed absolute probability scale and theme-native `color-mix`.
- [x] Add accessible stage/value labeling.
- [x] Format zero, whole percentages, and positive sub-1% values distinctly.
- [x] Make null rendering status-aware.
- [x] Add focused `HeatCell` tests.
- [x] Add `SortableHeader`.
- [x] Render the interaction as a button within the header.
- [x] Expose inactive, ascending, and descending states with `aria-sort`.
- [x] Add focused `SortableHeader` tests.
- [x] Run `pnpm build && pnpm test:run`.
- [x] Commit the primitives as one unit.

##### 4. Screen composition and interactions

- [x] Inspect NavContext and an existing team-selection callsite before wiring navigation.
- [x] Rebuild the screen header and explanatory copy.
- [x] Add the simulation-run metadata cluster.
- [x] Add All / AFC / NFC filters using `Pill`.
- [x] Memoize team metadata by abbreviation.
- [x] Enrich team identity with conference and division.
- [x] Add local sort state and immutable filter/sort derivation.
- [x] Implement default Win SB descending order.
- [x] Implement useful first-click direction and active-column toggling.
- [x] Keep null values last in both directions.
- [x] Render the five probability columns through `HeatCell`.
- [x] Render average wins to one decimal.
- [x] Add compact positive/negative/neutral Elo-delta treatment.
- [x] Surface no-prior-week Elo state visibly.
- [x] Preserve real clinched/eliminated pills and explain globally pending status.
- [x] Add team-profile navigation through the established custom-router pattern.
- [x] Correct the ErrorCard title to “Couldn't load projections.”
- [x] Preserve actionable empty-state copy.
- [x] Add horizontal overflow without silently hiding probability stages.
- [x] Add screen tests for sorting, filtering, metadata, missing metadata, Elo states, status pills, navigation, error, and empty results.
- [x] Run `pnpm build && pnpm test:run`.
- [x] Commit the screen rebuild as one unit.

##### 5. Real-data verification and Tier close-out

- [x] Run the API and frontend against the populated projections artifact.
- [x] Verify all 32 teams under the All filter.
- [x] Verify AFC and NFC subsets against team metadata.
- [x] Verify each sortable column in both directions.
- [x] Verify Week 1 Elo unavailability in normal and highlight modes.
- [x] Verify heat-cell readability across low and high probabilities.
- [x] Verify the normal centered layout and a narrow viewport.
- [x] Verify keyboard operation for filters, sort headers, and team navigation.
- [x] Run targeted Ruff checks for all Python files changed by Tier 1.
- [x] Run Pyrefly.
- [x] Run targeted projections backend tests.
- [x] Run the full frontend build and test suite.
- [x] Convert HeatCell from inset pill to full table-cell heat fill.
- [x] Update HeatCell tests for table-cell semantics.
- [ ] Replace conference pills with dependent conference/division selects.
- [ ] Add current Elo from the existing `/teams` composition.
- [ ] Add current record from the existing `/teams` composition.
- [ ] Add `/teams.as_of_week` beside the season metadata.
- [ ] Move Elo delta into a dedicated sortable column.
- [ ] Suppress repeated no-prior-snapshot warnings during Week 1.
- [ ] Add a single Week 1 Elo-delta caveat to the legend.
- [ ] Update PlayoffProjections screen tests.
- [ ] Verify the revised continuous heat matrix against the original static-table visual language.
- [ ] Run `pnpm build && pnpm test:run`.
- [ ] Commit the refinement in small units.
- [ ] Repository-wide Ruff: pre-existing PLR0917 findings remain outside W9.11 scope; track separately rather than refactoring unrelated APIs, CLI entrypoints, tests, and Numba simulation signatures.
- [ ] Collapse Tier 1 detail in PLAN.md to a completion summary.
- [ ] Mark Tier 1 complete in ROADMAP.md.
- [ ] Add CHANGELOG and HANDOFF updates if the final primitive inventory or public API contract changed.
- [ ] Commit the Tier 1 close-out documentation.

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
