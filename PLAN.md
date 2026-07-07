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

### Current Workstream: (none — between workstreams)

### Recently Completed

#### W9.6: GameDetail Full Fidelity — ✅ COMPLETE (2026-07-07)

Rebuilt GameDetail from placeholder-heavy skeleton to prototype
fidelity across 9 substeps in 4 tiers.

**Delivered:**
- Full-width game header with `TeamHero` primitives (colored marks
  + serif italic team names + AWAY/HOME context labels + kick date
  + venue placeholder + weather placeholder) — 2 substeps
- Model lean callout composed from `/edges` (recommendation + EV +
  confidence pill + WhyLink + slip button)
- Lines & Model Fair Value table (3 rows × 3 columns): Market row
  em-dashed pending W7; Model row from prediction data (spread,
  total, moneyline via `probToAmerican()`); Recommendation row
  composed from `/edges` per market with green highlight tint
- Win Probability card: 2-column with prob bands + labels on left,
  projected score + margin on right
- Team Comparison card: 8 metrics × 4 cohorts (Season/L4/Home/Away)
  via Pill primitive, "Open full comparison →" navigation to Compare
- Top Prop Edges card in right rail: 4-row compact list filtered
  to game_id, WhyLink dot per row, slip button per row

**Preserved as blocked placeholders:**
- Swing Factors (feature attribution workstream)
- Injuries (§5.3 injury data source)

**Data path adjustments:**
- Team city prefix stripped from `name` field in header
  (`stripCityPrefix` helper)
- `probToAmerican()` helper added to `utils/odds.ts`
- Client-side filtering for game-scoped composition (edges, props)
  since backend doesn't support per-game filter params

**Uses all 5 primitives from W9.5:**
- Pill (cohort tabs on Team Comparison)
- WhyLink (dot variant on model lean + prop edges)
- TeamMark (throughout, colored)
- Spark — not used in GameDetail (deferred to future win prob chart)
- TeamHero (both team heroes in header, plus prop row TeamMarks)

**W9.6 workstream complete.**

---

## Paused Workstreams

_(none currently paused)_

---

## Changelog

| Date | Change |
|------|--------|
| 2026-07-07 | **W9.6 complete.** GameDetail Full Fidelity shipped in 9 substeps across 4 tiers. All 5 W9.5 primitives consumed (heavy TeamHero + Pill + WhyLink usage). Composed cards from /edges, /games, /props, and `team_comparison` field from Step 7c. |
| 2026-07-06 | **W9.6 GameDetail Full Fidelity design.** Locked. 9 substeps across 4 tiers. Layout restructure + header composition + main column cards (lines table, win prob, team comparison) + right rail (prop edges + placeholders). Uses all 5 primitives from W9.5. |
