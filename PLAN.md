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

### Current Workstream: W9.8 — Dev Panel + Pending Highlight Mode

**Status:** Designing.

### What we are building

A floating dev panel (summoned by a bottom-right corner button) that
hosts development/verification tooling. First tool: a **Highlight
Pending & Blocked** toggle. When on, every pending/blocked element
across the app lights up in bright orange, making backend gaps
instantly visible during a visual pass.

The panel is a dev surface, not a user feature. It operationalizes
the frontend-as-verification-surface principle (ROADMAP §3, principle
6): the frontend only helps us see what's missing if the gaps are
visible, not hidden.

Alongside the toggle: an audit sweep across all built screens to
ensure every silently-missing piece of information is surfaced with
a visible pending/blocked marker — never skipped.

### Why we are building it now

The frontend exists at this stage as a verification surface — a way
to *see* what the backend can't yet provide. That only works if gaps
are visibly marked, not withheld. The near-miss on PlayerProp's
confidence/EV fields (almost dropped, instead surfaced as "pending")
is the canonical example of the discipline we want everywhere.

Building this toggle now, before more screen rebuilds:
1. Changes how we build — every subsequent substep gets designed with
   "does this register in highlight mode?"
2. Makes the visual pass effortless — flip on, walk the app, see every
   gap
3. Retroactively audits everything already built (Dashboard,
   GameDetail, TeamsScreen, PlayerProp, etc.)

Sequencing: toggle → audit sweep → then resume W9.10 (Compare) with
the highlight discipline baked in.

#### Success criteria

- Floating dev panel button in bottom-right corner; opens/closes a
  dev panel
- Panel contains a "Highlight Pending & Blocked" toggle
- Toggle state persisted to localStorage via new `DevPanelContext`
- When highlight on: all pending/blocked elements light up with bright
  orange outline + tint
- When highlight off: zero visual change from current app
- `PendingField`, `BlockedField`, `ComingSoonCard` all participate
- New `PendingChip` primitive for inline pending-text cases; existing
  ad-hoc inline markers retrofitted
- Audit sweep completed: every built screen walked in highlight mode,
  silently-missing items surfaced
- All quality gates pass

### Locked architectural decisions

| Decision | Choice |
|---|---|
| State container | New `DevPanelContext` (localStorage-persisted) |
| Toggle placement | Floating bottom-right button → opens dev panel |
| Button visibility | Always visible for now (gate behind `import.meta.env.DEV` if app ever ships to users — noted as future consideration) |
| Highlight mechanism | `usePendingHighlight()` hook returning styles (empty object when off) |
| Highlight color | Dedicated `--highlight` CSS var (bright orange), separate from `--warn` to avoid collision with legitimate warn usage |
| Retrofit targets | `PendingField`, `BlockedField`, `ComingSoonCard` |
| New primitive | `PendingChip` for inline pending-text cases |
| ComingSoonCard | Consolidate drifted per-screen copies into one shared primitive |
| Panel scope v1 | Highlight toggle only; room to grow (field-status demo could return here later) |

### Prerequisite

None. Builds on existing AppState/context patterns and field-status
components.

### Tiers

**Tier 1 — Dev panel infrastructure (2 substeps).**

- 1a: `DevPanelContext` with `highlightPending` boolean, localStorage
  persistence, provider wired into app root. `--highlight` CSS var
  defined.
- 1b: Floating bottom-right button + dev panel shell. Button toggles
  panel. Panel contains "Highlight Pending & Blocked" toggle. Styled
  as a distinct utilitarian dev surface (not matching app cards).

**Tier 2 — Highlight rendering (3 substeps).**

- 2a: `usePendingHighlight()` hook + retrofit `PendingField` and
  `BlockedField` to light up when highlight on.
- 2b: Consolidate `ComingSoonCard` variants (GameDetail, PlayerProp,
  TeamsScreen) into one shared `components/primitives/ComingSoonCard.tsx`;
  retrofit for highlight support.
- 2c: New `PendingChip` primitive + retrofit inline pending text
  (confidence/EV/line pending on PlayerProp, SituationalSplits empty
  states, etc.).

**Tier 3 — Audit sweep (4 substeps).**

Walk each built screen in highlight mode. Surface silently-missing
items with markers. Produce punch-list; fix trivial items inline,
note larger gaps.

- 3a: Dashboard + GamesList
- 3b: GameDetail
- 3c: TeamsScreen
- 3d: PlayerProp + PlayersExplorer + PlayoffProjections

**Tier 4 — Cleanup + close-out (1 substep).**

- 4a: Cleanup + workstream close-out.

### Disconfirming evidence

- **If ComingSoonCard copies have drifted significantly**, consolidation
  may surface subtle behavior differences. Reconcile to the most
  complete version; note any intentional per-screen variance.
- **If the audit sweep reveals a screen has a large silently-missing
  section** (not just a small inline gap), we note it as a follow-up
  workstream item rather than fixing inline — keeps the sweep bounded.
- **If `--highlight` orange clashes visually with team primary colors**
  on team-colored surfaces (hero bands), the outline may need a
  contrasting treatment (e.g., dashed border or inset shadow) rather
  than solid orange. Adjust during Tier 2.
- **If localStorage persistence of highlight mode causes confusion**
  (user leaves it on, forgets), consider defaulting to off on each
  session. Decide during 1a.

### Timeline

Total: ~9-10 substeps. Not tied to calendar; natural cadence.

### Success artifacts

By workstream close:

- Floating dev panel available app-wide
- Highlight toggle lights up all pending/blocked elements
- `PendingChip` primitive + consolidated `ComingSoonCard`
- Every built screen audited; silently-missing items surfaced
- Discipline established for future screen work: pending things must
  register in highlight mode
- Punch-list of any larger gaps found during the sweep (for future
  workstreams)
`

---

## Paused Workstreams

_(none currently paused)_

---

## Changelog

| Date | Change |
|------|--------|
| 2026-07-01 | **W9.8 Dev Panel + Pending Highlight Mode design.** Locked. ~9-10 substeps across 4 tiers. Floating dev panel with highlight toggle; retrofit field-status components + new PendingChip primitive; audit sweep across all built screens. Sequenced before W9.10 Compare so highlight discipline is baked in. |
