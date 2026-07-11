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

#### W9.8: Dev Panel + Pending Highlight Mode — ✅ COMPLETE (2026-07-01, audit deferred)

Shipped the highlight mechanism in Tiers 1-2:
- **Tier 1:** `DevPanelContext` + floating bottom-right dev panel with
  Highlight Pending & Blocked toggle. `--highlight` CSS var.
- **Tier 2:** `usePendingHighlight` hook; retrofit `PendingField`,
  `BlockedField`, consolidated `ComingSoonCard`; new `PendingChip`
  primitive for inline pending text.

**Tier 3 (audit sweep) DEFERRED** to a post-pipeline task. The audit
requires fully-populated backend data to distinguish "frontend forgot
to mark this" from "backend hasn't populated this yet." Running the
full retrain pipeline first, then walking every screen in highlight
mode, is the correct sequence.

**Discipline established:** All future pending/blocked UI routes through
PendingChip / ComingSoonCard / field-status components, so new gaps
auto-participate in highlight mode. The deferred sweep confirms
completeness once real data is flowing.

**W9.8 mechanism complete; audit sweep deferred.**

---

## Paused Workstreams

_(none currently paused)_

---

## Changelog

| Date | Change |
|------|--------|
| 2026-07-01 | **W9.8 Dev Panel + Pending Highlight Mode design.** Locked. ~9-10 substeps across 4 tiers. Floating dev panel with highlight toggle; retrofit field-status components + new PendingChip primitive; audit sweep across all built screens. Sequenced before W9.10 Compare so highlight discipline is baked in. |
