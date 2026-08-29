# Reproduction, Supersession, and Invalidation Roadmap

## Status

Boundary inspection is active. Implementation planning has not started.
No `PLAN.md` unit is active.

## Goal

Prove that later information updates current understanding without
corrupting historical truth.

## Entry condition

Workstream 3 is complete and supplies one reproducible vertical decision case
to revisit.

## Exit criterion

One complete historical case supports:

- an as-known-at-decision-time view;
- a latest-corrected view;
- an explicit difference view.

The complete case must preserve the original claim and decision, identify
later evidence, represent the appropriate downstream lifecycle or
recomputation outcome, and explain material differences without pretending
later facts were available earlier.

## Ways of Working

This workstream follows `docs/workstreams/WAYS_OF_WORKING.md`.

Program-level capability and sequence are locked here only after boundary
inspection closes. Exactly one bounded implementation unit will then be
activated in `PLAN.md`. No implementation unit is active during inspection.

## Inspection boundaries

1. Inventory and ownership
2. Temporal semantics and identities
3. Historical reproduction
4. Supersession and invalidation behavior
5. Latest-corrected view
6. Difference view and downstream impact
7. Evaluation and presentation
8. Consolidation and implementation planning

These boundaries are inspection structure, not production architecture.

## Boundary intentions

### Inventory and ownership

Locate current and candidate owners before classification.

### Temporal semantics and identities

Inspect effective time, system-known time, source-published time, evidence
cutoff, supersession time, selected-current semantics, corrected-record
semantics, and identity stability across recomputation.

### Historical reproduction

Follow one completed Workstream 3 case through exact source, product,
forecast, issuance, policy, recommendation, allocation, and evaluation
relationships.

### Supersession and invalidation behavior

Determine which domain-owned outcomes are required, such as unchanged,
expired, challenged, superseded, invalidated, recomputation required,
recomputed, unavailable, conflicting, or not applicable.

### Latest-corrected view

Identify what owns the best current historical account, which later evidence
it includes, and how original historical artifacts remain separate.

### Difference view and downstream impact

Determine how the system explains exact changed evidence, changed and
unchanged conclusions, recommendation eligibility, allocation, evaluation,
and the reason for every material difference.

This boundary is the evidence gate for reconsidering D36.

### Evaluation and presentation

Inspect API and frontend behavior for historical, current, corrected, and
difference states without frontend reimplementation of domain meaning.

### Consolidation and implementation planning

Produce the reuse map, confirmed gaps, D36 disposition, decision proposals,
dependency-ordered units, one active `PLAN.md` unit, fixed proof case, and
exit-evidence design.

## Explicit guardrails

- Do not treat file modification time as current-state identity.
- Do not use current selection as a substitute for latest-corrected semantics
  without inspecting its contract.
- Do not conflate correction, supersession, challenge, expiry, invalidation,
  and recomputation.
- Do not create a universal physical claim object.
- Do not build general downstream discovery without a concrete consumer.
- Do not allow later evidence to rewrite historical claims or decisions.
- Do not let the frontend recreate domain-owned temporal meaning.
- Do not treat realized outcome as corrected pregame evidence.
- Do not preserve development-era compatibility without a current contract.

## Deferred

D36, general forward-impact discoverability, remains deferred unless a
concrete downstream-discovery requirement is confirmed during inspection.
