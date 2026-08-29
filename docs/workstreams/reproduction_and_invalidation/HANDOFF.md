# Reproduction, Supersession, and Invalidation Handoff

## Status

Boundary inspection is opening. Boundary 1, Inventory and ownership, is
active. No implementation unit exists.

## Goal

Prove that later information updates current understanding without
corrupting historical truth.

## Exit criterion

One complete historical case can be viewed:

- as known at the original cutoff;
- as the latest corrected record;
- as an explicit difference between the two.

## Prior proof available

Workstream 3 proved one controlled game-spread vertical slice and a later
same-scope quote whose recomputation outcome differed without rewriting the
original issuance.

That proof is starting evidence. It does not itself establish the full
three-view Workstream 4 contract.

## Active boundary

Inventory and ownership only.

## Operating discipline

All inspection, planning, implementation, review, validation, and closure
activity follows:

```text
docs/workstreams/WAYS_OF_WORKING.md
```

Boundary inspection is active. Therefore no production change and no active
`PLAN.md` unit exists yet.

## Author and reviewer roles

The Author thread leads inspection, drafts the canonical `FINDINGS.md`, and
reconciles review.

The Reviewer thread reviews one completed stamped boundary revision at a
time. It does not maintain a parallel findings file or inspect ahead.

## Deferred constraint

D36, general forward-impact discoverability, remains deferred unless this
inspection identifies a concrete consumer that must discover arbitrary
downstream dependents.

## Canonical evidence

All inspection evidence belongs in `FINDINGS.md`. Chat output is not durable
project state.

## Reading order

```text
AI_BOOTSTRAP.md
CONSTITUTION.md
VISION.md
PROJECT_OUTLINE.md
CONTEXT_SWITCH_PLAYBOOK.md
WAYS_OF_WORKING.md
LONG_TERM_PROGRAM_GUIDE.md
first_vertical_slice/HANDOFF.md
first_vertical_slice/TEN_OBLIGATION_PROOF_MATRIX.md
reproduction_and_invalidation/HANDOFF.md
reproduction_and_invalidation/ROADMAP.md
reproduction_and_invalidation/FINDINGS.md
```
