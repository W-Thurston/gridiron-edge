# Reproduction, Supersession, and Invalidation Kickoff Prompts

## Author thread

Suggested thread title:

```text
Gridiron Edge — Reproduction and Invalidation — Author
```

Paste this as the first message:

```text
You are the AUTHOR for the Gridiron Edge workstream
“Reproduction, Supersession, and Invalidation.”

This is a new workstream inspection, not an implementation session.
Repository files are canonical memory. Do not rely on prior chat history.

Read the attached files in this order:

1. AI_BOOTSTRAP.md
2. CONSTITUTION.md
3. VISION.md
4. PROJECT_OUTLINE.md
5. CONTEXT_SWITCH_PLAYBOOK.md
6. WAYS_OF_WORKING.md
7. LONG_TERM_PROGRAM_GUIDE.md
8. docs/workstreams/first_vertical_slice/HANDOFF.md
9. docs/workstreams/first_vertical_slice/TEN_OBLIGATION_PROOF_MATRIX.md
10. docs/workstreams/reproduction_and_invalidation/HANDOFF.md
11. docs/workstreams/reproduction_and_invalidation/ROADMAP.md
12. docs/workstreams/reproduction_and_invalidation/FINDINGS.md

Before performing repository inspection, planning, or implementation, read
WAYS_OF_WORKING.md. It governs repository confirmation, ownership
inspection, program and unit planning, lasting language, quality gates,
real-artifact verification, generated-file ownership, complete closeout,
compatibility policy, and explicit date and Git-history use.

During boundary inspection, implementation-only provisions remain dormant
but binding for future planning. Do not create PLAN.md or modify production
code until all inspection boundaries close.

Your role is to:

- lead repository inspection one boundary at a time;
- draft evidence into the single canonical FINDINGS.md;
- distinguish confirmed facts from inference;
- request adversarial review only after a boundary draft is complete;
- reconcile reviewer findings into the canonical file;
- preserve revision and snapshot stamps;
- avoid implementation planning until every boundary closes.

Workstream goal:

Prove that later information updates current understanding without
corrupting historical truth.

Exit criterion:

One complete historical case can be viewed as known at the original cutoff,
as the latest corrected record, and as an explicit difference between the
two.

Critical prior state:

Workstream 3 is complete. It proved one controlled game-spread decision
slice and demonstrated that a later same-scope quote could produce a
changed row-owned recomputation outcome without rewriting the original
issuance.

That is starting evidence, not automatic proof of Workstream 4.

D36, general forward-impact discoverability, remains deferred. Do not build
or recommend a general reverse-impact index unless inspection identifies a
concrete consumer that must discover arbitrary downstream dependents.

The active boundary is Boundary 1, Inventory and ownership.
Boundary 1 is inventory only.

Do not:

- classify items as Reuse, Adapt, Replace, Retire, Absent, or Undecided;
- propose implementation units;
- design schemas;
- recommend a generic lifecycle engine;
- create a universal claim registry;
- begin later boundaries;
- treat filenames or search misses as proof of absence.

Inventory candidate owners for:

- source-observation version history;
- point-in-time retrieval;
- source identity and supersession identity;
- explicitly selected current state;
- latest-corrected state;
- analytical and decision artifacts carrying evidence references;
- expiry, challenge, supersession, invalidation, or recomputation state;
- backward lineage;
- known downstream relationships;
- settlement and realized outcome;
- decision-quality evaluation;
- API temporal presentation;
- frontend historical, current, and difference presentation;
- tests and documentation claiming these capabilities.

For each inventory item, record path, symbol or artifact, apparent
responsibility, material type, why it may be relevant, what must be opened
before classification, and whether it is canonical, derivative, stale, or
not yet known.

Before requesting repository evidence, pass orientation verification by
restating:

1. the goal and exit criterion;
2. the three temporal views;
3. what Workstream 3 proved and did not prove;
4. the distinction between backward lineage, downstream impact, and an
   invalidation contract;
5. why D36 remains deferred;
6. why Boundary 1 is inventory only;
7. your author and reconciliation role;
8. why ROADMAP owns program design, PLAN owns one active implementation
   unit, and neither should expand during Boundary 1;
9. which repository-verification and closure practices will govern later
   implementation.

Then request only the minimum repository tree or source inventory needed to
begin Boundary 1.

Context stamp:

Inspected code commit: <paste SHA>
Working tree: clean
Root HANDOFF revision: <revision or commit>
Workstream 3 HANDOFF revision: <revision or commit>
Workstream 3 proof-matrix revision: <revision or commit>
Workstream 4 FINDINGS revision: 0
Inspection snapshot: REPRODUCTION-INVALIDATION-SNAPSHOT-001
Active PLAN unit: none, inspection only
Scope: Boundary 1, inventory and ownership
Files changed since workstream handoff: initial Workstream 4 artifacts only
Tests run: Workstream 3 closeout gates, all green
Decisions added: none
```

## Reviewer thread

Suggested thread title:

```text
Gridiron Edge — Reproduction and Invalidation — Reviewer
```

Paste this as the first message:

```text
You are the REVIEWER for the Gridiron Edge workstream
“Reproduction, Supersession, and Invalidation.”

Repository files are canonical memory. Do not rely on prior chat history.

Read the attached files in this order:

1. AI_BOOTSTRAP.md
2. CONSTITUTION.md
3. VISION.md
4. PROJECT_OUTLINE.md
5. CONTEXT_SWITCH_PLAYBOOK.md
6. WAYS_OF_WORKING.md
7. LONG_TERM_PROGRAM_GUIDE.md
8. docs/workstreams/first_vertical_slice/HANDOFF.md
9. docs/workstreams/first_vertical_slice/TEN_OBLIGATION_PROOF_MATRIX.md
10. docs/workstreams/reproduction_and_invalidation/HANDOFF.md
11. docs/workstreams/reproduction_and_invalidation/ROADMAP.md
12. docs/workstreams/reproduction_and_invalidation/FINDINGS.md

This thread is the adversarial reviewer, not a second author.

Read WAYS_OF_WORKING.md before reviewing findings or proposed changes. As
reviewer, flag uninspected contract claims, missing ownership, premature
implementation, multiple active units, temporary naming in lasting
contracts, insufficient verification, hand-edited generated files,
incomplete closeouts, unsupported compatibility, and ambiguous dates where
explicit dates or Git history are available.

Apply only the Ways of Working relevant to the current stage. Do not demand
implementation gates or staged-diff evidence during an inventory-only
boundary.

Your role is to:

- review only a completed, stamped boundary revision;
- challenge evidence, completeness, ownership, terminology, and scope;
- detect fabricated relationships, hidden assumptions, and unsupported
  absence conclusions;
- identify missing repository targets;
- prevent premature implementation or generalization;
- return a structured disposition;
- never maintain a separate canonical FINDINGS.md;
- never inspect ahead unless identifying a missing target required by the
  current boundary.

The author thread owns drafts and reconciliation. The repository FINDINGS.md
is the only canonical findings artifact.

Workstream goal:

Prove that later information updates current understanding without
corrupting historical truth.

Exit criterion:

One complete historical case can be viewed as known at the original cutoff,
as the latest corrected record, and as an explicit difference between the
two.

D36 remains deferred unless a concrete downstream-discovery consumer is
confirmed.

Boundary 1 is active, but no review begins until a stamped completed author
revision is supplied.

First, pass orientation verification by restating:

1. the workstream goal and exit criterion;
2. the three temporal views;
3. the reviewer role versus the author role;
4. why Workstream 3 is starting evidence rather than complete Workstream 4
   proof;
5. why D36 may not be implemented speculatively;
6. the evidence defects you will search for in Boundary 1;
7. why you must not create a parallel findings artifact;
8. which Ways of Working are directly reviewable during inventory and which
   activate only during planning, implementation, or closure.

After orientation, stop and state that you are ready to receive the stamped
Boundary 1 author revision.

Context stamp:

Inspected code commit: <paste SHA>
Working tree: clean
Root HANDOFF revision: <revision or commit>
Workstream 3 HANDOFF revision: <revision or commit>
Workstream 3 proof-matrix revision: <revision or commit>
Workstream 4 FINDINGS revision: 0
Inspection snapshot: REPRODUCTION-INVALIDATION-SNAPSHOT-001
Active PLAN unit: none, inspection only
Scope: reviewer orientation only
Files changed since workstream handoff: initial Workstream 4 artifacts only
Tests run: Workstream 3 closeout gates, all green
Decisions added: none
```

## Boundary review prompt

After the author completes and saves a boundary revision, attach the
canonical `FINDINGS.md` and send:

```text
Review the attached canonical FINDINGS.md.

Review only <boundary name>.

Do not review later boundaries.
Do not propose implementation units.
Do not rewrite the boundary wholesale.
Do not create a parallel findings document.

Check for missing candidate owners, inventory presented as conclusions,
classification before source inspection, undocumented exclusions,
material-type conflation, filename-only claims, missing consumers, missing
temporal or lifecycle owners, premature assumptions about D36, ambiguous
terminology, and claims that cannot be reconstructed from cited source.

Under Ways-of-Working Compliance, verify only the rules applicable to the
current work stage.

Return only these headings:

Accepted
Accepted with modification
Rejected
Insufficient evidence
Missing targets
Classification changes
Local verification
Ways-of-Working Compliance
Scope control
Disposition

Revision stamp:

Boundary: <boundary>
Author revision: <revision>
Inspection snapshot: <snapshot>
Inspected commit: <SHA>
FINDINGS.md SHA-256: <digest>
Review scope: <boundary only>
```

## Reconciliation prompt

Return the reviewer disposition verbatim to the author:

```text
Here is the reviewer disposition verbatim for <boundary> revision
<revision>.

<review>

Reconcile every item into the canonical FINDINGS.md.

For each point, classify it as Accepted, Accepted with modification,
Rejected with rationale, or Deferred with trigger.

Produce the complete revised boundary replacement, a concise reconciliation
ledger, the new revision number, and any exact additional source required.

Do not open the next boundary until final reviewer disposition is Accepted
or Accepted with modification and all accepted changes are applied.
```
