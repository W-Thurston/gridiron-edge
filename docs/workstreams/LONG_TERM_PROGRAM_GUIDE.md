# Gridiron Edge Long-Term Program Guide

## Purpose

This guide describes how Gridiron Edge progressively inspects, proves, and
extends capabilities after the foundational evidence work and first complete
vertical decision slice.

It is a reusable framework for capability admission, workstream sequencing,
proof, generalization, and long-term product development. It is not current
project state, an active plan, a feature backlog, a technical specification,
or authority to amend the locked product definition.

## Authority

The authority order is:

1. `CONSTITUTION.md`
2. `VISION.md`
3. `PROJECT_OUTLINE.md`
4. `DECISIONS.md`
5. The active workstream's `ROADMAP.md`
6. The active `PLAN.md`
7. `WAYS_OF_WORKING.md`
8. This guide

`WAYS_OF_WORKING.md` governs repository execution, validation,
documentation, and closure discipline. This guide governs long-term
capability admission, progression, proof sequencing, and generalization.
The gates below supplement `WAYS_OF_WORKING.md`; they do not replace its
commands or closeout mechanics.

`CONTEXT_SWITCH_PLAYBOOK.md` governs movement among author, reviewer,
implementation, recovery, and amendment threads.

`initial_chatgpt_responses.md` is a non-authoritative idea archive. Its
concepts may become inspection targets only after reconciliation with
canonical product artifacts and current repository evidence.

## Program objective

Gridiron Edge should mature into a transparent NFL intelligence and
decision-support system whose central product is a traceable chain:

```text
time-valid evidence
→ uncertain analytical belief
→ explicit policy decision
→ portfolio consequence
→ execution record, when applicable
→ honest and separate evaluation
```

Progress is measured by proof outcomes, not feature count, model count,
interface area, engagement, wager volume, or favorable financial results.

## Program operating principles

### Preserve before deriving

Capture mutable evidence before building conclusions that must later prove
what was knowable. Claims may be recomputed from preserved inputs and pinned
methods. Missing historical observations generally cannot be reconstructed
honestly.

### Inspect ownership before designing

Locate source, identity, temporal, validation, persistence, retrieval,
computation, policy, lifecycle, evaluation, API, frontend, test, generated,
and documentation owners before proposing a new contract or abstraction.

### Progress by proof

A workstream exits only when its stated capability is demonstrated through
real owners and durable evidence.

### Use one complete case before generalizing

Prove the smallest coherent end-to-end case. Generalize only after another
concrete domain case and real consumer demonstrate what is shared.

### Keep kinds separate

```text
source observation
≠ derived fact
≠ interpretation
≠ prediction
≠ market price
≠ analytical edge
≠ recommendation
≠ allocation
≠ execution
≠ settlement
≠ evaluation
```

### Make uncertainty and limitation intrinsic

Estimated claims carry an appropriate representation of uncertainty,
support, scenario range, or limitation. Deterministic artifacts carry
provenance and validity rather than artificial distributions.

### Treat abstention and absence as explicit outcomes

Unavailable, insufficient, conflicting, stale, expired, challenged,
superseded, invalidated, out-of-support, and not-applicable states remain
explicit where the domain requires them.

### Preserve historical claims

Later evidence may evaluate, challenge, supersede, expire, invalidate, or
require recomputation. It never rewrites the original evidence, method,
decision, allocation, execution, or evaluation context.

### Keep traceability concerns distinct

- **Backward lineage:** Which exact upstream evidence and methods produced
  the claim?
- **Downstream impact:** Which known claims and decisions consumed it?
- **Invalidation contract:** Which future observations or state changes may
  expire, challenge, supersede, or require recomputation?

### Keep ownership domain-specific

Do not create a universal analytical-claim schema, lifecycle engine, event
bus, or registry solely to make unlike domains appear uniform.

### Keep critical meaning backend-owned

The frontend may compose and disclose backend-owned states. It does not
recreate analytics, policies, portfolio logic, invalidation meaning,
settlement rules, or decision-quality conclusions.

### Keep durable knowledge in versioned files

Chat assists reasoning. Repository artifacts own state, evidence, decisions,
plans, acceptance, proof, and handoff context.

## Standard workstream lifecycle

All repository activity follows `WAYS_OF_WORKING.md`.

### Candidate selection

A workstream candidate states:

- user job;
- governing constitutional boundary;
- Vision invariant advanced;
- concrete current limitation;
- dependency reason;
- smallest coherent proof case;
- exit criterion;
- explicit exclusions;
- broader capability not claimed;
- amendment or decision implications.

### Entry gate

Before inspection:

- prior exit proof exists;
- canonical docs reflect reality;
- commit and tree state are recorded;
- mirror is current;
- no conflicting plan is active;
- a fresh findings artifact exists;
- deferred decisions are carried forward;
- the request does not require an unrecorded amendment.

### Boundary inspection

Default boundaries:

1. Inventory and ownership
2. Identity and temporal semantics
3. Persistence and retrieval
4. Computation and policy boundaries
5. Traceability, validity, and lifecycle
6. Evaluation and assurance
7. API, presentation, and user comprehension
8. Consolidation, reuse map, and planning

Use one canonical `FINDINGS.md`, one boundary at a time, fixed author and
reviewer roles, evidence labels, stamped revisions, and reconciliation into
the same file.

### Classification gate

Every inspected owner receives one evidence-backed disposition:

```text
Reuse
Adapt
Replace
Retire
Absent
Undecided
```

Each disposition names evidence, contract implications, consumers,
temporal semantics, failure states, test depth, decision impact, and what
would change the classification.

### Consolidation gate

Before planning:

- all boundaries close;
- reuse map is reconciled;
- missing capability differs from missing test evidence;
- deferred questions are explicit;
- decisions are recorded or proposed;
- units are dependency ordered;
- proof case and exit evidence are fixed.

### Planning gate

- exactly one implementation unit is active;
- goal and acceptance are bounded;
- ownership and exclusions are named;
- focused, full, and artifact gates are known;
- documentation and generated owners are known;
- names describe lasting behavior.

### Implementation and unit closure

Implementation and closure follow `WAYS_OF_WORKING.md` in full. Each unit
must additionally prove that it advances the workstream exit criterion,
preserves temporal truth, keeps unavailable and conflicting states
explicit, remains reproducible from a clean checkout, and does not
generalize beyond inspected evidence.

### Workstream exit gate

A workstream closes only when:

- the exit criterion is directly demonstrated;
- promised capabilities have durable evidence;
- proof avoids local-only artifacts;
- identities and paths validate through owners;
- historical reproduction is verified where applicable;
- failure and abstention states are included;
- deferrals are explicit;
- documentation is reconciled;
- a new thread can reconstruct final state from files alone.

## Evidence and review discipline

Default evidence labels:

```text
SOURCE_CONFIRMED
TEST_CONFIRMED
ARTIFACT_CONFIRMED
RUNTIME_CONFIRMED
DOCUMENTATION_ONLY
INFERRED
ABSENT_CONFIRMED
UNDECIDED
```

Adversarial review searches for category mistakes, hidden assumptions,
duplicated ownership, missing states, temporal leakage, fabricated
relationships, unsupported absence, ambiguous language, premature
generalization, frontend ownership drift, and unreproducible claims.

When validation tightens, search every fixture and caller, correct invalid
placeholders rather than weakening checks, add load-bearing coverage, and
rerun focused and full gates.

## Capability Admission Test

A capability enters formal planning only when all answers are explicit:

1. Which user job does it serve?
2. Which constitutional boundary governs it?
3. Which Vision invariant does it advance?
4. What concrete limitation motivates it?
5. Why now?
6. Which preserved evidence does it require?
7. What is its smallest proof case?
8. Which owner may already own part of it?
9. What new artifact kind is required?
10. What must remain separate?
11. Which unavailable, conflicting, stale, or out-of-support states exist?
12. What uncertainty or limitation applies?
13. What is the evidence cutoff?
14. Which future evidence changes validity?
15. How does the original reproduce?
16. How does current understanding differ from historical truth?
17. How are outcomes evaluated without result-oriented reasoning?
18. Which meaning remains backend-owned?
19. What is excluded?
20. What evidence justifies generalization?
21. Does it trigger amendment?
22. What durable proof demonstrates completion?
23. How does a clean checkout reproduce it?
24. What may later work safely assume?

Unknown answers keep the capability in inspection, not implementation.

## Generalization Gate

Generalization requires:

- two concrete domain cases;
- a real consumer of shared behavior;
- stable semantics, not similar fields;
- common and domain-specific ownership;
- temporal, failure, absence, and conflict semantics;
- an explicit replacement or migration decision;
- adversarial review;
- proof the abstraction reduces duplication without hiding meaning.

Otherwise retain behavior in the domain owner.

## Surface-Readiness Gate

A consequential output appears only when all applicable levels exist:

1. Conclusion
2. Explanation
3. Quantitative context
4. Method
5. Evidence and reproduction

The surface truthfully communicates artifact kind, temporal view,
uncertainty, freshness, method, cutoff, policy, allocation, lifecycle,
execution, unavailable or conflicting evidence, and route to inspection.

## Temporal-Transparency Standard

### As known at decision time

The authoritative reproduction and evaluation view. It uses only evidence
visible within the original cutoff and pinned methods and policies.

### Latest corrected record

The best currently available historical account. It may include later
corrections but does not replace original artifacts.

### Difference view

The explicit comparison identifies changed source evidence, known time,
changed fields, changed and unchanged claims, lifecycle consequences,
recommendation and allocation differences, evaluation differences, and the
reason for every material change.

## Long-term capability progression

These horizons are candidate dependency directions, not automatically
active workstreams.

### Horizon 1 — Reproduction, supersession, and invalidation

Prove one complete case across as-known, latest-corrected, and difference
views. Preserve the original decision and avoid speculative global impact
infrastructure.

### Horizon 2 — Trust and inspection experience

Make one consequential claim followable from useful conclusion through all
applicable transparency levels to exact evidence and reproduction.

### Horizon 3 — Prediction intelligence and limitations

Deepen distributions, uncertainty integrity, calibration support,
forecast-horizon behavior, out-of-support states, model disagreement,
errors, stability, blind spots, and version differences without collapsing
model quality into betting performance.

### Horizon 4 — Market intelligence

Treat markets as exact executable observations and independent information
sources. Preserve quote history, price and line movement, freshness,
availability, disagreement, consensus limitations, vig treatment, market
definitions, and benchmark behavior.

### Horizon 5 — Football intelligence

Make time-valid, reproducible football analysis useful without a wagering
output. Cover team and unit performance, opponent adjustment, personnel,
usage, tendencies, matchup interaction, game state, conditions, and
limitations.

### Horizon 6 — Scenario intelligence

Keep observations separate from counterfactual assumptions. Support base
and alternate cases, conditional predictions, sensitivity, affected claims,
and decision consequences without claiming a scenario will occur.

### Horizon 7 — Portfolio intelligence

Deepen recorded exposure, concentration, common factors, explicit
correlation assumptions, incremental exposure, allocation comparison,
drawdown, and portfolio evaluation while preserving the execution boundary.

### Horizon 8 — Outcome, audit, and learning

Evaluate football analysis, feature stability, prediction, uncertainty,
calibration, market movement, recommendation, allocation, execution,
settlement, return, portfolio outcome, and decision quality separately.
Research proposes new versions rather than rewriting history.

### Horizon 9 — Product surfaces

Compose proven capabilities into candidate Today, Game, Teams and Players,
Models and Research, Markets and Portfolio, and Track Record and Evidence
surfaces. Navigation never defines architecture.

### Horizon 10 — Fantasy and additional decision lenses

Apply shared player distributions to league-specific roster decisions while
keeping season-long fantasy, prop betting, and DFS policies separate. DFS
remains deferred until explicitly activated.

## Cross-horizon assurance

Every horizon strengthens applicable assurance:

- provenance;
- point-in-time validity;
- data quality;
- uncertainty integrity;
- reproducibility;
- recommendation audit;
- portfolio audit;
- execution audit;
- outcome and decision-quality evaluation;
- responsible use.

## Workstream Closure Standard

A completed workstream records:

- **Completed:** lasting capability;
- **Goal:** objective and exit criterion;
- **Files Added/Removed/Changed:** every durable owner and proof artifact;
- **Tests:** actual focused, integration, frontend, artifact, and full gates;
- **Acceptance:** direct evidence for every exit condition;
- **Deferred:** related capability not proven and its trigger;
- **Decisions:** new, amended, superseded, or unchanged choices;
- **Evidence:** canonical proof and reproduction command;
- **Next dependency:** what later work may and may not assume.

Implementation-unit closeouts retain the shorter canonical structure in
`WAYS_OF_WORKING.md`.

## Thread and context-switch standard

Every thread declares role, work stage, reading order, active boundary or
unit, commit, tree state, revisions, tests, and decisions. Orientation must
pass before work. Author and reviewer use one canonical artifact and
stamped revisions.

## Recommended Workstream 4 inspection

1. Inventory and ownership
2. Temporal semantics and identities
3. Historical reproduction
4. Supersession and invalidation behavior
5. Latest-corrected view
6. Difference view and downstream impact
7. Evaluation and presentation
8. Consolidation and implementation planning

D36 remains deferred unless the Difference view boundary confirms a
consumer requiring arbitrary downstream discovery.

## Program health checks

Periodically verify:

- canonical authority remains consistent;
- idea archives are not shadow authority;
- deferrals have not become assumptions;
- runtime artifacts are not clean-checkout proof;
- test ownership follows production ownership;
- cutoffs are explicit UTC;
- later evidence cannot leak historically;
- current selection is not file recency;
- corrected records do not overwrite originals;
- edges, recommendations, allocations, and executions remain separate;
- full eligible universes, wins, losses, abstentions, and invalidations are
  retained;
- docs reflect actual state;
- a new thread can rehydrate from files alone.

## Long-term success standard

### Integrity

Historical outputs reproduce and superseded evidence remains visible.

### Epistemic honesty

Uncertainty, limitation, missing, conflicting, stale, unstable, and
out-of-support states remain explicit.

### Decision discipline

Edges do not automatically become recommendations, recommendations do not
automatically become allocations, and allocations do not imply execution.

### Evaluative honesty

Complete eligible universes remain preserved. Financial outcomes are
reported but do not independently define quality.

### Comprehension and utility

The intended user can determine what is claimed, what is unknown, why,
what would change it, whether a price is usable, how policy acted, what was
executed, what happened, and how to reproduce the chain.

## Final program rule

Build the smallest truthful capability that advances the locked product
vision, prove it through real owners, preserve historical evidence, and
generalize only when another concrete need demonstrates what is shared.

The repository is the memory. The evidence is the authority. The proof is
the progress.
