# Gridiron Edge Long-Term Program Guide

## Purpose

This guide describes how Gridiron Edge should progressively inspect,
prove, and extend capabilities after the foundational evidence work and
the first complete vertical decision slice.

It exists to preserve the larger product ambition while keeping future
work evidence-driven, bounded, reproducible, and consistent with the
project's established ways of working.

This guide is:

- a reusable framework for selecting and sequencing future workstreams;
- a bridge between the locked product vision and concrete repository work;
- a standard for inspection, planning, implementation, proof, and closure;
- a catalog of long-term capability horizons and the gates that govern them;
- a safeguard against losing durable product intent during thread changes.

This guide is not:

- a source of current project state;
- a replacement for `PROJECT_OUTLINE.md`;
- an active implementation plan;
- a conventional feature backlog;
- a technical specification;
- authority to amend `CONSTITUTION.md` or `VISION.md`;
- permission to build generalized infrastructure before a concrete need is
  demonstrated.

## Authority

The authority order is:

1. `CONSTITUTION.md`
2. `VISION.md`
3. `PROJECT_OUTLINE.md`
4. `DECISIONS.md`
5. The active workstream's `ROADMAP.md`
6. The active `PLAN.md`
7. This guide

Where this guide conflicts with a higher-authority artifact, the
higher-authority artifact wins.

`initial_chatgpt_responses.md` is a non-authoritative design-idea archive.
It preserves valuable product possibilities, conceptual distinctions,
questions, and future inspection targets. Its contents do not become
requirements until they are reconciled with the canonical product
artifacts, supported by repository evidence, scoped through an approved
workstream, and recorded in the appropriate authority file.

## Program objective

Gridiron Edge should mature into a transparent NFL intelligence and
decision-support system whose central product is a traceable chain from:

```text
time-valid evidence
→ uncertain analytical belief
→ explicit policy decision
→ portfolio consequence
→ execution record, when applicable
→ honest and separate evaluation
```

Progress is measured by proof outcomes, not feature count, interface area,
model count, or betting volume.

The long-term objective is not to produce more picks. It is to make every
consequential conclusion understandable, challengeable, reproducible, and
honestly evaluated while preserving what was knowable at the relevant time.

## Program operating principles

### Preserve before deriving

Capture mutable evidence before building conclusions that cannot later
prove what was knowable. Claims can often be recomputed from preserved
inputs and pinned methods. Missing historical observations generally
cannot be reconstructed honestly.

### Inspect ownership before designing

Before proposing a new contract, schema, service, surface, or abstraction,
locate the current owners of:

- source evidence;
- identity;
- temporal semantics;
- validation;
- persistence;
- retrieval;
- computation;
- policy;
- lifecycle state;
- evaluation;
- API presentation;
- frontend composition;
- test evidence;
- generated artifacts;
- documentation claims.

Repository inspection determines what should be reused, adapted, replaced,
retired, or created. Product ambition alone does not establish absence.

### Progress by proof, not feature count

A workstream exits only when its capability is demonstrated through real
boundaries and durable evidence. Completion is never inferred from code
volume, a green unit test suite alone, or accumulated partial features.

### Use one complete case before generalizing

Prove the smallest coherent end-to-end case first. Generalize only after a
second or third concrete case demonstrates what is genuinely shared.
Similar field names, serialization shapes, or implementation patterns are
not sufficient evidence for a universal abstraction.

### Keep kinds separate

Source observations, derived facts, interpretations, predictions, market
prices, analytical edges, recommendations, portfolio allocations,
executions, settlements, realized outcomes, and evaluations remain
distinct objects with distinct ownership.

Passing one boundary never implicitly passes the next:

```text
analytical edge
≠ recommendation
≠ portfolio allocation
≠ attempted execution
≠ actual execution
≠ settlement
```

### Make uncertainty and limitation intrinsic

Every estimated claim carries an appropriate representation of
uncertainty, support, scenarios, or limitation. Deterministic artifacts
carry provenance and validity rather than artificial probability
statements.

When evidence cannot support a precise conclusion, the correct output may
be directional, unstable, insufficient, unavailable, conflicting,
out-of-support, or not applicable.

### Treat abstention and absence as explicit outcomes

No conclusion, no edge, no recommendation, no allocation, stale evidence,
missing evidence, conflicting evidence, expired evidence, and unsupported
inference are meaningful states where applicable. They must not collapse
into blank UI, `None` without semantics, or generic failure.

### Preserve historical claims

Later evidence may evaluate, challenge, supersede, expire, invalidate, or
require recomputation. It must never rewrite the original evidence,
method, decision, allocation, execution record, or evaluation context.

Historical truth and current understanding are separate views over
preserved versions.

### Keep traceability concerns distinct

Every consequential claim should support three conceptually separate
questions:

- **Backward lineage:** Which exact upstream evidence and methods produced
  this claim?
- **Downstream impact:** Which known claims and decisions consumed this
  artifact?
- **Invalidation contract:** Which future observations or state changes
  could expire, challenge, supersede, or require recomputation of it?

A concrete need for one concern does not automatically justify a universal
mechanism for all three.

### Keep ownership domain-specific

Use shared conceptual contracts and capability protocols where evidence
supports them. Do not create a universal analytical-claim schema, global
lifecycle engine, generic event bus, or all-domain registry merely to make
unlike domains appear uniform.

The common conceptual spine should strengthen domain ownership, not erase
it.

### Keep critical meaning backend-owned

The frontend may compose, label, prioritize, and progressively disclose
backend-owned states. It must not recreate analytical formulas,
recommendation rules, portfolio policy, invalidation semantics,
settlement rules, or decision-quality conclusions.

### Keep durable knowledge in versioned files

Chat assists reasoning but does not own project state. Durable state,
decisions, evidence, plans, acceptance criteria, proof, and handoff context
belong in versioned repository artifacts.

End every working session by making the files true.

## Standard workstream lifecycle

Every future workstream follows the lifecycle below unless a documented
reason justifies a narrower variation.

### 1. Candidate selection

A proposed workstream must state:

- the user job it advances;
- the constitutional boundary that governs it;
- the Vision invariant or product promise it advances;
- the concrete current limitation or consumer need motivating it;
- why it belongs at this point in the dependency order;
- which earlier proof it depends on;
- the smallest coherent proof case;
- the proposed exit criterion;
- what it explicitly excludes;
- what broader capability it will not claim;
- whether it may require a constitutional amendment or new durable
  decision.

Attractive ideas do not enter formal planning without passing the
Capability Admission Test in this guide.

### 2. Entry gate

Before a workstream inspection begins, verify:

- the prior workstream's exit evidence exists;
- canonical documentation reflects the actual repository state;
- the inspected commit is recorded;
- the working tree state is recorded;
- the repository mirror is current;
- no conflicting `PLAN.md` unit is active;
- the workstream has a fresh findings artifact;
- the requested capability does not require an unrecorded amendment;
- current deferred decisions are carried forward explicitly;
- the opening context stamp is complete.

If the new thread cannot reconstruct the state from the supplied files,
stop and patch the files. Missing rehydration context is a project finding,
not a conversational inconvenience.

### 3. Boundary inspection

Use one canonical `FINDINGS.md`, one boundary at a time, with fixed author
and reviewer roles inside each boundary.

The default eight-boundary template is:

1. Inventory and ownership
2. Identity and temporal semantics
3. Persistence and retrieval
4. Computation and policy boundaries
5. Traceability, validity, and lifecycle
6. Evaluation and assurance
7. API, presentation, and user comprehension
8. Consolidation, reuse map, and implementation-unit derivation

This is a default template, not a universal physical architecture. A
workstream may merge or split boundaries when the domain warrants it, but
must retain the one-boundary-at-a-time inspection discipline.

For each boundary:

1. Open the owning source before classifying it.
2. Record evidence with explicit labels.
3. Distinguish production code, test evidence, documentation, generated
   output, and local runtime data.
4. Separate confirmed fact from inference.
5. Route the canonical boundary to adversarial review.
6. Reconcile the review into the same canonical file.
7. Close the boundary before opening the next one.

Do not design implementation units while ownership inspection remains
open.

### 4. Classification gate

Every discovered owner receives one evidence-backed disposition:

- **Reuse**
- **Adapt**
- **Replace**
- **Retire**
- **Absent**
- **Undecided**

Each classification must identify:

- supporting source evidence;
- contract implications;
- affected consumers;
- temporal implications;
- failure and absence semantics;
- migration relevance, if any;
- test depth required;
- whether a locked decision is implicated;
- evidence that would falsify or change the classification.

Do not conclude absence from filenames, indexes, search misses, or lack of
memory. Absence requires bounded exhaustive confirmation.

### 5. Consolidation gate

Before implementation planning:

- all inspection boundaries are closed;
- the reuse map is reconciled;
- missing capabilities are distinguished from missing tests;
- deferred questions are explicit;
- amendment triggers are identified;
- decisions are recorded or proposed;
- candidate implementation units are dependency ordered;
- the workstream proof case is fixed;
- the exit evidence format is defined.

### 6. Planning gate

Before activating `PLAN.md`:

- exactly one implementation unit is active;
- its goal is bounded and lasting;
- acceptance criteria are observable and testable;
- ownership boundaries are named;
- exclusions are explicit;
- required focused and full quality gates are known;
- artifact or response verification is specified;
- generated files and documentation owners are known;
- the unit can close coherently in one commit or a deliberately bounded
  sequence;
- the unit name does not rely on temporary phase or milestone language.

The roadmap owns future sequence. `PLAN.md` owns only the active unit.

### 7. Implementation gate

For every unit:

1. Confirm repository and working-tree state.
2. Read the owning boundaries and active plan.
3. Search for all affected call sites before moving or tightening
   ownership.
4. Implement through existing domain owners where appropriate.
5. Preserve immutability and point-in-time semantics.
6. Represent unavailable and conflicting states explicitly.
7. Add focused tests under the correct production-file owner.
8. Run focused quality gates.
9. Run the established full quality gates.
10. Inspect real artifacts, paths, identities, or responses.
11. Regenerate owned generated artifacts when applicable.
12. Perform adversarial review for consequential contract changes.
13. Recheck fixtures after tightening validation.
14. Inspect the staged inventory.
15. Inspect the staged diff.
16. Remove temporary scripts and accidental files.
17. Update `PLAN.md` directly in the same commit.
18. Update `HANDOFF.md`, `ROADMAP.md`, `DECISIONS.md`, and `CHANGELOG.md`
    where durable state changed.
19. Commit one coherent Conventional Commit unit.
20. Re-mirror canonical files.

### 8. Unit closure gate

An implementation unit closes only when:

- its acceptance criteria are directly proven;
- focused tests pass;
- full quality gates pass;
- lasting code and test names do not retain temporary unit terminology;
- persisted artifacts and paths are verified where applicable;
- documentation reflects actual behavior;
- the staged diff contains only intended files;
- `PLAN.md` uses the required closeout headings:

```text
Completed
Goal
Files Added/Removed/Changed
Tests
Acceptance
```

### 9. Workstream exit gate

A workstream closes only when:

- its stated exit criterion is directly demonstrated;
- every promised capability has durable evidence;
- the proof does not rely on local-only or untracked artifacts;
- every claimed path and identity is validated through its owner;
- historical reproduction is verified where applicable;
- failure, abstention, and unavailable states are included;
- deferred capabilities are named without implied completion;
- no temporary chronology leaked into lasting contracts;
- `HANDOFF.md`, `ROADMAP.md`, and proof artifacts are reconciled;
- a new thread can reconstruct the final state from files alone.

## Evidence and review discipline

### Evidence labels

Each findings artifact should define a bounded evidence vocabulary. The
default set is:

- `SOURCE_CONFIRMED`
- `TEST_CONFIRMED`
- `ARTIFACT_CONFIRMED`
- `RUNTIME_CONFIRMED`
- `DOCUMENTATION_ONLY`
- `INFERRED`
- `ABSENT_CONFIRMED`
- `UNDECIDED`

A statement may carry more than one label when different evidence types
support different parts of it.

### Author and reviewer roles

Within a boundary or consequential design unit:

- one model or contributor authors the canonical proposal;
- another performs adversarial review;
- the author classifies each critique as accepted, accepted with
  modification, rejected with rationale, or deferred;
- reconciliation updates the single canonical artifact;
- revision and snapshot stamps prevent parallel truths.

Adversarial review should search specifically for:

- category mistakes;
- hidden assumptions;
- duplicated ownership;
- missing failure states;
- leakage and temporal errors;
- fabricated identities or relationships;
- unverified absence conclusions;
- ambiguous terminology;
- product clutter;
- premature generalization;
- frontend recreation of backend meaning;
- claims that cannot be reproduced;
- ways the user could be misled.

### Validation-tightening rule

Whenever validation becomes stricter:

1. Search for every fixture and caller exercising the path.
2. Expect older placeholders to fail.
3. Correct invalid fixtures rather than weakening valid checks.
4. Add a test proving the new validation is load-bearing.
5. Re-run both focused and full gates.

## Capability Admission Test

A capability may enter formal workstream planning only when all answers
below are explicit.

1. Which user job does it serve?
2. Which constitutional boundary governs it?
3. Which Vision invariant does it advance?
4. What concrete current limitation motivates it?
5. Why is now the correct dependency point?
6. Which preserved evidence does it require?
7. What is its smallest coherent proof case?
8. Which existing owner may already own part of it?
9. What new artifact kind, if any, must exist?
10. What must remain separate from it?
11. What unavailable, conflicting, stale, or out-of-support states exist?
12. What uncertainty, support, scenario range, or limitation applies?
13. What is its evidence cutoff?
14. Which future evidence can expire, challenge, supersede, invalidate, or
    require recomputation?
15. How will the original conclusion reproduce?
16. How will current understanding differ from historical truth?
17. How will favorable and unfavorable outcomes be evaluated without
    result-oriented reasoning?
18. Which logic must remain backend-owned?
19. What is explicitly excluded?
20. What evidence would justify generalization?
21. Does it trigger a constitutional amendment?
22. What durable artifact or response will prove completion?
23. How will a clean checkout reproduce the proof?
24. What later work may safely assume after this capability closes?

If any answer is unknown, the capability remains an inspection candidate,
not an implementation commitment.

## Generalization Gate

Do not extract a universal contract merely because multiple objects share
field names or implementation patterns.

Generalization requires:

- at least two concrete domain cases;
- evidence that real consumers need shared behavior;
- stable shared semantics, not merely similar serialization;
- explicit ownership of common and domain-specific behavior;
- defined temporal semantics;
- defined failure, absence, and conflict semantics;
- an evidence-backed migration or clean-sheet replacement decision;
- adversarial review for category mistakes;
- proof that the abstraction reduces duplication without obscuring domain
  meaning;
- focused and cross-domain tests showing the shared owner remains truthful.

Otherwise, retain the behavior in its domain owner and record the possible
abstraction as an undecided future candidate.

## Surface-Readiness Gate

A consequential output may enter a composed interface only if all
applicable transparency levels exist:

1. Conclusion
2. Explanation
3. Quantitative context
4. Method
5. Evidence and reproduction

The surface must expose or truthfully summarize:

- the artifact kind;
- the temporal view;
- uncertainty, support, or limitation;
- source freshness and availability;
- method or model identity;
- evidence cutoff;
- policy state;
- allocation state;
- invalidation, challenge, or supersession state;
- execution state, if applicable;
- explicit unavailable or conflicting evidence;
- a route to deeper inspection.

The frontend may format and compose these states. It must not recreate
their analytical, policy, portfolio, invalidation, or evaluation meaning.

## Temporal-Transparency Standard

Long-term product surfaces should support three distinct temporal views
for consequential historical cases.

### As known at decision time

The authoritative view for reproducing and evaluating an original claim
or decision. It uses only evidence versions visible within the declared
cutoff and the methods and policies pinned to the original artifact.

### Latest corrected record

The best currently available historical account. It may include later
source corrections and newer evidence, but it must not replace or relabel
the original historical artifact.

### Difference view

The explicit comparison between the original and latest-corrected views.
It should identify:

- what source evidence changed;
- when the change became known;
- which fields differ;
- which claims differ;
- which claims remain unchanged;
- which conclusions expired, were challenged, were superseded, or require
  recomputation;
- whether recommendation eligibility changed;
- whether allocation changed;
- whether evaluation interpretation changed;
- why each material difference occurred.

A view switch without an explanation of the difference is incomplete
temporal transparency.

## Long-Term Capability Progression

The horizons below preserve the larger product ambition. They are
candidate dependency directions, not automatically active workstreams.
Every horizon must pass the Capability Admission Test and repository
inspection before entering formal planning.

## Horizon 1 — Reproduction, supersession, and invalidation

### Objective

Prove that later information updates current understanding without
corrupting historical truth.

### Required complete case

One historical case supports:

```text
as-known-at-decision-time view
latest-corrected view
explicit difference view
```

### Core proof areas

- later source version preserved beside the earlier version;
- explicit source supersession semantics;
- original claim and decision reproduction;
- latest-corrected interpretation;
- lifecycle outcome appropriate to each affected artifact;
- exact changed and unchanged conclusions;
- realized evidence that evaluates without rewriting;
- bounded downstream relationships;
- concrete evidence for or against revisiting general forward-impact
  discoverability.

### Guardrails

Do not assume the completed first vertical slice already proves the full
three-view product behavior. Do not build a global reverse-impact index,
general event bus, universal claim registry, or generic lifecycle engine
without a concrete consumer requirement.

## Horizon 2 — Trust and inspection experience

### Objective

Convert backend rigor into intended-user comprehension through progressive
disclosure.

### Candidate proof case

One consequential game-spread recommendation or abstention can be followed
from a useful product conclusion through all applicable transparency
levels to exact persisted evidence and a reproduction path.

### Candidate capabilities

- plain-language conclusion;
- explanation of major reasons and risks;
- quantitative context;
- method identity and limitations;
- exact evidence and reproduction;
- temporal-view labeling;
- invalidation and freshness visibility;
- distinction between edge, recommendation, allocation, execution, and
  outcome;
- challenge paths that do not require reading raw storage artifacts first.

### Exit principle

Trust is visible at the point of every claim, not confined to a separate
audit page.

## Horizon 3 — Prediction intelligence and limitations

### Objective

Deepen the model as an instrument without treating predictions as truth or
expanding model count for its own sake.

### Candidate proof areas

- predictive distributions or other defensible uncertainty forms;
- interval coverage and uncertainty integrity;
- leakage-safe calibration with visible support;
- forecast-horizon degradation;
- out-of-distribution or out-of-support status;
- champion and challenger disagreement;
- residual and error distributions;
- stability across training windows;
- known blind spots;
- model-version difference views;
- explicit insufficient or unstable calibration states;
- relevant benchmark comparison.

### Candidate first exit case

One prediction exposes its estimate, uncertainty, support status,
calibration evidence, model identity, limitations, and performance against
a valid benchmark without overstating sparse evidence.

### Guardrails

- Evaluate calibration at the broadest defensible level.
- Show sample support and uncertainty.
- Do not claim precise subgroup calibration without evidence.
- Keep model quality separate from betting profitability and decision
  quality.
- Do not present confidence tiers as guarantees.

## Horizon 4 — Market intelligence

### Objective

Treat betting markets as both collections of executable observations and
independent information sources.

### Candidate capabilities

- exact line and price history;
- quote freshness and availability;
- line movement separated from price movement;
- sportsbook disagreement;
- consensus methodology with explicit limitations;
- vig and no-vig transformations;
- provider-specific anomaly states;
- quote responsiveness;
- market definitions and settlement rules;
- opening, current, and eligible-closing semantics;
- market benchmark behavior;
- market disagreement that can remain non-actionable;
- point-in-time line shopping without rewriting observed history.

### Candidate first exit case

One market can be inspected as exact provider offers, a bounded market
summary, and an independent benchmark against the model while preserving
quote identity, freshness, availability, and uncertainty or limitation in
market interpretation.

### Guardrails

- An observed quote is not a consensus.
- A consensus is not necessarily an executable offer.
- Market disagreement is not automatically an analytical edge.
- An analytical edge is not automatically a recommendation.
- A quoted price is not claimed available without current evidence.

## Horizon 5 — Football intelligence

### Objective

Make football analysis useful in its own right, without requiring a model
prediction or betting narrative.

### Candidate capability sequence

1. Team and unit performance identities
2. Opponent adjustment
3. Personnel and usage
4. Tactical tendencies
5. Matchup interactions
6. Game-state effects
7. Venue, surface, rest, and travel context
8. Sustainable versus unstable performance
9. Explanatory evidence and limitations
10. Consumption by prediction and scenario domains without ownership drift

### Candidate first exit case

One football conclusion is time-valid, reproducible, limitation-aware,
useful without a wager, and consumable by prediction or scenario domains
without either domain redefining it.

### Guardrails

- Description is not prediction.
- Correlation is not automatically causal.
- Opponent adjustment must preserve method identity and evidence cutoff.
- Later corrections must not rewrite the original football claim.
- Betting relevance is optional, not required.

## Horizon 6 — Scenario intelligence

### Objective

Represent conditional questions explicitly without confusing assumptions
with observed evidence or forecasts of which scenario will occur.

### Candidate capabilities

- explicit scenario assumptions;
- base case and alternate cases;
- shared evidence cutoff;
- conditional prediction distributions;
- sensitivity to personnel, role, weather, pace, game script, or tactical
  assumptions;
- observed, assumed, and estimated values kept separate;
- scenario comparison;
- explanation of affected claims;
- decision consequences;
- scenario-aware invalidation conditions;
- no claim that the scenario itself will occur unless separately estimated.

### Candidate first exit case

One game supports a base case and one alternate scenario whose assumptions,
affected claims, uncertainty, and decision consequences are explicit and
reproducible.

### Guardrails

An injury report is an observation. “What happens if the player is
inactive?” is a counterfactual analytical operation. Those concerns may
interact, but they must not share ownership accidentally.

## Horizon 7 — Portfolio intelligence

### Objective

Deepen the proven separation between individual recommendation eligibility
and portfolio allocation when concrete portfolio needs justify it.

### Candidate capability sequence

1. Recorded open exposure
2. Game-level concentration
3. Team and player concentration
4. Common-factor declarations
5. Explicit correlation assumptions
6. Incremental exposure
7. Allocation-policy comparison
8. Drawdown scenarios
9. Risk-of-ruin analysis only if defensibly modeled
10. Portfolio outcome and policy evaluation

### Candidate first exit case

A set of individually eligible recommendations receives allocations or
explained zero allocations based on explicit, reproducible shared-risk and
concentration evidence.

### Guardrails

- Do not imply a sophisticated correlation model before it exists.
- Begin with explicit assumptions that are visible and challengeable.
- Keep recommended, allocated, attempted, executed, rejected, and settled
  statuses separate.
- Allocation is for the primary user's recorded portfolio context.
- Do not cross the constitutional execution boundary.

## Horizon 8 — Outcome, audit, and learning

### Objective

Broaden evaluation while preserving the separation between distinct
questions and preventing cherry-picking or result-oriented reasoning.

### Candidate evaluation domains

- football explanation quality;
- feature stability;
- prediction accuracy;
- probabilistic scoring;
- uncertainty coverage;
- calibration;
- market movement;
- recommendation-policy compliance;
- allocation-policy compliance;
- execution audit;
- settlement correctness;
- realized return;
- portfolio outcome;
- decision quality;
- repeated defect or limitation discovery.

### Candidate first exit case

One complete eligible universe is evaluated across prediction, market,
recommendation, allocation, execution, settlement, and decision-quality
dimensions without collapsing them into win/loss or ROI.

### Learning-loop rule

Research findings propose new method or policy versions. They do not
retroactively alter the original historical artifacts.

## Horizon 9 — Product surfaces

### Objective

Compose proven domain capabilities into coherent user experiences without
allowing navigation or pages to define architecture.

### Candidate surfaces

- Today
- Game
- Teams and Players
- Models and Research
- Markets and Portfolio
- Track Record and Evidence

These are candidate groupings, not locked navigation.

### Every consequential surface should answer

- What is being claimed?
- What kind of artifact is it?
- What is not known?
- Why was the conclusion reached?
- What uncertainty or limitation applies?
- What evidence could change it?
- Is the view historical, latest-corrected, or a difference view?
- Is the displayed quote current and usable?
- Did an analytical edge become a recommendation?
- Did a recommendation receive allocation?
- Was anything actually executed?
- What happened afterward?
- How can the output be reproduced?

### Guardrails

- Extreme transparency does not mean maximum information density.
- Users should not be forced through all five inspection levels.
- Nothing consequential may be hidden.
- Pages compose domain outputs and do not own shared truth.

## Horizon 10 — Fantasy and additional decision lenses

### Objective

Apply shared football and player outcome distributions to roster decisions
without duplicating prediction ownership.

### Candidate capability sequence

1. Player outcome distributions
2. League and scoring context
3. Replacement alternatives
4. Floor, median, and ceiling
5. Start and sit decisions
6. Waiver implications
7. Trade implications
8. Rest-of-season implications
9. Correlated lineup decisions
10. Outcome and decision-quality evaluation

### Candidate first exit case

One player distribution supports a transparent roster decision against a
specific alternative and league context, including uncertainty, opportunity
cost, assumptions, and later evaluation.

### Guardrails

- Season-long fantasy is a roster and opportunity-cost problem.
- Player prop betting is a price and probability problem.
- DFS is a constrained optimization and portfolio problem.
- Shared player predictions may feed all three, but their decision policies
  and responsibilities remain separate.
- DFS remains deferred until explicitly activated.

## Cross-Horizon Assurance Backlog

The following are not optional lenses. They are platform-wide assurance
concerns that should be strengthened through the workstreams that create
concrete demand for them.

### Provenance

Every consequential artifact identifies its source evidence, method,
creator or producer, schema, relevant timestamps, and exact upstream
versions.

### Point-in-time validity

Every conclusion declares its evidence cutoff and rejects information not
visible within it.

### Data quality

Malformed, incomplete, duplicated, conflicting, stale, or unsupported data
produces explicit domain outcomes and never silently enters consequential
claims.

### Uncertainty integrity

Estimated claims preserve appropriate uncertainty, support, scenario, or
limitation evidence through downstream consumption.

### Reproducibility

Historical claims reproduce from pinned evidence and method identity. A
failure to reproduce is a first-class assurance finding.

### Recommendation audit

Recommendation and abstention states agree with exact policy, quote,
prediction, eligibility, freshness, and invalidation evidence.

### Portfolio audit

Allocation state, amount, reason, and portfolio evidence remain mutually
consistent and independent of recommendation eligibility.

### Execution audit

A recommendation or allocation is never used as evidence that a wager was
placed. Intended, attempted, executed, rejected, canceled, voided, and
settled states remain distinct.

### Outcome and decision-quality evaluation

Outcomes evaluate prior claims but do not determine whether the original
process was valid.

### Responsible use

Consequential output language and presentation preserve uncertainty,
losses, abstention, stale or unavailable prices, and the separation between
decision support and execution.

## Technical and Product Debt Admission

Not every cleanup item deserves its own workstream. Debt may enter an
active unit only when:

- the unit naturally touches the owner;
- the debt creates real correctness, comprehension, maintenance, or proof
  risk;
- the change stays bounded;
- acceptance remains attributable;
- the change does not conceal an architectural decision;
- the additional diff can be reviewed coherently.

Otherwise, record the debt in the owning findings or backlog artifact with
its evidence and trigger condition.

Do not preserve temporary workstream, phase, or unit naming after the work
closes. Production names, tests, comments, and schemas should describe
lasting domain meaning.

## Workstream Closure Standard

A completed workstream must record:

### Completed

The lasting capability that now exists.

### Goal

The workstream goal and exit criterion.

### Files Added/Removed/Changed

Every durable owner, proof artifact, documentation owner, and retired
artifact.

### Tests

Focused, unit, integration, end-to-end, frontend, generated-artifact, and
full quality gates actually executed.

### Acceptance

Every exit criterion with direct evidence.

### Deferred

Every related capability not proven, with the reason it remains open and
the evidence that would reactivate it.

### Decisions

New, amended, superseded, or explicitly unchanged decisions.

### Evidence

The canonical proof artifact, exact identities or paths where applicable,
and reproduction command.

### Next dependency

What later work may safely assume and what it must not assume.

Implementation-unit closeouts continue to use the shorter required form:

```text
Completed
Goal
Files Added/Removed/Changed
Tests
Acceptance
```

## Thread and Context-Switch Standard

Every new thread begins with a context stamp:

```text
Inspected code commit:
Working tree: clean / intentionally dirty
Root HANDOFF revision:
Workstream HANDOFF revision:
FINDINGS revision:
Active PLAN unit:
Scope of this exchange:
Files changed since workstream handoff:
Tests run:
Decisions added:
```

The opening prompt must:

- state that repository files are canonical memory;
- provide a reading order;
- declare whether the session is inspection, planning, implementation,
  review, or amendment;
- state the exact active boundary or plan unit;
- carry forward locked and deferred decisions;
- require orientation verification before real work;
- prohibit reopening settled scope without new evidence.

A switch is complete only after the new thread reconstructs:

- the current workstream goal;
- the active boundary or unit;
- completed versus remaining work;
- locked constraints;
- deferred capabilities;
- the inspected commit and tree state;
- the next minimum repository evidence needed.

## Recommended Workstream 4 Inspection Shape

The next formal workstream is Reproduction, Supersession, and
Invalidation.

Its goal is to prove that later information updates current understanding
without corrupting historical truth.

Its exit criterion is one complete historical case viewable:

```text
as known at the original cutoff
as the latest corrected record
as an explicit difference between the two
```

The recommended inspection boundaries are:

### Boundary 1 — Inventory and ownership

Locate owners of source-version history, point-in-time retrieval,
supersession identity, selected-current state, corrected-record state,
evidence references, lifecycle outcomes, lineage, downstream relationships,
realized outcome, evaluation, API presentation, frontend composition, tests,
and documentation claims.

### Boundary 2 — Temporal semantics and identities

Inspect effective time, system-known time, source-published time, evidence
cutoff, supersession time, current-selection semantics, corrected-record
semantics, and identity stability across recomputation.

### Boundary 3 — Historical reproduction

Follow one completed vertical-slice case from source observation through
prediction, issuance, policy, recommendation, allocation, and evaluation.
Determine what reproduces from persisted artifacts and what still relies on
controlled reconstruction.

### Boundary 4 — Supersession and invalidation behavior

Determine which artifact owners need which outcomes:

```text
unchanged
expired
challenged
superseded
invalidated
recomputation_required
recomputed
unavailable
conflicting
not_applicable
```

Do not require every domain to support every outcome.

### Boundary 5 — Latest-corrected view

Identify the owner that can answer what the best current historical account
is, which later evidence it includes, which original artifacts remain
historical, which methods are rerun, and whether correction applies to
source facts, interpretation, or both.

### Boundary 6 — Difference view and downstream impact

Determine whether the system can explain exact old and new evidence,
changed fields, changed and unchanged claims, recommendation eligibility,
allocation, evaluation interpretation, and the reason for each difference.

This boundary is the primary evidence gate for reconsidering general
forward-impact discoverability.

### Boundary 7 — Evaluation and presentation

Inspect API and frontend behavior for historical versus current labels,
corrected-record labels, difference explanations, stale or invalid warnings,
original decision preservation, outcome separation, and progressive
disclosure.

### Boundary 8 — Consolidation and planning

Produce the reuse map, confirmed missing capabilities, deferred-decision
disposition, decision proposals, dependency-ordered units, one active
`PLAN.md` unit, one fixed proof case, and the exit-matrix design.

## Initial Workstream 4 Guardrails

- Workstream 3's later-spread-quote proof is evidence, not automatic proof
  of the full Workstream 4 exit criterion.
- Do not rewrite the completed Workstream 3 historical artifacts.
- Do not assume “latest” means file modification time.
- Do not use current selection as a substitute for latest-corrected
  semantics without inspecting its contract.
- Do not describe every changed output as invalidated.
- Do not conflate supersession, challenge, expiry, invalidation, and
  recomputation.
- Do not build a general impact index without a concrete downstream
  discovery consumer.
- Do not create a universal physical analytical-claim artifact solely to
  support the workstream.
- Do not let the frontend infer differences by independently comparing raw
  payloads if domain semantics are required.
- Do not treat realized outcome as corrected pregame evidence.

## Initial Workstream 4 Artifacts

Recommended directory:

```text
docs/workstreams/reproduction_and_invalidation/
```

Recommended starting artifacts:

```text
FINDINGS.md
HANDOFF.md
ROADMAP.md
WORKSTREAM_KICKOFF_PROMPT.md
```

The initial `FINDINGS.md` should contain only the workstream header,
snapshot, evidence labels, and open inventory boundary. It should not copy
prior findings or pre-classify owners.

## Decision and Amendment Triggers

Use an explicit decision or amendment process when work would:

- change the primary or secondary audience;
- cross the current execution boundary;
- alter the six locked invariants;
- redefine the epistemic vocabulary;
- weaken point-in-time correctness;
- change the transparency promise;
- make P&L or engagement a defining success metric;
- introduce user-specific bankroll or risk support for other people;
- require publication of protected, licensed, private, or security-sensitive
  evidence;
- reopen a recorded deferred decision;
- create a universal cross-domain owner with irreversible migration impact.

An amendment requires new evidence, rationale, consequences, rejected
alternatives, adversarial review, explicit canonical-file change, and
propagation to every dependent handoff or plan.

## Program Health Checks

Periodically verify:

### Authority health

- Canonical files do not conflict.
- Current state is absent from this guide.
- Deferred choices have not become accidental implementation assumptions.
- Idea archives have not become shadow authority.

### Repository health

- Generated runtime artifacts are not used as clean-checkout proof unless
  intentionally tracked.
- Stale caches and generated files are not masking import or schema changes.
- No retired schema or model identity is retained solely for nonexistent
  production compatibility.
- Test ownership follows production ownership.

### Temporal health

- Cutoffs are explicit and UTC.
- Later evidence cannot leak into historical evaluation.
- Current selections are versioned artifacts rather than implicit file
  recency.
- Corrected records do not overwrite historical records.

### Decision health

- Edges remain separate from recommendations.
- Recommendations remain separate from allocations.
- Allocations remain separate from executions.
- Zero allocations carry machine-readable reasons.
- Missing evidence remains distinct from completed policy rejection.

### Evaluation health

- Complete eligible universes are preserved.
- Wins and losses remain visible.
- Prediction quality, market performance, profitability, portfolio outcome,
  and decision quality remain separate.
- Favorable outcomes do not erase process defects.

### Documentation health

- `PLAN.md` contains exactly one active unit or an explicit no-active-unit
  state during inspection.
- `HANDOFF.md` reflects actual repository state.
- `ROADMAP.md` reflects completed and remaining scope.
- Temporary scripts and chronology terms have not become durable artifacts.
- A clean new thread can rehydrate without chat history.

## Long-Term Success Standard

The program succeeds when it earns trust across five dimensions.

### Integrity

Consequential historical outputs reproduce from the evidence and methods
available at the time. Superseded evidence remains preserved. Integrity
failures are visible.

### Epistemic honesty

Estimated claims carry uncertainty or limitation. Missing, conflicting,
stale, unstable, and out-of-support conditions remain explicit.

### Decision discipline

Edges do not automatically become recommendations. Recommendations do not
automatically become allocations. Allocations do not imply execution.

### Evaluative honesty

The complete eligible universe is retained. Original claims are evaluated
without retrospective rewriting. Financial outcomes are reported but do
not independently define quality.

### Comprehension and utility

The intended user can determine:

- what is claimed;
- what is not known;
- why the system reached the conclusion;
- what evidence could change it;
- whether a price remains usable;
- why an edge did or did not become a recommendation;
- why a recommendation did or did not receive allocation;
- what was actually executed;
- how the original conclusion performed;
- how to inspect and reproduce the evidence chain.

## Final Program Rule

Build the smallest truthful capability that advances the locked product
vision, prove it through its real owners, preserve its historical evidence,
and generalize only when a concrete second need demonstrates what is truly
shared.

The repository is the memory. The evidence is the authority. The proof is
the progress.
