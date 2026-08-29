# AI_BOOTSTRAP.md — How to work on Gridiron Edge in a fresh thread

> **Procedural only. Contains no project state.** State lives in HANDOFF.md,
> DECISIONS.md, PLAN.md, and workstream FINDINGS/HANDOFF files. This file changes
> rarely (method); those change often (state). Never copy current findings, active
> unit names, or decisions into this file.
> For this workstream, governance files are co-located in this folder, not repo root.

## First principle
**The conversation is not the memory. The repository is.** A new thread recovers
full context from versioned artifacts, not from any prior chat transcript. If it
isn't in a committed file, it doesn't exist.

## Governance authority (higher wins on conflict)
These documents **govern actions**:
1. `CONSTITUTION.md` — purpose, audience, boundaries, non-goals
2. `VISION.md` — architecture, the six invariants
3. `ROADMAP.md` — program scope, workstreams, proof progression
4. `DECISIONS.md` — durable cross-workstream irreversible decisions
5. Root `PLAN.md` — exactly ONE active implementation unit

## Supporting context
These documents **explain or rehydrate state** (they do not outrank the active
execution contract):
6. Root `HANDOFF.md` — repository-wide execution state
7. Active workstream `HANDOFF.md` — compressed workstream state
8. Active workstream `FINDINGS.md` — full evidence log

## Required reading order for a new thread
1. This `AI_BOOTSTRAP.md` (method)
2. Root `HANDOFF.md` (where the repo stands now)
3. Active workstream `HANDOFF.md` (compressed workstream state)
4. Root `PLAN.md` (the one active unit)
5. `DECISIONS.md` (locked constraints)
6. Workstream `FINDINGS.md` — **only** when evidence-level detail is required

CONSTITUTION.md and VISION.md are the constitutional core. Their binding consequences are distilled into DECISIONS and PLAN guardrails, so routine implementation threads do not load them. Read them in full for any inspection (Switch B), amendment (Switch F), or whenever a work item would touch an invariant or violate a PLAN guardrail — at which point it is an amendment, not an implementation choice.

*(Authority order and reading order are intentionally different: you read summaries
first to orient, but the active `PLAN.md` and `DECISIONS.md` govern over any
summary.)* Progressive disclosure: take the conclusion + authority first; drill into
evidence only when challenging a conclusion or implementing against it.

## Repository conventions
- **One active unit.** Root `PLAN.md` holds exactly one implementation unit at a
  time. Future units live in `ROADMAP.md`, never in `PLAN.md`.
- **Per-workstream folder:** `docs/workstreams/<name>/{FINDINGS.md, HANDOFF.md}`.
  `FINDINGS.md` = full evidence log ("why we believe this"); `HANDOFF.md` =
  compressed rehydration ("what a new thread must know"). The handoff links finding
  IDs; it does not restate their evidence.
- **PLAN closure convention:** update root `PLAN.md` **directly** to its completed
  form in the **same commit** as the implementation, using the established closure
  headings — **Completed · Goal · Files Added/Removed/Changed · Tests · Acceptance**.
  Do not "fold" the plan into findings; findings are not the execution authority.
- **Confirm repository state before acting.** Read the commit SHA and working-tree
  status first. Never reason from a stale artifact.
- **Real repository paths only** in plans/findings (e.g.
  `src/gridiron_edge/ingest/odds/store.py`). SharePoint `.txt` suffixes are mirror
  transport names, never repository paths.

## Evidence labels (use consistently; do not over-claim)
`VERIFIED_LOCAL_SOURCE` · `VERIFIED_LOCAL_TEST` · `VERIFIED_REAL_ARTIFACT` ·
`REVIEWED_FULL_ATTACHED_SOURCE` (full drag-and-drop attachment; not mirror, not
executed) · `REVIEWED_FULL_MIRROR` · `SUPPORTED_BY_MIRROR_SNIPPET` ·
`SUPPORTED_BY_FILENAME_ONLY` · `INDEXING_INCOMPLETE` ·
`LOCAL_VERIFICATION_REQUIRED` · `NOT_INSPECTED`
- Mirror reads are **not** verification. Reading a test is **not** running it —
  only a successful local run earns `VERIFIED_LOCAL_TEST`.
- The SharePoint mirror mangles some content (e.g. `int | None` → `int  None`,
  truncated docstrings); logic-level claims from the mirror are capped at
  `SUPPORTED_BY_MIRROR_SNIPPET` and flagged `LOCAL_VERIFICATION_REQUIRED`.

## Two-model workflow
- **One author, one adversarial reviewer, one canonical artifact.** Claude and
  Microsoft 365 Copilot (ChatGPT-side) alternate as author/reviewer; the shared
  file — not either chat log — is authoritative. Roles stay fixed within a bounded
  unit; whichever model has better byte-fidelity access to the relevant source
  leads that unit.
- **Reviewer response headers:** Accepted findings / Accepted with modification /
  Rejected findings / Insufficient evidence / Missing inspection targets /
  Classification changes / Local verification requests / Scope-control findings /
  Boundary disposition.
- **Version stamp on every cross-thread handoff** (see below). If a model is
  reading a different revision, stop and resynchronize — do not reason from the
  stale copy.
- **No re-litigation.** Closed decisions are not reopened without new repository
  evidence or an explicit amendment. Dialectic-for-its-own-sake is a failure mode.

## Context stamp (paste at the start of every exchange)
```
Inspected code commit:
Working tree:            clean / intentionally dirty
Root HANDOFF revision:
Workstream HANDOFF revision:
FINDINGS revision:
Active PLAN unit:
Scope of this exchange:
```
For implementation activity, add:
```
Files changed since workstream handoff:
Tests run:
Decisions added:
```

## Amendments
The constitutional core (CONSTITUTION invariants, VISION's six invariants, the
epistemic vocabulary, the transparency definition, the "success ≠ P&L" stance)
changes **only** by explicit amendment with recorded rationale, consequences, and
rejected alternatives — never incidentally inside an implementation unit.
