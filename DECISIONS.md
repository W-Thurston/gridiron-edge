# Architectural Decisions

Append-only log of architectural decisions made during development.
Each entry documents *why* a choice was made, not just *what* changed
(the latter belongs in `CHANGELOG.md`).

Format: newest entry at top. Each entry self-contained.

---

## D38 - Recurring season refresh preserves unrequested historical games

**Date:** 2026-09-18

### Decision

Recurring, non-all-years nflverse game refreshes preserve every unrequested
season in the registered raw games artifact and replace only the explicitly
requested season.

`fetch_nflverse_games()` remains the explicit replacement boundary for callers
that deliberately require a complete replacement or all-years fetch.
`refresh_nflverse_game_seasons()` owns preservation and replacement of selected
season slices. Shared CLI orchestration must route recurring refreshes through
that preservation-aware boundary rather than reproduce season-merging behavior.

`clean_nflverse_games()` continues to rebuild the canonical cleaned games
artifact from the complete registered raw artifact. It does not independently
merge current output with prior cleaned history.

### Context

A read-only moneyline prediction inspection completed on September 18, 2026
confirmed that the explicit-season `weekly-predict` refresh path called
`fetch_nflverse_games(seasons=[2026])`. That replacement function overwrote the
raw games artifact with the requested season. `clean_nflverse_games()` then
produced a nonempty cleaned artifact containing only the 16 completed 2026
Week 1 games.

The downstream Elo path rebuilt from that partial cleaned history and produced
a complete but historically reset Week 2 state. Every team began at the 1500
initial rating; Week 1 winners became 1510 and losers became 1490. Both
immutable 2026 Week 2 Win forecast runs persisted exactly that two-value Elo
state across all 16 games.

The repository already had a preservation-aware refresh implementation and
focused tests for it. The defect existed at the shared CLI composition seam,
where an explicit season selected the replacement function rather than the
recurring refresh function.

### Consequences

- `all_years=True` continues to select an explicit full replacement.
- A recurring refresh with an explicit season refreshes that season while
  retaining every other raw season.
- A recurring refresh without an explicit season delegates current-season
  resolution to the preservation-aware refresh wrapper.
- The shared pipeline owns refresh-mode selection.
- The nflverse ingest boundary owns selected-season replacement and historical
  preservation.
- The cleaner remains a deterministic transformation of the complete raw
  artifact rather than becoming a second historical merge owner.
- Composition-level regression tests must prove which fetch boundary the shared
  pipeline selects.
- Historical continuity must be validated through explicit season and game
  identities, not arbitrary row-count or rating-distribution thresholds.
- This decision does not define the future Elo reconstruction contract. That
  remains a separate corrective unit.
- An empty selected-season response is not a valid replacement artifact. It
  raises before any write, preserving the existing raw artifact unchanged.
  Upstream fetch exceptions likewise propagate without modifying the artifact.
- Shared orchestration distinguishes an explicitly supplied season with
  `season is not None`; workflow selection is not based on season-value
  truthiness.

### Alternatives considered and rejected

1. **Merge prior cleaned history inside `clean_nflverse_games()`.**
   Rejected because it would create a second independent season-merging
   implementation and allow the registered raw artifact to remain incomplete.

2. **Make every use of `fetch_nflverse_games(seasons=[...])` preserve history.**
   Rejected because the function has a valid explicit-replacement contract.
   Changing that contract would blur the distinction between replacement and
   recurring refresh behavior.

3. **Detect the defect from suspicious Elo distributions.**
   Rejected because numeric variance and rating-shape checks are heuristic.
   The durable invariant is preservation of explicit historical season and game
   identities.

4. **Repair the Elo update in the same implementation unit.**
   Rejected because games-history preservation and Elo reconstruction are
   separate ownership boundaries. Combining them would obscure which correction
   prevents the destructive refresh and make regression evidence less precise.

### References

- `src/gridiron_edge/cli/main.py`
- `src/gridiron_edge/ingest/nflverse/games.py`
- `src/gridiron_edge/transform/clean/games_nflverse.py`
- `tests/unit/cli/test_main.py`
- `tests/unit/cli/test_weekly_predict.py`
- `tests/unit/ingest/nflverse/test_games.py`
- `ROADMAP.md`
- `PLAN.md`
- `gridiron_edge_moneyline_inspection_v2.md`

---
### D37. Attribution operations are formally named and separated by authentication strength; `_closeout_matches` is corrected to re-derive the canonical digest

**Status:** Accepted
**Date:** 2026-08-26

#### Decision

Seven reference-attribution operations are formally named, confirmed from
full current source, and classified into two families that must not be
conflated:

**Canonical authentication** (re-derives or matches an exact,
digest-backed identity; a 1:1 relationship between one reference and one
exact upstream artifact):
1. **Canonical reference production** — `market_closeout.py::_candidate_reference`.
   Adapts one `CandidateIssuanceRow` into a `MarketCloseoutReference`,
   computing `reference_id` via `candidate_issuance_row_id`. Unconditional;
   cannot fail.
2. **Issuance-row resolution** — `recommendation_policy.py::_resolve_candidate`.
   Re-derives `candidate_issuance_row_id` for each row in an issuance,
   requires exactly one match against a supplied reference.
3. **Persisted-result self-validation** — `recommended_bet_result.py::validate_recommended_bet_result`
   (via `_resolve_row`). Re-derives the candidate reference from a
   *persisted* result's own embedded fields, dispatched through the
   recorded derivation version (Unit 4, D31).
4. **Structural closeout attribution** — `market_family_evaluation.py::_closeout_matches`.
   **Corrected in this unit** (see below).
5. **Recorded-wager reference authentication** — `bet_reference_matching.py::match_bet_references`.
   Newly formally named in this unit (previously uncounted among the six
   operations Boundary 4 identified). Validates a recorded wager's
   self-reported reference terms against real observed quote history by
   exact field-by-field equality (provider, provider_event_id, sportsbook,
   game_id, market, side, fetched_at, all required and exact), producing a
   five-state diagnostic (`MATCHED`/`MANUAL_BET`/`OBSERVATION_NOT_FOUND`/
   `AMBIGUOUS_OBSERVATION`/`REFERENCE_TERMS_CONFLICT`). Distinct from
   operation 1: it is not an unconditional adapter — it has genuine failure
   modes because it is validating a *claim* (what the bettor's own ledger
   row says the reference was) against *ground truth* (actual quote
   history), not merely transcribing already-known fields. Its own
   `reference_id` (`bet_id`, a ledger UUID) carries no digest, so it has no
   analogous suffix-authentication gap to close.

**Structural attribution** (matches materialized evidence to a *group* or
*aggregate* concept; not a 1:1 exact-reference relationship, and
correctly has no digest to check):
6. **Structural history-group attribution** — `market_family_evaluation.py::_history_matches`.
   Matches a `QuoteHistoryBoundary` (a descriptive statistical summary of
   observation depth/count for one market-side) to a candidate row via 6
   identity fields. **Confirmed, from full current source, to have no
   `reference_id` or digest field on `QuoteHistoryBoundary` at all** — this
   is not an omission or a narrower version of operation 4; it answers a
   structurally different question ("which group of historical
   observations describes this market-side's quote depth" vs. "which exact
   persisted result belongs to this exact row"). No digest-authentication
   mechanism applies here, and none should be added by analogy to
   operation 4's fix.
7. **Structural wager attribution** — `market_family_evaluation.py::_wager_return_for_row`.
   Attributes settled-wager return evidence (stake, PnL, realized return)
   to a candidate row via exact materialized-field matching against ledger
   columns (provider, provider_event_id, sportsbook, game_id, market_type,
   side, and all four immutable reference timestamp/price/line fields).
   Like operation 6, this is group/aggregate-style attribution over
   ledger rows, not a digest-backed 1:1 reference resolution.

**The `_closeout_matches` correction:** the function previously checked an
issuance-ID **prefix** (`reference_id.startswith(f"{issuance_id}:")`) plus
11 individually-compared materialized fields — every field the
`candidate_issuance_row_id` v1 hash payload covers, checked redundantly,
without ever re-deriving and comparing the digest **suffix** itself. This
meant a reference whose suffix was not the row's true digest, but whose 11
individual fields happened to equal the row's fields, would incorrectly be
attributed. The function now re-derives the canonical reference directly
(`reference.reference_id == candidate_issuance_row_id(issuance.issuance_id, row)`)
and separately checks `reference_kickoff`, the one field
`candidate_issuance_row_id`'s payload does not cover — kept as an explicit,
separate check, not folded into the digest or silently dropped.

#### Context

Boundary 4 (Workstream 2 inspection) confirmed six reference-attribution
operations and flagged operation 4's suffix-matching gap as a real
integrity ambiguity, not yet fixed. This unit's own source re-reading
(`market_closeout.py`, `bet_reference_matching.py`,
`market_family_evaluation.py`, and their focused tests, all read in full
this session) confirmed Boundary 4's finding exactly and surfaced a
seventh operation (`match_bet_references`) that Boundary 4's original
six-operation count did not include.

The test suite (`test_market_family_evaluation.py`) independently confirmed
the defect was live: its own `_closeout` fixture helper constructed every
closeout reference with a deliberately fake, non-digest suffix
(`f"{issuance.issuance_id}:test-{row.market}-{row.side}"`), and every
existing test using that helper passed — proof the function's
field-by-field checks alone were sufficient to satisfy every existing test,
independent of whether the digest suffix was genuine.

#### Alternatives considered and rejected
- **Option B (attribution-only, remove the suffix check entirely, rename
  the function to disclaim authentication).** Rejected: operation 4's
  actual callers (`_evaluate_row`) treat its result as establishing a 1:1
  relationship between one exact candidate row and one exact closeout
  result — removing the authentication guarantee here would weaken an
  existing real guarantee, not merely relabel it.
- **Option C (expose both a structural-match result and a digest-
  authenticated result separately, let the caller choose).** Rejected as
  unnecessary complexity: nothing in `_evaluate_row` or any other caller
  needs the weaker, structural-only guarantee — Option A's stronger
  guarantee is strictly better for every current consumer, with no
  observed need for the weaker variant to remain available.
- **Adding a digest/reference_id field to `QuoteHistoryBoundary` so
  `_history_matches` could adopt the same fix.** Rejected: `_history_matches`
  answers a structurally different question (group/aggregate attribution,
  not exact 1:1 reference resolution); it has no analogous gap, and adding
  a field to manufacture one would be scope creep with no defect to
  justify it.

#### Consequences
- `market_family_evaluation.py::_closeout_matches` now imports
  `candidate_issuance_row_id` from `candidate_issuance.py` and re-derives
  the canonical reference directly, rather than checking a prefix plus
  redundant individual fields.
- `test_market_family_evaluation.py`'s `_closeout` fixture helper now
  constructs genuine digest-backed references
  (`candidate_issuance_row_id(issuance.issuance_id, row)`), rather than a
  fake test suffix. All pre-existing tests using this helper continue to
  pass unchanged, since none of them asserted on the suffix's content —
  only on downstream evaluation behavior.
- One new test
  (`test_closeout_with_mismatched_digest_but_matching_fields_does_not_match`)
  proves the specific defect is closed: a forged reference with the
  correct materialized fields but an incorrect digest suffix is no longer
  attributed.
- `bet_reference_matching.py::match_bet_references` is formally named as
  the seventh attribution operation; no code changes were required —
  it already has no suffix/digest gap, since its identity
  (`bet_id`) is a ledger UUID, not a content digest.
- `market_family_evaluation.py::_history_matches` and `_wager_return_for_row`
  remain unchanged; their scope is confirmed correct as group/aggregate
  attribution, not exact 1:1 reference authentication.

#### Revisit triggers
- A future consumer of `_closeout_matches`'s result needs the weaker
  structural-only guarantee Option C would have provided (no such need
  observed today).
- `QuoteHistoryBoundary` or ledger wager rows ever gain their own
  digest-backed identity, at which point `_history_matches`/
  `_wager_return_for_row` should be re-evaluated against this same
  authentication-vs-attribution distinction.

#### References
- `src/gridiron_edge/market/market_closeout.py`
- `src/gridiron_edge/market/bet_reference_matching.py`
- `src/gridiron_edge/market/market_family_evaluation.py`
- `src/gridiron_edge/market/recommendation_policy.py`
- `src/gridiron_edge/market/recommended_bet_result.py`
- `tests/unit/market/test_market_family_evaluation.py`
- `docs/workstreams/analytical_claims/CLAIM_CAPABILITY_PROTOCOL.md`
  (Capability 8 — attribution, deferred to this unit by Unit 5)
- `docs/workstreams/analytical_claims/FINDINGS.md` (Boundary 4)
- `DECISIONS.md` D31 (candidate-reference derivation versioning — the
  mechanism operation 3 dispatches through)

### D36. Forward-impact discoverability is required but presently unimplemented

**Status:** Accepted
**Date:** 2026-08-26

#### Decision

The architecture requires a way to discover or assess downstream consumers
of a durable claim. Workstream 2's inspection found no current mechanism
satisfying this requirement anywhere in the codebase, and this decision
does not select one without a concrete use case. `production_chain_preflight.py`
remains a readiness auditor; it is explicitly not repositioned as this
capability's future implementation owner. A future mechanism, if and when
built, requires its own separate owner and its own decision.

#### Context

Boundary 3's original inspection found `production_chain_preflight.py`
resolves evidence *backward* (given a scope, find matching upstream
artifacts) and does not answer "given this exact reference, what
downstream artifacts consumed it." This unit's own full re-read of
`production_chain_preflight.py::_exact_candidate_issuance` confirmed this
directly: it scans persisted issuance files
(`directory.glob("*.json")`) and matches on season/week/product/run — the
same backward-resolution family as canonical reference authentication
(`_resolve_candidate`, `_resolve_row`), not a forward index.
`DECISIONS.md` D27 independently confirms this is intentional: preflight
is "the composite chronological audit artifact," explicitly scoped to
current-evidence readiness, not relationship discovery.

No concrete, present pain point in this codebase requires answering "what
consumed this reference" — the original seed incident (the motivation for
this entire workstream) required the opposite query ("what does this
reference depend on"), which backward lineage (Capability 5, strongly
satisfied) already answers.

#### Alternatives considered and rejected
- **Extend `production_chain_preflight.py` into a forward-index owner.**
  Rejected: preflight is scope- and chronology-oriented; extending it
  risks conflating audit with relationship discovery, a distinct
  responsibility.
- **Build a reverse index, relationship manifest, or query service now,
  speculatively.** Rejected: no concrete use case demonstrates the need;
  building infrastructure ahead of evidence repeats a mistake this
  workstream's own review process has caught and corrected twice already
  (Units 2 and 3's initial drafts).
- **Declare the capability satisfied by documentation alone.** Rejected:
  the program-level exit criterion requires traceability to be followed
  downstream in fact, not merely acknowledged as a gap. This decision
  states plainly that the capability is unimplemented and that
  Workstream 2's own downstream-traceability exit proof remains open.

#### Consequences
- No production code changes result from this decision.
- Every durable claim in `CLAIM_CONFORMANCE_MATRIX.md` is marked
  **Absent** for this capability — an honest, not hidden, gap.
- **Workstream 2's exit criterion (traceability followed both upstream and
  downstream) is not fully met by Unit 5's closure.** This must be carried
  forward explicitly in `HANDOFF.md` and ROADMAP.md, not silently
  implied as resolved.
- Five candidate mechanisms are named for whichever future unit implements
  this capability, none selected: embedded downstream references, a
  reverse index, a relationship manifest, a repository-scanning query
  service (reusing preflight's demonstrated technique without extending
  preflight's own role), or an explicit no-new-mechanism-at-current-scale
  deferral.

#### Revisit triggers
- A concrete use case demonstrates a real need to answer "what consumed
  this reference" (e.g. a future evaluation or audit unit requires it).
- Workstream scale or artifact count grows to a point where backward-only
  traceability becomes a demonstrated operational problem.

#### References
- `src/gridiron_edge/market/production_chain_preflight.py`
- `DECISIONS.md` D27 (production recommendation proof as an exact
  immutable evidence chain — the decision this finding is consistent with
  and does not contradict)
- `docs/workstreams/analytical_claims/FINDINGS.md` (Boundary 3)
- `docs/workstreams/analytical_claims/CLAIM_CAPABILITY_PROTOCOL.md`
  (Capability 10)
- `docs/workstreams/analytical_claims/HANDOFF.md` (must reflect this open
  exit-criterion item)

---

### D35. Validity and invalidation are required capabilities with artifact-owned mechanisms

**Status:** Accepted
**Date:** 2026-08-26

#### Decision

Durable claim specializations must state how readers distinguish
supported, internally consistent evidence from unsupported contract
semantics and from evidence corruption. Existing domain outcome/decision-
state enums (`RecommendedBetResultState`, `RecommendationDecisionState`,
`PolicyDerivationStatus`, etc.) are not replaced by a common lifecycle
enum. Unit 4's candidate-reference derivation-version dispatch
(`DECISIONS.md` D31) is one confirmed implementation precedent for one
field on one artifact — it is not established, by this decision, as a
universal mechanism other artifacts must adopt.

#### Context

Two independent artifacts — `RecommendationPolicyDecision`'s
`RecommendationDecisionState` and `RecommendedBetResult`'s
`RecommendedBetResultState` — were confirmed during this unit's review to
be structurally the same *kind* of thing: decision-outcome enums
answering "what did this evaluation conclude," not artifact-validity
lifecycles answering "is this persisted artifact still the authoritative
version of itself." This is a real, recurring gap, not an isolated
oversight on one artifact.

A hypothetical second test case for generalizing Unit 4's pattern — a
revised `RecommendationPolicyGovernance` producing a changed
`governance_fingerprint`, potentially invalidating prior policy
derivations the way the original seed incident invalidated candidate
references — was considered during this unit's pre-implementation review
and explicitly rejected as a confirmed analog. Full-source review of
`recommendation_governance.py` and `recommendation_policy.py` confirmed
both validate identity/fingerprint agreement against *currently* governed
content; no reader path exists that applies a *changed* derivation
algorithm to an *old* persisted fingerprint the way the candidate-
reference incident did. Generalizing Unit 4's specific mechanism from one
confirmed case plus one unconfirmed hypothetical would be premature.

#### Alternatives considered and rejected
- **Generalize Unit 4's version-dispatch pattern into a shared mechanism
  now**, using the governance-fingerprint scenario as a second proof case.
  Rejected: the scenario is not a confirmed instance; generalizing from
  one real case is not justified.
- **Introduce a common lifecycle enum (current/superseded/invalidated/
  corrupt) replacing existing domain states.** Rejected: existing domain
  states (decision outcomes, evidence availability) answer different
  questions than artifact validity; replacing them would lose real
  information, not add a missing capability.
- **Leave validity/invalidation entirely unaddressed as a capability.**
  Rejected: the program-level exit criterion explicitly requires an
  invalidation contract; documenting it as a required-but-mechanism-
  deferred capability is the honest middle path.

#### Consequences
- No production code changes result from this decision.
- `RecommendationPolicy` and `RecommendationGovernanceVersion` are marked
  **Absent** for this capability in the conformance matrix — an honest gap,
  not glossed over.
- A future unit implementing a second real validity/invalidation mechanism
  (for any artifact) does so against real evidence of need, choosing its
  own mechanism — not by assuming Unit 4's dispatch pattern is
  automatically correct for a different field or artifact.

#### Revisit triggers
- A real (not hypothetical) case of a method/governance/schema change
  needing to distinguish "superseded" from "corrupt" for an artifact other
  than the candidate reference.
- A future unit needs to query validity state across multiple artifact
  types, which would push toward a shared representation this decision
  currently declines to build.

#### References
- `src/gridiron_edge/market/recommendation_policy.py`
- `src/gridiron_edge/market/recommendation_governance.py`
- `src/gridiron_edge/market/recommended_bet_result.py`
- `DECISIONS.md` D31 (candidate-reference derivation versioning — the one
  confirmed precedent)
- `docs/workstreams/analytical_claims/CLAIM_CAPABILITY_PROTOCOL.md`
  (Capability 9)

---

### D34. Claim kind may be nominal or explicit

**Status:** Accepted
**Date:** 2026-08-26

#### Decision

Every durable claim specialization has a stable conceptual kind. Concrete
Python contract (type) identity is sufficient to represent that kind for
homogeneous internal paths. An explicit machine-readable discriminator
field is required only at heterogeneous storage, transport, or dispatch
boundaries — specifically when: heterogeneous claims share one envelope,
store, stream, or API collection; a consumer must dispatch without
importing concrete Python classes; forward relationships need typed
endpoints; or a common audit interface queries multiple claim categories.

#### Context

No current consumer in this codebase requires dispatching across claim
kinds without importing concrete types. Adding a `claim_kind` field to
every dataclass "for consistency" would be exactly the kind of blanket
production change this unit's own guardrails (and the pre-implementation
review) rejected.

#### Alternatives considered and rejected
- **Add an explicit `claim_kind` enum field to every durable claim
  dataclass now.** Rejected: no demonstrated consumer; would be a
  speculative field addition.
- **Rely solely on `isinstance`/type checks everywhere, forever.**
  Rejected as a blanket rule: this decision explicitly anticipates that a
  future heterogeneous boundary (a shared store, a common audit API) would
  need an explicit discriminator, and names the trigger conditions rather
  than ruling one out permanently.

#### Consequences
- No production code changes result from this decision.
- The conformance matrix (`CLAIM_CONFORMANCE_MATRIX.md`) serves as the
  documentation-level registry of currently-known claim kinds.
- Any future unit introducing a heterogeneous claim collection (a shared
  envelope, stream, or cross-kind query) must add an explicit
  discriminator at that point, per the trigger conditions above — this
  decision pre-authorizes that addition rather than requiring a fresh
  decision each time, provided the trigger condition is real and stated.

#### Revisit triggers
- Any of the four listed trigger conditions becomes concretely true for a
  real, implemented consumer (not a hypothetical one).

#### References
- `docs/workstreams/analytical_claims/CLAIM_CAPABILITY_PROTOCOL.md`
  (Capability 1)

---

### D33. Evidence usability remains domain-owned

**Status:** Accepted
**Date:** 2026-08-26

#### Decision

A claim or evidence representation must expose a machine-readable
usability state and, where the state alone is insufficient, a stable
reason explaining why it cannot be used for its declared purpose.
Domain-specific status enums and fields remain local to their owning
module. No shared uncertainty enum or Python `Protocol` is introduced
without a concrete cross-domain consumer demonstrating the need.

#### Context

Full-source review during Unit 5 confirmed three independently-evolved,
structurally distinct uncertainty/limitation shapes:
`EvaluationEvidenceStatus`/`MetricEstimate` (evidence availability,
`market_family_evaluation.py`), `PolicyCheckStatus`/`PolicyCheckResult`
(policy-check outcome, `recommendation_policy.py`, additionally carrying
`mandatory` and `required_value`), and
`EdgeResultState`/`EdgeDiagnosticBlocker` (analytical-result diagnostics,
`edge_diagnostics.py`). A fourth candidate, `CandidateOutcome`
(`market_family_evaluation.py`), was considered and rejected — it grades a
realized market-side result, not evidence usability, despite sharing
`UNAVAILABLE`/`CONFLICT`-shaped members.

These three shapes do not share a meaningfully common structural surface:
`PolicyCheckResult.reason` is required text while `MetricEstimate.reason`
is nullable; `MetricEstimate` carries `sample_size`/`value` fields
`PolicyCheckResult` has no equivalent for. A structural `Protocol` would
either be so weak as to add no value (requiring only `.status`) or would
force adapters/field additions that no present runtime consumer needs.

#### Alternatives considered and rejected
- **A shared uncertainty dataclass or enum.** Rejected: would either lose
  fields genuinely needed by one shape (e.g. `mandatory`) or force
  unrelated shapes to carry fields they don't use.
- **A Python `Protocol` for structural conformance.** Rejected: the three
  shapes' actual surfaces do not overlap enough to make a protocol useful;
  see Context above.
- **Treating `CandidateOutcome` as a fourth uncertainty shape.** Rejected:
  its primary semantic purpose is outcome grading, not usability
  reporting.

#### Consequences
- No production code changes. `PolicyCheckResult`, `MetricEstimate`,
  `EdgeDiagnostics`'s internal types, and `CandidateOutcome` remain
  exactly as they are.
- Any future artifact introducing a new usability/limitation
  representation should be checked against this decision's governing
  question ("can this be used for its declared purpose; if not, what
  machine-readable state and reason explain why") rather than against a
  shared type that does not exist.

#### Revisit triggers
- A concrete cross-domain consumer emerges that needs to inspect
  usability state across two or more of the three shapes without
  importing each shape's concrete type.

#### References
- `src/gridiron_edge/market/market_family_evaluation.py`
- `src/gridiron_edge/market/recommendation_policy.py`
- `src/gridiron_edge/market/edge_diagnostics.py`
- `docs/workstreams/analytical_claims/CLAIM_CAPABILITY_PROTOCOL.md`
  (Capability 7)

---

### D32. Analytical claims conform by capability profile, not inheritance

**Status:** Accepted
**Date:** 2026-08-26

#### Decision

Durable analytical claim specializations document conformance to a common
eleven-capability profile (`docs/workstreams/analytical_claims/CLAIM_CAPABILITY_PROTOCOL.md`).
They do not inherit from a universal base class and do not share one
physical persistence schema. Conformance is demonstrated by a per-artifact
mapping (`CLAIM_CONFORMANCE_MATRIX.md`), not by structural type identity.

#### Context

Workstream 2's inspection (Boundaries 1–8) found the analytical-evidence
substrate substantially reusable but lacking any common claim, identity, or
lineage primitive. Boundary 8's AD-1 established that nothing inspected
justifies a universal `AnalyticalClaim` class, one physical schema, or one
reference-resolution algorithm. This unit's own source re-reading
(`recommendation_policy.py`, `recommendation_governance.py`,
`market_family_evaluation.py`, `production_chain_preflight.py`,
`edge_diagnostics.py`) confirmed the same conclusion from a different
angle: the inspected artifacts have materially different architectural
roles (root evidence artifact, method/governance artifact, transient
decision, composite manifest, active evidence report, audit report), each
with distinct required and inapplicable capabilities. Forcing them into
one physical shape would misrepresent capabilities that are correctly Not
Applicable as Absent, and would collapse distinctions the codebase itself
maintains deliberately (e.g. `market_family_evaluation.py`'s AST-enforced
separation from policy/API/betting modules, Boundary 4).

#### Alternatives considered and rejected
- **A universal `AnalyticalClaim` base class.** Rejected: no artifact
  benefits from shared inheritance; the roles are too architecturally
  distinct.
- **One physical persistence schema across all claim-shaped artifacts.**
  Rejected: would conflate durable claims, evidence reports, and
  governance/method artifacts, which have different persistence
  obligations (see D35 for the validity distinction specifically).
- **Structural typing via Python `Protocol` at the whole-claim level.**
  Rejected for the same reason a narrower `Protocol` was rejected for
  uncertainty representation specifically (D33): the confirmed artifacts
  do not share a meaningful common surface beyond a handful of
  independently-typed fields.

#### Consequences
- No production code changes result from this decision. The profile and
  matrix are documentation artifacts.
- Future claim-shaped artifacts are expected to be checked against the
  eleven-capability profile and added to the conformance matrix, not
  forced to inherit from or match an existing artifact's physical shape.
- `RecommendedBetResult` remains the closest-to-fully-conforming
  specialization; it does not become a template class others inherit
  from.

#### Revisit triggers
- A future unit finds two or more artifacts share enough real structural
  surface (not just a name) that a shared type would eliminate genuine
  duplication without forcing an artificial fit.
- A common claim capability protocol (this decision) is itself superseded
  by evidence that a physical schema is needed for a demonstrated
  cross-artifact consumer.

#### References
- `docs/workstreams/analytical_claims/CLAIM_CAPABILITY_PROTOCOL.md`
- `docs/workstreams/analytical_claims/CLAIM_CONFORMANCE_MATRIX.md`
- `docs/workstreams/analytical_claims/FINDINGS.md` (Boundaries 1, 4, 8)
- `docs/workstreams/analytical_claims/PLAN.md` → Workstream 2, Unit 5

---

### D31. Candidate-reference derivation is independently versioned; recommended-result schema increments to 2

**Status:** Accepted
**Date:** 2026-08-26

#### Decision

Candidate-reference derivation (`candidate_issuance_row_id`) has an
independently owned version, separate from `CandidateIssuance`'s own
artifact schema version. Version 1 is the exact pre-D31 algorithm and
output — the version marker is not part of the v1 hash payload; it selects
which derivation implementation runs, it is not an input to that
implementation's digest. `RecommendedBetResult`'s schema increments from 1
to 2 to add `candidate_reference_derivation_version`, recording the version
used to derive each result's candidate reference. Readers dispatch
validation through the recorded version, not through comparison against
whatever version is currently default. A recorded version with no known
implementation raises a dedicated `UnsupportedCandidateReferenceVersionError`
(a `ValueError` subtype), which propagates directly and is not wrapped.
A recorded version with a known implementation whose re-derived reference
disagrees with stored evidence remains the existing content-corruption
`ValueError`, unchanged. The recommendation API's offer provenance exposes
`candidate_reference_derivation_version` alongside `issuance_id` and
`candidate_reference_id`, copied mechanically by the serializer.

#### Context

Unit 2 and Unit 3 of this workstream's persistence-hardening arc closed the
publication-atomicity and writer-coordination gaps in the betting ledger.
This decision addresses a different, longstanding gap: the exact seed
incident that motivated Workstream 2's opening (WS1 Unit 2 changed
`candidate_issuance_row_id`'s hashed payload to close a non-injectivity
gap; every previously-persisted `RecommendedBetResult` began failing its
own self-validation on read, because that validation re-derives the
candidate reference using whatever derivation logic is *currently*
installed, with no way to know the reference was computed under a
different, no-longer-current definition).

Tracing every consumer of `candidate_issuance_row_id` by actual data
lifecycle (not merely call graph) established that only one consumer is
exposed to this risk: `validate_recommended_bet_result`, which re-derives a
reference from a *persisted* result's own embedded fields, potentially long
after the derivation logic that originally produced it may have changed.
`market_closeout.py::_candidate_reference` and
`recommendation_policy.py::_resolve_candidate` both operate on fresh,
same-operation data and can never encounter this failure mode — confirmed
during implementation: both required zero source changes and their
existing test suites passed unmodified.

`RecommendedBetResult.issuance_schema_version` already exists but versions
`CandidateIssuance`'s own artifact schema (`CANDIDATE_ISSUANCE_SCHEMA_VERSION`),
which remained `1` throughout the entire WS1 incident even as the
row-reference payload changed — direct proof these are two independent
versioning axes that must not be conflated.

Because `recommended_bet_result_id` canonicalizes every dataclass field
(blanking only `result_id` itself), adding this new field to
`RecommendedBetResult` changes every newly computed result's identity,
which in turn changes `RecommendedBetEvaluation.evaluation_id` (built from
`result_ids`). This is an intentional, understood consequence, confirmed by
dedicated tests, not an oversight.

#### Alternatives considered and rejected
- **Embedding the version marker inside the v1 hash payload.** Rejected:
  this would change the v1 digest itself, invalidating every reference
  already computed under the current (pre-D31) definition.
- **Current-version-equality gating** (`if recorded != CURRENT: raise
  superseded`) instead of genuine dispatch. Rejected: not extensible — it
  provides no seam for a future, still-supported older version to be
  re-derived correctly.
- **Reusing `issuance_schema_version`** for this purpose. Rejected: proven
  to be a distinct axis by the seed incident itself.
- **Wrapping `UnsupportedCandidateReferenceVersionError` into a generic
  `ValueError`.** Rejected: discards the structural benefit of a dedicated
  exception type for no compatibility gain, since it already subclasses
  `ValueError`.
- **Keeping `RECOMMENDED_BET_RESULT_SCHEMA_VERSION` at 1** for the new
  field set. Rejected: two different physical field sets cannot both
  truthfully claim to be "schema 1."
- **Explicit schema-version-aware backward-compatible decoding.** Rejected
  as real migration machinery not justified here, consistent with the
  project's clean-sheet, never-live status.
- **Omitting the derivation version from the API.** Rejected: would
  recreate, at the API boundary, the exact omission this decision closes at
  the domain layer.
- **Constructing a fully-implemented fake v2 solely to exercise the
  unsupported-version test path.** Rejected: only one version has ever
  existed; the dispatcher's terminal branch is exercised directly with an
  unrecognized version number.

#### Consequences
- `RECOMMENDED_BET_RESULT_SCHEMA_VERSION` is 2. The store path is
  `schema=2/`. No schema-1 reader remains; schema-1 artifacts are
  unsupported, not migrated.
- All existing schema-1 development artifacts were regenerated and the
  `schema=1/` tree deleted, via the real, verified production-chain
  sequence:
  ```
  gridiron production-chain issue-candidates \
    --season <season> --week <week> --evaluated-at <ts> --write

  gridiron production-chain create-governance \
    --created-at <ts> [governance parameters...] --write

  gridiron production-chain derive-policy \
    --issuance-id <issuance_id> --governance-id <governance_id> \
    --created-at <ts> --write

  gridiron production-chain evaluate-recommendations \
    --issuance-id <issuance_id> --policy-id <policy_id> \
    --decision-at <ts> --write
  ```
  Executed against real production data (2026-2027 season, week 1; 1,680
  quote observations; 698 issued candidates) on 2026-08-26. All 698
  regenerated results passed `validate_recommended_bet_result` — the exact
  function this decision modified — with zero errors, confirming the
  schema-2 migration and version-dispatch machinery are mechanically sound
  against real data volume. (All 698 results were `result_state=unavailable`
  due to no historical outcome/closeout data being available to policy
  derivation in this environment at the time — this exercises the schema
  and dispatch machinery, not the reference-mismatch branches, which remain
  covered by this unit's own unit tests.)
- Every newly built `RecommendedBetResult` records
  `CURRENT_CANDIDATE_REFERENCE_DERIVATION_VERSION` (currently always `1`).
- `result_id` and `evaluation_id` for newly built artifacts differ from
  what identical inputs would have produced under schema 1 — expected and
  intentional.
- `market_closeout.py` and `recommendation_policy.py` required no source
  changes.
- The recommendation API's offer-provenance response includes the
  derivation version as read-only, mechanically-projected provenance. The
  checked-in `api-schema.json` OpenAPI snapshot and the frontend's
  generated `schema.ts` were regenerated to match; one frontend test
  fixture (`recommendationPresentation.test.ts`) required updating to
  include the new required field — a real scope item not anticipated in
  the original design, caught by the frontend's own TypeScript build gate.

#### Revisit triggers
- A genuine version 2 derivation is designed and needs to coexist with
  version 1 for already-persisted results.
- Old (schema-1 or v1-only) artifacts must become readable again without
  regeneration.
- Candidate references become interpreted by external (non-Gridiron-Edge)
  consumers.
- A caller needs to construct or evaluate a non-current but still-supported
  reference version in memory (not just at the persisted-result validation
  boundary) — no such path exists today.
- A common claim capability protocol (ROADMAP Unit 5, this workstream)
  supersedes this local per-artifact representation.

#### References
- `src/gridiron_edge/market/candidate_issuance.py`
- `src/gridiron_edge/market/recommended_bet_result.py`
- `src/gridiron_edge/market/recommended_bet_result_store.py` (unmodified —
  its reflective codec absorbed the schema change automatically)
- `src/gridiron_edge/api/schemas/recommendations.py`
- `src/gridiron_edge/api/serializers/recommendations.py`
- `frontend/src/components/recommendations/recommendationPresentation.test.ts`
- `tests/unit/market/test_candidate_issuance.py`
- `tests/unit/market/test_recommended_bet_result.py`
- `docs/workstreams/analytical_claims/PLAN.md` → Workstream 2, Unit 4
  (closed)
- DECISIONS.md D30 (bet-ledger writer coordination, for structural
  precedent)
- VISION.md → the versioned analytical claim's lifecycle status and
  invalidation-contract capabilities (this decision implements those
  existing invariants for one artifact type; it does not amend VISION.md)

### D30. Bet-ledger writer coordination uses an intra-process thread lock, not cross-process locking

**Status:** Accepted
**Date:** 2026-08-25

#### Decision

`betting/ledger.py`'s writer-coordination mechanism is a module-level
`threading.RLock`, held across the complete read-modify-write sequence in
`log_bet` and `settle_bet`, and reused by `betting/recording.py::record_wager`
across its full snapshot/write/restore sequence. This coordinates threads
within one running process only. It does not use file-based locking,
cross-process locking, or optimistic-concurrency/generation tracking.

#### Context

Unit 2 (bet-ledger atomic publication) made individual ledger writes
atomically visible but left open a distinct problem: two overlapping callers
that each read a valid ledger and each publish atomically can still silently
discard one another's update, because per-write atomicity does not make the
full read-modify-write transaction atomic across callers. Unit 2's own
review process had already rejected, once, the reasoning "no existing
coordination mechanism exists, therefore no coordination is needed" as
insufficient evidence that concurrent access cannot occur. This decision
resolves the question with direct evidence rather than repeating that error.

Evidence gathered before choosing a mechanism:
- No locking utility (file-based or otherwise) exists anywhere in this
  codebase, confirmed by a repository-wide search. The only superficially
  related match was a noisy-third-party-logger suppression entry
  unrelated to locking.
- No prior `DECISIONS.md` entry addresses ledger or writer concurrency.
- The CLI bet-recording commands (`gridiron bet log`, `gridiron bet settle`)
  are confirmed, directly by the project owner, to be unused in practice;
  the frontend, via the API, is the sole real path for recording wagers.
- The API is confirmed, directly by the project owner, to run as a single
  process (`uv run gridiron api serve --reload`) — not a multi-worker
  deployment. The only systemd-managed service in this repository (D26) is
  the unrelated quote-collection worker, which the same decision explicitly
  scopes as validated for that role only, not as a general API/frontend
  appliance.
- `api/routes/portfolio.py::record_portfolio_bet` is defined as a synchronous
  `def`, not `async def`. FastAPI/Starlette route synchronous handlers
  through a thread pool (`starlette.concurrency.run_in_threadpool` →
  `anyio.to_thread.run_sync`), confirmed directly from a production
  traceback encountered earlier in this project's own operation. Two
  near-simultaneous requests to the same endpoint (a double-click, two
  browser tabs, a frontend retry) therefore execute as two genuine threads
  of the single running API process.

This establishes the real, present risk as intra-process thread
concurrency, not cross-process concurrency. No evidence supports a
cross-process risk under the current, confirmed deployment and usage
pattern.

#### Alternatives considered and rejected

- **File-based or cross-process locking (e.g., a `filelock`-style
  dependency).** Rejected for now: no cross-process writer scenario is
  evidenced. Introducing this would add a new dependency and complexity to
  address a risk that does not currently exist, and would need to be
  revisited (not necessarily replaced) if the deployment model changes.
- **Optimistic concurrency (generation/fingerprint check, reject-and-retry
  on conflict).** Rejected: this is a low-frequency, single-operator
  personal betting ledger, not a high-throughput service; a blocking lock
  is simpler and sufficient for the confirmed risk, and optimistic
  concurrency's added retry/conflict-surfacing complexity is not justified
  by the actual call volume or contention pattern.
- **Documented-but-unenforced single-writer assumption.** Rejected outright:
  this was the original, incorrect draft of this decision, corrected during
  review. A documented assumption is not an enforced contract, and it does
  not survive contact with the confirmed thread-pool routing behavior above.

#### Consequences

- `log_bet`, `settle_bet`, and `record_wager` are serialized against one
  another within one process; two overlapping calls block rather than
  silently losing one write.
- This lock provides **no protection** if the API is ever run with multiple
  worker processes (e.g., `--workers N`, gunicorn, or a production
  multi-worker deployment), or if the CLI bet commands are ever used
  alongside a running API instance. This boundary is stated explicitly in
  the `ledger.py` module docstring.
- `record_wager` reuses `ledger.py`'s lock as an `RLock` specifically
  because it calls `log_bet` internally while already holding the lock; a
  non-reentrant `Lock` would deadlock on that call.

#### Revisit triggers
- If the API deployment model changes to multiple worker processes.
- If the CLI bet-recording commands are ever used in practice, especially
  alongside a running API instance.
- If ledger write volume or contention grows to a point where lock
  contention itself becomes an observed problem (no evidence of this
  today).

#### References
- `src/gridiron_edge/betting/ledger.py`
- `src/gridiron_edge/betting/recording.py`
- `tests/unit/betting/test_ledger.py::TestWriterCoordination`
- `tests/unit/betting/test_recording.py::test_concurrent_write_survives_a_failed_recorded_wager_rollback`
- `src/gridiron_edge/api/routes/portfolio.py::record_portfolio_bet`
- DECISIONS.md D26 (quote-collector deployment scope, for contrast)
- PLAN.md → Workstream 2, Unit 3 (persistence hardening — writer
  coordination)

## D29. System-known visibility is governed by `fetched_at`
Status: Accepted  Date: `<COMMIT_DATE>`  Workstream: Quote Observation (WS1)

### Decision
An observation is available to an "as-known-at-cutoff" view **only if** its local
system-known timestamp `fetched_at` satisfies the declared cutoff contract. For the
initial contract, visibility is **inclusive**: `fetched_at <= cutoff`.
- `sportsbook_updated_at` remains **source-provided update metadata**, not the
  system-known basis.
- `commence_time` remains the **event-start (kickoff) boundary**, used for pregame
  *eligibility*, which is a separate predicate from visibility.

### Context
WS1 inspection (FINDINGS rev 8) found the store carries three UTC world-times
(`fetched_at`, `sportsbook_updated_at`, `commence_time`) but exposes no
as-known-at-arbitrary-cutoff retrieval (F11/F29), and the production candidate path
consumes the full ledger with no cutoff filter (F35/F40, verified reachable leak).
A durable, unambiguous choice of *which timestamp defines system knowledge* is
required before any point-in-time retrieval is built.

### Consequences
- Visibility and pregame eligibility are **separate, composed predicates** — never
  fused as `min(cutoff, kickoff)`. Observed → Visible (`fetched_at <= cutoff`) →
  Eligible (`is_live is False and fetched_at < commence_time`).
- Cutoffs must be timezone-aware UTC; naïve/non-UTC values are rejected.
- No `decision_cutoff` or additional `effective_time` field is added to stored
  observation rows to satisfy this decision. Whether a further effective-time concept
  is ever needed is a separate, still-open design question (F11).

### References
- `docs/workstreams/quote_observations/FINDINGS.md` (F11, F29, F35, F40)
- root `PLAN.md` — Point-in-time quote evidence retrieval

---

## D28. Unresolved collection claims are not automatically retried
Status: Accepted  Date: `<COMMIT_DATE>`  Workstream: Quote Observation (WS1)

### Decision
When a collection claim exists without a terminal result (e.g., a crash between
`write_claim` and `write_result`), the system does **not** automatically retry or
reclaim it. Such claims surface as an explicit DEGRADED verification state. **Any**
future retry, lease, expiry, reconciliation, or manual-resolution mechanism requires
a separate explicit decision that defines how ambiguous prior execution and
provider-cost (duplicate-request) risk are handled.

### Context
WS1 inspection found unresolved-claim **detection** is implemented and tested (the
verifier reports `unresolved_claims` → DEGRADED), but **recovery** is absent and the
no-retry behavior is undocumented as policy (F23). Automatic retry under an ambiguous
prior outcome could double-spend paid provider requests, so committing to "no
automatic retry" locks safe present behavior without prematurely choosing the
long-term recovery model.

### Consequences
- Unit 4 (collection claim & receipt robustness — F22/F26/F27) may harden the
  claim/receipt lifecycle but must **not** introduce automatic retry/reclaim until
  this decision is superseded by an explicit recovery-policy decision.
- A stranded claim remains stranded (surfaced, not silently healed) until manually
  resolved.

### References
- `docs/workstreams/quote_observations/FINDINGS.md` (F23; related F22/F26/F27)
- `ROADMAP.md` — Unit 4

## D27. Production recommendation proof is an exact immutable evidence chain

**Status:** Accepted

**Date:** 2026-08-18

### Decision

Treat production recommendation proof as a chronological chain of exact
immutable identities rather than a request-time reconstruction or a directory
existence check.

The chain begins with one explicitly selected weekly product and its exact
forecast events. Candidate issuance consumes that product and canonical quote
history at an explicit UTC evaluation time. Recommendation governance is an
independent immutable, content-addressed artifact. Recommendation policy is
derived from exact empirical evidence plus that exact governance. Recommended-
bet evaluation requires an exact issuance, exact policy, and explicit UTC
decision time.

Production-chain preflight is the composite chronological audit artifact. One
assessment reads and validates the exact persisted evidence available at its
explicit UTC assessment time, classifies every component independently for
Moneyline, Spread, and Total, and may itself be persisted immutably. It does not
select candidates, policies, or evaluations by modification time or directory
presence.

Collection execution remains owned by the selected collection plan, due-state
evaluator, and immutable claim and terminal-result receipts. Manual quote
ingestion may establish quote-history depth but cannot prove selected-plan
execution.

Postgame proof is assembled through existing owners rather than a competing
postgame store. One assessment reuses selected-product outcome reconciliation,
exact candidate market closeout, market-specific CLV, historical quote
boundaries, market-family evaluation, cleaned games, and optional settled-wager
evidence. Before earliest kickoff, these boundaries are short-circuited and
remain not yet eligible.

The API and frontend mechanically serialize and present persisted recommended-
bet results. Positive expected value and candidate state cannot manufacture a
qualified or recommended result. Recording a wager remains an explicit local
operation and does not place a sportsbook wager.

### Rationale

The recommendation chain crosses selected forecasts, timestamped market
observations, candidate decisions, governed policy inputs, policy decisions,
optional wager evidence, outcomes, closeout, CLV, and realized performance.
Reconstructing any earlier decision from newer repository state would weaken
chronology and provenance.

Separate immutable identities make every decision input auditable and allow
exact replay. Strict relationship checks prevent unrelated artifacts from being
accepted merely because files exist. Independent family components prevent
Moneyline evidence from satisfying Spread or Total acceptance.

Using production-chain preflight as the composite audit snapshot avoids adding
a second persisted postgame report whose responsibilities would overlap the
outcome, closeout, CLV, market-family evaluation, and preflight owners.

### Consequences

- Candidate issuance requires an explicit selected product, exact forecast run,
  canonical quote history, and UTC evaluation time.
- Recommendation governance has its own deterministic content identity and
  immutable store.
- Policy derivation and recommendation evaluation operate on exact persisted
  identities and explicit UTC timestamps.
- Missing empirical evidence produces persisted unavailable policy and result
  states rather than invented thresholds or recommendations.
- Preflight validates artifact content and identity relationships, not file
  presence or recency.
- Multiple matching exact artifacts are conflicting unless a separate explicit
  selection contract is intentionally introduced.
- Manual quote ingestion does not count as selected-plan execution.
- Postgame evidence is computed once per assessment through existing domain
  owners and serialized by the immutable preflight snapshot.
- Moneyline price CLV, Spread point CLV, and Total point CLV remain distinct
  family evidence.
- Realized performance requires uniquely attributed settled-wager evidence;
  absent wagers remain unavailable rather than zero.
- The frontend presents persisted lifecycle state and does not infer
  recommendation eligibility.
- Gridiron Edge records wagers locally only through explicit user action and
  does not place sportsbook wagers.

### References

- `src/gridiron_edge/cli/production_chain.py`
- `src/gridiron_edge/market/recommendation_governance.py`
- `src/gridiron_edge/market/recommendation_governance_store.py`
- `src/gridiron_edge/market/production_chain_preflight.py`
- `src/gridiron_edge/market/production_chain_preflight_store.py`
- `src/gridiron_edge/market/candidate_issuance.py`
- `src/gridiron_edge/market/candidate_issuance_store.py`
- `src/gridiron_edge/market/recommendation_policy.py`
- `src/gridiron_edge/market/recommendation_policy_store.py`
- `src/gridiron_edge/market/recommended_bet_result.py`
- `src/gridiron_edge/market/recommended_bet_result_store.py`
- `src/gridiron_edge/market/collection_execution.py`
- `src/gridiron_edge/market/collection_receipt_store.py`
- `src/gridiron_edge/market/market_closeout.py`
- `src/gridiron_edge/market/market_family_evaluation.py`
- `src/gridiron_edge/evaluation/live_forecast_closeout.py`
- `PLAN.md`
- `ROADMAP.md`

## D26. Quote-worker deployment is repository-owned but operator-activated

**Status:** Accepted

**Date:** 2026-08-16

### Decision

Own the quote-collection worker's systemd service, timer, invocation wrapper,
installation logic, verification logic, and operational guidance in the
repository.

Keep weekly plan generation, active-plan selection, artifact transfer,
credential provisioning, installation, and timer activation as explicit
operator actions. Deployment does not infer a season, week, repository path,
runtime executable, deployment identity, or credential path.

The installed service remains a non-root `Type=oneshot` unit with no automatic
restart. Its wrapper generates the current UTC evaluation timestamp and invokes
the existing selected-plan executor. The timer wakes every five minutes, while
due-time eligibility, the inclusive fifteen-minute grace period, provider
quota, execution claims, terminal results, and quote persistence remain owned
by the existing domain boundary.

Installation validates the complete staged deployment before replacing live
files. If systemd reload fails after replacement, installation restores the
prior files and permission modes and reloads the restored state. Explicit timer
activation is a separate post-install operation and does not roll back a valid
installation if activation fails.

The provider key remains in a root-owned mode-0600 environment file. The
installer validates that the file contains exactly one nonempty
`ODDS_API_KEY` assignment. Read-only verification validates only file type,
ownership, and permissions and never opens the credential file.

### Rationale

Collection-plan and execution semantics were already validated independently of
deployment hardware. Repository ownership makes the worker reproducible and
recoverable without moving policy into systemd or creating a second execution
path.

Separating installation from activation prevents a valid deployed
configuration from being destroyed because activation fails. Staged validation
and restoration prevent malformed or unloadable units from replacing the last
known-good worker configuration.

Keeping credentials and weekly selection outside repository files preserves
secret isolation and explicit operational intent.

### Consequences

- Machine-specific inputs are required explicitly during installation and
  verification.
- The timer never selects or infers a weekly plan.
- The service never embeds a static evaluation timestamp.
- Installation may update deployed files without enabling the timer.
- A failed installation reload restores the prior deployment.
- A failed explicit activation leaves the installed files intact.
- Verification can report credential-file security without reading the secret.
- The Raspberry Pi deployment is validated only as a quote-collection worker,
  not as a full API, frontend, training, or prediction appliance.

### References

- `deploy/bin/install_quote_collection_worker.py`
- `deploy/bin/verify_quote_collection_worker.py`
- `deploy/systemd/gridiron-edge-collector.service`
- `deploy/systemd/gridiron-edge-collector.timer`
- `src/gridiron_edge/deployment/quote_collection_worker.py`
- `tests/unit/deployment/test_quote_collection_worker.py`
- `HANDOFF.md`
- `PLAN.md`

## D25. The Odds API v4 is the supported current-market provider

**Status:** Accepted

**Date:** 2026-08-05

### Decision

Use The Odds API v4 as the supported provider for current and upcoming NFL
moneyline, spread, and total quotes. Keep provider access explicit under the
`gridiron ingest` command group. `weekly-predict` remains a consumer of an
existing source-neutral snapshot and does not perform a paid or
network-dependent odds fetch.

The normalized quote row separates aggregator provenance from offered-price
provenance:

- `fetched_at`: local UTC observation time;
- `provider`: upstream data provider;
- `provider_event_id`: provider event identity;
- `sportsbook`: actual book offering the quote, nullable for truthful consensus
  sources;
- `sportsbook_updated_at`: provider-reported UTC update time for the book or
  market, when supplied;
- `commence_time`: event start time in UTC;
- `is_live`: whether the quote is in-play;
- canonical season, week, game, date, team, market, side, American odds, and
  line fields.

Current observations and future historical backfill use the same normalized
quote contract but different storage and operational semantics. A successful
current pull appends observed quotes to the local observation ledger and
atomically replaces the current snapshot. Historical provider backfill,
partitioning, retention, opening and closing definitions, and leakage-safe
evaluation are separate later work.

All returned sportsbooks are preserved. Ingestion does not pick a preferred or
best book. Downstream edge construction must evaluate complete same-book market
pairs and retain sportsbook provenance before ranking actionable offers.

### Provider rationale

The official NFL documentation shows a single NFL odds endpoint returning live
and upcoming games with commence time, team identity, bookmaker key and title,
bookmaker update time, moneyline, spread, total, point, and American price
fields. The provider states that historical NFL featured-market odds are
available from mid-2020.

Public self-service plans support development without a sales-led contract.
Exact request-credit consumption remains observable through provider response
headers and will be validated with the integration key before any automated
refresh cadence is introduced.

Official references:

- [The Odds API NFL coverage](https://the-odds-api.com/sports-odds-data/nfl-odds.html)
- [The Odds API v4 documentation](https://the-odds-api.com/liveapi/guides/v4/)
- [The Odds API plans](https://the-odds-api.com/)

### Failure and freshness boundaries

- Missing credentials fail before network access.
- Request, authentication, quota, malformed-payload, and zero-usable-match
  failures exit nonzero and do not replace a valid current snapshot.
- Partial usable coverage may be persisted with explicit fetch diagnostics;
  weekly readiness remains authoritative for coverage and eligibility.
- Storage records timestamps but does not invent a universal freshness limit.
  Consumers apply an explicit maximum age appropriate to their operation.
- Forecast publication remains valid when market ingestion fails or the current
  snapshot is stale.

### Consequences

- `sportsbook` can no longer stand in for both provider and book.
- The development odds schema and local Parquet artifacts may be replaced.
- nflverse schedule rows identify `provider=nflverse` and no fabricated
  sportsbook.
- Multi-book current snapshots require a sportsbook-aware recommendation pivot;
  game-only row overwrites are not valid.
- The legacy DraftKings adapter, resolver, and CLI command are retired rather
  than carried through the provider-aware quote contract.
- Historical backfill may use The Odds API or another compatible provider later
  without changing the normalized row contract.

### References

- `src/gridiron_edge/ingest/odds/store.py`
- `src/gridiron_edge/ingest/odds/nflverse_schedule.py`
- `src/gridiron_edge/market/recommendations.py`
- `src/gridiron_edge/market/weekly_edge_service.py`
- `PLAN.md`
- `ROADMAP.md`

## D24. Weekly operation uses immutable forecast events and explicitly selected weekly products

**Status:** Accepted

**Date:** 2026-08-05

### Decision

Weekly operation persists immutable forecast events and composes immutable, schedule-complete weekly products. Current state changes only through explicit season-and-week product selection.

Win and Total model families are inspected and selected independently before inference. Every selected family must produce one valid forecast for every scheduled game before events are written. Selected events from one invocation share a run ID and UTC generation timestamp while retaining exact model identity and role.

Forecast roles are explicit:

- `live` identifies forecasts issued by the operational weekly workflow before kickoff;
- `backfilled` identifies historical reconstruction used for evaluation and champion comparison.

The selected weekly product is the operational serialization boundary for API, forecast output, edge generation, readiness verification, and completed-week closeout. Consumers do not infer current state from newest files, event recency, champion lookup, or Elo fallback.

Prediction readiness and market readiness are independent. A prediction-ready selected product may publish forecast output when markets are missing. Edge diagnostics preserve blocked, non-calculable, no-positive, filtered, and positive states without fabricating prices or presenting blocked results as `No play`.

### Context

The previous game path mixed mutable archive selection, model-specific fallback behavior, prediction generation, and API loading. This made live provenance, current state, independent Win and Total selection, and market failure semantics difficult to prove.

The canonical one-row Away/Home game contract, model-specific availability inspection, policy-selected weekly execution, immutable event store, and weekly-product store now provide explicit identities and boundaries for each operation.

### Consequences

- Multiple coherent forecast runs and weekly products may coexist for one weekly scope.
- Writing a product does not select it.
- Missing current selection is an explicit error.
- Postgame closeout evaluates the exact `live` events referenced by the selected product.
- Backfilled events cannot substitute for missing live events.
- API request paths serialize persisted state and do not run inference or select a forecast.
- Missing market data does not invalidate prediction readiness.
- Operational recovery reruns a coherent workflow or explicitly selects an indexed product; it does not repair state through recency inference.

### Supersession

This decision supersedes D22 for current weekly operation. D22 remains as historical context for the earlier Elo-only upcoming-week path and API fallback.

### References

- `src/gridiron_edge/evaluation/forecast_store.py`
- `src/gridiron_edge/models/game_prediction/weekly_execution.py`
- `src/gridiron_edge/models/game_prediction/weekly_product_store.py`
- `src/gridiron_edge/cli/weekly_predict.py`
- `src/gridiron_edge/cli/post_week.py`
- `src/gridiron_edge/cli/verify_week.py`
- `HANDOFF.md`

---
## D23. BetSlip is a draft decision workspace with immutable recommendation provenance

**Date:** 2026-07-29

### Decision

BetSlip is a temporary decision-support workspace, not a sportsbook execution
surface and not the authoritative betting ledger.

Each staged selection uses a versioned discriminated BetLeg with:

- canonical producer-independent wager identity;
- immutable recommendation provenance;
- editable draft inputs.

Recommendation provenance records the model, reference price, reference
probability/value context, EV, edge strength, full-Kelly fraction, dollar Kelly
stake, bankroll, and Kelly multiplier available when the recommendation was
created.

Draft inputs record current odds, proposed stake, optional sportsbook text, and
notes. Editing draft inputs never mutates recommendation history.

### Price discipline

No producer may fabricate a sportsbook price.

Game edges preserve the exact `american_odds` returned by `/edges`. Prop
interests remain unpriced until a current price is manually entered or a future
verified odds source supplies one.

`market_value` is not a replacement for sportsbook odds. It retains its
market-specific meaning.

### Bankroll discipline

Dollar Kelly sizing requires an explicit bankroll basis.

`/edges` does not substitute a hidden bankroll when the query omits one.
Without bankroll, edge rows, EV, and full-Kelly fraction remain available while
`kelly_stake` remains null.

Tracked BetSlip sizing prefers `/portfolio/summary.bankroll`. A what-if
bankroll is allowed only as an explicitly selected source. Tracked, what-if,
unavailable, and zero bankroll states remain distinct. BetSlip does not fall
back to the legacy AppState calculator bankroll.

### Aggregate discipline

Singles report aggregate stake, payout, and profit only when every staged leg
has current odds and a proposed stake.

Parlays report quoted combined odds, payout, and profit only when every leg is
priced and an explicit parlay stake exists.

BetSlip does not report combined parlay model probability, EV, or Kelly because
leg correlation is not modeled.

### Persistence discipline

BetSlip and sizing persistence are versioned and runtime-validated. Malformed
legs or sizing state are rejected. Legacy prototype state is ignored rather
than migrated because it may contain fabricated prices, invalid prop variants,
incorrect identifiers, or producer-specific IDs.

### Consequences

- The same wager deduplicates across producer screens.
- Recommendation history remains auditable after current odds change.
- Missing price, probability, bankroll, or stake inputs produce explicit
  unavailable states instead of inferred values.
- BetSlip can support later draft export without implying execution.
- A future `Record Bet` workflow requires a separate backend design for ledger
  writes, duplicate protection, bankroll transactions, and partial failures.
- Multi-book line shopping remains a separate odds-ingestion capability.
- The interface must not render a `Place Bet` action.

### Revisit triggers

Revisit this decision if:

- a verified multi-book odds contract supplies current prop and game prices;
- a deliberately approved recorded-bet write API is added;
- correlation-aware parlay probability and EV models are implemented;
- local storage is replaced by authenticated server-side draft persistence.

### References

- `frontend/src/utils/betLegs.ts`
- `frontend/src/utils/betSlipSizing.ts`
- `frontend/src/utils/betSlipSummary.ts`
- `frontend/src/context/BetSlipContext.tsx`
- `frontend/src/hooks/useBetSlipSizing.ts`
- `frontend/src/components/betslip/`
- `src/gridiron_edge/api/routes/edges.py`
- `src/gridiron_edge/market/recommendations.py`

## D22. Elo is the canonical upcoming-week model; games API falls back champion→elo

**Status:** Superseded by D24

**Date:** 2026-07-12
**Workstream:** W9.10 (Compare) — surfaced during offseason-readiness work
**Status:** Accepted

### Decision

For **upcoming (unplayed) weeks**, the platform serves **Elo** win-prob
predictions. The games API resolves the `win_prob` champion first, then
**falls back to `elo`** when the champion has no archived rows for the
requested `(season, week)`. `weekly-predict`'s `predict-week` stage
archives upcoming weeks under `model_type="elo"` by design.

### Context

Trained models (logistic / random_forest / xgboost) predict from the
modeling file — a feature matrix built **only from completed games**.
They structurally cannot predict an upcoming week: no feature rows exist
for unplayed games (and many rolling features — e.g. L3 EPA — are
undefined for Week 1 of a new season). Elo, by contrast, predicts from
the Elo state table, which **carries a rating forward** before a team
plays. Elo is therefore the *only* model that can predict an upcoming
week without new machinery.

This surfaced in the offseason: with the 2026-2027 season not yet played,
`/games?season=2026-2027&week=1` returned empty — the champion
(logistic) had zero rows for the week, and the API filtered strictly by
champion. The frontend showed nothing (or the prior Super Bowl).

Three options:

1. **Fall back champion→elo for upcoming weeks** (chosen). Small loader
   change; serves the only predictions that exist.
2. **Build an upcoming-week feature matrix** so trained models predict
   upcoming weeks. Real workstream: fold the upcoming schedule into
   `build-features`, compute per-game features for unplayed games, run
   the champion predict path. Deferred — worth it only if trained-model
   upcoming projections are wanted (more useful mid-season, where
   rolling features exist for the next unplayed week).
3. **Serve empty for upcoming weeks.** Rejected — the frontend is a
   verification surface; showing nothing is strictly worse than showing
   the Elo signal that genuinely exists.

Option 1 is not a workaround — Elo is the *correct* upcoming-week signal,
especially for Week 1 where trained-model features are thin-to-undefined.

### Consequences

- Games serve Elo for upcoming weeks: win probability populates;
  `model_spread` / `model_total` / projected scores are null (they come
  from trained-model post-processing, absent for upcoming weeks) and are
  marked via `field_status` per D14.
- Completed weeks still serve the champion (backfilled). The fallback
  only triggers when the champion has no rows for the scope.
- Consistent with D21: the fallback reads a static artifact (the archive)
  and picks a `model_type` filter — no request-time compute.
- `resolve_current_season_week` prefers the upcoming schedule's earliest
  week once the completed archive ends on a season-ender (week ≥ 22), so
  the default view lands on the upcoming Week 1 rather than replaying the
  final completed game.
- Fallback lives in `api/loaders.py::load_games_for_week` + `load_game`.

### Disconfirming evidence (when to revisit)

- If trained-model projections for upcoming weeks become genuinely
  wanted (e.g. mid-season next-week edges), build the upcoming-week
  feature matrix (Option 2) and predict under the champion — the
  fallback then only covers Week 1 / true cold-starts.

### References

- HANDOFF.md → §7 W13 champion subsection (champion→elo fallback) + offseason data-coverage
- ROADMAP.md §9 (upcoming-week feature matrix future note)
- `api/loaders.py::load_games_for_week`, `load_game`
- DECISIONS.md D21 (serialization boundary — this is consistent with it)

## D21. API layer is a serialization boundary, not a compute boundary

**Date:** 2026-07-01 (Tier 2, W8)
**Workstream:** W8 (API Serving Layer)
**Status:** Accepted

### Decision

Every endpoint serves pre-computed static artifacts. The API layer reads
from disk, serializes through Pydantic, and returns. Any computation
(model predictions, Monte Carlo simulations, ranking passes, champion
selection, evaluation metrics, percentile ranks, cohort aggregations)
happens upstream in ingest, training, or scheduled batch jobs, and the
results are persisted as files. The API layer never computes.

### Context

The prototype-driven Tier 2 design initially treated some endpoints as
"compute on request" — /model/performance calls build_evaluation_df +
summarise at request time; the champion-model resolution question
implied comparing archived model outputs at request time. Both are
compute-on-request patterns.

The correct architecture is: the retrain pipeline writes model outputs
and champion manifests; the evaluation pipeline writes metric summaries;
the sim pipeline writes projection CSVs; the ingest pipeline writes odds
and predictions. The API reads all of these as static files.

### Consequences

- Response times are dominated by disk I/O and Pydantic overhead.
  Millisecond-scale, deterministic.
- Staleness is visible: every response can include the mtime of the
  underlying artifact.
- Missing artifacts surface as _meta.field_status entries pointing at
  the batch job that should have produced them.
- No hidden computation, no in-request model calls, no request-time
  ranking.
- Every new endpoint asks: "what static file does this read?" If the
  answer is "we'd have to compute it," the answer is instead "add a
  batch job to write it."
- Some existing endpoints deviate from this and require refactoring:
  /model/performance currently computes metrics at request time.

### References

- PLAN.md → W8 Tier 2 Step 5 pre-planning
- DECISIONS.md D17-D20 (serializer + placeholder conventions)

## D20. Extended placeholder convention: `Unavailable` slugs for data limits

**Date:** 2026-07-01 (Tier 2 Step 1, W8)
**Workstream:** W8 (API Serving Layer)
**Status:** Accepted (refines D14)

### Decision

The placeholder convention introduced in D14 distinguished two field states: populated, and null-with-`field_status`. Tier 2 endpoints surfaced a third: fields that are null because the specific request or dataset lacks what's needed to compute them, not because upstream workstream work is pending.

Add an `Unavailable` slug family alongside `Blocker`:

- `Blocker` — field is null because an upstream workstream is not yet built. Frontend renders a "coming soon" state.
- `Unavailable` — field is null because the source data or request doesn't support it. Frontend can render a "not available for this request" state.
- `"pending"` (from D14) — retained for cases where backend work is scheduled but not yet done. Distinct from Unavailable in that pending fields *will* eventually populate.

`Unavailable` slugs use `roadmap` values that describe the nature of the gap: `"data"` for source-data limits, `"request"` for missing query parameters.

### Consequences

- Serializers construct `_meta.field_status` entries for both `Blocker` and `Unavailable` cases.
- Completeness tests accept slugs from either registry.
- Frontend can distinguish "not yet built" from "not applicable to this request" without changing the wire shape.
- Every null in an API response continues to have a documented reason — D14 semantics preserved and extended.

### References

- DECISIONS.md D14 (original placeholder convention)
- PLAN.md → Tier 2 Step 1

## D19. API loaders thread `settings.repo_root` explicitly to domain loaders

**Date:** 2026-06-27 (Tier 2 Step 1, W8)
**Workstream:** W8 (API Serving Layer)
**Status:** Accepted

### Decision

`api/loaders.py` wrapper functions **always pass `repo=settings.repo_root` explicitly** to the underlying domain loaders (`ledger.load_bets`, `bankroll.load_transactions`, `bankroll.balance_history`, `bankroll.current_balance`, etc.). API loaders do not rely on the domain loaders' default behavior of falling back to `get_settings().repo_root` internally.

### Context

Domain loaders in `betting/ledger.py` and `betting/bankroll.py` accept an optional `repo: Path | None = None` kwarg. When `None`, they call `get_settings()` themselves and use `repo_root` from that. This is convenient for CLI usage but hides which `Settings` a loader is using.

The API layer already has `Settings` in hand (via FastAPI's `SettingsDep` dependency). Two options:

1. Rely on the domain loader default (pass nothing, let it call `get_settings()` again).
2. Pass `repo=settings.repo_root` explicitly.

Option 2 wins because:

- Tests can inject a stubbed `Settings` via FastAPI's dependency override, and the domain loader honors it. Option 1 would ignore the override and re-read the real `get_settings()`.
- The API layer's `SettingsDep` becomes the single source of truth for path resolution; every request flows through it.
- Avoids surprising behavior where two requests in the same process could see different `Settings` snapshots if `get_settings()` had different results at different times.

### Consequences

- Every `api/loaders.py` wrapper takes `Settings` as its first argument and passes `settings.repo_root` explicitly to the domain call.
- Test fixtures for the API layer can point `Settings` at a `MiniRepoBuilder` temp directory and the domain loaders will read from there without further plumbing.
- If a domain loader's signature changes to require `repo`, the API wrapper is the single file to update.

### References

- PLAN.md → Tier 2 Step 1
- DECISIONS.md D18 (serializer scope)
- `betting/ledger.py::load_bets`, `betting/bankroll.py::load_transactions`, `balance_history`, `current_balance`

## D18. API serializers own `_meta.field_status` construction

**Date:** 2026-06-27 (Tier 2 design phase, W8)
**Workstream:** W8 (API Serving Layer)
**Status:** Accepted

### Decision

In responses that mix populated and unpopulated fields, **the serializer constructs the `_meta.field_status` block**, not the route handler. The route is responsible only for invoking the loader, passing the result to the serializer, and returning the constructed response object.

### Context

D14 established the placeholder convention (`null` + `_meta.field_status`). D14 did not specify which layer is responsible for marking fields. Two options:

1. **Route owns the `_meta` block.** Route knows which fields the serializer can produce and stamps the rest as pending or blocked.
2. **Serializer owns the `_meta` block.** Serializer is the code that decides which fields it can populate, so it also decides what to mark pending.

Option 2 wins because:

- The serializer is the only code that knows what it can produce. Route-level marking duplicates that knowledge.
- Routes stay thin (5–10 lines) and consistent across endpoints.
- When a backend addition lands and the serializer can now populate a previously-pending field, only the serializer changes.

### Consequences

- Routes are uniformly small.
- Serializer signatures consistently return the final response object, not a tuple of (data, metadata).
- Tests for serializers check both the data fields and the `_meta.field_status` block; tests for routes are mostly reachability and dependency-injection.

### Disconfirming evidence (when to revisit)

- If a route ends up wanting to override field-status entries from the serializer (e.g., to mark something blocked at runtime that the serializer thought was populated), the abstraction has the wrong owner.

### References

- DECISIONS.md D14 (placeholder convention)
- PLAN.md → Tier 2 design phase

---

## D17. API serialization pattern: per-endpoint hand-written serializers

**Date:** 2026-06-27 (Tier 2 design phase, W8)
**Workstream:** W8 (API Serving Layer)
**Status:** Accepted

### Decision

API endpoints that translate domain data (DataFrames, dataclasses) into Pydantic response models use **hand-written serializer functions, one per endpoint**, living under `src/gridiron_edge/api/serializers/`. Each serializer is 5–15 lines and explicitly maps loader output fields to schema fields. No reflection, no column-mapping configuration, no shared serialization engine.

### Context

Three alternatives were considered:

1. **Per-endpoint hand-written serializers.** Explicit, testable, no magic. More total lines of code.
2. **`model_validate(row.to_dict())` reflection.** Less code; relies on column names exactly matching field names. Fragile when either side changes.
3. **Shared `DataFrameSerializer` utility with per-endpoint mapping specs.** Hides translation logic in configuration; hard to debug when a mapping breaks.

Option 1 wins because:

- Each serializer is small enough that boilerplate is not painful.
- Column renames in the data layer fail at a specific named function, not a hidden mapping spec.
- Tier 2 is the first time the API layer touches the data layer; transparency now will pay off many times later.
- The unit test for each serializer reads like a contract: input shape → output shape.

### Consequences

- ~9 serializer modules under `api/serializers/`, each with a small unit test file.
- New columns in the data layer don't appear in API responses until a serializer explicitly maps them. (This is a feature: deliberate evolution, not silent leakage.)
- Performance: serializers are pure functions; trivially cacheable if a hot path emerges.

### Disconfirming evidence (when to revisit)

- If two or more serializers end up being near-identical in structure (same field-mapping pattern repeated), a shared utility may genuinely be warranted.
- If the count of serializer files grows past ~20 and most are mechanical, the boilerplate cost may have crossed the threshold where Option 3 wins.

### References

- PLAN.md → Tier 2 design phase

---

## D16. API response envelope: uniform field-status, sub-resource routes per blocker

**Date:** 2026-06-23 (Tier 1 design phase, W8)
**Workstream:** W8 (API Serving Layer)
**Status:** Accepted

### Decision

Two API design choices made during W8 Tier 1:

1. **List endpoints surface blocked-list state through the same `_meta.field_status` mechanism as scalar/object fields.** A blocked list endpoint sets `_meta.field_status["items"]` to its `BlockedStatus`. No separate `_meta.list_status` envelope field.
2. **Tier 3 sub-resource endpoints on parent resources (e.g., `/games/{id}/injuries`, `/props/{id}/shop`) live in their own route files grouped by blocker, not on the parent resource's router.**

### Context

D14 established the field-level placeholder convention. Tier 1 implementation surfaced two ambiguities:

- **List shape:** when a list endpoint is blocked, does the blocker live on a list-level envelope or inside `field_status` keyed on `"items"`? Uniform-`field_status` wins on consistency (one lookup pattern across all response shapes) at minor cost in discoverability.
- **Sub-resource routing:** Tier 3 sub-resources like `/games/{id}/injuries` could attach to the existing games router or live in their own files. Separate files win on transition clarity — when a blocker clears, the unblock work is a single-file diff with a clean PR boundary.

### Consequences

- `BaseListResponse[T]` carries only `items` and `total` alongside the inherited `_meta`. No second envelope field.
- The Tier 3 route file count is higher (~9 files), but each file maps to exactly one blocker domain and ~one PR's worth of future unblock work.
- OpenAPI `/docs` groups each Tier 3 domain as its own collapsible section, improving navigation for both internal review and the W9 frontend.
- The `Blocker` slug registry in `api/meta.py` is the single source of truth for blocker identity; consistency tests assert every route uses a registered slug.

### References

- PLAN.md → Tier 1 design phase
- DECISIONS.md D14 (placeholder convention)
- ROADMAP.md §9.5 (backend gaps drives blocker slugs)

## D15. Prototype-driven endpoint contract for W8

**Date:** 2026-06-23
**Workstream:** W8 (API Serving Layer)
**Status:** Accepted

### Decision

The W8 endpoint inventory is derived from the Gridiron Edge frontend prototype, not from speculative backend capabilities. Every screen in the prototype gets the endpoints it needs. The inventory is fixed at workstream start and does not contract during implementation; what varies is **field population**, governed by the placeholder convention (D14).

The endpoint inventory is organized into three population tiers:

* **Tier 1:** Direct serialization from existing backend output. Most fields populate at W8 close.
* **Tier 2:** Small backend additions during W8 (percentile ranking, off/def decomposition, opponent-allowed-by-position, weekly snapshots, limited cohort splits). Each is a discrete addition with a clear "field X populates" success signal.
* **Tier 3:** Blocked on upstream workstreams (W4.5, W7, W10, feature attribution, news ingest). Endpoints return fully `null` shapes with structured `_meta.field_status` blockers.

### Context

Initial W8 planning produced a speculative endpoint inventory of six endpoints based on ROADMAP guesses. The frontend prototype (19 screens, including dashboard, game detail, explainability, compare, props, line shopping, live, bankroll, news, tools, settings, onboarding) expanded that to \~25 endpoints with concrete shapes.

Working backwards from the prototype produces a substantially different and better contract:

* Pydantic schema design becomes near-mechanical translation rather than speculation.
* The "what does this endpoint return" question is settled by what the screen consumes.
* Aggregation and grouping needs (top-N edges, by-confidence rollups, split tabs) surface at design time rather than implementation time.
* Backend gaps become visible: each prototype field that the backend cannot produce is a structured signal that ends up in ROADMAP §9.

The alternative of cutting endpoints from W8 to match current backend capability was rejected: it would force throwaway "coming soon" scaffolding in the frontend, fragment the API surface as capabilities shipped, and lose the verification value that comes from seeing every gap in one place.

### Consequences

* The full endpoint inventory is locked at W8 start; Tier classification can change (Tier 2 → Tier 3 demotion if a backend addition proves too large), but endpoints are not removed.
* The frontend (W9) can wire to the full surface from day one, even though most Tier 2/3 endpoints will return mostly-null shapes initially.
* ROADMAP §9.5 captures the backend gaps as a structured list, with each item explicitly classified as "W8 Tier 2," "Deferred — future workstream," or "Blocks on Wx."
* M4.5 (Visual output verification) is the natural milestone: walking every populated screen verifies the populated fields and surfaces the null ones as roadmap signals.

### Alternatives considered and rejected

* **Speculative endpoint inventory.** Rejected: produces over- and under-specified endpoints that need rework once the frontend is real.
* **Cut endpoints that cannot be populated today.** Rejected: forces frontend branching, scaffolding work, and loses the verification signal.
* **Endpoint-level 501 stubs.** Rejected: pollutes OpenAPI docs and gives the frontend no shape to render against. The field-level placeholder convention (D14) is the chosen alternative.

### References

* PLAN.md → Current Workstream → W8
* ROADMAP.md §9.5 Backend gaps surfaced by the prototype
* Gridiron Edge frontend prototype (source preserved separately)

---

## D14. Placeholder convention for unpopulated API fields

**Date:** 2026-06-23
**Workstream:** W8 (API Serving Layer)
**Status:** Accepted (provisional — explicitly revisitable)

### Decision

API responses use a uniform placeholder convention for fields the backend cannot yet populate:

1. The field returns `null`.
2. The response includes an optional top-level `_meta.field_status` dictionary keyed on field paths (dot notation).
3. Each entry is either the string `"pending"` (backend work scheduled but not done) or a structured object `{"status": "blocked", "blocker": <slug>, "roadmap": <reference>}`.
4. Granularity is **field-level**, not section-level.

Example:

```json
{
  "game_id": "sf-bal",
  "model": {"home_win_prob": 0.71, "home_win_lo": 0.62, "home_win_hi": 0.78},
  "injuries": null,
  "swing_factors": null,
  "_meta": {
    "field_status": {
      "injuries": {"status": "blocked", "blocker": "injury_data_source", "roadmap": "§5.3"},
      "swing_factors": {"status": "blocked", "blocker": "feature_attribution"}
    }
  }
}
```

This decision is **provisional**: if `_meta` proves noisy in practice during W8 or W9, the convention is revisited rather than entrenched.

### Context

The frontend prototype covers \~19 screens worth of analytics outputs. The backend can populate some fields today, others require small additions during W8, and others are blocked on future workstreams (W4.5 scenario engine, W7 multi-book odds, W10 live state, a possible feature-attribution workstream, and ROADMAP §5.3 injury data).

Two endpoint-level alternatives were rejected before settling on field-level placeholders:

* **Omit endpoints that can't be fully populated.** Rejected: forces the frontend to branch on endpoint existence; creates throwaway scaffolding work in W9; loses the "what's missing" signal that is the whole point of the verification surface.
* **Return 501 with structured metadata at the endpoint level.** Rejected: pollutes the API surface and the OpenAPI docs; gives the frontend no shape to render against.

Field-level placeholders inside a 200 response preserve:

* A single, consistent endpoint inventory that does not change as backend capabilities ship.
* The full prototype shape for the frontend, with placeholders rendering as dim/dash UI (the prototype already uses this pattern).
* An observable, structured inventory of "what's missing" — every walk of the UI surfaces gaps.

### Consequences

* All response models inherit a `BaseResponse` Pydantic shape that carries `_meta: ResponseMeta | None`.
* Backend code constructing responses must explicitly mark unpopulated fields with their status; silent `null` is treated as a bug.
* The frontend renders `null` with a uniform placeholder treatment regardless of cause; `_meta.field_status` is informational for development walkthroughs.
* ROADMAP §9 (Known Issues & Backlog) becomes the source of truth for which gaps map to which workstreams; the API does not duplicate this prioritization, only references it.
* If `_meta` proves noisy in practice, the convention is revisited — this is an explicit "ship it, learn from it" stance, not a permanent commitment.

### Disconfirming evidence (when to revisit)

* Frontend consumers report that `_meta` blocks make response inspection harder, not easier.
* The `_meta.field_status` dict regularly grows past \~10 entries per response.
* Constructing the envelope on the backend becomes a recurring source of bugs or test churn.

### References

* PLAN.md → Current Workstream → Locked architectural decisions → Placeholder convention
* ROADMAP.md §9.5 Backend gaps surfaced by the prototype

---

## D13. FastAPI + Pydantic v2 for the API serving layer

**Date:** 2026-06-23
**Workstream:** W8 (API Serving Layer)
**Status:** Accepted

### Decision

The W8 API serving layer is built with **FastAPI** as the framework and **Pydantic v2** for request validation and response models. Pydantic is confined to the `api/` boundary; no Pydantic imports outside `src/gridiron_edge/api/`.

### Context

W8 needs to expose every analytics output the platform produces, shaped to the Gridiron Edge frontend prototype. Three framework families were considered:

1. **FastAPI + Pydantic v2** — native integration, free OpenAPI/Swagger docs at `/docs`, request validation → 422 automatic, response model serialization with field filtering.
2. **FastAPI + stdlib dataclasses** — keeps the codebase Pydantic-free at the cost of weaker validation, awkward request body parsing, and degraded OpenAPI generation.
3. **Litestar** with msgspec/attrs/Pydantic interchangeable — smaller community and ecosystem; the framework flexibility is not worth the reduced StackOverflow and tooling coverage for a single-developer project.

The codebase is otherwise pandas/dataclass-shaped. Dragging Pydantic into `models/`, `evaluation/`, `market/`, or `features/` would be a costly cross-cutting change. Confining Pydantic to the API boundary preserves the existing domain idioms while gaining the FastAPI integration benefits where they matter.

Time-to-first-dashboard was the dominant prioritization signal: W8 is a verification surface for W9 (Frontend), and faster feedback compounds.

### Consequences

- New runtime dependency on Pydantic v2 and FastAPI.
- OpenAPI/Swagger docs at `/docs` ship for free; no separate API documentation effort.
- Response models live in `api/schemas/`; routes in `api/routes/`; both import from domain modules but the domain does not import from `api/`.
- A future migration off FastAPI would require rewriting `api/` but would not touch the rest of the codebase.
- A future decision to expose Pydantic models more broadly (e.g., for config validation) is not foreclosed but is not adopted here.

### Alternatives considered and rejected

- **FastAPI + stdlib dataclasses.** Rejected: the OpenAPI generation degrades materially for request bodies, and the validation gap costs more in W8 than the avoided dependency saves.
- **Litestar.** Rejected: smaller ecosystem, no compelling differentiator for a single-developer read-only API.
- **Starlette directly.** Rejected: too much boilerplate for the time-to-dashboard goal.
- **Flask.** Rejected: no native type-driven validation or OpenAPI generation.

### References

- PLAN.md → Current Workstream → Locked architectural decisions
- ROADMAP.md §W8

---

## D12 - Trainable Describes the Artifact Lifecycle, Not the Training Call

Date: 2026-06-20

Decision:
The Trainable protocol requires only:

    spec: ModelSpec
    is_trained(*, repo: Path | None = None) -> bool

The training call itself is intentionally NOT part of
the protocol because game and prop trainers have
legitimately different training signatures:

    GamesTrainer.train(df, *, model_type, repo, ...)
    PropTrainer.train(*, model_type, repo)

ModelSpec.trainable: bool is the canonical declarative
source of truth for whether a model has a training
step. ModelRegistry.register enforces consistency
between spec.trainable and the structural Trainable
check at registration time so the two signals cannot
drift apart at runtime.

Rationale:
The audit recommended deleting Trainable entirely and
relying solely on spec.trainable. After implementation,
we found that Trainable still provides real value:
without a structural check, a class can declare itself
trainable and then fail at first use due to missing
methods. The structural guarantee is what registration
enforces. The train signature, however, varies
legitimately across families; forcing uniformity would
have required a separate workstream to harmonize the
training call shapes across families.

Consequences:
- ModelRegistry.is_trainable and trainable_names read
  spec.trainable directly without instantiating models.
- Adding a new model that declares spec.trainable=True
  without implementing is_trained fails at import time.
- Family-specific training APIs remain free to evolve
  without affecting the registry contract.
- Harmonizing train(...) signatures across families
  remains a future workstream if desired.

## D11 - Task-discriminated Model Metadata

Date: 2026-06-20

Decision:
Model metadata records all holdout metrics in a single
``metrics`` dict on BaseModelMetadata. Task-appropriate
keys are chosen by the trainer at training time.

Classification metric keys:
    brier, ece, auc, log_loss, accuracy

Regression metric keys:
    mae, rmse, r2

Display surfaces dispatch on ``meta.task`` to pick the
right keys.

Rationale:
Previously, GameModelMetadata carried eight metric fields
and PropModelMetadata carried three. Each model populated
only the fields relevant to its task and the rest were
NaN-filled. The asymmetric layout produced confusing CLI
output (Brier displayed as NaN for regression models),
required brittle parallel branches in display code, and
made schema evolution risky.

Consequences:
- Trainers populate only the metrics they actually compute.
- Display surfaces read from ``meta.metrics`` and dispatch
  on ``meta.task``.
- Legacy artifacts written before Unit 9 are migrated
  silently on read via ``_migrate_legacy_metrics``.
- Absent metrics now signal "not recorded" rather than
  "recorded as NaN".
- schema_version is 3.

Out of scope:
- Promotion semantics for regression models. The current
  comparator remains classification-only. Extending
  promotion to regression is its own future workstream.

## D10 - Canonical Elo History Simulator

Date: 2026-06-20

Decision:
A single function is the source of truth for constructing
Elo history from games:

    simulate_elo_history(
        games,
        sorted_years,
        teams_by_year,
        expansion_start,
        ...,
    ) -> EloSimulationResult

The result contains both:
- the (team, year, week) Elo dict consumed by the state
  table builder, and
- the per-game predictions consumed by the tuner and the
  Elo predictor.

Rationale:
The state table builder and the tuner previously each
maintained their own near-identical Elo simulation
loops. The duplication produced a latent bug where one
path silently ignored cfg.divisor. The duplication also
made future Elo updates structurally risky because two
unrelated files had to be edited in lockstep.

Consequences:
- ratings/elo/table.py becomes data-shaping only.
- evaluation/tune.py becomes a thin tuner API around the
  canonical simulator.
- sim/_engine.py and sim/playoffs.py remain untouched
  (numba constraints) and continue to be pinned by
  existing parity tests.
- Future Elo work has a single file to modify.

Note on parity:
The numba kernels in sim/_engine.py and sim/playoffs.py
intentionally maintain their own _elo_win_prob and
_elo_update implementations because numba @njit cannot
call regular Python functions. Parity with the canonical
math is pinned by
tests/unit/ratings/test_elo_core.py::TestPythonNumbaParity.

## D9 - Prop CLI Becomes Archive- and Artifact-Driven

Date: 2026-06-20

Decision:
The prop CLI no longer retrains models inside its
evaluate / champion / projections flows.

- evaluate and champion read from the prop archive via
  build_prop_evaluation_df (Unit 7a).
- projections loads trained artifacts via ArtifactStore.
- A new PropTrainer.train_and_save provides the canonical
  "train and persist" entrypoint for prop models.

Rationale:
Prop CLI commands previously retrained on every call,
which conflated training with evaluation, created
stale-on-arrival outputs, and could not honestly report
historical performance. The post-Unit-5 prop archive
identity model and the Unit 7a canonical evaluation join
provide the foundation needed to make the prop CLI mirror
the game CLI architecture.

Consequences:
- Prop evaluation reports reflect what was actually
  archived, not the result of an ad-hoc retraining.
- Champion selection becomes data-driven.
- Projections require a saved artifact; this is the
  canonical workflow for prop predictions going forward.
- The prop integration spine is complete. Prop and game
  CLI surfaces now share the same operational shape.

## D8 - Prop Walk-Forward Backfill Uses train_through

Date: 2026-06-20

Decision:
Prop backfill is performed via a dedicated walk-forward
training entrypoint:

    PropTrainer.train_through(cutoff_season=...)

The CLI walks the requested season range:

    gridiron props backfill --start-season ... --end-season ...

and archives each season's predictions with the canonical
(model_name, model_type) identity established in Unit 5b.

Rationale:
The previous backfill path produced predictions only for
the holdout window using a model trained on all
non-holdout seasons. That conflated training and
evaluation and caused the archive to under-represent
historical performance. Walk-forward training was the
established game-side approach in Unit 2 and is the only
way to populate a historically honest prop archive.

Consequences:
- The prop archive can now grow honestly across all
  available seasons.
- Future evaluation surfaces (Unit 7a and Unit 7c)
  consume an archive whose semantics match the modelling
  intent.
- Train-time behaviour for the existing prop CLI
  workflows (evaluate / champion / projections) is
  unchanged until Unit 7c.

## D7 - Canonical Prop Evaluation Join

Date: 2026-06-20

Decision:
Prop archive evaluation goes through a single
canonical function:

    build_prop_evaluation_df(
        model_name,
        model_type,
        season,
        repo,
        actuals_df,
    )

Behavior:
- Reads predictions from the prop archive.
- Filters strictly by (model_name, model_type).
- Joins on (game_id, player_id) against an actuals
  DataFrame (injected or built via
  build_prop_features).
- Returns a normalized DataFrame whose actual stat
  column is named `actual` so downstream evaluators
  remain decoupled from per-stat naming.

Rationale:
Evaluation was previously coupled to training. Every
evaluate/champion call retrained the model, which made
honest archive-driven evaluation impossible and
duplicated work across model types. This canonical
join is the foundation for the rest of Unit 7 and any
future archive-driven analytics (CLV, ROI, drift).

Consequences:
- Trainers do not need to know about evaluation.
- Evaluation does not need to know about training.
- The prop CLI can move toward artifact-driven
  workflows in Unit 7c.
- Future analytics surfaces can reuse the same join.

## D6 - Artifact Metadata Uses Explicit `kind` Discriminator

Date: 2026-06-20

Decision:
Model artifact metadata identifies its subclass via an
explicit `kind` field on BaseModelMetadata:

    kind = "game"   # GameModelMetadata
    kind = "prop"   # PropModelMetadata

Rationale:
Previous behavior discriminated subclasses structurally
by the presence of `target_col`. That implicit signal
worked but was fragile: any future field overlap between
GameModelMetadata and PropModelMetadata could silently
break artifact reads.

Consequences:
- New artifacts persist `kind` directly.
- Legacy artifacts written before Unit 6b continue to
  load via a `target_col`-based fallback.
- No data migration is required.
- Future additions to either subclass remain decoupled
  from discrimination logic.

## D5 - Betting Ledger Uses Composite Model Identity

Date: 2026-06-20

Decision:
The bet ledger uses:

    model_name
    model_type

instead of:

    model_version

Rationale:
model_version cannot distinguish algorithm variants (ElasticNet,
RandomForest, XGBoost) of the same prediction family. Composite
identity preserves per-algorithm performance attribution and aligns
the ledger with the rest of the Gridiron Edge model architecture.

Consequences:
- The `gridiron bet log` CLI requires --model-name and --model-type.
- Performance analytics can now distinguish algorithm contributions.
- Old ledger data containing model_version is silently dropped at
  read time.

## D4 - Prop Archive Identity Uses Composite Model Keys

Date: 2026-06-20

Decision:
Prop archive identity is:

    (model_name, model_type)

instead of:

    model_version

Rationale:
model_version could not uniquely distinguish ElasticNet,
RandomForest, and XGBoost variants of the same prop family.
Composite identity preserves algorithm-specific historical
predictions and aligns prop archives with the game archive
architecture.

Deduplication key:

    game_id
    player_id
    stat_type
    model_name
    model_type

## D3 - Canonical Model Architecture

Date: 2026-06-20

Decision:
Adopt:

    Model
    ├── GameModel
    └── PropModel

with ModelRegistry as the canonical registry abstraction.

Rationale:
The previous Predictor / Trainer naming mixed capabilities,
workflows, and domain concepts. The system fundamentally manages
models. Model-based terminology aligns with GameModelMetadata,
PropModelMetadata, registry unification, and future expansion.

## D2 - Retain Unit 1 structural fix despite near-zero observed metric impact

**Date:** 2026-06-19
**Context:** Post-Unit-1 re-baseline outcome (prop_base/C1, prop_base/C2
  from audit_2026_06_18.md). The audit predicted 3-10% MAE inflation
  from holdout-as-validation leakage; the observed re-baseline showed
  metric changes below 1% across all four prop stat families and three
  model types. Unit 1b (game_base/H1, H2) showed similarly small impact
  in smoke tests.

**Decision:** Retain both fixes despite the small observed impact.
The fixes prevent a class of leakage rather than fixing observable
current leakage. They are structural protection against future
regressions, not corrections of current bias.

**Rationale:**

1. **The leakage was real but operationally small.** The audit
   correctly identified holdout-as-validation as a leakage path. The
   reason it produced small metric impact in practice is structural,
   not because the audit was wrong:
   - Prop HP grids are coarse (e.g., ElasticNet has 25 combinations
     across 5 alphas × 5 l1_ratios). Most combinations produce similar
     regularization. The "best on holdout" combination is often also
     "best on TimeSeriesSplit average" because the search space lacks
     resolution.
   - Models with strong regularization (ElasticNet, RF with
     `min_samples_leaf` floor) absorb HP differences as the
     regularizer dominates.
   - Train/holdout distributions are similar enough that what works on
     one works on the other.

2. **The fix is required for forward correctness.** As the codebase
   evolves - more HP combinations, less regularization, broader search
   spaces, new feature sets - the leakage gap may grow. The fix
   prevents this without requiring vigilance.

3. **Auditability.** A future reviewer asking "did you address the
   holdout-leakage finding from the 2026-06-18 audit?" should see a
   yes-this-was-fixed answer, not a yes-but-the-impact-was-small
   answer. The structural fix passes audit; the unmodified code does not.

4. **The cost is small.** TimeSeriesSplit inner CV adds 5× CV folds to
   each HP combination. This is a real cost (visible in Unit 1's
   ~5-hour champion sweep) but is acceptable given the protection it
   provides.

**Implications:**
- The walk-forward backfill in Unit 2 will use the new CV path.
- Any new prop model variant added later inherits the protection
  automatically.
- The audit's "leakage cascade" prediction (downstream metrics
  contaminated by upstream leakage) is now partially refuted: the
  cascade is real architecturally but operationally muted by the
  factors above. CLV analysis and ROI tracking against the new
  archive should be honest within practical noise bounds.

**Revisit triggers:**
- If a future feature set expansion or HP grid expansion shows a larger
  gap between training holdout metrics and live performance, the
  leakage protection becomes more important and any further inner-CV
  expansion (e.g., calibration_cv n_splits) should be revisited.
- If walk-forward backfill produces results materially different from
  the new TimeSeriesSplit baseline, the difference is attributable to
  model-weight leakage (the larger remaining leakage source) rather
  than HP-leakage.

---

## D1 - Walk-forward backfill with fixed hyperparameters

**Date:** 2026-06-19
**Context:** Post-audit remediation, Unit 2 (walk-forward backfill
  infrastructure). Resolves `backfill/C1` from
  `audit_2026_06_18.md`.

**Decision:** Historical predictions in the prediction archive are
generated by walk-forward retraining of model weights, using fixed
hyperparameters from the most recent tune. Intermediate model
artifacts are not persisted.

**Mechanism:**
For each historical season N, the model is retrained on data
strictly through season N-1 using the current spec's hyperparameters,
then used to predict season N. The trained intermediate artifact is
discarded after predictions are written.

**Alternatives considered:**

1. **Use current model for all historical predictions** (status quo):
   Cheapest, but introduces model-weight leakage for in-sample
   seasons. Predictions for season N use a model that saw season N's
   outcomes during training. Rejected - produces leakage-biased
   metrics across the entire historical archive.

2. **Walk-forward weights + walk-forward HP search** (full clean):
   Eliminates all leakage including HP-leakage. Cost is roughly 25×
   higher than fixed-HP walk-forward (~10 days continuous compute vs
   ~10 hours for a full backfill). Rejected - the marginal
   correctness gain over fixed-HP walk-forward is small (HPs are
   properties of data shape, not data content; they vary little
   across tune years), and the compute cost is disproportionate for
   the use case.

3. **Honest naming with `is_in_sample` column** (cheap transparent):
   No retraining; mark in-sample predictions in the archive; rely on
   downstream consumers to filter. Rejected - "remember to filter"
   is exactly the kind of implicit-contract pattern the audit found
   repeatedly. Also produces a permanently shrinking honest-analysis
   window as new seasons are added to training.

**Trade-off accepted:** Mild HP-leakage. Hyperparameters used in
historical retrains were selected with knowledge of the full
dataset including the seasons being predicted. The selected HPs
are roughly the same as those that would have been chosen with
honest walk-forward HP search (HPs are structural properties of
the data; they vary little year-to-year), so the bias is small
and bounded.

**Implications:**
- Metrics computed against the prediction archive are honest
  generalization estimates with respect to model weights.
- Visualizations like "how has model accuracy changed over years"
  produce trustworthy historical context.
- For external claims, regulatory review, or capital allocation
  decisions, full walk-forward HP search would be needed. This
  decision is appropriate for internal product development and
  internal performance tracking.

**Intermediate model persistence:** Not implemented. Trade-off
analysis: persisting intermediate models would enable retrospective
"what would the 2015 model have predicted for this specific game"
analysis, but at the cost of ~25× artifact storage and additional
artifact-management code paths. The predictions themselves are
persisted in the archive, which is sufficient for the use cases
identified (historical metric visualization, calibration analysis,
CLV tracking).

**Revisit triggers:**
- If CLV analysis shows systematic patterns suggesting HP-leakage
  is affecting bet-selection bias, escalate to full walk-forward
  HP search.
- If the codebase moves toward external-facing claims or
  regulatory review, escalate to full walk-forward HP search
  before publication.

---

---

## D22 - Exact-offer Line Shopping is an exhaustive analytical boundary

**Date:** 2026-08-07

**Context:** The current-market product needs to compare every sportsbook quote
without inheriting the recommendation pipeline's one-positive-side selection or
moving model calculations into the frontend.

**Decision:** Line Shopping preserves and evaluates every exact Moneyline,
Spread, and Total quote. An offer is model approved only when its expected value
is strictly greater than zero at its actual line and American price. The backend
owns model probability, expected value, approval, preferred-offer selection,
playable guidance, fair Moneyline prices, and product provenance.

Spread and Total outcome guidance uses -110 as an explicit explanatory reference
price. The resulting playable boundary remains continuous and is not rounded to
a sportsbook increment. Exact offers are always evaluated at their actual price,
so a favorable line at -110 can still be rejected at a worse price and an
unfavorable reference line can be approved at a better price. Spread guidance is
side-oriented for presentation. Maximum-EV approved ties are preserved.

The frontend owns preference persistence, visual presentation, Eastern kickoff
formatting, and deterministic explanations of wager mechanics, pushes, and
American-price stake examples. It does not calculate probability, expected
value, approval, or playable thresholds.

**Implications:** Negative-EV, break-even, unavailable-model, and partially
covered sportsbook offers remain visible. Disabling visual guidance leaves the
raw comparison intact. Current-market comparison remains separate from
arbitrage, middles, line movement, and historical market evaluation.
