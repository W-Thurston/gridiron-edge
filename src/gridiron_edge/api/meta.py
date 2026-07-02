# src/gridiron_edge/api/meta.py
"""Field-status metadata for API responses.

Implements the placeholder convention from DECISIONS.md D14: unpopulated
response fields return `null` accompanied by an optional `_meta.field_status`
entry describing why. Status is either the literal string `"pending"` (backend
work scheduled but not done) or a `BlockedStatus` object naming an upstream
blocker that maps to a ROADMAP.md item.

The `Blocker` class is a registry of stable `(slug, roadmap_ref)` tuples used
across the API surface. Per D16, every Tier 3 route uses a registered slug;
a consistency test enforces this.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field


class BlockedStatus(BaseModel):
    """A field blocked on an upstream workstream or external dependency.

    Attributes:
        status: Discriminator. Always the literal "blocked".
        blocker: Stable slug identifying the upstream gap. Must match a
            value registered in the `Blocker` class.
        roadmap: ROADMAP.md reference (e.g., "§5.3", "W7", "deferred").
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    status: Literal["blocked"] = "blocked"
    blocker: str = Field(description="Stable slug identifying the upstream gap.")
    roadmap: str = Field(description="ROADMAP.md reference for the blocker.")


# Discriminated union: a field's status is either the literal "pending" string
# or a BlockedStatus object. Pydantic v2 discriminates on `status` automatically
# for the object variant; the string variant has no discriminator and matches
# by literal value.
FieldStatus = Annotated[
    Literal["pending"] | BlockedStatus,
    Field(description="Status of an unpopulated field."),
]


class ResponseMeta(BaseModel):
    """Optional `_meta` envelope attached to API responses.

    `field_status` is keyed on dot-notation field paths within the response
    (e.g., "model.home_win_prob", "injuries", "splits.l4"). Granularity is
    field-level per D14.

    Builder methods (`with_pending`, `with_blocked`) return new instances
    rather than mutating in place, consistent with `frozen=True`.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    field_status: dict[str, FieldStatus] = Field(
        default_factory=dict,
        description="Map from field path to status (field-level per D14).",
    )

    def with_pending(self, field_path: str) -> ResponseMeta:
        """Return a new `ResponseMeta` with `field_path` marked pending."""
        new_map = {**self.field_status, field_path: "pending"}
        return ResponseMeta(field_status=new_map)

    def with_blocked(
        self,
        field_path: str,
        blocker: str,
        roadmap: str,
    ) -> ResponseMeta:
        """Return a new `ResponseMeta` with `field_path` marked blocked."""
        new_map = {
            **self.field_status,
            field_path: BlockedStatus(blocker=blocker, roadmap=roadmap),
        }
        return ResponseMeta(field_status=new_map)


class Blocker:
    """Registry of stable blocker slugs.

    Each attribute is a `(slug, roadmap_ref)` tuple. Use with the splat
    operator at call sites::

        meta = ResponseMeta().with_blocked("injuries", *Blocker.INJURY_DATA)

    The `all_slugs()` classmethod returns the set of registered slugs for
    consistency tests.

    Each entry maps to a row in ROADMAP.md §9.5 (Backend gaps surfaced by
    the prototype). Adding a new blocker means adding an entry there too.
    """

    INJURY_DATA: tuple[str, str] = ("injury_data_source", "§5.3")
    MULTI_BOOK: tuple[str, str] = ("multi_book_ingest", "W7")
    LIVE_STATE: tuple[str, str] = ("live_state_ingest", "W10")
    SCENARIO_ENGINE: tuple[str, str] = ("scenario_engine", "W4.5")
    FEATURE_ATTRIBUTION: tuple[str, str] = ("feature_attribution", "deferred")
    COMPARABLES: tuple[str, str] = ("comparables_retrieval", "deferred")
    HISTORICAL_LINES: tuple[str, str] = ("historical_line_movement", "W7")
    GAMEDAY_METADATA: tuple[str, str] = ("gameday_metadata", "deferred")
    NEWS_INGEST: tuple[str, str] = ("news_ingest", "deferred")
    WAR: tuple[str, str] = ("war_computation", "deferred")

    @classmethod
    def all_slugs(cls: type[Blocker]) -> frozenset:
        """Return every registered blocker slug.

        Used by the consistency test that asserts every Tier 3 route
        references a registered slug.
        """
        return frozenset(
            value[0]
            for name, value in vars(cls).items()
            if not name.startswith("_") and isinstance(value, tuple) and len(value) == 2
        )


class Unavailable:
    """Slugs for fields that are null because the source data doesn't support them.

    Distinct from `Blocker` (which points at an upstream workstream) and
    from `"pending"` (which means backend work is in progress). Fields
    marked with these slugs are null for a specific request because the
    underlying data lacks what's needed — a state that may resolve
    naturally as more bets accumulate, more CLV data lands, etc.
    """

    NO_CHAMPION_MANIFEST: tuple[str, str] = ("no_champion_manifest", "data")
    NO_CLV_DATA: tuple[str, str] = ("no_clv_data", "data")
    NO_EVALUATION_DATA: tuple[str, str] = ("no_evaluation_data", "data")
    NO_MODEL_CONTEXT: tuple[str, str] = ("no_model_context", "data")
    NO_ODDS_AVAILABLE: tuple[str, str] = ("no_odds_available", "data")
    NO_PRIOR_SNAPSHOT: tuple[str, str] = ("no_prior_snapshot", "data")
    NO_PROJECTIONS_DATA: tuple[str, str] = ("no_projections_data", "data")
    NO_STREAK_ACTIVITY: tuple[str, str] = ("no_streak_activity", "data")
    OFF_DEF_DECOMPOSITION: tuple[str, str] = ("off_def_decomposition", "data")
    PERIOD_NOT_REQUESTED: tuple[str, str] = ("period_not_requested", "request")
    SINGLE_CLASS_OUTCOME: tuple[str, str] = ("single_class_outcome", "data")

    @classmethod
    def all_slugs(cls: type[Unavailable]) -> frozenset:
        """Return every registered unavailable slug."""
        return frozenset(
            value[0]
            for name, value in vars(cls).items()
            if not name.startswith("_") and isinstance(value, tuple) and len(value) == 2
        )
