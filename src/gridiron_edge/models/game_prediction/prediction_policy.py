# src/gridiron_edge/models/game_prediction/prediction_policy.py

"""Availability-aware game prediction model policy."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class PredictionModelStatus(StrEnum):
    """Whether one prediction family can be produced."""

    SELECTED = "selected"
    UNAVAILABLE = "unavailable"


class PredictionPolicyRationale(StrEnum):
    """Machine-readable reason for one model decision."""

    CHAMPION_ELIGIBLE = "champion_eligible"
    OVERRIDE_ELIGIBLE = "override_eligible"
    ELO_ONLY_AVAILABLE = "elo_only_available"
    REQUIRED_INPUTS_UNAVAILABLE = "required_inputs_unavailable"
    CHAMPION_UNAVAILABLE = "champion_unavailable"
    OVERRIDE_INELIGIBLE = "override_ineligible"


class PredictionModelSource(StrEnum):
    """Origin of one selected model identity."""

    CHAMPION = "champion"
    OVERRIDE = "override"
    POLICY = "policy"


MetricValue = int | float | str | bool | None


@dataclass(frozen=True)
class PredictionAvailability:
    """Explicit input availability for one requested week."""

    season: str
    week: int
    elo_available: bool
    full_features_available: bool
    total_features_available: bool

    def __post_init__(self) -> None:
        """Validate prediction scope."""
        if not self.season.strip():
            raise ValueError("season must not be empty.")
        if self.week < 1:
            raise ValueError("week must be at least 1.")

    def to_dict(self) -> dict[str, object]:
        """Return a stable serialization representation."""
        return {
            "season": self.season,
            "week": self.week,
            "elo_available": self.elo_available,
            "full_features_available": self.full_features_available,
            "total_features_available": self.total_features_available,
        }


@dataclass(frozen=True)
class ModelProvenance:
    """Serializable model-selection provenance."""

    model_name: str
    model_type: str
    source: PredictionModelSource
    promoted_at: str | None = None
    source_run_id: str | None = None
    metrics: tuple[tuple[str, MetricValue], ...] = ()

    def __post_init__(self) -> None:
        """Validate model identity and optional metadata."""
        if not self.model_name.strip():
            raise ValueError("model_name must not be empty.")
        if not self.model_type.strip():
            raise ValueError("model_type must not be empty.")
        if self.promoted_at is not None and not self.promoted_at.strip():
            raise ValueError("promoted_at must not be empty when provided.")
        if self.source_run_id is not None and not self.source_run_id.strip():
            raise ValueError("source_run_id must not be empty when provided.")

    @classmethod
    def from_champion_entry(
        cls,
        *,
        model_name: str,
        entry: dict[str, Any],
    ) -> ModelProvenance:
        """Build provenance from one champion manifest entry."""
        model_type = str(entry.get("model_type", "")).strip()
        if not model_type:
            raise ValueError("Champion entry must contain model_type.")

        raw_metrics = entry.get("metrics", {})
        if not isinstance(raw_metrics, dict):
            raise ValueError("Champion metrics must be a mapping.")

        metrics: tuple[tuple[str, MetricValue], ...] = tuple(
            sorted((str(key), value) for key, value in raw_metrics.items())
        )
        promoted_at_value = entry.get("promoted_at")
        source_run_id_value = entry.get("source_run_id")

        return cls(
            model_name=model_name,
            model_type=model_type,
            source=PredictionModelSource.CHAMPION,
            promoted_at=(None if promoted_at_value is None else str(promoted_at_value)),
            source_run_id=(None if source_run_id_value is None else str(source_run_id_value)),
            metrics=metrics,
        )

    @classmethod
    def override(
        cls,
        *,
        model_name: str,
        model_type: str,
    ) -> ModelProvenance:
        """Build provenance for an explicit override."""
        return cls(
            model_name=model_name,
            model_type=model_type,
            source=PredictionModelSource.OVERRIDE,
        )

    @classmethod
    def policy(
        cls,
        *,
        model_name: str,
        model_type: str,
    ) -> ModelProvenance:
        """Build provenance for a policy-owned selection."""
        return cls(
            model_name=model_name,
            model_type=model_type,
            source=PredictionModelSource.POLICY,
        )

    def to_dict(self) -> dict[str, object]:
        """Return a stable serialization representation."""
        return {
            "model_name": self.model_name,
            "model_type": self.model_type,
            "source": self.source.value,
            "promoted_at": self.promoted_at,
            "source_run_id": self.source_run_id,
            "metrics": dict(self.metrics),
        }


@dataclass(frozen=True)
class PredictionModelDecision:
    """Selection result for one prediction family."""

    model_name: str
    model_type: str | None
    status: PredictionModelStatus
    rationale: PredictionPolicyRationale
    explanation: str
    provenance: ModelProvenance | None

    def __post_init__(self) -> None:
        """Validate decision-state invariants."""
        if not self.model_name.strip():
            raise ValueError("model_name must not be empty.")
        if not self.explanation.strip():
            raise ValueError("explanation must not be empty.")

        if self.status is PredictionModelStatus.SELECTED:
            if self.model_type is None:
                raise ValueError("Selected decision requires model_type.")
            if not self.model_type.strip():
                raise ValueError("Selected model_type must not be empty.")
            if self.provenance is None:
                raise ValueError("Selected decision requires provenance.")
            if self.provenance.model_name != self.model_name:
                raise ValueError("Decision and provenance model_name must match.")
            if self.provenance.model_type != self.model_type:
                raise ValueError("Decision and provenance model_type must match.")
            return

        if self.model_type is not None:
            raise ValueError("Unavailable decision must not contain model_type.")
        if self.provenance is not None:
            raise ValueError("Unavailable decision must not contain provenance.")

    def to_dict(self) -> dict[str, object]:
        """Return a stable serialization representation."""
        return {
            "model_name": self.model_name,
            "model_type": self.model_type,
            "status": self.status.value,
            "rationale": self.rationale.value,
            "explanation": self.explanation,
            "provenance": (None if self.provenance is None else self.provenance.to_dict()),
        }


@dataclass(frozen=True)
class PredictionPolicy:
    """Availability-aware policy for win and total predictions."""

    availability: PredictionAvailability
    win: PredictionModelDecision
    total: PredictionModelDecision

    def __post_init__(self) -> None:
        """Validate independent family identities."""
        if self.win.model_name != "win_prob":
            raise ValueError("win decision must use model_name 'win_prob'.")
        if self.total.model_name != "total":
            raise ValueError("total decision must use model_name 'total'.")

    def to_dict(self) -> dict[str, object]:
        """Return a stable serialization representation."""
        return {
            "availability": self.availability.to_dict(),
            "win": self.win.to_dict(),
            "total": self.total.to_dict(),
        }


def _selected_decision(
    provenance: ModelProvenance,
    *,
    rationale: PredictionPolicyRationale,
    explanation: str,
) -> PredictionModelDecision:
    """Create one valid selected decision."""
    return PredictionModelDecision(
        model_name=provenance.model_name,
        model_type=provenance.model_type,
        status=PredictionModelStatus.SELECTED,
        rationale=rationale,
        explanation=explanation,
        provenance=provenance,
    )


def _unavailable_decision(
    *,
    model_name: str,
    rationale: PredictionPolicyRationale,
    explanation: str,
) -> PredictionModelDecision:
    """Create one valid unavailable decision."""
    return PredictionModelDecision(
        model_name=model_name,
        model_type=None,
        status=PredictionModelStatus.UNAVAILABLE,
        rationale=rationale,
        explanation=explanation,
        provenance=None,
    )


def _resolve_win_decision(  # noqa: PLR0911
    availability: PredictionAvailability,
    *,
    champion: ModelProvenance | None,
    override: str | None,
) -> PredictionModelDecision:
    """Resolve the win-probability family independently."""
    if override is not None:
        normalized_override = override.strip()
        if not normalized_override:
            raise ValueError("win_override must not be empty when provided.")

        if normalized_override == "elo":
            if availability.elo_available:
                return _selected_decision(
                    ModelProvenance.override(
                        model_name="win_prob",
                        model_type="elo",
                    ),
                    rationale=PredictionPolicyRationale.OVERRIDE_ELIGIBLE,
                    explanation=(
                        "Explicit Elo override is eligible because Elo state is available."
                    ),
                )
            return _unavailable_decision(
                model_name="win_prob",
                rationale=PredictionPolicyRationale.OVERRIDE_INELIGIBLE,
                explanation=(
                    "Explicit Elo override is ineligible because Elo state is unavailable."
                ),
            )

        if availability.full_features_available:
            return _selected_decision(
                ModelProvenance.override(
                    model_name="win_prob",
                    model_type=normalized_override,
                ),
                rationale=PredictionPolicyRationale.OVERRIDE_ELIGIBLE,
                explanation=(
                    "Explicit win-model override is eligible because full "
                    "prediction features are available."
                ),
            )

        return _unavailable_decision(
            model_name="win_prob",
            rationale=PredictionPolicyRationale.OVERRIDE_INELIGIBLE,
            explanation=(
                "Explicit win-model override is ineligible because full "
                "prediction features are unavailable."
            ),
        )

    if champion is not None and availability.full_features_available:
        return _selected_decision(
            champion,
            rationale=PredictionPolicyRationale.CHAMPION_ELIGIBLE,
            explanation=(
                "The win-probability champion is eligible because full "
                "prediction features are available."
            ),
        )

    if availability.elo_available:
        return _selected_decision(
            ModelProvenance.policy(
                model_name="win_prob",
                model_type="elo",
            ),
            rationale=PredictionPolicyRationale.ELO_ONLY_AVAILABLE,
            explanation=("Full-feature win prediction is unavailable; Elo state is available."),
        )

    if champion is None:
        return _unavailable_decision(
            model_name="win_prob",
            rationale=PredictionPolicyRationale.CHAMPION_UNAVAILABLE,
            explanation=("No win-probability champion is available and Elo state is unavailable."),
        )

    return _unavailable_decision(
        model_name="win_prob",
        rationale=PredictionPolicyRationale.REQUIRED_INPUTS_UNAVAILABLE,
        explanation=(
            "The win-probability champion exists, but its required full "
            "prediction features and Elo fallback are unavailable."
        ),
    )


def _resolve_total_decision(
    availability: PredictionAvailability,
    *,
    champion: ModelProvenance | None,
    override: str | None,
) -> PredictionModelDecision:
    """Resolve the total-prediction family independently."""
    if override is not None:
        normalized_override = override.strip()
        if not normalized_override:
            raise ValueError("total_override must not be empty when provided.")

        if availability.total_features_available:
            return _selected_decision(
                ModelProvenance.override(
                    model_name="total",
                    model_type=normalized_override,
                ),
                rationale=PredictionPolicyRationale.OVERRIDE_ELIGIBLE,
                explanation=(
                    "Explicit total-model override is eligible because "
                    "total features are available."
                ),
            )

        return _unavailable_decision(
            model_name="total",
            rationale=PredictionPolicyRationale.OVERRIDE_INELIGIBLE,
            explanation=(
                "Explicit total-model override is ineligible because "
                "total features are unavailable."
            ),
        )

    if champion is not None and availability.total_features_available:
        return _selected_decision(
            champion,
            rationale=PredictionPolicyRationale.CHAMPION_ELIGIBLE,
            explanation=("The total champion is eligible because total features are available."),
        )

    if champion is None:
        return _unavailable_decision(
            model_name="total",
            rationale=PredictionPolicyRationale.CHAMPION_UNAVAILABLE,
            explanation="No total champion is available.",
        )

    return _unavailable_decision(
        model_name="total",
        rationale=PredictionPolicyRationale.REQUIRED_INPUTS_UNAVAILABLE,
        explanation=("The total champion exists, but its required features are unavailable."),
    )


def resolve_prediction_policy(
    availability: PredictionAvailability,
    *,
    win_champion: ModelProvenance | None,
    total_champion: ModelProvenance | None,
    win_override: str | None = None,
    total_override: str | None = None,
) -> PredictionPolicy:
    """Resolve independent win and total model decisions.

    Availability is supplied explicitly. This function performs no dataset
    loading, feature generation, champion-manifest reading, model execution,
    filesystem access, date lookup, or API inference.
    """
    win = _resolve_win_decision(
        availability,
        champion=win_champion,
        override=win_override,
    )
    total = _resolve_total_decision(
        availability,
        champion=total_champion,
        override=total_override,
    )
    return PredictionPolicy(
        availability=availability,
        win=win,
        total=total,
    )
