# src/gridiron_edge/api/schemas/explain.py
"""Schemas for per-game explainability endpoints (/games/{game_id}/explain).

`factors` serializes persisted Logistic explanation evidence (exact
scaled-feature-by-coefficient contributions in log-odds space) when it
exists for the game's selected Win forecast. `band`, `distribution`, and
`market_implied` remain null with structured `_meta.field_status` entries;
they are blocked on the scenario engine.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from gridiron_edge.api.schemas._base import BaseResponse


class ExplainFactor(BaseModel):
    """One feature's exact log-odds contribution to the headline prediction.

    A contribution is a linear decomposition of the fitted estimator's own
    decision function (`coefficient * transformed_value`), not a causal
    effect estimate. The intercept is represented as its own factor with
    `is_baseline=True` and `coefficient`/`transformed_value` unset.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    key: str | None = Field(default=None, description="Stable factor identifier.")
    label: str | None = Field(default=None, description="Human-readable label.")
    log_odds_contribution: float | None = Field(
        default=None,
        description="Contribution to the headline prediction, in log-odds space.",
    )
    coefficient: float | None = Field(
        default=None,
        description="The fitted estimator's coefficient for this feature.",
    )
    transformed_value: float | None = Field(
        default=None,
        description="The scaled feature value the coefficient was applied to.",
    )
    is_baseline: bool | None = Field(default=None)
    is_adjustable: bool | None = Field(default=None)


class CredibleBand(BaseModel):
    """90% credible interval around the headline probability."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    point: float | None = Field(default=None)
    lo: float | None = Field(default=None)
    hi: float | None = Field(default=None)


class ExplainDistribution(BaseModel):
    """Simulated outcome distribution for the explain view."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    samples: int | None = Field(default=None, description="Number of sims (e.g. 2000).")
    mean_margin: float | None = Field(
        default=None,
        description="Expected margin in points (home - away).",
    )
    sd: float | None = Field(default=None, description="Standard deviation of margin.")


class GameExplain(BaseResponse):
    """Response for GET /games/{game_id}/explain."""

    game_id: str
    headline_win_prob: float | None = Field(default=None)
    band: CredibleBand | None = Field(default=None)
    factors: list[ExplainFactor] | None = Field(default=None)
    distribution: ExplainDistribution | None = Field(default=None)
    market_implied: float | None = Field(
        default=None,
        description="De-vigged market implied prob for the favored side.",
    )
