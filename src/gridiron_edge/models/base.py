# src/gridiron_edge/models/base.py

"""Base types for the prediction model layer.

Defines the ``Predictor`` protocol that all prediction models must satisfy.
Any class with a ``spec`` attribute and a ``predict_historical`` method
matching this signature is a valid ``Predictor`` without explicit
inheritance — structural subtyping via Protocol.

This mirrors the ``Feature`` protocol in ``gridiron_edge.features.base``
and enables the evaluation framework to treat all models uniformly:

  - ``gridiron evaluate backfill --model-version elo_v1``
  - ``gridiron evaluate backfill --model-version logistic_v1``
  - ``gridiron evaluate backfill --model-version neural_v1``

Adding a new model requires only:
  1. Implementing this protocol
  2. Registering it with ``PredictorRegistry``
  3. Zero changes to evaluation, archiving, or CLI infrastructure
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from pathlib import Path

    import pandas as pd


@dataclass(frozen=True)
class PredictorSpec:
    """Metadata describing a predictor's identity.

    Attributes:
        name: Unique string key used to register and look up this predictor.
            Should follow the convention ``{model_type}_v{version}``,
            e.g. ``"elo_v1"``, ``"logistic_v1"``, ``"neural_v1"``.
        description: Human-readable description shown in CLI help and
            evaluation output.
    """

    name: str
    description: str


class Predictor(Protocol):
    """Protocol defining the interface all prediction models must satisfy.

    Any class with a ``spec`` attribute and a ``predict_historical`` method
    matching this signature is a valid ``Predictor`` without explicit
    inheritance.

    The key design constraint: ``predict_historical`` must be able to
    generate predictions for *every* historical game from scratch — not
    just the current week. This is required for the evaluation backfill
    to work correctly across all models.

    For models with a training step (logistic regression, neural networks),
    ``predict_historical`` should load a pre-trained artifact. The training
    step itself lives in ``models/{model_type}/train.py`` and is invoked
    separately via ``gridiron models train``.

    Attributes:
        spec: A ``PredictorSpec`` describing the predictor's identity.
    """

    spec: PredictorSpec

    def predict_historical(
        self,
        games: pd.DataFrame,
        *,
        repo: Path | None = None,
    ) -> pd.DataFrame:
        """Generate predictions for all historical games.

        Args:
            games: Canonical games DataFrame (``NFL_wk_by_wk_cleaned.csv``).
                Contains completed games only — rows with null WIN_OR_TIE
                have been filtered out by the caller.
            repo: Repository root path for loading auxiliary data (Elo state,
                trained model artifacts, etc.).

        Returns:
            DataFrame in prediction archive schema with columns:
                predicted_at, model_version, season, week, game_id,
                game_date, away_team, home_team, away_elo, home_elo,
                away_win_prob, home_win_prob.

            One row per game. Games where the model cannot produce a
            prediction (missing features, insufficient history) should be
            silently excluded rather than returning NaN probabilities.
        """
        ...

    def predict_upcoming(
        self,
        schedule: pd.DataFrame,
        *,
        repo: Path | None = None,
    ) -> pd.DataFrame:
        """Generate predictions for upcoming (unplayed) games.

        Args:
            schedule: Canonical upcoming schedule DataFrame.
            repo: Repository root path.

        Returns:
            DataFrame compatible with ``build_predictions_df()`` output:
                GAME_ID, GAME_DATE, AWAY_TEAM, HOME_TEAM,
                AWAY_TEAM_ELO, HOME_TEAM_ELO,
                AWAY_WIN_PROB, HOME_WIN_PROB,
                AWAY_TEAM_WIN_PROB, HOME_TEAM_WIN_PROB.

            The AWAY_TEAM_ELO / HOME_TEAM_ELO columns may be NaN for
            non-Elo models — they are used for display only.
        """
        ...
