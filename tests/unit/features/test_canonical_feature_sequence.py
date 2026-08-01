# tests/unit/features/test_canonical_feature_sequence.py

"""Tests for the canonical game-prediction feature sequence."""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd
from pandas import DataFrame
import pytest

from gridiron_edge.features.pipeline import (
    CANONICAL_FEATURES,
    canonical_feature_columns,
)
from gridiron_edge.features.registry import (
    FeatureRegistry,
    run_features,
    validate_ordering,
)

_EXPECTED_CANONICAL_FEATURES = [
    "home_away_elo",
    "home_away_epa",
    "home_away_rest",
    "home_away_record",
    "home_away_schedule_strength",
    "home_away_travel",
    "home_away_venue_hfa",
    "home_away_divisional",
    "home_away_primetime",
    "home_away_weather",
]


_RETIRED_COLUMNS = {
    "TEAM_A",
    "TEAM_B",
    "HOME_FIELD",
    "RESULT",
}


def test_canonical_sequence_is_exact() -> None:
    assert CANONICAL_FEATURES == _EXPECTED_CANONICAL_FEATURES


def test_all_canonical_features_are_registered() -> None:
    for name in CANONICAL_FEATURES:
        feature_class = FeatureRegistry.get(name)
        assert feature_class().spec.name == name


def test_canonical_sequence_satisfies_dependencies() -> None:
    validate_ordering(CANONICAL_FEATURES)


def test_schedule_strength_runs_after_elo() -> None:
    assert CANONICAL_FEATURES.index("home_away_elo") < CANONICAL_FEATURES.index(
        "home_away_schedule_strength"
    )


def test_declared_output_columns_match_feature_specs() -> None:
    expected = [
        column
        for name in CANONICAL_FEATURES
        for column in FeatureRegistry.get(name)().spec.produces
    ]

    assert canonical_feature_columns() == expected


def test_declared_output_columns_are_unique() -> None:
    columns = canonical_feature_columns()

    assert len(columns) == len(set(columns))


def test_declared_outputs_exclude_retired_orientation_columns() -> None:
    columns = canonical_feature_columns()

    assert _RETIRED_COLUMNS.isdisjoint(columns)
    assert not any(column.startswith("TEAM_A_") for column in columns)
    assert not any(column.startswith("TEAM_B_") for column in columns)


def test_declared_outputs_cover_every_canonical_feature_group() -> None:
    columns = set(canonical_feature_columns())

    expected = {
        "AWAY_ELO",
        "HOME_ELO",
        "AWAY_OFF_EPA_PER_PLAY",
        "HOME_OFF_EPA_PER_PLAY",
        "AWAY_DAYS_REST",
        "HOME_DAYS_REST",
        "DAYS_REST_DIFF",
        "AWAY_WIN_PCT",
        "HOME_WIN_PCT",
        "AWAY_SOS",
        "HOME_SOS",
        "GAME_SITE_ALTITUDE",
        "AWAY_KM_TRAVELED",
        "HOME_KM_TRAVELED",
        "HOME_FRANCHISE_HFA",
        "IS_DIV_GAME",
        "IS_PRIMETIME",
        "IS_DOME",
        "WIND_SPEED_MPH",
        "TEMP_F",
    }

    assert expected <= columns


def test_duplicate_declared_outputs_are_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "gridiron_edge.features.pipeline._feature_columns",
        lambda _names: ["AWAY_ELO", "AWAY_ELO"],
    )

    with pytest.raises(
        ValueError,
        match="Canonical features declare duplicate output columns: AWAY_ELO",
    ):
        canonical_feature_columns()


def _stub_compute(
    outputs: tuple[str, ...],
) -> Callable[..., DataFrame]:
    """Return a feature stub that preserves rows and adds null outputs."""

    def compute(
        _self: object,
        *,
        df: DataFrame,
        datasets: object,
    ) -> DataFrame:
        del datasets
        result = df.copy()
        for column in outputs:
            result[column] = pd.NA
        return result

    return compute


def test_complete_sequence_preserves_canonical_rows_and_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = DataFrame(
        {
            "GAME_ID": ["historical", "upcoming"],
            "YEAR": ["2024-2025", "2025-2026"],
            "WEEK_NUM": [18, 1],
            "AWAY_TEAM": ["Away One", "Away Two"],
            "HOME_TEAM": ["Home One", "Home Two"],
            "GAME_DATE": ["2025-01-05", "2025-09-07"],
            "IS_NEUTRAL_SITE": [0, 0],
            "AWAY_SCORE": [17.0, pd.NA],
            "HOME_SCORE": [24.0, pd.NA],
            "HOME_WIN": [1, pd.NA],
            "ACTUAL_MARGIN": [7.0, pd.NA],
            "ACTUAL_TOTAL": [41.0, pd.NA],
            "MARKER": ["historical", "upcoming"],
        }
    )
    expected = target.copy(deep=True)

    for name in CANONICAL_FEATURES:
        feature_class = FeatureRegistry.get(name)
        outputs = tuple(feature_class().spec.produces)
        monkeypatch.setattr(
            feature_class,
            "compute",
            _stub_compute(outputs),
        )

    result = run_features(
        df=target,
        feature_names=CANONICAL_FEATURES,
        datasets=object(),
    )

    pd.testing.assert_frame_equal(target, expected)
    assert result["GAME_ID"].tolist() == ["historical", "upcoming"]
    assert result["MARKER"].tolist() == ["historical", "upcoming"]
    assert len(result) == 2
    assert result.columns.is_unique
    assert set(canonical_feature_columns()) <= set(result.columns)
    assert _RETIRED_COLUMNS.isdisjoint(result.columns)
    assert result[canonical_feature_columns()].isna().all().all()
