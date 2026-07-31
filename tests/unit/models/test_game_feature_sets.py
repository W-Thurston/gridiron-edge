# tests/unit/models/test_game_feature_sets.py

"""Tests for canonical game-model feature declarations and builders."""

from __future__ import annotations

import pandas as pd
from pandas import DataFrame

from gridiron_edge.models.game_prediction._columns import (
    _COMBINED_FEATURES,
    _DIFF_FEATURES,
    _EPA_SUFFIXES,
    _EXPANDED_FEATURES,
    _GAME_FEATURES,
    _RAW_FEATURES,
    _TEAM_FEATURES,
)
from gridiron_edge.models.game_prediction._features import (
    FEATURE_SETS,
    _make_combined_features,
    _make_diff_features,
    _make_expanded_features,
    _make_raw_features,
)


def _canonical_frame() -> DataFrame:
    """Return one complete canonical feature row with distinct side values."""
    values: dict[str, list[float | int]] = {
        "AWAY_ELO": [1400.0],
        "HOME_ELO": [1600.0],
    }
    for index, suffix in enumerate(_EPA_SUFFIXES, start=1):
        values[f"AWAY_{suffix}"] = [float(index)]
        values[f"HOME_{suffix}"] = [float(index + 100)]

    for index, column in enumerate([*_GAME_FEATURES, *_TEAM_FEATURES], start=1):
        values[column] = [index]

    return DataFrame(values)


def test_feature_declaration_counts_and_uniqueness() -> None:
    expected = {
        "raw": (_RAW_FEATURES, 74),
        "diff": (_DIFF_FEATURES, 37),
        "combined": (_COMBINED_FEATURES, 111),
        "game": (_GAME_FEATURES, 15),
        "team": (_TEAM_FEATURES, 26),
        "expanded": (_EXPANDED_FEATURES, 152),
    }

    for columns, count in expected.values():
        assert len(columns) == count
        assert len(columns) == len(set(columns))


def test_model_features_exclude_retired_orientation() -> None:
    columns = _EXPANDED_FEATURES

    assert "HOME_FIELD" not in columns
    assert "RESULT" not in columns
    assert not any(column == "TEAM_A" or column.startswith("TEAM_A_") for column in columns)
    assert not any(column == "TEAM_B" or column.startswith("TEAM_B_") for column in columns)


def test_canonical_venue_and_rest_contract() -> None:
    assert "GAME_SITE_ALTITUDE" in _GAME_FEATURES
    assert "ALTITUDE" not in _GAME_FEATURES
    assert _TEAM_FEATURES.count("HOME_FRANCHISE_HFA") == 1
    assert "AWAY_FRANCHISE_HFA" not in _TEAM_FEATURES
    assert "DAYS_REST_DIFF" in _TEAM_FEATURES
    assert "TEAM_A_REST_DIFF" not in _TEAM_FEATURES
    assert "TEAM_B_REST_DIFF" not in _TEAM_FEATURES


def test_diff_features_use_home_minus_away() -> None:
    frame = _canonical_frame()

    result = _make_diff_features(frame)

    assert result.columns.tolist() == _DIFF_FEATURES
    assert result.iloc[0]["ELO_DIFF"] == 200.0
    for suffix in _EPA_SUFFIXES:
        assert result.iloc[0][f"{suffix}_DIFF"] == 100.0


def test_raw_features_preserve_declared_order_and_input() -> None:
    frame = _canonical_frame()
    expected = frame.copy(deep=True)

    result = _make_raw_features(frame)

    pd.testing.assert_frame_equal(frame, expected)
    assert result.columns.tolist() == _RAW_FEATURES


def test_combined_features_preserve_declared_order() -> None:
    result = _make_combined_features(_canonical_frame())

    assert result.columns.tolist() == _COMBINED_FEATURES
    assert result.columns.is_unique


def test_expanded_features_include_available_declared_columns() -> None:
    result = _make_expanded_features(_canonical_frame())

    assert result.columns.tolist() == _EXPANDED_FEATURES
    assert result.columns.is_unique


def test_expanded_features_tolerate_unavailable_extended_columns() -> None:
    frame = _canonical_frame().drop(columns=["WIND_SPEED_MPH", "AWAY_SOS"])

    result = _make_expanded_features(frame)

    expected = [
        column for column in _EXPANDED_FEATURES if column not in {"WIND_SPEED_MPH", "AWAY_SOS"}
    ]
    assert result.columns.tolist() == expected


def test_feature_set_keys_and_metadata_names_are_stable() -> None:
    assert list(FEATURE_SETS) == ["diff", "raw", "combined", "expanded"]
    assert FEATURE_SETS["diff"].name == "diff_37"
    assert FEATURE_SETS["raw"].name == "raw_74"
    assert FEATURE_SETS["combined"].name == "combined_111"
    assert FEATURE_SETS["expanded"].name == "expanded_152"


def test_feature_set_names_match_declarations() -> None:
    assert FEATURE_SETS["diff"].feature_names == _DIFF_FEATURES
    assert FEATURE_SETS["raw"].feature_names == _RAW_FEATURES
    assert FEATURE_SETS["combined"].feature_names == _COMBINED_FEATURES
    assert FEATURE_SETS["expanded"].feature_names == _EXPANDED_FEATURES


def test_feature_builders_do_not_mutate_input() -> None:
    frame = _canonical_frame()
    expected = frame.copy(deep=True)

    _make_diff_features(frame)
    _make_raw_features(frame)
    _make_combined_features(frame)
    _make_expanded_features(frame)

    pd.testing.assert_frame_equal(frame, expected)
