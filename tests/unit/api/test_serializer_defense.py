# tests/unit/api/test_serializer_defense.py

"""Unit tests for the defense-allowed serializer."""

from __future__ import annotations


def test_populated_cohorts() -> None:
    from gridiron_edge.api.serializers.defense import serialize_defense_allowed

    cohorts = {
        "season": {"avg_allowed": 57.8, "sample_size": 20, "rank_against_position": 21},
        "l4": {"avg_allowed": 46.25, "sample_size": 4, "rank_against_position": 10},
        "home": {"avg_allowed": 63.0, "sample_size": 12, "rank_against_position": 24},
        "away": {"avg_allowed": 50.0, "sample_size": 8, "rank_against_position": 12},
    }
    result = serialize_defense_allowed(
        team="NE", position="TE", stat_type="te_rec_yards", cohorts=cohorts
    )
    assert result.team == "NE"
    assert result.position == "TE"
    assert result.cohorts is not None
    assert result.cohorts["season"]["avg_allowed"] == 57.8
    assert result.response_meta is None


def test_empty_cohorts_marks_blocked() -> None:
    from gridiron_edge.api.serializers.defense import serialize_defense_allowed

    result = serialize_defense_allowed(
        team="NE", position="TE", stat_type="te_rec_yards", cohorts={}
    )
    assert result.cohorts is None
    assert result.response_meta is not None
    assert "cohorts" in result.response_meta.field_status
