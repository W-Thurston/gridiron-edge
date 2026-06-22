# tests/unit/ingest/test_draftkings_parse.py
"""Tests for gridiron_edge.ingest.odds.draftkings - odds normalization."""

from __future__ import annotations

from gridiron_edge.ingest.odds.draftkings import _norm_display_odds_american


class TestNormDisplayOddsAmerican:
    def test_ascii_negative_odds(self) -> None:
        sel: dict[str, dict[str, str]] = {"displayOdds": {"oddsAmerican": "-150"}}
        assert _norm_display_odds_american(sel) == -150

    def test_unicode_minus_odds(self) -> None:
        """DK sometimes returns U+2212 (−) instead of ASCII hyphen (-)."""  # noqa: RUF002
        sel: dict[str, dict[str, str]] = {"displayOdds": {"oddsAmerican": "\u2212150"}}
        assert _norm_display_odds_american(sel) == -150

    def test_positive_odds(self) -> None:
        sel: dict[str, dict[str, str]] = {"displayOdds": {"oddsAmerican": "130"}}
        assert _norm_display_odds_american(sel) == 130

    def test_int_passthrough(self) -> None:
        sel: dict[str, dict[str, int]] = {"displayOdds": {"oddsAmerican": -110}}
        assert _norm_display_odds_american(sel) == -110

    def test_float_coerced_to_int(self) -> None:
        sel: dict[str, dict[str, float]] = {"displayOdds": {"oddsAmerican": -110.0}}
        assert _norm_display_odds_american(sel) == -110

    def test_none_when_no_odds_key(self) -> None:
        sel: dict[str, dict] = {"displayOdds": {}}
        assert _norm_display_odds_american(sel) is None

    def test_none_when_no_display_odds(self) -> None:
        sel: dict = {}
        assert _norm_display_odds_american(sel) is None

    def test_fallback_key_price(self) -> None:
        sel: dict[str, dict[str, str]] = {"displayOdds": {"price": "-200"}}
        assert _norm_display_odds_american(sel) == -200

    def test_non_numeric_string_returned_as_is(self) -> None:
        sel: dict[str, dict[str, str]] = {"displayOdds": {"oddsAmerican": "EVEN"}}
        result: int | str | None = _norm_display_odds_american(sel)
        assert result == "EVEN"
