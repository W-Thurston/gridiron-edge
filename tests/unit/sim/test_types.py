# tests/unit/sim/test_types.py
"""Tests for gridiron_edge.sim._types - constants and dataclasses."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from gridiron_edge.sim._types import (
    AWAY_WIN,
    CONF_CODES,
    DIV_CODES,
    HOME_WIN,
    N_PLAYOFF_ROUNDS,
    N_TEAMS,
    N_WEEKS_REG,
    ROUND_CONF,
    ROUND_DIV,
    ROUND_SB,
    ROUND_WC,
    TIE,
    UNPLAYED,
    SimulationConfig,
    format_record,
)


class TestSimConstants:
    def test_n_teams_is_32(self) -> None:
        assert N_TEAMS == 32

    def test_n_weeks_reg_is_18(self) -> None:
        assert N_WEEKS_REG == 18

    def test_n_playoff_rounds_is_4(self) -> None:
        assert N_PLAYOFF_ROUNDS == 4


class TestGameOutcomeEncodings:
    def test_unplayed_is_neg_1(self) -> None:
        assert np.int8(-1) == UNPLAYED

    def test_away_win_is_0(self) -> None:
        assert np.int8(0) == AWAY_WIN

    def test_home_win_is_1(self) -> None:
        assert np.int8(1) == HOME_WIN

    def test_tie_is_2(self) -> None:
        assert np.int8(2) == TIE

    def test_all_distinct(self) -> None:
        values: set[int] = {int(UNPLAYED), int(AWAY_WIN), int(HOME_WIN), int(TIE)}
        assert len(values) == 4


class TestPlayoffRounds:
    def test_round_order(self) -> None:
        assert ROUND_WC < ROUND_DIV < ROUND_CONF < ROUND_SB

    def test_wildcard_is_0(self) -> None:
        assert ROUND_WC == 0

    def test_super_bowl_is_3(self) -> None:
        assert ROUND_SB == 3


class TestDivisionAndConferenceCodes:
    def test_8_divisions(self) -> None:
        assert len(DIV_CODES) == 8

    def test_2_conferences(self) -> None:
        assert len(CONF_CODES) == 2

    def test_afc_and_nfc_present(self) -> None:
        assert "AFC" in CONF_CODES
        assert "NFC" in CONF_CODES

    def test_division_names_contain_conference(self) -> None:
        for div_name in DIV_CODES:
            assert "AFC" in div_name or "NFC" in div_name

    def test_division_codes_are_unique(self) -> None:
        codes: list[int] = list(DIV_CODES.values())
        assert len(codes) == len(set(codes))


class TestSimulationConfig:
    def test_is_frozen(self) -> None:
        config = SimulationConfig(
            n_sims=1000,
            k_factor=20.0,
            divisor=400.0,
            p_tie=0.005,
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            config.n_sims = 500  # type: ignore[misc]


class TestFormatRecord:
    def test_no_ties(self) -> None:
        # 10 points, 6 games → 5W-1L
        result: str = format_record(pts=10, gp=6)
        assert result == "5-1"

    def test_with_tie(self) -> None:
        # 9 points, 6 games → 4W-1L-1T
        result: str = format_record(pts=9, gp=6)
        assert result == "4-1-1"

    def test_all_wins(self) -> None:
        result: str = format_record(pts=6, gp=3)
        assert result == "3-0"

    def test_all_losses(self) -> None:
        result: str = format_record(pts=0, gp=5)
        assert result == "0-5"
