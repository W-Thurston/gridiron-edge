# tests/unit/api/test_schemas_injuries.py

"""Unit tests for game-injuries response schemas."""

from __future__ import annotations

from pydantic import ValidationError
import pytest

from gridiron_edge.api.meta import BlockedStatus, Blocker, ResponseMeta
from gridiron_edge.api.schemas.injuries import GameInjuries, InjuryReport


class TestGameInjuriesConstruction:
    def test_minimal(self) -> None:
        injuries = GameInjuries(game_id="sf-bal")
        assert injuries.game_id == "sf-bal"
        assert injuries.response_meta is None

    def test_with_meta(self) -> None:
        meta = ResponseMeta().with_blocked("reports", *Blocker.INJURY_DATA)
        injuries = GameInjuries(game_id="sf-bal", response_meta=meta)
        status = injuries.response_meta.field_status["reports"]
        assert isinstance(status, BlockedStatus)
        assert status.blocker == "injury_data_source"
        assert status.roadmap == "injury data source"

    def test_meta_serializes_with_wire_alias(self) -> None:
        meta = ResponseMeta().with_blocked("reports", *Blocker.INJURY_DATA)
        injuries = GameInjuries(game_id="sf-bal", response_meta=meta)
        dumped = injuries.model_dump(by_alias=True)
        assert "_meta" in dumped


class TestGameInjuriesStrict:
    def test_rejects_unknown_fields(self) -> None:
        with pytest.raises(ValidationError):
            GameInjuries(game_id="sf-bal", unexpected="x")

    def test_is_frozen(self) -> None:
        injuries = GameInjuries(game_id="sf-bal")
        with pytest.raises(ValidationError):
            injuries.game_id = "other"


class TestInjuryReport:
    def test_default(self) -> None:
        assert InjuryReport() is not None

    def test_populated(self) -> None:
        report = InjuryReport(
            team="SF",
            player="C. McCaffrey",
            position="RB",
            status="Questionable",
            note="Knee · DNP Wed/Thu",
        )
        assert report.player == "C. McCaffrey"
        assert report.status == "Questionable"

    def test_is_frozen(self) -> None:
        report = InjuryReport()
        with pytest.raises(ValidationError):
            report.team = "BAL"

    def test_rejects_unknown_fields(self) -> None:
        with pytest.raises(ValidationError):
            InjuryReport(unexpected="x")


class TestGameInjuriesComposition:
    def test_holds_reports(self) -> None:
        injuries = GameInjuries(
            game_id="sf-bal",
            reports=[
                InjuryReport(team="SF", player="C. McCaffrey", status="Questionable"),
                InjuryReport(team="SF", player="N. Bosa", status="OUT"),
            ],
        )
        assert len(injuries.reports) == 2
        assert injuries.reports[1].status == "OUT"
