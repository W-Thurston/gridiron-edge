# tests/integration/test_archive_roundtrip.py

"""Integration tests for the prediction archive layer.

Exercises archive read/write semantics without involving a model. Tests
the dedup key, schema preservation, and round-trip fidelity for both
game and prop archives.

Marked ``@pytest.mark.integration``. Runtime: under 5 seconds.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest
from tests.fixtures.dataframes import make_predictions
from tests.fixtures.helpers import assert_archive_schema_valid

from gridiron_edge.evaluation.archive import (
    _ARCHIVE_COLUMNS,
    append_to_prediction_log,
    load_prediction_log,
)

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def archive_repo(tmp_path: Path) -> Path:
    """Empty repo with the predictions directory pre-created.

    Function-scoped because each test wants a fresh archive — sharing state
    across tests would obscure dedup behavior.
    """
    (tmp_path / "data" / "output" / "predictions").mkdir(parents=True)
    return tmp_path


# ---------------------------------------------------------------------------
# Test 1: Round-trip preserves schema and content
# ---------------------------------------------------------------------------


class TestArchiveRoundTrip:
    """Write predictions, read them back, verify identical data and schema."""

    def test_writes_then_reads_back(self, archive_repo: Path) -> None:
        predictions = make_predictions(n=5)

        append_to_prediction_log(
            predictions,
            model_name="win_prob",
            model_type="logistic",
            season="2025-2026",
            week=1,
            repo=archive_repo,
        )

        log = load_prediction_log(repo=archive_repo)

        assert len(log) == 5
        assert (log["model_name"] == "win_prob").all()
        assert (log["model_type"] == "logistic").all()
        assert (log["season"] == "2025-2026").all()
        assert (log["week"] == 1).all()

    def test_round_trip_preserves_schema(self, archive_repo: Path) -> None:
        predictions = make_predictions(n=3)

        append_to_prediction_log(
            predictions,
            model_name="win_prob",
            model_type="logistic",
            season="2025-2026",
            week=1,
            repo=archive_repo,
        )

        log = load_prediction_log(repo=archive_repo)
        assert_archive_schema_valid(log)

        # Every canonical column should be present
        for col in _ARCHIVE_COLUMNS:
            assert col in log.columns, f"missing archive column: {col}"

    def test_predictions_data_preserved(self, archive_repo: Path) -> None:
        """Numeric prediction columns survive parquet round-trip exactly."""
        predictions = make_predictions(n=3)
        # Custom probability values to verify exact preservation
        predictions["AWAY_WIN_PROB"] = [0.42, 0.61, 0.27]
        predictions["HOME_WIN_PROB"] = [0.58, 0.39, 0.73]

        append_to_prediction_log(
            predictions,
            model_name="win_prob",
            model_type="logistic",
            season="2025-2026",
            week=1,
            repo=archive_repo,
        )

        log = load_prediction_log(repo=archive_repo)
        # Sort by game_id to handle order changes during parquet round-trip
        log_sorted = log.sort_values("game_id").reset_index(drop=True)

        # The archive uses lowercase column names — verify those values
        assert log_sorted["away_win_prob"].tolist() == pytest.approx([0.42, 0.61, 0.27])
        assert log_sorted["home_win_prob"].tolist() == pytest.approx([0.58, 0.39, 0.73])


# ---------------------------------------------------------------------------
# Test 2: Dedup on (game_id, model_name, model_type)
# ---------------------------------------------------------------------------


class TestArchiveDedup:
    """Verify the multi-column dedup key prevents duplicate rows."""

    def test_dedup_overwrites_same_game_same_model(self, archive_repo: Path) -> None:
        predictions = make_predictions(n=3)

        # Write 1: baseline
        append_to_prediction_log(
            predictions,
            model_name="win_prob",
            model_type="logistic",
            season="2025-2026",
            week=1,
            predicted_at=dt.datetime(2025, 9, 1),
            repo=archive_repo,
        )

        # Write 2: same games, same model — should replace, not duplicate
        predictions_updated = make_predictions(n=3)
        predictions_updated["AWAY_WIN_PROB"] = 0.99
        predictions_updated["HOME_WIN_PROB"] = 0.01

        append_to_prediction_log(
            predictions_updated,
            model_name="win_prob",
            model_type="logistic",
            season="2025-2026",
            week=1,
            predicted_at=dt.datetime(2025, 9, 5),
            repo=archive_repo,
        )

        log = load_prediction_log(repo=archive_repo)

        # Still 3 rows total, not 6 (dedup happened)
        assert len(log) == 3
        # The newer values replaced the old ones
        assert (log["away_win_prob"] == 0.99).all()

    def test_different_model_types_coexist(self, archive_repo: Path) -> None:
        """Same games with different (model_name, model_type) pairs both archive."""
        predictions = make_predictions(n=3)

        # Write 1: logistic
        append_to_prediction_log(
            predictions,
            model_name="win_prob",
            model_type="logistic",
            season="2025-2026",
            week=1,
            repo=archive_repo,
        )

        # Write 2: same games, different model_type
        append_to_prediction_log(
            predictions,
            model_name="win_prob",
            model_type="random_forest",
            season="2025-2026",
            week=1,
            repo=archive_repo,
        )

        log = load_prediction_log(repo=archive_repo)

        # 6 rows total: 3 games x 2 model_types
        assert len(log) == 6
        assert set(log["model_type"].unique()) == {"logistic", "random_forest"}

    def test_different_model_names_coexist(self, archive_repo: Path) -> None:
        """Same games with different model_names (purposes) both archive."""
        predictions = make_predictions(n=3)

        # Write 1: win_prob
        append_to_prediction_log(
            predictions,
            model_name="win_prob",
            model_type="logistic",
            season="2025-2026",
            week=1,
            repo=archive_repo,
        )

        # Write 2: same games, different model_name (e.g., a future "total" model)
        append_to_prediction_log(
            predictions,
            model_name="total",
            model_type="logistic",
            season="2025-2026",
            week=1,
            repo=archive_repo,
        )

        log = load_prediction_log(repo=archive_repo)

        # 6 rows total: 3 games x 2 model_names
        assert len(log) == 6
        assert set(log["model_name"].unique()) == {"win_prob", "total"}

    def test_different_game_ids_accumulate(self, archive_repo: Path) -> None:
        """Predictions for different games append rather than overwrite."""
        week1 = make_predictions(n=3, week=1, game_id_prefix="2025_01")
        week2 = make_predictions(n=3, week=2, game_id_prefix="2025_02")

        append_to_prediction_log(
            week1,
            model_name="win_prob",
            model_type="logistic",
            season="2025-2026",
            week=1,
            repo=archive_repo,
        )

        append_to_prediction_log(
            week2,
            model_name="win_prob",
            model_type="logistic",
            season="2025-2026",
            week=2,
            repo=archive_repo,
        )

        log = load_prediction_log(repo=archive_repo)
        assert len(log) == 6
        assert set(log["week"].unique()) == {1, 2}


# ---------------------------------------------------------------------------
# Test 3: Filters narrow the loaded archive correctly
# ---------------------------------------------------------------------------


class TestArchiveFilters:
    """load_prediction_log filters reduce results without losing schema."""

    def test_filter_by_season(self, archive_repo: Path) -> None:
        # Write predictions for two different seasons
        for season in ("2024-2025", "2025-2026"):
            append_to_prediction_log(
                make_predictions(n=2, season=season, game_id_prefix=season[:4]),
                model_name="win_prob",
                model_type="logistic",
                season=season,
                week=1,
                repo=archive_repo,
            )

        filtered = load_prediction_log(season="2025-2026", repo=archive_repo)

        assert (filtered["season"] == "2025-2026").all()
        assert len(filtered) == 2

    def test_filter_by_model_pair(self, archive_repo: Path) -> None:
        # Write predictions for two different (model_name, model_type) pairs
        predictions = make_predictions(n=2)

        append_to_prediction_log(
            predictions,
            model_name="win_prob",
            model_type="logistic",
            season="2025-2026",
            week=1,
            repo=archive_repo,
        )
        append_to_prediction_log(
            predictions,
            model_name="win_prob",
            model_type="random_forest",
            season="2025-2026",
            week=1,
            repo=archive_repo,
        )

        # Filter to one specific pair
        filtered = load_prediction_log(
            model_name="win_prob",
            model_type="logistic",
            repo=archive_repo,
        )

        assert len(filtered) == 2
        assert (filtered["model_type"] == "logistic").all()
