# src/gridiron_edge/evaluation/comparable_games_corpus_builder.py
"""Build and persist one comparable-games historical corpus.

Replays the exact same feature-construction machinery live prediction and
walk-forward evaluation already trust — ``load_modeling_file``,
``_rebuild_features_with_window`` (D1/U3's fix), and
``FEATURE_SETS["combined"]`` — over every historical game, then scales the
result with the currently deployed ``win_prob``/``logistic`` champion's own
fitted scaler. No feature-engineering code is introduced by this module: it
only orchestrates already-production-trusted pieces and persists their
output as one immutable, content-addressed corpus.

Historical market outcome fields (``VEGAS_LINE``, ``FAVORITED``) come from
``data/cleaned/NFL_wk_by_wk_cleaned.csv``, joined on ``GAME_ID`` — these are
display/outcome fields, never model inputs, and require no Tier 6 market
backfill since they are already the full historical corpus.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd

from gridiron_edge.evaluation.comparable_games_corpus import (
    ComparableGamesCorpus,
    ComparableGamesFrameReference,
    create_comparable_games_corpus,
    frame_content_digest,
)
from gridiron_edge.evaluation.comparable_games_corpus_store import (
    write_comparable_games_corpus,
)
from gridiron_edge.evaluation.prediction_input_evidence import create_prediction_feature_schema

_MODEL_NAME = "win_prob"
_MODEL_TYPE = "logistic"
_LEAVE_ONE_OUT_PERCENTILE = 90.0
_META_COLUMNS: tuple[str, ...] = (
    "game_id",
    "season",
    "week",
    "game_date",
    "away_team",
    "home_team",
    "away_score",
    "home_score",
    "favorite_team",
    "spread_magnitude",
)


@dataclass(frozen=True, slots=True)
class ComparableGamesCorpusBuildResult:
    """Accounting for one persisted comparable-games corpus."""

    corpus: ComparableGamesCorpus
    manifest_path: Path
    row_count: int


def build_and_write_comparable_games_corpus(  # noqa: PLR0915
    *,
    generated_at: datetime,
    repo: Path,
) -> ComparableGamesCorpusBuildResult:
    """Build and persist the ``win_prob``/``logistic`` comparable-games corpus.

    Args:
        generated_at: Timezone-aware UTC generation timestamp (informational
            only — the corpus's identity does not depend on it).
        repo: Repository root.

    Raises:
        FileNotFoundError: If the live ``win_prob``/``logistic`` model or
            scaler artifact is not trained yet.
        ValueError: If the live artifact's persisted feature order does not
            match ``FEATURE_SETS["combined"]``.
    """
    from gridiron_edge.datasets.loaders import load_modeling_file
    from gridiron_edge.models.artifact import ArtifactStore
    from gridiron_edge.models.game_prediction._epa_window import (
        _rebuild_features_with_window,
    )
    from gridiron_edge.models.game_prediction._features import FEATURE_SETS
    from gridiron_edge.models.game_prediction.game_schema import HOME_WIN_TARGET

    store = ArtifactStore(repo)
    metadata = store.read_metadata(_MODEL_NAME, _MODEL_TYPE)
    model_path = store.model_path(_MODEL_NAME, _MODEL_TYPE)
    scaler_path = store.scaler_path(_MODEL_NAME, _MODEL_TYPE)
    if not scaler_path.exists():
        raise ValueError(f"{_MODEL_NAME}/{_MODEL_TYPE} has no persisted scaler artifact.")
    scaler = store.load_scaler(_MODEL_NAME, _MODEL_TYPE)
    if scaler is None:
        raise ValueError(f"{_MODEL_NAME}/{_MODEL_TYPE} has no persisted scaler artifact.")

    epa_window = int(metadata.parameters["epa_window"])
    modeling_schema_version = int(metadata.parameters["modeling_schema_version"])
    feature_set_name = str(metadata.parameters["feature_set"])
    feature_set = FEATURE_SETS["combined"]
    if feature_set.name != feature_set_name:
        raise ValueError(
            f"{_MODEL_NAME}/{_MODEL_TYPE} is trained on feature set {feature_set_name!r}, "
            f"but this builder only supports {feature_set.name!r}."
        )
    ordered_columns = tuple(metadata.feature_columns)
    if ordered_columns != tuple(feature_set.feature_names):
        raise ValueError(
            f"{_MODEL_NAME}/{_MODEL_TYPE}'s persisted feature order does not match "
            "FEATURE_SETS['combined']."
        )

    df = load_modeling_file(repo)
    df = _rebuild_features_with_window(df, window=epa_window, repo=repo)
    raw_features = feature_set.feature_fn(df)
    valid = raw_features.notna().all(axis=1) & df[HOME_WIN_TARGET].notna()
    valid_index = raw_features.index[valid]

    meta = df.loc[
        valid_index,
        [
            "GAME_ID",
            "YEAR",
            "WEEK_NUM",
            "GAME_DATE",
            "AWAY_TEAM",
            "HOME_TEAM",
            "AWAY_SCORE",
            "HOME_SCORE",
        ],
    ].copy()
    meta.columns = pd.Index(
        [
            "game_id",
            "season",
            "week",
            "game_date",
            "away_team",
            "home_team",
            "away_score",
            "home_score",
        ]
    )

    market = pd.read_csv(
        repo / "data" / "cleaned" / "NFL_wk_by_wk_cleaned.csv",
        usecols=["GAME_ID", "VEGAS_LINE", "FAVORITED"],
    )
    meta = meta.merge(market, left_on="game_id", right_on="GAME_ID", how="left").drop(
        columns=["GAME_ID"]
    )
    meta["favorite_team"] = meta["FAVORITED"]
    meta["spread_magnitude"] = meta["VEGAS_LINE"].abs()
    meta = meta.drop(columns=["FAVORITED", "VEGAS_LINE"]).reset_index(drop=True)

    raw = raw_features.reindex(valid_index).reset_index(drop=True)
    scaled = scaler.transform(raw.to_numpy())

    nn_distances = _leave_one_out_nearest_distances(scaled)
    distance_threshold = float(np.percentile(nn_distances, _LEAVE_ONE_OUT_PERCENTILE))
    median_distance = float(np.median(nn_distances))

    feature_frame = pd.DataFrame(scaled, columns=pd.Index(ordered_columns))
    frame = pd.concat([meta, feature_frame], axis=1)
    frame["season"] = frame["season"].astype(str)
    frame["week"] = frame["week"].astype(int)
    frame["game_date"] = frame["game_date"].astype(str)
    frame["away_score"] = frame["away_score"].astype(int)
    frame["home_score"] = frame["home_score"].astype(int)

    model_digest, _ = _file_identity(model_path)
    scaler_digest, _ = _file_identity(scaler_path)

    feature_schema = create_prediction_feature_schema(
        model_name=_MODEL_NAME,
        model_type=_MODEL_TYPE,
        task="classification",
        modeling_schema_version=modeling_schema_version,
        epa_window=epa_window,
        feature_set_name=feature_set_name,
        ordered_columns=ordered_columns,
    )

    columns = tuple(frame.columns)
    digest = frame_content_digest(columns, list(frame.itertuples(index=False, name=None)))
    # The frame's actual on-disk path is corpus_id-addressed (one frame per
    # corpus; see `comparable_games_corpus_frame_path`). `artifact` here is
    # informational only — the store never resolves it — so it is a fixed
    # template rather than a value depending on the corpus_id it appears in.
    frame_reference = ComparableGamesFrameReference(
        artifact="schema=1/frames/{corpus_id}.parquet",
        row_count=len(frame),
        columns=columns,
        content_digest=digest,
    )

    corpus = create_comparable_games_corpus(
        model_name=_MODEL_NAME,
        model_type=_MODEL_TYPE,
        model_content_digest=model_digest,
        scaler_content_digest=scaler_digest,
        feature_schema=feature_schema,
        generated_at=generated_at,
        frame=frame_reference,
        distance_threshold=distance_threshold,
        leave_one_out_median_distance=median_distance,
        leave_one_out_percentile=_LEAVE_ONE_OUT_PERCENTILE,
    )

    manifest_path = write_comparable_games_corpus(corpus, frame=frame, repo=repo)
    return ComparableGamesCorpusBuildResult(
        corpus=corpus, manifest_path=manifest_path, row_count=len(frame)
    )


def _leave_one_out_nearest_distances(vectors: np.ndarray) -> np.ndarray:
    """Return each row's distance to its nearest *other* row."""
    from sklearn.neighbors import NearestNeighbors

    neighbors = NearestNeighbors(n_neighbors=2, metric="euclidean")
    neighbors.fit(vectors)
    distances, _ = neighbors.kneighbors(vectors)
    return distances[:, 1]


def _file_identity(path: Path) -> tuple[str, int]:
    digest = sha256()
    size = 0
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size
