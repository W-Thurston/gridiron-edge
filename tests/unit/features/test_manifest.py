# tests/unit/features/test_manifest.py

from pathlib import Path


def test_manifest_includes_data_version(tmp_path: Path) -> None:
    """write_manifest stores data_version in the JSON."""
    import pandas as pd

    from gridiron_edge.features.manifest import (
        CURRENT_DATA_VERSION,
        read_manifest,
        write_manifest,
    )

    df = pd.DataFrame({"GAME_ID": ["x"], "TEAM_A": ["a"], "TEAM_B": ["b"]})
    write_manifest(
        df,
        feature_names=["home_field"],
        feature_columns=["HOME_FIELD"],
        modeling_dir=tmp_path,
    )

    manifest = read_manifest(tmp_path)
    assert manifest["data_version"] == CURRENT_DATA_VERSION


def test_manifest_accepts_custom_data_version(tmp_path: Path) -> None:
    """write_manifest accepts a non-default data_version."""
    import pandas as pd

    from gridiron_edge.features.manifest import read_manifest, write_manifest

    df = pd.DataFrame({"GAME_ID": ["x"]})
    write_manifest(
        df,
        feature_names=[],
        feature_columns=[],
        modeling_dir=tmp_path,
        data_version=42,
    )

    manifest = read_manifest(tmp_path)
    assert manifest["data_version"] == 42
