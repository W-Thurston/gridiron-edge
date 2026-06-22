# tests/unit/features/test_pipeline.py

from pathlib import Path


def test_data_version_changed_returns_true_when_no_manifest(tmp_path: Path) -> None:
    from gridiron_edge.features.pipeline import _data_version_changed

    assert _data_version_changed(tmp_path) is True


def test_data_version_changed_returns_true_when_version_differs(tmp_path: Path) -> None:
    import json

    from gridiron_edge.features.pipeline import _data_version_changed

    manifest_path = tmp_path / "modeling_file_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 4,
                "data_version": 0,
                "feature_names": [],
                "feature_columns": [],
                "all_columns": [],
                "row_count": 0,
            }
        )
    )

    assert _data_version_changed(tmp_path) is True


def test_data_version_changed_returns_false_when_versions_match(tmp_path: Path) -> None:
    import json

    from gridiron_edge.features.manifest import CURRENT_DATA_VERSION
    from gridiron_edge.features.pipeline import _data_version_changed

    manifest_path = tmp_path / "modeling_file_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 4,
                "data_version": CURRENT_DATA_VERSION,
                "feature_names": [],
                "feature_columns": [],
                "all_columns": [],
                "row_count": 0,
            }
        )
    )

    assert _data_version_changed(tmp_path) is False


def test_data_version_changed_returns_true_when_field_missing(tmp_path: Path) -> None:
    import json

    from gridiron_edge.features.pipeline import _data_version_changed

    manifest_path = tmp_path / "modeling_file_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 4,
                "feature_names": [],
                "feature_columns": [],
                "all_columns": [],
                "row_count": 0,
            }
        )
    )

    assert _data_version_changed(tmp_path) is True
