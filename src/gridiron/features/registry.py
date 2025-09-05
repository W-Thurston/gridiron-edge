from __future__ import annotations

from collections.abc import Callable
from typing import TypedDict

import pandas as pd

# A feature takes a DataFrame and returns a Series or DataFrame (adjust if needed).
FeatureFn = Callable[..., pd.Series | pd.DataFrame]


class FeatureMeta(TypedDict):
    fn: FeatureFn
    tags: list[str]


_FEATURES: dict[str, FeatureMeta] = {}


def register_feature(name: str, tags: list[str] | None = None) -> Callable[[FeatureFn], FeatureFn]:
    """Decorator: register a feature by name with optional tags."""
    tag_list = tags or []

    def deco(fn: FeatureFn) -> FeatureFn:
        _FEATURES[name] = {"fn": fn, "tags": tag_list}
        return fn

    return deco


def list_features() -> list[str]:
    """Return the list of registered feature names."""
    return list(_FEATURES.keys())


def get_feature_fn(name: str) -> FeatureFn:
    """Look up a registered feature function by name."""
    try:
        return _FEATURES[name]["fn"]
    except KeyError as e:
        raise KeyError(f"Unknown feature: {name!r}") from e
