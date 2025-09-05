from __future__ import annotations

import pandas as pd


class BinaryProbModel:
    def fit(self, x: pd.DataFrame, y: pd.Series) -> BinaryProbModel:
        """Fit on features x and binary target y (1=home win)."""
        return self

    def predict_proba(self, x: pd.DataFrame) -> pd.Series:
        """Return probability of home team winning."""
        raise NotImplementedError
