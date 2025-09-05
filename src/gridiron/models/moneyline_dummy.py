from __future__ import annotations

import pandas as pd

from gridiron.models.interfaces import BinaryProbModel


class MoneylineDummyModel(BinaryProbModel):
    def __init__(self, mode: str = "coinflip") -> None:
        self.mode = mode
        self.p_home = 0.5

    def fit(self, x: pd.DataFrame, y: pd.Series) -> MoneylineDummyModel:
        if self.mode == "home-bias":
            self.p_home = float(y.mean()) if len(y) else 0.5
        else:
            self.p_home = 0.5
        return self

    def predict_proba(self, x: pd.DataFrame) -> pd.Series:
        return pd.Series(self.p_home, index=x.index, dtype="float")
