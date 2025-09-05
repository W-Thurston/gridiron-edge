from gridiron.api.pipelines import walk_forward
from gridiron.data.schedules import load_schedules_standardized
from gridiron.utils.time import SnapshotPolicy


def test_no_future_leakage_in_training():
    std = load_schedules_standardized([2021, 2022])
    reg = std[std["game_type"] == "REG"].copy()
    train = reg[reg["season"] == 2021]
    predict = reg[reg["season"] == 2022]
    preds = walk_forward(train, predict, SnapshotPolicy("EARLY_WED_10ET"), mode="coinflip")
    # Predictions should exist for 2022 weeks and none for 2021
    assert (preds["season"] == 2021).sum() == 0
    assert (preds["season"] == 2022).sum() > 0
