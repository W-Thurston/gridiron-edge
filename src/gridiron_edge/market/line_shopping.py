"""Pure current-market comparison and model guidance for line shopping."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Final, Literal

import pandas as pd
from pandas import DataFrame, Series

# pyrefly: ignore [missing-import]
from scipy.stats import norm

from gridiron_edge.market.edge import (
    expected_value,
    spread_cover_prob,
    total_cover_prob,
)
from gridiron_edge.market.odds_math import (
    american_to_implied_prob,
    decimal_to_american,
)

GuidanceStatus = Literal[
    "available",
    "model_unavailable",
    "uncertainty_unavailable",
]

REFERENCE_ODDS: Final[int] = -110
_EFFECTIVE_EV_TOLERANCE: Final[float] = 1e-12
_REQUIRED_COLUMNS: tuple[str, ...] = (
    "game_id",
    "market",
    "side",
    "line",
    "odds",
    "sportsbook",
)
_REQUIRED_PRODUCT_COLUMNS: tuple[str, ...] = (
    "game_id",
    "win_status",
    "away_win_prob",
    "home_win_prob",
    "spread_status",
    "model_spread",
    "spread_uncertainty",
    "total_status",
    "model_total",
    "total_uncertainty",
    "product_id",
    "product_run_id",
)
_MODEL_OUTPUT_COLUMNS: tuple[str, ...] = (
    "model_status",
    "model_value",
    "model_probability",
    "expected_value",
    "is_model_approved",
    "is_best_model_approved_offer",
    "product_id",
    "product_run_id",
)
_GUIDANCE_COLUMNS: tuple[str, ...] = (
    "game_id",
    "market",
    "side",
    "model_status",
    "model_value",
    "playable_line",
    "reference_odds",
    "fair_american_odds",
    "product_id",
    "product_run_id",
)
_SUPPORTED_MARKETS = frozenset({"moneyline", "spread", "total"})
_SUPPORTED_SIDES = frozenset({"away", "home", "over", "under"})


@dataclass(frozen=True, slots=True)
class LineShoppingGuidanceResult:
    """Every exact offer plus one model-guidance row per displayed outcome."""

    offers: DataFrame
    guidance: DataFrame


def classify_line_shopping_offers(quotes: DataFrame) -> DataFrame:
    """Return deterministically ordered quotes with best-line and price flags.

    Best price is evaluated only among equivalent outcomes at an identical line.
    Best line is outcome-aware: the highest spread for either team side, the
    lowest total for ``over``, and the highest total for ``under``. Moneyline has
    no line-quality classification. Ties are preserved for both flags.

    The input frame is never mutated and all source columns are preserved.
    """
    _require_columns(quotes.columns, _REQUIRED_COLUMNS, label="quotes")
    output = quotes.copy(deep=True)
    if output.empty:
        output["is_best_price"] = Series(dtype="bool")
        output["is_best_line"] = Series(dtype="bool")
        return output

    _validate_values(output)
    output["is_best_price"] = _best_price_mask(output)
    output["is_best_line"] = _best_line_mask(output)

    return output.sort_values(
        ["game_id", "market", "side", "line", "sportsbook", "odds"],
        ascending=[True, True, True, True, True, False],
        na_position="first",
        kind="stable",
    ).reset_index(drop=True)


def evaluate_line_shopping_guidance(
    product: DataFrame,
    quotes: DataFrame,
    *,
    reference_odds: int = REFERENCE_ODDS,
) -> LineShoppingGuidanceResult:
    """Attach exhaustive selected-product guidance to every exact quote.

    The function retains negative, break-even, and positive-EV offers. Model
    approval is strict positive expected value at the offer's actual line and
    price. Preferred approved offers are all effective maximum-EV ties within
    one game, market, and side. No repository I/O or recommendation filtering is
    performed.
    """
    if reference_odds == 0:
        raise ValueError("Line-shopping reference odds cannot be zero")
    _require_columns(quotes.columns, _REQUIRED_COLUMNS, label="quotes")
    _require_columns(product.columns, _REQUIRED_PRODUCT_COLUMNS, label="product")
    _validate_product(product)

    classified = classify_line_shopping_offers(quotes)
    if classified.empty:
        empty_offers = classified.copy()
        for column in _MODEL_OUTPUT_COLUMNS:
            empty_offers[column] = _empty_model_column(column)
        return LineShoppingGuidanceResult(
            offers=empty_offers,
            guidance=DataFrame(columns=list(_GUIDANCE_COLUMNS)),
        )

    product_columns = list(_REQUIRED_PRODUCT_COLUMNS)
    merged = classified.merge(
        product.loc[:, product_columns],
        on="game_id",
        how="left",
        validate="many_to_one",
        indicator=True,
    )
    missing_games = sorted(
        merged.loc[merged["_merge"] == "left_only", "game_id"].astype(str).unique().tolist()
    )
    if missing_games:
        raise ValueError(
            "Line-shopping quotes have no selected-product row for game IDs: "
            + ", ".join(missing_games)
        )
    merged = merged.drop(columns="_merge")

    evaluations = [
        _evaluate_offer(row, reference_odds=reference_odds) for _, row in merged.iterrows()
    ]
    evaluation_frame = DataFrame(evaluations, index=merged.index)
    output = pd.concat([merged, evaluation_frame], axis=1)
    output["is_best_model_approved_offer"] = _preferred_approved_mask(output)

    guidance = _build_outcome_guidance(output, reference_odds=reference_odds)
    ordered = output.sort_values(
        ["game_id", "market", "side", "line", "sportsbook", "odds"],
        ascending=[True, True, True, True, True, False],
        na_position="first",
        kind="stable",
    ).reset_index(drop=True)
    return LineShoppingGuidanceResult(offers=ordered, guidance=guidance)


def _require_columns(
    columns: Iterable[str],
    required: Iterable[str],
    *,
    label: str,
) -> None:
    available = set(columns)
    missing = [column for column in required if column not in available]
    if missing:
        raise ValueError(f"Line-shopping {label} missing columns: {missing}")


def _validate_product(product: DataFrame) -> None:
    if product["game_id"].isna().any():
        raise ValueError("Line-shopping product game_id cannot be null")
    if product["game_id"].astype(str).str.strip().eq("").any():
        raise ValueError("Line-shopping product game_id cannot be empty")
    if product["game_id"].astype(str).duplicated().any():
        raise ValueError("Line-shopping product contains duplicate game_id values")


def _validate_values(quotes: DataFrame) -> None:
    markets = set(quotes["market"].dropna().astype(str))
    unsupported_markets = sorted(markets - _SUPPORTED_MARKETS)
    if unsupported_markets:
        raise ValueError(f"Unsupported line-shopping markets: {unsupported_markets}")

    sides = set(quotes["side"].dropna().astype(str))
    unsupported_sides = sorted(sides - _SUPPORTED_SIDES)
    if unsupported_sides:
        raise ValueError(f"Unsupported line-shopping sides: {unsupported_sides}")

    valid_sides = {
        "moneyline": {"away", "home"},
        "spread": {"away", "home"},
        "total": {"over", "under"},
    }
    invalid_pairs = [
        f"{market}:{side}"
        for market, side in quotes[["market", "side"]].itertuples(index=False, name=None)
        if str(side) not in valid_sides[str(market)]
    ]
    if invalid_pairs:
        raise ValueError(
            "Unsupported line-shopping market-side pairs: " + ", ".join(sorted(set(invalid_pairs)))
        )

    if quotes[["game_id", "market", "side", "sportsbook", "odds"]].isna().any().any():
        raise ValueError("Line-shopping identity and odds columns cannot be null")
    if (quotes["odds"] == 0).any():
        raise ValueError("Line-shopping American odds cannot be zero")

    point_markets = quotes["market"].isin(("spread", "total"))
    if quotes.loc[point_markets, "line"].isna().any():
        raise ValueError("Spread and total offers require a line")
    if quotes.loc[quotes["market"] == "moneyline", "line"].notna().any():
        raise ValueError("Moneyline offers must not contain a line")


def _evaluate_offer(row: Series, *, reference_odds: int) -> dict[str, object]:
    market = str(row["market"])
    side = str(row["side"])
    status = _guidance_status(row, market=market)
    base: dict[str, object] = {
        "model_status": status,
        "model_value": None,
        "model_probability": None,
        "expected_value": None,
        "is_model_approved": None,
    }
    if status != "available":
        return base

    if market == "moneyline":
        probability = float(row[f"{side}_win_prob"])
        model_value = probability
    elif market == "spread":
        model_value = float(row["model_spread"])
        line = float(row["line"])
        home_market_spread = line if side == "home" else -line
        home_probability = spread_cover_prob(
            model_value,
            home_market_spread,
            float(row["spread_uncertainty"]),
        )
        probability = home_probability if side == "home" else 1.0 - home_probability
    else:
        model_value = float(row["model_total"])
        over_probability = total_cover_prob(
            model_value,
            float(row["line"]),
            float(row["total_uncertainty"]),
        )
        probability = over_probability if side == "over" else 1.0 - over_probability

    ev = expected_value(probability, int(row["odds"]))
    return {
        "model_status": status,
        "model_value": model_value,
        "model_probability": probability,
        "expected_value": ev,
        "is_model_approved": ev > 0.0,
    }


def _guidance_status(row: Series, *, market: str) -> GuidanceStatus:
    status_by_market = {
        "moneyline": _moneyline_guidance_status,
        "spread": _spread_guidance_status,
        "total": _total_guidance_status,
    }
    return status_by_market[market](row)


def _moneyline_guidance_status(row: Series) -> GuidanceStatus:
    available = (
        str(row["win_status"]) == "available"
        and not pd.isna(row["away_win_prob"])
        and not pd.isna(row["home_win_prob"])
    )
    return "available" if available else "model_unavailable"


def _spread_guidance_status(row: Series) -> GuidanceStatus:
    if str(row["spread_status"]) != "available" or pd.isna(row["model_spread"]):
        return "model_unavailable"
    return "uncertainty_unavailable" if pd.isna(row["spread_uncertainty"]) else "available"


def _total_guidance_status(row: Series) -> GuidanceStatus:
    status = str(row["total_status"])
    if status not in {"available", "uncertainty_unavailable"}:
        return "model_unavailable"
    if pd.isna(row["model_total"]):
        return "model_unavailable"
    return (
        "uncertainty_unavailable"
        if status != "available" or pd.isna(row["total_uncertainty"])
        else "available"
    )


def _preferred_approved_mask(offers: DataFrame) -> Series:
    result = Series(False, index=offers.index, dtype="bool")
    approved = offers.loc[offers["is_model_approved"].eq(True), :]
    for _, group in approved.groupby(["game_id", "market", "side"], sort=False):
        maximum = float(group["expected_value"].max())
        tied = group["expected_value"].sub(maximum).abs().le(_EFFECTIVE_EV_TOLERANCE)
        result.loc[group.index] = tied
    return result


def _build_outcome_guidance(offers: DataFrame, *, reference_odds: int) -> DataFrame:
    break_even = american_to_implied_prob(reference_odds)
    z_value = float(norm.ppf(break_even))
    rows: list[dict[str, object]] = []
    for (game_id, market, side), group in offers.groupby(
        ["game_id", "market", "side"],
        sort=True,
    ):
        first = group.iloc[0]
        status = str(first["model_status"])
        model_value = (
            _outcome_model_value(
                first,
                market=str(market),
                side=str(side),
            )
            if status == "available"
            else None
        )
        playable_line: float | None = None
        fair_american_odds: int | None = None
        outcome_reference_odds: int | None = None
        if status == "available":
            if market == "moneyline":
                probability = float(first["model_probability"])
                fair_american_odds = decimal_to_american(1.0 / probability)
            elif market == "spread":
                uncertainty = float(first["spread_uncertainty"])
                spread = float(first["model_spread"])
                playable_line = (
                    spread + uncertainty * z_value
                    if side == "home"
                    else -spread + uncertainty * z_value
                )
                outcome_reference_odds = reference_odds
            else:
                uncertainty = float(first["total_uncertainty"])
                total = float(first["model_total"])
                playable_line = (
                    total - uncertainty * z_value
                    if side == "over"
                    else total + uncertainty * z_value
                )
                outcome_reference_odds = reference_odds
        rows.append(
            {
                "game_id": str(game_id),
                "market": str(market),
                "side": str(side),
                "model_status": status,
                "model_value": model_value,
                "playable_line": playable_line,
                "reference_odds": outcome_reference_odds,
                "fair_american_odds": fair_american_odds,
                "product_id": str(first["product_id"]),
                "product_run_id": str(first["product_run_id"]),
            }
        )
    return DataFrame(rows, columns=list(_GUIDANCE_COLUMNS))


def _outcome_model_value(
    row: Series,
    *,
    market: str,
    side: str,
) -> float:
    """Return the point estimate from the displayed outcome's perspective."""
    if market == "moneyline":
        return float(row["model_probability"])
    if market == "spread":
        spread = float(row["model_spread"])
        return spread if side == "home" else -spread
    return float(row["model_total"])


def _empty_model_column(column: str) -> Series:
    if column in {"is_model_approved", "is_best_model_approved_offer"}:
        return Series(dtype="boolean")
    if column in {"model_value", "model_probability", "expected_value"}:
        return Series(dtype="float64")
    return Series(dtype="object")


def _best_price_mask(quotes: DataFrame) -> Series:
    group_columns = ["game_id", "market", "side", "line"]
    best_prices = quotes.groupby(group_columns, dropna=False)["odds"].transform("max")
    return quotes["odds"].eq(best_prices)


def _best_line_mask(quotes: DataFrame) -> Series:
    result = Series(False, index=quotes.index, dtype="bool")
    group_columns = ["game_id", "market", "side"]

    for _, group in quotes.groupby(
        group_columns,
        dropna=False,
        sort=False,
    ):
        market = str(group["market"].iloc[0])
        side = str(group["side"].iloc[0])
        if market == "spread":
            best_line = group["line"].max()
        elif market == "total" and side == "over":
            best_line = group["line"].min()
        elif market == "total" and side == "under":
            best_line = group["line"].max()
        else:
            continue
        result.loc[group.index] = group["line"].eq(best_line)

    return result
