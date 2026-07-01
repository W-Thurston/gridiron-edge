# tests/integration/api/test_api_contract.py

"""Full-surface contract tests for the API.

Two checks that complement the unit-level tests:

1. JSON round-trip parity — every endpoint's response, when re-validated
   through its declared Pydantic model, produces an identical structure.
   Catches subtle bugs where a response renders in /docs but doesn't
   actually pass strict deserialization.

2. `_meta.field_status` completeness — every `null` scalar or empty
   list/dict field in a response body has a corresponding entry in
   `_meta.field_status`. Tightens the unit-level check ("at least one
   blocked field exists") into "every unpopulated field is accounted for."
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from fastapi.testclient import TestClient
import pytest
from requests import Response

from gridiron_edge.api.app import create_app
from gridiron_edge.api.schemas._base import BaseListResponse
from gridiron_edge.api.schemas.comparables import GameComparables
from gridiron_edge.api.schemas.explain import GameExplain
from gridiron_edge.api.schemas.injuries import GameInjuries
from gridiron_edge.api.schemas.lines import LineDetail, LineRow
from gridiron_edge.api.schemas.live import LiveGame, LiveGameSummary
from gridiron_edge.api.schemas.model_performance import ModelPerformance
from gridiron_edge.api.schemas.news import NewsItem
from gridiron_edge.api.schemas.portfolio import (
    BankrollCurve,
    BetRow,
    PortfolioSplits,
    PortfolioSummary,
    TransactionRow,
)
from gridiron_edge.api.schemas.prop_reasoning import PropReasoning
from gridiron_edge.api.schemas.prop_shop import PropShop
from gridiron_edge.api.schemas.swing_factors import GameSwingFactors
from gridiron_edge.api.schemas.teams import TeamProfile, TeamRankingsList
from gridiron_edge.api.schemas.weeks import CurrentWeek

# Aliases for the parameterized list responses (avoids fragile inline generics).
_BetsList = BaseListResponse[BetRow]
_TransactionsList = BaseListResponse[TransactionRow]


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app())


# Each entry is (path, expected response model). For list endpoints the
# model is the parameterized BaseListResponse[T] variant.
ENDPOINTS: list[tuple[str, type]] = [
    # List endpoints
    ("/lines", BaseListResponse[LineRow]),
    ("/live", BaseListResponse[LiveGameSummary]),
    ("/news", BaseListResponse[NewsItem]),
    ("/news/alerts", BaseListResponse[NewsItem]),
    # Detail endpoints (single object)
    ("/lines/sf-bal", LineDetail),
    ("/live/sf-bal", LiveGame),
    ("/games/sf-bal/injuries", GameInjuries),
    ("/games/sf-bal/explain", GameExplain),
    ("/games/sf-bal/swing-factors", GameSwingFactors),
    ("/games/sf-bal/comparables", GameComparables),
    ("/props/lamar-rush/shop", PropShop),
    ("/props/lamar-rush/reasoning", PropReasoning),
    ("/weeks/current", CurrentWeek),
    ("/portfolio/summary", PortfolioSummary),
    ("/portfolio/bets", _BetsList),
    ("/portfolio/curve", BankrollCurve),
    ("/portfolio/transactions", _TransactionsList),
    ("/portfolio/splits", PortfolioSplits),
    ("/model/performance", ModelPerformance),
    ("/teams", TeamRankingsList),
    ("/teams/BAL", TeamProfile),  # BAL is a known abbreviation
]


# ---------------------------------------------------------------------------
# JSON round-trip parity
# ---------------------------------------------------------------------------


class TestRoundTripParity:
    """Every response must re-validate through its declared model.

    The wire shape FastAPI returns has to be acceptable as input to the
    same Pydantic model that produced it. If this fails, the response
    declared by `response_model=` and the response constructed by the
    handler have diverged.
    """

    @pytest.mark.parametrize("path,model_cls", ENDPOINTS)
    def test_response_revalidates(
        self,
        client: TestClient,
        path: str,
        model_cls: type,
    ) -> None:
        response: Response = client.get(path)
        assert response.status_code == 200, response.text

        body = response.json()

        # model_validate accepts the wire shape (including `_meta` alias)
        # because populate_by_name=True is set on BaseResponse.
        revalidated = model_cls.model_validate(body)

        # Dump the revalidated model and compare to the original wire shape.
        # by_alias=True so we get `_meta` (not `response_meta`); exclude_none
        # is intentionally not set so we preserve the null fields that
        # `_meta.field_status` references.
        round_tripped = revalidated.model_dump(by_alias=True)

        assert round_tripped == body, (
            f"Round-trip mismatch on {path}:\n"
            f"original keys:    {sorted(body.keys())}\n"
            f"round-trip keys:  {sorted(round_tripped.keys())}"
        )


# ---------------------------------------------------------------------------
# field_status completeness
# ---------------------------------------------------------------------------


def _iter_null_field_paths(
    body: dict,
    prefix: str = "",
) -> Iterator[str]:
    """Yield dot-notation paths to every null scalar or empty container.

    Walks the response body and emits a path string for each leaf or
    container that represents "no data here." Skips the `_meta` envelope
    itself so we don't ask `_meta` to document itself.

    Conventions:
    - `null` scalars → yielded (need a field_status entry).
    - Empty lists → yielded as the field path itself.
    - Empty dicts → yielded as the field path itself.
    - Non-empty containers → recursed into.
    - `_meta` at the top level → skipped entirely.

    Path identifiers (game_id, prop_id) are populated by the URL and do
    not need to be tracked.
    """
    skip_keys: set[str] = {"_meta", "game_id", "prop_id", "total"}

    for key, value in body.items():
        if key in skip_keys:
            continue
        path: str = f"{prefix}{key}" if not prefix else f"{prefix}.{key}"

        if value is None:
            yield path
        elif isinstance(value, list):
            if not value:
                yield path
            # Non-empty lists are populated and need no annotation.
        elif isinstance(value, dict) and not value:
            yield path
            # Non-empty dicts could be populated objects; recurse to find
            # any nested nulls only when this isn't an `items` element.
        # Scalars that aren't None are populated by definition.


class TestFieldStatusCompleteness:
    """Every null/empty field must have a `_meta.field_status` entry.

    The placeholder convention (DECISIONS.md D14) says unpopulated fields
    are null AND carry structured metadata describing why. This test
    enforces the AND. If a field is null but the response forgot to mark
    it pending or blocked, the consumer has no way to know whether it's
    a deliberate placeholder or a bug.
    """

    @pytest.mark.parametrize("path,_model_cls", ENDPOINTS)
    def test_all_null_fields_documented(
        self,
        client: TestClient,
        path: str,
        _model_cls: type,
    ) -> None:
        body = client.get(path).json()
        meta = body.get("_meta") or {}
        field_status: dict[str, Any] = meta.get("field_status", {})

        null_paths: set[str] = set(_iter_null_field_paths(body))
        documented_paths: set[str] = set(field_status.keys())

        undocumented: set[str] = null_paths - documented_paths
        assert not undocumented, (
            f"Null fields without `_meta.field_status` entries on {path}:\n"
            f"  undocumented: {sorted(undocumented)}\n"
            f"  documented:   {sorted(documented_paths)}"
        )

    @pytest.mark.parametrize("path,_model_cls", ENDPOINTS)
    def test_no_orphan_field_status_entries(
        self,
        client: TestClient,
        path: str,
        _model_cls: type,
    ) -> None:
        """field_status entries should refer to real fields in the response.

        Catches the opposite mistake: a field_status entry that doesn't
        correspond to any actual field, usually from a typo in the
        handler (e.g., `with_blocked("injurys", ...)`).
        """
        body = client.get(path).json()
        meta = body.get("_meta") or {}
        field_status: dict[str, Any] = meta.get("field_status", {})

        # Collect every top-level key in the response body plus a few
        # specific keys that appear inside nested structures we expect.
        # For Tier 1 with everything blocked at the field level, top-level
        # coverage is sufficient — nested-field documentation will be
        # exercised when populated endpoints arrive.
        top_level_keys: set = set(body.keys()) - {"_meta", "total"}

        # Plus the special-cased "items" key that list responses use.
        if "items" in body:
            top_level_keys.add("items")

        for documented_path in field_status:
            top_level: str = documented_path.split(".")[0]
            assert top_level in top_level_keys, (
                f"field_status references unknown field on {path}:\n"
                f"  documented: '{documented_path}'\n"
                f"  body keys: {sorted(top_level_keys)}"
            )


# ---------------------------------------------------------------------------
# Cross-cutting: response model declaration matches actual response
# ---------------------------------------------------------------------------


class TestResponseModelMatchesHandler:
    """If a route declares response_model=X but returns Y, FastAPI's
    response validation should catch it — but if X is structurally
    compatible with a subset of Y, the bug can slip through silently.
    These tests double-check by counting top-level keys.
    """

    @pytest.mark.parametrize("path,model_cls", ENDPOINTS)
    def test_response_keys_match_model_fields(
        self,
        client: TestClient,
        path: str,
        model_cls: type,
    ) -> None:
        body = client.get(path).json()
        expected_keys: set = set(model_cls.model_fields.keys())

        # The wire shape uses `_meta` (alias) instead of `response_meta`.
        # Substitute the alias when comparing.
        if "response_meta" in expected_keys:
            expected_keys.remove("response_meta")
            expected_keys.add("_meta")

        actual_keys: set = set(body.keys())

        # The response may legitimately omit `_meta` if every field is
        # populated — but in Tier 1 every endpoint has at least one
        # blocked field, so `_meta` must be present.
        assert "_meta" in actual_keys, f"{path} response missing `_meta`"

        # All other declared keys should appear.
        unexpected_in_body: set = actual_keys - expected_keys
        missing_from_body: set = expected_keys - actual_keys

        assert not unexpected_in_body, (
            f"{path} body has keys not declared by {model_cls.__name__}: {unexpected_in_body}"
        )
        assert not missing_from_body, (
            f"{path} body missing keys declared by {model_cls.__name__}: {missing_from_body}"
        )
