# src/gridiron_edge/api/app.py
"""FastAPI app factory for the Gridiron Edge API.

The app is read-only at W8. Routes attach in later steps. The factory
shape lets tests build isolated app instances with stubbed dependencies.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# OpenAPI tag inventory — one entry per route domain. Tier 1 tags
# describe populated endpoints; Tier 3 tags describe blocked endpoints
# whose unblock work is tracked in ROADMAP §9.5.
_OPENAPI_TAGS: list[dict[str, str]] = [
    # Populated endpoints.
    {"name": "weeks", "description": "Current week and season."},
    {"name": "games", "description": "Game list, detail, and predictions."},
    {"name": "edges", "description": "Model edges across markets."},
    {"name": "teams", "description": "Power rankings and team profiles."},
    {"name": "projections", "description": "Season and playoff projections."},
    {"name": "props", "description": "Player prop list and detail."},
    {"name": "portfolio", "description": "Bankroll, bets, performance."},
    {"name": "compare", "description": "Team vs team and player vs defense."},
    {"name": "model", "description": "Model performance metrics."},
    {"name": "weeks", "description": "Current NFL season and week."},
    {"name": "portfolio", "description": "Bankroll, bets, curve, transactions, splits."},
    # Endpoints blocked on upstream gaps; tags exist now so /docs groups
    # them correctly even while routes return placeholder shapes.
    {
        "name": "lines",
        "description": "Multi-book line shopping. Blocked on multi-book odds ingest.",
    },
    {"name": "live", "description": "Live game state. Blocked on live state ingest."},
    {"name": "news", "description": "Injury and market news. Blocked on news ingest."},
    {"name": "injuries", "description": "Game injuries. Blocked on injury data source (§5.3)."},
    {
        "name": "explain",
        "description": "Win-probability explainability. Blocked on scenario engine.",
    },
    {
        "name": "swing-factors",
        "description": "Per-game swing factors. Blocked on feature attribution.",
    },
    {
        "name": "comparables",
        "description": "Historical comparable games. Blocked on comparables retrieval.",
    },
    {
        "name": "prop-shop",
        "description": "Per-prop multi-book shopping. Blocked on multi-book odds ingest.",
    },
    {
        "name": "prop-reasoning",
        "description": "Per-prop model reasoning. Blocked on feature attribution.",
    },
    {"name": "model", "description": "Model prediction quality and betting performance."},
    {"name": "teams", "description": "Power rankings and per-team profiles."},
]


def create_app() -> FastAPI:
    """Build a Gridiron Edge API app instance.

    Returns a fresh `FastAPI` instance with OpenAPI metadata and CORS
    middleware configured. Routes are not attached here; callers (or
    later steps in W8) wire routers via `app.include_router(...)`.

    Returns:
        Configured FastAPI app, ready to mount routers on.
    """
    app = FastAPI(
        title="Gridiron Edge API",
        version="0.1.0",
        description=(
            "Read-only REST API exposing Gridiron Edge analytics outputs. "
            "See PLAN.md for the active workstream and ROADMAP.md §9.5 for "
            "the backend-gaps backlog that drives field-level placeholders."
        ),
        openapi_tags=_OPENAPI_TAGS,
    )

    # Permissive CORS for local development. Tightening is a W9 concern
    # once a deployment surface exists.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["GET"],
        allow_headers=["*"],
    )

    from gridiron_edge.api.routes import (
        comparables,
        explain,
        injuries,
        lines,
        live,
        model,
        news,
        portfolio,
        prop_reasoning,
        prop_shop,
        swing_factors,
        teams,
        weeks,
    )

    app.include_router(comparables.router)
    app.include_router(explain.router)
    app.include_router(injuries.router)
    app.include_router(lines.router)
    app.include_router(live.router)
    app.include_router(model.router)
    app.include_router(news.router)
    app.include_router(portfolio.router)
    app.include_router(prop_reasoning.router)
    app.include_router(prop_shop.router)
    app.include_router(swing_factors.router)
    app.include_router(teams.router)
    app.include_router(weeks.router)

    return app


# Module-level app for uvicorn's import string (`gridiron_edge.api.app:app`).
app: FastAPI = create_app()
