"""`gridiron api` commands.

- ``serve`` launches the API via uvicorn.
- ``export-schema`` serializes the FastAPI OpenAPI spec to a JSON file
  that the frontend consumes to generate its typed API client.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import FastAPI
import typer
import uvicorn

api_app = typer.Typer(
    no_args_is_help=True,
    help="Run the Gridiron Edge API.",
)


@api_app.command("serve")
def serve(
    host: str = typer.Option(
        "127.0.0.1",
        help="Bind host. Use 0.0.0.0 to expose on the local network.",
    ),
    port: int = typer.Option(
        8000,
        help="Bind port.",
    ),
    reload: bool = typer.Option(
        False,
        help="Reload on source change. Local development only.",
    ),
    log_level: str = typer.Option(
        "info",
        help="uvicorn log level (critical, error, warning, info, debug, trace).",
    ),
) -> None:
    """Run the Gridiron Edge API."""
    uvicorn.run(
        "gridiron_edge.api.app:app",
        host=host,
        port=port,
        reload=reload,
        log_level=log_level,
    )


@api_app.command("export-schema")
def export_schema(
    output: Path = typer.Option(  # noqa: B008
        Path("api-schema.json"),
        "--output",
        "-o",
        help="Where to write the OpenAPI JSON schema.",
    ),
) -> None:
    """Serialize the FastAPI OpenAPI spec to a JSON file.

    Used by the frontend to generate a typed API client from the
    checked-in schema. Rerun after any API surface change to keep the
    frontend client in sync.
    """
    from gridiron_edge.api.app import create_app
    from gridiron_edge.core.console import console, step

    console.header("api export-schema", subtitle=str(output))

    with step("Serialize OpenAPI spec") as s:
        app: FastAPI = create_app()
        schema: dict[str, Any] = app.openapi()
        s.set_detail(f"{len(schema.get('paths', {}))} paths")

    with step("Write to disk") as s:
        output.write_text(json.dumps(schema, indent=2, sort_keys=True))
        s.set_detail(f"{output.stat().st_size / 1024:.1f} KB")

    console.summary()
