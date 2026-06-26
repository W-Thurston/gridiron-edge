# src/gridiron_edge/cli/api.py
"""`gridiron api serve` command.

Launches the Gridiron Edge API via uvicorn, pointing at
`gridiron_edge.api.app:app`. Reload mode is opt-in for local
development.
"""

from __future__ import annotations

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
