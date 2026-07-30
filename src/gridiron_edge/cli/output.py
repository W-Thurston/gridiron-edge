# src/gridiron_edge/cli/output.py
"""CLI commands for output generation."""

from __future__ import annotations

# pyrefly: ignore [missing-import]
import typer

output_app = typer.Typer(
    help="Write reports and outputs.",
    no_args_is_help=True,
)


@output_app.command("predictions")
def output_predictions(
    *,
    year: str = typer.Option(
        ...,
        help="NFL season label like '2026-2027'.",
    ),
    week: int = typer.Option(
        ...,
        help="Week number to render.",
    ),
    format: list[str] = typer.Option(  # noqa: B008  # type: ignore[assignment]
        [],
        help="Output format(s): png, html. Repeatable. Defaults to both.",
    ),
) -> None:
    r"""Render weekly matchup predictions image and/or HTML.

    \b
    Examples:
      gridiron output predictions --year 2026-2027 --week 1
      gridiron output predictions --year 2026-2027 --week 1 --format png
      gridiron output predictions --year 2026-2027 --week 1 --format html
    """
    from gridiron_edge.core.console import console, step
    from gridiron_edge.viz.predictions import (
        build_predictions_df,
        render_predictions_html,
        render_predictions_image,
    )

    formats = set(format) or {"png", "html"}
    console.header(
        "output predictions",
        subtitle=f"{year}  week {week}",
    )

    with step("Build predictions dataframe") as s:
        df = build_predictions_df(
            year=year,
            week=week,
        )
        if df.empty:
            s.set_detail("no data - check schedule and Elo state")
        else:
            s.set_detail(f"{len(df)} games")

    if not df.empty:
        if "png" in formats:
            with step("Render PNG") as s:
                path = render_predictions_image(
                    df,
                    year=year,
                    week=week,
                )
                s.set_detail(str(path))

        if "html" in formats:
            with step("Render HTML") as s:
                path = render_predictions_html(
                    df,
                    year=year,
                    week=week,
                )
                s.set_detail(str(path))

    console.summary()
