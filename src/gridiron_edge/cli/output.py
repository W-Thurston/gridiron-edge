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
    season: str = typer.Option(
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
      gridiron output predictions --season 2026-2027 --week 1
      gridiron output predictions --season 2026-2027 --week 1 --format png
      gridiron output predictions --season 2026-2027 --week 1 --format html
    """
    from gridiron_edge.core.console import console, step
    from gridiron_edge.core.settings import get_settings
    from gridiron_edge.datasets.loaders import load_current_weekly_product
    from gridiron_edge.evaluation.forecast_evidence_disposition import (
        require_operational_weekly_product,
    )
    from gridiron_edge.evaluation.forecast_evidence_disposition_store import (
        list_forecast_evidence_dispositions,
    )
    from gridiron_edge.viz.predictions import (
        build_weekly_product_display_frame,
        render_predictions_html,
        render_predictions_image,
    )

    formats = set(format) or {"png", "html"}
    unsupported = sorted(formats - {"png", "html"})
    if unsupported:
        raise typer.BadParameter(
            "Unsupported format(s): " + ", ".join(unsupported),
            param_hint="--format",
        )

    console.header(
        "output predictions",
        subtitle=f"{season}  week {week}",
    )
    repo = get_settings().repo_root

    with step("Load selected weekly product") as s:
        product = load_current_weekly_product(
            repo,
            season=season,
            week=week,
        )

        try:
            if "product_id" not in product.columns:
                raise ValueError("Selected weekly product is missing required column: product_id")

            product_ids = tuple(
                sorted(product["product_id"].dropna().astype(str).str.strip().unique())
            )
            if len(product_ids) != 1 or product_ids == ("",):
                raise ValueError("Selected weekly product must contain one nonempty product_id.")

            product_id = next(iter(product_ids))
            dispositions = list_forecast_evidence_dispositions(
                season=season,
                week=week,
                product_id=product_id,
                repo=repo,
            )
            require_operational_weekly_product(
                product,
                dispositions,
            )
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from exc

        df = build_weekly_product_display_frame(product)
        s.set_detail(f"{len(df)} games · {product_id}")

    if "png" in formats:
        with step("Render PNG") as s:
            path = render_predictions_image(
                df,
                year=season,
                week=week,
                repo=repo,
            )
            s.set_detail(str(path))

    if "html" in formats:
        with step("Render HTML") as s:
            path = render_predictions_html(
                df,
                year=season,
                week=week,
                repo=repo,
            )
            s.set_detail(str(path))

    console.summary()
