from __future__ import annotations

from pathlib import Path

import typer
from dotenv import load_dotenv
from hydra import compose, initialize_config_dir

from gridiron.api.backtest import BacktestConfig, run_backtest
from gridiron.data.schedules import load_schedules_standardized
from gridiron.utils.io import ensure_dir, write_parquet
from gridiron.utils.time import SnapshotPolicy

app = typer.Typer(name="gridiron")


def _parse_seasons_arg(arg: str) -> list[int]:
    """
    Accepts "2010-2024" or "2021" or "2018,2019,2020".
    """
    arg = arg.strip()
    if "-" in arg:
        a, b = arg.split("-", 1)
        return list(range(int(a), int(b) + 1))
    if "," in arg:
        return [int(x) for x in arg.split(",")]
    return [int(arg)]


@app.command("data-schedules")
def data_schedules(
    seasons: str = typer.Option(..., help='e.g., "2010-2024"'),
    configs_dir: str = typer.Option("configs", help="Configs directory"),
) -> None:
    load_dotenv()
    cfg_dir = str(Path(configs_dir).resolve())
    with initialize_config_dir(version_base=None, config_dir=cfg_dir):
        paths = compose(config_name="paths")

    years = _parse_seasons_arg(seasons)
    df = load_schedules_standardized(
        years,
        csv_url=paths.nfl_schedules_csv,
        verbose=True,
    )
    out_dir = Path(paths.data_standardized)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_parquet(df, out_dir / "schedules.parquet")
    typer.echo(f"Wrote {len(df)} rows → {out_dir / 'schedules.parquet'}")


@app.command("backtest")
def backtest(
    train_seasons: str = typer.Option(..., help='e.g., "2010-2021"'),
    predict_seasons: str = typer.Option(..., help='e.g., "2022"'),
    snapshot: str = typer.Option("EARLY_WED_10ET", help="EARLY_WED_10ET | T_MINUS_24H_ET"),
    mode: str = typer.Option("coinflip", help="coinflip | home-bias (stub)"),
    configs_dir: str = typer.Option("configs", help="Configs directory"),
    force_remote: bool = typer.Option(
        False, help="Ignore local parquet; fetch schedules from remote CSV"
    ),
) -> None:
    load_dotenv()
    cfg_dir = str(Path(configs_dir).resolve())
    with initialize_config_dir(version_base=None, config_dir=cfg_dir):
        paths = compose(config_name="paths")

    art_dir = ensure_dir(paths.data_artifacts)

    cfg = BacktestConfig(
        train_seasons=_parse_seasons_arg(train_seasons),
        predict_seasons=_parse_seasons_arg(predict_seasons),
        snapshot_policy=SnapshotPolicy(snapshot),
        mode=mode,
        artifacts_dir=Path(art_dir),
        schedules_parquet_path=Path(paths.data_standardized) / "schedules.parquet",
        schedules_csv_url=paths.nfl_schedules_csv,
        force_remote=force_remote,
    )

    result = run_backtest(cfg)
    typer.echo(f"Backtest complete → {result['artifacts_dir']}")


if __name__ == "__main__":
    app()
